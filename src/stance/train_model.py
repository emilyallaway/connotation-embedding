import torch, sys, os, argparse
sys.path.append('./modeling')
import baseline_models as bm
import data_utils, model_utils, datasets
import input_models as im
import torch.optim as optim
import torch.nn as nn

SEED  = 0
LOCAL = False
NUM_GPUS = 2  # None
use_cuda = torch.cuda.is_available()


def train(model_handler, num_epochs, verbose=True, corpus_samplers=None, dev_data=None,
          use_score='f_macro'):
    '''
    Trains the given model using the given data for the specified
    number of epochs. Prints training loss and evaluation starting
    after 10 epochs. Saves at most 10 checkpoints plus a final one.
    :param model_handler: a holder with a model and data to be trained.
                            Assuming the model is a pytorch model.
    :param num_epochs: the number of epochs to train the model for.
    :param verbose: whether or not to print train results while training.
                    Default (True): do print intermediate results.
    :param corpus_samplers: list of samplers for individual corpora, None
                            if only evaling on the full corpus.
    '''
    for epoch in range(num_epochs):
        model_handler.train_step()

        if verbose:
            # print training loss and training (& dev) scores, ignores the first few epochs
            if epoch >= 0:
                print("training loss: {}".format(model_handler.loss))
                # eval model on training data
                trn_scores = eval_helper(model_handler, corpus_samplers, data_name='TRAIN')
                # update best scores
                if dev_data is not None:
                    dev_scores = eval_helper(model_handler, corpus_samplers, data_name='DEV',
                                             data=dev_data)
                    model_handler.save_best(scores=dev_scores)
                else:
                    model_handler.save_best(scores=trn_scores)

    # save final checkpoint
    model_handler.save(num="FINAL")

    # print final training (& dev) scores
    eval_helper(model_handler, corpus_samplers, data_name='TRAIN')
    if dev_data is not None:
        eval_helper(model_handler, corpus_samplers, data_name='DEV', data=dev_data)



def eval_helper(model_handler, corpus_samplers, data_name, data=None):
    '''
    Helper function for evaluating the model during training.
    Can evaluate on all the data or just a subset of corpora.
    :param model_handler: the holder for the model
    :param corpus_samplers: a list of samplers for individual corpora,
                            None if only evaling on the full corpus
    :return: the scores from running on all the data
    '''
    # eval on full corpus
    scores = model_handler.eval_and_print(data=data, data_name=data_name)
    if corpus_samplers is not None:
        # eval on src subsets of the corpus
        for sampler in corpus_samplers:
            model_handler.eval_and_print(data=sampler, data_name=sampler.corpus)
    return scores


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--config_file', help='Name of the cofig data file', required=False)
    parser.add_argument('-i', '--trn_data', help='Name of the training data file', required=False)
    parser.add_argument('-d', '--dev_data', help='Name of the dev data file', default=None, required=False)
    parser.add_argument('-c', '--corpora', help='Corpora to evaluate on individually. Separate with ;',
                        default=None, required=False)
    parser.add_argument('-b', '--sampling', help='Whether to do balanced sample weighting in the loss.',
                        default=None, required=False)
    parser.add_argument('-w', '--weighting', help='Whether to weight the loss per topic',
                        default=None, required=False)
    parser.add_argument('-n', '--name', help='something to add to the saved model name',
                        required=False, default='')
    args = vars(parser.parse_args())

    torch.manual_seed(SEED)

    args['sampling'] = None
    args['weighting'] = None

    ####################
    # load config file #
    ####################
    with open(args['config_file'], 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]

    ################
    # load vectors #
    ################
    vec_path='../../resources/stance'

    vec_name = config['vec_name']
    vec_dim  = int(config['vec_dim'])

    vecs = data_utils.load_vectors('{}/{}.vectorsF.npy'.format(vec_path, vec_name),
                                   dim=vec_dim, seed=SEED)

    if 'conn_vec_name' in config:
        pos2conn_vecs = dict()
        for d,t in zip(config['conn_vec_dims'].split(','), config['conn_vec_tags'].split(',')):
            conn_vec_path = '{}/{}-{}_vectorsF.npy'.format(vec_path, config['conn_vec_name'], t)
            conn_vecs = data_utils.load_vectors(conn_vec_path, dim=int(d), unk_rand=False)
            pos2conn_vecs[t] = conn_vecs

    #############
    # LOAD DATA #
    #############
    # load training data
    vocab_name = '{}/{}.vocabF.pkl'.format(vec_path, vec_name)
    if 'conn_vec_name' not in config:
        data = datasets.StanceData(args['trn_data'], vocab_name, pad_val=len(vecs) - 1)
    else:
        trim_n = int(config['truncate']) if 'truncate' in config else None
        data = datasets.StanceDataPos(args['trn_data'], vocab_name, pad_val=len(vecs) - 1,
                                      pos_lst=config['conn_vec_tags'].split(','), truncate_data=trim_n)

    dataloader = data_utils.DataSampler(data, batch_size=int(config['b']), sampling=args['sampling'],
                                        weighting=args['weighting'])

    # load dev data if specified
    if args['dev_data'] is not None:
        if 'conn_vec_name' not in config:
            dev_data = datasets.StanceData(args['dev_data'], vocab_name,
                                           pad_val=len(vecs) - 1)
        else:
            trim_dev = int(config['truncate_dev']) if 'truncate_dev' in config else None
            dev_data = datasets.StanceDataPos(args['dev_data'], vocab_name, pad_val=len(vecs)-1,
                                              pos_lst=config['conn_vec_tags'].split(','),
                                              truncate_data=trim_dev)

        dev_dataloader = data_utils.DataSampler(dev_data, batch_size=int(config['b']), shuffle=False,
                                                sampling=args['sampling'],
                                                weighting=args['weighting'])
    else:
        dev_dataloader = None

    corpus_sampler_lst = None

    # RUN
    print("Using cuda?: {}".format(use_cuda))
    if 'BiCondLSTM' in config['name']:
        batch_args = {}
        input_layer = im.BasicWordEmbedLayer(vecs=vecs, use_cuda=use_cuda)
        setup_fn = model_utils.setup_helper_bicond

        nl = 3
        loss_fn = nn.CrossEntropyLoss()

        model = bm.BiCondLSTMModel(int(config['h']), input_layer.dim,
                                   drop_prob=float(config['dropout']), use_cuda=use_cuda,
                                   num_labels=nl)
        o = optim.Adam(model.parameters(), lr=.001)

        kwargs = {'model': model, 'embed_model': input_layer, 'dataloader': dataloader,
                  'batching_fn': data_utils.prepare_batch_with_reverse,
                  'batching_kwargs': batch_args, 'name': config['name'] + args['name'],
                  'loss_function': loss_fn,
                  'optimizer': o,
                  'setup_fn': setup_fn}

        model_handler = model_utils.TorchModelHandler(use_cuda=use_cuda, **kwargs)

    elif 'BiCondAttention' in config['name']:
        batch_args = {'use_conn': True}
        input_layer = im.ConnVecsLayerSeparate(vecs, pos2conn_vecs, use_random=('random' in config))

        nl = 3
        loss_fn = nn.CrossEntropyLoss()

        model = bm.BiCondLSTMAttentionModel(int(config['h']), input_layer.dim,
                                            conn_dim=input_layer.conn_dim,
                                   drop_prob=float(config['dropout']), use_cuda=use_cuda,
                                   num_labels=nl,
                                            att_type=config['att'])

        kwargs = {'model': model, 'embed_model': input_layer, 'dataloader': dataloader,
                  'batching_fn': data_utils.prepare_batch_with_reverse,
                  'batching_kwargs': batch_args, 'name': config['name'] + args['name'],
                  'loss_function': loss_fn,
                  'optimizer': optim.Adam(model.parameters()),
                  'setup_fn': model_utils.setup_helper_bicond_connvecs}

        model_handler = model_utils.TorchModelHandler(use_cuda=use_cuda, **kwargs)


    train(model_handler, int(config['epochs']), corpus_samplers=corpus_sampler_lst,
              dev_data=dev_dataloader)