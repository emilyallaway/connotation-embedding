import torch, sys, os, argparse
sys.path.append('./modeling')
import embedding_models as em
import data_utils, model_utils, datasets, utils
import model_layers as ml
import torch.optim as optim
import torch.nn as nn
from itertools import chain

SEED  = 0
LOCAL = False
NUM_GPUS = 1
use_cuda = torch.cuda.is_available()

ENC_TYPE='lstm'
USE_WORD=True
CONN_LOSS_FACTOR = 0.01
SCALE_ATT='scale'

L2_FACTOR = 1.0

CONN_LOSS = nn.CrossEntropyLoss
CONN_EMO_LOSS = nn.MultiLabelSoftMarginLoss
WEIGHTS = [.3, .5, .3, 1/6., 1/6.] + \
          [1., 1., 1., 1., 1., 3., 3., 1., 1., 3., 3.] + [3] # Emo is last # W7

dim2i = {'Social Val': 0, 'Polite': 1, 'Impact': 2, 'Fact': 3,
         'Sent':4, 'Emo': 5}

verbna_dim2i = {'Social Val': 0, 'Polite': 1, 'Impact': 2, 'Fact': 3, 'Sent':4,
              'P(wt)': 5, 'P(wa)': 6, 'P(at)': 7, 'E(t)': 8, 'E(a)': 9, 'V(t)': 10, 'V(a)': 11,
              'S(t)': 12, 'S(a)': 13, 'power': 14, 'agency': 15, 'Emo': 16}

NUM_VERB_DIMS = 9 # 12



def train(model_handler, num_epochs, verbose=True, dev_data=None,
          use_score='f_macro', num_prep=None):
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
    if num_prep is None:
        num_prep = num_epochs
    is_eval = False

    for epoch in range(num_epochs):
        model_handler.train_step()

        if verbose:
            # print training loss and training (& dev) scores, ignores the first few epochs
            print("training loss: {}".format(model_handler.loss))
            # print("training loss on verbs: {}".format(model_handler.verb_loss))
            if epoch >= 0:
                # eval model on training data
                trn_scores = eval_helper(model_handler, data_name='TRAIN', is_eval=is_eval)
                # update best scores
                if dev_data is not None:
                    dev_scores = eval_helper(model_handler, data_name='DEV',
                                             data=dev_data, is_eval=is_eval)
                    model_handler.save_best(scores=dev_scores)
                else:
                    model_handler.save_best(scores=trn_scores)

    # save final checkpoint
    model_handler.save(num="FINAL")

    # print final training (& dev) scores
    eval_helper(model_handler, data_name='TRAIN')
    if dev_data is not None:
        eval_helper(model_handler, data_name='DEV', data=dev_data)


def eval_helper(model_handler, data_name, data=None, is_eval=False):
    '''
    Helper function for evaluating the model during training.
    Can evaluate on all the data or just a subset of corpora.
    :param model_handler: the holder for the model
    :param corpus_samplers: a list of samplers for individual corpora,
                            None if only evaling on the full corpus
    :return: the scores from running on all the data
    '''
    # eval on full corpus
    scores = model_handler.eval_and_print(data=data, data_name=data_name,
                                          is_eval=is_eval)
    return scores


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--config_file', help='Name of the cofig data file', required=False)
    parser.add_argument('-i', '--trn_data', help='Name of the training data file', required=False)
    parser.add_argument('-d', '--dev_data', help='Name of the dev data file', default=None, required=False)
    parser.add_argument('-b', '--sampling', help='Whether to use balanced class weights for sampling',
                        default=None, required=False)
    parser.add_argument('-n', '--dims', help='The names of the dims to use', required=False,
                        default='Social Val;Polite;Impact;Fact;Sent;Emo')
    parser.add_argument('-m', '--verb_dims', help='The verb dimensions to use', required=False,
                        default='P(wt);P(wa);P(at);E(t);E(a);V(t);V(a);S(t);S(a);power;agency')
    parser.add_argument('-r', '--use_related', help='Whether to use related words', default=True,
                        required=False)
    args = vars(parser.parse_args())

    torch.manual_seed(SEED)

    ####################
    # load config file #
    ####################
    with open(args['config_file'], 'r') as f:
        config = dict()
        for l in f.readlines(): config[l.strip().split(":")[0]] = l.strip().split(":")[1]

    ##############
    # parse args #
    ##############
    ori_dims = args['dims'].split(';') if args['dims'] != '' else []
    num_dims = len(ori_dims)
    weight_lst = WEIGHTS
    loss_fn_lst = [CONN_LOSS for _ in range(5)] + [CONN_EMO_LOSS]
    name = config['name']

    verb_dims = args['verb_dims'].split(';') if args['verb_dims'] != '' else []
    if 'use_verb' in config:
        num_dims += len(verb_dims)
        if 'Emo' in ori_dims:
            dims = ori_dims[:-1] # remove Emo, which is LAST
            dims += verb_dims + ['Emo']
        else:
            dims = ori_dims + verb_dims
        loss_fn_lst = [CONN_LOSS for _ in range(5 + NUM_VERB_DIMS + 2)] + [CONN_EMO_LOSS]  # Emo is LAST
    else:
        dims = ori_dims

    if 'verb_only' in config:
        dims = verb_dims

    if args['sampling'] is not None:
        dim_weights = weight_lst
        name += '-{}'.format(args['sampling'])
    else:
        dim_weights = None

    if ('use_verb' not in config and ';'.join(dims) != 'Social Val;Polite;Impact;Fact;Sent;Emo') or \
            ('use_verb' in config and
                     ';'.join(dims) != 'Social Val;Polite;Impact;Fact;Sent;' +
                     'P(wt);P(wa);P(at);E(t);E(a);V(t);V(a);S(t);S(a);power;agency;Emo'):
        name += '-' + '.'.join(['_'.join(d.split()) for d in dims])
        if 'use_verb' not in config:
            use_dim2i = dim2i
        else:
            use_dim2i = verbna_dim2i

        weight_lst = [weight_lst[use_dim2i[d]] for d in dims]
        loss_fn_lst = [loss_fn_lst[use_dim2i[d]] for d in dims]
        num_dims = len(weight_lst)
        if 'verb_only' in config:
            ori_dims = []
        if 'use_verb' not in config:
            verb_dims = []

        if len(weight_lst) == 1:
            weight_lst[0] = 1.

    ################
    # load vectors #
    ################
    vec_path = '../../resources/connotations'

    vec_name = config['vec_name']
    vec_dim = int(config['vec_dim'])

    vecs = data_utils.load_vectors('{}/{}.vectorsF.npy'.format(vec_path, vec_name),
                                   dim=vec_dim, seed=SEED)

    #############
    # LOAD DATA #
    #############
    if 'dict_data' in config:
        rd_task = True
    else:
        rd_task = False
    # load training data
    vocab_name = '{}/{}.vocabF.pkl'.format(vec_path, vec_name)
    use_related = args['use_related']
    data = datasets.ConnData(args['trn_data'], vocab_name, pad_val=len(vecs) - 1,
                                 dims=ori_dims, use_related=use_related,
                             dict_task=rd_task, verb_task=('use_verb' in config), verb_dims=verb_dims)

    dataloader = data_utils.POSDataSampler(data, batch_size=int(config['b']),
                                            dim_weights=dim_weights, num_resets=0,
                                            sampling=args['sampling'], verb_only=('verb_only' in config),
                                           na_only=('use_verb' not in config))
    print(len(dataloader.ori_verb_indices))
    # load dev data if specified
    if args['dev_data'] is not None:
        dev_data = datasets.ConnData(args['dev_data'], vocab_name,
                                       pad_val=len(vecs) - 1, dims=ori_dims,
                                     use_related=use_related,
                                     dict_task=rd_task, verb_task=('use_verb' in config), verb_dims=verb_dims)
        dev_dataloader = data_utils.POSDataSampler(dev_data, batch_size=int(config['b']), shuffle=False,
                                                       num_resets=0, verb_only=('verb_only' in config),
                                                   na_only=('use_verb' not in config))
        print(len(dev_dataloader.ori_verb_indices))
    else:
        dev_dataloader = None

    ##############
    # initialize #
    ##############
    print("Using cuda?: {}".format(use_cuda))

    num_prep = 0
    if 'BiLSTMEmbedder' in config['name']:
        ##################
        # Basic Embedder #
        ##################
        input_layer = ml.BasicWordEmbedSeqsLayer(vecs=vecs, use_cuda=use_cuda)

        model = em.MultiTaskConnotationEmbedder(int(config['h']), input_layer.dim,
                                                dims, 'Emo' in dims,
                                                drop_prob=float(config['dropout']), use_cuda=use_cuda, enc_type=ENC_TYPE,
                                                use_word=USE_WORD,
                                                use_verb=('use_verb' in config))

        if 'use_verb' not in config:
            # initialize loss functions, potentially using weights
            loss_fn_lst = [lf() if args['sampling'] != 'balanced'
                           else lf(weight=dataloader.dim2weight[d]) for d,lf in zip(dims, loss_fn_lst)]
            handler_fn = model_utils.TorchModelHandler

        else:
            # initialize loss functions, potentially using weights
            loss_fn_lst = [lf(reduction='none') if args['sampling'] != 'balanced'
                           else lf(weight=dataloader.dim2weight[d], reduction='none')
                           for d, lf in zip(dims, loss_fn_lst)]
            handler_fn = model_utils.VerbTorchModelHandler


        kwargs = {'model': model, 'input_model': input_layer, 'dataloader': dataloader,
                  'batching_fn': data_utils.prepare_batch_with_reverse,
                  'batching_kwargs': {'num_dims': len(dims),
                                      'use_labels': True}, 'name': name,
                  'loss_function_lst': loss_fn_lst,
                  'weight_lst': weight_lst,
                  'optimizer': optim.Adam(model.parameters(), lr=.001),
                  'setup_fn': model_utils.setup_helper_multitask_conn_embedder,
                  'verb_only': 'verb_only' in config,
                  'l2': L2_FACTOR}

        if 'use_verb' in config:
            if 'verb_only' not in config:
                kwargs['verb_params'] = model.predictor.conn_pred[5:].parameters()
                kwargs['verb_optimizer'] = optim.Adam(chain(model.def_encoder.parameters(), kwargs['verb_params']))

                kwargs['pred_params'] = chain(model.predictor.conn_pred[:5].parameters(),
                                                 model.predictor.conn_pred_emo.parameters())
                kwargs['optimizer'] = optim.Adam(chain(model.def_encoder.parameters(), kwargs['pred_params']))
            else:
                kwargs['verb_optimizer'] = optim.Adam(model.parameters())
                kwargs['verb_params'] = model.predictor.parameters()
        print("NAME: {}".format(name))
        model_handler =handler_fn(dims, use_cuda=use_cuda, **kwargs)

    elif "BiLSTMRelEmbedder" in config['name']:
        #########################
        # Embedder with Related #
        #########################
        input_layer = ml.BasicWordEmbedSeqsLayer(vecs=vecs, use_cuda=use_cuda)

        model = em.MultiTaskConnotationEmbedderWithRelated(int(config['h']), input_layer.dim,
                                                           dims, 'Emo' in dims, drop_prob=float(config['dropout']),
                                                           use_cuda=use_cuda, enc_type=ENC_TYPE, use_word=USE_WORD,
                                                           att_type=SCALE_ATT, use_verb=('use_verb' in config))

        if 'use_verb' not in config:
            # initialize loss functions, potentially using weights
            loss_fn_lst = [lf() if args['sampling'] != 'balanced'
                           else lf(weight=dataloader.dim2weight[d]) for d, lf in zip(dims, loss_fn_lst)]
            handler_fn = model_utils.TorchModelHandler
        else:
            # initialize loss functions, potentially using weights
            loss_fn_lst = [lf(reduction='none') if args['sampling'] != 'balanced'
                           else lf(weight=dataloader.dim2weight[d], reduction='none')
                           for d, lf in zip(dims, loss_fn_lst)]
            handler_fn = model_utils.VerbTorchModelHandler

        kwargs = {'model': model, 'input_model': input_layer, 'dataloader': dataloader,
                  'batching_fn': data_utils.prepare_batch_with_reverse,
                  'batching_kwargs': {'num_dims': len(dims),
                                      'use_related': 'v1',
                                      'use_labels': True},
                  'name': name,
                  'loss_function_lst': loss_fn_lst,
                  'weight_lst': weight_lst,
                  'optimizer': optim.Adam(model.parameters(), lr=.001),
                  'setup_fn': model_utils.setup_helper_multitask_conn_embedder,
                  'verb_only': 'verb_only' in config,
                  'l2': L2_FACTOR,
                  'pred_params': model.predictor.parameters()}

        if 'use_verb' in config:
            if 'verb_only' not in config:
                kwargs['verb_params'] = model.predictor.conn_pred[5:].parameters()
                kwargs['verb_optimizer'] = optim.Adam(chain(model.def_encoder.parameters(), kwargs['verb_params']))

                kwargs['pred_params'] = chain(model.predictor.conn_pred[:5].parameters(),
                                                 model.predictor.conn_pred_emo.parameters())
                kwargs['optimizer'] = optim.Adam(chain(model.def_encoder.parameters(), kwargs['pred_params']))
            else:
                kwargs['verb_optimizer'] = optim.Adam(model.parameters())
                kwargs['verb_params'] = model.predictor.parameters()

        print("NAME: {}".format(name))
        model_handler = handler_fn(dims, use_cuda=use_cuda, **kwargs)
        model_handler.dataloader.data.use_related = True
        dev_dataloader.data.use_related = True

    #######
    # RUN #
    #######
    train(model_handler, int(config['epochs']), dev_data=dev_dataloader, num_prep=num_prep)
