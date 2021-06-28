import torch, os, sys, argparse
sys.path.append('./modeling')
import baseline_models as bm
import data_utils, model_utils, datasets
import input_models as im
import torch.nn as nn
import torch.optim as optim
import pandas as pd

LOCAL = False
VECTOR_NAME = 'glove.6B.100d'
SEED = 0
NUM_GPUS = 1
use_cuda = torch.cuda.is_available()

def eval(model_handler, dev_data, class_wise=False):
    '''
    Evaluates the given model on the given data, by computing
    macro-averaged F1, precision, and recall scores. Can also
    compute class-wise scores. Prints the resulting scores
    :param class_wise: whether to return class-wise scores. Default(False):
                        does not return class-wise scores.
    :return: a dictionary from score names to the score values.
    '''
    model_handler.eval_and_print(data_name='TRAIN', class_wise=class_wise)

    model_handler.eval_and_print(data=dev_data, data_name='DEV',
                                 class_wise=class_wise)


def save_predictions(model_handler, dev_data, out_name, is_test=False):
    trn_preds, _, _ = model_handler.predict()
    dev_preds, _, _ = model_handler.predict(data=dev_data)
    if is_test:
        dev_name = 'test'
    else:
        dev_name = 'dev'

    predict_helper(trn_preds, model_handler.dataloader.data).to_csv(out_name + '-train.csv', index=False)
    predict_helper(dev_preds, dev_data.data).to_csv(out_name + '-{}.csv'.format(dev_name), index=False)


def predict_helper(pred_lst, pred_data):
    out_data = []
    cols = list(pred_data.data_file.columns)
    for i in pred_data.data_file.index:
        row = pred_data.data_file.iloc[i]
        temp = [row[c] for c in cols]
        temp.append(pred_lst[i])
        out_data.append(temp)
    cols += ['pred label']
    return pd.DataFrame(out_data, columns=cols)




if __name__ == '__main__':
    '''
    first arg: config file name
    second arg: data file name
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--config_file', help='Name of the cofig data file', required=False)
    parser.add_argument('-i', '--trn_data', help='Name of the training data file', required=False)
    parser.add_argument('-d', '--dev_data', help='Name of the dev data file', default=None, required=False)
    parser.add_argument('-c', '--corpora', help='Corpora to evaluate on individually. Separate with ;',
                        default=None, required=False)
    parser.add_argument('-k', '--ckp_name', help='Checkpoint name', required=False)
    parser.add_argument('-m', '--mode', help='What to do', required=True)
    parser.add_argument('-n', '--name', help='something to add to the saved model name',
                        required=False, default='')
    parser.add_argument('-o', '--out', help='Ouput file name', default='')
    args = vars(parser.parse_args())

    torch.manual_seed(SEED)

    with open(args['config_file'], 'r') as f:
        config = dict()
        for l in f.readlines(): config[l.strip().split(":")[0]] = l.strip().split(":")[1]

    # load vectors
    vec_path = '../../resources/stance'

    vec_name = config['vec_name']
    vec_dim = int(config['vec_dim'])

    vecs = data_utils.load_vectors('{}/{}.vectorsF.npy'.format(vec_path, vec_name),
                                   dim=vec_dim, seed=SEED)
    if 'conn_vec_name' in config:
        pos2conn_vecs = dict()
        for d,t in zip(config['conn_vec_dims'].split(','), config['conn_vec_tags'].split(',')):
            conn_vec_path = '{}/{}-{}_vectorsF.npy'.format(vec_path, config['conn_vec_name'], t)
            conn_vecs = data_utils.load_vectors(conn_vec_path, dim=int(d), unk_rand=False)
            pos2conn_vecs[t] = conn_vecs


    # load training and dev data
    vocab_name = '{}/{}.vocabF.pkl'.format(vec_path, vec_name)
    if 'conn_vec_name' not in config:
        trn_data = datasets.StanceData(args['trn_data'], vocab_name,
                                       pad_val=len(vecs) - 1)
        dev_data = datasets.StanceData(args['dev_data'], vocab_name,
                                       pad_val=len(vecs) - 1)
    else:
        trn_data = datasets.StanceDataPos(args['trn_data'], vocab_name, pad_val=len(vecs) - 1,
                                      pos_lst=config['conn_vec_tags'].split(','))
        dev_data = datasets.StanceDataPos(args['dev_data'], vocab_name, pad_val=len(vecs) - 1,
                                          pos_lst=config['conn_vec_tags'].split(','))

    trn_dataloader = data_utils.DataSampler(trn_data, batch_size=int(config['b']),
                                            shuffle=True)
    dev_dataloader = data_utils.DataSampler(dev_data, batch_size=int(config['b']),
                                            shuffle=False)

    print("Using cuda?: {}".format(use_cuda))


    if 'BiCondLSTM' in config['name']:
        batch_args = {}
        input_layer = im.BasicWordEmbedSeqsLayer_WithRev(vecs=vecs, use_cuda=use_cuda)

        model = bm.BiCondLSTMModel(int(config['h']), input_layer.dim,
                                   drop_prob=float(config['dropout']), use_cuda=use_cuda)

        kwargs = {'model': model, 'embed_model': input_layer, 'dataloader': trn_dataloader,
                  'batching_fn': data_utils.prepare_batch_with_reverse,
                  'batching_kwargs': batch_args, 'name': config['name'],
                  'loss_function': nn.CrossEntropyLoss(),
                  'optimizer': optim.Adam(model.parameters(), lr=.001),
                  'setup_fn': model_utils.setup_helper_bicond}

        model_handler = model_utils.TorchModelHandler(use_cuda=use_cuda, **kwargs)

    elif 'BiCondAttention' in config['name']:
        batch_args = {'use_conn': True}
        input_layer = im.ConnVecsLayerSeparate(vecs, pos2conn_vecs)

        nl = 3
        model = bm.BiCondLSTMAttentionModel(int(config['h']), input_layer.dim,
                                            conn_dim=input_layer.conn_dim,
                                            drop_prob=float(config['dropout']), use_cuda=use_cuda,
                                            num_labels=nl,
                                            att_type=config['att'])

        kwargs = {'model': model, 'embed_model': input_layer, 'dataloader': trn_dataloader,
                  'batching_fn': data_utils.prepare_batch_with_reverse,
                  'batching_kwargs': batch_args, 'name': config['name'] + args['name'],
                  'loss_function': nn.CrossEntropyLoss(),
                  'optimizer': optim.Adam(model.parameters()),
                  'setup_fn': model_utils.setup_helper_bicond_connvecs}

        model_handler = model_utils.TorchModelHandler(use_cuda=use_cuda, **kwargs)

    cname = '../../checkpoints/ckp-[NAME]-{}.tar'.format(args['ckp_name'])
    model_handler.load(filename=cname)

    if args['mode'] == 'eval':
        eval(model_handler, dev_dataloader, class_wise=True)
    elif args['mode'] == 'predict':
        save_predictions(model_handler, dev_dataloader, out_name=args['out'], is_test=('test' in args['dev_data']))