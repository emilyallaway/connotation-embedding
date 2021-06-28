import torch, sys, os, argparse
sys.path.append('./modeling')
import embedding_models as em
import data_utils, model_utils, datasets
import model_layers as ml
import torch.optim as optim
import torch.nn as nn
import json
import pandas as pd
from itertools import chain


SEED = 0
NUM_GPUS = 1
use_cuda = False

ENC_TYPE='lstm'
USE_WORD=True
CONN_LOSS_FACTOR = 0.01
SCALE_ATT='scale'


dim2i = {'Social Val': 0, 'Polite': 1, 'Impact': 2, 'Fact': 3,
         'Sent':4, 'Emo': 5}

WEIGHTS = [.3, .5, .3, 1/6., 1/6.] + \
          [1., 1., 1., 1., 1., 3., 3., 1., 1., 3., 3.] + [3] # Emo is last # W7
CONN_LOSS = nn.CrossEntropyLoss
CONN_EMO_LOSS = nn.MultiLabelSoftMarginLoss

verbna_dim2i = {'Social Val': 0, 'Polite': 1, 'Impact': 2, 'Fact': 3, 'Sent':4,
              'P(wt)': 5, 'P(wa)': 6, 'P(at)': 7, 'E(t)': 8, 'E(a)': 9, 'V(t)': 10, 'V(a)': 11,
              'S(t)': 12, 'S(a)': 13, 'power': 14, 'agency': 15, 'Emo': 16}

NUM_VERB_DIMS = 9 # 12


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


def extract_embeddings(model_handler, data, dev_data, data_name='train'):
    model_handler.get_embeddings(data=dev_data, data_name=data_name)


def save_predictions(model_handler, dev_data, out_name, class_wise=False, is_test=False):
    trn_pred_lst, _ = model_handler.predict(class_wise=class_wise) # predict train
    trn_out, trn_cols = predict_helper(trn_pred_lst, model_handler.dataloader.data)
    pd.DataFrame(trn_out, columns=trn_cols).to_csv(out_name + '-train.csv', index=False)

    if dev_data is not None:
        if is_test:
            dev_name = 'test'
        else:
            dev_name = 'dev'
        print("is test? {}".format(is_test))
        dev_pred_lst, _ = model_handler.predict(data=dev_data, class_wise=class_wise) # predict dev
        dev_out, dev_cols = predict_helper(dev_pred_lst, dev_data.data)
        pd.DataFrame(dev_out, columns=dev_cols).to_csv(out_name + '-{}.csv'.format(dev_name), index=False)


def predict_helper(pred_lst, pred_data):
    #
    # NOTE: we want to make predictions for everything, even non-valid dimensions
    out_data = []
    cols = list(pred_data.data_file.columns)
    for i in pred_data.data_file.index:
        temp = []
        row = pred_data.data_file.iloc[i]
        for c in cols:
            temp.append(row[c])

        for ci in range(len(pred_lst)):
            if isinstance(pred_lst[ci][i], int):
                temp.append(pred_lst[ci][i])
            else:
                temp.append(json.dumps(pred_lst[ci][i].tolist()))
        out_data.append(temp)
    if 'P(wt)' in pred_data.data_file.columns:
        cols += ['pred Social Val', 'pred Polite', 'pred Impact', 'pred Fact', 'pred Sent', 'pred P(wt)'] + \
                ['pred P(wa)', 'pred P(at)', 'pred E(t)', 'pred E(a)', 'pred V(t)', 'predV(a)'] +  \
                ['pred S(t)', 'pred S(a)', 'pred power', 'pred agency', 'pred Emo']
    else:
        cols += ['pred Social Val', 'pred Polite', 'pred Impact', 'pred Fact', 'pred Sent', 'pred Emo']
    return out_data, cols


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--config_file', help='Name of the cofig data file', required=False)
    parser.add_argument('-i', '--trn_data', help='Name of the training data file', required=False)
    parser.add_argument('-d', '--dev_data', help='Name of the dev data file', default=None, required=False)
    parser.add_argument('-c', '--corpora', help='Corpora to evaluate on individually. Separate with ;',
                        default=None, required=False)
    parser.add_argument('-b', '--sampling', help='Whether to use balanced class weights for sampling',
                        default=None, required=False)
    parser.add_argument('-n', '--dims', help='The names of the dims to use', required=False,
                        default='Social Val;Polite;Impact;Fact;Sent;Emo')
    parser.add_argument('-v', '--verb_dims', help='The verb dimensions to use', required=False,
                        default='P(wt);P(wa);P(at);E(t);E(a);V(t);V(a);S(t);S(a);power;agency')
    parser.add_argument('-k', '--ckp_name', help='Checkpoint name', required=False)
    parser.add_argument('-m', '--mode', help='Whether to extract embeds or eval', default='eval',
                        required=False)
    parser.add_argument('-o', '--out_name', help='The name of the file to save predictions to, if used',
                        required=False)
    parser.add_argument('-e', '--embed_name', help='Embedding name', required=False, default='train')
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
            dims = ori_dims[:-1]  # remove Emo, which is LAST
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
    vocab_name = '{}/{}.vocabF.pkl'.format(vec_path, vec_name)
    use_related = args['use_related']
    trn_data = datasets.ConnData(args['trn_data'], vocab_name, pad_val=len(vecs) - 1,
                                 dims=ori_dims, use_related=use_related,
                                 verb_task=('use_verb' in config), verb_dims=verb_dims)
    dev_data = datasets.ConnData(args['dev_data'], vocab_name, pad_val=len(vecs) - 1,
                                 dims=ori_dims, use_related=use_related,
                                 verb_task=('use_verb' in config), verb_dims=verb_dims)

    trn_dataloader = data_utils.POSDataSampler(trn_data, batch_size=int(config['b']),
                                              num_resets=0, shuffle=False, verb_only=('verb_only' in config),
                                               na_only=('use_verb' not in config))
    dev_dataloader = data_utils.POSDataSampler(dev_data, batch_size=int(config['b']),
                                              num_resets=0, shuffle=False, verb_only=('verb_only' in config),
                                               na_only=('use_verb' not in config))

    print("Using cuda?: {}".format(use_cuda))

    if 'BiLSTMEmbedder' in config['name']:
        batch_args = {}
        input_layer = ml.BasicWordEmbedSeqsLayer(vecs=vecs, use_cuda=use_cuda)

        model = em.MultiTaskConnotationEmbedder(int(config['h']), input_layer.dim,
                                                dims, 'Emo' in dims,
                                                drop_prob=float(config['dropout']), use_cuda=use_cuda,
                                                use_word=USE_WORD,
                                                use_verb=('use_verb' in config))

        if 'use_verb' not in config:
            # initialize loss functions, potentially using weights
            loss_fn_lst = []
            handler_fn = model_utils.TorchModelHandler

        else:
            loss_fn_lst = []
            handler_fn = model_utils.VerbTorchModelHandler


        kwargs = {'model': model, 'input_model': input_layer, 'dataloader': trn_dataloader,
                  'batching_fn': data_utils.prepare_batch_with_reverse,
                  'batching_kwargs': {'num_dims': len(dims)}, 'name': name,
                  'loss_function_lst': loss_fn_lst,
                  'weight_lst': weight_lst,
                  'optimizer': optim.Adam(model.parameters(), lr=.001),
                  'setup_fn': model_utils.setup_helper_multitask_conn_embedder,
                  'verb_only': 'verb_only' in config,
                  'l2': 0.}

        if 'use_verb' in config:
            if 'verb_only' not in config:
                kwargs['verb_optimizer'] = optim.Adam(chain(model.def_encoder.parameters(),
                                                      model.predictor.conn_pred[5:].parameters()),
                                                      lr=0.001)
                kwargs['optimizer'] = optim.Adam(chain(model.def_encoder.parameters(),
                                                 model.predictor.conn_pred[:5].parameters(),
                                                 model.predictor.conn_pred_emo.parameters()))
            else:
                kwargs['verb_optimizer'] = optim.Adam(model.predictor.parameters(), lr=0.01)

        if args['mode'] == 'embeds' or args['mode'] == 'predict':
            kwargs['batching_kwargs']['use_labels'] = False

        if args['mode'] == 'predict':
            handler_fn = model_utils.TorchModelHandler
        
        print('Name: {}'.format(name))
        model_handler = handler_fn(dims, use_cuda=use_cuda, **kwargs)

    elif "BiLSTMRelEmbedder" in config['name']:
        input_layer = ml.BasicWordEmbedSeqsLayer(vecs=vecs, use_cuda=use_cuda)

        model = em.MultiTaskConnotationEmbedderWithRelated(int(config['h']), input_layer.dim,
                                                           dims, 'Emo' in dims, drop_prob=float(config['dropout']),
                                                           use_cuda=use_cuda, enc_type=ENC_TYPE, use_word=USE_WORD,
                                                           att_type=SCALE_ATT, use_verb=('use_verb' in config))

    if 'use_verb' not in config:
        # initialize loss functions, potentially using weights
        loss_fn_lst = []
        handler_fn = model_utils.TorchModelHandler
    else:
        # initialize loss functions, potentially using weights
        loss_fn_lst = []
        handler_fn = model_utils.VerbTorchModelHandler

    kwargs = {'model': model, 'input_model': input_layer, 'dataloader': trn_dataloader,
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
              'l2': 1.0,
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

        if args['mode'] == 'embeds' or args['mode'] == 'predict':
            kwargs['batching_kwargs']['use_labels'] = False

        if args['mode'] == 'predict':
            handler_fn = model_utils.TorchModelHandler

    print("NAME: {}".format(name))
    model_handler = handler_fn(dims, use_cuda=use_cuda, **kwargs)
    model_handler.dataloader.data.use_related = True
    dev_dataloader.data.use_related = True

    cname = '../../checkpoints/connotations/ckp-[NAME]-{}.tar'.format(args['ckp_name'])
    model_handler.load(filename=cname)

    if args['mode'] == 'eval':
        eval(model_handler, dev_dataloader, class_wise=False)
    elif args['mode'] == 'embeds':
        model_handler.dataloader.data.use_labels = False
        if args['trn_data'] == args['dev_data']:
            dev_dataloader = None
        extract_embeddings(model_handler, trn_dataloader, dev_dataloader, data_name=args['embed_name'])
    elif args['mode'] == 'predict':
        model_handler.dataloader.data.use_labels = False
        if args['trn_data'] == args['dev_data']:
            dev_dataloader = None
        save_predictions(model_handler, dev_dataloader, args['out_name'], is_test=('test' in args['dev_data']))
