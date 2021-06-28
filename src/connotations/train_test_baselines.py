from modeling.baselines import ConnMaxEntModel, MajorityBaseline
import sys
import argparse
import json
import numpy as np
import pandas as pd
sys.path.append('..')
from processing.postprocessing import load_word_order, find_lst_el

dim2emo = {0: 'anger', 1: 'anticip', 2: 'disgust', 3: 'fear', 4: 'joy',
           5: 'sadness', 6: 'surprise', 7: 'trust'}

EVAL_KEY = 'test'

def eval_only(m, dims, pos, model_dir, data_name, name, vec_name='deps.words'):
    model = m(dimensions=dims, pos=pos, name=name, vec_file=vec_name)
    model.load(model_dir)
    print("{} DATA".format(data_name))
    model.eval(data_name)


def train_and_eval(m, dims, pos, name, output_dir, data_name, outname, vec_name='deps.words'):
    model = m(dimensions=dims, pos=pos, name=name, vec_file=vec_name)
    model.train()
    print("TRAIN DATA")
    print("   {} examples".format(len(model.word2conn['train'])))
    model.eval('train')
    print()
    print("{} DATA".format(EVAL_KEY))
    print("   {} examples".format(len(model.word2conn[EVAL_KEY])))
    model.eval(EVAL_KEY)
    print()

    np.random.seed(0)
    df = pd.read_csv(data_name, keep_default_na=False)
    unk_words = {'N': set(), 'A': set()}
    pred_res = {d: [] for d in dims}
    data = []
    for i in df.index:
        row = df.iloc[i]
        pos = row['POS']
        w = row['word']
        if pos == 'V': continue
        v, c_filtered = model.load_example(w, EVAL_KEY)
        if v is not None:
            p, _ = model.predict(in_data=[v], labels=[c_filtered])
            idx = 0  # find_lst_el(w, wlst)
        else:
            idx = -1
        for d in dims:
            if d != 'Emo':
                if idx != -1:
                    pred_res[d] = int(p[d][0])
                else:
                    unk_words[pos].add(w)
                    pred_res[d] = np.random.randint(0, 3)  # -1
            else:
                ep = [0] * 8
                if idx != -1:
                    for ei in range(8):
                        ep[ei] = int(p['Emo'][ei][0])
                pred_res['Emo'] = json.dumps(ep)
        temp = [w, pos] + [pred_res[d] for d in['Social Val', 'Polite', 'Impact', 'Fact', 'Sent', 'Emo']]
        data.append(temp)

    cols = ['word', 'POS', 'pred Social Val', 'pred Polite', 'pred Impact', 'pred Fact', 'pred Sent', 'pred Emo']
    pd.DataFrame(data, columns=cols).to_csv(outname, index=False)

    # save the model
    model.save(output_dir)


def make_predictions_all(m, dims, model_dir, data_name, name, outpath, vec_name='deps.words'):
    model_N = m(dimensions=dims['N'], pos='N', name=name, vec_file=vec_name, extra_words=None)
    model_N.load(model_dir)
    N_in_data, N_labels, N_wlst = model_N.make_data(data_split='test')#'dev')
    assert len(N_in_data) == len(N_wlst)
    N_pred, _ = model_N.predict(in_data=N_in_data, labels=N_labels)

    model_A = m(dimensions=dims['A'], pos='A', name=name, vec_file=vec_name, extra_words=None)
    model_A.load(model_dir)
    A_in_data, A_labels, A_wlst = model_A.make_data(data_split='test')#'dev')
    assert len(A_in_data) == len(A_wlst)
    A_pred, _ = model_A.predict(in_data=A_in_data, labels=A_labels)

    np.random.seed(0)
    wlst = load_word_order(data_name)
    preds = {get_dim(d): [] for d in set(dims['N']) | set(dims['A']) - {'Emo'}}
    unk_words = {'N': set(), 'A': set()}
    for w,pos in wlst:
        if pos != 'V':
            if pos == 'N':
                p = N_pred
                idx = find_lst_el(w, N_wlst)
                dim_set = set(dims['N']) - {'Emo'}
            else:
                p = A_pred
                idx = find_lst_el(w, A_wlst)
                dim_set = set(dims['A']) - {'Emo'}

            for d in dim_set:
                new_d = get_dim(d)
                if idx != -1:
                    preds[new_d].append(p[d][idx])
                else:
                    unk_words[pos].add(w)
                    preds[new_d].append(-1)
            ep = []
            if idx != -1:
                for ei in range(8):
                    if p['Emo'][ei][idx] == 1:
                        ep.append(dim2emo[ei])
            if len(ep) == 0:
                ep.append('NONE')
            preds['Emo'].append(','.join(set(ep)))
        else: # for verbs
            for d in preds:
                preds[d].append(-1)

    print(unk_words)
    for d in preds:
        outf = open(outpath + '-{}.txt'.format(d), 'w')
        for p in preds[d]:
            outf.write('{}\n'.format(p))

def make_predictions_detailed(m, dims, model_dir, data_name, name, outname, vec_name='deps.words'):
    if 'train' in data_name:
        s = 'train'
    elif 'dev' in data_name:
        s = 'dev'
    else:
        s = 'test'

    model = m(dimensions=dims, pos='O', name=name, vec_file=vec_name, extra_words=None)
    model.load(model_dir)
    # in_data, labels, wlst = model.make_data(data_split=s)
    # preds, _ = model.predict(in_data=in_data, labels=labels)


def get_dim(d):
    if d == 'Social Stat' or d == 'Value':
        return 'Social Val'
    elif d == 'Social Impact':
        return 'Impact'
    else:
        return d


def get_rev_dim(d, pos):
    if d == 'Social Val':
        if pos == 'N':
            return 'Social Stat'
        else:
            return 'Value'
    elif d == 'Impact':
        if pos == 'N':
            return 'Social Impact'
    return d


def save_embeds(model_dir, name, vec_name, input_data, dims, out_name):
    words = set([l.strip() for l in open('/proj/nlp/users/eallaway/stance/connotations/data/conns/lexica/' +
                                     'connotation_frames/mpqa_vocab.txt', 'r').readlines()])

    model = m(dimensions=dims, pos='O', name=name, vec_file=vec_name, extra_words=words)
    model.load(model_dir)

    in_data, _, wlst = model.make_data(data_split=words)
    in_preds, _ = model.predict(in_data=in_data)

    data = []
    for wi, w in enumerate(wlst):
        idx = find_lst_el(w, wlst)
        cvec = [0] * (len(model.dimensions) + (8 if model.emo else 0))
        for d in model.dimensions:
            cidx = model.dim_map[model.pos][d]
            cvec[cidx] = int(in_preds[d][idx])
        if model.emo:
            for i, di in enumerate(model.dim_map[model.pos]['Emo']):
                cvec[di] = int(in_preds['Emo'][i][idx])
        cvec += [0] * 11
        data.append([w, 'O', json.dumps(cvec)])

    new_data = pd.DataFrame(data, columns=['word', 'POS', 'lex_vec'])
    new_data.to_csv(out_name, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='What to do', required=True)
    parser.add_argument('-p', '--pos', help='Part of speech to train and test on', required=False,
                        default='O')
    parser.add_argument('-d', '--dims', help='Dimensions of connotation to use, separated by ;', required=False,
                        default='All')
    parser.add_argument('-v', '--vec_name', help='The name of the word vectors to use',
                        default='deps.words', required=False)
    parser.add_argument('-n', '--name', help='Name of the model, not including dimensions',
                        default='', required=False)
    parser.add_argument('-s', '--data_splt', help='The name of the data split to evaluate on',
                        default='dev', required=False)
    parser.add_argument('-e', '--model_dir', help='The directory where the trained model is saved',
                        default='../../saved_models/', required=False)
    parser.add_argument('-i', '--input_data', help='The input data name')
    parser.add_argument('-o', '--output_name', help='Name of the output file')
    args = vars(parser.parse_args())

    if args['pos'] == 'N':
        if args['dims'] == 'All':
            dims = ['Social Stat', 'Polite', 'Fact', 'Sent', 'Social Impact', 'Emo']
        else:
            dims = args['dims'].split(';')
    elif args['pos'] == 'A':
        if args['dims'] == 'All':
            dims = ['Polite', 'Fact', 'Sent', 'Value', 'Impact', 'Emo']
        else:
            dims = args['dims'].split(';')
    elif args['pos'] == 'O':
        dims = {'N': ['Social Stat', 'Polite', 'Fact', 'Sent', 'Social Impact', 'Emo'],
                'A': ['Polite', 'Fact', 'Sent', 'Value', 'Impact', 'Emo']}
    else:
        print("ERROR: POS was {} but should have been N or A. Exiting.".format(args['pos']))
        sys.exit(1)

    if 'LogReg' in args['name']:
        print("Training logreg")
        m = ConnMaxEntModel
    elif 'Maj' in args['name']:
        print("training Maj")
        m = MajorityBaseline
    else:
        print("ERROR: model name was {} but should have had LogReg or Maj.Exiting".format(args['name']))
        sys.exit(1)

    if args['mode'] == '1':
        print("Training and evaluating")
        dims = ['Social Val', 'Polite', 'Impact', 'Fact', 'Sent', 'Emo']
        train_and_eval(m, dims, pos, args['name'], vec_name=args['vec_name'],
                       output_dir=args['model_dir'], data_name=args['input_data'], outname=args['output_name'])
    elif args['mode'] == '2':
        print("Evaluating only")
        eval_only(m, dims, args['pos'], data_name=args['data_splt'], model_dir=args['model_dir'], name=args['name'],
                  vec_name=args['vec_name'])
    elif args['mode'] == '3':
        print("Making predictions")
        make_predictions_all(m, dims, data_name=args['input_data'], model_dir=args['model_dir'], name=args['name'],
                             outpath=args['output_name'], vec_name=args['vec_name'])
    elif args['mode'] == '4':
        print("Making predictions")
        dims = ['Social Val', 'Polite', 'Impact', 'Fact', 'Sent', 'Emo']
        make_predictions_detailed(m, dims, data_name=args['input_data'], model_dir=args['model_dir'], name=args['name'],
                             outname=args['output_name'], vec_name=args['vec_name'])

    elif args['mode'] == '5':
        print("Saving lexicon vectors")
        dims = ['Social Val', 'Polite', 'Impact', 'Fact', 'Sent', 'Emo']
        save_embeds(model_dir=args['model_dir'], name=args['name'], vec_name=args['vec_name'],
                    input_data=args['input_data'], dims=dims, out_name=args['output_name'])

    else:
        print("ERROR: doing nothing")