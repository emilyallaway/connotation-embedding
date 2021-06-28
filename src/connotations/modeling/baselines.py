from sklearn.linear_model import LogisticRegression
import os
import pickle
import json
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from data_processing.dim_mapping_v2 import dim2emo
import utils

VEC_PATH = '../../../resources/'
CONN_FILE = '../../data/conn_input'
CONN_MODEL_NAME='BiLSTMRelEmbedderVERB-cn-w7-s4-42w-norm-learn-20r-balanced'


class ConnMaxEntModel:
    def __init__(self, dimensions, pos, full_path=CONN_FILE, extra_words=None,
                 vec_file='deps.words',use_all=False, name=''):
        self.dim_map = {'O':
                            {'Social Val': 0, 'Polite': 1, 'Impact': 2, 'Fact': 3, 'Sent': 4,
                        'Emo': [5, 6, 7, 8, 9, 10, 11, 12]}}
        self.use_partial = use_all
        self.pos = pos
        self.emo = 'Emo' in dimensions
        self.dimensions = list(set(dimensions) - {'Emo'}) # can specify adjectives and nouns, does NOT include Emo

        # initialized models
        self.models = [[] for _ in range(len(self.dim_map[self.pos]) + 7)]  # +7 is for emotions
        for d in dimensions:
            if d != 'Emo':
                self.models[self.dim_map[self.pos][d]] =LogisticRegression(class_weight='balanced',solver='lbfgs',
                                                                           multi_class='multinomial',
                                                                                        max_iter=5000)
        if self.emo:
            for i in self.dim_map[self.pos]['Emo']:
                self.models[i] =LogisticRegression(class_weight='balanced', solver='lbfgs',
                                                    max_iter=5000)

        self.missing_vecs = {'train': set(), 'dev': set(), 'test': set()}

        self.vec_name = vec_file
        self.path = full_path
        nstr = ['_'.join(d.split(' ')) for d in dimensions]
        self.name = (name + '-' + '.'.join(nstr) + '/').strip('-')  # use original parameter so include Emo

        self.load_data(extra_words)


    def load_helper(self, s, allwords, w2c):
        df = pd.read_csv(os.path.join(self.path) + '-{}.csv'.format(s), keep_default_na=False)
        for i in df.index:
            row = df.iloc[i]
            if row['POS'] != "N" and row['POS'] != 'A': continue

            w = row['word'].strip(',').replace('/', ' ')

            conn_lst = [0. for _ in range(len(self.models))]
            for d in self.dim_map[self.pos]:
                if d != 'Emo':
                    conn_lst[self.dim_map[self.pos][d]] = row[d]  # conn[d]
                else:  # d == Emo
                    conn= json.loads(row[d])
                    for ei in dim2emo:
                        conn_lst[self.dim_map[self.pos][d][0] + ei] = conn[ei]  # conn[d][ei]
            w2c[w] = conn_lst
            allwords.add(w)
            for wi in w.split():
                allwords.add(wi)

    def load_data(self, extra_words):
        print("...loading data")
        self.word2conn = {'train': {}, 'dev': {}, 'test': {}}
        allwords = set()
        for s in self.word2conn:
            self.load_helper(s, allwords, self.word2conn[s])
        if extra_words is not None:
            keep_words = allwords | extra_words
        else:
            keep_words = allwords
        # get and filter word vectors
        vec_path = os.path.join(VEC_PATH, self.vec_name)
        self.word2idx, self.vecs = utils.load_vectors(vec_path, keep_words=keep_words)

    def make_data(self, data_split='train'):
        print("...making data")
        word_lst = []
        input_data = []
        labels = []
        if isinstance(data_split, str):
            for w in self.word2conn[data_split]:
                v, c_filtered = self.load_example(w, data_split)
                if v is not None:
                    input_data.append(v)
                    labels.append(c_filtered)
            labels = np.array(labels)
        else:
            for w in data_split:
                if w not in self.word2idx: continue
                input_data.append(self.vecs[self.word2idx[w]])
                word_lst.append(w)
        return np.array(input_data), labels, word_lst

    def load_example(self, w, s):
        v = None
        if w not in self.word2idx:
            if ' ' not in w:
                print(w)
                pass
            else:
                vlst = []
                for wi in w.split():
                    if wi in self.word2idx:
                        vlst.append(self.vecs[self.word2idx[wi]])
                if len(vlst) > 0:
                    v = np.mean(vlst, axis=0)
                else:
                    print(w)
                    pass
        else:
            v = self.vecs[self.word2idx[w]]
        c = self.word2conn[s][w]
        c_filtered = [0 for _ in range(len(self.models))]
        for d, i in self.dim_map[self.pos].items():
            if d != 'Emo':
                c_filtered[i] = 0 if c[i] < 0 else 1 if c[i] > 0 else 2
        if self.emo:
            for ei in self.dim_map[self.pos]['Emo']:
                c_filtered[ei] = int(c[ei])
        return v, c_filtered


    def train(self):
        trn_data, trn_labels, _ = self.make_data()
        print("training models:")
        # j = 0
        for d, i in self.dim_map[self.pos].items():
            if d not in self.dimensions: continue
            print("    ... {} model".format(d))
            if d != 'Emo':
                self.models[self.dim_map[self.pos][d]].fit(trn_data, trn_labels[:, i])
        if self.emo:
            for di in self.dim_map[self.pos]['Emo']:
                try:
                    self.models[di].fit(trn_data, trn_labels[:, di])
                except:
                    print("ERROR: revert to Majority Emo-{}".format(di))

    def predict(self, data_split='train', in_data=None, labels=None):
        if in_data is None:
            in_data, labels, _ = self.make_data(data_split)
        preds = {}
        for d, i in self.dim_map[self.pos].items():
            if d not in self.dimensions: continue
            preds[d] = self.models[self.dim_map[self.pos][d]].predict(in_data)
        if self.emo:
            emo_preds = []
            for di in self.dim_map[self.pos]['Emo']:
                emo_preds.append(self.models[di].predict(in_data))
            preds['Emo'] = emo_preds
        return preds, labels

    def eval(self, data_split='train'):
        preds, labels = self.predict(data_split)
        print("evaluating models: ")
        for d,i in self.dim_map[self.pos].items():
            if d not in self.dimensions: continue
            fscore = f1_score(labels[:, i], preds[d], average='macro')
            prec = precision_score(labels[:,i], preds[d], average='macro')
            rec = recall_score(labels[:, i], preds[d], average='macro')
            acc = accuracy_score(labels[:, i], preds[d])
            print("   {}: f1={:.4f}, prec={:.4f}, rec={:.4f}, acc={:.4f}".format(d, fscore, prec, rec, acc))
        if self.emo:
            tot_f, tot_p, tot_r, tot_a = 0., 0., 0., 0.
            for i, di in enumerate(self.dim_map[self.pos]['Emo']):
                fscore = f1_score(labels[:, di], preds['Emo'][i], average='macro')
                tot_f += fscore
                prec = precision_score(labels[:, di], preds['Emo'][i], average='macro')
                tot_p += prec
                rec = recall_score(labels[:, di], preds['Emo'][i], average='macro')
                tot_r += rec
                acc = accuracy_score(labels[:, di], preds['Emo'][i])
                tot_a += acc
                print("   Emo-{}: f1={:.4f}, prec={:.4f}, rec={:.4f}, acc={:.4f}".format(dim2emo[i], fscore,
                                                                                         prec, rec, acc))
            print("   Emo: avg-f1={:.4f}, avg-p={:.4f}, avg-r={:.4f}, avg-a={:.4f}".format(tot_f/8, tot_p/8,
                                                                                           tot_r/8, tot_a/8))

    def save(self, output_dir):
        model_path = os.path.join(output_dir, self.name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        for d in self.dimensions:
            n = '_'.join(d.split(' '))
            pickle.dump(self.models[self.dim_map[self.pos][d]],
                        open(os.path.join(model_path, '{}-{}.pkl'.format(self.vec_name, n)), 'wb'))
        if self.emo:
            emo_lst = []
            for di in self.dim_map[self.pos]['Emo']:
                emo_lst.append(self.models[di])
            pickle.dump(emo_lst, open(os.path.join(model_path, '{}-Emo-lst.pkl'.format(self.vec_name)), 'wb'))

    def load(self, dir):
        model_path = os.path.join(dir, self.name)

        for d in self.dimensions:
            n = '_'.join(d.split(' '))
            self.models[self.dim_map[self.pos][d]] = pickle.load(open(os.path.join(model_path,
                                                                                   '{}-{}.pkl'.format(self.vec_name, n)),
                                                                      'rb'))
        if self.emo:
            emo_lst = pickle.load(open(os.path.join(model_path, '{}-Emo-lst.pkl'.format(self.vec_name)), 'rb'))
            for i, m in enumerate(emo_lst):
                self.models[len(self.dim_map[self.pos]) - 1 + i] = m


class MajorityBaseline:
    def __init__(self, dimensions, pos, full_path=CONN_FILE,vec_file='deps.words',use_all=False, name=''):
        self.dim_map = {'N': {'Social Stat': 0, 'Polite': 1, 'Fact': 2, 'Sent': 3, 'Social Impact': 4,
                              'Emo': [5, 6, 7, 8, 9, 10, 11, 12]},
                        'A': {'Polite': 0, 'Fact': 1, 'Sent': 2, 'Value': 3, 'Impact': 4,
                              'Emo': [5, 6, 7, 8, 9, 10, 11, 12]},
                        'V': {'Polite': 0, 'Fact': 1, 'Sent': 2, 'Value': 3, 'Impact': 4, 'Social Stat': 5,
                              'Emo': [6, 7, 8, 9, 10, 11, 12, 13]}}
        self.use_partial = use_all
        self.pos = pos
        self.emo = 'Emo' in dimensions
        self.dimensions = list(set(dimensions) - {'Emo'}) # can specify adjectives and nouns, does NOT include Emo

        # init models
        self.models = [0 for _ in range(len(self.dim_map[self.pos]) + 7)]  # +7 is for emotions

        self.vec_name = vec_file
        self.path = full_path
        self.__load_data()
        nstr = ['_'.join(d.split(' ')) for d in dimensions]
        self.name = (name + '-' + '.'.join(nstr) + '/').strip('-')  # use original parameter so include Emo

    def fit(self):
        dim2c = dict()
        for d in self.dim_map[self.pos]:
            dim2c[d] = dict()
        if self.emo:
            dim2c['Emo'] = dict()
            for i in range(8):
                dim2c['Emo'][i] = dict()

        df = pd.read_csv(os.path.join(self.path), keep_default_na=False)
        for i in df.index:
            row = df.iloc[i]
            if (not self.use_partial and row['partial?'] != 0) \
                    or row['train/dev/test'] != 'train' or row['POS'] != self.pos:
                continue
            conn = json.loads(row['conn'])
            for c in conn:
                if c != 'Emo':
                    if conn[c] > 0:
                        dim2c[c]['+'] = dim2c[c].get('+', 0) + 1
                    elif conn[c] < 0:
                        dim2c[c]['-'] = dim2c[c].get('-', 0) + 1
                    else:
                        dim2c[c]['0'] = dim2c[c].get('0', 0) + 1
                else:
                    for i, ei in enumerate(conn['Emo']):
                        if ei > 0:
                            dim2c['Emo'][i]['0'] = dim2c['Emo'][i].get('0', 0) + 1
                        else:
                            dim2c['Emo'][i]['-'] = dim2c['Emo'][i].get('-', 0) + 1

        def find_max(di, d_info):
            mk = ''
            mv = 0
            for k in d_info:
                if d_info[k] > mv:
                    mk = k
                    mv = d_info[k]
            print(di, mv, d_info)
            if mk == '+':
                self.models[di] = 2
            elif mk == '-':
                self.models[di] = 0
            else:
                self.models[di] = 1

        for d in dim2c:
            if d != 'Emo':
                find_max(di=self.dim_map[self.pos][d], d_info=dim2c[d])
            else:
                for i in dim2c[d]:
                    find_max(di=self.dim_map[self.pos][d][i],
                             d_info=dim2c[d][i])

    def __load_data(self):
        print("...loading data")
        self.word2conn = {'train': {}, 'dev': {}, 'test': {}}

        df = pd.read_csv(os.path.join(self.path), keep_default_na=False)
        for i in df.index:
            row = df.iloc[i]
            if row['POS'] != self.pos or (not self.use_partial and row['partial?'] != 0): continue

            w = row['word'].lower().strip(',').replace('/', ' ') # NEW
            conn = json.loads(row['conn'])
            conn_lst = [0. for _ in range(len(self.models))]
            for d,i in self.dim_map[self.pos].items():
                if d != 'Emo':
                    conn_lst[i] = conn[d]
                else: # d == Emo
                    for ei in dim2emo:
                        conn_lst[self.dim_map[self.pos][d][0] + ei] = conn[d][ei]

            self.word2conn[row['train/dev/test']][w] = conn_lst

    def __make_data(self, data_split='train'):
        print("...making data")
        input_data = []
        labels = []
        for w in self.word2conn[data_split]:
            input_data.append(w)
            c = self.word2conn[data_split][w]
            c_filtered = [2 for _ in range(len(self.models))]
            for d, i in self.dim_map[self.pos].items():
                if d != 'Emo':
                    c_filtered[i] = utils.make_label(c[i])
            if self.emo:
                for ei in self.dim_map[self.pos]['Emo']:
                    c_filtered[ei] = utils.make_label(c[ei]) - 1
            labels.append(c_filtered)
        return input_data, np.array(labels)

    def train(self):
        print("training models:")
        self.fit()

    def predict(self, data_split='train', in_data=None, labels=None):
        if in_data is None:
            in_data, labels = self.__make_data(data_split)
        print("make model predictions:")
        preds = {}
        for d,i in self.dim_map[self.pos].items():
            if d not in self.dimensions: continue
            print("    ... {} model".format(d))
            preds[d] = [self.models[self.dim_map[self.pos][d]] for _ in range(len(in_data))]
        if self.emo:
            emo_preds = []
            for ei in self.dim_map[self.pos]['Emo']:
                emo_preds.append([self.models[ei] for _ in range(len(in_data))])
            preds['Emo'] = emo_preds
        return preds, labels

    def eval(self, data_split='train'):
        preds, labels = self.predict(data_split)
        print("evaluating models: ")
        for d,i in self.dim_map[self.pos].items():
            if d not in self.dimensions: continue
            fscore = f1_score(labels[:, i], preds[d], average='macro')
            prec = precision_score(labels[:,i], preds[d], average='macro')
            rec = recall_score(labels[:, i], preds[d], average='macro')
            acc = accuracy_score(labels[:, i], preds[d])
            print(str(i) + "   {} ({}): f1={:.4f}, prec={:.4f}, rec={:.4f}, acc={:.4f}".format(d,
                                                                                          self.models[self.dim_map[self.pos][d]],
                                                                                          fscore, prec, rec, acc))
        if self.emo:
            tot_f, tot_p, tot_r, tot_a = 0., 0., 0., 0.
            for i, di in enumerate(self.dim_map[self.pos]['Emo']):
                fscore = f1_score(labels[:, di], preds['Emo'][i], average='macro')
                tot_f += fscore
                prec = precision_score(labels[:, di], preds['Emo'][i], average='macro')
                tot_p += prec
                rec = recall_score(labels[:, di], preds['Emo'][i], average='macro')
                tot_r += rec
                acc = accuracy_score(labels[:, di], preds['Emo'][i])
                tot_a += acc
                print("   Emo-{}: f1={:.4f}, prec={:.4f}, rec={:.4f}, acc={:.4f}".format(dim2emo[i], fscore,
                                                                                         prec, rec, acc))
            print("   Emo: avg-f1={:.4f}, avg-p={:.4f}, avg-r={:.4f}, avg-a={:.4f}".format(tot_f / 8, tot_p / 8,
                                                                                           tot_r / 8, tot_a / 8))
    def save(self, output_dir):
        model_path = os.path.join(output_dir, self.name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        for d in self.dimensions:
            n = '_'.join(d.split(' '))
            pickle.dump(self.models[self.dim_map[self.pos][d]],
                        open(os.path.join(model_path, '{}.pkl'.format(n)), 'wb'))
        if self.emo:
            emo_lst = []
            for di in self.dim_map[self.pos]['Emo']:
                emo_lst.append(self.models[di])
            pickle.dump(emo_lst, open(os.path.join(model_path, 'Emo-lst.pkl'), 'wb'))

    def load(self, dir):
        model_path = os.path.join(dir, self.name)

        for d in self.dimensions:
            n = '_'.join(d.split(' '))
            self.models[self.dim_map[self.pos][d]] = pickle.load(open(os.path.join(model_path,
                                                                                   '{}.pkl'.format(n)),
                                                                      'rb'))
        if self.emo:
            emo_lst = pickle.load(open(os.path.join(model_path, 'Emo-lst.pkl'), 'rb'))
            for i, m in enumerate(emo_lst):
                self.models[len(self.dim_map[self.pos]) - 1 + i] = m