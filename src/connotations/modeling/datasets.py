import torch, pickle, json
from torch.utils.data import Dataset
import pandas as pd
from functools import reduce
import numpy as np


class ConnData(Dataset):
    '''
    Holds the stance dataset.
    '''
    def __init__(self, data_name, vocab_name, name='noun_adj_full-only', corpus=None,
                 max_sen_len=5, max_tok_len=42, max_word_len=1, max_rel_words=20, keep_sen=False,
                 pad_val=0, dims=['Social Val', 'Polite', 'Impact', 'Fact', 'Sent', 'Emo'],
                 use_related=False,
                 dict_task=False,
                 use_labels=True,
                 verb_task=False,
                 verb_dims=['P(wt)', 'P(wa)', 'P(at)', 'E(t)', 'E(a)', 'V(t)', 'V(a)',
                'S(t)', 'S(a)', 'P(rt)', 'P(ra)', 'P(ta)', 'power', 'agency']):
        self.data_file = pd.read_csv(data_name, keep_default_na=False)
        self.word2i = pickle.load(open(vocab_name, 'rb'))
        self.name = name
        self.max_sen_len = max_sen_len
        self.max_tok_len = max_tok_len
        self.max_word_len = max_word_len
        self.keep_sen = keep_sen
        self.pad_value = pad_val
        self.dims = dims
        self.use_related = use_related
        self.max_rel_words = max_rel_words
        self.dict_task = dict_task
        self.use_labels = use_labels

        self.verb_task = verb_task
        self.verb_dims = verb_dims

        np.random.seed(0)

        if corpus is not None:
            # filter the corpus if requested
            self.data_file = self.data_file.loc[self.data_file['src'] == corpus]


    def __reg_label_helper(self, v):
        if float(v) < 0:
            return 0
        elif float(v) > 0:
            return 1
        else:
            return 2 # Neutral, DIFFERENT from REGULAR connotation stuff

    def make_labels(self, row):
        l = []
        for d in self.dims:
            if isinstance(row[d], str):
                l.append(json.loads(row[d]))
            else:
                l.append(self.__reg_label_helper(row[d]))
        return l

    def __verb_label_helper(self, v):
        if float(v) == -1:
            return 0
        elif float(v) == 1:
            return 1
        elif float(v) == 0:
            return 2
        else:
            return 3  # row[d] == 3

    def get_class(self, v, d):
        if d in self.dims:
            return self.__reg_label_helper(v)
        else:
            return self.__verb_label_helper(v)

    def make_labels_verb(self, row):
        l = []
        for d in self.verb_dims:
            l.append(self.__verb_label_helper(row[d]))
        return l

    def make_labels_both(self, row):
        l = []
        for d in self.dims:
            if d == 'Emo': continue
            l.append(self.__reg_label_helper(row[d]))
        for d in self.verb_dims:
            l.append(self.__verb_label_helper(row[d]))
        if 'Emo' in self.dims:
            l.append(json.loads(row['Emo'])) # emo is last label
        return l

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx, corpus=None):
        row = self.data_file.iloc[idx]

        if corpus is not None and row['src'] != corpus:
            # check if the row is from the corpus and return None if not
            return None

        # load word
        ori_word = row['word'].strip(',').replace('/', ' ').split()

        # load definition list
        def_lst = json.loads(row['def_lst'])

        def get_index(word):
            return self.word2i[word] if word in self.word2i else len(self.word2i)

        # index def text & word
        def_lst = [[get_index(w) for w in s] for s in def_lst]
        word = [get_index('_'.join(ori_word))]

        # truncate def text
        if self.keep_sen:
            def_lst = def_lst[:self.max_sen_len]
            def_lst = [d[:self.max_tok_len] for d in def_lst]  # truncates the length of each sentence, by # tokens
            def_lens = [len(d) for d in def_lst]  # compute lens (before padding)
        else:
            def_lst = reduce(lambda x, y: x + y, def_lst)
            def_lst = def_lst[:self.max_tok_len]
            def_lens = len(def_lst)  # compute combined text len

        # pad def text
        if self.keep_sen:
            for d in def_lst:
                while len(d) < self.max_tok_len:
                    d.append(self.pad_value)
            while len(def_lst) < self.max_sen_len:
                def_lens.append(1)
        else:
            while len(def_lst) < self.max_tok_len:
                def_lst.append(self.pad_value)

        # compute word len
        word_lens = len(word)  # get len (before padding)

        # process related words
        if self.use_related:
            rel_lst = json.loads(row['rel_lst'])[: self.max_rel_words]
            rel_lst = [get_index(w) for w in rel_lst]
            while len(rel_lst) < self.max_rel_words:
                rel_lst.append(self.pad_value)
            num_rel = len(rel_lst)
        else:
            rel_lst = []
            num_rel = 1.

        # make labels
        if self.use_labels:
            if self.verb_task:
                l = self.make_labels_both(row)
            else:
                l = self.make_labels(row)
        else:
            l = []

        if self.dict_task:
            l.append(word)

        sample = {'def_lst': def_lst, 'word': word, 'labels': l,
                  'def_l': def_lens, 'word_l': word_lens,
                  'ori_word': ' '.join(ori_word), 'pos': row['POS'],
                  'rel_lst': rel_lst, 'rel_l': num_rel}

        if self.verb_task and self.use_labels:
            sample['loss_mask'] = json.loads(row['mask'])

        return sample
