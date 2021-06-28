import torch, pickle, json
from torch.utils.data import Dataset
import pandas as pd
from functools import reduce
from IPython import embed
import numpy as np


class StanceData(Dataset):
    '''
    Holds the stance dataset.
    '''
    def __init__(self, data_name, vocab_name, name='CFpersp-train-full', corpus=None,
                 max_sen_len=7, max_tok_len=35, max_top_len=5, binary=False, keep_sen=False,
                 pad_val=0):
        self.data_file = pd.read_csv(data_name)
        self.word2i = pickle.load(open(vocab_name, 'rb'))
        self.name = name
        self.max_sen_len = max_sen_len
        self.max_tok_len = max_tok_len
        self.max_top_len = max_top_len
        self.binary = binary
        self.keep_sen = keep_sen
        self.pad_value = pad_val

        if corpus is not None:
            # filter the corpus if requested
            self.data_file = self.data_file.loc[self.data_file['src'] == corpus]

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx, corpus=None):
        row = self.data_file.iloc[idx]

        if corpus is not None and row['src'] != corpus:
            # check if the row is from the corpus and return None if not
            return None

        # load text
        ori_text = json.loads(row['text'])

        # load topic
        ori_topic = json.loads(row['topic'])

        def get_index(word):
            return self.word2i[word] if word in self.word2i else len(self.word2i)

        # index text & topic
        text = [[get_index(w) for w in s] for s in ori_text]  # [get_index(w) for w in text]
        topic = [get_index(w) for w in ori_topic][:self.max_top_len]

        # truncate text
        if self.keep_sen:
            text = text[:self.max_sen_len]
            text = [t[:self.max_tok_len] for t in text]  # truncates the length of each sentence, by # tokens
            text_lens = [len(t) for t in text]  # compute lens (before padding)
        else:
            text = reduce(lambda x, y: x + y, text)
            text = text[:self.max_tok_len]
            text_lens = len(text)  # compute combined text len

        # pad text
        if self.keep_sen:
            for t in text:
                while len(t) < self.max_tok_len:
                    t.append(self.pad_value)
            while len(text_lens) < self.max_sen_len:
                text_lens.append(1)
        else:
            while len(text) < self.max_tok_len:
                text.append(self.pad_value)

        # compute topic len
        topic_lens = len(topic)  # get len (before padding)

        # pad topic
        while len(topic) < self.max_top_len:
            topic.append(self.pad_value)

        if self.binary:
            if float(row['label']) == 0:
                l = 0
            else:
                l = 1
        else:
            if float(row['label']) == 0:
                l = 0
            elif float(row['label']) == 1:
                l = 1
            else:
                l = 2

        sample = {'text': text, 'topic': topic, 'label': l,
                  'txt_l': text_lens, 'top_l': topic_lens,
                  'ori_topic':  ' '.join(ori_topic),
                  'ori_text': ' '.join([' '.join(ti) for ti in ori_text])}
        return sample


class StanceDataPos(Dataset):

    '''
    Holds the stance dataset where the text is also returned with only certain POS kept.
    '''
    def __init__(self, data_name, vocab_name, name='CFpersp-train-full', corpus=None,
                 pos_lst=['N', 'V', 'A', 'LY', 'O'], max_sen_len=7, max_tok_len=35,
                 max_top_len=5,
                 binary=False, keep_sen=False, pad_val=0, truncate_data=None):
        self.data_file = pd.read_csv(data_name)
        self.word2i = pickle.load(open(vocab_name, 'rb'))
        self.name = name
        self.pos_lst = pos_lst
        self.max_sen_len = max_sen_len
        self.max_tok_len = max_tok_len
        self.max_top_len = max_top_len
        self.binary = binary
        self.keep_sen = keep_sen
        self.pad_value = pad_val
        self.VERB_TAGS = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VH', 'VHD', 'VHG', 'VHN', 'VHP', 'VHZ',
                     'VV', 'VVD', 'VVG', 'VVN', 'VVP', 'VVZ'}
        self.NOUN_TAGS = {'NN', 'NNS', 'NNSZ', 'NNZ', 'NP', 'NPS', 'NPSZ', 'NPZ'}
        self.ADJ_TAGS = {'JJ', 'JJR', 'JJS'}
        self.ADV_TAGS = {'RB', 'RBR', 'RBS'}

        if corpus is not None:
            # filter the corpus if requested
            self.data_file = self.data_file.loc[self.data_file['src'] == corpus]
        if truncate_data is not None:
            self.trim_data(num_examples=truncate_data)


    def trim_data(self, num_examples=2000):
        t2i = dict()
        for i in self.data_file.index:
            t = self.data_file.iloc[i]['topic']
            t2i[t] = t2i.get(t, [])
            t2i[t].append(i)
        np.random.seed(0)
        allidxs = []
        for t in t2i:
            idxs = t2i[t]
            np.random.shuffle(idxs)
            allidxs += idxs[: num_examples]
        print("trimming data, original lenght: {}".format(len(self.data_file)))
        self.data_file = self.data_file.iloc[allidxs]
        print("new length: {}".format(len(self.data_file)))

    def __convert_pos(self, t):

        if t in self.VERB_TAGS:
            return 'V'
        elif t in self.NOUN_TAGS:
            return 'N'
        elif t in self.ADJ_TAGS:
            return 'A'
        # elif t in ADV_TAGS:
        #     return 'LY'
        else:
            return 'O'

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx, corpus=None):
        row = self.data_file.iloc[idx]

        if corpus is not None and row['src'] != corpus:
            # check if the row is from the corpus and return None if not
            return None

        # load text
        ori_text = json.loads(row['text'])

        # load topic
        ori_topic = json.loads(row['topic'])

        # load pos tags
        pos_text = json.loads(row['pos_text'])
        lem_text = json.loads(row['lem_text'])

        def get_index(word):
            return self.word2i[word] if word in self.word2i else len(self.word2i)

        def get_index_filter_pos(word, pos, keep_tag):
            t = self.__convert_pos(pos)
            if word in self.word2i and t == keep_tag:
                return self.word2i[word]
            else:
                return self.pad_value # unk token = vector of all zeros

        # index pos filtered topic
        text_pos2filtered = dict()
        for pos in self.pos_lst:
            text_pos2filtered[pos] = [[get_index_filter_pos(w, p, pos) for w, p in zip(ti, pi)] for ti, pi in zip(lem_text, pos_text)]

        # index text
        text = [[get_index(w) for w in s] for s in ori_text]  # [get_index(w) for w in text]
        
        # truncate text
        if self.keep_sen:
            text = text[:self.max_sen_len]
            text = [t[:self.max_tok_len] for t in text]  # truncates the length of each sentence, by # tokens
            text_lens = [len(t) for t in text]  # compute lens (before padding)
            for pos in text_pos2filtered:
                text_pos2filtered[pos] = [t[:self.max_tok_len] for t in text_pos2filtered[pos][:self.max_sen_len]]
        else:
            text = reduce(lambda x, y: x + y, text)
            text = text[:self.max_tok_len]
            text_lens = len(text)  # compute combined text len
            for pos in text_pos2filtered:
                text_pos2filtered[pos] = reduce(lambda x,y: x+y, text_pos2filtered[pos])[:self.max_tok_len]

        # index topic
        topic = [get_index(w) for w in ori_topic][:self.max_top_len]

        # compute topic len
        topic_lens = len(topic)  # get len (before padding)

        # pad topic
        while len(topic) < self.max_top_len:
            topic.append(self.pad_value)

        # pad text and pos filtered text
        if self.keep_sen:
            for t in text:
                while len(t) < self.max_tok_len:
                    t.append(self.pad_value)
            # pad pos filtered
            for pos in text_pos2filtered:
                for t in text_pos2filtered[pos]:
                    while len(t) < self.max_tok_len:
                        t.append(self.pad_value)

            while len(text_lens) < self.max_sen_len:
                text_lens.append(1)
        else:
            while len(text) < self.max_tok_len:
                text.append(self.pad_value)
            # pad pos filtered
            for pos in text_pos2filtered:
                while len(text_pos2filtered[pos]) < self.max_tok_len:
                    text_pos2filtered[pos].append(self.pad_value)
        if self.binary:
            if float(row['label']) == 0:
                l = 0
            else:
                l = 1
        else:
            if float(row['label']) == 0:
                l = 0
            elif float(row['label']) == 1:
                l = 1
            else:
                l = 2

        sample = {'text': text, 'topic': topic, 'label': l,
                  'txt_l': text_lens, 'top_l': topic_lens,
                  'text_pos2filtered': text_pos2filtered,
                  'ori_topic': ' '.join(ori_topic),
                  'ori_text': ' '.join([' '.join(ti) for ti in ori_text])}
        return sample


class StanceDataBoW(Dataset):
    '''
    Holds and loads stance data sets with vectors instead of word indices and for
    BoWV model. Does NOT actually store the vectors.
    '''
    def __init__(self, data_name, text_vocab_file, topic_vocab_file, binary=False):
        self.data_file = pd.read_csv(data_name)
        self.text_vocab2i = dict()
        self.topic_vocab2i = dict()

        self.unk_index = 0

        self.__load_vocab(self.text_vocab2i, text_vocab_file)
        self.__load_vocab(self.topic_vocab2i, topic_vocab_file)
        self.binary = binary

    def __load_vocab(self, vocab2i, vocab_file):
        f = open(vocab_file, 'r')
        lines = f.readlines()
        i = 1
        print(len(lines))
        for l in lines:
            w = str(l.strip())
            vocab2i[w] = i
            i += 1

    def __len__(self):
        return len(self.data_file)

    def get_vocab_size(self):
        '''
        Gets the size of the vocabulary.
        :return: The vocabulary size
        '''
        return len(self.vocab2i)

    def __convert_text(self, input_data, col):
        '''
        Converts text data to BoW
        :param input_data: tokenized text, as a list
        :return: BoW version of input using the stored vocabulary
        '''
        # make BoW representation
        if col == 'topic':
            vocab2i = self.topic_vocab2i
        else:
            vocab2i = self.text_vocab2i

        text_bow = [0 for _ in range(len(vocab2i) + 1)]
        for w in input_data:
            if w in vocab2i:
                try:
                    text_bow[vocab2i[w]] = 1
                except: embed()
            else:
                text_bow[self.unk_index] = 1
        return text_bow

    def __getitem__(self, idx):
        row = self.data_file.iloc[idx]

        text = reduce(lambda x,y: x + y, json.loads(row['text'])) # collapse the sentences
        topic = json.loads(row['topic'])

        text = self.__convert_text(text, 'text')
        topic = self.__convert_text(topic, 'topic')

        if self.binary:
            if float(row['label']) == 0:
                l = [1, 0]
            else:
                l = [0, 1]
        else:
            if float(row['label']) == 0:
                l = [1, 0, 0]
            elif float(row['label']) == 1:
                l = [0, 1, 0]
            else:
                l = [0, 0, 1]

        sample = {'text': text, 'topic': topic, 'label': l}
        return sample
