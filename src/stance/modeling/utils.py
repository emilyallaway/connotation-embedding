import torch, os, pickle, json, random
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import numpy as np


def load_vectors(vecfile):
    '''
    Loads saved vectors;
    :param vecfile: the name of the file to load the vectors from.
    :return: a numpy array of all the vectors.
    '''
    vecs = np.load(vecfile)

    vecs = np.vstack((vecs, np.random.randn(300))) # <unk> -> V-2
    vecs = np.vstack((vecs, np.zeros(300))) # pad -> V-1 ???
    vecs = vecs.astype(float, copy=False)

    return vecs


def prepare_batch(sample_batched):
    '''
    Prepares a batch of data to be used in training or evaluation.
    :param sample_batched: a list of dictionaries, where each is a sample
    :return: a list of all the post instances (sorted in decreasing order of len),
            a list of all topic instances (corresponding to the sorted posts),
            a list of labels for the post,topic instances
    '''
    text_lens = np.array([len(sample['text']) for sample in sample_batched])
    indices = (-text_lens).argsort()
    text_batch = [torch.tensor(sample_batched[i]['text']) for i in indices]
    topic_batch = [torch.tensor(sample_batched[i]['topic']) for i in indices]
    labels = [sample_batched[i]['label'] for i in indices]
    return text_batch, topic_batch, labels


def prepare_batch_with_reverse(sample_batched):
    '''
    Prepares a batch of data to be used in training or evaluation.
    :param sample_batched: a list of dictionaries, where each is a sample
    :return: a list of all the post instances (sorted in decreasing order of len),
            a list of all topic instances (corresponding to the sorted posts),
            a list of all the post instances (sored in decreasing order of len), reversed
            a list of all topic instances (corresponding to the sorted posts, reversed
            a list of labels for the post,topic instances
    '''
    text_lens = np.array([len(sample['text']) for sample in sample_batched])
    indices = (-text_lens).argsort()
    text_batch = [torch.tensor(sample_batched[i]['text']) for i in indices]
    topic_batch = [torch.tensor(sample_batched[i]['topic']) for i in indices]
    labels = [sample_batched[i]['label'] for i in indices]

    # get the flipped text and topic for backward lstm
    rev_text_batch = [torch.flip(torch.tensor(sample_batched[i]['text']), dims=[0]) for i in indices]
    rev_topic_batch = [torch.flip(torch.tensor(sample_batched[i]['topic']), dims=[0]) for i in indices]

    return text_batch, topic_batch, rev_text_batch, rev_topic_batch, labels


class StanceData(Dataset):
    '''
    Holds the stance dataset.
    '''
    def __init__(self, data_name, vocab_name):
        self.data_file = pd.read_csv(data_name)
        self.word2i = pickle.load(open(vocab_name, 'rb'))

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx, binary=True):
        row = self.data_file.iloc[idx]

        text = json.loads(row['text'])
        topic = json.loads(row['topic'])

        def get_index(word):
            return self.word2i[word] if word in self.word2i else len(self.word2i)

        text = [get_index(w) for w in text]
        topic = [get_index(w) for w in topic]

        if binary:
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

        sample =  {'text': text, 'topic': topic, 'label': l}
        return sample


class StanceDataBoW(Dataset):
    '''
    Holds and loads stance data sets with vectors instead of word indices and for
    BoWV model. Does NOT actually store the vectors.
    '''
    def __init__(self, data_name, text_vocab_file, topic_vocab_file):
        self.data_file = pd.read_csv(data_name)
        self.text_vocab2i = dict()
        self.topic_vocab2i = dict()

        self.unk_index = 0

        self.__load_vocab(self.text_vocab2i, text_vocab_file)
        self.__load_vocab(self.topic_vocab2i, topic_vocab_file)


    def __load_vocab(self, vocab2i, vocab_file):
        f = open(vocab_file, 'r')
        lines = f.readlines()
        i = 1
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
                text_bow[vocab2i[w]] = 1
            else:
                text_bow[self.unk_index] = 1
        return text_bow

    def __getitem__(self, idx, binary=True):
        row = self.data_file.iloc[idx]

        text = json.loads(row['text'])
        topic = json.loads(row['topic'])

        text = self.__convert_text(text, 'text')
        topic = self.__convert_text(topic, 'topic')

        if binary:
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


class DataSampler():
    '''
    A sampler for a dataset. Can get samples of differents sizes.
    Is iterable. By default shuffles the data each time all the data
    has been used through iteration.
    '''
    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = list(range(len(data)))
        if shuffle: random.shuffle(self.indices)
        self.batch_num = 0

    def __iter__(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle: random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.indices != []:
            idxs = self.indices[:self.batch_size]
            batch = [self.data.__getitem__(i) for i in idxs]
            self.indices = self.indices[self.batch_size:]
            return batch
        else:
            raise StopIteration

    def get(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle: random.shuffle(self.indices)
        return self.__next__()
