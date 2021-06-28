import torch, random, json
import numpy as np

def load_vectors(vecfile, dim=300, unk_rand=True, seed=0):
    '''
    Loads saved vectors;
    :param vecfile: the name of the file to load the vectors from.
    :return: a numpy array of all the vectors.
    '''
    vecs = np.load(vecfile)
    np.random.seed(seed)

    if unk_rand:
        vecs = np.vstack((vecs, np.random.randn(dim))) # <unk> -> V-2 ??
    else:
        vecs = np.vstack((vecs, np.zeros(dim))) # <unk> -> V - 2??
    vecs = np.vstack((vecs, np.zeros(dim))) # pad -> V-1 ???
    vecs = vecs.astype(float, copy=False)

    return vecs


def prepare_batch_with_sentences_in_rev(sample_batched, max_tok_len=35, pos_filtered=False, **kwargs):
    '''
    Prepares a batch of data, preserving sentence boundaries and returning the sentences in reverse order.
    Also returns reversed text and topic.
    :param sample_batched: a list of dictionaries, where each is a sample
    :param max_sen_len: the maximum # sentences allowed
    :param padding_idx: the index for padding for dummy sentences
    :param pos_filtered: a flag determining whether to return pos filtered text
    :return: the text batch, has shape (S, B, Tij) where S is the max sen len, B is the batch size,
                and Tij is the number of tokens in sentence i of batch element j;
            the topic batch, a list of all topic instances;
            the a list of labels for the post, topic instances
            AND (depending on flag)
            the text pos filtered, with the same shape as the text batches
    '''
    topic_batch = torch.tensor([b['topic'] for b in sample_batched])
    labels = [torch.tensor(b['label']) for b in sample_batched]

    max_sen_len = kwargs['max_sen_len']
    padding_idx = kwargs['padding_idx']

    text_batch = []
    text_batch_pos2filter = dict()
    sens = []
    txt_lens_lst = []
    for i in range(1, max_sen_len + 1):
        sn_idx = max_sen_len - i
        s_lst = []
        s_lst_pos_D = dict()
        si = []
        s_len_lst = []
        for b in sample_batched:
            if len(b['text']) > sn_idx:
                s_lst.append(torch.tensor([b['text'][sn_idx]]))
                si.append(1)
            else:
                s_lst.append(torch.tensor([[padding_idx] * max_tok_len]))
                si.append(0)
            s_len_lst.append(b['txt_l'][sn_idx])

            if kwargs.get('use_conn', False):
                for pos in b['text_pos2filtered']:
                    s_lst_pos = s_lst_pos_D.get(pos, [])
                    if len(b['text_pos_filtered'][pos]) > sn_idx and len(b['text_pos_filtered'][pos][sn_idx]) > 0:
                        s_lst_pos.append(torch.tensor([b['text_pos_filtered'][pos][sn_idx]]))
                    else:
                        s_lst_pos.append(torch.tensor([[padding_idx] * max_tok_len]))

        text_batch.append(torch.cat(s_lst, dim=0))

        if kwargs.get('use_conn', False):
            for t in s_lst_pos_D:
                text_batch_pos2filter[t] = text_batch_pos2filter.get(t, [])
                text_batch_pos2filter[t].append(torch.cat(s_lst_pos_D[t], dim=0))

        sens.append(si)
        txt_lens_lst.append(s_len_lst)

    txt_lens = txt_lens_lst # (S, B, T)?
    top_lens = [b['top_l'] for b in sample_batched]

    args = {'text': text_batch, 'topic': topic_batch,
            'labels': labels, 'sentence_mask': sens,
            'txt_l': txt_lens, 'top_l': top_lens}
    if pos_filtered:
        args['text_pos2filter'] = text_batch_pos2filter
    return args



def prepare_batch_with_reverse(sample_batched, **kwargs):
    '''
    Prepares a batch of data to be used in training or evaluation. Includes the text reversed.
    :param sample_batched: a list of dictionaries, where each is a sample
    :param pos_filtered: a flag determining whether to return pos filtered text
    :return: a list of all the post instances (sorted in decreasing order of len),
            a list of all topic instances (corresponding to the sorted posts),
            a list of all the post instances (sored in decreasing order of len), reversed
            a list of all topic instances (corresponding to the sorted posts, reversed
            a list of labels for the post,topic instances
            AND (depending on flag)
            a list of all posts instances with only certain POS (sorted in dec order of len)
            a list of all post instances reversed with only certain POS (sorted in dec order of len)
    '''
    text_lens = np.array([b['txt_l'] for b in sample_batched])
    text_batch = torch.tensor([b['text'] for b in sample_batched])
    topic_batch = torch.tensor([b['topic'] for b in sample_batched])
    labels = [b['label'] for b in sample_batched]
    top_lens = [b['top_l'] for b in sample_batched]

    raw_text_batch = [b['ori_text'] for b in sample_batched]
    raw_top_batch = [b['ori_topic'] for b in sample_batched]

    args = {'text': text_batch, 'topic': topic_batch, 'labels': labels,
            'txt_l': text_lens, 'top_l': top_lens,
            'ori_text': raw_text_batch, 'ori_topic': raw_top_batch}

    if kwargs.get('use_conn', False):
        text_pos2filtered = dict()
        for t in sample_batched[0]['text_pos2filtered']:
            text_pos2filtered[t] = torch.tensor([b['text_pos2filtered'][t] for b in sample_batched])
        args['text_pos2filter'] = text_pos2filtered

    return args


def prepare_batch_raw(sample_batched, **kwargs):
    text_lens = np.array([b['txt_l'] for b in sample_batched])
    top_lens = [b['top_l'] for b in sample_batched]

    text_batch = [b['ori_text'] for b in sample_batched]
    top_batch = [b['ori_topic'] for b in sample_batched]

    args = {'ori_topic': top_batch, 'ori_text': text_batch,
            'txt_l': text_lens, 'top_l': top_lens}
    return args


class DataSampler:
    '''
    A sampler for a dataset. Can get samples of differents sizes.
    Is iterable. By default shuffles the data each time all the data
    has been used through iteration.
    '''
    def __init__(self, data, batch_size, shuffle=True, sampling=None, weighting=None):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        random.seed(0)

        self.sampling = sampling
        self.weighting = weighting

        if self.sampling is not None:
            self.__compute_weights()

        if self.weighting is not None:
            self.__compute_topic_weights()

        self.indices = list(range(len(data)))
        if shuffle: random.shuffle(self.indices)
        self.batch_num = 0

    def __len__(self):
        return len(self.data)

    def __compute_weights(self):
        c2count = dict()
        for i in self.data.data_file.index:
            l = self.data.data_file.iloc[i]['label']
            c2count[l] = c2count.get(l, 0.) + 1.

        clst = [1.] * len(c2count)
        for c in c2count:
            if c2count[c] != 0:
                clst[c] = len(self.data) / (len(c2count) * c2count[c])
        self.weight = torch.tensor(clst)

    def __compute_topic_weights(self):
        t2count = dict()
        t2c2count = dict()
        for i in self.data.data_file.index:
            t = ' '.join(json.loads(self.data.data_file.iloc[i]['topic']))
            l = self.data.data_file.iloc[i]['label']
            t2count[t] = t2count.get(t, 0.) + 1. # count of the topic
            t2c2count[t] = t2c2count.get(t, dict())
            t2c2count[t][l] = t2c2count[t].get(l, 0.) + 1. # count of label for topic

        t2w = dict()
        t2c2w = dict()
        for t in t2count:
            t2w[t] = len(self.data) / (len(t2count) * t2count[t])
            for c in t2c2count[t]:
                t2c2w[t] = t2c2w.get(t, dict())
                t2c2w[t][c] = t2count[t] / (len(t2c2count[t]) * t2c2count[t][c])
        self.topic_weight = t2w
        self.topic2c2w = t2c2w


    def num_batches(self):
        return len(self.data) / float(self.batch_size)

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
        self.reset()
        return self.__next__()

    def reset(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle: random.shuffle(self.indices)
