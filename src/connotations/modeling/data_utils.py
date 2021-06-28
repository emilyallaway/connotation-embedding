import torch, random, json, copy
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


def prepare_batch_with_defs_sep(sample_batched, max_tok_len=35, pos_filtered=False, **kwargs):
    '''
    Prepares a batch of data, preserving sentence boundaries and returning the sentences in reverse order.
    Collapses all sentences together.
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
    word_batch = torch.tensor([b['word'] for b in sample_batched])
    labels = [torch.tensor(b['labels']) for b in sample_batched]

    max_sen_len = kwargs['max_sen_len']
    padding_idx = kwargs['padding_idx']

    text_batch = []
    sens = []
    txt_lens_lst = []
    for i in range(0, max_sen_len):
        s_lst = []
        si = []
        s_len_lst = []
        for b in sample_batched:
            if len(b['def_lst']) < i:
                s_lst.append(torch.tensor([b['def_lst'][i]]))
                si.append(1)
            else:
                s_lst.append(torch.tensor([[padding_idx] * max_tok_len]))
                si.append(0)
            s_len_lst.append(b['def_l'][i])

        text_batch.append(torch.cat(s_lst, dim=0))

        sens.append(si)
        txt_lens_lst.append(s_len_lst)

    txt_lens = txt_lens_lst # (S, B, T)?
    word_lens = [b['word_l'] for b in sample_batched]

    args = {'def': text_batch, 'word': word_batch,
            'labels': labels, 'sentence_mask': sens,
            'def_l': txt_lens, 'def_l': word_lens}

    return args


def prepare_batch_with_reverse(sample_batched, pos_filtered=False, **kwargs):
    '''
    Prepares a batch of data to be used in training or evaluation.
    :param sample_batched: a list of dictionaries, where each is a sample
    :param pos_filtered: a flag determining whether to return pos filtered text
    :return: a list of all the post instances (sorted in decreasing order of len),
            a list of all topic instances (corresponding to the sorted posts),
            a list of labels for the post,topic instances
            AND (depending on flag)
            a list of all posts instances with only certain POS (sorted in dec order of len)
    '''
    text_lens = np.array([b['def_l'] for b in sample_batched])
    text_batch = torch.tensor([b['def_lst'] for b in sample_batched])
    word_batch = torch.tensor([b['word'] for b in sample_batched])

    labels = []
    if kwargs.get('use_labels', True):
        for i in range(kwargs['num_dims']):
            labels.append([b['labels'][i] for b in sample_batched])

        if kwargs.get('dict_task', False):
            labels.append([b['labels'][0] for b in sample_batched])

    # labels = [b['labels'] for b in sample_batched]
    word_lens = [b['word_l'] for b in sample_batched]

    args = {'def': text_batch, 'word': word_batch, 'labels': labels,
            'def_l': text_lens, 'word_l': word_lens, 'use_keys': ['def', 'word']}

    if pos_filtered:
        args['def_pos'] = torch.tensor([b['def_post_filtered'] for b in sample_batched])

    if kwargs.get('use_related', None) is not None:
        args['rel'] = torch.tensor([b['rel_lst'] for b in sample_batched])
        args['rel_l'] = torch.tensor([b['rel_l'] for b in sample_batched])
        args['use_keys'] = args['use_keys'] + ['rel']

    if kwargs.get('verb_task', False):
        args['mask'] = [b['mask'] for b in sample_batched]

    return args


def prepare_batch_RD(sample_batched, pos_filtered=False, **kwargs):
    '''
    Prepares a batch of data to be used in training or evaluation.
    :param sample_batched: a list of dictionaries, where each is a sample
    :param pos_filtered: a flag determining whether to return pos filtered text
    :return: a list of all the post instances (sorted in decreasing order of len),
            a list of all topic instances (corresponding to the sorted posts),
            a list of labels for the post,topic instances
            AND (depending on flag)
            a list of all posts instances with only certain POS (sorted in dec order of len)
    '''
    text_lens = np.array([b['def_l'] for b in sample_batched])
    text_batch = torch.tensor([b['def_lst'] for b in sample_batched])
    word_batch = torch.tensor([b['word'] for b in sample_batched])
    word_lens = [b['word_l'] for b in sample_batched]

    labels = [b['labels'][0] for b in sample_batched]

    args = {'def': text_batch, 'word': word_batch, 'labels': [labels],
            'def_l': text_lens, 'word_l': word_lens, 'use_keys': ['def', 'word']}

    if pos_filtered:
        args['def_pos'] = torch.tensor([b['def_post_filtered'] for b in sample_batched])

    if kwargs.get('use_related', None) is not None:
        args['rel'] = torch.tensor([b['rel_lst'] for b in sample_batched])
        args['rel_l'] = torch.tensor([b['rel_l'] for b in sample_batched])
        args['use_keys'] = args['use_keys'] + ['rel']

    # args['rw'] = torch.tensor([b['rw'] for b in sample_batched])
    return args


class DataSampler:
    '''
    A sampler for a Dataset. Can get samples of differents sizes.
    Is iterable. By default shuffles the data each time all the data
    has been used through iteration.
    '''
    def __init__(self, data, batch_size, shuffle=True, dim_weights=None, sampling='weird',
                 count_pos=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.count_pos = count_pos
        random.seed(0)

        self.all_data = list(range(len(data)))

        if dim_weights is not None:
            self.dim_weights = dim_weights
            if sampling == 'weird':
                self.__weird_sampling()
            else:
                self.__balanced_sampling()

        self.indices = copy.deepcopy(self.all_data)
        if shuffle: random.shuffle(self.indices)
        self.batch_num = 0
        self.mode = 1


    def eval(self):
        pass


    def train(self):
        pass


    def __len__(self):
        return len(self.data)

    def __compute_weights(self):
        dim2c2count = dict()
        for i in self.data.data_file.index:
            row = self.data.data_file.iloc[i]
            for d in self.data.dims:
                c = self.__get_class(row[d])
                dim2c2count[d] = dim2c2count.get(d, dict())
                dim2c2count[d][c] = dim2c2count[d].get(c, 0.) + 1.
        dim2c2weight = dict()
        for d in dim2c2count:
            dim2c2weight[d] = dim2c2weight.get(d, dict())
            num_classes = len(dim2c2count[d])
            for c in dim2c2count[d]:
                dim2c2weight[d][c] = len(self.data) / (num_classes * dim2c2count[d][c])

        np.random.seed(0)
        new_indices = []
        for i in self.data.data_file.index:
            row = self.data.data_file.iloc[i]
            v = sum([self.dim_weights[i] * dim2c2weight[d][self.__get_class(row[d])]
                 for i, d in enumerate(self.data.dims)])
            n = np.random.random()
            if n <= v:
                new_indices.append(i)

        self.all_data += new_indices

    def __balanced_sampling(self):
        # NOTE: this will break if there are no samples for a particular class
        dim2c2count = dict()

        use_dims = copy.deepcopy(self.data.dims)
        if self.data.verb_task:
            use_dims += self.data.verb_dims

        na_count = 0
        verb_cf_count = 0
        verb_powa_count = 0
        for i in self.data.data_file.index:
            row = self.data.data_file.iloc[i]
            if row['POS'] == 'V':
                if 'powa' in row['verb-label']:
                    verb_powa_count += 1
                if 'cf' in row['verb-label']:
                    verb_cf_count += 1
            else:
                na_count += 1
            for d in use_dims:
                c = self.__get_class(row[d], d, use_all=True)
                dim2c2count[d] = dim2c2count.get(d, dict())

                if (row['POS'] != 'V' and d in {'Social Val', 'Polite', 'Impact', 'Sent', 'Fact'}) or \
                        (row['POS'] == 'V' and ('powa' in row['verb-label'] and d in {'power', 'agency'}) or
                             ('cf' in row['verb-label'] and d in {'P(wt)', 'P(wa)', 'P(at)', 'E(t)', 'E(a)', 'V(t)', 'V(a)', 'S(t)', 'S(a)'})):
                # if d != 'Emo' and row['POS'] != 'V':
                    dim2c2count[d][c] = dim2c2count[d].get(c, 0.) + 1.
                # else:
                elif row['POS'] != 'V' and d == 'Emo':
                    for ei, e in enumerate(c):
                        dim2c2count[d][ei] = dim2c2count[d].get(ei, 0.) + e
        dim2c2weight = dict()
        for d in dim2c2count:
            dim2c2weight[d] = dim2c2weight.get(d, dict())
            num_classes = len(dim2c2count[d])
            for c in dim2c2count[d]:
                # if d == 'Emo': embed()
                if dim2c2count[d][c] != 0:
                    if not self.count_pos:
                        data_size = len(self.data)
                    else:
                        if d in {'Social Val', 'Polite', 'Impact', 'Sent', 'Fact', 'Emo'}:
                            data_size = na_count
                        elif d in {'power', 'agency'}:
                            data_size = verb_powa_count
                        else:
                            data_size = verb_cf_count

                    dim2c2weight[d][c] = data_size / (num_classes * dim2c2count[d][c])
                else:
                    dim2c2weight[d][c] = 1.
        dim2weight = dict()
        for d in dim2c2weight:
            c_lst = [1.] * len(dim2c2weight[d])
            if d != 'Emo':
                for c in dim2c2weight[d]:
                    c_lst[c] = dim2c2weight[d][c]
            dim2weight[d] = torch.tensor(c_lst)

        self.dim2weight = dim2weight

    def __get_class(self, v, d, use_all=False):
        if isinstance(v, str):
            l = json.loads(v)
            if not use_all:
                if sum(l) > 0:
                    return 1
                else:
                    return 0
            else:
                return l
        else:
            return self.data.get_class(v, d)


    def num_batches(self):
        return len(self.indices) / float(self.batch_size)

    def __iter__(self):
        self.indices = copy.deepcopy(self.all_data)
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
        self.indices = copy.deepcopy(self.all_data)
        if self.shuffle: random.shuffle(self.indices)

    def train(self):
        pass

    def eval(self):
        pass


class POSDataSampler(DataSampler):
    def __init__(self, data, batch_size, shuffle=True, dim_weights=None, sampling='weird', num_resets=10,
                 verb_only=False, na_only=False):
        DataSampler.__init__(self, data=data, batch_size=batch_size, shuffle=shuffle, dim_weights=dim_weights,
                             sampling=sampling, count_pos=True)
        self.split_data()
        self.ori_num_verb_resets = num_resets
        self.num_verb_resets = self.ori_num_verb_resets
        self.verb_batch = False
        self.verb_only = verb_only
        self.na_only = na_only
        self.num_verb_iters = 0

    def split_data(self):
        self.ori_verb_indices = []
        self.ori_na_indices = []
        for i in self.data.data_file.index:
            row = self.data.data_file.iloc[i]
            if row['POS'] == 'V':
                self.ori_verb_indices.append(i)
            elif row['POS'] == 'N' or row['POS'] == 'A':
                self.ori_na_indices.append(i)

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        idxs = None
        if len(self.verb_indices) == 0 and self.num_verb_iters < self.num_verb_resets:
            self.num_verb_iters += 1
            self.reset_verb()

        if self.na_indices != []:
            idxs = self.na_indices[: self.batch_size]
            self.na_indices = self.na_indices[self.batch_size:]
            self.verb_batch = False
        elif self.verb_indices != []:
            idxs = self.verb_indices[: self.batch_size]
            self.verb_indices = self.verb_indices[self.batch_size:]
            self.verb_batch = True
        else:
            self.verb_batch = False

        if idxs is not None:
            batch = [self.data.__getitem__(i) for i in idxs]
            return batch
        else:
            raise StopIteration

    def reset_verb(self):
        self.verb_indices = copy.deepcopy(self.ori_verb_indices)
        if self.shuffle: random.shuffle(self.verb_indices)

    def reset_na(self):
        self.na_indices = copy.deepcopy(self.ori_na_indices)
        if self.shuffle:
            random.shuffle(self.na_indices)


    def reset(self):
        if self.verb_only:
            self.reset_verb()
            self.num_verb_iters = 0
            self.na_indices = []
        elif self.na_only:
            self.verb_indices = []
            self.reset_na()
        else:
            self.reset_verb()
            self.num_verb_iters = 0
            self.reset_na()

    def train(self):
        self.num_verb_resets = self.ori_num_verb_resets

    def eval(self):
        self.num_verb_resets = 0
