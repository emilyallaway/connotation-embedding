import numpy as np
import pickle


def myround(n):
    '''
    Rounds a number to 1,-1, or 0, based on the cutoffs noted in Rashkin et al. 2016 (ACL).
    :param n: the number to round
    :return: 1 if number is positive, -1 if number is  negative, else 0
    '''
    if isinstance(n, int): return n
    if n < -0.25: return -1
    elif n > 0.25 and n < 2: return 1
    else: return 0


def make_label(n):
    if n < 0:
        return 0
    elif n == 1:
        return 1
    elif n == 0:
        return 2
    else:
        return 3


def reverse_label(d):
    if d == 0: return -1
    elif d == 1: return 0
    else: return 1


def load_vectors(vec_file, keep_words=None, d=300):
    word2idx = dict()
    vec_mat = [[]]
    idx = 1
    allwords = set()
    with open(vec_file) as f:
        for line in f:
            temp = line.split(' ')
            if len(temp) != d +1:
                print(temp[0])
                continue
            allwords.add(temp[0])
            if keep_words is not None and temp[0] not in keep_words:
                continue
            vec = list(map(lambda x: float(x), temp[1:]))
            word2idx[temp[0]] = idx
            idx += 1
            vec_mat.append(vec)
            if idx % 100000 == 0:
                print(idx)
    vec_mat[0] = [0] * len(vec_mat[1])
    print("finished making matrix: {}".format(len(vec_mat)))
    return word2idx, np.array(vec_mat, dtype='float32')


def load_vectors_mine(vec_File, keep_words=None, d=300, ignore_pos=None):
    conn_embeds = np.load(vec_File + '.vecs.npy')
    word2pos2i = pickle.load(open(vec_File + '.vocab.pkl', 'rb'))

    word2idx = dict()
    vec_mat = [[]]
    idx = 0
    for w in word2pos2i:
        if keep_words is not None and w not in keep_words:
            continue
        vlst = []
        for pos in word2pos2i[w]:
            if ignore_pos is not None and pos in ignore_pos: continue
            vlst.append(conn_embeds[word2pos2i[w][pos]])
        if len(vlst) == 0:
            if 'O' in word2pos2i[w]:
                vlst.append(conn_embeds[word2pos2i[w]['O']])
            else: continue
        vec_mat.append(np.mean(vlst, axis=0))
        word2idx[w] = idx
        idx += 1
    vec_mat[0] = np.zeros(d)
    print("finished making matrix: {}".format(len(vec_mat)))
    return word2idx, np.array(vec_mat, dtype='float32')


def load_vectors_both(vec_File1, vec_File2, keep_words=None, pos='V'):
    my_word2idx, my_vecs = load_vectors_mine(vec_File1, keep_words=keep_words,pos=pos)
    pre_word2idx, pre_vecs = load_vectors(vec_File2, keep_words=keep_words)
    vec_mat = [[]]
    word2idx = dict()
    idx = 0
    n = 0
    d = my_vecs.shape[1]
    for w in pre_word2idx:
        if w in my_word2idx:
            mv = my_vecs[my_word2idx[w]]
            n += 1
        else:
            mv = np.zeros(my_vecs.shape[1])
        v = np.mean((pre_vecs[pre_word2idx[w]], mv), axis=0)
        vec_mat.append(v)
        word2idx[w] = idx
        idx += 1
    vec_mat[0] = np.zeros(d)
    vec_mat = np.array(vec_mat, dtype='float32')
    print("finished making matrix: {}".format(vec_mat.shape))
    print(n)
    return word2idx, vec_mat


def load_mine_train_plus(vec_file, keep_words=None, pos='N'):
    print("loading {}".format(vec_file + '.train'))
    trn_word2idx, trn_vecs = load_vectors_mine(vec_file + '.train', keep_words=keep_words, pos=pos)
    print("loading {}".format(vec_file + '.dev'))
    tst_word2idx, tst_vecs = load_vectors_mine(vec_file + '.dev', keep_words=keep_words, pos=pos)

    word2idx = dict()
    vec_mat = [[]]
    idx = 0
    for w in trn_word2idx:
        word2idx[w] = idx
        idx += 1
        vec_mat.append(trn_vecs[trn_word2idx[w]])
    for w in tst_word2idx:
        word2idx[w] = idx
        idx += 1
        vec_mat.append(tst_vecs[tst_word2idx[w]])
    vec_mat[0] = np.zeros(trn_vecs.shape[1])
    return word2idx, np.array(vec_mat, dtype='float32')

