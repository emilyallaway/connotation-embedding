import pickle as cPickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from IPython import embed
from functools import reduce
import json, sys, argparse
from functools import reduce

'''
Loads and saves files for all the words that will be needed in training/dev/test. Make sure
that there is a vector file. Can change which file it is below on line 49. Can also change
the names of the input files (lines 38-40) and the output location (lines 62-63). The words
are saved as a numpy array of vectors and a map from words to indices in the array. 
'''

VECTOR_PATH = '../../../resources/connotations'
VECTOR_NAME = 'numberbatch-en-19.08'

abbs = ["'nt", "'d", "'ve", "'s"]

COLUMNS = ['word', 'def_lst', 'rel_lst']

DATA_NAME = '../../../data/lexicon/embedding/conn_input'


def load_words(inname, cols=COLUMNS):
    df = pd.read_csv(inname, keep_default_na=False)
    vocab = set()
    word2freq = dict()
    for i in df.index:
        row = df.iloc[i]
        for c in cols:
            if c == 'def_lst':
                words = json.loads(row[c])
                if len(words) != 0:
                    words = reduce(lambda x,y: x+y, words)
            else:
                word_temp = row[c]
                if type(word_temp) != list:
                    words = ['_'.join(word_temp.lower().strip(',').replace('/', ' ').split())] # CLEAN
            for w in words:
                word2freq[w] = word2freq.get(w, 0) + 1
            for w in words:
                for wi in w.split('_'):
                    vocab.add(wi)
                vocab.add(' '.join(w.split('_'))) # CLEAN
    return vocab, word2freq


def save_vectors_vocab_full(suffix='conn'):
    all_vocab = get_vocabs()

    vecs = []
    word2i = dict()
    allw2i = set()
    i = 0
    with open('{}{}.txt'.format(VECTOR_PATH, VECTOR_NAME), 'r') as f:
        for l in f:
            fields = l.strip().split()
            w = fields[0]
            allw2i.add(w)

            if w not in all_vocab: continue

            if i % 10000 == 0: print(i)
            v = fields[1:]
            vecs.append(v)
            word2i[w] = i
            i += 1
    print(len(vecs))

    # INCLUDES a single vector for every multi-word expression, ex: give_birth
    i = len(word2i)
    for w in all_vocab:
        if w not in word2i and ' ' in w:
            wlst = []
            for wi in w.split(' '):
                if wi in word2i:
                    wlst.append(vecs[word2i[wi]])
            if len(wlst) == 0: continue
            wlst = np.array(wlst).astype(float)
            vecs.append(np.mean(wlst, axis=0))
            word2i[w] = i
            i += 1
    print(len(vecs))

    np.save('{}{}-{}.vectorsF.npy'.format(VECTOR_PATH, VECTOR_NAME, suffix), vecs)
    cPickle.dump(word2i, open('{}{}-{}.vocabF.pkl'.format(VECTOR_PATH, VECTOR_NAME, suffix), 'wb'),
                 protocol=cPickle.HIGHEST_PROTOCOL)
    del vecs


def get_vocabs(cols=COLUMNS):
    # get vocab for text only
    train_vocab, train2freq = load_words(DATA_NAME + '-train.csv', cols=cols)
    dev_vocab, dev2freq = load_words(DATA_NAME + '-dev.csv', cols=cols)
    test_vocab, test2freq = load_words(DATA_NAME + '-test.csv', cols=cols)
    all_vocab = train_vocab.union(dev_vocab, test_vocab)
    return all_vocab


def save_vocabs_only():
    word_vocab = get_vocabs(cols=['word'])
    out_file = open('{}word_vocab.txt'.format(VECTOR_PATH), 'w')
    for w in word_vocab:
        out_file.write('{}\n'.format(w))

    def_vocb = get_vocabs(cols=['def_lst'])
    out_file = open('{}def_vocab.txt'.format(VECTOR_PATH), 'w')
    for w in def_vocb:
        out_file.write('{}\n'.format(w))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='What to do', required=True)
    parser.add_argument('-d', '--data_file', help='Input data file', required=False)
    parser.add_argument('-n', '--vec_name', help='Name of vectors', required=False)
    args = vars(parser.parse_args())

    if args['mode'] == '1':
        save_vectors_vocab_full()
    elif args['mode'] == '2':
        save_vocabs_only()
    elif args['mode'] == '3':
        save_vectors_vocab_full(suffix='defs')
    else:
        print("ERROR: doing nothing")
