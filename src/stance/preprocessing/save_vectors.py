import pickle as cPickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from IPython import embed
import json, sys, argparse, os, re, string, pickle
from functools import reduce
from nltk.corpus import stopwords

'''
Loads and saves files for all the words that will be needed in training/dev/test. Make sure
that there is a vector file. Can change which file it is below on line 49. Can also change
the names of the input files (lines 38-40) and the output location (lines 62-63). The words
are saved as a numpy array of vectors and a map from words to indices in the array. 
'''

DATA_PATH = '../../../data/stance/IACv2_stance'

VECTOR_PATH = '../../../resources/stance/'
VECTOR_NAME = 'glove.6B.100d'

DEF_DIR = ''

MODEL_NAME = 'BiLSTMRelEmbedderVERB-cn-w7-s4-42w-norm-learn-20r-balanced'
MODEL_SHORT = 'allwords_defrel'
MODEL_DIM = 300

abbs = ["'nt", "'d", "'ve", "'s", "n't"]


def load_words(inname, cols=['topic', 'text']):
    df = pd.read_csv(inname)
    vocab = set()
    word2freq = dict()
    for i in df.index:
        row = df.iloc[i]
        for c in cols:
            if c == 'text':
                words = reduce(lambda x,y: x+y, json.loads(row[c]))
            else:
                try:
                    words = json.loads(row[c])
                except: embed()
            words = list(map(lambda x: x.strip().lower(), words))

            for w in words:
                word2freq[w] = word2freq.get(w, 0) + 1
            for w in words: vocab.add(w)
    return vocab, word2freq


def save_vectors_vocab_full():
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

    np.save('{}{}.vectorsF.npy'.format(VECTOR_PATH, VECTOR_NAME), vecs)
    cPickle.dump(word2i, open('{}{}.vocabF.pkl'.format(VECTOR_PATH, VECTOR_NAME), 'wb'),
                 protocol=cPickle.HIGHEST_PROTOCOL)
    del vecs


def get_vocabs(cols=['text', 'topic']):
    # get vocab for text only
    train_vocab, train2freq = load_words('{}-train.csv'.format(DATA_PATH), cols=cols)
    dev_vocab, dev2freq = load_words('{}-dev.csv'.format(DATA_PATH), cols=cols)
    test_vocab, test2freq = load_words('{}-test.csv'.format(DATA_PATH), cols=cols)
    all_vocab = train_vocab | dev_vocab | test_vocab
    return all_vocab


def get_vocab_freq(cols):
    train_vocab, train2freq = load_words('{}-train.csv'.format(DATA_PATH), cols=cols)
    dev_vocab, dev2freq = load_words('{}-dev.csv'.format(DATA_PATH), cols=cols)
    test_vocab, test2freq = load_words('{}-test.csv'.format(DATA_PATH), cols=cols)
    all_vocab = train_vocab | dev_vocab | test_vocab
    all2freq = dict()
    for w in all_vocab:
        n = 0
        if w in train_vocab:
            n += train2freq[w]
        if w in dev_vocab:
            n += dev2freq[w]
        if w in test_vocab:
            n += test2freq[w]
        all2freq[w] = n
    return all_vocab, all2freq


def save_vocabs_only(vocab_size=5000):
    text_vocab, text2freq = get_vocab_freq(cols=['text'])
    print(len(text_vocab))
    sorted_vocab = sorted(text2freq.items(), key=lambda kv: kv[1], reverse=True)[:vocab_size]
    print(len(sorted_vocab))

    out_file = open('{}text_vocab_top{}.txt'.format(VECTOR_PATH, vocab_size), 'w')
    for w,_ in sorted_vocab:
        out_file.write('{}\n'.format(w.strip()))

    topic_vocab= get_vocabs(cols=['topic'])
    print(len(topic_vocab))
    out_file = open('{}topic_vocab.txt'.format(VECTOR_PATH), 'w')
    for w in topic_vocab:
        out_file.write('{}\n'.format(w))


def make_embedding_file(outf, max_defs=10):
    sw = set(stopwords.words('english'))

    word2i = cPickle.load(open('{}{}.vocabF.pkl'.format(VECTOR_PATH, VECTOR_NAME), 'rb'))

    word2pos2defs = dict()
    for w in word2i:
        wc = '_'.join(w.lower().strip(',').replace('/', ' ').split())
        word = ' '.join(wc.split('_'))
        try:
            l = wc[0]
        except:
            continue
        for t in ['N', 'V', 'A', 'O']:
            p = os.path.join(DEF_DIR, l, '{}.{}.txt'.format(wc, t))
            if os.path.exists(p):
                f = open(p, 'r')
                lines = f.readlines()
                def_l = []
                for fl in lines:
                    ltemp = re.sub(r'<[^<>]*>', '', fl.strip().lower())
                    wlst = list(filter(lambda x: x not in sw and x not in string.punctuation and x != word,
                                       word_tokenize(ltemp)))
                    if len(wlst) > 0:
                        def_l.append(wlst)
                word2pos2defs[w] = word2pos2defs.get(w, dict())
                word2pos2defs[w][t] = word2pos2defs[w].get(t, [])
                word2pos2defs[w][t] = word2pos2defs[w][t] + def_l
    print("data size (in words) before removing 0 defs: {}".format(len(word2pos2defs)))
    data = []
    t2c = dict()
    for w in word2pos2defs:
        for t in word2pos2defs[w]:
            defl = word2pos2defs[w][t][:max_defs]
            if len(defl) == 0: continue
            data.append([w, t, json.dumps(defl)])
            t2c[t] = t2c.get(t, 0) + 1
    print("data size: {}".format(len(data)))
    print(t2c)
    df = pd.DataFrame(data, columns=['word', 'POS', 'def_lst'])
    df.to_csv(outf, index=False)


def save_myembed(vec_name):
    conn_embeds = np.load(VECTOR_PATH + 'myvecs/' + '{}_conn-embeds.{}.vecs.npy'.format(MODEL_NAME, MODEL_SHORT))
    word2pos2i = pickle.load(open(VECTOR_PATH + 'myvecs/'
                                  + '{}_conn-embeds.{}.vocab.pkl'.format(MODEL_NAME, MODEL_SHORT), 'rb'))  # all POS are O

    word2i = cPickle.load(open('{}{}.vocabF.pkl'.format(VECTOR_PATH, VECTOR_NAME), 'rb'))

    num_cols = MODEL_DIM
    for pos in ['N', 'A', 'V', 'O']:
        conn_vecs = [0] * len(word2i)
        for w in word2i:
            if w not in word2pos2i or pos not in word2pos2i.get(w, []):
                vec = [0] * num_cols
            else:
                vec = conn_embeds[word2pos2i[w][pos]]
            conn_vecs[word2i[w]] = vec
        np.save('{}{}{}-{}_vectorsF.npy'.format(VECTOR_PATH, 'lexicon-connvectors/', vec_name, pos), conn_vecs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='What to do', required=True)
    parser.add_argument('-d', '--data_file', help='Input data file, possibly a ; separated list',
                        required=False)
    parser.add_argument('-n', '--vec_name', help='Name of vectors', required=False)
    parser.add_argument('-o', '--out_name', help='Output file name', required=False)
    args = vars(parser.parse_args())

    if args['mode'] == '1':
        save_vectors_vocab_full()

    elif args['mode'] == '2':
        save_vocabs_only()

    elif args['mode'] == '4':
        # make embedding input file
        make_embedding_file(args['out_name'])

    elif args['mode'] == '5':
        # make my embeddings into the right format
        save_myembed(args['vec_name'])

    else:
        print("ERROR: doing nothing")
