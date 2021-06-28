import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
import argparse
from functools import reduce

def load_original(inname, col, k=None):
    df = pd.read_csv(inname)

    str_data = []
    for i in df.index:
        words = reduce(lambda x, y: x + y, json.loads(df.iloc[i][col]))
        str_data.append(' '.join(words))

    if k is None:
        CV = CountVectorizer()
    else:
        CV = CountVectorizer(max_features=k)

    CV.fit(str_data)

    return CV.vocabulary_

def prune_vocab_topk(inname, col, outname, k=10000):
    vocab = load_original(inname, col, k)
    out_file = open(outname, 'w')
    for w in vocab:
        out_file.write('{}\n'.format(w))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='What to do', required=True)
    parser.add_argument('-d', '--data', help='Data file', required=False)
    parser.add_argument('-o', '--out_file', help='Output file name prefix', required=False)
    parser.add_argument('-c', '--col', help='Name of data column to use', required=False)
    parser.add_argument('-k', '--k', help='Number of words to keep in vocabulary', required=False)
    args = vars(parser.parse_args())

    if args['mode'] == '1':
        k = int(args['k'])
        outname = '{}_top{}.txt'.format(args['out_file'], k)
        prune_vocab_topk(args['data'], args['col'], outname, k=k)
    else:
        print("ERROR: doing nothing")






