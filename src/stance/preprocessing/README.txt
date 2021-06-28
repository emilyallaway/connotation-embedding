HOW TO MAKE VECTORS AND MAPS
1: to make vectors and maps:
    run: python save_vectors.py -m 1
    makes: glove.42B.300d.vectorsF.npy
            glove.42B.300d.vocabF.npy

2: to make only the vocabs for text and topic individually
    run: python save_vectors.py -m 2
    makes:
        text_vocab.txt
        topic_vocab.txt


TO PRUNE VOCABULARY
1. with top k most frequent words
    run: python make_vocab.py -m 1 -d <data_name> -c <data_col_name> -o <outname_prefix> -k <k>