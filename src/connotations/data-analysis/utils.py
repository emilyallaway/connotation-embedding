import numpy as np
import pickle

def load_embeddings(data_dir, model_name):
    trn_conn_embeds = np.load(data_dir + '{}_conn-embeds.train.vecs.npy'.format(model_name))
    trn_word2pos2i = pickle.load(open(data_dir + '{}_conn-embeds.train.vocab.pkl'.format(model_name), 'rb'))
    dev_conn_embeds = np.load(data_dir + '{}_conn-embeds.dev.vecs.npy'.format(model_name))
    dev_word2pos2i = pickle.load(open(data_dir + '{}_conn-embeds.dev.vocab.pkl'.format(model_name),'rb'))
    return trn_conn_embeds, dev_conn_embeds, trn_word2pos2i, dev_word2pos2i