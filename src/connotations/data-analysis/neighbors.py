from sklearn.neighbors import NearestNeighbors
import utils, json, argparse, copy
import pandas as pd
import numpy as np
import pickle

def find_neighbors(trn_embeds, dev_embeds, trn_i2wordpos, dev_i2wordpos, model_name, out_name,
                   dist_metric='minkowski', num_neighbors=50):
    knn = NearestNeighbors(n_neighbors=num_neighbors, metric=dist_metric)
    print("fitting KNN ...")
    knn.fit(trn_embeds)

    # get train nearest neighbors
    find_neighbors_helper(knn, trn_embeds, trn_i2wordpos, trn_i2wordpos,
                          'train', out_name, model_name, num_neighbors)

    # get dev nearest neighbors
    find_neighbors_helper(knn, dev_embeds, trn_i2wordpos, dev_i2wordpos,
                   'dev', out_name, model_name, num_neighbors)


def find_neighbors_helper(knn, pred_embeds, trn_i2wordpos, pred_i2wordpos,
                          data_name, out_name, model_name, num_neighbors):
    print("... getting {} {}-NN".format(data_name, num_neighbors))
    neigh_data = []
    dists, neighs = knn.kneighbors(pred_embeds)
    for i in range(len(pred_embeds)):
        nlst = [(trn_i2wordpos[nidx], str(nd)) for nidx, nd in zip(neighs[i], dists[i])]
        neigh_data.append([pred_i2wordpos[i][0], pred_i2wordpos[i][1], json.dumps(nlst)])
    temp = pd.DataFrame(neigh_data, columns=['word', 'POS', 'neighbor_lst'])
    temp.to_csv(out_name + '{}.{}nn-{}.csv'.format(model_name, num_neighbors, data_name), index=False)


def find_neighbors_all(embeds, i2wordpos, model_name, out_name,
                   dist_metric='minkowski', num_neighbors=50):
    knn = NearestNeighbors(n_neighbors=num_neighbors, metric=dist_metric)
    print("fitting KNN ...")
    knn.fit(embeds)

    find_neighbors_helper(knn, embeds, i2wordpos, i2wordpos, 'train+dev',
                          out_name, model_name, num_neighbors)


def nearest_neighbors_optimal_distance(data_dir, model_name, out_name, use_all=False):
    trn_conn_embeds, dev_conn_embeds, trn_word2pos2i, dev_word2pos2i = utils.load_embeddings(data_dir, model_name)
    if use_all:
        in_embeds = np.concatenate((trn_conn_embeds, dev_conn_embeds), axis=0)
        in_word2pos2i = copy.deepcopy(trn_word2pos2i)
        for w in dev_word2pos2i:
            in_word2pos2i[w] = in_word2pos2i.get(w, dict())
            for t in dev_word2pos2i[w]:
                in_word2pos2i[w][t] = dev_word2pos2i[w][t] + len(trn_conn_embeds)
        in_i2wordpos= {in_word2pos2i[w][t]: (w, t) for w in in_word2pos2i for t in in_word2pos2i[w]}

    trn_i2wordpos = {trn_word2pos2i[w][t]: (w, t) for w in trn_word2pos2i for t in trn_word2pos2i[w]}
    dev_i2wordpos = {dev_word2pos2i[w][t]: (w, t) for w in dev_word2pos2i for t in dev_word2pos2i[w]}

    for m in ['cosine', 'euclidean']:
        print("Distance: {}".format(m))
        if not use_all:
            find_neighbors(trn_conn_embeds, dev_conn_embeds, trn_i2wordpos, dev_i2wordpos,
                            model_name + '.{}'.format(m), out_name, dist_metric=m)
        else:
            find_neighbors_all(in_embeds, in_i2wordpos, model_name + '.{}'.format(m), out_name,
                               dist_metric=m)
        print()


def get_wordvec_neighbors(data_dir, model_name, vec_name, out_name, num_neighbors=50):
    vecs = np.load('../resources/{}.vectorsF.npy'.format(vec_name))
    word2i = pickle.load(open('../resources/{}.vocabF.pkl'.format(vec_name), 'rb'))

    _, _, trn_word2pos2i, dev_word2pos2i = utils.load_embeddings(data_dir, model_name)
    uni_words = (set(trn_word2pos2i.keys()) | set(dev_word2pos2i.keys())) & set(word2i.keys())

    new_word2i = dict()
    new_vecs = []
    for i, w in enumerate(uni_words):
        new_vecs.append(vecs[word2i[w]])
        new_word2i[w] = i
    in_embeds = np.array(new_vecs)

    i2word = {i:w for w,i in new_word2i.items()}

    for m in ['cosine', 'euclidean']:
        print("Distance: {}".format(m))
        knn = NearestNeighbors(n_neighbors=num_neighbors, metric=m)
        print("fitting KNN ...")
        knn.fit(in_embeds)

        print("... getting {} {}-NN".format('train+dev', num_neighbors))
        neigh_data = []
        dists, neighs = knn.kneighbors(in_embeds)
        for i in range(len(in_embeds)):
            nlst = [(i2word[nidx], str(nd)) for nidx, nd in zip(neighs[i], dists[i])]
            neigh_data.append([i2word[i], json.dumps(nlst)])
        temp = pd.DataFrame(neigh_data, columns=['word', 'neighbor_lst'])
        temp.to_csv(out_name + '{}.{}nn-{}.csv'.format(model_name + '.{}'.format(m), num_neighbors, 'train+dev'), index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='What to do', required=True)
    parser.add_argument('-s', '--config_file', help='config file name', required=True)
    parser.add_argument('-d', '--data_dir', help='Location of the saved embeddings', required=False,
                        default='../data/vectors/')
    parser.add_argument('-b', '--balanced', help='Whether the original model was made with balanced sampling',
                        required=False, default='balanced')
    parser.add_argument('-o', '--out_name', help='Beginning of output file path', required=False)
    parser.add_argument('-v', '--vec_name', help='Name of the vectors file', required=False)
    args = vars(parser.parse_args())

    with open(args['config_file'], 'r') as f:
        config = dict()
        for l in f.readlines(): config[l.strip().split(":")[0]] = l.strip().split(":")[1]

    name = config['name']
    if args['balanced'] is not None:
        name += '-balanced'

    if args['mode'] == '1':
        print("NN FIT ONLY ON TRAIN")
        nearest_neighbors_optimal_distance(args['data_dir'], name, args['out_name'])
    elif args['mode'] == '2':
        print("NN FIT ON TRAIN+DEV")
        nearest_neighbors_optimal_distance(args['data_dir'], name, args['out_name'], use_all=True)
    elif args['mode'] == '3':
        print("NN FIT with word vectors {}".format(args['vec_name']))
        get_wordvec_neighbors(args['data_dir'], name, args['vec_name'], args['out_name'])



