import pandas as pd
import json, sys

MODEL_NAME = 'BiLSTMRelEmbedderVERB-cn-w7-s4-42w-norm-learn-20r-balanced'

DATA_DIR = '../../../data'
DATA_FILE = 'train+dev{}.[METRIC].50nn-train+dev.csv'.format(MODEL_NAME)
CN_DATA_FILE = 'numberbatch.[METRIC].50nn-train+dev.csv'
G_DATA_FILE = 'glove.[METRIC].50nn-train+dev.csv'
CONN_FILE = '../../../data/lexicon/noun_adj_connotation_lexicon.csv'


def load_data(dist_metric):
    df = pd.read_csv(DATA_DIR + DATA_FILE.replace('[METRIC]', dist_metric))
    word2pos2neighs = dict()
    for i in df.index:
        row = df.iloc[i]
        w = row['word']
        if row['POS'] != 'N' and row['POS'] != 'A': continue
        word2pos2neighs[w] = word2pos2neighs.get(w, dict())
        word2pos2neighs[w][row['POS']] = list(map(lambda x: x[0], json.loads(row['neighbor_lst'])))
    return word2pos2neighs


def load_data_wordvecs(data_file, dist_metric):
    df = pd.read_csv(DATA_DIR + data_file.replace('[METRIC]', dist_metric))
    word2neighs = dict()
    for i in df.index:
        row = df.iloc[i]
        # if row['POS'] != 'N' and row['POS'] != 'A': continue
        w = row['word']
        word2neighs[w] = list(map(lambda x: x[0], json.loads(row['neighbor_lst'])))
    return word2neighs



def load_connotations():
    df = pd.read_csv(CONN_FILE, keep_default_na=False)
    word2pos2conn = dict()
    for i in df.index:
        row = df.iloc[i]
        w = row['word']
        word2pos2conn[w] = word2pos2conn.get(w, dict())
        word2pos2conn[w][row['POS']] = json.loads(row['conn'])
    return word2pos2conn


def analyze_neighbor_conns_EMO(ratio_val1, ratio_val2, word2pos2conn, word2pos2neighs):
    pol2ratio = dict()
    emos = ['anger', 'antic', 'disgust', 'fear', 'joy', 'sad', 'surp', 'trust']
    for e in emos:
        for p in [0, 1]:
            pol2ratio['{}-{}'.format(e, p)] = [0, 0]

    for w in word2pos2neighs:
        for t in word2pos2neighs[w]:
            nlst = word2pos2neighs[w][t]
            for i, e in enumerate(emos):
                cv1, cv2 = 0, 0
                for nw, nt in nlst:
                    if nw == w and nt == t: continue

                    if word2pos2conn[nw][nt]['Emo'][i] == ratio_val1:
                        cv1 +=1
                    elif word2pos2conn[nw][nt]['Emo'][i] == ratio_val2:
                        cv2 += 1
                p = word2pos2conn[nw][nt]['Emo'][i]
                k = '{}-{}'.format(e, int(p))
                pol2ratio[k] = pol2ratio.get(k, [0, 0])
                pol2ratio[k][0] = pol2ratio[k][0] + cv1
                pol2ratio[k][1] = pol2ratio[k][1] + cv2
    return pol2ratio


def analyze_neighbors_wordvecs_EMO(ratio_val1, ratio_val2, word2pos2conn, word2neighs):
    pol2ratio = dict()
    emos = ['anger', 'antic', 'disgust', 'fear', 'joy', 'sad', 'surp', 'trust']
    for e in emos:
        for p in [0, 1]:
            pol2ratio['{}-{}'.format(e, p)] = [0, 0]

    for w in word2neighs:
        nlst = word2neighs[w]
        for i, e in enumerate(emos):
            cv1, cv2 = 0, 0
            for nw in nlst:
                if nw == w: continue
                my_conn = get_conn('Emo', word2pos2conn[nw], ei=i)
                if my_conn == ratio_val1:
                    cv1 += 1
                elif my_conn == ratio_val2:
                    cv2 += 1
                k = '{}-{}'.format(e, int(my_conn))
                pol2ratio[k] = pol2ratio.get(k, [0, 0])
                pol2ratio[k][0] = pol2ratio[k][0] + cv1
                pol2ratio[k][1] = pol2ratio[k][1] + cv2
    return pol2ratio


def analyze_neighbor_conns(ratio_val1, ratio_val2, conn_dim, word2pos2conn, word2pos2neighs, verbose=False):
    pol2ratio = dict()
    pol2max = dict()
    pol2min = dict()
    for w in word2pos2neighs:
        for t in word2pos2neighs[w]:
            nlst = word2pos2neighs[w][t]
            cv1, cv2 = 0, 0
            for nw, nt in nlst:
                if nw == w and nt == t: continue
                if mysign(word2pos2conn[nw][nt][get_dim_name(conn_dim, nt)]) == ratio_val1:
                    cv1 += 1
                elif mysign(word2pos2conn[nw][nt][get_dim_name(conn_dim, nt)]) == ratio_val2:
                    cv2 += 1
            p = mysign(word2pos2conn[w][t][get_dim_name(conn_dim, t)])

            pol2ratio[p] = pol2ratio.get(p, [0, 0])
            pol2ratio[p][0] = pol2ratio[p][0] + cv1
            pol2ratio[p][1] = pol2ratio[p][1] + cv2

            pol2max[p] = pol2max.get(p, [0, None, None])
            pol2min[p] = pol2min.get(p, [float('inf'),None, None])
            if abs(cv1 - cv2) > pol2max[p][0]:
                pol2max[p] = [abs(cv1 - cv2), (cv1, cv2), (w, t)]
            if abs(cv1 - cv2) < pol2min[p][0]:
                pol2min[p] = [abs(cv1 - cv2), (cv1, cv2), (w, t)]
    if verbose:
        print("ratio: words {}: words {}".format(ratio_val1, ratio_val2))
        for p in pol2ratio:
            print("   for p={}:  {}:{}".format(p, pol2ratio[p][0], pol2ratio[p][1]))
            print("              max= {}:{} with word {}".format(pol2max[p][1][0], pol2max[p][1][1], pol2max[p][2]))
            print("              min= {}:{} with word {}".format(pol2min[p][1][0], pol2min[p][1][1], pol2min[p][2]))
        print()
    return pol2ratio


def analyze_neighbors_wordvecs(ratio_val1, ratio_val2, conn_dim, word2pos2conn, word2neighs,
                               verbose=False):
    pol2ratio = dict()
    pol2max = dict()
    pol2min = dict()
    for w in word2neighs:
        nlst = word2neighs[w]
        cv1, cv2 = 0, 0
        for nw in nlst:
            if nw == w: continue
            if get_conn(conn_dim, word2pos2conn[nw]) == ratio_val1:
                cv1 += 1
            elif get_conn(conn_dim, word2pos2conn[nw]) == ratio_val2:
                cv2 += 1
        p = get_conn(conn_dim, word2pos2conn[w])

        pol2ratio[p] = pol2ratio.get(p, [0, 0])
        pol2ratio[p][0] = pol2ratio[p][0] + cv1
        pol2ratio[p][1] = pol2ratio[p][1] + cv2

        pol2max[p] = pol2max.get(p, [0, None, None])
        pol2min[p] = pol2min.get(p, [float('inf'), None, None])
        if abs(cv1 - cv2) > pol2max[p][0]:
            pol2max[p] = [abs(cv1 - cv2), (cv1, cv2), w]
        if abs(cv1 - cv2) < pol2min[p][0]:
            pol2min[p] = [abs(cv1 - cv2), (cv1, cv2), w]

    if verbose:
        print("ratio: words {}: words {}".format(ratio_val1, ratio_val2))
        for p in pol2ratio:
            print("   for p={}:  {}:{}".format(p, pol2ratio[p][0], pol2ratio[p][1]))
            print("              max= {}:{} with word {}".format(pol2max[p][1][0], pol2max[p][1][1], pol2max[p][2]))
            print("              min= {}:{} with word {}".format(pol2min[p][1][0], pol2min[p][1][1], pol2min[p][2]))
        print()
    return pol2ratio


def mysign(n):
    if n > 0: return 1
    elif n < 0: return -1
    else: return 0


def get_dim_name(d, t):
    if d == 'Social Val':
        if t == 'N': return 'Social Stat'
        else: return 'Value'
    elif d == 'Impact':
        if t == 'N': return 'Social Impact'
        else: return 'Impact'
    else: return d


def get_conn(conn_dim, wordconn, ei=0):
    if conn_dim != 'Emo':
        n = 0.
        for t in wordconn:
            d = get_dim_name(conn_dim, t)
            n += mysign(wordconn[t][d])
    else:
        n = 0.
        for t in wordconn:
            n += wordconn[t]['Emo'][ei]
    return mysign(n / len(wordconn))


def get_neighbor_comparisons(m='euclidean'):

    word2pos2conn = load_connotations()
    word2pos2neighs = load_data(m)
    cn_word2neighs = load_data_wordvecs(CN_DATA_FILE, m)
    g_word2neighs = load_data_wordvecs(G_DATA_FILE, m)

    data = []
    for conn_d in ['Social Val', 'Polite', 'Impact', 'Fact', 'Sent']:
        p2temp = {1: [conn_d, 1], -1: [conn_d, -1], 0: [conn_d, 0]}
        for n1, n2 in [(1, -1), (1, 0), (-1, 0)]:
            c_word2pol = analyze_neighbor_conns(n1, n2, conn_d, word2pos2conn, word2pos2neighs)
            cn_word2pol = analyze_neighbors_wordvecs(n1, n2, conn_d, word2pos2conn, cn_word2neighs)
            g_word2pol = analyze_neighbors_wordvecs(n1, n2, conn_d, word2pos2conn, g_word2neighs)
            for p in p2temp:
                for w2p in [c_word2pol, cn_word2pol, g_word2pol]:
                    p2temp[p] = p2temp[p] + [float(w2p[p][0]) / w2p[p][1]]
        for p in [1, -1, 0]:
            data.append(p2temp[p])

    emos = ['anger', 'antic', 'disgust', 'fear', 'joy', 'sad', 'surp', 'trust']

    emo_p2temp = dict()
    for e in emos:
        emo_p2temp[e] = {'{}-0'.format(e): ['Emo-{}'.format(e), 0], '{}-1'.format(e): ['Emo-{}'.format(e), 1]}

    for n1, n2 in [(1, 0), (1, 1), (0, 0)]:#[(1, -1), (1, 0), (-1, 0)]:
        c_word2pol = analyze_neighbor_conns_EMO(n1, n2, word2pos2conn, word2pos2neighs)
        cn_word2pol = analyze_neighbors_wordvecs_EMO(n1, n2, word2pos2conn, cn_word2neighs)
        g_word2pol = analyze_neighbors_wordvecs_EMO(n1, n2, word2pos2conn, g_word2neighs)
        for e in emos:
            p2temp = emo_p2temp[e]
            for p in p2temp:
                for w2p in [c_word2pol, cn_word2pol, g_word2pol]:

                    p2temp[p] = p2temp[p] + [(float(w2p[p][0]) / w2p[p][1]) if w2p[p][1] > 0 else w2p[p][0]]

    temp = {'Avg-1': ['Emo', 'Avg-1'], 'Avg-0': ['Emo', 'Avg-0']}
    for p in [0, 1]:
        for i in range(2, 11):
            n = sum([emo_p2temp[e]['{}-{}'.format(e, p)][i] for e in emos]) / float(len(emos))
            temp['Avg-{}'.format(p)].append(n)

    for e in ['Avg'] + emos:
        for p in [0, 1]:
            k = '{}-{}'.format(e, p)
            if k in temp:
                data.append(temp[k])
            else:
                data.append(emo_p2temp[e][k])

    cols = ['dim', 'base_pol'] + ['{}_{}'.format(n, c) for c in ['+:-', '+:=', '-:='] for n in ['conn', 'cn', 'glove']]
    df = pd.DataFrame(data, columns=cols)
    out_name = DATA_DIR + 'RATIOS_{}.{}.50nn-train+dev.csv'.format(MODEL_NAME, m)
    df.to_csv(out_name, index=False)


def get_neighbors_examples(thresh=.5, vec_name='glove', m='cosine'):
    model_name = '{}.{}.50nn-train+dev.csv'.format(MODEL_NAME, m)
    out_name = 'interesting_50neighbors.{}.{}.{}.thresh{}.csv'.format(model_name, m, vec_name, thresh)
    e_df = pd.read_csv(DATA_DIR + 'train+dev' + model_name)
    if vec_name == 'glove':
        vec_file = G_DATA_FILE.replace('[METRIC]', m)
    else:
        vec_file = CN_DATA_FILE.replace('[METRIC', m)

    g_df = pd.read_csv(DATA_DIR + vec_file)
    conn_df = pd.read_csv(CONN_FILE)

    data = []
    for d in ['Social Val', 'Polite', 'Impact', 'Fact', 'Sent']:
        n = len(data)
        print("DIM = {}".format(d))
        word2pos = dict()
        for i in conn_df.index:
            row = conn_df.iloc[i]
            dim = get_dim_name(d, row['POS'])

            if json.loads(row['conn']).get(dim, 0) == 0: continue
            e_row_df = e_df.loc[(e_df['word'] == row['word']) & (e_df['POS'] == row['POS'])]
            g_row_df = g_df.loc[g_df['word'] == row['word']]
            if len(e_row_df) == 0 or len(g_row_df) == 0: continue
            e_nlst = set(map(lambda x: x[0][0], json.loads(e_row_df.iloc[0]['neighbor_lst']))) - {row['word']}
            g_nlst = set(map(lambda x: x[0], json.loads(g_row_df.iloc[0]['neighbor_lst']))) - {row['word']}
            if len(e_nlst - g_nlst) / 50. > thresh:
                word2pos[row['word']] = word2pos.get(row['word'], [])
                word2pos[row['word']].append(row['POS'])

        for w in word2pos:
            for t in word2pos[w]:
                e_row_val = e_df.loc[(e_df['word'] == w) & (e_df['POS'] == t)].iloc[0]['neighbor_lst']
                e_nlst = set(map(lambda x: x[0][0], json.loads(e_row_val))) - {w}

                g_row_val = g_df.loc[g_df['word'] == w].iloc[0]['neighbor_lst']
                g_nlst = set(map(lambda x: x[0], json.loads(g_row_val))) - {w}
                data.append([d, w, t,
                             json.dumps(list(e_nlst - g_nlst)),
                             json.dumps(list(g_nlst - e_nlst)),
                             json.dumps(list(e_nlst & g_nlst))
                             ])
        print("... {} new examples".format(len(data) - n))

    cols = ['conn_dim', 'word', 'POS', 'only_conn', 'only_wordvec', 'both']
    pd.DataFrame(data, columns=cols).to_csv(DATA_DIR + out_name, index=False)


if __name__ == '__main__':
    if sys.argv[1] == '1':
        for m in ['cosine', 'euclidean']:
            print(m)
            get_neighbor_comparisons(m=m)
    elif sys.argv[1] == '2':
        get_neighbors_examples(m='euclidean')
    else:
        print("ERROR. doing nothing")
