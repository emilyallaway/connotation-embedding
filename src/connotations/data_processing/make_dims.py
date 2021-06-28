from parse_lexica import *
from dim_mapping_v2 import *
from nltk.corpus import wordnet as wn
from collections import Counter
import json, argparse
import pandas as pd
import numpy as np


###################
# set random seed #
np.random.seed(0)


def get_and_tag_words():
    cat2words, word2def, word2cats, _, word2src = load_general_inquirer()
    word2emos, emo2words, word2rel = load_nrc_emotion()
    word2pol, pol2words, _ = load_cwn()
    word2cat2score = load_dal()

    pos2words = {'N': set(), 'A': set(), 'V': set(), 'LY': set()}

    pos2word2cats = {'N': dict(), 'A': dict(), 'V': dict(), 'LY': dict()}
    pos2word2emos = {'N': dict(), 'A': dict(), 'V': dict(), 'LY': dict()}
    pos2word2pol = {'N': dict(), 'A': dict(), 'V': dict(), 'LY': dict()}
    pos2word2cat2score = {'N': dict(), 'A': dict(), 'V': dict(), 'LY': dict()}
    pos2word2src = {'N': dict(), 'A': dict(), 'V': dict(), 'LY': dict()}

    # split by POS from GI
    for w,d in word2def.items():
        if d is None or isinstance(d, float): continue
        t = None
        if 'LY' in d or 'adv' in d:
            t = 'LY'
        elif 'SUPV' in d or 'verb' in d:
            t = 'V'
        elif 'Modif' in d or 'adj' in d:
            t = 'A'
        elif 'Noun' in d or 'noun' in d:
            t = 'N'
        if t is not None:
            pos2words[t].add(w)
            pos2word2src[t][w] = 'GI' + word2src[w]
            pos2word2cats[t][w] = word2cats[w]

    # split by POS from CWN
    for w in word2pol:
        if '_' in w:
            t = w.split('_')[-1]
            w_only = ' '.join(w.split('_')[:-1])
        else:
            t = w.split('.')[-2]
            w_only = '.'.join(w.split('.')[:-2])

        k = None
        if 'n' in t:
            pos2words['N'].add(w)
            k = 'N'
        elif 'a' in t or 's' in t:
            pos2words['A'].add(w)
            k = 'A'
        elif 'v' in t:
            pos2words['V'].add(w)
            k = 'V'
        elif 'r' in t:
            pos2words['LY'].add(w)
            k = 'LY'

        if k is not None:
            if 'Cwn' not in pos2word2src[k].get(w_only, ''):
                s = (pos2word2src[k].get(w_only, '') + '-CWn').strip('-')
                pos2word2src[k][w_only] = s
            pos2word2pol[k][w] = word2pol[w]

    # split by POS from NRC
    for w, rlst in word2rel.items():
        rpos_lst = []
        for r in rlst + [w]:
            temp = [ss.pos() for ss in wn.synsets(r)]
            if len(temp) == 0: continue
            rpos_lst.append(Counter(temp).most_common(1)[0][0])
        if len(rpos_lst) == 0: continue
        rpos = Counter(rpos_lst).most_common(1)[0]
        t = None
        if 'v' in rpos:
            pos2words['V'].add(w)
            t = 'V'
        elif 'n' in rpos:
            pos2words['N'].add(w)
            t = 'N'
        elif 'a' in rpos or 's' in rpos:
            pos2words['A'].add(w)
            t = 'A'
        elif 'r' in rpos:
            pos2words['LY'].add(w)
            t = 'LY'
        if t is not None:
            if 'NRC' not in pos2word2src[t].get(w, ''):
                s = (pos2word2src[t].get(w, '') + '-NRC').strip('-')
                pos2word2src[t][w] = s
            pos2word2emos[t][w] = word2emos[w]

    # split by POS from DAL
    for w in word2cat2score:
        pos_lst = [ss.pos() for ss in wn.synsets(w)]
        for t in pos_lst:
            k = None
            if 'v' in t:
                pos2words['V'].add(w)
                k = 'V'
            elif 'n' in t:
                pos2words['N'].add(w)
                k = 'N'
            elif 'a' in t or 's' in t:
                pos2words['A'].add(w)
                k = 'A'
            elif 'r' in t:
                pos2words['LY'].add(w)
                k = 'LY'

            if k is not None:
                if'DAL' not in pos2word2src[k].get(w, ''):
                    s = (pos2word2src[k].get(w, '') + '-DAL').strip('-')
                    pos2word2src[k][w] = s
                pos2word2cat2score[k][w] = word2cat2score[w]

    args = {'word2def': word2def, 'word2cats': pos2word2cats, #GI
            'word2emos': pos2word2emos, #NRC
            'word2pol': pos2word2pol, # CWN
            'word2cat2score': pos2word2cat2score, # DAL
            'pos2words': pos2words,
            'word2src': pos2word2src}
    return args


def get_sim_cwn(w, t, cwn_words):
    for pw in cwn_words:
        if '_' in pw:
            pw_temp = pw.split('_')
            pw_t = pw_temp[-1]
            pw_only = ' '.join(pw_temp[:-1])
        elif '.' in pw:
            pw_temp = pw.split('.')
            pw_t = pw_temp[-2]
            pw_only = '.'.join(pw_temp[:-2])

        if pw_only == w and pw_t == t:
            return pw
    return None


def select_getter(k):
    if k == 'N':
        fn = get_noun_conn
    elif k == 'A':
        fn = get_adj_conn
    elif k == 'V':
        fn = get_verb_conn
    else:
        fn = None
    return fn


def update_confidence_etc(word2tag2conn, conn_dims, w, w_only, k, word2src):
    s = word2src[w]
    if w_only in word2src and w != w_only:
        s = s + '-' + word2src[w_only]

    conn_dims.source = s
    if 'H4' in s and 'Lvd' not in s:  # H4 only
        conn_dims.confidence = 0.6
    elif 'Lvd' in s and 'H4' not in s:  # Lvd only
        conn_dims.confidence = 0.4
    if 'GI' not in s:
        conn_dims.partial = 1

    word2tag2conn[w_only] = word2tag2conn.get(w_only, dict())
    word2tag2conn[w_only][k] = word2tag2conn[w_only].get(k, [])
    word2tag2conn[w_only][k].append(conn_dims)


def get_dims(args, use_all=True):
    word2tag2conn = dict()
    for k in args['word2cats']:
        if k not in ['N', 'A', 'V']: continue
        for w in args['word2cats'][k]: # GI words
            w_only = w.split("#")[0]

            if w == 'make, made sense':
                w_only = 'make sense'
            elif w == 'year, month old':
                w_only = 'old'
            elif w == 'make, have a go':
                w_only = 'have a go'
            elif w == 'pick on, a fight':
                w_only = 'pick on'

            fn = select_getter(k)

            if w_only in args['word2emos'][k]:
                word_emos = args['word2emos'][k][w_only]
            else:
                word_emos = []

            if w_only in args['word2cat2score'][k]:
                word_score = args['word2cat2score'][k][w_only]['im']
            else:
                word_score = 0.

            cwn_w = get_sim_cwn(w_only, k.lower(), args['word2pol'][k])#, args['pos2words'][k])
            if cwn_w is not None:
                word_pol = args['word2pol'][k][cwn_w]
            else:
                word_pol = 0.
            conn_dims = fn(wordcats=args['word2cats'][k][w], wordemos=word_emos,
                           wordpol=word_pol, wordscore=word_score) # a object of connotations
            # set confidence and source of labeling and whether partial
            update_confidence_etc(word2tag2conn, conn_dims, w, w_only, k, args['word2src'][k])
    # HERE: all words with any GI info have been labeled

    if not use_all:
        return word2tag2conn

    for k in args['word2emos']:
        if k not in ['N', 'A', 'V']: continue  # check pos is valid
        for w in args['word2emos'][k]: # NRC words
            if w in word2tag2conn and k in word2tag2conn[w]: continue # there's already a conn for word-pos combo

            fn = select_getter(k)

            word_emos = args['word2emos'][k][w] # emotion
            # factuality info
            if w in args['word2cat2score'][k]:
                word_score = args['word2cat2score'][k][w]['im']
            else:
                word_score = 0.
            # sentiment info
            cwn_w = get_sim_cwn(w, k.lower(), args['word2pol'][k])
            if cwn_w is not None:
                word_pol = args['word2pol'][k][cwn_w]
            else:
                word_pol = 0.

            conn_dims = fn(wordcats=[], wordemos=word_emos,
                           wordpol=word_pol, wordscore=word_score)  # a list of connotations
            update_confidence_etc(word2tag2conn, conn_dims, w, w, k, args['word2src'][k])
    # HERE: all words with any GI or emotion have been labeled

    for k in args['word2cat2score']:
        if k not in ['N', 'A', 'V']: continue  # check pos is valid
        for w in args['word2cat2score'][k]: # DAL words
            if w in word2tag2conn and k in word2tag2conn[w]: continue # already a conn

            fn = select_getter(k)

            word_score = args['word2cat2score'][k][w]['im']
            # sentiment info
            cwn_w = get_sim_cwn(w, k.lower(), args['word2pol'][k])
            if cwn_w is not None:
                word_pol = args['word2pol'][k][cwn_w]
            else:
                word_pol = 0.

            conn_dims = fn(wordcats=[], wordemos=[],
                           wordpol=word_pol, wordscore=word_score)  # a list of connotations
            update_confidence_etc(word2tag2conn, conn_dims, w, w, k, args['word2src'][k])
    # HERE: all words with any GI, emotion, or factuality have been labeled

    for k in args['word2pol']:
        if k not in ['N', 'A', 'V']: continue # check valid pos
        for w in args['word2pol'][k]: # CWN words
            if '_' in w:
                w_only = ' '.join(w.split('_')[:-1])
            else:
                w_only = '.'.join(w.split('.')[:-2])

            if w_only in word2tag2conn and k in word2tag2conn[w_only]: continue # already a conn

            fn = select_getter(k)

            word_pol = args['word2pol'][k][w]

            conn_dims = fn(wordcats=[], wordemos=[],
                           wordpol=word_pol, wordscore=0.)  # a list of connotations

            update_confidence_etc(word2tag2conn, conn_dims, w_only, w_only, k, args['word2src'][k])
    # HERE: all words with GI, emotion, factuality, or sentiment have been labeled

    return word2tag2conn


def assign_helper(data, indices, split2i, name):
    for i in indices:
        w = data[i][0]
        data[i].append(name)
        split2i[w] = split2i.get(w, [])
        split2i[w].append(i)


def check_and_fix(data, dict1, dict2, name1, name2):
    int_words = set(dict1.keys()) & set(dict2.keys())
    if len(int_words) != 0:
        for w in int_words:
            n = np.random.random()
            if n < 0.5:
                # move to dict1
                mv_from_dict = dict2
                mv_to_dict = dict1
                new_name = name1
            else:
                # move to dict2
                mv_from_dict = dict1
                mv_to_dict = dict2
                new_name = name2

            for i in mv_from_dict[w]:
                data[i][-1] = new_name

            mv_to_dict[w] = mv_to_dict[w] + mv_from_dict[w]
            del mv_from_dict[w]


def split_helper(data, train_split):
    #############################
    # assign fully labeled data #
    #############################
    trn_num = int(len(data) * train_split)  # train is only fully labeled instances, train_split%
    dev_num = int((len(data) - trn_num) / 2)  # dev gets % of fully labeled instances

    indices_full = [i for i in range(len(data))]
    np.random.shuffle(indices_full)

    trn2i = dict()
    dev2i = dict()
    test2i = dict()
    assign_helper(data, indices_full[:trn_num], trn2i, 'train')
    assign_helper(data, indices_full[trn_num: trn_num + dev_num], dev2i, 'dev')
    assign_helper(data, indices_full[trn_num + dev_num:], test2i, 'test')

    check_and_fix(data, trn2i, dev2i, 'train', 'dev')
    assert len(set(trn2i.keys()) & set(dev2i.keys())) == 0
    check_and_fix(data, trn2i, test2i, 'train', 'test')
    assert len(set(trn2i.keys()) & set(test2i.keys())) == 0
    check_and_fix(data, dev2i, test2i, 'dev', 'test')
    assert len(set(dev2i.keys()) & set(test2i.keys())) == 0


def split_full_and_partial(data, train_split):
    data_full = []
    data_partial = []
    for r in data:
        if r[-1] == 0:
            data_full.append(r)
        else:
            data_partial.append(r)

    split_helper(data_full, train_split)
    split_helper(data_partial, train_split)
    new_data = data_full + data_partial
    return new_data


def assign_data_split(data, train_split):
    noun_data = []
    adj_data = []
    verb_data = []
    for r in data:
        if r[1] == 'N':
            noun_data.append(r)
        elif r[1] == 'A':
            adj_data.append(r)
        else:
            verb_data.append(r)

    # split nouns, adjectives, and verbs into train/dev/test
    split_noun_data = split_full_and_partial(noun_data, train_split)
    split_adj_data = split_full_and_partial(adj_data, train_split)
    split_verb_data = split_full_and_partial(verb_data, train_split)

    new_data = split_noun_data + split_adj_data + split_verb_data
    return new_data


def label_words_simple(outname):
    myargs = get_and_tag_words()
    word2tag2conn = get_dims(myargs, use_all=False) # NOTE: this is very slow, unsurprisingly

    data = []
    for word in word2tag2conn:
        for t in word2tag2conn[word]:
            data.append([word, t, json.dumps([c.get() for c in word2tag2conn[word][t]])])

    df = pd.DataFrame(data, columns=['word', 'POS', 'conn'])
    df.to_csv(outname, index=False)


def label_words_complete(outname, train_split=0.6):
    myargs = get_and_tag_words()
    word2tag2conn = get_dims(myargs)  # NOTE: this is very slow, unsurprisingly

    data = []
    for word in word2tag2conn:
        for t in word2tag2conn[word]:
            temp = dict()
            for k in word2tag2conn[word][t][0].get():
                if k != 'Emo':
                    temp[k] = 0.
                else:
                    temp[k] = [0.] * 8

            src = ''
            confidence = word2tag2conn[word][t][0].confidence
            par = 1
            for cv in word2tag2conn[word][t]:
                cv_dict = cv.get()
                for k in cv_dict:
                    if k != 'Emo':
                        temp[k] += cv_dict[k]
                    else: # k = Emo
                        for i,e in enumerate(cv_dict[k]):
                            if e != 0:
                                temp[k][i] = e

                if ('H4' in src and 'Lvd' in cv.source) or ('Lvd' in src and 'H4' in cv.source):
                    # only reason to change confidence to 1 is if get both H4 and Lvd
                    confidence = 1.0

                for s in cv.source.split('-'):
                    if s not in src:
                        src += '-' + s

                if cv.partial == 0:
                    # only change partial if get some example which is not partial
                    par = 0

            src = src.strip('-')

            for k in temp:
                if k != 'Emo':
                    temp[k] = temp[k] / len(word2tag2conn[word][t])

            data.append([word, t, json.dumps(temp), confidence, src, par])

    print("got dims")
    # assign train/dev/test split
    new_data = assign_data_split(data, train_split)
    print("assigned splits")

    df = pd.DataFrame(new_data, columns=['word', 'POS', 'conn', 'confidence', 'source', 'partial?', 'train/dev/test'])
    df.to_csv(outname, index=False)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='What to do', required=True)
    parser.add_argument('-o', '--outf', help='Full name of the output file', required=True)
    args = vars(parser.parse_args())

    if args['mode'] == '1':
        label_words_simple(args['outf'])
    elif args['mode'] == '2':
        label_words_complete(args['outf'])
    else:
        print("ERROR: doing nothing")