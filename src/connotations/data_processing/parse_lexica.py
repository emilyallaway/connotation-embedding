import pandas as pd
import os, sys, re

LEXICON_PATH = ''


def load_general_inquirer(categories=None, fname='harvard-general-inquirer/inquirerbasic.csv'):
    print("loading general inquirer ...")
    cat2words = dict()
    word2def = dict()
    word2cats = dict()
    allwords = set()
    word2src = dict()

    df = pd.read_csv(os.path.join(LEXICON_PATH, fname), encoding='utf-8', low_memory=False)
    for i in df.index:
        row = df.iloc[i]
        w = row['Entry'].lower()

        d = row['Othtags']
        if not pd.isna(row['Defined']) and row['Defined'] != '|':
            temp = row['Defined'].split(':')
            if len(temp) > 1 and temp[1].strip().startswith("\""):
                a = temp[1].strip()
                w = a[1: a[1:].find("\"") + 1].lower() # change the word, b/c lexicon in lemmatized
                w = ' '.join(re.sub(r'\([^()]*\)', '', w).strip(', ').split()) # remove text between parens

        if d == 'Handels':
            d = row['Defined'] if not pd.isna(row['Defined']) else None
        word2def[w] = d

        allwords.add(w.split('#')[0])

        word2cats[w] = word2cats.get(w, set())
        for c in df.columns:
            if categories is not None and c not in categories: continue
            cat = row[c]
            cat2words[cat] = cat2words.get(cat, set())
            cat2words[cat].add(w)

            word2cats[w].add(cat)

        word2src[w] = row['Source']
    return cat2words, word2def, word2cats, allwords, word2src


def load_nrc_emotion(lex_level='Senselevel',
                     fname='NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-[type]-v0.92.txt'):
    print("loading nrc emotion lexicon ...")
    word2emos = dict()
    emo2words = dict()
    word2rel = dict()

    f = open(os.path.join(LEXICON_PATH, fname.replace('[type]', lex_level)), 'r', encoding='utf-8')
    lines = f.readlines()
    for l in lines:
        temp = l.strip().split('\t')
        if len(temp) <= 1: continue

        e = temp[1]
        if temp[2] == '1':
            if 'Sense' in lex_level:
                # see ReadMe in the folder if need to write this code
                w, rel_str = temp[0].split('--')
                rel_lst = list(map(lambda x: x.strip(), rel_str.split(',')))
                word2rel[w] = rel_lst
            else:
                w = temp[0]

            word2emos[w] = word2emos.get(w, set())
            word2emos[w].add(e)

            emo2words[e] = emo2words.get(e, set())
            emo2words[e].add(w)
    return word2emos, emo2words, word2rel


def load_cwn(pos_cutoff=0.25, neg_cutoff=-0.25,fname='connotation-wordnet/cwn.txt'):
    print("loading connotation wordnet ...")
    word2pol = dict()
    pol2words = dict()
    allwords = set()

    f = open(os.path.join(LEXICON_PATH, fname), 'r')
    lines = f.readlines()
    for l in lines:
        score, w = l.strip().split('\t')
        if '_' in w:
            allwords.add(' '.join(w.split('_')[:-1]))
        else: # . in w
            allwords.add('.'.join(w.split('.')[:-2]))

        if float(score) >= pos_cutoff:
            p = 1
        elif float(score) <= neg_cutoff:
            p = -1
        else:
            p = 0
        word2pol[w] = float(score)#p
        pol2words[p] = pol2words.get(p, set())
        pol2words[p].add(w)

    return word2pol, pol2words, allwords


def load_dal(fname='Whissel-DAL/dictionary_English_readable.txt'):
    print("loading dictionary of affect in language ...")
    word2cat2score = dict()

    f = open(os.path.join(LEXICON_PATH, fname), encoding='utf-8')
    lines = f.readlines()

    for l in lines:
        w, pl, ac, im = l.strip().split()
        word2cat2score[w] = {'pl': float(pl), 'ac': float(ac), 'im': float(im)}

    return word2cat2score


def get_vocab():
    _, _, _, gi_allwords, _ = load_general_inquirer()
    word2emos, _, _ = load_nrc_emotion()
    _, _,  cwn_allwords = load_cwn()
    word2cat2score = load_dal()

    allwords = gi_allwords | set(word2emos.keys()) | cwn_allwords | set(word2cat2score.keys())

    outf = open(os.path.join(LEXICON_PATH, 'cwn-gi-emolex-dal-vocab.txt'), 'w', encoding='utf-8')
    for w in allwords:
        outf.write('{}\n'.format(w))


if __name__ == '__main__':
    if sys.argv[1] == '1':
        get_vocab()





