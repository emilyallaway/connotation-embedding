
mydims = {'N': {'Use': 'Usefulness', 'EmoA': 'Emotional Association', 'SoIm': 'Societal Impact',
                'P': 'Politeness', 'SoSt': 'Social Status', 'F': 'Factuality'},
          'A': {'V': 'Value', 'P': 'Politeness', 'EmoA': 'Emotional Association', 'Im': 'Impact',
                'F': 'Factuality'}}

emo2dim = {'anger': 0, 'anticip': 1, 'disgust': 2, 'fear': 3, 'joy': 4,
           'sadness': 5, 'surprise': 6, 'trust': 7}

dim2emo = {0: 'anger', 1: 'anticip', 2: 'disgust', 3: 'fear', 4: 'joy',
           5: 'sadness', 6: 'surprise', 7: 'trust'}


def compute_weighted_score(weight_lst, vals):
    vpos = sum([vals[i][0] * weight_lst[i] for i in range(len(weight_lst))])
    vneg = sum([vals[i][1] * weight_lst[i] for i in range(len(weight_lst))])

    if vpos > vneg and (vpos != 0 or vneg != 0):
        return vpos
    elif vneg >= vpos and (vpos != 0 or vneg != 0):
        return -vneg
    else:
        return 0


def myround(n):
    if n <= -0.25:
        return -1
    elif n >= 0.25:
        return 1
    else:
        return 0


def get_social_status_power(wordcats):
    p = [0, 0]
    if 'PowGain' in wordcats:
        if 'Negativ' in wordcats or 'Hostile' in wordcats \
                or 'Weak' in wordcats:
            p[1] += 1.
        else:
            p[0] += 1.
    elif 'PowLoss' in wordcats:
        if 'Positiv' in wordcats:
            p[0] += 1.
        else:
            p[1] += 1.
    if 'PowEnds' in wordcats:
        if 'Weak' in wordcats:
            p[1] += .5
        else:
            p[0] += .5

    if 'PowCon' in wordcats:
        if 'Negativ' not in wordcats and 'Hostile' not in wordcats and 'Weak' not in wordcats:
            p[0] += .75
        else:
            p[1] += .75
    elif 'PowCoop' in wordcats:
        if 'Negativ' in wordcats or 'Weak' in wordcats or 'Hostile' in wordcats:
            p[1] += .75
        else:
            p[0] += .75
    if 'PowAuPt' in wordcats:
        p[0] += 1.
    elif 'PowPt' in wordcats:
        if 'Power' in wordcats:
            p[0] += 1
        else:
            p[1] += 1

    if 'PowAuth' in wordcats:
        if 'Negativ' in wordcats or 'Weak' in wordcats or 'Hostile' in wordcats:
            p[1] += .75
        else:
            p[0] += .75
    if 'PowOth' in wordcats:
        if 'Positiv' in wordcats or ('Positiv' in wordcats and 'Strong' in wordcats) or \
                ('Positiv' in wordcats and 'Power' in wordcats and 'Negativ' not in wordcats):
            p[0] += .5
        elif 'Negativ' in wordcats or 'Weak' in wordcats:
            p[1] += .5
    return p


def get_social_status_moral(wordcats):
    temp = [0., 0.]
    # rectitude
    if 'RcEthic' in wordcats:
        if 'Negativ' in wordcats or 'Hostile' in wordcats or 'Weak' in wordcats or ('Submit' in wordcats and 'Positiv' not in wordcats):
            temp[1] += .5
        else:
            temp[0] += .5
    for c,v in [('RcRelig', .5), ('RcGain', 1.), ('RcEnds', 1)]:
        if c in wordcats:
            if 'Negativ' in wordcats or 'Hostile' in wordcats or 'Weak' in wordcats :
                temp[1] += v
            else:
                temp[0] += v
    if 'RcLoss' in wordcats and 'RcGain' not in wordcats: # mutually exclusive
        temp[1] += 1

    # vice and virtue
    if 'Virtue' in wordcats:
        temp[0] += 1
    elif 'Vice' in wordcats:
        temp[1] += 1

    return temp


def get_social_status_material(wordcats):
    temp = [0., 0.]
    if 'WltPt' in wordcats:
        if 'Strong' in wordcats or 'Power' in wordcats or 'COLL' in wordcats or 'SocRel' in wordcats or 'Active' in wordcats:
            temp[0] += 1
        elif 'Negativ' in wordcats or 'Weak' in wordcats:
            temp[1] += 1

    if 'WltTran' in wordcats:
        if 'Negativ' in wordcats or 'Weak' in wordcats:
            temp[1] += .5
        else:
            temp[0] += .5

    if 'WltOth' in wordcats:
        if 'Strong' in wordcats or 'Power' in wordcats:
            temp[0] += 1
        elif 'Weak' in wordcats:
            temp[1] += 1

        for c in ['Food', 'Object', 'Doctrin', 'Academ', 'Work', 'NatrObj', 'Vehicle']:
            if c in wordcats:
                temp[0] += .5
                break
        if 'Econ@' in wordcats:
            if 'Positiv' in wordcats:
                temp[0] += .5
            elif 'Negativ' in wordcats:
                temp[1] += .5

    return temp


def get_social_status_ability(wordcats):
    temp = [0., 0.]
    # enlightenment & skills
    if 'Goal' in wordcats:
        if 'Negativ' in wordcats:
            temp[1] += .5
        else:
            temp[0] += .5

    if 'EnlPt' in wordcats:
        if 'Submit' in wordcats or 'Passive' in wordcats:
            temp[1] += 1
        else:
            temp[0] += 1
    for c,v in [('EnlGain', 1), ('SklAsth', 0.5)]:
        if c in wordcats:
            if 'Negativ' in wordcats:
                temp[1] += v
            else:
                temp[0] += v
    if 'EnlOth' in wordcats:
        if 'Negativ' in wordcats or 'Weak' in wordcats:
            temp[1] += .5
        elif 'Strong' in wordcats or 'Power' in wordcats or 'Active' in wordcats:
            temp[0] += .5
    if 'EnlLoss' in wordcats: # mutually exclusive
        if 'Postiv' in wordcats:
            temp[0] += 1
        else:
            temp[1] += 1

    if 'SklPt' in wordcats:
        if 'Exprsv' in wordcats or 'Submit' in wordcats or 'Active' in wordcats or \
                ('Strong' in wordcats and 'Power' not in wordcats) or 'Weak' in wordcats:
            temp[1] += .5
        elif 'Legal' in wordcats or 'COLL' in wordcats or 'Power' in wordcats:
            temp[0] += .5

    if 'SklOth' in wordcats:
        if 'Weak' in wordcats or ('Strong' in wordcats and 'Power' not in wordcats) or \
                ('Strong' in wordcats and 'Active' in wordcats):
            temp[1] += .5
        else:
            temp[0] += .5

    return temp


def get_social_status(wordcats):
    vals = []
    # power
    vals.append(get_social_status_power(wordcats))
    # moral
    vals.append(get_social_status_moral(wordcats))
    # material
    vals.append(get_social_status_material(wordcats))
    # ability
    vals.append(get_social_status_ability(wordcats))

    weight_lst = [2/7, 2/7, 2/7, 1/7]
    return compute_weighted_score(weight_lst, vals)


def get_politeness(wordcats):
    vals = []
    # respect
    temp = [0., 0.]
    if 'RspGain' in wordcats:
        temp[0] += 1
    elif 'RspLoss' in wordcats:
        temp[1] += 1
    if 'RspOth' in wordcats:
        if 'Negativ' in wordcats or 'Hostile' in wordcats or ('Negativ' in wordcats and 'Weak' in wordcats) or \
                ('Negativ' in wordcats and 'Submit' in wordcats):
            temp[1] += .5
        else:
            temp[0] += .5
    vals.append(temp)

    # affection
    temp = [0., 0.]
    if 'AffGain' in wordcats:
        temp[0] += 1
    elif 'AffLoss' in wordcats and 'Hostile' in wordcats:
        temp[1] += .75
    if 'AffOth' in wordcats:
        if 'Negativ' in wordcats and 'HU' in wordcats:
            temp[1] += .5
        else:
            temp[0] += .5
    vals.append(temp)

    # participants
    temp = [0., 0.]
    for c in ['WlbPt', 'SklPt', 'EnlPt', 'Relig']:
        if c in wordcats:
            if 'Negativ' in wordcats:
                temp[1] += .75
            else:
                temp[0] += .75
    if 'WltPt' in wordcats:
        if 'Negativ' in wordcats:
            temp[1] += .75
        elif 'Positiv' in wordcats:
            temp[0] += .75

    if 'Polit@' in wordcats and 'HU' in wordcats:
        if 'Negativ' in wordcats and 'Hostile' in wordcats:
            temp[1] += .5
        else:
            temp[0] += .5

    for c in ['Milit', 'Legal']:
        if c in wordcats and 'HU' in wordcats:
            if 'Negativ' in wordcats or 'Hostile' in wordcats or 'Weak' in wordcats:
                temp[1] += .75
            elif 'Positiv' in wordcats:
                temp[0] += .75
    if 'Academ' in wordcats:
        temp[0] += .75
    if 'Doctrin' in wordcats:
        if 'Negativ' in wordcats:
            temp[1] += .75
        else:
            temp[0] += .75

    vals.append(temp)

    weight_lst = [3/6, 1/6, 2/6]
    return compute_weighted_score(weight_lst, vals)


def get_factuality(wordscore):
    return myround(wordscore - 2)  # noramlizes first to range [-1, 1]


def get_emo_association(wordemos):
    vals = [0] * 8
    for e in emo2dim:
        if e in wordemos:
            vals[emo2dim[e]] = 1.
    return vals


def get_sentiment(wordpol):
    return myround(wordpol * 2 - 1)  # noramlizes first to range [-1, 1]


def get_helper(check_cats, wordcats):
    temp = [0., 0.]
    for c, v in check_cats:
        if c in wordcats:
            if 'Negativ' in wordcats or 'Weak' in wordcats or 'Hostile' in wordcats:
                temp[1] += v
            else:
                temp[0] += v
    return temp


def get_usefulness_value(wordcats):
    vals = []
    # means
    temp = [0., 0.]
    for c in ['Means', 'MeansLw']:
        if c in wordcats:
            if 'Negativ' in wordcats or 'Weak' in wordcats:
                temp[1] += 1
            else:
                temp[0] += 1
    if 'Fail' in wordcats:
        temp[1] += 1
    if 'Solve' in wordcats:
        if 'Negativ' in wordcats or 'Weak' in wordcats:
            temp[1] += .5
        else:
            temp[0] += .5
    for c in ['EndsLw', 'Try']:
        if c in wordcats:
            if 'Negativ' in wordcats:
                temp[1] += .5
            else:
                temp[0] += .5
    if 'Goal' in wordcats:
        if 'Negativ' in wordcats:
            temp[1] += 1
        else:
            temp[0] += 1
    vals.append(temp)

    # skills
    temp = get_helper([('SklOth', .5)], wordcats)
    if 'SklPt' in wordcats:
        temp[0] += .75
    vals.append(temp)

    # enlightenment
    temp = get_helper([('EnlOth', 0.5)], wordcats)
    for c in ['EnlGain', 'EnlEnds', 'EnlPt']:
        if c in wordcats:
            temp[0] += 1
    if 'EnlLoss' in wordcats and 'EnlGain' not in wordcats:
        temp[1] += 1
    vals.append(temp)

    # wealth
    temp = get_helper([('WltPt', 1), ('WltTran', 0.5), ('WltOth', 0.5)], wordcats)
    if 'Quality' in wordcats:
        if 'Econ@' in wordcats or 'Positiv' in wordcats:
            temp[0] += .5
        elif 'Negativ' in wordcats:
            temp[1] += .5
    vals.append(temp)

    # well-being
    temp= [0., 0.]
    if 'WlbPhys' in wordcats:
        if 'Negativ' in wordcats or 'Hostile' in wordcats or ('Weak' in wordcats and 'Submit' in wordcats) or 'Vice' in wordcats: # MODIFICATION
            temp[1] += .75
        else:
            temp[0] += .75
    if 'WlbGain' in wordcats:
        temp[0] += 1
    if 'WlbPt' in wordcats:
        if 'Negativ' in wordcats or 'Weak' in wordcats or 'Hostile':
            temp[1] += 1
        else:
            temp[0] += 1
    if 'WlbLoss' in wordcats:
        temp[1] += 1
    if 'WlbPsyc' in wordcats:
        if 'Negativ' in wordcats or 'Hostile' in wordcats or 'Weak' in wordcats or 'Anomie' in wordcats or \
                        'Pain' in wordcats or 'Vice' in wordcats:
            temp[1] += .75
        else:
            temp[0] += .75
    if 'Vice' in wordcats: # MODIFICATION
        temp[1] += 1.
    elif 'Virtue' in wordcats:
        temp[0] += 1.
    vals.append(temp)

    weight_lst = [2/7, 1/7, 1/7, 1/7, 2/7]
    return compute_weighted_score(weight_lst, vals)


def get_social_value(wordcats):
    vals = []
    # power
    vals.append(get_social_status_power(wordcats))
    # moral
    vals.append(get_social_status_moral(wordcats))
    # material
    vals.append(get_social_status_material(wordcats))
    # ability
    vals.append(get_social_status_ability(wordcats))

    if 'HU' not in wordcats and 'Role' not in wordcats and 'COLL' not in wordcats:
        v = get_usefulness_value(wordcats)
        if v > 0:
            vals.append([1, 0])
        elif v < 0:
            vals.append([0, 1])
        else:
            vals.append([0, 0])
    else:
        vals.append([0, 0])
    weight_lst = [2/9, 2/9, 2/9, 1/9, 2/9]
    return compute_weighted_score(weight_lst, vals)


def get_impact(wordcats):
    '''
    Notes pg. 193 - 194
    :param wordcats:
    :return:
    '''
    vals = []
    # affect & pleasure/palin
    temp = [0., 0.]
    for c in ['PosAff', 'Pleasur']:
        if c in wordcats:
            temp[0] += 1
    for c in ['Pain', 'NegAff']:
        if c in wordcats:
            temp[1] += 1
    vals.append(temp)

    # moral & denial
    temp = [0., 0.]
    for c in ['Anomie', 'NotLw', 'Vice']:
        if c in wordcats:
            temp[1] += 1
    if 'Virtue' in wordcats:
        temp[0] += 1
    vals.append(temp)

    # "bettering" society
    temp = [0., 0.]

    if 'RcGain' in wordcats:
        if 'Negativ' in wordcats:
            temp[1] += .75
        else:
            temp[0] += .75

    for c in ['RcLoss', 'RspLoss']:
        if c in wordcats:
            if 'Positiv' in wordcats:
                temp[0] += .75
            else:
                temp[1] += .75
    for c in ['RcEthic', 'RspOth']:
        if c in wordcats:
            if 'Negativ' in wordcats and 'Submit' not in wordcats:
                temp[1] += .5
            elif 'Positiv' in wordcats or ('Negativ' in wordcats and 'Submit' in wordcats):
                temp[0] += .5

    for c,v in [('WlbPsyc', .5), ('RcEnds', .5), ('EnlOth', .5)]:
        if c in wordcats:
            if 'Negativ' in wordcats:
                temp[1] += v
            elif 'Positiv' in wordcats:
                temp[0] += v

    for c,v in [('WlbGain', 1), ('RspGain', 1), ('EnlGain', 1), ('EnlEnds', 1), ('EnlPt', 1)]:
        if c in wordcats:
            temp[0] += v

    if 'WlbLoss' in wordcats:
        if 'Positiv' in wordcats:
            temp[0] += 1
        else:
            temp[1] += 1
    if 'WlbPt' in wordcats:
        if 'Hostile' in wordcats and 'Submit' not in wordcats:
            temp[1] += 1
        else:
            temp[0] += 1

    if 'EnlLoss' in wordcats:
        temp[1] += 1

    if 'SklOth' in wordcats:
        if 'Positiv' in wordcats or 'Strong' in wordcats or 'Work' in wordcats:
            temp[0] += .5
        elif 'Negativ' in wordcats or 'Weak' in wordcats:
            temp[1] += .5

    if 'WlbPhys' in wordcats:
        if 'Negativ' in wordcats or 'Weak' in wordcats:
            temp[1] += .5
        elif 'Positiv' in wordcats or 'Strong' in wordcats:
            temp[0] += .5

    if 'Try' in wordcats:
        if 'Negativ' in wordcats:
            temp[1] += .5
        else:
            temp[0] += .5

    if 'Goal' in wordcats:
        temp[0] += .5

    vals.append(temp)

    weight_lst = [2/5, 2/5, 1/5]
    return compute_weighted_score(weight_lst, vals)


def get_noun_conn(wordcats, wordemos, wordpol, wordscore):
    return NounConn(wordcats, wordemos, wordpol, wordscore)


def get_adj_conn(wordcats, wordemos, wordpol, wordscore):
    return AdjConn(wordcats, wordemos, wordpol, wordscore)


def get_verb_conn(wordcats, wordemos, wordpol, wordscore):
    return DummyVerbConn(wordcats, wordemos, wordpol, wordscore)


class NounConn:
    def __init__(self, wordcats, wordemos, wordpol, wordscore):
        self.inputs = {'wordcats': wordcats, 'wordemos': wordemos,
                       'wordpol': wordpol, 'wordimg': wordscore}

        self.social_stat = get_social_value(wordcats)
        self.polite = get_politeness(wordcats)
        self.factuality = get_factuality(wordscore)
        self.emo = get_emo_association(wordemos)
        self.sen = get_sentiment(wordpol)
        self.social_imp = get_impact(wordcats)
        self.confidence = 1.0
        self.source = ''
        self.partial = 0

    def get(self):
        sd = {'Social Val': self.social_stat, 'Polite': self.polite,
              'Fact': self.factuality, 'Emo': self.emo, 'Sent': self.sen,
              'Impact': self.social_imp}
        return sd

    def __str__(self):
        sd = self.get()
        s = ''
        for d in sd:
            s += '{}: {}\n'.format(d, sd[d])
        return s.strip()


class AdjConn:
    def __init__(self, wordcats, wordemos, wordpol, wordscore):
        self.inputs = {'wordcats': wordcats, 'wordemos': wordemos,
                       'wordpol': wordpol, 'wordimg': wordscore}

        self.polite = get_politeness(wordcats)
        self.factuality = get_factuality(wordscore)
        self.emo = get_emo_association(wordemos)
        self.sen = get_sentiment(wordpol)
        self.value = get_social_value(wordcats)
        self.imp = get_impact(wordcats)
        self.confidence = 1.0
        self.source = None
        self.partial = 0

    def get(self):
        sd = {'Polite': self.polite, 'Fact': self.factuality, 'Emo': self.emo,
              'Sent': self.sen, 'Social Value': self.value, 'Impact': self.imp}
        return sd

    def __str__(self):
        sd = self.get()
        s = ''
        for d in sd:
            s += '{}: {}\n'.format(d,sd[d])
        return s.strip()


class DummyVerbConn:
    def __init__(self, wordcats, wordemos, wordpol, wordscore):
        self.inputs = {'wordcats': wordcats, 'wordemos': wordemos,
                       'wordpol': wordpol, 'wordimg': wordscore}

        self.polite = get_politeness(wordcats)
        self.factuality = get_factuality(wordscore)
        self.emo = get_emo_association(wordemos)
        self.sen = get_sentiment(wordpol)
        self.value = get_usefulness_value(wordcats)
        self.imp = get_impact(wordcats)
        self.social_stat = get_social_status(wordcats)
        self.confidence = 1.0
        self.source = None
        self.partial = 0

    def get(self):
        sd = {'Polite': self.polite, 'Fact': self.factuality, 'Emo': self.emo,
              'Sent': self.sen, 'Value': self.value, 'Impact': self.imp,
              'Social Stat': self.social_stat}
        return sd

    def __str__(self):
        sd = self.get()
        s = ''
        for d in sd:
            s += '{}: {}\n'.format(d,sd[d])
        return s.strip()
