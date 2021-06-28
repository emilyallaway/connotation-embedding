# Data for 'A Unified Feature Representation for Lexical Connotations'
Submission to EACL 2021

## Lexicon
Combination of new data released for this submission for nouns and adjectives.

Short column descriptions
- word: the word
- POS: the part of speech for the word. One of N (noun) and A (adjective)
- conn: the connotation aspect values (see section 3)
- source: the sources from which the labels are taken. The sources are - separated.
    GIH4, GILvd, and GIH4Lvd indicate General Inquirer, CWn indicates Connotation Wordnet,
    NRC indicates NRC emotion lexicon, and DAL indicates The Dictionary of Affect in 
    Language
- partial?: 1 indicates the word is considered partially labeled, 0 is fully labeled
- train/dev/test: which split the word belongs to


## Embedding
The input files for training the connotation embeddings.
These files include values from the Connotation Frames lexicon 
(https://www.aclweb.org/anthology/P16-1030.pdf) and from the Connotation
Frames of Power and Agency lexicon (https://www.aclweb.org/anthology/D17-1247.pdf).

Short column descriptions
- word: the word
- POS: the part of speech
- def_lst: the processed definitions
- Social Val: the Social Value connotation aspect score
- Polite: the Politeness connotation aspect score
- Impact: the Impact connotation aspect score
- Fact: the Factuality connotation aspect score
- Sent: the Sentiment connotation aspect score
- Emo: the Emotional Association connotation aspect value
- partial?: 1 indicates the word is considered partially labeled, 0 is fully labeled
- source: the sources from which the labels are taken. The sources are - separated.
    GIH4, GILvd, and GIH4Lvd indicate General Inquirer, CWn indicates Connotation Wordnet,
    NRC indicates NRC emotion lexicon, and DAL indicates The Dictionary of Affect in 
    Language
- P(wt): the Perspective of the writer on the theme connotation aspect value
- P(wa): the Perspective of the writer on the agent connotation aspect value
- P(at): the Perspective of the agent on the theme connotation aspect value
- E(t): the Effect on the theme connotation aspect value 
- E(a): the Effect on the agent connotation aspect value
- V(t): the Value of the theme connotation aspect value
- V(a): the Value of the agent connotation aspect value
- S(t): the State of the theme connotation aspect value
- S(a): the State of the agent connotation aspect value
- power: the power connotation aspect value
- agency: the agency connotation aspect value
- mask: a mask over connotation dimensions that don't apply to a particular word
- verb-label: indicates the type of verb label for the word. 'nolabel' indicates none 
    (for nouns and adjectives), 'cf' indicates Pwt through Sa only, 'powa', indicates only
    power and agency, 'cfpowa' indicates Pwt through agency
- rel_lst: list of related words


## Stance
Processed files for running stance experiments. These are processed from the Internet
Argument Corpus v2 (https://nlds.soe.ucsc.edu/iac2).

Short column descriptions
- author_id: id of the author
- text_id: id of the original text
- text: processed text
- label: stance label. 0 is con, 1 is prob, 2 is neutral
- topic: the topic
- src: forum source of the original text
- pos_text: part-of-speech tags with a one-to-one correspondence to the processed text
- lem_text: lemmatized text with a one-to-one correspondence to the pos_text and text
- full_txt_id: unique id for the text/topic combination
- topic_id: id of the topic
