import sys, os, time, argparse
sys.path.append('..')
import data_utils, model_utils, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np

VEC_NAME='glove100'

BINARY = False

VOCAB_SIZE = 10000

DATA_DIR  = '../../../resources/stance/'

MODEL_TYPE = 'LogReg'#'Maj'#'LogReg'

def baseline_BoWV(model_type, trn_name, dev_name):
    '''
    Loads, trains, and evaluates a logistics regression model
    on the training and dev data. Currently does BINARY classification.
    Prints the scores to the console. Saves the trained model to
    'stance-internal-git/saved_models/'.
    :param trn_name: the name of the training data file
    :param dev_name: the name of the dev data file
    '''
    print("Loading data")
    trn_data = datasets.StanceDataBoW(trn_name,
                                      text_vocab_file='{}text_vocab_top{}.txt'.format(DATA_DIR, VOCAB_SIZE),
                                      topic_vocab_file='{}topic_vocab.txt'.format(DATA_DIR),
                                      binary=BINARY)

    trn_datasampler = data_utils.DataSampler(trn_data, batch_size=len(trn_data))

    print("Initializing model")
    #########
    # MODEL #
    #########
    # model = LogisticRegression(solver='lbfgs', class_weight='balanced',
    #                            max_iter=500)
    if model_type == 'LogReg':
        model = LogisticRegression(solver='lbfgs', class_weight='balanced',
                                   multi_class='multinomial',
                                   max_iter=600)
    elif model_type == 'SVM':
        model = SVC(gamma='scale', max_iter=600)
    elif model_type == 'Maj':
        model = MajorityBaseline(len(trn_data.topic_vocab2i),
                                 {v:k for k,v in trn_data.topic_vocab2i.items()})
    elif model_type == 'Rand':
        model = RandomBaseline()


    model_handler = model_utils.ModelHandler(model=model, name='LogReg.multi.balanced',
                                             dataloader=trn_datasampler)

    print("Training model")
    st = time.time()
    model_handler.train_step()
    et = time.time()
    print("   took: {:.1f} minutes".format((et - st) / 60.))

    print("Evaluating model on train data")
    trn_scores = model_handler.eval_model(data=None, class_wise=True)
    for s in trn_scores:
        print('{}: {}'.format(s, trn_scores[s]))

    print("Evaluating model on dev data")
    dev_data = datasets.StanceDataBoW(dev_name,
                                      text_vocab_file='{}text_vocab_top{}.txt'.format(DATA_DIR, VOCAB_SIZE),
                                      topic_vocab_file='{}topic_vocab.txt'.format(DATA_DIR),
                                      binary=BINARY)
    dev_datasampler = data_utils.DataSampler(dev_data, batch_size=len(dev_data))
    dev_scores = model_handler.eval_model(data=dev_datasampler, class_wise=True)
    for s in dev_scores:
        print('{}: {}'.format(s, dev_scores[s]))

    print("Saving model")
    if trn_data.binary:
        model_handler.save('../../saved_models/binaryClass.{}'.format(VEC_NAME))
    else:
        model_handler.save('../../saved_models/3Class.{}'.format(VEC_NAME))


class MajorityBaseline():
    def __init__(self, topic_vocab_size, i2topic):
        self.topic_vocab_size = topic_vocab_size
        self.idx2topic = i2topic

    def convert_data(self, data):
        temp = [np.nonzero(row[-self.topic_vocab_size:])[0] for row in data]
        topics = []
        for row in temp:
            topics.append(' '.join([self.idx2topic[ti + 1] for ti in row]))
        return topics

    def fit(self, data, labels):
        topic_rows = self.convert_data(data)

        self.t2counts = dict()
        for t, l in zip(topic_rows, labels):
            self.t2counts[t] = self.t2counts.get(t, [0, 0, 0])
            self.t2counts[t][l] = self.t2counts[t][l] + 1.

    def predict(self, data):
        topic_rows = self.convert_data(data)
        labels = []
        for t in topic_rows:
            labels.append(np.argmax(self.t2counts[t]))
        return labels



class RandomBaseline():
    def __init__(self):
        pass

    def fit(self, data, labels):
        pass

    def predict(self, data):
        return [np.random.randint(0, 3) for _ in range(len(data))]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='What to do')
    parser.add_argument('-i', '--trn_data', help='Training data file')
    parser.add_argument('-d', '--dev_data', help='Dev data file')
    parser.add_argument('-t', '--model_type', help='type of model to train', default='LogReg')
    args = vars(parser.parse_args())

    if args['mode'] == '1':
        print("training {} model".format(args['model_type']))
        baseline_BoWV(args['model_type'], args['trn_data'], args['dev_data'])

    elif args['mode'] == '2':
        print("training all 3 baselines")
        for mt in ['Rand', 'Maj', 'LogReg']:
            print("MODEL {}".format(mt))
            baseline_BoWV(mt, args['trn_data'], args['dev_data'])
            print()
    else:
        print("ERROR: doing nothing")

