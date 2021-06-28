import torch, data_utils, pickle, time, json, copy
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import torch.nn as nn

class ModelHandler():
    '''
    Class that holds a model and provides the functionality to train it,
    save it, load it, and evaluate it. The model used here is assumed to be an
    sklearn model. Use TorchModelHandler for a model written in pytorch.
    '''
    def __init__(self, model, dataloader, name):
        self.model = model
        self.dataloader = dataloader

        self.name = name

        self.score_dict = dict()

    def prepare_data(self, data=None):
        '''
        Prepares data to be used for training or dev by formatting it
        correctly.
        :param data: the data to be formatted, in a DataHandler. If this is None (default) then
                        the data used is in self.dataloader.
        :return: the formatted input data as a numpy array,
                the formatted labels as a numpy array
        '''
        if data == None:
            data = self.dataloader

        data = data.get()

        concat_data = []
        labels = []
        for s in data:
            concat_data.append(s['text'] + s['topic'])
            labels.append(np.argmax(s['label']))

        input_data = np.array(concat_data)
        input_labels = np.array(labels)

        return input_data, input_labels

    def train_step(self):
        print("   preparing data")
        input_data, input_labels = self.prepare_data()

        print("   training")
        self.model.fit(input_data, input_labels)

    def save(self, out_prefix):
        pickle.dump(self.model, open('{}-{}.pkl'.format(out_prefix, self.name), 'wb'))
        print("model saved")

    def load(self, name_prefix):
        self.model = pickle.load(open('{}-{}.pkl'.format(name_prefix, self.name), 'rb'))
        print("model loaded")

    def compute_scores(self, score_fn, true_labels, pred_labels, class_wise, name, binary=False):
        if binary:
            vals = score_fn(true_labels, pred_labels, labels=[0, 1], average=None)
            self.score_dict['{}_macro'.format(name)] = sum(vals) / 2.
        else:
            vals = score_fn(true_labels, pred_labels, labels=[0, 1, 2], average=None)
            self.score_dict['{}_macro'.format(name)] = sum(vals) / 3.

        if class_wise:
            self.score_dict['{}_anti'.format(name)] = vals[0]
            self.score_dict['{}_pro'.format(name)] = vals[1]
            if not binary:
                self.score_dict['{}_none'.format(name)] = vals[2]

    def eval_model(self, data, class_wise=False):
        if data is None:
            b = self.dataloader.data.binary
        else:
            b = data.data.binary

        print("   preparing data")
        input_data, true_labels = self.prepare_data(data)
        print("   making predictions")
        pred_labels = self.model.predict(input_data)
        # pred_labels = [np.random.randint(0, 3) for _ in range(len(true_labels))]

        print("   computing scores")
        self.compute_scores(f1_score, true_labels, pred_labels, class_wise, 'f',
                            binary=b)
        # calculate class-wise and macro-average precision
        self.compute_scores(precision_score, true_labels, pred_labels, class_wise, 'p',
                            binary=b)
        # calculate class-wise and macro-average recall
        self.compute_scores(recall_score, true_labels, pred_labels, class_wise, 'r',
                            binary=b)

        return self.score_dict


class TorchModelHandler():
    '''
    Class that holds a model and provides the functionality to train it,
    save it, load it, and evaluate it. The model used here is assumed to be
    written in pytorch.
    '''
    # def __init__(self, model, loss_function, dataloader, optimizer, name, num_ckps=10,
    #              use_score='f_macro', device='cpu', use_last_batch=True):
    def __init__(self, num_ckps=10, use_score='f_macro', use_cuda=False, use_last_batch=True,
                 num_gpus=None, **params):
        super(TorchModelHandler, self).__init__()
        # data fields
        self.model = params['model']
        self.embed_model = params['embed_model']
        self.dataloader = params['dataloader']
        self.batching_fn = params['batching_fn']
        self.batching_kwargs = params['batching_kwargs']
        self.setup_fn = params['setup_fn']


        self.num_labels = self.model.num_labels
        self.name = params['name']
        self.use_last_batch = use_last_batch

        # optimization fields
        self.loss_function = params['loss_function']
        self.optimizer = params['optimizer']

        # stats fields
        self.checkpoint_num = 0
        self.num_ckps = num_ckps
        self.epoch = 0

        # evaluation fields
        self.score_dict = dict()
        self.max_score = 0.
        self.max_lst = []  # to keep top 5 scores
        self.score_key = use_score

        # GPU support
        self.use_cuda = use_cuda
        # self.model.to(self.device) # transfers the model to GPU, if available

        if self.use_cuda:
            # move model and loss function to GPU, NOT the embedder
            self.model = self.model.to('cuda')
            self.loss_function = self.loss_function.to('cuda')

        if num_gpus is not None:
            # self.model = self.model.to('cuda:1')
            self.model = nn.DataParallel(self.model, device_ids=[0,1])

    def save_best(self, data=None, scores=None, data_name=None, class_wise=False):
        '''
        Evaluates the model on data and then updates the best scores and saves the best model.
        :param data: data to evaluate and update based on. Default (None) will evaluate on the internally
                        saved data. Otherwise, should be a DataSampler. Only used if scores is not None.
        :param scores: a dictionary of precomputed scores. Default (None) will compute a list of scores
                        using the given data, name and class_wise flag.
        :param data_name: the name of the data evaluating and updating on. Only used if scores is not None.
        :param class_wise: lag to determine whether to compute class-wise scores in
                            addition to macro-averaged scores. Only used if scores is not None.
        '''
        if scores is None:
            # evaluate and print
            scores = self.eval_and_print(data=data, data_name=data_name,
                                         class_wise=class_wise)
        scores = copy.deepcopy(scores)  # copy the scores, otherwise storing a pointer which won't track properly

        # update list of top scores
        curr_score = scores[self.score_key]
        score_updated = False
        if len(self.max_lst) < 5:
            score_updated = True
            self.max_lst.append((scores, self.epoch - 1))
        elif curr_score > self.max_lst[0][0][self.score_key]:
            score_updated = True
            self.max_lst[0] = (scores, self.epoch - 1)

        # update best saved model and file with top scores
        if score_updated:
            prev_max = self.max_lst[-1][0][self.score_key]
            # sort the scores
            self.max_lst = sorted(self.max_lst, key=lambda p: p[0][self.score_key]) # lowest first
            # write top 5 scores
            f = open('data/{}.top5_{}.txt'.format(self.name, self.score_key), 'w') # overrides
            for p in self.max_lst:
                f.write('Epoch: {}\nScore: {}\nAll Scores: {}\n'.format(p[1], p[0][self.score_key],
                                                                      json.dumps(p[0])))
            # save best model step, if its this one
            if curr_score > prev_max:
                self.save(num='BEST')

    def save(self, num=None):
        '''
        Saves the pytorch model in a checkpoint file.
        :param num: The number to associate with the checkpoint. By default uses
                    the internally tracked checkpoint number but this can be changed.
        '''
        if num is None:
            check_num = self.checkpoint_num
        else: check_num = num

        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss
        }, '../../checkpoints/ckp-{}-{}.tar'.format(self.name, check_num))

        if num is None:
            self.checkpoint_num = (self.checkpoint_num + 1) % self.num_ckps

    def load(self, filename='../../checkpoints/ckp-[NAME]-FINAL.tar', use_cpu=False):
        '''
        Loads a saved pytorch model from a checkpoint file.
        :param filename: the name of the file to load from. By default uses
                        the final checkpoint for the model of this' name.
        '''
        filename = filename.replace('[NAME]', self.name)
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def train_step(self):
        '''
        Runs one epoch of training on this model.
        '''
        print("epoch {}".format(self.epoch))
        self.model.train()
        self.loss = 0. # clear the loss
        start_time = time.time()
        for i_batch, sample_batched in enumerate(self.dataloader):
            # zero gradients before EVERY optimizer step
            self.model.zero_grad()

            y_pred, labels = self.get_pred_with_grad(sample_batched)

            label_tensor = torch.tensor(labels)
            if self.use_cuda:
                # move labels to cuda if necessary
                label_tensor = label_tensor.to('cuda')

            if self.dataloader.weighting:
                batch_loss = self.loss_function(y_pred, label_tensor)
                weight_lst = torch.tensor([self.dataloader.topic2c2w[b['ori_topic']][b['label']] for b in sample_batched])
                if self.use_cuda:
                    weight_lst = weight_lst.to('cuda')
                graph_loss = torch.mean(batch_loss * weight_lst)
            else:
                graph_loss = self.loss_function(y_pred, label_tensor)

            self.loss += graph_loss.item()# update loss

            graph_loss.backward()

            self.optimizer.step()

        end_time = time.time()
        print("   took: {:.1f} min".format((end_time - start_time)/60.))
        self.epoch += 1

    def compute_scores(self, score_fn, true_labels, pred_labels, class_wise, name):
        '''
        Computes scores using the given scoring function of the given name. The scores
        are stored in the internal score dictionary.
        :param score_fn: the scoring function to use.
        :param true_labels: the true labels.
        :param pred_labels: the predicted labels.
        :param class_wise: flag to determine whether to compute class-wise scores in
                            addition to macro-averaged scores.
        :param name: the name of this score function, to be used in storing the scores.
        '''
        labels = [i for i in range(self.num_labels)]
        n = float(len(labels))

        vals = score_fn(true_labels, pred_labels, labels=labels, average=None)
        self.score_dict['{}_macro'.format(name)] = sum(vals) / n

        if class_wise:
            self.score_dict['{}_anti'.format(name)] = vals[0]
            self.score_dict['{}_pro'.format(name)] = vals[1]
            if n > 2:
                self.score_dict['{}_none'.format(name)] = vals[2]

    def eval_model(self, data=None, class_wise=False):
        '''
        Evaluates this model on the given data. Stores computed
        scores in the field "score_dict". Currently computes macro-averaged
        F1 scores, precision and recall. Can also compute scores on a class-wise basis.
        :param data: the data to use for evaluation. By default uses the internally stored data
                    (should be a DataSampler if passed as a parameter).
        :param class_wise: flag to determine whether to compute class-wise scores in
                            addition to macro-averaged scores.
        :return: a map from score names to values
        '''
        pred_labels, true_labels, t2pred = self.predict(data)
        self.score(pred_labels, true_labels, class_wise, t2pred)

        return self.score_dict

    def predict(self, data=None):
        all_y_pred = None
        all_labels = None

        self.model.eval()

        if data is None:
            data = self.dataloader

        t2pred = dict()
        for sample_batched in data:
            with torch.no_grad():
                y_pred, labels = self.get_pred_noupdate(sample_batched)
                y_pred_arr = y_pred.detach().cpu().numpy()
                ls = np.array(labels)

                for bi, b in enumerate(sample_batched):
                    t = b['ori_topic']
                    t2pred[t] = t2pred.get(t, ([], []))
                    t2pred[t][0].append(y_pred_arr[bi, :])
                    t2pred[t][1].append(ls[bi])

                if all_y_pred is None:
                    all_y_pred = y_pred_arr
                    all_labels = ls
                else:
                    all_y_pred = np.concatenate((all_y_pred, y_pred_arr), 0)
                    all_labels = np.concatenate((all_labels, ls), 0)

        for t in t2pred:
            t2pred[t] = (np.argmax(t2pred[t][0], axis=1), t2pred[t][1])

        pred_labels = all_y_pred.argmax(axis=1)
        true_labels = all_labels
        return pred_labels, true_labels, t2pred

    def eval_and_print(self, data=None, data_name=None, class_wise=False):
        '''
        Evaluates this model on the given data. Stores computed
        scores in the field "score_dict". Currently computes macro-averaged.
        Prints the results to the console.
        F1 scores, precision and recall. Can also compute scores on a class-wise basis.
        :param data: the data to use for evaluation. By default uses the internally stored data
                    (should be a DataSampler if passed as a parameter).
        :param data_name: the name of the data evaluating.
        :param class_wise: flag to determine whether to compute class-wise scores in
                            addition to macro-averaged scores.
        :return: a map from score names to values
        '''
        scores = self.eval_model(data=data, class_wise=class_wise)
        print("Evaling on \"{}\" data".format(data_name))
        for s_name, s_val in scores.items():
            print("{}: {}".format(s_name, s_val))
        return scores

    def score(self, pred_labels, true_labels, class_wise, t2pred):
        '''
        Helper Function to compute scores. Stores updated scores in
        the field "score_dict".
        :param pred_labels: the predicted labels
        :param true_labels: the correct labels
        :param class_wise: flag to determine whether to compute class-wise scores in
                            addition to macro-averaged scores.
        '''
        # calculate class-wise and macro-averaged F1
        self.compute_scores(f1_score, true_labels, pred_labels, class_wise, 'f')
        # calculate class-wise and macro-average precision
        self.compute_scores(precision_score, true_labels, pred_labels, class_wise, 'p')
        # calculate class-wise and macro-average recall
        self.compute_scores(recall_score, true_labels, pred_labels, class_wise, 'r')

        for t in t2pred:
            self.compute_scores(f1_score, t2pred[t][1], t2pred[t][0], class_wise,
                                '{}-f'.format(t))

    def get_pred_with_grad(self, sample_batched):
        '''
        Helper function for getting predictions while tracking gradients.
        Used for training the model.
        OVERRIDES: super method.
        :param sample_batched: the batch of data samples
        :return: the predictions for the batch (as a tensor) and the true
                    labels for the batch (as a numpy array)
        '''
        args = self.batching_fn(sample_batched, **self.batching_kwargs)

        # EMBEDDING
        embed_args = self.embed_model(**args)
        args.update(embed_args)

        # PREDICTION
        y_pred = self.model(*self.setup_fn(args, self.use_cuda))
        labels = args['labels']

        return y_pred, labels

    def get_pred_noupdate(self, sample_batched):
        '''
        Helper function for getting predictions without tracking gradients.
        Used for evaluating the model or getting predictions for other reasons.
        OVERRIDES: super method.
        :param sample_batched: the batch of data samples
        :return: the predictions for the batch (as a tensor) and the true labels
                    for the batch (as a numpy array)
        '''
        args = self.batching_fn(sample_batched, **self.batching_kwargs)

        with torch.no_grad():
            # EMBEDDING
            embed_args = self.embed_model(**args)
            args.update(embed_args)

            # PREDICTION
            y_pred = self.model(*self.setup_fn(args, self.use_cuda))
            labels = args['labels']

        return y_pred, labels



def setup_helper_bicond(args, use_cuda):
    if use_cuda:
        txt_E= args['txt_E'].to('cuda')  # (B,T,E)
        top_E = args['top_E'].to('cuda')  # (B,C,E)
        txt_l = torch.tensor(args['txt_l']).to('cuda')  # (B, S)
        top_l = torch.tensor(args['top_l']).to('cuda')  # (B)
    else:
        txt_E = args['txt_E']  # (B,T,E)
        top_E = args['top_E']  # (B,C,E)
        txt_l = torch.tensor(args['txt_l'])
        top_l = torch.tensor(args['top_l'])
    return txt_E, top_E, txt_l, top_l

def setup_helper_bicond_connvecs(args, use_cuda):
    if use_cuda:
        txt_E= args['txt_E'].to('cuda')  # (B,T,E)
        top_E = args['top_E'].to('cuda')  # (B,C,E)
        txt_l = torch.tensor(args['txt_l']).to('cuda')  # (B, S)
        top_l = torch.tensor(args['top_l']).to('cuda')  # (B)
        conn_E = args['conn_E'].to('cuda')
        pos_E = args['pos_txt_E'].to('cuda')
    else:
        txt_E = args['txt_E']  # (B,T,E)
        top_E = args['top_E']  # (B,C,E)
        txt_l = torch.tensor(args['txt_l'])
        top_l = torch.tensor(args['top_l'])
        conn_E = args['conn_E']
        pos_E = args['pos_txt_E'].to('cuda')
    return txt_E, top_E, txt_l, top_l, conn_E, pos_E