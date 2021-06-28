import torch, data_utils, pickle, time, json, copy
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import torch.nn as nn
import pickle


class TorchModelHandler():
    '''
    Class that holds a model and provides the functionality to train it,
    save it, load it, and evaluate it. The model used here is assumed to be
    written in pytorch.
    '''
    # def __init__(self, model, loss_function, dataloader, optimizer, name, num_ckps=10,
    #              use_score='f_macro', device='cpu', use_last_batch=True):
    def __init__(self, dims, num_ckps=10, use_score='f', use_cuda=False, use_last_batch=True,
                 num_gpus=None, **params):
        super(TorchModelHandler, self).__init__()
        # data fields
        self.model = params['model']
        self.input_model = params['input_model']
        self.dataloader = params['dataloader']
        self.batching_fn = params['batching_fn']
        self.batching_kwargs = params['batching_kwargs']
        self.setup_fn = params['setup_fn']

        self.num_labels = [3, 3, 3, 3, 3, 2]
        self.name = params['name']
        self.use_last_batch = use_last_batch
        self.dim_map = {0: 'Social_Val', 1: 'Polite', 2: 'Impact', 3: 'Fact', 4: 'Sent', 5: 'Emo'}
        self.dims = dims

        # optimization fields
        self.loss_function_lst = params['loss_function_lst']
        self.optimizer = params['optimizer']
        self.weight_lst = params['weight_lst']

        # stats fields
        self.checkpoint_num = 0
        self.num_ckps = num_ckps
        self.epoch = 0

        # evaluation fields
        self.score_dict = dict()
        self.max_score = 0.
        self.max_lst = []  # to keep top 5 scores
        self.score_key = use_score
        self.thresh = 0.5

        # GPU support
        self.use_cuda = use_cuda
        # self.model.to(self.device) # transfers the model to GPU, if available

        if self.use_cuda:
            # move model and loss function to GPU, NOT the embedder
            self.model = self.model.to('cuda')
            for i in range(len(self.loss_function_lst)):
                self.loss_function_lst[i] = self.loss_function_lst[i].to('cuda')

        if num_gpus is not None:
            # self.model = self.model.to('cuda:1')
            self.model = nn.DataParallel(self.model, device_ids=[0,1])

    def get_score_conn(self, scores):
        curr_score_lst = list(map(lambda x: x[1],
                                  filter(lambda x: x[0].startswith(self.score_key), scores.items())))
        curr_score = sum(curr_score_lst) / 6#float(len(self.weight_lst))
        return curr_score

    def get_score(self, scores):
        return self.get_score_conn(scores)

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
        curr_score = self.get_score(scores)

        score_updated = False
        if len(self.max_lst) < 5:
            score_updated = True
            self.max_lst.append((curr_score, scores, self.epoch - 1))
        elif curr_score > self.max_lst[0][0]:
            score_updated = True
            self.max_lst[0] = (curr_score, scores, self.epoch - 1)

        # update best saved model and file with top scores
        if score_updated:
            prev_max = self.max_lst[-1][0]
            # sort the scores
            self.max_lst = sorted(self.max_lst, key=lambda p: p[0]) # lowest first
            # write top 5 scores
            f = open('data/{}.top5_{}.txt'.format(self.name, self.score_key), 'w') # overrides
            for p in self.max_lst:
                f.write('Epoch: {}\nScore: {}\nAll Scores: {}\n'.format(p[2], p[0], json.dumps(p[1])))
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
        }, '../../checkpoints/connotations/ckp-{}-{}.tar'.format(self.name, check_num))

        if num is None:
            self.checkpoint_num = (self.checkpoint_num + 1) % self.num_ckps

    def load(self, filename='../../checkpoints/connotations/ckp-[NAME]-FINAL.tar', use_cpu=False):
        '''
        Loads a saved pytorch model from a checkpoint file.
        :param filename: the name of the file to load from. By default uses
                        the final checkpoint for the model of this' name.
        '''
        filename = filename.replace('[NAME]', self.name)
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def compute_conn_loss(self, y_pred_lst, labels, sample_batched):
        '''
        Compute the loss for the connotation tasks
        :param y_pred_lst:
        :param labels:
        :return:
        '''
        graph_loss = 0
        for i in range(len(self.weight_lst)):
            label_tensor = torch.tensor(labels[i])
            if 'Emo' in self.dims and i == len(self.dims) - 1:
                label_tensor = label_tensor.float()
            if self.use_cuda:
                # move labels to cuda if necessary
                label_tensor = label_tensor.to('cuda')
            graph_loss += self.weight_lst[i] * self.loss_function_lst[i](y_pred_lst[i], label_tensor)
        return graph_loss

    def train_step(self):
        '''
        Runs one epoch of training on this model.
        '''
        print("epoch {}".format(self.epoch))
        self.model.train()
        self.loss = 0. # clear the loss
        start_time = time.time()
        self.dataloader.train()
        for i_batch, sample_batched in enumerate(self.dataloader):
            # zero gradients before EVERY optimizer step
            self.model.zero_grad()

            y_pred_lst, labels = self.get_pred_with_grad(sample_batched)
            graph_loss = self.compute_conn_loss(y_pred_lst=y_pred_lst, labels=labels,
                                                sample_batched=sample_batched)

            # self.loss = graph_loss.item()
            self.loss += graph_loss.item()# update loss

            graph_loss.backward()

            self.optimizer.step()

        end_time = time.time()
        print("   took: {:.1f} min".format((end_time - start_time)/60.))
        self.epoch += 1

    def eval_model(self, data=None, class_wise=False, is_eval=False):
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
        pred_labels_lst, true_labels = self.predict(data, class_wise, is_eval=is_eval)
        self.score(pred_labels_lst, true_labels, class_wise, is_eval)

        return self.score_dict

    def eval_and_print(self, data=None, data_name=None, class_wise=False, is_eval=False):
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
        scores = self.eval_model(data=data, class_wise=class_wise, is_eval=is_eval)
        print("Evaling on \"{}\" data".format(data_name))
        for s_name, s_val in scores.items():
            print("{}: {}".format(s_name, s_val))
        return scores

    def predict_connotation(self, data, all_y_pred_lst, all_labels):
        '''
        Predict connotation labels for the entire dataset
        :param data:
        :param all_y_pred_lst:
        :param all_labels:
        :return:
        '''
        for sample_batched in data:
            with torch.no_grad():
                y_pred_lst, labels = self.get_pred_noupdate(sample_batched)
                y_pred_lst_arr = [y_pred.detach().cpu().numpy() for y_pred in y_pred_lst]

                ls = [np.array(l) for l in labels]

                if all_y_pred_lst is None:
                    all_y_pred_lst = y_pred_lst_arr
                    all_labels = ls
                else:
                    all_y_pred_lst = [np.concatenate((all_y_p, y_arr), 0) for all_y_p, y_arr in
                                      zip(all_y_pred_lst, y_pred_lst_arr)]
                    all_labels = [np.concatenate((all_l, l), 0) for all_l, l in zip(all_labels, ls)]

        pred_labels_lst = [all_y_pred_lst[i].argmax(axis=1) for i in range(len(set(self.dims) - {'Emo'}))]
        if 'Emo' in self.dims:
            pred_labels_lst.append(np.where(all_y_pred_lst[-1] > self.thresh, 1, 0))
        return pred_labels_lst, all_labels

    def predict(self, data=None, class_wise=False, is_eval=False):
        '''
        Make predicts using the specified data. If no data is specified, using the
        internally stored training data.
        :param data:
        :param class_wise:
        :param is_eval:
        :return:
        '''
        all_y_pred_lst = None
        all_labels = None

        self.model.eval()

        if data is None:
            data = self.dataloader
        data.eval()

        pred_labels_lst, all_labels = self.predict_connotation(data, all_y_pred_lst, all_labels)

        return pred_labels_lst, all_labels

    def score_connotation(self, pred_labels, true_labels, class_wise, is_eval):
        '''
        Compute metrics for the connotation data for each dimension.
        :param pred_labels:
        :param true_labels:
        :param class_wise:
        :param is_eval:
        :return:
        '''
        # score other connotations
        for i in range(len(set(self.dims) - {'Emo'})):
            self.score_helper(true_labels[i], pred_labels[i], self.num_labels[i], self.dims[i],
                          class_wise, is_eval)

        if 'Emo' in self.dims:
            # score emotion
            totals = [0., 0., 0.]
            class_totals = [[0., 0.], [0., 0.], [0., 0.]]

            edim = len(set(self.dims) - {'Emo'})
            for ei in range(8):
                score_lst, class_score_lst = self.score_helper(true_labels[edim][:, ei], pred_labels[edim][:, ei],
                                                               self.num_labels[edim], 'Emo',
                                                               class_wise, is_eval)
                self.score_dict['f-Emo-c{}'.format(ei)] = score_lst[0]
                for i in range(3):
                    if i == 0 or is_eval:
                        totals[i] += score_lst[i]
                    if class_wise and (i == 0 or is_eval):
                        for j in range(2):
                            class_totals[i][j] += class_score_lst[i][j]
            for i, n in enumerate(['f', 'p', 'r']):
                if i == 0 or is_eval:
                    v = totals[i] / 8.
                    self.score_dict['{}-Emo_macro'.format(n)] = v
                if class_wise and (i == 0 or is_eval):
                    for j in range(2):
                        v = class_totals[i][j] / 8.
                        self.score_dict['{}-Emo_{}'.format(n, j)] = v

    def score(self, pred_labels, true_labels, class_wise, is_eval=False):
        '''
        Helper Function to compute scores. Stores updated scores in
        the field "score_dict".
        :param pred_labels: the predicted labels
        :param true_labels: the correct labels
        :param class_wise: flag to determine whether to compute class-wise scores in
                            addition to macro-averaged scores.
        '''
        self.score_connotation(pred_labels, true_labels, class_wise, is_eval)

    def score_helper(self, true_labels, pred_labels, num_labels, dim_name, class_wise=False, is_eval=False):
        f1v, f1_class_vs = self.compute_scores(f1_score, true_labels, pred_labels, class_wise,
                                               'f-{}'.format(dim_name), num_labels)
        pv, p_class_vs, rv, r_class_vs = None, None, None, None
        if is_eval:
            # calculate class-wise and macro-average precision
            pv, p_class_vs =  self.compute_scores(precision_score, true_labels, pred_labels, class_wise,
                                                  'p-{}'.format(dim_name), num_labels)
            # calculate class-wise and macro-average recall
            rv, r_class_vs = self.compute_scores(recall_score, true_labels, pred_labels, class_wise,
                                                 'r-{}'.format(dim_name), num_labels)
        return [f1v, pv, rv],  [f1_class_vs, p_class_vs, r_class_vs]

    def compute_scores(self, score_fn, true_labels, pred_labels, class_wise, name, num_labels):
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
        labels = list(set(true_labels))

        vals = score_fn(true_labels, pred_labels, labels=labels, average=None)
        score = sum(vals) /len(labels)
        self.score_dict['{}_macro'.format(name)] = score

        class_scores = None
        if class_wise:
            class_scores = [vals[0], vals[1]]
            self.score_dict['{}_0'.format(name)] = vals[0]
            self.score_dict['{}_1'.format(name)] = vals[1]
            if num_labels > 2:
                class_scores.append(vals[2])
                self.score_dict['{}_none'.format(name)] = vals[2]
        return score, class_scores

    def get_embeddings(self, data=None, data_name='train', out_path='data/vectors/'):
        # data should be a Dataloader
        if data is None:
            data = self.dataloader

        embed_lst = []
        word2pos2i = dict()
        self.model.eval()
        i = 0
        for sample_batched in data:
            args = self.batching_fn(sample_batched, **self.batching_kwargs)

            with torch.no_grad():
                # EMBEDDING
                input_args = self.input_model(**args)
                args.update(input_args)

                embed_lst.append(self.model.get_hidden(*self.setup_fn(args, self.use_cuda)))

            for b in sample_batched:
                word2pos2i[b['ori_word']] = word2pos2i.get(b['ori_word'], dict())
                word2pos2i[b['ori_word']][b['pos']] = i
                i += 1
        conn_embeds = torch.cat(embed_lst, dim=0)
        conn_embeds = conn_embeds.detach().cpu().numpy()
        vec_name = out_path + '{}_conn-embeds.{}.vecs.npy'.format(self.name, data_name)
        np.save(vec_name, conn_embeds)
        print("saved embeddings to {}".format(vec_name))
        vocab_name = out_path + '{}_conn-embeds.{}.vocab.pkl'.format(self.name, data_name)
        pickle.dump(word2pos2i, open(vocab_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        print("saved vocab to {}".format(vocab_name))

    def get_pred_with_grad(self, sample_batched, batching_fn=None,
                           batching_kwargs=None, setup_fn=None):
        '''
        Helper function for getting predictions while tracking gradients.
        Used for training the model.
        OVERRIDES: super method.
        :param sample_batched: the batch of data samples
        :return: the predictions for the batch (as a tensor) and the true
                    labels for the batch (as a numpy array)
        '''
        if batching_fn is None:
            batching_fn = self.batching_fn
            batching_kwargs = self.batching_kwargs
            setup_fn = self.setup_fn

        args = batching_fn(sample_batched, **batching_kwargs)

        # EMBEDDING
        input_args = self.input_model(**args)
        args.update(input_args)

        # PREDICTION
        y_pred_lst = self.model(*setup_fn(args, self.use_cuda))
        labels = args['labels']

        return y_pred_lst, labels

    def get_pred_noupdate(self, sample_batched, batching_fn=None,
                           batching_kwargs=None, setup_fn=None):
        '''
        Helper function for getting predictions without tracking gradients.
        Used for evaluating the model or getting predictions for other reasons.
        OVERRIDES: super method.
        :param sample_batched: the batch of data samples
        :return: the predictions for the batch (as a tensor) and the true labels
                    for the batch (as a numpy array)
        '''
        if batching_fn is None:
            batching_fn = self.batching_fn
            batching_kwargs = self.batching_kwargs
            setup_fn = self.setup_fn

        args = batching_fn(sample_batched, **batching_kwargs)

        with torch.no_grad():
            # EMBEDDING
            input_args = self.input_model(**args)
            args.update(input_args)

            # PREDICTION
            y_pred_lst = self.model(*setup_fn(args, self.use_cuda))
            labels = args['labels']

        return y_pred_lst, labels


class VerbTorchModelHandler(TorchModelHandler):
    def __init__(self, dims, num_ckps=10, use_score='f', use_cuda=False, use_last_batch=True,
                 num_gpus=None, **params):
        TorchModelHandler.__init__(self, dims=dims, num_ckps=num_ckps, use_score=use_score, use_cuda=use_cuda,
                                   use_last_batch=use_last_batch, num_gpus=num_gpus,
                                   **params)

        self.num_verb_dims = 9
        self.num_powa_dims = 2
        self.num_labels = [3] * (5 + self.num_verb_dims) + [4, 4, 2] # Emo is LAST
        self.dim_map = {0: 'Social_Val', 1: 'Polite', 2: 'Impact', 3: 'Fact', 4: 'Sent', 5:  'P(wt)', 6: 'P(wa)',
                        7: 'P(at)', 8: 'E(t)', 9: 'E(a)', 10: 'V(t)', 11: 'V(a)', 12: 'S(t)',
                        13: 'S(a)', 14: 'power', 15: 'agency', 16: 'Emo'}
        self.mask_map = {'Social Val': 0, 'Polite': 1, 'Impact': 2, 'Fact': 3, 'Sent': 4, 'P(wt)': 5, 'P(wa)': 6,
                        'P(at)': 7,  'E(t)': 8, 'E(a)': 9, 'V(t)': 10, 'V(a)': 11, 'S(t)': 12,
                        'S(a)': 13, 'power': 14, 'agency': 15, 'Emo': 16}
        self.eps = .0001
        self.verb_optimizer = params['verb_optimizer']
        self.verb_only = params['verb_only']
        self.l2 = params['l2']

    def compute_conn_loss(self, y_pred_lst, labels, sample_batched):
        mask = torch.tensor([b['loss_mask'] for b in sample_batched])
        if self.use_cuda:
            mask = mask.to('cuda')

        mask_offset = 0
        if self.verb_only:
            mask = mask[:, 5: 5+ self.num_verb_dims + self.num_powa_dims]
            mask_offset = 5

        graph_loss = 0
        for i in range(len(self.weight_lst)):
            label_tensor = torch.tensor(labels[i])
            if 'Emo' in self.dims and i == len(self.dims) - 1:
                label_tensor = label_tensor.float()
            if self.use_cuda:
                # move labels to cuda if necessary
                label_tensor = label_tensor.to('cuda')
            midx = self.mask_map[self.dims[i]] - mask_offset
            batch_loss = mask[:, midx] * self.loss_function_lst[i](y_pred_lst[i], label_tensor)
            avg_batch_loss = torch.sum(batch_loss) / (torch.sum(batch_loss != 0).float() + self.eps)

            graph_loss += self.weight_lst[i] * avg_batch_loss

        return graph_loss

    def train_step(self):
        '''
        Runs one epoch of training on this model.
        '''
        print("epoch {}".format(self.epoch))
        self.model.train()
        self.loss = 0. # clear the N&A loss
        self.verb_loss = 0. # clear verb loss
        num_verb_batches = 0
        num_na_batches = 0

        start_time = time.time()
        self.dataloader.train()
        for i_batch, sample_batched in enumerate(self.dataloader):
            # zero gradients before EVERY optimizer step
            self.model.zero_grad()

            y_pred_lst, labels = self.get_pred_with_grad(sample_batched)
            graph_loss = self.compute_conn_loss(y_pred_lst=y_pred_lst, labels=labels,
                                                sample_batched=sample_batched)

            if not self.dataloader.verb_batch:
                self.loss += graph_loss.item()# update loss
                num_na_batches += 1
            else:
                self.verb_loss += graph_loss.item()
                num_verb_batches += 1
                if self.verb_only:
                    self.loss += graph_loss.item()

            graph_loss.backward()
            if self.dataloader.verb_batch:
                self.verb_optimizer.step()
            else:
                self.optimizer.step()

        end_time = time.time()
        print("   took: {:.1f} min".format((end_time - start_time)/60.))
        self.epoch += 1
        if num_verb_batches > 0:
            print("avg training loss on verbs: {}".format(self.verb_loss / float(num_verb_batches)))
        if num_na_batches > 0:
            print("avg training loss on N & A: {}".format(self.loss / float(num_na_batches)))

    def predict_connotation(self, data, all_y_pred_lst, all_labels):
        all_y_pred_lst = [[] for _ in range(len(self.dims))]
        all_labels = [[] for _ in range(len(self.dims))]

        self.model.eval()
        data.eval()
        for sample_batched in data:
            with torch.no_grad():
                y_pred_lst, labels = self.get_pred_noupdate(sample_batched)
                y_pred_lst_arr = [y_pred.detach().cpu().numpy() for y_pred in y_pred_lst]
                ls = [np.array(l) for l in labels]
                mask = np.array([b['loss_mask'] for b in sample_batched])
                mask_offset = 0
                if self.verb_only:
                    mask = mask[:, 5: 5+self.num_verb_dims + self.num_powa_dims]
                    mask_offset = 5

                if data.verb_batch:
                    if not self.verb_only:
                        dim_range = range(5, 5+self.num_verb_dims+self.num_powa_dims)
                    else:
                        dim_range = range(0, self.num_verb_dims + self.num_powa_dims)
                    if len(self.dims) != self.num_verb_dims + self.num_powa_dims and \
                                    len(self.dims) != self.num_verb_dims + self.num_powa_dims + 6:
                        dim_range = range(0, 1)

                else:
                    dim_range = range(0, 5)

                for i in dim_range:
                    temp_p = []
                    temp_l = []
                    midx = self.mask_map[self.dims[i]] - mask_offset
                    for pj in range(len(y_pred_lst_arr[i])):
                        # if mask[pj][i] != 0:
                        if mask[pj][midx] != 0:
                            temp_p.append(y_pred_lst_arr[i][pj])
                            if self.batching_kwargs['use_labels']:
                                temp_l.append(ls[i][pj])

                    temp_p = np.array(temp_p)
                    temp_l = np.array(temp_l)
                    if len(all_y_pred_lst[i]) != 0 and len(temp_p) != 0:
                        all_y_pred_lst[i] = np.concatenate((all_y_pred_lst[i], temp_p), 0)
                        if self.batching_kwargs['use_labels']:
                            all_labels[i] = np.concatenate((all_labels[i], temp_l), 0)
                    elif len(temp_p) != 0:
                        all_y_pred_lst[i] = temp_p
                        all_labels[i] = temp_l


                if not self.dataloader.verb_batch  and 'Emo' in self.dims:# emo preds
                    all_y_pred_lst[-1] = np.concatenate((all_y_pred_lst[-1], y_pred_lst_arr[-1]), 0) \
                        if len(all_y_pred_lst[-1]) != 0 else y_pred_lst_arr[-1]
                    if self.batching_kwargs['use_labels']:
                        all_labels[-1] = np.concatenate((all_labels[-1], ls[-1]), 0) if len(all_labels[-1]) != 0 else ls[-1]

        pred_labels_lst = [all_y_pred_lst[i].argmax(axis=1) if len(all_y_pred_lst[i]) != 0 else all_y_pred_lst[i] for i in range(len(set(self.dims) - {'Emo'}))]
        if 'Emo' in self.dims:
            pred_labels_lst.append(np.where(all_y_pred_lst[-1] > self.thresh, 1, 0))

        return pred_labels_lst, all_labels


def setup_helper_multitask_conn_embedder(args, use_cuda):
    if use_cuda:
        def_E= args['def_E'].to('cuda')  # (B,T,E)
        word_E = args['word_E'].to('cuda')  # (B,C,E)
        def_l = torch.tensor(args['def_l']).to('cuda')  # (B)
        word_l = torch.tensor(args['word_l']).to('cuda')  # (B)
    else:
        def_E = args['def_E']  # (B,T,E)
        word_E = args['word_E']  # (B,C,E)
        def_l = torch.tensor(args['def_l'])
        word_l = torch.tensor(args['word_l'])
    if 'rel_l' in args:
        if use_cuda:
            rel_E = torch.tensor(args['rel_E']).to('cuda')
        else:
            rel_E = args['rel_E']
        return def_E, word_E, def_l, word_l, rel_E
    else:
        return def_E, word_E, def_l, word_l, None

