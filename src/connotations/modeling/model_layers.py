import torch, math
import torch.nn as nn

# B: batch size
# T: max sequence length
# E: word embedding size
# C: conn embeddings size
# H: hidden size
# Y: output size
# N_dir: num directions
# N_layer: num layers
# L_i: length of sequence i

################
# Model Layers #
################


class PredictionLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, pred_fn, use_cuda=False,
                 use_orthog=False):
        super(PredictionLayer, self).__init__()
        self.use_cuda = use_cuda

        self.input_dim = input_size
        self.output_dim = output_size
        self.pred_fn = pred_fn

        self.use_orthog = use_orthog

        if use_orthog:
            # set the projection matrices to be orthogonal
            self.W = torch.empty(self.input_dim, self.output_dim)
            nn.init.orthogonal_(self.W)
            self.b = torch.empty(self.output_dim)
            nn.init.normal_(self.b)
            if self.use_cuda:
                self.W = self.W.to('cuda')
                self.b = self.b.to('cuda')
        else:
            self.model = nn.Linear(self.input_dim, self.output_dim)
            if self.use_cuda:
                self.model = self.model.to('cuda')

    def forward(self, input_data):
        if self.use_orthog:
            res = torch.einsum('bh,ho->bo', input_data, self.W) #(B, output_dim)
            res = res + self.b
            return res
        else:
            return self.model(input_data) # (B, output_dim)


class MultiTaskPredictor(torch.nn.Module):
    def __init__(self, input_dim, dims, use_emo, use_cuda=False,
                 use_word=False):
        super(MultiTaskPredictor, self).__init__()

        self.use_cuda = use_cuda
        self.input_dim = input_dim
        self.use_emo = use_emo
        self.use_word = use_word
        self.size = int(self.input_dim / len(dims))

        if self.use_word:
            self.input_dim = 2 * self.input_dim


        self.conn_pred = nn.ModuleList(PredictionLayer(self.size, 3,
                                                       pred_fn=nn.Tanh(),
                                                       use_cuda=self.use_cuda)
                                      for _ in range(len(set(dims) - {'Emo'})))
        if self.use_emo:
            self.conn_pred_emo = PredictionLayer(self.size, 8, pred_fn=nn.Tanh(),
                                                 use_cuda=self.use_cuda)

    def forward(self, hn, word):
        y_pred_lst = []
        if self.use_word:
            h_res = torch.cat((hn, word.squeeze(1)), dim=1)
        else:
            h_res = hn
        i = 0
        for m in self.conn_pred:
            in_h_res = h_res[:, i*self.size: i*self.size + self.size]
            y_pred_lst.append(m(in_h_res))
            i += 1
        if self.use_emo:
            in_h_res =  h_res[:, i*self.size: i*self.size + self.size]
            y_pred_lst.append(self.conn_pred_emo(in_h_res))
        return y_pred_lst


class MultiTaskPredictorVerbs(torch.nn.Module):
    def __init__(self, input_dim, dims, use_emo, use_cuda=False,
                 use_word=False):
        super(MultiTaskPredictorVerbs, self).__init__()

        self.use_cuda = use_cuda
        self.input_dim = input_dim
        self.use_emo = use_emo
        self.use_word = use_word
        self.size = int(self.input_dim / len(dims))

        if self.use_word:
            self.input_dim = 2 * self.input_dim

        self.size = self.input_dim

        conn_pred_lst = [PredictionLayer(self.size, 3, pred_fn=nn.Tanh(),
                                         use_cuda=self.use_cuda)
                                       for _ in range(len(set(dims) - {'Emo', 'power', 'agency'}))]
        if self.use_emo:
            self.conn_pred_emo = PredictionLayer(self.size, 8, pred_fn=nn.Tanh(),
                                                 use_cuda=self.use_cuda)
        if 'power' in dims:
            conn_pred_lst.append(PredictionLayer(self.size, 4, pred_fn=nn.Tanh(),
                                                 use_cuda=self.use_cuda))
        if 'agency' in dims:
            conn_pred_lst.append(PredictionLayer(self.size, 4, pred_fn=nn.Tanh(),
                                                 use_cuda=self.use_cuda))
        self.conn_pred = nn.ModuleList(conn_pred_lst)

    def forward(self, hn, word):
        y_pred_lst = []
        if self.use_word:
            h_res = torch.cat((hn, word.squeeze(1)), dim=1)
        else:
            h_res = hn
        i = 0
        for m in self.conn_pred:
            in_h_res = h_res
            # in_h_res = h_res[:, i * self.size: i * self.size + self.size]
            y_pred_lst.append(m(in_h_res))
            i += 1
        if self.use_emo:
            in_h_res = h_res
            # in_h_res = h_res[:, i * self.size: i * self.size + self.size]
            y_pred_lst.append(self.conn_pred_emo(in_h_res))
        return y_pred_lst


class AttentionLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, use_cuda=False, att_type='reg'):
        super(AttentionLayer, self).__init__()

        self.input_dim = input_dim
        self.use_cuda = use_cuda

        if att_type == 'reg':
            self.scale = 1.
        else:
            self.scale = math.sqrt(2 * self.input_dim)

        self.ff = nn.Sequential(nn.Linear(input_dim*2, hidden_dim, bias=False), nn.Tanh())
        if self.use_cuda:
            self.ff = self.ff.to('cuda')

    def forward(self, inputs, query, last_hidden, rel_words):
        sim = torch.einsum('blh,bh->bl', inputs, query) / self.scale # (B, L)
        att_weights = nn.functional.softmax(sim, dim=1)#(B, L)
        context_vec = torch.einsum('blh,bl->bh', inputs, att_weights) #(B, 2*H)
        att_vec = self.ff(context_vec) # (B, 2*H)
        return att_vec


################
# Input Layers #
################


class BasicWordEmbedSeqsLayer(torch.nn.Module):
    def __init__(self, vecs, static_embeds=True, use_cuda=False):
        super(BasicWordEmbedSeqsLayer, self).__init__()
        vec_tensor = torch.tensor(vecs)

        self.embeds = nn.Embedding.from_pretrained(vec_tensor, freeze=static_embeds)

        self.dim = vecs.shape[1]
        self.vocab_size = float(vecs.shape[0])
        self.use_cuda = use_cuda

    def forward(self, **kwargs):
        embed_args = dict()
        for k in kwargs['use_keys']:
            embed_args['{}_E'.format(k)] = self.embeds(kwargs[k]).type(torch.FloatTensor) # (B, T, E)
        return embed_args