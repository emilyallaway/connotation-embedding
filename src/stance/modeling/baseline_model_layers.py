import torch, time, math
import torch.nn as nn
import torch.nn.utils.rnn as rnn

# B: batch size
# T: max sequence length
# E: word embedding size
# C: conn embeddings size
# H: hidden size
# Y: output size
# N_dir: num directions
# N_layer: num layers
# L_i: length of sequence i
# S: number of sentences

class TwoLayerFFNNLayer(torch.nn.Module):
    '''
    2-layer FFNN with specified nonlinear function
    must be followed with some kind of prediction layer for actual prediction
    '''
    def __init__(self, input_dim, hidden_dim, out_dim, nonlinear_fn):
        super(TwoLayerFFNNLayer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   nonlinear_fn,
                                   nn.Linear(hidden_dim, out_dim))

    def forward(self, input):
        return self.model(input)

class PredictionLayer(torch.nn.Module):
    '''
    Predicition layer. linear projection followed by the specified functions
    ex: pass pred_fn=nn.Tanh()
    '''
    def __init__(self, input_size, output_size, pred_fn, use_cuda=False):
        super(PredictionLayer, self).__init__()

        self.use_cuda = use_cuda

        self.input_dim = input_size
        self.output_dim = output_size
        self.pred_fn = pred_fn

        self.model = nn.Sequential(nn.Linear(self.input_dim, self.output_dim, bias=False), nn.Tanh())

        if self.use_cuda:
            self.model = self.model.to('cuda')#cuda()

    def forward(self, input_data):
        return self.model(input_data)


class BiCondLSTMLayer(torch.nn.Module):
    '''
        Bidirection Conditional Encoding (Augenstein et al. 2016 EMNLP).
        Bidirectional LSTM with initial states from topic encoding.
        Topic encoding is also a bidirectional LSTM.
        '''

    def __init__(self, hidden_dim, embed_dim, drop_prob=0, num_layers=1, use_cuda=False):
        super(BiCondLSTMLayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_cuda = use_cuda

        self.topic_lstm = nn.LSTM(self.embed_dim, self.hidden_dim, bidirectional=True) #LSTM
        self.text_lstm = nn.LSTM(self.embed_dim, self.hidden_dim, bidirectional=True)


    def forward(self, txt_e, top_e, top_l, txt_l):
        ####################
        # txt_e = (Lx, B, E), top_e = (Lt, B, E), top_l=(B), txt_l=(B)
        ########################

        p_text_embeds = rnn.pack_padded_sequence(txt_e, txt_l, enforce_sorted=False) # these are sorted
        p_top_embeds = rnn.pack_padded_sequence(top_e, top_l, enforce_sorted=False) # LSTM

        # feed topic
        _, last_top_hn_cn = self.topic_lstm(p_top_embeds) # ((2, B, H), (2, B, H)) #LSTM
        last_top_hn = last_top_hn_cn[0] # LSTM

        # feed text conditioned on topic
        output, (txt_last_hn, _)  = self.text_lstm(p_text_embeds, last_top_hn_cn) # (2, B, H)
        txt_fw_bw_hn = txt_last_hn.transpose(0, 1).reshape((-1, 2 * self.hidden_dim))
        padded_output, _ = rnn.pad_packed_sequence(output)
        return padded_output, txt_fw_bw_hn, last_top_hn


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, input_dim, use_cuda=False):
        super(ScaledDotProductAttention, self).__init__()
        self.input_dim = input_dim

        self.scale = math.sqrt(2 * self.input_dim)

    def forward(self, inputs, query):
        # inputs = (B, L, 2*H), query = (B, 2*H), last_hidden=(B, 2*H)
        sim = torch.einsum('blh,bh->bl', inputs, query) / self.scale  # (B, L)
        att_weights = nn.functional.softmax(sim, dim=1)  # (B, L)
        context_vec = torch.einsum('blh,bl->bh', inputs, att_weights)  # (B, 2*H)
        return context_vec