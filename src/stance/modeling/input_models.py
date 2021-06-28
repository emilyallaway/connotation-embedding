import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from transformers import BertTokenizer, BertModel

from IPython import embed

# B: batch size
# T: max sequence length
# E: word embedding size
# C: conn embeddings size
# H: hidden size
# Y: output size
# N_dir: num directions
# N_layer: num layers
# L_i: length of sequence i


class BasicWordEmbedLayer(torch.nn.Module):
    def __init__(self, vecs, static_embeds=True, use_cuda=False):
        super(BasicWordEmbedLayer, self).__init__()
        vec_tensor = torch.tensor(vecs)

        self.embeds = nn.Embedding.from_pretrained(vec_tensor, freeze=static_embeds)

        self.dim = vecs.shape[1]
        self.vocab_size = float(vecs.shape[0])
        self.use_cuda = use_cuda

    def forward(self, **kwargs):
        embed_args = {'txt_E': self.embeds(kwargs['text']).type(torch.FloatTensor), # (B, T, E)
                      'top_E': self.embeds(kwargs['topic']).type(torch.FloatTensor)} #(B, C, E)
        return embed_args


class ConnVecsLayerSeparate(torch.nn.Module):
    def __init__(self, vecs, pos2conn_vecs, static_word=True, static_conn=True, use_random=False):
        super(ConnVecsLayerSeparate, self).__init__()

        self.pos2conn_embeds = dict()
        self.conn_dim = 0
        self.pos_lst = list(pos2conn_vecs.keys())
        for t in pos2conn_vecs:
            conn_tensor = torch.tensor(pos2conn_vecs[t])
            self.pos2conn_embeds[t] = nn.Embedding.from_pretrained(conn_tensor, freeze=static_conn)
            self.conn_dim = conn_tensor.shape[1]

        vec_tensor = torch.tensor(vecs)
        self.word_embeds = nn.Embedding.from_pretrained(vec_tensor, freeze=static_word)
        self.random_embeds = nn.Embedding(vec_tensor.shape[0], vec_tensor.shape[1])

        self.dim = vecs.shape[1]# embedding size == num connotation features
        self.vocab_size = float(vecs.shape[0])
        self.use_random = use_random

    def forward(self, **kwargs):
        word_E = self.word_embeds(kwargs['text']).type(torch.FloatTensor)  # (B, Lx, E)
        top_E = self.word_embeds(kwargs['topic']).type(torch.FloatTensor)  # (B, Lt, E)
        conn_E = torch.zeros(word_E.shape[0], word_E.shape[1], self.conn_dim, dtype=torch.float32)  # (B, Lx, C_dim)
        pos_E = torch.zeros(word_E.shape[0], word_E.shape[1], self.dim, dtype=torch.float32)
        for t in self.pos_lst:
            conn_E = conn_E + self.pos2conn_embeds[t](kwargs['text_pos2filter'][t]).type(torch.FloatTensor)
            if not self.use_random:
                pos_E = pos_E + self.word_embeds(kwargs['text_pos2filter'][t]).type(torch.FloatTensor)
            else:
                pos_E = pos_E + self.random_embeds(kwargs['text_pos2filter'][t]).type(torch.FloatTensor)

        embed_args = {'txt_E': word_E, 'top_E': top_E,
                      'conn_E': conn_E,
                      'txt_l': kwargs['txt_l'], 'top_l': kwargs['top_l'],
                      'pos_txt_E': pos_E}
        return embed_args

