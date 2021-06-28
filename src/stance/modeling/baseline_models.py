import torch, time
import torch.nn as nn

import baseline_model_layers as bml


# B: batch size
# T: max sequence length
# E: word embedding size
# C: conn embeddings size
# H: hidden size
# Y: output size
# N_dir: num directions
# N_layer: num layers
# L_i: length of sequence i


class BiCondLSTMModel(torch.nn.Module):
    '''
    Bidirectional Coniditional Encoding LSTM (Augenstein et al, 2016, EMNLP)
    Single layer bidirectional LSTM where initial states are from the topic encoding.
    Topic is also with a bidirectional LSTM. Prediction done with a single layer FFNN with
    tanh then softmax, to use cross-entropy loss.
    '''
    def __init__(self, hidden_dim, embed_dim, drop_prob=0, num_layers=1, num_labels=3,
                 use_cuda=False, max_seq_len=35, max_top_len=5):
        super(BiCondLSTMModel, self).__init__()
        self.use_cuda = use_cuda
        self.num_labels = num_labels

        # self.input_dim = int(embed_dim  - 300)
        self.input_dim = embed_dim
        self.W = torch.empty((embed_dim, self.input_dim))
        self.W = nn.Parameter(nn.init.xavier_normal_(self.W))

        self.bilstm = bml.BiCondLSTMLayer(hidden_dim, self.input_dim, drop_prob, num_layers,
                                      use_cuda=use_cuda)
        self.dropout = nn.Dropout(p=drop_prob) # so we can have dropouts on last layer
        self.pred_layer = bml.PredictionLayer(input_size=2 * num_layers * hidden_dim,
                                          output_size=self.num_labels,
                                          pred_fn=nn.Tanh(), use_cuda=use_cuda)



    def forward(self, text, topic, text_l, topic_l):
        # text = nn.functional.tanh(torch.einsum('ble,eh->blh', text, self.W))
        text = torch.einsum('ble,eh->blh', text, self.W)


        text = text.transpose(0, 1)  # (T, B, E)
        topic = topic.transpose(0, 1)  # (C,B,E)

        _, combo_fb_hn, _ = self.bilstm(text, topic, topic_l, text_l)  # (B, H*N_dir*N_layer)

        # dropout
        combo_fb_hn = self.dropout(combo_fb_hn) #(B, H*N, dir*N_layers)

        y_pred = self.pred_layer(combo_fb_hn)  # (B, 2)
        return y_pred
        # normed_y_pred = nn.functional.softmax(y_pred, dim=1)
        #
        # return normed_y_pred


class BiCondLSTMAttentionModel(torch.nn.Module):
    def __init__(self, hidden_dim, embed_dim, conn_dim, drop_prob=0, num_layers=1, num_labels=3,
                 use_cuda=False, max_seq_len=35, max_top_len=5, att_type='cn'):
        super(BiCondLSTMAttentionModel, self).__init__()
        self.use_cuda = use_cuda
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim

        ## Attention mechanism init
        self.att_type = att_type
        if self.att_type == 'hncnconcat':
            self.att_in_dim = 2 * hidden_dim * num_layers + conn_dim
            self.att_out_dim = self.att_in_dim
            self.query_W = torch.empty((2 * self.hidden_dim * num_layers, self.att_in_dim))
            self.query_W = nn.Parameter(nn.init.xavier_normal_(self.query_W))
            self.att_layer = bml.ScaledDotProductAttention(self.att_in_dim, use_cuda)


        elif att_type == 'cn+w':
            self.att_out_dim = 2 * self.hidden_dim + 2 * self.hidden_dim
            self.query_W = torch.empty((2 * self.hidden_dim * num_layers, embed_dim))
            self.query_W_c = torch.empty((2 * self.hidden_dim * num_layers, conn_dim))
            self.query_W = nn.Parameter(nn.init.xavier_normal_(self.query_W))
            self.query_W_c = nn.Parameter(nn.init.xavier_normal_(self.query_W_c))
            self.att_layer_w = bml.ScaledDotProductAttentionLearned(embed_dim, use_cuda)
            self.att_layer_c = bml.ScaledDotProductAttentionLearned(conn_dim, use_cuda)
            self.ff = nn.Sequential(nn.Linear(conn_dim + embed_dim, 2*self.hidden_dim), nn.Tanh())
        elif att_type == 'cnlearn':
            # self.att_in_dim = embed_dim  # word only
            # self.att_in_dim = conn_dim # conn only
            self.att_in_dim = conn_dim + embed_dim # word + conn
            self.att_out_dim = self.att_in_dim + 2 * self.hidden_dim * num_layers
            self.query_W = torch.empty((2 * self.hidden_dim * num_layers, self.att_in_dim))
            self.query_W = nn.Parameter(nn.init.xavier_normal_(self.query_W))
            self.att_layer = bml.ScaledDotProductAttentionLearned(self.att_in_dim, use_cuda)

        elif att_type == 'cn-w':
            self.att_in_dim = embed_dim # word only
            self.att_out_dim = self.att_in_dim + 2 * self.hidden_dim * num_layers
            self.query_W = torch.empty((2 * self.hidden_dim * num_layers, self.att_in_dim))
            self.query_W = nn.Parameter(nn.init.xavier_normal_(self.query_W))
            self.att_layer = bml.ScaledDotProductAttention(self.att_in_dim, use_cuda)
        elif att_type == 'cn-c':
            self.att_in_dim = conn_dim # conn only
            self.att_out_dim = self.att_in_dim + 2 * self.hidden_dim * num_layers
            self.query_W = torch.empty((2 * self.hidden_dim * num_layers, self.att_in_dim))
            self.query_W = nn.Parameter(nn.init.xavier_normal_(self.query_W))
            self.att_layer = bml.ScaledDotProductAttention(self.att_in_dim, use_cuda)

        elif self.att_type == 'cn':
            # self.att_in_dim = embed_dim # word only
            # self.att_in_dim = conn_dim # conn only
            self.att_in_dim = conn_dim + embed_dim # word + conn
            self.att_out_dim = self.att_in_dim + 2 * self.hidden_dim * num_layers
            self.query_W = torch.empty((2 * self.hidden_dim * num_layers, self.att_in_dim))
            self.query_W = nn.Parameter(nn.init.xavier_normal_(self.query_W))
            self.att_layer = bml.ScaledDotProductAttention(self.att_in_dim, use_cuda)

        elif self.att_type == 'hn':
            self.att_in_dim = 2 * self.hidden_dim * num_layers
            self.att_out_dim = self.att_in_dim
            self.att_layer = bml.ScaledDotProductAttention(self.att_in_dim, use_cuda)
            self.query_W = torch.empty((2 * self.hidden_dim * num_layers, self.att_in_dim))
            self.query_W = nn.Parameter(nn.init.xavier_normal_(self.query_W))

        else: #hncn
            # self.att_in_dim = conn_dim # conn only
            # self.att_in_dim = embed_dim # word only
            self.att_in_dim = conn_dim + embed_dim # word+conn
            self.att_out_dim = self.att_in_dim + 2 * self.hidden_dim * num_layers
            # self.att_out_dim = self.att_in_dim + embed_dim
            self.query_W = torch.empty((2 * self.hidden_dim * num_layers, self.att_in_dim))
            # self.query_W = torch.empty((embed_dim, conn_dim))
            self.query_W = nn.Parameter(nn.init.xavier_normal_(self.query_W))
            self.att_layer_cn = bml.ScaledDotProductAttention(self.att_in_dim, use_cuda)

            self.att_layer_hn = bml.ScaledDotProductAttention(2 * self.hidden_dim * num_layers, use_cuda)
            # self.att_layer_hn = bml.ScaledDotProductAttention(embed_dim, use_cuda)

        ## LSTM, dropout, pred init
        self.bilstm = bml.BiCondLSTMLayer(hidden_dim, embed_dim, drop_prob, num_layers,
                                      use_cuda=use_cuda)
        self.dropout = nn.Dropout(p=drop_prob) # so we can have dropouts on last layer
        self.pred_layer = bml.PredictionLayer(input_size=self.att_out_dim,
                                          output_size=self.num_labels,
                                          pred_fn=nn.Tanh(), use_cuda=use_cuda)

    def forward(self, text, topic, text_l, topic_l, conn, pos_txt):
        # topic: (B, C, E), text: (B, T, E)
        text = text.transpose(0, 1)  # (T, B, E)
        topic = topic.transpose(0, 1)  # (C,B,E)
        # conn: (B, T, Ec)
        # pos_txt: (B, T, E)

        ### dropout on the input
        text = self.dropout(text)

        ### Bicond LSTM
        output, combo_fb_hn, last_top_hn = self.bilstm(text, topic, topic_l, text_l)  # (B, H*N_dir*N_layer)

        ### dropout on the LSTM
        combo_fb_hn = self.dropout(combo_fb_hn) #(B, H*N, dir*N_layers)
        output = self.dropout(output)

        ### transform the output
        output = output.transpose(0, 1) # (B, L, 2H)

        ### transofrm the topic
        query_hn = last_top_hn.transpose(0, 1).reshape(-1, 2 * self.hidden_dim) # (B, 2H)
        top_query = torch.einsum('bh,ha->ba', query_hn, self.query_W) # (B, A), where A dimension of attention input
        # query_hn = torch.mean(topic, dim=0)
        # top_query = torch.einsum('bh,ha->ba', query_hn, self.query_W)

        ### apply attention
        if self.att_type == 'hncnconcat':
            att_inputs = torch.cat((conn, output), dim=2) # (B, T, 2H+Ec)
            context_vec = self.att_layer(att_inputs, top_query) #(B, 2H+Ec)

        elif self.att_type == 'cn+w':
            # top_query (B,embed_dim)
            top_query_c = torch.einsum('bh,ha->ba', query_hn, self.query_W_c) #(B, conn_dim)
            cvec = self.att_layer_c(conn, top_query_c)
            wvec = self.att_layer_w(pos_txt, top_query)
            v = self.ff(torch.cat((cvec, wvec), dim=1)) #(B, 2H)
            context_vec = torch.cat((combo_fb_hn, v), dim=1)

        elif self.att_type == 'cnlearn':
            # cv = self.att_layer(conn, top_query) # (B, E) # conn only
            # cv = self.att_layer(pos_txt, top_query)  # word only
            cv = self.att_layer(torch.cat((pos_txt, conn), dim=2), top_query) # Word+conn
            context_vec = torch.cat((combo_fb_hn, cv), dim=1)  # (B, 2H+E)

        elif self.att_type == 'cn-w':
            cv = self.att_layer(pos_txt, top_query) # word only
            context_vec = torch.cat((combo_fb_hn, cv), dim=1)  # (B, 2H+E)
        elif self.att_type == 'cn-c':
            cv = self.att_layer(conn, top_query) # (B, E) # conn only
            context_vec = torch.cat((combo_fb_hn, cv), dim=1)  # (B, 2H+E)

        elif self.att_type == 'cn':
            # cv = self.att_layer(conn, top_query) # (B, E) # conn only
            # cv = self.att_layer(pos_txt, top_query) # word only
            cv = self.att_layer(torch.cat((pos_txt, conn), dim=2), top_query) # Word+conn
            context_vec = torch.cat((combo_fb_hn, cv), dim=1) #(B, 2H+E)

        elif self.att_type == 'hn':
            context_vec = self.att_layer(output, query_hn)
        else: # hncn
            cvec_hn = self.att_layer_hn(output, query_hn) #(B, 2H)

            # cvec_cn = self.att_layer_cn(conn, top_query) # (B, E) CONN-only
            # cvec_cn = self.att_layer_cn(pos_txt, top_query) # WORD-only
            cvec_cn = self.att_layer_cn(torch.cat((pos_txt, conn), dim=2), top_query) # concat WORD+CONN

            context_vec = torch.cat((cvec_cn, cvec_hn), dim=1)

        y_pred = self.pred_layer(context_vec) #(B, num_labels)
        return y_pred
