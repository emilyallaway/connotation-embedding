import torch
import torch.nn as nn
from model_layers import MultiTaskPredictor, AttentionLayer, MultiTaskPredictorVerbs
import torch.nn.utils.rnn as rnn
import transformer as tm


class MultiTaskConnotationEmbedder(torch.nn.Module):
    def __init__(self, hidden_dim, embed_dim, dims, use_emo, drop_prob=0, use_cuda=False,
                 enc_type='lstm', use_word=False, use_verb=False):
        super(MultiTaskConnotationEmbedder, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.enc_type = enc_type

        print("drop prob: {}".format(drop_prob))
        self.dropout = nn.Dropout(p=drop_prob)

        # if self.enc_type == 'lstm':
        self.def_encoder = nn.GRU(self.embed_dim, self.hidden_dim, bidirectional=True,
                                     batch_first=True)
        # else:
        #     self.encoder_layer = tm.TransformerEncoderLayer(self.embed_dim, nhead=4,
        #                                                     dim_feedforward=self.hidden_dim)
        #     self.def_encoder = tm.TransformerEncoder(self.encoder_layer, num_layers=2)
        #     self.cls = nn.Parameter(torch.zeros(self.embed_dim)) # `final' hidden
        #     self.hidden_dim = self.embed_dim // 2

        if not use_verb:
            self.predictor = MultiTaskPredictor(self.hidden_dim * 2, dims, use_emo, use_cuda,
                                                use_word=use_word)
        else:
            self.predictor = MultiTaskPredictorVerbs(self.hidden_dim * 2, dims, use_emo, use_cuda,
                                                     use_word=use_word)

        self.use_cuda = use_cuda


    def forward(self, def_text, word, def_l, word_l, foo):
        hn = self.get_hidden(def_text, word, def_l, word_l, foo)

        y_pred_lst = self.predictor(hn, word)
        return y_pred_lst

    def get_hidden(self, def_text, word, def_l, word_l, foo):
        def_text = self.dropout(def_text)
        # def_text = (B, T, E)
        # if self.enc_type == 'lstm':
        p_def_text = rnn.pack_padded_sequence(def_text, def_l, enforce_sorted=False, batch_first=True)
        _, combo_fb_hn = self.def_encoder(p_def_text)  # (num_dir, B, H) # FOR GRU
        hn = combo_fb_hn.transpose(0, 1).reshape(-1, self.hidden_dim * 2)  # (B, 2*H)
        # else:
        #     full_input = torch.cat((self.cls.unsqueeze(0).repeat(def_text.shape[0], 1).unsqueeze(1), def_text), dim=1)
        #     mask = (full_input == 0)[:, :, 0]
        #     res = self.def_encoder(full_input.transpose(0, 1), src_key_padding_mask=mask)
        #     hn = res[0]

        hn = nn.functional.normalize(hn, dim=1)
        return hn


class MultiTaskConnotationEmbedderWithRelated(torch.nn.Module):
    def __init__(self, hidden_dim, embed_dim, dims, use_emo, drop_prob=0, use_cuda=False,
                 enc_type='lstm', use_word=False, att_type='reg', use_verb=False):
        super(MultiTaskConnotationEmbedderWithRelated, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.enc_type = enc_type

        self.att = AttentionLayer(self.hidden_dim, 2 * self.hidden_dim, use_cuda,
                                  att_type=att_type)

        print("drop prob: {}".format(drop_prob))
        self.dropout = nn.Dropout(p=drop_prob)

        # if self.enc_type == 'lstm':
        self.def_encoder = nn.GRU(self.embed_dim, self.hidden_dim, bidirectional=True,
                                     batch_first=True)
        # else:
        #     self.encoder_layer = tm.TransformerEncoderLayer(self.embed_dim, nhead=4,
        #                                                     dim_feedforward=self.hidden_dim)
        #     self.def_encoder = tm.TransformerEncoder(self.encoder_layer, num_layers=2)
        #     self.cls = nn.Parameter(torch.zeros(self.embed_dim)) # `final' hidden
        #     self.hidden_dim = self.embed_dim // 2

        if not use_verb:
            self.predictor = MultiTaskPredictor(self.hidden_dim * 2, dims, use_emo, use_cuda,
                                                use_word=use_word)
        else:
            self.predictor = MultiTaskPredictorVerbs(self.hidden_dim * 2, dims, use_emo, use_cuda,
                                                     use_word=use_word)

        self.use_cuda = use_cuda

    def forward(self, def_text, word, def_l, word_l, rel):
        hn = self.get_hidden(def_text, word, def_l, word_l, rel)

        y_pred_lst = self.predictor(hn, word)
        return y_pred_lst

    def get_hidden(self, def_text, word, def_l, word_l, rel):
        def_text = self.dropout(def_text)
        # def_text = (B, T, E)
        # if self.enc_type == 'lstm':
        p_def_text = rnn.pack_padded_sequence(def_text, def_l, enforce_sorted=False, batch_first=True)
        _, combo_fb_hn = self.def_encoder(p_def_text)  # (num_dir, B, H) # FOR GRU
        hn = combo_fb_hn.transpose(0, 1).reshape(-1, self.hidden_dim * 2)  # (B, 2*H)
        # else:
        #     full_input = torch.cat((self.cls.unsqueeze(0).repeat(def_text.shape[0], 1).unsqueeze(1), def_text), dim=1)
        #     mask = (full_input == 0)[:, :, 0]
        #     res = self.def_encoder(full_input.transpose(0, 1), src_key_padding_mask=mask)
        #     hn = res[0]

        avec = self.att(rel, hn, hn, rel)
        hn = nn.functional.normalize(hn + avec, dim=1)
        return hn
