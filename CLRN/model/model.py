import sys
import torch.nn as nn
import torch
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))

from nn_utils import LSTM, pooling, build_mask


class Model(nn.Module):
    def __init__(self, d_bert, d_h, n_layers, dropout_prob):
        super(Model, self).__init__()
        self.d_bert = d_bert
        self.d_h = d_h
        self.dropout_prob = dropout_prob

        # Bilstm for e_question pair
        self.encoder_e_q = LSTM(d_input=d_bert, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                                dropout=dropout_prob, birnn=True)
        # Bilstm for e_r pair extracted from kg
        self.encoder_e_r = LSTM(d_input=d_bert, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                                dropout=dropout_prob, birnn=True)

        # Bilstm for path
        self.encoder_p = LSTM(d_input=d_bert, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)

        self.W_att = nn.Linear(d_h * 2, 1)

        self.softmax = nn.Softmax(dim=-1)

        self.cos = nn.CosineSimilarity()

    def forward(self,
                e_q_bert_enc, e_q_lens,
                p_bert_enc, p_lens,
                e_r_bert_enc, e_r_lens, e_r_nums,
                pooling_type="last"):
        """
        :param e_q_bert_enc: [total_e_num, max_e_len, 768]
        :param e_q_lens: [total_e_num]
        :param p_bert_enc: [total_e_num, max_e_len, 768]
        :param p_lens: [total_e_num]
        :param e_r_bert_enc: [total_e_num]
        :param e_r_lens: [total_e_num]
        :param e_r_nums: [total_e_num]
        :param pooling_type:
        :return score: [total_e_num]
        """
        # step1: rid of path information from question
        # q_e_enc: [total_e_num, max_e_len, d_h]
        e_q_enc, _ = self.encoder_e_q(e_q_bert_enc, e_q_lens)

        # p_enc: [total_e_num, max_e_len, d_h]
        p_enc, _ = self.encoder_p(p_bert_enc, p_lens)
        # [total_e_num, d_h], pooling operation
        p_pooling = pooling(p_enc, p_lens, pooling_type)

        # p_pooling: [total_r_num, d_h] --> [total_r_num, max_q_len, d_h]
        # concat q_enc and p_pooling: [total_r_num, max_q_len, d_h] cat [total_r_num, max_q_len, d_h] --> [total_r_num, max_q_len, d_h * 2]
        # att_weights: [total_r_num, max_q_len] attention weights of each question word.
        att_weights = self.W_att(
            torch.cat([e_q_enc, p_pooling.unsqueeze(1).expand(*e_q_enc.size())], dim=-1)
        ).squeeze(-1)
        # att_mask: [total_r_num, max_q_len]
        att_mask = build_mask(att_weights, e_q_lens, dim=-2)
        # softmax, att_weights: [total_r_num, max_q_len]
        att_weights = self.softmax(att_weights.masked_fill(att_mask == 0, -float("inf")))

        # [total_r_num, d_h]
        e_q_pooling = pooling(e_q_enc, e_q_lens, pooling_type)

        # get question vector
        # att_weights: [total_e_num, max_q_len] --> [total_r_num, max_q_len, 1]
        # [total_e_num, max_q_len, d_h] * [total_e_num, max_q_len, 1] --> [total_e_num, max_q_len, d_h]
        # q_context: [total_e_num, max_q_len, d_h] --> [total_r_num, d_h]
        e_q_used_context = torch.sum(torch.mul(e_q_enc, att_weights.unsqueeze(-1)), dim=1)
        # [total_e_num, d_h]
        e_q_contexts = torch.sub(e_q_pooling, e_q_used_context)

        # e_r_enc: [total_e_num * e_r_nums, max_e_len, d_h]
        e_r_enc, _ = self.encoder_e_r(e_r_bert_enc, e_r_lens)
        e_r_pooling = pooling(e_r_enc, e_r_lens, pooling_type)

        e_q_context_cat = []
        for i, e_q_context in enumerate(e_q_contexts):
            e_r_num = e_r_nums[i]
            e_q_context_ext = e_q_context.unsqueeze(0).expand(e_r_num, -1)
            e_q_context_cat.append(e_q_context_ext)
        e_q_context_cat = torch.cat(e_q_context_cat, dim=0).squeeze(-1)
        # [total_e_num]
        score = self.cos(e_q_context_cat, e_r_pooling)

        return score
