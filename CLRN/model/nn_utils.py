import torch.nn as nn
import torch
import pdb


class LSTM(nn.Module):
    """
    LSTM that can perform mask
    """

    def __init__(self, d_input, d_h, n_layers=1, batch_first=True, birnn=True, dropout=0.3):
        """
        :param d_input: input dimension
        :param d_h: hidden dimension
        :param n_layers: layer number of LSTM
        :param batch_first: if True the input is [bs, max_seq_len, d_input] else [max_seq_len, bs, d_input]
        :param birnn: if True, BiLstm else LSTM
        :param dropout: probability of the dropout layer
        """
        super(LSTM, self).__init__()

        n_dir = 2 if birnn else 1
        self.init_h = nn.Parameter(torch.Tensor(n_layers * n_dir, d_h))
        self.init_c = nn.Parameter(torch.Tensor(n_layers * n_dir, d_h))

        INI = 1e-2
        torch.nn.init.uniform_(self.init_h, -INI, INI)
        torch.nn.init.uniform_(self.init_c, -INI, INI)

        self.lstm = nn.LSTM(
            input_size=d_input,
            hidden_size=d_h,
            num_layers=n_layers,
            bidirectional=birnn,
            batch_first=not batch_first
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, seqs, seq_lens=None, init_states=None):
        """
        :param seqs: [bs, max_seq_len, d_input] or [max_seq_len, bs, d_input]
        :param seq_lens: [bs]
        :param init_states: hidden and cell
        :return: [bs, max_seq_len, d_h]
        """

        bs = seqs.size(0)
        bf = self.lstm.batch_first

        if not bf:
            seqs = seqs.transpose(0, 1)

        seqs = self.dropout(seqs)

        size = (self.init_h.size(0), bs, self.init_h.size(1))
        if init_states is None:
            init_states = (self.init_h.unsqueeze(1).expand(*size).contiguous(),
                           self.init_c.unsqueeze(1).expand(*size).contiguous())

        if seq_lens is not None:
            assert bs == len(seq_lens)
            sort_ind = sorted(range(len(seq_lens)), key=lambda i: seq_lens[i], reverse=True)
            seq_lens = [seq_lens[i] for i in sort_ind]
            seqs = self.reorder_sequence(seqs, sort_ind, bf)
            init_states = self.reorder_init_states(init_states, sort_ind)

            packed_seq = nn.utils.rnn.pack_padded_sequence(seqs, seq_lens)
            packed_out, final_states = self.lstm(packed_seq, init_states)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out)

            back_map = {ind: i for i, ind in enumerate(sort_ind)}
            reorder_ind = [back_map[i] for i in range(len(seq_lens))]
            lstm_out = self.reorder_sequence(lstm_out, reorder_ind, bf)
            final_states = self.reorder_init_states(final_states, reorder_ind)
        else:
            lstm_out, final_states = self.lstm(seqs)
        return lstm_out.transpose(0, 1), final_states

    def reorder_sequence(self, seqs, order, batch_first=False):
        """
        seqs: [max_seq_len, bs, d] if not batch_first
        order: list of sequence length
        """
        batch_dim = 0 if batch_first else 1
        assert len(order) == seqs.size()[batch_dim]
        order = torch.LongTensor(order).to(seqs.device)
        sorted_seqs = seqs.index_select(index=order, dim=batch_dim)
        return sorted_seqs

    def reorder_init_states(self, states, order):
        """
        lstm_states: (H, C) of tensor [layer, batch, hidden]
        order: list of sequence length
        """
        assert isinstance(states, tuple)
        assert len(states) == 2
        assert states[0].size() == states[1].size()
        assert len(order) == states[0].size()[1]

        order = torch.LongTensor(order).to(states[0].device)
        sorted_states = (states[0].index_select(index=order, dim=1),
                         states[1].index_select(index=order, dim=1))
        return sorted_states


def pooling(emb, lens, type):
    """
    pooling operation with mask
    :param emb: [bs, max_seq_len, d]
    :param lens: [bs] list of length
    :param type: last --> output last one of hidden states
                 avg  --> output average of all embeddings
                 max  --> output
    :return:
    """
    assert type in ["last", "avg", "max"]
    bs = len(emb)
    d_h = emb.size(-1)
    pooling_emb = torch.zeros(bs, d_h)

    if emb.is_cuda:
        pooling_emb = pooling_emb.to(emb.device)

    if type == "last":
        for i in range(bs):
            pooling_emb[i] = emb[i, lens[i] - 1]
    else:
        mask = build_mask(emb, lens)
        if type == "max":
            emb = emb.masked_fill(mask == 0, -float("inf"))
            pooling_emb = emb.max(dim=1)
        else:
            emb = emb.masked_fill(mask == 0, 0.0)
            pooling_emb = emb.mean(dim=1)
    return pooling_emb


def build_mask(seq, seq_lens, dim=-2):
    """
    :param seq: [bs, max_seq_len, d] or [bs, max_seq_len]
    :param seq_lens: [bs]
    :param dim:  the dimension to be masked
    :return: [bs, max_seq_len, d] or [bs, max_seq_len]
    """
    mask = torch.zeros_like(seq)
    if dim == -1:
        mask.transpose_(-2, -1)
    for i, l in enumerate(seq_lens):
        mask[i, :l].fill_(1)
    if dim == -1:
        mask.transpose_(-2, -1)
    return mask


def cal_loss(pos_score, neg_score, neg_e_nums, margin=0.5):
    loss_function = nn.MarginRankingLoss(margin=margin)

    pos_score_ext = []

    for ib, neg_e_num in enumerate(neg_e_nums):
        s_ext = pos_score[ib].expand(neg_e_num)
        pos_score_ext.append(s_ext)

    pos_score_ext = torch.cat(pos_score_ext, dim=0)
    ones = torch.ones(len(pos_score_ext))
    if pos_score_ext.is_cuda:
        ones = ones.to(pos_score_ext.device)
    loss = loss_function(pos_score_ext, neg_score, ones)
    return loss


def predict(scores, candidate_batch, candidate_nums):
    preds = []
    start = end = 0
    for ib, num in enumerate(candidate_nums):

        end += num
        idx = torch.argmax(scores[start:end])
        pred = candidate_batch[ib][idx]
        preds.append(pred)
        start = end

    return preds