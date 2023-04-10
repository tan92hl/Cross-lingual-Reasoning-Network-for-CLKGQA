import sys
import torch

sys.path.append("..")


def get_bert_input(bert_tokenizer, q, p, r):
    tokens = []
    segment_ids = []

    tokens.append("[CLS]")
    segment_ids.append(0)

    # Question
    q_index_st = len(tokens)
    for i, tok in enumerate(q):
        sub_toks = bert_tokenizer.tokenize(tok)
        tokens += sub_toks
        segment_ids += [0] * len(sub_toks)
    q_index_ed = len(tokens)
    tokens.append("[SEP]")
    segment_ids.append(0)
    q_index = (q_index_st, q_index_ed)

    # Path
    p_index_st = len(tokens)
    for i, tok in enumerate(p):
        sub_toks = bert_tokenizer.tokenize(tok)
        tokens += sub_toks
        segment_ids += [1] * len(sub_toks)
    p_index_ed = len(tokens)
    p_index = (p_index_st, p_index_ed)
    tokens.append("[SEP]")
    segment_ids.append(1)

    # Relation
    r_index_st = len(tokens)
    for i, tok in enumerate(r):
        sub_toks = bert_tokenizer.tokenize(tok)
        tokens += sub_toks
        segment_ids += [1] * len(sub_toks)
    r_index_ed = len(tokens)
    r_index = (r_index_st, r_index_ed)
    tokens.append("[SEP]")
    segment_ids.append(1)

    return tokens, segment_ids, q_index, p_index, r_index


def get_bert_output(bert_model, bert_tokenizer, q_batch, p_batch, rs_batch, device=-1):
    input_ids = []
    tokens = []
    segment_ids = []
    input_mask = []

    q_index = []
    p_index = []
    r_index = []

    q_lens = []
    p_lens = []
    r_lens = []

    max_seq_len = 0

    r_nums = []

    for ib, q in enumerate(q_batch):
        rs = rs_batch[ib]
        # debug

        p = p_batch[ib]
        r_nums.append(len(rs))

        for r in rs:
            tokens_one, segment_ids_one, q_index_one, p_index_one, r_index_one = get_bert_input(bert_tokenizer, q, p, r)

            q_index.append(q_index_one)
            q_lens.append(q_index_one[1] - q_index_one[0])
            p_index.append(p_index_one)
            p_lens.append(p_index_one[1] - p_index_one[0])
            r_index.append(r_index_one)
            r_lens.append(r_index_one[1] - r_index_one[0])

            max_seq_len = max(max_seq_len, len(tokens_one))

            input_ids_one = bert_tokenizer.convert_tokens_to_ids(tokens_one)
            input_mask_one = [1] * len(input_ids_one)

            input_ids.append(input_ids_one)
            tokens.append(tokens_one)
            segment_ids.append(segment_ids_one)
            input_mask.append(input_mask_one)

    # padding to max length
    for ib in range(len(input_ids)):
        while len(input_ids[ib]) < max_seq_len:
            input_ids[ib].append(0)
            input_mask[ib].append(0)
            segment_ids[ib].append(0)

    # convert to tensor
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    input_mask_tensor = torch.tensor(input_mask, dtype=torch.long)
    segment_ids_tensor = torch.tensor(segment_ids, dtype=torch.long)

    if device != -1:
        input_ids_tensor = input_ids_tensor.to(device)
        input_mask_tensor = input_mask_tensor.to(device)
        segment_ids_tensor = segment_ids_tensor.to(device)

    all_layers_bert_enc, pooling_output = bert_model(input_ids=input_ids_tensor,
                                                     token_type_ids=segment_ids_tensor,
                                                     attention_mask=input_mask_tensor,
                                                     return_dict=False)
    all_layers_bert_enc = [all_layers_bert_enc]

    return all_layers_bert_enc, pooling_output, tokens, \
           q_lens, q_index, p_lens, p_index, r_lens, r_index, r_nums


def get_bert_enc(all_layers_bert_enc, index, lens, d_h_bert,
                 n_layers_bert, n_layers_bert_out, device=-1):
    n_layers_bert = 1
    bs = len(lens)
    max_len = max(lens)
    if device != -1:
        enc = torch.zeros([bs, max_len, d_h_bert * n_layers_bert_out]).to(device)
    else:
        enc = torch.zeros([bs, max_len, d_h_bert * n_layers_bert_out])

    for b in range(bs):
        index_one = index[b]
        for j in range(n_layers_bert_out):
            # index of last j-th layer
            i_layer = n_layers_bert - 1 - j
            st = j * d_h_bert
            ed = (j + 1) * d_h_bert
            enc[b, 0:(index_one[1] - index_one[0]), st:ed] = \
                all_layers_bert_enc[i_layer][b, index_one[0]:index_one[1], :]
    return enc


def bert_encode_qpr(bert_config, bert_model, bert_tokenizer, q_batch, p_batch, r_batch,
                    n_layers_bert_out=1, device=-1):
    all_layers_bert_enc, \
    pooling_output, tokens, \
    q_lens, q_index, \
    p_lens, p_index, \
    r_lens, r_index, r_nums = get_bert_output(bert_model, bert_tokenizer, q_batch, p_batch, r_batch, device=device)

    q_enc = get_bert_enc(all_layers_bert_enc, q_index, q_lens,
                         bert_config.hidden_size,
                         bert_config.num_hidden_layers,
                         n_layers_bert_out,
                         device=device)

    p_enc = get_bert_enc(all_layers_bert_enc, p_index, p_lens,
                         bert_config.hidden_size,
                         bert_config.num_hidden_layers,
                         n_layers_bert_out,
                         device=device)

    r_enc = get_bert_enc(all_layers_bert_enc, r_index, r_lens,
                         bert_config.hidden_size,
                         bert_config.num_hidden_layers,
                         n_layers_bert_out,
                         device=device)
    return q_enc, q_lens, p_enc, p_lens, r_enc, r_lens, r_nums


def input_to_bert(bert_model, bert_config, input_ids, input_masks, input_segments, e_lens, max_seq_len, device=-1):
    for ib in range(len(e_lens)):
        e_len = len(input_ids[ib])
        if e_len < max_seq_len:
            dif = max_seq_len - e_len
            input_ids[ib] += [0] * dif
            input_masks[ib] += [0] * dif
            input_segments[ib] += [0] * dif

    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    input_masks_tensor = torch.tensor(input_masks, dtype=torch.float)
    input_segments_tensor = torch.tensor(input_segments, dtype=torch.long)

    if device != -1:
        input_ids_tensor = input_ids_tensor.to(device)
        input_masks_tensor = input_masks_tensor.to(device)
        input_segments_tensor = input_segments_tensor.to(device)

    last_layer_bert_enc, _ = bert_model(input_ids=input_ids_tensor,
                                        token_type_ids=input_segments_tensor,
                                        attention_mask=input_masks_tensor,
                                        return_dict=False)

    if device == -1:
        entity_enc_batch = torch.zeros(len(e_lens), bert_config.hidden_size)
    else:
        entity_enc_batch = torch.zeros(len(e_lens), bert_config.hidden_size).to(device)

    for ib, e_len in enumerate(e_lens):
        entity_enc_batch[ib] = torch.mean(last_layer_bert_enc[ib, :e_len, :], dim=0)

    return entity_enc_batch
