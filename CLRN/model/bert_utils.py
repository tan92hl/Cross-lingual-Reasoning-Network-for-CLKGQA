import sys
import torch
import pdb

sys.path.append("..")


def input_to_bert(bert_model, bert_config, input_ids, input_masks, input_segments, max_seq_len,
                  e_context_lens=None, p_lens=None, device=-1):
    for ib, input_id in enumerate(input_ids):
        if len(input_id) < max_seq_len:
            dif = max_seq_len - len(input_id)
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

    if e_context_lens and p_lens:

        if device == -1:
            e_q_enc_batch = torch.zeros(len(e_context_lens), max(e_context_lens), bert_config.hidden_size)
            p_enc_batch = torch.zeros(len(p_lens), max(p_lens), bert_config.hidden_size)
        else:
            e_q_enc_batch = torch.zeros(len(e_context_lens), max(e_context_lens),
                                        bert_config.hidden_size).to(device)
            p_enc_batch = torch.zeros(len(p_lens), max(p_lens), bert_config.hidden_size).to(device)

        for ib, e_len in enumerate(e_context_lens):
            p_len = p_lens[ib]
            e_q_enc_batch[ib][:e_len] = last_layer_bert_enc[ib, :e_len, :]
            p_enc_batch[ib][:p_len] = last_layer_bert_enc[ib, e_len:e_len + p_len, :]

        return e_q_enc_batch, p_enc_batch
    else:
        return last_layer_bert_enc


def get_bert_output(bert_model, bert_config, bert_tokenizer, entity_context_batch, is_e_r, device=-1):
    input_ids = []
    input_masks = []
    input_segments = []
    e_context_lens = []
    e_r_nums = []

    p_lens = []
    max_seq_len = 0

    for e_cs in entity_context_batch:
        e_r_nums.append(len(e_cs))
        for e_c in e_cs:
            if is_e_r:
                encoded_dict = bert_tokenizer(e_c[0] + " " + e_c[1])

            else:
                encoded_dict = bert_tokenizer(e_c[0], e_c[1])
                p_lens.append(encoded_dict['token_type_ids'].count(1))
            e_context_lens.append(encoded_dict['token_type_ids'].count(0))
            input_ids.append(encoded_dict['input_ids'])
            input_masks.append(encoded_dict['attention_mask'])
            input_segments.append(encoded_dict['token_type_ids'])
            max_seq_len = max(max_seq_len, len(encoded_dict['token_type_ids']))

    if is_e_r:
        e_r_enc_batch = input_to_bert(bert_model, bert_config,
                                      input_ids, input_masks, input_segments,
                                      max_seq_len, device=device)
        return e_r_enc_batch, e_context_lens, e_r_nums

    else:
        e_q_enc_batch, p_enc_batch = input_to_bert(bert_model, bert_config,
                                                   input_ids, input_masks, input_segments,
                                                   max_seq_len,
                                                   e_context_lens=e_context_lens, p_lens=p_lens, device=device)
        return e_q_enc_batch, e_context_lens, p_enc_batch, p_lens
