import pdb
import sys
from CLRN.model.bert_utils import get_bert_output
import numpy
from bert_utils import bert_encode_qpr
from utils import *

sys.path.append("..")


def predict_gold_relation(model, bert_config, bert_model, bert_tokenizer,
                          q_batch, p_batch, rs_batch, pooling_type="last", device=-1):
    # DONE
    # TODO:make one step prediction based on the input

    q_enc, q_lens, \
    p_enc, p_lens, \
    r_enc, r_lens, r_nums = bert_encode_qpr(bert_config, bert_model, bert_tokenizer,
                                            q_batch, p_batch, rs_batch,
                                            n_layers_bert_out=1, device=device)

    score = model(q_enc, q_lens,
                  p_enc, p_lens,
                  r_enc, r_lens,
                  pooling_type=pooling_type)

    gold_r_batch, prediction_bad_batch = select_gold_r(score, r_nums, rs_batch)

    return gold_r_batch, prediction_bad_batch


def predict_tri(scores, candidate_tris, candidate_nums, pred_scores):
    matched_es = []
    pred_rs = []
    pred_es = []

    start = end = 0
    for ib, num in enumerate(candidate_nums):
        if num == 0:
            matched_es.append(" ")
            pred_rs.append(" ")
            pred_es.append(" ")
            pred_scores[ib].append(-1)
        else:
            end += num
            idx = torch.argmax(scores[start:end])
            pred_scores[ib].append(scores[start:end][idx])
            matched_es.append(candidate_tris[start + idx][0])
            pred_rs.append(candidate_tris[start + idx][1])
            pred_es.append(candidate_tris[start + idx][2])
            start = end

    return matched_es, pred_rs, pred_es


def predict_one_step(model, bert,
                     e_batch, p_batch, q_batch,
                     pred_scores,
                     hop1_complete_path,
                     kg, is_first=False,
                     NMN_alignment=None,
                     pooling_type="last", device=-1):
    kg_tris, kg_adapt_es, kg_dicts = kg

    bert_model, bert_tokenizer, bert_config = bert

    trans_e, candidate_tris, candidate_e_r_batch, complete_e_r_nums = \
        get_candidate(e_batch, kg_tris, kg_adapt_es, kg_dicts, NMN_alignment=NMN_alignment, is_first=is_first)

    if len(candidate_tris) == 0:
        scores = None

    else:

        e_q_p_batch = combine_e_q_p(trans_e, q_batch, p_batch, complete_e_r_nums)

        e_q_enc, e_q_lens, p_enc, p_lens = get_bert_output(bert_model, bert_config, bert_tokenizer,
                                                           e_q_p_batch,
                                                           is_e_r=False, device=device)

        candidate_e_r_enc, candidate_e_r_lens, effective_e_r_nums = get_bert_output(bert_model, bert_config,
                                                                                    bert_tokenizer,
                                                                                    candidate_e_r_batch,
                                                                                    is_e_r=True, device=device)

        scores = model(e_q_enc, e_q_lens, p_enc, p_lens,
                       candidate_e_r_enc, candidate_e_r_lens, effective_e_r_nums,
                       pooling_type=pooling_type)

    matched_es, pred_rs, pred_es = predict_tri(scores, candidate_tris, complete_e_r_nums, pred_scores)

    for idx, p in enumerate(p_batch):
        hop1_complete_path[idx].append(matched_es[idx])
        hop1_complete_path[idx].append(pred_rs[idx])
        hop1_complete_path[idx].append(pred_es[idx])
        # hop1_complete_path[idx].append(str(float(pred_scores[idx][-1])))
        p.append(pred_rs[idx])

    return p_batch, pred_es, matched_es


def predict(data_loader, model, bert, dual_kg, dual_alignment_dict, n_hop, ranker,
            NMN_alignment=None, pooling_type="last", device=-1):
    # Get the 4 possible path in the 2-hop scenario
    all_paths = []

    alignments = set()
    batch_len = len(data_loader)
    count = 0
    hits_top1 = 0
    hits_topk = 0
    # debug
    """dual_kg = [[dual_kg[0][0], dual_kg[0][1], dual_kg[1][2], dual_kg[0][3], dual_kg[0][4]],
               [dual_kg[1][0], dual_kg[1][1], dual_kg[0][2], dual_kg[1][3], dual_kg[1][4]]]"""

    for i, batch in enumerate(data_loader):
        alignment_once = {"hop1_h": [], "hop1_t": []}
        print("Having completed the {}hop data by {:.2f}%...".format(n_hop, i / batch_len * 100), end='\r')
        paths_batch = []
        complete_batch = []
        scores_batch = []

        q_batch, head_batch, p_batch_origin, gold_path, path_scores_batch, complete_gold, complete_path = \
            get_from_batch(batch)

        for kg in dual_kg:
            p_batch = copy_path(p_batch_origin)
            hop1_score_batch = copy_path(path_scores_batch)
            hop1_complete_path = copy_path(complete_path)
            hop1_p_batch_origin, hop1_t_batch, hop1_h_batch = \
                predict_one_step(model, bert, head_batch, p_batch, q_batch,
                                 hop1_score_batch, hop1_complete_path, kg, NMN_alignment=NMN_alignment, is_first=True,
                                 pooling_type=pooling_type, device=device)
            alignment_once["hop1_h"].append(hop1_h_batch)
            alignment_once["hop1_t"].append(hop1_t_batch)
            if n_hop == 1:
                paths_batch.append(hop1_p_batch_origin)
                complete_batch.append(hop1_complete_path)
                scores_batch.append(hop1_score_batch)
                continue

            for kg_2itr in dual_kg:
                hop2_score_batch = copy_path(hop1_score_batch)
                hop1_p_batch = copy_path(hop1_p_batch_origin)
                hop2_complete_path = copy_path(hop1_complete_path)
                hop2_p_batch_origin, hop2_t_batch, hop2_h_batch = \
                    predict_one_step(model, bert, hop1_t_batch, hop1_p_batch, q_batch,
                                     hop2_score_batch, hop2_complete_path, kg_2itr, NMN_alignment=NMN_alignment,
                                     pooling_type=pooling_type, device=device)

                if "hop2_t" not in alignment_once:
                    alignment_once["hop2_t"] = []
                alignment_once["hop2_t"].append(hop2_t_batch)
                if n_hop == 2:
                    complete_batch.append(hop2_complete_path)
                    paths_batch.append(hop2_p_batch_origin)
                    scores_batch.append(hop2_score_batch)
                    continue

                for kg_3itr in dual_kg:
                    hop3_score_batch = copy_path(hop2_score_batch)
                    hop2_p_batch = copy_path(hop2_p_batch_origin)
                    hop3_complete_path = copy_path(hop2_complete_path)
                    hop3_p_batch, hop3_t_batch, hop3_h_batch = \
                        predict_one_step(model, bert, hop2_t_batch, hop2_p_batch, q_batch,
                                         hop3_score_batch, hop3_complete_path, kg_3itr, NMN_alignment=NMN_alignment,
                                         pooling_type=pooling_type, device=device)

                    if "hop3_t" not in alignment_once:
                        alignment_once["hop3_t"] = []
                    alignment_once["hop3_t"].append(hop3_t_batch)
                    if n_hop == 3:
                        paths_batch.append(hop3_p_batch)
                        complete_batch.append(hop3_complete_path)
                        scores_batch.append(hop3_score_batch)

        align(alignment_once, dual_alignment_dict, alignments, n_hop)
        hits_top1, hits_topk, reordered_path = check_and_reorder(bert, hits_top1, hits_topk, paths_batch, q_batch,
                                                                 dual_alignment_dict, ranker,
                                                                 gold_path, complete_gold, complete_batch,
                                                                 device=device)
        all_paths.extend(reordered_path)

        count += len(reordered_path)
        # rank_path(bert, one_q_paths_batch, q_batch, all_q_paths, i, device=device)
    print(f"The estimated top 1 accuracy is: {hits_top1 / count:.5}")
    print(f"The estimated top k accuracy is: {hits_topk / count:.5}")
    return all_paths, alignments


def align(alignment_once, dual_alignment_dict, alignments, n_hop):
    hop_alignment_batch = []
    hop_alignment_batch.append(alignment_once["hop1_h"])
    hop_alignment_batch.append(alignment_once["hop1_t"])
    if n_hop > 1:
        hop_alignment_batch.append(alignment_once["hop2_t"])
    if n_hop > 2:
        hop_alignment_batch.append(alignment_once["hop3_t"])
    for hop_alignment in hop_alignment_batch:
        for i in range(len(hop_alignment[0])):
            pairs = [es[i] for es in hop_alignment]
            for pair_idx in range(int(len(pairs) / 2)):
                e0 = pairs[pair_idx * 2]
                e1 = pairs[pair_idx * 2 + 1]
                alignment_add(alignments, dual_alignment_dict, e0, e1)


def alignment_add(alignments, dual_alignment_dict, e0, e1):
    if e0 != e1 and e0 != " " and e1 != " ":
        if e0 in dual_alignment_dict[0]:
            alignments.add((dual_alignment_dict[0][e0], dual_alignment_dict[1][e1]))
        else:
            alignments.add((dual_alignment_dict[0][e1], dual_alignment_dict[1][e0]))


def align_map(dual_alignment_dict, complete_path):
    mapped_complete_path = []

    for idx in range(int(len(complete_path) / 3)):
        dict_n = 0
        for i in range(3):
            if complete_path[3 * idx + i] not in dual_alignment_dict[0]:
                dict_n = 1
                break
        for i in range(3):
            mapped_complete_path.append(dual_alignment_dict[dict_n][complete_path[3 * idx + i]])

    return mapped_complete_path


def check_and_reorder(bert, hits_top1, hits_topk, paths_batch, q_batch,
                      dual_alignment_dict, ranker,
                      gold_path, complete_gold, complete_path, device=-1):
    bert_model, bert_tokenizer, bert_config = bert
    reodered_path = []
    for idx, gold_p in enumerate(gold_path):
        hit_top1 = False
        hit_topk = False

        preds = [path_batch[idx] for path_batch in paths_batch if " " not in path_batch[idx]]

        if len(preds) == 0:
            continue
        pred_complete = [c[idx] for c in complete_path if " " not in c[idx]]

        questions = [q_batch[idx]] * len(preds)
        pairs = zip(questions, preds)
        pairs_feature = get_features(bert_model, bert_config, bert_tokenizer, [pairs], device=device)
        pred_scores = [y for _, y in ranker.predict_proba(pairs_feature)]
        for ib, pred in enumerate(preds):
            if pred[0] != pred_complete[ib][0]:
                pred_scores[ib] = min(pred_scores) - 1

        c_p = []
        for complete in pred_complete:
            c_p.append(align_map(dual_alignment_dict, complete))

        preds = [p for p, _ in sorted(zip(preds, pred_scores), key=lambda pair: pair[1], reverse=True)]
        pred_complete = [p for p, _ in sorted(zip(c_p, pred_scores), key=lambda pair: pair[1])]
        for ib, pred in enumerate(preds):
            if gold_p == pred:
                if ib == 0:
                    hit_top1 = True
                hit_topk = True

        hits_top1 += hit_top1
        hits_topk += hit_topk

        pred_complete.insert(0, complete_gold[idx])
        reodered_path.append(pred_complete)

    return hits_top1, hits_topk, reodered_path
