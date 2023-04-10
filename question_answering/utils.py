from collections import OrderedDict
from Levenshtein import distance
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
import re
import torch
from transformers import BertTokenizer, BertModel, BertConfig
import pickle
from deduction.bert_utils import input_to_bert
from torch.utils.data import DataLoader
import chinese_converter
from unidecode import unidecode
import os
from CLRN.utils import get_bert_output
from matching.utils import get_features
from bidict import bidict

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# debug
import pdb


def load_one_hop_data(data_path):
    # TODO:load question
    print('loading one hop data from directory ...\n')
    test_data = []
    for line in open(data_path):
        line = re.split('@@@|>', line)
        q = line[0]
        head_e = line[1].split('resource/')[-1]
        head_r = line[2].split('property/')[-1]
        head_t = line[3].split('resource/')[-1]
        complete_gold = [line[1], line[2], line[3]]
        gold_path = [head_e, head_r]
        path = [head_e]
        test_data.append(dict([("question", q),
                               ("head_e", head_e),
                               ("path", path),
                               ("gold_path", gold_path),
                               ("complete_gold", complete_gold)]))

    return test_data


def load_alignment(dual_alignment_path):
    alignment = bidict()
    for alignment_path in dual_alignment_path:
        for line in open(alignment_path):
            line = line.split('>')
            e1 = line[0].split('resource/')[-1]
            e2 = line[1].split('resource/')[-1]
            if e2 not in alignment.values():
                alignment[e1] = e2
    return alignment


def get_path_ranker(bert, dual_dest, tri_pair_path, device=-1, nhop=2, type=""):
    if os.path.exists("{}hop_ranker".format(nhop) + str(dual_dest) + type):
        print("loading ranker...\n")
        ranker_file = open("{}hop_ranker".format(nhop) + str(dual_dest) + type, 'rb')
        ranker = pickle.load(ranker_file)

    else:
        print("processing train data...\n")
        rank_pairs, rank_labels = get_ranking_data(tri_pair_path)

        rank_data_loader = DataLoader(dataset=rank_pairs, batch_size=128,
                                      shuffle=False, num_workers=1, collate_fn=lambda x: x)
        bert_model, bert_tokenizer, bert_config = bert
        rank_features = get_features(bert_model, bert_config, bert_tokenizer, rank_data_loader, device=device)
        if nhop == 2:
            ranker = MLPClassifier(max_iter=1000, alpha=0.05, learning_rate='adaptive', hidden_layer_sizes=(50, 50, 50))
        else:
            ranker = MLPClassifier(max_iter=1000, alpha=0.0001, learning_rate='adaptive', activation='tanh')

        ranker.fit(rank_features, rank_labels)

        ranker_file = open("{}hop_ranker".format(nhop) + str(dual_dest) + type, 'wb')
        pickle.dump(ranker, ranker_file)
    ranker_file.close()
    return ranker


def get_ranking_data(tri_pair_path):
    questions = []
    equal_paths = []
    labels = []
    equal_nums = []
    count = 0
    for pair_path in tri_pair_path:
        question_path, equal_path = pair_path
        for line in open(equal_path):
            if line == "\n":
                equal_nums.append(count)
                continue

            line = line.split("\t")
            if "gold" in line[0]:
                h = re.split(">|resource/", line[1])[-1]
                rs = [re.split("property/|>", ele)[-1] for ele in line if "property/" in ele]
                labels.append(1)
                count = 0
                gold_h = h
                gold_rs = rs
            else:
                h = re.split(">|resource/", line[1])[-2]

                rs = [re.split("property/|>", ele)[-2] for ele in line if "property/" in ele]

                if gold_h == h and gold_rs == rs:
                    continue

                labels.append(0)

            path = [h] + rs
            equal_paths.append(path)
            count += 1

    count = 0
    for pair_path in tri_pair_path:
        question_path, equal_path = pair_path
        for idx, line in enumerate(open(question_path)):
            line = line.split("@@@")
            question = [line[0]]
            questions.extend([question] * equal_nums[count])
            count += 1

    pairs = list(zip(questions, equal_paths))
    return pairs, labels


def load_two_hop_data(data_path):
    # TODO:load question
    print('loading 2 hop data from directory ...\n')
    test_data = []
    for line in open(data_path):
        line = re.split('@@@|>', line)
        q = line[0]
        head_e = line[1].split('resource/')[-1]
        head_r = line[2].split('property/')[-1]
        head_tail = line[3].split('resource/')[-1]
        mid_e = line[4].split('resource/')[-1]
        mid_r = line[5].split('property/')[-1]
        mid_tail = line[6].split('resource/')[-1]
        gold_path = [head_e, head_r, mid_r]
        complete_gold = [line[1], line[2], line[3], line[4], line[5], line[6]]
        path = [head_e]
        test_data.append(dict([("question", q),
                               ("head_e", head_e),
                               ("path", path),
                               ("gold_path", gold_path),
                               ("complete_gold", complete_gold)]))

    return test_data


def load_three_hop_data(data_path):
    # TODO:load question
    print('loading 3 hop data from directory ...\n')
    test_data = []
    for line in open(data_path):
        line = re.split('@@@|>', line)
        q = line[0]
        head_e = line[1].split('resource/')[-1]
        head_r = line[2].split('property/')[-1]
        head_t = line[3].split('resource/')[-1]
        mid_e = line[4].split('resource/')[-1]
        mid_r = line[5].split('property/')[-1]
        mid_t = line[6].split('resource/')[-1]
        tail_h = line[7].split('resource/')[-1]
        tail_r = line[8].split('property/')[-1]
        tail_t = line[9].split('resource/')[-1]
        gold_path = [head_e, head_r, mid_r, tail_r]
        complete_gold = [line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9]]
        path = [head_e]
        test_data.append(dict([("question", q),
                               ("head_e", head_e),
                               ("path", path),
                               ("gold_path", gold_path),
                               ("complete_gold", complete_gold)]))

    return test_data


def load_dual_kg(kg_dual_path, dual_dest, bert, device=-1):
    if os.path.exists("../data/dual_kg" + str(dual_dest)):
        dual_kg_file = open("../data/dual_kg" + str(dual_dest), 'rb')
        dual_kg, dual_alignment_dict = pickle.load(dual_kg_file)
        dual_kg_file.close()
    else:
        print('loading data from directory {} ...\n'.format(kg_dual_path))
        dual_kg = []
        dual_alignment_dict = []
        for kg_path in kg_dual_path:
            alignment_dict = dict()
            tris = []
            adapt_es = []
            for line in open(kg_path):
                line = line.split()
                h = re.split('resource/|>', line[0])[-2]

                if 'zh.' in line[0]:
                    adapt_e = chinese_converter.to_simplified(h)
                elif 'fr.' in line[0]:
                    adapt_e = unidecode(h)
                else:
                    adapt_e = unidecode(h)
                adapt_e = matching_adapt(adapt_e)

                r = re.split('property/|>', line[1])[-2]
                t = re.split('resource/|>', line[2])[-2]
                tris.append((h, r, t))

                adapt_es.append(adapt_e)
                alignment_dict[h] = line[0]
                alignment_dict[r] = line[1]
                alignment_dict[t] = line[2]

            kg = [tris, adapt_es]

            # debug
            dual_kg.append(kg)
            dual_alignment_dict.append(alignment_dict)

        if os.path.exists("../data/dual_kg_dict" + str(dual_dest)):
            dual_kg_dict_file = open("../data/dual_kg_dict" + str(dual_dest), 'rb')
            dual_kg_dict = pickle.load(dual_kg_dict_file)
            dual_kg_dict_file.close()
            print("post process dual kgs to bert encode and adapt them...")

            # dual_kg = post_process_kg(dual_kg, dual_kg_dict, bert, device=device)
            dual_kg[0].append(dual_kg_dict[0])
            dual_kg[1].append(dual_kg_dict[1])
            dual_kg_pair = (dual_kg, dual_alignment_dict)
            dual_kg_file = open("../data/dual_kg" + str(dual_dest), 'wb')
            pickle.dump(dual_kg_pair, dual_kg_file)
            dual_kg_file.close()
        else:
            exit(0)

    return dual_kg, dual_alignment_dict


def post_process_kg(dual_kg, dual_kg_dict, bert, device=-1):
    bert_model, bert_tokenizer, bert_config = bert
    # TODO:bert encode kg, and get kg ordered
    post_dual_kg = []
    for ib, kg in enumerate(dual_kg):
        kg_tris, kg_adapt_es, e_r_loader = kg
        kg_encs = []
        kg_lens = []
        loader_len = len(e_r_loader)
        for i, e_r_batch in enumerate(e_r_loader):
            print("posting processing the {} kg with progress {:.2f}%".format(ib + 1, i / loader_len * 100), end='\r')
            e_r_enc, e_r_lens, _ = get_bert_output(bert_model, bert_config, bert_tokenizer,
                                                   e_r_batch, is_e_r=True, device=device)
            kg_encs.extend([torch.squeeze(enc) for enc in torch.tensor_split(e_r_enc, len(e_r_enc))])
            kg_lens.extend(e_r_lens)
        print(f'kg {ib + 1} finished')
        kg = (kg_tris, kg_adapt_es, dual_kg_dict[ib], kg_encs, kg_lens)
        post_dual_kg.append(kg)

    return post_dual_kg


def load_fine_tuned_bert(fine_tubed_bert_path, device=-1):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
    bert_config = BertConfig.from_pretrained('bert-base-multilingual-cased')

    # debug

    if device == -1:
        bert_model.load_state_dict(torch.load(fine_tubed_bert_path, map_location=torch.device('cpu')))
    else:
        bert_model.load_state_dict(torch.load(fine_tubed_bert_path, map_location="cuda:{}".format(device)))

        bert_model.to(device)

    return bert_model, bert_tokenizer, bert_config


def get_candidate_from_id(eId_batch, eId_r_map):
    # TODO:
    candidate_batch = []

    for eId in eId_batch:
        if eId == -1:
            candidate_batch.append(['no_such_relation'])
        else:
            candidate_batch.append(eId_r_map[eId])

    return candidate_batch


def copy_path(p_batch):
    # deep copy the path since each path deduced from the question is unique
    copy_p_batch = []

    for ib, p in enumerate(p_batch):
        copy_p_batch.append([])
        for element in p:
            copy_p_batch[ib].append(element)

    return copy_p_batch


def matching_adapt(origin_e):
    if origin_e[0] == '(':
        adapt_e = origin_e.split(')')[1]
    else:
        adapt_e = origin_e.split('(')[0]
    adapt_e = "".join(t_e for t_e in adapt_e if t_e.isalnum()).lower()
    if len(adapt_e) == 0 or len(origin_e.split(')')) > 2:
        adapt_e = origin_e
    return adapt_e


def get_candidate(e_batch, kg_tris, kg_adapt_es, kg_dicts, NMN_alignment=None, is_first=False):
    candidate_tris = []
    candidate_e_r_nums = []
    trans_e_batch = []
    candidate_e_r_batch = []

    for match in e_batch:
        if match == " ":
            candidate_e_r_nums.append(0)
            continue

        e_r_num = 0
        candidate_e_r = []
        is_tail = False
        trans_e = match
        for ib, e_r_e in enumerate(kg_tris):
            if match == e_r_e[0]:
                candidate_e_r.append([e_r_e[0], e_r_e[1]])
                candidate_tris.append(e_r_e)
                e_r_num += 1
            if match == e_r_e[2]:
                is_tail = True

        if e_r_num == 0:
            # if is_tail or is_first:
            if is_first or (match not in kg_dicts):
                candidate_e_r_nums.append(0)
                continue

            trans_e = kg_dicts[match]
            adapt_e = matching_adapt(kg_dicts[match])
            dist = [distance(adapt_e, e) /
                    max(len(adapt_e), len(e))
                    for e in kg_adapt_es]
            dist = torch.tensor(dist)

            can_num = 15
            aligned_len = 0
            aligned_e = None
            if NMN_alignment:
                if match in NMN_alignment:
                    aligned_e = NMN_alignment[match]
                    can_num = 5
                elif match in NMN_alignment.inverse:
                    aligned_e = NMN_alignment.inverse[match]
                    can_num = 5
            if aligned_e:
                aligned_tris = [[h, r, t] for h, r, t in kg_tris if h == aligned_e]
                aligned_pairs = [[h, r] for h, r, t in kg_tris if h == aligned_e]
                candidate_tris.extend(aligned_tris)
                candidate_e_r.extend(aligned_pairs)
                aligned_len = len(aligned_tris)
                can_idx = torch.topk(dist, can_num, largest=False, sorted=True)
            else:
                while True:
                    can_idx = torch.topk(dist, can_num, largest=False, sorted=True)
                    if can_idx.values[0] < 0.2 and can_idx.values[0] != can_idx.values[-1]:
                        break
                    elif can_idx.values[-1] <= 0.7 and can_num < 30:
                        can_num = 2 * can_num
                    else:
                        break

            candidate_tris.extend([kg_tris[idx] for idx in can_idx.indices])
            candidate_e_r.extend([[kg_tris[idx][0], kg_tris[idx][1]] for idx in can_idx.indices])
            e_r_num = can_num + aligned_len

        trans_e_batch.append(trans_e)
        candidate_e_r_nums.append(e_r_num)
        candidate_e_r_batch.append(candidate_e_r)

    return trans_e_batch, candidate_tris, candidate_e_r_batch, candidate_e_r_nums


def get_from_batch(batch):
    # TODO: extract the question and head entity batch from the
    #  given input original batch
    q_batch = []
    head_batch = []
    p_batch = []
    gold_path = []
    path_scores_batch = []
    complete_path = []
    complete_gold = []

    for sample in batch:
        q_batch.append(sample["question"])
        head_batch.append(sample["head_e"])
        p_batch.append(sample["path"])
        gold_path.append(sample["gold_path"])
        complete_gold.append(sample["complete_gold"])
        complete_path.append([])
        path_scores_batch.append([])

    return q_batch, head_batch, p_batch, gold_path, path_scores_batch, complete_gold, complete_path


def get_next_entity(eId_batch, r_batch, eId_dict_map):
    # TODO: get the next entity batch given the previous entity and relation
    next_entity_batch = []
    next_eId_batch = []
    for ib, eId in enumerate(eId_batch):
        # debug
        if r_batch[ib] == 'no_such_relation':
            next_e, next_eId = 'no_such_entity', -1
        else:
            next_e, next_eId = eId_dict_map[eId][r_batch[ib]]

        next_entity_batch.append(next_e)
        next_eId_batch.append(next_eId)
    return next_entity_batch, next_eId_batch


def rank_path(bert, one_q_paths_batch, q_batch, all_q_paths, batch_id, device=-1):
    # TODO: Rank the paths based on similarity to the question
    bert_model, bert_tokenizer, bert_config = bert

    for ib, q in enumerate(q_batch):
        one_q_paths = []

        for path_idx in range(4):
            final_p_batch, middle_e_batch, tail_e_batch, pred_bad_batch = one_q_paths_batch[path_idx]

            if not pred_bad_batch[ib]:
                one_q_paths.append((final_p_batch[ib][1], middle_e_batch[ib], final_p_batch[ib][2], tail_e_batch[ib]))

            # one_q_paths[ib]["Batch{} Q{} P{}:".format(batch_id, ib, path_idx)] = dict()
            # one_q_paths[ib]["Batch{} Q{} P{}:".format(batch_id, ib, path_idx)]["final_p"] = final_p_batch[ib]
            # one_q_paths[ib]["Batch{} Q{} P{}:".format(batch_id, ib, path_idx)]["mid_e"] = middle_e_batch[ib]
            # one_q_paths[ib]["Batch{} Q{} P{}:".format(batch_id, ib, path_idx)]["pred_tail"] = tail_e_batch[ib]
            # one_q_paths[ib]["Batch{} Q{} P{}:".format(batch_id, ib, path_idx)]["pred_bad"] = pred_bad_batch[ib]

        all_q_paths.append(one_q_paths)


def combine_e_q_p(e_batch, q_batch, p_batch, e_r_nums):
    e_q_p_batch = []
    count = 0
    for idx, num in enumerate(e_r_nums):
        if num != 0:
            e_q_p = [[e_batch[count] + " " + q_batch[idx], " ".join(p_batch[idx])]]
            e_q_p_batch.append(e_q_p)
            count += 1
    return e_q_p_batch


def select_gold_r(score, r_nums, rs_batch):
    pred_bad_batch = []

    score_list = score.split(r_nums)
    preds = []

    for i, s in enumerate(score_list):

        idx = torch.argmax(s, dim=-1).item()
        # TODO: the bar to judge if the the positive relation is available
        preds.append(rs_batch[i][idx])
        if rs_batch[i][idx] == 'no_such_relation':
            pred_bad_batch.append(True)
            print("bad score:{}\n".format(s[idx]))
        else:
            pred_bad_batch.append(False)
            print("good score:{}\n".format(s[idx]))

    return preds, pred_bad_batch


def save_result(all_paths, alignments, dual_dest, n_hop, pair_n=""):
    f = open("saved_{}hop_alignments".format(n_hop) + str(dual_dest) + pair_n, "w")
    for alignment in alignments:
        f.write("{}\t{}\n".format(alignment[0], alignment[1]))
    f.close()

    f = open("{}hop_equivalent_path".format(n_hop) + str(dual_dest) + pair_n, "w")
    for idx, paths in enumerate(all_paths):

        for ib, path in enumerate(paths):
            if ib == 0:
                filling = f"{idx} gold\t"

            else:
                filling = f"{idx} equal\t"

            count = 1
            for p in path:
                if count % 3 == 0:
                    padding = "@"
                    count = 0
                else:
                    padding = "\t"
                filling += p + padding
                count += 1
            filling += "\n"
            f.write(filling)
        f.write("\n")
    f.close()
