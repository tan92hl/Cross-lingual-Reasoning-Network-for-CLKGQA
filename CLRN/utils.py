import pdb
import re
import chinese_converter
import torch
from unidecode import unidecode
import sys
import os
import pathlib
from trans_utils import translate

sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
# from trans_utils import *
from model.bert_utils import get_bert_output
from bidict import bidict
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertModel, BertConfig
from Levenshtein import distance

import pickle

sys.path.append("..")


def get_data(origin_e_q_p_data, goldmap_e_r_data,
             trans_dual_kg, dual_kg_e, dual_kg, dual_kg_dict, dual_dest,
             alignment):
    if os.path.exists("../data/train_data_3q" + str(dual_dest)):
        print("loading training data from saved files...\n")
        train_data_file = open("../data/train_data_3q" + str(dual_dest), 'rb')
        train_data = pickle.load(train_data_file)

    else:
        print("processing train data...\n")
        train_data = get_train_data(origin_e_q_p_data[0], goldmap_e_r_data[0],
                                    trans_dual_kg, dual_kg_e, dual_kg, dual_kg_dict,
                                    is_train=True, alignment=alignment, dual_dest=dual_dest)
        train_data_file = open("../data/train_data_3q" + str(dual_dest), 'wb')
        pickle.dump(train_data, train_data_file)
    train_data_file.close()

    if os.path.exists("../data/dev_data_3q" + str(dual_dest)):
        print("loading dev data from saved files...\n")
        dev_data_file = open("../data/dev_data_3q" + str(dual_dest), 'rb')
        dev_data = pickle.load(dev_data_file)

    else:
        print("processing dev data...\n")
        dev1 = get_train_data(origin_e_q_p_data[1], goldmap_e_r_data[1],
                              trans_dual_kg, dual_kg_e, dual_kg, dual_kg_dict,
                              is_train=False, alignment=alignment, switch=False, dual_dest=dual_dest)
        dev2 = get_train_data(origin_e_q_p_data[2], goldmap_e_r_data[2],
                              trans_dual_kg, dual_kg_e, dual_kg, dual_kg_dict,
                              is_train=False, alignment=alignment, switch=True, dual_dest=dual_dest)
        dev_data = (dev1, dev2)
        dev_data_file = open("../data/dev_data_3q" + str(dual_dest), 'wb')
        pickle.dump(dev_data, dev_data_file)
    dev_data_file.close()

    if os.path.exists("../data/test_data_3q" + str(dual_dest)):
        print("loading test data from saved files...\n")
        test_data_file = open("../data/test_data_3q" + str(dual_dest), 'rb')
        test_data = pickle.load(test_data_file)

    else:
        print("processing test data...\n")
        test1 = get_train_data(origin_e_q_p_data[3], goldmap_e_r_data[3],
                               trans_dual_kg, dual_kg_e, dual_kg, dual_kg_dict,
                               is_train=False, alignment=alignment, switch=False, dual_dest=dual_dest)
        test2 = get_train_data(origin_e_q_p_data[4], goldmap_e_r_data[4],
                               trans_dual_kg, dual_kg_e, dual_kg, dual_kg_dict,
                               is_train=False, alignment=alignment, switch=True, dual_dest=dual_dest)
        test_data = (test1, test2)
        test_data_file = open("../data/test_data_3q" + str(dual_dest), 'wb')
        pickle.dump(test_data, test_data_file)
    test_data_file.close()

    return train_data, dev_data, test_data


def load_optimizer(model, lr, bert_model, bert_lr):
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0)
    bert_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, bert_model.parameters()), lr=bert_lr, weight_decay=0)

    return opt, bert_opt


def load_alignment(dual_alignment_path):
    alignment = bidict()
    for alignment_path in dual_alignment_path:
        for line in open(alignment_path,encoding='utf-8'):
            line = line.split('>')
            e1 = line[0].split('resource/')[-1]
            e2 = line[1].split('resource/')[-1]
            if e2 not in alignment.values():
                alignment[e1] = e2
    return alignment


def load_bert():
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
    bert_config = BertConfig.from_pretrained('bert-base-multilingual-cased')
    # debug

    return bert_model, bert_tokenizer, bert_config


def one_time_fix_dict(kg_dual_path, dual_dest):
    dual_kg_dict_file = open("../data/dual_kg_dict" + str(dual_dest), 'rb')
    dual_kg_dict = pickle.load(dual_kg_dict_file)
    dual_kg_dict_file.close()

    for ib, kg_path in enumerate(kg_dual_path):
        kg_dict = dual_kg_dict[ib]

        for line in open(kg_path,encoding='utf-8'):
            line = line.split()
            h = re.split('resource/|>', line[0])[-2]
            t = re.split('resource/|>', line[2])[-2]

            if h not in kg_dict:
                translated_e = translate(h.replace('_', ' '), dest=dual_dest[ib - 1], src=dual_dest[ib])
                kg_dict[h] = translated_e
            if t not in kg_dict:
                translated_e = translate(t.replace('_', ' '), dest=dual_dest[ib - 1], src=dual_dest[ib])
                kg_dict[t] = translated_e

    dual_kg_dict_file = open("../data/dual_kg_dict" + str(dual_dest), 'wb')
    pickle.dump(dual_kg_dict, dual_kg_dict_file)
    dual_kg_dict_file.close()



def load_dual_kg(kg_dual_path, dual_dest):
    if os.path.exists("../data/saved_kg" + str(dual_dest)):
        kg_data_file = open("../data/saved_kg" + str(dual_dest), 'rb')
        kg_data = pickle.load(kg_data_file)
        dual_kg, dual_kg_e, trans_dual_kg = kg_data
        kg_data_file.close()

    else:
        # TODO:load kg and question
        dual_kg = []
        dual_kg_e = []
        trans_dual_kg = []
        for ib, kg_path in enumerate(kg_dual_path):
            kg_e = set()
            kg = []
            trans_kg = []
            for line in open(kg_path,encoding='utf-8'):
                line = line.split()
                e = re.split('resource/|>', line[0])[-2]
                r = re.split('property/|>', line[1])[-2]
                if e[0] == '(':
                    trans_e = e.split(')')[1]
                else:
                    trans_e = e.split('(')[0]
                if len(trans_e) == 0 or len(e.split(')')) > 2:
                    trans_e = e
                if 'zh.' in line[0]:
                    trans_e = chinese_converter.to_simplified(trans_e)
                elif 'fr.' in line[0]:
                    trans_e = unidecode(trans_e)
                else:
                    trans_e = unidecode(trans_e)

                trans_e = ''.join(t_e for t_e in trans_e if t_e.isalnum()).lower()
                trans_kg.append([trans_e, r])
                kg.append([e, r])
                kg_e.add(e)
                kg_e.add(re.split('resource/|>', line[2])[-2])

            trans_dual_kg.append(trans_kg)
            dual_kg.append(kg)
            dual_kg_e.append(kg_e)
        kg_data = (dual_kg, dual_kg_e, trans_dual_kg)
        kg_data_file = open("../data/saved_kg" + str(dual_dest), 'wb')
        pickle.dump(kg_data, kg_data_file)
        kg_data_file.close()

    if os.path.exists("../data/dual_kg_dict" + str(dual_dest)):
        dual_kg_dict_file = open("../data/dual_kg_dict" + str(dual_dest), 'rb')
        dual_kg_dict = pickle.load(dual_kg_dict_file)
        dual_kg_dict_file.close()

    else:
        dual_kg_dict = translate_kg(dual_kg_e, dual_dest)
        dual_kg_dict_file = open("../data/dual_kg_dict" + str(dual_dest), 'wb')
        pickle.dump(dual_kg_dict, dual_kg_dict_file)
        dual_kg_dict_file.close()
    return dual_kg, dual_kg_e, trans_dual_kg, dual_kg_dict


def translate_kg(dual_kg_e, dual_dest):
    dual_kg_dict = []

    kg_lens = len(dual_kg_e[0]) + len(dual_kg_e[1])
    count = 0
    for idx, kg_e in enumerate(dual_kg_e):
        kg_dict = dict()
        for e in kg_e:
            translated_e = translate(e.replace('_', ' '), dest=dual_dest[idx - 1], src=dual_dest[idx])
            kg_dict[e] = translated_e
            count += 1
            print("Translating 2 kgs with progress:{:.2f}%".format(count / kg_lens * 100), end='\r')
        dual_kg_dict.append(kg_dict)
    return dual_kg_dict


def load_data(path_2hop, path_3hop, dual_dest):
    if 'fr' in dual_dest:
        mark = 'fr.'
    else:
        mark = 'zh.'
    train_eqp = []
    train_gold_er = []
    dev_eqp = []
    dev_gold_er = []
    dev_s_eqp = []
    dev_s_gold_er = []
    test_eqp = []
    test_gold_er = []
    test_s_eqp = []
    test_s_gold_er = []
    eqp = [train_eqp, dev_eqp, dev_s_eqp, test_eqp, test_s_eqp]
    er = [train_gold_er, dev_gold_er, dev_s_gold_er, test_gold_er, test_s_gold_er]

    for idx, tri_data_path in enumerate(path_2hop):
        for data_path in tri_data_path:
            for line in open(data_path,encoding='utf-8'):
                line = re.split('@@@|>', line)
                q = line[0]
                head_e = line[1].split('resource/')[-1]
                head_r = line[2].split('property/')[-1]
                head_tail = line[3].split('resource/')[-1]
                mid_e = line[4].split('resource/')[-1]
                mid_r = line[5].split('property/')[-1]
                p = head_e + " " + head_r
                # mid_tail = line[6].split('/')[-1]
                if idx == 0:
                    idx1 = idx2 = idx
                elif idx == 1:
                    idx1 = idx
                    idx2 = idx + 1
                else:
                    idx1 = 3
                    idx2 = 4
                prepare_data(eqp[idx1], er[idx1],
                             head_e, q, head_e, mark in line[1],
                             head_e, head_r)
                prepare_data(eqp[idx2], er[idx2],
                             head_tail, q, p, mark in line[3],
                             mid_e, mid_r)
                prepare_data(eqp[idx1], er[idx1],
                             mid_e, q, p, mark in line[4],
                             mid_e, mid_r)

    for idx, tri_data_path in enumerate(path_3hop):
        for data_path in tri_data_path:
            for line in open(data_path,encoding='utf-8'):
                line = re.split('@@@|>', line)
                q = line[0]
                head_e = line[1].split('resource/')[-1]
                head_r = line[2].split('property/')[-1]
                head_tail = line[3].split('resource/')[-1]
                mid_e = line[4].split('resource/')[-1]
                mid_r = line[5].split('property/')[-1]
                mid_t = line[6].split('resource/')[-1]
                tail_e = line[7].split('resource/')[-1]
                tail_r = line[8].split('property/')[-1]
                path1 = head_e + " " + head_r
                path2 = head_e + " " + head_r + " " + mid_r
                if idx == 0:
                    idx1 = idx2 = idx
                elif idx == 1:
                    idx1 = idx
                    idx2 = idx + 1
                else:
                    idx1 = 3
                    idx2 = 4
                prepare_data(eqp[idx1], er[idx1],
                             head_e, q, head_e, mark in line[1],
                             head_e, head_r)

                if head_tail != mid_e:
                    prepare_data(eqp[idx1], er[idx1],
                                 mid_e, q, path1, mark in line[4],
                                 mid_e, mid_r)
                    ib = idx2
                else:
                    ib = idx1
                prepare_data(eqp[ib], er[ib],
                             head_tail, q, path1, mark in line[3],
                             mid_e, mid_r)
                if mid_t != tail_e:
                    prepare_data(eqp[idx1], er[idx1],
                                 tail_e, q, path2, mark in line[7],
                                 tail_e, tail_r)
                    ib = idx2
                else:
                    ib = idx1
                prepare_data(eqp[ib], er[ib],
                             mid_t, q, path2, mark in line[6],
                             tail_e, tail_r)

    return eqp, er


def prepare_data(e_q_p, gold_e_r, e, q, p, b_lang, gold_e, gold_r):
    e_q_p.append([e, q, p, b_lang])
    gold_e_r.append([gold_e, gold_r])


def match(e_enc, kg_enc, kg_e):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # TODO: return matched entity
    # Using the iterative method, can be replaced by the batch method
    # kg_enc: [entity_num, bert_config.hidden_size]
    entity_enc = e_enc.tile((kg_enc.shape[0], 1))

    match_score = cos(entity_enc, kg_enc)
    match_id = torch.argmax(match_score).item()

    return kg_e[match_id]


def get_result(dual_e, dual_e_enc, dual_kg_enc, dual_kg_e):
    match_count = 0

    for ib, e in enumerate(dual_e):

        match_result = match(dual_e_enc[ib], dual_kg_enc[ib], dual_kg_e[ib])
        if match_result == dual_e[ib - 1]:
            match_count += 1

    return match_count





def get_ground_truth(batch, is_train=True):
    pos_kg_e_batch = []
    neg_kg_e_batch = []
    e_q_p_batch = []

    for example in batch:
        pos_kg_e = example["positive_samples"]
        neg_kg_e = example["negative_samples"]
        e_q_p = example["e_q_p"]

        if (len(pos_kg_e) > 0) and (len(neg_kg_e) > 0):
            if is_train:
                pos_kg_e_batch.append([pos_kg_e])
            else:
                pos_kg_e_batch.append(pos_kg_e)
            neg_kg_e_batch.append(neg_kg_e)
            e_q_p_batch.append([e_q_p])

    return pos_kg_e_batch, neg_kg_e_batch, e_q_p_batch


def get_train_data(origin_e_q_p, goldmap_e_r, trans_dual_kg, dual_kg_e, dual_kg, dual_kg_dict,
                   is_train=True, alignment=None, switch=True, dual_dest=None):
    train_data = []
    e_q_len = len(origin_e_q_p)
    count = 0
    valid_count = 0
    for ib, e_q_p in enumerate(origin_e_q_p):
        if is_train:
            print("Getting the training data with progress {:.3f}%".format(ib / e_q_len * 100), end='\r')
        else:
            if switch:
                print("{} dev/test data with progress {:.3f}%".
                      format('Switch kg', ib / e_q_len * 100), end='\r')
            else:
                print("{} dev/test data with progress {:.3f}%".
                      format('No switch kg', ib / e_q_len * 100), end='\r')
        gold_e_r = goldmap_e_r[ib]
        if e_q_p[3]:
            idx = 1
        else:
            idx = 0
        if e_q_p[0] == gold_e_r[0]:
            if is_train:
                neg = [[e, r] for e, r in dual_kg[idx - 1] if (e == e_q_p[0] and (e, r) != gold_e_r)]
                neg += [[e, r] for e, r in dual_kg[idx] if (e == e_q_p[0] and (e, r) != gold_e_r)]
                if len(neg) == 0:
                    continue
            else:
                neg = [[e, r] for e, r in dual_kg[idx - 1] if e == e_q_p[0]]
                neg += [[e, r] for e, r in dual_kg[idx] if e == e_q_p[0]]
            train_once = dict()
            train_once["e_q_p"] = (e_q_p[0] + ' ' + e_q_p[1], e_q_p[2])
            train_once["negative_samples"] = neg
            train_once["positive_samples"] = gold_e_r
            valid = True

        else:
            train_once, valid = get_one_train_data(e_q_p, trans_dual_kg[idx], dual_kg[idx], dual_kg_e[idx],
                                                   dual_kg_dict[idx - 1], gold_e_r,
                                                   is_train, alignment, idx, dual_dest)
        count += 1
        valid_count += valid
        train_data.append(train_once)
    print(f"valid: {valid_count / count:.4}\n")
    return train_data


def get_one_train_data(e_q_p, trans_kg, kg, kg_e, kg_dict, gold_e_r, is_train, alignment, idx, dual_dest):
    train_once = dict()
    origin_e = e_q_p[0]
    aligned_e = None
    if origin_e in alignment:
        aligned_e = alignment[origin_e]
    elif origin_e in alignment.inverse:
        aligned_e = alignment.inverse[origin_e]
    translated_e = kg_dict[e_q_p[0]]

    train_once["e_q_p"] = (translated_e + ' ' + e_q_p[1], e_q_p[2])
    if e_q_p[0] in kg_e:
        if is_train:
            train_once["negative_samples"], valid = \
                [(e, r) for e, r in kg if (e == e_q_p[0] and (e, r) != gold_e_r)], True
        else:
            train_once["negative_samples"], valid = \
                [(e, r) for e, r in kg if e == e_q_p[0]], True
    else:
        train_once["negative_samples"], valid = get_negative_samples(translated_e, trans_kg, kg, gold_e_r=gold_e_r,
                                                                     is_train=is_train, aligned_e=aligned_e)

    train_once["positive_samples"] = gold_e_r
    return train_once, valid


def get_negative_samples(origin_e, trans_kg, kg, gold_e_r=None, is_train=True, aligned_e=None):
    if origin_e[0] == '(':
        trans_e = origin_e.split(')')[1]
    else:
        trans_e = origin_e.split('(')[0]
    trans_e = "".join(t_e for t_e in trans_e if t_e.isalnum()).lower()
    if len(trans_e) == 0 or len(origin_e.split(')')) > 2:
        trans_e = origin_e
    valid = False
    neg_num = 15

    neg_dist = [distance(trans_e, e) /
                max(len(trans_e), len(e))
                for e, _ in trans_kg]
    neg_dist = torch.tensor(neg_dist)
    while True:
        negative_idx = torch.topk(neg_dist, neg_num, largest=False, sorted=True)
        if negative_idx.values[0] < 0.2 and negative_idx.values[0] != negative_idx.values[-1]:

            break
        elif negative_idx.values[-1] <= 0.7 and neg_num < 30:
            neg_num = 2 * neg_num
        else:
            break

    negative = [kg[idx] for idx in negative_idx.indices]
    if aligned_e:
        negative.extend([[e, r] for e, r in kg if e == aligned_e])

    if gold_e_r in negative:
        valid = True
    elif is_train:
        negative.extend([[e, r] for e, r in kg if e == gold_e_r[0]])

    if is_train:
        negative = [[e, r] for e, r in negative if (e, r) != gold_e_r]
    return negative, valid
