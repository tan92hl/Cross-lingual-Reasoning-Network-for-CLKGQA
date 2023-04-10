import pathlib
import sys
import os
import inspect

import pickle
import time
import jsonlines
from pargs import pargs
from predictor import *
from CLRN.model.model import Model

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

if __name__ == "__main__":
    args = pargs()

    device = -1
    if torch.cuda.is_available():
        device = args.gpu
        torch.cuda.set_device(args.gpu)

    else:
        print("cuda not available, please run with cuda\n")

    # TODO: 1. load the bert and clkqqa model
    # DONE
    print("Loading pretrained models ...\n")
    bert_model, bert_tokenizer, bert_config = load_fine_tuned_bert(args.fine_tuned_bert_path, device=device)
    bert = (bert_model, bert_tokenizer, bert_config)
    model = Model(d_bert=bert_config.hidden_size,
                  d_h=args.d_h,
                  n_layers=args.n_layers_lstm,
                  dropout_prob=0.0)
    model.eval()
    bert_model.eval()

    if device != -1:
        model.load_state_dict(torch.load(args.trained_clkgqa_path, map_location="cuda:{}".format(device)))
        model.cuda()
        bert_model.cuda()
    else:
        model.load_state_dict(torch.load(args.trained_clkgqa_path, map_location=torch.device('cpu')))

    with torch.no_grad():
        hop2_ranker_en = get_path_ranker(bert, args.dual_dest,
                                         ((args.hop2_question, args.hop2_equal_path),
                                          (args.hop2_question_zh, args.hop2_equal_path_zh),
                                          (args.hop2_question_fr, args.hop2_equal_path_fr)),
                                         device=device, nhop=2)

        hop3_ranker_en = get_path_ranker(bert, args.dual_dest,
                                         [(args.hop3_question, args.hop3_equal_path),
                                          (args.hop3_question_zh, args.hop3_equal_path_zh),
                                          (args.hop3_question_fr, args.hop3_equal_path_fr)
                                          ],
                                         device=device, nhop=3)

    NMN_alignment = load_alignment((args.alignment_path_test, args.alignment_path_dev))

    with torch.no_grad():
        dual_kg, dual_alignment_dict = load_dual_kg((args.kg_path1, args.kg_path2), args.dual_dest, bert, device=device)

    # TODO: 2. load the data to go through the deduction
    # TODO: 2.1 load 2 kgs
    # TODO: 2.2 load question
    two_hop_data_dev_en = load_two_hop_data(args.data_path_2h_dev_en)
    two_hop_data_dev_fr = load_two_hop_data(args.data_path_2h_dev_fr)
    two_hop_data_dev_zh = load_two_hop_data(args.data_path_2h_dev_zh)
    two_hop_data_test_en = load_two_hop_data(args.data_path_2h_test_en)
    two_hop_data_test_fr = load_two_hop_data(args.data_path_2h_test_fr)
    two_hop_data_test_zh = load_two_hop_data(args.data_path_2h_test_zh)

    three_hop_data_dev_en = load_three_hop_data(args.data_path_3h_dev_en)
    three_hop_data_dev_fr = load_three_hop_data(args.data_path_3h_dev_fr)
    three_hop_data_dev_zh = load_three_hop_data(args.data_path_3h_dev_zh)
    three_hop_data_test_en = load_three_hop_data(args.data_path_3h_test_en)
    three_hop_data_test_fr = load_three_hop_data(args.data_path_3h_test_fr)
    three_hop_data_test_zh = load_three_hop_data(args.data_path_3h_test_zh)

    """two_hop_loader_dev_zh = DataLoader(dataset=two_hop_data_dev_zh, batch_size=args.bs, shuffle=False,
                                       num_workers=1, collate_fn=lambda x: x)
    two_hop_loader_dev_en = DataLoader(dataset=two_hop_data_dev_en, batch_size=args.bs, shuffle=False,
                                       num_workers=1, collate_fn=lambda x: x)
    two_hop_loader_dev_fr = DataLoader(dataset=two_hop_data_dev_fr, batch_size=args.bs, shuffle=False,
                                       num_workers=1, collate_fn=lambda x: x)"""

    two_hop_loader_test_zh = DataLoader(dataset=two_hop_data_test_zh, batch_size=args.bs, shuffle=False,
                                        num_workers=1, collate_fn=lambda x: x)
    two_hop_loader_test_en = DataLoader(dataset=two_hop_data_test_en, batch_size=args.bs, shuffle=False,
                                        num_workers=1, collate_fn=lambda x: x)
    two_hop_loader_test_fr = DataLoader(dataset=two_hop_data_test_fr, batch_size=args.bs, shuffle=False,
                                        num_workers=1, collate_fn=lambda x: x)

    """three_hop_loader_dev_zh = DataLoader(dataset=three_hop_data_dev_zh, batch_size=args.bs, shuffle=False,
                                         num_workers=1, collate_fn=lambda x: x)
    three_hop_loader_dev_en = DataLoader(dataset=three_hop_data_dev_en, batch_size=args.bs, shuffle=False,
                                         num_workers=1, collate_fn=lambda x: x)
    three_hop_loader_dev_fr = DataLoader(dataset=three_hop_data_dev_fr, batch_size=args.bs, shuffle=False,
                                         num_workers=1, collate_fn=lambda x: x)"""

    three_hop_loader_test_zh = DataLoader(dataset=three_hop_data_test_zh, batch_size=args.bs, shuffle=False,
                                          num_workers=1, collate_fn=lambda x: x)
    three_hop_loader_test_en = DataLoader(dataset=three_hop_data_test_en, batch_size=args.bs, shuffle=False,
                                          num_workers=1, collate_fn=lambda x: x)
    three_hop_loader_test_fr = DataLoader(dataset=three_hop_data_test_fr, batch_size=args.bs, shuffle=False,
                                          num_workers=1, collate_fn=lambda x: x)

    # TODO: 3. consider 4 situations for the 2-hop deduction
    '''
    There are 4 possible predicted paths in 2 2-hop kgs:
    1. (e1)kg1->(e2)kg1 --> (e2)kg1->(e3)kg1
    2. (e1)kg1->(e2)kg1 --> (e2)kg2->(e3)kg2
    3. (e1)kg2->(e2)kg2 --> (e2)kg1->(e3)kg1
    4. (e1)kg2->(e2)kg2 --> (e2)kg2->(e3)kg2
    The path length is <=4 because a deduction might fail 
    due to the knowledge may not exist in a kg
    '''
    print("All prerequisites loaded, entering the deducting stage...")
    with torch.no_grad():


        """"two_hop_paths_dev_fr, two_hop_alignments_dev_fr = predict(two_hop_loader_dev_fr, model, bert, dual_kg,
                                                                  dual_alignment_dict, 2, hop2_ranker_en,
                                                                  pooling_type=args.pooling_type, device=device)
        save_result(two_hop_paths_dev_fr, two_hop_alignments_dev_fr, args.dual_dest, 2, "fr")"""

        """two_hop_paths_dev_zh, two_hop_alignments_dev_zh = predict(two_hop_loader_dev_zh, model, bert, dual_kg,
                                                                  dual_alignment_dict, 2, hop2_ranker_en,
                                                                  pooling_type=args.pooling_type, device=device)
        save_result(two_hop_paths_dev_zh, two_hop_alignments_dev_zh, args.dual_dest, 2, "zh")"""

        """two_hop_paths_dev_en, two_hop_alignments_dev_en = predict(two_hop_loader_dev_en, model, bert, dual_kg,
                                                                  dual_alignment_dict, 2, hop2_ranker_en,
                                                                  pooling_type=args.pooling_type, device=device)
        save_result(two_hop_paths_dev_en, two_hop_alignments_dev_en, args.dual_dest, 2, "en")"""

        """three_hop_paths_test_zh, three_hop_alignments_test_zh = predict(three_hop_loader_test_zh, model, bert, dual_kg,
                                                                        dual_alignment_dict, 3, hop3_ranker_en,
                                                                        NMN_alignment=NMN_alignment,
                                                                        pooling_type=args.pooling_type, device=device)
        save_result(three_hop_paths_test_zh, three_hop_alignments_test_zh, args.dual_dest, 3, "zh")"""

        """three_hop_paths_test_fr, three_hop_alignments_test_fr = predict(three_hop_loader_test_fr, model, bert, dual_kg,
                                                                        dual_alignment_dict, 3, hop3_ranker_en,
                                                                        NMN_alignment=NMN_alignment,
                                                                        pooling_type=args.pooling_type, device=device)
        save_result(three_hop_paths_test_fr, three_hop_alignments_test_fr, args.dual_dest, 3, "fr")

        three_hop_paths_test_en, three_hop_alignments_test_en = predict(three_hop_loader_test_en, model, bert, dual_kg,
                                                                        dual_alignment_dict, 3, hop3_ranker_en,
                                                                        NMN_alignment=NMN_alignment,
                                                                        pooling_type=args.pooling_type, device=device)
        save_result(three_hop_paths_test_en, three_hop_alignments_test_en, args.dual_dest, 3, "en")"""

        two_hop_paths_test_zh, two_hop_alignments_test_zh = predict(two_hop_loader_test_zh, model, bert, dual_kg,
                                                                    dual_alignment_dict, 2, hop2_ranker_en,
                                                                    NMN_alignment=NMN_alignment,
                                                                    pooling_type=args.pooling_type, device=device)
        save_result(two_hop_paths_test_zh, two_hop_alignments_test_zh, args.dual_dest, 2, "zh")

        two_hop_paths_test_fr, two_hop_alignments_test_fr = predict(two_hop_loader_test_fr, model, bert, dual_kg,
                                                                    dual_alignment_dict, 2, hop2_ranker_en,
                                                                    NMN_alignment=NMN_alignment,
                                                                    pooling_type=args.pooling_type, device=device)
        save_result(two_hop_paths_test_fr, two_hop_alignments_test_fr, args.dual_dest, 2, "fr")

        two_hop_paths_test_en, two_hop_alignments_test_en = predict(two_hop_loader_test_en, model, bert, dual_kg,
                                                                    dual_alignment_dict, 2, hop2_ranker_en,
                                                                    NMN_alignment=NMN_alignment,
                                                                    pooling_type=args.pooling_type, device=device)
        save_result(two_hop_paths_test_en, two_hop_alignments_test_en, args.dual_dest, 2, "en")

        """three_hop_paths_dev_zh, three_hop_alignments_dev_zh = predict(three_hop_loader_dev_zh, model, bert, dual_kg,
                                                                      dual_alignment_dict, 3, hop3_ranker_en,
                                                                      pooling_type=args.pooling_type, device=device)
        save_result(three_hop_paths_dev_zh, three_hop_alignments_dev_zh, args.dual_dest, 3, "zh")"""

        """three_hop_paths_dev_fr, three_hop_alignments_dev_fr = predict(three_hop_loader_dev_fr, model, bert, dual_kg,
                                                                      dual_alignment_dict, 3, hop3_ranker_en,
                                                                      NMN_alignment=NMN_alignment,
                                                                      pooling_type=args.pooling_type, device=device)
        save_result(three_hop_paths_dev_fr, three_hop_alignments_dev_fr, args.dual_dest, 3, "fr")"""

        """three_hop_paths_dev_en, three_hop_alignments_dev_en = predict(three_hop_loader_dev_en, model, bert, dual_kg,
                                                                      dual_alignment_dict, 3, hop3_ranker_en,
                                                                      NMN_alignment=NMN_alignment,
                                                                      pooling_type=args.pooling_type, device=device)
        save_result(three_hop_paths_dev_en, three_hop_alignments_dev_en, args.dual_dest, 3, "en")"""


