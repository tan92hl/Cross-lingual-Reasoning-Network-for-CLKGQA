import os.path
import sys
import argparse

sys.path.append("..")


def pargs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dual_dest', type=tuple,
                        default=("fr", "en"),
                        choices=[("zh", "en"), ("fr", "zh"), ("fr", "en")],
                        help='Input language types')

    parser.add_argument('--fine_tuned_bert_path', type=str,
                        default=os.path.abspath(
                            "../trained_model/fr_en/best_bert_model_epoch_4_best_dev_acc_0.656_.pt"),
                        help='Input the fine tuned bert path')

    parser.add_argument('--trained_clkgqa_path', type=str,
                        default=os.path.abspath(
                            "../trained_model/fr_en/best_model_epoch_4_best_dev_acc_0.656_.pt"),
                        help='Input the trained clkgqa model path')

    parser.add_argument('--hop2_question', type=str,
                        default=os.path.abspath(
                            "../data/selected_useful_trainingset/useful_train_en_fr_2h_en_question.txt"),
                        help='Input the 2-hop data path')
    parser.add_argument('--hop2_equal_path', type=str,
                        default=os.path.abspath(
                            "../equal_path/fr_en/2hop_equivalent_path('fr', 'en')"),
                        help='Input the 2-hop data path')

    parser.add_argument('--hop3_question', type=str,
                        default=os.path.abspath(
                            "../data/selected_useful_trainingset/useful_train_en_fr_3h_en_question.txt"),
                        help='Input the 3-hop data path')
    parser.add_argument('--hop3_equal_path', type=str,
                        default=os.path.abspath(
                            "../equal_path/fr_en/3hop_equivalent_path('fr', 'en')"),
                        help='Input the 3-hop data path')

    parser.add_argument('--hop2_question_zh', type=str,
                        default=os.path.abspath(
                            "../data/rank_trainer/fr_en/selected_2hop_questions('fr', 'en')zh"),
                        help='Input the 2-hop data path')
    parser.add_argument('--hop2_equal_path_zh', type=str,
                        default=os.path.abspath(
                            "../data/rank_trainer/fr_en/2hop_equivalent_path('fr', 'en')zh"),
                        help='Input the 2-hop data path')

    parser.add_argument('--hop3_question_zh', type=str,
                        default=os.path.abspath(
                            "../data/rank_trainer/fr_en/selected_3hop_questions('fr', 'en')zh"),
                        help='Input the 3-hop data path')
    parser.add_argument('--hop3_equal_path_zh', type=str,
                        default=os.path.abspath(
                            "../data/rank_trainer/fr_en/3hop_equivalent_path('fr', 'en')zh"),
                        help='Input the 3-hop data path')

    parser.add_argument('--hop2_question_fr', type=str,
                        default=os.path.abspath(
                            "../data/rank_trainer/fr_en/selected_2hop_questions('fr', 'en')fr"),
                        help='Input the 2-hop data path')
    parser.add_argument('--hop2_equal_path_fr', type=str,
                        default=os.path.abspath(
                            "../data/rank_trainer/fr_en/2hop_equivalent_path('fr', 'en')fr"),
                        help='Input the 2-hop data path')

    parser.add_argument('--hop3_question_fr', type=str,
                        default=os.path.abspath(
                            "../data/rank_trainer/fr_en/selected_3hop_questions('fr', 'en')fr"),
                        help='Input the 3-hop data path')
    parser.add_argument('--hop3_equal_path_fr', type=str,
                        default=os.path.abspath(
                            "../data/rank_trainer/fr_en/3hop_equivalent_path('fr', 'en')fr"),
                        help='Input the 3-hop data path')



    parser.add_argument('--data_path_2h_dev_en', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/en_question/dev_en_fr_2h_en_question.txt"),
                        help='Input the 2-hop data path')

    parser.add_argument('--data_path_2h_test_en', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/en_question/test_en_fr_2h_en_question.txt"),
                        help='Input the 2-hop data path')

    parser.add_argument('--data_path_2h_dev_fr', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/fr_question/dev_en_fr_2h_fr_question.txt"),
                        help='Input the 2-hop data path')

    parser.add_argument('--data_path_2h_test_fr', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/fr_question/test_en_fr_2h_fr_question.txt"),
                        help='Input the 2-hop data path')

    parser.add_argument('--data_path_2h_dev_zh', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/zh_question/dev_en_fr_2h_zh_question.txt"),
                        help='Input the 2-hop data path')

    parser.add_argument('--data_path_2h_test_zh', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/zh_question/test_en_fr_2h_zh_question.txt"),
                        help='Input the 2-hop data path')

    parser.add_argument('--data_path_3h_dev_en', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/en_question/dev_en_fr_3h_en_question.txt"),
                        help='Input the 3-hop data path')

    parser.add_argument('--data_path_3h_test_en', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/en_question/test_en_fr_3h_en_question.txt"),
                        help='Input the 3-hop data path')

    parser.add_argument('--data_path_3h_dev_fr', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/fr_question/dev_en_fr_3h_fr_question.txt"),
                        help='Input the 3-hop data path')

    parser.add_argument('--data_path_3h_test_fr', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/fr_question/test_en_fr_3h_fr_question.txt"),
                        help='Input the 3-hop data path')

    parser.add_argument('--data_path_3h_dev_zh', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/zh_question/dev_en_fr_3h_zh_question.txt"),
                        help='Input the 3-hop data path')

    parser.add_argument('--data_path_3h_test_zh', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/zh_question/test_en_fr_3h_zh_question.txt"),
                        help='Input the 3-hop data path')

    parser.add_argument('--alignment_path_dev', type=str,
                        default=os.path.abspath("../data/NMN_alignment_00/NMN_alignment_list_dev_result_ids_en_fr"),
                        help='NMN alignment for dev')

    parser.add_argument('--alignment_path_test', type=str,
                        default=os.path.abspath("../data/NMN_alignment_00/NMN_alignment_list_test_result_ids_en_fr"),
                        help='NMN alignment for test')

    parser.add_argument('--kg_path1', type=str,
                        default=os.path.abspath("../data/MLPQ/KGs/sampled_monolingual_KGs/Sampled_en.txt"),
                        help='Input the 2-hop data path1')
    parser.add_argument('--kg_path2', type=str,
                        default=os.path.abspath("../data/MLPQ/KGs/sampled_monolingual_KGs/Sampled_fr.txt"),
                        help='Input the 2-hop data path2')

    parser.add_argument('--debug', default=False, action="store_true", dest='debug',
                        help='The length of the data to test the deduction')

    parser.add_argument("--bs", type=int, default=128, help="batch size")
    parser.add_argument("--d_h", type=int, default=100, help="the hidden dimension of the LSTMs")
    parser.add_argument("--n_layers_lstm", type=int, default=1, help="the number of the layers of the LSTMs")
    parser.add_argument("--pooling_type", type=str, choices=["max", "last", "avg"], default="avg",
                        help="the type of all pooling operation, "
                             "including max-pooling, last-pooling, "
                             "and average-pooling")

    parser.add_argument("--gpu", type=int, default=3, help="the index of the used GPU")

    return parser.parse_args()
