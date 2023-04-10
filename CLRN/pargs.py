import argparse
import os.path
import sys
import inspect


def pargs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dual_dest', type=tuple,
                        default=("fr", "en"),
                        choices=[("fr", "en"), ("zh", "en"), ("fr", "zh")],
                        help='Input language types')

    parser.add_argument('--data_path_2hop_train0', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/fr_question/train_en_fr_2h_fr_question.txt"),
                        help='Input the 2-hop data path')

    parser.add_argument('--data_path_2hop_train1', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/en_question/train_en_fr_2h_en_question.txt"),
                        help='Input the 2-hop data path')

    parser.add_argument('--data_path_2hop_train2', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/zh_question/train_en_fr_2h_zh_question.txt"),
                        help='Input the 2-hop data path')

    parser.add_argument('--data_path_2hop_dev0', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/fr_question/dev_en_fr_2h_fr_question.txt"),
                        help='Input the 2-hop data path')

    parser.add_argument('--data_path_2hop_dev1', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/en_question/dev_en_fr_2h_en_question.txt"),
                        help='Input the 2-hop data path')

    parser.add_argument('--data_path_2hop_dev2', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/zh_question/dev_en_fr_2h_zh_question.txt"),
                        help='Input the 2-hop data path')

    parser.add_argument('--data_path_2hop_test0', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/fr_question/test_en_fr_2h_fr_question.txt"),
                        help='Input the 2-hop data path')

    parser.add_argument('--data_path_2hop_test1', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/en_question/test_en_fr_2h_en_question.txt"),
                        help='Input the 2-hop data path')

    parser.add_argument('--data_path_2hop_test2', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/zh_question/test_en_fr_2h_zh_question.txt"),
                        help='Input the 2-hop data path')

    parser.add_argument('--data_path_3hop_train0', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/fr_question/train_en_fr_3h_fr_question.txt"),
                        help='Input the 3-hop data path')

    parser.add_argument('--data_path_3hop_train1', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/en_question/train_en_fr_3h_en_question.txt"),
                        help='Input the 3-hop data path')

    parser.add_argument('--data_path_3hop_train2', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/zh_question/train_en_fr_3h_zh_question.txt"),
                        help='Input the 3-hop data path')

    parser.add_argument('--data_path_3hop_dev0', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/fr_question/dev_en_fr_3h_fr_question.txt"),
                        help='Input the 3-hop data path')

    parser.add_argument('--data_path_3hop_dev1', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/en_question/dev_en_fr_3h_en_question.txt"),
                        help='Input the 3-hop data path')

    parser.add_argument('--data_path_3hop_dev2', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/zh_question/dev_en_fr_3h_zh_question.txt"),
                        help='Input the 3-hop data path')

    parser.add_argument('--data_path_3hop_test0', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/fr_question/test_en_fr_3h_fr_question.txt"),
                        help='Input the 3-hop data path')

    parser.add_argument('--data_path_3hop_test1', type=str,
                        default=os.path.abspath(
                            "../data/MLPQ_train_dev_test/en_fr/en_question/test_en_fr_3h_en_question.txt"),
                        help='Input the 3-hop data path')

    parser.add_argument('--data_path_3hop_test2', type=str,
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
                        default=os.path.abspath("../data/MLPQ/KGs/sampled_monolingual_KGs/Sampled_fr.txt"),
                        help='Input the 2-hop data path1')
    parser.add_argument('--kg_path2', type=str,
                        default=os.path.abspath("../data/MLPQ/KGs/sampled_monolingual_KGs/Sampled_en.txt"),
                        help='Input the 2-hop data path2')

    parser.add_argument("--gpu", type=int, default=0, help="the index of the used GPU")


    # Model Hyper-Parameters
    parser.add_argument("--dropout_prob", type=float, default=0.3, help="probability of the dropout layer")

    parser.add_argument("--n_layers_lstm", type=int, default=1, help="the number of the layers of the LSTMs")
    parser.add_argument("--d_h", type=int, default=100, help="the hidden dimension of the LSTMs")
    parser.add_argument("--bs", type=int, default=8, help="batch size")
    parser.add_argument("--bert_lr", type=float, default=9e-7, help="learning rate for training BERT")
    parser.add_argument("--pooling_type", type=str, choices=["max", "last", "avg"], default="avg", help="the type of all pooling operation, "
                                                                                                         "including max-pooling, last-pooling, "
                                                                                                         "and average-pooling")

    parser.add_argument("--n_epochs", type=int, default=15, help="the number of total epochs")
    parser.add_argument("--ag", type=int, default=1, help="accumulate gradients for training")
    parser.add_argument("--lr", type=float, default=7e-4, help="learning rate for training the Seq2SQL model")

    return parser.parse_args()
