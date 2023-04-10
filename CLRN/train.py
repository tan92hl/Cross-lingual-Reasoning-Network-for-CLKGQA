# -*- coding: utf-8 -*-
import random
import time
import glob
import sys
import jsonlines

from pargs import pargs

sys.path.append("..")

from training_utils import *

from model.model import Model
from utils import *

if __name__ == "__main__":
    args = pargs()
    device = -1
    if torch.cuda.is_available():
        device = args.gpu
        torch.cuda.set_device(args.gpu)
    else:
        print("cuda not available, please run with cuda\n")

    # Load model and data
    bert_model, bert_tokenizer, bert_config = load_bert()
    bert = (bert_model, bert_tokenizer, bert_config)
    print('bert_config.hidden_size:',bert_config.hidden_size)
    model = Model(d_bert=bert_config.hidden_size, d_h=args.d_h,
                  n_layers=args.n_layers_lstm, dropout_prob=args.dropout_prob)

    if torch.cuda.is_available():
#        model.cuda()
#        bert_model.cuda()
        model.cuda().half()
        bert_model.cuda().half()

    # one_time_fix_dict((args.kg_path1, args.kg_path2), args.dual_dest)

    # TODO: load and bert encode kg
    print("loading 2 kgs...\n")
    dual_kg, dual_kg_e, trans_dual_kg, dual_kg_dict = load_dual_kg((args.kg_path1, args.kg_path2), args.dual_dest)


    # TODO: load data
    print("Loading {} Data...\n".format(str(args.dual_dest)))
    origin_e_q_p_data, goldmap_e_r_data = \
        load_data(((args.data_path_2hop_train0, args.data_path_2hop_train1, args.data_path_2hop_train2),
                   (args.data_path_2hop_dev0, args.data_path_2hop_dev1, args.data_path_2hop_dev2),
                   (args.data_path_2hop_test0, args.data_path_2hop_test1, args.data_path_2hop_test2)),
                  ((args.data_path_3hop_train0, args.data_path_3hop_train1, args.data_path_3hop_train2),
                   (args.data_path_3hop_dev0, args.data_path_3hop_dev1, args.data_path_3hop_dev2),
                   (args.data_path_3hop_test0, args.data_path_3hop_test1, args.data_path_3hop_test2)),
                  args.dual_dest
                  )


    # TODO: load the alignment info
    alignment = load_alignment((args.alignment_path_test, args.alignment_path_dev))
    train_data, dev_data, test_data = get_data(origin_e_q_p_data, goldmap_e_r_data, trans_dual_kg,
                                               dual_kg_e, dual_kg, dual_kg_dict, args.dual_dest,
                                               alignment)
    random.shuffle(train_data)

    train_loader = DataLoader(dataset=train_data, batch_size=args.bs,
                              shuffle=False, num_workers=0, collate_fn=lambda x: x)
    dev_no_switch_loader = DataLoader(dataset=dev_data[0], batch_size=args.bs,
                                      shuffle=False, num_workers=0, collate_fn=lambda x: x)
    dev_switch_loader = DataLoader(dataset=dev_data[1], batch_size=args.bs,
                                   shuffle=False, num_workers=0, collate_fn=lambda x: x)
    test_no_switch_loader = DataLoader(dataset=test_data[0], batch_size=args.bs,
                                       shuffle=False, num_workers=0, collate_fn=lambda x: x)
    test_switch_loader = DataLoader(dataset=test_data[1], batch_size=args.bs,
                                    shuffle=False, num_workers=0, collate_fn=lambda x: x)

    opt, bert_opt = load_optimizer(model, args.lr, bert_model, args.bert_lr)

    # Build saved checkpoint directory
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "saved_model", timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print("\nSaving model to \"{}\".".format(out_dir))
    with open(os.path.join(out_dir, "param.log"), "w", encoding="utf-8") as f:
        for x, y in vars(args).items():
            f.write("{}: {}\n".format(x, y))
    checkpoint_dir = os.path.join(out_dir, "checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    header = '\n  Time   Epoch       Loss        No_switch_acc   Switch_Acc'
    log = " ".join("{:>6.0f},{:>5.0f},{:>12.6f},{:>12.4f},{:>14.4f}".split(","))
    best_model_prefix = os.path.join(checkpoint_dir, "best_model")
    best_bert_model_prefix = os.path.join(checkpoint_dir, "best_bert_model")
    best_acc = 0
    start = time.time()
    for i_epoch in range(1, args.n_epochs + 1):
        print(f"Start to train the {i_epoch} epoch...\n")
        """loss = train_one_epoch(train_loader,
                               model, opt,
                               bert_model, bert_config, bert_tokenizer, bert_opt,
                               accumulate_gradients=args.ag,
                               pooling_type=args.pooling_type,
                               device=device)"""
        loss = train_one_epoch(train_loader,
                               model, opt,
                               bert_model, bert_config, bert_tokenizer, bert_opt,
                               accumulate_gradients=args.ag,
                               pooling_type=args.pooling_type,
                               device=device)
        # TODO: model need to be evaluated
        # Evaluation on development data
        with torch.no_grad():
            print("entering the post kg processing stage of epoch:\n", i_epoch)
            """acc1, result = test(dev_no_switch_loader, model,
                                bert_model, bert_config, bert_tokenizer,
                                pooling_type=args.pooling_type,
                                device=device)"""
            acc2, result_s = test(dev_switch_loader, model,
                                  bert_model, bert_config, bert_tokenizer,
                                  pooling_type=args.pooling_type,
                                  device=device)
        print(header)
        print(log.format(time.time() - start, i_epoch, loss, acc1, acc2))
        with jsonlines.open("./results.json", mode="w") as fout:
            for res in result:
                fout.write(res)
        with jsonlines.open("./results_switch.json", mode="w") as fout:
            for res in result_s:
                fout.write(res)
        # Update the best checkpoint.
        if i_epoch > 3 and best_acc <= (acc1 + acc2) / 2:
            best_acc = (acc1 + acc2) / 2

            snapshot_path = best_model_prefix + "_epoch_{}_best_dev_acc_{:.3f}_.pt".format(i_epoch, best_acc)
            snapshot_path_bert = best_bert_model_prefix + "_epoch_{}_best_dev_acc_{:.3f}_.pt".format(i_epoch, best_acc)
            torch.save(model.state_dict(), snapshot_path)
            torch.save(bert_model.state_dict(), snapshot_path_bert)
            # Remove previous checkpoints
            for f in glob.glob(best_model_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)
            for f in glob.glob(best_bert_model_prefix + '*'):
                if f != snapshot_path_bert:
                    os.remove(f)

    print("Training finished.")
    print("Best Acc: {:.2f}\nModel writing to \"{}\".".format(best_acc, out_dir))
    with torch.no_grad():
        print("Final testing\n")
        acc1, _ = test(test_no_switch_loader, model,
                       bert_model, bert_config, bert_tokenizer,
                       pooling_type=args.pooling_type,
                       device=device)
        acc2, _ = test(test_switch_loader, model,
                       bert_model, bert_config, bert_tokenizer,
                       pooling_type=args.pooling_type,
                       device=device)
    print("The final no switch acc is: ", acc1, "The final switch acc is: ", acc2)
