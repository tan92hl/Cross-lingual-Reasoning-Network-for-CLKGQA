import pdb

from model.bert_utils import get_bert_output
from utils import get_ground_truth
from model.nn_utils import cal_loss
from model.nn_utils import predict


def train_one_epoch(train_loader, model, opt,
                    bert_model, bert_config, bert_tokenizer, bert_opt,
                    accumulate_gradients=1, pooling_type="avg", device=-1):
    model.train()
    bert_model.train()

    avg_loss = 0
    count = 0
    train_len = len(train_loader)
    for i, batch in enumerate(train_loader):
        print("This epoch has been completed by {:.2f}%".format(i / train_len * 100), end="\r")
        pos_e_r_batch, neg_e_r_batch, e_q_p_batch = get_ground_truth(batch, is_train=True)
        count += len(batch)

        e_q_enc, e_q_lens, p_enc, p_lens = get_bert_output(bert_model, bert_config, bert_tokenizer,
                                                           e_q_p_batch, is_e_r=False, device=device)

        pos_e_r_enc, pos_e_r_lens, pos_e_r_nums = get_bert_output(bert_model, bert_config, bert_tokenizer,
                                                                  pos_e_r_batch, is_e_r=True, device=device)

        neg_e_r_enc, neg_e_r_lens, neg_e_r_nums = get_bert_output(bert_model, bert_config, bert_tokenizer,
                                                                  neg_e_r_batch, is_e_r=True, device=device)

        pos_score = model(e_q_enc, e_q_lens, p_enc, p_lens,
                          pos_e_r_enc, pos_e_r_lens, pos_e_r_nums,
                          pooling_type=pooling_type)

        neg_score = model(e_q_enc, e_q_lens, p_enc, p_lens,
                          neg_e_r_enc, neg_e_r_lens, neg_e_r_nums,
                          pooling_type=pooling_type)

        loss = cal_loss(pos_score, neg_score, neg_e_r_nums)

        # Caculate gradients and update parameters
        if i % accumulate_gradients == 0:
            opt.zero_grad()
            bert_opt.zero_grad()
            loss.backward()

            if accumulate_gradients == 1:
                opt.step()
                bert_opt.step()
        elif i % accumulate_gradients == (accumulate_gradients - 1):
            loss.backward()
            opt.step()
            bert_opt.step()
        else:
            loss.backward()
        avg_loss += loss.item()

    avg_loss /= count
    return avg_loss


def test(dev_loader, model,
         bert_model, bert_config, bert_tokenizer,
         pooling_type="last", device=-1):
    model.eval()
    bert_model.eval()

    count = 0
    correct = 0
    result = []
    for i, batch in enumerate(dev_loader):

        gold_e_r_batch, candidate_e_r_batch, e_p_q_batch = get_ground_truth(batch, is_train=False)
        if not candidate_e_r_batch:
            continue
        e_q_enc, e_q_lens, p_enc, p_lens = get_bert_output(bert_model, bert_config, bert_tokenizer,
                                                           e_p_q_batch,
                                                           is_e_r=False, device=device)

        candidate_e_r_enc, candidate_e_r_lens, candidate_e_r_nums = get_bert_output(bert_model, bert_config,
                                                                                    bert_tokenizer,
                                                                                    candidate_e_r_batch,
                                                                                    is_e_r=True, device=device)

        scores = model(e_q_enc, e_q_lens, p_enc, p_lens,
                       candidate_e_r_enc, candidate_e_r_lens, candidate_e_r_nums,
                       pooling_type=pooling_type)

        preds = predict(scores, candidate_e_r_batch, candidate_e_r_nums)
        pdb.set_trace()
        for j, pred_e_r in enumerate(preds):
            gold_e_r = gold_e_r_batch[j]
            result.append((gold_e_r, pred_e_r))
            if (gold_e_r[0] == pred_e_r[0]) and (gold_e_r[1] == pred_e_r[1]):
                correct += 1

        count += len(preds)
    acc = correct / count * 1.0
    return acc, result
