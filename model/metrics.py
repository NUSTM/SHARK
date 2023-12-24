from fastNLP import MetricBase
from fastNLP.core.metrics import _compute_f_pre_rec
import numpy as np


class Seq2SeqSpanMetric(MetricBase):
    def __init__(self, add_ERC, eos_token_id, num_labels):
        super(Seq2SeqSpanMetric, self).__init__()
        self.add_ERC = add_ERC
        self.eos_token_id = eos_token_id
        self.num_labels_ori = num_labels # 7
        self.num_labels = num_labels - 1 # 6
        self.word_start_index = self.num_labels + 2  # shift for sos and eos

        self.triple_fp = 0
        self.triple_tp = 0
        self.triple_fn = 0
        self.em = 0
        self.invalid = 0
        self.total = 0

        self.eval_class = True
        self.triple_fp_class = [0] * (self.num_labels+1)
        self.triple_tp_class = [0] * (self.num_labels+1)
        self.triple_fn_class = [0] * (self.num_labels+1)
        self.output_label_list = [] 
        self.num_ins_class = [0] * self.num_labels
        
        self.confusion_mat = np.zeros([self.num_labels_ori, self.num_labels_ori])


    # The input is a batch of data.
    def evaluate(self, target_span, pred, tgt_tokens, tgt_emotions, dia_utt_num):
        if self.add_ERC:
            self.evaluate_emo(pred['result_emo'], tgt_emotions, dia_utt_num)
        
        self.evaluate_ectec(target_span, pred['result_ectec'], tgt_tokens)
    

    def evaluate_emo(self, pred, tgt_tokens, dia_utt_num):
        for i, (ts, ps) in enumerate(zip(tgt_tokens.tolist(), pred.tolist())):
            for j in range(dia_utt_num[i]):
                if j < len(ps):
                    self.confusion_mat[ts[j]][ps[j]] += 1
                else:
                    print('len_pred error: ', dia_utt_num[i], len(ps))


    def evaluate_ectec(self, target_span, pred, tgt_tokens):
        self.total += pred.size(0)
        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # delete </s>
        tgt_tokens = tgt_tokens[:, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1)
        target_seq_len = (target_seq_len - 2).tolist()
        pred_spans = []
        for i, (ts, ps) in enumerate(zip(target_span, pred.tolist())):
            em = 0
            ps = ps[:pred_seq_len[i]]
            if pred_seq_len[i] == target_seq_len[i]:
                em = int(
                    tgt_tokens[i, :target_seq_len[i]].eq(pred[i, :target_seq_len[i]]).sum().item() == target_seq_len[i])
            self.em += em
            invalid = 0
            pairs = []
            cur_pair = []
            if len(ps):
                for index, j in enumerate(ps):
                    if j < self.word_start_index:
                        cur_pair.append(j)
                        if len(cur_pair) != 3:
                            invalid = 1
                        else:
                            pairs.append(tuple(cur_pair))
                        cur_pair = []
                    else:
                        cur_pair.append(j)
            pred_spans.append(pairs.copy())
            self.invalid += invalid

            ts = set([tuple(t) for t in ts])
            ps = set(pairs)
            for p in list(ts):
                self.num_ins_class[int(p[-1])-2] += 1
            for p in list(ps): 
                if p in ts:
                    ts.remove(p)
                    self.triple_tp += 1
                    self.triple_tp_class[int(p[-1])-2] += 1
                    self.triple_tp_class[-1] += 1
                    if p[-1] not in self.output_label_list:
                        self.output_label_list.append(p[-1])
                else:
                    self.triple_fp += 1
                    self.triple_fp_class[int(p[-1])-2] += 1
                    self.triple_fp_class[-1] += 1
                    if p[-1] not in self.output_label_list:
                        self.output_label_list.append(p[-1])
            self.triple_fn += len(ts)


            self.triple_fn_class[-1] += len(ts)
            for p in list(ts):
                self.triple_fn_class[int(p[-1])-2] += 1
                if p[-1] not in self.output_label_list:
                    self.output_label_list.append(p[-1])


    # Call get_metric() to obtain the final evaluation results after all batch data is processed.
    def get_metric(self, reset=True):
        res = {}
        f, pre, rec = _compute_f_pre_rec(1, self.triple_tp, self.triple_fn, self.triple_fp)
        res['triple_f'] = round(f, 4)*1
        res['triple_rec'] = round(rec, 4)*1
        res['triple_pre'] = round(pre, 4)*1

        if self.eval_class:
            res['triple_f_all'], res['triple_rec_all'], res['triple_pre_all'] = [], [], []
            # 6 emotion categories
            weight = np.array(self.num_ins_class) / sum(self.num_ins_class)
            # 4 main emotion categories except Disgust and Fear
            idx = [0,3,4,5] # ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
            weight4 = np.array(self.num_ins_class)[idx] / sum(np.array(self.num_ins_class)[idx])
            weight_f,  weight_pre, weight_rec = 0, 0, 0
            weight_f_avg4,  weight_pre_avg4, weight_rec_avg4 = 0, 0, 0
            for ii in range(len(self.triple_tp_class)):
                f, pre, rec = _compute_f_pre_rec(1, self.triple_tp_class[ii], self.triple_fn_class[ii], self.triple_fp_class[ii])
                res['triple_f_all'].append(round(f, 4)*1)
                res['triple_rec_all'].append(round(rec, 4)*1)
                res['triple_pre_all'].append(round(pre, 4)*1)
                if ii < self.num_labels:
                    weight_f += weight[ii] * f 
                    weight_pre += weight[ii] * pre 
                    weight_rec += weight[ii] * rec
                    if ii in idx:
                        weight_f_avg4 += weight4[idx.index(ii)] * f 
                        weight_pre_avg4 += weight4[idx.index(ii)] * pre
                        weight_rec_avg4 += weight4[idx.index(ii)] * rec

            res['ECTEC'] = res['triple_f_all'][:6] + [round(weight_pre, 4)*1, round(weight_rec, 4)*1, round(weight_f, 4)*1, round(weight_pre_avg4, 4)*1, round(weight_rec_avg4, 4)*1, round(weight_f_avg4, 4)*1]
            res['triple_f_w-avg-6'] = round(weight_f, 4)*1
            res['triple_f_w-avg-4'] = round(weight_f_avg4, 4)*1

            if self.add_ERC:
                p = np.diagonal(self.confusion_mat / np.reshape(np.sum(self.confusion_mat, axis = 0) + 1e-8, [1,7]) )
                r = np.diagonal(self.confusion_mat / np.reshape(np.sum(self.confusion_mat, axis = 1) + 1e-8, [7,1]) )
                f = 2*p*r/(p+r+1e-8)
                weight = np.sum(self.confusion_mat, axis = 1) / np.sum(self.confusion_mat)
                w_avg_f = np.sum(f * weight)
                res['emocate_f'] = np.around(np.append(f, w_avg_f), decimals=4)

        res['em'] = round(self.em / self.total, 4)
        res['invalid'] = round(self.invalid / self.total, 4)

        if reset:
            self.triple_fp = 0
            self.triple_tp = 0
            self.triple_fn = 0
            self.em = 0
            self.invalid = 0
            self.total = 0

            self.eval_class = True
            self.triple_fp_class = [0] * (self.num_labels+1)
            self.triple_tp_class = [0] * (self.num_labels+1)
            self.triple_fn_class = [0] * (self.num_labels+1)
            self.output_label_list = [] 
            self.num_ins_class = [0] * self.num_labels

            self.confusion_mat = np.zeros([self.num_labels_ori, self.num_labels_ori])

        return res
