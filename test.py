import torch, sys, os, datetime, warnings, fitlog, argparse, json, random
sys.path.append('../')
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

warnings.filterwarnings('ignore')
from data.pipe import BartECTECPipe
from fastNLP import MetricBase, Tester
from model.metrics import Seq2SeqSpanMetric
from fastNLP import cache_results
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='ECF', type=str)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--test_batch_size', default=16, type=int)
parser.add_argument('--bart_name', default='facebook/bart-base', type=str)
parser.add_argument('--max_len', default=50, type=int)
parser.add_argument('--metric_key', default='triple_f_w-avg-6', type=str)
parser.add_argument('--seed', type=int, default=2023, help='seed')
parser.add_argument('--save_path', default='./output/', type=str)
parser.add_argument('--add_ERC', type=int, default=1)
parser.add_argument('--model_file', default='', type=str)

args= parser.parse_args()
print(args)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


def run():
    
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fitlog.set_log_dir(log_dir, new_log=True)

    print_time()
    
    if args.seed:
        print('\nSet seed: {}\n'.format(args.seed))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

    fitlog.add_hyper(args)


    cache_fn = f"caches/data_{args.bart_name}_{args.dataset_name}.pt"

    @cache_results(cache_fn, _refresh=False)
    def get_data():
        pipe = BartECTECPipe(tokenizer=args.bart_name)
        data_bundle = pipe.process_from_file(f'./data/{args.dataset_name}')
        return data_bundle, pipe.tokenizer, pipe.mapping2id

    data_bundle, tokenizer, mapping2id = get_data()
    print(mapping2id)

    eos_token_id = 1
    label_ids = list(mapping2id.values())

    if args.model_file:
        print('Load {} .'.format(args.model_file))
        model = torch.load(args.model_file)
    else:
        print("Please provide the model file!")
    # print(model)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model.to(device)

    metric = Seq2SeqSpanMetric(args.add_ERC, eos_token_id, num_labels=len(label_ids))


    test = data_bundle.get_dataset('test')
    test.set_ignore_type('raw_words', 'emo_utts', 'cau_utts')
    test.set_target('raw_words', 'emo_utts', 'cau_utts')

    class WriteResultToFileMetric(MetricBase):
        def __init__(self, add_ERC, fp, eos_token_id, num_labels=len(label_ids)):
            super(WriteResultToFileMetric, self).__init__()
            self.add_ERC = add_ERC
            self.fp = fp
            self.eos_token_id = eos_token_id
            self.num_labels = num_labels - 1
            self.word_start_index = self.num_labels + 2  # shift for sos and eos

            self.raw_words = []
            self.emotions = []
            self.emotions_pred = []
            self.tgt_triplets = []
            self.pred_triplets = []


        def evaluate(self, target_span, raw_words, emo_utts, cau_utts, pred, tgt_emotions, dia_utt_num):
            if self.add_ERC:
                pred_emo = pred['result_emo']
            else:
                pred_emo = torch.tensor([])
            pred = pred['result_ectec']
            
            pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

            pred = pred[:, 1:]  # delete </s>
            pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
            pred_seq_len = (pred_seq_len - 2).tolist()
            
            if self.add_ERC:
                for i, (_cau_utts, _raw_words, ts, ps, te, pe) in enumerate(zip(cau_utts, raw_words, target_span, pred.tolist(), tgt_emotions.tolist(), pred_emo.tolist())):
                    self.emotions.append(te[:dia_utt_num[i]])
                    self.emotions_pred.append(pe[:dia_utt_num[i]])

                    ps = ps[:pred_seq_len[i]]
                    pairs = []
                    cur_pair = []  # each pair with the format (e_start, c_start, e_class)
                    if len(ps):
                        for index, j in enumerate(ps):
                            if j < self.word_start_index:
                                cur_pair.append(j)
                                if len(cur_pair) != 3:
                                    pass
                                else:
                                    pairs.append(tuple(cur_pair))
                                cur_pair = []
                            else:
                                cur_pair.append(j)
                    self.pred_triplets.append(pairs)
                    
                self.raw_words.extend(raw_words.tolist())
                self.tgt_triplets.extend(target_span.tolist())
            
            else:
                for i, (_cau_utts, _raw_words, ts, ps, te) in enumerate(zip(cau_utts, raw_words, target_span, pred.tolist(), tgt_emotions.tolist())):
                    self.emotions.append(te[:dia_utt_num[i]])

                    ps = ps[:pred_seq_len[i]]
                    pairs = []
                    cur_pair = []  # each pair with the format (e_start, c_start, e_class)
                    if len(ps):
                        for index, j in enumerate(ps):
                            if j < self.word_start_index:
                                cur_pair.append(j)
                                if len(cur_pair) != 3:
                                    pass
                                else:
                                    pairs.append(tuple(cur_pair))
                                cur_pair = []
                            else:
                                cur_pair.append(j)
                    self.pred_triplets.append(pairs)
                    

                self.raw_words.extend(raw_words.tolist())
                self.tgt_triplets.extend(target_span.tolist())

        def get_metric(self, reset=True):
            data = []
            if self.add_ERC:
                for raw_words, tgt_emotions, pred_emotions, tgt_triplets, pred_triplets in zip(
                self.raw_words, self.emotions, self.emotions_pred, self.tgt_triplets, self.pred_triplets):
                    data.append({
                        'words': raw_words,
                        'tgt_emotions': tgt_emotions,
                        'pred_emotions': pred_emotions,
                        'tgt_triplets': tgt_triplets,
                        'pred_triplets': pred_triplets
                    })
            else:
                for raw_words, tgt_emotions, tgt_triplets, pred_triplets in zip(
                self.raw_words, self.emotions, self.tgt_triplets, self.pred_triplets):
                    data.append({
                        'words': raw_words,
                        'tgt_emotions': tgt_emotions,
                        'tgt_triplets': tgt_triplets,
                        'pred_triplets': pred_triplets
                    })
            
            line = json.dumps(data, indent=1)
            with open(self.fp, 'w', encoding='utf-8') as f:
                f.write(line)
            return {}


    if args.save_path:
        pred_path = args.save_path + 'save_predictions/{}/'.format(args.seed)
    else:
        pred_path = './save_predictions/'
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    fp = pred_path + datetime.datetime.now().strftime('%Y%m%d%H%M_') + os.path.split(args.dataset_name)[-1] + '_{}_{}.txt'.format(args.metric_key, args.max_len)
    
    tester = Tester(test, model, metrics=[metric, WriteResultToFileMetric(args.add_ERC, fp, eos_token_id, num_labels=len(label_ids))], batch_size=args.test_batch_size, device=device)
    
    eval_results = tester.test()
    res = eval_results['Seq2SeqSpanMetric']['ECTEC']
    print(fp)

    print_time()

    return res


def print_time():
    print(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))

    
if __name__ == "__main__":
    run()
