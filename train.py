import torch, os, datetime, warnings, fitlog, argparse, json, random
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

warnings.filterwarnings('ignore')
from data.pipe import BartECTECPipe
from model.bart_ectec import BartSeq2SeqModel
from fastNLP import Trainer, MetricBase, Tester
from model.metrics import Seq2SeqSpanMetric
from model.losses import Seq2SeqLoss
from torch import optim
from fastNLP import BucketSampler, GradientClipCallback, cache_results, WarmupCallback
from fastNLP import FitlogCallback
from fastNLP.core.sampler import SortedSampler
from model.generator import SequenceGeneratorModel
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='ECF', type=str)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--test_batch_size', default=16, type=int)
parser.add_argument('--num_beams', default=4, type=int)
parser.add_argument('--n_epochs', default=50, type=int)
parser.add_argument('--bart_name', default='facebook/bart-base', type=str)
parser.add_argument('--max_len', default=50, type=int)
parser.add_argument('--metric_key', default='triple_f_w-avg-6', type=str)
parser.add_argument('--seed', type=int, default=0, help='seed')

parser.add_argument('--num_run', type=int, default=5)
parser.add_argument('--save_model', type=int, default=0)
parser.add_argument('--save_path', default='./output/', type=str)
parser.add_argument('--test', action='store_true', default=False)

parser.add_argument('--fuse_type', default='gat', type=str)
parser.add_argument('--use_gate', type=int, default=1)
parser.add_argument('--use_retrieval_CSK', type=int, default=1)
parser.add_argument('--use_generated_CSK', type=int, default=1)
parser.add_argument('--use_CSK', type=int, default=1)
parser.add_argument('--add_ERC', type=int, default=1)

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

    bos_token_id = 0
    eos_token_id = 1
    label_ids = list(mapping2id.values())
    
    model = BartSeq2SeqModel.build_model(args.bart_name, tokenizer, label_ids=label_ids)
    model = SequenceGeneratorModel(model, args.use_CSK, args.add_ERC, args.use_gate, args.fuse_type, args.use_retrieval_CSK, args.use_generated_CSK, bos_token_id=bos_token_id,
                                eos_token_id=eos_token_id,
                                max_length=args.max_len, max_len_a=1, num_beams=args.num_beams, do_sample=False,
                                repetition_penalty=1, length_penalty=1.0, pad_token_id=eos_token_id,
                                restricter=None)
    # print(model)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    parameters = []
    params = {'lr':args.lr, 'weight_decay':1e-2}
    params['params'] = [param for name, param in model.named_parameters() if not ('bart_encoder' in name or 'bart_decoder' in name)]
    parameters.append(params)

    params = {'lr':args.lr, 'weight_decay':1e-2}
    params['params'] = []
    for name, param in model.named_parameters():
        if ('bart_encoder' in name or 'bart_decoder' in name) and not ('layernorm' in name or 'layer_norm' in name):
            params['params'].append(param)
    parameters.append(params)

    params = {'lr':args.lr, 'weight_decay':0}
    params['params'] = []
    for name, param in model.named_parameters():
        if ('bart_encoder' in name or 'bart_decoder' in name) and ('layernorm' in name or 'layer_norm' in name):
            params['params'].append(param)
    parameters.append(params)

    optimizer = optim.AdamW(parameters)


    callbacks = []
    callbacks.append(GradientClipCallback(clip_value=5, clip_type='value'))
    callbacks.append(WarmupCallback(warmup=0.01, schedule='linear'))
    callbacks.append(FitlogCallback(data_bundle.get_dataset('test')))

    sampler = None
    sampler = BucketSampler(seq_len_field_name='src_seq_len')
    
    metric = Seq2SeqSpanMetric(args.add_ERC, eos_token_id, num_labels=len(label_ids))


    model_path = None
    if args.save_model:
        if args.save_path:
            model_path = args.save_path + 'save_models/{}/'.format(args.seed) 
        else:
            model_path = './save_models/{}/'.format(args.seed)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

    trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, optimizer=optimizer,
                    loss=Seq2SeqLoss(args.add_ERC),
                    batch_size=args.batch_size, sampler=sampler, drop_last=False, update_every=args.gradient_accumulation_steps,
                    num_workers=2, n_epochs=args.n_epochs, print_every=1,
                    dev_data=data_bundle.get_dataset('dev'), metrics=metric, metric_key=args.metric_key,
                    validate_every=-1, save_path=model_path, use_tqdm=True, device=device,
                    callbacks=callbacks, check_code_level=-1, test_use_tqdm=False,
                    test_sampler=SortedSampler('src_seq_len'), dev_batch_size=args.batch_size)

    res = []
    if not args.test:
        trainer.train(load_best_model=False)
    else:
        trainer.train(load_best_model=True)
        print_time()


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
                pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)
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
    
    seed_pool = [2023, 153, 42, 43, 52]
    all_res = []

    for ii in range(args.num_run):
        args.seed = seed_pool[ii]
        print('\n\n****** run {} begin ******\n'.format(ii+1))
        res = run()
        all_res.append(res)
        print('\n****** run {} end ******\n\n'.format(ii+1))
    
        if ii > 0:
            print('\n\n{}\n\n{} runs average:\n{}\n{}'.format(np.array(all_res), ii+1, np.array(all_res).mean(axis=0), np.array(all_res).std(axis=0)))