from fastNLP.io import Pipe, DataBundle, Loader
import os, json
from fastNLP import DataSet, Instance
from transformers import AutoTokenizer
import numpy as np
from itertools import chain
from functools import cmp_to_key


emotion_idx = dict(zip(['neutral','anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'], range(7)))
max_utt_num = 35


def cmp_emo_utt(v1, v2):
    if v1[0]['from']==v2[0]['from']:
        return v1[1]['from'] - v2[1]['from']
    return v1[0]['from'] - v2[0]['from']



class BartECTECPipe(Pipe):
    def __init__(self, tokenizer='facebook/bart-base'):
        super(BartECTECPipe, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        
        self.mapping = {
            'anger': '<<anger>>', 
            'disgust': '<<disgust>>', 
            'fear': '<<fear>>',
            'joy': '<<joy>>', 
            'sadness': '<<sadness>>', 
            'surprise': '<<surprise>>',
            'neutral': '<<neutral>>'
            }
        
        cur_num_tokens = self.tokenizer.vocab_size
        self.cur_num_token = cur_num_tokens

        unique_no_split_tokens = self.tokenizer.unique_no_split_tokens

        add_tokens = list(self.mapping.values())
        self.max_utt_num = max_utt_num
        utt_prefix_list = ['<<U{}>>'.format(i) for i in range(self.max_utt_num)]
        add_tokens += utt_prefix_list
        add_tokens += ['<<react>>', '<</react>>']
        add_tokens += ['<<xReact{}>>'.format(i) for i in range(self.max_utt_num)]
        add_tokens += ['<<oReact{}>>'.format(i) for i in range(self.max_utt_num)]
        for tok in add_tokens:
            assert self.tokenizer.convert_tokens_to_ids([tok])[0]==self.tokenizer.unk_token_id
        self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + add_tokens

        self.tokenizer.add_tokens(add_tokens)
        self.mapping2id = {}
        self.mapping2targetid = {}

        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            assert key_id[0] >= cur_num_tokens
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid)
        
        self.start_prefix_id = self.tokenizer.convert_tokens_to_ids('<<U0>>')
        self.last_prefix_id = self.tokenizer.convert_tokens_to_ids('<<U{}>>'.format(self.max_utt_num-1))
    
    def tokenize_tokens(self, raw_words):
        w_bpes = [[self.tokenizer.bos_token_id]]
        for word in raw_words:
            bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
            bpes = self.tokenizer.convert_tokens_to_ids(bpes)
            w_bpes.append(bpes)
        w_bpes.append([self.tokenizer.eos_token_id])
        all_bpes = list(chain(*w_bpes))
        return w_bpes, all_bpes


    def process(self, data_bundle: DataBundle) -> DataBundle:
        """
        words: List[str]
        emo_utts: [{
            'index': int
            'from': int
            'to': int
            "category": str
            'term': List[str]
        }],
        cau_utts: [{
            'index': int
            'from': int
            'to': int
            'term': List[str]
        }]

        输出为[o_s, o_e, a_s, a_e, c]或者[a_s, a_e, o_s, o_e, c]
        :param data_bundle:
        :return:
        """

        target_shift = len(self.mapping)-1 + 2  # Remove "neutral", add the start token (sos) and end token (eos)

        def prepare_target(ins):
            emotions = ins["emotions"]
            speakers = ins["speakers"]
            knowledge_relations = ins['knowledge_relations']
            knowledge_tokens_dict = ins['knowledge_tokens_dict']
            
            raw_words = ins['raw_words']
            ##            <s>   w0    w1     w2
            ##             0   1 2   3 4 5   6 7
            ## word_bpes=[[x],[x,x],[x,x,x],[x,x]]  len = [1,2,3,2]
            word_bpes, _word_bpes = self.tokenize_tokens(raw_words)
            word_bpes_xReact, _word_bpes_xReact = self.tokenize_tokens(knowledge_tokens_dict['xReact'])
            word_bpes_oReact, _word_bpes_oReact = self.tokenize_tokens(knowledge_tokens_dict['oReact'])
            word_bpes_xReact_retrieval, _word_bpes_xReact_retrieval = self.tokenize_tokens(knowledge_tokens_dict['xReact_retrieval'])
            word_bpes_oReact_retrieval, _word_bpes_oReact_retrieval = self.tokenize_tokens(knowledge_tokens_dict['oReact_retrieval'])

            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(list(lens)).tolist() # [1,3,6,8]
            target = [0]  # special sos
            target_spans = []
            
            def get_prefix_ids(_word_bpes, start_prefix_id, last_prefix_id):
                utt_prefix_ids = []
                for ii, w_id in enumerate(_word_bpes):
                    if w_id >= start_prefix_id and w_id <= last_prefix_id:
                        utt_prefix_ids.append(ii)
                return utt_prefix_ids
            
            utt_prefix_ids = get_prefix_ids(_word_bpes, self.start_prefix_id, self.last_prefix_id)
            utt_prefix_ids_xReact = get_prefix_ids(_word_bpes_xReact, self.tokenizer.convert_tokens_to_ids('<<xReact0>>'), self.tokenizer.convert_tokens_to_ids('<<xReact{}>>'.format(self.max_utt_num-1)))
            utt_prefix_ids_oReact = get_prefix_ids(_word_bpes_oReact, self.tokenizer.convert_tokens_to_ids('<<oReact0>>'), self.tokenizer.convert_tokens_to_ids('<<oReact{}>>'.format(self.max_utt_num-1)))
            utt_prefix_ids_xReact_retrieval = get_prefix_ids(_word_bpes_xReact_retrieval, self.tokenizer.convert_tokens_to_ids('<<xReact0>>'), self.tokenizer.convert_tokens_to_ids('<<xReact{}>>'.format(self.max_utt_num-1)))
            utt_prefix_ids_oReact_retrieval = get_prefix_ids(_word_bpes_oReact_retrieval, self.tokenizer.convert_tokens_to_ids('<<oReact0>>'), self.tokenizer.convert_tokens_to_ids('<<oReact{}>>'.format(self.max_utt_num-1)))

            emo_utts_cau_utts = [(e, c) for e, c in zip(ins["emo_utts"], ins["cau_utts"])]

            emo_utts_cau_utts = sorted(emo_utts_cau_utts, key=cmp_to_key(cmp_emo_utt))

            for emo_utts, cau_utts in emo_utts_cau_utts:
                assert emo_utts['index'] == cau_utts['index']
                a_start_bpe = cum_lens[emo_utts['from']]
                a_end_bpe = cum_lens[emo_utts['to']-1]
                o_start_bpe = cum_lens[cau_utts['from']]
                o_end_bpe = cum_lens[cau_utts['to']-1] 

                # Validate alignment between idx and word
                for idx, word in zip((o_start_bpe, o_end_bpe, a_start_bpe, a_end_bpe), (cau_utts['term'][0], cau_utts['term'][-1], emo_utts['term'][0], emo_utts['term'][-1])):
                    assert _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[:1])[0] or \
                           _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[-1:])[0]

                target_spans.append([a_start_bpe+target_shift,  o_start_bpe+target_shift])
                target_spans[-1].append(self.mapping2targetid[emo_utts["category"]]+2) 
                target_spans[-1] = tuple(target_spans[-1])
            target.extend(list(chain(*target_spans)))
            target.append(1)  # special eos

            tgt_emotions = [emotion_idx[x] for x in ins['emotions']]
            
            return {'tgt_tokens': target, 'target_span': target_spans, 'src_tokens': list(chain(*word_bpes)), 'utt_prefix_ids': utt_prefix_ids, 'tgt_emotions': tgt_emotions, 'src_tokens_xReact': list(chain(*word_bpes_xReact)), 'utt_prefix_ids_xReact': utt_prefix_ids_xReact, 'src_tokens_oReact': list(chain(*word_bpes_oReact)), 'utt_prefix_ids_oReact': utt_prefix_ids_oReact, 'src_tokens_xReact_retrieval': list(chain(*word_bpes_xReact_retrieval)), 'utt_prefix_ids_xReact_retrieval': utt_prefix_ids_xReact_retrieval, 'src_tokens_oReact_retrieval': list(chain(*word_bpes_oReact_retrieval)), 'utt_prefix_ids_oReact_retrieval': utt_prefix_ids_oReact_retrieval}


        data_bundle.apply_more(prepare_target, use_tqdm=True, tqdm_desc='Pre. tgt.')
        data_bundle.set_ignore_type('target_span')
        data_bundle.set_pad_val('tgt_tokens', 1)  # eos
        data_bundle.set_pad_val('tgt_emotions', 0)
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='utt_prefix_ids', new_field_name='dia_utt_num')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_emotions', new_field_name='tgt_emo_seq_len')
        
        data_bundle.set_pad_val('src_tokens_xReact', self.tokenizer.pad_token_id)
        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens_xReact', new_field_name='src_seq_len_xReact')
        data_bundle.set_pad_val('src_tokens_oReact', self.tokenizer.pad_token_id)
        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens_oReact', new_field_name='src_seq_len_oReact')
        
        data_bundle.set_pad_val('src_tokens_xReact_retrieval', self.tokenizer.pad_token_id)
        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens_xReact_retrieval', new_field_name='src_seq_len_xReact_retrieval')
        data_bundle.set_pad_val('src_tokens_oReact_retrieval', self.tokenizer.pad_token_id)
        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens_oReact_retrieval', new_field_name='src_seq_len_oReact_retrieval')
        
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'utt_prefix_ids', 'dia_utt_num', 'tgt_emotions','tgt_emo_seq_len', 'src_tokens_xReact', 'src_seq_len_xReact', 'src_tokens_oReact', 'src_seq_len_oReact', 'utt_xReact_mask', 'utt_oReact_mask', 'utt_prefix_ids_xReact', 'utt_prefix_ids_oReact', 'src_tokens_xReact_retrieval', 'src_seq_len_xReact_retrieval', 'src_tokens_oReact_retrieval', 'src_seq_len_oReact_retrieval', 'utt_prefix_ids_xReact_retrieval', 'utt_prefix_ids_oReact_retrieval')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', 'target_span', 'utt_prefix_ids', 'dia_utt_num', 'tgt_emotions','tgt_emo_seq_len')
        return data_bundle


    def process_from_file(self, paths) -> DataBundle:
        """
        :param paths: Refer to the load function in `fastNLP.io.loader.ConllLoader` for supported path types.
        :return: DataBundle
        """
        data_bundle = ABSALoader().load(paths) # Transform Dataset-type data into a data_bundle
        data_bundle = self.process(data_bundle) # Transform the data_bundle into the data required for the task.
        return data_bundle



class ABSALoader(Loader):
    """
    Store raw data in Dataset format using Loader
    """
    def __init__(self):
        super().__init__()
        self.max_utt_num = max_utt_num

    def _load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ds = DataSet()
        for ins in data:
            tokens = ins["words"]
            emotions = ins["emotions"]
            speakers = ins["speakers"]
            emo_utts = ins["emo_utts"]
            cau_utts = ins["cau_utts"]
            assert len(emo_utts)==len(cau_utts)
            knowledge_relations = ins["knowledge_relations"]
            knowledge_tokens_dict = {}
            for ii,rel in enumerate(knowledge_relations):
                knowledge_tokens_dict[rel] = ins[rel]
                knowledge_tokens_dict[rel+'_retrieval'] = ins[rel+'_retrieval']
                
            ins = Instance(raw_words=tokens, emo_utts=emo_utts, cau_utts=cau_utts, emotions=emotions, speakers=speakers, knowledge_relations=knowledge_relations, knowledge_tokens_dict=knowledge_tokens_dict)
            
            utt_xReact_mask = [[0]*self.max_utt_num for _ in range(self.max_utt_num)]
            utt_oReact_mask = [[0]*self.max_utt_num for _ in range(self.max_utt_num)]

            for ii in range(len(speakers)):
                start_id = 0
                for jj in np.arange(start_id, ii+1):
                    if speakers[ii] == speakers[jj]:
                        utt_xReact_mask[ii][jj] = 1
                    else:
                        utt_oReact_mask[ii][jj] = 1
            ins.add_field('utt_xReact_mask', utt_xReact_mask)
            ins.add_field('utt_oReact_mask', utt_oReact_mask)
            
            ds.append(ins)
        return ds
