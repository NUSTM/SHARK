import torch
from .modeling_bart import BartEncoder, BartDecoder, BartModel
from transformers import BartTokenizer
from fastNLP import seq_len_to_mask
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
import torch.nn.functional as F
from fastNLP.models import Seq2SeqModel
from torch import nn



def get_utt_representation(all_word_state, utt_prefix_ids, dia_utt_num, transformer_unit, use_trans_layer=False):
    bz, _, h = all_word_state.size() # bsz x max_word_len x hidden_size
    
    output = all_word_state.gather(index=utt_prefix_ids.unsqueeze(2).repeat(1, 1, all_word_state.size(-1)), dim=1) # bsz x max_utt_len x hidden_size
    utt_mask = seq_len_to_mask(dia_utt_num, max_len=output.size(1)).eq(0) # bsz x max_utt_len
    utt_mask_ = utt_mask.unsqueeze(2).repeat(1, 1, output.size(-1))
    output = output.masked_fill(utt_mask_, 0)
    
    # cls_tokens, _ = torch.max(hidden_states, dim=1)  # max pooling
    if use_trans_layer:
        output = transformer_unit(output)

    return output  # bsz x max_utt_len(35) x hidden_size


class FBartEncoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        assert isinstance(encoder, BartEncoder)
        self.bart_encoder = encoder

    def forward(self, src_tokens, src_seq_len):
        mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        dict = self.bart_encoder(input_ids=src_tokens, attention_mask=mask, return_dict=True,
                                 output_hidden_states=True)
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        return encoder_outputs, mask, hidden_states


# The generated position index sequence needs to be converted back to the original word sequence before being fed into the decoder.
class FBartDecoder(Seq2SeqDecoder):
    def __init__(self, decoder, pad_token_id, label_ids):
        super().__init__()
        assert isinstance(decoder, BartDecoder)
        self.decoder = decoder
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer('causal_masks', causal_mask.float())
        self.pad_token_id = pad_token_id
        self.label_start_id = label_ids[0] 
        self.label_end_id = label_ids[-1]
        mapping = torch.LongTensor([0, 2]+label_ids)
        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping) - 1
        hidden_size = decoder.embed_tokens.weight.size(1)
        self.encoder_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.Dropout(0.3),
                                             nn.ReLU(),
                                             nn.Linear(hidden_size, hidden_size))
        
        
    def forward(self, tokens, utt_prefix_ids, dia_utt_num, state):
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        first = state.first

        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # mapping to the BART token index
        mapping_token_mask = tokens.lt(self.src_start_index)  #
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)

        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)

        eos_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[2:3])  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id])  # bsz x max_len x num_class

        # bsz x max_word_len x hidden_size
        src_outputs = state.encoder_output
        src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)

        mask = mask.unsqueeze(1).__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        return logits


    def decode(self, tokens, utt_prefix_ids, dia_utt_num, state):
        return self(tokens, utt_prefix_ids, dia_utt_num, state)[:, -1]
        

class CaGFBartDecoder(FBartDecoder):
    # Copy and generate
    def __init__(self, decoder, pad_token_id, label_ids):
        super().__init__(decoder, pad_token_id, label_ids)
        self.dropout_layer = nn.Dropout(0.3)
        

    def forward(self, tokens, utt_prefix_ids, dia_utt_num, state):
        if tokens.size(0) != utt_prefix_ids.size(0):
            beam_size = tokens.size(0)//utt_prefix_ids.size(0)
            utt_prefix_ids = utt_prefix_ids.repeat_interleave(beam_size, dim=0)
            dia_utt_num = dia_utt_num.repeat_interleave(beam_size, dim=0)
        
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask
        tgt_tokens_copy = tokens

        first = state.first
        
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # mapping to the BART token index
        mapping_token_mask = tokens.lt(self.src_start_index)
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)  # bsz x max_len
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)
        
        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
        
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size. 
        hidden_state = self.dropout_layer(hidden_state)
        if not self.training:
            state.past_key_values = dict.past_key_values

        
        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)), fill_value=-1e24) # bsz x max_len x (max_word_len+2+num_labels)

        eos_scores = F.linear(hidden_state, self.dropout_layer(self.decoder.embed_tokens.weight[2:3]))  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.dropout_layer(self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id]))  # bsz x max_len x num_class
        
        src_outputs = state.encoder_output
        src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)

        mask = mask.unsqueeze(1)
        input_embed = self.dropout_layer(self.decoder.embed_tokens(src_tokens))  # bsz x max_word_len x hidden_size

        src_outputs = (src_outputs + input_embed)/2
        
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        mask = mask.__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        return logits



class BartSeq2SeqModel(Seq2SeqModel):
    @classmethod
    def build_model(cls, bart_model, tokenizer, label_ids, decoder_type='avg_feature', copy_gate=False, use_recur_pos=False, tag_first=False):
        """
        Define the encoder and decoder
        Initialize the custom tokens
        """
        model = BartModel.from_pretrained(bart_model)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens)+num_tokens)
        encoder = model.encoder
        decoder = model.decoder

        if use_recur_pos:
            decoder.set_position_embedding(label_ids[0], tag_first)

        _tokenizer = BartTokenizer.from_pretrained(bart_model)
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':
                index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                if len(index)>1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]
                assert index>=num_tokens, (index, num_tokens, token)
                indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(token[2:-2]))
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                model.decoder.embed_tokens.weight.data[index] = embed

        encoder = FBartEncoder(encoder)
        
        if decoder_type is None:
            assert copy_gate is False
            decoder = FBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids)
        elif decoder_type == 'avg_feature':
            decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids)
        else:
            raise RuntimeError("Unsupported feature.")
        

        return cls(encoder=encoder, decoder=decoder)


    def prepare_state(self, src_tokens, src_tokens_xReact, src_seq_len_xReact, src_tokens_oReact, src_seq_len_oReact, utt_xReact_mask, utt_oReact_mask, utt_prefix_ids_xReact, utt_prefix_ids_oReact, src_tokens_xReact_retrieval, src_seq_len_xReact_retrieval, src_tokens_oReact_retrieval, src_seq_len_oReact_retrieval, utt_prefix_ids_xReact_retrieval, utt_prefix_ids_oReact_retrieval, gat, graph_att_layer, use_CSK, add_ERC, use_gate, fuse_type, use_retrieval_CSK, use_generated_CSK, linear_layer, linear_layer1, utt_prefix_ids, dia_utt_num, transformer_unit, emo_ffn, src_seq_len=None, first=None, tgt_seq_len=None):
        encoder_outputs, encoder_mask, hidden_states = self.encoder(src_tokens, src_seq_len)
        src_embed_outputs = hidden_states[0]

        if use_CSK:
            encoder_outputs_utt = get_utt_representation(encoder_outputs, utt_prefix_ids, dia_utt_num, transformer_unit, use_trans_layer=False)

            if use_generated_CSK:
                encoder_outputs_xReact, encoder_mask_xReact, hidden_states_xReact = self.encoder(src_tokens_xReact, src_seq_len_xReact)
                encoder_outputs_oReact, encoder_mask_oReact, hidden_states_oReact = self.encoder(src_tokens_oReact, src_seq_len_oReact)
                encoder_outputs_utt_xReact = get_utt_representation(encoder_outputs_xReact, utt_prefix_ids_xReact, dia_utt_num, transformer_unit, use_trans_layer=False)
                encoder_outputs_utt_oReact = get_utt_representation(encoder_outputs_oReact, utt_prefix_ids_oReact, dia_utt_num, transformer_unit, use_trans_layer=False)

                if not use_retrieval_CSK:
                    new_xReact_encoder_outputs_utt, new_oReact_encoder_outputs_utt = encoder_outputs_utt_xReact, encoder_outputs_utt_oReact
            
            if use_retrieval_CSK:
                encoder_outputs_xReact_retrieval, encoder_mask_xReact_retrieval, hidden_states_xReact_retrieval = self.encoder(src_tokens_xReact_retrieval, src_seq_len_xReact_retrieval)
                encoder_outputs_oReact_retrieval, encoder_mask_oReact_retrieval, hidden_states_oReact_retrieval = self.encoder(src_tokens_oReact_retrieval, src_seq_len_oReact_retrieval)
                encoder_outputs_utt_xReact_retrieval = get_utt_representation(encoder_outputs_xReact_retrieval, utt_prefix_ids_xReact_retrieval, dia_utt_num, transformer_unit, use_trans_layer=False)
                encoder_outputs_utt_oReact_retrieval = get_utt_representation(encoder_outputs_oReact_retrieval, utt_prefix_ids_oReact_retrieval, dia_utt_num, transformer_unit, use_trans_layer=False)

                if not use_generated_CSK:
                    new_xReact_encoder_outputs_utt, new_oReact_encoder_outputs_utt = encoder_outputs_utt_xReact_retrieval, encoder_outputs_utt_oReact_retrieval

            if use_retrieval_CSK and use_generated_CSK:
                if use_gate:
                    xReact_encoder_outputs_utt = torch.cat((encoder_outputs_utt, encoder_outputs_utt_xReact, encoder_outputs_utt_xReact_retrieval), -1)
                    oReact_encoder_outputs_utt = torch.cat((encoder_outputs_utt, encoder_outputs_utt_oReact, encoder_outputs_utt_oReact_retrieval), -1)
                    indicator_x = linear_layer(xReact_encoder_outputs_utt)
                    indicator_o = linear_layer(oReact_encoder_outputs_utt)
                    
                    indicator_x_ = F.softmax(indicator_x, dim=-1)
                    indicator_o_ = F.softmax(indicator_o, dim=-1)
                    indicator_x_ = indicator_x_[:,:,0].unsqueeze(2).repeat(1, 1, encoder_outputs_utt.size(-1))
                    indicator_o_ = indicator_o_[:,:,0].unsqueeze(2).repeat(1, 1, encoder_outputs_utt.size(-1))

                    new_xReact_encoder_outputs_utt = indicator_x_ * encoder_outputs_utt_xReact + (1 - indicator_x_) * encoder_outputs_utt_xReact_retrieval
                    new_oReact_encoder_outputs_utt = indicator_o_ * encoder_outputs_utt_oReact + (1 - indicator_o_) * encoder_outputs_utt_oReact_retrieval
                else:
                    new_xReact_encoder_outputs_utt = linear_layer1(torch.cat((encoder_outputs_utt_xReact, encoder_outputs_utt_xReact_retrieval), -1))
                    new_oReact_encoder_outputs_utt = linear_layer1(torch.cat((encoder_outputs_utt_oReact, encoder_outputs_utt_oReact_retrieval), -1))
                    
            if fuse_type == 'gat':
                new_encoder_outputs_utt = graph_att_layer(encoder_outputs_utt, new_xReact_encoder_outputs_utt, new_oReact_encoder_outputs_utt, utt_xReact_mask, utt_oReact_mask)
            elif fuse_type == 'gat1':
                new_encoder_outputs_utt = gat(encoder_outputs_utt, new_xReact_encoder_outputs_utt, new_oReact_encoder_outputs_utt, utt_xReact_mask, utt_oReact_mask)
            else:
                new_encoder_outputs_utt = encoder_outputs_utt + new_xReact_encoder_outputs_utt + new_oReact_encoder_outputs_utt
            
            utt_mask = seq_len_to_mask(dia_utt_num, max_len=encoder_outputs_utt.size(1)) # bsz x max_utt_len
            new_encoder_outputs_utt = new_encoder_outputs_utt.masked_fill(utt_mask.eq(0).unsqueeze(2).repeat(1, 1, encoder_outputs_utt.size(-1)), 0)
            
            logits = emo_ffn(new_encoder_outputs_utt)
            
            bz, _, _ = new_encoder_outputs_utt.size()
            new_encoder_outputs = encoder_outputs.clone()
            for ii in range(bz):
                for jj in range(dia_utt_num[ii]):
                    new_encoder_outputs[ii, utt_prefix_ids[ii][jj]] = encoder_outputs[ii, utt_prefix_ids[ii][jj]] + new_encoder_outputs_utt[ii, jj] # Add the origin prefix representation to the knowledge-enhanced utterance representation
                    
            state = BartState(new_encoder_outputs, encoder_mask, src_tokens, first, src_embed_outputs)
        else:
            logits = torch.tensor([])
            if add_ERC:
                encoder_outputs_utt = get_utt_representation(encoder_outputs, utt_prefix_ids, dia_utt_num, transformer_unit, use_trans_layer=False)
                logits = emo_ffn(encoder_outputs_utt)
            state = BartState(encoder_outputs, encoder_mask, src_tokens, first, src_embed_outputs)

        return state, logits


    def forward(self, src_tokens, src_tokens_xReact, src_seq_len_xReact, src_tokens_oReact, src_seq_len_oReact, utt_xReact_mask, utt_oReact_mask, utt_prefix_ids_xReact, utt_prefix_ids_oReact, src_tokens_xReact_retrieval, src_seq_len_xReact_retrieval, src_tokens_oReact_retrieval, src_seq_len_oReact_retrieval, utt_prefix_ids_xReact_retrieval, utt_prefix_ids_oReact_retrieval, gat, graph_att_layer, use_CSK, add_ERC, use_gate, fuse_type, use_retrieval_CSK, use_generated_CSK, linear_layer, linear_layer1, tgt_tokens, utt_prefix_ids, dia_utt_num, transformer_unit, emo_ffn, src_seq_len, tgt_seq_len, first):
        """
        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        
        state, emotion_pred_output = self.prepare_state(src_tokens, src_tokens_xReact, src_seq_len_xReact, src_tokens_oReact, src_seq_len_oReact, utt_xReact_mask, utt_oReact_mask, utt_prefix_ids_xReact, utt_prefix_ids_oReact, src_tokens_xReact_retrieval, src_seq_len_xReact_retrieval, src_tokens_oReact_retrieval, src_seq_len_oReact_retrieval, utt_prefix_ids_xReact_retrieval, utt_prefix_ids_oReact_retrieval, gat, graph_att_layer, use_CSK, add_ERC, use_gate, fuse_type, use_retrieval_CSK, use_generated_CSK, linear_layer, linear_layer1, utt_prefix_ids, dia_utt_num, transformer_unit, emo_ffn, src_seq_len, first, tgt_seq_len)
        
        decoder_output = self.decoder(tgt_tokens, utt_prefix_ids, dia_utt_num, state)

        pred = {}
        if add_ERC:
            if isinstance(emotion_pred_output, torch.Tensor):
                pred['pred_emo'] = emotion_pred_output
            elif isinstance(emotion_pred_output, (tuple, list)):
                pred['pred_emo'] = emotion_pred_output[0]
            else:
                raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")
        
        if isinstance(decoder_output, torch.Tensor):
            pred['pred_ectec'] = decoder_output
        elif isinstance(decoder_output, (tuple, list)):
            pred['pred_ectec'] = decoder_output[0]
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")

        return {'pred': pred}



class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, first, src_embed_outputs):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs, indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(layer[key1][key2], indices)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new