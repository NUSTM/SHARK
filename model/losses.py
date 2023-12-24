
from fastNLP import LossBase
import torch.nn.functional as F
from fastNLP import seq_len_to_mask


class Seq2SeqLoss(LossBase):
    def __init__(self, add_ERC):
        super().__init__()
        self.add_ERC = add_ERC

    def get_loss(self, tgt_tokens, tgt_seq_len, tgt_emotions, tgt_emo_seq_len, pred):
        """
        :param tgt_tokens: bsz x max_len, [sos, tokens, eos]
        :param pred: bsz x max_len-1 x vocab_size
        :return:
        """
        tgt_seq_len = tgt_seq_len - 1
        mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
        tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)
        loss = F.cross_entropy(target=tgt_tokens, input=pred['pred_ectec'].transpose(1, 2))
        
        if self.add_ERC:
            mask_emo = seq_len_to_mask(tgt_emo_seq_len, max_len=tgt_emotions.size(1)).eq(0)
            tgt_emotions = tgt_emotions.masked_fill(mask_emo, -100)
            loss_emo = F.cross_entropy(target=tgt_emotions, input=pred['pred_emo'].transpose(1, 2))
            return loss + loss_emo
        else:
            return loss

