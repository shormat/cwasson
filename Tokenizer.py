import sentencepiece as spm
import torch

class Tokenizer():
    def __init__(self, model_path, max_len):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.max_len = max_len

        self.voc_size = self.sp.vocab_size()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()


    def encode(self, src, tgt):
        src = self.sp.encode(src)
        tgt = self.sp.encode(tgt)

        # add special tokens
        src = [self.bos_id] + src + [self.eos_id]
        tgt = [self.bos_id] + tgt + [self.eos_id]

        # padding / truncation
        src = src[:self.max_len] + [self.pad_id] * max(0, self.max_len - len(src))
        tgt = tgt[:self.max_len] + [self.pad_id] * max(0, self.max_len - len(tgt))

        return torch.LongTensor(src), torch.LongTensor(tgt)

