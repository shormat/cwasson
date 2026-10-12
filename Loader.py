import sentencepiece as spm
from torch.utils.data import Dataset, random_split
import torch
from Tokenizer import Tokenizer

class DatasetLoader(Dataset):
    def __init__(self, src_path, tgt_path, tokenizer):
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.tokenizer = tokenizer

        # None here, each workers will open its own file descriptor
        self.src_fd = None
        self.tgt_fd = None
    
        # Offsets of the start of each line in the source and target files
        self.offsets = []
        with open(self.src_path, 'r', encoding='utf-8') as f_src, \
             open(self.tgt_path, 'r', encoding='utf-8') as f_tgt:
            src_off = f_src.tell()
            tgt_off = f_tgt.tell()
            while True:
                src_line = f_src.readline()
                tgt_line = f_tgt.readline()
                if not src_line or not tgt_line:
                    break
                if src_line.strip() and tgt_line.strip():
                    self.offsets.append((src_off, tgt_off))
                src_off = f_src.tell()
                tgt_off = f_tgt.tell()
    
    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        if self.src_fd is None:
            self.src_fd = open(self.src_path, 'r', encoding='utf-8')
        if self.tgt_fd is None:
            self.tgt_fd = open(self.tgt_path, 'r', encoding='utf-8')
        
        src_off, tgt_off = self.offsets[idx]

        self.src_fd.seek(src_off)
        self.tgt_fd.seek(tgt_off)
        src_line = self.src_fd.readline().strip()
        tgt_line = self.tgt_fd.readline().strip()

        # return the tokenized
        return self.tokenizer.encode(src_line, tgt_line)
    
    def __del__(self):
        if self.src_fd:
            self.src_fd.close()
        if self.tgt_fd:
            self.tgt_fd.close()

    def get_splits(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

        # Close any open file descriptors before splitting
        if self.src_fd is not None:
            self.src_fd.close()
            self.src_fd = None
        if self.tgt_fd is not None:
            self.tgt_fd.close()
            self.tgt_fd = None

        total = len(self)
        train_len = int(total * train_ratio)
        val_len = int(total * val_ratio)
        test_len = total - train_len - val_len
        
        return random_split(
            self,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(seed)
        )