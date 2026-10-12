import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:x.size(1)]

        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('scale', torch.sqrt(torch.tensor(self.d_k, dtype=torch.float)))

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear projections
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        attn  = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = self.dropout(F.softmax(attn, dim=-1))

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        out = self.w_o(out)

        return out
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        l1 = self.linear1(x)
        relu = F.relu(l1)
        drop = self.dropout(relu)
        l2 = self.linear2(drop)
        
        return l2

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, src_mask=None):
        attn_output = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoding_out, src_mask=None, tgt_mask=None):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        attn_output = self.cross_attn(x, encoding_out, encoding_out, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, voc_size, d_model, n_layers, n_heads, d_ff, max_len=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(voc_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('scale', torch.sqrt(torch.tensor(d_model, dtype=torch.float)))

    def forward(self, x, src_mask=None):
        x = self.embedding(x) * self.scale
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x
    
class Decoder(nn.Module):
    def __init__(self, voc_size, d_model, n_layer, n_heads, d_ff, max_len=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(voc_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layer)])
        self.fc_out = nn.Linear(d_model, voc_size)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('scale', torch.sqrt(torch.tensor(d_model, dtype=torch.float)))

    def forward(self, x, encoding_out, src_mask=None, tgt_mask=None):
        x = self.embedding(x) * self.scale
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, encoding_out, src_mask, tgt_mask)

        out = self.fc_out(x)

        return out
    
class Transformer(nn.Module):
    def __init__(self, voc_size, pad_idx, d_model=512, n_layers=6, n_heads=8, d_ff=2048, max_len=128, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(voc_size, d_model, n_layers, n_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(voc_size, d_model, n_layers, n_heads, d_ff, max_len, dropout)
        self.src_pad_idx = pad_idx
        self.tgt_pad_idx = pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        device = tgt.device
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        encoding_out = self.encoder(src, src_mask)
        decoding_out = self.decoder(tgt, encoding_out, src_mask, tgt_mask)

        return decoding_out