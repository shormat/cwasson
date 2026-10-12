<div align="center">

# 🥐 Cwasson

**A French → English neural machine translator, built from scratch in PyTorch.**

Every piece of the Transformer is written by hand: multi-head attention, positional encoding, encoder/decoder stacks, masking, greedy decoding.
No `nn.Transformer`, no pre-trained weights.

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-from%20scratch-EE4C2C?logo=pytorch&logoColor=white)
![Accelerate](https://img.shields.io/badge/🤗%20Accelerate-multi--GPU-FFD21E)
![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-optional-FFBE00?logo=weightsandbiases&logoColor=black)

[Architecture](#-architecture) · [Quick start](#-quick-start) · [Results](#-results)

</div>

---

## ✨ Highlights

- **Transformer from scratch**: the architecture from [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), implemented module by module in [`Transformer.py`](Transformer.py).
- **Shared BPE vocabulary**: a 32k-token SentencePiece model trained jointly on French and English ([`shuffled.model`](shuffled.model)).
- **Streaming dataset**: [`Loader.py`](Loader.py) indexes byte offsets instead of loading the corpus into RAM, so it scales to millions of sentence pairs.
- **Multi-GPU training** with 🤗 Accelerate (plus a plain `DataParallel` fallback).
- **Evaluation**: validation loss and corpus-level **SacreBLEU**, gathered across processes.
- **Experiment tracking**: optional, through Weights & Biases.
- **Interactive mode**: type French, get English.

## 🧠 Architecture

```text
 French sentence                                     English sentence (shifted right)
       │                                                         │
 SentencePiece BPE (32k, shared)                     SentencePiece BPE (32k, shared)
       │                                                         │
 Embedding × √d_model + sinusoidal PE                Embedding × √d_model + sinusoidal PE
       │                                                         │
 ┌─────▼─────────────── × N ┐                       ┌────────────▼──────────── × N ┐
 │ Multi-Head Self-Attention│                       │ Masked Multi-Head Self-Attn  │
 │ Add & LayerNorm          │                       │ Add & LayerNorm              │
 │ Feed-Forward (ReLU)      │  ── encoder output ─▶ │ Multi-Head Cross-Attention   │
 │ Add & LayerNorm          │                       │ Add & LayerNorm              │
 └──────────────────────────┘                       │ Feed-Forward (ReLU)          │
                                                    │ Add & LayerNorm              │
                                                    └────────────┬─────────────────┘
                                                                 │
                                                       Linear → vocab logits
```

Default training configuration (see [`main.py`](main.py)):

| Hyperparameter | Value | | Hyperparameter | Value |
|---|---|---|---|---|
| `d_model` | 256 | | Max sequence length | 128 |
| Attention heads | 4 | | Batch size | 32 |
| `d_ff` | 1024 | | Optimizer | Adam, lr = 3e-4 |
| Encoder / decoder layers | 3 / 3 | | Loss | Cross-entropy, label smoothing 0.1 |
| Dropout | 0.2 | | Split | 90 / 5 / 5 |

## 📁 Project structure

```text
cwasson/
├── Transformer.py          # Positional encoding, multi-head attention, encoder/decoder, full model
├── Cwasson.py              # Trainer: training, validation, BLEU, checkpoints, translation
├── Tokenizer.py            # SentencePiece wrapper (BOS/EOS, padding, truncation)
├── Loader.py               # Streaming parallel-corpus Dataset based on byte offsets
├── WandbLogger.py          # Thin, optional Weights & Biases wrapper
├── main.py                 # Training entry point
├── main_translate.py       # Inference / interactive translation entry point
├── downloader_opus.ipynb   # Download and clean OPUS-100 (en-fr)
├── cleaner_dataset.py      # fastText language-ID filtering (Tatoeba)
└── shuffled.model          # Trained SentencePiece BPE model (32k vocab)
```

## 🚀 Quick start

### 1. Install

```bash
git clone https://github.com/shormat/cwasson.git
cd cwasson
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare the data

The model trains on about **873k clean sentence pairs** taken from [Tatoeba](https://tatoeba.org) and [OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100). The corpora are not included in this repository.

1. **OPUS-100**: run [`downloader_opus.ipynb`](downloader_opus.ipynb). It keeps pairs of at most 50 words with a length ratio between 0.5 and 2, removes duplicates and filters out unusual characters. It writes `train.fr` and `train.en`.
2. **Tatoeba**: download the Moses-format `Tatoeba.en-fr.{fr,en}` files from [OPUS](https://opus.nlpl.eu/Tatoeba.php) and the fastText [`lid.176.bin`](https://fasttext.cc/docs/en/language-identification.html) model. Then run:
   ```bash
   python cleaner_dataset.py   # → filtered_french.txt / filtered_english.txt
   ```
3. Concatenate and shuffle both corpora, keeping the lines aligned, into **`shuffled_fr.txt`** and **`shuffled_en.txt`**: line *n* of one file must be the translation of line *n* of the other.

### 3. Train

```bash
python main.py                      # single process
accelerate launch main.py           # multi-GPU (run `accelerate config` first)
```

Checkpoints are saved under `checkpoints/{train,epochs,val,test}/`. To log runs to W&B, set `USE_WANDB = True` in `main.py`.

### 4. Translate

```bash
python main_translate.py checkpoints/epochs/model_epoch_20_<timestamp>.pth
```

```text
Starting interactive translation session...
Type 'quit' or 'exit' to end the session
> Si j'avais le temps, je le ferai.

Translation: If I had time, I'll do it.
```

> [!NOTE]
> The trainer builds its data loaders when it is created, so `shuffled_fr.txt` and `shuffled_en.txt` must exist even for inference.

## 📊 Results

After 20 epochs, the validation loss fell steadily from **~4.7 to ~3.85** and never plateaued. The training and validation losses converge, so the model shows no sign of overfitting. BLEU rises throughout training but stays modest in absolute terms, because the outputs are often correct yet more literal than the references.

| 🇫🇷 French | 🇬🇧 Cwasson |
|---|---|
| Si j'avais le temps, je le ferai. | If I had time, I'll do it. |
| Bonjour, comment allez vous ? | Hello, how are you going? |
| Il fait ses devoirs. | He makes his homework |
| Il a raté tous ses examens | He missed all his exams. |

## 👤 Author

**Saïd Hormat-Allah** ([@shormat](https://github.com/shormat)). Built as a deep learning course project at the **University of Liège**.

## 📚 References

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), NeurIPS 2017
- Kudo & Richardson, [*SentencePiece*](https://arxiv.org/abs/1808.06226), EMNLP 2018
- Zhang et al., [*OPUS-100*](https://arxiv.org/abs/2004.11867), ACL 2020
- Post, [*A Call for Clarity in Reporting BLEU Scores*](https://arxiv.org/abs/1804.08771), WMT 2018
