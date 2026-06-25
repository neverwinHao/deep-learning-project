import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import time
import math
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')

# ── Vocabulary ─────────────────────────────────────────────────

PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN = '<PAD>', '<BOS>', '<EOS>', '<UNK>'
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


def build_vocab(sentences, max_size=30000, min_freq=1):
    counter = Counter()
    for s in sentences:
        counter.update(s)
    vocab = SPECIAL_TOKENS[:]
    for word, freq in counter.most_common():
        if freq < min_freq:
            break
        vocab.append(word)
        if len(vocab) >= max_size:
            break
    word2idx = {w: i for i, w in enumerate(vocab)}
    return vocab, word2idx


# ── Data Loading ───────────────────────────────────────────────

def load_parallel_data(zh_path, en_path):
    with open(zh_path, 'r', encoding='utf-8') as f:
        zh_lines = [line.strip().split() for line in f if line.strip()]
    with open(en_path, 'r', encoding='utf-8') as f:
        en_lines = [line.strip().split() for line in f if line.strip()]
    assert len(zh_lines) == len(en_lines)
    return zh_lines, en_lines


def load_dev_data(path):
    lines = open(path, 'r', encoding='utf-8').read().strip().split('\n')
    zh, en = [], []
    i = 0
    while i < len(lines):
        if lines[i].strip():
            zh.append(lines[i].strip().split())
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines):
                en.append(lines[i].strip().split())
                i += 1
        else:
            i += 1
    return zh, en


def load_test_data(test_path, ref_path):
    with open(test_path, 'r', encoding='utf-8') as f:
        test_zh = [line.strip().split() for line in f if line.strip()]
    # Reference file format: chinese line, blank line, english line (repeating)
    with open(ref_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    refs = []
    i = 0
    while i < len(lines):
        if lines[i].strip():
            # Skip Chinese line
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines):
                refs.append([lines[i].strip()])
                i += 1
        else:
            i += 1
    assert len(refs) == len(test_zh), f'Mismatch: {len(refs)} refs vs {len(test_zh)} test'
    return test_zh, refs


# ── Dataset ────────────────────────────────────────────────────

class TranslationDataset(Dataset):
    def __init__(self, src_sents, tgt_sents, src_word2idx, tgt_word2idx, max_len=60):
        self.pairs = []
        for src, tgt in zip(src_sents, tgt_sents):
            if len(src) > max_len or len(tgt) > max_len:
                continue
            src_ids = [BOS_IDX] + [src_word2idx.get(w, UNK_IDX) for w in src] + [EOS_IDX]
            tgt_ids = [BOS_IDX] + [tgt_word2idx.get(w, UNK_IDX) for w in tgt] + [EOS_IDX]
            self.pairs.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) for t in tgt_batch]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)
    src_padded = [s + [PAD_IDX] * (max_src - len(s)) for s in src_batch]
    tgt_padded = [t + [PAD_IDX] * (max_tgt - len(t)) for t in tgt_batch]
    return torch.tensor(src_padded, dtype=torch.long), torch.tensor(tgt_padded, dtype=torch.long)


# ── Positional Encoding ────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ── Transformer Model ──────────────────────────────────────────

class TransformerMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        out = self.transformer(src_emb, tgt_emb,
                               src_mask=src_mask, tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        return self.fc_out(out)

    def encode(self, src, src_key_padding_mask=None):
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None,
               memory_key_padding_mask=None):
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask)


# ── Label Smoothing ────────────────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, padding_idx=PAD_IDX, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = (target == self.padding_idx)
        true_dist[mask] = 0
        loss = self.criterion(pred, true_dist)
        return loss / (~mask).sum()


# ── Scheduler ──────────────────────────────────────────────────

class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self):
        self._step += 1
        lr = self.d_model ** (-0.5) * min(self._step ** (-0.5),
                                           self._step * self.warmup_steps ** (-1.5))
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        return lr

    def get_lr(self):
        if self._step == 0:
            return 0
        return self.d_model ** (-0.5) * min(self._step ** (-0.5),
                                             self._step * self.warmup_steps ** (-1.5))


# ── BLEU ───────────────────────────────────────────────────────

def compute_bleu(hypotheses, references, max_n=4):
    """Compute corpus BLEU-4 score."""
    from collections import Counter
    import math

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for hyp, refs in zip(hypotheses, references):
        hyp_tokens = hyp.split()
        hyp_len += len(hyp_tokens)
        ref_lens = [len(r.split()) for r in refs]
        closest_ref_len = min(ref_lens, key=lambda r: (abs(r - len(hyp_tokens)), r))
        ref_len += closest_ref_len

        for n in range(1, max_n + 1):
            hyp_ngrams = Counter()
            for i in range(len(hyp_tokens) - n + 1):
                ngram = tuple(hyp_tokens[i:i+n])
                hyp_ngrams[ngram] += 1

            max_ref_ngrams = Counter()
            for ref in refs:
                ref_tokens = ref.split()
                ref_ngrams = Counter()
                for i in range(len(ref_tokens) - n + 1):
                    ngram = tuple(ref_tokens[i:i+n])
                    ref_ngrams[ngram] += 1
                for ngram, count in ref_ngrams.items():
                    max_ref_ngrams[ngram] = max(max_ref_ngrams[ngram], count)

            clipped = sum(min(count, max_ref_ngrams[ngram])
                          for ngram, count in hyp_ngrams.items())
            clipped_counts[n-1] += clipped
            total_counts[n-1] += max(len(hyp_tokens) - n + 1, 0)

    precisions = []
    for n in range(max_n):
        if total_counts[n] == 0:
            precisions.append(0)
        else:
            precisions.append(clipped_counts[n] / total_counts[n])

    if min(precisions) == 0:
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return bp * math.exp(log_avg) * 100


# ── Inference ──────────────────────────────────────────────────

def translate_sentence(model, src_ids, tgt_word2idx, tgt_vocab, device, max_len=80, beam_size=1):
    model.eval()
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad_mask = (src == PAD_IDX)
    memory = model.encode(src, src_key_padding_mask=src_pad_mask)

    if beam_size == 1:
        # Greedy
        ys = torch.tensor([[BOS_IDX]], dtype=torch.long, device=device)
        for _ in range(max_len):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1), device=device)
            out = model.decode(ys, memory, tgt_mask=tgt_mask,
                               memory_key_padding_mask=src_pad_mask)
            logits = model.fc_out(out[:, -1, :])
            next_token = logits.argmax(dim=-1).item()
            if next_token == EOS_IDX:
                break
            ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)
        return [tgt_vocab[idx] for idx in ys[0, 1:].tolist()]
    else:
        # Beam search
        beams = [(torch.tensor([[BOS_IDX]], dtype=torch.long, device=device), 0.0)]
        completed = []
        for _ in range(max_len):
            candidates = []
            for seq, score in beams:
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq.size(1), device=device)
                out = model.decode(seq, memory, tgt_mask=tgt_mask,
                                   memory_key_padding_mask=src_pad_mask)
                logits = model.fc_out(out[:, -1, :])
                log_probs = torch.log_softmax(logits, dim=-1)
                topk = log_probs.topk(beam_size, dim=-1)
                for i in range(beam_size):
                    token = topk.indices[0, i].item()
                    new_score = score + topk.values[0, i].item()
                    new_seq = torch.cat([seq, torch.tensor([[token]], device=device)], dim=1)
                    if token == EOS_IDX:
                        completed.append((new_seq, new_score / (new_seq.size(1) - 1)))
                    else:
                        candidates.append((new_seq, new_score))
            if not candidates:
                break
            candidates.sort(key=lambda x: x[1] / (x[0].size(1) - 1), reverse=True)
            beams = candidates[:beam_size]
            if len(completed) >= beam_size:
                break
        if completed:
            completed.sort(key=lambda x: x[1], reverse=True)
            best = completed[0][0]
        else:
            best = beams[0][0]
        return [tgt_vocab[idx] for idx in best[0, 1:].tolist() if idx != EOS_IDX]


def evaluate_bleu(model, src_sents, references, src_word2idx, tgt_word2idx, tgt_vocab,
                  device, beam_size=1):
    model.eval()
    hypotheses = []
    with torch.no_grad():
        for src in src_sents:
            src_ids = [BOS_IDX] + [src_word2idx.get(w, UNK_IDX) for w in src] + [EOS_IDX]
            tokens = translate_sentence(model, src_ids, tgt_word2idx, tgt_vocab, device,
                                        beam_size=beam_size)
            hypotheses.append(' '.join(tokens))
    bleu = compute_bleu(hypotheses, references)
    return bleu, hypotheses


# ── Training ───────────────────────────────────────────────────

def train_model(config, save_prefix='transformer'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load data
    zh_train, en_train = load_parallel_data(
        os.path.join(DATA_DIR, 'chinese.txt'),
        os.path.join(DATA_DIR, 'english.txt'))
    dev_zh, dev_en = load_dev_data(os.path.join(DATA_DIR, 'Niu.dev.txt'))
    test_zh, test_refs = load_test_data(
        os.path.join(DATA_DIR, 'Niu.test.txt'),
        os.path.join(DATA_DIR, 'Niu.test.reference'))

    print(f'Train: {len(zh_train)} pairs, Dev: {len(dev_zh)} pairs, Test: {len(test_zh)} sentences')

    # Build vocabularies
    src_vocab, src_word2idx = build_vocab(zh_train, max_size=config['src_vocab_size'], min_freq=2)
    tgt_vocab, tgt_word2idx = build_vocab(en_train, max_size=config['tgt_vocab_size'], min_freq=2)
    print(f'Src vocab: {len(src_vocab)}, Tgt vocab: {len(tgt_vocab)}')

    # Dataset
    train_dataset = TranslationDataset(zh_train, en_train, src_word2idx, tgt_word2idx,
                                       max_len=config['max_len'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    print(f'Train pairs (filtered): {len(train_dataset)}')

    # Model
    model = TransformerMT(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_layers'],
        num_decoder_layers=config['num_layers'],
        dim_feedforward=config['dim_ff'],
        dropout=config['dropout']
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {param_count:,}')

    optimizer = optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, config['d_model'], warmup_steps=config['warmup_steps'])
    criterion = LabelSmoothingLoss(len(tgt_vocab), padding_idx=PAD_IDX, smoothing=0.1)

    history = {'train_loss': [], 'dev_bleu': [], 'lr': [], 'epoch_time': []}
    best_bleu = 0.0

    for epoch in range(config['epochs']):
        model.train()
        total_loss, n_batches, t0 = 0.0, 0, time.time()

        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_input.size(1), device=device)
            src_pad_mask = (src == PAD_IDX)
            tgt_pad_mask = (tgt_input == PAD_IDX)

            output = model(src, tgt_input, tgt_mask=tgt_mask,
                           src_key_padding_mask=src_pad_mask,
                           tgt_key_padding_mask=tgt_pad_mask)

            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        elapsed = time.time() - t0
        cur_lr = scheduler.get_lr()

        # Evaluate dev BLEU every epoch (400 sentences greedy is fast enough)
        dev_refs = [[' '.join(e)] for e in dev_en]
        dev_bleu, _ = evaluate_bleu(model, dev_zh, dev_refs, src_word2idx,
                                     tgt_word2idx, tgt_vocab, device, beam_size=1)

        history['train_loss'].append(avg_loss)
        history['dev_bleu'].append(dev_bleu)
        history['lr'].append(cur_lr)
        history['epoch_time'].append(elapsed)

        print(f'Epoch {epoch+1}/{config["epochs"]}  loss={avg_loss:.4f}  '
              f'dev_BLEU={dev_bleu:.2f}  lr={cur_lr:.6f}  time={elapsed:.1f}s')

        if dev_bleu > best_bleu:
            best_bleu = dev_bleu
            torch.save(model.state_dict(), os.path.join(BASE, f'{save_prefix}_best.pth'))
            print(f'  -> New best BLEU: {best_bleu:.2f}')

    # Always save last epoch
    torch.save(model.state_dict(), os.path.join(BASE, f'{save_prefix}_last.pth'))

    # Final test evaluation with beam search
    # Compare best (dev-selected) and last checkpoint
    print('\nFinal evaluation on test set...')
    dev_refs_full = [[' '.join(e)] for e in dev_en]
    best_ckpt_bleu = 0
    for ckpt in ['best', 'last']:
        p = os.path.join(BASE, f'{save_prefix}_{ckpt}.pth')
        if os.path.exists(p):
            model.load_state_dict(torch.load(p, weights_only=True))
            b, _ = evaluate_bleu(model, dev_zh, dev_refs_full, src_word2idx,
                                  tgt_word2idx, tgt_vocab, device, beam_size=1)
            print(f'  {ckpt} ckpt - Full Dev BLEU: {b:.2f}')
            if b > best_ckpt_bleu:
                best_ckpt_bleu = b
                best_ckpt = ckpt

    # Use the better checkpoint
    model.load_state_dict(torch.load(os.path.join(BASE, f'{save_prefix}_{best_ckpt}.pth'), weights_only=True))
    print(f'Using {best_ckpt} checkpoint for final eval')

    # Test with greedy (fast)
    test_bleu, test_hyps = evaluate_bleu(model, test_zh, test_refs, src_word2idx,
                                          tgt_word2idx, tgt_vocab, device, beam_size=1)
    print(f'Test BLEU-4 (greedy): {test_bleu:.2f}')

    # Beam search comparison on subset
    beam_bleus = {}
    test_subset = test_zh[:200]
    refs_subset = test_refs[:200]
    for bw in [1, 3, 5, 8]:
        b, _ = evaluate_bleu(model, test_subset, refs_subset, src_word2idx,
                              tgt_word2idx, tgt_vocab, device, beam_size=bw)
        beam_bleus[bw] = b
        print(f'  beam={bw}: BLEU={b:.2f} (on 200 samples)')
    history['beam_bleus'] = beam_bleus
    history['test_bleu'] = test_bleu

    # Save sample translations
    samples = []
    for i in range(min(10, len(test_zh))):
        src_ids = [BOS_IDX] + [src_word2idx.get(w, UNK_IDX) for w in test_zh[i]] + [EOS_IDX]
        tokens = translate_sentence(model, src_ids, tgt_word2idx, tgt_vocab, device, beam_size=5)
        samples.append({
            'source': ' '.join(test_zh[i]),
            'hypothesis': ' '.join(tokens),
            'references': test_refs[i]
        })
    history['samples'] = samples

    # Save
    with open(os.path.join(BASE, f'{save_prefix}_history.json'), 'w') as f:
        json.dump(history, f, ensure_ascii=False)

    # Save vocabs for visualization
    vocab_info = {
        'src_vocab_size': len(src_vocab),
        'tgt_vocab_size': len(tgt_vocab),
        'src_vocab': src_vocab[:100],
        'tgt_vocab': tgt_vocab[:100],
    }
    with open(os.path.join(BASE, f'{save_prefix}_vocab.json'), 'w') as f:
        json.dump(vocab_info, f, ensure_ascii=False)

    return history


# ── Main ───────────────────────────────────────────────────────

if __name__ == '__main__':
    configs = {
        'base': {
            'src_vocab_size': 30000,
            'tgt_vocab_size': 30000,
            'max_len': 60,
            'batch_size': 64,
            'd_model': 512,
            'nhead': 8,
            'num_layers': 6,
            'dim_ff': 2048,
            'dropout': 0.1,
            'warmup_steps': 4000,
            'epochs': 30,
        },
        'small': {
            'src_vocab_size': 30000,
            'tgt_vocab_size': 30000,
            'max_len': 60,
            'batch_size': 64,
            'd_model': 256,
            'nhead': 8,
            'num_layers': 3,
            'dim_ff': 1024,
            'dropout': 0.1,
            'warmup_steps': 2000,
            'epochs': 30,
        },
        'large': {
            'src_vocab_size': 30000,
            'tgt_vocab_size': 30000,
            'max_len': 60,
            'batch_size': 48,
            'd_model': 512,
            'nhead': 8,
            'num_layers': 8,
            'dim_ff': 2048,
            'dropout': 0.15,
            'warmup_steps': 4000,
            'epochs': 30,
        },
    }

    all_history = {}
    for name, cfg in configs.items():
        print(f'\n{"="*60}\nTraining {name} model\n{"="*60}')
        h = train_model(cfg, save_prefix=name)
        all_history[name] = h

    with open(os.path.join(BASE, 'all_history.json'), 'w') as f:
        json.dump(all_history, f, ensure_ascii=False)

    print('\nDone! All models trained.')
