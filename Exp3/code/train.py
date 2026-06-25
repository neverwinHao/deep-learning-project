import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import time
import math

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '实验3：数据集tang.npz')
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..')


# ── Models ──────────────────────────────────────────────────────

class PoetryModelBasic(nn.Module):
    """Baseline: single-layer LSTM."""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.embeddings(x)
        B, T = x.size()
        if hidden is None:
            h0 = x.new_zeros(self.num_layers, B, self.hidden_dim).float()
            c0 = x.new_zeros(self.num_layers, B, self.hidden_dim).float()
        else:
            h0, c0 = hidden
        out, hidden = self.lstm(embeds, (h0, c0))
        out = self.fc(out)
        return out.reshape(B * T, -1), hidden


class PoetryModelDeep(nn.Module):
    """3-layer LSTM with dropout and residual-like connection."""
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=3, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.embeddings(x)
        B, T = x.size()
        if hidden is None:
            h0 = x.new_zeros(self.num_layers, B, self.hidden_dim).float()
            c0 = x.new_zeros(self.num_layers, B, self.hidden_dim).float()
        else:
            h0, c0 = hidden
        out, hidden = self.lstm(embeds, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out)
        return out.reshape(B * T, -1), hidden


class PoetryModelGRU(nn.Module):
    """2-layer GRU for comparison."""
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.embeddings(x)
        B, T = x.size()
        if hidden is None:
            h0 = x.new_zeros(self.num_layers, B, self.hidden_dim).float()
        else:
            h0 = hidden
        out, hidden = self.gru(embeds, h0)
        out = self.dropout(out)
        out = self.fc(out)
        return out.reshape(B * T, -1), hidden


# ── Data ────────────────────────────────────────────────────────

def load_data(batch_size=64):
    d = np.load(DATA_PATH, allow_pickle=True)
    data = torch.from_numpy(d['data']).long()
    ix2word = d['ix2word'].item()
    word2ix = d['word2ix'].item()
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return loader, ix2word, word2ix


# ── Generation ──────────────────────────────────────────────────

def generate(model, start_words, ix2word, word2ix, device, max_len=125, temperature=1.0):
    model.eval()
    results = list(start_words)
    start_len = len(start_words)
    inp = torch.tensor([[word2ix['<START>']]]).long().to(device)
    hidden = None
    with torch.no_grad():
        for i in range(max_len):
            output, hidden = model(inp, hidden)
            if i < start_len:
                w = results[i]
                inp = torch.tensor([[word2ix.get(w, word2ix.get('<START>'))]]).long().to(device)
            else:
                logits = output[0] / temperature
                probs = torch.softmax(logits, dim=0)
                top_index = torch.multinomial(probs, 1).item()
                w = ix2word[top_index]
                results.append(w)
                inp = torch.tensor([[top_index]]).long().to(device)
            if w == '<EOP>':
                results.pop()
                break
    return ''.join(results)


def generate_acrostic(model, head_chars, ix2word, word2ix, device, max_len=125, temperature=0.8):
    """Generate acrostic poem (藏头诗)."""
    model.eval()
    results = []
    inp = torch.tensor([[word2ix['<START>']]]).long().to(device)
    hidden = None
    head_idx = 0
    with torch.no_grad():
        for i in range(max_len):
            output, hidden = model(inp, hidden)
            if head_idx < len(head_chars) and (i == 0 or (len(results) > 0 and results[-1] in '。，！？')):
                w = head_chars[head_idx]
                head_idx += 1
            else:
                logits = output[0] / temperature
                probs = torch.softmax(logits, dim=0)
                top_index = torch.multinomial(probs, 1).item()
                w = ix2word[top_index]
            results.append(w)
            inp = torch.tensor([[word2ix.get(w, 0)]]).long().to(device)
            if w == '<EOP>':
                results.pop()
                break
            if head_idx >= len(head_chars) and w in '。':
                break
    return ''.join(results)


# ── Training ────────────────────────────────────────────────────

def train_model(model_name, model, loader, word2ix, ix2word, device,
                epochs=30, lr=1e-3, save_prefix=''):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    pad_idx = word2ix['</s>']

    history = {'train_loss': [], 'perplexity': [], 'lr': [], 'epoch_time': []}

    for epoch in range(epochs):
        model.train()
        total_loss, total_tokens, t0 = 0.0, 0, time.time()

        for batch in loader:
            x = batch[0].to(device)
            inp = x[:, :-1]
            tgt = x[:, 1:]

            output, _ = model(inp)
            loss = criterion(output, tgt.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            mask = (tgt.reshape(-1) != pad_idx)
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()

        scheduler.step()
        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))
        elapsed = time.time() - t0

        history['train_loss'].append(avg_loss)
        history['perplexity'].append(ppl)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['epoch_time'].append(elapsed)

        sample = generate(model, '湖光秋月两相和', ix2word, word2ix, device)
        print(f'[{model_name}] Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  '
              f'ppl={ppl:.2f}  time={elapsed:.1f}s')
        print(f'  Sample: {sample}')

    # Save
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'{save_prefix}model.pth'))
    with open(os.path.join(SAVE_DIR, f'{save_prefix}history.json'), 'w') as f:
        json.dump(history, f)

    return history


# ── Main ────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    loader, ix2word, word2ix = load_data(batch_size=64)
    vocab_size = len(word2ix)
    print(f'Vocab size: {vocab_size}, Poems: {len(loader.dataset)}')

    models_cfg = {
        'basic': (PoetryModelBasic, dict(vocab_size=vocab_size, embed_dim=128, hidden_dim=256)),
        'deep':  (PoetryModelDeep, dict(vocab_size=vocab_size, embed_dim=256, hidden_dim=512, num_layers=3, dropout=0.3)),
        'gru':   (PoetryModelGRU, dict(vocab_size=vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.2)),
    }

    all_history = {}
    for name, (cls, kwargs) in models_cfg.items():
        print(f'\n{"="*60}\nTraining {name}\n{"="*60}')
        model = cls(**kwargs)
        param_count = sum(p.numel() for p in model.parameters())
        print(f'Parameters: {param_count:,}')
        h = train_model(name, model, loader, word2ix, ix2word, device,
                        epochs=30, lr=1e-3, save_prefix=f'{name}_')
        all_history[name] = h

        # Generate samples with the final model
        model.to(device)
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'{name}_model.pth'), weights_only=True))
        print(f'\n--- {name} Final Samples ---')
        prompts = ['湖光秋月两相和', '大漠沙如雪', '床前明月光', '春风又绿江南岸']
        for p in prompts:
            poem = generate(model, p, ix2word, word2ix, device, temperature=0.8)
            print(f'  [{p}] → {poem}')

        acrostic = generate_acrostic(model, '人工智能', ix2word, word2ix, device)
        print(f'  [藏头诗:人工智能] → {acrostic}')

    with open(os.path.join(SAVE_DIR, 'all_history.json'), 'w') as f:
        json.dump(all_history, f)

    print('\nDone!')
