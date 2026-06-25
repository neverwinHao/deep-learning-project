"""
Experiment 7: Neural Network Language Model (LSTM on PTB)
Train LSTM language model on Penn Treebank, target PPL < 80.
"""
import os
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import urllib.request
import tarfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
os.makedirs(DATA_DIR, exist_ok=True)


# ── Data ───────────────────────────────────────────────────────

def download_ptb():
    url = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'
    tgz_path = os.path.join(DATA_DIR, 'simple-examples.tgz')
    if not os.path.exists(os.path.join(DATA_DIR, 'ptb.train.txt')):
        if not os.path.exists(tgz_path):
            print('Downloading PTB dataset...')
            urllib.request.urlretrieve(url, tgz_path)
        print('Extracting...')
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(DATA_DIR)
        # Move PTB files to data dir
        ptb_dir = os.path.join(DATA_DIR, 'simple-examples', 'data')
        for split in ['train', 'valid', 'test']:
            src = os.path.join(ptb_dir, f'ptb.{split}.txt')
            dst = os.path.join(DATA_DIR, f'ptb.{split}.txt')
            if os.path.exists(src) and not os.path.exists(dst):
                os.rename(src, dst)
    return (os.path.join(DATA_DIR, 'ptb.train.txt'),
            os.path.join(DATA_DIR, 'ptb.valid.txt'),
            os.path.join(DATA_DIR, 'ptb.test.txt'))


def build_vocab(filepath):
    with open(filepath, 'r') as f:
        words = f.read().replace('\n', ' <eos> ').split()
    word_counts = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1
    vocab = sorted(word_counts.keys())
    word2idx = {w: i for i, w in enumerate(vocab)}
    return word2idx, vocab


def tokenize(filepath, word2idx):
    with open(filepath, 'r') as f:
        words = f.read().replace('\n', ' <eos> ').split()
    return torch.tensor([word2idx.get(w, word2idx.get('<unk>', 0)) for w in words], dtype=torch.long)


class PTBDataset(Dataset):
    def __init__(self, data, seq_len=35):
        self.seq_len = seq_len
        n_seqs = len(data) // seq_len
        self.data = data[:n_seqs * seq_len].view(n_seqs, seq_len)

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx + 1] if idx + 1 < len(self.data) else self.data[idx]


def batchify(data, batch_size):
    n_batch = len(data) // batch_size
    data = data[:n_batch * batch_size]
    return data.view(batch_size, -1).t().contiguous()


# ── Model ──────────────────────────────────────────────────────

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=650, hidden_dim=650, num_layers=2, dropout=0.5, tie_weights=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout, batch_first=False)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        if tie_weights and embed_dim == hidden_dim:
            self.decoder.weight = self.encoder.weight
        self._init_weights()

    def _init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, x, hidden):
        emb = self.drop(self.encoder(x))
        output, hidden = self.lstm(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)


# ── Training ───────────────────────────────────────────────────

def evaluate(model, data, batch_size, seq_len, criterion, device):
    model.eval()
    total_loss = 0.0
    total_len = 0
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(0, data.size(0) - 1, seq_len):
            seq_end = min(i + seq_len, data.size(0) - 1)
            x = data[i:seq_end].to(device)
            y = data[i+1:seq_end+1].to(device)
            hidden = tuple(h.detach() for h in hidden)
            output, hidden = model(x, hidden)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            total_loss += loss.item() * x.numel()
            total_len += x.numel()
    return total_loss / total_len


def train_epoch(model, data, batch_size, seq_len, criterion, optimizer, clip, device):
    model.train()
    total_loss = 0.0
    total_len = 0
    hidden = model.init_hidden(batch_size, device)
    for i in range(0, data.size(0) - 1, seq_len):
        seq_end = min(i + seq_len, data.size(0) - 1)
        x = data[i:seq_end].to(device)
        y = data[i+1:seq_end+1].to(device)
        hidden = tuple(h.detach() for h in hidden)
        optimizer.zero_grad()
        output, hidden = model(x, hidden)
        loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item() * x.numel()
        total_len += x.numel()
    return total_loss / total_len


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Data
    train_path, valid_path, test_path = download_ptb()
    word2idx, vocab = build_vocab(train_path)
    vocab_size = len(vocab)
    print(f'Vocabulary size: {vocab_size}')

    train_data = batchify(tokenize(train_path, word2idx), batch_size=20).to(device)
    valid_data = batchify(tokenize(valid_path, word2idx), batch_size=20).to(device)
    test_data = batchify(tokenize(test_path, word2idx), batch_size=20).to(device)

    print(f'Train: {train_data.size()}, Valid: {valid_data.size()}, Test: {test_data.size()}')

    # Configs to compare
    configs = {
        'medium': {'embed_dim': 650, 'hidden_dim': 650, 'num_layers': 2, 'dropout': 0.5, 'lr': 20.0, 'epochs': 40},
        'small': {'embed_dim': 200, 'hidden_dim': 200, 'num_layers': 2, 'dropout': 0.3, 'lr': 20.0, 'epochs': 30},
        'large': {'embed_dim': 1500, 'hidden_dim': 1500, 'num_layers': 2, 'dropout': 0.65, 'lr': 20.0, 'epochs': 40},
    }

    all_history = {}
    seq_len = 35
    batch_size = 20
    clip = 0.25

    for name, cfg in configs.items():
        print(f'\n{"="*60}\nTraining {name} model\n{"="*60}')
        model = LSTMLanguageModel(
            vocab_size, embed_dim=cfg['embed_dim'], hidden_dim=cfg['hidden_dim'],
            num_layers=cfg['num_layers'], dropout=cfg['dropout']
        ).to(device)

        param_count = sum(p.numel() for p in model.parameters())
        print(f'Parameters: {param_count:,}')

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'])
        best_val_ppl = float('inf')
        history = {'train_ppl': [], 'valid_ppl': [], 'lr': [], 'epoch_time': []}
        lr = cfg['lr']

        for epoch in range(cfg['epochs']):
            t0 = time.time()
            train_loss = train_epoch(model, train_data, batch_size, seq_len,
                                      criterion, optimizer, clip, device)
            val_loss = evaluate(model, valid_data, batch_size, seq_len, criterion, device)

            train_ppl = math.exp(train_loss)
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - t0

            history['train_ppl'].append(train_ppl)
            history['valid_ppl'].append(val_ppl)
            history['lr'].append(lr)
            history['epoch_time'].append(elapsed)

            print(f'Epoch {epoch+1}/{cfg["epochs"]}  train_ppl={train_ppl:.2f}  '
                  f'val_ppl={val_ppl:.2f}  lr={lr:.4f}  time={elapsed:.1f}s')

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                torch.save(model.state_dict(), os.path.join(BASE, f'{name}_best.pth'))
            else:
                lr /= 4.0
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        # Test evaluation
        model.load_state_dict(torch.load(os.path.join(BASE, f'{name}_best.pth'), weights_only=True))
        test_loss = evaluate(model, test_data, batch_size, seq_len, criterion, device)
        test_ppl = math.exp(test_loss)
        history['test_ppl'] = test_ppl
        print(f'\n{name} Test PPL: {test_ppl:.2f}')

        all_history[name] = history

    with open(os.path.join(BASE, 'all_history.json'), 'w') as f:
        json.dump(all_history, f)

    print('\nDone!')


if __name__ == '__main__':
    main()
