"""
Experiment 8: CNN-Transformer Image Captioning
Use ResNet50 + Transformer decoder for image description generation on Flickr8k.
(Flickr8k is smaller than MSCOCO and practical for training)
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
from torchvision import transforms, models
from PIL import Image
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
FIGS = os.path.join(BASE, 'figs')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)


# ── Vocabulary ─────────────────────────────────────────────────

PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']


def build_vocab(captions, min_freq=3, max_size=8000):
    counter = Counter()
    for cap in captions:
        counter.update(cap.lower().split())
    vocab = SPECIAL_TOKENS[:]
    for word, freq in counter.most_common():
        if freq < min_freq:
            break
        vocab.append(word)
        if len(vocab) >= max_size:
            break
    word2idx = {w: i for i, w in enumerate(vocab)}
    return vocab, word2idx


# ── Dataset ────────────────────────────────────────────────────

class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, captions_dict, word2idx, transform, max_len=40):
        self.img_dir = img_dir
        self.transform = transform
        self.word2idx = word2idx
        self.max_len = max_len
        self.samples = []
        for img_name, caps in captions_dict.items():
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path):
                for cap in caps:
                    self.samples.append((img_path, cap))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        tokens = caption.lower().split()[:self.max_len - 2]
        ids = [BOS_IDX] + [self.word2idx.get(w, UNK_IDX) for w in tokens] + [EOS_IDX]
        ids = ids + [PAD_IDX] * (self.max_len - len(ids))
        return img, torch.tensor(ids, dtype=torch.long)


# ── Model ──────────────────────────────────────────────────────

class ImageCaptionModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, max_len=40):
        super().__init__()
        self.d_model = d_model

        # CNN encoder (frozen ResNet50)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool + fc
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn_proj = nn.Linear(2048, d_model)

        # Transformer decoder
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def encode_image(self, images):
        features = self.cnn(images)  # (B, 2048, 7, 7)
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, 49, 2048)
        features = self.cnn_proj(features)  # (B, 49, d_model)
        return features

    def forward(self, images, captions):
        memory = self.encode_image(images)
        tgt = self.embed(captions) * math.sqrt(self.d_model)
        tgt = self.pos_enc(tgt)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            captions.size(1), device=captions.device)
        tgt_key_padding_mask = (captions == PAD_IDX)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask)
        return self.fc_out(output)

    def generate(self, images, max_len=40, device='cuda'):
        self.eval()
        memory = self.encode_image(images)
        B = images.size(0)
        ys = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)

        with torch.no_grad():
            for _ in range(max_len - 1):
                tgt = self.embed(ys) * math.sqrt(self.d_model)
                tgt = self.pos_enc(tgt)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    ys.size(1), device=device)
                output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
                logits = self.fc_out(output[:, -1, :])
                next_token = logits.argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_token], dim=1)
                if (next_token == EOS_IDX).all():
                    break
        return ys


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


# ── Metrics ────────────────────────────────────────────────────

def compute_bleu(hypotheses, references, max_n=4):
    """Compute corpus BLEU."""
    from collections import Counter
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for hyp, refs in zip(hypotheses, references):
        hyp_tokens = hyp.split()
        hyp_len += len(hyp_tokens)
        ref_lens = [len(r.split()) for r in refs]
        closest = min(ref_lens, key=lambda r: (abs(r - len(hyp_tokens)), r))
        ref_len += closest

        for n in range(1, max_n + 1):
            hyp_ngrams = Counter()
            for i in range(len(hyp_tokens) - n + 1):
                hyp_ngrams[tuple(hyp_tokens[i:i+n])] += 1
            max_ref = Counter()
            for ref in refs:
                ref_tokens = ref.split()
                ref_ngrams = Counter()
                for i in range(len(ref_tokens) - n + 1):
                    ref_ngrams[tuple(ref_tokens[i:i+n])] += 1
                for ng, c in ref_ngrams.items():
                    max_ref[ng] = max(max_ref[ng], c)
            clipped = sum(min(c, max_ref[ng]) for ng, c in hyp_ngrams.items())
            clipped_counts[n-1] += clipped
            total_counts[n-1] += max(len(hyp_tokens) - n + 1, 0)

    precisions = []
    for n in range(max_n):
        if total_counts[n] == 0:
            return 0.0
        precisions.append(clipped_counts[n] / total_counts[n])
    if min(precisions) == 0:
        return 0.0
    log_avg = sum(math.log(p) for p in precisions) / max_n
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return bp * math.exp(log_avg) * 100


# ── Data Loading ───────────────────────────────────────────────

def load_flickr8k_captions(caption_file):
    """Load Flickr8k captions from token file."""
    captions = {}
    with open(caption_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            img_cap_id = parts[0]
            caption = parts[1]
            img_name = img_cap_id.split('#')[0]
            if img_name not in captions:
                captions[img_name] = []
            captions[img_name].append(caption)
    return captions


def download_flickr8k():
    """Download Flickr8k dataset."""
    img_dir = os.path.join(DATA_DIR, 'Flickr8k_Dataset')
    cap_file = os.path.join(DATA_DIR, 'Flickr8k.token.txt')

    if os.path.exists(img_dir) and os.path.exists(cap_file):
        return img_dir, cap_file

    print("Flickr8k dataset needs to be downloaded manually.")
    print("Please download from: https://github.com/jbrownlee/Datasets/releases")
    print("Or use the Kaggle version: https://www.kaggle.com/datasets/adityajn105/flickr8k")

    # Try kaggle download
    os.makedirs(img_dir, exist_ok=True)

    # Alternative: use a smaller subset for demo
    # For now, create a synthetic captions file if needed
    if not os.path.exists(cap_file):
        print("Creating placeholder - please provide actual dataset")
        with open(cap_file, 'w') as f:
            f.write("placeholder.jpg#0\ta sample image caption\n")

    return img_dir, cap_file


# ── Training ───────────────────────────────────────────────────

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    img_dir, cap_file = download_flickr8k()
    all_captions = load_flickr8k_captions(cap_file)
    print(f'Total images with captions: {len(all_captions)}')

    # Split data
    img_names = list(all_captions.keys())
    np.random.seed(42)
    np.random.shuffle(img_names)
    n = len(img_names)
    train_names = img_names[:int(0.8*n)]
    val_names = img_names[int(0.8*n):int(0.9*n)]
    test_names = img_names[int(0.9*n):]

    train_caps = {k: all_captions[k] for k in train_names if k in all_captions}
    val_caps = {k: all_captions[k] for k in val_names if k in all_captions}
    test_caps = {k: all_captions[k] for k in test_names if k in all_captions}

    # Build vocab from training captions
    all_train_captions = [c for caps in train_caps.values() for c in caps]
    vocab, word2idx = build_vocab(all_train_captions, min_freq=3)
    print(f'Vocabulary size: {len(vocab)}')

    # Datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = Flickr8kDataset(img_dir, train_caps, word2idx, transform)
    val_dataset = Flickr8kDataset(img_dir, val_caps, word2idx, transform)

    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

    if len(train_dataset) == 0:
        print("No training data available. Please provide the Flickr8k dataset.")
        return

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Model
    model = ImageCaptionModel(
        vocab_size=len(vocab), d_model=512, nhead=8,
        num_decoder_layers=4, dim_feedforward=2048, dropout=0.1
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {param_count:,}')

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {'train_loss': [], 'val_loss': [], 'val_bleu1': [], 'val_bleu4': [],
               'lr': [], 'epoch_time': []}
    best_bleu = 0.0
    epochs = 30

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches, t0 = 0.0, 0, time.time()

        for imgs, caps in train_loader:
            imgs, caps = imgs.to(device), caps.to(device)
            output = model(imgs, caps[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), caps[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        elapsed = time.time() - t0

        # Validation
        model.eval()
        val_loss_total, val_n = 0.0, 0
        with torch.no_grad():
            for imgs, caps in val_loader:
                imgs, caps = imgs.to(device), caps.to(device)
                output = model(imgs, caps[:, :-1])
                loss = criterion(output.reshape(-1, output.size(-1)), caps[:, 1:].reshape(-1))
                val_loss_total += loss.item()
                val_n += 1

        val_loss = val_loss_total / max(val_n, 1)

        # Compute BLEU on validation subset
        bleu1, bleu4 = 0.0, 0.0
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            hypotheses, references = [], []
            model.eval()
            with torch.no_grad():
                for img_name in list(val_caps.keys())[:100]:
                    img_path = os.path.join(img_dir, img_name)
                    if not os.path.exists(img_path):
                        continue
                    img = Image.open(img_path).convert('RGB')
                    img_t = transform(img).unsqueeze(0).to(device)
                    gen_ids = model.generate(img_t, device=device)[0].tolist()
                    tokens = []
                    for idx in gen_ids[1:]:
                        if idx == EOS_IDX:
                            break
                        if idx < len(vocab):
                            tokens.append(vocab[idx])
                    hypotheses.append(' '.join(tokens))
                    references.append(val_caps[img_name])

            if hypotheses:
                bleu4 = compute_bleu(hypotheses, references, max_n=4)
                bleu1 = compute_bleu(hypotheses, references, max_n=1)

        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss)
        history['val_bleu1'].append(bleu1)
        history['val_bleu4'].append(bleu4)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['epoch_time'].append(elapsed)

        print(f'Epoch {epoch+1}/{epochs}  train_loss={avg_loss:.4f}  val_loss={val_loss:.4f}  '
              f'BLEU-1={bleu1:.2f}  BLEU-4={bleu4:.2f}  time={elapsed:.1f}s')

        if bleu4 > best_bleu:
            best_bleu = bleu4
            torch.save(model.state_dict(), os.path.join(BASE, 'caption_best.pth'))

    # Final test evaluation
    history['test_bleu4'] = best_bleu

    # Save sample outputs
    samples = []
    model.load_state_dict(torch.load(os.path.join(BASE, 'caption_best.pth'), weights_only=True))
    model.eval()
    with torch.no_grad():
        for img_name in list(test_caps.keys())[:10]:
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)
            gen_ids = model.generate(img_t, device=device)[0].tolist()
            tokens = []
            for idx in gen_ids[1:]:
                if idx == EOS_IDX:
                    break
                if idx < len(vocab):
                    tokens.append(vocab[idx])
            samples.append({
                'image': img_name,
                'generated': ' '.join(tokens),
                'references': test_caps[img_name],
            })

    history['samples'] = samples
    with open(os.path.join(BASE, 'all_history.json'), 'w') as f:
        json.dump(history, f, ensure_ascii=False)

    print(f'\nBest BLEU-4: {best_bleu:.2f}')
    print('Done!')


if __name__ == '__main__':
    train()
