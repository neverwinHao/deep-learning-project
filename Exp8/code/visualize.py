"""Visualization for Experiment 8: CNN-Transformer Image Captioning"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

fm.fontManager.addfont('/home/v-haoliu3/.local/share/fonts/NotoSansSC.ttf')
fm.fontManager.addfont('/home/v-haoliu3/.local/share/fonts/NotoSerifSC.otf')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Noto Serif CJK SC', 'DejaVu Serif', 'Times New Roman', 'serif'],
    'mathtext.fontset': 'dejavuserif', 'axes.unicode_minus': False,
    'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 16, 'axes.titlesize': 20, 'axes.titleweight': 'bold',
    'axes.labelsize': 17, 'axes.labelweight': 'bold', 'axes.linewidth': 1.5,
    'axes.grid': True, 'grid.color': '#E0E0E0', 'grid.linewidth': 0.5,
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'legend.fontsize': 14, 'legend.framealpha': 0.92,
    'lines.linewidth': 2.5, 'lines.markersize': 6,
})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(BASE, 'figs')
os.makedirs(FIGS, exist_ok=True)

with open(os.path.join(BASE, 'all_history.json')) as f:
    hist = json.load(f)

C1, C2, C3 = '#3274A1', '#C03D3E', '#3A923A'
epochs = list(range(1, len(hist['train_loss']) + 1))

# Fig 1: Training loss
print('Fig 1: Training loss...')
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, hist['train_loss'], color=C1, label='Train Loss')
ax.plot(epochs, hist['val_loss'], color=C2, label='Val Loss')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Training & Validation Loss')
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig01_loss.png')); plt.close()

# Fig 2: BLEU scores
print('Fig 2: BLEU scores...')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(epochs, hist['val_bleu1'], color=C1, marker='o', markersize=4)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('BLEU-1'); ax1.set_title('Validation BLEU-1')
ax2.plot(epochs, hist['val_bleu4'], color=C2, marker='o', markersize=4)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('BLEU-4'); ax2.set_title('Validation BLEU-4')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig02_bleu.png')); plt.close()

# Fig 3: Learning rate
print('Fig 3: LR...')
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, hist['lr'], color=C3)
ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate'); ax.set_title('Learning Rate Schedule')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig03_lr.png')); plt.close()

# Fig 4: Epoch time
print('Fig 4: Epoch time...')
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, hist['epoch_time'], color=C1, marker='s', markersize=3)
ax.set_xlabel('Epoch'); ax.set_ylabel('Time (s)'); ax.set_title('Training Time per Epoch')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig04_epoch_time.png')); plt.close()

# Fig 5: Loss vs BLEU
print('Fig 5: Loss vs BLEU...')
fig, ax = plt.subplots(figsize=(8, 6))
bleu_nonzero = [(l, b) for l, b in zip(hist['val_loss'], hist['val_bleu4']) if b > 0]
if bleu_nonzero:
    losses, bleus = zip(*bleu_nonzero)
    ax.scatter(losses, bleus, color=C1, s=80)
ax.set_xlabel('Validation Loss'); ax.set_ylabel('BLEU-4')
ax.set_title('Loss vs BLEU-4 Correlation')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig05_loss_vs_bleu.png')); plt.close()

# Fig 6: Sample translations table
print('Fig 6: Sample captions...')
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')
samples = hist.get('samples', [])
if samples:
    table_data = []
    for i, s in enumerate(samples[:6]):
        gen_short = s['generated'][:60] + ('...' if len(s['generated']) > 60 else '')
        ref_short = s['references'][0][:60] + ('...' if len(s['references'][0]) > 60 else '')
        table_data.append([str(i+1), s['image'][:20], gen_short, ref_short])
    table = ax.table(cellText=table_data,
                     colLabels=['#', 'Image', 'Generated', 'Reference'],
                     cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#3274A1')
            cell.set_text_props(color='white', weight='bold')
ax.set_title('Sample Generated Captions', pad=20)
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig06_samples.png')); plt.close()

print(f'\nAll figures saved to {FIGS}')
