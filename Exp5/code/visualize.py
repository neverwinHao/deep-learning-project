"""Visualization for Experiment 5: YOLOv5 Object Detection"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

fm.fontManager.addfont('/home/v-haoliu3/.local/share/fonts/NotoSansSC.ttf')
fm.fontManager.addfont('/home/v-haoliu3/.local/share/fonts/NotoSerifSC.otf')

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Noto Serif CJK SC', 'DejaVu Serif', 'Times New Roman', 'serif'],
    'mathtext.fontset': 'dejavuserif',
    'axes.unicode_minus': False,
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

with open(os.path.join(BASE, 'training_history.json')) as f:
    hist = json.load(f)

C1, C2, C3 = '#3274A1', '#C03D3E', '#3A923A'
epochs = hist.get('epochs', list(range(1, 51)))

# Fig 1: Training Loss (box + cls + dfl)
print('Fig 1: Training losses...')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
for ax, key, title in [(ax1, 'train/box_loss', 'Box Loss'),
                        (ax2, 'train/cls_loss', 'Classification Loss'),
                        (ax3, 'train/dfl_loss', 'DFL Loss')]:
    if key in hist:
        ax.plot(epochs, hist[key], color=C1, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig01_train_loss.png'))
plt.close()

# Fig 2: Validation Loss
print('Fig 2: Validation losses...')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
for ax, key, title in [(ax1, 'val/box_loss', 'Val Box Loss'),
                        (ax2, 'val/cls_loss', 'Val Cls Loss'),
                        (ax3, 'val/dfl_loss', 'Val DFL Loss')]:
    if key in hist:
        ax.plot(epochs, hist[key], color=C2, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig02_val_loss.png'))
plt.close()

# Fig 3: mAP curves
print('Fig 3: mAP curves...')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
if 'metrics/mAP50(B)' in hist:
    ax1.plot(epochs, hist['metrics/mAP50(B)'], color=C1, marker='o', markersize=3)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('mAP@0.5')
ax1.set_title('mAP@0.5 Over Training')

if 'metrics/mAP50-95(B)' in hist:
    ax2.plot(epochs, hist['metrics/mAP50-95(B)'], color=C2, marker='o', markersize=3)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('mAP@0.5:0.95')
ax2.set_title('mAP@0.5:0.95 Over Training')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig03_map_curves.png'))
plt.close()

# Fig 4: Precision & Recall
print('Fig 4: Precision & Recall...')
fig, ax = plt.subplots(figsize=(10, 5))
if 'metrics/precision(B)' in hist:
    ax.plot(epochs, hist['metrics/precision(B)'], color=C1, label='Precision', marker='o', markersize=3)
if 'metrics/recall(B)' in hist:
    ax.plot(epochs, hist['metrics/recall(B)'], color=C2, label='Recall', marker='s', markersize=3)
ax.set_xlabel('Epoch')
ax.set_ylabel('Score')
ax.set_title('Precision and Recall')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig04_precision_recall.png'))
plt.close()

# Fig 5: Learning Rate
print('Fig 5: Learning rate...')
fig, ax = plt.subplots(figsize=(10, 5))
if 'lr/pg0' in hist:
    ax.plot(epochs, hist['lr/pg0'], color=C3)
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedule')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig05_lr.png'))
plt.close()

# Fig 6: Final metrics summary
print('Fig 6: Summary...')
fig, ax = plt.subplots(figsize=(8, 5))
metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
values = [hist.get('map50', 0), hist.get('map50_95', 0),
          hist.get('precision', 0), hist.get('recall', 0)]
colors_bar = [C1, C2, C3, '#E5AE38']
bars = ax.bar(metrics, [v*100 for v in values], color=colors_bar)
ax.set_ylabel('Score (%)')
ax.set_title('Final Detection Metrics')
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{v*100:.1f}%', ha='center', fontsize=12)
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig06_summary.png'))
plt.close()

# Fig 7: Loss convergence comparison
print('Fig 7: Total loss convergence...')
fig, ax = plt.subplots(figsize=(10, 5))
if all(k in hist for k in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']):
    total = [a+b+c for a,b,c in zip(hist['train/box_loss'], hist['train/cls_loss'], hist['train/dfl_loss'])]
    ax.plot(epochs, total, color=C1, label='Train Total Loss')
if all(k in hist for k in ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']):
    val_total = [a+b+c for a,b,c in zip(hist['val/box_loss'], hist['val/cls_loss'], hist['val/dfl_loss'])]
    ax.plot(epochs, val_total, color=C2, label='Val Total Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Total Loss')
ax.set_title('Total Loss Convergence')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig07_total_loss.png'))
plt.close()

print(f'\nAll figures saved to {FIGS}')
