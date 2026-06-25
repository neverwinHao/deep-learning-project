"""Visualization for Experiment 6: SegNet Semantic Segmentation"""
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

# Fig 1: Training and validation loss
print('Fig 1: Loss curves...')
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, hist['train_loss'], color=C1, label='Train Loss')
ax.plot(epochs, hist['val_loss'], color=C2, label='Val Loss')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Training & Validation Loss')
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig01_loss.png')); plt.close()

# Fig 2: Pixel Accuracy
print('Fig 2: Pixel accuracy...')
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, hist['val_pa'], color=C1, marker='o', markersize=3)
ax.set_xlabel('Epoch'); ax.set_ylabel('Pixel Accuracy'); ax.set_title('Validation Pixel Accuracy')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig02_pixel_accuracy.png')); plt.close()

# Fig 3: Mean IoU
print('Fig 3: mIoU...')
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, hist['val_miou'], color=C3, marker='o', markersize=3)
ax.set_xlabel('Epoch'); ax.set_ylabel('mIoU'); ax.set_title('Validation Mean IoU')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig03_miou.png')); plt.close()

# Fig 4: mPA
print('Fig 4: mPA...')
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, hist['val_mpa'], color=C2, marker='o', markersize=3)
ax.set_xlabel('Epoch'); ax.set_ylabel('Mean Pixel Accuracy'); ax.set_title('Validation mPA')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig04_mpa.png')); plt.close()

# Fig 5: All metrics combined
print('Fig 5: All metrics...')
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, hist['val_pa'], color=C1, label='PA')
ax.plot(epochs, hist['val_mpa'], color=C2, label='mPA')
ax.plot(epochs, hist['val_miou'], color=C3, label='mIoU')
ax.set_xlabel('Epoch'); ax.set_ylabel('Score'); ax.set_title('Segmentation Metrics Over Training')
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig05_all_metrics.png')); plt.close()

# Fig 6: Learning rate
print('Fig 6: LR...')
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, hist['lr'], color=C1)
ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate'); ax.set_title('Learning Rate Schedule')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig06_lr.png')); plt.close()

# Fig 7: Training time
print('Fig 7: Epoch time...')
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, hist['epoch_time'], color=C1, marker='s', markersize=3)
ax.set_xlabel('Epoch'); ax.set_ylabel('Time (s)'); ax.set_title('Training Time per Epoch')
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig07_epoch_time.png')); plt.close()

# Fig 8: Final test metrics bar chart
print('Fig 8: Test results...')
fig, ax = plt.subplots(figsize=(8, 5))
test_metrics = ['PA', 'mPA', 'mIoU']
test_values = [hist.get('test_pa', 0), hist.get('test_mpa', 0), hist.get('test_miou', 0)]
bars = ax.bar(test_metrics, test_values, color=[C1, C2, C3])
for bar, v in zip(bars, test_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{v:.4f}', ha='center', fontsize=14)
ax.set_ylabel('Score'); ax.set_title('Test Set Segmentation Metrics')
ax.set_ylim(0, 1.0)
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig08_test_metrics.png')); plt.close()

print(f'\nAll figures saved to {FIGS}')
