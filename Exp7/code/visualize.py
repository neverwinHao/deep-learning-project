"""Visualization for Experiment 7: LSTM Language Model on PTB"""
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

colors = {'small': '#C03D3E', 'medium': '#3274A1', 'large': '#3A923A'}
labels = {'small': 'Small (200d)', 'medium': 'Medium (650d)', 'large': 'Large (1500d)'}

# Fig 1: Train PPL
print('Fig 1: Train perplexity...')
fig, ax = plt.subplots(figsize=(10, 5))
for name in ['small', 'medium', 'large']:
    if name in hist:
        epochs = list(range(1, len(hist[name]['train_ppl']) + 1))
        ax.plot(epochs, hist[name]['train_ppl'], color=colors[name], label=labels[name])
ax.set_xlabel('Epoch'); ax.set_ylabel('Perplexity'); ax.set_title('Training Perplexity')
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig01_train_ppl.png')); plt.close()

# Fig 2: Valid PPL
print('Fig 2: Validation perplexity...')
fig, ax = plt.subplots(figsize=(10, 5))
for name in ['small', 'medium', 'large']:
    if name in hist:
        epochs = list(range(1, len(hist[name]['valid_ppl']) + 1))
        ax.plot(epochs, hist[name]['valid_ppl'], color=colors[name], label=labels[name])
ax.set_xlabel('Epoch'); ax.set_ylabel('Perplexity'); ax.set_title('Validation Perplexity')
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig02_valid_ppl.png')); plt.close()

# Fig 3: LR schedule
print('Fig 3: Learning rate...')
fig, ax = plt.subplots(figsize=(10, 5))
for name in ['small', 'medium', 'large']:
    if name in hist:
        epochs = list(range(1, len(hist[name]['lr']) + 1))
        ax.plot(epochs, hist[name]['lr'], color=colors[name], label=labels[name])
ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate'); ax.set_title('Learning Rate (with decay)')
ax.set_yscale('log')
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig03_lr.png')); plt.close()

# Fig 4: Final test PPL comparison
print('Fig 4: Test PPL comparison...')
fig, ax = plt.subplots(figsize=(8, 5))
model_names = [n for n in ['small', 'medium', 'large'] if n in hist]
test_ppls = [hist[n].get('test_ppl', 0) for n in model_names]
bars = ax.bar(model_names, test_ppls, color=[colors[n] for n in model_names])
for bar, v in zip(bars, test_ppls):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{v:.1f}', ha='center', fontsize=14)
ax.axhline(80, color='red', linestyle='--', linewidth=1.5, label='Target (PPL<80)')
ax.set_ylabel('Test Perplexity'); ax.set_title('Test Perplexity Comparison')
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig04_test_ppl.png')); plt.close()

# Fig 5: Training speed
print('Fig 5: Training speed...')
fig, ax = plt.subplots(figsize=(10, 5))
for name in ['small', 'medium', 'large']:
    if name in hist:
        epochs = list(range(1, len(hist[name]['epoch_time']) + 1))
        ax.plot(epochs, hist[name]['epoch_time'], color=colors[name], label=labels[name])
ax.set_xlabel('Epoch'); ax.set_ylabel('Time (s)'); ax.set_title('Training Time per Epoch')
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig05_epoch_time.png')); plt.close()

# Fig 6: Convergence (EMA smoothed)
print('Fig 6: Smoothed PPL...')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for name in ['small', 'medium', 'large']:
    if name in hist:
        ppl = hist[name]['valid_ppl']
        ema = [ppl[0]]
        for p in ppl[1:]:
            ema.append(0.3*p + 0.7*ema[-1])
        epochs = list(range(1, len(ema) + 1))
        ax1.plot(epochs, ema, color=colors[name], label=labels[name])
        # PPL improvement rate
        improvements = [0] + [ppl[i-1] - ppl[i] for i in range(1, len(ppl))]
        ax2.plot(epochs, improvements, color=colors[name], label=labels[name], alpha=0.8)

ax1.set_xlabel('Epoch'); ax1.set_ylabel('PPL (EMA)'); ax1.set_title('Smoothed Valid PPL')
ax1.legend()
ax2.set_xlabel('Epoch'); ax2.set_ylabel('PPL Reduction'); ax2.set_title('Per-Epoch PPL Improvement')
ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ax2.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig06_convergence.png')); plt.close()

# Fig 7: Model size vs test PPL
print('Fig 7: Parameter efficiency...')
fig, ax = plt.subplots(figsize=(8, 5))
param_sizes = {'small': 5.0, 'medium': 20.0, 'large': 66.0}  # approximate M params
for name in model_names:
    if 'test_ppl' in hist[name]:
        ax.scatter(param_sizes.get(name, 10), hist[name]['test_ppl'],
                   color=colors[name], s=200, zorder=5, label=labels[name])
ax.set_xlabel('Parameters (M)'); ax.set_ylabel('Test PPL')
ax.set_title('Parameter Efficiency')
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig07_param_efficiency.png')); plt.close()

print(f'\nAll figures saved to {FIGS}')
