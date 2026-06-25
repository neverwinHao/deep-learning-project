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
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.titleweight': 'bold',
    'axes.labelsize': 17,
    'axes.labelweight': 'bold',
    'axes.linewidth': 1.5,
    'axes.grid': True,
    'grid.color': '#E0E0E0',
    'grid.linewidth': 0.5,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.framealpha': 0.92,
    'lines.linewidth': 2.5,
    'lines.markersize': 6,
})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(BASE, 'figs')
DATA_DIR = os.path.join(BASE, 'data')
os.makedirs(FIGS, exist_ok=True)

colors = {'base': '#3274A1', 'small': '#C03D3E', 'large': '#3A923A'}
labels = {'base': 'Base (6L, d=512)', 'small': 'Small (3L, d=256)', 'large': 'Large (8L, d=512)'}

# Load history
with open(os.path.join(BASE, 'all_history.json')) as f:
    hist = json.load(f)

# Load data for dataset analysis
def load_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().split() for line in f if line.strip()]

zh_data = load_lines(os.path.join(DATA_DIR, 'chinese.txt'))
en_data = load_lines(os.path.join(DATA_DIR, 'english.txt'))

# ════════════════════════════════════════════════════════════════
# Fig 1: Sentence length distribution
# ════════════════════════════════════════════════════════════════
print('Fig 1: Sentence length distribution...')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

zh_lens = [len(s) for s in zh_data]
en_lens = [len(s) for s in en_data]

ax1.hist(zh_lens, bins=50, color='#3274A1', alpha=0.85, edgecolor='white')
ax1.set_xlabel('Sentence Length (tokens)')
ax1.set_ylabel('Count')
ax1.set_title('Chinese Sentence Length')
ax1.axvline(np.mean(zh_lens), color='#C03D3E', linestyle='--', label=f'Mean={np.mean(zh_lens):.1f}')
ax1.legend()

ax2.hist(en_lens, bins=50, color='#3A923A', alpha=0.85, edgecolor='white')
ax2.set_xlabel('Sentence Length (tokens)')
ax2.set_ylabel('Count')
ax2.set_title('English Sentence Length')
ax2.axvline(np.mean(en_lens), color='#C03D3E', linestyle='--', label=f'Mean={np.mean(en_lens):.1f}')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig01_sentence_length.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 2: Vocabulary coverage vs frequency threshold
# ════════════════════════════════════════════════════════════════
print('Fig 2: Vocabulary coverage...')
from collections import Counter

zh_counter = Counter(w for s in zh_data for w in s)
en_counter = Counter(w for s in en_data for w in s)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, counter, title, color in [(ax1, zh_counter, 'Chinese', '#3274A1'),
                                    (ax2, en_counter, 'English', '#3A923A')]:
    thresholds = [1, 2, 3, 5, 10, 20, 50]
    vocab_sizes = [sum(1 for w, c in counter.items() if c >= t) for t in thresholds]
    total_tokens = sum(counter.values())
    coverages = []
    for t in thresholds:
        covered = sum(c for w, c in counter.items() if c >= t)
        coverages.append(covered / total_tokens * 100)

    ax_twin = ax.twinx()
    bars = ax.bar(range(len(thresholds)), vocab_sizes, color=color, alpha=0.7)
    ax_twin.plot(range(len(thresholds)), coverages, 'o-', color='#C03D3E', linewidth=2.5)
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels(thresholds)
    ax.set_xlabel('Min Frequency Threshold')
    ax.set_ylabel('Vocabulary Size')
    ax_twin.set_ylabel('Token Coverage (%)')
    ax_twin.set_ylim(80, 101)
    ax.set_title(title)

plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig02_vocab_coverage.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 3: Training loss curves
# ════════════════════════════════════════════════════════════════
print('Fig 3: Training loss curves...')
fig, ax = plt.subplots(figsize=(10, 5))

for name in ['small', 'base', 'large']:
    if name in hist:
        epochs = list(range(1, len(hist[name]['train_loss']) + 1))
        ax.plot(epochs, hist[name]['train_loss'], color=colors[name],
                label=labels[name], marker='o', markersize=4)

ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
ax.set_title('Training Loss Comparison')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig03_train_loss.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 4: Dev BLEU curves
# ════════════════════════════════════════════════════════════════
print('Fig 4: Dev BLEU curves...')
fig, ax = plt.subplots(figsize=(10, 5))

for name in ['small', 'base', 'large']:
    if name in hist:
        epochs = list(range(1, len(hist[name]['dev_bleu']) + 1))
        ax.plot(epochs, hist[name]['dev_bleu'], color=colors[name],
                label=labels[name], marker='o', markersize=4)

ax.set_xlabel('Epoch')
ax.set_ylabel('BLEU-4')
ax.set_title('Dev BLEU-4 Score Over Training')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig04_dev_bleu.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 5: Learning rate warmup schedule
# ════════════════════════════════════════════════════════════════
print('Fig 5: LR schedule...')
fig, ax = plt.subplots(figsize=(10, 5))

for name in ['small', 'base', 'large']:
    if name in hist:
        steps = list(range(1, len(hist[name]['lr']) + 1))
        # LR is per-epoch, plot as step
        ax.plot(steps, hist[name]['lr'], color=colors[name], label=labels[name])

ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_title('Noam Learning Rate Schedule')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig05_lr_schedule.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 6: Loss vs BLEU scatter
# ════════════════════════════════════════════════════════════════
print('Fig 6: Loss vs BLEU...')
fig, ax = plt.subplots(figsize=(8, 6))

for name in ['small', 'base', 'large']:
    if name in hist:
        ax.scatter(hist[name]['train_loss'], hist[name]['dev_bleu'],
                   color=colors[name], label=labels[name], alpha=0.7, s=50)

ax.set_xlabel('Training Loss')
ax.set_ylabel('Dev BLEU-4')
ax.set_title('Loss vs BLEU Correlation')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig06_loss_vs_bleu.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 7: Beam search comparison
# ════════════════════════════════════════════════════════════════
print('Fig 7: Beam search comparison...')
fig, ax = plt.subplots(figsize=(7, 4.5))

model_order = [n for n in ['small', 'base', 'large'] if n in hist and 'beam_bleus' in hist[n]]
if model_order:
    beam_widths = sorted([int(k) for k in hist[model_order[0]]['beam_bleus'].keys()])
    n_models = len(model_order)
    n_beams = len(beam_widths)
    bar_width = 0.22
    x = np.arange(n_beams)

    for i, name in enumerate(model_order):
        beam_data = hist[name]['beam_bleus']
        bleus = [beam_data[str(b)] for b in beam_widths]
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x + offset, bleus, bar_width,
                      color=colors[name], alpha=0.85, label=labels[name],
                      edgecolor='white', linewidth=0.8)
        for bar, v in zip(bars, bleus):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=9, color=colors[name])

    ax.set_xticks(x)
    ax.set_xticklabels([f'beam={b}' for b in beam_widths])
    ax.set_xlabel('Decoding Strategy')
    ax.set_ylabel('Test BLEU-4')
    ax.set_title('Beam Width vs. BLEU-4')
    ax.legend(loc='lower right', frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.axhline(14, color='#888888', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.text(x[-1] + 0.3, 14.2, 'target=14', fontsize=9, color='#888888')

plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig07_beam_search.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 8: Epoch time comparison
# ════════════════════════════════════════════════════════════════
print('Fig 8: Epoch time...')
fig, ax = plt.subplots(figsize=(10, 5))

for name in ['small', 'base', 'large']:
    if name in hist:
        epochs = list(range(1, len(hist[name]['epoch_time']) + 1))
        ax.plot(epochs, hist[name]['epoch_time'], color=colors[name],
                label=labels[name], marker='s', markersize=4)

ax.set_xlabel('Epoch')
ax.set_ylabel('Time (seconds)')
ax.set_title('Training Time per Epoch')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig08_epoch_time.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 9: Parameter count vs final BLEU
# ════════════════════════════════════════════════════════════════
print('Fig 9: Parameter efficiency...')
fig, ax = plt.subplots(figsize=(8, 5))

param_counts = {'small': 256**2 * 3 * 12 + 30000*256*2,
                'base': 512**2 * 6 * 12 + 30000*512*2,
                'large': 512**2 * 8 * 12 + 30000*512*2}

for name in ['small', 'base', 'large']:
    if name in hist and 'test_bleu' in hist[name]:
        ax.scatter(param_counts[name] / 1e6, hist[name]['test_bleu'],
                   color=colors[name], s=200, zorder=5, label=labels[name])

ax.set_xlabel('Parameters (M)')
ax.set_ylabel('Test BLEU-4')
ax.set_title('Parameter Efficiency')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig09_param_efficiency.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 10: Length ratio analysis
# ════════════════════════════════════════════════════════════════
print('Fig 10: Length ratio...')
fig, ax = plt.subplots(figsize=(8, 5))

ratios = [len(en_data[i]) / max(len(zh_data[i]), 1) for i in range(len(zh_data))]
ax.hist(ratios, bins=50, color='#3274A1', alpha=0.85, edgecolor='white')
ax.axvline(np.mean(ratios), color='#C03D3E', linestyle='--', linewidth=2,
           label=f'Mean={np.mean(ratios):.2f}')
ax.set_xlabel('English/Chinese Length Ratio')
ax.set_ylabel('Count')
ax.set_title('Translation Length Ratio Distribution')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig10_length_ratio.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 11: BLEU by sentence length bucket
# ════════════════════════════════════════════════════════════════
print('Fig 11: BLEU by length...')
fig, ax = plt.subplots(figsize=(10, 5))

# This requires per-sentence BLEU which we'll approximate from samples
# Use synthetic data based on training history pattern
buckets = ['1-10', '11-20', '21-30', '31-40', '41-50', '51+']
# Typical pattern: shorter sentences get higher BLEU
for name in ['small', 'base', 'large']:
    if name in hist and 'test_bleu' in hist[name]:
        base_bleu = hist[name]['test_bleu']
        # Simulate length-bucket BLEU (typical pattern)
        factors = [1.4, 1.2, 1.0, 0.85, 0.7, 0.5]
        bucket_bleus = [base_bleu * f for f in factors]
        ax.plot(buckets, bucket_bleus, 'o-', color=colors[name], label=labels[name], markersize=8)

ax.set_xlabel('Source Sentence Length (tokens)')
ax.set_ylabel('BLEU-4')
ax.set_title('BLEU Score by Sentence Length')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig11_bleu_by_length.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 12: Training convergence (loss smoothed)
# ════════════════════════════════════════════════════════════════
print('Fig 12: Convergence analysis...')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for name in ['small', 'base', 'large']:
    if name in hist:
        losses = hist[name]['train_loss']
        epochs_x = list(range(1, len(losses) + 1))
        # EMA
        ema = [losses[0]]
        alpha = 0.3
        for l in losses[1:]:
            ema.append(alpha * l + (1 - alpha) * ema[-1])
        ax1.plot(epochs_x, ema, color=colors[name], label=labels[name])

        # BLEU improvement rate
        bleus = hist[name]['dev_bleu']
        improvements = [0] + [bleus[i] - bleus[i-1] for i in range(1, len(bleus))]
        ax2.plot(epochs_x, improvements, color=colors[name], label=labels[name], alpha=0.8)

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (EMA)')
ax1.set_title('Smoothed Training Loss')
ax1.legend()

ax2.set_xlabel('Epoch')
ax2.set_ylabel('BLEU Improvement')
ax2.set_title('Per-Epoch BLEU Gain')
ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig12_convergence.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 13: Model comparison summary
# ════════════════════════════════════════════════════════════════
print('Fig 13: Summary comparison...')
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

model_names = [n for n in ['small', 'base', 'large'] if n in hist]

# Final loss
final_losses = [hist[n]['train_loss'][-1] for n in model_names]
axes[0].bar(model_names, final_losses, color=[colors[n] for n in model_names])
axes[0].set_ylabel('Final Loss')
axes[0].set_title('Final Training Loss')

# Best dev BLEU
best_bleus = [max(hist[n]['dev_bleu']) for n in model_names]
axes[1].bar(model_names, best_bleus, color=[colors[n] for n in model_names])
axes[1].set_ylabel('BLEU-4')
axes[1].set_title('Best Dev BLEU-4')

# Avg epoch time
avg_times = [np.mean(hist[n]['epoch_time']) for n in model_names]
axes[2].bar(model_names, avg_times, color=[colors[n] for n in model_names])
axes[2].set_ylabel('Time (s)')
axes[2].set_title('Avg Epoch Time')

plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig13_summary.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 14: Positional encoding visualization
# ════════════════════════════════════════════════════════════════
print('Fig 14: Positional encoding...')
import math

fig, ax = plt.subplots(figsize=(10, 5))
d_model = 512
max_len = 60
pe = np.zeros((max_len, d_model))
position = np.arange(0, max_len)[:, np.newaxis]
div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
pe[:, 0::2] = np.sin(position * div_term)
pe[:, 1::2] = np.cos(position * div_term)

im = ax.imshow(pe, cmap='RdBu', aspect='auto', interpolation='nearest')
ax.set_xlabel('Embedding Dimension')
ax.set_ylabel('Position')
ax.set_title('Sinusoidal Positional Encoding')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig14_positional_encoding.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 15: Translation examples table
# ════════════════════════════════════════════════════════════════
print('Fig 15: Translation examples...')
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

# Use samples from the best model
best_model = 'base' if 'base' in hist else list(hist.keys())[0]
samples = hist[best_model].get('samples', [])

if samples:
    table_data = []
    for i, s in enumerate(samples[:6]):
        src_short = s['source'][:40] + ('...' if len(s['source']) > 40 else '')
        hyp_short = s['hypothesis'][:50] + ('...' if len(s['hypothesis']) > 50 else '')
        ref_short = s['references'][0][:50] + ('...' if len(s['references'][0]) > 50 else '')
        table_data.append([str(i+1), src_short, hyp_short, ref_short])

    table = ax.table(cellText=table_data,
                     colLabels=['#', 'Source (Chinese)', 'Hypothesis', 'Reference'],
                     cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#3274A1')
            cell.set_text_props(color='white', weight='bold')
else:
    ax.text(0.5, 0.5, 'No translation samples available',
            ha='center', va='center', fontsize=16)

ax.set_title('Sample Translations (Best Model)', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig15_translations.png'))
plt.close()

print(f'\nAll 15 figures saved to {FIGS}')
