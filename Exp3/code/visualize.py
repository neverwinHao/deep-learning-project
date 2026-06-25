import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import sys
from collections import Counter

import matplotlib.font_manager as fm
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
    'grid.linestyle': '-',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.framealpha': 0.92,
    'lines.linewidth': 2.5,
    'lines.markersize': 6,
})

BASE = os.path.dirname(os.path.dirname(__file__))
FIGS = os.path.join(BASE, 'figs')
DATA_PATH = os.path.join(BASE, '实验3：数据集tang.npz')
os.makedirs(FIGS, exist_ok=True)

with open(os.path.join(BASE, 'all_history.json')) as f:
    hist = json.load(f)

d = np.load(DATA_PATH, allow_pickle=True)
ix2word = d['ix2word'].item()
word2ix = d['word2ix'].item()
data = d['data']

colors = {'basic': '#3274A1', 'deep': '#C03D3E', 'gru': '#3A923A'}
labels = {'basic': 'Basic (1-layer LSTM)', 'deep': 'Deep (3-layer LSTM)', 'gru': 'GRU (2-layer)'}
epochs = list(range(1, 31))

# ════════════════════════════════════════════════════════════════
# Fig 1: Dataset analysis — poem length distribution
# ════════════════════════════════════════════════════════════════
print('Fig 1: Dataset poem length distribution...')
pad_id = word2ix['</s>']
start_id = word2ix['<START>']
eop_id = word2ix['<EOP>']

poem_lengths = []
for row in data:
    non_pad = np.sum(row != pad_id)
    poem_lengths.append(non_pad - 2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.hist(poem_lengths, bins=60, color='#3274A1', edgecolor='white', alpha=0.95)
ax.axvline(np.mean(poem_lengths), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(poem_lengths):.1f}')
ax.axvline(np.median(poem_lengths), color='orange', linestyle='--', linewidth=2, label=f'Median={np.median(poem_lengths):.0f}')
ax.set_xlabel('Poem Length (chars)')
ax.set_ylabel('Count')
ax.set_title('Poem Length Distribution')
ax.legend()

ax2 = axes[1]
type_names = ['五言绝句', '七言绝句', '五言律诗', '七言律诗', '其他']
type_vals = [0, 0, 0, 0, 0]
for l in poem_lengths:
    if 18 <= l <= 22: type_vals[0] += 1
    elif 26 <= l <= 30: type_vals[1] += 1
    elif 38 <= l <= 42: type_vals[2] += 1
    elif 54 <= l <= 58: type_vals[3] += 1
    else: type_vals[4] += 1

bar_colors = ['#3274A1','#2EABB8','#E1812C','#9372B2','#7F7F7F']
bars = ax2.bar(range(len(type_names)), type_vals, color=bar_colors, edgecolor='white', linewidth=1.5)
ax2.set_xticks(range(len(type_names)))
ax2.set_xticklabels(type_names, rotation=30, ha='right')
for bar, v in zip(bars, type_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
             f'{v}', ha='center', va='bottom', fontweight='bold')
ax2.set_ylabel('Count')
ax2.set_title('Poem Type Distribution (by length)')
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig01_dataset_length.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 2: Character frequency analysis
# ════════════════════════════════════════════════════════════════
print('Fig 2: Character frequency...')
all_chars = []
for row in data:
    for idx in row:
        if idx not in (pad_id, start_id, eop_id):
            all_chars.append(ix2word[idx])

char_freq = Counter(all_chars)
top50 = char_freq.most_common(50)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
chars = [c for c, _ in top50[:30]]
freqs = [f for _, f in top50[:30]]
ax.barh(range(len(chars)), freqs, color='#3274A1', alpha=0.95)
ax.set_yticks(range(len(chars)))
ax.set_yticklabels(chars, fontsize=13)
ax.invert_yaxis()
ax.set_xlabel('Frequency')
ax.set_title('Top 30 Most Frequent Characters')

ax2 = axes[1]
all_freqs = sorted(char_freq.values(), reverse=True)
ranks = np.arange(1, len(all_freqs) + 1)
ax2.loglog(ranks, all_freqs, 'o', markersize=2, color='#C03D3E', alpha=0.7)
coeffs = np.polyfit(np.log(ranks[:500]), np.log(all_freqs[:500]), 1)
fit_line = np.exp(coeffs[1]) * ranks[:500] ** coeffs[0]
ax2.loglog(ranks[:500], fit_line, '--', color='black', linewidth=2, label=f'Zipf fit (α={-coeffs[0]:.2f})')
ax2.set_xlabel('Rank (log)')
ax2.set_ylabel('Frequency (log)')
ax2.set_title("Zipf's Law of Character Frequency")
ax2.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig02_char_frequency.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 3: Training loss curves with EMA smoothing
# ════════════════════════════════════════════════════════════════
print('Fig 3: Loss curves with EMA...')
def ema(values, alpha=0.3):
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
for name in hist:
    ax.plot(epochs, hist[name]['train_loss'], color=colors[name], alpha=0.3, linewidth=1)
    ax.plot(epochs, ema(hist[name]['train_loss']), color=colors[name], label=labels[name], linewidth=2.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss (with EMA smoothing)')
ax.legend()


ax2 = axes[1]
for name in hist:
    ax2.plot(epochs, hist[name]['perplexity'], color=colors[name], alpha=0.3, linewidth=1)
    ax2.plot(epochs, ema(hist[name]['perplexity']), color=colors[name], label=labels[name], linewidth=2.5)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Perplexity')
ax2.set_title('Perplexity Curve')
ax2.legend()

fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig03_loss_ppl.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 4: Convergence rate and training dynamics
# ════════════════════════════════════════════════════════════════
print('Fig 4: Convergence dynamics...')
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

ax = axes[0]
for name in hist:
    losses = hist[name]['train_loss']
    improvement = [losses[i-1] - losses[i] for i in range(1, len(losses))]
    ax.plot(epochs[1:], improvement, color=colors[name], label=labels[name], linewidth=2, marker='o', markersize=4)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss Improvement')
ax.set_title('Per-Epoch Loss Improvement')
ax.legend()


ax2 = axes[1]
for name in hist:
    losses = hist[name]['train_loss']
    initial = losses[0]
    cum_pct = [(initial - l) / initial * 100 for l in losses]
    ax2.plot(epochs, cum_pct, color=colors[name], label=labels[name], linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Cumulative Improvement (%)')
ax2.set_title('Cumulative Loss Reduction (%)')
ax2.legend()


ax3 = axes[2]
for name in hist:
    ax3.semilogy(epochs, hist[name]['train_loss'], color=colors[name], label=labels[name], linewidth=2)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss (log scale)')
ax3.set_title('Loss on Log Scale')
ax3.legend()

fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig04_convergence.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 5: LR schedule and time
# ════════════════════════════════════════════════════════════════
print('Fig 5: LR and time...')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
for name in hist:
    ax.plot(epochs, hist[name]['lr'], color=colors[name], label=labels[name], linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedule (StepLR)')
ax.legend()

for ep in [10, 20]:
    ax.axvline(x=ep, color='gray', linestyle=':', alpha=0.5)
    ax.annotate(f'×0.5 at ep {ep}', xy=(ep, hist['basic']['lr'][ep-1]), fontsize=11)

ax2 = axes[1]
for name in hist:
    ax2.plot(epochs, hist[name]['epoch_time'], color=colors[name], label=labels[name], linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Time (s)')
ax2.set_title('Training Time per Epoch')
ax2.legend()

fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig05_lr_time.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 6: Model comparison bar chart
# ════════════════════════════════════════════════════════════════
print('Fig 6: Summary bars...')
fig, axes = plt.subplots(1, 4, figsize=(14, 5))
names = list(hist.keys())
metrics = [
    ([hist[n]['train_loss'][-1] for n in names], 'Final Loss', 'Loss'),
    ([hist[n]['perplexity'][-1] for n in names], 'Final Perplexity', 'PPL'),
    ([sum(hist[n]['epoch_time']) for n in names], 'Total Training Time', 'Time (s)'),
    ([sum(hist[n]['epoch_time']) / hist[n]['perplexity'][-1] for n in names], 'Time / PPL Efficiency', 'Time/PPL'),
]

for ax, (vals, title, ylabel) in zip(axes, metrics):
    bars = ax.bar([labels[n].split('(')[0] for n in names], vals,
                  color=[colors[n] for n in names], edgecolor='white', linewidth=1.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{v:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig06_summary.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 7: Parameter efficiency analysis
# ════════════════════════════════════════════════════════════════
print('Fig 7: Parameter efficiency...')
params = {'basic': 3588069, 'deep': 12156773, 'gru': 9135973}
final_ppl = {n: hist[n]['perplexity'][-1] for n in hist}
total_time = {n: sum(hist[n]['epoch_time']) for n in hist}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for name in hist:
    ax.scatter(params[name]/1e6, final_ppl[name], color=colors[name], s=250, zorder=5,
              edgecolors='black', linewidth=1.5)
    ax.annotate(labels[name], (params[name]/1e6, final_ppl[name]),
               textcoords="offset points", xytext=(12, 12), fontsize=12)
ax.set_xlabel('Parameters (M)')
ax.set_ylabel('Final PPL')
ax.set_title('Parameters vs Perplexity')

ax.invert_yaxis()

ax2 = axes[1]
for name in hist:
    ax2.scatter(total_time[name], final_ppl[name], color=colors[name], s=250, zorder=5,
               edgecolors='black', linewidth=1.5)
    ax2.annotate(labels[name], (total_time[name], final_ppl[name]),
                textcoords="offset points", xytext=(12, 12), fontsize=12)
ax2.set_xlabel('Total Training Time (s)')
ax2.set_ylabel('Final PPL')
ax2.set_title('Training Time vs Perplexity')

ax2.invert_yaxis()
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig07_efficiency.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 8: PPL convergence speed — epochs to reach thresholds
# ════════════════════════════════════════════════════════════════
print('Fig 8: PPL threshold convergence...')
thresholds = [10, 8, 7, 6.5, 6, 5.5]
fig, ax = plt.subplots(figsize=(10, 5))
width = 0.25
x = np.arange(len(thresholds))

for i, name in enumerate(hist):
    epochs_to = []
    for thr in thresholds:
        reached = None
        for ep, ppl in enumerate(hist[name]['perplexity'], 1):
            if ppl <= thr:
                reached = ep
                break
        epochs_to.append(reached if reached else 31)
    ax.bar(x + i * width, epochs_to, width, label=labels[name], color=colors[name], alpha=0.95, edgecolor='white')

ax.set_xticks(x + width)
ax.set_xticklabels([f'PPL≤{t}' for t in thresholds])
ax.set_ylabel('Epochs Required')
ax.set_title('Epochs to Reach PPL Thresholds')
ax.legend()
ax.axhline(y=30, color='gray', linestyle='--', alpha=0.4)
ax.grid(True, alpha=0.2, axis='y')
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig08_ppl_threshold.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 9: Radar chart — multi-dimensional model comparison
# ════════════════════════════════════════════════════════════════
print('Fig 9: Radar chart...')
categories = ['Final PPL\n(lower=better)', 'Convergence\n(first 5 ep)', 'Param Efficiency\n(PPL/params)', 'Training Speed\n(faster=better)', 'Stability\n(last 10 ep var)']

def normalize(vals, invert=False):
    mn, mx = min(vals), max(vals)
    if mx == mn: return [0.5] * len(vals)
    normed = [(v - mn) / (mx - mn) for v in vals]
    if invert: normed = [1 - n for n in normed]
    return normed

raw = {}
for name in hist:
    ppl_val = hist[name]['perplexity'][-1]
    early_improve = hist[name]['train_loss'][0] - hist[name]['train_loss'][4]
    ppl_per_param = ppl_val / (params[name] / 1e6)
    speed = 1.0 / np.mean(hist[name]['epoch_time'])
    stability = np.std(hist[name]['train_loss'][-10:])
    raw[name] = [ppl_val, early_improve, ppl_per_param, speed, stability]

normed = {}
for name in hist:
    normed[name] = []
for dim in range(5):
    dim_vals = [raw[n][dim] for n in hist]
    if dim in [0, 2, 4]:
        n_vals = normalize(dim_vals, invert=True)
    else:
        n_vals = normalize(dim_vals, invert=False)
    for i, name in enumerate(hist):
        normed[name].append(n_vals[i])

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

for name in hist:
    vals = normed[name] + normed[name][:1]
    ax.plot(angles, vals, 'o-', linewidth=2, label=labels[name], color=colors[name])
    ax.fill(angles, vals, alpha=0.1, color=colors[name])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_title('Multi-Dimensional Model Comparison', fontsize=16, pad=25)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig09_radar.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 10: Punctuation and structure analysis of dataset
# ════════════════════════════════════════════════════════════════
print('Fig 10: Punctuation structure...')
puncts = set('，。！？、；：')
lines_per_poem = []
chars_per_line_all = []

for row in data:
    text = ''.join([ix2word[idx] for idx in row if idx not in (pad_id, start_id, eop_id)])
    lines = []
    current = 0
    for ch in text:
        if ch in puncts:
            lines.append(current)
            current = 0
        else:
            current += 1
    if current > 0:
        lines.append(current)
    lines_per_poem.append(len(lines))
    chars_per_line_all.extend(lines)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.hist(chars_per_line_all, bins=range(1, 15), color='#2EABB8', edgecolor='white', alpha=0.95, align='left')
ax.set_xlabel('Characters per Line')
ax.set_ylabel('Frequency')
ax.set_title('Verse Length Distribution')
ax.set_xticks(range(1, 15))

ax2 = axes[1]
ax2.hist(lines_per_poem, bins=range(1, 30), color='#9372B2', edgecolor='white', alpha=0.95, align='left')
ax2.set_xlabel('Lines per Poem')
ax2.set_ylabel('Frequency')
ax2.set_title('Lines per Poem Distribution')
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig10_structure.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 11: Semantic category thematic words
# ════════════════════════════════════════════════════════════════
print('Fig 11: Thematic word analysis...')
themes = {
    '山水自然': list('山水云风月日天雪花草树林江河海湖溪泉石峰松柏竹'),
    '人物情感': list('人君子客僧友妻儿愁恨悲怜思忆别离梦泪情'),
    '宫廷政治': list('帝王臣官宫殿朝廷诏龙凤'),
    '战争边塞': list('兵马剑戈关塞城烽征战胡'),
    '时间季节': list('春秋冬夏朝暮晨夕昔今年岁'),
    '色彩': list('白青红黄碧紫翠绿金银'),
}

theme_counts = {}
for theme, chars in themes.items():
    cnt = sum(char_freq.get(c, 0) for c in chars)
    theme_counts[theme] = cnt

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
theme_names = list(theme_counts.keys())
theme_vals = list(theme_counts.values())
theme_colors_list = ['#2EABB8', '#3274A1', '#E1812C', '#9372B2', '#A9AA35', '#C03D3E']
bars = ax.barh(theme_names, theme_vals, color=theme_colors_list, edgecolor='white', linewidth=1.5)
ax.set_xlabel('Cumulative Frequency')
ax.set_title('Thematic Word Frequency Analysis')
for bar, v in zip(bars, theme_vals):
    ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
            f'{v:,}', ha='left', va='center', fontsize=12)

ax2 = axes[1]
for i, (theme, chars) in enumerate(themes.items()):
    sorted_chars = sorted(chars, key=lambda c: char_freq.get(c, 0), reverse=True)[:5]
    text = ' '.join([f'{c}({char_freq.get(c,0)})' for c in sorted_chars])
    ax2.text(0.05, 0.88 - i * 0.16, f'【{theme}】{text}',
            transform=ax2.transAxes, fontsize=14, color=theme_colors_list[i], fontweight='bold')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.set_title('Top 5 Characters per Theme')
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig11_themes.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 12: Learning rate impact — loss at decay points
# ════════════════════════════════════════════════════════════════
print('Fig 12: LR decay impact...')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, decay_ep in enumerate([10, 20]):
    ax = axes[idx]
    window = range(max(1, decay_ep - 3), min(31, decay_ep + 4))
    for name in hist:
        losses = hist[name]['train_loss']
        ax.plot(list(window), [losses[e-1] for e in window],
                color=colors[name], label=labels[name], linewidth=2, marker='o', markersize=7)
    ax.axvline(x=decay_ep, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'LR decay (ep {decay_ep})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss around LR Decay Point (ep {decay_ep})')
    ax.legend(fontsize=10)
    

fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig12_lr_decay_impact.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 13: Training phase analysis (3 phases)
# ════════════════════════════════════════════════════════════════
print('Fig 13: Training phases...')
fig, ax = plt.subplots(figsize=(10, 5.5))
name = 'gru'
losses = hist[name]['train_loss']
ax.plot(epochs, losses, color=colors[name], linewidth=2.5, label='GRU Loss')

ax.axvspan(1, 5, alpha=0.15, color='red', label='Phase I: Rapid Learning')
ax.axvspan(5, 15, alpha=0.15, color='orange', label='Phase II: Steady Improvement')
ax.axvspan(15, 30, alpha=0.15, color='green', label='Phase III: Fine Convergence')

ax.annotate(f'Initial Loss: {losses[0]:.3f}', xy=(1, losses[0]), fontsize=13,
           arrowprops=dict(arrowstyle='->', color='black'), xytext=(3, losses[0]+0.15))
ax.annotate(f'End of Phase I: {losses[4]:.3f}\n{(losses[0]-losses[4])/losses[0]*100:.1f}% reduction',
           xy=(5, losses[4]), fontsize=12, xytext=(7, losses[4]+0.15),
           arrowprops=dict(arrowstyle='->', color='black'))
ax.annotate(f'Final Loss: {losses[-1]:.3f}\nPPL: {hist[name]["perplexity"][-1]:.2f}',
           xy=(30, losses[-1]), fontsize=13, xytext=(24, losses[-1]+0.1),
           arrowprops=dict(arrowstyle='->', color='black'))

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('GRU Training: Three-Phase Analysis')
ax.legend(fontsize=12, loc='upper right')

fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig13_phases.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 14: Architecture parameter comparison
# ════════════════════════════════════════════════════════════════
print('Fig 14: Architecture comparison...')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
components = ['Embedding', 'RNN Layers', 'FC Layer']
basic_parts = [128*8293, (128+256)*256*4, 256*8293]
deep_parts = [256*8293, (256+512)*512*4*3, 512*8293]
gru_parts = [256*8293, (256+512)*512*3*2, 512*8293]

x = np.arange(len(components))
width = 0.25
ax1.bar(x - width, [p/1e6 for p in basic_parts], width, label='Basic', color=colors['basic'], edgecolor='white')
ax1.bar(x, [p/1e6 for p in deep_parts], width, label='Deep', color=colors['deep'], edgecolor='white')
ax1.bar(x + width, [p/1e6 for p in gru_parts], width, label='GRU', color=colors['gru'], edgecolor='white')
ax1.set_xticks(x)
ax1.set_xticklabels(components)
ax1.set_ylabel('Parameters (M)')
ax1.set_title('Parameter Distribution by Module')
ax1.legend()

ax2 = axes[1]
gate_info = {'LSTM\n(4 gates)': 4, 'GRU\n(3 gates)': 3}
ax2.bar(gate_info.keys(), gate_info.values(), color=['#C03D3E', '#3A923A'],
       width=0.5, edgecolor='white', linewidth=2)
ax2.set_ylabel('Matrix Multiplications / Step')
ax2.set_title('LSTM vs GRU Computation Cost\n(Gate Matrices per Timestep)')
for i, (k, v) in enumerate(gate_info.items()):
    ax2.text(i, v + 0.1, f'{v}', ha='center', fontsize=16, fontweight='bold')

fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig14_architecture.png'))
plt.close()

# ════════════════════════════════════════════════════════════════
# Fig 15: PPL trajectory scatter
# ════════════════════════════════════════════════════════════════
print('Fig 15: PPL trajectory scatter...')
fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(hist['basic']['perplexity'], hist['deep']['perplexity'],
          c=epochs, cmap='viridis', s=80, edgecolors='black', linewidth=0.5, zorder=5)
mn = min(min(hist['basic']['perplexity']), min(hist['deep']['perplexity'])) - 0.5
mx = max(max(hist['basic']['perplexity']), max(hist['deep']['perplexity'])) + 0.5
ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.5, label='y=x (equal performance)')
ax.set_xlabel('Basic LSTM PPL')
ax.set_ylabel('Deep LSTM PPL')
ax.set_title('Basic vs Deep: PPL Trajectory\n(color = epoch, yellow = later)')
ax.legend()
cbar = plt.colorbar(sc, ax=ax, label='Epoch')
ax.set_aspect('equal')
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig15_ppl_scatter.png'))
plt.close()

print(f'\nAll 15 figures saved to {FIGS}')
