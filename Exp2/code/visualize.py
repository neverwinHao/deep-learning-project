"""
Publication-quality visualizations for ViT CIFAR-10 experiment.
Style: NeurIPS / ICML / CVPR level figures.
"""
import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, precision_score, recall_score)
import torch
import torchvision
import torchvision.transforms as transforms

# ======================== Global Config ========================

CLASSES = ('Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
           'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
CLASSES_SHORT = ('plane', 'car', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck')

# ---- Publication color scheme (colorblind-friendly) ----
PAL = {
    'blue':   '#3274A1',
    'orange': '#E1812C',
    'green':  '#3A923A',
    'red':    '#C03D3E',
    'purple': '#9372B2',
    'brown':  '#845B53',
    'pink':   '#D684BD',
    'gray':   '#7F7F7F',
    'olive':  '#A9AA35',
    'cyan':   '#2EABB8',
}
CLASS_PAL = list(PAL.values())

# ---- LaTeX-like style ----
def set_pub_style():
    matplotlib.rcParams.update({
        'figure.dpi': 200,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08,
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'],
        'mathtext.fontset': 'dejavuserif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.linewidth': 0.8,
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': '#E5E5E5',
        'grid.linewidth': 0.4,
        'grid.linestyle': '-',
        'legend.frameon': True,
        'legend.framealpha': 0.92,
        'legend.edgecolor': '#CCCCCC',
        'legend.fontsize': 9,
        'legend.handlelength': 1.8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,
        'figure.facecolor': 'white',
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
    })


def _smooth(y, weight=0.85):
    """Exponential moving average for smoother curves."""
    s = []
    last = y[0]
    for v in y:
        last = weight * last + (1 - weight) * v
        s.append(last)
    return np.array(s)


def _savefig(fig, path, name):
    full = os.path.join(path, name)
    fig.savefig(full)
    plt.close(fig)
    print(f'  -> {name}')


# ======================== 1. Loss Curve ========================

def plot_loss(history, save_dir):
    epochs = np.arange(1, len(history['train_loss']) + 1)
    tl = np.array(history['train_loss'])
    vl = np.array(history['test_loss'])
    tl_s = _smooth(tl, 0.8)
    vl_s = _smooth(vl, 0.8)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    ax.plot(epochs, tl, color=PAL['blue'], alpha=0.25, linewidth=0.8)
    ax.plot(epochs, vl, color=PAL['orange'], alpha=0.25, linewidth=0.8)
    ax.plot(epochs, tl_s, color=PAL['blue'], linewidth=2, label='Train')
    ax.plot(epochs, vl_s, color=PAL['orange'], linewidth=2, label='Val')

    # fill gap
    ax.fill_between(epochs, tl_s, vl_s, where=vl_s >= tl_s,
                    alpha=0.08, color=PAL['orange'], interpolate=True)
    ax.fill_between(epochs, tl_s, vl_s, where=vl_s < tl_s,
                    alpha=0.08, color=PAL['blue'], interpolate=True)

    # annotate min val loss
    mi = np.argmin(vl)
    ax.scatter(epochs[mi], vl[mi], color=PAL['orange'], s=50, zorder=5,
              edgecolors='white', linewidth=1.2)
    ax.annotate(f'{vl[mi]:.3f}', xy=(epochs[mi], vl[mi]),
                xytext=(8, 10), textcoords='offset points',
                fontsize=8, fontweight='bold', color=PAL['orange'],
                arrowprops=dict(arrowstyle='-', color=PAL['orange'], lw=0.8))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('(a) Training & Validation Loss')
    ax.legend(loc='upper right')
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(bottom=0)

    _savefig(fig, save_dir, 'fig1_loss.png')


# ======================== 2. Accuracy Curve ========================

def plot_accuracy(history, save_dir):
    epochs = np.arange(1, len(history['train_acc']) + 1)
    ta = np.array(history['train_acc'])
    va = np.array(history['test_acc'])
    ta_s = _smooth(ta, 0.8)
    va_s = _smooth(va, 0.8)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    ax.plot(epochs, ta, color=PAL['blue'], alpha=0.25, linewidth=0.8)
    ax.plot(epochs, va, color=PAL['orange'], alpha=0.25, linewidth=0.8)
    ax.plot(epochs, ta_s, color=PAL['blue'], linewidth=2, label='Train')
    ax.plot(epochs, va_s, color=PAL['orange'], linewidth=2, label='Val')

    # 80% target
    ax.axhline(80, color=PAL['green'], ls='--', lw=1.2, alpha=0.7, label='Target (80%)')

    # best val
    bi = np.argmax(va)
    ax.scatter(epochs[bi], va[bi], color=PAL['orange'], s=50, zorder=5,
              edgecolors='white', linewidth=1.2)
    ax.annotate(f'{va[bi]:.1f}%', xy=(epochs[bi], va[bi]),
                xytext=(-35, -18), textcoords='offset points',
                fontsize=8.5, fontweight='bold', color=PAL['orange'],
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=PAL['orange'], lw=0.8),
                arrowprops=dict(arrowstyle='->', color=PAL['orange'], lw=0.8))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(b) Training & Validation Accuracy')
    ax.legend(loc='lower right')
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(35, 98)

    _savefig(fig, save_dir, 'fig2_accuracy.png')


# ======================== 3. Generalization Gap ========================

def plot_gap(history, save_dir):
    epochs = np.arange(1, len(history['train_acc']) + 1)
    ta = np.array(history['train_acc'])
    va = np.array(history['test_acc'])
    gap = ta - va

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    ax.fill_between(epochs, 0, gap, where=gap <= 5, color=PAL['green'], alpha=0.35, label='Healthy (<5%)')
    ax.fill_between(epochs, 0, gap, where=(gap > 5) & (gap <= 10), color=PAL['orange'], alpha=0.35, label='Warning (5-10%)')
    ax.fill_between(epochs, 0, gap, where=gap > 10, color=PAL['red'], alpha=0.35, label='Overfitting (>10%)')
    ax.plot(epochs, gap, color='#333333', linewidth=1.2)

    ax.axhline(5, color=PAL['orange'], ls=':', lw=0.8, alpha=0.6)
    ax.axhline(10, color=PAL['red'], ls=':', lw=0.8, alpha=0.6)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Acc $-$ Val Acc (%)')
    ax.set_title('(c) Generalization Gap')
    ax.legend(fontsize=8, loc='upper left', ncol=1)
    ax.set_xlim(1, len(epochs))

    _savefig(fig, save_dir, 'fig3_gap.png')


# ======================== 4. LR Schedule ========================

def plot_lr(history, save_dir):
    n = len(history['train_loss'])
    epochs = np.arange(1, n + 1)
    lr_init = 3e-4
    lrs = np.array([lr_init * 0.5 * (1 + np.cos(np.pi * t / n)) for t in range(n)])

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.plot(epochs, lrs * 1e4, color=PAL['purple'], linewidth=2)
    ax.fill_between(epochs, 0, lrs * 1e4, alpha=0.12, color=PAL['purple'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate ($\\times 10^{-4}$)')
    ax.set_title('(d) Cosine Annealing Schedule')
    ax.set_xlim(1, n)
    ax.set_ylim(bottom=0)

    _savefig(fig, save_dir, 'fig4_lr.png')


# ======================== 5. Confusion Matrix ========================

def plot_cm(preds, targets, save_dir):
    cm = confusion_matrix(targets, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # custom cmap
    cmap = LinearSegmentedColormap.from_list('custom',
        ['#FFFFFF', '#D6E4F0', '#6AAED6', '#2171B5', '#08306B'])

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100, aspect='equal')

    # text annotations
    for i in range(10):
        for j in range(10):
            val = cm_pct[i, j]
            color = 'white' if val > 55 else '#333333'
            weight = 'bold' if i == j else 'normal'
            size = 9 if i == j else 7.5
            txt = f'{val:.0f}' if val >= 1 else ''
            if i == j:
                txt = f'{val:.1f}'
            ax.text(j, i, txt, ha='center', va='center',
                    color=color, fontsize=size, fontweight=weight)

    # diagonal highlight
    for i in range(10):
        rect = plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                              edgecolor=PAL['orange'], linewidth=1.8)
        ax.add_patch(rect)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=40, ha='right', fontsize=8.5)
    ax.set_yticklabels(CLASSES, fontsize=8.5)
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('True', fontweight='bold')
    ax.set_title('(e) Normalized Confusion Matrix (%)')

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label('Classification Rate (%)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    _savefig(fig, save_dir, 'fig5_confusion.png')


# ======================== 6. Per-class Accuracy (Lollipop) ========================

def plot_per_class(preds, targets, save_dir):
    cm = confusion_matrix(targets, preds)
    acc = np.diag(cm) / cm.sum(axis=1) * 100
    avg = acc.mean()

    order = np.argsort(acc)
    s_acc = acc[order]
    s_names = [CLASSES[i] for i in order]
    s_colors = [CLASS_PAL[i] for i in order]

    fig, ax = plt.subplots(figsize=(5, 4.5))

    for i, (a, n, c) in enumerate(zip(s_acc, s_names, s_colors)):
        ax.barh(i, a, height=0.55, color=c, alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.plot(a, i, 'o', color=c, markersize=7, markeredgecolor='white', markeredgewidth=1.2, zorder=5)
        ax.text(a + 1.2, i, f'{a:.1f}', va='center', fontsize=8.5, fontweight='bold', color='#333')

    ax.axvline(80, color=PAL['green'], ls='--', lw=1.2, alpha=0.6, label='80% target')
    ax.axvline(avg, color=PAL['red'], ls='--', lw=1.2, alpha=0.6, label=f'Mean={avg:.1f}%')

    ax.set_yticks(range(len(s_names)))
    ax.set_yticklabels(s_names, fontsize=9)
    ax.set_xlabel('Accuracy (%)')
    ax.set_xlim(0, 102)
    ax.set_title('(f) Per-Class Test Accuracy')
    ax.legend(fontsize=8, loc='lower right')

    _savefig(fig, save_dir, 'fig6_per_class.png')


# ======================== 7. Precision / Recall / F1 ========================

def plot_prf(preds, targets, save_dir):
    rpt = classification_report(targets, preds, target_names=CLASSES, output_dict=True)
    prec = np.array([rpt[c]['precision'] * 100 for c in CLASSES])
    rec  = np.array([rpt[c]['recall'] * 100 for c in CLASSES])
    f1   = np.array([rpt[c]['f1-score'] * 100 for c in CLASSES])

    x = np.arange(10)
    w = 0.22

    fig, ax = plt.subplots(figsize=(7, 3.8))

    ax.bar(x - w, prec, w, label='Precision', color=PAL['blue'], alpha=0.85, edgecolor='white', lw=0.5)
    ax.bar(x,     rec,  w, label='Recall',    color=PAL['green'], alpha=0.85, edgecolor='white', lw=0.5)
    ax.bar(x + w, f1,   w, label='F1',        color=PAL['orange'], alpha=0.85, edgecolor='white', lw=0.5)

    for i in range(10):
        ax.text(x[i] + w, f1[i] + 1.2, f'{f1[i]:.0f}', ha='center', va='bottom',
                fontsize=7, fontweight='bold', color=PAL['orange'])

    ax.axhline(80, color=PAL['gray'], ls=':', lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=30, ha='right', fontsize=8.5)
    ax.set_ylabel('Score (%)')
    ax.set_ylim(55, 103)
    ax.set_title('(g) Per-Class Precision, Recall & F1-Score')
    ax.legend(fontsize=8, ncol=3, loc='lower right')

    _savefig(fig, save_dir, 'fig7_prf.png')


# ======================== 8. Error Analysis ========================

def plot_errors(preds, targets, save_dir):
    cm = confusion_matrix(targets, preds)
    np.fill_diagonal(cm, 0)

    pairs = []
    for i in range(10):
        for j in range(10):
            if cm[i][j] > 0:
                pairs.append((i, j, cm[i][j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    top = pairs[:10]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    labels = [f'{CLASSES[p[0]]} $\\rightarrow$ {CLASSES[p[1]]}' for p in top]
    counts = [p[2] for p in top]
    y = range(len(top))

    cmap_r = plt.cm.Reds(np.linspace(0.3, 0.8, len(top)))
    bars = ax.barh(y, counts[::-1], height=0.6, color=cmap_r, edgecolor='white', lw=0.5)

    for bar, c in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                str(c), va='center', fontsize=8, fontweight='bold', color='#555')

    ax.set_yticks(y)
    ax.set_yticklabels(labels[::-1], fontsize=8.5)
    ax.set_xlabel('Count')
    ax.set_title('(h) Top-10 Misclassification Pairs')

    _savefig(fig, save_dir, 'fig8_errors.png')


# ======================== 9. Sample Predictions ========================

def plot_samples(preds, targets, save_dir):
    trans = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root=os.path.join(save_dir, 'data'), train=False, download=True, transform=trans)

    correct_idx, wrong_idx = [], []
    seen_c, seen_w = set(), set()
    for i in range(len(preds)):
        if preds[i] == targets[i] and targets[i] not in seen_c and len(correct_idx) < 10:
            correct_idx.append(i)
            seen_c.add(targets[i])
        elif preds[i] != targets[i] and targets[i] not in seen_w and len(wrong_idx) < 10:
            wrong_idx.append(i)
            seen_w.add(targets[i])
        if len(correct_idx) >= 10 and len(wrong_idx) >= 10:
            break

    fig, axes = plt.subplots(2, 10, figsize=(12, 3.2))
    fig.subplots_adjust(hspace=0.55, wspace=0.08)

    for row, (indices, tag) in enumerate([(correct_idx, 'correct'), (wrong_idx, 'wrong')]):
        for col in range(10):
            ax = axes[row, col]
            if col < len(indices):
                idx = indices[col]
                img = testset[idx][0].permute(1, 2, 0).numpy()
                ax.imshow(np.clip(img, 0, 1), interpolation='lanczos')

                pred_l = CLASSES_SHORT[preds[idx]]
                true_l = CLASSES_SHORT[targets[idx]]
                is_ok = preds[idx] == targets[idx]

                bc = PAL['green'] if is_ok else PAL['red']
                for sp in ax.spines.values():
                    sp.set_edgecolor(bc)
                    sp.set_linewidth(2)

                if is_ok:
                    ax.set_title(true_l, fontsize=7, color=PAL['green'], fontweight='bold', pad=2)
                else:
                    ax.set_title(f'{true_l}$\\to${pred_l}', fontsize=6.5, color=PAL['red'],
                                fontweight='bold', pad=2)
            else:
                ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])

    # Row labels
    fig.text(0.005, 0.72, 'Correct', fontsize=9, fontweight='bold', color=PAL['green'],
             rotation=90, va='center')
    fig.text(0.005, 0.28, 'Wrong', fontsize=9, fontweight='bold', color=PAL['red'],
             rotation=90, va='center')

    fig.suptitle('(i) Sample Predictions', fontsize=11, fontweight='bold', y=1.02)
    _savefig(fig, save_dir, 'fig9_samples.png')


# ======================== 10. Radar Chart ========================

def plot_radar(preds, targets, save_dir):
    cm = confusion_matrix(targets, preds)
    acc = np.diag(cm) / cm.sum(axis=1) * 100
    rpt = classification_report(targets, preds, target_names=CLASSES, output_dict=True)
    f1 = np.array([rpt[c]['f1-score'] * 100 for c in CLASSES])

    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

    vals_acc = acc.tolist() + [acc[0]]
    vals_f1 = f1.tolist() + [f1[0]]

    ax.plot(angles, vals_acc, 'o-', color=PAL['blue'], linewidth=1.8, markersize=5, label='Accuracy')
    ax.fill(angles, vals_acc, alpha=0.12, color=PAL['blue'])
    ax.plot(angles, vals_f1, 's--', color=PAL['orange'], linewidth=1.8, markersize=5, label='F1-Score')
    ax.fill(angles, vals_f1, alpha=0.08, color=PAL['orange'])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CLASSES, fontsize=8)
    ax.set_ylim(50, 100)
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.set_yticklabels(['60', '70', '80', '90', '100'], fontsize=7, color='#666')

    ax.set_title('(j) Per-Class Accuracy & F1 Radar', fontsize=11, fontweight='bold', pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1.15, -0.05), fontsize=8)

    _savefig(fig, save_dir, 'fig10_radar.png')


# ======================== 11. Training Dynamics (Loss rate of change) ========================

def plot_dynamics(history, save_dir):
    tl = np.array(history['train_loss'])
    vl = np.array(history['test_loss'])
    epochs = np.arange(1, len(tl) + 1)

    # gradient of loss (rate of change)
    dtl = -np.gradient(tl)
    dvl = -np.gradient(vl)
    dtl_s = _smooth(dtl, 0.75)
    dvl_s = _smooth(dvl, 0.75)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    # -- Left: loss improvement rate --
    ax1.plot(epochs, dtl_s, color=PAL['blue'], linewidth=1.8, label='Train')
    ax1.plot(epochs, dvl_s, color=PAL['orange'], linewidth=1.8, label='Val')
    ax1.fill_between(epochs, dtl_s, alpha=0.1, color=PAL['blue'])
    ax1.fill_between(epochs, dvl_s, alpha=0.1, color=PAL['orange'])
    ax1.axhline(0, color='#999', ls='-', lw=0.6)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('$-\\nabla_{\\mathrm{epoch}}\\mathcal{L}$ (loss decrease rate)')
    ax1.set_title('(k) Loss Improvement Rate')
    ax1.legend(fontsize=8)
    ax1.set_xlim(1, len(epochs))

    # -- Right: train vs val loss scatter --
    ax2.scatter(tl, vl, c=epochs, cmap='viridis', s=20, alpha=0.8, edgecolors='white', linewidth=0.5)
    # ideal line
    mn, mx = min(tl.min(), vl.min()), max(tl.max(), vl.max())
    ax2.plot([mn, mx], [mn, mx], '--', color=PAL['gray'], lw=1, alpha=0.6, label='$y=x$ (no gap)')
    ax2.set_xlabel('Train Loss')
    ax2.set_ylabel('Val Loss')
    ax2.set_title('(l) Train vs Val Loss Trajectory')
    ax2.legend(fontsize=8, loc='upper left')

    cbar = fig.colorbar(ax2.collections[0], ax=ax2, shrink=0.85, pad=0.02)
    cbar.set_label('Epoch', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout(w_pad=2)
    _savefig(fig, save_dir, 'fig11_dynamics.png')


# ======================== 12. Semantic Similarity (Misclassification Heatmap) ========================

def plot_semantic(preds, targets, save_dir):
    """Off-diagonal confusion as a symmetric misclassification affinity matrix."""
    cm = confusion_matrix(targets, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    np.fill_diagonal(cm_pct, 0)

    # make symmetric
    sym = (cm_pct + cm_pct.T) / 2

    mask = np.triu(np.ones_like(sym, dtype=bool), k=0)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    sns.heatmap(sym, mask=mask, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=CLASSES, yticklabels=CLASSES,
                linewidths=0.8, linecolor='white',
                cbar_kws={'label': 'Mutual Confusion (%)', 'shrink': 0.8},
                ax=ax, vmin=0, annot_kws={'size': 7.5})

    ax.set_title('(m) Pairwise Semantic Confusion', fontweight='bold')
    ax.tick_params(labelsize=8)
    plt.yticks(rotation=0)
    plt.xticks(rotation=40, ha='right')

    _savefig(fig, save_dir, 'fig12_semantic.png')


# ======================== 13. Summary Dashboard ========================

def plot_summary(history, preds, targets, save_dir):
    cm = confusion_matrix(targets, preds)
    acc = np.diag(cm) / cm.sum(axis=1) * 100
    best_va = max(history['test_acc'])
    best_ep = np.argmax(history['test_acc']) + 1
    final_ta = history['train_acc'][-1]
    mf1 = f1_score(targets, preds, average='macro') * 100
    mp  = precision_score(targets, preds, average='macro') * 100
    mr  = recall_score(targets, preds, average='macro') * 100
    total = len(preds)
    correct = sum(1 for p, t in zip(preds, targets) if p == t)

    fig = plt.figure(figsize=(10, 4.5))
    gs = GridSpec(2, 6, figure=fig, hspace=0.6, wspace=0.8)

    # metric cards
    cards = [
        ('Best Val Acc', f'{best_va:.2f}%', PAL['orange']),
        ('Macro F1',     f'{mf1:.2f}%',     PAL['blue']),
        ('Macro Prec',   f'{mp:.2f}%',      PAL['green']),
        ('Macro Recall', f'{mr:.2f}%',      PAL['purple']),
        ('Best Epoch',   f'{best_ep}',      PAL['brown']),
        ('Params',       '4.76M',           PAL['cyan']),
    ]

    for i, (lbl, val, clr) in enumerate(cards):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.60, val, fontsize=16, fontweight='bold', ha='center', va='center', color=clr)
        ax.text(0.5, 0.20, lbl, fontsize=7.5, ha='center', va='center', color='#666')
        for sp in ax.spines.values():
            sp.set_edgecolor(clr); sp.set_linewidth(1.5)
        ax.set_facecolor('#FAFAFA')
        ax.set_xticks([]); ax.set_yticks([])

    # radar
    ax_r = fig.add_subplot(gs[1, 0:3], polar=True)
    angles = np.linspace(0, 2*np.pi, 10, endpoint=False).tolist() + [0]
    vals = acc.tolist() + [acc[0]]
    ax_r.plot(angles, vals, 'o-', color=PAL['blue'], linewidth=1.5, markersize=4)
    ax_r.fill(angles, vals, alpha=0.15, color=PAL['blue'])
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(CLASSES, fontsize=6.5)
    ax_r.set_ylim(50, 100)
    ax_r.set_yticks([60, 70, 80, 90])
    ax_r.set_yticklabels(['60', '70', '80', '90'], fontsize=6, color='#999')

    # donut
    ax_d = fig.add_subplot(gs[1, 3:6])
    sizes = [correct, total - correct]
    colors_d = [PAL['green'], PAL['red']]
    wedges, _, autotexts = ax_d.pie(sizes, colors=colors_d, autopct='%1.1f%%',
                                     startangle=90, pctdistance=0.75,
                                     wedgeprops=dict(width=0.35, edgecolor='white', linewidth=2),
                                     textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax_d.text(0, 0, f'{correct}\n/{total}', ha='center', va='center',
              fontsize=12, fontweight='bold', color='#333')
    ax_d.legend(['Correct', 'Wrong'], fontsize=8, loc='lower right')

    fig.suptitle('(n) Experiment Summary', fontsize=13, fontweight='bold', y=1.02)
    _savefig(fig, save_dir, 'fig13_summary.png')


# ======================== MAIN ========================

def main():
    set_pub_style()
    work_dir = os.path.dirname(os.path.abspath(__file__))

    print('Loading data...')
    with open(os.path.join(work_dir, 'history.json')) as f:
        history = json.load(f)
    with open(os.path.join(work_dir, 'best_preds.json')) as f:
        data = json.load(f)
        preds, targets = data['preds'], data['targets']

    print('Generating publication-quality figures:')
    plot_loss(history, work_dir)
    plot_accuracy(history, work_dir)
    plot_gap(history, work_dir)
    plot_lr(history, work_dir)
    plot_cm(preds, targets, work_dir)
    plot_per_class(preds, targets, work_dir)
    plot_prf(preds, targets, work_dir)
    plot_errors(preds, targets, work_dir)
    plot_samples(preds, targets, work_dir)
    plot_radar(preds, targets, work_dir)
    plot_dynamics(history, work_dir)
    plot_semantic(preds, targets, work_dir)
    plot_summary(history, preds, targets, work_dir)

    print(f'\nDone! 13 figures saved to {work_dir}')
    print(f'Best test accuracy: {max(history["test_acc"]):.2f}%')


if __name__ == '__main__':
    main()
