"""Generate all report figures from the saved OOF probability files.

Run from repo root:
    python src/local/make_figures.py --data ../.. --out report/figures
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score

INK = '#101418'
GREY = '#8A9199'
LIGHT = '#E6E9EC'
ACCENT = '#1F4E79'

plt.rcParams.update({
    'figure.dpi': 160, 'savefig.dpi': 160, 'font.size': 9,
    'axes.edgecolor': INK, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK, 'ytick.color': INK, 'axes.titleweight': 'bold',
    'font.family': 'DejaVu Sans', 'axes.spines.top': False,
    'axes.spines.right': False,
})


def nrm(P):
    P = np.clip(np.asarray(P, dtype=np.float64), 1e-9, None)
    return P / P.sum(-1, keepdims=True)


def fig_architecture(out):
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.set_xlim(0, 10); ax.set_ylim(-0.06, 6.46); ax.axis('off')

    def box(x, y, w, h, label, fc='white', fs=7.6, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05,rounding_size=0.10',
                                    linewidth=1.0, edgecolor=INK, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=fs,
                fontweight='bold' if bold else 'normal', linespacing=1.4)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=9,
                                     linewidth=0.9, color=INK, shrinkA=0, shrinkB=0))

    box(0.30, 5.60, 9.40, 0.66, 'Romanized Bangla text   ·   8,195 train / 2,048 test',
        fc=LIGHT, fs=8.4, bold=True)

    box(0.30, 3.40, 4.30, 1.55, '', fc='white')
    ax.text(2.45, 4.68, 'LEVEL 1a — sparse lexical', ha='center', fontsize=7.8, fontweight='bold')
    ax.text(2.45, 4.28, 'word TF-IDF 1–3   +   char_wb TF-IDF 2–5', ha='center', fontsize=7.0, color=GREY)
    ax.text(2.45, 3.92, 'six linear learners (LR x2, SVC, SGD, ridge, CNB)', ha='center',
            fontsize=6.9, color=GREY)
    ax.text(2.45, 3.58, '+ binary specialist  19E1A1 / 72A2C8', ha='center', fontsize=6.9, color=GREY)

    box(5.40, 4.98, 4.30, 0.46, 'Avro phonetic transliteration', fc='white', fs=7.2)
    box(5.40, 3.40, 4.30, 1.25, '', fc='white')
    ax.text(7.55, 4.36, 'LEVEL 1b — pretrained encoders', ha='center', fontsize=7.8, fontweight='bold')
    ax.text(7.55, 4.00, 'BanglaBERT (transliterated)  ·  MuRIL  ·  mDeBERTa-v3', ha='center',
            fontsize=6.9, color=GREY)
    ax.text(7.55, 3.66, '5-fold  ·  5–8 epochs  ·  2 seeds each', ha='center', fontsize=6.9, color=GREY)

    box(1.90, 2.20, 6.20, 0.66,
        'out-of-fold probability matrix  ·  37 columns', fc=LIGHT, bold=True, fs=8.2)
    box(1.90, 0.96, 6.20, 0.66,
        'LEVEL 2 — LogReg + LightGBM meta-learner', fc=ACCENT, bold=True, fs=8.2)
    box(1.90, 0.18, 6.20, 0.50,
        'constrained per-class bias  →  submission.csv', fc='white', fs=7.2)

    arrow(2.45, 5.60, 2.45, 4.97)
    arrow(7.55, 5.60, 7.55, 5.48)
    arrow(7.55, 4.94, 7.55, 4.68)
    arrow(2.45, 3.38, 2.45, 2.88)
    arrow(7.55, 3.38, 7.55, 2.88)
    arrow(5.00, 2.18, 5.00, 1.64)
    arrow(5.00, 0.94, 5.00, 0.70)

    ax.text(8.25, 1.34, 'OOF 0.7542', fontsize=7.4, fontweight='bold')
    ax.text(8.25, 1.10, 'private 0.78786', fontsize=7.4)
    ax.text(8.25, 0.87, 'rank 1', fontsize=7.0, color=GREY)

    fig.tight_layout()
    fig.savefig(os.path.join(out, 'architecture.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def fig_scores(out):
    rows = [
        ('best_v2', 0.73825, 0.75858), ('best_v3', 0.73936, 0.76765),
        ('best_v4', 0.74806, 0.78078), ('v9', 0.75744, 0.78768),
        ('v8', 0.75881, 0.78340), ('v7', 0.76023, 0.78786),
    ]
    names = [r[0] for r in rows]
    pub = [r[1] for r in rows]; priv = [r[2] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    ax.plot(x, pub, 'o-', color=GREY, label='Public LB (30%)', markersize=5, linewidth=1.2)
    ax.plot(x, priv, 's-', color=ACCENT, label='Private LB (70%)', markersize=5, linewidth=1.6)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel('macro-F1'); ax.set_ylim(0.730, 0.797)
    ax.grid(axis='y', color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)
    for i, (p, q) in enumerate(zip(pub, priv)):
        ax.annotate(f'{p:.4f}', (i, p), textcoords='offset points', xytext=(0, -12),
                    ha='center', fontsize=6.6, color=GREY)
        ax.annotate(f'{q:.4f}', (i, q), textcoords='offset points', xytext=(0, 7),
                    ha='center', fontsize=6.8, color=ACCENT, fontweight='bold')
    ax.legend(frameon=False, fontsize=7.5, loc='lower right')
    ax.set_title('Submission progression — public vs private', fontsize=9.5, pad=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'scores.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def fig_confusion(out, data, y, classes, oof):
    cm = confusion_matrix(y, oof.argmax(1), normalize='true')
    fig, ax = plt.subplots(figsize=(3.9, 3.5))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=7.6)
    ax.set_yticklabels(classes, fontsize=7.6)
    ax.set_xlabel('predicted', fontsize=8); ax.set_ylabel('true', fontsize=8)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center', fontsize=7.4,
                    color='white' if cm[i, j] > 0.55 else INK)
    ax.spines[:].set_visible(False)
    ax.set_title(f'OOF confusion (macro-F1 {f1_score(y, oof.argmax(1), average="macro"):.4f})',
                 fontsize=8.8, pad=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'confusion.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def fig_class_dist(out, y, classes):
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    counts = np.bincount(y, minlength=len(classes))
    ax.bar(classes, counts, color=[ACCENT, '#4A7AA6', '#95B3C9', '#C9D6E0'], edgecolor=INK, linewidth=0.6)
    for i, c in enumerate(counts):
        ax.text(i, c + 40, str(c), ha='center', fontsize=7.4)
    ax.set_ylabel('rows'); ax.set_ylim(0, counts.max() * 1.16)
    ax.grid(axis='y', color=LIGHT, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_title('Training class distribution (imbalanced 2.2:1)', fontsize=8.8, pad=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'class_dist.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../..')
    ap.add_argument('--out', default='report/figures')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tr = pd.read_csv(os.path.join(a.data, 'train.csv'))
    classes = sorted(tr.label.unique())
    y = tr.label.map({l: i for i, l in enumerate(classes)}).values

    L1 = nrm(np.load(os.path.join(a.data, 'oof_level1.npy')))
    BN = nrm(np.load(os.path.join(a.data, 'oof_banglabert_e5x2.npy')))
    MU = nrm(np.load(os.path.join(a.data, 'oof_muril_e8x2.npy')))
    MD = nrm(np.load(os.path.join(a.data, 'oof_mdeberta_e6x2.npy')))
    X = np.concatenate([np.log(p) for p in (L1, BN, MU, MD)], axis=1)
    oof = np.zeros((len(y), 4))
    for tri, vai in StratifiedKFold(5, shuffle=True, random_state=202).split(X, y):
        m = LogisticRegression(max_iter=3000).fit(X[tri], y[tri])
        oof[vai] = m.predict_proba(X[vai])

    fig_architecture(a.out)
    fig_scores(a.out)
    fig_confusion(a.out, a.data, y, classes, oof)
    fig_class_dist(a.out, y, classes)
    print('figures written to', a.out)
    print('nested OOF =', round(f1_score(y, oof.argmax(1), average='macro'), 4))
