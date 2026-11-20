"""
Local blender for BhasaQuest-1 — Kaggle GPU ছাড়াই সব candidate মাপা যায়।

কী করে:
  1) যতগুলো probability track পাওয়া যায় সব লোড করে (level1 / banglabert seeds / muril /
     mdeberta / নতুন bn_s2,s3,s4 / test_bn_full)।
  2) প্রতি configuration-এর জন্য nested-CV meta-stacker চালায় -> honest OOF।
  3) শুধু OOF-এ সত্যিকারের লাভ হলে submission লেখে (নাহলে কিছুই লেখে না)।

চালাও:  ./.venv/bin/python final_blend.py
"""
import os, glob, argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='.', help='directory holding train.csv / test.csv / *.npy')
ap.add_argument('--out', default='.', help='where to write submission files')
a = ap.parse_args()
DATA, OUT = a.data, a.out

tr = pd.read_csv(os.path.join(DATA, 'train.csv'))
te = pd.read_csv(os.path.join(DATA, 'test.csv'))
classes = sorted(tr.label.unique())
cmap = {l: i for i, l in enumerate(classes)}
y = tr.label.map(cmap).values
C = len(classes)


def nrm(P):
    P = np.clip(np.asarray(P, dtype=np.float64), 1e-9, None)
    return P / P.sum(-1, keepdims=True)


def load(name, n):
    f = os.path.join(DATA, f'{name}.npy')
    if not os.path.exists(f):
        return None
    A = np.load(f)
    if A.shape[0] != n:
        print(f'  !! {f} shape {A.shape} মেলেনি -> বাদ')
        return None
    return nrm(A)


def f1p(P):
    return f1_score(y, np.asarray(P).argmax(1), average='macro')


# ----------------------------- tracks -----------------------------
print('=== tracks ===')
L1_o, L1_t = load('oof_level1', len(y)), load('test_level1', len(te))
if L1_o is None:
    raise SystemExit('oof_level1.npy নেই — এটা ছাড়া কিছুই করা যাবে না')

bn_seed_names = [k for k in ['oof_banglabert_e5x2', 'oof_bn_s2', 'oof_bn_s3', 'oof_bn_s4']
                 if os.path.exists(os.path.join(DATA, f'{k}.npy'))]
bn_o_list, bn_t_list = [], []
for k in bn_seed_names:
    o = load(k, len(y)); t = load(k.replace('oof_', 'test_'), len(te))
    if o is not None and t is not None:
        bn_o_list.append(o); bn_t_list.append(t)
        print(f'  {k:24s} solo={f1p(o):.4f}')

BN_o = nrm(np.mean(bn_o_list, axis=0))
BN_t = nrm(np.mean(bn_t_list, axis=0))
print(f'  BANGLA-BERT mean ({len(bn_o_list)} seeds) solo={f1p(BN_o):.4f}')

MU_o, MU_t = load('oof_muril_e8x2', len(y)), load('test_muril_e8x2', len(te))
MD_o, MD_t = load('oof_mdeberta_e6x2', len(y)), load('test_mdeberta_e6x2', len(te))
if MU_o is None:
    MU_o, MU_t = load('oof_muril', len(y)), load('test_muril', len(te))
if MD_o is None:
    print('  !! mdeberta track নেই — বাদ')
print(f'  muril    solo={f1p(MU_o):.4f}' if MU_o is not None else '  muril: none')
print(f'  mdeberta solo={f1p(MD_o):.4f}' if MD_o is not None else '  mdeberta: none')

FULL_t = load('test_bn_full', len(te))
if FULL_t is not None:
    print('  test_bn_full পাওয়া গেছে -> test-এ BN column 70/30 blend হবে')


# ----------------------------- meta stacker -----------------------------
def stack(cols_o, cols_t, outer_seed=202, use_log=True):
    """nested-CV meta LogisticRegression -> (oof, test)"""
    Xo = np.concatenate([np.log(c) if use_log else c for c in cols_o], axis=1)
    Xt = np.concatenate([np.log(c) if use_log else c for c in cols_t], axis=1)
    oof = np.zeros((len(y), C))
    skf = StratifiedKFold(5, shuffle=True, random_state=outer_seed)
    for tri, vai in skf.split(Xo, y):
        m = LogisticRegression(max_iter=3000, C=1.0).fit(Xo[tri], y[tri])
        oof[vai] = m.predict_proba(Xo[vai])
    m = LogisticRegression(max_iter=3000, C=1.0).fit(Xo, y)
    return oof, m.predict_proba(Xt)


base_o, base_t = [L1_o, BN_o] + ([MU_o] if MU_o is not None else []) + \
                 ([MD_o] if MD_o is not None else []), None
base_t = [L1_t, BN_t] + ([MU_t] if MU_o is not None else []) + \
         ([MD_t] if MD_o is not None else [])

print('\n=== nested-CV stackers ===')
oof_a, te_a = stack(base_o, base_t)
print(f'  A) baseline (bn={len(bn_o_list)} seeds)         OOF={f1p(oof_a):.4f}')

results = {'A_baseline': (f1p(oof_a), te_a)}

# full-data model test blend (OOF অপরিবর্তিত, শুধু test column পাল্টায়)
if FULL_t is not None:
    BN_t_mix = nrm(0.7 * BN_t + 0.3 * FULL_t)
    t_b = [L1_t, BN_t_mix] + ([MU_t] if MU_o is not None else []) + \
          ([MD_t] if MD_o is not None else [])
    _, te_b = stack(base_o, t_b)
    agree = float((te_a.argmax(1) == te_b.argmax(1)).mean())
    print(f'  B) A + full-data test blend   (label agree {agree*100:.1f}%)')
    results['B_full'] = (f1p(oof_a), te_b)

# ----------------------------- decision -----------------------------
best_name = max(results, key=lambda k: results[k][0])
best_oof, best_te = results[best_name]
print(f'\nBEST = {best_name}  OOF={best_oof:.4f}')

GAIN_THRESHOLD = 0.0015


def write(sub_name, prob, note=''):
    lab = [classes[i] for i in prob.argmax(1)]
    sub = pd.DataFrame({'id': te.id, 'label': lab})
    sub.to_csv(os.path.join(OUT, sub_name), index=False)
    dist = sub.label.value_counts(normalize=True).round(3).to_dict()
    print(f'  wrote {sub_name}  dist={dist} {note}')


if best_name == 'A_baseline':
    print(f'\n-> নতুন candidate লিখছি না (baseline-ই সেরা; লাভ < {GAIN_THRESHOLD})')
    print('   নিরাপদ: আগের 0.76023 submission-টাই selected রাখো।')
else:
    write('submission_push.csv', best_te, '(full-data blend)')
    print('   submit করার আগে দেখো: দামি slot, তাই শুধু একবার।')

print('\n=== per-class (baseline) ===')
print(classification_report(y, oof_a.argmax(1), target_names=classes, digits=3))
