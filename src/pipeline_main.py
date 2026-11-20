"""
BhasaQuest-1 --- main pipeline: two-level hybrid stacking ensemble.

Level 1a  sparse   : word TF-IDF (1-3) + char_wb TF-IDF (2-5), six linear learners,
                     plus a binary specialist for the 19E1A1 / 72A2C8 boundary.
Level 1b  encoders : BanglaBERT (transliterated), MuRIL (romanized), mDeBERTa-v3 (romanized).
Level 2            : LogReg + LightGBM meta-models over 37 OOF probability columns,
                     then a constrained per-class bias search.

Design notes that were measured, not assumed:
  1) MAXLEN 256 with dynamic padding. At 128 tokens, 19.4% of class 19E1A1 was truncated
     (mean length 87.9, max 514). 256 covers 98.8% of the data at almost the same cost,
     because padding is per batch.
  2) Label smoothing 0.05 --- there is label noise between 19E1A1 and 72A2C8.
  3) Best-epoch checkpointing on fold validation F1, not the final epoch.
  4) A COVERED mask so a partial run cannot silently produce a meaningless score.

Runtime on a single T4 with fp16: level-1 ~4 min, each encoder 30-40 min, level-2 ~2 min.
Prints Bengali diagnostics by design; the numbers it reports match the report.

Usage:
    python src/pipeline_main.py        # expects train.csv / test.csv beside it or on Kaggle
"""
import os, re, time, math
import numpy as np, pandas as pd

# ----------------------------- config -----------------------------
# v6 = three distinct transformers (diversity) + TF-IDF stacker
# Why: v4/v5 carried only one transformer (BanglaBERT, solo 0.7231). Raising diversity
# is the most reliable source of gain in text classification.
#   track 1: csebuetnlp/banglabert   -> transliterated Bangla script (proven, 0.7231)
#   track 2: google/muril-base-cased -> ROMANIZED text directly (new)
#            MuRIL is pretrained on transliterated data; PANX bn_tr F1 72.94 vs mBERT 41.77
#   track 3: microsoft/mdeberta-v3-base -> ROMANIZED (new)
#            XNLI avg 79.8 vs XLM-R-base 76.2 -- strongest base-size multilingual model
# BanglishBERT dropped: its blend weight was exactly 0.00 in v4/v5.
FAST_DEV_RUN = False
RUN_BANGLISHBERT_TRACK = False
RUN_BANGLABERT_TRACK = True     # track 1 (transliterated)
RUN_MURIL_TRACK = True          # track 2 (romanized)
RUN_MDEBERTA_TRACK = True       # track 3 (romanized)
BB_SEEDS, BN_SEEDS, MURIL_SEEDS, MDEB_SEEDS = 2, 2, 2, 2
# v7: per-model epochs. MuRIL's first run proved 5 epochs is not enough --
# in fold 1, seed 1 was stuck at 0.2805 while the loss kept falling (1.3725 -> 1.2219).
# That is clear underfitting, so MuRIL is given 8 epochs.
BN_EPOCHS, MURIL_EPOCHS, MDEB_EPOCHS = 5, 8, 6
MAXLEN, EPOCHS, LR, BS = 256, 5, 3e-5, 32
MDEB_BS = 16                    # mDeBERTa-v3 embeddings are ~190M params, so a smaller batch
LABEL_SMOOTH = 0.05
NSPLIT, SEED, N_CLASSES = 5, 42, 4
np.random.seed(SEED)

# ----------------------------- data -----------------------------
def find(fname):
    for pat in [fname, f'/kaggle/input/**/{fname}', f'**/{fname}']:
        g = glob.glob(pat, recursive=True)
        if g:
            return g[0]
    raise FileNotFoundError(fname)

import glob
train = pd.read_csv(find('train.csv'))
test = pd.read_csv(find('test.csv'))
classes = sorted(train.label.unique())
cmap = {l: i for i, l in enumerate(classes)}
y = train.label.map(cmap).values
Ttr = train.text.astype(str).values
Tte = test.text.astype(str).values
I19, I72 = cmap['19E1A1'], cmap['72A2C8']
print(f'train={len(train)} test={len(test)} classes={classes}')

from sklearn.model_selection import StratifiedKFold
FOLDS = list(StratifiedKFold(NSPLIT, shuffle=True, random_state=SEED).split(Ttr, y))
if FAST_DEV_RUN:
    FOLDS = FOLDS[:2]
print(f'using {len(FOLDS)} folds')

# A smoke test runs only 2 folds, leaving 60% of the OOF rows at zero.
# Training level-2 on those zero rows makes the score meaningless (it showed 0.44).
# So we keep a coverage mask: every metric is computed only on rows that were filled.
COVERED = np.zeros(len(y), dtype=bool)

def f1p(P, yy=None, mask=None):
    from sklearn.metrics import f1_score
    if yy is None and mask is None:
        return f1_score(y, P.argmax(1), average='macro')
    if mask is None:
        return f1_score(yy, P.argmax(1), average='macro')
    return f1_score(y[mask], P[mask].argmax(1), average='macro')

# ----------------------------- deps -----------------------------
# mDeBERTa-v3's tokenizer will not load without sentencepiece/protobuf
for _pkg, _mod in [('sentencepiece', 'sentencepiece'), ('protobuf', 'google.protobuf')]:
    try:
        __import__(_mod)
    except ImportError:
        print(f'installing {_pkg} ...', flush=True)
        os.system(f'pip install -q {_pkg}')

# ----------------------------- transliteration -----------------------------
try:
    from avro import parse as _avro
except ImportError:
    os.system('pip install -q avro.py'); from avro import parse as _avro
try:
    from normalizer import normalize as _bn_norm
except ImportError:
    os.system('pip install -q csebuetnlp-normalizer'); from normalizer import normalize as _bn_norm

def to_bangla(texts):
    out = []
    for t in texts:
        try:
            out.append(_bn_norm(_avro(t)))
        except Exception:
            out.append(t)
    return out

# ----------------------------- Level-1 -----------------------------
def run_level1():
    global COVERED
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.naive_bayes import ComplementNB
    from scipy.sparse import hstack
    from scipy.special import softmax

    def feats(A, B, C):
        vw = TfidfVectorizer(ngram_range=(1, 3), min_df=1, sublinear_tf=True,
                             token_pattern=r"[a-z']+|[0-9]+", max_features=150000)
        vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2,
                             sublinear_tf=True, max_features=200000)
        Xa = hstack([vw.fit_transform(A), vc.fit_transform(A)]).tocsr()
        return Xa, hstack([vw.transform(B), vc.transform(B)]).tocsr(), \
               hstack([vw.transform(C), vc.transform(C)]).tocsr()

    model_names = ['lr_unweighted', 'lr_balanced', 'svc_calibrated', 'sgd_elasticnet',
                   'ridge', 'complement_nb']
    oofs = {n: np.zeros((len(y), N_CLASSES)) for n in model_names}
    tests = {n: np.zeros((len(Tte), N_CLASSES)) for n in model_names}
    spec_oof = np.zeros(len(y)); spec_test = np.zeros(len(Tte))

    for fi, (tri, vai) in enumerate(FOLDS):
        t0 = time.time()
        COVERED[vai] = True
        Xtr, Xva, Xte = feats([Ttr[i] for i in tri], [Ttr[i] for i in vai], Tte)
        ytr = y[tri]

        m = LogisticRegression(C=3.0, max_iter=4000).fit(Xtr, ytr)
        oofs['lr_unweighted'][vai] = m.predict_proba(Xva)
        tests['lr_unweighted'] += m.predict_proba(Xte) / len(FOLDS)

        m = LogisticRegression(C=3.0, max_iter=4000, class_weight='balanced').fit(Xtr, ytr)
        oofs['lr_balanced'][vai] = m.predict_proba(Xva)
        tests['lr_balanced'] += m.predict_proba(Xte) / len(FOLDS)

        m = CalibratedClassifierCV(LinearSVC(C=0.5, class_weight='balanced'), cv=3,
                                   method='sigmoid').fit(Xtr, ytr)
        oofs['svc_calibrated'][vai] = m.predict_proba(Xva)
        tests['svc_calibrated'] += m.predict_proba(Xte) / len(FOLDS)

        m = SGDClassifier(loss='log_loss', penalty='elasticnet', l1_ratio=0.15, alpha=1e-5,
                          max_iter=3000, class_weight='balanced', random_state=SEED).fit(Xtr, ytr)
        oofs['sgd_elasticnet'][vai] = m.predict_proba(Xva)
        tests['sgd_elasticnet'] += m.predict_proba(Xte) / len(FOLDS)

        m = RidgeClassifier(alpha=1.0, class_weight='balanced', random_state=SEED).fit(Xtr, ytr)
        oofs['ridge'][vai] = softmax(m.decision_function(Xva), axis=1)
        tests['ridge'] += softmax(m.decision_function(Xte), axis=1) / len(FOLDS)

        m = ComplementNB(alpha=0.3).fit(Xtr, ytr)
        oofs['complement_nb'][vai] = m.predict_proba(Xva)
        tests['complement_nb'] += m.predict_proba(Xte) / len(FOLDS)

        mask = np.isin(ytr, [I19, I72])
        if mask.sum() > 10:
            ybin = (ytr[mask] == I72).astype(int)
            sp = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(Xtr[mask], ybin)
            spec_oof[vai] = sp.predict_proba(Xva)[:, 1]
            spec_test += sp.predict_proba(Xte)[:, 1] / len(FOLDS)

        print(f'  fold {fi+1}/{len(FOLDS)} done ({time.time()-t0:.0f}s)', flush=True)

    for n in model_names:
        print(f'  {n:16s} OOF macroF1={f1p(oofs[n], mask=COVERED):.4f}')
    level1_oof = np.concatenate([oofs[n] for n in model_names] + [spec_oof.reshape(-1, 1)], axis=1)
    level1_test = np.concatenate([tests[n] for n in model_names] + [spec_test.reshape(-1, 1)], axis=1)
    print(f'  Level-1 uniform avg macroF1={f1p(np.mean([oofs[n] for n in model_names], axis=0), mask=COVERED):.4f}')
    return level1_oof, level1_test

# ----------------------------- transformer with dynamic padding -----------------------------
def run_transformer(model_name, train_texts, test_texts, tag, n_seeds=1, bs=32, ep=None):
    """n_seeds > 1: train n_seeds times per fold with different random seeds and average
    the probabilities (bagging) -- the most proven way to cut transformer fine-tuning
    variance on small data. bs = per-model batch size (lower it for large-vocab models)."""
    BS_ = int(bs)
    epochs = 1 if FAST_DEV_RUN else int(ep if ep is not None else EPOCHS)
    safe = re.sub(r'[^a-zA-Z0-9]+', '_', tag).strip('_').lower()
    # The cache key includes the epoch and seed count -- otherwise an older (underfit)
    # file trained with fewer epochs would be loaded by mistake. Legacy filenames are
    # only usable when their epoch/seed settings match the current defaults.
    key = f'{safe}_e{epochs}x{int(n_seeds)}'
    cands = [key] + ([safe] if (epochs == EPOCHS and int(n_seeds) == 2) else [])
    for _k in cands:
        c_o, c_t = f'oof_{_k}.npy', f'test_{_k}.npy'
        if os.path.exists(c_o) and os.path.exists(c_t):
            _o, _t = np.load(c_o), np.load(c_t)
            if _o.shape == (len(y), N_CLASSES) and _t.shape == (len(Tte), N_CLASSES):
                print(f'[{tag}] CACHE found ({_k}) -> loaded without retraining '
                      f'(solo OOF={f1p(_o, mask=COVERED):.4f})', flush=True)
                return _o, _t
            print(f'[{tag}] {_k} cache shape mismatch -> skipped', flush=True)
    print(f'[{tag}] no cache -> starting training ({epochs} epochs x {n_seeds} seeds)', flush=True)

    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              get_linear_schedule_with_warmup)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = (dev == 'cuda')
    print(f'[{tag}] device={dev} maxlen={MAXLEN} dynamic-pad=on smooth={LABEL_SMOOTH}')
    if dev == 'cpu':
        print(f'[{tag}] !! no GPU -- set Settings > Accelerator > GPU T4, do not run this otherwise')

    tok = AutoTokenizer.from_pretrained(model_name)

    class DS(Dataset):
        def __init__(self, texts, labels=None):
            enc = tok([str(t) for t in texts], truncation=True, max_length=MAXLEN, padding=False)
            self.ids = enc['input_ids']; self.att = enc['attention_mask']
            self.lab = None if labels is None else [int(v) for v in labels]
            self.lens = np.array([len(x) for x in self.ids])
        def __len__(self): return len(self.ids)
        def __getitem__(self, i):
            d = {'input_ids': torch.tensor(self.ids[i]), 'attention_mask': torch.tensor(self.att[i])}
            if self.lab is not None: d['labels'] = torch.tensor(self.lab[i])
            return d

    def collate(batch):
        mx = max(len(b['input_ids']) for b in batch)
        ids = torch.zeros(len(batch), mx, dtype=torch.long)
        att = torch.zeros(len(batch), mx, dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b['input_ids']); ids[i, :n] = b['input_ids']; att[i, :n] = b['attention_mask']
        out = {'input_ids': ids, 'attention_mask': att}
        if 'labels' in batch[0]:
            out['labels'] = torch.stack([b['labels'] for b in batch])
        return out

    def make_batches(lens, bs, rng=None, jitter=False):
        idx = np.argsort(lens + (rng.rand(len(lens)) * 8 if jitter else 0))
        bs_list = [idx[i:i + bs].tolist() for i in range(0, len(idx), bs)]
        if jitter and rng is not None:
            rng.shuffle(bs_list)
        return bs_list

    @torch.no_grad()
    def predict(model, ds):
        """Length-sorted batching (faster), but predictions are written back to the original
        indices. The first v3 build returned them in sorted order, which scrambled oof[vai]
        and produced an OOF macro-F1 of 0.24."""
        model.eval()
        n = len(ds)
        out = np.zeros((n, N_CLASSES), dtype=np.float32)
        for idx in make_batches(ds.lens, 64):
            batch = collate([ds[i] for i in idx])
            b = {k: v.to(dev) for k, v in batch.items() if k != 'labels'}
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                logits = model(**b).logits
            out[np.asarray(idx)] = torch.softmax(logits.float(), -1).cpu().numpy()
        assert out.sum(1).min() > 0.5, 'predict: some rows were left unfilled'
        return out

    ds_test = DS(test_texts)
    oof = np.zeros((len(y), N_CLASSES))
    tp = np.zeros((len(Tte), N_CLASSES))

    for f, (tri, vai) in enumerate(FOLDS):
        t0 = time.time()
        ds_tr_base = DS([train_texts[i] for i in tri], y[tri])
        ds_va = DS([train_texts[i] for i in vai])
        seed_val, seed_test = [], []

        for s in range(n_seeds):
            seed = SEED + f * 100 + s
            torch.manual_seed(seed)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=N_CLASSES).to(dev)
            # Some mDeBERTa-v3 checkpoint weights are stored as fp16. In that case the AMP
            # scaler.unscale_() call crashes: "Attempting to unscale FP16 gradients".
            # Casting every parameter to fp32 fixes it while fp16 AMP stays enabled.
            n_fp16 = sum(1 for p in model.parameters() if p.dtype != torch.float32)
            if n_fp16:
                for p in model.parameters():
                    if p.dtype != torch.float32:
                        p.data = p.data.float()
                print(f'  [{tag}] {n_fp16} tensors were fp16 -> cast to fp32', flush=True)
            ds_tr = ds_tr_base
            rng = np.random.RandomState(seed)

            head = [p for n, p in model.named_parameters() if n.startswith('classifier')]
            hid = {id(p) for p in head}
            body = [p for p in model.parameters() if id(p) not in hid]
            opt = torch.optim.AdamW([{'params': body, 'lr': LR},
                                     {'params': head, 'lr': LR * 5}], weight_decay=0.01)
            steps = math.ceil(len(ds_tr) / BS_) * epochs
            sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
            scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

            best_f1, best_val, best_test = -1.0, None, None
            for ep in range(epochs):
                model.train(); tot = 0.0; nb = 0
                for batch_idx in make_batches(ds_tr.lens, BS_, rng=rng, jitter=True):
                    batch = [ds_tr[i] for i in batch_idx]
                    b = collate(batch)
                    labels = b.pop('labels').to(dev)
                    b = {k: v.to(dev) for k, v in b.items()}
                    opt.zero_grad()
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                        logits = model(**b).logits
                        loss = F.cross_entropy(logits.float(), labels, label_smoothing=LABEL_SMOOTH)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update(); sch.step()
                    tot += loss.item(); nb += 1
                p = predict(model, ds_va)
                ef1 = f1p(p, y[vai])
                note = ''
                if ef1 > best_f1:
                    best_f1, best_val = ef1, p
                    best_test = predict(model, ds_test)
                    note = '  <- best'
                print(f'  [{tag}] fold{f+1} seed{s+1}/{n_seeds} ep{ep+1} loss={tot/max(nb,1):.4f} '
                      f'f1={ef1:.4f} ({time.time()-t0:.0f}s){note}', flush=True)
            seed_val.append(best_val); seed_test.append(best_test)
            print(f'  [{tag}] fold{f+1} seed{s+1}/{n_seeds} BEST={best_f1:.4f}', flush=True)
            del model
            if dev == 'cuda':
                torch.cuda.empty_cache()

        fold_val = np.mean(seed_val, axis=0)
        oof[vai] = fold_val
        tp += np.mean(seed_test, axis=0) / len(FOLDS)
        print(f'  [{tag}] fold{f+1} seed-avg F1={f1p(fold_val, y[vai]):.4f} '
              f'(single-seed best was {best_f1:.4f})', flush=True)

    print(f'[{tag}] OOF macroF1={f1p(oof, mask=COVERED):.4f}')
    np.save(f'oof_{key}.npy', oof); np.save(f'test_{key}.npy', tp)
    print(f'  saved oof_{key}.npy / test_{key}.npy (cache)', flush=True)
    return oof, tp

# ----------------------------- Level-2 -----------------------------
def run_level2(L1_oof, L1_test, deep):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    try:
        from lightgbm import LGBMClassifier
        HAS_LGB = True
    except ImportError:
        HAS_LGB = False

    def nrm(P):
        P = np.clip(P, 1e-9, None); return P / P.sum(-1, keepdims=True)

    X = np.concatenate([L1_oof] + [nrm(o) for o, _ in deep], axis=1)
    Xt = np.concatenate([L1_test] + [nrm(t) for _, t in deep], axis=1)
    print(f'\nLevel-2 meta-features: {X.shape[1]}')

    # train/score on COVERED rows only (with all 5 folds, COVERED = every row)
    rows = np.where(COVERED)[0]
    Xc, yc = X[rows], y[rows]
    print(f'Level-2 uses {len(rows)}/{len(y)} rows with valid OOF')

    skf = StratifiedKFold(5, shuffle=True, random_state=7)
    oof_lr = np.zeros((len(y), N_CLASSES)); te_lr = np.zeros((len(Tte), N_CLASSES))
    oof_gb = np.zeros((len(y), N_CLASSES)); te_gb = np.zeros((len(Tte), N_CLASSES))
    for tri, vai in skf.split(Xc, yc):
        gi, gv = rows[tri], rows[vai]
        m = LogisticRegression(max_iter=3000, C=1.0).fit(Xc[tri], yc[tri])
        oof_lr[gv] = m.predict_proba(Xc[vai]); te_lr += m.predict_proba(Xt) / 5
        if HAS_LGB:
            g = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15,
                               subsample=0.8, colsample_bytree=0.8, verbose=-1,
                               random_state=SEED)
            g.fit(Xc[tri], yc[tri])
            oof_gb[gv] = g.predict_proba(Xc[vai]); te_gb += g.predict_proba(Xt) / 5

    print(f'meta LR       OOF macroF1={f1p(oof_lr, mask=COVERED):.4f}')
    if HAS_LGB:
        print(f'meta LightGBM OOF macroF1={f1p(oof_gb, mask=COVERED):.4f}')

    # weighted log-prob blend of stacker outputs
    def Lg(P): return np.log(nrm(P))
    best = (-1, None)
    grid = np.arange(0, 1.01, 0.1) if HAS_LGB else [1.0]
    for w in grid:
        cand = w * Lg(oof_gb) + (1 - w) * Lg(oof_lr) if HAS_LGB else Lg(oof_lr)
        v = f1p(cand, mask=COVERED)
        if v > best[0]:
            best = (v, w)
    v, w = best
    print(f'best LGB/LR weight={w:.1f} -> blend OOF={v:.4f}')
    L_meta = w * Lg(oof_gb) + (1 - w) * Lg(oof_lr) if HAS_LGB else Lg(oof_lr)
    Lt_meta = w * Lg(te_gb) + (1 - w) * Lg(te_lr) if HAS_LGB else Lg(te_lr)

    # ----------------------------- Alt: direct weighted log-prob blend -----------------------------
    # Another route that bypasses the meta-stacker: search weights directly over the
    # components (stacker + each transformer). It scored better in an earlier local run.
    # We take whichever is better on OOF, with a 0.001 margin to avoid overfitting.
    def softmax(P):
        Z = np.exp(P - P.max(1, keepdims=True)); return Z / Z.sum(1, keepdims=True)

    # component = meta-stacker output + each transformer track (separately)
    comps_oof = [Lg(softmax(L_meta))] + [Lg(o) for o, _ in deep]
    comps_te = [Lg(softmax(Lt_meta))] + [Lg(t) for _, t in deep]

    # greedy forward selection with replacement (Caruana-style), scored on OOF
    n_models = len(comps_oof)
    counts = np.zeros(n_models)
    cur = np.zeros_like(comps_oof[0])
    best_score = -1.0
    for step in range(12):
        best_j, best_v = -1, best_score
        for j in range(n_models):
            cand = (cur * step + comps_oof[j]) / (step + 1)
            val = f1p(cand, mask=COVERED)
            if val > best_v + 1e-9:
                best_j, best_v = j, val
        if best_j < 0:
            break
        cur = (cur * step + comps_oof[best_j]) / (step + 1)
        counts[best_j] += 1
        best_score = best_v
        print(f'  greedy step {step+1}: add component{best_j} -> OOF={best_v:.4f}', flush=True)

    wts = counts / counts.sum()
    print('  greedy component weights:', np.round(wts, 3).tolist())
    L_greedy = sum(wts[j] * comps_oof[j] for j in range(n_models))
    Lt_greedy = sum(wts[j] * comps_te[j] for j in range(n_models))
    print(f'  meta-blend OOF={f1p(L_meta, mask=COVERED):.4f} | '
          f'greedy-blend OOF={f1p(L_greedy, mask=COVERED):.4f}')

    if f1p(L_greedy, mask=COVERED) > f1p(L_meta, mask=COVERED) + 0.001:
        print('  -> greedy blend chosen')
        return L_greedy, Lt_greedy
    print('  -> meta blend chosen')
    return L_meta, Lt_meta

def tune_bias(L, rounds=8):
    from sklearn.metrics import f1_score
    b = np.zeros(N_CLASSES)
    yc = y[COVERED]
    cur = f1_score(yc, (L[COVERED] + b).argmax(1), average='macro')
    for _ in range(rounds):
        improved = False
        for c in range(N_CLASSES):
            for d in np.arange(-0.9, 0.95, 0.05):
                nb = b.copy(); nb[c] = d
                v = f1_score(yc, (L[COVERED] + nb).argmax(1), average='macro')
                if v > cur + 1e-6:
                    cur, b, improved = v, nb, True
        if not improved:
            break
    return b, cur

# ----------------------------- main -----------------------------
if __name__ == '__main__':
    t0 = time.time()
    print('\n=== Level-1: 7 classical models ===')
    if os.path.exists('oof_level1.npy') and os.path.exists('test_level1.npy'):
        L1_oof = np.load('oof_level1.npy'); L1_test = np.load('test_level1.npy')
        if L1_oof.shape[0] == len(y) and L1_test.shape[0] == len(Tte):
            COVERED[:] = True   # loaded from cache, so run_level1 no longer sets COVERED
            print(f'  [CACHE] Level-1 loaded ({L1_oof.shape[1]} cols) -- not refitting')
        else:
            L1_oof, L1_test = run_level1()
            np.save('oof_level1.npy', L1_oof); np.save('test_level1.npy', L1_test)
    else:
        L1_oof, L1_test = run_level1()
        # so that a mid-run session death does not destroy the Level-1 work
        np.save('oof_level1.npy', L1_oof); np.save('test_level1.npy', L1_test)
        print('  saved oof_level1.npy / test_level1.npy')
    print(f'  COVERED rows = {COVERED.sum()}/{len(y)}')

    deep = []
    if RUN_BANGLISHBERT_TRACK:
        print(f'\n=== Deep: BanglishBERT (romanized), {BB_SEEDS}-seed ===')
        deep.append(run_transformer('csebuetnlp/banglishbert', Ttr, Tte, 'banglishbert',
                                    n_seeds=1 if FAST_DEV_RUN else BB_SEEDS, bs=BS, ep=EPOCHS))
    else:
        print('\n[SKIP] BanglishBERT -- its blend weight was 0.00 in v4/v5')

    if RUN_BANGLABERT_TRACK:
        print(f'\n=== Deep 1/3: BanglaBERT (transliterated), {BN_SEEDS}-seed, {BN_EPOCHS}ep ===')
        Ttr_bn = to_bangla(Ttr); Tte_bn = to_bangla(Tte)
        print('  sample:', Ttr[0][:50], '->', Ttr_bn[0][:50])
        deep.append(run_transformer('csebuetnlp/banglabert', Ttr_bn, Tte_bn, 'banglabert',
                                    n_seeds=1 if FAST_DEV_RUN else BN_SEEDS, bs=BS,
                                    ep=BN_EPOCHS))

    if RUN_MURIL_TRACK:
        print(f'\n=== Deep 2/3: MuRIL (romanized), {MURIL_SEEDS}-seed, {MURIL_EPOCHS}ep ===')
        deep.append(run_transformer('google/muril-base-cased', Ttr, Tte, 'muril',
                                    n_seeds=1 if FAST_DEV_RUN else MURIL_SEEDS, bs=BS,
                                    ep=MURIL_EPOCHS))

    if RUN_MDEBERTA_TRACK:
        print(f'\n=== Deep 3/3: mDeBERTa-v3 (romanized), {MDEB_SEEDS}-seed, '
              f'{MDEB_EPOCHS}ep, bs={MDEB_BS} ===')
        deep.append(run_transformer('microsoft/mdeberta-v3-base', Ttr, Tte, 'mdeberta',
                                    n_seeds=1 if FAST_DEV_RUN else MDEB_SEEDS, bs=MDEB_BS,
                                    ep=MDEB_EPOCHS))

    print('\n=== Level-2: stacking ===')
    L, Lt = run_level2(L1_oof, L1_test, deep)
    b, final_f1 = tune_bias(L)
    print(f'\nFINAL OOF macroF1={final_f1:.4f}  bias={np.round(b,2)}')

    from sklearn.metrics import classification_report
    print(classification_report(y[COVERED], (L[COVERED] + b).argmax(1),
                                target_names=classes, digits=3))

    sub = pd.DataFrame({'id': test.id, 'label': [classes[i] for i in (Lt + b).argmax(1)]})
    sub.to_csv('submission.csv', index=False)
    print('dist:', sub.label.value_counts(normalize=True).round(3).to_dict())
    print('wrote submission.csv')
    print(f'total {time.time()-t0:.0f}s')
