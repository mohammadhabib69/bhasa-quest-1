"""
BhasaQuest-1 — FINAL PUSH: BanglaBERT extra seeds + full-data model

Why only these two runs:
  every other lever has already been measured (TF-IDF views, pseudo-labelling, trio
  specialist, bias tuning, per-class weights, rank averaging, an LLM) and none produced a
  real gain. What is left is variance reduction: BanglaBERT is the only strong transformer
  (solo OOF 0.7227) and the previous run used only 2 seeds. Three more seeds cut
  seed noise -- the most proven gain available on small data.

Outputs (existing files are never touched):
  oof_bn_s2.npy / test_bn_s2.npy      -> new seed (per-fold seed = 42 + f*100 + 2)
  oof_bn_s3.npy / test_bn_s3.npy      -> seed s=3
  oof_bn_s4.npy / test_bn_s4.npy      -> seed s=4
  test_bn_full.npy                    -> model trained on 100% of train (no OOF)

Kaggle settings: GPU T4 x1 (or x2), Internet ON, about 2 hours.
When the run finishes, download these 7 files and hand them over.
"""
import os, re, glob, math, time

T_ALL = time.time()
import numpy as np
import pandas as pd

# ----------------------------- toggles -----------------------------
RUN_EXTRA_SEEDS = True      # 3 new seeds      (~70-90 min)
RUN_FULL_DATA   = True      # 100% data model  (~15 min)
SEEDS_TO_RUN    = [2, 3, 4] # 0 and 1 already exist (oof_banglabert_e5x2.npy)
FULL_SEEDS      = 2

MODEL_NAME = 'csebuetnlp/banglabert'
BN_EPOCHS  = 5
MAXLEN     = 256
LR         = 3e-5
BS         = 32
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

train = pd.read_csv(find('train.csv'))
test = pd.read_csv(find('test.csv'))
classes = sorted(train.label.unique())
cmap = {l: i for i, l in enumerate(classes)}
y = train.label.map(cmap).values
Ttr = train.text.astype(str).values
Tte = test.text.astype(str).values
print(f'train={len(train)} test={len(test)} classes={classes}', flush=True)

from sklearn.model_selection import StratifiedKFold
FOLDS = list(StratifiedKFold(NSPLIT, shuffle=True, random_state=SEED).split(Ttr, y))
print(f'folds={len(FOLDS)} (exactly the v7 split, random_state={SEED})', flush=True)

# ----------------------------- transliteration -----------------------------
for _pkg, _mod in [('sentencepiece', 'sentencepiece')]:
    try:
        __import__(_mod)
    except ImportError:
        os.system(f'pip install -q {_pkg}')
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

print('transliterating ...', flush=True)
Ttr_bn = to_bangla(Ttr)
Tte_bn = to_bangla(Tte)
print('  sample:', Ttr[0][:50], '->', Ttr_bn[0][:50], flush=True)

# ----------------------------- torch -----------------------------
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
use_amp = (dev == 'cuda')
print(f'device={dev}', flush=True)
if dev == 'cuda':
    print(f'  gpu={torch.cuda.get_device_name(0)} '
          f'mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB', flush=True)
else:
    raise SystemExit('!! no GPU -- set Settings > Accelerator > GPU T4, do not run this otherwise')

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f'tokenizer ok (vocab={tok.vocab_size})', flush=True)

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

ds_test = DS(Tte_bn)

def new_model(seed):
    torch.manual_seed(seed)
    m = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=N_CLASSES).to(dev)
    n_fp16 = sum(1 for p in m.parameters() if p.dtype != torch.float32)
    if n_fp16:
        for p in m.parameters():
            if p.dtype != torch.float32:
                p.data = p.data.float()
    return m

def build_opt(m):
    head = [p for n_, p in m.named_parameters() if n_.startswith('classifier')]
    hid = {id(p) for p in head}
    body = [p for p in m.parameters() if id(p) not in hid]
    return torch.optim.AdamW([{'params': body, 'lr': LR},
                              {'params': head, 'lr': LR * 5}], weight_decay=0.01)

def train_epoch(model, ds_tr, opt, sch, scaler, rng):
    model.train(); tot = 0.0; nb = 0
    for batch_idx in make_batches(ds_tr.lens, BS, rng=rng, jitter=True):
        batch = collate([ds_tr[i] for i in batch_idx])
        b = {k: v.to(dev) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            out = model(**b)
            loss = F.cross_entropy(out.logits, b['labels'], label_smoothing=LABEL_SMOOTH)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        tot += loss.item(); nb += 1
    return tot / max(nb, 1)

from sklearn.metrics import f1_score

# ============================ EXTRA SEEDS ============================
if RUN_EXTRA_SEEDS:
    print(f'\n=== BanglaBERT extra seeds {SEEDS_TO_RUN} @ {BN_EPOCHS}ep ===', flush=True)
    for s in SEEDS_TO_RUN:
        c_o, c_t = f'oof_bn_s{s}.npy', f'test_bn_s{s}.npy'
        if os.path.exists(c_o) and os.path.exists(c_t):
            o = np.load(c_o)
            print(f'  [s={s}] cache exists (solo OOF={f1_score(y, o.argmax(1), average="macro"):.4f}) -> skip', flush=True)
            continue
        t_start = time.time()
        oof = np.zeros((len(y), N_CLASSES), dtype=np.float32)
        tp = np.zeros((len(Tte), N_CLASSES), dtype=np.float32)
        for f, (tri, vai) in enumerate(FOLDS):
            t0 = time.time()
            ds_tr = DS([Ttr_bn[i] for i in tri], y[tri])
            ds_va = DS([Ttr_bn[i] for i in vai])
            seed = SEED + f * 100 + s
            model = new_model(seed)
            opt = build_opt(model)
            steps = math.ceil(len(ds_tr) / BS) * BN_EPOCHS
            sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
            scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
            rng = np.random.RandomState(seed)
            best_f1, best_val, best_test = -1.0, None, None
            for ep in range(BN_EPOCHS):
                l = train_epoch(model, ds_tr, opt, sch, scaler, rng)
                p = predict(model, ds_va)
                ef1 = f1_score(y[vai], p.argmax(1), average='macro')
                note = ''
                if ef1 > best_f1:
                    best_f1, best_val = ef1, p
                    best_test = predict(model, ds_test)
                    note = '  <- best'
                print(f'  [s={s}] fold{f+1} ep{ep+1} loss={l:.4f} f1={ef1:.4f} '
                      f'({time.time()-t0:.0f}s){note}', flush=True)
            oof[vai] = best_val
            tp += best_test / len(FOLDS)
            print(f'  [s={s}] fold{f+1}/5 done ({time.time()-t0:.0f}s) BEST={best_f1:.4f} '
                  f'| total {(time.time()-t_start)/60:.1f} min', flush=True)
            del model
            torch.cuda.empty_cache()
        solo = f1_score(y, oof.argmax(1), average='macro')
        print(f'  [s={s}] SOLO OOF={solo:.4f}', flush=True)
        np.save(c_o, oof); np.save(c_t, tp)
        print(f'  [s={s}] saved {c_o} / {c_t}', flush=True)

# ============================ FULL DATA ============================
if RUN_FULL_DATA:
    print(f'\n=== BanglaBERT FULL DATA ({len(Ttr)} rows), {FULL_SEEDS} seeds @ {BN_EPOCHS}ep ===', flush=True)
    if os.path.exists('test_bn_full.npy'):
        print('  cache exists -> skip', flush=True)
    else:
        ds_all = DS(Ttr_bn, y)
        t_start = time.time()
        tp = np.zeros((len(Tte), N_CLASSES), dtype=np.float32)
        for s in range(FULL_SEEDS):
            seed = 1000 + s
            model = new_model(seed)
            opt = build_opt(model)
            steps = math.ceil(len(ds_all) / BS) * BN_EPOCHS
            sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
            scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
            rng = np.random.RandomState(seed)
            for ep in range(BN_EPOCHS):
                l = train_epoch(model, ds_all, opt, sch, scaler, rng)
                print(f'  [full] seed{s+1} ep{ep+1} loss={l:.4f} '
                      f'({time.time()-t_start:.0f}s)', flush=True)
            tp += predict(model, ds_test) / FULL_SEEDS
            del model
            torch.cuda.empty_cache()
        np.save('test_bn_full.npy', tp)
        print('  saved test_bn_full.npy', flush=True)

print('\n=== DONE -- download these files ===', flush=True)
for f in sorted(glob.glob('oof_bn_s*.npy') + glob.glob('test_bn_s*.npy') + glob.glob('test_bn_full.npy')):
    print('  ', f, np.load(f).shape, flush=True)
print(f'total {(time.time()-T_ALL)/60:.1f} min', flush=True)
