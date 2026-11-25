import os
from paths import DATA, RAW, FIGS, TABLES
import json, time, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from datasets import load
from core import *
rng = np.random.default_rng(0)
res = {}
for name in ('pendigits', 'digits', 'mnist'):
    X, y, _ = load(name); E, _ = knn_connected(X, 10)
    sub = E
    lab = (y[sub[:, 0]] != y[sub[:, 1]]).astype(int)
    res[name] = []
    for k in (10, 20, 40, 80, 160):
        idx = neighborhoods(X, k)
        t0 = time.perf_counter(); kG = gorc(X, sub, idx, batched=True); tb = time.perf_counter() - t0
        t0 = time.perf_counter(); kGl = gorc(X, sub, idx, batched=False); tl = time.perf_counter() - t0
        t0 = time.perf_counter(); kU = uorc(X, sub, idx); tu = time.perf_counter() - t0
        r = dict(ne=len(sub), k=k, m=X.shape[1], tb=tb, tl=tl, tu=tu, aucG=roc_auc_score(lab, -kG), aucU=roc_auc_score(lab, -kU),
                 rho=spearmanr(kG, kU)[0])
        res[name].append(r); print(name, r, flush=True)
        json.dump(res, open(f'{RAW}/kscan.json', 'w'), indent=1)
