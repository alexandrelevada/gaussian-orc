import os
from paths import DATA, RAW, FIGS, TABLES
import json, numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from datasets import load, ALL
from core import incidence, vertex_score, neighborhoods
X_ = {}
for n in ALL:
    D = np.load(f'{RAW}/{n}.npz'); E = D['E']; y = D['y']; bnd = (D['nb_lab'] > 0).astype(int); N = len(y)
    lab = (y[E[:, 0]] != y[E[:, 1]]).astype(int)
    X, _, _ = load(n); idx = neighborhoods(X, 20); S = [set(r) for r in idx]
    jd = np.array([1 - len(S[i] & S[j]) / len(S[i] | S[j]) for i, j in E])
    c = rankdata(-D['kG']) + rankdata(D['length'])
    tq = lambda s, L, q=0.05: L[np.argsort(-s)[:int(np.ceil(q * len(s)))]].mean()
    inc = incidence(N, E); bj = vertex_score(inc, -jd); bl = vertex_score(inc, -D['length'])
    X_[n] = dict(combo=dict(auroc=roc_auc_score(lab, c), p5=tq(c, lab)),
                 v_jaccard=dict(auroc=roc_auc_score(bnd, bj), p5=tq(bj, bnd)),
                 v_length=dict(auroc=roc_auc_score(bnd, bl), p5=tq(bl, bnd)))
json.dump(X_, open(f'{RAW}/extra.json', 'w'), indent=1)
