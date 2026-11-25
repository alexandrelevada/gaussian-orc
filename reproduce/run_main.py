import os
from paths import DATA, RAW, FIGS, TABLES
import sys, json, time, numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from datasets import load, ALL
from core import *

KG, K, Q, H = 10, 20, 0.05, 2

def topq(s, q):
    return np.argsort(-s)[:max(1, int(np.ceil(q * len(s))))]

def jac(a, b):
    a, b = set(map(int, a)), set(map(int, b)); return len(a & b) / max(1, len(a | b))

def edge_metrics(s, lab):
    t = topq(s, Q)
    return dict(auroc=roc_auc_score(lab, s), ap=average_precision_score(lab, s), p5=lab[t].mean())

def run(name):
    X, y, _ = load(name); n, m = X.shape
    t0 = time.perf_counter(); E, added = knn_connected(X, KG); idx = neighborhoods(X, K); tgraph = time.perf_counter() - t0
    t0 = time.perf_counter(); kG = gorc(X, E, idx, batched=True); tGb = time.perf_counter() - t0
    t0 = time.perf_counter(); kG2 = gorc(X, E, idx, batched=False); tGl = time.perf_counter() - t0
    t0 = time.perf_counter(); kU = uorc(X, E, idx, p=1); tU = time.perf_counter() - t0
    assert np.allclose(kG, kG2, atol=1e-6)
    inter = (y[E[:, 0]] != y[E[:, 1]]).astype(int)
    length = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
    S = [set(r) for r in idx]
    jacd = np.array([1 - len(S[i] & S[j]) / len(S[i] | S[j]) for i, j in E])
    R = dict(name=name, n=n, m=m, C=int(y.max() + 1), E=len(E), added=len(added), form='kxk' if K <= m else 'mxm',
             t_graph=tgraph, t_gorc_batched=tGb, t_gorc_loop=tGl, t_uorc=tU,
             base_edge=inter.mean(),
             edge={nm: edge_metrics(s, inter) for nm, s in
                   [('gorc', -kG), ('uorc', -kU), ('length', length), ('jaccard', jacd)]},
             spearman_GU=spearmanr(kG, kU)[0], pearson_GU=np.corrcoef(kG, kU)[0, 1],
             jac_top_edges=jac(topq(-kG, Q), topq(-kU, Q)),
             rho_s_length=spearmanr(-kG, length)[0])
    if len(added):
        pos = {tuple(e): t for t, e in enumerate(E)}; pct = (-kG).argsort().argsort() / (len(E) - 1)
        R['added_pct'] = float(np.mean([pct[pos[tuple(a)]] for a in added]))
    inc = incidence(n, E)
    nb_lab = np.array([np.mean(y[np.where(E[e, 0] == i, E[e, 1], E[e, 0])] != y[i]) for i, e in enumerate(inc)])
    bnd = (nb_lab > 0).astype(int)
    rad = np.linalg.norm(X - X[idx[:, -1]], axis=1)          # raio k-NN (densidade inversa)
    R['base_vertex'] = bnd.mean()
    V = {}
    for nm, kap in (('gorc', kG), ('uorc', kU)):
        for how in ('mean', 'max', 'quantile', 'softmax', 'top_frac'):
            b = vertex_score(inc, kap, how)
            V[f'{nm}_{how}'] = dict(auroc=roc_auc_score(bnd, b), p5=bnd[topq(b, Q)].mean(),
                                   rho_imp=spearmanr(b, nb_lab)[0], rho_rad=spearmanr(b, rad)[0])
    V['radius'] = dict(auroc=roc_auc_score(bnd, rad), p5=bnd[topq(rad, Q)].mean(), rho_imp=spearmanr(rad, nb_lab)[0], rho_rad=1.0)
    R['vertex'] = V
    bG = vertex_score(inc, kG); bU = vertex_score(inc, kU)
    R['jac_top_vertices'] = jac(topq(bG, Q), topq(bU, Q)); R['spearman_bGU'] = spearmanr(bG, bU)[0]
    A = {}
    for nm, b in (('gorc', bG), ('uorc', bU), ('radius', rad)):
        a = anchors(n, E, b, H)
        glob = np.argsort(b)[:len(a)]
        A[nm] = dict(count=len(a), purity=1 - bnd[a].mean(), coverage=len(set(y[a])) / (y.max() + 1),
                     glob_purity=1 - bnd[glob].mean(), glob_coverage=len(set(y[glob])) / (y.max() + 1),
                     rho_rad=spearmanr(b[a], rad[a])[0] if len(a) > 2 else np.nan)
    A['jac_GU'] = jac(anchors(n, E, bG, H), anchors(n, E, bU, H))
    R['anchors'] = A; R['base_pure'] = 1 - bnd.mean()
    np.savez_compressed(f'{RAW}/{name}.npz', E=E, added=added, kG=kG, kU=kU, y=y, bG=bG, bU=bU,
                        aG=anchors(n, E, bG, H), aU=anchors(n, E, bU, H), rad=rad, nb_lab=nb_lab, length=length)
    return R

if __name__ == '__main__':
    pass
    for name in (sys.argv[1:] or ALL):
        R = run(name)
        json.dump(R, open(f'{RAW}/{name}.json', 'w'), indent=1, default=float)
        print(f"{name:14s} n={R['n']:5d} |E|={R['E']:6d} G={R['t_gorc_batched']:.2f}s Gloop={R['t_gorc_loop']:.2f}s "
              f"U={R['t_uorc']:.2f}s  edgeAUC G={R['edge']['gorc']['auroc']:.3f} U={R['edge']['uorc']['auroc']:.3f} "
              f"len={R['edge']['length']['auroc']:.3f} rhoGU={R['spearman_GU']:.3f} "
              f"vAUC G={R['vertex']['gorc_softmax']['auroc']:.3f} U={R['vertex']['uorc_softmax']['auroc']:.3f} "
              f"anchPur G={R['anchors']['gorc']['purity']:.2f} base={R['base_pure']:.2f}", flush=True)
