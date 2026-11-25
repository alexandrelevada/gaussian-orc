#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimentos do artigo (Seção 5) em novos conjuntos do OpenML: G-ORC × ORC uniforme.

@author: Alexandre L. M. Levada

Para cada conjunto, com o MESMO grafo k-NN conexo e as MESMAS vizinhanças:
  * curvaturas das arestas: G-ORC (κ_G, W₂ de Bures, forma fechada) e ORC uniforme (κ₁ᵁ, W₁ exato);
  * tempos (mediana de --repeats): ORC uniforme, G-ORC em laço por aresta e G-ORC em lote;
  * métricas das Tabelas 3–9 do artigo (rótulos usados SÓ na avaliação):
      T3 conjuntos      n, m, C, |E|, forma, % arestas inter-classe, % vértices de borda
      T4 concordância   Spearman (arestas, vértices) e Jaccard (top 5% arestas, top 5% vértices, âncoras)
      T5 arestas        AUROC e precisão do top 5%: G-ORC, ORC, comprimento, Jaccard, G+L
      T6 vértices       AUROC e precisão do top 5%: G-ORC, ORC, Jaccard agregado, raio k-NN; ρ_rad
      T7 agregação      AUROC e precisão do top 5% por agregação (mean, max, quantile, softmax, top_frac)
      T8 âncoras        nº, pureza e cobertura de classes: G, U, modos de densidade, corte global
      T9 tempo          tempos e speed-ups
  * figura do grafo: arestas mais negativas (G-ORC e ORC) e vértices de interface/âncoras;
  * CSV por vértice (pontuações e seleções).

Os resultados de cada conjunto vão para <out>/results/<nome>.json (padrão: results/openml/). Depois
de cada execução, as tabelas LaTeX são REFEITAS a partir de TODOS os JSON da pasta (<out>/tables/),
com as médias e os ranks médios recalculados. Os JSON dos 12 conjuntos do artigo
(results/paper_results/) são copiados automaticamente para a pasta (desative com --no_paper).

Dependências: numpy scipy scikit-learn pandas matplotlib networkx joblib pot
    pip install pot            (ou: conda install -c conda-forge pot)

Exemplos
--------
    # (os resultados vão para results/openml/; os 12 conjuntos do artigo entram automaticamente nas tabelas)
    # um conjunto tabular (padronização z-score, como no artigo)
    python gorc_openml_experiments.py --openml vehicle
    # vários conjuntos de uma vez
    python gorc_openml_experiments.py --openml vehicle waveform-5000 spambase
    # por data_id; dados de imagem/pixels: escala para [0, 1] (como MNIST/Digits no artigo)
    python gorc_openml_experiments.py --openml_id 41082 --preprocess max --name USPS
    # conjuntos grandes: amostra de até 10 000 pontos (padrão)
    python gorc_openml_experiments.py --openml Fashion-MNIST --preprocess max --max_samples 10000
    # só refazer as tabelas com os JSON já existentes
    python gorc_openml_experiments.py --tables_only
    # teste rápido sem internet
    python gorc_openml_experiments.py --sklearn wine --repeats 1
"""
import argparse
import glob
import json
import os
import re
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from gaussian_orc import (gaussian_orc, uniform_orc, knn_graph_edges, neighborhoods,
                          node_boundary_score, local_minima)

AGGS = ['mean', 'max', 'quantile', 'softmax', 'top_frac']


# ──────────────────────────────────────────────────────────────────────────────
# 0) Leitura e pré-processamento (como na Seção 5.1)
# ──────────────────────────────────────────────────────────────────────────────

def _frame_to_numeric(df):
    """Atributos numéricos; nominais com categorias numéricas são convertidos; os demais descartados."""
    cols, n_conv, dropped = [], 0, []
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_numeric_dtype(col) and not isinstance(col.dtype, pd.CategoricalDtype):
            cols.append(col.astype(float)); continue
        conv = pd.to_numeric(col.astype(object), errors='coerce')
        if conv.notna().sum() == col.notna().sum():
            cols.append(conv.astype(float)); n_conv += 1
        else:
            dropped.append(c)
    if n_conv:
        print(f'  aviso: {n_conv} atributo(s) nominal(is) numérico(s) convertido(s)')
    if dropped:
        print(f'  aviso: {len(dropped)} atributo(s) categórico(s) descartado(s): {", ".join(map(str, dropped[:5]))}'
              f'{" ..." if len(dropped) > 5 else ""}')
    if not cols:
        raise SystemExit('nenhum atributo numérico utilizável')
    return pd.concat(cols, axis=1).to_numpy(dtype=float)


def load_dataset(src, ident, version=1, target=None):
    """Retorna X (n, m), rótulos brutos e nome."""
    import sklearn.datasets as skd
    if src == 'sklearn':
        X, y = getattr(skd, f'load_{ident}')(return_X_y=True)
        return np.asarray(X, float), np.asarray(y), ident
    kw = dict(as_frame=True, parser='auto')
    if target is not None:
        kw['target_column'] = target
    if src == 'openml_id':
        ds = skd.fetch_openml(data_id=int(ident), **kw)
        name = (ds.details or {}).get('name', f'openml_{ident}') if hasattr(ds, 'details') else f'openml_{ident}'
    else:
        v = version if version == 'active' else int(version)
        ds = skd.fetch_openml(name=ident, version=v, **kw)
        name = ident
    if ds.target is None:
        raise SystemExit(f'{name}: sem atributo-alvo padrão no OpenML (use --target)')
    X = _frame_to_numeric(ds.data)
    y = np.asarray(ds.target).ravel()
    return X, y, name


def preprocess(X, y, mode, max_samples, seed):
    """Remove ausentes, duplicatas exatas e atributos constantes; amostra; escala."""
    ok = ~np.isnan(X).any(1) & pd.notna(pd.Series(y, dtype=object)).to_numpy()
    if (~ok).sum():
        print(f'  aviso: {(~ok).sum()} amostra(s) com valores ausentes removida(s)')
    X, y = X[ok], y[ok]
    _, u = np.unique(X, axis=0, return_index=True); u = np.sort(u)
    if len(u) < len(X):
        print(f'  aviso: {len(X) - len(u)} duplicata(s) exata(s) removida(s)')
    X, y = X[u], y[u]
    X = X[:, X.std(0) > 0]
    if max_samples and len(X) > max_samples:
        s = np.sort(np.random.default_rng(seed).choice(len(X), max_samples, replace=False))
        X, y = X[s], y[s]
        print(f'  amostra aleatória de {max_samples} pontos (semente {seed})')
    if mode == 'standard':
        X = StandardScaler().fit_transform(X)
    elif mode == 'max':
        X = X / np.abs(X).max()
    elif mode == 'minmax':
        X = (X - X.min(0)) / (X.max(0) - X.min(0))
    y = LabelEncoder().fit_transform(np.asarray(y).astype(str))
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# 1) Métricas
# ──────────────────────────────────────────────────────────────────────────────

def topq(s, q):
    return np.argsort(-np.asarray(s), kind='stable')[:max(1, int(np.ceil(q * len(s))))]


def jac(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / max(1, len(a | b))


def detection(score, lab, q):
    return dict(auroc=float(roc_auc_score(lab, score)), p5=float(np.mean(lab[topq(score, q)])))


def timed(fun, repeats):
    ts, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter(); r = fun(); ts.append(time.perf_counter() - t0)
        out = r if out is None else out
    return out, float(np.median(ts))


def run_dataset(X, y, name, display, args):
    n, m = X.shape
    C = int(y.max() + 1)
    k, q, h = args.k_gauss, args.q, args.hops
    print(f'\n=== {display}: n={n} m={m} C={C}  (k_G={args.k_graph}, k={k}, forma {"k×k" if k <= m else "m×m"})')

    t0 = time.perf_counter()
    E, added = knn_graph_edges(X, args.k_graph, connect='components', return_added=True)
    idx = neighborhoods(X, k)
    t_graph = time.perf_counter() - t0
    (_, kG), tb = timed(lambda: gaussian_orc(X, E, idx=idx, n_jobs=args.n_jobs, batched=True), args.repeats)
    print(f'  G-ORC (lote):            {tb:8.3f} s')
    (_, kGl), tl = timed(lambda: gaussian_orc(X, E, idx=idx, n_jobs=args.n_jobs, batched=False), args.repeats)
    print(f'  G-ORC (laço por aresta): {tl:8.3f} s')
    (_, kU), tu = timed(lambda: uniform_orc(X, E, idx=idx, n_jobs=args.n_jobs, p=1), args.repeats)
    print(f'  ORC uniforme (POT):      {tu:8.3f} s   →  speed-up {tu / tb:.1f}× (lote), {tu / tl:.1f}× (laço)')
    assert np.allclose(kG, kGl, atol=1e-5), 'lote e laço divergem'

    # --- arestas ---------------------------------------------------------------
    inter = (y[E[:, 0]] != y[E[:, 1]]).astype(int)
    length = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
    S = [set(r) for r in idx]
    jacd = np.array([1 - len(S[i] & S[j]) / len(S[i] | S[j]) for i, j in E])
    combo = rankdata(-kG) + rankdata(length)
    R = dict(name=name, display=display, n=n, m=m, C=C, E=len(E), added=len(added),
             form='kxk' if k <= m else 'mxm', k_graph=args.k_graph, k_gauss=k, q=q, hops=h,
             preprocess=args.preprocess, t_graph=t_graph,
             timing=dict(b=tb, l=tl, u=tu, E=len(E), repeats=args.repeats, n_jobs=args.n_jobs,
                         cpus=os.cpu_count()),
             base_edge=float(inter.mean()))
    if 0 < inter.sum() < len(inter):
        R['edge'] = {nm: detection(s, inter, q) for nm, s in
                     [('gorc', -kG), ('uorc', -kU), ('length', length), ('jaccard', jacd), ('combo', combo)]}
    R['spearman_GU'] = float(spearmanr(kG, kU)[0])
    R['pearson_GU'] = float(np.corrcoef(kG, kU)[0, 1])
    R['jac_top_edges'] = jac(topq(-kG, q), topq(-kU, q))
    R['rho_s_length'] = float(spearmanr(-kG, length)[0])
    if len(added):
        pos = {tuple(e): t for t, e in enumerate(E.tolist())}
        R['added_pct'] = float(np.mean([np.mean(-kG <= -kG[pos[tuple(a)]]) for a in added.tolist()]))

    # --- vértices --------------------------------------------------------------
    inc = [[] for _ in range(n)]
    for e, (i, j) in enumerate(E):
        inc[i].append(e); inc[j].append(e)
    nb_lab = np.array([np.mean(y[np.where(E[e, 0] == i, E[e, 1], E[e, 0])] != y[i]) for i, e in enumerate(inc)])
    bnd = (nb_lab > 0).astype(int)
    rad = np.linalg.norm(X - X[idx[:, -1]], axis=1)       # raio k-NN (inverso da densidade)
    R['base_vertex'] = float(bnd.mean()); R['base_pure'] = float(1 - bnd.mean())
    V = {}
    if 0 < bnd.sum() < n:
        for nm, kap in (('gorc', kG), ('uorc', kU)):
            for how in AGGS:
                b = node_boundary_score(n, E, kap, how=how)
                V[f'{nm}_{how}'] = dict(**detection(b, bnd, q), rho_imp=float(spearmanr(b, nb_lab)[0]),
                                        rho_rad=float(spearmanr(b, rad)[0]))
        V['radius'] = dict(**detection(rad, bnd, q), rho_imp=float(spearmanr(rad, nb_lab)[0]), rho_rad=1.0)
        V['jaccard'] = detection(node_boundary_score(n, E, -jacd), bnd, q)
        V['length'] = detection(node_boundary_score(n, E, -length), bnd, q)
    R['vertex'] = V
    bG = node_boundary_score(n, E, kG); bU = node_boundary_score(n, E, kU)
    R['jac_top_vertices'] = jac(topq(bG, q), topq(bU, q))
    R['spearman_bGU'] = float(spearmanr(bG, bU)[0])

    # --- âncoras ---------------------------------------------------------------
    A, sel = {}, {}
    for nm, b in (('gorc', bG), ('uorc', bU), ('radius', rad)):
        a = local_minima(n, E, b, h); sel[nm] = a
        glob_ = np.argsort(b, kind='stable')[:len(a)]
        A[nm] = dict(count=int(len(a)), purity=float(1 - bnd[a].mean()), coverage=len(set(y[a])) / C,
                     glob_purity=float(1 - bnd[glob_].mean()), glob_coverage=len(set(y[glob_])) / C,
                     rho_rad=float(spearmanr(b[a], rad[a])[0]) if len(a) > 2 else float('nan'))
    A['jac_GU'] = jac(sel['gorc'], sel['uorc'])
    R['anchors'] = A

    if args.check_gelbrich:                                # Proposição 3: κ_G ≥ κ₂ᵁ
        _, k2 = uniform_orc(X, E, idx=idx, n_jobs=args.n_jobs, p=2)
        R['gelbrich_max_violation'] = float(np.max(k2 - kG))
        print(f'  Prop. 3: máx(κ₂ᵁ − κ_G) = {R["gelbrich_max_violation"]:.1e} (≤ 0 até arredondamento)')

    if 'edge' in R:
        e = R['edge']; v = V
        print(f'  arestas  AUROC  G={e["gorc"]["auroc"]:.3f} U={e["uorc"]["auroc"]:.3f} '
              f'len={e["length"]["auroc"]:.3f} jacc={e["jaccard"]["auroc"]:.3f} G+L={e["combo"]["auroc"]:.3f}'
              f'   Spearman(κ_G, κ_U)={R["spearman_GU"]:.3f}')
        print(f'  vértices AUROC  G={v["gorc_softmax"]["auroc"]:.3f} U={v["uorc_softmax"]["auroc"]:.3f} '
              f'raio={v["radius"]["auroc"]:.3f}   P@{q:.0%}: G={v["gorc_softmax"]["p5"]:.2f} base={R["base_vertex"]:.2f}')
        print(f'  âncoras  G: {A["gorc"]["count"]}  pureza={A["gorc"]["purity"]:.2f} (base {R["base_pure"]:.2f})  '
              f'cobertura={A["gorc"]["coverage"]:.2f}')
    arrays = dict(E=E, added=added, kG=kG, kU=kU, y=y, bG=bG, bU=bU, aG=sel['gorc'], aU=sel['uorc'],
                  rad=rad, nb_lab=nb_lab, length=length)
    return R, arrays


# ──────────────────────────────────────────────────────────────────────────────
# 2) Figura do grafo (arestas e vértices destacados)
# ──────────────────────────────────────────────────────────────────────────────

def layout_positions(X, E, how, seed):
    n = len(X)
    if how == 'auto':
        how = 'spring' if n <= 3000 else 'tsne'
    if how == 'tsne':
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        Z = PCA(50, random_state=seed).fit_transform(X) if X.shape[1] > 50 else X
        return TSNE(2, init='pca', random_state=seed).fit_transform(Z), 't-SNE (só visualização)'
    import networkx as nx                                  # layout do grafo no espaço original
    d = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
    s_med = np.median(d) if np.median(d) > 0 else 1.0
    G = nx.Graph(); G.add_nodes_from(range(n))
    for (i, j), dij in zip(E.tolist(), d):
        G.add_edge(i, j, sim=float(np.exp(-(dij / s_med) ** 2)), dist_n=max(dij / s_med, 1e-6))
    p = nx.spring_layout(G, seed=seed, iterations=300)     # sem pesos: evita que pontos isolados
    if how in ('kawai', 'kamada_kawai'):                   # comprimam o restante do desenho
        p = nx.kamada_kawai_layout(G, weight='dist_n', pos=p)
        nm_ = 'Kamada–Kawai do grafo k-NN'
    else:
        nm_ = 'spring do grafo k-NN'
    Z = np.array([p[v] for v in range(n)])
    lo, hi = np.percentile(Z, 1, 0), np.percentile(Z, 99, 0)  # recorta pontos muito afastados à borda
    return np.clip(Z, lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)), nm_


def plot_graph(X, arr, R, args, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    RED, BLUE = '#b2182b', '#2166ac'
    E, y = arr['E'], arr['y']
    Z, lay = layout_positions(X, E, args.layout, args.seed)
    n = len(X); C = R['C']
    cls = plt.cm.tab10(y % 10) if C <= 10 else plt.cm.tab20(y % 20)
    ps = 6 if n < 3000 else 1.2
    qe = args.q if len(E) * args.q <= 3000 else 3000 / len(E)          # legibilidade em grafos grandes
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, key, ttl in ((axs[0], 'kG', 'G-ORC'), (axs[1], 'kU', 'ORC uniforme')):
        t = topq(-arr[key], qe)
        ax.add_collection(LineCollection(Z[E], colors='#c9c8c3', linewidths=0.2, alpha=0.35, rasterized=True))
        ax.scatter(*Z.T, s=ps, c=cls, alpha=0.45, lw=0, rasterized=True)
        ax.add_collection(LineCollection(Z[E[t]], colors=RED, linewidths=0.6, alpha=0.85, rasterized=True))
        frac = np.mean(y[E[t, 0]] != y[E[t, 1]])
        ax.set_title(f'{ttl}: {qe:.0%} arestas mais negativas ({len(t)})\n'
                     f'{frac:.0%} inter-classe (base {R["base_edge"]:.0%})', fontsize=11)
    ax = axs[2]
    I = topq(arr['bG'], args.q); A = arr['aG']
    ax.add_collection(LineCollection(Z[E], colors='#c9c8c3', linewidths=0.2, alpha=0.3, rasterized=True))
    ax.scatter(*Z.T, s=ps, c=cls, alpha=0.3, lw=0, rasterized=True)
    ax.scatter(*Z[I].T, s=ps * 3 + 8, c=RED, edgecolors='#5a1010', lw=0.3, label=f'interface (top {args.q:.0%})')
    ax.scatter(*Z[A].T, s=ps * 4 + 18, c=BLUE, marker='D', edgecolors='white', lw=0.4,
               label=f'âncoras (h={args.hops}, {len(A)})')
    ax.legend(loc='lower left', fontsize=9, framealpha=0.85)
    v = R['vertex']; a = R['anchors']['gorc']
    ax.set_title(f'G-ORC: vértices — interface {v["gorc_softmax"]["p5"]:.0%} borda (base {R["base_vertex"]:.0%})\n'
                 f'âncoras: pureza {a["purity"]:.0%}, cobertura {a["coverage"]:.0%} das classes', fontsize=11)
    for a_ in axs:
        a_.set_xticks([]); a_.set_yticks([]); [s.set_visible(False) for s in a_.spines.values()]
        a_.autoscale()
    T = R['timing']
    fig.suptitle(f'{R["display"]}  (n={n}, m={R["m"]}, C={C}, |E|={R["E"]}, k_G={args.k_graph}, k={args.k_gauss})'
                 f'   —   tempo das curvaturas: ORC {T["u"]:.2f}s, G-ORC {T["b"]:.2f}s ({T["u"] / T["b"]:.1f}×)',
                 fontsize=12)
    fig.text(0.01, 0.01, f'Layout: {lay}. Cores = classes (só exibição); rótulos não usados nas seleções.',
             fontsize=8.5, color='#666')
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path + '.png', dpi=200); fig.savefig(path + '.pdf', dpi=300)
    plt.close(fig)
    print(f'  figura: {path}.png/.pdf')


# ──────────────────────────────────────────────────────────────────────────────
# 3) Tabelas LaTeX (mesmo formato das Tabelas 3–9 do artigo)
# ──────────────────────────────────────────────────────────────────────────────

def f2(x, d=2):
    return f'{x:.{d}f}'


def bold_row(vals, d=2):
    r = [round(v, d) for v in vals]; best = max(r)
    return [('\\textbf{' + f2(v, d) + '}') if rv == best else f2(v, d) for v, rv in zip(vals, r)]


def mean_rank(Z):
    return np.mean([rankdata(-np.round(z, 2)) for z in Z], 0)          # empates → rank médio


def write_table(path, head, rows, foot, cols, new_rows=None):
    s = '\\begin{tabular}{' + cols + '}\n\\toprule\n' + head + '\\midrule\n' + '\n'.join(r + ' \\\\' for r in rows) + '\n'
    if foot:
        s += '\\midrule\n' + '\n'.join(r + ' \\\\' for r in foot) + '\n'
    s += '\\bottomrule\n\\end{tabular}\n'
    open(path, 'w').write(s)


def build_tables(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, 'results', '*.json')))
    if not files:
        print('nenhum resultado em', os.path.join(out_dir, 'results')); return
    allR = [json.load(open(f)) for f in files]
    allR.sort(key=lambda r: (r.get('order', 1e9), r.get('created', 0)))
    R = [r for r in allR if not r.get('timing_only') and 'edge' in r and r.get('vertex')]
    tdir = os.path.join(out_dir, 'tables'); os.makedirs(tdir, exist_ok=True)
    nm = lambda r: r['display']
    newset = {r['name'] for r in allR if not r.get('paper')}
    rows_new = {}

    def add_rows(key, rows, recs):
        rows_new[key] = [row for row, r in zip(rows, recs) if r['name'] in newset]

    # T3
    rows = [f"{nm(r)} & {r['n']} & {r['m']} & {r['C']} & {r['E']} & {r['form'].replace('x', '$\\times$')} & "
            f"{100 * r['base_edge']:.1f} & {100 * r['base_vertex']:.1f}" for r in R]
    write_table(f'{tdir}/tab_datasets.tex', 'Dataset & $n$ & $m$ & $C$ & $|E|$ & Form & Inter-class edges (\\%) & '
                'Boundary vertices (\\%) \\\\\n', rows, None, '@{}lrrrrcrr@{}'); add_rows('T3 datasets', rows, R)
    # T4
    keys = ['spearman_GU', 'jac_top_edges', 'spearman_bGU', 'jac_top_vertices']
    rows = [f"{nm(r)} & " + ' & '.join(f2(r[k]) for k in keys) + f" & {f2(r['anchors']['jac_GU'])}" for r in R]
    mean = [np.mean([r[k] for r in R]) for k in keys] + [np.mean([r['anchors']['jac_GU'] for r in R])]
    write_table(f'{tdir}/tab_agreement.tex',
                ' & \\multicolumn{2}{c}{Edges} & \\multicolumn{2}{c}{Vertices} & Anchors \\\\\n'
                '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-6}\n'
                'Dataset & Spearman $\\rho$ & Jaccard top 5\\% & Spearman $\\rho$ & Jaccard top 5\\% & Jaccard \\\\\n',
                rows, ['Mean & ' + ' & '.join(f2(x) for x in mean)], '@{}lccccc@{}'); add_rows('T4 agreement', rows, R)
    # T5
    M = ['gorc', 'uorc', 'length', 'jaccard', 'combo']
    A_ = np.array([[r['edge'][k]['auroc'] for k in M] for r in R]); P_ = np.array([[r['edge'][k]['p5'] for k in M] for r in R])
    rows = [f"{nm(r)} & " + ' & '.join(bold_row(a)) + ' & ' + ' & '.join(bold_row(p)) + f" & {f2(r['base_edge'])}"
            for r, a, p in zip(R, A_, P_)]
    foot = ['Mean & ' + ' & '.join(f2(x) for x in A_.mean(0)) + ' & ' + ' & '.join(f2(x) for x in P_.mean(0)) +
            ' & ' + f2(np.mean([r['base_edge'] for r in R])),
            'Mean rank & ' + ' & '.join(f2(x, 1) for x in mean_rank(A_)) + ' & ' +
            ' & '.join(f2(x, 1) for x in mean_rank(P_)) + ' & ']
    write_table(f'{tdir}/tab_edges.tex', ' & \\multicolumn{5}{c}{AUROC} & \\multicolumn{5}{c}{Precision of top 5\\%} & \\\\\n'
                '\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n'
                'Dataset & G-ORC & ORC & Len. & Jacc. & G+L & G-ORC & ORC & Len. & Jacc. & G+L & Base \\\\\n',
                rows, foot, '@{}lcccccccccc c@{}'); add_rows('T5 edges', rows, R); rows_new['T5 edges (Mean/Mean rank)'] = foot
    # T6
    M = ['gorc_softmax', 'uorc_softmax', 'jaccard', 'radius']
    A_ = np.array([[r['vertex'][k]['auroc'] for k in M] for r in R]); P_ = np.array([[r['vertex'][k]['p5'] for k in M] for r in R])
    rows = [f"{nm(r)} & " + ' & '.join(bold_row(a)) + ' & ' + ' & '.join(bold_row(p)) +
            f" & {f2(r['base_vertex'])} & {r['vertex']['gorc_softmax']['rho_rad']:+.2f}" for r, a, p in zip(R, A_, P_)]
    foot = ['Mean & ' + ' & '.join(f2(x) for x in A_.mean(0)) + ' & ' + ' & '.join(f2(x) for x in P_.mean(0)) +
            ' & ' + f2(np.mean([r['base_vertex'] for r in R])) + ' & ',
            'Mean rank & ' + ' & '.join(f2(x, 1) for x in mean_rank(A_)) + ' & ' +
            ' & '.join(f2(x, 1) for x in mean_rank(P_)) + ' & & ']
    write_table(f'{tdir}/tab_vertices.tex', ' & \\multicolumn{4}{c}{AUROC} & \\multicolumn{4}{c}{Precision of top 5\\%} & & \\\\\n'
                '\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n'
                'Dataset & G-ORC & ORC & Jaccard & Radius & G-ORC & ORC & Jaccard & Radius & Base & $\\rho_{\\mathrm{rad}}$ \\\\\n',
                rows, foot, '@{}lcccccccccc@{}'); add_rows('T6 vertices', rows, R); rows_new['T6 vertices (Mean/Mean rank)'] = foot
    # T7
    A_ = np.array([[r['vertex'][f'gorc_{h}']['auroc'] for h in AGGS] for r in R])
    P_ = np.array([[r['vertex'][f'gorc_{h}']['p5'] for h in AGGS] for r in R])
    rows = [f"{nm(r)} & " + ' & '.join(bold_row(a)) + ' & ' + ' & '.join(bold_row(p)) for r, a, p in zip(R, A_, P_)]
    foot = ['Mean & ' + ' & '.join(f2(x) for x in A_.mean(0)) + ' & ' + ' & '.join(f2(x) for x in P_.mean(0)),
            'Mean rank & ' + ' & '.join(f2(x, 1) for x in mean_rank(A_)) + ' & ' + ' & '.join(f2(x, 1) for x in mean_rank(P_))]
    hd = 'Mean & Max & Quantile & Soft max & Top frac.'
    write_table(f'{tdir}/tab_aggregation.tex', ' & \\multicolumn{5}{c}{AUROC} & \\multicolumn{5}{c}{Precision of top 5\\%} \\\\\n'
                '\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n' f'Dataset & {hd} & {hd} \\\\\n',
                rows, foot, '@{}lcccccccccc@{}'); add_rows('T7 aggregation', rows, R); rows_new['T7 aggregation (Mean/Mean rank)'] = foot
    # T8
    rows, Pu, Co = [], [], []
    for r in R:
        a = r['anchors']
        pu = [a['gorc']['purity'], a['uorc']['purity'], a['radius']['purity'], a['gorc']['glob_purity']]
        co = [a['gorc']['coverage'], a['uorc']['coverage'], a['radius']['coverage'], a['gorc']['glob_coverage']]
        Pu.append(pu); Co.append(co)
        rows.append(f"{nm(r)} & {a['gorc']['count']} & {a['uorc']['count']} & {a['radius']['count']} & " +
                    ' & '.join(f2(x) for x in pu) + f" & {f2(r['base_pure'])} & " + ' & '.join(f2(x) for x in co))
    foot = ['Mean & & & & ' + ' & '.join(f2(x) for x in np.mean(Pu, 0)) + ' & ' + f2(np.mean([r['base_pure'] for r in R])) +
            ' & ' + ' & '.join(f2(x) for x in np.mean(Co, 0))]
    write_table(f'{tdir}/tab_anchors.tex', ' & \\multicolumn{3}{c}{Count} & \\multicolumn{4}{c}{Purity} & & '
                '\\multicolumn{4}{c}{Class coverage} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\\cmidrule(lr){10-13}\n'
                'Dataset & G & U & Mode & G & U & Mode & G$_{\\mathrm{glob}}$ & Base & G & U & Mode & G$_{\\mathrm{glob}}$ \\\\\n',
                rows, foot, '@{}lrrrccccccccc@{}'); add_rows('T8 anchors', rows, R); rows_new['T8 anchors (Mean)'] = foot
    # T9 (inclui também entradas só de tempo)
    RT = [r for r in allR if 'timing' in r]
    rows = [f"{nm(r)} & {r['timing']['E']} & {r['timing']['u']:.2f} & {r['timing']['l']:.2f} & {r['timing']['b']:.2f} & "
            f"{r['timing']['u'] / r['timing']['l']:.1f}$\\times$ & {r['timing']['u'] / r['timing']['b']:.1f}$\\times$" for r in RT]
    write_table(f'{tdir}/tab_timing.tex', ' & & \\multicolumn{3}{c}{Edge-curvature time (s)} & \\multicolumn{2}{c}{Speed-up} \\\\\n'
                '\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\n'
                'Dataset & $|E|$ & ORC & G-ORC (loop) & G-ORC (batched) & Loop & Batched \\\\\n', rows, None, '@{}lrrrrrr@{}')
    add_rows('T9 timing', rows, RT)

    with open(os.path.join(tdir, 'new_rows.tex'), 'w') as fh:            # só as linhas novas, para colar
        fh.write('% Linhas dos conjuntos NOVOS (não presentes no artigo) e linhas Mean/Mean rank\n'
                 '% recalculadas sobre TODOS os conjuntos em results/. Tabelas: 3 datasets, 4 agreement,\n'
                 '% 5 edges, 6 vertices, 7 aggregation, 8 anchors, 9 timing.\n')
        for key, rows in rows_new.items():
            fh.write(f'\n% ---- {key} ----\n' + '\n'.join(r + ' \\\\' for r in rows) + '\n')
    pd.DataFrame([dict(dataset=r['display'], n=r['n'], m=r['m'], C=r['C'], E=r['E'],
                       edge_auroc_G=r['edge']['gorc']['auroc'], edge_auroc_U=r['edge']['uorc']['auroc'],
                       edge_auroc_len=r['edge']['length']['auroc'], edge_auroc_jacc=r['edge']['jaccard']['auroc'],
                       edge_auroc_GL=r['edge']['combo']['auroc'], spearman_GU=r['spearman_GU'],
                       vertex_auroc_G=r['vertex']['gorc_softmax']['auroc'], vertex_auroc_U=r['vertex']['uorc_softmax']['auroc'],
                       vertex_auroc_rad=r['vertex']['radius']['auroc'], vertex_p5_G=r['vertex']['gorc_softmax']['p5'],
                       anchors_purity_G=r['anchors']['gorc']['purity'], anchors_coverage_G=r['anchors']['gorc']['coverage'],
                       base_pure=r['base_pure'], t_uorc=r['timing']['u'], t_gorc_loop=r['timing']['l'],
                       t_gorc_batched=r['timing']['b']) for r in R]).to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    print(f'\ntabelas refeitas com {len(R)} conjuntos ({len(newset)} novo(s)): {tdir}/  (linhas novas em new_rows.tex)')


# ──────────────────────────────────────────────────────────────────────────────
# 4) Programa principal
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--openml', nargs='*', default=[], help='nomes de conjuntos do OpenML')
    ap.add_argument('--openml_id', nargs='*', default=[], help='data_ids do OpenML')
    ap.add_argument('--sklearn', nargs='*', default=[], help='iris wine breast_cancer digits (teste offline)')
    ap.add_argument('--version', default=1, help="versão no OpenML ('active' = mais recente)")
    ap.add_argument('--target', default=None, help='coluna-alvo no OpenML, se não houver padrão')
    ap.add_argument('--name', nargs='*', default=None, help='nome(s) de exibição nas tabelas, na mesma ordem')
    ap.add_argument('--preprocess', default='standard', choices=['standard', 'max', 'minmax', 'none'],
                    help="'standard' (z-score; tabulares, padrão) ou 'max' (pixels → [0,1], como MNIST)")
    ap.add_argument('--max_samples', type=int, default=10000, help='amostra aleatória de no máximo N pontos')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--k_graph', type=int, default=10)
    ap.add_argument('--k_gauss', type=int, default=20)
    ap.add_argument('--q', type=float, default=0.05, help='fração selecionada (top q)')
    ap.add_argument('--hops', type=int, default=2, help='raio das âncoras (saltos)')
    ap.add_argument('--repeats', type=int, default=3, help='repetições para a mediana dos tempos')
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--check_gelbrich', action='store_true', help='confere κ_G ≥ κ₂ᵁ (Prop. 3; custa um ORC extra)')
    ap.add_argument('--no_plot', action='store_true')
    ap.add_argument('--layout', default='auto', choices=['auto', 'spring', 'kawai', 'tsne'],
                    help="auto: spring (grafo no espaço original) se n ≤ 3000, senão t-SNE")
    ap.add_argument('--out', default=os.path.join('results', 'openml'), help='pasta de saída')
    ap.add_argument('--no_paper', action='store_true',
                    help='não incluir nas tabelas os 12 conjuntos do artigo (results/paper_results)')
    ap.add_argument('--tables_only', action='store_true', help='só refaz as tabelas a partir dos JSON')
    args = ap.parse_args()

    for sub in ('results', 'figs', 'tables', 'csv'):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    paper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'paper_results')
    if not args.no_paper and os.path.isdir(paper_dir):     # resultados dos 12 conjuntos do artigo
        import shutil
        for f in glob.glob(os.path.join(paper_dir, '*.json')):
            dst = os.path.join(args.out, 'results', os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copy(f, dst)
    if args.tables_only:
        build_tables(args.out); return
    jobs = [('openml', d) for d in args.openml] + [('openml_id', d) for d in args.openml_id] + \
           [('sklearn', d) for d in args.sklearn]
    if not jobs:
        ap.error('informe --openml, --openml_id ou --sklearn (ou use --tables_only)')
    if args.name and len(args.name) != len(jobs):
        ap.error('--name precisa de um nome por conjunto')
    for t, (src, ident) in enumerate(jobs):
        X, y_raw, name = load_dataset(src, ident, args.version, args.target)
        X, y = preprocess(X, y_raw, args.preprocess, args.max_samples, args.seed)
        if args.k_gauss >= len(X) or args.k_graph >= len(X):
            print(f'{name}: poucos pontos para k; pulando'); continue
        display = args.name[t] if args.name else name
        slug = re.sub(r'[^\w\-]+', '_', str(name))
        old = os.path.join(args.out, 'results', f'{slug}.json')
        if os.path.exists(old) and json.load(open(old)).get('paper'):
            slug += '_openml'                              # não sobrescreve um resultado do artigo
            display = display if args.name else f'{name} (OpenML)'
        R, arr = run_dataset(X, y, slug, display, args)
        R['source'] = f'{src}:{ident}'; R['created'] = time.time(); R['paper'] = False
        json.dump(R, open(os.path.join(args.out, 'results', f'{slug}.json'), 'w'), indent=1)
        np.savez_compressed(os.path.join(args.out, 'results', f'{slug}.npz'), **arr)
        pd.DataFrame(dict(indice=np.arange(len(y)), rotulo=y, borda_verdadeira=(arr['nb_lab'] > 0).astype(int),
                          b_gorc=arr['bG'], b_uorc=arr['bU'], raio_knn=arr['rad'],
                          interface_gorc=np.isin(np.arange(len(y)), topq(arr['bG'], args.q)).astype(int),
                          ancora_gorc=np.isin(np.arange(len(y)), arr['aG']).astype(int))
                     ).to_csv(os.path.join(args.out, 'csv', f'{slug}_vertices.csv'), index=False)
        if not args.no_plot:
            plot_graph(X, arr, R, args, os.path.join(args.out, 'figs', f'{slug}_graph'))
        build_tables(args.out)


if __name__ == '__main__':
    main()
