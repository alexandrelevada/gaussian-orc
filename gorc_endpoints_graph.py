#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vértices extremos das arestas de menor G-ORC (sem agregação).

Seleciona as arestas com as menores curvaturas de Ollivier–Ricci gaussiana (padrão: 5%)
e destaca no grafo k-NN apenas os VÉRTICES que são extremidades dessas arestas.
Visualização não supervisionada: os rótulos só aparecem como referência (nome da classe
no centro de cada grupo) e, com --validate, na conferência impressa no terminal.

Exemplos:
    python gorc_endpoints_graph.py                       # digits, 5% das arestas
    python gorc_endpoints_graph.py --top 0.02 --show_edges
    python gorc_endpoints_graph.py --min_hits 3              # só vértices tocados por ≥ 3 arestas selecionadas
    python gorc_endpoints_graph.py --top_nodes 0.10          # 10% dos vértices de maior pontuação de borda
    python gorc_endpoints_graph.py --top_nodes 0.05 --agg mean --smooth 1
    python gorc_endpoints_graph.py --bottom_nodes 0.10       # 10% dos vértices de MENOR pontuação (interior)
    python gorc_endpoints_graph.py --local_minima --lm_hops 2  # mínimos locais da pontuação (um 'fundo' por região)
    python gorc_endpoints_graph.py --both                    # azul: mínimos locais (2 saltos); vermelho: top 5%
    python gorc_endpoints_graph.py --both --lm_hops 3 --top_nodes 0.10

Dois modos de seleção:
    arestas  (padrão)      extremidades das --top arestas de menor κ_G, com ≥ --min_hits arestas
    vértices (--top_nodes) os --top_nodes vértices de maior pontuação de borda b_i, obtida
                           agregando −κ_G das arestas incidentes (--agg; maior b = κ mais negativo)
    vértices (--bottom_nodes) os de MENOR b_i (κ mais positivo): interior das regiões
    vértices (--local_minima) vértices com b_i ≤ b_j para todo j a até --lm_hops saltos no grafo
    python gorc_endpoints_graph.py --validate --layout kawai

Saídas:
    <conjunto>_gorc_endpoints[_h<min_hits>].png   grafo com os vértices destacados
    <conjunto>_gorc_endpoints.csv   índice, rótulo, destacado (0/1), nº de arestas selecionadas incidentes
"""
import argparse
import re
import time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from sklearn.preprocessing import LabelEncoder, StandardScaler
from gaussian_orc import gaussian_orc, uniform_orc, knn_graph_edges, node_boundary_score, graph_smooth

ap = argparse.ArgumentParser()
src = ap.add_mutually_exclusive_group()
src.add_argument('--dataset', default='digits',
                 choices=['digits', 'iris', 'wine', 'breast_cancer'], help='conjunto do scikit-learn')
src.add_argument('--openml', default=None, help='nome do conjunto no OpenML (ex.: mfeat-karhunen)')
src.add_argument('--openml_id', type=int, default=None, help='data_id do conjunto no OpenML')
ap.add_argument('--version', default=1, help="versão do conjunto no OpenML ('active' = mais recente)")
ap.add_argument('--scale', action='store_true', help='padroniza os atributos (média 0, desvio 1)')
ap.add_argument('--k_graph', type=int, default=10, help='vizinhos do grafo k-NN')
ap.add_argument('--k_gauss', '--k_meas', dest='k_gauss', type=int, default=20,
                help='vizinhos de cada medida local (gaussiana ou uniforme)')
ap.add_argument('--orc', default='gaussian', choices=['gaussian', 'uniform'],
                help="curvatura: 'gaussian' (G-ORC, W₂ de Bures) ou 'uniform' (ORC clássico, W₁ por transporte ótimo)")
ap.add_argument('--top', type=float, default=0.05,
                help='arestas destacadas: fração (<1, ex. 0.03) ou número (≥1, ex. 200)')
ap.add_argument('--layout', default='spring', choices=['spring', 'kamada_kawai', 'kawai'])
ap.add_argument('--seed', type=int, default=0, help='semente do layout')
ap.add_argument('--min_hits', type=int, default=1,
                help='destaca só os vértices com pelo menos este nº de arestas selecionadas incidentes')
ap.add_argument('--top_nodes', type=float, default=None,
                help='modo vértices: fração (<1, ex. 0.05) ou número (≥1) de vértices de maior pontuação de borda')
ap.add_argument('--bottom_nodes', type=float, default=None,
                help='modo vértices: fração (<1) ou número (≥1) de vértices de MENOR pontuação de borda (interior)')
ap.add_argument('--both', action='store_true',
                help='na mesma figura: mínimos locais (interior) em azul e os --top_nodes (padrão 0.05) '
                     'vértices de maior pontuação de borda em vermelho')
ap.add_argument('--local_minima', action='store_true',
                help='modo vértices: destaca os mínimos locais da pontuação de borda (pontos puros de cada região)')
ap.add_argument('--lm_hops', type=int, default=2,
                help='raio (em saltos no grafo) da vizinhança usada para definir o mínimo local')
ap.add_argument('--agg', default='softmax', choices=['mean', 'max', 'quantile', 'softmax', 'top_frac'],
                help='agregação aresta→vértice no modo vértices (padrão softmax)')
ap.add_argument('--smooth', type=int, default=0, help='passos de suavização no grafo (modo vértices)')
ap.add_argument('--show_edges', action='store_true', help='desenha também as arestas selecionadas (finas)')
ap.add_argument('--validate', action='store_true',
                help='compara os vértices destacados com a borda verdadeira (usa rótulos; só no terminal)')
ap.add_argument('--out', default=None, help='arquivo da figura')
args = ap.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 0) Leitura dos dados
# ──────────────────────────────────────────────────────────────────────────────

def load_data(args):
    """Retorna X (n, m) float, y (n,) inteiros 0..C-1, nomes das classes e nome do conjunto."""
    if args.openml is not None or args.openml_id is not None:
        version = args.version if args.version == 'active' else int(args.version)
        if args.openml_id is not None:
            ds = skdata.fetch_openml(data_id=args.openml_id, as_frame=True, parser='auto')
            name = ds.details.get('name', f'openml_{args.openml_id}') if hasattr(ds, 'details') else f'openml_{args.openml_id}'
        else:
            ds = skdata.fetch_openml(name=args.openml, version=version, as_frame=True, parser='auto')
            name = args.openml
        df = ds.data
        # só atributos numéricos; categóricos são descartados (avisa)
        # Atributos nominais cujas categorias são números (ex.: mfeat-pixel, '0'..'6')
        # são convertidos para float; os realmente categóricos são descartados.
        cols, n_conv, dropped = [], 0, []
        for c in df.columns:
            col = df[c]
            if pd.api.types.is_numeric_dtype(col) and not isinstance(col.dtype, pd.CategoricalDtype):
                cols.append(col.astype(float))
                continue
            conv = pd.to_numeric(col.astype(object), errors='coerce')
            if conv.notna().sum() == col.notna().sum():      # todos os valores presentes são números
                cols.append(conv.astype(float)); n_conv += 1
            else:
                dropped.append(c)
        if n_conv:
            print(f'aviso: {n_conv} atributo(s) nominal(is) com valores numéricos convertido(s) para número')
        if dropped:
            print(f'aviso: {len(dropped)} atributo(s) categórico(s) não numérico(s) descartado(s): '
                  f'{", ".join(map(str, dropped[:5]))}{" ..." if len(dropped) > 5 else ""}')
        if not cols:
            raise SystemExit('nenhum atributo numérico utilizável neste conjunto')
        X = pd.concat(cols, axis=1).to_numpy(dtype=float)
        target = ds.target
        if target is None:
            raise ValueError('o conjunto não tem atributo-alvo padrão no OpenML')
        y_raw = np.asarray(target).ravel()
    else:
        loader = getattr(skdata, f'load_{args.dataset}')
        X, y_raw = loader(return_X_y=True)
        X = X.astype(float)
        name = args.dataset

    # remove linhas com valores ausentes
    ok = ~np.isnan(X).any(axis=1) & ~(np.asarray(y_raw, dtype=object) == None)  # noqa: E711
    if (~ok).sum():
        print(f'aviso: {(~ok).sum()} amostra(s) com valores ausentes removida(s)')
    X, y_raw = X[ok], np.asarray(y_raw)[ok]

    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    classes = le.classes_
    # ordena rótulos numéricos ('10' depois de '9')
    if all(re.fullmatch(r'-?\d+(\.\d+)?', c) for c in classes):
        order = np.argsort(classes.astype(float))
        remap = np.empty_like(order); remap[order] = np.arange(len(order))
        y, classes = remap[y], classes[order]
    return X, y, classes, re.sub(r'[^\w\-]+', '_', str(name))


X, y, classes, name = load_data(args)
if args.scale:
    X = StandardScaler().fit_transform(X)
keep = X.std(axis=0) > 0
if (~keep).sum():
    print(f'aviso: {(~keep).sum()} atributo(s) constante(s) removido(s)')
    X = X[:, keep]
n, m = X.shape
print(f'conjunto: {name}  n={n}  m={m}  classes={len(classes)}')
if args.k_gauss > n - 1 or args.k_graph > n - 1:
    raise SystemExit('k_graph e k_gauss precisam ser menores que n')


# ──────────────────────────────────────────────────────────────────────────────
# 1) Curvatura de Ollivier–Ricci gaussiana
# ──────────────────────────────────────────────────────────────────────────────

ORC = 'G-ORC' if args.orc == 'gaussian' else 'ORC uniforme'
TAG = 'gorc' if args.orc == 'gaussian' else 'uorc'
t0 = time.perf_counter()
E = knn_graph_edges(X, args.k_graph)
t_graph = time.perf_counter() - t0
t0 = time.perf_counter()                                 # só a curvatura das arestas (inclui as medidas locais)
if args.orc == 'gaussian':
    E, kappa = gaussian_orc(X, E, k_gauss=args.k_gauss, n_jobs=-1)
else:
    E, kappa = uniform_orc(X, E, k_meas=args.k_gauss, n_jobs=-1)
t_curv = time.perf_counter() - t0
t_sel0 = time.perf_counter()
print(f'tempo do grafo k-NN: {t_graph:.3f}s  |  tempo SÓ do cálculo das curvaturas das arestas ({ORC}): '
      f'{t_curv:.3f}s  ({1e3 * t_curv / len(E):.3f} ms/aresta)')
print(f'{ORC}: {len(E)} arestas  '
      f'κ: mín={kappa.min():.3f}  mediana={np.median(kappa):.3f}  máx={kappa.max():.3f}  '
      f'negativas={np.mean(kappa < 0):.0%}')

n_top = int(round(args.top * len(E))) if args.top < 1 else int(args.top)
n_top = max(1, min(n_top, len(E)))
top = np.argsort(kappa)[:n_top]                         # mais negativas = mais "ponte"

# vértices extremos das arestas selecionadas (sem agregação: basta tocar uma delas)
hits = np.bincount(E[top].ravel(), minlength=n)          # nº de arestas selecionadas incidentes
if args.min_hits < 1:
    raise SystemExit('--min_hits deve ser ≥ 1')
b_node = None
both_top = None
if args.both:                                            # azul = mínimos locais, vermelho = maiores pontuações
    if args.bottom_nodes is not None:
        raise SystemExit('--both não se combina com --bottom_nodes')
    both_top = args.top_nodes if args.top_nodes is not None else 0.05
    args.top_nodes, args.local_minima = None, True
if sum(v is not None and v is not False for v in (args.top_nodes, args.bottom_nodes, args.local_minima or None)) > 1:
    raise SystemExit('use só uma de --top_nodes, --bottom_nodes, --local_minima')
lmin = args.local_minima
bottom = args.bottom_nodes is not None or lmin
if args.bottom_nodes is not None:
    args.top_nodes = args.bottom_nodes
if args.top_nodes is not None or lmin:                   # modo vértices: pontuação de borda agregada
    b_node = node_boundary_score(n, E, kappa, how=args.agg)
    if args.smooth:
        b_node = graph_smooth(n, E, b_node, steps=args.smooth)
    ends = np.zeros(n, bool)
    if lmin:                                             # mínimo da bola de lm_hops saltos, por propagação
        nbrs = [[] for _ in range(n)]
        for i, j in E:
            nbrs[i].append(j); nbrs[j].append(i)
        ball_min = b_node.copy()
        for _ in range(max(1, args.lm_hops)):
            ball_min = np.array([min(ball_min[i], ball_min[v].min()) if len(v) else ball_min[i]
                                 for i, v in enumerate(nbrs)])
        ends = b_node <= ball_min + 1e-12
    else:
        n_nodes = int(round(args.top_nodes * n)) if args.top_nodes < 1 else int(args.top_nodes)
        n_nodes = max(1, min(n_nodes, n))
        ends[np.argsort(b_node if bottom else -b_node, kind='stable')[:n_nodes]] = True
    red = np.zeros(n, bool)
    if both_top is not None:
        n_red = int(round(both_top * n)) if both_top < 1 else int(both_top)
        red[np.argsort(-b_node, kind='stable')[:max(1, min(n_red, n))]] = True
        if (red & ends).any():
            print(f'aviso: {(red & ends).sum()} vértice(s) são mínimos locais e também estão entre as maiores '
                  f'pontuações; desenhados em vermelho')
            ends &= ~red
else:
    ends = hits >= args.min_hits
    red = np.zeros(n, bool)
if not ends.any():
    raise SystemExit(f'nenhum vértice tem ≥ {args.min_hits} arestas selecionadas (máximo = {hits.max()}); '
                     f'reduza --min_hits ou aumente --top')
if b_node is not None:
    print(f'modo vértices: agregação {args.agg}' + (f' + suavização {args.smooth}' if args.smooth else '') +
          f'  →  {ends.sum()} vértices destacados ({ends.mean():.1%}) com b '
          + (f'mínimos locais (raio {args.lm_hops} salto(s))' if lmin else
             f'≤ {b_node[ends].max():.3f}  [menor pontuação = interior]' if bottom else f'≥ {b_node[ends].min():.3f}'))
else:
    print(f'arestas selecionadas: {n_top} ({n_top / len(E):.1%}) com κ_G ≤ {kappa[top].max():.3f}  →  '
          f'{(hits > 0).sum()} vértices extremos; destacados com ≥ {args.min_hits}: {ends.sum()} ({ends.mean():.1%} dos vértices)')
if red.any():
    print(f'vermelho: {red.sum()} vértices ({red.mean():.1%}) de maior pontuação de borda, b ≥ {b_node[red].min():.3f}')
t_sel = time.perf_counter() - t_sel0
print(f'tempo da análise ({ORC}): grafo {t_graph:.3f}s + curvatura {t_curv:.3f}s + '
      f'agregação/seleção {t_sel:.3f}s = {t_graph + t_curv + t_sel:.3f}s')
csv = {'indice': np.arange(n), 'rotulo': y, 'destacado': ends.astype(int), 'n_arestas': hits}
if red.any():
    csv['destacado_borda'] = red.astype(int)
if b_node is not None:
    csv['b_borda'] = b_node
pd.DataFrame(csv).to_csv(f'{name}_{TAG}_endpoints.csv', index=False)

if args.validate:                                         # rótulos: só para conferência no terminal
    from sklearn.neighbors import NearestNeighbors
    Ak = NearestNeighbors(n_neighbors=args.k_graph).fit(X).kneighbors(return_distance=False)
    border = (y[Ak] != y[:, None]).any(1)
    print(f'[validação] borda verdadeira (algum vizinho de outra classe): {border.mean():.1%} dos vértices')
    if bottom:
        rk = NearestNeighbors(n_neighbors=args.k_graph).fit(X).kneighbors()[0][:, -1]
        print(f'   destacados que são interior (sem vizinho de outra classe): {(~border[ends]).mean():.1%}  '
              f'(interior no total: {(~border).mean():.1%})')
        print(f'   raio k-NN mediano: destacados={np.median(rk[ends]):.2f}  todos={np.median(rk):.2f}  '
              f'(percentil médio do raio entre os destacados = {np.mean([np.mean(rk <= r) for r in rk[ends]]):.0%})')
    print(f'   destacados que são borda: {border[ends].mean():.1%}  ({border[ends].mean() / border.mean():.1f}× o acaso)')
    print(f'   bordas cobertas pelos destacados: {ends[border].mean():.1%}')
    print('   destacados por classe: ' + '  '.join(f'{classes[c]}:{ends[y == c].sum()}' for c in range(len(classes))))
    if red.any():
        print(f'   [vermelho] destacados que são borda: {border[red].mean():.1%}  '
              f'({border[red].mean() / border.mean():.1f}× o acaso); bordas cobertas: {red[border].mean():.1%}')
        print('   [vermelho] por classe: ' + '  '.join(f'{classes[c]}:{red[y == c].sum()}' for c in range(len(classes))))
    if lmin:                                             # comparação com o corte global do mesmo tamanho
        glob = np.zeros(n, bool); glob[np.argsort(b_node, kind='stable')[:ends.sum()]] = True
        print(f'   corte global com o mesmo nº ({ends.sum()}): interior = {(~border[glob]).mean():.1%}')
        print('   corte global por classe:   ' + '  '.join(f'{classes[c]}:{glob[y == c].sum()}' for c in range(len(classes))))
        for nm, sel in (('mínimos locais', ends), ('corte global', glob)):
            cnt = np.array([sel[y == c].sum() for c in range(len(classes))])
            print(f'   {nm:<15} classes cobertas: {(cnt > 0).sum()}/{len(classes)}  '
                  f'mín/máx por classe: {cnt.min()}/{cnt.max()}')


# ──────────────────────────────────────────────────────────────────────────────
# 2) Layout do grafo (espaço original, sem redução de dimensionalidade)
# ──────────────────────────────────────────────────────────────────────────────

layout = 'kamada_kawai' if args.layout == 'kawai' else args.layout
G = nx.Graph()
G.add_nodes_from(range(n))
d = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
s_med = np.median(d) if np.median(d) > 0 else 1.0
for (i, j), dij in zip(E, d):
    G.add_edge(int(i), int(j), sim=float(np.exp(-(dij / s_med) ** 2)), dist_n=max(dij / s_med, 1e-6))

t0 = time.time()
comps = sorted(nx.connected_components(G), key=len, reverse=True)
if len(comps) > 1:
    print(f'aviso: grafo com {len(comps)} componentes conexas; posicionadas separadamente')
pos = np.zeros((n, 2))
ncol_c = int(np.ceil(np.sqrt(len(comps))))
cell = 2.2 * np.sqrt(len(comps[0]) / n)
for ci, comp in enumerate(comps):
    H = G.subgraph(comp)
    if H.number_of_nodes() <= 2:
        p = nx.spring_layout(H, seed=args.seed)
    else:
        p = nx.spring_layout(H, weight='sim', seed=args.seed, iterations=200)
        if layout == 'kamada_kawai':
            p = nx.kamada_kawai_layout(H, weight='dist_n', pos=p)
    P_ = np.array([p[v] for v in comp]); P_ -= P_.mean(0)
    P_ = P_ / (np.abs(P_).max() or 1.0) * np.sqrt(len(comp) / n)
    if len(comps) > 1:
        r_, c_ = divmod(ci, ncol_c); P_ += np.array([c_ * cell, -r_ * cell])
    pos[list(comp)] = P_
print(f'layout {layout}: {time.time() - t0:.1f}s')


# ──────────────────────────────────────────────────────────────────────────────
# 3) Figura: só os vértices extremos destacados
# ──────────────────────────────────────────────────────────────────────────────

C_HI, C_EDGE = ('#1f5fbf', '#0d2f66') if bottom else ('#c62828', '#5a1010')
fig, ax = plt.subplots(figsize=(11, 9), dpi=150)
segs = pos[E]
ax.add_collection(LineCollection(segs, colors='#c9c8c3', linewidths=0.25, alpha=0.5, zorder=1))
if args.show_edges:
    ax.add_collection(LineCollection(segs[top], colors=C_HI, linewidths=0.6, alpha=0.35, zorder=2))
rest = ~ends & ~red
ax.scatter(pos[rest, 0], pos[rest, 1], s=7, c='#b9b8b3', alpha=0.7, linewidths=0, zorder=3)
if b_node is not None:                                    # tamanho cresce com o posto de b entre os destacados
    rb = np.argsort(np.argsort(-b_node[ends] if bottom else b_node[ends])) / max(ends.sum() - 1, 1)
    sz = 14 + 40 * rb
else:
    sz = 14 + 10 * np.minimum(hits[ends] - args.min_hits, 4)              # tamanho cresce com nº de arestas selecionadas
ax.scatter(pos[ends, 0], pos[ends, 1], s=sz, c=C_HI, edgecolors=C_EDGE, linewidths=0.4,
           alpha=0.9, zorder=4)
if red.any():                                             # vermelho: maiores pontuações de borda
    rr = np.argsort(np.argsort(b_node[red])) / max(red.sum() - 1, 1)
    ax.scatter(pos[red, 0], pos[red, 1], s=14 + 40 * rr, c='#c62828', edgecolors='#5a1010',
               linewidths=0.4, alpha=0.9, zorder=4)
if len(classes) <= 20:
    for c, cname in enumerate(classes):
        cx, cy = np.median(pos[y == c], axis=0)
        ax.text(cx, cy, str(cname), fontsize=12 if len(str(cname)) < 3 else 8, weight='bold',
                color='#2b2b28', ha='center', va='center', zorder=6,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#b9b8b3', lw=0.6, alpha=0.85))
from matplotlib.lines import Line2D as _L
ax.legend(handles=([_L([], [], ls='', marker='o', ms=6, mfc='#c62828', mec='#5a1010',
                       label=f'{red.mean():.0%} dos vértices com maior pontuação de borda ({red.sum()} vértices)')]
                   if red.any() else []) + [_L([], [], ls='', marker='o', ms=6, mfc=C_HI, mec=C_EDGE,
                      label=(f'mínimos locais da pontuação de borda (raio {args.lm_hops}) ({args.agg} de −κ)' if lmin else
                             f'{ends.mean():.0%} dos vértices com {"menor" if bottom else "maior"} pontuação de borda ({args.agg} de −κ)'
                             if b_node is not None else
                             f'extremidade de aresta com κ_G no 1º {n_top / len(E):.0%}' if args.min_hits == 1 else
                             f'tocado por ≥ {args.min_hits} arestas com κ_G no 1º {n_top / len(E):.0%}')
                            + f' ({ends.sum()} vértices)'),
                   _L([], [], ls='', marker='o', ms=4, mfc='#b9b8b3', mec='none',
                      label='demais vértices' if (args.min_hits == 1 or b_node is not None) else
                      f'demais vértices (incl. {((hits > 0) & ~ends).sum()} com < {args.min_hits} arestas selecionadas)')],
          loc='lower left', fontsize=9, frameon=True, framealpha=0.9,
          title=('tamanho ∝ força do sinal (borda ou interior)' if red.any() else
                 ('tamanho ∝ quão interior (κ mais positivo)' if bottom else 'tamanho ∝ pontuação de borda')
                 if b_node is not None else
                 'tamanho ∝ nº de arestas selecionadas incidentes'), title_fontsize=8)
if b_node is not None:
    ttl = (f'{name} — grafo {args.k_graph}-NN: {ends.sum()} mínimos locais (raio {args.lm_hops}) da pontuação de borda '
           f'{ORC} ({args.agg}' + (f' + suavização {args.smooth}' if args.smooth else '') + f', k_gauss={args.k_gauss})'
           ) if lmin else (f'{name} — grafo {args.k_graph}-NN: {ends.sum()} vértices ({ends.mean():.0%}) de '
           f'{"MENOR pontuação de borda (interior)" if bottom else "maior pontuação de borda"} '
           f'{ORC} ({args.agg}' + (f' + suavização {args.smooth}' if args.smooth else '') + f', k_gauss={args.k_gauss})')
if red.any():
    ttl = (f'{name} — grafo {args.k_graph}-NN, pontuação de borda {ORC} ({args.agg}'
           + (f' + suavização {args.smooth}' if args.smooth else '') + f', k_gauss={args.k_gauss}): '
           f'azul = mínimos locais (raio {args.lm_hops}), vermelho = {red.mean():.0%} maiores')
ax.set_title(ttl if b_node is not None else f'{name} — grafo {args.k_graph}-NN: vértices tocados por '
             f'{"≥ " + str(args.min_hits) + " das " if args.min_hits > 1 else ""}{n_top} arestas de menor {ORC} '
             f'(k_gauss={args.k_gauss})', fontsize=11.5, color='#2b2b28', loc='left')
ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.autoscale()
fig.text(0.01, 0.0, f'Layout {"Kamada–Kawai" if layout == "kamada_kawai" else "spring"} do grafo k-NN no espaço '
         f'original ℝ^{m}. Rótulos só no centro de cada classe (referência); não usados na seleção.  '
         f'Cálculo das curvaturas ({ORC}, {len(E)} arestas): {t_curv:.2f}s.',
         fontsize=8.5, color='#7a7974', va='bottom')
plt.tight_layout(rect=(0, 0.02, 1, 1))
suffix = (f'_both_lm{args.lm_hops}_top{int(round(100 * red.mean()))}_{args.agg}' if red.any() else
          f'_localmin{args.lm_hops}_{args.agg}' if lmin else
          f'_{"bottom" if bottom else "nodes"}{int(round(100 * ends.mean()))}_{args.agg}' if b_node is not None else
          (f'_h{args.min_hits}' if args.min_hits > 1 else ''))
out = args.out or f'{name}_{TAG}_endpoints{suffix}.png'
plt.savefig(out, bbox_inches='tight')
print('salvo em', out)
