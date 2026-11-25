#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaussian Ollivier–Ricci curvature (G-ORC) on k-NN graphs  — versão 2 (artigo)

@author: Alexandre L. M. Levada

Cada vértice i é representado pela gaussiana local N(μ_i, Σ_i), estimada nos
k_gauss vizinhos mais próximos (incluindo o próprio ponto). Para cada aresta (i, j):

    κ_G(i, j) = 1 − W₂(N_i, N_j) / ‖x_i − x_j‖

    W₂² = ‖μ_i − μ_j‖² + tr Σ_i + tr Σ_j − 2 tr( (Σ_i^½ Σ_j Σ_i^½)^½ )

Novidades da versão 2 (compatível com a versão anterior)
---------------------------------------------------------
* ddof=0 (padrão): covariância com normalização 1/k, como no artigo (necessária para a
  Proposição 3, κ_G ≥ κ₂ᵁ). ddof=1 reproduz a versão anterior (1/(k−1)).
* batched=True (padrão): a forma fechada é avaliada em LOTES de arestas com álgebra
  linear empilhada (é o "G-ORC (batched)" das tabelas); batched=False é o laço por aresta
  ("G-ORC (loop)"), com a mesma estrutura do ORC uniforme.
* forma k×k: fidelidade = ‖P_i P_jᵀ‖_* = Σ √λ(M Mᵀ), M = P_i P_jᵀ (eigvalsh em vez de SVD).
* idx=: vizinhanças pré-calculadas (para G-ORC e ORC uniforme usarem EXATAMENTE as mesmas).
* uniform_orc(p=1|2): W₁ (ORC clássico) ou W₂ (κ₂ᵁ, para conferir a Proposição 3).
* local_minima(): âncoras (mínimos locais da pontuação em h saltos), por propagação de mínimos.

Escolha automática da forma (method='auto'): k×k se k ≤ m, m×m se k > m;
custo por aresta O(min(k, m)³) (+ O(k² m) para formar as matrizes na forma k×k).
"""

import os
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed


def _nj(n_jobs):
    return max(1, os.cpu_count() if n_jobs in (-1, None) else n_jobs)


def knn_graph_edges(X, k_graph, connect='components', return_lengths=False, return_added=False):
    """
    Arestas não direcionadas (i < j) do grafo k-NN, com garantia de conexidade.

    connect:
      None          grafo k-NN puro (pode ser desconexo)
      'components'  (padrão) liga as componentes do k-NN pelas arestas mais curtas entre
                    elas (Borůvka sobre as componentes = arestas da MST euclidiana que
                    atravessam componentes)
      'mst'         une o k-NN com a MST euclidiana completa (forma a matriz n×n)
    return_added: devolve também as arestas adicionadas para conectar (pontes de longo alcance).
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    A = NearestNeighbors(n_neighbors=k_graph).fit(X).kneighbors(return_distance=False)
    E = {(min(i, int(j)), max(i, int(j))) for i in range(n) for j in A[i]}
    knn = set(E)
    if connect == 'mst':
        T = minimum_spanning_tree(squareform(pdist(X))).tocoo()
        E |= {(min(i, j), max(i, j)) for i, j in zip(T.row.tolist(), T.col.tolist())}
    elif connect == 'components':
        while True:
            I, J = np.array(list(E)).T
            G = coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n))
            nc, lab = connected_components(G, directed=False)
            if nc == 1:
                break
            for c in range(nc):
                inside = np.flatnonzero(lab == c)
                outside = np.flatnonzero(lab != c)
                d, idx = NearestNeighbors(n_neighbors=1).fit(X[outside]).kneighbors(X[inside])
                a = int(np.argmin(d[:, 0]))
                i, j = int(inside[a]), int(outside[idx[a, 0]])
                E.add((min(i, j), max(i, j)))
    elif connect is not None:
        raise ValueError("connect deve ser None, 'components' ou 'mst'")
    added = np.array(sorted(E - knn), dtype=int).reshape(-1, 2)
    E = np.array(sorted(E))
    out = [E]
    if return_lengths:
        out.append(np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1))
    if return_added:
        out.append(added)
    return out[0] if len(out) == 1 else tuple(out)


def neighborhoods(X, k):
    """Índices dos k vizinhos mais próximos de cada ponto, incluindo o próprio ponto."""
    return NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X, return_distance=False)


def local_gaussians(X, k_gauss, ddof=0):
    """μ_i (n×m) e fatores P_i (n×k×m) com Σ_i = P_iᵀ P_i (normalização 1/(k − ddof))."""
    idx = neighborhoods(X, k_gauss)
    Nb = X[idx]
    mu = Nb.mean(axis=1)
    P = (Nb - mu[:, None, :]) / np.sqrt(k_gauss - ddof)
    return mu, P


def gaussian_orc(X, E=None, k_graph=10, k_gauss=20, n_jobs=1, method='auto', max_mem_gb=4.0,
                 batched=True, ddof=0, idx=None, chunk=512):
    """
    Curvatura de Ollivier–Ricci gaussiana de cada aresta.

    Parameters
    ----------
    X        : ndarray (n, m)
    E        : ndarray (n_e, 2) de arestas; None → grafo k_graph-NN conexo
    k_graph  : vizinhos do grafo (se E for None)
    k_gauss  : vizinhos de cada gaussiana local (ignorado se idx for dado)
    n_jobs   : 1 sequencial, -1 todas as CPUs (threads)
    method   : 'auto' (k×k se k ≤ m, m×m se k > m), 'kxk' ou 'mxm'
    max_mem_gb : limite para guardar os fatores P_i (n·k·m) ou Σ_i, Σ_i^½ (2·n·m²);
                 na forma k×k, acima do limite os fatores são formados por lote (mais lento)
    batched  : True → lotes de arestas (álgebra linear empilhada); False → laço por aresta
    ddof     : 0 → covariância 1/k (artigo); 1 → 1/(k−1) (versão anterior)
    idx      : (n, k) vizinhanças pré-calculadas (incluindo o próprio ponto)

    Returns
    -------
    E (n_e, 2), kappa (n_e,)  — κ_G; quanto mais negativo, mais "ponte".
    """
    X = np.asarray(X, dtype=float)
    n, m = X.shape
    if E is None:
        E = knn_graph_edges(X, k_graph)
    E = np.asarray(E)
    if idx is None:
        idx = neighborhoods(X, k_gauss)
    k = idx.shape[1]
    if method == 'auto':
        method = 'kxk' if k <= m else 'mxm'
    if method not in ('kxk', 'mxm'):
        raise ValueError("method deve ser 'auto', 'kxk' ou 'mxm'")
    nj = _nj(n_jobs)
    nodes, inv = np.unique(E, return_inverse=True)        # só os vértices que aparecem em E
    El = inv.reshape(E.shape)
    I = idx[nodes]
    c = 1.0 / np.sqrt(k - ddof)
    mu = np.concatenate([X[I[s:s + 256]].mean(1) for s in range(0, len(nodes), 256)])
    sq = (X ** 2).sum(1)
    tr = (sq[I].sum(1) - k * (mu ** 2).sum(1)) / (k - ddof)   # tr Σ_i
    Xn = X[nodes]
    d = np.linalg.norm(Xn[El[:, 0]] - Xn[El[:, 1]], axis=1)
    dm = ((mu[El[:, 0]] - mu[El[:, 1]]) ** 2).sum(1)

    if method == 'kxk':
        def P0(i):                                          # fatores (lote, k, m)
            return (X[I[i]] - mu[i][:, None, :]) * c
        Pst = None
        if len(nodes) * k * m * 8 / 1e9 <= max_mem_gb:
            Pst = np.concatenate([P0(np.arange(s, min(s + 256, len(nodes)))) for s in range(0, len(nodes), 256)])
        P = (lambda i: Pst[i]) if Pst is not None else P0

        def fid(i, j):                                      # lote
            M = P(i) @ np.swapaxes(P(j), 1, 2)
            return np.sqrt(np.clip(np.linalg.eigvalsh(M @ np.swapaxes(M, 1, 2)), 0, None)).sum(-1)

        def fid1(i, j):                                     # uma aresta
            M = (Pst[i] if Pst is not None else P0(np.array([i]))[0]) @ \
                (Pst[j] if Pst is not None else P0(np.array([j]))[0]).T
            return np.sqrt(np.clip(np.linalg.eigvalsh(M @ M.T), 0, None)).sum()
    else:
        mem = 2 * len(nodes) * m * m * 8 / 1e9
        if mem > max_mem_gb:
            raise MemoryError(f"forma 'mxm' precisa de ~{mem:.1f} GB (limite {max_mem_gb} GB); "
                              f"aumente max_mem_gb ou use method='kxk'")
        S = np.empty((len(nodes), m, m))
        for s in range(0, len(nodes), 256):
            Nb = (X[I[s:s + 256]] - mu[s:s + 256, None, :]) * c
            S[s:s + 256] = np.swapaxes(Nb, 1, 2) @ Nb
        w, V = np.linalg.eigh(S)
        Sh = (V * np.sqrt(np.clip(w, 0, None))[:, None, :]) @ np.swapaxes(V, 1, 2)

        def fid(i, j):
            lam = np.linalg.eigvalsh(Sh[i] @ S[j] @ Sh[i])
            return np.sqrt(np.clip(lam, 0, None)).sum(-1)

        def fid1(i, j):
            return np.sqrt(np.clip(np.linalg.eigvalsh(Sh[i] @ S[j] @ Sh[i]), 0, None)).sum()

    if batched:
        ch = max(1, min(chunk, int(1e7 / (k * m))))
        parts = [(s, min(s + ch, len(El))) for s in range(0, len(El), ch)]
        jobs = [p for p in np.array_split(np.array(parts, dtype=object), nj) if len(p)]

        def run(ps):
            return np.concatenate([fid(El[a:b, 0], El[a:b, 1]) for a, b in ps])
    else:
        jobs = [c_ for c_ in np.array_split(El, nj) if len(c_)]

        def run(c_):
            return np.array([fid1(i, j) for i, j in c_])
    if nj == 1:
        f = np.concatenate([run(j_) for j_ in jobs])
    else:
        f = np.concatenate(Parallel(n_jobs=nj, prefer='threads')(delayed(run)(j_) for j_ in jobs))
    w2 = np.maximum(dm + tr[El[:, 0]] + tr[El[:, 1]] - 2.0 * f, 0.0)
    return E, 1.0 - np.sqrt(w2) / np.maximum(d, 1e-12)


def _chunk_uniform(E, X, idx, sq, p):
    import ot
    k = idx.shape[1]
    w = np.full(k, 1.0 / k)
    it = max(100000, 50 * k * k)
    out = np.empty(len(E))
    for e, (i, j) in enumerate(E):
        ii, jj = idx[i], idx[j]
        # ‖a − b‖² = ‖a‖² + ‖b‖² − 2 aᵀb : uma multiplicação k×m · m×k, sem tensor k×k×m
        D2 = np.maximum(sq[ii][:, None] + sq[jj][None, :] - 2.0 * (X[ii] @ X[jj].T), 0.0)
        if p == 1:
            W = ot.emd2(w, w, np.sqrt(D2), numItermax=it)
        else:
            W = np.sqrt(max(ot.emd2(w, w, D2, numItermax=it), 0.0))
        out[e] = 1.0 - W / max(np.linalg.norm(X[i] - X[j]), 1e-12)
    return out


def uniform_orc(X, E=None, k_graph=10, k_meas=20, n_jobs=1, idx=None, p=1):
    """
    Curvatura de Ollivier–Ricci com a medida UNIFORME sobre os k_meas vizinhos (incluindo i).
        κ_p(i, j) = 1 − W_p(μ_i, μ_j) / ‖x_i − x_j‖   (p=1: ORC clássico; p=2: κ₂ᵁ da Prop. 3)
    W_p por transporte ótimo exato (POT, ot.emd2: simplex de rede).
    """
    X = np.asarray(X, dtype=float)
    if E is None:
        E = knn_graph_edges(X, k_graph)
    E = np.asarray(E)
    if idx is None:
        idx = neighborhoods(X, k_meas)
    sq = (X ** 2).sum(1)
    nj = _nj(n_jobs)
    if nj == 1:
        return E, _chunk_uniform(E, X, idx, sq, p)
    parts = Parallel(n_jobs=nj, prefer='threads')(
        delayed(_chunk_uniform)(c, X, idx, sq, p) for c in np.array_split(E, nj) if len(c))
    return E, np.concatenate(parts)


def node_boundary_score(n, E, kappa, how='softmax', q=0.8, tau=None, top_frac=0.05):
    """
    Pontuação de borda dos vértices b_i (maior = mais "borda"), a partir de −κ das arestas
    incidentes. how: 'mean' | 'max' | 'quantile' | 'softmax' (normalizado; τ=None → desvio-padrão
    de −κ) | 'top_frac'.  (Padrão agora é 'softmax', como no artigo.)
    """
    nk = -np.asarray(kappa, dtype=float)
    inc = [[] for _ in range(n)]
    for e, (i, j) in enumerate(E):
        inc[i].append(e); inc[j].append(e)
    if how == 'top_frac':
        thr = np.quantile(nk, 1 - top_frac)
    if how == 'softmax' and tau is None:
        tau = float(np.std(nk)) or 1.0
    out = np.full(n, np.nan)
    for i in range(n):
        if not inc[i]:
            continue
        v = nk[inc[i]]
        if how == 'mean':
            out[i] = v.mean()
        elif how == 'max':
            out[i] = v.max()
        elif how == 'quantile':
            out[i] = np.quantile(v, q)
        elif how == 'softmax':
            mx = v.max()
            out[i] = mx + tau * np.log(np.mean(np.exp((v - mx) / tau)))
        elif how == 'top_frac':
            out[i] = np.mean(v >= thr)
        else:
            raise ValueError(f'agregação desconhecida: {how}')
    return out


def graph_smooth(n, E, b, steps=1, alpha=0.5):
    """Suavização no grafo: b ← (1−α)·b + α·média dos vizinhos, repetida `steps` vezes."""
    nb = [[] for _ in range(n)]
    for i, j in E:
        nb[i].append(j); nb[j].append(i)
    b = np.array(b, dtype=float)
    for _ in range(steps):
        m_ = np.array([b[v].mean() if v else b[i] for i, v in enumerate(nb)])
        b = (1 - alpha) * b + alpha * m_
    return b


def local_minima(n, E, b, hops=2):
    """Âncoras: vértices com b_i ≤ b_j para todo j a até `hops` saltos (propagação de mínimos)."""
    E = np.asarray(E)
    c = np.asarray(b, dtype=float).copy()
    for _ in range(max(1, hops)):
        c2 = c.copy()
        np.minimum.at(c2, E[:, 0], c[E[:, 1]])
        np.minimum.at(c2, E[:, 1], c[E[:, 0]])
        c = c2
    return np.flatnonzero(np.asarray(b) <= c + 1e-12)


if __name__ == '__main__':
    from scipy.linalg import sqrtm
    rng = np.random.default_rng(0)
    print('1) Exatidão: formas k×k e m×m, lote e laço, contra a fórmula de Bures explícita')
    X = rng.normal(size=(300, 8)) @ rng.normal(size=(8, 8))
    for k in (6, 12, 40):
        E, kk = gaussian_orc(X, k_graph=5, k_gauss=k, method='kxk')
        _, km = gaussian_orc(X, E, k_gauss=k, method='mxm')
        _, kl = gaussian_orc(X, E, k_gauss=k, batched=False, n_jobs=2)
        mu, P = local_gaussians(X, k)
        ref = []
        for i, j in E[:60]:
            Si, Sj = P[i].T @ P[i], P[j].T @ P[j]
            Sh = sqrtm(Si).real
            w2 = np.sum((mu[i] - mu[j]) ** 2) + np.trace(Si + Sj - 2 * sqrtm(Sh @ Sj @ Sh).real)
            ref.append(1 - np.sqrt(max(w2, 0)) / np.linalg.norm(X[i] - X[j]))
        ref = np.array(ref)
        print(f'   k={k:<3} m=8: erro kxk={np.abs(kk[:60] - ref).max():.1e}  mxm={np.abs(km[:60] - ref).max():.1e}'
              f'  lote vs laço={np.abs(kk - kl).max():.1e}')
    print('2) Proposição 3: κ_G ≥ κ₂ᵁ ≤ κ₁ᵁ')
    E = knn_graph_edges(X, 5)[:200]
    idx = neighborhoods(X, 12)
    _, kg = gaussian_orc(X, E, idx=idx)
    _, k1 = uniform_orc(X, E, idx=idx, p=1)
    _, k2 = uniform_orc(X, E, idx=idx, p=2)
    print(f'   máx(κ₂ᵁ − κ_G) = {np.max(k2 - kg):.1e} (≤ 0)   máx(κ₂ᵁ − κ₁ᵁ) = {np.max(k2 - k1):.1e} (≤ 0)')
