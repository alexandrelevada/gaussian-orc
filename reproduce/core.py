"""Funções usadas pelos scripts de reprodução. Toda a matemática está em ../gaussian_orc.py;
este módulo só adapta as assinaturas (mesmo grafo e mesmas vizinhanças para G-ORC e ORC uniforme)."""
import os
import numpy as np
from paths import ROOT  # noqa: F401  (coloca a raiz do repositório no sys.path)
import gaussian_orc as G

NJ = os.cpu_count()


def knn_connected(X, kG):
    """Grafo kG-NN simétrico e conexo; devolve (E, arestas adicionadas para conectar)."""
    return G.knn_graph_edges(X, kG, connect='components', return_added=True)


def neighborhoods(X, k):
    return G.neighborhoods(X, k)


def gorc(X, E, idx, batched=True, n_jobs=NJ):
    """κ_G (covariância 1/k), forma k×k ou m×m escolhida automaticamente."""
    return G.gaussian_orc(X, E, idx=idx, batched=batched, n_jobs=n_jobs)[1]


def uorc(X, E, idx, p=1, n_jobs=NJ):
    """κ_p com medidas uniformes e W_p exato (POT)."""
    return G.uniform_orc(X, E, idx=idx, p=p, n_jobs=n_jobs)[1]


def incidence(n, E):
    inc = [[] for _ in range(n)]
    for e, (i, j) in enumerate(E):
        inc[i].append(e); inc[j].append(e)
    return [np.array(v) for v in inc]


def vertex_score(inc, kappa, how='softmax', tau=None, q=0.8, top_frac=0.05):
    """Pontuação de borda por vértice a partir das listas de incidência (ver G.node_boundary_score)."""
    s = -np.asarray(kappa)
    if tau is None:
        tau = float(np.std(s))
    thr = np.quantile(s, 1 - top_frac)
    out = np.empty(len(inc))
    for i, e in enumerate(inc):
        v = s[e]
        if how == 'mean': out[i] = v.mean()
        elif how == 'max': out[i] = v.max()
        elif how == 'quantile': out[i] = np.quantile(v, q)
        elif how == 'top_frac': out[i] = np.mean(v >= thr)
        else:
            u = v.max(); out[i] = u + tau * np.log(np.mean(np.exp((v - u) / tau)))
    return out


def anchors(n, E, b, h=2):
    return G.local_minima(n, E, b, h)
