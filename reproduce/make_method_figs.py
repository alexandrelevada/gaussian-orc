"""Figuras da seção de método (dados 2D sintéticos, G-ORC com covariância 1/k)."""
import os
from paths import DATA, RAW, FIGS, TABLES
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'font.family': 'DejaVu Sans'})
INK, MUTED, GRID = '#2b2b28', '#7a7974', '#c9c8c3'
RED, BLUE = '#c62828', '#1f5fbf'
DIV = LinearSegmentedColormap.from_list('div', ['#8f1f1e', '#e34948', '#f2a5a4', '#e6e5e1',
                                                '#9ec5f4', '#3987e5', '#104281'])   # κ: neg = vermelho
SEQ = LinearSegmentedColormap.from_list('seq', ['#104281', '#3987e5', '#9ec5f4', '#e6e5e1',
                                                '#f2a5a4', '#e34948', '#8f1f1e'])   # b: alto = vermelho
rng = np.random.default_rng(3)

# ------------------------------------------------------------------ dados
def rot(a): return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
from sklearn.datasets import make_moons
Mo, ml = make_moons(520, noise=0.10, random_state=2)
A, B = Mo[ml == 0], Mo[ml == 1]
C = rng.normal(size=(110, 2)) * [0.16, 0.10] + [2.40, 0.80]
X = np.vstack([A, B, C]); n = len(X)
lab = np.r_[np.zeros(len(A)), np.ones(len(B)), 2 * np.ones(len(C))]
KG, K = 8, 15

def knn_edges(X, kg):
    Aidx = NearestNeighbors(n_neighbors=kg).fit(X).kneighbors(return_distance=False)
    E = {(min(i, int(j)), max(i, int(j))) for i in range(len(X)) for j in Aidx[i]}
    while True:                                    # conecta componentes (Borůvka, como no código)
        I, J = np.array(list(E)).T
        nc, cl = connected_components(coo_matrix((np.ones(len(I)), (I, J)), shape=(len(X),) * 2), directed=False)
        if nc == 1: break
        for c in range(nc):
            ins, outs = np.flatnonzero(cl == c), np.flatnonzero(cl != c)
            d, ix = NearestNeighbors(n_neighbors=1).fit(X[outs]).kneighbors(X[ins])
            a = int(np.argmin(d[:, 0])); E.add(tuple(sorted((int(ins[a]), int(outs[ix[a, 0]])))))
    return np.array(sorted(E))

E = knn_edges(X, KG)
idx = NearestNeighbors(n_neighbors=K).fit(X).kneighbors(X, return_distance=False)
MU = X[idx].mean(1)
P = (X[idx] - MU[:, None]) / np.sqrt(K)
SIG = np.einsum('nki,nkj->nij', P, P)

def gorc(i, j):
    d = np.linalg.norm(X[i] - X[j])
    dm = np.linalg.norm(MU[i] - MU[j])
    nuc = np.linalg.svd(P[i] @ P[j].T, compute_uv=False).sum()
    b2 = max(np.trace(SIG[i]) + np.trace(SIG[j]) - 2 * nuc, 0)
    return 1 - np.sqrt(dm ** 2 + b2) / d, dm / d, np.sqrt(b2) / d

R = np.array([gorc(i, j) for i, j in E]); kap = R[:, 0]

def node_scores(kap, tau=None):
    nk = -kap; tau = tau or np.std(nk)
    inc = [[] for _ in range(n)]
    for e, (i, j) in enumerate(E): inc[i].append(e); inc[j].append(e)
    b = np.empty(n)
    for i in range(n):
        v = nk[inc[i]]; mx = v.max(); b[i] = mx + tau * np.log(np.mean(np.exp((v - mx) / tau)))
    return b, inc, tau

b, inc, tau0 = node_scores(kap)
nbrs = [[] for _ in range(n)]
for i, j in E: nbrs[i].append(j); nbrs[j].append(i)
bm = b.copy()
for _ in range(2): bm = np.array([min(bm[i], bm[v].min()) for i, v in enumerate(nbrs)])
cores = b <= bm + 1e-12
top = np.zeros(n, bool); top[np.argsort(-b)[:int(round(0.05 * n))]] = True
print(f'n={n} |E|={len(E)} κ: min={kap.min():.2f} med={np.median(kap):.2f} max={kap.max():.2f}; '
      f'interface={top.sum()} núcleos={cores.sum()} (por grupo: {[int(cores[lab == c].sum()) for c in range(4)]})')

def ellipse(ax, mu, S, color, lw=1.4, ls='-'):
    w, V = np.linalg.eigh(S); ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    ax.add_patch(Ellipse(mu, 2 * 2 * np.sqrt(max(w[1], 1e-12)), 2 * 2 * np.sqrt(max(w[0], 1e-12)),
                         angle=ang, fill=False, ec=color, lw=lw, ls=ls, zorder=5))

def base(ax, alpha_e=0.35):
    ax.add_collection(LineCollection(X[E], colors=GRID, lw=0.4, alpha=alpha_e, zorder=1))
    ax.scatter(*X.T, s=4, c='#9a9994', lw=0, zorder=2)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

# ------------------------------------------------------------------ Figura 1: intuição
cA = np.flatnonzero(lab == 0)
def typical(cands):                               # aresta de κ mediano entre as candidatas
    cands = [e for e in cands if len(set(idx[E[e][0]]) & set(idx[E[e][1]])) < K - 1]
    ks = kap[cands]; return cands[int(np.argsort(ks)[len(ks) // 2])]
radA = np.linalg.norm(X - [0, 0], axis=1); angA = np.arctan2(X[:, 1], X[:, 0])
mid = (lab == 0) & (np.abs(angA - np.pi / 2) < 0.35)
inner = set(np.flatnonzero(mid & (np.abs(radA - 1.0) < 0.05)))
outer = set(np.flatnonzero(mid & (radA > 1.08)))
e_int = typical([e for e, (i, j) in enumerate(E) if i in inner or j in inner and lab[i] == lab[j] == 0])
e_per = typical([e for e, (i, j) in enumerate(E) if i in outer and j in outer])
e_br = typical([e for e, (i, j) in enumerate(E) if lab[i] != lab[j]])
cases = [(e_int, '(a) Interior edge'), (e_per, '(b) Peripheral edge (facing empty space)'),
         (e_br, '(c) Edge between two regions')]
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), dpi=200)
for ax, (e, ttl) in zip(axes, cases):
    i, j = E[e]; kk, rm, rs = R[e]
    base(ax, 0.25)
    for v, col in ((i, BLUE), (j, RED)):
        ax.scatter(*X[idx[v]].T, s=14, facecolors='none', edgecolors=col, lw=0.8, zorder=3)
        ellipse(ax, MU[v], SIG[v], col)
        ax.scatter(*MU[v], marker='+', s=60, c=col, lw=1.6, zorder=6)
    ax.plot(*X[[i, j]].T, c=INK, lw=2.4, zorder=7)
    ax.scatter(*X[[i, j]].T, s=36, c=[BLUE, RED], edgecolors=INK, lw=0.8, zorder=8)
    c = X[[i, j]].mean(0); half = 0.75
    ax.set_xlim(c[0] - half * 1.35, c[0] + half * 1.35); ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_title(ttl, loc='left', color=INK)
    sign = 'positive' if kk > 0.05 else ('negative' if kk < -0.05 else '≈ 0')
    ax.text(0.02, 0.03, f'$\\rho_\\mu$ = {rm:.2f}   $\\rho_\\Sigma$ = {rs:.2f}\n'
            f'$\\kappa_G$ = {kk:+.2f}  ({sign})', transform=ax.transAxes, fontsize=9.5, color=INK,
            va='bottom', bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=GRID, alpha=0.95))
axes[0].legend(handles=[plt.Line2D([], [], c=INK, lw=2.4, label='edge $(i,j)$'),
                        plt.Line2D([], [], c=BLUE, lw=1.4, label='$\\mathcal{N}(\\bar{x}_i,\\Sigma_i)$, 2σ'),
                        plt.Line2D([], [], c=RED, lw=1.4, label='$\\mathcal{N}(\\bar{x}_j,\\Sigma_j)$, 2σ'),
                        plt.Line2D([], [], ls='', marker='+', ms=8, mew=1.6, c=MUTED, label='local means')],
               loc='upper left', fontsize=7.5, frameon=True, framealpha=0.95)
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig_gorc_intuition.pdf'), bbox_inches='tight'); plt.savefig(os.path.join(FIGS, 'fig_gorc_intuition.png'), bbox_inches='tight')
print('fig1 casos (κ, ρμ, ρΣ):', [tuple(np.round(R[e], 3)) for e, _ in cases])

# ------------------------------------------------------------------ Figura 2: pipeline
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), dpi=200)
ax = axes[0]; base(ax, 0.0)
lim = np.quantile(np.abs(kap), 0.98)
o = np.argsort(-kap)
lc = LineCollection(X[E][o], cmap=DIV, norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
                    lw=0.4 + 2.0 * (kap[o] < 0), zorder=3)
lc.set_array(kap[o]); ax.add_collection(lc)
cb = fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.01, extend='both'); cb.set_label('$\\kappa_G(i,j)$'); cb.outline.set_visible(False)
ax.set_title('(a) Edge curvature $\\kappa_G$', loc='left')
ax = axes[1]; base(ax)
r = np.argsort(np.argsort(b)) / (n - 1); o = np.argsort(r)
sc = ax.scatter(*X[o].T, c=r[o], cmap=SEQ, s=12, edgecolors='#8a8985', lw=0.2, zorder=4)
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01); cb.set_label('boundary score $b_i$ (percentile)'); cb.outline.set_visible(False)
ax.set_title('(b) Vertex boundary score (soft max)', loc='left')
ax = axes[2]; base(ax)
ax.scatter(*X[top].T, s=34, c=RED, edgecolors='#5a1010', lw=0.5, zorder=5, label=f'interface: top 5% of $b_i$ ({top.sum()})')
ax.scatter(*X[cores].T, s=60, c=BLUE, marker='D', edgecolors='#0d2f66', lw=0.6, zorder=5, label=f'anchors: local minima of $b_i$, $h=2$ ({cores.sum()})')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=1, fontsize=7.5, frameon=False)
ax.set_title('(c) Interface vertices and anchors', loc='left')
for a in axes: a.autoscale()
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig_gorc_pipeline.pdf'), bbox_inches='tight'); plt.savefig(os.path.join(FIGS, 'fig_gorc_pipeline.png'), bbox_inches='tight')

# ------------------------------------------------------------------ Figura 3: agregação
iface = int(np.argmax(b))
ints = [v for v in np.flatnonzero(lab == 0) if len(inc[v]) >= 6]
interior = min(ints, key=lambda v: abs(b[v] - np.median(b)))
mixed = min([v for v in range(n) if len(inc[v]) >= 8], key=lambda v: abs((-kap[inc[v]]).max() - 0.9))
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), dpi=200, gridspec_kw={'width_ratios': [1, 1.25]})
ax = axes[0]
ex = [(interior, 'interior vertex', BLUE), (mixed, 'vertex with a mild bridge', '#e8710a'), (iface, 'interface vertex', RED)]
for yy, (v, nm, col) in enumerate(ex):
    vals = -kap[inc[v]]
    ax.scatter(vals, np.full(len(vals), yy) + rng.uniform(-0.12, 0.12, len(vals)), s=22, c=col, alpha=0.85, lw=0)
    ax.scatter([vals.mean()], [yy + 0.32], marker='v', s=40, c=MUTED); ax.scatter([vals.max()], [yy + 0.32], marker='v', s=40, c=INK)
ax.axvline(0, color=GRID, lw=0.8)
ax.set_yticks(range(3)); ax.set_yticklabels([e[1] for e in ex]); ax.set_ylim(-0.5, 2.6)
ax.set_xlabel('$-\\kappa_G$ of incident edges'); ax.set_title('(a) Incident edge scores (▼ grey: mean, ▼ black: max)', loc='left', fontsize=9)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax = axes[1]
ts = np.logspace(-2.5, 1.5, 200)
for v, nm, col in ex:
    vals = -kap[inc[v]]; mx = vals.max()
    Sv = [mx + t * np.log(np.mean(np.exp((vals - mx) / t))) for t in ts]
    ax.plot(ts, Sv, c=col, lw=2, label=nm)
    ax.plot(ts[[0, -1]], [vals.max()] * 2, c=col, lw=0.8, ls=':'); ax.plot(ts[[0, -1]], [vals.mean()] * 2, c=col, lw=0.8, ls='--')
ax.axvline(tau0, c=INK, lw=1, ls='-.'); ax.text(tau0, ax.get_ylim()[1], '  default $\\tau$', va='top', fontsize=8, color=INK)
ax.set_xscale('log'); ax.set_xlabel('temperature $\\tau$'); ax.set_ylabel('$S_\\tau$ (soft max of $-\\kappa_G$)')
ax.set_title('(b) Soft max interpolates between max (dotted) and mean (dashed)', loc='left', fontsize=9)
ax.legend(fontsize=7.5, frameon=False, loc='center right')
for s in ('top', 'right'): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(FIGS, 'fig_gorc_aggregation.pdf'), bbox_inches='tight'); plt.savefig(os.path.join(FIGS, 'fig_gorc_aggregation.png'), bbox_inches='tight')
print('fig3 vértices (v, grau, max -κ, média -κ, S_tau):', [(int(v), len(inc[v]), round(float((-kap[inc[v]]).max()), 2), round(float((-kap[inc[v]]).mean()), 2), round(float(b[v]), 2)) for v, _, _ in ex], 'tau0=', round(tau0, 3))
