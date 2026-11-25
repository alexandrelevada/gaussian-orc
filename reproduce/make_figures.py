import os
from paths import DATA, RAW, FIGS, TABLES
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from datasets import load, ALL

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
RED, BLUE, GREY, INK = '#b2182b', '#2166ac', '#9a9a9a', '#333333'
C_G, C_U, C_B = '#b2182b', '#2166ac', '#7a7a7a'      # G-ORC, ORC uniforme, baseline
NAMES = {'iris': 'Iris', 'wine': 'Wine', 'breast_cancer': 'Breast cancer', 'digits': 'Digits', 'segmentation': 'Segmentation',
         'satimage': 'Satimage', 'texture': 'Texture', 'optdigits': 'Optdigits', 'pendigits': 'Pendigits',
         'letter': 'Letter', 'mnist': 'MNIST', 'fashion_mnist': 'Fashion-MNIST'}
OUT = FIGS + '/'
Q = 0.05
def topq(s, q=Q): return np.argsort(-s)[:max(1, int(np.ceil(q * len(s))))]
def npz(n): return np.load(f'{RAW}/{n}.npz')

# ---------------------------------------------------------------- 1. qualitativo (t-SNE só para exibição)
def qualitative():
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.0))
    for r, name in enumerate(('digits', 'mnist')):
        X, y, _ = load(name); D = npz(name); E = D['E']
        Z = TSNE(2, init='pca', random_state=0, perplexity=30).fit_transform(PCA(50, random_state=0).fit_transform(X) if X.shape[1] > 50 else X)
        cls = plt.cm.tab10(y % 10)
        ps = 3 if len(X) < 3000 else 0.6
        for c, (key, ttl) in enumerate((('kG', 'G-ORC: top 5% edges'), ('kU', 'Uniform ORC: top 5% edges'))):
            ax = axs[r, c]; t = topq(-D[key], Q if name == 'digits' else 0.01)
            ax.scatter(*Z.T, s=ps, c=cls, alpha=0.35, lw=0, rasterized=True)
            inter = y[E[t, 0]] != y[E[t, 1]]
            ax.add_collection(LineCollection(Z[E[t]], colors=RED, lw=0.35, alpha=0.8, rasterized=True))
            ttl = ttl if name == 'digits' else ttl.replace('5%', '1%')
            ax.set_title(f'{NAMES[name]} — {ttl}\n{inter.mean()*100:.0f}% inter-class (base {np.mean(y[E[:,0]]!=y[E[:,1]])*100:.0f}%)')
        ax = axs[r, 2]; b = D['bG']; I = topq(b); A = D['aG']
        ax.scatter(*Z.T, s=ps, c=cls, alpha=0.25, lw=0, rasterized=True)
        ax.scatter(*Z[I].T, s=ps * 4 + 2, c=RED, lw=0, label=f'interface (top 5%)', rasterized=True)
        ax.scatter(*Z[A].T, s=ps * 6 + 6, c=BLUE, marker='D', lw=0.3, edgecolors='white', label=f'anchors ($h$=2)', rasterized=True)
        ax.set_title(f'{NAMES[name]} — G-ORC vertices'); ax.legend(loc='lower left', frameon=False, markerscale=1.5, handletextpad=0.2)
        for a in axs[r]: a.set_xticks([]); a.set_yticks([]); [sp.set_visible(False) for sp in a.spines.values()]
    fig.text(0.5, 0.005, 'Points coloured by class (labels used only for display). t-SNE coordinates used only for visualization; all computations are in the original space.',
             ha='center', fontsize=6.5, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(OUT + 'fig_exp_qualitative.pdf', dpi=300); plt.close(fig)

# ---------------------------------------------------------------- 2. grades de imagens
def image_grids():
    fig = plt.figure(figsize=(7.2, 3.6))
    sub = fig.subfigures(2, 1, hspace=0.04)
    for r, name in enumerate(('mnist', 'fashion_mnist')):
        X, y, _, X0 = load(name, raw=True); D = npz(name); E = D['E']; L = D['length']
        from core import incidence, vertex_score, anchors as anch
        b = D['bG']; A = D['aG']; rng = np.random.default_rng(1)
        img = lambda i: X0[i].reshape(28, 28)
        axs = sub[r].subplots(2, 18, gridspec_kw=dict(wspace=0.05, hspace=0.05))
        t = rng.choice(topq(-D['kG'], 0.01), 6, replace=False)          # amostra das arestas do top 1%
        I = rng.choice(topq(b, Q), 12, replace=False)                    # amostra da interface (top 5%)
        Aa = rng.choice(A, 12, replace=False)                            # amostra das âncoras
        for c in range(6):
            for rr in range(2):
                ax = axs[rr, c]; i = E[t[c], rr]; ax.imshow(img(i), cmap='gray_r'); ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_color(RED); sp.set_visible(True)
                ax.set_xlabel(str(y[i]), fontsize=6, labelpad=1)
        for c in range(6):
            for rr in range(2):
                ax = axs[rr, 6 + c]; i = I[rr * 6 + c]; ax.imshow(img(i), cmap='gray_r'); ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_color(RED); sp.set_visible(True); sp.set_linestyle(':')
                ax.set_xlabel(str(y[i]), fontsize=6, labelpad=1)
                ax = axs[rr, 12 + c]; i = Aa[rr * 6 + c]; ax.imshow(img(i), cmap='gray_r'); ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_color(BLUE); sp.set_visible(True)
                ax.set_xlabel(str(y[i]), fontsize=6, labelpad=1)
        sub[r].suptitle(f'{NAMES[name]}:   edges from the top 1% (columns = endpoint pairs)'
                        f'   |   interface vertices (top 5%)   |   anchors', fontsize=8, y=0.99)
    fig.savefig(OUT + 'fig_exp_images.pdf', dpi=300); plt.close(fig)

# ---------------------------------------------------------------- 3. concordância G-ORC × uniforme
def agreement():
    names = ('breast_cancer', 'digits', 'letter', 'mnist')
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.3))
    for ax, n in zip(axs, names):
        D = npz(n); R = json.load(open(f'{RAW}/{n}.json'))
        a, b = D['kU'], D['kG']; lo, hi = np.quantile(np.r_[a, b], [0.002, 0.998])
        ax.hexbin(a, b, gridsize=45, extent=(lo, hi, lo, hi), bins='log', cmap='Greys', mincnt=1, lw=0)
        ax.plot([lo, hi], [lo, hi], color=RED, lw=0.7, ls='--')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
        ax.set_title(f"{NAMES[n]}\nSpearman $\\rho$={R['spearman_GU']:.3f}")
        ax.set_xlabel(r'uniform ORC $\kappa_1^{\mathrm{U}}$')
    axs[0].set_ylabel(r'G-ORC $\kappa_G$')
    fig.tight_layout(); fig.savefig(OUT + 'fig_exp_agreement.pdf'); plt.close(fig)

# ---------------------------------------------------------------- 4. tempo
def timing():
    T = json.load(open(f'{RAW}/timing.json')); K = json.load(open(f'{RAW}/kscan.json'))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    ax = axs[0]; ns = [n for n in ALL + ['mnist_pca50', 'fashion_mnist_pca50'] if n in T]; x = np.arange(len(ns))
    sb = [T[n]['u'] / T[n]['b'] for n in ns]; sl = [T[n]['u'] / T[n]['l'] for n in ns]
    ax.bar(x - 0.2, sb, 0.38, color=C_G, label='G-ORC, batched'); ax.bar(x + 0.2, sl, 0.38, color='#f4a582', label='G-ORC, per-edge loop')
    ax.axhline(1, color=INK, lw=0.6); ax.set_xticks(x); ax.set_xticklabels([NAMES.get(n, n.replace('fashion_mnist_pca50', 'F-MNIST PCA-50').replace('mnist_pca50', 'MNIST PCA-50')) for n in ns], rotation=60, ha='right', fontsize=6.2)
    ax.set_ylabel('speed-up over uniform ORC (×)'); ax.set_ylim(0, 26); ax.legend(frameon=False, loc='upper left', ncol=2, fontsize=6.5); ax.set_title('(a) Speed-up, $k$=20')
    mk = {'pendigits': 'o', 'digits': 's', 'mnist': '^'}
    for n, rs in K.items():
        k = [r['k'] for r in rs]; E = rs[0]['ne']
        axs[1].plot(k, [r['tu'] / E * 1e3 for r in rs], color=C_U, marker=mk[n], ms=3, lw=1)
        axs[1].plot(k, [r['tb'] / E * 1e3 for r in rs], color=C_G, marker=mk[n], ms=3, lw=1)
        axs[2].plot(k, [r['aucU'] for r in rs], color=C_U, marker=mk[n], ms=3, lw=1)
        axs[2].plot(k, [r['aucG'] for r in rs], color=C_G, marker=mk[n], ms=3, lw=1)
    for ax in axs[1:]:
        ax.set_xscale('log', base=2); ax.set_xlabel('measure neighbors $k$'); ax.set_xticks([10, 20, 40, 80, 160]); ax.set_xticklabels(['10', '20', '40', '80', '160']); ax.minorticks_off()
    axs[1].set_yscale('log'); axs[1].set_ylabel('ms per edge'); axs[1].set_title('(b) Cost vs. $k$')
    axs[2].set_ylabel('edge AUROC (inter-class)'); axs[2].set_title('(c) Detection vs. $k$')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C_G, lw=2.2, label='G-ORC'), Line2D([], [], color=C_U, lw=2.2, label='uniform ORC')] + \
        [Line2D([], [], color=INK, marker=mk[n], ls='', ms=3, label=NAMES[n] + f" ($m$={K[n][0]['m']})") for n in K]
    fig.legend(handles=h, frameon=False, loc='lower center', ncol=5, fontsize=6.5, bbox_to_anchor=(0.66, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT + 'fig_exp_timing.pdf'); plt.close(fig)

# ---------------------------------------------------------------- 5. precisão@q
def precision():
    names = ('wine', 'digits', 'letter', 'mnist')
    qs = np.linspace(0.01, 0.30, 30)
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.6), sharex=True)
    for c, n in enumerate(names):
        D = npz(n); y = D['y']; E = D['E']; inter = y[E[:, 0]] != y[E[:, 1]]; bnd = D['nb_lab'] > 0
        for s, col, lab in ((-D['kG'], C_G, 'G-ORC'), (-D['kU'], C_U, 'uniform ORC'), (D['length'], C_B, 'edge length')):
            axs[0, c].plot(qs * 100, [inter[topq(s, q)].mean() for q in qs], color=col, lw=1.1, label=lab, ls='-' if col != C_B else '--')
        axs[0, c].axhline(inter.mean(), color=INK, lw=0.6, ls=':')
        for s, col, lab in ((D['bG'], C_G, 'G-ORC'), (D['bU'], C_U, 'uniform ORC'), (D['rad'], C_B, '$k$-NN radius')):
            axs[1, c].plot(qs * 100, [bnd[topq(s, q)].mean() for q in qs], color=col, lw=1.1, label=lab, ls='-' if col != C_B else '--')
        axs[1, c].axhline(bnd.mean(), color=INK, lw=0.6, ls=':', label='base rate')
        axs[0, c].set_title(NAMES[n]); axs[1, c].set_xlabel('selected fraction $q$ (%)')
    axs[0, 0].set_ylabel('inter-class edges\namong selected'); axs[1, 0].set_ylabel('boundary vertices\namong selected')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C_G, lw=1.5, label='G-ORC'), Line2D([], [], color=C_U, lw=1.5, label='uniform ORC'),
         Line2D([], [], color=C_B, lw=1.2, ls='--', label='baseline: edge length (top) / $k$-NN radius (bottom)'),
         Line2D([], [], color=INK, lw=0.8, ls=':', label='base rate')]
    fig.legend(handles=h, frameon=False, loc='lower center', ncol=4, fontsize=6.8)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(OUT + 'fig_exp_precision.pdf'); plt.close(fig)

if __name__ == '__main__':
    import sys
    for f in (sys.argv[1:] or ['agreement', 'precision', 'image_grids', 'qualitative', 'timing']):
        globals()[f](); print('ok', f, flush=True)
