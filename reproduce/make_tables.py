import os
from paths import DATA, RAW, FIGS, TABLES
import json, numpy as np
from datasets import ALL
from make_figures import NAMES
NAMES = dict(NAMES, mnist_pca50='MNIST (PCA-50)', fashion_mnist_pca50='Fashion-MNIST (PCA-50)')
OUT = TABLES + '/'
R = {n: json.load(open(f'{RAW}/{n}.json')) for n in ALL}
XT = json.load(open(f'{RAW}/extra.json'))
for n in ALL:
    R[n]['edge']['combo'] = XT[n]['combo']; R[n]['vertex']['jaccard'] = XT[n]['v_jaccard']; R[n]['vertex']['length'] = XT[n]['v_length']
def f(x, d=2): return f'{x:.{d}f}'
def bold_row(vals, d=2, hi=True):
    r = [round(v, d) for v in vals]; best = max(r) if hi else min(r)
    return [('\\textbf{' + f(v, d) + '}') if rv == best else f(v, d) for v, rv in zip(vals, r)]
def write(name, head, rows, foot=None, cols=None):
    s = '\\begin{tabular}{' + cols + '}\n\\toprule\n' + head + '\\midrule\n' + '\n'.join(r + ' \\\\' for r in rows) + '\n'
    if foot: s += '\\midrule\n' + '\n'.join(r + ' \\\\' for r in foot) + '\n'
    s += '\\bottomrule\n\\end{tabular}\n'
    open(OUT + name + '.tex', 'w').write(s)

# datasets
rows = [f"{NAMES[n]} & {R[n]['n']} & {R[n]['m']} & {R[n]['C']} & {R[n]['E']} & {R[n]['form'].replace('x','$\\times$')} & {100*R[n]['base_edge']:.1f} & {100*R[n]['base_vertex']:.1f}" for n in ALL]
write('tab_datasets', 'Dataset & $n$ & $m$ & $C$ & $|E|$ & Form & Inter-class edges (\\%) & Boundary vertices (\\%) \\\\\n', rows, cols='@{}lrrrrcrr@{}')

# agreement
keys = ['spearman_GU', 'jac_top_edges', 'spearman_bGU', 'jac_top_vertices']
rows = [f"{NAMES[n]} & " + ' & '.join(f(R[n][k]) for k in keys) + f" & {f(R[n]['anchors']['jac_GU'])}" for n in ALL]
mean = [np.mean([R[n][k] for n in ALL]) for k in keys] + [np.mean([R[n]['anchors']['jac_GU'] for n in ALL])]
write('tab_agreement', ' & \\multicolumn{2}{c}{Edges} & \\multicolumn{2}{c}{Vertices} & Anchors \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-6}\n'
      'Dataset & Spearman $\\rho$ & Jaccard top 5\\% & Spearman $\\rho$ & Jaccard top 5\\% & Jaccard \\\\\n', rows,
      ['Mean & ' + ' & '.join(f(x) for x in mean)], cols='@{}lccccc@{}')

# edges
M = ['gorc', 'uorc', 'length', 'jaccard', 'combo']
rows, A, P = [], [], []
for n in ALL:
    e = R[n]['edge']; a = [e[k]['auroc'] for k in M]; p = [e[k]['p5'] for k in M]; A.append(a); P.append(p)
    rows.append(f"{NAMES[n]} & " + ' & '.join(bold_row(a)) + ' & ' + ' & '.join(bold_row(p)) + f" & {f(R[n]['base_edge'])}")
A, P = np.array(A), np.array(P)
from scipy.stats import rankdata
rk = lambda Z: np.mean([rankdata(-np.round(z, 2)) for z in Z], 0)   # empates -> rank médio
foot = ['Mean & ' + ' & '.join(f(x) for x in A.mean(0)) + ' & ' + ' & '.join(f(x) for x in P.mean(0)) + ' & ' + f(np.mean([R[n]['base_edge'] for n in ALL])),
        'Mean rank & ' + ' & '.join(f(x, 1) for x in rk(A)) + ' & ' + ' & '.join(f(x, 1) for x in rk(P)) + ' & ']
write('tab_edges', ' & \\multicolumn{5}{c}{AUROC} & \\multicolumn{5}{c}{Precision of top 5\\%} & \\\\\n\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n'
      'Dataset & G-ORC & ORC & Len. & Jacc. & G+L & G-ORC & ORC & Len. & Jacc. & G+L & Base \\\\\n', rows, foot, cols='@{}lcccccccccc c@{}')

# vertices
M = ['gorc_softmax', 'uorc_softmax', 'jaccard', 'radius']
rows, A, P = [], [], []
for n in ALL:
    v = R[n]['vertex']; a = [v[k]['auroc'] for k in M]; p = [v[k]['p5'] for k in M]; A.append(a); P.append(p)
    rows.append(f"{NAMES[n]} & " + ' & '.join(bold_row(a)) + ' & ' + ' & '.join(bold_row(p)) + f" & {f(R[n]['base_vertex'])} & {v['gorc_softmax']['rho_rad']:+.2f}")
A, P = np.array(A), np.array(P)
foot = ['Mean & ' + ' & '.join(f(x) for x in A.mean(0)) + ' & ' + ' & '.join(f(x) for x in P.mean(0)) + ' & ' + f(np.mean([R[n]['base_vertex'] for n in ALL])) + ' & ',
        'Mean rank & ' + ' & '.join(f(x, 1) for x in rk(A)) + ' & ' + ' & '.join(f(x, 1) for x in rk(P)) + ' & & ']
write('tab_vertices', ' & \\multicolumn{4}{c}{AUROC} & \\multicolumn{4}{c}{Precision of top 5\\%} & & \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n'
      'Dataset & G-ORC & ORC & Jaccard & Radius & G-ORC & ORC & Jaccard & Radius & Base & $\\rho_{\\mathrm{rad}}$ \\\\\n', rows, foot, cols='@{}lcccccccccc@{}')

# anchors
rows, Pu, Co = [], [], []
for n in ALL:
    a = R[n]['anchors']; pu = [a['gorc']['purity'], a['uorc']['purity'], a['radius']['purity'], a['gorc']['glob_purity']]
    co = [a['gorc']['coverage'], a['uorc']['coverage'], a['radius']['coverage'], a['gorc']['glob_coverage']]
    Pu.append(pu); Co.append(co)
    rows.append(f"{NAMES[n]} & {a['gorc']['count']} & {a['uorc']['count']} & {a['radius']['count']} & " + ' & '.join(f(x) for x in pu) + f" & {f(R[n]['base_pure'])} & " + ' & '.join(f(x) for x in co))
Pu, Co = np.array(Pu), np.array(Co)
foot = ['Mean & & & & ' + ' & '.join(f(x) for x in Pu.mean(0)) + ' & ' + f(np.mean([R[n]['base_pure'] for n in ALL])) + ' & ' + ' & '.join(f(x) for x in Co.mean(0))]
write('tab_anchors', ' & \\multicolumn{3}{c}{Count} & \\multicolumn{4}{c}{Purity} & & \\multicolumn{4}{c}{Class coverage} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\\cmidrule(lr){10-13}\n'
      'Dataset & G & U & Mode & G & U & Mode & G$_{\\mathrm{glob}}$ & Base & G & U & Mode & G$_{\\mathrm{glob}}$ \\\\\n', rows, foot, cols='@{}lrrrccccccccc@{}')

# aggregation
H = ['mean', 'max', 'quantile', 'softmax', 'top_frac']
Z = np.array([[R[n]['vertex'][f'gorc_{h}']['auroc'] for h in H] for n in ALL])
Zp = np.array([[R[n]['vertex'][f'gorc_{h}']['p5'] for h in H] for n in ALL])
rows = [f"{NAMES[n]} & " + ' & '.join(bold_row(z)) + ' & ' + ' & '.join(bold_row(zp)) for n, z, zp in zip(ALL, Z, Zp)]
foot = ['Mean & ' + ' & '.join(f(x) for x in Z.mean(0)) + ' & ' + ' & '.join(f(x) for x in Zp.mean(0)),
        'Mean rank & ' + ' & '.join(f(x, 1) for x in rk(Z)) + ' & ' + ' & '.join(f(x, 1) for x in rk(Zp))]
hd = ' & '.join(['Mean', 'Max', 'Quantile', 'Soft max', 'Top frac.'])
write('tab_aggregation', ' & \\multicolumn{5}{c}{AUROC} & \\multicolumn{5}{c}{Precision of top 5\\%} \\\\\n\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n'
      'Dataset & ' + hd + ' & ' + hd + ' \\\\\n', rows, foot, cols='@{}lcccccccccc@{}')

# timing
T = json.load(open(f'{RAW}/timing.json'))
ns = [n for n in ALL + ['mnist_pca50', 'fashion_mnist_pca50'] if n in T]
rows = [f"{NAMES[n]} & {T[n]['E']} & {T[n]['u']:.2f} & {T[n]['l']:.2f} & {T[n]['b']:.2f} & {T[n]['u']/T[n]['l']:.1f}$\\times$ & {T[n]['u']/T[n]['b']:.1f}$\\times$" for n in ns]
write('tab_timing', ' & & \\multicolumn{3}{c}{Edge-curvature time (s)} & \\multicolumn{2}{c}{Speed-up} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\n'
      'Dataset & $|E|$ & ORC & G-ORC (loop) & G-ORC (batched) & Loop & Batched \\\\\n', rows, cols='@{}lrrrrrr@{}')
print('tabelas em', OUT)
