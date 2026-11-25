import os
from paths import DATA, RAW, FIGS, TABLES
import numpy as np, json
from datasets import load, ALL
from core import uorc, neighborhoods
out = {}
for n in ALL:
    X, y, _ = load(n); D = np.load(f'{RAW}/{n}.npz'); E = D['E']; idx = neighborhoods(X, 20)
    k2 = uorc(X, E, idx, p=2)
    out[n] = dict(max_viol=float(np.max(k2 - D['kG'])), frac_viol=float(np.mean(k2 > D['kG'] + 1e-9)),
                  max_viol_w1=float(np.max(k2 - D['kU'])))
    print(n, out[n], flush=True)
json.dump(out, open(f'{RAW}/gelbrich.json', 'w'), indent=1)
