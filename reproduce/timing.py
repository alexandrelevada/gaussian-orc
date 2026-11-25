import os
from paths import DATA, RAW, FIGS, TABLES
import json, time, sys, numpy as np
from datasets import load, ALL
from core import *
R = {}
for name in ALL + ['mnist_pca50', 'fashion_mnist_pca50']:
    X, y, _ = load(name); E, _ = knn_connected(X, 10); idx = neighborhoods(X, 20)
    t = {'b': [], 'l': [], 'u': []}
    for r in range(3):
        for key, f in (('b', lambda: gorc(X, E, idx, batched=True)), ('l', lambda: gorc(X, E, idx, batched=False)),
                       ('u', lambda: uorc(X, E, idx))):
            t0 = time.perf_counter(); f(); t[key].append(time.perf_counter() - t0)
    R[name] = {k: float(np.median(v)) for k, v in t.items()}; R[name]['E'] = len(E)
    print(name, R[name], 'speedup', R[name]['u'] / R[name]['b'], R[name]['u'] / R[name]['l'], flush=True)
    json.dump(R, open(f'{RAW}/timing.json', 'w'), indent=1)
