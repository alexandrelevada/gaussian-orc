"""Carregamento e pré-processamento das bases (rótulos usados apenas na avaliação)."""
import os
from paths import DATA, RAW, FIGS, TABLES
import gzip, numpy as np, pandas as pd
from sklearn import datasets as skd
from sklearn.preprocessing import StandardScaler

D = DATA + '/'
NMAX = 10000

def _idx(path):
    with gzip.open(path, 'rb') as f:
        b = f.read()
    nd = b[3]; dims = [int.from_bytes(b[4+4*i:8+4*i], 'big') for i in range(nd)]
    return np.frombuffer(b, np.uint8, offset=4+4*nd).reshape(dims)

def _pmlb(name):
    df = pd.read_csv(D + f'{name}.tsv.gz', sep='\t')
    return df.drop(columns='target').values.astype(float), df['target'].values

def load(name, seed=0, raw=False):
    if name.endswith('_pca50'):
        from sklearn.decomposition import PCA
        X, y, img, X0 = load(name[:-6], seed, True)
        Z = PCA(50, random_state=0).fit_transform(X)
        return (Z, y, img, X0) if raw else (Z, y, img)
    img = False
    if name == 'iris':   X, y = skd.load_iris(return_X_y=True)
    elif name == 'wine': X, y = skd.load_wine(return_X_y=True)
    elif name == 'breast_cancer': X, y = skd.load_breast_cancer(return_X_y=True)
    elif name == 'digits': X, y = skd.load_digits(return_X_y=True); img = True
    elif name == 'fashion_mnist':
        X = _idx(D + 'fmnist_t10k-images-idx3-ubyte.gz').reshape(-1, 784).astype(float)
        y = _idx(D + 'fmnist_t10k-labels-idx1-ubyte.gz').astype(int); img = True
    else:
        X, y = _pmlb(name); img = name in ('mnist', 'optdigits')
    X = np.asarray(X, float); y = np.asarray(y)
    # remove duplicatas exatas e atributos constantes
    _, u = np.unique(X, axis=0, return_index=True); u = np.sort(u)
    X, y = X[u], y[u]
    X0 = X
    X = X[:, X.std(0) > 0]
    if len(X) > NMAX:                         # amostragem estratificada simples
        rng = np.random.default_rng(seed)
        s = np.sort(rng.choice(len(X), NMAX, replace=False)); X, y, X0 = X[s], y[s], X0[s]
    X = X / X.max() if img else StandardScaler().fit_transform(X)
    _, y = np.unique(y, return_inverse=True)
    return (X, y, img, X0) if raw else (X, y, img)

ALL = ['iris', 'wine', 'breast_cancer', 'digits', 'segmentation', 'satimage', 'texture',
       'optdigits', 'pendigits', 'letter', 'mnist', 'fashion_mnist']

if __name__ == '__main__':
    for n in ALL:
        X, y, img = load(n); print(f'{n:14s} n={X.shape[0]:6d} m={X.shape[1]:4d} C={y.max()+1:3d} img={img}')
