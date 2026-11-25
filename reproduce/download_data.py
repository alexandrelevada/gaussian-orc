"""Baixa as bases do primeiro grupo que não vêm com o scikit-learn (PMLB e Fashion-MNIST) para ../data/.
Iris, Wine, Breast cancer e Digits são carregados diretamente do scikit-learn."""
import os, urllib.request
from paths import DATA

PMLB = 'https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/{0}/{0}.tsv.gz'
FMNIST = 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/{}'
FILES = {f'{d}.tsv.gz': PMLB.format(d) for d in
         ('segmentation', 'satimage', 'texture', 'optdigits', 'pendigits', 'letter', 'mnist')}
FILES.update({f'fmnist_{f}': FMNIST.format(f) for f in ('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')})

if __name__ == '__main__':
    for name, url in FILES.items():
        path = os.path.join(DATA, name)
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            print(f'ok (já existe)  {name}'); continue
        print(f'baixando        {name}  <-  {url}')
        urllib.request.urlretrieve(url, path)
    print('dados em', DATA)
