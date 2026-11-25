"""Caminhos do repositório (tudo relativo à raiz, independente de onde o script é chamado)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
RAW = os.path.join(ROOT, 'results', 'raw')
FIGS = os.path.join(ROOT, 'results', 'figures')
TABLES = os.path.join(ROOT, 'results', 'tables')
for d in (DATA, RAW, FIGS, TABLES):
    os.makedirs(d, exist_ok=True)
if ROOT not in sys.path:                      # permite "from gaussian_orc import ..."
    sys.path.insert(0, ROOT)
