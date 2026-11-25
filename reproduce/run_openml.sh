#!/usr/bin/env bash
# Segundo grupo de bases (OpenML) do artigo. Requer acesso à internet (openml.org).
# Os resultados vão para ../results/openml/ e as Tabelas 3-9 com os 22 conjuntos são refeitas
# automaticamente em ../results/openml/tables/ (linhas novas em new_rows.tex).
set -e
cd "$(dirname "$0")/.."
python gorc_openml_experiments.py --openml coil-20 semeion mfeat-factors mfeat-fourier \
       mfeat-karhunen mfeat-pixel mfeat-zernike Indian_pines har
python gorc_openml_experiments.py --openml_id 41082 --name USPS
