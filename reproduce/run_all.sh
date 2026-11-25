#!/usr/bin/env bash
# Reproduz todos os resultados do primeiro grupo de bases (Seções 4 e 5 do artigo).
# Uso:  cd reproduce && bash run_all.sh            (tudo, ~1-2 h em 2 CPUs)
#       bash run_all.sh quick                      (só Iris, Wine, Breast cancer e Digits)
set -e
cd "$(dirname "$0")"
python download_data.py
python make_method_figs.py                                   # Figuras 1-3 (método, dados sintéticos)
if [ "$1" == "quick" ]; then
    python run_main.py iris wine breast_cancer digits
    echo "Modo rápido concluído: resultados em ../results/raw/"; exit 0
fi
python run_main.py                                           # métricas das 12 bases (+ curvaturas salvas)
python run_main.py mnist_pca50 fashion_mnist_pca50           # variantes PCA-50 (Seção 5.6)
python extra.py                                              # baselines extras (G+L, Jaccard e comprimento nos vértices)
python gelbrich_check.py                                     # verificação da Proposição 3 (κ_G ≥ κ₂ᵁ)
python timing.py                                             # Tabela 9 (mediana de 3 execuções)
python kscan.py                                              # Figura 8(b,c): custo e AUROC em função de k
python make_tables.py                                        # Tabelas 3-9 em LaTeX (../results/tables)
python make_figures.py                                       # Figuras 4-8 (../results/figures)
echo "Concluído. Tabelas em ../results/tables, figuras em ../results/figures, dados brutos em ../results/raw"
