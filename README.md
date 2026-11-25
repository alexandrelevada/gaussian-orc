# Gaussian Ollivier–Ricci Curvature (G-ORC)

Código oficial do artigo **"Gaussian Ollivier–Ricci Curvature for Unsupervised Graph-Based Learning"**
(Alexandre L. M. Levada, Universidade Federal de São Carlos).

A **curvatura de Ollivier–Ricci gaussiana (G-ORC)** detecta, sem supervisão, as arestas de um grafo
*k*-NN que ligam regiões diferentes dos dados (*pontes*), os vértices nas regiões de transição
(*interface*) e vértices representativos de uma única região (*âncoras*). Cada vizinhança é resumida
pela gaussiana com a mesma média e covariância, e o transporte ótimo entre as gaussianas das duas
extremidades de uma aresta tem forma fechada (distância de Bures–Wasserstein). Por isso a curvatura
de cada aresta é **exata, sem otimização iterativa**, e muito mais rápida que a ORC clássica.

```math
\kappa_G(i,j) \;=\; 1-\frac{W_2\big(\mathcal{N}(\bar{\mathbf{x}}_i,\boldsymbol{\Sigma}_i),\,\mathcal{N}(\bar{\mathbf{x}}_j,\boldsymbol{\Sigma}_j)\big)}{\lVert\mathbf{x}_i-\mathbf{x}_j\rVert}
```

```math
W_2^2=\lVert\bar{\mathbf{x}}_i-\bar{\mathbf{x}}_j\rVert^2 +tr\boldsymbol{\Sigma}_i + tr\boldsymbol{\Sigma}_j-2 tr\big(\boldsymbol{\Sigma}_i^{1/2}\boldsymbol{\Sigma}_j\boldsymbol{\Sigma}_i^{1/2}\big)^{1/2}
```

- **κ_G muito negativo** → as vizinhanças das extremidades estão mais distantes (em transporte) do que
  as próprias extremidades: a aresta é uma **ponte** entre regiões.
- **κ_G positivo** → aresta no interior de uma região.
- Custo por aresta **O(min(k, m)³)**: o termo de Bures é calculado pela norma nuclear de uma matriz
  *k×k* (quando *k ≤ m*) ou pelos autovalores de uma matriz *m×m* (quando *k > m*), com escolha
  automática.

---

## Sumário

1. [Estrutura do repositório](#estrutura-do-repositório)
2. [Instalação](#instalação)
3. [Uso rápido como biblioteca](#uso-rápido-como-biblioteca)
4. [Visualização em um conjunto de dados](#visualização-em-um-conjunto-de-dados)
5. [Reproduzindo os resultados do artigo](#reproduzindo-os-resultados-do-artigo)
6. [Estendendo os experimentos com bases do OpenML](#estendendo-os-experimentos-com-bases-do-openml)
7. [Correspondência entre scripts, tabelas e figuras](#correspondência-entre-scripts-tabelas-e-figuras)
8. [Parâmetros](#parâmetros)
9. [Observações importantes](#observações-importantes)
10. [Citação](#citação)

---

## Estrutura do repositório

```
gaussian-orc/
├── gaussian_orc.py               # BIBLIOTECA: G-ORC, ORC uniforme, pontuação dos vértices, âncoras
├── gorc_endpoints_graph.py       # visualização interativa (linha de comando) em um conjunto
├── gorc_openml_experiments.py    # experimentos completos em bases do OpenML + tabelas LaTeX
├── reproduce/                    # scripts que reproduzem as figuras e tabelas do artigo
│   ├── paths.py                  #   caminhos (data/, results/) relativos à raiz
│   ├── download_data.py          #   baixa PMLB (UCI, MNIST) e Fashion-MNIST para data/
│   ├── datasets.py               #   leitura e pré-processamento das 12 bases do 1º grupo
│   ├── core.py                   #   adaptação das funções de gaussian_orc.py para os scripts
│   ├── run_main.py               #   métricas por base (arestas, vértices, âncoras, concordância)
│   ├── extra.py                  #   baselines adicionais (G+L, Jaccard/comprimento nos vértices)
│   ├── gelbrich_check.py         #   verificação numérica da Proposição 3 (κ_G ≥ κ₂ᵁ)
│   ├── timing.py                 #   tempos (mediana de 3 execuções) → Tabela 9
│   ├── kscan.py                  #   custo e AUROC em função de k → Figura 8(b,c)
│   ├── make_tables.py            #   Tabelas 3–9 em LaTeX
│   ├── make_figures.py           #   Figuras experimentais (concordância, precisão, tempo, imagens)
│   ├── make_method_figs.py       #   Figuras 1–3 da seção de método (dados sintéticos)
│   ├── run_all.sh                #   executa tudo na ordem certa (1º grupo de bases)
│   └── run_openml.sh             #   executa o 2º grupo de bases (OpenML)
├── results/
│   ├── paper_results/            # métricas (JSON) das 12 bases do 1º grupo, como no artigo
│   ├── tables/                   # tabelas LaTeX geradas
│   └── figures/                  # figuras geradas (PDF/PNG)
├── requirements.txt
├── LICENSE
└── README.md
```

As pastas `data/`, `results/raw/` e `results/openml/` são criadas na execução e não são versionadas.

---

## Instalação

Requer Python ≥ 3.9. Testado com Python 3.13, NumPy 2.5, SciPy 1.18, scikit-learn 1.9 e POT 0.9.7.

```bash
git clone https://github.com/alexandrelevada/gaussian-orc.git
cd gaussian-orc
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

O pacote [POT (Python Optimal Transport)](https://pythonot.github.io/) é necessário apenas para a ORC
uniforme (comparação). Se `pip install pot` falhar, use `conda install -c conda-forge pot`.

Teste rápido da instalação (confere as formas k×k/m×m, lote/laço e a Proposição 3):

```bash
python gaussian_orc.py
```

---

## Uso rápido como biblioteca

```python
import numpy as np
from sklearn.datasets import load_digits
from gaussian_orc import (knn_graph_edges, gaussian_orc, uniform_orc,
                          node_boundary_score, local_minima)

X, y = load_digits(return_X_y=True)
X = X / 16.0                                     # pixels em [0, 1]

E = knn_graph_edges(X, k_graph=10)                # grafo 10-NN simétrico e conexo
E, kappa = gaussian_orc(X, E, k_gauss=20, n_jobs=-1)        # κ_G de cada aresta

# 1) pontes: 5% das arestas com menor curvatura
bridges = E[np.argsort(kappa)[:int(0.05 * len(E))]]

# 2) pontuação de borda dos vértices (soft max normalizado de −κ das arestas incidentes)
b = node_boundary_score(len(X), E, kappa, how='softmax')
interface = np.argsort(-b)[:int(0.05 * len(X))]   # 5% dos vértices de maior pontuação

# 3) âncoras: mínimos locais da pontuação em 2 saltos
anchors = local_minima(len(X), E, b, hops=2)

# comparação com a ORC clássica (mesmo grafo e mesmas vizinhanças)
_, kappa_u = uniform_orc(X, E, k_meas=20, n_jobs=-1)
```

Nesse exemplo, 34% das pontes ligam dígitos diferentes (contra 4% de todas as arestas).

Principais funções de `gaussian_orc.py`:

| Função | Descrição |
|---|---|
| `knn_graph_edges(X, k_graph, connect='components')` | grafo *k*-NN simétrico; conecta componentes pelas arestas mais curtas (arestas da MST euclidiana) |
| `gaussian_orc(X, E, k_gauss=20, method='auto', batched=True, ddof=0, n_jobs=1)` | G-ORC de cada aresta; `method` = `'auto'`, `'kxk'` ou `'mxm'`; `batched=False` usa um laço por aresta |
| `uniform_orc(X, E, k_meas=20, p=1, n_jobs=1)` | ORC com medidas uniformes e *W_p* exato (POT); `p=2` dá κ₂ᵁ |
| `node_boundary_score(n, E, kappa, how='softmax')` | pontuação dos vértices: `'mean'`, `'max'`, `'quantile'`, `'softmax'`, `'top_frac'` |
| `local_minima(n, E, b, hops=2)` | âncoras (mínimos locais de `b` em `hops` saltos) |
| `graph_smooth(n, E, b, steps, alpha)` | suavização opcional da pontuação no grafo |

---

## Visualização em um conjunto de dados

`gorc_endpoints_graph.py` desenha o grafo *k*-NN (layout no espaço original) destacando as arestas
ou os vértices selecionados. Os rótulos são usados apenas para conferência (`--validate`).

```bash
# extremidades das 5% arestas mais negativas (Digits)
python gorc_endpoints_graph.py --dataset digits

# vermelho: 5% vértices de interface; azul: âncoras (mínimos locais, 2 saltos)
python gorc_endpoints_graph.py --dataset digits --both --validate

# mesma figura com a ORC uniforme (para comparar resultados e tempos)
python gorc_endpoints_graph.py --dataset digits --both --orc uniform

# conjuntos do OpenML
python gorc_endpoints_graph.py --openml mfeat-karhunen --scale --both
```

Opções úteis: `--top`, `--min_hits`, `--top_nodes`, `--bottom_nodes`, `--local_minima`, `--lm_hops`,
`--agg`, `--smooth`, `--layout spring|kawai`, `--k_graph`, `--k_gauss`. Veja `--help`.

---

## Reproduzindo os resultados do artigo

O artigo usa **22 bases** em dois grupos:

| Grupo | Bases | Origem |
|---|---|---|
| 1º (12 bases) | Iris, Wine, Breast cancer, Digits | scikit-learn |
| | Segmentation, Satimage, Texture, Optdigits, Pendigits, Letter, MNIST | PMLB (GitHub) |
| | Fashion-MNIST (conjunto de teste) | GitHub (Zalando Research) |
| 2º (10 bases) | COIL-20, Semeion, USPS, Mfeat-factors/-fourier/-karhunen/-pixel/-zernike, Indian Pines, HAR | OpenML |

Bases com mais de 10 000 amostras são subamostradas aleatoriamente (semente 0). Duplicatas exatas e
atributos constantes são removidos; dados de pixels são escalados para [0, 1] e os demais
padronizados (média 0, desvio 1). Parâmetros padrão: `k_G = 10`, `k = 20`, `q = 5%`, `h = 2`.

### 1º grupo

```bash
cd reproduce
bash run_all.sh quick     # teste rápido: Iris, Wine, Breast cancer e Digits (poucos minutos)
bash run_all.sh           # tudo: ~1–2 h em 2 CPUs (a ORC uniforme e o kscan dominam o tempo)
```

`run_all.sh` executa, nesta ordem:

```bash
python download_data.py                         # baixa as bases para ../data/
python make_method_figs.py                      # Figuras 1–3
python run_main.py                              # métricas das 12 bases
python run_main.py mnist_pca50 fashion_mnist_pca50
python extra.py                                 # baselines G+L e Jaccard/comprimento nos vértices
python gelbrich_check.py                        # Proposição 3
python timing.py                                # Tabela 9
python kscan.py                                 # Figura 8(b,c)
python make_tables.py                           # Tabelas 3–9 → ../results/tables/
python make_figures.py                          # Figuras 4–8 → ../results/figures/
```

Cada script também pode ser executado isoladamente; `run_main.py` aceita os nomes das bases
(`python run_main.py digits mnist`).

### 2º grupo (OpenML)

```bash
bash reproduce/run_openml.sh
```

Os resultados vão para `results/openml/`, e as Tabelas 3–9 com **todas as 22 bases** (médias e ranks
recalculados) são geradas em `results/openml/tables/`.

---

## Estendendo os experimentos com bases do OpenML

`gorc_openml_experiments.py` calcula, para qualquer base do OpenML, todas as métricas do artigo,
a figura do grafo e as linhas das tabelas. Os resultados das 12 bases do 1º grupo
(`results/paper_results/`) entram automaticamente nas tabelas, de modo que médias, ranks médios e
os testes de Wilcoxon passam a considerar todas as bases.

```bash
# bases tabulares (padronização z-score, padrão)
python gorc_openml_experiments.py --openml vehicle waveform-5000 spambase

# por data_id; dados de pixels → escala [0, 1]
python gorc_openml_experiments.py --openml_id 41082 --preprocess max --name USPS

# bases grandes: amostra de até N pontos (padrão 10000)
python gorc_openml_experiments.py --openml Fashion-MNIST --preprocess max --max_samples 10000

# verificar também a Proposição 3 (custa uma ORC extra)
python gorc_openml_experiments.py --openml vehicle --check_gelbrich

# apenas refazer as tabelas a partir dos JSON existentes
python gorc_openml_experiments.py --tables_only

# teste offline, sem internet
python gorc_openml_experiments.py --sklearn wine --repeats 1
```

Saídas (padrão `results/openml/`):

| Arquivo | Conteúdo |
|---|---|
| `tables/tab_*.tex` | Tabelas 3–9 completas, no formato do artigo |
| `tables/new_rows.tex` | apenas as linhas das bases novas + linhas *Mean*/*Mean rank* recalculadas |
| `figs/<base>_graph.png/.pdf` | pontes (G-ORC e ORC), vértices de interface e âncoras |
| `csv/<base>_vertices.csv` | pontuações e seleções por vértice |
| `results/<base>.json` | todas as métricas; `summary.csv` resume todas as bases |

---

## Correspondência entre scripts, tabelas e figuras

| Item do artigo | Conteúdo | Script |
|---|---|---|
| Figuras 1–3 | pipeline, intuição, agregação (dados sintéticos) | `reproduce/make_method_figs.py` |
| Tabela 3 | bases de dados | `run_main.py` → `make_tables.py` |
| Tabela 4, Figura 4 | concordância G-ORC × ORC uniforme | `run_main.py` → `make_tables.py`, `make_figures.py agreement` |
| Tabela 5, Figura 5 | detecção de arestas inter-classe | `run_main.py`, `extra.py` → `make_tables.py`, `make_figures.py precision` |
| Tabela 6 | vértices de borda | `run_main.py`, `extra.py` → `make_tables.py` |
| Tabela 7 | agregação aresta → vértice | `run_main.py` → `make_tables.py` |
| Tabela 8, Figuras 6 e 7 | âncoras e resultados qualitativos | `run_main.py` → `make_tables.py`, `make_figures.py qualitative image_grids` |
| Tabela 9, Figura 8 | tempo de execução e efeito de *k* | `timing.py`, `kscan.py` → `make_tables.py`, `make_figures.py timing` |
| Linhas do 2º grupo | 10 bases do OpenML | `gorc_openml_experiments.py` (`reproduce/run_openml.sh`) |

A numeração segue a versão atual do manuscrito e pode mudar na versão publicada.

---

## Parâmetros

| Parâmetro | Papel | Padrão |
|---|---|---|
| `k_graph` (*k_G*) | vizinhos que definem as arestas do grafo | 10 |
| `k_gauss` (*k*) | vizinhos que definem cada gaussiana (escala da curvatura) | 20 |
| `how` / `--agg` | agregação das arestas para os vértices | `softmax` |
| `tau` | temperatura do soft max | desvio-padrão de −κ |
| `q` | fração de vértices de interface | 0.05 |
| `hops` (*h*) | raio, em saltos, das âncoras | 2 |
| `ddof` | normalização da covariância: 0 → 1/k (artigo), 1 → 1/(k−1) | 0 |
| `batched` | avaliação em lotes de arestas (álgebra linear empilhada) ou laço por aresta | `True` |

Recomenda-se `k_gauss ≥ k_graph`. Valores de *k* entre 20 e 40 deram os melhores resultados.

---

## Observações importantes

- **Rótulos não são usados** em nenhuma etapa do método; servem apenas para avaliar.
- **Mesmo grafo e mesmas vizinhanças:** nos experimentos, G-ORC e ORC uniforme usam exatamente o
  mesmo grafo e os mesmos `idx` de vizinhança (`idx=` em `gaussian_orc`/`uniform_orc`), de modo
  que a única diferença é a medida (gaussiana com *W₂* × empírica com *W₁*).
- **Tempos dependem do hardware.** No artigo, o 1º grupo foi medido em uma máquina com 2 vCPUs e o
  2º grupo em outra máquina; os speed-ups são comparáveis apenas dentro de cada grupo. Em dimensão
  muito alta (por exemplo, COIL-20, *m* = 1024) o modo em lote pode ser mais lento que o laço; nesse
  caso use `batched=False` ou projete os dados (por exemplo, PCA-50) antes.
- **Threads:** os scripts paralelizam as arestas com threads (`n_jobs=-1`). Se a biblioteca de
  álgebra linear também usar muitas threads, pode haver competição; limite-as com
  `export OMP_NUM_THREADS=1` se observar lentidão.
- **Arestas muito curtas** (quase-duplicatas) podem dominar a cauda extrema do score; veja a seção
  de limitações do artigo.
- **Reprodutibilidade:** amostragens usam semente fixa (0). Pequenas diferenças numéricas
  (~10⁻⁶) entre máquinas são esperadas e não alteram as tabelas.

---

## Licença

Distribuído sob a licença MIT (veja `LICENSE`).

## Contato

Alexandre L. M. Levada — Departamento de Computação, Universidade Federal de São Carlos (UFSCar) —
alexandre.levada at ufscar br
