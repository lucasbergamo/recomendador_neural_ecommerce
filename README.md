# Recomendador Neural E-commerce

Sistema de recomendação de produtos baseado no comportamento de navegação dos usuários, implementado com **Neural Collaborative Filtering (NCF/NeuMF)** em PyTorch. Pipeline completo containerizado com Docker, dados versionados com DVC e experimentos rastreados no MLflow.

> **Tech Challenge Fase 02 — FIAP Pós-Tech MLET**

---

## Arquitetura

```
Usuário → Embedding ─┐
                      ├─ GMF (element-wise ×) ──┐
Item    → Embedding ─┘                           ├─ Linear → Score
                                                 │
Usuário → Embedding ─┐                           │
                      ├─ Concat → MLP layers ────┘
Item    → Embedding ─┘
```

O modelo combina **Generalized Matrix Factorization (GMF)** — que captura interações lineares — com um **MLP** que captura não-linearidades, resultando na arquitetura **NeuMF** (He et al., 2017).

### Pipeline de ponta a ponta

```
data/bronze (CSV) → preprocess → data/silver → feature_eng → data/gold
                                                                  │
                              ┌───────────────────────────────────┤
                              ▼                                   ▼
                    train_baselines (SVD/Popularity/Random)  train (NCF)
                              │                                   │
                              └──────────────┬────────────────────┘
                                              ▼
                                     evaluate (test set)
                                              │
                                              ▼
                    MLflow (tracking, runs) → Model Registry (@staging/@production)
```

5 stages no `dvc.yaml` (`preprocess`, `feature_eng`, `train_baselines`, `train`, `evaluate`) — `dvc dag` mostra o grafo completo, `dvc repro` executa tudo automaticamente respeitando as dependências entre eles.

---

## Resultados

| Modelo | NDCG@10 | Precision@10 | Recall@10 | HR@10 | MAP@10 |
|---|---|---|---|---|---|
| NCF (NeuMF) | 0.0345 | 0.0231 | 0.0445 | 0.1898 | 0.0148 |
| Popularity | **0.0350** | 0.0244 | 0.0396 | 0.1983 | 0.0153 |
| SVD | 0.0079 | 0.0051 | 0.0132 | 0.0509 | 0.0031 |
| Random | 0.0061 | 0.0052 | 0.0064 | 0.0509 | 0.0020 |

O NCF fica marginalmente abaixo do Popularity (dataset pequeno e denso favorece recomendação não-personalizada) — mesmo depois de testar 3 configs de tuning. Histórico completo do tuning, evidência dos runs no MLflow e leitura honesta desse trade-off no [Model Card](docs/model_card.md).

---

## Stack

| Componente | Tecnologia |
|---|---|
| Modelo | PyTorch 2.x — NeuMF (GMF + MLP) |
| Baselines | Scikit-Learn — SVD, Popularity, Random |
| Experimentos | MLflow — tracking + Model Registry |
| Versionamento de dados | DVC — pipeline reprodutível |
| Dependências | Poetry — lock file commitado |
| Containerização | Docker multi-stage + docker-compose |
| Qualidade de código | Ruff + pre-commit hooks |
| Configuração | Pydantic Settings + .env |
| Testes | Pytest |

---

## Início Rápido

**Pré-requisitos:** Python 3.11+, [Poetry](https://python-poetry.org/), Docker + Docker Compose, git.

Caminho recomendado — do zero até o modelo registrado:

```bash
git clone https://github.com/lucasbergamo/recomendador_neural_ecommerce.git
cd recomendador_neural_ecommerce

poetry install
cp .env.example .env

make docker-up      # sobe o MLflow em container (localhost:5000)
dvc repro           # pipeline completo: preprocess → feature_eng →
                     # {train_baselines, train} → evaluate

make test           # 26 testes, roda local (rápido, não precisa de container)
make register       # registra o modelo final no MLflow Model Registry
```

Depois disso, em `http://localhost:5000`:
- aba **Runs** (ou **Evaluation runs**, dependendo da versão da UI) do experimento `recommendation-system` — todos os runs de treino/baseline
- **Models → ncf-recommender** — o modelo registrado, aliases `@staging` e `@production`

> **Nota sobre o remote do DVC:** é local (`~/dvc-storage`, simula um bucket S3) — quem clona o repo não tem acesso a essa pasta, então `dvc pull` **não vai funcionar**. Isso é intencional: `dvc repro` reconstrói tudo do zero a partir dos CSVs brutos do MovieLens já commitados em `data/bronze/`, sem depender de nenhum remote externo.

<details>
<summary>Alternativas — sem Docker, ou passo a passo manual</summary>

**Sem Docker** (MLflow local via Poetry, precisa estar rodando antes dos comandos de treino):

```bash
poetry install && cp .env.example .env
make mlflow &        # ou outro terminal — acesse http://localhost:5000

make data
make train-baselines
make train
make eval
make register
```

**Docker, passo a passo** (equivalente ao `dvc repro`, mas manual):

```bash
make docker-up
make docker-data
make docker-train-baselines
make docker-train
```

</details>

---

## Estrutura do Projeto

```
recomendador_neural_ecommerce/
├── src/
│   ├── data/
│   │   ├── load.py          # Download MovieLens 100K
│   │   ├── preprocess.py    # Bronze → Silver
│   │   ├── features.py      # Silver → Gold (interaction matrix, splits)
│   │   └── pipeline.py      # Orquestrador do pipeline de dados
│   ├── models/
│   │   ├── base.py          # Template Method: BaseRecommender
│   │   ├── ncf.py           # NeuMF — GMF + MLP
│   │   ├── baselines.py     # SVD, Popularity, Random
│   │   └── factory.py       # Factory Pattern: cria modelos por nome
│   ├── training/
│   │   ├── trainer.py       # Loop de treino + early stopping + MLflow
│   │   ├── train_baselines.py
│   │   └── strategies.py    # Strategy Pattern: Adam, AdamW, SGD
│   ├── evaluation/
│   │   └── metrics.py       # NDCG@K, Precision@K, Recall@K, HR@K, MAP@K
│   └── utils/
│       ├── config.py        # Pydantic Settings + paths
│       ├── logger.py        # Structlog
│       └── reproducibility.py
├── tests/                   # Pytest: smoke, unit, integration
├── data/
│   ├── bronze/              # Dados brutos (DVC)
│   ├── silver/              # Dados limpos (DVC)
│   └── gold/                # Features prontas (DVC)
├── docs/
│   ├── model_card.md        # Documentação do modelo
│   ├── dataset.md           # Documentação do dataset
│   └── monitoring_plan.md   # Plano de monitoramento
├── scripts/
│   ├── validate_env.py      # Valida ambiente + cria diretórios
│   └── register_model.py    # Registra o modelo final no MLflow Model Registry
├── Dockerfile               # Multi-stage: builder + runtime + ci (lint/test em container)
├── docker-compose.yml       # MLflow server + data-pipeline + train-baselines + train-model + ci
├── dvc.yaml                 # 5 stages: preprocess → feature_eng → {train_baselines, train} → evaluate
├── pyproject.toml           # Poetry — deps prod/dev separadas
└── Makefile                 # Comandos de desenvolvimento
```

---

## Design Patterns

| Pattern | Onde | Por quê |
|---|---|---|
| **Template Method** | `src/models/base.py` | `BaseRecommender` define `predict()` e `recommend_top_k()` — subclasses só implementam `forward()` |
| **Factory** | `src/models/factory.py` | `RecommenderFactory.create(ModelType.NCF, ...)` — desacopla criação de uso |
| **Strategy** | `src/training/strategies.py` | Troca Adam/AdamW/SGD sem mudar o trainer |

---

## Métricas de Avaliação

Todas as métricas são calculadas no test set com **leave-time-out split** (últimas interações por usuário):

| Métrica | Descrição |
|---|---|
| **NDCG@K** | Ranking quality — penaliza hits em posições piores |
| **Precision@K** | Fração dos K recomendados que são relevantes |
| **Recall@K** | Fração dos relevantes encontrados nos K recomendados |
| **HR@K** (Hit Rate) | 1 se pelo menos 1 item relevante está nos K recomendados |
| **MAP@K** | Mean Average Precision — combina precision e ordem |

---

## Dataset

Documentação completa em [`docs/dataset.md`](docs/dataset.md).

- **MovieLens 100K**: 100.000 ratings brutos (99.287 após filtro de cold-start), 943 usuários, 1.682 itens brutos (1.349 após filtro)
- Download automático via `make data` (ou já commitado em `data/bronze/` — funciona sem internet)
- Alternativas: RetailRocket, Instacart, Amazon Reviews (ver docs/dataset.md)

---

## Testes

```bash
make test        # todos os testes
make test-cov    # com relatório de cobertura
```

Os testes cobrem:
- Smoke tests (imports e inicializações)
- Modelos (forward pass, shapes, ranges)
- Métricas (NDCG, Precision, Recall, HR, MAP)
- Pipeline de dados (preprocessing, feature engineering)

---

## Etapas de Desenvolvimento

| Etapa | Descrição | Status |
|---|---|---|
| 1 — Clean Code | Estrutura, design patterns, linting | ✅ Concluída |
| 2 — Dependências | Poetry, Pydantic Settings, validate_env | ✅ Concluída |
| 3 — Containerização | Docker multi-stage, DVC pipeline, MLflow | ✅ Concluída |
| 4 — Modelo Neural | NCF treinado, tuning (3 configs), Model Registry, Model Card | ✅ Concluída |

---

## Contribuindo

```bash
# Instalar hooks de pre-commit
pre-commit install

# Verificar linting
make lint

# Formatar código
make format
```
