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

### Opção 1 — Local (Poetry)

```bash
git clone https://github.com/lucascbergamo/recomendador_neural_ecommerce.git
cd recomendador_neural_ecommerce

# Instalar dependências
poetry install

# Configurar variáveis de ambiente
cp .env.example .env

# Validar ambiente (cria diretórios automaticamente)
make validate

# Pipeline completo: download → preprocess → features → train → evaluate
make data
make train-baselines
make train
make eval

# Visualizar experimentos
make mlflow   # acesse http://localhost:5000
```

### Opção 2 — Docker

```bash
# Subir MLflow + pipeline de dados
make docker-up
make docker-data

# Treinar modelo
make docker-train

# Visualizar em http://localhost:5000
```

### Opção 3 — DVC (pipeline reprodutível)

```bash
poetry install
dvc repro      # executa todos os stages automaticamente
dvc metrics show
```

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
│   └── validate_env.py      # Valida ambiente + cria diretórios
├── Dockerfile               # Multi-stage: builder + runtime
├── docker-compose.yml       # MLflow server + trainer
├── dvc.yaml                 # Pipeline: preprocess → feature_eng → train → evaluate
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

- **MovieLens 100K**: 100.836 interações, 943 usuários, 1.682 itens
- Download automático via `make data`
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
| 4 — Modelo Neural | NCF treinado, Model Registry, Model Card | 🔄 Em andamento |

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
