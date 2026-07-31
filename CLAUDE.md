# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
make install        # poetry install
cp .env.example .env
make validate        # sanity check: Python, pacotes-chave, diretórios, .env

# Qualidade de código
make lint          # ruff check + format check
make format        # ruff autofix

# Testes
make test                            # todos os testes + cobertura (--cov=src)
pytest tests/test_models.py -v      # arquivo específico
pytest tests/test_models.py::test_ncf_forward_shape  # teste único

# Pipeline de dados (requer internet — faz download do MovieLens 100K)
make data          # cria diretórios + download + bronze→silver→gold

# Treinamento
make train-baselines   # SVD, Popularity, Random via MLflow
make train             # NeuralCF (NCF) via MLflow
make eval              # avalia modelo salvo em models/ncf.pt

# Model Registry
make register       # registra models/ncf.pt no MLflow Model Registry,
                     # promove via aliases @staging → @production

# Serving
make serve          # sobe a API (uvicorn) em localhost:8000

# Pipeline completo reprodutível (5 stages: preprocess, feature_eng,
# train_baselines, train, evaluate)
make pipeline       # poetry run dvc repro

# Docker
make docker-up               # sobe MLflow server em localhost:5000 (imagem oficial)
make docker-data              # roda pipeline de dados em container (serviço data-pipeline)
make docker-train-baselines   # baselines em container (requer --profile training)
make docker-train             # NCF em container (requer --profile training)
make docker-eval              # avalia models/ncf.pt em container (requer --profile training)
make docker-register          # registra no MLflow Model Registry em container (requer --profile training, MLflow de pé)
make docker-lint              # ruff em container (stage ci do Dockerfile)
make docker-test              # pytest + cobertura em container (stage ci do Dockerfile)
make docker-serve             # sobe a API em container (localhost:8000, modelo embutido na imagem)
```

## Arquitetura

### Fluxo de dados

```
data/bronze/   →   data/silver/   →   data/gold/
(CSV raw)          (Parquet limpo)     (Parquet pronto para treino)
load.py            preprocess.py       features.py
```

`data/bronze/` contém os CSVs brutos do MovieLens 100K (baixados automaticamente). `data/silver/` é a camada limpa (sem cold-start, IDs re-indexados a partir de 0 — obrigatório para índices de embedding PyTorch). `data/gold/` contém `train/val/test.parquet` com colunas `user_id`, `item_id`, `interaction` (binário), e `metadata.parquet` com `n_users`/`n_items`.

O split é **temporal por usuário** (leave-time-out): as últimas 10% interações de cada usuário vão para test, as 10% anteriores para val, o resto para train. Os negativos (interaction=0) são amostrados com ratio 1:4 e ficam apenas no treino.

### Modelo principal — NeuralCF (`src/models/ncf.py`)

Implementa NeuMF (He et al., 2017): dois caminhos paralelos que se fundem na camada final.
- **GMF path**: `gmf_user_emb ⊙ gmf_item_emb` (produto element-wise — captura interações lineares)
- **MLP path**: `concat(mlp_user_emb, mlp_item_emb) → Linear(128)→ReLU→Linear(64)→ReLU→Linear(32)`
- **Saída**: `concat(GMF_out, MLP_out) → Linear(1)` → logit (sigmoid só na inferência)

O modelo herda de `BaseRecommender` (`src/models/base.py`), que implementa o **Template Method**: `forward()` é abstrato; `predict()` e `recommend_top_k()` estão prontos na base.

### Design Patterns

| Pattern | Arquivo | Uso |
|---|---|---|
| Template Method | `src/models/base.py` | `BaseRecommender` — toda subclasse só implementa `forward()` |
| Factory | `src/models/factory.py` | `RecommenderFactory.create(ModelType.NCF, n_users=..., n_items=...)` |
| Strategy | `src/training/strategies.py` | `get_optimizer_strategy("adam"|"adamw"|"sgd")` |

### Configuração

Todas as settings são lidas via **Pydantic Settings** (`src/utils/config.py`). Os valores padrão funcionam sem `.env`. Para sobrescrever, edite o `.env` — as variáveis mapeiam diretamente para os campos de `Settings` (ex: `EMBEDDING_DIM=128`).

Os caminhos de diretório (`DATA_BRONZE_DIR`, `MODELS_DIR`, etc.) são constantes derivadas de `PROJECT_ROOT` e devem ser importadas de `src.utils.config`, não hardcodadas.

### MLflow + Model Registry

Todos os runs (NCF + baselines) vão para o mesmo experimento `recommendation-system`. O MLflow precisa estar rodando (`make mlflow` ou `make docker-up`) antes de treinar. O tracking URI padrão é `http://localhost:5000`. Na UI (MLflow 3.x), os runs ficam em **Runs**/**Evaluation runs** — a aba **Traces** é observabilidade de LLM/GenAI, não se aplica aqui.

O modelo final é registrado no Model Registry via `scripts/register_model.py` (`make register`) — nome `ncf-recommender`, promovido por **aliases** (`@staging`, `@production`), não pelos stages antigos (Staging/Production/Archived), descontinuados desde o MLflow 2.9. Script separado do `trainer.py` de propósito: registrar/promover é uma decisão sobre um modelo já escolhido, não deveria rodar a cada `make train`.

### DVC pipeline

`dvc.yaml` define 5 stages com dependências explícitas entre arquivos: `preprocess` → `feature_eng` → `{train_baselines, train}` → `evaluate` (`train_baselines` e `train` rodam em paralelo, ambos dependem só de `feature_eng`). Ao modificar qualquer `dep`, `dvc repro` re-executa apenas os stages afetados — inclui `src/utils/config.py` como dep do stage `train`, já que hiperparâmetros de lá (`embedding_dim`, `learning_rate` etc.) afetam o modelo final. Os arquivos `data/silver/`, `data/gold/` e `models/ncf.pt` são gerenciados pelo DVC (não pelo git). O remote é local (`~/dvc-storage`) — simula um bucket S3 para fins de reprodutibilidade, mas não está disponível pra quem clona o repo (`dvc pull` não funciona); `dvc repro` reconstrói tudo a partir dos CSVs de `data/bronze/`, já commitados.

## Convenções

- **Commits**: padrão conventional commits (`feat:`, `fix:`, `docs:`, `style:`, `refactor:`), com escopo opcional em parênteses — ex: `feat(models): add attention layer`. Nunca incluir co-autoria de ferramentas na mensagem.
- **Linting**: ruff com `line-length=100`. `N803`/`N806` ignorados (convenção ML para matrizes X maiúsculo).
- **Seeds**: sempre chamar `set_global_seed()` no início de qualquer script de treino ou geração de dados.
- **Logits vs probabilidades**: modelos retornam logits em `forward()`; sigmoid só em `predict()` ou na inferência — nunca dentro do `forward()`.
