# Stage 1 — builder: instala dependências em ambiente isolado
FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir poetry==2.4.1

COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction --no-ansi

# Stage 2 — runtime: imagem final enxuta sem ferramentas de build
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copia apenas os pacotes instalados do builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Cria diretórios de dados em tempo de build — resolve o problema de dirs faltantes
RUN mkdir -p data/bronze data/silver data/gold models metrics

COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Valida ambiente antes de qualquer execução
RUN python scripts/validate_env.py || true

CMD ["python", "-m", "src.training.trainer"]

# Stage 3 — lint: só o ruff, NÃO parte do builder — não precisa de PyTorch/MLflow/etc.
# pra checar sintaxe e estilo. Isolado de propósito: mais rápido, e não depende do
# builder ter sucesso (menos ponto de falha, não mais).
FROM python:3.11-slim AS lint

WORKDIR /app

RUN pip install --no-cache-dir "ruff>=0.7"

COPY pyproject.toml ./
COPY src/ ./src/
COPY tests/ ./tests/

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Stage 4 — ci: runtime + pytest, para rodar os testes em container (precisa das
# dependências reais, já que os testes importam e executam o código de verdade).
FROM python:3.11-slim AS ci

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

RUN pip install --no-cache-dir "pytest>=8.3" "pytest-cov>=5.0"

RUN mkdir -p data/bronze data/silver data/gold models metrics

COPY pyproject.toml ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY configs/ ./configs/

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Stage 5 — serve: runtime + modelo treinado embutido, expõe a API HTTP.
# Diferente dos outros serviços (que recebem data/models via volume), este stage
# empacota o modelo DENTRO da imagem — é o que roda sozinho na AWS, sem disco local.
FROM runtime AS serve

COPY models/ncf.pt ./models/ncf.pt
COPY data/gold/metadata.parquet ./data/gold/metadata.parquet
COPY data/silver/items.parquet ./data/silver/items.parquet
COPY data/silver/item_id_map.parquet ./data/silver/item_id_map.parquet

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
