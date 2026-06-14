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

# Stage 3 — ci: runtime + ferramentas de dev (ruff, pytest) para rodar lint/testes em container
FROM python:3.11-slim AS ci

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

RUN pip install --no-cache-dir "ruff>=0.7" "pytest>=8.3" "pytest-cov>=5.0"

RUN mkdir -p data/bronze data/silver data/gold models metrics

COPY pyproject.toml ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY configs/ ./configs/

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
