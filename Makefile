.PHONY: install lint format test validate data train-baselines train eval docker-up docker-down docker-lint docker-test clean

# ── Setup ──────────────────────────────────────────────────────────
install:
	poetry install

validate:
	poetry run python scripts/validate_env.py

# ── Qualidade de código ────────────────────────────────────────────
lint:
	poetry run ruff check src/ tests/
	poetry run ruff format --check src/ tests/

format:
	poetry run ruff check --fix src/ tests/
	poetry run ruff format src/ tests/

# ── Testes ────────────────────────────────────────────────────────
test:
	poetry run pytest tests/ -v --tb=short

test-cov:
	poetry run pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# ── Pipeline de dados ─────────────────────────────────────────────
data:
	mkdir -p data/bronze data/silver data/gold models metrics
	poetry run python -m src.data.pipeline

# ── Treinamento ───────────────────────────────────────────────────
train-baselines:
	poetry run python -m src.training.train_baselines

train:
	poetry run python -m src.training.trainer

eval:
	poetry run python -m src.evaluation.metrics

# Pipeline completo via DVC
pipeline:
	dvc repro

# ── MLflow ────────────────────────────────────────────────────────
mlflow:
	poetry run mlflow ui --port 5000

# ── Docker ────────────────────────────────────────────────────────
docker-build:
	docker compose build

docker-lint:
	docker compose --profile ci run --rm ci ruff check src/ tests/
	docker compose --profile ci run --rm ci ruff format --check src/ tests/

docker-test:
	docker compose --profile ci run --rm ci pytest tests/ -v --tb=short

docker-up:
	docker compose up -d mlflow

docker-data:
	docker compose run --rm trainer

docker-train:
	docker compose run --rm --profile training train-model

docker-down:
	docker compose down -v

# ── Limpeza ───────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
