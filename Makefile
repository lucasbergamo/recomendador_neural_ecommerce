.PHONY: install lint format test validate data train-baselines train eval docker-up docker-down clean

# ── Setup ──────────────────────────────────────────────────────────
install:
	poetry install

validate:
	python scripts/validate_env.py

# ── Qualidade de código ────────────────────────────────────────────
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

# ── Testes ────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# ── Pipeline de dados ─────────────────────────────────────────────
data:
	mkdir -p data/bronze data/silver data/gold models metrics
	python -m src.data.pipeline

# ── Treinamento ───────────────────────────────────────────────────
train-baselines:
	python -m src.training.train_baselines

train:
	python -m src.training.trainer

eval:
	python -m src.evaluation.metrics

# Pipeline completo via DVC
pipeline:
	dvc repro

# ── MLflow ────────────────────────────────────────────────────────
mlflow:
	mlflow ui --port 5000

# ── Docker ────────────────────────────────────────────────────────
docker-build:
	docker compose build

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
