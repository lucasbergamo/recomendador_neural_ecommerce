.PHONY: install lint format test validate data train-baselines train eval register serve pipeline mlflow docker-up docker-down docker-lint docker-test docker-eval docker-register docker-serve clean

export CURRENT_UID := $(shell id -u)
export CURRENT_GID := $(shell id -g)

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

# ── Model Registry ────────────────────────────────────────────────
register:
	poetry run python -m scripts.register_model

# ── Serving ───────────────────────────────────────────────────────
serve:
	poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Pipeline completo via DVC
pipeline:
	poetry run dvc repro

# ── MLflow ────────────────────────────────────────────────────────
mlflow:
	poetry run mlflow ui --port 5000

# ── Docker ────────────────────────────────────────────────────────
check-resources:
	bash scripts/check_docker_resources.sh

docker-build: check-resources
	docker compose build

docker-lint:
	docker compose --profile ci run --rm lint ruff check src/ tests/
	docker compose --profile ci run --rm lint ruff format --check src/ tests/

docker-test:
	docker compose --profile ci run --rm ci pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

docker-up:
	docker compose up -d mlflow

docker-data:
	docker compose run --rm data-pipeline

docker-train-baselines:
	docker compose --profile training run --rm train-baselines

docker-train:
	docker compose --profile training run --rm train-model

docker-eval:
	docker compose --profile training run --rm evaluate

docker-register:
	docker compose --profile training run --rm register

docker-serve:
	docker compose up -d serve

docker-down:
	docker compose down

docker-down-clean:
	docker compose down -v

# ── Limpeza ───────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
