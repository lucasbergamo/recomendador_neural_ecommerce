from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    seed: int = 42
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "recommendation-system"

    embedding_dim: int = 64
    mlp_layers: list[int] = [128, 64, 32]
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 10

    top_k: int = 10
    dataset_name: str = "ml-100k"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_BRONZE_DIR = PROJECT_ROOT / "data" / "bronze"
DATA_SILVER_DIR = PROJECT_ROOT / "data" / "silver"
DATA_GOLD_DIR = PROJECT_ROOT / "data" / "gold"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"
