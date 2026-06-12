"""Treina e avalia os baselines (SVD, Popularity, Random) com MLflow tracking."""

import json

import mlflow
import pandas as pd

from src.data.features import load_gold
from src.evaluation.metrics import evaluate_baseline
from src.models.baselines import PopularityRecommender, SVDRecommender, RandomRecommender
from src.utils.config import METRICS_DIR, settings
from src.utils.logger import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger(__name__)


def _run_baseline(
    name: str,
    model: object,
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_items: int,
) -> dict:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=name):
        model.fit(train)
        metrics = evaluate_baseline(model, test, n_items, settings.top_k)
        mlflow.log_params({"model": name, "top_k": settings.top_k})
        mlflow.log_metrics(metrics)
        logger.info(f"{name}_evaluated", **metrics)
    return metrics


def run() -> None:
    set_global_seed()
    train, _, test, meta = load_gold()
    n_items = int(meta["n_items"])

    baselines = {
        "svd": SVDRecommender(top_k=settings.top_k),
        "popularity": PopularityRecommender(top_k=settings.top_k),
        "random": RandomRecommender(n_items=n_items, top_k=settings.top_k),
    }

    all_metrics: dict[str, dict] = {}
    for name, model in baselines.items():
        all_metrics[name] = _run_baseline(name, model, train, test, n_items)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "baseline_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("baselines_done", models=list(all_metrics))


if __name__ == "__main__":
    run()
