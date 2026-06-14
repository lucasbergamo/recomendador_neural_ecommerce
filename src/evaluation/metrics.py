"""Métricas de avaliação para sistemas de recomendação: NDCG, Precision, Recall, HR, MAP."""

import json
from typing import Protocol

import numpy as np
import pandas as pd
import torch

from src.utils.config import METRICS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Recommender(Protocol):
    """Protocol para qualquer modelo que suporte predict por user_id."""

    def predict(self, user_id: int, k: int) -> list[int]: ...


def ndcg_at_k(relevant: set[int], recommended: list[int], k: int) -> float:
    dcg = sum(
        1.0 / np.log2(rank + 2) for rank, item in enumerate(recommended[:k]) if item in relevant
    )
    ideal_dcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def precision_at_k(relevant: set[int], recommended: list[int], k: int) -> float:
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(relevant: set[int], recommended: list[int], k: int) -> float:
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant) if relevant else 0.0


def hit_rate_at_k(relevant: set[int], recommended: list[int], k: int) -> float:
    return float(any(item in relevant for item in recommended[:k]))


def average_precision_at_k(relevant: set[int], recommended: list[int], k: int) -> float:
    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant), k) if relevant else 0.0


def _compute_mean_metrics(
    user_metrics: list[dict[str, float]],
) -> dict[str, float]:
    keys = user_metrics[0].keys()
    return {k: round(float(np.mean([m[k] for m in user_metrics])), 4) for k in keys}


def evaluate_recommender(
    model: object,
    test_df: pd.DataFrame,
    n_items: int,
    k: int,
    device: torch.device,
) -> dict[str, float]:
    """Avalia o modelo NCF no test set usando leave-one-out por usuário."""

    from src.models.base import BaseRecommender

    assert isinstance(model, BaseRecommender)
    model.eval()

    user_metrics = []
    test_positives = test_df[test_df["interaction"] == 1]

    for user_id, group in test_positives.groupby("user_id"):
        relevant = set(group["item_id"].tolist())
        recommended = model.recommend_top_k(int(user_id), k * 2, device)[:k]
        user_metrics.append(
            {
                f"ndcg_at_{k}": ndcg_at_k(relevant, recommended, k),
                f"precision_at_{k}": precision_at_k(relevant, recommended, k),
                f"recall_at_{k}": recall_at_k(relevant, recommended, k),
                f"hit_rate_at_{k}": hit_rate_at_k(relevant, recommended, k),
                f"map_at_{k}": average_precision_at_k(relevant, recommended, k),
            }
        )

    metrics = _compute_mean_metrics(user_metrics)
    logger.info("ncf_evaluated", **metrics)
    return metrics


def evaluate_baseline(
    model: object,
    test_df: pd.DataFrame,
    n_items: int,
    k: int,
) -> dict[str, float]:
    """Avalia baselines sklearn no test set."""
    user_metrics = []
    test_positives = test_df[test_df["interaction"] == 1]

    for user_id, group in test_positives.groupby("user_id"):
        relevant = set(group["item_id"].tolist())
        recommended = model.predict(int(user_id), k=k)
        user_metrics.append(
            {
                f"ndcg_at_{k}": ndcg_at_k(relevant, recommended, k),
                f"precision_at_{k}": precision_at_k(relevant, recommended, k),
                f"recall_at_{k}": recall_at_k(relevant, recommended, k),
                f"hit_rate_at_{k}": hit_rate_at_k(relevant, recommended, k),
                f"map_at_{k}": average_precision_at_k(relevant, recommended, k),
            }
        )

    return _compute_mean_metrics(user_metrics)


def save_eval_metrics(metrics: dict[str, float]) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    from src.data.features import load_gold
    from src.models.factory import RecommenderFactory
    from src.utils.config import MODELS_DIR, settings

    _, _, test_df, meta = load_gold()
    n_users, n_items = int(meta["n_users"]), int(meta["n_items"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RecommenderFactory.create_neural(n_users=n_users, n_items=n_items)
    model.load(MODELS_DIR / "ncf.pt")
    model = model.to(device)

    metrics = evaluate_recommender(model, test_df, n_items, settings.top_k, device)
    save_eval_metrics(metrics)
    print("Métricas finais (NCF):", metrics)
