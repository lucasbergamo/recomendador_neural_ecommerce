"""Silver → Gold: feature engineering e splits para o modelo de recomendação."""

import numpy as np
import pandas as pd

from src.data.preprocess import load_silver
from src.utils.config import DATA_GOLD_DIR
from src.utils.logger import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger(__name__)

NEGATIVE_RATIO = 4  # negativos implícitos por positivo (leave-one-out)


def create_interaction_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Converte ratings explícitos em feedback implícito binário (1 = interagiu)."""
    df = ratings[["user_id", "item_id", "timestamp"]].copy()
    df["interaction"] = 1
    return df


def add_negative_samples(interactions: pd.DataFrame, n_items: int) -> pd.DataFrame:
    """Adiciona amostras negativas para treino — itens não visitados pelo usuário."""
    set_global_seed()
    positives = interactions.groupby("user_id")["item_id"].apply(set).to_dict()
    negatives = []

    for user_id, pos_items in positives.items():
        all_items = set(range(n_items))
        neg_candidates = list(all_items - pos_items)
        n_neg = min(len(pos_items) * NEGATIVE_RATIO, len(neg_candidates))
        sampled = np.random.choice(neg_candidates, size=n_neg, replace=False)
        negatives.extend(
            {"user_id": user_id, "item_id": item, "interaction": 0} for item in sampled
        )

    neg_df = pd.DataFrame(negatives)
    return pd.concat([interactions, neg_df], ignore_index=True)


def temporal_split(
    interactions: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leave-time-out split: últimas interações por usuário → val/test."""
    positives = interactions[interactions["interaction"] == 1].copy()
    negatives = interactions[interactions["interaction"] == 0].copy()

    positives_sorted = positives.sort_values(["user_id", "timestamp"])
    user_groups = positives_sorted.groupby("user_id")

    train_pos, val_pos, test_pos = [], [], []
    for _, group in user_groups:
        n = len(group)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        train_pos.append(group.iloc[: n - n_test - n_val])
        val_pos.append(group.iloc[n - n_test - n_val : n - n_test])
        test_pos.append(group.iloc[n - n_test :])

    train_df = pd.concat([*train_pos, negatives], ignore_index=True)
    val_df = pd.concat(val_pos, ignore_index=True)
    test_df = pd.concat(test_pos, ignore_index=True)

    return train_df, val_df, test_df


def save_gold(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    n_users: int,
    n_items: int,
) -> None:
    DATA_GOLD_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(DATA_GOLD_DIR / "train.parquet", index=False)
    val.to_parquet(DATA_GOLD_DIR / "val.parquet", index=False)
    test.to_parquet(DATA_GOLD_DIR / "test.parquet", index=False)

    metadata = pd.DataFrame([{"n_users": n_users, "n_items": n_items}])
    metadata.to_parquet(DATA_GOLD_DIR / "metadata.parquet", index=False)
    logger.info(
        "gold_saved",
        train=len(train),
        val=len(val),
        test=len(test),
        n_users=n_users,
        n_items=n_items,
    )


def load_gold() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train = pd.read_parquet(DATA_GOLD_DIR / "train.parquet")
    val = pd.read_parquet(DATA_GOLD_DIR / "val.parquet")
    test = pd.read_parquet(DATA_GOLD_DIR / "test.parquet")
    meta = pd.read_parquet(DATA_GOLD_DIR / "metadata.parquet").iloc[0].to_dict()
    return train, val, test, meta


def main() -> None:
    set_global_seed()
    ratings, _, _ = load_silver()
    n_users = ratings["user_id"].nunique()
    n_items = ratings["item_id"].nunique()

    interactions = create_interaction_matrix(ratings)
    interactions = add_negative_samples(interactions, n_items)
    train, val, test = temporal_split(interactions)
    save_gold(train, val, test, n_users, n_items)


if __name__ == "__main__":
    main()
