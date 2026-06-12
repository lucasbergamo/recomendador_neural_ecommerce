"""Orquestra o pipeline completo de dados: download → bronze → silver → gold."""

from src.data.features import (
    add_negative_samples,
    create_interaction_matrix,
    save_gold,
    temporal_split,
)
from src.data.load import download_movielens, load_bronze
from src.data.preprocess import preprocess_items, preprocess_ratings, preprocess_users, save_silver
from src.utils.logger import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger(__name__)


def run() -> None:
    set_global_seed()
    logger.info("pipeline_started")

    download_movielens()
    ratings_raw, items_raw, users_raw = load_bronze()

    ratings = preprocess_ratings(ratings_raw)
    items = preprocess_items(items_raw)
    users = preprocess_users(users_raw)
    save_silver(ratings, items, users)

    n_users = ratings["user_id"].nunique()
    n_items = ratings["item_id"].nunique()

    interactions = create_interaction_matrix(ratings)
    interactions = add_negative_samples(interactions, n_items)
    train, val, test = temporal_split(interactions)
    save_gold(train, val, test, n_users, n_items)

    logger.info("pipeline_finished", n_users=n_users, n_items=n_items)


if __name__ == "__main__":
    run()
