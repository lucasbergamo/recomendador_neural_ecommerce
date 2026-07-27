"""Bronze → Silver: limpeza e normalização dos dados brutos."""

import pandas as pd

from src.data.load import load_bronze
from src.utils.config import DATA_SILVER_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5


def preprocess_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    df = ratings.copy()
    df = df.dropna(subset=["user_id", "item_id", "rating"])
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = _filter_cold_start(df)
    df = _reindex_ids(df)
    logger.info(
        "ratings_preprocessed",
        rows=len(df),
        users=df["user_id"].nunique(),
        items=df["item_id"].nunique(),
    )
    return df


def preprocess_items(items: pd.DataFrame) -> pd.DataFrame:
    df = items.copy()
    df = df.dropna(subset=["item_id", "title"])
    df["title"] = df["title"].str.strip()
    return df


def preprocess_users(users: pd.DataFrame) -> pd.DataFrame:
    df = users.copy()
    df = df.dropna(subset=["user_id"])
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median())
    df["gender"] = df["gender"].map({"M": 0, "F": 1}).fillna(0).astype(int)
    return df


def _filter_cold_start(df: pd.DataFrame) -> pd.DataFrame:
    user_counts = df["user_id"].value_counts()
    item_counts = df["item_id"].value_counts()
    valid_users = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
    valid_items = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index
    return df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]


def _build_id_maps(df: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    user_map = {uid: idx for idx, uid in enumerate(sorted(df["user_id"].unique()))}
    item_map = {iid: idx for idx, iid in enumerate(sorted(df["item_id"].unique()))}
    return user_map, item_map


def _reindex_ids(df: pd.DataFrame) -> pd.DataFrame:
    user_map, item_map = _build_id_maps(df)
    df = df.copy()
    df["user_id"] = df["user_id"].map(user_map)
    df["item_id"] = df["item_id"].map(item_map)
    return df


def save_silver(ratings: pd.DataFrame, items: pd.DataFrame, users: pd.DataFrame) -> None:
    DATA_SILVER_DIR.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(DATA_SILVER_DIR / "ratings.parquet", index=False)
    items.to_parquet(DATA_SILVER_DIR / "items.parquet", index=False)
    users.to_parquet(DATA_SILVER_DIR / "users.parquet", index=False)
    logger.info("silver_saved", path=str(DATA_SILVER_DIR))


def save_item_id_map(item_map: dict[int, int]) -> None:
    """Persiste item_id original → reindexado — permite recuperar título depois do treino."""
    DATA_SILVER_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {"item_id_original": list(item_map.keys()), "item_id": list(item_map.values())}
    )
    df.to_parquet(DATA_SILVER_DIR / "item_id_map.parquet", index=False)
    logger.info("item_id_map_saved", n_items=len(df))


def load_silver() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_parquet(DATA_SILVER_DIR / "ratings.parquet")
    items = pd.read_parquet(DATA_SILVER_DIR / "items.parquet")
    users = pd.read_parquet(DATA_SILVER_DIR / "users.parquet")
    return ratings, items, users


def main() -> None:
    ratings_raw, items_raw, users_raw = load_bronze()
    ratings = preprocess_ratings(ratings_raw)
    items = preprocess_items(items_raw)
    users = preprocess_users(users_raw)
    save_silver(ratings, items, users)

    clean = ratings_raw.dropna(subset=["user_id", "item_id", "rating"])
    filtered = _filter_cold_start(clean)
    _, item_map = _build_id_maps(filtered)
    save_item_id_map(item_map)


if __name__ == "__main__":
    main()
