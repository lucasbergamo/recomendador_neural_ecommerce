"""Bronze → Silver: limpeza e normalização dos dados brutos."""

import pandas as pd

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


def _reindex_ids(df: pd.DataFrame) -> pd.DataFrame:
    user_map = {uid: idx for idx, uid in enumerate(sorted(df["user_id"].unique()))}
    item_map = {iid: idx for idx, iid in enumerate(sorted(df["item_id"].unique()))}
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


def load_silver() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_parquet(DATA_SILVER_DIR / "ratings.parquet")
    items = pd.read_parquet(DATA_SILVER_DIR / "items.parquet")
    users = pd.read_parquet(DATA_SILVER_DIR / "users.parquet")
    return ratings, items, users
