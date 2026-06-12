"""Download e carregamento do dataset MovieLens 100K (bronze layer)."""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import DATA_BRONZE_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
RATINGS_FILE = "ml-100k/u.data"
ITEMS_FILE = "ml-100k/u.item"
USERS_FILE = "ml-100k/u.user"


def download_movielens(dest_dir: Path = DATA_BRONZE_DIR) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    ratings_path = dest_dir / "ratings.csv"

    if ratings_path.exists():
        logger.info("dataset_already_exists", path=str(ratings_path))
        return

    logger.info("downloading_dataset", url=MOVIELENS_URL)
    response = requests.get(MOVIELENS_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        _extract_ratings(zf, dest_dir)
        _extract_items(zf, dest_dir)
        _extract_users(zf, dest_dir)

    logger.info("dataset_downloaded", dest=str(dest_dir))


def _extract_ratings(zf: zipfile.ZipFile, dest_dir: Path) -> None:
    with zf.open(RATINGS_FILE) as f:
        df = pd.read_csv(
            f,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
        )
    df.to_csv(dest_dir / "ratings.csv", index=False)


def _extract_items(zf: zipfile.ZipFile, dest_dir: Path) -> None:
    with zf.open(ITEMS_FILE) as f:
        df = pd.read_csv(
            f,
            sep="|",
            encoding="latin-1",
            names=[
                "item_id", "title", "release_date", "video_release_date", "imdb_url",
                "unknown", "action", "adventure", "animation", "children", "comedy",
                "crime", "documentary", "drama", "fantasy", "film_noir", "horror",
                "musical", "mystery", "romance", "sci_fi", "thriller", "war", "western",
            ],
            usecols=["item_id", "title", "release_date",
                     "action", "adventure", "comedy", "drama", "romance", "thriller"],
        )
    df.to_csv(dest_dir / "items.csv", index=False)


def _extract_users(zf: zipfile.ZipFile, dest_dir: Path) -> None:
    with zf.open(USERS_FILE) as f:
        df = pd.read_csv(
            f,
            sep="|",
            names=["user_id", "age", "gender", "occupation", "zip_code"],
        )
    df.to_csv(dest_dir / "users.csv", index=False)


def load_bronze() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(DATA_BRONZE_DIR / "ratings.csv")
    items = pd.read_csv(DATA_BRONZE_DIR / "items.csv")
    users = pd.read_csv(DATA_BRONZE_DIR / "users.csv")
    logger.info("bronze_loaded", ratings=len(ratings), items=len(items), users=len(users))
    return ratings, items, users
