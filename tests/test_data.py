"""Testes para o pipeline de dados."""

import pandas as pd

from src.data.features import create_interaction_matrix
from src.data.preprocess import (
    _filter_cold_start,
    _reindex_ids,
    preprocess_ratings,
    preprocess_users,
)


def _make_ratings(n: int = 50) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "user_id": rng.integers(0, 10, n),
            "item_id": rng.integers(0, 20, n),
            "rating": rng.integers(1, 6, n).astype(float),
            "timestamp": rng.integers(800000000, 900000000, n),
        }
    )


def test_filter_cold_start_removes_sparse():
    df = _make_ratings(100)
    filtered = _filter_cold_start(df)
    assert len(filtered) <= len(df)


def test_reindex_ids_contiguous():
    df = _make_ratings(50)
    reindexed = _reindex_ids(df)
    user_ids = sorted(reindexed["user_id"].unique())
    item_ids = sorted(reindexed["item_id"].unique())
    assert user_ids == list(range(len(user_ids)))
    assert item_ids == list(range(len(item_ids)))


def test_preprocess_ratings_types():
    df = _make_ratings(50)
    result = preprocess_ratings(df)
    assert result["rating"].dtype == float
    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])


def test_create_interaction_matrix_binary():
    df = _make_ratings(30)
    interactions = create_interaction_matrix(df)
    assert set(interactions["interaction"].unique()) == {1}


def test_preprocess_users_gender_encoding():
    users = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "age": [25, 30, 45],
            "gender": ["M", "F", "M"],
            "occupation": ["student", "engineer", "doctor"],
            "zip_code": ["00000", "11111", "22222"],
        }
    )
    result = preprocess_users(users)
    assert set(result["gender"].unique()).issubset({0, 1})
