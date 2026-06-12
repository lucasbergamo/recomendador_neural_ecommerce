"""Baselines de recomendação baseados em Scikit-Learn e heurísticas."""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class PopularityRecommender:
    """Recomenda os itens mais populares globalmente (baseline trivial)."""

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k
        self._popular_items: list[int] = []

    def fit(self, train: pd.DataFrame) -> "PopularityRecommender":
        self._popular_items = (
            train[train["interaction"] == 1]["item_id"]
            .value_counts()
            .head(self.top_k)
            .index.tolist()
        )
        return self

    def predict(self, user_id: int, k: int | None = None) -> list[int]:
        return self._popular_items[: k or self.top_k]


class SVDRecommender:
    """Matrix Factorization via TruncatedSVD — baseline clássico para rec-sys."""

    def __init__(self, n_components: int = 50, top_k: int = 10) -> None:
        self.n_components = n_components
        self.top_k = top_k
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._n_items: int = 0

    def fit(self, train: pd.DataFrame) -> "SVDRecommender":
        positives = train[train["interaction"] == 1]
        n_users = positives["user_id"].max() + 1
        self._n_items = positives["item_id"].max() + 1

        matrix = np.zeros((n_users, self._n_items))
        for _, row in positives.iterrows():
            matrix[int(row["user_id"]), int(row["item_id"])] = 1.0

        self._user_factors = self._svd.fit_transform(matrix)
        self._item_factors = self._svd.components_.T
        self._user_factors = normalize(self._user_factors)
        self._item_factors = normalize(self._item_factors)
        return self

    def predict_scores(self, user_id: int) -> np.ndarray:
        if self._user_factors is None or self._item_factors is None:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")
        return self._user_factors[user_id] @ self._item_factors.T

    def predict(self, user_id: int, k: int | None = None) -> list[int]:
        scores = self.predict_scores(user_id)
        top = int(k or self.top_k)
        return np.argsort(scores)[::-1][:top].tolist()


class RandomRecommender:
    """Recomenda itens aleatoriamente — dummy baseline (lower bound)."""

    def __init__(self, n_items: int, top_k: int = 10, seed: int = 42) -> None:
        self.n_items = n_items
        self.top_k = top_k
        self._rng = np.random.default_rng(seed)

    def fit(self, train: pd.DataFrame) -> "RandomRecommender":
        return self

    def predict(self, user_id: int, k: int | None = None) -> list[int]:
        top = int(k or self.top_k)
        return self._rng.choice(self.n_items, size=top, replace=False).tolist()
