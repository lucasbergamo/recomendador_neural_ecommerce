"""Factory Pattern: cria instâncias de modelos de recomendação por nome."""

from enum import StrEnum

from src.models.base import BaseRecommender
from src.models.baselines import PopularityRecommender, RandomRecommender, SVDRecommender
from src.models.ncf import NeuralCF
from src.utils.config import settings


class ModelType(StrEnum):
    NCF = "ncf"
    SVD = "svd"
    POPULARITY = "popularity"
    RANDOM = "random"


class RecommenderFactory:
    """Cria modelos de recomendação sem expor a lógica de instanciação ao chamador."""

    @staticmethod
    def create_neural(n_users: int, n_items: int) -> BaseRecommender:
        return NeuralCF(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=settings.embedding_dim,
            mlp_layers=settings.mlp_layers,
            dropout=settings.dropout,
        )

    @staticmethod
    def create_svd(top_k: int = 10) -> SVDRecommender:
        return SVDRecommender(top_k=top_k)

    @staticmethod
    def create_popularity(top_k: int = 10) -> PopularityRecommender:
        return PopularityRecommender(top_k=top_k)

    @staticmethod
    def create_random(n_items: int, top_k: int = 10) -> RandomRecommender:
        return RandomRecommender(n_items=n_items, top_k=top_k)

    @classmethod
    def create(cls, model_type: ModelType | str, **kwargs: object) -> object:
        """Cria modelo por tipo — ponto único de entrada para todos os modelos."""
        match ModelType(model_type):
            case ModelType.NCF:
                return cls.create_neural(**kwargs)
            case ModelType.SVD:
                return cls.create_svd(**kwargs)
            case ModelType.POPULARITY:
                return cls.create_popularity(**kwargs)
            case ModelType.RANDOM:
                return cls.create_random(**kwargs)
