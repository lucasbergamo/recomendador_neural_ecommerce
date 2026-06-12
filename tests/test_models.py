"""Testes unitários para os modelos de recomendação."""

import torch
import pytest

from src.models.ncf import NeuralCF
from src.models.baselines import PopularityRecommender, SVDRecommender, RandomRecommender
from src.models.factory import ModelType, RecommenderFactory


N_USERS, N_ITEMS = 50, 100


def test_ncf_forward_shape():
    model = NeuralCF(n_users=N_USERS, n_items=N_ITEMS)
    users = torch.randint(0, N_USERS, (32,))
    items = torch.randint(0, N_ITEMS, (32,))
    logits = model(users, items)
    assert logits.shape == (32,), f"Esperado (32,), obtido {logits.shape}"


def test_ncf_predict_range():
    model = NeuralCF(n_users=N_USERS, n_items=N_ITEMS)
    users = torch.randint(0, N_USERS, (16,))
    items = torch.randint(0, N_ITEMS, (16,))
    probs = model.predict(users, items)
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilidades fora de [0, 1]"


def test_ncf_recommend_top_k():
    model = NeuralCF(n_users=N_USERS, n_items=N_ITEMS)
    device = torch.device("cpu")
    recs = model.recommend_top_k(user_id=0, k=10, device=device)
    assert len(recs) == 10
    assert len(set(recs)) == 10, "Recomendações com duplicatas"
    assert all(0 <= r < N_ITEMS for r in recs)


def test_factory_creates_ncf():
    model = RecommenderFactory.create(ModelType.NCF, n_users=N_USERS, n_items=N_ITEMS)
    assert isinstance(model, NeuralCF)


def test_factory_creates_svd():
    model = RecommenderFactory.create(ModelType.SVD, top_k=5)
    assert isinstance(model, SVDRecommender)


def test_factory_creates_popularity():
    model = RecommenderFactory.create(ModelType.POPULARITY, top_k=5)
    assert isinstance(model, PopularityRecommender)


def test_factory_creates_random():
    model = RecommenderFactory.create(ModelType.RANDOM, n_items=N_ITEMS, top_k=5)
    assert isinstance(model, RandomRecommender)
