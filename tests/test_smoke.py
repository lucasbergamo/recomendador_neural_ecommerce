"""Smoke tests: verifica que os módulos importam e inicializam sem erros."""

import torch


def test_import_ncf():
    from src.models.ncf import NeuralCF
    model = NeuralCF(n_users=10, n_items=20)
    assert model is not None


def test_import_factory():
    from src.models.factory import RecommenderFactory, ModelType
    model = RecommenderFactory.create(ModelType.RANDOM, n_items=20, top_k=5)
    assert model is not None


def test_import_settings():
    from src.utils.config import settings
    assert settings.seed == 42


def test_import_metrics():
    from src.evaluation.metrics import ndcg_at_k, precision_at_k
    assert ndcg_at_k({1}, [1, 2, 3], k=3) > 0


def test_ncf_forward_no_crash():
    from src.models.ncf import NeuralCF
    model = NeuralCF(n_users=10, n_items=20)
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([5, 6, 7])
    out = model(users, items)
    assert out.shape == (3,)
