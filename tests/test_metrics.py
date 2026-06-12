"""Testes unitários para as métricas de avaliação."""

import pytest

from src.evaluation.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_ndcg_perfect_ranking():
    relevant = {1, 2, 3}
    recommended = [1, 2, 3, 4, 5]
    score = ndcg_at_k(relevant, recommended, k=5)
    assert score == pytest.approx(1.0), "Ranking perfeito deve ter NDCG = 1.0"


def test_ndcg_no_hits():
    relevant = {10, 11}
    recommended = [1, 2, 3, 4, 5]
    assert ndcg_at_k(relevant, recommended, k=5) == 0.0


def test_precision_at_k_perfect():
    relevant = {1, 2, 3}
    recommended = [1, 2, 3]
    assert precision_at_k(relevant, recommended, k=3) == pytest.approx(1.0)


def test_precision_at_k_partial():
    relevant = {1, 2, 3}
    recommended = [1, 4, 5]
    assert precision_at_k(relevant, recommended, k=3) == pytest.approx(1 / 3)


def test_recall_at_k_all_found():
    relevant = {1, 2}
    recommended = [1, 2, 3, 4]
    assert recall_at_k(relevant, recommended, k=4) == pytest.approx(1.0)


def test_recall_at_k_none_found():
    relevant = {10, 11}
    recommended = [1, 2, 3]
    assert recall_at_k(relevant, recommended, k=3) == 0.0


def test_hit_rate_at_k_hit():
    relevant = {5}
    recommended = [1, 2, 5, 8]
    assert hit_rate_at_k(relevant, recommended, k=4) == 1.0


def test_hit_rate_at_k_miss():
    relevant = {99}
    recommended = [1, 2, 3]
    assert hit_rate_at_k(relevant, recommended, k=3) == 0.0


def test_map_at_k():
    relevant = {1, 3}
    recommended = [1, 2, 3, 4, 5]
    ap = average_precision_at_k(relevant, recommended, k=5)
    assert 0.0 < ap <= 1.0
