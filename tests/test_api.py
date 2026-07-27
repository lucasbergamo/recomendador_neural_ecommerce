"""Testes da API de serving — cobre o contrato HTTP do modelo NCF."""

from fastapi.testclient import TestClient

from src.api.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_recommend_valid_user():
    with TestClient(app) as client:
        response = client.get("/recommend/0?k=5")
        assert response.status_code == 200
        body = response.json()
        assert body["user_id"] == 0
        assert len(body["recommended_items"]) == 5
        first = body["recommended_items"][0]
        assert "item_id" in first
        assert "title" in first
        assert first["title"] != "desconhecido"


def test_recommend_invalid_user():
    with TestClient(app) as client:
        response = client.get("/recommend/999999")
        assert response.status_code == 404
