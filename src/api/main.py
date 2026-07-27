"""API HTTP para servir o modelo NCF treinado — expõe recomend_top_k via endpoint."""

from contextlib import asynccontextmanager

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException

from src.models.base import BaseRecommender
from src.models.factory import RecommenderFactory
from src.utils.config import DATA_GOLD_DIR, DATA_SILVER_DIR, MODELS_DIR, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_state: dict[str, object] = {}


def _load_item_titles() -> dict[int, str]:
    id_map = pd.read_parquet(DATA_SILVER_DIR / "item_id_map.parquet")
    items = pd.read_parquet(DATA_SILVER_DIR / "items.parquet")
    merged = id_map.merge(items, left_on="item_id_original", right_on="item_id")
    return dict(zip(merged["item_id_x"], merged["title"], strict=True))


@asynccontextmanager
async def lifespan(app: FastAPI):
    meta = pd.read_parquet(DATA_GOLD_DIR / "metadata.parquet").iloc[0].to_dict()
    n_users, n_items = int(meta["n_users"]), int(meta["n_items"])

    model = RecommenderFactory.create_neural(n_users=n_users, n_items=n_items)
    model.load(MODELS_DIR / "ncf.pt")
    model.eval()

    _state["model"] = model
    _state["device"] = torch.device("cpu")
    _state["titles"] = _load_item_titles()
    logger.info("model_loaded", n_users=n_users, n_items=n_items)
    yield
    _state.clear()


app = FastAPI(
    title="Recomendador Neural E-commerce",
    description="Serving do modelo NCF (NeuMF) treinado — Tech Challenge Fase 02 FIAP",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = settings.top_k) -> dict:
    model: BaseRecommender = _state["model"]  # type: ignore[assignment]
    device: torch.device = _state["device"]  # type: ignore[assignment]
    titles: dict[int, str] = _state["titles"]  # type: ignore[assignment]

    if not 0 <= user_id < model.n_users:
        raise HTTPException(
            status_code=404,
            detail=f"user_id inválido — esperado entre 0 e {model.n_users - 1}",
        )

    item_ids = model.recommend_top_k(user_id, k, device)
    recommended_items = [
        {"item_id": item_id, "title": titles.get(item_id, "desconhecido")} for item_id in item_ids
    ]
    return {"user_id": user_id, "recommended_items": recommended_items}
