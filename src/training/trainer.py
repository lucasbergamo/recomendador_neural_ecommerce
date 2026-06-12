"""Loop de treinamento do NCF com early stopping e MLflow tracking."""

import copy
import json

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.features import load_gold
from src.evaluation.metrics import evaluate_recommender
from src.models.factory import RecommenderFactory
from src.models.ncf import NeuralCF
from src.training.strategies import get_optimizer_strategy
from src.utils.config import METRICS_DIR, MODELS_DIR, settings
from src.utils.logger import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger(__name__)


def _build_tensors(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    users = torch.tensor(df["user_id"].values, dtype=torch.long)
    items = torch.tensor(df["item_id"].values, dtype=torch.long)
    labels = torch.tensor(df["interaction"].values, dtype=torch.float32)
    return users, items, labels


def run() -> None:
    set_global_seed()
    train_df, val_df, test_df, meta = load_gold()
    n_users, n_items = int(meta["n_users"]), int(meta["n_items"])

    model: NeuralCF = RecommenderFactory.create_neural(n_users=n_users, n_items=n_items)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    strategy = get_optimizer_strategy("adam")
    optimizer = strategy.create(model.parameters(), lr=settings.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    train_users, train_items, train_labels = _build_tensors(train_df)
    loader = DataLoader(
        TensorDataset(train_users, train_items, train_labels),
        batch_size=settings.batch_size,
        shuffle=True,
    )

    val_users, val_items, val_labels = _build_tensors(val_df[val_df["interaction"] == 1])

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="ncf_pytorch"):
        mlflow.log_params(
            {
                "model": "NeuralCF",
                "embedding_dim": settings.embedding_dim,
                "mlp_layers": str(settings.mlp_layers),
                "dropout": settings.dropout,
                "learning_rate": settings.learning_rate,
                "batch_size": settings.batch_size,
                "max_epochs": settings.max_epochs,
                "patience": settings.patience,
                "optimizer": "adam",
                "n_users": n_users,
                "n_items": n_items,
            }
        )

        best_val_loss = np.inf
        best_weights = copy.deepcopy(model.state_dict())
        no_improve = 0
        stopped_at = settings.max_epochs

        for epoch in range(1, settings.max_epochs + 1):
            model.train()
            train_loss = 0.0
            for u, i, y in loader:
                u, i, y = u.to(device), i.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(u, i), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(u)
            train_loss /= len(train_df)

            model.eval()
            with torch.no_grad():
                val_logits = model(val_users.to(device), val_items.to(device))
                val_loss = criterion(val_logits, val_labels.to(device)).item()

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= settings.patience:
                stopped_at = epoch
                logger.info("early_stopping", epoch=epoch, best_val_loss=round(best_val_loss, 4))
                break

        model.load_state_dict(best_weights)
        mlflow.log_param("stopped_at_epoch", stopped_at)

        metrics = evaluate_recommender(model, test_df, n_items, settings.top_k, device)
        mlflow.log_metrics(metrics)
        logger.info("ncf_trained", **metrics)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "ncf.pt"
        model.save(model_path)
        mlflow.log_artifact(str(model_path))

        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        with open(METRICS_DIR / "train_metrics.json", "w") as f:
            json.dump({"stopped_at_epoch": stopped_at, "best_val_loss": best_val_loss}, f)

        logger.info("model_saved", path=str(model_path))


if __name__ == "__main__":
    run()
