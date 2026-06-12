"""Template Method Pattern: define o esqueleto de treinamento de qualquer recomendador."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn


class BaseRecommender(ABC, nn.Module):
    """Classe base para todos os modelos de recomendação.

    Implementa o Template Method para padronizar o fluxo:
    predict → score → rank → recommend.
    """

    def __init__(self, n_users: int, n_items: int) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items

    @abstractmethod
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Retorna scores brutos (logits) para pares user-item."""

    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Template method: score → sigmoid → probabilidade de interação."""
        logits = self.forward(user_ids, item_ids)
        return torch.sigmoid(logits)

    def recommend_top_k(self, user_id: int, k: int, device: torch.device) -> list[int]:
        """Retorna os top-k itens recomendados para um usuário."""
        self.eval()
        with torch.no_grad():
            users = torch.full((self.n_items,), user_id, dtype=torch.long, device=device)
            items = torch.arange(self.n_items, dtype=torch.long, device=device)
            scores = self.predict(users, items).cpu().numpy()
        return sorted(range(self.n_items), key=lambda i: scores[i], reverse=True)[:k]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Path) -> None:
        self.load_state_dict(torch.load(path, weights_only=True))
