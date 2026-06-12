"""Neural Collaborative Filtering (NCF): GMF + MLP combinados (He et al., 2017)."""

import torch
import torch.nn as nn

from src.models.base import BaseRecommender
from src.utils.config import settings


class NeuralCF(BaseRecommender):
    """Combina Generalized Matrix Factorization (GMF) e MLP para recomendação.

    Arquitetura NeuMF:
    - GMF path: user_emb ⊙ item_emb (element-wise product)
    - MLP path: [user_emb || item_emb] → camadas densas
    - Saída: concat(GMF, MLP) → Linear(1) → logit

    Por que NeuMF em vez de MF puro:
    - MF captura interações lineares; MLP captura não-linearidades
    - A combinação supera ambos individualmente em benchmarks de rec-sys
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int | None = None,
        mlp_layers: list[int] | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__(n_users, n_items)
        emb_dim = embedding_dim or settings.embedding_dim
        layers = mlp_layers or settings.mlp_layers
        drop = dropout or settings.dropout

        # GMF embeddings
        self.gmf_user = nn.Embedding(n_users, emb_dim)
        self.gmf_item = nn.Embedding(n_items, emb_dim)

        # MLP embeddings (tamanho separado para GMF e MLP = flexibilidade)
        self.mlp_user = nn.Embedding(n_users, emb_dim)
        self.mlp_item = nn.Embedding(n_items, emb_dim)

        # MLP layers
        mlp_input_dim = emb_dim * 2
        mlp_blocks: list[nn.Module] = []
        in_dim = mlp_input_dim
        for out_dim in layers:
            mlp_blocks += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_blocks)

        # Camada final: GMF output (emb_dim) + MLP output (layers[-1]) → logit
        self.output = nn.Linear(emb_dim + layers[-1], 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # GMF path
        gmf_out = self.gmf_user(user_ids) * self.gmf_item(item_ids)

        # MLP path
        mlp_in = torch.cat([self.mlp_user(user_ids), self.mlp_item(item_ids)], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # Saída combinada — retorna logits (sigmoid aplicado apenas na inferência)
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.output(combined).squeeze(-1)
