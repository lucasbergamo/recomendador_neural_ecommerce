"""Strategy Pattern: intercambia algoritmos de otimização sem mudar o trainer."""

from abc import ABC, abstractmethod

import torch


class OptimizerStrategy(ABC):
    """Interface para estratégias de otimização."""

    @abstractmethod
    def create(self, parameters: object, lr: float) -> torch.optim.Optimizer:
        pass


class AdamStrategy(OptimizerStrategy):
    """Adam — padrão para modelos baseados em embeddings."""

    def create(self, parameters: object, lr: float) -> torch.optim.Optimizer:
        return torch.optim.Adam(parameters, lr=lr)


class AdamWStrategy(OptimizerStrategy):
    """AdamW — Adam com weight decay desacoplado (melhor regularização)."""

    def __init__(self, weight_decay: float = 1e-4) -> None:
        self.weight_decay = weight_decay

    def create(self, parameters: object, lr: float) -> torch.optim.Optimizer:
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=self.weight_decay)


class SGDStrategy(OptimizerStrategy):
    """SGD com momentum — mais lento mas útil para comparação."""

    def __init__(self, momentum: float = 0.9) -> None:
        self.momentum = momentum

    def create(self, parameters: object, lr: float) -> torch.optim.Optimizer:
        return torch.optim.SGD(parameters, lr=lr, momentum=self.momentum)


def get_optimizer_strategy(name: str = "adam") -> OptimizerStrategy:
    strategies: dict[str, OptimizerStrategy] = {
        "adam": AdamStrategy(),
        "adamw": AdamWStrategy(),
        "sgd": SGDStrategy(),
    }
    if name not in strategies:
        raise ValueError(f"Estratégia desconhecida: {name}. Opções: {list(strategies)}")
    return strategies[name]
