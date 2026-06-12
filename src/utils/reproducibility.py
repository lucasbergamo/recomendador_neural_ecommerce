import random

import numpy as np
import torch

from src.utils.config import settings


def set_global_seed(seed: int | None = None) -> None:
    s = seed if seed is not None else settings.seed
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
