"""Reproducibility utilities shared across model_v10 scripts.

Every script should call ``seed_everything`` **once** at startup before any
tensor allocation.  The helper also sets environment variables that influence
cuBLAS workspace behaviour to guarantee bitwise-identical results across runs
on the same hardware.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, *, deterministic: bool = True) -> None:
    """Set all random seeds and configure PyTorch for reproducibility.

    Parameters
    ----------
    seed:
        Master seed propagated to Python, NumPy, and PyTorch.
    deterministic:
        When *True* (default), enables ``torch.use_deterministic_algorithms``
        and disables cuDNN benchmarking so that results are bitwise-reproducible
        across runs on the *same* GPU.  Cross-GPU reproducibility still depends
        on identical hardware.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS workspace — prevents non-deterministic reductions
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` that re-seeds each worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
