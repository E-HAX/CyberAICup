"""Deterministic 5-fold split over the 200 training pairs."""
from __future__ import annotations

import numpy as np

from .config import N_FOLDS, N_TRAIN


def fold_of(idx: int) -> int:
    return idx % N_FOLDS


def split(fold: int) -> tuple[list[int], list[int]]:
    """Return (train_indices, val_indices) for a fold."""
    rng = np.random.RandomState(0)
    order = rng.permutation(N_TRAIN)
    val = sorted(int(i) for k, i in enumerate(order) if k % N_FOLDS == fold)
    train = sorted(int(i) for i in range(N_TRAIN) if i not in set(val))
    return train, val
