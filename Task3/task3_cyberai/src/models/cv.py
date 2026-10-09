"""Cross-validation protocol shared by every container.

Two schemes run side by side:

``plain``  repeated stratified K-fold - the optimistic number, comparable with
           what most published baselines report.
``group``  stratified group K-fold over the pseudo-group key from io_utils -
           near-duplicate flows (the visible trace of several flows coming from
           one source call) are kept inside a single fold. Because the test
           split is drawn from held-out calls, this is the pessimistic and
           therefore decisive number for model selection.

Folds are generated once, stored in the work Volume and reloaded by every
worker so that scores computed in different containers are comparable.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold

import config as C

FOLDS_PATH = C.FEATURES_DIR / "folds.npz"


def build_folds(y: np.ndarray, groups: np.ndarray) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    plain = list(
        RepeatedStratifiedKFold(
            n_splits=C.N_SPLITS, n_repeats=C.N_REPEATS, random_state=C.SEED
        ).split(np.zeros(len(y)), y)
    )
    gcodes = np.unique(groups, return_inverse=True)[1]
    group_folds: list[tuple[np.ndarray, np.ndarray]] = []
    for rep in range(2):
        rng = np.random.RandomState(C.SEED + rep)
        perm = rng.permutation(len(y))
        splitter = StratifiedGroupKFold(n_splits=C.N_SPLITS, shuffle=True, random_state=C.SEED + rep)
        for tr, va in splitter.split(np.zeros(len(y)), y[perm], gcodes[perm]):
            group_folds.append((perm[tr], perm[va]))
    return {"plain": plain, "group": group_folds}


def save_folds(folds: dict, path=FOLDS_PATH) -> None:
    payload = {}
    for scheme, splits in folds.items():
        for i, (tr, va) in enumerate(splits):
            payload[f"{scheme}_{i}_tr"] = tr
            payload[f"{scheme}_{i}_va"] = va
        payload[f"{scheme}_n"] = np.array([len(splits)])
    np.savez_compressed(path, **payload)


def load_folds(path=FOLDS_PATH) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    z = np.load(path, allow_pickle=False)
    out = {}
    for scheme in ("plain", "group"):
        n = int(z[f"{scheme}_n"][0])
        out[scheme] = [(z[f"{scheme}_{i}_tr"], z[f"{scheme}_{i}_va"]) for i in range(n)]
    return out


def n_repeats_of(scheme: str, n_folds: int) -> int:
    return max(1, n_folds // C.N_SPLITS)


def limit_repeats(splits: list, n_repeats: int) -> list:
    """Take only the first ``n_repeats`` repeats of a fold list."""
    return splits[: n_repeats * C.N_SPLITS]
