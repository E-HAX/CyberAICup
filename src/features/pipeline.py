"""Cached artifacts shared by every worker container.

`build_and_cache` runs once and writes to the work Volume:

    features/base_train.parquet / base_test.parquet   blocks F1-F8
    features/folds.npz                                identical folds for all workers
    features/selection.npz                            per-fold mutual-information ranking
    features/meta.json                                shapes, column names, checksums

Computing these once matters because the hyperparameter search fans out over
hundreds of containers; each of them loads a ready matrix instead of repeating
the same deterministic work. The mutual-information ranking is stored per fold
and computed only from that fold's training rows, so the "compact feature set"
option stays leakage-free.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

import config as C
import io_utils as IO
from features import base as fbase
from models import cv as cvmod

BASE_TRAIN = C.FEATURES_DIR / "base_train.parquet"
BASE_TEST = C.FEATURES_DIR / "base_test.parquet"
SELECTION = C.FEATURES_DIR / "selection.npz"
META = C.FEATURES_DIR / "meta.json"


def build_and_cache() -> dict:
    C.ensure_dirs()
    train, test = IO.load_train(), IO.load_test()
    y = IO.labels_to_ids(train["label"])
    groups = IO.group_keys(train)

    base_tr = fbase.build(train[C.FEATURE_COLS])
    base_te = fbase.build(test[C.FEATURE_COLS])
    base_tr.to_parquet(BASE_TRAIN)
    base_te.to_parquet(BASE_TEST)

    folds = cvmod.build_folds(y, groups)
    cvmod.save_folds(folds)

    ranking = {}
    for scheme, splits in folds.items():
        for i, (tr_idx, _) in enumerate(splits):
            mi = mutual_info_classif(
                base_tr.iloc[tr_idx].to_numpy(), y[tr_idx], random_state=C.SEED
            )
            ranking[f"{scheme}_{i}"] = np.argsort(mi)[::-1].astype(np.int32)
    np.savez_compressed(SELECTION, **ranking)

    meta = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_base_features": int(base_tr.shape[1]),
        "base_columns": list(base_tr.columns),
        "n_pseudo_groups": int(len(set(groups))),
        "folds": {k: len(v) for k, v in folds.items()},
    }
    META.write_text(json.dumps(meta, indent=2))
    return meta


class Cache:
    """Lazy per-container handle on the cached artifacts."""

    _instance: "Cache | None" = None

    def __init__(self):
        self.train = IO.load_train()
        self.test = IO.load_test()
        self.y = IO.labels_to_ids(self.train["label"])
        self.base_train = pd.read_parquet(BASE_TRAIN)
        self.base_test = pd.read_parquet(BASE_TEST)
        self.folds = cvmod.load_folds()
        self.selection = dict(np.load(SELECTION))
        # Feature block F12, present only after the probe stage has run. The
        # probes are trained on auxiliary-corpus targets, never on competition
        # labels, so they are safe to compute once outside the fold loop.
        self.probes_train, self.probes_test = self._load_probes()

    @staticmethod
    def _load_probes():
        tr = C.FEATURES_DIR / "probes_train.parquet"
        te = C.FEATURES_DIR / "probes_test.parquet"
        if tr.exists() and te.exists():
            return pd.read_parquet(tr), pd.read_parquet(te)
        return None, None

    @property
    def has_probes(self) -> bool:
        return self.probes_train is not None

    @classmethod
    def get(cls) -> "Cache":
        if cls._instance is None:
            cls._instance = Cache()
        return cls._instance

    def top_columns(self, scheme: str, fold: int, k: int) -> list[str]:
        order = self.selection[f"{scheme}_{fold}"]
        cols = self.base_train.columns.to_numpy()
        return list(cols[order[:k]])
