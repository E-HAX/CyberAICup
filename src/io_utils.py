"""Loading, validation and label handling for the RTC dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def load_train() -> pd.DataFrame:
    df = pd.read_csv(C.TRAIN_CSV)
    _validate(df, labelled=True)
    return df


def load_test() -> pd.DataFrame:
    df = pd.read_csv(C.TEST_CSV)
    _validate(df, labelled=False)
    return df


def _validate(df: pd.DataFrame, labelled: bool) -> None:
    missing = [c for c in C.FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    if labelled:
        if "label" not in df.columns:
            raise ValueError("labelled frame has no 'label' column")
        unknown = set(df["label"]) - set(C.LABELS)
        if unknown:
            raise ValueError(f"unknown label strings: {sorted(unknown)}")
    if not np.allclose(df[C.TIME_COLS[0]].to_numpy(), 0.0):
        raise ValueError("relative_time_0 is not identically zero")


def labels_to_ids(labels: pd.Series) -> np.ndarray:
    return labels.map(C.LABEL_TO_ID).to_numpy(dtype=np.int64)


def ids_to_labels(ids: np.ndarray) -> np.ndarray:
    return np.array([C.ID_TO_LABEL[int(i)] for i in ids], dtype=object)


def split_app_mode(labels: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split 'Zoom_video' into ('Zoom', 'video')."""
    parts = labels.str.rsplit("_", n=1, expand=True)
    return parts[0], parts[1]


def lengths(df: pd.DataFrame) -> np.ndarray:
    return df[C.LEN_COLS].to_numpy(dtype=np.float64)


def times(df: pd.DataFrame) -> np.ndarray:
    return df[C.TIME_COLS].to_numpy(dtype=np.float64)


def iats(df: pd.DataFrame) -> np.ndarray:
    """Inter-arrival times d_1..d_4 derived from the cumulative timestamps."""
    return np.diff(times(df), axis=1)


def group_keys(df: pd.DataFrame) -> np.ndarray:
    """Pseudo-group key for leakage-aware CV.

    No source-call identifier is provided with the dataset, yet flows produced
    by one call are near-identical and the test split comes from held-out
    calls. Flows sharing a coarsely rounded length tuple and duration decade
    are therefore treated as one group so that they cannot straddle a fold
    boundary and inflate the validation score.
    """
    L = lengths(df)
    t = df[C.TIME_COLS[-1]].to_numpy(dtype=np.float64)
    Lq = (np.round(L / C.GROUP_LEN_ROUND) * C.GROUP_LEN_ROUND).astype(np.int64)
    tq = np.round(np.log10(np.maximum(t, 1e-9)), C.GROUP_LOG_TIME_ROUND)
    keys = [f"{'-'.join(map(str, row))}|{tv}" for row, tv in zip(Lq, tq)]
    return np.asarray(keys, dtype=object)
