"""Tensor encoding and augmentation for the competition flows.

Encoding is shared with the auxiliary corpus so a model pretrained on MIRAGE
sees the competition data in exactly the same input space: per-packet
continuous descriptors plus a quantised size bin.

Augmentations are chosen to be plausible transport-level perturbations rather
than generic noise - a capture of the same call on a different path would show
slightly different pacing and padding, not arbitrary feature jitter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import config as C
from transformer.model import N_LEN_BINS

# Fixed bin edges over the observed payload-size range. They are dataset
# independent by construction (geometric spacing from 20 B to the MTU) so the
# same bin index means the same thing in both corpora.
LEN_EDGES = np.unique(np.round(np.geomspace(20, 1400, N_LEN_BINS)).astype(int))


def size_bins(lengths: np.ndarray) -> np.ndarray:
    return np.digitize(lengths, LEN_EDGES).astype(np.int64)


def encode_flow(lengths: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Continuous per-packet descriptors, shape (n_flows, n_packets, N_CONT).

    Columns
        0 log1p(length) scaled           4 gap share of the flow's time span
        1 length / 1500                  5 sub-millisecond gap flag
        2 log1p(gap in milliseconds)     6 path-MTU-regime flag
        3 cumulative time share          7 small-packet (STUN/RTCP/DTX) flag
    """
    n, t = lengths.shape
    gaps = np.zeros_like(times)
    gaps[:, 1:] = np.diff(times, axis=1)
    span = np.maximum(times[:, -1:], 1e-9)
    out = np.stack(
        [
            np.log1p(lengths) / 7.5,
            lengths / 1500.0,
            np.log1p(gaps * 1e3),
            times / span,
            gaps / span,
            (gaps < 1e-3).astype(np.float64),
            (lengths >= 1100).astype(np.float64),
            (lengths < 100).astype(np.float64),
        ],
        axis=-1,
    )
    return out.astype(np.float32)


def encode_frame(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    L = df[C.LEN_COLS].to_numpy(dtype=np.float64)
    T = df[C.TIME_COLS].to_numpy(dtype=np.float64)
    return encode_flow(L, T), size_bins(L)


class FlowDataset(torch.utils.data.Dataset):
    """Flows with optional traffic-plausible augmentation."""

    def __init__(
        self,
        lengths: np.ndarray,
        times: np.ndarray,
        y: np.ndarray | None = None,
        augment: bool = False,
        time_jitter: float = 0.10,
        len_jitter: float = 2.0,
        drop_prob: float = 0.10,
        seed: int = C.SEED,
    ):
        self.lengths = lengths.astype(np.float64)
        self.times = times.astype(np.float64)
        self.y = None if y is None else y.astype(np.int64)
        self.augment = augment
        self.time_jitter = time_jitter
        self.len_jitter = len_jitter
        self.drop_prob = drop_prob
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.lengths)

    def _view(self, i: int):
        L = self.lengths[i].copy()
        T = self.times[i].copy()
        keep = np.ones(len(L), dtype=bool)
        if self.augment:
            # Path-dependent pacing: stretch or compress all gaps together.
            gaps = np.diff(T)
            gaps = gaps * self.rng.uniform(1 - self.time_jitter, 1 + self.time_jitter)
            T = np.concatenate([[0.0], np.cumsum(gaps)])
            # Padding-quantum jitter, never enough to change the size band.
            L = np.maximum(1.0, L + self.rng.uniform(-self.len_jitter, self.len_jitter, len(L)))
            if self.rng.rand() < self.drop_prob and len(L) > 2:
                j = self.rng.randint(1, len(L))
                keep[j] = False
        cont = encode_flow(L[None, :], T[None, :])[0]
        bins = size_bins(L)
        pad = ~keep
        return cont, bins, pad

    def __getitem__(self, i: int):
        cont, bins, pad = self._view(i)
        item = {
            "cont": torch.from_numpy(cont),
            "bins": torch.from_numpy(bins),
            "pad": torch.from_numpy(pad),
        }
        if self.y is not None:
            item["y"] = torch.tensor(self.y[i])
        return item


def frame_to_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (
        df[C.LEN_COLS].to_numpy(dtype=np.float64),
        df[C.TIME_COLS].to_numpy(dtype=np.float64),
    )
