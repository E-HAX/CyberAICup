"""Deterministic, target-free feature blocks F1-F8.

These transforms are pure row-wise functions of a flow's own five packets, so
they carry no fitted state and cannot leak label information. They are computed
once and cached to the Volume; the fitted, fold-dependent blocks (F9-F11) live
in features/likelihood.py and run inside the cross-validation loop instead.

Block map
    F1  raw and log-transformed columns
    F2  inter-arrival times and their distribution summary
    F3  packet-length distribution summary
    F4  sequence shape: pairwise differences/ratios, trends, monotonicity
    F5  protocol-semantics flags (length bands, padding residues, RTP/SRTP)
    F6  burst / frame-group structure at several inter-arrival thresholds
    F7  rate and throughput
    F8  spectral transforms of the length and log-inter-arrival sequences
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.fft import dct, rfft
from scipy.stats import kurtosis, skew

import config as C

EPS = 1e-9


def _stats(x: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    """Distribution summary of a per-row vector."""
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    q25, q50, q75 = np.percentile(x, [25, 50, 75], axis=1)
    return {
        f"{prefix}_min": x.min(axis=1),
        f"{prefix}_max": x.max(axis=1),
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_median": q50,
        f"{prefix}_q25": q25,
        f"{prefix}_q75": q75,
        f"{prefix}_iqr": q75 - q25,
        f"{prefix}_range": x.max(axis=1) - x.min(axis=1),
        f"{prefix}_sum": x.sum(axis=1),
        f"{prefix}_cv": std / (np.abs(mean) + EPS),
        f"{prefix}_skew": skew(x, axis=1, bias=False),
        f"{prefix}_kurt": kurtosis(x, axis=1, bias=False),
    }


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Return blocks F1-F8 as one frame aligned to df's index."""
    L = df[C.LEN_COLS].to_numpy(dtype=np.float64)
    T = df[C.TIME_COLS].to_numpy(dtype=np.float64)
    D = np.diff(T, axis=1)  # inter-arrival times d_1..d_4
    duration = T[:, -1]
    n_rows, n_pk = L.shape

    f: dict[str, np.ndarray] = {}

    # -- F1: raw and log ---------------------------------------------------
    for i in range(n_pk):
        f[f"len_{i}"] = L[:, i]
        f[f"log_len_{i}"] = np.log1p(L[:, i])
        f[f"t_{i}"] = T[:, i]
        f[f"log_t_{i}"] = np.log1p(T[:, i])

    # -- F2: inter-arrival -------------------------------------------------
    for i in range(n_pk - 1):
        f[f"iat_{i+1}"] = D[:, i]
        f[f"log_iat_{i+1}"] = np.log1p(D[:, i])
        f[f"iat_frac_{i+1}"] = D[:, i] / (duration + EPS)
        f[f"tfrac_{i+1}"] = T[:, i + 1] / (duration + EPS)
    f.update(_stats(D, "iat"))
    f.update(_stats(np.log1p(D), "logiat"))
    f["duration"] = duration
    f["log_duration"] = np.log1p(duration)

    # -- F3: length distribution ------------------------------------------
    f.update(_stats(L, "len"))
    n_distinct = np.array([len(set(row)) for row in L], dtype=np.float64)
    f["len_n_distinct"] = n_distinct
    f["len_all_equal"] = (n_distinct == 1).astype(np.float64)
    f["len_argmax"] = L.argmax(axis=1).astype(np.float64)
    f["len_argmin"] = L.argmin(axis=1).astype(np.float64)
    # Normalised Shannon entropy of the length multiset: 0 for a constant-size
    # keepalive flow, 1 when all five packets differ.
    ent = np.empty(n_rows)
    for r, row in enumerate(L):
        _, counts = np.unique(row, return_counts=True)
        p = counts / counts.sum()
        ent[r] = -(p * np.log(p)).sum() / np.log(n_pk)
    f["len_entropy"] = ent

    # -- F4: sequence shape ------------------------------------------------
    for i in range(n_pk):
        for j in range(i + 1, n_pk):
            f[f"len_diff_{i}{j}"] = L[:, j] - L[:, i]
            f[f"len_ratio_{i}{j}"] = L[:, j] / (L[:, i] + EPS)
    dL = np.diff(L, axis=1)
    for i in range(n_pk - 1):
        f[f"len_step_{i+1}"] = dL[:, i]
        f[f"len_step_sign_{i+1}"] = np.sign(dL[:, i])
    f["len_n_increase"] = (dL > 0).sum(axis=1).astype(np.float64)
    f["len_n_decrease"] = (dL < 0).sum(axis=1).astype(np.float64)
    f["len_n_equal_step"] = (dL == 0).sum(axis=1).astype(np.float64)
    f["len_longest_monotone"] = _longest_monotone_run(dL)
    lmax = L.max(axis=1)
    for i in range(n_pk):
        f[f"len_over_first_{i}"] = L[:, i] / (L[:, 0] + EPS)
        f[f"len_over_max_{i}"] = L[:, i] / (lmax + EPS)
    idx = np.arange(n_pk, dtype=np.float64)
    slope, r2 = _linfit(np.tile(idx, (n_rows, 1)), L)
    f["len_trend_slope"] = slope
    f["len_trend_r2"] = r2
    cum_bytes = np.cumsum(L, axis=1)
    slope_b, r2_b = _linfit(T, cum_bytes)
    f["bytes_time_slope"] = slope_b
    f["bytes_time_r2"] = r2_b

    # -- F5: protocol semantics -------------------------------------------
    for name, (lo, hi) in {
        "tiny": C.BAND_TINY,
        "audio": C.BAND_AUDIO,
        "mid": C.BAND_MID,
        "mtu": C.BAND_MTU,
    }.items():
        inband = ((L >= lo) & (L < hi)).astype(np.float64)
        f[f"n_{name}"] = inband.sum(axis=1)
        f[f"frac_{name}"] = inband.mean(axis=1)
        f[f"first_is_{name}"] = inband[:, 0]
    f["n_len47"] = (L == 47).sum(axis=1).astype(np.float64)
    f["is_constant_keepalive"] = ((n_distinct == 1) & (L[:, 0] < 100)).astype(np.float64)
    for m in (4, 8, 16):
        res = L % m
        for i in range(n_pk):
            f[f"len_mod{m}_{i}"] = res[:, i]
        f[f"len_mod{m}_n_zero"] = (res == 0).sum(axis=1).astype(np.float64)
        f[f"len_mod{m}_n_distinct"] = np.array(
            [len(set(row)) for row in res], dtype=np.float64
        )
    # Payload size after removing the fixed RTP header (12 B) and, in turn, the
    # SRTP authentication tag (10 B); apps differ in which overheads they add.
    f["rtp_payload_mean"] = (L - 12).clip(min=0).mean(axis=1)
    f["srtp_payload_mean"] = (L - 22).clip(min=0).mean(axis=1)
    # The first packet of a flow is often a binding/handshake packet and
    # structurally unlike packets 1..4.
    rest_mean = L[:, 1:].mean(axis=1)
    f["first_vs_rest_diff"] = L[:, 0] - rest_mean
    f["first_vs_rest_ratio"] = L[:, 0] / (rest_mean + EPS)

    # -- F6: burst / frame-group structure ---------------------------------
    for thr in C.BURST_THRESHOLDS:
        tag = f"{int(thr * 1e6)}us"
        below = (D < thr).astype(np.float64)
        f[f"n_iat_below_{tag}"] = below.sum(axis=1)
        f[f"frac_iat_below_{tag}"] = below.mean(axis=1)
        f[f"longest_burst_{tag}"] = _longest_true_run(D < thr) + 1.0
        g_count, g_max_pk, g_max_bytes, g_gap_mean = _group_stats(L, D, thr)
        f[f"n_groups_{tag}"] = g_count
        f[f"group_max_packets_{tag}"] = g_max_pk
        f[f"group_max_bytes_{tag}"] = g_max_bytes
        f[f"group_gap_mean_{tag}"] = g_gap_mean
        f[f"packets_per_group_{tag}"] = n_pk / g_count

    # -- F7: rate and throughput -------------------------------------------
    total_bytes = L.sum(axis=1)
    f["total_bytes"] = total_bytes
    f["bytes_per_sec"] = total_bytes / (duration + EPS)
    f["packets_per_sec"] = n_pk / (duration + EPS)
    f["log_bytes_per_sec"] = np.log1p(total_bytes / (duration + EPS))
    inst_rate = L[:, 1:] / (D + EPS)
    f.update(_stats(inst_rate, "inst_rate"))
    f.update(_stats(np.log1p(inst_rate), "log_inst_rate"))

    # -- F8: spectral ------------------------------------------------------
    len_dct = dct(L, axis=1, norm="ortho")
    logiat_dct = dct(np.log1p(D), axis=1, norm="ortho")
    for i in range(len_dct.shape[1]):
        f[f"len_dct_{i}"] = len_dct[:, i]
    for i in range(logiat_dct.shape[1]):
        f[f"logiat_dct_{i}"] = logiat_dct[:, i]
    len_fft = np.abs(rfft(L, axis=1))
    for i in range(len_fft.shape[1]):
        f[f"len_fftmag_{i}"] = len_fft[:, i]

    out = pd.DataFrame(f, index=df.index)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


# ------------------------------------------------------------- helpers ----


def _linfit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row least-squares slope and R^2 of y against x."""
    xm = x.mean(axis=1, keepdims=True)
    ym = y.mean(axis=1, keepdims=True)
    xc, yc = x - xm, y - ym
    sxx = (xc * xc).sum(axis=1)
    sxy = (xc * yc).sum(axis=1)
    syy = (yc * yc).sum(axis=1)
    slope = sxy / (sxx + EPS)
    r2 = (sxy * sxy) / ((sxx * syy) + EPS)
    return slope, r2


def _longest_true_run(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape[0], dtype=np.float64)
    for r, row in enumerate(mask):
        best = run = 0
        for v in row:
            run = run + 1 if v else 0
            best = max(best, run)
        out[r] = best
    return out


def _longest_monotone_run(d: np.ndarray) -> np.ndarray:
    out = np.zeros(d.shape[0], dtype=np.float64)
    for r, row in enumerate(d):
        best = run = 0
        prev = 0.0
        for v in row:
            s = np.sign(v)
            run = run + 1 if (s != 0 and s == prev) else 1
            prev = s
            best = max(best, run)
        out[r] = best
    return out


def _group_stats(
    L: np.ndarray, D: np.ndarray, thr: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Segment each flow into frame groups split at inter-arrival gaps >= thr.

    A video frame fragmented across the path MTU arrives as one tight group of
    packets; audio paced at a 20 ms packetisation interval arrives as singleton
    groups. Group structure is therefore a direct media-type signal.
    """
    n_rows, n_pk = L.shape
    count = np.zeros(n_rows)
    max_pk = np.zeros(n_rows)
    max_bytes = np.zeros(n_rows)
    gap_mean = np.zeros(n_rows)
    for r in range(n_rows):
        groups_pk, groups_bytes, gaps = [], [], []
        cur_pk, cur_bytes = 1, L[r, 0]
        for i in range(n_pk - 1):
            if D[r, i] < thr:
                cur_pk += 1
                cur_bytes += L[r, i + 1]
            else:
                groups_pk.append(cur_pk)
                groups_bytes.append(cur_bytes)
                gaps.append(D[r, i])
                cur_pk, cur_bytes = 1, L[r, i + 1]
        groups_pk.append(cur_pk)
        groups_bytes.append(cur_bytes)
        count[r] = len(groups_pk)
        max_pk[r] = max(groups_pk)
        max_bytes[r] = max(groups_bytes)
        gap_mean[r] = float(np.mean(gaps)) if gaps else 0.0
    return count, max_pk, max_bytes, gap_mean
