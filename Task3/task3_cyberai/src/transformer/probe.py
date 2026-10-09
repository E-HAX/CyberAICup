"""Customized linear probes on the frozen pretrained encoder - feature block F12.

The probes are the mechanism that turns the auxiliary corpus into new columns
for the tree models. Each probe is trained on a target taken from MIRAGE, never
from the competition labels, so applying it to the competition flows creates
genuinely new information rather than a leak:

    P1  application-family probe   which MIRAGE application does this flow
                                   resemble
    P2  media-modality probe       a "video-likeness" score learned from an
                                   auxiliary proxy for interactive video
    P3  bitrate / pacing probes    regression heads whose *residuals* say how
                                   unlike the auxiliary corpus a flow paces
    P4  transport-fingerprint      which padding quantum the payload sizes obey
    P5  embedding compression      principal components of the frozen
                                   representation, label-free

All auxiliary flows are truncated to the same five packets the competition
provides, so a probe fitted on MIRAGE is being applied to inputs of exactly the
shape it was trained on.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

import config as C
import io_utils as IO
from transformer.dataset_rtc import encode_flow, size_bins
from transformer.mirage_stream import load_shards
from transformer.model import PacketFormer

N_ALIGN_PACKETS = C.N_PACKETS  # probes see exactly five packets, like the task
TOP_APPS = 20
PCA_DIM = 16
MAX_AUX = 120000


def _load_encoder(ckpt_path: str) -> tuple[PacketFormer, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    model = PacketFormer(
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dim_ff=cfg["dim_ff"],
        dropout=0.0,
        n_classes=10,
        max_len=cfg.get("max_packets", 20) + 4,
    )
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


@torch.no_grad()
def _embed(model: PacketFormer, lengths: np.ndarray, gaps: np.ndarray, device, bs: int = 4096):
    times = np.cumsum(gaps, axis=1)
    out = []
    for i in range(0, len(lengths), bs):
        L = lengths[i : i + bs]
        T = times[i : i + bs]
        cont = torch.from_numpy(encode_flow(L, T)).to(device)
        bins = torch.from_numpy(size_bins(L)).to(device)
        out.append(model.encode(cont, bins).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def _aux_targets(lengths: np.ndarray, gaps: np.ndarray) -> dict[str, np.ndarray]:
    """Auxiliary supervision derived from the flows themselves, not from labels."""
    span = np.maximum(gaps.sum(axis=1), 1e-9)
    total_bytes = lengths.sum(axis=1)
    frac_mtu = (lengths >= 1100).mean(axis=1)
    sub_ms = (gaps[:, 1:] < 1e-3).mean(axis=1)
    log_bitrate = np.log1p(total_bytes / span)
    log_mean_gap = np.log1p(gaps[:, 1:].mean(axis=1) * 1e3)

    # A flow looks like interactive video when large frames are fragmented
    # across the path MTU and those fragments arrive back to back.
    video_like = ((frac_mtu > 0.2) & (sub_ms > 0.4)).astype(np.int64)

    residues = np.stack(
        [(lengths % m == 0).mean(axis=1) for m in (2, 4, 8, 16)], axis=1
    )
    padding_class = residues.argmax(axis=1)
    return {
        "log_bitrate": log_bitrate,
        "log_mean_gap": log_mean_gap,
        "frac_mtu": frac_mtu,
        "video_like": video_like,
        "padding_class": padding_class,
    }


def build_probe_features(pretrained: str, max_aux: int = MAX_AUX) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = _load_encoder(pretrained)
    model.to(device)

    # -- auxiliary side, truncated to the competition's five packets ---------
    aux = load_shards()
    n = min(len(aux["lengths"]), max_aux)
    rng = np.random.RandomState(C.SEED)
    sel = rng.permutation(len(aux["lengths"]))[:n]
    keep = aux["mask"][sel][:, :N_ALIGN_PACKETS].all(axis=1)
    sel = sel[keep]
    aux_L = aux["lengths"][sel][:, :N_ALIGN_PACKETS].astype(np.float64)
    aux_G = aux["gaps"][sel][:, :N_ALIGN_PACKETS].astype(np.float64)
    aux_G[:, 0] = 0.0
    aux_labels = aux["label_code"][sel]
    aux_names = aux["label_names"]

    E_aux = _embed(model, aux_L, aux_G, device)
    targets = _aux_targets(aux_L, aux_G)

    # -- competition side ---------------------------------------------------
    train, test = IO.load_train(), IO.load_test()
    rtc_L = np.vstack([train[C.LEN_COLS].to_numpy(float), test[C.LEN_COLS].to_numpy(float)])
    rtc_T = np.vstack([train[C.TIME_COLS].to_numpy(float), test[C.TIME_COLS].to_numpy(float)])
    rtc_G = np.zeros_like(rtc_T)
    rtc_G[:, 1:] = np.diff(rtc_T, axis=1)
    E_rtc = _embed(model, rtc_L, rtc_G, device)

    scaler = StandardScaler().fit(E_aux)
    A, R = scaler.transform(E_aux), scaler.transform(E_rtc)
    feats: dict[str, np.ndarray] = {}
    report: dict = {"aux_flows": int(len(A)), "checkpoint": pretrained}

    # -- P1: application-family probe ---------------------------------------
    counts = np.bincount(aux_labels, minlength=len(aux_names))
    top = np.argsort(counts)[::-1][:TOP_APPS]
    keep_rows = np.isin(aux_labels, top)
    if keep_rows.sum() > 200:
        remap = {v: i for i, v in enumerate(top)}
        y_app = np.array([remap[v] for v in aux_labels[keep_rows]])
        p1 = LogisticRegression(max_iter=2000, C=1.0, n_jobs=1).fit(A[keep_rows], y_app)
        proba = p1.predict_proba(R)
        for i, cls in enumerate(p1.classes_):
            feats[f"probe_app_{aux_names[top[cls]]}"[:60]] = proba[:, i]
        feats["probe_app_top1"] = proba.max(axis=1)
        feats["probe_app_entropy"] = -(proba * np.log(proba + 1e-9)).sum(axis=1)
        srt = np.sort(proba, axis=1)
        feats["probe_app_margin"] = srt[:, -1] - srt[:, -2]
        report["p1_train_acc"] = float(p1.score(A[keep_rows], y_app))
        report["p1_classes"] = [str(aux_names[top[c]]) for c in p1.classes_]

    # -- P2: media-modality probe -------------------------------------------
    y_vid = targets["video_like"]
    if 0 < y_vid.mean() < 1:
        p2 = LogisticRegression(max_iter=2000, C=1.0).fit(A, y_vid)
        feats["probe_video_like"] = p2.predict_proba(R)[:, 1]
        report["p2_train_acc"] = float(p2.score(A, y_vid))

    # -- P3: bitrate and pacing regressions, predictions and residuals -------
    observed = _aux_targets(rtc_L, rtc_G)
    for name in ("log_bitrate", "log_mean_gap", "frac_mtu"):
        reg = Ridge(alpha=1.0).fit(A, targets[name])
        pred = reg.predict(R)
        feats[f"probe_pred_{name}"] = pred
        # The residual is the informative part: it measures how far this flow
        # sits from what the auxiliary corpus would expect of it.
        feats[f"probe_resid_{name}"] = observed[name] - pred
        report[f"p3_{name}_r2"] = float(reg.score(A, targets[name]))

    # -- P4: transport fingerprint ------------------------------------------
    y_pad = targets["padding_class"]
    if len(np.unique(y_pad)) > 1:
        p4 = LogisticRegression(max_iter=2000, C=1.0).fit(A, y_pad)
        proba = p4.predict_proba(R)
        for i, cls in enumerate(p4.classes_):
            feats[f"probe_pad_mod{2 ** (cls + 1)}"] = proba[:, i]
        report["p4_train_acc"] = float(p4.score(A, y_pad))

    # -- P5: label-free embedding compression --------------------------------
    pca = PCA(n_components=min(PCA_DIM, R.shape[1]), random_state=C.SEED).fit(R)
    Z = pca.transform(R)
    for i in range(Z.shape[1]):
        feats[f"probe_emb_pc{i}"] = Z[:, i]
    report["p5_explained_variance"] = float(pca.explained_variance_ratio_.sum())

    frame = pd.DataFrame(feats).astype(np.float32)
    n_train = len(train)
    frame.iloc[:n_train].reset_index(drop=True).to_parquet(
        C.FEATURES_DIR / "probes_train.parquet"
    )
    frame.iloc[n_train:].reset_index(drop=True).to_parquet(
        C.FEATURES_DIR / "probes_test.parquet"
    )
    report["n_probe_features"] = int(frame.shape[1])
    (C.REPORTS_DIR / "probe_report.json").write_text(json.dumps(report, indent=2, default=float))
    return report
