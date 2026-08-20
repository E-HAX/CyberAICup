"""Self-supervised pretraining of PacketFormer on the auxiliary MIRAGE corpus.

Three objectives run together:

masked packet modelling
    15% of packet tokens are replaced by a learned mask token and the model
    reconstructs the quantised payload size (cross-entropy) and the log gap
    (Huber). This is the ET-BERT / YaTC idea adapted to a per-packet token.

contrastive
    Two traffic-plausible augmentations of the same flow are pulled together
    and other flows in the batch pushed apart, which is what makes the
    representation tolerant of the pacing differences between a mobile capture
    and the lab Wi-Fi capture used by the competition data.

domain adversarial
    A gradient-reversal head tries to tell auxiliary flows from competition
    flows. Because the encoder is trained to defeat it, the representation is
    pushed toward features the two corpora share - the transfer would otherwise
    be dominated by capture-environment artefacts.

Only unlabelled competition flows are used for the domain head; no competition
label is touched anywhere in this file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
import io_utils as IO
from transformer.dataset_rtc import encode_flow, size_bins
from transformer.mirage_stream import load_shards
from transformer.model import PacketFormer, cosine_schedule, grad_reverse

CKPT_DIR = C.CACHE / "pretrain"

DEFAULT_CFG = {
    "d_model": 96,
    "n_layers": 3,
    "n_heads": 4,
    "dim_ff": 192,
    "dropout": 0.1,
    "steps": 8000,
    "batch_size": 256,
    "lr": 3e-4,
    "weight_decay": 0.02,
    "mask_ratio": 0.15,
    "w_mlm_bin": 1.0,
    "w_mlm_iat": 0.5,
    "w_contrastive": 0.5,
    "w_domain": 0.1,
    "temperature": 0.2,
    "max_packets": 20,
    "max_flows": 400000,
    "shards": None,
    "seed": C.SEED,
}


def _times_from_gaps(gaps: np.ndarray) -> np.ndarray:
    return np.cumsum(gaps, axis=1)


def _augment(lengths: np.ndarray, gaps: np.ndarray, rng: np.random.RandomState):
    g = gaps * rng.uniform(0.9, 1.1, size=(len(gaps), 1))
    L = np.maximum(1.0, lengths + rng.uniform(-2, 2, size=lengths.shape))
    return L, g


def _encode(lengths: np.ndarray, gaps: np.ndarray):
    times = _times_from_gaps(gaps)
    cont = encode_flow(lengths, times)
    bins = size_bins(lengths)
    return cont, bins


class Corpus:
    """Auxiliary flows plus unlabelled competition flows for the domain head."""

    def __init__(self, cfg: dict):
        data = load_shards(limit=cfg["shards"])
        n = min(len(data["lengths"]), cfg["max_flows"])
        rng = np.random.RandomState(cfg["seed"])
        idx = rng.permutation(len(data["lengths"]))[:n]
        self.lengths = data["lengths"][idx].astype(np.float64)
        self.gaps = data["gaps"][idx].astype(np.float64)
        self.mask = data["mask"][idx]
        self.labels = data["label_code"][idx]
        self.label_names = data["label_names"]

        train, test = IO.load_train(), IO.load_test()
        rtc_L = np.vstack(
            [train[C.LEN_COLS].to_numpy(float), test[C.LEN_COLS].to_numpy(float)]
        )
        rtc_T = np.vstack(
            [train[C.TIME_COLS].to_numpy(float), test[C.TIME_COLS].to_numpy(float)]
        )
        pad = self.lengths.shape[1] - rtc_L.shape[1]
        self.rtc_lengths = np.pad(rtc_L, ((0, 0), (0, max(0, pad))))
        rtc_gaps = np.zeros_like(rtc_T)
        rtc_gaps[:, 1:] = np.diff(rtc_T, axis=1)
        self.rtc_gaps = np.pad(rtc_gaps, ((0, 0), (0, max(0, pad))))
        self.rtc_mask = np.zeros_like(self.rtc_lengths, dtype=bool)
        self.rtc_mask[:, : rtc_L.shape[1]] = True

    def batch(self, rng: np.random.RandomState, size: int):
        i = rng.randint(0, len(self.lengths), size)
        j = rng.randint(0, len(self.rtc_lengths), max(8, size // 4))
        return i, j


def pretrain(cfg_overrides: dict | None = None) -> dict:
    cfg = {**DEFAULT_CFG, **(cfg_overrides or {})}
    torch.manual_seed(cfg["seed"])
    rng = np.random.RandomState(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus = Corpus(cfg)
    model = PacketFormer(
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dim_ff=cfg["dim_ff"],
        dropout=cfg["dropout"],
        n_classes=10,
        max_len=cfg["max_packets"] + 4,
    ).to(device)
    domain_head = nn.Sequential(nn.Linear(cfg["d_model"], 64), nn.GELU(), nn.Linear(64, 2)).to(device)

    params = list(model.parameters()) + list(domain_head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    t0 = time.time()

    for step in range(cfg["steps"]):
        i, j = corpus.batch(rng, cfg["batch_size"])
        L, G, M = corpus.lengths[i].copy(), corpus.gaps[i].copy(), corpus.mask[i].copy()

        # Random truncation, so the encoder is equally at home on the five
        # packets the competition provides and on longer auxiliary flows.
        keep_n = rng.choice([C.N_PACKETS, 8, 12, cfg["max_packets"]], size=len(L))
        cut = np.arange(L.shape[1])[None, :] >= keep_n[:, None]
        L[cut] = 0.0
        G[cut] = 0.0
        M[cut] = False

        # Two augmented views for the contrastive term.
        L1, G1 = _augment(L, G, rng)
        L2, G2 = _augment(L, G, rng)
        cont1, bins1 = _encode(L1, G1)
        cont2, bins2 = _encode(L2, G2)
        pad = torch.from_numpy(~M).to(device)
        cont1_t = torch.from_numpy(cont1).to(device)
        bins1_t = torch.from_numpy(bins1).to(device)
        cont2_t = torch.from_numpy(cont2).to(device)
        bins2_t = torch.from_numpy(bins2).to(device)

        # -- masked packet modelling ---------------------------------------
        mask_sel = torch.rand(bins1_t.shape, device=device) < cfg["mask_ratio"]
        mask_sel = mask_sel & torch.from_numpy(M).to(device)
        if mask_sel.sum() == 0:
            mask_sel[:, 0] = torch.from_numpy(M[:, 0]).to(device)
        bin_logits, iat_pred = model.masked_forward(cont1_t, bins1_t, mask_sel, pad)
        target_bins = bins1_t[mask_sel]
        loss_bin = F.cross_entropy(bin_logits[mask_sel], target_bins)
        target_iat = torch.from_numpy(np.log1p(G1 * 1e3).astype(np.float32)).to(device)
        loss_iat = F.huber_loss(iat_pred[mask_sel], target_iat[mask_sel])

        # -- contrastive ----------------------------------------------------
        z1 = model.project(cont1_t, bins1_t, pad)
        z2 = model.project(cont2_t, bins2_t, pad)
        loss_con = _nt_xent(z1, z2, cfg["temperature"])

        # -- domain adversarial --------------------------------------------
        # Both sides are compared as five-packet views. Feeding twenty-packet
        # auxiliary flows against five-packet competition flows would let the
        # discriminator win on sequence length alone, which teaches the encoder
        # nothing about the actual capture-environment gap.
        rtc_cont, rtc_bins = _encode(corpus.rtc_lengths[j], corpus.rtc_gaps[j])
        rtc_pad = torch.from_numpy(~corpus.rtc_mask[j]).to(device)
        aux5_L = corpus.lengths[i].copy()
        aux5_G = corpus.gaps[i].copy()
        aux5_L[:, C.N_PACKETS :] = 0.0
        aux5_G[:, C.N_PACKETS :] = 0.0
        aux5_mask = corpus.mask[i].copy()
        aux5_mask[:, C.N_PACKETS :] = False
        aux5_cont, aux5_bins = _encode(aux5_L, aux5_G)
        h_aux = model.encode(
            torch.from_numpy(aux5_cont).to(device),
            torch.from_numpy(aux5_bins).to(device),
            torch.from_numpy(~aux5_mask).to(device),
        )
        h_rtc = model.encode(
            torch.from_numpy(rtc_cont).to(device), torch.from_numpy(rtc_bins).to(device), rtc_pad
        )
        h = torch.cat([h_aux, h_rtc])
        d_target = torch.cat(
            [torch.zeros(len(h_aux), dtype=torch.long), torch.ones(len(h_rtc), dtype=torch.long)]
        ).to(device)
        lam = 2.0 / (1.0 + np.exp(-5 * step / cfg["steps"])) - 1.0
        loss_dom = F.cross_entropy(domain_head(grad_reverse(h, lam)), d_target)

        loss = (
            cfg["w_mlm_bin"] * loss_bin
            + cfg["w_mlm_iat"] * loss_iat
            + cfg["w_contrastive"] * loss_con
            + cfg["w_domain"] * loss_dom
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g in opt.param_groups:
            g["lr"] = cfg["lr"] * cosine_schedule(step, cfg["steps"], warmup=cfg["steps"] // 20)
        opt.step()

        if step % 250 == 0 or step == cfg["steps"] - 1:
            entry = {
                "step": step,
                "loss": float(loss.item()),
                "mlm_bin": float(loss_bin.item()),
                "mlm_iat": float(loss_iat.item()),
                "contrastive": float(loss_con.item()),
                "domain": float(loss_dom.item()),
            }
            history.append(entry)
            print(json.dumps(entry), flush=True)
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "step": step},
                CKPT_DIR / "encoder.pt",
            )

    ckpt = CKPT_DIR / "encoder.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": cfg["steps"]}, ckpt)
    (CKPT_DIR / "history.json").write_text(json.dumps(history, indent=2))
    return {
        "checkpoint": str(ckpt),
        "aux_flows": int(len(corpus.lengths)),
        "aux_labels": int(len(corpus.label_names)),
        "steps": cfg["steps"],
        "seconds": round(time.time() - t0, 1),
        "final": history[-1] if history else None,
    }


def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z = torch.cat([z1, z2])
    sim = z @ z.t() / temperature
    n = len(z1)
    sim.fill_diagonal_(-1e9)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def latest_checkpoint() -> str | None:
    p = CKPT_DIR / "encoder.pt"
    return str(p) if p.exists() else None
