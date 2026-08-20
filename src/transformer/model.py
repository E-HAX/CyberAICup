"""PacketFormer: a small transformer encoder over a packet sequence.

A flow is a very short sequence - five packets in the competition data, up to
twenty in the auxiliary corpus - so the model is deliberately small and heavily
regularised. Each packet becomes one token built from continuous descriptors
(size, inter-arrival, position in the flow) plus a learned embedding of the
quantised packet size, which is what makes the discrete, codec-driven size
modes visible to attention.

The same encoder serves three roles: supervised classifier on the competition
data, self-supervised model during pretraining on the auxiliary corpus, and
frozen feature extractor for the linear probes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

N_CONT = 8  # continuous descriptors per packet, see dataset_rtc.encode_flow
N_LEN_BINS = 24


class PacketFormer(nn.Module):
    def __init__(
        self,
        d_model: int = 96,
        n_layers: int = 3,
        n_heads: int = 4,
        dim_ff: int = 192,
        dropout: float = 0.15,
        n_classes: int = 10,
        max_len: int = 24,
        n_len_bins: int = N_LEN_BINS,
        n_cont: int = N_CONT,
    ):
        super().__init__()
        self.d_model = d_model
        self.cont_proj = nn.Linear(n_cont, d_model)
        self.bin_emb = nn.Embedding(n_len_bins + 1, d_model)
        self.pos_emb = nn.Embedding(max_len + 1, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

        # Self-supervised heads, unused when the model is only classifying.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.mlm_bin_head = nn.Linear(d_model, n_len_bins + 1)
        self.mlm_iat_head = nn.Linear(d_model, 1)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 64)
        )

    # ----------------------------------------------------------- encode ----

    def embed_tokens(self, cont: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        x = self.cont_proj(cont) + self.bin_emb(bins)
        b, t, _ = x.shape
        pos = torch.arange(1, t + 1, device=x.device).unsqueeze(0).expand(b, t)
        x = x + self.pos_emb(pos)
        return x

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = tokens.shape[0]
        cls = self.cls.expand(b, 1, self.d_model) + self.pos_emb(
            torch.zeros(b, 1, dtype=torch.long, device=tokens.device)
        )
        x = torch.cat([cls, tokens], dim=1)
        x = self.input_drop(self.input_norm(x))
        if pad_mask is not None:
            pad_mask = torch.cat(
                [torch.zeros(b, 1, dtype=torch.bool, device=tokens.device), pad_mask], dim=1
            )
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.out_norm(h)

    def encode(
        self, cont: torch.Tensor, bins: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sequence representation taken from the [CLS] position."""
        h = self.forward_tokens(self.embed_tokens(cont, bins), pad_mask)
        return h[:, 0]

    # ------------------------------------------------------- objectives ----

    def forward(
        self,
        cont: torch.Tensor,
        bins: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        mixup_lam: float | None = None,
        mixup_perm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.embed_tokens(cont, bins)
        if mixup_lam is not None and mixup_perm is not None:
            # Mixup in embedding space keeps the discrete size bins meaningful
            # while still smoothing the decision boundary.
            tokens = mixup_lam * tokens + (1.0 - mixup_lam) * tokens[mixup_perm]
        h = self.forward_tokens(tokens, pad_mask)
        return self.head(h[:, 0])

    def masked_forward(
        self,
        cont: torch.Tensor,
        bins: torch.Tensor,
        mask: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked packet modelling: predict the size bin and the log-gap."""
        tokens = self.embed_tokens(cont, bins)
        tokens = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens)
        h = self.forward_tokens(tokens, pad_mask)[:, 1:]
        return self.mlm_bin_head(h), self.mlm_iat_head(h).squeeze(-1)

    def project(
        self, cont: torch.Tensor, bins: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Contrastive projection of the sequence representation."""
        z = self.proj_head(self.encode(cont, bins, pad_mask))
        return nn.functional.normalize(z, dim=-1)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, lambd)


def cosine_schedule(step: int, total: int, warmup: int = 0) -> float:
    if warmup and step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
