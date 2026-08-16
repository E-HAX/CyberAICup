"""CenterNet focal loss, masked L1 and Dice."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_focal(pred_logits: torch.Tensor, target: torch.Tensor,
                   alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    # Always in float32: under bfloat16 autocast, 1 - 1e-4 rounds to exactly 1.0,
    # so log(1 - pred) becomes -inf and the zero-weighted term becomes NaN.
    pred_logits = pred_logits.float()
    target = target.float()
    pred = torch.sigmoid(pred_logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1.0 - pos
    neg_weight = torch.pow(1.0 - target, beta)
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weight * neg
    n = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.expand_as(pred)
    return (F.l1_loss(pred * m, target * m, reduction="sum") / m.sum().clamp(min=1.0))


def dice_bce(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_logits = pred_logits.float()
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    p = torch.sigmoid(pred_logits)
    num = 2 * (p * target).sum() + 1.0
    den = p.sum() + target.sum() + 1.0
    return bce + (1 - num / den)


def total_loss(out: dict, batch: dict, w_hm: float = 1.0, w_wh: float = 0.2,
               w_off: float = 1.0, w_seg: float = 0.5):
    losses = {
        "hm": w_hm * gaussian_focal(out["hm"], batch["hm"]),
        "wh": w_wh * masked_l1(out["wh"], batch["wh"], batch["reg"]),
        "off": w_off * masked_l1(out["off"], batch["off"], batch["reg"]),
        "seg": w_seg * dice_bce(out["seg"], batch["seg"]),
    }
    losses["total"] = sum(losses.values())
    return losses
