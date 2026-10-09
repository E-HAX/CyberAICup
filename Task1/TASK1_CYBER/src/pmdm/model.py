"""Siamese change detector with CenterNet-style heads.

Shared-weight encoder over both streams, concat + absolute-difference fusion at
every scale (the U-Net SiamDiff/SiamConc shape that stays competitive with much
heavier change-detection transformers), U-Net decoder down to output stride 2
because the targets are median 22 px wide, and CenterNet heads so box corners
come from sub-pixel regression rather than from thresholded blobs.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import IN_CHANS_PER_STREAM


def conv_bn(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class Fuse(nn.Module):
    """concat(a, b, |a-b|) -> conv."""

    def __init__(self, ch: int, out: int):
        super().__init__()
        self.block = nn.Sequential(conv_bn(ch * 3, out, 1), conv_bn(out, out, 3))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.block(torch.cat([a, b, (a - b).abs()], 1))


class UpBlock(nn.Module):
    def __init__(self, cin: int, skip: int, out: int):
        super().__init__()
        self.block = nn.Sequential(conv_bn(cin + skip, out, 3), conv_bn(out, out, 3))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], 1))


class SiamCenterNet(nn.Module):
    def __init__(self, backbone: str = "convnext_tiny", pretrained: bool = True,
                 decoder_ch: tuple[int, ...] = (192, 128, 96, 64, 48)):
        super().__init__()
        self.encoder = timm.create_model(
            backbone, features_only=True, pretrained=pretrained,
            in_chans=IN_CHANS_PER_STREAM,
        )
        chs = self.encoder.feature_info.channels()          # strides 4, 8, 16, 32
        self.fuse = nn.ModuleList([Fuse(c, min(c, 256)) for c in chs])
        fused = [min(c, 256) for c in chs]

        # Stride-2 stem straight off the stacked input, so the decoder can reach
        # output stride 2 without inventing detail.
        self.stem2 = nn.Sequential(
            conv_bn(IN_CHANS_PER_STREAM * 2 + 2, 32, 3),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.up3 = UpBlock(fused[3], fused[2], decoder_ch[0])
        self.up2 = UpBlock(decoder_ch[0], fused[1], decoder_ch[1])
        self.up1 = UpBlock(decoder_ch[1], fused[0], decoder_ch[2])
        self.up0 = UpBlock(decoder_ch[2], 32, decoder_ch[3])

        head_ch = decoder_ch[3]
        self.hm = nn.Sequential(conv_bn(head_ch, head_ch), nn.Conv2d(head_ch, 1, 1))
        self.wh = nn.Sequential(conv_bn(head_ch, head_ch), nn.Conv2d(head_ch, 2, 1))
        self.off = nn.Sequential(conv_bn(head_ch, head_ch), nn.Conv2d(head_ch, 2, 1))
        self.seg = nn.Sequential(conv_bn(head_ch, head_ch), nn.Conv2d(head_ch, 1, 1))
        nn.init.constant_(self.hm[-1].bias, -4.0)   # rare positives
        nn.init.constant_(self.seg[-1].bias, -4.0)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
        signed = (b[:, 3:4] - a[:, 3:4])                      # photo ink minus template ink
        absdiff = (b[:, :3] - a[:, :3]).abs().mean(1, keepdim=True)
        stem = self.stem2(torch.cat([a, b, signed, absdiff], 1))

        fa = self.encoder(a)
        fb = self.encoder(b)
        f = [self.fuse[i](fa[i], fb[i]) for i in range(len(fa))]

        x = self.up3(f[3], f[2])
        x = self.up2(x, f[1])
        x = self.up1(x, f[0])
        x = self.up0(x, stem)
        return {"hm": self.hm(x), "wh": self.wh(x), "off": self.off(x), "seg": self.seg(x)}
