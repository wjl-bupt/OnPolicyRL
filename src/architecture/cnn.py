# -*- encoding: utf-8 -*-
"""Channels-first image feature extractor for the network axis (S0).

Two conv geometries, chosen by the assembler from the spatial size:

* NATURE  (Mnih 2015) for Atari-scale frames (>= ~36 px): 8/4, 4/2, 3/1.
* MINATAR for MinAtar's 10x10 grids: a single 3x3 conv (the Nature stack
  underflows below 36 px -- its second conv would see a <1px map).

uint8 pixels are scaled by `scale` (255 for uint8 spaces, 1 for already-[0,1]
float channels like MinAtar); the flat conv output size is inferred by a dummy
forward so no geometry arithmetic is hand-maintained.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch as th
import torch.nn as nn
from torch import Tensor

from architecture.init import layer_init

# (out_channels, kernel, stride)
_ConvSpec = Sequence[tuple[int, int, int]]


def _conv_stack(obs_shape, specs: _ConvSpec, features_dim: int, ortho: bool,
                scale: float) -> tuple[nn.Sequential, nn.Sequential, int]:
    in_c = int(obs_shape[0])
    layers: list[nn.Module] = []
    for out_c, k, s in specs:
        layers += [nn.Conv2d(in_c, out_c, kernel_size=k, stride=s), nn.ReLU()]
        in_c = out_c
    layers.append(nn.Flatten())
    cnn = nn.Sequential(*layers)
    with th.no_grad():
        n_flat = cnn(th.zeros(1, *obs_shape).float() / scale).shape[1]
    linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())
    if ortho:
        for m in (*cnn, *linear):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                layer_init(m)
    return cnn, linear, features_dim


class ImageExtractor(nn.Module):
    """`obs [B, C, H, W] -> features [B, features_dim]`. Pick `specs` by size."""

    NATURE: _ConvSpec = ((32, 8, 4), (64, 4, 2), (64, 3, 1))
    MINATAR: _ConvSpec = ((16, 3, 1),)

    def __init__(self, obs_shape, features_dim: int = 512, ortho: bool = True,
                 scale: float = 1.0, specs: _ConvSpec | None = None):
        super().__init__()
        self.scale = float(scale)
        self.cnn, self.linear, self.out_dim = _conv_stack(
            obs_shape, specs or self.NATURE, features_dim, ortho, self.scale)

    def forward(self, obs: Tensor) -> Tensor:
        return self.linear(self.cnn(obs.float() / self.scale))
