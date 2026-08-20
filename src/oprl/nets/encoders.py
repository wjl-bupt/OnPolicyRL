"""Default network components -- optional defaults, not a required path.

Any `nn.Module` satisfying the Policy protocol works directly (DESIGN.md §4.5,
level 3); importing `oprl.nets` is entirely optional.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..registry import register


def orthogonal_init(layer: nn.Module, std: float = np.sqrt(2), bias: float = 0.0):
    """Orthogonal initialization -- one of PPO's known-important details."""
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias)
    return layer


_ACT = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU}


@register("encoder", "mlp")
class MLPEncoder(nn.Module):
    """The contract is just two things: an `out_dim` attribute and `forward`."""

    def __init__(
        self,
        in_dim: int,
        hidden: tuple[int, ...] = (64, 64),
        activation: str = "tanh",
    ):
        super().__init__()
        act = _ACT[activation]
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [orthogonal_init(nn.Linear(d, h)), act()]
            d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


@register("encoder", "cnn")
class CNNEncoder(nn.Module):
    """Nature-CNN style, for Atari / MinAtar. Input is [B, C, H, W]."""

    def __init__(self, in_shape: tuple[int, ...], out_dim: int = 512, scale: bool = True):
        super().__init__()
        c, h, w = in_shape
        self.scale = scale
        if h <= 16:  # MinAtar scale (10x10): use a small kernel.
            self.conv = nn.Sequential(
                orthogonal_init(nn.Conv2d(c, 16, 3, stride=1)), nn.ReLU(), nn.Flatten()
            )
        else:  # Atari 84x84
            self.conv = nn.Sequential(
                orthogonal_init(nn.Conv2d(c, 32, 8, stride=4)), nn.ReLU(),
                orthogonal_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
                orthogonal_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
                nn.Flatten(),
            )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, c, h, w)).shape[1]
        self.head = nn.Sequential(orthogonal_init(nn.Linear(n_flat, out_dim)), nn.ReLU())
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.scale and x.max() > 1.0:
            # Only 8-bit pixel encodings need /255. MinAtar is bool (0/1), and dividing
            # it would collapse the signal to ~zero before the first layer.
            x = x / 255.0
        return self.head(self.conv(x))
