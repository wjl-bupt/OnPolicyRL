# -*- encoding: utf-8 -*-
"""MLP feature extractor for the network axis (S0).

`obs -> features`: a stack of Linear+activation whose last width is `out_dim`,
which the actor/value heads read. Vector obs only -- image obs go through
architecture/cnn.py, dict obs need a combined extractor (not wired yet).

Not normalized here: scaling vector observations is a normalizer's job (S0,
config), not the extractor's. Obs are only cast to float.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch.nn as nn
from torch import Tensor

from architecture.init import layer_init


class MlpExtractor(nn.Module):
    """`in_dim -> net_arch` hidden stack; `out_dim` is the final hidden width.

    Example::

        ex = MlpExtractor(4, (64, 64))     # ex.out_dim == 64
    """

    def __init__(
        self,
        in_dim: int,
        net_arch: Sequence[int] = (64, 64),
        activation: type[nn.Module] = nn.Tanh,
        ortho: bool = True,
    ):
        super().__init__()
        sizes = [int(in_dim), *[int(h) for h in net_arch]]
        layers: list[nn.Module] = []
        for i, o in zip(sizes[:-1], sizes[1:]):
            lin = nn.Linear(i, o)
            if ortho:
                layer_init(lin, gain=np.sqrt(2))
            layers += [lin, activation()]
        self.net = nn.Sequential(*layers)
        self.out_dim = sizes[-1]

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs.float())
