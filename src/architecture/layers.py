# -*- encoding: utf-8 -*-
'''
@File :layers.py
@Created-Time :2026-09-16 11:39:15
@Author  :june
@Description   : base layers implement
@Modified-Time : 2026-09-16 11:39:15
'''

import torch as th
import torch.nn as nn
from typing import List

class MlpLayers(nn.Module):
    def __init__(
        self, 
        in_d : int,
        out_d : int, 
        hidden_dims: List | int = 256,
        activation_fn: type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
        use_bias: bool = True,
        init_func: None = None,
        pre_linear_modules: list[type[nn.Module]] | None = None,
        post_linear_modules: list[type[nn.Module]] | None = None,
    ):
        self.in_d = in_d
        self.out_d = out_d
        self.use_bias = use_bias
        self.init_func = init_func
        self.activation_fn = activation_fn
        self.squash_output = squash_output
        self.hidden_dims = hidden_dims if isinstance(list, hidden_dims) else [hidden_dims]
        self.net_arch = [self.in_d] + self.hidden_dims + [self.out_d]
        self.layers = []

        # Hidden layers
        for in_d, out_d in zip(self.net_arch[:-2], self.net_arch[1:-1]):
            self.layers.extend(pre_linear_modules)
            self.layers.append(
                nn.Linear(in_d, out_d, bias=self.use_bias)
            )
            self.layers.extend(post_linear_modules)
            self.layers.append(self.activation_fn)
        # Output layer
        self.layers.append(
            nn.Linear(
                self.net_arch[-2],
                self.net_arch[-1],
                bias=self.use_bias,
            )
        )
        if self.squash_output:
            self.layers.append(nn.Tanh())

        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.layers(x)
        
