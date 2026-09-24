# -*- encoding: utf-8 -*-
'''
@File :util.py
@Created-Time :2026-09-16 17:51:17
@Author  :june
@Description   : small shared helpers
@Modified-Time :2026-09-16 17:51:17
'''

import torch as th


def th2nparray(x):
    """Tensor -> CPU numpy; anything else passes through unchanged.

    Note the boundary rule: the env adapter hands the framework **tensors on
    device** (DESIGN.md §4.2), so conversions belong at the edges (logging,
    saving), not in the training path.
    """
    if isinstance(x, th.Tensor):
        return x.detach().cpu().numpy()
    return x
