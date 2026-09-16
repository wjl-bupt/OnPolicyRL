# -*- encoding: utf-8 -*-
'''
@File :actor_critic.py
@Created-Time :2026-09-16 14:42:39
@Author  :june
@Description   : Actor-Critic Archtecture
@Modified-Time : 2026-09-16 14:42:39
'''

import torch as th
import torch.nn as nn

class ShareActorCritic(nn.Module):
    def __init__(self, ):
        pass

class ActorCritic(nn.Module):
    def __init__(self, use_share = True):
        if use_share:
            ShareActorCritic()
        else:
            pass
    
    def forward(self, x):
        pass

