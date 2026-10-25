"""Finite-window causal attention, with optional GTrXL-style GRU gates.

Histories are explicit rollout observations, not hidden mutable module state.
This uses absolute positions and recomputation, not Transformer-XL segment memory.
"""
import math
import numpy as np
import torch
from torch import nn


class GRUGate(nn.Module):
    def __init__(self, width, bias=2.):
        super().__init__()
        self.wr=nn.Linear(width,width,bias=False);self.ur=nn.Linear(width,width,bias=False)
        self.wz=nn.Linear(width,width,bias=False);self.uz=nn.Linear(width,width,bias=False)
        self.wh=nn.Linear(width,width,bias=False);self.uh=nn.Linear(width,width,bias=False)
        self.bias=nn.Parameter(torch.full((width,),float(bias)))
    def forward(self,x,y):
        r=torch.sigmoid(self.wr(y)+self.ur(x))
        z=torch.sigmoid(self.wz(y)+self.uz(x)-self.bias)
        h=torch.tanh(self.wh(y)+self.uh(r*x))
        return (1-z)*x+z*h


class CausalBlock(nn.Module):
    def __init__(self,width,heads,expansion=4,gated=True,eps=1e-6):
        super().__init__()
        self.heads,self.head_dim=heads,width//heads
        self.norm1=nn.LayerNorm(width,eps=eps);self.norm2=nn.LayerNorm(width,eps=eps)
        self.qkv=nn.Linear(width,width*3);self.projection=nn.Linear(width,width)
        self.ff=nn.Sequential(nn.Linear(width,width*expansion),nn.ReLU(),nn.Linear(width*expansion,width))
        self.gate1=GRUGate(width) if gated else None
        self.gate2=GRUGate(width) if gated else None
    def forward(self,x):
        b,t,d=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(b,t,3,self.heads,self.head_dim).unbind(2)
        q,k,v=[a.transpose(1,2) for a in (q,k,v)]
        score=q@k.transpose(-1,-2)/math.sqrt(self.head_dim)
        future=torch.ones(t,t,device=x.device,dtype=torch.bool).triu(1)
        weights=score.masked_fill(future,float('-inf')).softmax(-1)
        y=self.projection((weights@v).transpose(1,2).reshape(b,t,d))
        x=self.gate1(x,torch.relu(y)) if self.gate1 is not None else x+y
        y=self.ff(self.norm2(x))
        return self.gate2(x,torch.relu(y)) if self.gate2 is not None else x+y


class CausalExtractor(nn.Module):
    def __init__(self,shape,dtype,cfg,branch='actor'):
        super().__init__()
        self.context=cfg.context_length
        if len(shape) not in (1,3) or shape[0]%self.context:
            raise ValueError('Causal Transformer expects frame_stack=context_length on vector or CHW observations')
        self.frame_shape=(shape[0]//self.context,*shape[1:])
        width=getattr(cfg,branch+'_hidden_dim') or cfg.hidden_dim
        depth=getattr(cfg,branch+'_num_blocks') or cfg.num_blocks
        self.out_dim=width
        if len(shape)==3:
            from architecture.cnn import ImageExtractor
            specs=ImageExtractor.NATURE if min(shape[1:])>=36 else ImageExtractor.MINATAR
            if cfg.image_encoder=='minatar_reference':specs=((32,3,1),(32,3,1))
            elif cfg.image_encoder!='auto':raise ValueError('Unknown image_encoder')
            self.encoder=ImageExtractor(self.frame_shape,cfg.features_dim,cfg.ortho_init,
                255. if np.dtype(dtype)==np.uint8 else 1.,specs)
            self.embedding=nn.Linear(cfg.features_dim,width)
        else:
            self.encoder=nn.Identity();self.embedding=nn.Linear(self.frame_shape[0],width)
        pos=torch.arange(self.context).float()[:,None]
        freq=torch.exp(torch.arange(0,width,2).float()*(-math.log(10000.)/width))
        pe=torch.zeros(self.context,width)
        pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq[:width//2])
        self.register_buffer('positions',pe)
        self.blocks=nn.Sequential(*(CausalBlock(width,cfg.attention_heads,cfg.expansion,cfg.transformer_gated,cfg.norm_eps) for _ in range(depth)))
        self.final_norm=nn.LayerNorm(width,eps=cfg.norm_eps)
    def sequence_features(self,obs):
        # Each minibatch element retains its own complete chronological window.
        batch=obs.shape[0]
        frames=obs.reshape(batch*self.context,*self.frame_shape)
        encoded=self.encoder(frames.float())
        tokens=self.embedding(encoded).reshape(batch,self.context,self.out_dim)
        return self.final_norm(self.blocks(tokens+self.positions))
    def forward(self,obs):
        return self.sequence_features(obs)[:,-1]
