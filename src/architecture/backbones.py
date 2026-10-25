"""Native vector/feature backbones. See recipe/NETWORKS.md for audited sources.

RSNorm belongs to the observation transform, never a minibatch forward.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SpectralLinear(nn.Linear):
    """Exact differentiable SN: no mutable power-iteration buffers in vmap/grad.

SVD is more expensive than approximate power iteration; benchmark before scaling.
    """
    def forward(self, x):
        sigma = torch.linalg.matrix_norm(self.weight, ord=2).clamp_min(1e-6)
        return F.linear(x, self.weight / sigma, self.bias)


class SimBaBlock(nn.Module):
    def __init__(self, width, expansion=4, eps=1e-6):
        super().__init__()
        self.branch = nn.Sequential(nn.LayerNorm(width, eps=eps), nn.Linear(width, expansion*width),
                                    nn.ReLU(), nn.Linear(expansion*width, width))
    def forward(self, x):
        return x + self.branch(x)


class BroBlock(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.branch = nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width, eps=eps),
                                    nn.ReLU(), nn.Linear(width, width), nn.LayerNorm(width, eps=eps))
    def forward(self, x):
        return x + self.branch(x)


class SpectralBlock(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.branch = nn.Sequential(nn.LayerNorm(width, eps=eps), SpectralLinear(width,width),
                                    nn.ReLU(), SpectralLinear(width,width))
    def forward(self, x):
        return x + self.branch(x)


class ResidualExtractor(nn.Module):
    def __init__(self, in_dim, architecture, width, blocks, expansion=4, norm_eps=1e-6):
        super().__init__()
        self.out_dim = width
        if architecture == 'simba':
            layers = [nn.Linear(in_dim,width)]
            layers += [SimBaBlock(width,expansion,norm_eps) for _ in range(blocks)]
            layers += [nn.LayerNorm(width,eps=norm_eps)]
        elif architecture == 'bronet':
            layers = [nn.Linear(in_dim,width), nn.LayerNorm(width,eps=norm_eps), nn.ReLU()]
            layers += [BroBlock(width,norm_eps) for _ in range(blocks)]
        elif architecture == 'spectralnet':
            layers = [nn.Linear(in_dim,width),nn.ReLU()]
            layers += [SpectralBlock(width,norm_eps) for _ in range(blocks)]
        else:
            raise ValueError(f'Unknown residual architecture: {architecture}')
        self.net = nn.Sequential(*layers)
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                if architecture == 'bronet':nn.init.orthogonal_(layer.weight,math.sqrt(2))
                else:nn.init.kaiming_normal_(layer.weight,nonlinearity='relu')
                nn.init.zeros_(layer.bias)
    def forward(self, obs):
        return self.net(obs.float())


def validate_backbone(cfg):
    if cfg.architecture not in ('mlp','simba','bronet','spectralnet','causal_transformer'):
        raise ValueError(f'Unknown architecture: {cfg.architecture}')
    for key in ('hidden_dim','num_blocks','expansion','actor_hidden_dim','critic_hidden_dim','actor_num_blocks','critic_num_blocks'):
        val=getattr(cfg,key)
        if val is not None and (type(val) is not int or val < 1):raise ValueError(f'{key} must be a positive integer')
    if not math.isfinite(cfg.norm_eps) or cfg.norm_eps <= 0:raise ValueError('norm_eps must be positive')
    for key in ('context_length','attention_heads','diffusion_steps','diffusion_hidden_dim','diffusion_blocks','diffusion_time_dim'):
        value=getattr(cfg,key)
        if type(value) is not int or value < 1:raise ValueError(f'{key} must be a positive integer')
    if cfg.diffusion_time_dim < 4 or cfg.diffusion_time_dim % 2:
        raise ValueError('diffusion_time_dim must be even and >= 4')
    if cfg.action_head not in ('auto','diffusion'):raise ValueError('Unknown action_head')
    for key in ('diffusion_min_std','diffusion_clip_denoised'):
        value=getattr(cfg,key)
        if not math.isfinite(value) or value<=0:raise ValueError(f'{key} must be finite and positive')
    if type(cfg.transformer_gated) is not bool:raise ValueError('transformer_gated must be boolean')
    if cfg.architecture=='causal_transformer':
        for width in (cfg.actor_hidden_dim or cfg.hidden_dim,cfg.critic_hidden_dim or cfg.hidden_dim):
            if width % cfg.attention_heads:raise ValueError('Transformer width must be divisible by attention_heads')
    if cfg.rsnorm is not None and type(cfg.rsnorm) is not bool:raise ValueError('rsnorm must be boolean or null')


def resolve_normalization(network, environment):
    """Resolve model's input-normalization requirement before job hashing."""
    from copy import deepcopy
    result=deepcopy(environment)
    if network.get('architecture')=='causal_transformer':
        context=network.get('context_length',16)
        wrappers=result.setdefault('wrappers',{})
        if 'frame_stack' in wrappers and wrappers['frame_stack'] != context:
            raise ValueError('Transformer requires environment.wrappers.frame_stack == model.context_length')
        wrappers['frame_stack']=context
    if network.get('action_head')=='diffusion':
        wrappers=result.setdefault('wrappers',{})
        if 'action_mode' in wrappers and wrappers['action_mode'] != 'rescale':
            raise ValueError('Diffusion requires environment.wrappers.action_mode=rescale')
        wrappers['action_mode']='rescale'
    requested=network.get('rsnorm')
    if requested is None and network.get('architecture','mlp')=='simba':requested=True
    if requested is not None:
        wrappers=result.setdefault('wrappers',{})
        if 'normalize_observation' in wrappers and wrappers['normalize_observation'] != requested:
            raise ValueError('model.rsnorm conflicts with environment.wrappers.normalize_observation')
        wrappers['normalize_observation']=requested
    return result
