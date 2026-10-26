"""Adaptive Advantage Estimation (Chen et al., IJCNN 2021).

Equations 19, 22-30 and 36. Categorical and diagonal Normal policies.
Timing convention: previous completed update's regression accuracy/KL and
current rollout entropy determine this rollout's fixed actor/critic lambdas.
The paper does not specify this timing or numerical defaults for b_k/c_k.
"""
from dataclasses import dataclass
import math
import torch as th
from torch.distributions import Categorical, Normal, kl_divergence
from architecture.actor_critic import build_policy as build_base
from onpolicy.ppo import PPOConfig
from onpolicy._runtime import NativeLoop, estimates, run_algorithm


@dataclass
class AAEConfig(PPOConfig):
    aae_variant: int = 1
    accuracy_metric: str = 'ev'
    kl_threshold: float = .01  # engineering default, not specified in paper
    kl_scale: float = 1.0      # engineering default, not specified in paper
    indicator_eps: float = 1e-8

    def __post_init__(self):
        super().__post_init__()
        if type(self.aae_variant) is not int or self.aae_variant not in (1, 2):
            raise ValueError('aae_variant must be 1 or 2')
        if self.accuracy_metric not in ('r2', 'ev'):
            raise ValueError('accuracy_metric must be r2 or ev')
        if any(not math.isfinite(x) or x < 0 for x in (self.kl_threshold, self.kl_scale)):
            raise ValueError('AAE KL threshold/scale must be finite and nonnegative')
        if not math.isfinite(self.indicator_eps) or self.indicator_eps <= 0:
            raise ValueError('indicator_eps must be positive and finite')


def detached_distribution(dist):
    if isinstance(dist, Categorical):
        return Categorical(logits=dist.logits.detach().clone())
    if isinstance(dist, Normal):
        return Normal(dist.loc.detach().clone(), dist.scale.detach().clone())
    raise TypeError('AAE requires Categorical or diagonal Normal distributions')


def joint_stat(value, dist):
    return value.sum(-1) if isinstance(dist, Normal) else value


def regression_accuracy(target, prediction, metric, eps):
    variance = target.var(unbiased=False)
    degenerate = variance <= eps
    error = target - prediction
    numerator = error.square().mean() if metric == 'r2' else error.var(unbiased=False)
    score = th.where(degenerate, th.zeros_like(variance), 1 - numerator / variance.clamp_min(eps))
    return score, degenerate


def adaptive_lambdas(entropy, regression, kl, variant, threshold, scale):
    convergence = th.sigmoid(-entropy)
    penalty = scale * (kl - threshold).clamp_min(0)
    accuracy = (1 - penalty).clamp(0, 1) * regression.clamp(0, 1)
    actor = ((1 - .5 * accuracy) * convergence if variant == 1
             else 1 - accuracy * (1 - .5 * convergence))
    critic = 1 - .5 * accuracy * convergence
    return actor, critic, convergence, accuracy


class AAE(NativeLoop):
    @th.no_grad()
    def _compute_advantages_(self):
        self.policy.train()
        c, p, b = self.config, self.policy, self.rollout_buffer
        ids = self.indices()
        if not len(ids):
            raise ValueError('AAE rollout contains no valid transitions')
        dist = p.dist(self.flat('obs')[ids])
        self._behavior_dist = detached_distribution(dist)
        entropy = joint_stat(dist.entropy(), dist).mean()
        actor, critic, conv, acc = adaptive_lambdas(
            entropy, p.aae_regression, p.aae_kl, c.aae_variant, c.kl_threshold, c.kl_scale)
        adv, _ = estimates(b, c.gamma, float(actor))
        _, returns = estimates(b, c.gamma, float(critic))
        b.set_estimates(adv, returns)
        for key, value in dict(lambda_actor=actor, lambda_critic=critic, entropy=entropy,
                               convergence=conv, accuracy=acc,
                               previous_regression=p.aae_regression, previous_kl=p.aae_kl).items():
            self.log.record('aae/' + key, float(value))

    def _update_(self):
        stats = super()._update_()
        with th.no_grad():
            ids = self.indices()
            obs = self.flat('obs')[ids]
            current = self.policy.dist(obs)
            # KL(old || new), summed across independent continuous action dimensions.
            kl = joint_stat(kl_divergence(self._behavior_dist, current), current).mean().clamp_min(0)
            score, degenerate = regression_accuracy(self.flat('returns')[ids],
                self.policy.predict_value(obs), self.config.accuracy_metric, self.config.indicator_eps)
            self.policy.aae_regression.copy_(score)
            self.policy.aae_kl.copy_(kl)
            self.policy.aae_initialized.fill_(True)
            self.log.record('aae/regression_after_update', float(score))
            self.log.record('aae/kl_after_update', float(kl))
            self.log.record('aae/target_variance_degenerate', float(degenerate))
        self._behavior_dist = None
        return stats


def build_policy(obs, action, network, algo):
    if network.action_head == 'diffusion':
        raise ValueError('AAE requires a Categorical or Gaussian policy, not diffusion')
    p = build_base(obs, action, network)
    p.register_buffer('aae_regression', th.tensor(0., dtype=th.float64))
    p.register_buffer('aae_kl', th.tensor(0., dtype=th.float64))
    p.register_buffer('aae_initialized', th.tensor(False))
    return p


CONFIG_CLASS = AAEConfig


def train(cfg, env, policy, log=None, monitor=None, resume=None):
    return run_algorithm(AAE, cfg, env, policy, log, monitor, resume)
