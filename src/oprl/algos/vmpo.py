"""V-MPO -- On-Policy Maximum a Posteriori Policy Optimization (ICLR 2020).
https://openreview.net/forum?id=SylOlp4FvH

**Why this is its own file**: it is not a surrogate swap (DESIGN.md §4.6, tier two) but
a restructuring of the update rule itself:
  1. E-step: keep only the **top half** of samples by advantage and weight them by
     exp(A/eta) to form a non-parametric target distribution psi.
  2. M-step: pull the parametric policy toward psi under a KL trust-region constraint.
  3. eta and alpha are **Lagrangian multipliers learned through dual losses**, not
     hyperparameters.

There is no ratio and no clipping, so none of ppo.py's objectives can be reused.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, Normal

from ..advantages import get_estimator
from ..buffer import RolloutBuffer
from ..config import Config
from ..logger import Logger
from ..metrics import Timer, explained_variance
from ..norm import ObsNormalizer, RewardNormalizer
from ..rollout import collect
from ..seeding import seed_everything
from ..tree import tree_flatten_time
from ..types import EnvAdapter, Policy


@dataclass
class VMPOConfig(Config):
    # Note: V-MPO's E-step keeps only the top half of samples, so it carries less
    # effective gradient signal than PPO and needs more epochs for a comparable update.
    # num_epochs=1 leads to visible under-training.
    num_epochs: int = 8
    num_minibatches: int = 4
    lr: float = 3e-4
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    advantage: str = "gae"     # shares the estimator registry with PPO
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # --- E-step ---
    top_k_frac: float = 0.5          # the paper uses the top half by advantage
    eps_eta: float = 0.02            # KL bound epsilon_eta on the temperature

    # --- M-step trust region ---
    eps_alpha: float = 0.1           # discrete actions: policy KL bound
    eps_alpha_mu: float = 0.005      # continuous: KL bound on the mean
    eps_alpha_sigma: float = 1e-5    # continuous: KL bound on the variance

    # --- Initial Lagrangian multipliers (the dual losses adjust them) ---
    init_eta: float = 1.0
    init_alpha: float = 1.0
    min_multiplier: float = 1e-8     # projection floor keeping multipliers positive

    norm_obs: bool = False
    norm_reward: bool = False


class VMPODuals(nn.Module):
    """Lagrangian multipliers, log-parameterized for positivity and then projected to
    >= min_multiplier.

    These multipliers are **learned**: eta controls the E-step temperature and alpha the
    tightness of the M-step trust region. They share the policy's optimizer but are
    driven by their own dual losses.
    """

    def __init__(self, cfg: VMPOConfig, continuous: bool):
        super().__init__()
        self.continuous = continuous
        self.log_eta = nn.Parameter(torch.tensor(float(cfg.init_eta)).log())
        if continuous:
            self.log_alpha_mu = nn.Parameter(torch.tensor(float(cfg.init_alpha)).log())
            self.log_alpha_sigma = nn.Parameter(torch.tensor(float(cfg.init_alpha)).log())
        else:
            self.log_alpha = nn.Parameter(torch.tensor(float(cfg.init_alpha)).log())

    @property
    def eta(self) -> Tensor:
        return F.softplus(self.log_eta) + 1e-8

    def alphas(self) -> tuple[Tensor, ...]:
        if self.continuous:
            return (F.softplus(self.log_alpha_mu) + 1e-8,
                    F.softplus(self.log_alpha_sigma) + 1e-8)
        return (F.softplus(self.log_alpha) + 1e-8,)


def e_step(advantages: Tensor, eta: Tensor, cfg: VMPOConfig) -> tuple[Tensor, Tensor, Tensor]:
    """E-step: top-k samples -> softmax(A/eta) weights plus eta's dual loss.

    Returns:
        (psi weights, eta dual loss, mask of selected samples)
    """
    n = advantages.numel()
    k = max(1, int(n * cfg.top_k_frac))
    topk_val, topk_idx = torch.topk(advantages, k)

    # psi(s,a) ~ exp(A/eta), normalized over the top-k only.
    scaled = topk_val / eta
    weights = torch.softmax(scaled, dim=0).detach()   # weights do not backprop into eta

    # eta's dual loss (paper eq. 8): eta*eps + eta*log(1/k * sum exp(A/eta)),
    # computed with logsumexp for numerical stability.
    lse = torch.logsumexp(scaled, dim=0) - torch.log(
        torch.tensor(float(k), device=advantages.device)
    )
    eta_loss = eta * cfg.eps_eta + eta * lse

    mask = torch.zeros(n, dtype=torch.bool, device=advantages.device)
    mask[topk_idx] = True
    return weights, eta_loss, mask


def _kl_categorical(old: Categorical, new: Categorical) -> Tensor:
    return torch.distributions.kl_divergence(old, new)


def _kl_gaussian_decoupled(
    old: Normal, new: Normal
) -> tuple[Tensor, Tensor]:
    """Continuous actions: split the KL into a mean-only and a variance-only term.

    A key implementation detail from the V-MPO paper -- combining them lets the mean and
    variance trust regions interfere with each other.
    """
    mu_o, sd_o = old.loc.detach(), old.scale.detach()
    mu_n, sd_n = new.loc, new.scale

    # Mean only: hold sigma at sigma_old.
    kl_mu = 0.5 * (((mu_n - mu_o) / sd_o) ** 2).sum(-1)
    # Variance only: hold mu at mu_old.
    ratio = sd_n / sd_o
    kl_sigma = (torch.log(ratio) + 0.5 * (1.0 / ratio**2 - 1.0)).sum(-1)
    return kl_mu, kl_sigma


def vmpo_loss(
    policy: Policy, duals: VMPODuals, mb, cfg: VMPOConfig
) -> tuple[Tensor, dict]:
    """E-step + M-step + both dual losses, combined into one differentiable scalar."""
    obs, action = mb.obs, mb["action"]
    adv = mb["advantages"]

    # --- E-step: non-parametric target distribution ---
    eta = duals.eta
    weights, eta_loss, mask = e_step(adv, eta, cfg)

    # --- M-step: weighted maximum likelihood pulling the policy toward psi ---
    logp, entropy, value = policy.evaluate(obs, action, valid=mb.get("valid"))
    pi_loss = -(weights * logp[mask]).sum()

    # --- Trust region: KL(pi_old || pi_new) <= eps_alpha ---
    dist_new = policy.dist(obs) if hasattr(policy, "dist") else None
    kl_stats: dict[str, float] = {}
    alpha_loss = torch.zeros((), device=logp.device)

    if dist_new is not None and "old_dist" in mb:
        old = mb["old_dist"]
        if isinstance(dist_new, Normal):
            a_mu, a_sd = duals.alphas()
            kl_mu, kl_sd = _kl_gaussian_decoupled(old, dist_new)
            kl_mu_m, kl_sd_m = kl_mu.mean(), kl_sd.mean()
            # Multipliers enter the policy term detached and the KL enters the dual
            # term detached; the two must not backprop into each other.
            alpha_loss = (
                a_mu * (cfg.eps_alpha_mu - kl_mu_m.detach())
                + a_sd * (cfg.eps_alpha_sigma - kl_sd_m.detach())
            )
            pi_loss = pi_loss + a_mu.detach() * kl_mu_m + a_sd.detach() * kl_sd_m
            kl_stats = {"diag/kl_mu": kl_mu_m.item(), "diag/kl_sigma": kl_sd_m.item()}
        else:
            (a,) = duals.alphas()
            kl = _kl_categorical(old, dist_new).mean()
            alpha_loss = a * (cfg.eps_alpha - kl.detach())
            pi_loss = pi_loss + a.detach() * kl
            kl_stats = {"diag/kl": kl.item()}

    v_loss = 0.5 * ((value - mb["returns"]) ** 2).mean()
    total = pi_loss + eta_loss + alpha_loss + cfg.vf_coef * v_loss

    stats = {
        "loss/policy": pi_loss.item(),
        "loss/value": v_loss.item(),
        "loss/eta_dual": eta_loss.item(),
        "loss/alpha_dual": float(alpha_loss.item()),
        "loss/total": total.item(),
        "loss/entropy": entropy.mean().item(),
        "vmpo/eta": eta.item(),
        "vmpo/n_selected": int(mask.sum().item()),
        "_value": value.detach(),
        "_returns": mb["returns"],
        **kl_stats,
    }
    for i, a in enumerate(duals.alphas()):
        stats[f"vmpo/alpha_{i}"] = a.item()
    return total, stats


def train(
    cfg: VMPOConfig,
    env: EnvAdapter,
    policy: Policy,
    log: Logger | None = None,
    estimator=None,
) -> Policy:
    device = cfg.resolve_device()
    # Seed everything before any parameter is created: cfg.seed must determine the whole
    # run, not just the environment (see oprl/seeding.py).
    seed_everything(cfg.seed, cfg.deterministic)
    log = log or Logger(run_dir=cfg.run_dir)
    timer = Timer()
    estimator = get_estimator(estimator or cfg.advantage, cfg)

    continuous = type(env.action_space).__name__ != "Discrete"
    duals = VMPODuals(cfg, continuous).to(device)

    obs_norm = ObsNormalizer(env.obs_space.shape, device) if cfg.norm_obs else None
    reward_norm = (
        RewardNormalizer(env.num_envs, cfg.gamma, device) if cfg.norm_reward else None
    )

    buf = RolloutBuffer(
        cfg.rollout_len, env.num_envs, env.obs_space, env.action_space, device,
        extra=getattr(estimator, "extra_fields", None) or None,
    )
    # Multipliers and policy share one optimizer but follow their own dual losses.
    opt = torch.optim.Adam(
        list(policy.parameters()) + list(duals.parameters()), lr=cfg.lr, eps=1e-5
    )

    batch = cfg.rollout_len * env.num_envs
    n_iters = max(1, cfg.total_steps // batch)
    obs = env.reset(seed=cfg.seed)
    global_step = 0
    stats: dict = {}

    for it in range(n_iters):
        if cfg.anneal_lr:
            frac = 1.0 - it / n_iters
            for g in opt.param_groups:
                g["lr"] = frac * cfg.lr

        obs, steps = collect(env, policy, buf, obs, log, timer, obs_norm, reward_norm)
        global_step += steps

        adv, ret, diag = estimator.compute(buf, policy=policy, cfg=cfg)
        buf.advantages, buf.returns = adv, ret
        log.add(**diag)

        # Record pi_old's full distribution for the KL constraint (V-MPO needs the whole
        # distribution, not just logprobs). Note the time axis must be flattened: the
        # encoder expects [B, ...] while buf.obs is [T, N, ...].
        with torch.no_grad():
            old_dist = None
            if hasattr(policy, "dist"):
                old_dist = policy.dist(tree_flatten_time(buf.obs))

        for epoch in range(cfg.num_epochs):
            estimator.on_epoch_start(epoch, buf, policy=policy, cfg=cfg)
            for mb in buf.iter_minibatches(cfg.num_minibatches):
                if old_dist is not None:
                    mb["old_dist"] = _slice_dist(old_dist, mb)
                with timer("bwd"):
                    loss, stats = vmpo_loss(policy, duals, mb, cfg)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    gn = nn.utils.clip_grad_norm_(
                        list(policy.parameters()) + list(duals.parameters()),
                        cfg.max_grad_norm,
                    )
                log.add(**{k: v for k, v in stats.items() if not k.startswith("_")})
                log.record("grad/norm", float(gn))
                opt.step()
                _project_duals(duals, cfg)

        if stats:
            log.record("diag/explained_variance",
                       explained_variance(stats["_value"], stats["_returns"]))
        log.record("train/lr", opt.param_groups[0]["lr"])
        log.add(**timer.drain())
        if it % cfg.log_interval == 0:
            log.dump(global_step)

    log.dump(global_step)
    log.close()
    return policy


@torch.no_grad()
def _project_duals(duals: VMPODuals, cfg: VMPOConfig) -> None:
    """Multipliers must stay positive; softplus guarantees it, this clamps against
    numerical underflow."""
    floor = torch.tensor(cfg.min_multiplier).log()
    for p in duals.parameters():
        p.clamp_(min=float(floor))


def _slice_dist(dist, mb):
    """Slice pi_old by the minibatch's flat indices. `dist` is already [T*N, ...]."""
    idx = mb.get("_flat_idx")
    if idx is None:
        return dist
    if isinstance(dist, Normal):
        return Normal(dist.loc[idx], dist.scale[idx])
    return Categorical(logits=dist.logits[idx])
