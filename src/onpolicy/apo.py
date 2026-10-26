"""
APO: Anchored policy optimization by leveraging unsampled actions in continuous spaces
https://www.sciencedirect.com/science/article/abs/pii/S0893608026009342


"""

from dataclasses import dataclass
import time
import torch as th
from torch import Tensor
from onpolicy.ppo import PPOConfig
from onpolicy._review_metrics import ReviewConfig, ReviewMixin, describe, alignment
from onpolicy._runtime import NativeLoop, make_buffer
from protocol.buffer import Field
from architecture.actor_critic import ActorCritic
from architecture.actor_critic import PolicyConfig

@dataclass
class APOConfig(ReviewConfig, PPOConfig):
    ua_nums: int = 2
    uarr_coef: float = -1.0  # >=0: fixed weight; <0: old action density

    def __post_init__(self):
        super().__post_init__()
        if type(self.ua_nums) is not int or self.ua_nums < 1:
            raise ValueError('ua_nums must be a positive integer')


@dataclass
class APOPolicyConfig(PolicyConfig):
    ua_nums: int = None

@dataclass
class APOBatch:
    """What one minibatch looks like inside the update (algo_design §4.2).

    Everything the losses need is here, evaluated exactly once -- append-only:
    new fields join at the end, signatures never break.
    """

    observations: Tensor
    actions: Tensor
    old_log_prob: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor
    logp: Tensor          # current policy, from the single evaluate
    entropy: Tensor
    value: Tensor
    # APO specific param
    uact : Tensor
    uact_logprobs : Tensor
    uact_old_logprobs : Tensor
    
    progress: float = 0.0



class APOActorCritic(ActorCritic):
    def __init__(self, extractors, policy_head, value_head, share, cfg):
        super().__init__(extractors, policy_head, value_head, share)
        self.ua_nums = cfg.ua_nums

    def act_with_anchors(self, obs):
        """Return [B, K+1, D] actions; only the first action reaches the env.

        Keep inherited act/evaluate single-action protocols for evaluation.
        Distribution sample axes precede batch axes: sample [K+1, B, D],
        then move the sample axis to the buffer's [B, K+1, D] layout.
        """
        feat_pi, feat_v = self._features(obs)
        dist = self.policy_head.dist(feat_pi)
        if isinstance(dist, th.distributions.Categorical):
            # Weighted sampling without replacement from the complement of the
            # executed action. Gumbel top-k remains stable for tiny probabilities.
            executed = dist.sample()
            scores = dist.logits - th.empty_like(dist.logits).exponential_().log()
            scores.scatter_(1, executed[:, None], -th.inf)
            unused = scores.topk(self.ua_nums, dim=-1).indices
            actions = th.cat((executed[:, None], unused), dim=1).T
        else:
            actions = dist.sample((self.ua_nums + 1,))
        logp = self.policy_head.log_prob(dist, actions)
        return actions.movedim(0, 1), logp.movedim(0, 1), self._value(feat_v)

    def evaluate_anchors(self, obs, actions):
        """Evaluate [B, K+1, D] stored actions against each corresponding state."""
        feat_pi, feat_v = self._features(obs)
        dist = self.policy_head.dist(feat_pi)
        logp = self.policy_head.log_prob(dist, actions.movedim(1, 0)).movedim(0, 1)
        return logp, self.policy_head.entropy(dist), self._value(feat_v)

class APOPolicyLoss:
    """Clipped surrogate plus a fixed or old-density weighted anchor penalty.

    Anchors are actions sampled at rollout time but never sent to the env.
    Their actions and old joint log densities stay fixed throughout an update.
    A negative uarr_coef selects the original old-density weighting.
    """

    def __init__(self, epsilon: float = 0.2, schedule: str = "constant",
                 epsilon_min: float = 1e-3, uarr_coef: float = -1.0):
        self.epsilon = epsilon
        self.schedule = schedule
        self.epsilon_min = epsilon_min
        self.uarr_coef = uarr_coef

    def _current_eps(self, progress: float) -> float:
        """The clip range at this point in training (`progress` in [0, 1])."""
        if self.schedule == "linear":
            eps = self.epsilon_min + (self.epsilon - self.epsilon_min) * (1.0 - progress)
            return max(self.epsilon_min, eps)
        return self.epsilon

    def __call__(self, batch: APOBatch) -> tuple[Tensor, dict[str, float]]:
        eps = self._current_eps(batch.progress)
        ratio = (batch.logp - batch.old_log_prob).exp()
        pg1 = -batch.advantages * ratio
        pg2 = -batch.advantages * ratio.clamp(1 - eps, 1 + eps)
        policy_loss = th.max(pg1, pg2).mean()
        # APO regularization
        ua_ratio = (batch.uact_logprobs - batch.uact_old_logprobs).exp()
        uarr_coef = (th.full_like(batch.uact_old_logprobs, self.uarr_coef)
                     if self.uarr_coef >= 0 else batch.uact_old_logprobs.exp())
        uarr_loss = (0.5 * uarr_coef * (ua_ratio - 1).square()).mean()
        loss =  policy_loss + uarr_loss
        clipfrac = (pg2 > pg1).float().mean().item()
        return loss, {
            "train/clipfrac": clipfrac, 
            "train/clip_range": eps,
            "apo/ratio": float(ratio.detach().mean()),
            "apo/ua_ratio": float(ua_ratio.detach().mean()),
            "train/clip_policy_loss": float(policy_loss.detach()),
            "apo/uarr_loss" : float(uarr_loss.detach()),
            "apo/uarr_coef" : float(uarr_coef.mean()),
        }

class APO(ReviewMixin, NativeLoop):
    def __init__(self, env, policy, rollout_buffer, config, log = None, monitor=None):
        super().__init__(env, policy, rollout_buffer, config, log, monitor)
        self.policy_loss = APOPolicyLoss(config.epsilon, config.clip_schedule,
                                         config.clip_range_min, config.uarr_coef)

    def _sample_policy(self, obs):
        actions, logp, value = self.policy.act_with_anchors(obs)
        return actions[:, 0], logp[:, 0], value, {
            'uact': actions[:, 1:], 'ua_logprobs': logp[:, 1:]}

    # NativeLoop collects next_obs/next_value and handles timeout bootstraps.
    # PPO's update loop consumes the extended batch and the APO policy loss.

    def _make_batch_(self, data) -> APOBatch:
        """S8: ONE evaluate over the minibatch, then assemble the context. The
        policy owns the distribution (doc §9.2 S8): stored obs + actions -> the
        current logp / entropy / value in a single forward. `progress` (fraction
        of total_steps consumed) is stamped here so scheduled components -- e.g. a
        decaying clip range -- can read it off the batch."""
        used_act, unused_act = data.actions, data.uact
        total_actions = th.cat((used_act.unsqueeze(1), unused_act), dim=1)
        logp, entropy, value = self.policy.evaluate_anchors(data.observations, total_actions)
        used_log_prob, unused_log_prob = logp[:,0], logp[:, 1:]
        progress = min(1.0, self.global_step / max(1, self.config.total_steps))
        batch = APOBatch(
            observations=data.observations, actions=data.actions,
            old_log_prob=data.old_log_prob, old_values=data.old_values,
            advantages=data.advantages, returns=data.returns,
            logp=used_log_prob, entropy=entropy, value=value, progress=progress,
            uact_logprobs=unused_log_prob, uact=unused_act, uact_old_logprobs=data.ua_logprobs,
        )
        interval = self.config.review_gradient_interval
        if self.config.review_metrics and interval and self.num_updates % interval == 0:
            diagnostic_start=time.perf_counter()
            # Diagnostic autograd.grad does not populate .grad or affect training.
            from dataclasses import replace
            diagnostic = replace(batch)
            if self.config.normalize_advantage:
                a = batch.advantages
                diagnostic.advantages = (a-a.mean())/(a.std(unbiased=False)+1e-8)
            total, _ = self.policy_loss(diagnostic)
            ratio = (diagnostic.logp-diagnostic.old_log_prob).exp()
            eps = self.policy_loss._current_eps(progress)
            pg = -th.min(diagnostic.advantages*ratio, diagnostic.advantages*ratio.clamp(1-eps,1+eps)).mean()
            params = list(self.policy.parameters())
            def grad(loss):
                gs=th.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
                return th.cat([g.flatten() if g is not None else th.zeros_like(p).flatten() for p,g in zip(params,gs)])
            gp,gr=grad(pg),grad(total-pg)
            self.log.record('review/pg_gradient_norm',float(gp.norm()))
            self.log.record('review/regularizer_gradient_norm',float(gr.norm()))
            self.log.record('review/regularizer_pg_norm_ratio',float(gr.norm()/(gp.norm()+1e-12)))
            alignment(self.log,'review/regularizer_vs_pg',gr,gp)
            self._review_gradient_done=True
            self.log.record('review_time/gradient_diagnostic_seconds',time.perf_counter()-diagnostic_start)
        return batch

    def _update_(self):
        self._review_gradient_done=False
        return super()._update_()


CONFIG_CLASS = APOConfig

def build_policy(obs,action,network,algo):
    from architecture.actor_critic import build_policy as build
    if type(action).__name__ == 'Discrete':
        if getattr(action,'start',0) != 0:raise ValueError('APO requires zero-based Discrete actions')
        if not 1 <= algo.ua_nums <= action.n-1:
            raise ValueError('Discrete APO requires 1 <= ua_nums <= |A|-1')
    elif type(action).__name__ != 'Box' or len(action.shape) != 1:
        raise ValueError('APO requires flat Box or Discrete actions')
    if network.action_head == 'diffusion':
        raise ValueError('APO requires a Gaussian policy; diffusion requires DPPO')
    apo = APOPolicyConfig(**{**vars(network), "ua_nums": algo.ua_nums})
    return build(obs,action,apo,policy_cls=APOActorCritic)

def make_apo_buffer(cfg, env):
    return make_buffer(cfg, env, extra_schema={
        "uact": Field((cfg.ua_nums,) + env.action_space.shape, th.int64 if hasattr(env.action_space,"n") else th.float32),
        "ua_logprobs": Field((cfg.ua_nums,)),
    }, extra_samples=('uact', 'ua_logprobs'))


def train(cfg, env, policy, log=None, monitor=None, resume=None):
    algo = APO(env, policy, make_apo_buffer(cfg, env), cfg, log=log, monitor=monitor)
    if resume:
        algo.global_step = int(resume['global_step'])
        algo.num_updates = int(resume['iteration'])
    algo.train()
    return {'global_step': algo.global_step, 'iteration': algo.num_updates}
