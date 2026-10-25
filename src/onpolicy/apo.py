"""
APO: Anchored policy optimization by leveraging unsampled actions in continuous spaces
https://www.sciencedirect.com/science/article/abs/pii/S0893608026009342


"""

from dataclasses import dataclass
import torch as th
from torch import Tensor
from onpolicy.ppo import PPOConfig
from onpolicy._runtime import NativeLoop, make_buffer
from protocol.buffer import Field
from architecture.actor_critic import ActorCritic
from architecture.actor_critic import PolicyConfig

@dataclass
class APOConfig(PPOConfig):
    ua_nums: int = 2

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
    """Clipped surrogate plus the existing old-density weighted anchor penalty.

    Anchors are actions sampled at rollout time but never sent to the env.
    Their actions and old joint log densities stay fixed throughout an update.
    This integration repair preserves the repository's regularization formula.
    """

    def __init__(self, epsilon: float = 0.2, schedule: str = "constant",
                 epsilon_min: float = 1e-3):
        self.epsilon = epsilon
        self.schedule = schedule
        self.epsilon_min = epsilon_min

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
        uarr_loss = (0.5 * batch.uact_old_logprobs.exp() * (ua_ratio - 1).square()).mean()
        loss =  policy_loss + uarr_loss
        clipfrac = (pg2 > pg1).float().mean().item()
        return loss, {
            "train/clipfrac": clipfrac, 
            "train/clip_range": eps,
            "train/ratio": float(ratio.detach().mean()),
            "train/ua_ratio": float(ua_ratio.detach().mean()),
            "train/clip_policy_loss": float(policy_loss.detach()),
            "train/uarr_loss" : float(uarr_loss.detach()),
        }

class APO(NativeLoop):
    def __init__(self, env, policy, rollout_buffer, config, log = None, monitor=None):
        super().__init__(env, policy, rollout_buffer, config, log, monitor)
        self.policy_loss = APOPolicyLoss(config.epsilon, config.clip_schedule,
                                         config.clip_range_min)

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
        return APOBatch(
            observations=data.observations, actions=data.actions,
            old_log_prob=data.old_log_prob, old_values=data.old_values,
            advantages=data.advantages, returns=data.returns,
            logp=used_log_prob, entropy=entropy, value=value, progress=progress,
            uact_logprobs=unused_log_prob, uact=unused_act, uact_old_logprobs=data.ua_logprobs,
        )



CONFIG_CLASS = APOConfig

def build_policy(obs,action,network,algo):
    from architecture.actor_critic import build_policy as build
    if type(action).__name__ != 'Box' or len(action.shape) != 1:
        raise ValueError('APO requires a flat continuous Box action space')
    if network.action_head == 'diffusion':
        raise ValueError('APO requires a Gaussian policy; diffusion requires DPPO')
    apo = APOPolicyConfig(**{**vars(network), "ua_nums": algo.ua_nums})
    return build(obs,action,apo,policy_cls=APOActorCritic)

def make_apo_buffer(cfg, env):
    return make_buffer(cfg, env, extra_schema={
        "uact": Field((cfg.ua_nums,) + env.action_space.shape),
        "ua_logprobs": Field((cfg.ua_nums,)),
    }, extra_samples=('uact', 'ua_logprobs'))


def train(cfg, env, policy, log=None, monitor=None, resume=None):
    algo = APO(env, policy, make_apo_buffer(cfg, env), cfg, log=log, monitor=monitor)
    if resume:
        algo.global_step = int(resume['global_step'])
        algo.num_updates = int(resume['iteration'])
    algo.train()
    return {'global_step': algo.global_step, 'iteration': algo.num_updates}
