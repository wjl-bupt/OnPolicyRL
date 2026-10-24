"""SPO's soft quadratic penalty, implemented on the native PPO Batch protocol."""
from onpolicy.ppo import PPOConfig, ClipPolicyLoss
from onpolicy._runtime import NativeLoop, run_algorithm

class SoftPolicyLoss(ClipPolicyLoss):
    def __call__(self, batch):
        eps = max(1e-3,self._current_eps(batch.progress))
        ratio = (batch.logp-batch.old_log_prob).exp()
        loss = -(batch.advantages*ratio-batch.advantages.abs()*(ratio-1).square()/(2*eps)).mean()
        return loss, {'train/clip_range':eps}

class SPO(NativeLoop):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        c=self.config
        self.policy_loss=SoftPolicyLoss(c.epsilon,c.clip_schedule,c.clip_range_min)

def train(cfg,env,policy,log=None,monitor=None,resume=None):
    return run_algorithm(SPO,cfg,env,policy,log,monitor,resume)


# Algorithm discovery contract (no central name/class mapping).
CONFIG_CLASS = PPOConfig
