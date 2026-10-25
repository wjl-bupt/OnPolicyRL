"""Native DPPO: clipped per-denoising-transition likelihood ratios.

A single-action DDPM implementation of the two-level diffusion MDP objective,
not a claim to reproduce every fine-tuning heuristic of the upstream code.
"""
from dataclasses import dataclass,replace
from types import SimpleNamespace
import torch as th
from onpolicy.ppo import PPOConfig
from onpolicy._runtime import NativeLoop,make_buffer
from protocol.buffer import Field


@dataclass
class DPPOConfig(PPOConfig):
    gamma_denoising: float = .99
    def __post_init__(self):
        super().__post_init__()
        if not 0<=self.gamma_denoising<=1:raise ValueError('gamma_denoising must be in [0,1]')
        if self.ent_coef != 0:raise ValueError('DPPO has no marginal action entropy; ent_coef must be 0')


class DPPO(NativeLoop):
    def _sample_policy(self,obs):
        action,chain,logps,value=self.policy.sample_chain(obs)
        return action,logps.sum(-1),value,{'denoising_chain':chain,'denoising_logprob':logps}
    def _update_(self):
        cfg=self.config;stats={}
        progress=min(1.,self.global_step/cfg.total_steps)
        weights=cfg.gamma_denoising**th.arange(self.policy.denoising_steps-1,-1,-1,device=self.rollout_buffer.device)
        for ids in self.batches():
            with self.timer('bwd'):
                logp,value=self.policy.evaluate_chain(self.flat('obs')[ids],self.flat('denoising_chain')[ids])
                adv=self.flat('advantages')[ids]
                if cfg.normalize_advantage:adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
                old=self.flat('denoising_logprob')[ids]
                batch=SimpleNamespace(logp=logp,old_log_prob=old,advantages=adv[:,None]*weights,
                    progress=progress,value=value,old_values=self.flat('value')[ids],returns=self.flat('returns')[ids])
                pg,pg_stats=self.policy_loss(batch)
                vl,v_stats=self.value_loss(batch)
                lr=logp-old;kl=(lr.exp()-1-lr).mean()
                loss=pg+cfg.vf_coef*vl+cfg.kl_coef*kl
                norm=self.step_loss(loss)
            stats={**pg_stats,**v_stats,'loss/policy':float(pg.detach()),'loss/value':float(vl.detach()),
                'loss/total':float(loss.detach()),'train/kl':float(kl.detach()),'train/grad_norm':norm,
                'diffusion/denoising_steps':float(self.policy.denoising_steps)}
            self.log.add(**stats)
            if cfg.target_kl is not None and stats['train/kl']>cfg.target_kl:break
        return stats


def build_policy(obs,action,network,algo):
    from architecture.actor_critic import build_policy as build
    if network.action_head not in ('auto','diffusion'):raise ValueError('DPPO requires diffusion action head')
    return build(obs,action,replace(network,action_head='diffusion'))


def train(cfg,env,policy,log=None,monitor=None,resume=None):
    from architecture.diffusion import DiffusionPolicy
    if not isinstance(policy,DiffusionPolicy):raise ValueError('DPPO requires a DiffusionPolicy')
    k,a=policy.denoising_steps,policy.action_dim
    buf=make_buffer(cfg,env,extra_schema={'denoising_chain':Field((k+1,a)), 'denoising_logprob':Field((k,))})
    algo=DPPO(env,policy,buf,cfg,log=log,monitor=monitor)
    if resume:algo.global_step=int(resume['global_step']);algo.num_updates=int(resume['iteration'])
    algo.train()
    return {'global_step':algo.global_step,'iteration':algo.num_updates}

CONFIG_CLASS=DPPOConfig
