"""Public tensor factory: one explicit backend and shared wrapper semantics."""
from common.env.adapter import GymVectorAdapter
from common.env.make_env import make_env
from common.env.transforms import TensorTransforms


def make_tensor_env(env_id, num_envs=1, seed=0, device='cpu', backend='gymnasium',
                    wrappers=None, implementation=None, **options):
    # implementation is recorded, and validated against actual provider below.
    if backend == 'jax':
        if env_id.startswith(('MinAtar/','MJX/')):
            from common.env.jax_adapter import JaxVectorAdapter
            env = JaxVectorAdapter(env_id,num_envs=num_envs,seed=seed,device=device,**options)
        else:
            from common.env.functional import FunctionalAdapter
            env = FunctionalAdapter(env_id,num_envs=num_envs,seed=seed,device=device,**options)
    elif backend in ('gymnasium','auto'):
        if env_id.startswith('MinAtar/'):
            options.setdefault('float_observation', True)
        if env_id.startswith('Craftax'):
            # Gymnasium interface, explicitly backed by the SAME Craftax JAX engine.
            from common.env.gym_bridge import make_craftax_gym
            env = GymVectorAdapter(make_craftax_gym(env_id,num_envs,seed,**options),device)
            env.provenance = dict(implementation='craftax', interface='gymnasium', simulation='jax-cpu')
        elif env_id.startswith('DMControl/'):
            # Load Triton's LLVM before EGL; reversed order can segfault on this host.
            import torch._dynamo  # optimizer initialization dependency
            import gymnasium as gym
            from common.env.dmcontrol import DMControlEnv
            from common.env.wrappers import vectorize
            mode = options.pop('autoreset_mode','SameStep')
            limit = options.pop('max_episode_steps',1000)
            def one():
                e = DMControlEnv(env_id,**options)
                return gym.wrappers.RecordEpisodeStatistics(gym.wrappers.TimeLimit(e,limit))
            env = GymVectorAdapter(vectorize(one,num_envs,seed,False,mode),device)
            env.provenance = dict(implementation='dm-control', task=env_id)
        else:
            env = GymVectorAdapter(make_env(env_id,num_envs=num_envs,seed=seed,backend=backend,**options),device)
            env.provenance = dict(implementation='gymnasium',task=env_id)
    else:
        raise ValueError(f'tensor factory supports gymnasium/jax, got {backend}')
    actual = getattr(env,'provenance',{}).get('implementation',getattr(env,'provenance',{}).get('backend'))
    if implementation is not None and implementation != actual:
        env.close()
        raise ValueError(f'configured implementation {implementation!r} differs from actual {actual!r}')
    env.provenance = dict(getattr(env, 'provenance', {}), interface=backend, requested_options=options)
    return TensorTransforms(env,wrappers) if wrappers is not None else env
