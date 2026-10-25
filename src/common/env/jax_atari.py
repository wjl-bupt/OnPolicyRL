"""JAXAtari preprocessing that stops repeats at terminal and retains final obs.

Do not use upstream PixelObsWrapper: it autoresets before exposing the terminal
observation. Our shared adapter owns autoreset and frame stacking.
"""
import numpy as np
from gymnasium import spaces


def atari_provider(name, options):
    import jax
    import jax.numpy as jp
    from jaxtari.core import make
    from jaxtari.wrappers import AtariWrapper
    options = dict(options)
    repeat = options.pop('frame_skip',4)
    size = options.pop('screen_size',84)
    gray = options.pop('grayscale',True)
    sticky = options.pop('repeat_action_probability',.25)
    if not 0 <= sticky <= 1 or repeat < 1: raise ValueError('invalid Atari sticky/repeat')
    env = AtariWrapper(make(name),sticky_actions=sticky,
        episodic_life=options.pop('terminal_on_life_loss',False),
        first_fire=options.pop('fire_reset',False),noop_max=options.pop('noop_max',30),
        full_action_space=options.pop('full_action_space',False),
        max_frames_per_episode=options.pop('max_frames_per_episode',108000))
    if options: raise ValueError(f'unknown JAXAtari options: {options}')
    def area_weights(source, target):
        scale = source/target
        left = jp.arange(target)[:,None]*scale
        right = left + scale
        return jp.maximum(0,jp.minimum(right,jp.arange(source)[None,:]+1)-jp.maximum(left,jp.arange(source)[None,:]))/scale
    h,w,_ = env.image_space().shape
    wy,wx = area_weights(h,size),area_weights(w,size)
    def encode(image):
        x = image.astype(jp.float32)
        if gray: x = jp.rint(x @ jp.array([.299,.587,.114]))[...,None]
        x = jp.einsum('ih,hwc,jw->ijc',wy,x,wx)
        return jp.transpose(jp.clip(jp.rint(x),0,255).astype(jp.uint8),(2,0,1))
    def render(state): return env.render(state.env_state)
    def reset(key):
        _,s = env.reset(key); return encode(render(s)),s
    def step(key,state,action):
        _,s,r,t,tr,_ = env.step(state,action)
        image = render(s)
        prev = image
        for _ in range(repeat-1):
            def advance():
                _,ns,nr,nt,ntr,_ = env.step(s,action)
                return ns,r+nr,nt,ntr,image,render(ns)
            s,r,t,tr,prev,image = jax.lax.cond(t|tr,lambda:(s,r,t,tr,prev,image),advance)
        return encode(jp.maximum(prev,image)),s,r,t,tr,{}
    return reset,step,spaces.Box(0,255,(1 if gray else 3,size,size),dtype=np.uint8),spaces.Discrete(env.action_space().n),'jaxtari'
