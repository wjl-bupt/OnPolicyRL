import torch
from loss.value_init import init_values_optimal

def gae_fast(agent, td, lam, gamma, value_gae_init, args):

    ''' Calculate GAE for a batch of trajectories. '''

    obs          = td["obs"]         # (T, actors, obs.shape)
    next_obs     = td["next_obs"]    # (T, actors, obs.shape)
    rewards      = td["rewards"]     # (T, actors)
    current_done = td["done"]        # (T, actors) # if obs of timestep t is done
    next_done    = td["next_done"]   # (T, actors) # if next_obs of timestep t is done

    device    = obs.device
    T, actors = obs.shape[:2]

    # ------------------------------------------------------------------
    with torch.no_grad():

        # reshape obs for relative value prediction
        x_i     = obs.reshape(T*actors, *obs.shape[2:])         # (T*actors, ... )
        x_j     = next_obs.reshape(T*actors, *obs.shape[2:])    # (T*actors, ... )
        delta   = agent.get_rv(x_j, x_i).reshape(T, actors)     # (T, actors)


        if value_gae_init == 'optimal':

            # relatives values
            v = torch.zeros(T+1, actors, device=device)     # (T+1, actors)
            offset = init_values_optimal(obs, current_done, agent, args)  # (T+1, actors)

            # offsets (trajetory ranking enabled)
            running_v = torch.zeros(actors, device=device) + offset[0]
            for t in range(T):
                v[t] = running_v
                running_v += delta[t]
                running_v = running_v * (1 - next_done[t])  * (1 - current_done[t]) # episode ended
                running_v += offset[t+1]
            v[T] = running_v
            td_delta = rewards + gamma * v[1:] * (1 - next_done) - v[:-1] # (T, actors)

        elif value_gae_init == 'zeros':

            # relatives values
            v = torch.zeros(T+1, actors, device=device)     # (T+1, actors)

            # zeros (no trajectory ranking)
            running_v = torch.zeros(actors, device=device)
            for t in range(T):
                v[t] = running_v
                running_v += delta[t]
                running_v = running_v * (1 - next_done[t])  * (1 - current_done[t]) # episode ended
            v[T] = running_v

            # TD residuals
            td_delta = rewards + gamma * v[1:] * (1 - next_done) - v[:-1] # (T, actors)


        # GAE
        advantages = torch.zeros_like(td_delta)     # (T, actors)
        gae = torch.zeros(actors, device=device)    # (actors,)

        for t in reversed(range(T)):
            gae = gae * (1 - next_done[t])
            gae = td_delta[t] + lam * gamma * gae
            advantages[t] = gae

    td["advantages"] = advantages # (T, actors)

