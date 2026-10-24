import torch
from tensordict import TensorDict

def rv_1_step_target(data_i, data_j, agent, gamma):

    # Delta(i,j) = V(i) - V(j)

    obs_i       = data_i["encoded_obs"]
    obs_j       = data_j["encoded_obs"]
    rewards_i   = data_i["rewards"]
    rewards_j   = data_j["rewards"]
    next_obs_i  = data_i["encoded_next_obs"]
    next_obs_j  = data_j["encoded_next_obs"]
    next_done_i = data_i["next_done"]
    next_done_j = data_j["next_done"]

    with torch.no_grad():

        # build one big (4 x B) batch of state pairs
        pair_first  = torch.cat([next_obs_i, next_obs_i,  obs_i,      obs_i], dim=0)
        pair_second = torch.cat([next_obs_j, obs_j,       next_obs_j, obs_j], dim=0)

        delta_raw   = agent.encoding_to_rv(pair_first, pair_second)       # (4B, ...)
        delta_raw   = delta_raw.view(4, obs_i.size(0), *delta_raw.shape[1:])

        # unpack in the order we stacked
        delta_next, delta_raw_j, delta_raw_i, delta_raw_ij = delta_raw

        # adjust with rewards
        delta_next_j  = delta_raw_j  + rewards_j
        delta_next_i  = delta_raw_i  - rewards_i
        delta_next_ij = delta_raw_ij + rewards_j - rewards_i

        delta_r = rewards_i - rewards_j

        delta_target     = delta_r + gamma * delta_next
        delta_target_i   = delta_r + gamma * delta_next_i
        delta_target_j   = delta_r + gamma * delta_next_j
        #delta_target_ij  = delta_r + gamma * delta_next_ij # noisy
        delta_target_ij  = torch.zeros_like(delta_target) # no bootstrapping

        # masks
        mask_i  = next_done_i.to(torch.bool)     # s_i+1 terminal?
        mask_j  = next_done_j.to(torch.bool)     # s_j+1 terminal?
        mask_ij = mask_i & mask_j                # both terminal

        # priority: ij > i > j > none
        delta_target = torch.where(
            mask_ij, delta_target_ij,
            torch.where(
                mask_i,  delta_target_i,
                torch.where(mask_j, delta_target_j, delta_target)
            )
        )

        # save for relative value clipping
        old_delta = delta_raw_ij

    # target, target_observation, old_delta
    return delta_target, data_j["obs"], old_delta
