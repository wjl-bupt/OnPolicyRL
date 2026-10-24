import torch

def find_start_states_in_batch(current_done):

    ''' Arguments:
            current_done: (T, actors) tensor of done flags (uint8) for each observation, if the observation is done.
        Returns:
            state_mask: (T, actors) boolean tensor, True if the observation is a start state.
    '''

    done = current_done.to(torch.bool)  # ensure dones are boolean
    T, actors = done.shape[:2]
    state_mask = torch.zeros_like(done, dtype=torch.bool)  # (T, actors)

    # each first state of a traj should be considered, that is not done
    state_mask[0] = ~done[0]

    # get the states that follow a done state (start state of new episode)
    if T > 1:
        state_mask[1:] = done[:-1]

    return state_mask


def init_values_optimal(obs, current_done, agent, args):
    #obs shape (T, actors, *obs_shape)
    #current_done shape (T, actors)

    T, A = obs.shape[:2]
    done = current_done.to(torch.bool)  # ensure bool
    state_mask = find_start_states_in_batch(current_done)  # (T, actors)
    start_pos = torch.nonzero(state_mask, as_tuple=False)  # (N, 2)

    # get all start states
    start_t = start_pos[:, 0]
    start_a = start_pos[:, 1]
    start_states = obs[start_t, start_a]  # (N, *obs_shape)
    N = start_states.shape[0]

    # create matrix of all (i, j) pairs of start states
    s_i = start_states.unsqueeze(1).expand(N, N, *start_states.shape[1:])
    s_j = start_states.unsqueeze(0).expand(N, N, *start_states.shape[1:])
    flat_i = s_i.reshape(N * N, *start_states.shape[1:])
    flat_j = s_j.reshape(N * N, *start_states.shape[1:])
    deltas = agent.get_rv(flat_i, flat_j).view(N, N)  # (N, N)

    # calculate the offsets
    offsets = deltas.mean(dim=1)  # (N,)

    # shift offsets to be non-negative
    offsets = offsets - offsets.min()

    # create value offset tensor
    value_offset = torch.zeros(T + 1, A, device=obs.device)  # (T+1, actors)

    # put the value offsets into the correct positions
    value_offset[start_t, start_a] = offsets

    return value_offset
