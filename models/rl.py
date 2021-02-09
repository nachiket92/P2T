import torch
import math
import utils as u
from typing import Tuple
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MDP(object):

    def __init__(self, grid_dim, horizon=40, gamma=0.99, actions=4):

        self.horizon = horizon
        self.gamma = gamma

        if actions == 4:
            # Actions: [D, R, U, L, end]
            self.actions = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1)]
        else:
            # Actions: [D, R, U, L, DR, UR, DL, UL, end]
            self.actions = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (1, 1, 0),
                            (-1, 1, 0), (1, -1, 0), (-1, -1, 0), (0, 0, 1)]

        # Grid dimensions
        self.grid_dim = grid_dim

        # Transition tables
        s_next, s_next_inv, s_prev, s_prev_inv = self.get_transition_table()
        self.s_next = s_next.to(device)
        self.s_next_inv = s_next_inv.to(device)
        self.s_prev = s_prev.to(device)
        self.s_prev_inv = s_prev_inv.to(device)
        self.s_next.requires_grad = False
        self.s_next_inv.requires_grad = False
        self.s_prev.requires_grad = False
        self.s_prev_inv.requires_grad = False

    def get_transition_table(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns look up tables for next state and previous state for a state-action pair. Additionally returns
        look up tables with invalid flags for invalid transitions
        """
        s_next = torch.zeros((len(self.actions), 2 * self.grid_dim[0] * self.grid_dim[1])).long()
        s_prev = torch.zeros((len(self.actions), 2 * self.grid_dim[0] * self.grid_dim[1])).long()

        # Some actions are invalid for certain states, eg. at the boundaries of the grid, or transitions out of goal
        # states. Flags for these are stored in s_next_inv and s_prev_inv
        s_next_inv = torch.zeros((len(self.actions), 2 * self.grid_dim[0] * self.grid_dim[1]))
        s_prev_inv = torch.ones((len(self.actions), 2 * self.grid_dim[0] * self.grid_dim[1]))

        # Pointers to the next state given current state and action:
        for t in range(2):
            for m in range(self.grid_dim[0]):
                for n in range(self.grid_dim[1]):
                    for k in range(len(self.actions)):
                        s_type = self.actions[k][2]
                        row = m + self.actions[k][0]
                        col = n + self.actions[k][1]
                        orig_id = trc2id(t, m, n, self.grid_dim)
                        if 0 <= row < self.grid_dim[0] and 0 <= col < self.grid_dim[1] and t == 0:
                            next_id = trc2id(s_type, row, col, self.grid_dim)
                            s_next[k, orig_id] = int(next_id)
                        else:
                            s_next_inv[k, orig_id] = -math.inf

        # Pointers to previous state, given current state and action taken to get there:
        for t in range(2):
            for m in range(self.grid_dim[0]):
                for n in range(self.grid_dim[1]):
                    for k in range(len(self.actions)):
                        s_type = t - self.actions[k][2]
                        row = m - self.actions[k][0]
                        col = n - self.actions[k][1]
                        orig_id = trc2id(t, m, n, self.grid_dim)
                        if s_type == 0 and 0 <= row < self.grid_dim[0] and 0 <= col < self.grid_dim[1]:
                            prev_id = trc2id(s_type, row, col, self.grid_dim)
                            s_prev[k, orig_id] = int(prev_id)
                        else:
                            s_prev_inv[k, orig_id] = 0

        return s_next, s_next_inv, s_prev, s_prev_inv


def solve(mdp: MDP, r: torch.Tensor, initial_state: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MaxEnt RL. Solves for MaxEnt policy given current reward. Returns MaxEnt policy and state visitation frequencies
    given initial state
    :param mdp: MDP object
    :param r: Reward tensor, shape [batch_size, 2, grid_dim, grid_dim]
    :param initial_state: Tuple representing row and column of initial state
    :return svf: Tensor of SVFs of MaxEnt policy, shape [batch_size, 2, grid_dim, grid_dim]
    :return pi: Policy tensor, shape [batch_size, horizon, num_actions, num_states]
    """

    pi = backward(mdp, r)
    svf = forward(mdp, pi, initial_state)
    svf = vec2grid(svf, mdp.grid_dim)
    return svf, pi


def backward(mdp: MDP, r: torch.Tensor) -> torch.Tensor:
    """
    Approximate value iteration with goal and path states
    :param mdp: MDP object
    :param r: Reward tensor, shape [batch_size, 2, grid_dim, grid_dim]
    :return pi: Policy tensor, shape [batch_size, horizon, num_actions, num_states]
    """
    # Convert r to a vector
    r = grid2vec(r)

    # Initialize V tensor
    v = torch.zeros_like(r)
    num_path_states = int(v.shape[1]/2)

    # Initialize path state V to -inf
    v[:, 0:num_path_states] = -math.inf

    # Initialize goal state V to goal rewards
    v[:, num_path_states:] = r[:, num_path_states:]

    # Batch indices
    batch_idcs = torch.tensor(range(r.shape[0])).unsqueeze(1).unsqueeze(2)
    batch_idcs = batch_idcs.repeat(1, len(mdp.actions), r.shape[1])

    # Initialize Q and pi tensors
    q = torch.zeros_like(v[batch_idcs, mdp.s_next])
    pi = torch.zeros(q.shape[0], mdp.horizon, q.shape[1], q.shape[2]).to(device)

    # Set requires grad to 0 to save memory
    pi.requires_grad = False
    v.requires_grad = False
    q.requires_grad = False

    # Backward pass of DP
    for k in range(mdp.horizon):
        q = r.unsqueeze(1).repeat(1, q.shape[1], 1) + mdp.gamma * v[batch_idcs, mdp.s_next] + mdp.s_next_inv
        v = torch.logsumexp(q, dim=1)
        v[:, num_path_states:] = r[:, num_path_states:]
        pi[:, mdp.horizon-k-1, :, :] = torch.exp(q - v.unsqueeze(1).repeat(1, q.shape[1], 1))
    pi[torch.isnan(pi)] = 0

    return pi


# Policy propagation (see algorithm 4 in the paper)
def forward(mdp: MDP, pi: torch.Tensor, initial_state: Tuple[int, int]) -> torch.Tensor:
    """
    Policy propagation. Propagates MaxEnt policy given initial state to return SVFs
    :param mdp: MDP object
    :param pi: Policy tensor, shape [batch_size, horizon, num_actions, num_states]
    :param initial_state: Tuple for initial state
    :return svf: Tensor of SVFs, shape [batch_size, num_states]
    """

    # Initialize state visitation frequencies
    c_row = initial_state[0]
    c_col = initial_state[1]
    c_id = trc2id(0, c_row, c_col, mdp.grid_dim)
    svf_t = torch.zeros(pi.shape[0], mdp.horizon, pi.shape[3]).to(device)
    svf_t.requires_grad = False
    svf_t[:, 0, c_id] = 1

    # Action indices
    a_idcs = torch.arange(len(mdp.actions)).repeat(pi.shape[3], 1).permute(1, 0)

    # Batch indices
    batch_idcs = torch.tensor(range(pi.shape[0])).unsqueeze(1).unsqueeze(2).repeat(1, pi.shape[2], pi.shape[3])

    # Forward pass of DP to compute state visitation frequencies
    for t in range(mdp.horizon - 1):
        d_next = pi[:, t, a_idcs, mdp.s_prev] * svf_t[:, t, :][batch_idcs, mdp.s_prev] * mdp.s_prev_inv
        svf_t[:, t + 1, :] = torch.sum(d_next, dim=1)

    # Sum over MDP horizon
    svf = torch.sum(svf_t, dim=1)
    svf.requires_grad = False

    return svf


def sample_policy(pi, mdp, num_samples, grid_extent, initial_state, scene_tensor, agent_tensor):
    """
    Samples state sequences from MaxEnt policy:

    Inputs
    pi: policy obtained by solving approximate value iteration
    mdp: Markov decision process object
    num_samples: number of state sequences to sample from policy
    grid_extents: (x_min, x_max, y_min, y_max)
    initial state: (row, column)
    scene_tensor: Tensor of image features extracted by CNN backbone
    agent_tensor: Tensor of surrounding agent states

    Output
    waypts: BEV coordinates for states in the sampled plan
    scene_feats: sequences of scene features corresponding to sampled plan
    agent_feats: sequences of agent features corresponding to sampled plan
    """

    pi = pi.permute(0, 1, 3, 2)
    pi[:, :, pi.shape[2] // 2:, -1] = 1
    s_next = mdp.s_next.permute(1, 0)
    s_next[s_next.shape[0] // 2:, -1] = torch.tensor(range(s_next.shape[0] // 2)) + s_next.shape[0] // 2

    pi = pi.to(device)
    s_next = s_next.to(device)

    batch_size = pi.shape[0]
    grid_idcs = torch.zeros(batch_size, num_samples, mdp.horizon, 2).to(device)
    waypts = torch.zeros(batch_size, num_samples, mdp.horizon, 2).to(device)
    s0 = trc2id(0, initial_state[0], initial_state[1], mdp.grid_dim)
    s = torch.ones(batch_size * num_samples).long().to(device) * s0
    grid_size = grid_extent[1] - grid_extent[0]
    row_centers = torch.tensor(np.linspace(grid_extent[3] - grid_size / (mdp.grid_dim[0] * 2),
                                           grid_extent[2] + grid_size / (mdp.grid_dim[0] * 2),
                                           mdp.grid_dim[0])).to(device)
    col_centers = torch.tensor(np.linspace(grid_extent[0] + grid_size / (mdp.grid_dim[1] * 2),
                                           grid_extent[1] - grid_size / (mdp.grid_dim[1] * 2),
                                           mdp.grid_dim[1])).to(device)

    for n in range(mdp.horizon):
        # Populate grid_idcs and waypts
        t, r, c = id2trc(s, mdp.grid_dim)
        r = r * (1 - t)
        c = c * (1 - t)
        waypts[:, :, n, 0] = (row_centers[r.long()] * (1 - t)).reshape(batch_size, num_samples)
        waypts[:, :, n, 1] = (col_centers[c.long()] * (1 - t)).reshape(batch_size, num_samples)
        grid_idcs[:, :, n, 0] = r.reshape(batch_size, num_samples)
        grid_idcs[:, :, n, 1] = c.reshape(batch_size, num_samples)

        # Sample next actions
        idcs = torch.tensor(range(batch_size)).unsqueeze(0).repeat(num_samples, 1).permute(1, 0).reshape(-1).to(device)
        pi_s = pi[idcs, n, s]
        a = Categorical(pi_s).sample()

        # Obtain next states
        s = s_next[s, a]

    all_grid_idcs = grid_idcs.reshape(batch_size * num_samples, mdp.horizon, 2)
    scene_tensor = scene_tensor.unsqueeze(0).repeat(num_samples, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
    scene_tensor = scene_tensor.reshape(-1, scene_tensor.shape[2], scene_tensor.shape[3], scene_tensor.shape[4])
    agent_tensor = agent_tensor.unsqueeze(0).repeat(num_samples, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
    agent_tensor = agent_tensor.reshape(-1, agent_tensor.shape[2], agent_tensor.shape[3], agent_tensor.shape[4])
    scene_feats, agent_feats = u.get_plan_feats(all_grid_idcs,
                                                scene_tensor,
                                                agent_tensor)
    scene_feats = scene_feats.reshape(batch_size, num_samples, mdp.horizon, scene_feats.shape[2])
    agent_feats = agent_feats.reshape(batch_size, num_samples, mdp.horizon, agent_feats.shape[2])

    return waypts, scene_feats, agent_feats


def vec2grid(v, grid_dim):
    """
    Convert vector to grid
    """
    new_shape = tuple(list(v.shape)[:-1] + [2, grid_dim[0], grid_dim[1]])
    g = v.view(new_shape)
    return g


def grid2vec(g):
    """
    Convert grid to vector
    """
    vec_len = g.shape[-1] * g.shape[-2] * g.shape[-3]
    new_shape = tuple(list(g.shape)[:-3] + [vec_len])
    v = g.view(new_shape)
    return v


def trc2id(t, row, col, grid_dim):
    """
    Convert row, col, cell type to state id
    """
    s_id = (t * grid_dim[0] * grid_dim[1] + row * grid_dim[1] + col) // 1
    return s_id


def id2trc(s_id, grid_dim):
    """
    Convert state id to row, col, cell type
    """
    t = s_id // (grid_dim[0] * grid_dim[1])
    r = (s_id - t * grid_dim[0] * grid_dim[1]) // grid_dim[1]
    c = (s_id - t * grid_dim[0] * grid_dim[1] - r * grid_dim[1]) // 1
    return t, r, c
