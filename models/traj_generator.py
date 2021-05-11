from __future__ import division
import torch.nn as nn
import torch

# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TrajGenerator(nn.Module):

    def __init__(self, args):
        """
        Trajectory generator that outputs a trajectory conditioned on grid based plan

        args to include
        'coord_emb_size': int Embedding fc layer size for x-y coordinates
        'traj_enc_size': int Size of trajectory encoder
        'waypt_enc_size': int Size of waypoint encoder
        'att_size': int Attention layer size
        'op_length': int Length (prediction horizon) of output trajectories
        'use_agents': int Whether or not to use agents (0 or 1)
        'use_scene': int Whether or not to use scene feats (0 or 1)
        'use_motion_states': int Whether or not to use vel, acc, turn rate for encoding history (0 or 1)
        'scene_feat_size': int Size of scene features at each grid cell
        'agent_feat_size': int Size of agent features at each grid cell
        'scene_emb_size': int Embedding fc layer size for scene features
        'agent_emb_size': int Embedding fc layer size for surrounding agent features
        """

        super(TrajGenerator, self).__init__()

        self.coord_emb_size = args['coord_emb_size']
        self.traj_enc_size = args['traj_enc_size']
        self.waypt_enc_size = args['waypt_enc_size']
        self.att_size = args['att_size']
        self.op_length = args['op_length']
        self.use_agents = args['use_agents']  # 0 or 1
        self.use_scene = args['use_scene']  # 0 or 1
        self.use_motion_states = args['use_motion_states']  # 0 or 1
        self.scene_feat_size = args['scene_feat_size']
        self.scene_emb_size = args['scene_emb_size']
        self.agent_feat_size = args['agent_feat_size']
        self.agent_emb_size = args['agent_emb_size']

        # Track history encoder:
        if self.use_motion_states:
            self.hist_emb = nn.Linear(5, self.coord_emb_size)
        else:
            self.hist_emb = nn.Linear(2, self.coord_emb_size)
        self.hist_enc_gru = nn.GRU(self.coord_emb_size, self.traj_enc_size)

        # Waypoint encoder:
        self.waypt_coord_emb = nn.Linear(2, self.coord_emb_size)
        if self.use_scene:
            self.waypt_scene_emb = nn.Linear(self.scene_feat_size, self.scene_emb_size)
        if self.use_agents:
            self.waypt_agent_emb = nn.Linear(self.agent_feat_size, self.agent_emb_size)
        self.waypt_enc_gru = nn.GRU(self.coord_emb_size + self.use_scene * self.scene_emb_size
                                    + self.use_agents * self.agent_emb_size, self.waypt_enc_size, bidirectional=True)

        # Decoder:
        self.dec_gru = nn.GRUCell(2*self.waypt_enc_size, self.traj_enc_size)
        self.attn1 = nn.Linear(2*self.waypt_enc_size + self.traj_enc_size, self.att_size)
        self.attn2 = nn.Linear(self.att_size, 1)
        self.op_traj = nn.Linear(self.traj_enc_size, 2)

        # Non-linearities:
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.softmax_att = nn.Softmax(dim=0)
        self.tanh = nn.Tanh()

    def forward(self,
                hist: torch.Tensor,
                waypts: torch.Tensor,
                scene_feats: torch.Tensor,
                agent_feats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for trajectory generator
        :param hist: Tensor of track histories, shape [batch_size, hist_len, 2 or 5]
        :param waypts: Tensor of planned way points, shape [batch_size, MDP horizon, 2]
        :param scene_feats: Tensor of scene feats along waypoints, shape [batch_size, MDP horizon, scene_feat_size]
        :param agent_feats: Tensor of agent feats along waypoints, shape [batch_size, MDP horizon, agent_feat_size]
        :return traj: Tensor of predicted trajectories, shape [batch_size, op_length, 2]
        """

        # Obtain sequence lengths and sort indices for pack padded sequence
        tmp = torch.sum(torch.abs(waypts), dim=2)
        tmp = tmp[1:, :] != 0
        waypt_lengths = torch.sum(tmp, dim=0) + 1
        waypt_lengths_sorted, argsort = torch.sort(waypt_lengths, descending=True)
        _, argargsort = torch.sort(argsort)

        # Encode history:
        _, h_hist = self.hist_enc_gru(self.leaky_relu(self.hist_emb(hist)))

        # Encode waypoints:
        waypt_coord_emb = self.waypt_coord_emb(waypts)
        emb_cat = waypt_coord_emb
        if self.use_scene:
            waypt_scene_emb = self.waypt_scene_emb(scene_feats)
            emb_cat = torch.cat((emb_cat, waypt_scene_emb), dim=2)
        if self.use_agents:
            waypt_agent_emb = self.waypt_agent_emb(agent_feats)
            emb_cat = torch.cat((emb_cat, waypt_agent_emb), dim=2)
        emb = self.leaky_relu(emb_cat)

        emb_sorted = emb[:, argsort, :]
        emb_packed = nn.utils.rnn.pack_padded_sequence(emb_sorted, waypt_lengths_sorted.cpu(), batch_first=False)
        h_waypt_packed, _ = self.waypt_enc_gru(emb_packed)
        h_waypt_unpacked, _ = nn.utils.rnn.pad_packed_sequence(h_waypt_packed)
        h_waypt = h_waypt_unpacked[:, argargsort, :]

        # Attention decoder:
        traj = torch.zeros(self.op_length, hist.shape[1], 2).float().to(device)

        h = h_hist.squeeze()
        for k in range(self.op_length):
            att_wts = self.softmax_att(self.attn2(self.tanh(self.attn1(torch.cat((h.repeat(h_waypt.shape[0], 1, 1),
                                                                                  h_waypt), dim=2)))))
            ip = att_wts.repeat(1, 1, h_waypt.shape[2])*h_waypt
            ip = ip.sum(dim=0)
            h = self.dec_gru(ip, h)
            traj[k] = self.op_traj(h)

        traj = traj.permute(1, 0, 2)

        return traj
