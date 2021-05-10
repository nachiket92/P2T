from __future__ import print_function
import torch
from torch.utils.data import DataLoader
from models.traj_generator import TrajGenerator
from models.reward_model import RewardModel
import models.rl as rl
import yaml
import utils as u
import numpy as np
from datasets.sdd import SDD as DS
import multiprocessing as mp
import scipy.io as scp


# Read config file
config_file = 'configs/sdd.yml'
with open(config_file, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)


# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Initialize dataset:
ts_set = DS(config['dataroot'],
            config['test'],
            t_h=config['t_h'],
            t_f=4.8,
            grid_dim=config['args_mdp']['grid_dim'][0],
            img_size=config['img_size'],
            horizon=config['args_mdp']['horizon'],
            grid_extent=config['grid_extent'],
            num_actions=config['args_mdp']['actions'])


# Initialize data loader:
ts_dl = DataLoader(ts_set,
                   batch_size=4,
                   shuffle=True,
                   num_workers=config['num_workers'])

or_lbls = scp.loadmat(config['dataroot'] + '/' + config['or_labels'])
or_lbls = or_lbls['img_lbls']


# Initialize Models:
net_r = RewardModel(config['args_r']).float().to(device)
net_r.load_state_dict(torch.load(config['opt_r']['checkpt_dir'] + '/' + 'best.tar')['model_state_dict'])
for param in net_r.parameters():
    param.requires_grad = False
net_r.eval()

net_t = TrajGenerator(config['args_t']).float().to(device)
net_t.load_state_dict(torch.load(config['opt_finetune']['checkpt_dir'] + '/' + 'best.tar')['model_state_dict'])
net_t.eval()

mdp = rl.MDP(config['args_mdp']['grid_dim'],
             horizon=config['args_mdp']['horizon'],
             gamma=config['args_mdp']['gamma'],
             actions=config['args_mdp']['actions'])
initial_state = config['args_mdp']['initial_state']


# Sampling parameters for policy roll-outs:
num_samples = 1000
K = [5, 20]


# Variables to track metrics
agg_min_ade_k = torch.zeros(len(K))
agg_min_fde_k = torch.zeros(len(K))
agg_or_k = torch.zeros(len(K))


counts = 0

with mp.Pool(8) as process_pool:

    # Load batch
    for i, data in enumerate(ts_dl):

        # Process inputs
        hist, fut, img, svf_e, motion_feats, waypts_e, agents, grid_idcs, _, img_vis, ref_pos, ds_ids, _ = data
        img = img.float().to(device)
        motion_feats = motion_feats.float().to(device)
        agents = agents.float().to(device)
        hist = hist.permute(1, 0, 2).float().to(device)
        fut = fut.float().to(device)

        # Calculate reward:
        r, scene_tensor = net_r(motion_feats, img)
        r_detached = r.detach()
        svf, pi = rl.solve(mdp, r_detached, initial_state)

        # Sample policy:
        waypts, scene_feats, agent_feats = rl.sample_policy(pi, mdp, num_samples, config['grid_extent'],
                                                            initial_state, scene_tensor, agents)

        # Generate trajectories:
        horizon = config['args_mdp']['horizon']
        waypts_stacked = waypts.view(-1, horizon, 2)
        waypts_stacked = waypts_stacked.permute(1, 0, 2).to(device)
        scene_feats_stacked = scene_feats.view(-1, horizon, config['args_t']['scene_feat_size'])
        scene_feats_stacked = scene_feats_stacked.permute(1, 0, 2).to(device)
        agent_feats_stacked = agent_feats.view(-1, horizon, config['args_t']['agent_feat_size'])
        agent_feats_stacked = agent_feats_stacked.permute(1, 0, 2).to(device)
        hist_stacked = hist.reshape(hist.shape[0], hist.shape[1], 1, hist.shape[2])
        hist_stacked = hist_stacked.repeat(1, 1, num_samples, 1)
        hist_stacked = hist_stacked.view(hist_stacked.shape[0], -1, hist_stacked.shape[3])
        traj = net_t(hist_stacked, waypts_stacked, scene_feats_stacked, agent_feats_stacked)
        traj = traj.reshape(-1, num_samples, traj.shape[1], traj.shape[2])

        # Cluster
        for n in range(len(K)):
            traj_vec = traj.reshape(traj.shape[0], traj.shape[1], -1).detach().cpu().numpy()
            params = [(traj_vec[i], K[n]) for i in range(len(traj_vec))]
            labels = process_pool.starmap(u.km_cluster, params)
            traj_clustered = torch.zeros(traj.shape[0], K[n], traj.shape[2], traj.shape[3])
            counts_clustered = torch.zeros(traj.shape[0], K[n])
            for m in range(traj.shape[0]):
                clusters = set(labels[m])
                tmp1 = torch.zeros(len(clusters), traj.shape[2], traj.shape[3])
                tmp2 = torch.zeros(len(clusters))
                for idx, c in enumerate(clusters):
                    tmp = np.where(labels[m] == c)
                    tmp1[idx] = torch.mean(traj[m, tmp[0]], dim=0)
                    tmp2[idx] = len(tmp[0])
                traj_clustered[m, :len(tmp2)] = tmp1
                counts_clustered[m, :len(tmp2)] = tmp2

            # 4.8s horizon
            masks = torch.zeros_like(counts_clustered).to(device)
            masks[counts_clustered == 0] = np.inf
            traj_clustered = traj_clustered.to(device)
            agg_min_ade_k[n] += u.min_ade_k(traj_clustered[:, :, 0:12, :], fut, masks).item() * fut.shape[0]
            agg_min_fde_k[n] += u.min_fde_k(traj_clustered[:, :, 0:12, :], fut, masks).item() * fut.shape[0]
            agg_or_k[n] += u.offroad_rate(traj_clustered[:, :, 0:12, :], or_lbls, ref_pos, ds_ids, fut, masks) \
                           * fut.shape[0]

        counts += fut.shape[0]
        print(i)

print('Results for K=5: \n' +
      'MinADEK: ' + str(agg_min_ade_k[0].item()/counts) +
      ' MinFDEK: ' + str(agg_min_fde_k[0].item()/counts) +
      'Offroad rate: ' + str(agg_or_k[0].item()/counts))

print('Results for K=10: \n' +
      'MinADEK: ' + str(agg_min_ade_k[1].item()/counts) +
      ' MinFDEK: ' + str(agg_min_fde_k[1].item()/counts) +
      'Offroad rate: ' + str(agg_or_k[1].item()/counts))
