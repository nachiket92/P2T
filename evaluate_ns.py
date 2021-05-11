from __future__ import print_function
import torch
from torch.utils.data import DataLoader
from models.traj_generator import TrajGenerator
from models.reward_model import RewardModel
import models.rl as rl
import yaml
import utils as u
import numpy as np
from datasets.ns import NS as DS
import multiprocessing as mp
import json
from nuscenes.eval.prediction.config import PredictionConfig
from nuscenes.prediction.helper import convert_local_coords_to_global
from nuscenes.eval.prediction.data_classes import Prediction
from nuscenes.eval.prediction.compute_metrics import compute_metrics


# Read config file
config_file = 'configs/ns.yml'
with open(config_file, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)


# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Initialize dataset:
ts_set = DS(config['dataroot'],
            config['test'],
            t_h=config['t_h'],
            t_f=config['t_f'],
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


# Prediction helper and configs:
helper = ts_set.helper

with open('configs/prediction5_cfg.json', 'r') as f:
    pred_config = json.load(f)
pred_config5 = PredictionConfig.deserialize(pred_config, helper)

with open('configs/prediction10_cfg.json', 'r') as f:
    pred_config = json.load(f)
pred_config10 = PredictionConfig.deserialize(pred_config, helper)


# Lists of predictions
preds5 = []
preds10 = []


with mp.Pool(8) as process_pool:

    # Load batch
    for i, data in enumerate(ts_dl):

        # Process inputs
        hist, fut, img, svf_e, motion_feats, waypts_e, agents, _, _, img_vis, instance_token, sample_token, _ = data
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

        # Cluster (K=5)
        traj_vec = traj.reshape(traj.shape[0], traj.shape[1], -1).detach().cpu().numpy()
        params = [(traj_vec[i], 5) for i in range(len(traj_vec))]
        labels = process_pool.starmap(u.km_cluster, params)
        traj_clustered = torch.zeros(traj.shape[0], 5, traj.shape[2], traj.shape[3])
        counts_clustered = torch.zeros(traj.shape[0], 5)
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

        # Append to list of predictions (K=5):
        for n in range(traj_clustered.shape[0]):

            pred_local = traj_clustered[n].detach()
            counts = counts_clustered[n].detach()
            pred_local = pred_local[counts != 0]
            counts = counts[counts != 0]
            counts = counts.numpy()
            pred_local = pred_local.numpy()

            starting_annotation = helper.get_sample_annotation(instance_token[n], sample_token[n])
            pred_global = np.zeros_like(pred_local)
            for m in range(pred_local.shape[0]):
                pred_global[m] = convert_local_coords_to_global(pred_local[m],
                                                                starting_annotation['translation'],
                                                                starting_annotation['rotation'])

            preds5.append(Prediction(instance=instance_token[n], sample=sample_token[n], prediction=pred_global,
                                     probabilities=counts).serialize())

        # Cluster (K=10)
        params = [(traj_vec[i], 10) for i in range(len(traj_vec))]
        labels = process_pool.starmap(u.km_cluster, params)
        traj_clustered = torch.zeros(traj.shape[0], 10, traj.shape[2], traj.shape[3])
        counts_clustered = torch.zeros(traj.shape[0], 10)
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

        # Append to list of predictions (K=10):
        for n in range(traj_clustered.shape[0]):

            pred_local = traj_clustered[n].detach()
            counts = counts_clustered[n].detach()
            pred_local = pred_local[counts != 0]
            counts = counts[counts != 0]
            counts = counts.numpy()
            pred_local = pred_local.numpy()

            starting_annotation = helper.get_sample_annotation(instance_token[n], sample_token[n])
            pred_global = np.zeros_like(pred_local)
            for m in range(pred_local.shape[0]):
                pred_global[m] = convert_local_coords_to_global(pred_local[m],
                                                                starting_annotation['translation'],
                                                                starting_annotation['rotation'])

            preds10.append(Prediction(instance=instance_token[n], sample=sample_token[n], prediction=pred_global,
                                      probabilities=counts).serialize())

        print("Batch " + str(i) + " of " + str(len(ts_dl)))

results5 = compute_metrics(preds5, helper, pred_config5)
print('Results for K=5: \n' + str(results5))

results10 = compute_metrics(preds10, helper, pred_config10)
print('Results for K=10: \n' + str(results10))
