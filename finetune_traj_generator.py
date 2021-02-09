from __future__ import print_function
import torch
from torch.utils.data import DataLoader
from models.traj_generator import TrajGenerator
from models.reward_model import RewardModel
import models.rl as rl
import time
import math
import yaml
from torch.utils.tensorboard import SummaryWriter
import utils as u
import numpy as np
import multiprocessing as mp


config_file = 'configs/ns.yml'


# Read config file
with open(config_file, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)


# Import appropriate dataset
if config['ds'] == 'sdd':
    from datasets.sdd import SDD as DS
elif config['ds'] == 'ns':
    from datasets.ns import NS as DS


# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Tensorboard summary writer:
writer = SummaryWriter(log_dir=config['opt_finetune']['log_dir'])


# Initialize datasets:
tr_set = DS(config['dataroot'],
            config['train'],
            t_h=config['t_h'],
            t_f=config['t_f'],
            grid_dim=config['args_mdp']['grid_dim'][0],
            img_size=config['img_size'],
            horizon=config['args_mdp']['horizon'],
            grid_extent=config['grid_extent'],
            num_actions=config['args_mdp']['actions'])

val_set = DS(config['dataroot'],
             config['val'],
             t_h=config['t_h'],
             t_f=config['t_f'],
             grid_dim=config['args_mdp']['grid_dim'][0],
             img_size=config['img_size'],
             horizon=config['args_mdp']['horizon'],
             grid_extent=config['grid_extent'],
             num_actions=config['args_mdp']['actions'])


# Initialize data loaders:
tr_dl = DataLoader(tr_set,
                   batch_size=config['opt_finetune']['batch_size'],
                   shuffle=True,
                   num_workers=config['num_workers'])

val_dl = DataLoader(val_set,
                    batch_size=config['opt_finetune']['batch_size'],
                    shuffle=True,
                    num_workers=config['num_workers'])


# Initialize Models:
net_r = RewardModel(config['args_r']).float().to(device)
net_r.load_state_dict(torch.load(config['opt_r']['checkpt_dir'] + '/' + 'best.tar')['model_state_dict'])
for param in net_r.parameters():
    param.requires_grad = False
net_r.eval()

net_t = TrajGenerator(config['args_t']).float().to(device)
net_t.load_state_dict(torch.load(config['opt_t']['checkpt_dir'] + '/' + 'best.tar')['model_state_dict'])

mdp = rl.MDP(config['args_mdp']['grid_dim'],
             horizon=config['args_mdp']['horizon'],
             gamma=config['args_mdp']['gamma'],
             actions=config['args_mdp']['actions'])

initial_state = config['args_mdp']['initial_state']


# Sampling parameters for policy roll-outs:
num_samples = config['opt_finetune']['num_samples']
num_clusters = config['opt_finetune']['num_clusters']


# Initialize Optimizer:
num_epochs = config['opt_finetune']['num_epochs']
optimizer = torch.optim.Adam(net_t.parameters(), lr=config['opt_finetune']['lr'])


# Load checkpoint if specified in config:
if config['opt_finetune']['load_checkpt']:
    checkpoint = torch.load(config['opt_finetune']['checkpt_path'])
    net_t.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    val_loss = checkpoint['loss']
    min_val_loss = checkpoint['min_val_loss']
else:
    start_epoch = 1
    val_loss = math.inf
    min_val_loss = math.inf


# ======================================================================================================================
# Main Loop
# ======================================================================================================================

# Forever increasing counter to keep track of iterations (for tensorboard log).
iters_epoch = len(tr_set) // config['opt_finetune']['batch_size']
iters = (start_epoch - 1) * iters_epoch

with mp.Pool(8) as process_pool:

    for epoch in range(start_epoch, start_epoch + num_epochs):

        # __________________________________________________________________________________________________________________
        # Train
        # __________________________________________________________________________________________________________________

        # Set batchnorm layers to train mode
        net_t.train()

        # Variables to track training performance
        tr_loss = 0
        tr_time = 0

        # For tracking training time
        st_time = time.time()

        # Load batch
        for i, data in enumerate(tr_dl):

            # Process inputs
            hist, fut, img, svf_e, motion_feats, _, agents, _, _, img_vis, _, _, _ = data
            img = img.float().to(device)
            motion_feats = motion_feats.float().to(device)
            agents = agents.float().to(device)
            hist = hist.permute(1, 0, 2).float().to(device)
            fut = fut.float().to(device)

            # Compute reward and solve for policy:
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
            traj_vec = traj.reshape(traj.shape[0], traj.shape[1], -1).detach().cpu().numpy()
            params = [(traj_vec[ii], num_clusters) for ii in range(len(traj_vec))]
            labels = process_pool.starmap(u.km_cluster, params)
            traj_clustered = torch.zeros(traj.shape[0], num_clusters, traj.shape[2], traj.shape[3])
            counts_clustered = torch.zeros(traj.shape[0], num_clusters)
            for n in range(traj.shape[0]):
                clusters = set(labels[n])
                tmp1 = torch.zeros(len(clusters), traj.shape[2], traj.shape[3])
                tmp2 = torch.zeros(len(clusters))
                for idx, c in enumerate(clusters):
                    tmp = np.where(labels[n] == c)
                    tmp1[idx] = torch.mean(traj[n, tmp[0]], dim=0)
                    tmp2[idx] = len(tmp[0])
                traj_clustered[n, :len(tmp2)] = tmp1
                counts_clustered[n, :len(tmp2)] = tmp2

            # Calculate loss
            masks = torch.zeros_like(counts_clustered).to(device)
            masks[counts_clustered == 0] = np.inf
            traj_clustered = traj_clustered.to(device)
            l_batch = u.min_ade_k(traj_clustered, fut, masks)

            # Backprop:
            optimizer.zero_grad()
            l_batch.backward()
            a = torch.nn.utils.clip_grad_norm_(net_t.parameters(), 10)
            optimizer.step()

            # Track train loss and train time
            batch_time = time.time() - st_time
            tr_loss += l_batch.item()
            tr_time += batch_time
            st_time = time.time()

            # Tensorboard train metrics
            writer.add_scalar('train/loss', l_batch.item(), iters)
            writer.close()

            # Increment global iteration counter for tensorboard
            iters += 1

            # Print/log train loss (path SVFs) and ETA for epoch after pre-defined steps
            iters_log = config['opt_finetune']['steps_to_log_train_loss']
            if i % iters_log == iters_log - 1:
                eta = tr_time / iters_log * (len(tr_set) / config['opt_finetune']['batch_size'] - i)
                print("Epoch no:", epoch,
                      "| Epoch progress:", format(i / (len(tr_set)/config['opt_finetune']['batch_size']) * 100, '0.2f'),
                      "| Train loss:", format(tr_loss / iters_log, '0.5f'),
                      "| Val loss prev epoch", format(val_loss, '0.5f'),
                      "| Min val loss", format(min_val_loss, '0.5f'),
                      "| ETA(s):", int(eta))

                # Log images from train batch into tensorboard:
                tb_fig_train = u.tb_traj_ft_plots(img_vis[0:8],
                                                  hist[:, 0:8, :].permute(1, 0, 2).detach().cpu(),
                                                  traj_clustered[0:8].detach().cpu(),
                                                  fut[0:8].detach().cpu(),
                                                  svf[0:8].detach().cpu(),
                                                  counts_clustered[0:8],
                                                  extent=config['grid_extent'])
                writer.add_figure('train/trajectories', tb_fig_train, iters)
                writer.close()

                # Reset variables to track training performance
                tr_loss = 0
                tr_time = 0

        # __________________________________________________________________________________________________________________
        # Validate
        # __________________________________________________________________________________________________________________
        print('Calculating validation loss...')

        # Set batchnorm/dropout layers to eval mode, stop tracking gradients
        net_t.eval()
        with torch.no_grad():

            # Variables to track validation performance
            agg_val_loss = 0
            val_batch_count = 0

            # Load batch
            for k, data_val in enumerate(val_dl):

                # Process inputs
                hist, fut, img, svf_e, motion_feats, waypts_e, agents, grid_idcs, _, img_vis, _, _, _ = data_val
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
                traj_vec = traj.reshape(traj.shape[0], traj.shape[1], -1).detach().cpu().numpy()
                params = [(traj_vec[ii], num_clusters) for ii in range(len(traj_vec))]
                labels = process_pool.starmap(u.km_cluster, params)
                traj_clustered = torch.zeros(traj.shape[0], num_clusters, traj.shape[2], traj.shape[3])
                counts_clustered = torch.zeros(traj.shape[0], num_clusters)
                for n in range(traj.shape[0]):
                    clusters = set(labels[n])
                    tmp1 = torch.zeros(len(clusters), traj.shape[2], traj.shape[3])
                    tmp2 = torch.zeros(len(clusters))
                    for idx, c in enumerate(clusters):
                        tmp = np.where(labels[n] == c)
                        tmp1[idx] = torch.mean(traj[n, tmp[0]], dim=0)
                        tmp2[idx] = len(tmp[0])
                    traj_clustered[n, :len(tmp2)] = tmp1
                    counts_clustered[n, :len(tmp2)] = tmp2

                # Calculate minADE_K
                masks = torch.zeros_like(counts_clustered).to(device)
                masks[counts_clustered == 0] = np.inf
                traj_clustered = traj_clustered.to(device)
                l_batch = u.min_ade_k(traj_clustered, fut, masks)
                agg_val_loss += l_batch.item()
                val_batch_count += 1

                # Log images from first val batch into tensorboard
                if k == 0:
                    tb_fig_val = u.tb_traj_ft_plots(img_vis[0:8],
                                                    hist[:, 0:8, :].permute(1, 0, 2).detach().cpu(),
                                                    traj_clustered[0:8].detach().cpu(),
                                                    fut[0:8].detach().cpu(),
                                                    svf[0:8].detach().cpu(),
                                                    counts_clustered[0:8],
                                                    extent=config['grid_extent'])
                    writer.add_figure('val/trajectories', tb_fig_val, iters)
                    writer.close()

        # Print validation losses
        print('Val loss :', format(agg_val_loss / val_batch_count, '0.4f'))
        val_loss = agg_val_loss / val_batch_count

        # Tensorboard validation metrics
        writer.add_scalar('val/loss', val_loss, iters)
        writer.close()

        # Save checkpoint
        if config['opt_finetune']['save_checkpoints']:
            model_path = config['opt_finetune']['checkpt_dir'] + '/' + str(epoch) + '.tar'
            torch.save({
                'epoch': epoch,
                'model_state_dict': net_t.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'min_val_loss': min(val_loss, min_val_loss)
            }, model_path)

        # Save best model if applicable
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            model_path = config['opt_finetune']['checkpt_dir'] + '/best.tar'
            torch.save({
                'epoch': epoch,
                'model_state_dict': net_t.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'min_val_loss': min_val_loss
            }, model_path)
