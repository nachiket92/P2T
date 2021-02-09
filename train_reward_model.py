from __future__ import print_function
import torch
from torch.utils.data import DataLoader
from models.reward_model import RewardModel
import time
import math
import models.rl as rl
import yaml
from torch.utils.tensorboard import SummaryWriter
import utils as u


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
writer = SummaryWriter(log_dir=config['opt_r']['log_dir'])


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
tr_collate_fn = tr_set.collate_fn if config['ds'] == 'sdd' else None
tr_dl = DataLoader(tr_set,
                   batch_size=config['opt_r']['batch_size'],
                   shuffle=True,
                   num_workers=config['num_workers'],
                   collate_fn=tr_collate_fn)

val_collate_fn = val_set.collate_fn if config['ds'] == 'sdd' else None
val_dl = DataLoader(val_set,
                    batch_size=config['opt_r']['batch_size'],
                    shuffle=True,
                    num_workers=config['num_workers'],
                    collate_fn=val_collate_fn)


# Initialize Models:
net = RewardModel(config['args_r']).float().to(device)

mdp = rl.MDP(config['args_mdp']['grid_dim'],
             horizon=config['args_mdp']['horizon'],
             gamma=config['args_mdp']['gamma'],
             actions=config['args_mdp']['actions'])

initial_state = config['args_mdp']['initial_state']


# Initialize Optimizer:
num_epochs = config['opt_r']['num_epochs']
optimizer = torch.optim.Adam(net.parameters(), lr=config['opt_r']['lr'])


# Load checkpoint if specified in config:
if config['opt_r']['load_checkpt']:
    checkpoint = torch.load(config['opt_r']['checkpt_path'])
    net.load_state_dict(checkpoint['model_state_dict'])
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
iters_epoch = len(tr_set) // config['opt_r']['batch_size']
iters = (start_epoch - 1) * iters_epoch

for epoch in range(start_epoch, start_epoch + num_epochs):

    # __________________________________________________________________________________________________________________
    # Train
    # __________________________________________________________________________________________________________________

    # Set batchnorm layers to train mode
    net.train()

    # Variables to track training performance
    tr_svf_diff_path = 0
    tr_svf_diff_goal = 0
    tr_time = 0

    # For tracking training time
    st_time = time.time()

    # Load batch
    for i, data in enumerate(tr_dl):

        # Process inputs
        _, _, img, svf_e, motion_feats, _, _, _, _, img_vis, _, _, _ = data
        svf_e = svf_e.float().to(device)
        img = img.float().to(device)
        motion_feats = motion_feats.float().to(device)

        # Calculate reward over grid using model
        r, _ = net(motion_feats, img)

        # Forward RL (solve for maxent policy and SVF)
        r_detached = r.detach()
        svf, _ = rl.solve(mdp, r_detached, initial_state=initial_state)

        # Calculate difference in state visitation frequencies
        svf = svf.to(device)
        svf_diff = svf - svf_e

        # Backprop
        optimizer.zero_grad()
        torch.autograd.backward(r, svf_diff)
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        # Track difference in state visitation frequencies and train time
        batch_time = time.time() - st_time
        tr_svf_diff_path += torch.mean(torch.abs(svf_diff[:, 0, :, :])).item()
        tr_svf_diff_goal += torch.mean(torch.abs(svf_diff[:, 1, :, :])).item()
        tr_time += batch_time
        st_time = time.time()

        # Tensorboard train metrics
        writer.add_scalar('train/SVF diff (goals)', torch.mean(torch.abs(svf_diff[:, 1, :, :])).item(), iters)
        writer.add_scalar('train/SVF diff (paths)', torch.mean(torch.abs(svf_diff[:, 0, :, :])).item(), iters)
        writer.close()

        # Increment global iteration counter for tensorboard
        iters += 1

        # Print/log train loss (path SVFs) and ETA for epoch after pre-defined steps
        iters_log = config['opt_r']['steps_to_log_train_loss']
        if i % iters_log == iters_log - 1:
            eta = tr_time / iters_log * (len(tr_set) / config['opt_r']['batch_size'] - i)
            print("Epoch no:", epoch,
                  "| Epoch progress(%):", format(i / (len(tr_set) / config['opt_r']['batch_size']) * 100, '0.2f'),
                  "| Train SVF diff (paths):", format(tr_svf_diff_path / iters_log, '0.5f'),
                  "| Train SVF diff (goals):", format(tr_svf_diff_goal / iters_log, '0.7f'),
                  "| Val loss prev epoch", format(val_loss, '0.7f'),
                  "| Min val loss", format(min_val_loss, '0.5f'),
                  "| ETA(s):", int(eta))

            # Log images from train batch into tensorboard:
            tb_fig_train = u.tb_reward_plots(img_vis[0:8],
                                             r[0:8].detach().cpu(),
                                             svf[0:8].detach().cpu(),
                                             svf_e[0:8].detach().cpu())
            writer.add_figure('train/SVFs_and_rewards', tb_fig_train, iters)
            writer.close()

            # Reset variables to track training performance
            tr_svf_diff_path = 0
            tr_svf_diff_goal = 0
            tr_time = 0

    # __________________________________________________________________________________________________________________
    # Validate
    # __________________________________________________________________________________________________________________
    print('Calculating validation loss...')

    # Set batchnorm layers to eval mode, stop tracking gradients
    net.eval()
    with torch.no_grad():

        # Variables to track validation performance
        val_svf_diff_path = 0
        val_svf_diff_goal = 0
        val_batch_count = 0

        # Load batch
        for k, data_val in enumerate(val_dl):

            # Process inputs
            _, _, img, svf_e, motion_feats, _, _, _, _, img_vis, _, _, _ = data_val
            svf_e = svf_e.float().to(device)
            img = img.float().to(device)
            motion_feats = motion_feats.float().to(device)

            # Calculate reward over grid using model
            r, _ = net(motion_feats, img)

            # Forward RL (solve for maxent policy and SVF)
            r_detached = r.detach()
            svf, pi = rl.solve(mdp, r_detached, initial_state=initial_state)

            # Calculate difference in state visitation frequencies
            svf = svf.to(device)
            svf_diff = svf - svf_e
            val_svf_diff_path += torch.mean(torch.abs(svf_diff[:, 0, :, :])).item()
            val_svf_diff_goal += torch.mean(torch.abs(svf_diff[:, 1, :, :])).item()
            val_batch_count += 1

            # Log images from first val batch into tensorboard
            if k == 0:
                tb_fig_val = u.tb_reward_plots(img_vis[0:8],
                                               r[0:8].detach().cpu(),
                                               svf[0:8].detach().cpu(),
                                               svf_e[0:8].detach().cpu())
                writer.add_figure('val/SVFs_and_rewards', tb_fig_val, iters)
                writer.close()

    # Print validation losses
    print('Val SVF diff (paths) :', format(val_svf_diff_path / val_batch_count, '0.5f'),
          ', Val SVF diff (goals) :', format(val_svf_diff_goal / val_batch_count, '0.7f'))
    val_loss = val_svf_diff_path / val_batch_count

    # Tensorboard val metrics
    writer.add_scalar('val/SVF_diff_goals', val_svf_diff_goal / val_batch_count, iters)
    writer.add_scalar('val/SVF_diff_paths', val_svf_diff_path / val_batch_count, iters)
    writer.close()

    # Save checkpoint
    if config['opt_r']['save_checkpoints']:
        model_path = config['opt_r']['checkpt_dir'] + '/' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'min_val_loss': min(val_loss, min_val_loss)
        }, model_path)

    # Save best model if applicable
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        model_path = config['opt_r']['checkpt_dir'] + '/best.tar'
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'min_val_loss': min_val_loss
        }, model_path)
