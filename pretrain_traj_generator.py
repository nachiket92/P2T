from __future__ import print_function
import torch
from torch.utils.data import DataLoader
from models.traj_generator import TrajGenerator
from models.reward_model import RewardModel
import time
import math
from utils import get_plan_feats
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
writer = SummaryWriter(log_dir=config['opt_t']['log_dir'])


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
                   batch_size=config['opt_t']['batch_size'],
                   shuffle=True,
                   num_workers=config['num_workers'])

val_dl = DataLoader(val_set,
                    batch_size=config['opt_t']['batch_size'],
                    shuffle=True,
                    num_workers=config['num_workers'])


# Initialize Models:
net_r = RewardModel(config['args_r']).float().to(device)
net_r.load_state_dict(torch.load(config['opt_r']['checkpt_dir'] + '/' + 'best.tar')['model_state_dict'])
for param in net_r.parameters():
    param.requires_grad = False
net_r.eval()

net_t = TrajGenerator(config['args_t']).float().to(device)
loss = torch.nn.MSELoss()


# Initialize Optimizer:
num_epochs = config['opt_t']['num_epochs']
optimizer = torch.optim.Adam(net_t.parameters(), lr=config['opt_t']['lr'])


# Load checkpoint if specified in config:
if config['opt_t']['load_checkpt']:
    checkpoint = torch.load(config['opt_t']['checkpt_path'])
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
iters_epoch = len(tr_set) // config['opt_t']['batch_size']
iters = (start_epoch - 1) * iters_epoch

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
        hist, fut, img, svf_e, motion_feats, waypts_e, agents, grid_idcs, _, img_vis, _, _, _ = data
        img = img.float().to(device)
        motion_feats = motion_feats.float().to(device)
        agents = agents.float().to(device)
        hist = hist.permute(1, 0, 2).float().to(device)
        fut = fut.float().to(device)
        waypts_e = waypts_e.permute(1, 0, 2).float().to(device)
        grid_idcs = grid_idcs.float().to(device)

        # Get scene and agent features:
        _, scene_tensor = net_r(motion_feats, img)
        scene_feats, agent_feats = get_plan_feats(grid_idcs, scene_tensor, agents)
        scene_feats = scene_feats.permute(1, 0, 2)
        agent_feats = agent_feats.permute(1, 0, 2)

        # Forward pass:
        op_traj = net_t(hist, waypts_e, scene_feats, agent_feats)
        l_batch = loss(op_traj, fut)

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
        iters_log = config['opt_t']['steps_to_log_train_loss']
        if i % iters_log == iters_log - 1:
            eta = tr_time / iters_log * (len(tr_set) / config['opt_t']['batch_size'] - i)
            print("Epoch no:", epoch,
                  "| Epoch progress(%):", format(i / (len(tr_set) / config['opt_t']['batch_size']) * 100, '0.2f'),
                  "| Train loss:", format(tr_loss / iters_log, '0.5f'),
                  "| Val loss prev epoch", format(val_loss, '0.5f'),
                  "| Min val loss", format(min_val_loss, '0.5f'),
                  "| ETA(s):", int(eta))

            # Log images from train batch into tensorboard:
            tb_fig_train = u.tb_traj_pt_plots(img_vis[0:8],
                                              hist[:, 0:8, :].permute(1, 0, 2).detach().cpu(),
                                              op_traj[0:8].detach().cpu(),
                                              fut[0:8].detach().cpu(),
                                              svf_e[0:8],
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
            waypts_e = waypts_e.permute(1, 0, 2).float().to(device)
            grid_idcs = grid_idcs.float().to(device)

            # Get scene and agent features:
            _, scene_tensor = net_r(motion_feats, img)
            scene_feats, agent_feats = get_plan_feats(grid_idcs, scene_tensor, agents)
            scene_feats = scene_feats.permute(1, 0, 2)
            agent_feats = agent_feats.permute(1, 0, 2)

            # Forward pass:
            op_traj = net_t(hist, waypts_e, scene_feats, agent_feats)
            l_batch = loss(op_traj, fut)
            agg_val_loss += l_batch.item()
            val_batch_count += 1

            # Log images from first val batch into tensorboard
            if k == 0:
                tb_fig_val = u.tb_traj_pt_plots(img_vis[0:8],
                                                hist[:, 0:8, :].permute(1, 0, 2).detach().cpu(),
                                                op_traj[0:8].detach().cpu(),
                                                fut[0:8].detach().cpu(),
                                                svf_e[0:8],
                                                extent=config['grid_extent'])
                writer.add_figure('val/trajectories', tb_fig_val, iters)
                writer.close()

    # Print validation losses
    print('Val loss :', format(agg_val_loss / val_batch_count, '0.5f'))
    val_loss = agg_val_loss / val_batch_count

    # Tensorboard validation metrics
    writer.add_scalar('val/loss', val_loss, iters)
    writer.close()

    # Save checkpoint
    if config['opt_t']['save_checkpoints']:
        model_path = config['opt_t']['checkpt_dir'] + '/' + str(epoch) + '.tar'
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
        model_path = config['opt_t']['checkpt_dir'] + '/best.tar'
        torch.save({
            'epoch': epoch,
            'model_state_dict': net_t.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'min_val_loss': min_val_loss
        }, model_path)
