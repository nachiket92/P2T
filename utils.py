import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np


def get_plan_feats(plans, scene_tensor, agent_tensor):
    """
    Returns location coordinates, map and agent features for a given batch of plans

    Inputs
    plans: Sequences of row and column values on grid. shape: (Batchsize, horizon, 2)
    scene_tensor: Tensor of scene features: (Batchsize, C_s, H, W)
    agent_tensor: Tensor of agent features: (Batchsize, C_a, H, W)

    Output
    scene_feats: Scene features along plan (Batchsize, horizon, C_s)
    agent_feats: Agent features along plan (Batchsize, horizon, C_a)
    """
    h = scene_tensor.shape[2]
    scene_tensor = scene_tensor.reshape(scene_tensor.shape[0], scene_tensor.shape[1], -1)
    agent_tensor = agent_tensor.reshape(agent_tensor.shape[0], agent_tensor.shape[1], -1)
    plans = plans[:, :, 0] * h + plans[:, :, 1]
    plans_s = plans[:, None, :].repeat(1, scene_tensor.shape[1], 1).long()
    plans_a = plans[:, None, :].repeat(1, agent_tensor.shape[1], 1).long()
    scene_feats = torch.gather(scene_tensor, 2, plans_s)
    agent_feats = torch.gather(agent_tensor, 2, plans_a)
    scene_feats = scene_feats.permute(0, 2, 1)
    agent_feats = agent_feats.permute(0, 2, 1)
    return scene_feats, agent_feats


@ignore_warnings(category=ConvergenceWarning)
def km_cluster(data, num_clusters):
    """
    Performs K-means clustering on a set of sampled trajectories
    """
    clustering = KMeans(num_clusters, n_init=1, max_iter=100).fit(data)
    return clustering.labels_


def min_ade_k(y_pred, y_gt, masks):
    """
    minADE_k loss for cases where k can vary across a batch.

    Inputs
    y_pred: Predicted trajectories, Tensor shape: (Batchsize, maxK, prediction horizon, 2).
     Includes dummy values when K< maxK
    y_gt: Ground truth trajectory, Tensor shape: (Batchsize, prediction horizon, 2)
    masks: 0 or inf values depending on value of K for each sample in the batch, Tensor shape: (Batchsize, maxK)

    Output
    loss: minADE_k loss for batch
    """
    y_gt = y_gt.reshape([y_gt.shape[0], 1, y_gt.shape[1], y_gt.shape[2]])
    y_gt_repeated = y_gt.repeat([1, y_pred.shape[1], 1, 1])
    loss = torch.pow(y_gt_repeated - y_pred[:, :, :, 0:2], 2)
    loss = torch.sum(loss, 3)
    loss = torch.pow(loss, 0.5)
    loss = torch.mean(loss, 2) + masks
    loss, ids = torch.min(loss, 1)
    loss = torch.mean(loss)
    return loss


def min_fde_k(y_pred, y_gt, masks, all_timestamps=False):
    """
    minFDE_k loss for cases where k can vary across a batch.

    Inputs
    y_pred: Predicted trajectories, Tensor shape: (Batchsize, maxK, prediction horizon, 2).
     Includes dummy values when K< maxK
    y_gt: Ground truth trajectory, Tensor shape: (Batchsize, prediction horizon, 2)
    masks: 0 or inf values depending on value of K for each sample in the batch, Tensor shape: (Batchsize, maxK)
    all_timestamps: Flag, if true, returns displacement error for each timestamp over prediction horizon,
    for best of k FDE trajectory

    Output
    l: minFDE_k loss for batch
    """
    y_gt = y_gt.reshape([y_gt.shape[0], 1, y_gt.shape[1], y_gt.shape[2]])
    y_gt_last = y_gt[:, :, y_gt.shape[2] - 1, :]
    y_pred_last = y_pred[:, :, y_pred.shape[2] - 1, :]
    y_gt_last_repeated = y_gt_last.repeat([1, y_pred_last.shape[1], 1])
    loss = torch.pow(y_gt_last_repeated - y_pred_last[:, :, 0:2], 2)
    loss = torch.sum(loss, 2)
    loss = torch.pow(loss, 0.5) + masks
    loss, ids = torch.min(loss, 1)
    loss = torch.mean(loss)
    if all_timestamps:
        ids = ids.repeat(1, y_pred.shape[2], y_pred.shape[3], 1)
        ids = ids.permute(3, 0, 1, 2)
        y_pred_best = y_pred.gather(1, ids)
        loss = torch.pow(y_gt - y_pred_best[:, :, :, 0:2], 2)
        loss = torch.sum(loss, 3)
        loss = torch.pow(loss, 0.5)
        loss = torch.squeeze(loss)
        loss = torch.mean(loss, 0)
        return loss
    else:
        return loss


def sdd_local2global(traj, ref_pos):
    """
    Transforms trajectory to global coordinates for SDD
    """

    # Flip
    traj = traj[:, [1, 0]]
    ref_pos = ref_pos[[1, 0, 2]]

    # Rotate
    theta = ref_pos[2]
    r_mat = torch.zeros(2, 2)
    r_mat[0, 0] = np.cos(np.pi * theta / 180)
    r_mat[0, 1] = np.sin(np.pi * theta / 180)
    r_mat[1, 0] = -np.sin(np.pi * theta / 180)
    r_mat[1, 1] = np.cos(np.pi * theta / 180)
    traj = torch.mm(r_mat, traj.t()).t()

    # Translate
    traj = traj + ref_pos[:2]

    return traj


def offroad_rate(y_pred, img_lbls, ref_pos, ds_ids, y_gt, masks, all_timestamps=False):
    """
    Computes offroad rate for Stanford drone dataset

    Inputs
    y_pred, y_gt, all_timestamps: Similar to minADE_k and minFDE_k functions
    img_lbls: path/obstacle labels, binary images from SDD
    ref_pos: global co-ordinates of agent location at the time of prediction, for each instance in the batch
    dsIds: scene Ids for each instance in the batch

    Output
    offroad rate for batch
    """

    # Transform to global co-ordinates
    y_gt_global = torch.zeros_like(y_gt)
    y_pred_global = torch.zeros_like(y_pred)
    for k in range(y_pred.shape[0]):

        # Transform ground_truth
        y_gt_global[k] = sdd_local2global(y_gt[k].cpu(), ref_pos[k].cpu())

        # Transform predictions
        for n in range(y_pred.shape[1]):
            y_pred_global[k, n] = sdd_local2global(y_pred[k, n].cpu(), ref_pos[k].cpu())

    # Compute offroad rate
    num_path = torch.zeros(y_pred.shape[2])
    counts = torch.zeros(y_pred.shape[2])
    for k in range(y_pred.shape[0]):
        lbl_img = img_lbls[0][ds_ids[k] - 1]

        for n in range(y_pred.shape[1]):
            for m in range(y_pred.shape[2]):

                row = int(y_pred_global[k, n, m, 1].item())
                col = int(y_pred_global[k, n, m, 0].item())
                row_gt = int(y_gt_global[k, m, 1].item())
                col_gt = int(y_gt_global[k, m, 0].item())

                # If mask is 0:
                if masks[k, n] == 0:
                    # If ground truth future location is on a path and within the image boundaries:
                    if row_gt < lbl_img.shape[0] and col_gt < lbl_img.shape[1]:
                        if lbl_img[row_gt, col_gt]:
                            counts[m] += 1
                            # If predicted location is on a path and within image boundaries:
                            if row < lbl_img.shape[0] and col < lbl_img.shape[1]:
                                if lbl_img[row, col]:
                                    num_path[m] += 1
    if all_timestamps:
        return torch.ones_like(num_path) - num_path / counts
    else:
        return torch.tensor(1) - torch.sum(num_path) / torch.sum(counts)


def tb_reward_plots(img_vis, r, svf, svf_e):
    """
    Returns matplotlib figure showing rewards and SVFs for visualizing reward model training progress
    """
    fig, ax = plt.subplots(r.shape[0], 7, figsize=(15, 15))
    for i in range(r.shape[0]):
        ax[i, 0].imshow(img_vis[i].permute(1, 2, 0))
        ax[i, 1].imshow(r[i, 0], cmap='viridis')
        ax[i, 2].imshow(r[i, 1], cmap='viridis')
        ax[i, 3].imshow(svf[i, 0], cmap='viridis')
        ax[i, 4].imshow(svf[i, 1], cmap='viridis')
        ax[i, 5].imshow(svf_e[i, 0], cmap='viridis')
        ax[i, 6].imshow(svf_e[i, 1], cmap='viridis')
    return fig


def tb_traj_pt_plots(img_vis, hist, traj, fut, svf_e, extent):
    """
    Returns matplotlib figure showing trajectory conditioned on ground truth plan.
    Helps visualize pre-training progress for trajectory generator
    """
    fig, ax = plt.subplots(2, img_vis.shape[0], figsize=(16, 5))
    for i in range(img_vis.shape[0]):
        ax[0, i].imshow(img_vis[i].permute(1, 2, 0), extent=extent)
        ax[0, i].plot(traj[i, :, 0], traj[i, :, 1], color='r', lw=0.5, marker='o',
                      markeredgecolor='r', markersize=2, alpha=0.8)
        ax[0, i].plot(fut[i, :, 0], fut[i, :, 1], color='k', lw=0.5, marker='o',
                      markeredgecolor='k', markersize=2, alpha=1)
        ax[0, i].plot(hist[i, :, 0], hist[i, :, 1], color='k', lw=0.5, marker='s',
                      markeredgecolor='k', markersize=2, alpha=1)
        ax[1, i].imshow(svf_e[i, 0], cmap='viridis', extent=extent)
    return fig


def tb_traj_ft_plots(img_vis, hist, traj, fut, svf, counts, extent):
    """
    Returns matplotlib figure showing trajectories conditioned on sampled plans along with SVFs for maxEnt policy.
    Helps visualize fine-tuning progress for trajectory generator
    """
    fig, ax = plt.subplots(3, img_vis.shape[0], figsize=(16, 5))
    for i in range(img_vis.shape[0]):
        ax[0, i].imshow(img_vis[i].permute(1, 2, 0), extent=extent)
        ax[0, i].plot(fut[i, :, 0], fut[i, :, 1], color='k', lw=0.5, marker='o',
                      markeredgecolor='k', markersize=2, alpha=1)
        ax[0, i].plot(hist[i, :, 0], hist[i, :, 1], color='k', lw=0.5, marker='s',
                      markeredgecolor='k', markersize=2, alpha=1)
        for n in range(torch.sum(counts[i] != 0).item()):
            ax[0, i].plot(traj[i, n, :, 0], traj[i, n, :, 1], color='r', lw=0.5, marker='o',
                          markeredgecolor='r', markersize=2, alpha=0.8)
        ax[1, i].imshow(svf[i, 0], cmap='viridis', extent=extent)
        ax[2, i].imshow(svf[i, 1], cmap='viridis', extent=extent)
    return fig
