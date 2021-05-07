from __future__ import print_function, division
from torch.utils.data import Dataset
import scipy.io as scp
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as tsfm
import torch

normalize_imagenet = tsfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# Dataset class for the stanford drone dataset
class SDD(Dataset):

    def __init__(self, dataroot, mat_file, t_h=84, t_f=144, grid_dim=25, img_size=200, horizon=40,
                 grid_extent=(-500, 500, -200, 800), num_actions=4):
        self.D = scp.loadmat(dataroot + '/' + mat_file)['D']
        self.T = scp.loadmat(dataroot + '/' + mat_file)['tracks']
        self.imgs = scp.loadmat(dataroot + '/' + mat_file)['imgs']
        self.t_h = int(t_h * 30)  # length of track history
        self.t_f = int(t_f * 30)  # length of predicted trajectory
        self.traj_d_s = 12  # down sampling rate of sequences for trajectories (SDD raw data is 30 Hz,
        #  but prior work uses 2.5 Hz)
        self.grid_dim = grid_dim  # grid dimensions
        self.img_size = img_size  # size of image mapped to grid (images are downsampled by a factor of 5,
        #  so crop size is img_size x 5)
        self.horizon = horizon  # MDP horizon
        self.grid_extent = grid_extent  # The extent of the cropped scene around the agent (l, r, d, u)
        self.num_actions = num_actions  # number of actions in MDP (4 or 8)

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        """
        Outputs returned for each prediction instance:
        hist: snippet of track history
        fut: ground truth future trajectory
        img: Imagenet normalized bird's eye view map around the target vehicle
        svf_e: Goal and path state visitation frequencies for expert demonstration, ie. path from train set
        motion_feats: motion and position features used for reward model
        waypts_e: (x,y) BEV co-ordinates corresponding to grid cells of svf_e
        agents: tensor of surrounding agent states populated in grid around target agent
        grid_idcs: grid co-ordinates of svf_e
        bc_targets: ground truth actions for training behavior cloning model
        img_vis: Raw image (unnormalized) for visualization
        ref_pos: global coordinates of agent's current location (helpful for computing offroad rate)
        ds_id: SDD scene id (helpful for computing offroad rate)
        idx: instance id (mainly for debugging)
        """

        # Dataset (scene) id
        ds_id = self.D[idx, 0].astype(int)

        # Track id
        t_id = self.D[idx, 1].astype(int)

        # Frame number
        frame = self.D[idx, 2]

        # Orientation of agent (based on instantaneous velocity)
        theta = self.D[idx, 4]

        # Trajectory history
        hist, ref_pos = self.get_history(t_id, frame, ds_id, -theta)
        hist = torch.from_numpy(hist)
        ref_pos = torch.from_numpy(ref_pos)

        # True future trajectory
        fut_indefinite = self.get_future(t_id, frame, ds_id, -theta, fixed_horizon=False)
        fut_definite = self.get_future(t_id, frame, ds_id, -theta, fixed_horizon=True)
        fut_definite = fut_definite[1:, :]
        fut = torch.from_numpy(fut_definite)

        # Cropped image around current position
        imgret = self.get_img(t_id, frame, ds_id, 90-theta)
        img_vis = tsfm.ToTensor()(imgret)  # Raw image for visualization of results
        img = tsfm.ToTensor()(imgret)
        img = normalize_imagenet(img)  # Imagenet normalized image, used for training

        # surrounding agent states
        agents = torch.zeros(3, self.grid_dim, self.grid_dim)
        # TODO

        # get expert SVF:
        svf_e, waypts_e, grid_idcs = self.get_expert_waypoints(fut_indefinite)
        svf_e = torch.from_numpy(svf_e)
        waypts_e = torch.from_numpy(waypts_e)
        grid_idcs = torch.from_numpy(grid_idcs)

        # get motion and position feats:
        motion_feats = self.get_motion_feats(hist)
        motion_feats = torch.from_numpy(motion_feats)

        # Targets for behavior cloning model:
        bc_targets = self.get_bc_targets(fut_indefinite, self.num_actions)
        bc_targets = torch.from_numpy(bc_targets)

        return hist, fut, img, svf_e, motion_feats, waypts_e, agents, grid_idcs, bc_targets, img_vis, \
            ref_pos, ds_id, idx

    def get_future(self, t_id, t, ds_id, theta, fixed_horizon=False):
        """
        helper function to get future trajectory:
        """
        track = self.T[ds_id - 1][t_id]
        ref_pos = track[np.where(track[:, 0] == t)][0][1:3]
        stpt = np.argwhere(track[:, 0] == t).item()
        if fixed_horizon:
            enpt = np.argwhere(track[:, 0] == t).item() + self.t_f + 1
            fut = track[stpt:enpt:self.traj_d_s, 1:3] - ref_pos
        else:
            fut = track[stpt:, 1:3] - ref_pos
        r_mat = np.empty([2, 2])
        r_mat[0, 0] = np.cos(np.pi * theta / 180)
        r_mat[0, 1] = np.sin(np.pi * theta / 180)
        r_mat[1, 0] = -np.sin(np.pi * theta / 180)
        r_mat[1, 1] = np.cos(np.pi * theta / 180)
        fut = np.matmul(r_mat, fut.transpose()).transpose()
        fut = fut[:, [1, 0]]
        return fut
    
    def get_history(self, t_id, t, ds_id, theta):
        """
        helper function to get track history:
        """
        track = self.T[ds_id - 1][t_id]
        ref_pos = track[np.where(track[:, 0] == t)][0][1:3]
        stpt = np.argwhere(track[:, 0] == t).item() - self.t_h
        enpt = np.argwhere(track[:, 0] == t).item() + 1
        hist = track[stpt:enpt:self.traj_d_s, 1:3] - ref_pos
        r_mat = np.empty([2, 2])
        r_mat[0, 0] = np.cos(np.pi * theta / 180)
        r_mat[0, 1] = np.sin(np.pi * theta / 180)
        r_mat[1, 0] = -np.sin(np.pi * theta / 180)
        r_mat[1, 1] = np.cos(np.pi * theta / 180)
        hist = np.matmul(r_mat, hist.transpose()).transpose()
        hist = hist[:, [1, 0]]
        ref_pos = ref_pos[[1, 0]]
        ref_pos = np.append(ref_pos, -theta)
        return hist, ref_pos

    def get_img(self, t_id, t, ds_id, theta):
        """
        helper function to get cropped scene around agent being predicted
        """
        img = Image.fromarray(self.imgs[0][ds_id - 1])
        padsize = int(self.img_size)
        img_expanded = ImageOps.expand(img, (padsize, padsize, padsize, padsize))
        img_rotated = img_expanded.rotate(theta, expand=1)
        track = self.T[ds_id - 1][t_id]
        ref_pos = track[np.where(track[:, 0] == t)][0][1:3]
        x0 = ref_pos[0] / 5
        y0 = ref_pos[1] / 5
        cx = img.size[0] / 2
        cy = img.size[1] / 2
        x1 = x0 - cx
        y1 = y0 - cy
        r_mat = np.empty([2, 2])
        r_mat[0, 0] = np.cos(np.pi * theta / 180)
        r_mat[0, 1] = np.sin(np.pi * theta / 180)
        r_mat[1, 0] = -np.sin(np.pi * theta / 180)
        r_mat[1, 1] = np.cos(np.pi * theta / 180)
        x_1 = np.empty([2, 1])
        x_1[0] = x1
        x_1[1] = y1
        x_2 = np.matmul(r_mat, x_1)
        x2 = x_2[0]
        y2 = x_2[1]
        cex = img_rotated.size[0] / 2
        cey = img_rotated.size[1] / 2
        x3 = cex + x2
        y3 = cey + y2
        x3 = int(x3)
        y3 = int(y3)
        img_cropped = img_rotated.crop(
            (x3 - self.img_size / 2, y3 - self.img_size / 2, x3 + self.img_size / 2,
             y3 + self.img_size / 2))
        return img_cropped

    def get_expert_waypoints(self, fut):
        """
        Helper function to get the expert's state visitation frequencies based on their trajectory
        """
        grid_extent = self.grid_extent
        svf_e = np.zeros((2, self.grid_dim, self.grid_dim))
        waypts_e = np.zeros((self.horizon, 2))
        grid_idcs = np.zeros((self.horizon, 2))
        count = 0
        row_prev = np.nan
        column_prev = np.nan
        grid_size = grid_extent[1] - grid_extent[0]
        row_centers = np.linspace(grid_extent[3] - grid_size/(self.grid_dim*2), grid_extent[2] +
                                  grid_size/(self.grid_dim*2), self.grid_dim)
        col_centers = np.linspace(grid_extent[0] + grid_size/(self.grid_dim*2), grid_extent[1] -
                                  grid_size/(self.grid_dim*2), self.grid_dim)

        for k in range(fut.shape[0]):
            # Convert trajectory (x,y) co-ordinates to grid locations:
            column = np.argmin(np.absolute(fut[k, 0] - col_centers))
            row = np.argmin(np.absolute(fut[k, 1] - row_centers))
            # Demonstration ends when expert leaves the image crop corresponding to the grid:
            if grid_extent[0] <= fut[k, 0] <= grid_extent[1] and \
                    grid_extent[2] <= fut[k, 1] <= grid_extent[3]:
                # Check if cell location has changed
                if row != row_prev or column != column_prev:
                    # Add cell location to path states of expert
                    svf_e[0, row.astype(int), column.astype(int)] = 1
                    if count < self.horizon:
                        # Get BEV coordinates corresponding to cell locations
                        waypts_e[count, 0] = row_centers[row]
                        waypts_e[count, 1] = col_centers[column]
                        grid_idcs[count, 0] = row
                        grid_idcs[count, 1] = column
                        count += 1
            else:
                break
            column_prev = column
            row_prev = row

        # Last cell location where demonstration terminates is the goal state:
        svf_e[1, row_prev.astype(int), column_prev.astype(int)] = 1
        return svf_e, waypts_e, grid_idcs

    def get_motion_feats(self, hist):
        """
        Helper function to get motion and position features over grid for reward model
        """
        grid_extent = self.grid_extent
        feats = np.zeros((3, self.grid_dim, self.grid_dim))
        grid_size = grid_extent[1] - grid_extent[0]
        y = (np.linspace(grid_extent[3] - grid_size/(self.grid_dim*2), grid_extent[2] +
                         grid_size/(self.grid_dim*2), self.grid_dim)).reshape(-1, 1).repeat(self.grid_dim, axis=1)
        x = (np.linspace(grid_extent[0] + grid_size/(self.grid_dim*2), grid_extent[1] -
                         grid_size/(self.grid_dim*2), self.grid_dim)).reshape(-1, 1).repeat(self.grid_dim,
                                                                                            axis=1).transpose()
        feats[0] = ((hist[-1, 0] - hist[-2, 0]) ** 2 + (hist[-1, 1] - hist[-2, 1]) ** 2) ** 0.5
        feats[1] = x/grid_size
        feats[2] = y/grid_size

        return feats

    def get_bc_targets(self, fut, num_actions):
        """
        Helper function to get targets for behavior cloning model
        """
        grid_extent = self.grid_extent
        bc_targets = np.zeros((num_actions + 1, self.grid_dim, self.grid_dim))
        grid_size = grid_extent[1] - grid_extent[0]
        row_centers = np.linspace(grid_extent[3] - grid_size / (self.grid_dim * 2), grid_extent[2] +
                                  grid_size / (self.grid_dim * 2), self.grid_dim)
        col_centers = np.linspace(grid_extent[0] + grid_size / (self.grid_dim * 2), grid_extent[1] -
                                  grid_size / (self.grid_dim * 2), self.grid_dim)
        row_prev = np.argmin(np.absolute(fut[0, 1] - row_centers))
        column_prev = np.argmin(np.absolute(fut[0, 0] - col_centers))

        for k in range(fut.shape[0]):
            # Convert trajectory (x,y) co-ordinates to grid locations:
            column = np.argmin(np.absolute(fut[k, 0] - col_centers))
            row = np.argmin(np.absolute(fut[k, 1] - row_centers))
            # Demonstration ends when expert leaves the image crop corresponding to the grid:
            if grid_extent[0] <= fut[k, 0] <= grid_extent[1] and grid_extent[2] <= fut[k, 1] <= \
                    grid_extent[3]:
                # Check if cell location has changed
                if row != row_prev or column != column_prev:
                    bc_targets[:, int(row_prev), int(column_prev)] = 0
                    d_x = column - column_prev
                    d_y = row - row_prev
                    theta = np.arctan2(d_y, d_x)
                    # Assign ground truth actions for expert demonstration
                    if num_actions == 4:  # [D,R,U,L,end]
                        if np.pi / 4 <= theta < 3 * np.pi / 4:
                            bc_targets[0, int(row_prev), int(column_prev)] = 1
                        elif -np.pi / 4 <= theta < np.pi / 4:
                            bc_targets[1, int(row_prev), int(column_prev)] = 1
                        elif -3 * np.pi / 4 <= theta < -np.pi / 4:
                            bc_targets[2, int(row_prev), int(column_prev)] = 1
                        else:
                            bc_targets[3, int(row_prev), int(column_prev)] = 1
                    else:  # [D, R, U, L, DR, UR, DL, UL, end]
                        if 3 * np.pi / 8 <= theta < 5 * np.pi / 8:
                            bc_targets[0, int(row_prev), int(column_prev)] = 1
                        elif -np.pi / 8 <= theta < np.pi / 8:
                            bc_targets[1, int(row_prev), int(column_prev)] = 1
                        elif -5 * np.pi / 8 <= theta < -3 * np.pi / 8:
                            bc_targets[2, int(row_prev), int(column_prev)] = 1
                        elif np.pi / 8 <= theta < 3 * np.pi / 8:
                            bc_targets[4, int(row_prev), int(column_prev)] = 1
                        elif -3 * np.pi / 8 <= theta < -np.pi / 8:
                            bc_targets[5, int(row_prev), int(column_prev)] = 1
                        elif 5 * np.pi / 8 <= theta < 7 * np.pi / 8:
                            bc_targets[6, int(row_prev), int(column_prev)] = 1
                        elif -7 * np.pi / 8 <= theta < -5 * np.pi / 8:
                            bc_targets[7, int(row_prev), int(column_prev)] = 1
                        else:
                            bc_targets[3, int(row_prev), int(column_prev)] = 1
            else:
                break
            column_prev = column
            row_prev = row

        # Final action is the end action to transition to the goal state:
        bc_targets[num_actions, int(row_prev), int(column_prev)] = 1
        return bc_targets

    def collate_fn(self, samples):
        """
        Collate function to get rid of stationary agents while training reward model:
        """
        batch_size = 0
        for _, _, _, svf_e, _, _, _, _, _, _, _, _, _ in samples:
            if svf_e[1, int((self.grid_dim - 1) / 2), int((self.grid_dim - 1) / 2)] == 0:
                batch_size += 1

        hist_batch = torch.zeros(batch_size, samples[0][0].shape[0], samples[0][0].shape[1])
        fut_batch = torch.zeros(batch_size, samples[0][1].shape[0], samples[0][1].shape[1])
        img_batch = torch.zeros(batch_size, samples[0][2].shape[0], samples[0][2].shape[1], samples[0][2].shape[2])
        svf_e_batch = torch.zeros(batch_size, samples[0][3].shape[0], samples[0][3].shape[1],
                                  samples[0][3].shape[2])
        motion_feats_batch = torch.zeros(batch_size, samples[0][4].shape[0], samples[0][4].shape[1],
                                         samples[0][4].shape[2])
        waypts_e_batch = torch.zeros(batch_size, samples[0][5].shape[0], samples[0][5].shape[1])
        agents_batch = torch.zeros(batch_size, samples[0][6].shape[0], samples[0][6].shape[1], samples[0][6].shape[2])
        grid_idcs_batch = torch.zeros(batch_size, samples[0][7].shape[0], samples[0][7].shape[1])
        bc_targets_batch = torch.zeros(batch_size, samples[0][8].shape[0], samples[0][8].shape[1],
                                       samples[0][8].shape[2])
        img_vis_batch = torch.zeros(batch_size, samples[0][9].shape[0], samples[0][9].shape[1], samples[0][9].shape[2])
        ref_pos_batch = torch.zeros(batch_size, samples[0][10].shape[0])
        ds_id_batch = torch.zeros(batch_size)
        idx_batch = torch.zeros(batch_size)

        count = 0
        for sampleId, (hist, fut, img, svf_e, motion_feats, waypts_e, agents, grid_idcs, bc_targets, img_vis, ref_pos,
                       ds_id, idx) in enumerate(samples):
            if svf_e[1, int((self.grid_dim - 1) / 2), int((self.grid_dim - 1) / 2)] == 0:
                hist_batch[count, :, :] = hist
                fut_batch[count, :, :] = fut
                img_batch[count, :, :, :] = img
                svf_e_batch[count, :, :, :] = svf_e
                motion_feats_batch[count, :, :, :] = motion_feats
                waypts_e_batch[count, :, :] = waypts_e
                agents_batch[count, :, :, :] = agents
                grid_idcs_batch[count, :, :] = grid_idcs
                bc_targets_batch[count, :, :, :] = bc_targets
                img_vis_batch[count, :, :, :] = img_vis
                ref_pos_batch[count, :] = ref_pos
                ds_id_batch[count] = float(ds_id)
                idx_batch[count] = float(idx)
                count += 1

        return hist_batch, fut_batch, img_batch, svf_e_batch, motion_feats_batch, waypts_e_batch, agents_batch,\
            grid_idcs_batch, bc_targets_batch, img_vis_batch, ref_pos_batch, ds_id_batch, idx_batch
