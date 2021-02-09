from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as tsfm
import torch
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation, AgentRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer
from nuscenes.prediction.input_representation.utils import convert_to_pixel_coords, get_crops, get_rotation_matrix
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction.helper import quaternion_yaw
from pyquaternion import Quaternion
from PIL import Image
import cv2
import torch.nn.functional as f
from typing import Any, Dict, List, Tuple
import os


normalize_imagenet = tsfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class NS(Dataset):

    def __init__(self,
                 dataroot: str,
                 split: str,
                 t_h: float = 2,
                 t_f: float = 6,
                 grid_dim: int = 25,
                 img_size: int = 200,
                 horizon: int = 40,
                 grid_extent: Tuple[int, int, int, int] = (-25, 25, -10, 40),
                 num_actions: int = 4,
                 image_extraction_mode: bool = False):
        """
        Initializes dataset class for nuScenes prediction

        :param dataroot: Path to tables and data
        :param split: Dataset split for prediction benchmark ('train'/'train_val'/'val')
        :param t_h: Track history in seconds
        :param t_f: Prediction horizon in seconds
        :param grid_dim: Size of grid, default: 25x25
        :param img_size: Size of raster map image in pixels, default: 200x200
        :param horizon: MDP horizon
        :param grid_extent: Map extents in meters, (-left, right, -behind, front)
        :param num_actions: Number of actions for each state (4: [D,R,U,L] or 8: [D, R, U, L, DR, UR, DL, UL])
        :param image_extraction_mode: Whether dataset class is being used for image extraction
        """

        # Nuscenes dataset and predict helper
        self.dataroot = dataroot
        self.ns = NuScenes('v1.0-trainval', dataroot=dataroot)
        self.helper = PredictHelper(self.ns)
        self.token_list = get_prediction_challenge_split(split, dataroot=dataroot)

        # Useful parameters
        self.grid_dim = grid_dim
        self.grid_extent = grid_extent
        self.img_size = img_size
        self.t_f = t_f
        self.t_h = t_h
        self.horizon = horizon
        self.num_actions = num_actions

        # Map row, column and velocity states to actual values
        grid_size_m = self.grid_extent[1] - self.grid_extent[0]
        self.row_centers = np.linspace(self.grid_extent[3] - grid_size_m / (self.grid_dim * 2),
                                       self.grid_extent[2] + grid_size_m / (self.grid_dim * 2),
                                       self.grid_dim)

        self.col_centers = np.linspace(self.grid_extent[0] + grid_size_m / (self.grid_dim * 2),
                                       self.grid_extent[1] - grid_size_m / (self.grid_dim * 2),
                                       self.grid_dim)

        # Surrounding agent input representation: populate grid with velocity, acc, yaw-rate
        self.agent_ip = AgentMotionStatesOnGrid(self.helper,
                                                resolution=grid_size_m / img_size,
                                                meters_ahead=grid_extent[3],
                                                meters_behind=-grid_extent[2],
                                                meters_left=-grid_extent[0],
                                                meters_right=grid_extent[1])

        # Image extraction mode is used for extracting map images offline prior to training
        self.image_extraction_mode = image_extraction_mode
        if self.image_extraction_mode:

            # Raster map representation
            self.map_ip = StaticLayerRasterizer(self.helper,
                                                resolution=grid_size_m / img_size,
                                                meters_ahead=grid_extent[3],
                                                meters_behind=-grid_extent[2],
                                                meters_left=-grid_extent[0],
                                                meters_right=grid_extent[1])

            # Raster map with agent boxes. Only used for visualization
            static_layer_rasterizer = StaticLayerRasterizer(self.helper,
                                                            resolution=grid_size_m / img_size,
                                                            meters_ahead=grid_extent[3],
                                                            meters_behind=-grid_extent[2],
                                                            meters_left=-grid_extent[0],
                                                            meters_right=grid_extent[1])

            agent_rasterizer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=1,
                                                          resolution=grid_size_m / img_size,
                                                          meters_ahead=grid_extent[3],
                                                          meters_behind=-grid_extent[2],
                                                          meters_left=-grid_extent[0],
                                                          meters_right=grid_extent[1])

            self.map_ip_agents = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

    def __len__(self):
        return len(self.token_list)

    def __getitem__(self, idx):
        """
        Returns inputs, ground truth values and other utilities for data point at given index

        :return hist: snippet of track history, default 2s at 0.5 Hz sampling frequency
        :return fut: ground truth future trajectory, default 6s at 0.5 Hz sampling frequency
        :return img: Imagenet normalized bird's eye view map around the target vehicle
        :return svf_e: Goal and path state visitation frequencies for expert demonstration, ie. path from train set
        :return motion_feats: motion and position features used for reward model
        :return waypts_e: (x,y) BEV co-ordinates corresponding to grid cells of svf_e
        :return agents: tensor of surrounding agent states populated in grid around target agent
        :return grid_idcs: grid co-ordinates of svf_e
        :return bc_targets: ground truth actions for training behavior cloning model
        :return img_agents: image with agent boxes for visualization / debugging
        :return instance_token: nuScenes instance token for prediction instance
        :return sample_token: nuScenes sample token for prediction instance
        :return idx: instance id (mainly for debugging)
        """

        # Nuscenes instance and sample token for prediction data point
        instance_token, sample_token = self.token_list[idx].split("_")

        # If dataset is being used for image extraction
        grid_size_m = self.grid_extent[1] - self.grid_extent[0]
        if self.image_extraction_mode:

            # Make directory to store raster map images
            img_dir = os.path.join(self.dataroot, 'prediction_raster_maps',
                                   'images' + str(self.img_size) + "_" + str(int(grid_size_m)) + 'm')
            if not os.path.isdir(img_dir):
                os.mkdir(img_dir)

            # Generate and save raster map image with just static elements
            img = self.map_ip.make_representation(instance_token, sample_token)
            img_save = Image.fromarray(img)
            img_save.save(os.path.join(img_dir, instance_token + "_" + sample_token + '.png'))

            # Generate and save raster map image with static elements and agent boxes (for visualization only)
            img_agents = self.map_ip_agents.make_input_representation(instance_token, sample_token)
            img_agents_save = Image.fromarray(img_agents)
            img_agents_save.save(os.path.join(img_dir, instance_token + "_" + sample_token + 'agents.png'))

            # Return dummy values
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        # If dataset is being used for training/validation/testing
        else:

            # Get track history for agent:
            hist = self.get_hist(instance_token, sample_token)
            hist = torch.from_numpy(hist)

            # Get ground truth future for agent:
            fut = self.helper.get_future_for_agent(instance_token,
                                                   sample_token,
                                                   seconds=self.t_f,
                                                   in_agent_frame=True)
            fut = torch.from_numpy(fut)

            # Get indefinite future for computing expert State visitation frequencies (SVF):
            fut_indefinite = self.helper.get_future_for_agent(instance_token,
                                                              sample_token,
                                                              seconds=300,
                                                              in_agent_frame=True)

            # Up sample indefinite future by a factor of 10
            fut_interpolated = np.zeros((fut_indefinite.shape[0] * 10 + 1, 2))
            param_query = np.linspace(0, fut_indefinite.shape[0], fut_indefinite.shape[0] * 10 + 1)
            param_given = np.linspace(0, fut_indefinite.shape[0], fut_indefinite.shape[0] + 1)
            val_given_x = np.concatenate(([0], fut_indefinite[:, 0]))
            val_given_y = np.concatenate(([0], fut_indefinite[:, 1]))
            fut_interpolated[:, 0] = np.interp(param_query, param_given, val_given_x)
            fut_interpolated[:, 1] = np.interp(param_query, param_given, val_given_y)

            # Read pre-extracted raster map image
            img_dir = os.path.join(self.dataroot, 'prediction_raster_maps',
                                   'images' + str(self.img_size) + "_" + str(int(grid_size_m)) + 'm')
            img = cv2.imread(os.path.join(img_dir, instance_token + "_" + sample_token + '.png'))

            # Pre-process image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img)
            img = img.permute((2, 0, 1)).float() / 255

            # Normalize using Imagenet stats
            img = normalize_imagenet(img)

            # Read pre-extracted raster map with agent boxes (for visualization + debugging)
            img_agents = cv2.imread(os.path.join(img_dir, instance_token + "_" + sample_token + 'agents.png'))

            # Pre-process image
            img_agents = cv2.cvtColor(img_agents, cv2.COLOR_BGR2RGB)
            img_agents = torch.from_numpy(img_agents)
            img_agents = img_agents.permute((2, 0, 1)).float() / 255

            # Get surrounding agent states
            agents = torch.from_numpy(self.agent_ip.make_representation(instance_token, sample_token))
            agents = agents.permute((2, 0, 1)).float()

            # Sum pool states to down-sample to grid dimensions
            agents = f.avg_pool2d(agents[None, :, :, :], self.img_size // self.grid_dim)
            agents = agents.squeeze(dim=0) * ((self.img_size // self.grid_dim) ** 2)

            # Get expert SVF:
            svf_e, waypts_e, grid_idcs = self.get_expert_waypoints(fut_interpolated)
            svf_e = torch.from_numpy(svf_e)
            waypts_e = torch.from_numpy(waypts_e)
            grid_idcs = torch.from_numpy(grid_idcs)

            # Get motion and position feats:
            motion_feats = self.get_motion_feats(instance_token, sample_token)
            motion_feats = torch.from_numpy(motion_feats)

            # Targets for behavior cloning model:
            bc_targets = self.get_bc_targets(fut_interpolated)
            bc_targets = torch.from_numpy(bc_targets)

            return hist, fut, img, svf_e, motion_feats, waypts_e, agents, grid_idcs, bc_targets, img_agents, \
                instance_token, sample_token, idx

    def get_hist(self, instance_token: str, sample_token: str):
        """
        Function to get track history of agent
        :param instance_token: nuScenes instance token for datapoint
        :param sample_token nuScenes sample token for datapoint
        """
        # x, y co-ordinates in agent's frame of reference
        xy = self.helper.get_past_for_agent(instance_token,
                                            sample_token,
                                            seconds=self.t_h,
                                            in_agent_frame=True)

        # Get all history records for obtaining velocity, acceleration and turn rate values
        hist_records = self.helper.get_past_for_agent(instance_token,
                                                      sample_token,
                                                      seconds=self.t_h,
                                                      in_agent_frame=True,
                                                      just_xy=False)
        if xy.shape[0] > self.t_h * 2:
            xy = xy[0:int(self.t_h) * 2]
        if len(hist_records) > self.t_h * 2:
            hist_records = hist_records[0:int(self.t_h) * 2]

        # Initialize hist tensor and set x and y co-ordinates returned by prediction helper
        hist = np.zeros((xy.shape[0], 5))
        hist[:, 0:2] = xy

        # Instance and sample tokens from history records
        i_tokens = [hist_records[i]['instance_token'] for i in range(len(hist_records))]
        i_tokens.insert(0, instance_token)
        s_tokens = [hist_records[i]['sample_token'] for i in range(len(hist_records))]
        s_tokens.insert(0, sample_token)

        # Set velocity, acc and turn rate values for hist
        for k in range(hist.shape[0]):
            i_t = i_tokens[k]
            s_t = s_tokens[k]
            v = self.helper.get_velocity_for_agent(i_t, s_t)
            a = self.helper.get_acceleration_for_agent(i_t, s_t)
            theta = self.helper.get_heading_change_rate_for_agent(i_t, s_t)

            # If function returns nan values due to short tracks, set corresponding value to 0
            if np.isnan(v):
                v = 0
            if np.isnan(a):
                a = 0
            if np.isnan(theta):
                theta = 0
            hist[k, 2] = v
            hist[k, 3] = a
            hist[k, 4] = theta

        # Zero pad for track histories shorter than t_h
        hist_zeropadded = np.zeros((int(self.t_h) * 2, 5))

        # Flip to have correct order of timestamps
        hist = np.flip(hist, 0)
        hist_zeropadded[-hist.shape[0]:] = hist

        return hist_zeropadded

    def get_expert_waypoints(self, fut: np.ndarray):
        """
        Function to get the expert's state visitation frequencies based on their trajectory
        :param fut: numpy array with future trajectory of for all available future timestamps, up-sampled by 10
        """

        # Expert state visitation frequencies for training reward model, waypoints in meters and grid indices
        svf_e = np.zeros((2, self.grid_dim, self.grid_dim))
        waypts_e = np.zeros((self.horizon, 2))
        grid_idcs = np.zeros((self.horizon, 2))

        count = 0
        row_prev = np.nan
        column_prev = np.nan
        for k in range(fut.shape[0]):

            # Convert trajectory (x,y) co-ordinates to grid locations:
            column = np.argmin(np.absolute(fut[k, 0] - self.col_centers))
            row = np.argmin(np.absolute(fut[k, 1] - self.row_centers))

            # Demonstration ends when expert leaves the image crop corresponding to the grid:
            if self.grid_extent[0] <= fut[k, 0] <= self.grid_extent[1] and \
                    self.grid_extent[2] <= fut[k, 1] <= self.grid_extent[3]:

                # Check if cell location has changed
                if row != row_prev or column != column_prev:

                    # Add cell location to path states of expert
                    svf_e[0, row.astype(int), column.astype(int)] = 1

                    if count < self.horizon:

                        # Get BEV coordinates corresponding to cell locations
                        waypts_e[count, 0] = self.row_centers[row]
                        waypts_e[count, 1] = self.col_centers[column]
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

    def get_motion_feats(self, instance_token: str, sample_token: str):
        """
        Function to get motion and position features over grid for reward model
        :param instance_token: NuScenes instance token for datapoint
        :param sample_token: NuScenes sample token for datapoint
        """
        feats = np.zeros((3, self.grid_dim, self.grid_dim))

        # X and Y co-ordinates over grid
        grid_size_m = self.grid_extent[1] - self.grid_extent[0]
        y = (np.linspace(self.grid_extent[3] - grid_size_m/(self.grid_dim*2),
                         self.grid_extent[2] + grid_size_m/(self.grid_dim*2),
                         self.grid_dim)).reshape(-1, 1).repeat(self.grid_dim, axis=1)
        x = (np.linspace(self.grid_extent[0] + grid_size_m/(self.grid_dim*2),
                         self.grid_extent[1] - grid_size_m/(self.grid_dim*2),
                         self.grid_dim)).reshape(-1, 1).repeat(self.grid_dim, axis=1).transpose()

        # Velocity of agent
        v = self.helper.get_velocity_for_agent(instance_token, sample_token)
        if np.isnan(v):
            v = 0

        # Normalize X and Y co-ordinates over grid
        feats[0] = v
        feats[1] = x/grid_size_m
        feats[2] = y/grid_size_m

        return feats

    def get_bc_targets(self, fut: np.ndarray):
        """
        Function to get targets for behavior cloning model
        :param fut: numpy array with future trajectory of for all available future timestamps, up-sampled by 10
        """
        bc_targets = np.zeros((self.num_actions + 1, self.grid_dim, self.grid_dim))
        column_prev = np.argmin(np.absolute(fut[0, 0] - self.col_centers))
        row_prev = np.argmin(np.absolute(fut[0, 1] - self.row_centers))

        for k in range(fut.shape[0]):

            # Convert trajectory (x,y) co-ordinates to grid locations:
            column = np.argmin(np.absolute(fut[k, 0] - self.col_centers))
            row = np.argmin(np.absolute(fut[k, 1] - self.row_centers))

            # Demonstration ends when expert leaves the image crop corresponding to the grid:
            if self.grid_extent[0] <= fut[k, 0] <= self.grid_extent[1] and self.grid_extent[2] <= fut[k, 1] <= \
                    self.grid_extent[3]:

                # Check if cell location has changed
                if row != row_prev or column != column_prev:
                    bc_targets[:, int(row_prev), int(column_prev)] = 0
                    d_x = column - column_prev
                    d_y = row - row_prev
                    theta = np.arctan2(d_y, d_x)

                    # Assign ground truth actions for expert demonstration
                    if self.num_actions == 4:  # [D,R,U,L,end]
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
        bc_targets[self.num_actions, int(row_prev), int(column_prev)] = 1

        return bc_targets


# Utilities for surrounding agent representation for NuScenes
class AgentMotionStatesOnGrid(AgentRepresentation):
    """
    Represents surrounding agents as state vectors (velocity, acc and yaw-rate and binary flag for whether an agent
    exists at location) over a 2-D grid.
    """
    def __init__(self, helper: PredictHelper,
                 resolution: float = 0.1,  # meters / pixel
                 meters_ahead: float = 40, meters_behind: float = 10,
                 meters_left: float = 25, meters_right: float = 25,):

        self.helper = helper
        if not resolution > 0:
            raise ValueError(f"Resolution must be positive. Received {resolution}.")
        self.resolution = resolution
        self.meters_ahead = meters_ahead
        self.meters_behind = meters_behind
        self.meters_left = meters_left
        self.meters_right = meters_right

    def make_representation(self, instance_token: str, sample_token: str) -> np.ndarray:
        """
        Represents agents as a 4 channel image. Channel 0: 1 if an agent is present at this location 0 otherwise
        Channel 1, 2, 3 velocity, acc and yaw-rate of agent

        If multiple agents are present at same location for given resolution, channel 0 is summed, min of other
        channels retained. (Probably need a better solution than min). All values are scalars, direction can be
        inferred using lane direction from static layer.

        :param instance_token: Instance token.
        :param sample_token: Sample token.
        :return: np.ndarray representing a 4 channel image.
        """

        # Taking radius around track before to ensure all actors are in image
        buffer = max([self.meters_ahead, self.meters_behind,
                      self.meters_left, self.meters_right]) * 2

        image_side_length = int(buffer / self.resolution)

        # We will center the track in the image
        central_track_pixels = (image_side_length / 2, image_side_length / 2)
        base_image = np.zeros((image_side_length, image_side_length, 4))

        annotations = self.helper.get_annotations_for_sample(sample_token)
        center_agent_annotation = self.helper.get_sample_annotation(instance_token, sample_token)

        populate_agent_states(center_agent_annotation, central_track_pixels,
                              annotations, base_image, self.helper, resolution=self.resolution)

        # Rotate and crop representation:
        center_agent_yaw = quaternion_yaw(Quaternion(center_agent_annotation['rotation']))
        rotation_mat = get_rotation_matrix(base_image.shape, center_agent_yaw)
        rotated_image = cv2.warpAffine(base_image, rotation_mat, (base_image.shape[1],
                                                                  base_image.shape[0]))

        row_crop, col_crop = get_crops(self.meters_ahead, self.meters_behind,
                                       self.meters_left, self.meters_right, self.resolution,
                                       image_side_length)

        return rotated_image[row_crop, col_crop]


def populate_agent_states(center_agent_annotation: Dict[str, Any],
                          center_agent_pixels: Tuple[float, float],
                          annotations: List[Dict[str, Any]],
                          base_image: np.ndarray,
                          helper: PredictHelper,
                          resolution: float = 0.1) -> None:
    """
    Adds agent states to 4 channel base_image
    :param center_agent_annotation: Annotation record for the agent
        that is in the center of the image.
    :param center_agent_pixels: Pixel location of the agent in the
        center of the image.
    :param annotations: Annotation records for other agents
    :param base_image: 4 channel image to populate with agent states.
    :param helper: Predict helper
    :param resolution: Size of the image in pixels / meter.
    :return: None.
    """

    agent_x, agent_y = center_agent_annotation['translation'][:2]

    for i, annotation in enumerate(annotations):
        if annotation['instance_token'] != center_agent_annotation['instance_token']:
            location = annotation['translation'][:2]
            row_pixel, column_pixel = convert_to_pixel_coords(location,
                                                              (agent_x, agent_y),
                                                              center_agent_pixels, resolution)

            if 0 <= row_pixel < base_image.shape[0] and 0 <= column_pixel < base_image.shape[1]:

                v = helper.get_velocity_for_agent(annotation['instance_token'],
                                                  annotation['sample_token'])

                a = helper.get_acceleration_for_agent(annotation['instance_token'],
                                                      annotation['sample_token'])

                omega = helper.get_heading_change_rate_for_agent(annotation['instance_token'],
                                                                 annotation['sample_token'])

                if base_image[row_pixel, column_pixel, 0] == 0:
                    base_image[row_pixel, column_pixel, 0] = 1
                    if not np.isnan(v):
                        base_image[row_pixel, column_pixel, 1] = v
                    if not np.isnan(a):
                        base_image[row_pixel, column_pixel, 2] = a
                    if not np.isnan(omega):
                        base_image[row_pixel, column_pixel, 3] = omega

                else:
                    base_image[row_pixel, column_pixel, 0] += 1
                    if not np.isnan(v):
                        base_image[row_pixel, column_pixel, 1] = min(v, base_image[row_pixel, column_pixel, 1])
                    if not np.isnan(a):
                        base_image[row_pixel, column_pixel, 2] = min(a, base_image[row_pixel, column_pixel, 2])
                    if not np.isnan(omega):
                        base_image[row_pixel, column_pixel, 3] = min(omega, base_image[row_pixel, column_pixel, 3])
