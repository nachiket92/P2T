from __future__ import division
import torch.nn as nn
import torch
import torchvision.models as mdl
from typing import Tuple

# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RewardModel(nn.Module):

    def __init__(self, args):
        """
        Reward model over 2-D grid
        args to include
        'backbone': str CNN backbone to use (resnet34 or resnet50)
        'scene_feat_size': int Size of scene features at each grid cell
        'grid_cell_size': int Kernel size to pool scene features to map to grid dimensions
        'agg_sizes': Tuple[int, int] 1x1 conv layer channel dimensions to output rewards at each cell
        """

        super(RewardModel, self).__init__()

        # Unpack arguments:
        self.grid_cell_size = args['grid_cell_size']
        self.scene_feat_size = args['scene_feat_size']
        self.agg_sizes = args['agg_sizes']

        # Initialize model parameters:
        self.feats, feat_size = init_conv_layers(args['backbone'])

        # Conv layers to aggregate context at each grid location
        self.conv_grid = nn.Conv2d(feat_size, self.scene_feat_size, self.grid_cell_size, stride=self.grid_cell_size)

        # Layers for aggregating motion and scene features:
        self.agg1_goal = nn.Conv2d(self.scene_feat_size + 3, self.agg_sizes[0], 1)
        self.agg2_goal = nn.Conv2d(self.agg_sizes[0], self.agg_sizes[1], 1)
        self.agg1_path = nn.Conv2d(self.scene_feat_size + 3, self.agg_sizes[0], 1)
        self.agg2_path = nn.Conv2d(self.agg_sizes[0], self.agg_sizes[1], 1)

        # Output layers:
        self.op_goal = nn.Conv2d(self.agg_sizes[1], 1, 1)
        self.op_path = nn.Conv2d(self.agg_sizes[1], 1, 1)

        # Non-linearities:
        self.relu = torch.nn.ReLU()
        self.logsigmoid = torch.nn.LogSigmoid()
        self.sigmoid = torch.nn.Sigmoid()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, motion_feats: torch.Tensor, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for reward model
        :param motion_feats: Tensor of motion features shape [batch_size, motion_feat_size, grid_dim, grid_dim]
        :param img: Tensor raster map shape [batch_size, 3, img_size, img_size]
        :return r: Tensor of reward maps shape [batch_size, 2, grid_dim, grid_dim]
        :return img_feats: Tensor of scene features shape [batch_size, scene_feat_size, grid_dim, grid_dim]
        """
        # Encode map/scene
        img_feats = self.feats(img)
        img_feats = self.relu(self.conv_grid(img_feats))

        # Concatenate motion and position features
        feats = torch.cat((img_feats, motion_feats), dim=1)

        # Output goal and path rewards
        r_goal = self.logsigmoid(self.op_goal(self.relu(self.agg2_goal(self.relu(self.agg1_goal(feats))))))
        r_path = self.logsigmoid(self.op_path(self.relu(self.agg2_path(self.relu(self.agg1_path(feats))))))

        # Concatenate along channel dimension
        r = torch.cat((r_path, r_goal), 1)

        return r, img_feats


def init_conv_layers(backbone: str) -> Tuple[nn.Sequential, int]:
    """
    Helper function to initialize conv layers for extracting map features
    :param backbone: CNN backbone to use
    :return: Tuple with nn.Sequential for first few layers of  backbone and channel dim of final returned layer
    """
    if backbone == 'resnet34':
        resnet34 = mdl.resnet34(pretrained=True)
        return nn.Sequential(resnet34.conv1, resnet34.bn1, resnet34.relu, resnet34.maxpool, resnet34.layer1), 64
    elif backbone == 'resnet50':
        resnet50 = mdl.resnet50(pretrained=True)
        return nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool, resnet50.layer1), 256
