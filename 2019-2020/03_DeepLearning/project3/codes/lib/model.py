import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

from lib import pointnet_utils


class PointNetClassificationMSG(nn.Module):
    """ Self defined model for point cloud classification. 
    
        The set abstraction layer in this implementation uses Multi-Scale Grouping.
    """
    def __init__(self, args):
        super(PointNetClassificationMSG, self).__init__()

        self.args = args

        self.SetAbstractionModules = nn.ModuleList()

        self.SetAbstractionModules.append(
            PointnetSA_MSG(
                npoint=512,
                radius_list=[0.1, 0.2, 0.4],
                nsample_list=[16, 32, 128],
                mlp_list=[
                    [0, 32, 32, 64], 
                    [0, 64, 64, 128], 
                    [0, 64, 96, 128]
                ],
                use_xyz=self.args.use_xyz,
            )
        )

        input_channels = 64 + 128 + 128
        self.SetAbstractionModules.append(
            PointnetSA_MSG(
                npoint=128,
                radius_list=[0.2, 0.4, 0.8],
                nsample_list=[32, 64, 128],
                mlp_list=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=self.args.use_xyz,
            )
        )

        self.SetAbstractionModules.append(
            PointnetSA_MSG(
                npoint=None,
                radius_list=[None],
                nsample_list=[None],
                mlp_list=[[128 + 256 + 256, 256, 512, 1024]],
                use_xyz=self.args.use_xyz,
            )
        )
    
        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 40),
        )

    def forward(self, xyz, features=None):
        """ Forward pass of the network
        
        Args:
            xyz <torch.FloatTensor>: (B, N, 3), points in the point-cloud
            features <torch.FloatTensor>: (B, N, input_channels), default None
        """
        for module in self.SetAbstractionModules:
            xyz, features = module(xyz, features)
        
        # print(features.shape)
        # print(features.squeeze().shape)
        # sys.exit(0)

        return self.fc_layer(features.squeeze())


class PointnetSA_MSG(nn.Module):
    """ Multi-Scale Grouping Layer in Set Abstraction 
        
        Args:
            npoint <int> : Number of random picked points
            radius_list <list[float]> : List of "radius of sampling ball"
            nsample_list <list[int]> : List of "number of samples in each ball query"
            mlp_list <list<list[int'>> : Spec of the pointnet before the global max_pool for each scale
            bn <bool> : Use batchnorm or not.
    """
    def __init__(self, npoint, radius_list, nsample_list, mlp_list, bn=True, use_xyz=True):
        super(PointnetSA_MSG, self).__init__()

        assert len(radius_list) == len(nsample_list) == len(mlp_list), "Must be same groups!"

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for radius, nsample, mlp_spec in zip(radius_list, nsample_list, mlp_list):
            self.groupers.append(
                pointnet_utils.SampleAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet_utils.GroupAll(use_xyz)
            )

            if use_xyz: mlp_spec[0] += 3

            self.mlps.append(pointnet_utils.build_shared_mlp(mlp_spec, bn))

    def forward(self, xyz, features):
        r"""
        Agrs:
            xyz <torch.Tensor> : (B, N, 3) tensor of the xyz coordinates of the features
            features <torch.Tensor> : (B, N, C) tensor of the descriptors of the the features
        
        Returns:
            new_xyz <torch.Tensor> : (B, npoint, 3) tensor of the new features' xyz
            new_features <torch.Tensor> : (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        # farthest point sampling
        new_xyz = (
            pointnet_utils.gather_operation(
                xyz, pointnet_utils.farthest_point_sample(xyz, self.npoint)
            )                                                               # (B, npoints, 3)
            if self.npoint is not None
            else None
        )
        torch.cuda.empty_cache()

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)         # (B, npoint, nsample, C)
            new_features = new_features.permute(0, 3, 1, 2)                 # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)                       # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)])        # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)                         # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1).permute(0,2,1)   # (B, npoint, \sum_k(mlps[k][-1]))

        return new_xyz, new_features



class cls_3d(nn.Module):
    """ basic model for point cloud classification """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.fc = nn.Linear(256, 40)
        self.relu = nn.ReLU()

    def forward(self, x, features=None):
        if x.shape[1] != 3:
            B, N, C = x.shape
            x = torch.transpose(x, 1, 2)
            assert x.shape == (B, C, N)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x
