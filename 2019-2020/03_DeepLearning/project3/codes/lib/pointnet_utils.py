import torch
import torch.nn as nn


def build_shared_mlp(mlp_spec, bn=True):
    """ Build mlp layer
    
    Agrs:
        mlp_spec <List[int]>
        bn <bool>
    """
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(
            nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        )
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)


class SampleAndGroup(nn.Module):
    """ Sampling points and Grouping them.
    Args:
        radius <float>: Radius of ball
        nsample <int>: Maximum number of features to gather in the ball
    """
    def __init__(self, radius, nsample, use_xyz=True):
        super(SampleAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
    
    def forward(self, xyz, new_xyz, features=None):
        """
        Args:
            xyz <torch.Tensor>: xyz coordinates of the features (B, N, 3)
            new_xyz <torch.Tensor>: centriods (B, npoint, 3)
            features <torch.Tensor>: Descriptors of the features (B, N, C)
        
        Returns:
            new_features <torch.Tensor>: (B, npoint, nsample, 3 + C) tensor
        """
        B, N, _ = xyz.shape
        _, npoint, _ = new_xyz.shape

        idx = ball_query(self.radius, self.nsample, xyz, new_xyz) # (B, npoint, nsample)
        torch.cuda.empty_cache()
        grouped_xyz = gather_operation(xyz, idx)                  # (B, npoint, nsample, 3)
        torch.cuda.empty_cache()
        grouped_xyz -= new_xyz.view(B, npoint, 1, 3)              # (B, npoint, nsample, 3)
        torch.cuda.empty_cache()

        if features is not None:
            grouped_features = gather_operation(features, idx)    # (B, npoint, nsample, C)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=-1
                )                                                 # (B, npoint, nsample, C + 3)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz                            # (B, npoint, nsample, 0 + 3)

        return new_features  # (B, npoint, nsample, C + 3)


class GroupAll(nn.Module):
    """ Groups all features """

    def __init__(self, use_xyz=True):
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        """
        Args: 
            xyz <torch.Tensor> : coordinates of the features (B, N, 3)
            new_xyz <torch.Tensor> : Ignored
            features <torch.Tensor> : Descriptors of the features (B, N, C)
        
        Returns:
            new_features <torch.Tensor>: (B, 1, N, C + 3) tensor
        """
        B, N, _ = xyz.shape
        grouped_xyz = xyz.view(B, 1, N, 3)
        
        if features is not None:
            grouped_features = features.view(B, 1, N, -1)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=-1
                )                                   # (B, 1, N, 3 + C)
            else:
                new_features = grouped_features     # (B, 1, N, 0 + C)
        else:
            new_features = grouped_xyz              # (B, 1, N, 3 + 0)

        return new_features                         # (B, 1, N, 3 + C)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, __ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def gather_operation(points, idx):
    """
    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, point]
    
    Returns:
        new_points:, indexed points data, [B, point, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def ball_query(radius, nsample, xyz, new_xyz):
    """
    Args:
        radius <int> : local region radius
        nsample <int> : max sample number in local region
        xyz <torch.Tensor> : all points, [B, N, 3]
        new_xyz <torch.Tensor> : query points, [B, npoint, 3]
    
    Returns:
        group_idx <torch.Tensor> : grouped points index, [B, npoint, nsample]
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, npoint, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, npoint, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, npoint, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist