from __future__ import absolute_import

import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


def grid_sample(input, grid, canvas=None):
    output = F.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        input_mask = input.data.new(input.size()).fill_(1)
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output


# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :,
                                                                         1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix


# output_ctrl_pts are specified, according to our task.
def build_output_control_points(num_control_points, margins):
    margin_x, margin_y = margins
    num_ctrl_pts_per_side = num_control_points // 2
    ctrl_pts_x = np.linspace(margin_x, 1.0 - margin_x, num_ctrl_pts_per_side)
    ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * margin_y
    ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1.0 - margin_y)
    ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
    ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
    # ctrl_pts_top = ctrl_pts_top[1:-1,:]
    # ctrl_pts_bottom = ctrl_pts_bottom[1:-1,:]
    output_ctrl_pts_arr = np.concatenate(
        [ctrl_pts_top, ctrl_pts_bottom], axis=0)
    output_ctrl_pts = torch.Tensor(output_ctrl_pts_arr)
    return output_ctrl_pts


# demo: ~/test/models/test_tps_transformation.py
class TPSSpatialTransformer(nn.Module):
    def __init__(self,
                 output_image_size=None,
                 num_control_points=None,
                 margins=None):
        super(TPSSpatialTransformer, self).__init__()
        self.output_image_size = output_image_size
        self.num_control_points = num_control_points
        self.margins = margins

        self.target_height, self.target_width = output_image_size
        target_control_points = build_output_control_points(num_control_points,
                                                            margins)
        N = num_control_points
        # N = N - 4

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(
            target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        HW = self.target_height * self.target_width
        target_coordinate = list(
            itertools.product(
                range(self.target_height), range(self.target_width)))
        target_coordinate = torch.Tensor(target_coordinate)  # HW x 2
        Y, X = target_coordinate.split(1, dim=1)
        Y = Y / (self.target_height - 1)
        X = X / (self.target_width - 1)
        target_coordinate = torch.cat([X, Y],
                                      dim=1)  # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = compute_partial_repr(
            target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
        ],
                                           dim=1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)
        self.register_buffer('target_control_points', target_control_points)

    def forward(self, input, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_control_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)

        Y = torch.cat([
            source_control_points, self.padding_matrix.expand(batch_size, 3, 2)
        ], 1)
        mapping_matrix = torch.matmul(self.inverse_kernel, Y)
        source_coordinate = torch.matmul(self.target_coordinate_repr,
                                         mapping_matrix)

        grid = source_coordinate.view(-1, self.target_height, self.target_width,
                                      2)
        grid = torch.clamp(grid, 0,
                           1)  # the source_control_points may be out of [0, 1].
        # the input to grid_sample is normalized [-1, 1], but what we get is [0, 1]
        grid = 2.0 * grid - 1.0
        output_maps = grid_sample(input, grid, canvas=None)
        return output_maps, source_coordinate


if __name__ == "__main__":
    from stn_torch import STNHead
    in_planes = 3
    num_ctrlpoints = 20
    torch.manual_seed(10)
    activation = 'none'  # 'sigmoid'
    stn_head = STNHead(in_planes, num_ctrlpoints, activation)
    np.random.seed(100)
    data = np.random.randn(10, 3, 32, 64).astype("float32")
    input = torch.tensor(data)
    control_points = stn_head(input)
    tps = TPSSpatialTransformer(
        output_image_size=[32, 320],
        num_control_points=20,
        margins=[0.05, 0.05])
    out = tps(input, control_points[1])
    print("out 0 :", out[0].shape)
    print("out 1:", out[1].shape)
