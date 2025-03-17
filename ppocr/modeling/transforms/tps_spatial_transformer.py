# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is refer from:
https://github.com/ayumiymk/aster.pytorch/blob/master/lib/models/tps_spatial_transformer.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import nn, ParamAttr
from paddle.nn import functional as F
import numpy as np
import itertools


def grid_sample(input, grid, canvas=None):
    input.stop_gradient = False

    is_fp16 = False
    if grid.dtype != paddle.float32:
        data_type = grid.dtype
        input = input.cast(paddle.float32)
        grid = grid.cast(paddle.float32)
        is_fp16 = True
    output = F.grid_sample(input, grid)
    if is_fp16:
        output = output.cast(data_type)
        grid = grid.cast(data_type)

    if canvas is None:
        return output
    else:
        input_mask = paddle.ones(shape=input.shape)
        if is_fp16:
            input_mask = input_mask.cast(paddle.float32)
            grid = grid.cast(paddle.float32)
        output_mask = F.grid_sample(input_mask, grid)
        if is_fp16:
            output_mask = output_mask.cast(data_type)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output


# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    N = input_points.shape[0]
    M = control_points.shape[0]
    pairwise_diff = paddle.reshape(input_points, shape=[N, 1, 2]) - paddle.reshape(
        control_points, shape=[1, M, 2]
    )
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * paddle.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = np.array(repr_matrix != repr_matrix)
    repr_matrix[mask] = 0
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
    output_ctrl_pts_arr = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
    output_ctrl_pts = paddle.to_tensor(output_ctrl_pts_arr)
    return output_ctrl_pts


class TPSSpatialTransformer(nn.Layer):
    def __init__(self, output_image_size=None, num_control_points=None, margins=None):
        super(TPSSpatialTransformer, self).__init__()
        self.output_image_size = output_image_size
        self.num_control_points = num_control_points
        self.margins = margins

        self.target_height, self.target_width = output_image_size
        target_control_points = build_output_control_points(num_control_points, margins)
        N = num_control_points

        # create padded kernel matrix
        forward_kernel = paddle.zeros(shape=[N + 3, N + 3])
        target_control_partial_repr = compute_partial_repr(
            target_control_points, target_control_points
        )
        target_control_partial_repr = paddle.cast(
            target_control_partial_repr, forward_kernel.dtype
        )
        forward_kernel[:N, :N] = target_control_partial_repr
        forward_kernel[:N, -3] = 1
        forward_kernel[-3, :N] = 1
        target_control_points = paddle.cast(target_control_points, forward_kernel.dtype)
        forward_kernel[:N, -2:] = target_control_points
        forward_kernel[-2:, :N] = paddle.transpose(target_control_points, perm=[1, 0])
        # compute inverse matrix
        inverse_kernel = paddle.inverse(forward_kernel)

        # create target coordinate matrix
        HW = self.target_height * self.target_width
        target_coordinate = list(
            itertools.product(range(self.target_height), range(self.target_width))
        )
        target_coordinate = paddle.to_tensor(target_coordinate)  # HW x 2
        Y, X = paddle.split(target_coordinate, target_coordinate.shape[1], axis=1)
        Y = Y / (self.target_height - 1)
        X = X / (self.target_width - 1)
        target_coordinate = paddle.concat(
            [X, Y], axis=1
        )  # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = compute_partial_repr(
            target_coordinate, target_control_points
        )
        target_coordinate_repr = paddle.concat(
            [
                target_coordinate_partial_repr,
                paddle.ones(shape=[HW, 1]),
                target_coordinate,
            ],
            axis=1,
        )

        # register precomputed matrices
        self.inverse_kernel = inverse_kernel
        self.padding_matrix = paddle.zeros(shape=[3, 2])
        self.target_coordinate_repr = target_coordinate_repr
        self.target_control_points = target_control_points

    def forward(self, input, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.shape[1] == self.num_control_points
        assert source_control_points.shape[2] == 2
        batch_size = source_control_points.shape[0]

        padding_matrix = paddle.expand(self.padding_matrix, shape=[batch_size, 3, 2])
        Y = paddle.concat(
            [source_control_points.astype(padding_matrix.dtype), padding_matrix], 1
        )
        mapping_matrix = paddle.matmul(self.inverse_kernel, Y)
        source_coordinate = paddle.matmul(self.target_coordinate_repr, mapping_matrix)

        grid = paddle.reshape(
            source_coordinate, shape=[-1, self.target_height, self.target_width, 2]
        )
        grid = paddle.clip(
            grid, 0, 1
        )  # the source_control_points may be out of [0, 1].
        # the input to grid_sample is normalized [-1, 1], but what we get is [0, 1]
        grid = 2.0 * grid - 1.0
        output_maps = grid_sample(input, grid, canvas=None)
        return output_maps, source_coordinate
