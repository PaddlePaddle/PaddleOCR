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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import nn, ParamAttr
from paddle.nn import functional as F
import numpy as np


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        bn_name = "bn_" + name
        self.bn = nn.BatchNorm(
            out_channels,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class LocalizationNetwork(nn.Layer):
    def __init__(self, in_channels, num_fiducial, loc_lr, model_name):
        super(LocalizationNetwork, self).__init__()
        self.F = num_fiducial
        F = num_fiducial
        if model_name == "large":
            num_filters_list = [64, 128, 256, 512]
            fc_dim = 256
        else:
            num_filters_list = [16, 32, 64, 128]
            fc_dim = 64

        self.block_list = []
        for fno in range(0, len(num_filters_list)):
            num_filters = num_filters_list[fno]
            name = "loc_conv%d" % fno
            conv = self.add_sublayer(
                name,
                ConvBNLayer(
                    in_channels=in_channels,
                    out_channels=num_filters,
                    kernel_size=3,
                    act='relu',
                    name=name))
            self.block_list.append(conv)
            if fno == len(num_filters_list) - 1:
                pool = nn.AdaptiveAvgPool2D(1)
            else:
                pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
            in_channels = num_filters
            self.block_list.append(pool)
        name = "loc_fc1"
        stdv = 1.0 / math.sqrt(num_filters_list[-1] * 1.0)
        self.fc1 = nn.Linear(
            in_channels,
            fc_dim,
            weight_attr=ParamAttr(
                learning_rate=loc_lr,
                name=name + "_w",
                initializer=nn.initializer.Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(name=name + '.b_0'),
            name=name)

        # Init fc2 in LocalizationNetwork
        initial_bias = self.get_initial_fiducials()
        initial_bias = initial_bias.reshape(-1)
        name = "loc_fc2"
        param_attr = ParamAttr(
            learning_rate=loc_lr,
            initializer=nn.initializer.Assign(np.zeros([fc_dim, F * 2])),
            name=name + "_w")
        bias_attr = ParamAttr(
            learning_rate=loc_lr,
            initializer=nn.initializer.Assign(initial_bias),
            name=name + "_b")
        self.fc2 = nn.Linear(
            fc_dim,
            F * 2,
            weight_attr=param_attr,
            bias_attr=bias_attr,
            name=name)
        self.out_channels = F * 2

    def forward(self, x):
        """
           Estimating parameters of geometric transformation
           Args:
               image: input
           Return:
               batch_C_prime: the matrix of the geometric transformation
        """
        B = x.shape[0]
        i = 0
        for block in self.block_list:
            x = block(x)
        x = x.squeeze(axis=2).squeeze(axis=2)
        x = self.fc1(x)

        x = F.relu(x)
        x = self.fc2(x)
        x = x.reshape(shape=[-1, self.F, 2])
        return x

    def get_initial_fiducials(self):
        """ see RARE paper Fig. 6 (a) """
        F = self.F
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return initial_bias


class GridGenerator(nn.Layer):
    def __init__(self, in_channels, num_fiducial):
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.F = num_fiducial

        name = "ex_fc"
        initializer = nn.initializer.Constant(value=0.0)
        param_attr = ParamAttr(
            learning_rate=0.0, initializer=initializer, name=name + "_w")
        bias_attr = ParamAttr(
            learning_rate=0.0, initializer=initializer, name=name + "_b")
        self.fc = nn.Linear(
            in_channels,
            6,
            weight_attr=param_attr,
            bias_attr=bias_attr,
            name=name)

    def forward(self, batch_C_prime, I_r_size):
        """
        Generate the grid for the grid_sampler.
        Args:
            batch_C_prime: the matrix of the geometric transformation
            I_r_size: the shape of the input image
        Return:
            batch_P_prime: the grid for the grid_sampler
        """
        C = self.build_C_paddle()
        P = self.build_P_paddle(I_r_size)

        inv_delta_C_tensor = self.build_inv_delta_C_paddle(C).astype('float32')
        P_hat_tensor = self.build_P_hat_paddle(
            C, paddle.to_tensor(P)).astype('float32')

        inv_delta_C_tensor.stop_gradient = True
        P_hat_tensor.stop_gradient = True

        batch_C_ex_part_tensor = self.get_expand_tensor(batch_C_prime)

        batch_C_ex_part_tensor.stop_gradient = True

        batch_C_prime_with_zeros = paddle.concat(
            [batch_C_prime, batch_C_ex_part_tensor], axis=1)
        batch_T = paddle.matmul(inv_delta_C_tensor, batch_C_prime_with_zeros)
        batch_P_prime = paddle.matmul(P_hat_tensor, batch_T)
        return batch_P_prime

    def build_C_paddle(self):
        """ Return coordinates of fiducial points in I_r; C """
        F = self.F
        ctrl_pts_x = paddle.linspace(-1.0, 1.0, int(F / 2), dtype='float64')
        ctrl_pts_y_top = -1 * paddle.ones([int(F / 2)], dtype='float64')
        ctrl_pts_y_bottom = paddle.ones([int(F / 2)], dtype='float64')
        ctrl_pts_top = paddle.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = paddle.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = paddle.concat([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # F x 2

    def build_P_paddle(self, I_r_size):
        I_r_height, I_r_width = I_r_size
        I_r_grid_x = paddle.divide(
            paddle.arange(
                -I_r_width, I_r_width, 2, dtype='float64') + 1.0,
            paddle.to_tensor(
                I_r_width, dtype='float64'))
        I_r_grid_y = paddle.divide(
            paddle.arange(
                -I_r_height, I_r_height, 2, dtype='float64') + 1.0,
            paddle.to_tensor(
                I_r_height, dtype='float64'))  # self.I_r_height
        # P: self.I_r_width x self.I_r_height x 2
        P = paddle.stack(paddle.meshgrid(I_r_grid_x, I_r_grid_y), axis=2)
        P = paddle.transpose(P, perm=[1, 0, 2])
        # n (= self.I_r_width x self.I_r_height) x 2
        return P.reshape([-1, 2])

    def build_inv_delta_C_paddle(self, C):
        """ Return inv_delta_C which is needed to calculate T """
        F = self.F
        hat_C = paddle.zeros((F, F), dtype='float64')  # F x F
        for i in range(0, F):
            for j in range(i, F):
                if i == j:
                    hat_C[i, j] = 1
                else:
                    r = paddle.norm(C[i] - C[j])
                    hat_C[i, j] = r
                    hat_C[j, i] = r
        hat_C = (hat_C**2) * paddle.log(hat_C)
        delta_C = paddle.concat(  # F+3 x F+3
            [
                paddle.concat(
                    [paddle.ones(
                        (F, 1), dtype='float64'), C, hat_C], axis=1),  # F x F+3
                paddle.concat(
                    [
                        paddle.zeros(
                            (2, 3), dtype='float64'), paddle.transpose(
                                C, perm=[1, 0])
                    ],
                    axis=1),  # 2 x F+3
                paddle.concat(
                    [
                        paddle.zeros(
                            (1, 3), dtype='float64'), paddle.ones(
                                (1, F), dtype='float64')
                    ],
                    axis=1)  # 1 x F+3
            ],
            axis=0)
        inv_delta_C = paddle.inverse(delta_C)
        return inv_delta_C  # F+3 x F+3

    def build_P_hat_paddle(self, C, P):
        F = self.F
        eps = self.eps
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        # P_tile: n x 2 -> n x 1 x 2 -> n x F x 2
        P_tile = paddle.tile(paddle.unsqueeze(P, axis=1), (1, F, 1))
        C_tile = paddle.unsqueeze(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        # rbf_norm: n x F
        rbf_norm = paddle.norm(P_diff, p=2, axis=2, keepdim=False)

        # rbf: n x F
        rbf = paddle.multiply(
            paddle.square(rbf_norm), paddle.log(rbf_norm + eps))
        P_hat = paddle.concat(
            [paddle.ones(
                (n, 1), dtype='float64'), P, rbf], axis=1)
        return P_hat  # n x F+3

    def get_expand_tensor(self, batch_C_prime):
        B, H, C = batch_C_prime.shape
        batch_C_prime = batch_C_prime.reshape([B, H * C])
        batch_C_ex_part_tensor = self.fc(batch_C_prime)
        batch_C_ex_part_tensor = batch_C_ex_part_tensor.reshape([-1, 3, 2])
        return batch_C_ex_part_tensor


class TPS(nn.Layer):
    def __init__(self, in_channels, num_fiducial, loc_lr, model_name):
        super(TPS, self).__init__()
        self.loc_net = LocalizationNetwork(in_channels, num_fiducial, loc_lr,
                                           model_name)
        self.grid_generator = GridGenerator(self.loc_net.out_channels,
                                            num_fiducial)
        self.out_channels = in_channels

    def forward(self, image):
        image.stop_gradient = False
        batch_C_prime = self.loc_net(image)
        batch_P_prime = self.grid_generator(batch_C_prime, image.shape[2:])
        batch_P_prime = batch_P_prime.reshape(
            [-1, image.shape[2], image.shape[3], 2])
        batch_I_r = F.grid_sample(x=image, grid=batch_P_prime)
        return batch_I_r
