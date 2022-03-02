#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.param_attr import ParamAttr
import numpy as np


class LocalizationNetwork(object):
    def __init__(self, params):
        super(LocalizationNetwork, self).__init__()
        self.F = params['num_fiducial']
        self.loc_lr = params['loc_lr']
        self.model_name = params['model_name']

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None):
        conv = layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        bn_name = "bn_" + name
        return layers.batch_norm(
            input=conv,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

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

    def __call__(self, image):
        """
        Estimating parameters of geometric transformation
        Args:
            image: input
        Return: 
            batch_C_prime: the matrix of the geometric transformation
        """
        F = self.F
        loc_lr = self.loc_lr
        if self.model_name == "large":
            num_filters_list = [64, 128, 256, 512]
            fc_dim = 256
        else:
            num_filters_list = [16, 32, 64, 128]
            fc_dim = 64
        for fno in range(len(num_filters_list)):
            num_filters = num_filters_list[fno]
            name = "loc_conv%d" % fno
            if fno == 0:
                conv = self.conv_bn_layer(
                    image, num_filters, 3, act='relu', name=name)
            else:
                conv = self.conv_bn_layer(
                    pool, num_filters, 3, act='relu', name=name)

            if fno == len(num_filters_list) - 1:
                pool = layers.adaptive_pool2d(
                    input=conv, pool_size=[1, 1], pool_type='avg')
            else:
                pool = layers.pool2d(
                    input=conv,
                    pool_size=2,
                    pool_stride=2,
                    pool_padding=0,
                    pool_type='max')
        name = "loc_fc1"
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        fc1 = layers.fc(input=pool,
                        size=fc_dim,
                        param_attr=fluid.param_attr.ParamAttr(
                            learning_rate=loc_lr,
                            initializer=fluid.initializer.Uniform(-stdv, stdv),
                            name=name + "_w"),
                        act='relu',
                        name=name)

        initial_bias = self.get_initial_fiducials()
        initial_bias = initial_bias.reshape(-1)
        name = "loc_fc2"
        param_attr = fluid.param_attr.ParamAttr(
            learning_rate=loc_lr,
            initializer=fluid.initializer.NumpyArrayInitializer(
                np.zeros([fc_dim, F * 2])),
            name=name + "_w")
        bias_attr = fluid.param_attr.ParamAttr(
            learning_rate=loc_lr,
            initializer=fluid.initializer.NumpyArrayInitializer(initial_bias),
            name=name + "_b")
        fc2 = layers.fc(input=fc1,
                        size=F * 2,
                        param_attr=param_attr,
                        bias_attr=bias_attr,
                        name=name)
        batch_C_prime = layers.reshape(x=fc2, shape=[-1, F, 2], inplace=False)
        return batch_C_prime


class GridGenerator(object):
    def __init__(self, params):
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.F = params['num_fiducial']

    def build_C(self):
        """ Return coordinates of fiducial points in I_r; C """
        F = self.F
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # F x 2

    def build_P(self, I_r_size):
        I_r_width, I_r_height = I_r_size
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0)\
            / I_r_width  # self.I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0)\
            / I_r_height  # self.I_r_height
        # P: self.I_r_width x self.I_r_height x 2
        P = np.stack(np.meshgrid(I_r_grid_x, I_r_grid_y), axis=2)
        # n (= self.I_r_width x self.I_r_height) x 2
        return P.reshape([-1, 2])

    def build_inv_delta_C(self, C):
        """ Return inv_delta_C which is needed to calculate T """
        F = self.F
        hat_C = np.zeros((F, F), dtype=float)  # F x F
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C**2) * np.log(hat_C)
        # print(C.shape, hat_C.shape)
        delta_C = np.concatenate(  # F+3 x F+3
            [
                np.concatenate(
                    [np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
                np.concatenate(
                    [np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
                np.concatenate(
                    [np.zeros((1, 3)), np.ones((1, F))], axis=1)  # 1 x F+3
            ],
            axis=0)
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # F+3 x F+3

    def build_P_hat(self, C, P):
        F = self.F
        eps = self.eps
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        #P_tile: n x 2 -> n x 1 x 2 -> n x F x 2
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        #rbf_norm: n x F
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)
        #rbf: n x F
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + eps))
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x F+3

    def get_expand_tensor(self, batch_C_prime):
        name = "ex_fc"
        initializer = fluid.initializer.ConstantInitializer(value=0.0)
        param_attr = fluid.param_attr.ParamAttr(
            learning_rate=0.0, initializer=initializer, name=name + "_w")
        bias_attr = fluid.param_attr.ParamAttr(
            learning_rate=0.0, initializer=initializer, name=name + "_b")
        batch_C_ex_part_tensor = fluid.layers.fc(input=batch_C_prime,
                                                 size=6,
                                                 param_attr=param_attr,
                                                 bias_attr=bias_attr,
                                                 name=name)
        batch_C_ex_part_tensor = fluid.layers.reshape(
            x=batch_C_ex_part_tensor, shape=[-1, 3, 2])
        return batch_C_ex_part_tensor

    def __call__(self, batch_C_prime, I_r_size):
        """
        Generate the grid for the grid_sampler.
        Args:
            batch_C_prime: the matrix of the geometric transformation
            I_r_size: the shape of the input image
        Return: 
            batch_P_prime: the grid for the grid_sampler 
        """
        C = self.build_C()
        P = self.build_P(I_r_size)
        inv_delta_C = self.build_inv_delta_C(C).astype('float32')
        P_hat = self.build_P_hat(C, P).astype('float32')

        inv_delta_C_tensor = layers.create_tensor(dtype='float32')
        layers.assign(inv_delta_C, inv_delta_C_tensor)
        inv_delta_C_tensor.stop_gradient = True
        P_hat_tensor = layers.create_tensor(dtype='float32')
        layers.assign(P_hat, P_hat_tensor)
        P_hat_tensor.stop_gradient = True

        batch_C_ex_part_tensor = self.get_expand_tensor(batch_C_prime)
        #         batch_C_ex_part_tensor = create_tmp_var(
        #             fluid.default_main_program(),
        #             name='batch_C_ex_part_tensor', 
        #             dtype='float32', shape=[-1, 3, 2])
        #         layers.py_func(func=get_batch_C_expand, 
        #             x=[batch_C_prime], out=[batch_C_ex_part_tensor])

        batch_C_ex_part_tensor.stop_gradient = True

        batch_C_prime_with_zeros = layers.concat(
            [batch_C_prime, batch_C_ex_part_tensor], axis=1)
        batch_T = layers.matmul(inv_delta_C_tensor, batch_C_prime_with_zeros)
        batch_P_prime = layers.matmul(P_hat_tensor, batch_T)
        return batch_P_prime


class TPS(object):
    def __init__(self, params):
        super(TPS, self).__init__()
        self.loc_net = LocalizationNetwork(params)
        self.grid_generator = GridGenerator(params)

    def __call__(self, image):
        batch_C_prime = self.loc_net(image)
        I_r_size = [image.shape[3], image.shape[2]]
        batch_P_prime = self.grid_generator(batch_C_prime, I_r_size)
        batch_P_prime = layers.reshape(
            x=batch_P_prime, shape=[-1, image.shape[2], image.shape[3], 2])
        batch_I_r = layers.grid_sampler(x=image, grid=batch_P_prime)
        image.stop_gradient = False
        return batch_I_r
