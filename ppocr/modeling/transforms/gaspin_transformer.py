# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
import functools
from .tps import GridGenerator

'''This code is refer from:
https://github.com/hikopensource/DAVAR-Lab-OCR/davarocr/davar_rcg/models/transformations/gaspin_transformation.py
'''

class SP_TransformerNetwork(nn.Layer):
    """
    Sturture-Preserving Transformation (SPT) as Equa. (2) in Ref. [1]
    Ref: [1] SPIN: Structure-Preserving Inner Offset Network for Scene Text Recognition. AAAI-2021.
    """

    def __init__(self, nc=1, default_type=5):
        """ Based on SPIN
        Args:
            nc (int): number of input channels (usually in 1 or 3)
            default_type (int): the complexity of transformation intensities (by default set to 6 as the paper)
        """
        super(SP_TransformerNetwork, self).__init__()
        self.power_list = self.cal_K(default_type)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.InstanceNorm2D(nc)

    def cal_K(self, k=5):
        """

        Args:
            k (int): the complexity of transformation intensities (by default set to 6 as the paper)

        Returns:
            List: the normalized intensity of each pixel in [0,1], denoted as \beta [1x(2K+1)]

        """
        from math import log
        x = []
        if k != 0:
            for i in range(1, k+1):
                lower = round(log(1-(0.5/(k+1))*i)/log((0.5/(k+1))*i), 2)
                upper = round(1/lower, 2)
                x.append(lower)
                x.append(upper)
        x.append(1.00)
        return x

    def forward(self, batch_I, weights, offsets, lambda_color=None):
        """

        Args:
            batch_I (Tensor): batch of input images [batch_size x nc x I_height x I_width]
            weights:
            offsets: the predicted offset by AIN, a scalar
            lambda_color: the learnable update gate \alpha in Equa. (5) as
                          g(x) = (1 - \alpha) \odot x + \alpha \odot x_{offsets}

        Returns:
            Tensor: transformed images by SPN as Equa. (4) in Ref. [1]
                        [batch_size x I_channel_num x I_r_height x I_r_width]

        """
        batch_I = (batch_I + 1) * 0.5
        if offsets is not None:
            batch_I = batch_I*(1-lambda_color) + offsets*lambda_color
        batch_weight_params = paddle.unsqueeze(paddle.unsqueeze(weights, -1), -1)
        batch_I_power = paddle.stack([batch_I.pow(p) for p in self.power_list], axis=1)

        batch_weight_sum = paddle.sum(batch_I_power * batch_weight_params, axis=1)
        batch_weight_sum = self.bn(batch_weight_sum)
        batch_weight_sum = self.sigmoid(batch_weight_sum)
        batch_weight_sum = batch_weight_sum * 2 - 1
        return batch_weight_sum

class GA_SPIN_Transformer(nn.Layer):
    """
    Geometric-Absorbed SPIN Transformation (GA-SPIN) proposed in Ref. [1]


    Ref: [1] SPIN: Structure-Preserving Inner Offset Network for Scene Text Recognition. AAAI-2021.
    """

    def __init__(self, in_channels=1,
                 I_r_size=(32, 100),
                 offsets=False,
                 norm_type='BN',
                 default_type=6,
                 loc_lr=1,
                 stn=True):
        """
        Args:
            in_channels (int): channel of input features,
                                set it to 1 if the grayscale images and 3 if RGB input
            I_r_size (tuple): size of rectified images (used in STN transformations)
            offsets (bool): set it to False if use SPN w.o. AIN,
                            and set it to True if use SPIN (both with SPN and AIN)
            norm_type (str): the normalization type of the module,
                            set it to 'BN' by default, 'IN' optionally
            default_type (int): the K chromatic space,
                                set it to 3/5/6 depend on the complexity of transformation intensities
            loc_lr (float): learning rate of location network
            stn (bool): whther to use stn.

        """
        super(GA_SPIN_Transformer, self).__init__()
        self.nc = in_channels
        self.spt = True
        self.offsets = offsets
        self.stn = stn  # set to True in GA-SPIN, while set it to False in SPIN
        self.I_r_size = I_r_size
        self.out_channels = in_channels
        if norm_type == 'BN':
            norm_layer = functools.partial(nn.BatchNorm2D, use_global_stats=True)
        elif norm_type == 'IN':
            norm_layer = functools.partial(nn.InstanceNorm2D, weight_attr=False,
                                           use_global_stats=False)
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

        if self.spt:
            self.sp_net = SP_TransformerNetwork(in_channels,
                                                default_type)
            self.spt_convnet = nn.Sequential(
                                  # 32*100
                                  nn.Conv2D(in_channels, 32, 3, 1, 1, bias_attr=False),
                                  norm_layer(32), nn.ReLU(),
                                  nn.MaxPool2D(kernel_size=2, stride=2),
                                  # 16*50
                                  nn.Conv2D(32, 64, 3, 1, 1, bias_attr=False),
                                  norm_layer(64), nn.ReLU(),
                                  nn.MaxPool2D(kernel_size=2, stride=2),
                                  # 8*25
                                  nn.Conv2D(64, 128, 3, 1, 1, bias_attr=False),
                                  norm_layer(128), nn.ReLU(),
                                  nn.MaxPool2D(kernel_size=2, stride=2),
                                  # 4*12
            )
            self.stucture_fc1 = nn.Sequential(
                                  nn.Conv2D(128, 256, 3, 1, 1, bias_attr=False),
                                  norm_layer(256), nn.ReLU(),
                                  nn.MaxPool2D(kernel_size=2, stride=2),
                                  nn.Conv2D(256, 256, 3, 1, 1, bias_attr=False),
                                  norm_layer(256), nn.ReLU(),  # 2*6
                                  nn.MaxPool2D(kernel_size=2, stride=2),
                                  nn.Conv2D(256, 512, 3, 1, 1, bias_attr=False),
                                  norm_layer(512), nn.ReLU(),  # 1*3
                                  nn.AdaptiveAvgPool2D(1),
                                  nn.Flatten(1, -1),  # batch_size x 512
                                  nn.Linear(512, 256, weight_attr=nn.initializer.Normal(0.001)),
                                  nn.BatchNorm1D(256), nn.ReLU()
                                )
            self.out_weight = 2*default_type+1
            self.spt_length = 2*default_type+1
            if offsets:
                self.out_weight += 1
            if self.stn:
                self.F = 20
                self.out_weight += self.F * 2
                self.GridGenerator = GridGenerator(self.F*2, self.F)
                
            # self.out_weight*=nc
            # Init structure_fc2 in LocalizationNetwork
            initial_bias = self.init_spin(default_type*2)
            initial_bias = initial_bias.reshape(-1)
            param_attr = ParamAttr(
                learning_rate=loc_lr,
                initializer=nn.initializer.Assign(np.zeros([256, self.out_weight])))
            bias_attr = ParamAttr(
                learning_rate=loc_lr,
                initializer=nn.initializer.Assign(initial_bias))
            self.stucture_fc2 = nn.Linear(256, self.out_weight,
                                weight_attr=param_attr,
                                bias_attr=bias_attr)
            self.sigmoid = nn.Sigmoid()

            if offsets:
                self.offset_fc1 = nn.Sequential(nn.Conv2D(128, 16,
                                                          3, 1, 1,
                                                          bias_attr=False),
                                                norm_layer(16),
                                                nn.ReLU(),)
                self.offset_fc2 = nn.Conv2D(16, in_channels,
                                            3, 1, 1)
                self.pool = nn.MaxPool2D(2, 2)

    def init_spin(self, nz):
        """
        Args:
            nz (int): number of paired \betas exponents, which means the value of K x 2

        """
        init_id = [0.00]*nz+[5.00]
        if self.offsets:
            init_id += [-5.00]
            # init_id *=3
        init = np.array(init_id)

        if self.stn:
            F = self.F
            ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
            ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
            ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
            ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
            ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
            initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
            initial_bias = initial_bias.reshape(-1)
            init = np.concatenate([init, initial_bias], axis=0)
        return init

    def forward(self, x, return_weight=False):
        """
        Args:
            x (Tensor): input image batch
            return_weight (bool): set to False by default,
                                  if set to True return the predicted offsets of AIN, denoted as x_{offsets}

        Returns:
            Tensor: rectified image [batch_size x I_channel_num x I_height x I_width], the same as the input size
        """

        if self.spt:
            feat = self.spt_convnet(x)
            fc1 = self.stucture_fc1(feat)
            sp_weight_fusion = self.stucture_fc2(fc1)
            sp_weight_fusion = sp_weight_fusion.reshape([x.shape[0], self.out_weight, 1])
            if self.offsets:  # SPIN w. AIN
                lambda_color = sp_weight_fusion[:, self.spt_length, 0]
                lambda_color = self.sigmoid(lambda_color).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                sp_weight = sp_weight_fusion[:, :self.spt_length, :]
                offsets = self.pool(self.offset_fc2(self.offset_fc1(feat)))

                assert offsets.shape[2] == 2  # 2
                assert offsets.shape[3] == 6  # 16
                offsets = self.sigmoid(offsets)  # v12

                if return_weight:
                    return offsets
                offsets = nn.functional.upsample(offsets, size=(x.shape[2], x.shape[3]), mode='bilinear')

                if self.stn:
                    batch_C_prime = sp_weight_fusion[:, (self.spt_length + 1):, :].reshape([x.shape[0], self.F, 2])
                    build_P_prime = self.GridGenerator(batch_C_prime, self.I_r_size)
                    build_P_prime_reshape = build_P_prime.reshape([build_P_prime.shape[0],
                                                                   self.I_r_size[0],
                                                                   self.I_r_size[1],
                                                                   2])

            else:  # SPIN w.o. AIN
                sp_weight = sp_weight_fusion[:, :self.spt_length, :]
                lambda_color, offsets = None, None

                if self.stn:
                    batch_C_prime = sp_weight_fusion[:, self.spt_length:, :].reshape([x.shape[0], self.F, 2])
                    build_P_prime = self.GridGenerator(batch_C_prime, self.I_r_size)
                    build_P_prime_reshape = build_P_prime.reshape([build_P_prime.shape[0],
                                                                   self.I_r_size[0],
                                                                   self.I_r_size[1],
                                                                   2])

            x = self.sp_net(x, sp_weight, offsets, lambda_color)
            if self.stn:
                is_fp16 = False
                if build_P_prime_reshape.dtype != paddle.float32:
                    data_type = build_P_prime_reshape.dtype
                    x = x.cast(paddle.float32)
                    build_P_prime_reshape = build_P_prime_reshape.cast(paddle.float32)
                    is_fp16 = True
                x = F.grid_sample(x=x, grid=build_P_prime_reshape, padding_mode='border')
                if is_fp16:
                    x = x.cast(data_type)
        return x
