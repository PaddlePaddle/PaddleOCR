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


class DBHead(object):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, params):
        self.k = params['k']
        self.inner_channels = params['inner_channels']
        self.C, self.H, self.W = params['image_shape']
        print(self.C, self.H, self.W)

    def binarize(self, x):
        conv1 = fluid.layers.conv2d(
            input=x,
            num_filters=self.inner_channels // 4,
            filter_size=3,
            padding=1,
            param_attr=fluid.initializer.MSRAInitializer(uniform=False),
            bias_attr=False)
        conv_bn1 = fluid.layers.batch_norm(
            input=conv1,
            param_attr=fluid.initializer.ConstantInitializer(value=1.0),
            bias_attr=fluid.initializer.ConstantInitializer(value=1e-4),
            act="relu")
        conv2 = fluid.layers.conv2d_transpose(
            input=conv_bn1,
            num_filters=self.inner_channels // 4,
            filter_size=2,
            stride=2,
            param_attr=fluid.initializer.MSRAInitializer(uniform=False),
            bias_attr=self._get_bias_attr(0.0004, conv_bn1.shape[1], "conv2"),
            act=None)
        conv_bn2 = fluid.layers.batch_norm(
            input=conv2,
            param_attr=fluid.initializer.ConstantInitializer(value=1.0),
            bias_attr=fluid.initializer.ConstantInitializer(value=1e-4),
            act="relu")
        conv3 = fluid.layers.conv2d_transpose(
            input=conv_bn2,
            num_filters=1,
            filter_size=2,
            stride=2,
            param_attr=fluid.initializer.MSRAInitializer(uniform=False),
            bias_attr=self._get_bias_attr(0.0004, conv_bn2.shape[1], "conv3"),
            act=None)
        out = fluid.layers.sigmoid(conv3)
        return out

    def thresh(self, x):
        conv1 = fluid.layers.conv2d(
            input=x,
            num_filters=self.inner_channels // 4,
            filter_size=3,
            padding=1,
            param_attr=fluid.initializer.MSRAInitializer(uniform=False),
            bias_attr=False)
        conv_bn1 = fluid.layers.batch_norm(
            input=conv1,
            param_attr=fluid.initializer.ConstantInitializer(value=1.0),
            bias_attr=fluid.initializer.ConstantInitializer(value=1e-4),
            act="relu")
        conv2 = fluid.layers.conv2d_transpose(
            input=conv_bn1,
            num_filters=self.inner_channels // 4,
            filter_size=2,
            stride=2,
            param_attr=fluid.initializer.MSRAInitializer(uniform=False),
            bias_attr=self._get_bias_attr(0.0004, conv_bn1.shape[1], "conv2"),
            act=None)
        conv_bn2 = fluid.layers.batch_norm(
            input=conv2,
            param_attr=fluid.initializer.ConstantInitializer(value=1.0),
            bias_attr=fluid.initializer.ConstantInitializer(value=1e-4),
            act="relu")
        conv3 = fluid.layers.conv2d_transpose(
            input=conv_bn2,
            num_filters=1,
            filter_size=2,
            stride=2,
            param_attr=fluid.initializer.MSRAInitializer(uniform=False),
            bias_attr=self._get_bias_attr(0.0004, conv_bn2.shape[1], "conv3"),
            act=None)
        out = fluid.layers.sigmoid(conv3)
        return out

    def _get_bias_attr(self, l2_decay, k, name, gradient_clip=None):
        regularizer = fluid.regularizer.L2Decay(l2_decay)
        stdv = 1.0 / math.sqrt(k * 1.0)
        initializer = fluid.initializer.Uniform(-stdv, stdv)
        bias_attr = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=initializer,
            name=name + "_b_attr")
        return bias_attr

    def step_function(self, x, y):
        return fluid.layers.reciprocal(1 + fluid.layers.exp(-self.k * (x - y)))

    def __call__(self, conv_features, mode="train"):
        c2, c3, c4, c5 = conv_features
        param_attr = fluid.initializer.MSRAInitializer(uniform=False)
        in5 = fluid.layers.conv2d(
            input=c5,
            num_filters=self.inner_channels,
            filter_size=1,
            param_attr=param_attr,
            bias_attr=False)
        in4 = fluid.layers.conv2d(
            input=c4,
            num_filters=self.inner_channels,
            filter_size=1,
            param_attr=param_attr,
            bias_attr=False)
        in3 = fluid.layers.conv2d(
            input=c3,
            num_filters=self.inner_channels,
            filter_size=1,
            param_attr=param_attr,
            bias_attr=False)
        in2 = fluid.layers.conv2d(
            input=c2,
            num_filters=self.inner_channels,
            filter_size=1,
            param_attr=param_attr,
            bias_attr=False)

        out4 = fluid.layers.elementwise_add(
            x=fluid.layers.resize_nearest(
                input=in5, scale=2), y=in4)  # 1/16
        out3 = fluid.layers.elementwise_add(
            x=fluid.layers.resize_nearest(
                input=out4, scale=2), y=in3)  # 1/8
        out2 = fluid.layers.elementwise_add(
            x=fluid.layers.resize_nearest(
                input=out3, scale=2), y=in2)  # 1/4

        p5 = fluid.layers.conv2d(
            input=in5,
            num_filters=self.inner_channels // 4,
            filter_size=3,
            padding=1,
            param_attr=param_attr,
            bias_attr=False)
        p5 = fluid.layers.resize_nearest(input=p5, scale=8)
        p4 = fluid.layers.conv2d(
            input=out4,
            num_filters=self.inner_channels // 4,
            filter_size=3,
            padding=1,
            param_attr=param_attr,
            bias_attr=False)
        p4 = fluid.layers.resize_nearest(input=p4, scale=4)
        p3 = fluid.layers.conv2d(
            input=out3,
            num_filters=self.inner_channels // 4,
            filter_size=3,
            padding=1,
            param_attr=param_attr,
            bias_attr=False)
        p3 = fluid.layers.resize_nearest(input=p3, scale=2)
        p2 = fluid.layers.conv2d(
            input=out2,
            num_filters=self.inner_channels // 4,
            filter_size=3,
            padding=1,
            param_attr=param_attr,
            bias_attr=False)

        fuse = fluid.layers.concat(input=[p5, p4, p3, p2], axis=1)
        shrink_maps = self.binarize(fuse)
        if mode != "train":
            return {"maps": shrink_maps}
        threshold_maps = self.thresh(fuse)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = fluid.layers.concat(
            input=[shrink_maps, threshold_maps, binary_maps], axis=1)
        predicts = {}
        predicts['maps'] = y
        return predicts
