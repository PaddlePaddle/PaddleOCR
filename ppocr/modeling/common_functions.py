#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
import math


def get_para_bias_attr(l2_decay, k, name):
    regularizer = fluid.regularizer.L2Decay(l2_decay)
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = fluid.initializer.Uniform(-stdv, stdv)
    para_attr = fluid.ParamAttr(
        regularizer=regularizer, initializer=initializer, name=name + "_w_attr")
    bias_attr = fluid.ParamAttr(
        regularizer=regularizer, initializer=initializer, name=name + "_b_attr")
    return [para_attr, bias_attr]


def conv_bn_layer(input,
                  num_filters,
                  filter_size,
                  stride=1,
                  groups=1,
                  act=None,
                  name=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=(filter_size - 1) // 2,
        groups=groups,
        act=None,
        param_attr=ParamAttr(name=name + "_weights"),
        bias_attr=False,
        name=name + '.conv2d')

    bn_name = "bn_" + name
    return fluid.layers.batch_norm(
        input=conv,
        act=act,
        name=bn_name + '.output',
        param_attr=ParamAttr(name=bn_name + '_scale'),
        bias_attr=ParamAttr(bn_name + '_offset'),
        moving_mean_name=bn_name + '_mean',
        moving_variance_name=bn_name + '_variance')


def deconv_bn_layer(input,
                    num_filters,
                    filter_size=4,
                    stride=2,
                    act='relu',
                    name=None):
    deconv = fluid.layers.conv2d_transpose(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=1,
        act=None,
        param_attr=ParamAttr(name=name + "_weights"),
        bias_attr=False,
        name=name + '.deconv2d')
    bn_name = "bn_" + name
    return fluid.layers.batch_norm(
        input=deconv,
        act=act,
        name=bn_name + '.output',
        param_attr=ParamAttr(name=bn_name + '_scale'),
        bias_attr=ParamAttr(bn_name + '_offset'),
        moving_mean_name=bn_name + '_mean',
        moving_variance_name=bn_name + '_variance')


def create_tmp_var(program, name, dtype, shape, lod_level=0):
    return program.current_block().create_var(
        name=name, dtype=dtype, shape=shape, lod_level=lod_level)
