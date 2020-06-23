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

import paddle.fluid as fluid
from ..common_functions import conv_bn_layer, deconv_bn_layer
from collections import OrderedDict


class EASTHead(object):
    """
    EAST: An Efficient and Accurate Scene Text Detector
        see arxiv: https://arxiv.org/abs/1704.03155
    args:
        params(dict): the super parameters for network build
    """

    def __init__(self, params):

        self.model_name = params['model_name']

    def unet_fusion(self, inputs):
        f = inputs[::-1]
        if self.model_name == "large":
            num_outputs = [128, 128, 128, 128]
        else:
            num_outputs = [64, 64, 64, 64]
        g = [None, None, None, None]
        h = [None, None, None, None]
        for i in range(4):
            if i == 0:
                h[i] = f[i]
            else:
                h[i] = fluid.layers.concat([g[i - 1], f[i]], axis=1)
                h[i] = conv_bn_layer(
                    input=h[i],
                    num_filters=num_outputs[i],
                    filter_size=3,
                    stride=1,
                    act='relu',
                    name="unet_h_%d" % (i))
            if i <= 2:
                #can be replaced with unpool
                g[i] = deconv_bn_layer(
                    input=h[i],
                    num_filters=num_outputs[i],
                    name="unet_g_%d" % (i))
            else:
                g[i] = conv_bn_layer(
                    input=h[i],
                    num_filters=num_outputs[i],
                    filter_size=3,
                    stride=1,
                    act='relu',
                    name="unet_g_%d" % (i))
        return g[3]

    def detector_header(self, f_common):
        if self.model_name == "large":
            num_outputs = [128, 64, 1, 8]
        else:
            num_outputs = [64, 32, 1, 8]
        f_det = conv_bn_layer(
            input=f_common,
            num_filters=num_outputs[0],
            filter_size=3,
            stride=1,
            act='relu',
            name="det_head1")
        f_det = conv_bn_layer(
            input=f_det,
            num_filters=num_outputs[1],
            filter_size=3,
            stride=1,
            act='relu',
            name="det_head2")
        #f_score
        f_score = conv_bn_layer(
            input=f_det,
            num_filters=num_outputs[2],
            filter_size=1,
            stride=1,
            act=None,
            name="f_score")
        f_score = fluid.layers.sigmoid(f_score)
        #f_geo
        f_geo = conv_bn_layer(
            input=f_det,
            num_filters=num_outputs[3],
            filter_size=1,
            stride=1,
            act=None,
            name="f_geo")
        f_geo = (fluid.layers.sigmoid(f_geo) - 0.5) * 2 * 800
        return f_score, f_geo

    def __call__(self, inputs):
        f_common = self.unet_fusion(inputs)
        f_score, f_geo = self.detector_header(f_common)
        predicts = OrderedDict()
        predicts['f_score'] = f_score
        predicts['f_geo'] = f_geo
        return predicts
