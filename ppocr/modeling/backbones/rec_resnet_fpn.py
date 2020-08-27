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

import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

__all__ = [
    "ResNet", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"
]

Trainable = True
w_nolr = fluid.ParamAttr(trainable=Trainable)
train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}


class ResNet():
    def __init__(self, params):
        self.layers = params['layers']
        self.params = train_parameters

    def __call__(self, input):
        layers = self.layers
        supported_layers = [18, 34, 50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        stride_list = [(2, 2), (2, 2), (1, 1), (1, 1)]
        num_filters = [64, 128, 256, 512]

        conv = self.conv_bn_layer(
            input=input,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu',
            name="conv1")
        F = []
        if layers >= 50:
            for block in range(len(depth)):
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    conv = self.bottleneck_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=stride_list[block] if i == 0 else 1,
                        name=conv_name)
                F.append(conv)
        else:
            for block in range(len(depth)):
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)

                    if i == 0 and block != 0:
                        stride = (2, 1)
                    else:
                        stride = (1, 1)

                    conv = self.basic_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=stride,
                        if_first=block == i == 0,
                        name=conv_name)
                F.append(conv)

        base = F[-1]
        for i in [-2, -3]:
            b, c, w, h = F[i].shape
            if (w, h) == base.shape[2:]:
                base = base
            else:
                base = fluid.layers.conv2d_transpose(
                    input=base,
                    num_filters=c,
                    filter_size=4,
                    stride=2,
                    padding=1,
                    act=None,
                    param_attr=w_nolr,
                    bias_attr=w_nolr)
                base = fluid.layers.batch_norm(
                    base, act="relu", param_attr=w_nolr, bias_attr=w_nolr)
            base = fluid.layers.concat([base, F[i]], axis=1)
            base = fluid.layers.conv2d(
                base,
                num_filters=c,
                filter_size=1,
                param_attr=w_nolr,
                bias_attr=w_nolr)
            base = fluid.layers.conv2d(
                base,
                num_filters=c,
                filter_size=3,
                padding=1,
                param_attr=w_nolr,
                bias_attr=w_nolr)
            base = fluid.layers.batch_norm(
                base, act="relu", param_attr=w_nolr, bias_attr=w_nolr)

        base = fluid.layers.conv2d(
            base,
            num_filters=512,
            filter_size=1,
            bias_attr=w_nolr,
            param_attr=w_nolr)

        return base

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=2 if stride == (1, 1) else filter_size,
            dilation=2 if stride == (1, 1) else 1,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=ParamAttr(
                name=name + "_weights", trainable=Trainable),
            bias_attr=False,
            name=name + '.conv2d.output.1')

        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            name=bn_name + '.output.1',
            param_attr=ParamAttr(
                name=bn_name + '_scale', trainable=Trainable),
            bias_attr=ParamAttr(
                bn_name + '_offset', trainable=Trainable),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance', )

    def shortcut(self, input, ch_out, stride, is_first, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1 or is_first == True:
            if stride == (1, 1):
                return self.conv_bn_layer(input, ch_out, 1, 1, name=name)
            else:  #stride == (2,2)
                return self.conv_bn_layer(input, ch_out, 1, stride, name=name)

        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, name):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + "_branch2a")
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b")
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            name=name + "_branch2c")

        short = self.shortcut(
            input,
            num_filters * 4,
            stride,
            is_first=False,
            name=name + "_branch1")

        return fluid.layers.elementwise_add(
            x=short, y=conv2, act='relu', name=name + ".add.output.5")

    def basic_block(self, input, num_filters, stride, is_first, name):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=3,
            act='relu',
            stride=stride,
            name=name + "_branch2a")
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            act=None,
            name=name + "_branch2b")
        short = self.shortcut(
            input, num_filters, stride, is_first, name=name + "_branch1")
        return fluid.layers.elementwise_add(x=short, y=conv1, act='relu')
