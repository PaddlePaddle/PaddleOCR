# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append('/home/zhoujun20/PaddleOCR')

import paddle
from paddle import nn
from ppocr.modeling.transform import build_transform
from ppocr.modeling.backbones import build_backbone
from ppocr.modeling.necks import build_neck
from ppocr.modeling.heads import build_head

__all__ = ['Model']


class Model(nn.Layer):
    def __init__(self, config):
        """
        Detection module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(Model, self).__init__()
        algorithm = config['algorithm']
        self.type = config['type']
        self.model_name = '{}_{}'.format(self.type, algorithm)

        in_channels = config.get('in_channels', 3)
        # build transfrom,
        # for rec, transfrom can be TPS,None
        # for det and cls, transfrom shoule to be None,
        #                  if you make model differently, you can use transfrom in det and cls
        if 'Transform' not in config or config['Transform'] is None:
            self.use_transform = False
        else:
            self.use_transform = True
            config['Transform']['in_channels'] = in_channels
            self.transform = build_transform(config['Transform'])
            in_channels = self.transform.out_channels

        # build backbone, backbone is need for del, rec and cls
        config["Backbone"]['in_channels'] = in_channels
        self.backbone = build_backbone(config["Backbone"], self.type)
        in_channels = self.backbone.out_channels

        # build neck
        # for rec, neck can be cnn,rnn or reshape(None)
        # for det, neck can be FPN, BIFPN and so on.
        # for cls, neck should be none
        if 'Neck' not in config or config['Neck'] is None:
            self.use_neck = False
        else:
            self.use_neck = True
            config['Neck']['in_channels'] = in_channels
            self.neck = build_neck(config['Neck'])
            in_channels = self.neck.out_channels
        # # build head, head is need for det, rec and cls
        config["Head"]['in_channels'] = in_channels
        self.head = build_head(config["Head"])

    def forward(self, x):
        if self.use_transform:
            x = self.transform(x)
        x = self.backbone(x)
        if self.use_neck:
            x = self.neck(x)
        x = self.head(x)
        return x


def check_static():
    import numpy as np
    from ppocr.utils.save_load import load_dygraph_pretrain
    from ppocr.utils.logging import get_logger
    from tools import program

    config = program.load_config('configs/det/det_r50_vd_db.yml')

    logger = get_logger()
    np.random.seed(0)
    data = np.random.rand(1, 3, 640, 640).astype(np.float32)

    paddle.disable_static()

    x = paddle.to_tensor(data)

    config['Architecture']['in_channels'] = 3
    config['Architecture']["Head"]['out_channels'] = 37
    model = Model(config['Architecture'])
    model.eval()
    load_dygraph_pretrain(
        model, logger, 'det_r50_vd_db/best_accuracy', load_static_weights=True)

    y = model(x)
    for y1 in y:
        print(y1.shape)

    static_out = np.load('static_out.npy')
    diff = y.numpy() - static_out
    print(y.shape, static_out.shape, diff.mean())


if __name__ == '__main__':
    check_static()
