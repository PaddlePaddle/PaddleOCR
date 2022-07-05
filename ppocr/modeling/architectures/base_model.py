# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import nn
from ppocr.modeling.transforms import build_transform
from ppocr.modeling.backbones import build_backbone
from ppocr.modeling.necks import build_neck
from ppocr.modeling.heads import build_head

__all__ = ['BaseModel']


class BaseModel(nn.Layer):
    def __init__(self, config):
        """
        the module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(BaseModel, self).__init__()
        in_channels = config.get('in_channels', 3)
        model_type = config['model_type']
        # build transfrom,
        # for rec, transfrom can be TPS,None
        # for det and cls, transfrom shoule to be None,
        # if you make model differently, you can use transfrom in det and cls
        if 'Transform' not in config or config['Transform'] is None:
            self.use_transform = False
        else:
            self.use_transform = True
            config['Transform']['in_channels'] = in_channels
            self.transform = build_transform(config['Transform'])
            in_channels = self.transform.out_channels

        # build backbone, backbone is need for del, rec and cls
        config["Backbone"]['in_channels'] = in_channels
        self.backbone = build_backbone(config["Backbone"], model_type)
        self.backbone.training = False
        # in_channels = self.backbone.out_channels

        """
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
        if 'Head' not in config or config['Head'] is None:
            self.use_head = False
        else:
            self.use_head = True
            config["Head"]['in_channels'] = in_channels
            self.head = build_head(config["Head"])

        self.return_all_feats = config.get("return_all_feats", False)
        """
    def forward(self, x, data=None):
        import numpy as np
        hr_img = x[0]
        lr_img = x[1]
        length = x[2]
        input_tensor = x[3]
        label = x[4]
        # print("length:", length)
        # print("lr image:", lr_img.shape)
        # print("hr image:", hr_img.shape)
        pred = {}
        if self.use_transform:
            x = self.transform(lr_img)
            sr_img = x
            pred["sr_img"] = sr_img
            pred["hr_img"] = hr_img

        if self.training:
            # print("hr img cuda:{}, numpy:{}".format(paddle.sum(hr_img), np.sum(hr_img.numpy())))
            # print("sr img cuda:{}, numpy:{}".format(paddle.sum(sr_img), np.sum(sr_img.numpy())))

            sr_pred, word_attention_map_pred, sr_correct_list = self.backbone(sr_img, length,
                                                                            input_tensor, test=False)
            
            
            hr_pred, word_attention_map_gt, hr_correct_list = self.backbone(hr_img, length,
                                                                input_tensor, test=False)

            pred["input_tensor"] = input_tensor
            pred["hr_pred"] = hr_pred
            pred["word_attention_map_gt"] = word_attention_map_gt
            pred["hr_correct_list"] = hr_correct_list
            pred["sr_pred"] = sr_pred
            pred["word_attention_map_pred"] = word_attention_map_pred
            pred["sr_correct_list"] = sr_correct_list
        return pred



