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

from paddle import nn
from ppocr.modeling.transforms import build_transform
from ppocr.modeling.backbones import build_backbone
from ppocr.modeling.necks import build_neck
from ppocr.modeling.heads import build_head
from .base_model import BaseModel
from ppocr.utils.save_load import load_dygraph_pretrain

__all__ = ['DistillationModel']


class DistillationModel(nn.Layer):
    def __init__(self, config):
        """
        the module for OCR distillation.
        args:
            config (dict): the super parameters for module.
        """
        super().__init__()

        freeze_params = config["freeze_params"]
        pretrained = config["pretrained"]
        if not isinstance(freeze_params, list):
            freeze_params = [freeze_params]
        assert len(config["Models"]) == len(freeze_params)

        if not isinstance(pretrained, list):
            pretrained = [pretrained] * len(config["Models"])
        assert len(config["Models"]) == len(pretrained)

        self.model_dict = dict()
        index = 0
        for key in config["Models"]:
            model_config = config["Models"][key]
            model = BaseModel(model_config)
            if pretrained[index] is not None:
                load_dygraph_pretrain(model, path=pretrained[index])
            if freeze_params[index]:
                for param in model.parameters():
                    param.trainable = False
            self.model_dict[key] = self.add_sublayer(key, model)
            index += 1

    def forward(self, x):
        result_dict = dict()
        for key in self.model_dict:
            result_dict[key] = self.model_dict[key](x)
        return result_dict
