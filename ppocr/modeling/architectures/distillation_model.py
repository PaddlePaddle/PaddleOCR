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

from paddle import nn
from ppocr.modeling.transforms import build_transform
from ppocr.modeling.backbones import build_backbone
from ppocr.modeling.necks import build_neck
from ppocr.modeling.heads import build_head
from .base_model import BaseModel

__all__ = ['DistillationModel']


class DistillationModel(nn.Layer):
    def __init__(self, config):
        """
        the module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(DistillationModel, self).__init__()
        config["Teacher"]["model_type"] = config["model_type"]
        config["Teacher"]["algorithm"] = config["algorithm"]

        config["Student"]["model_type"] = config["model_type"]
        config["Student"]["algorithm"] = config["algorithm"]
        self.teacher = BaseModel(config["Teacher"])

        for param in self.teacher.parameters():
            param.trainable = False

        self.student = BaseModel(config["Student"])

    def forward(self, x):
        teacher_out = self.teacher.forward(x)
        student_out = self.student.forward(x)

        # teacher_out.stop_gradient = True

        result = {
            "teacher_out": teacher_out,
            "student_out": student_out,
        }
        return result
