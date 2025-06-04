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

"""
OmniParser: A Unified Framework for Text Spotting, Key Information Extraction and Table Recognition
Based on paper: https://arxiv.org/abs/xxxx.xxxxx
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D

from ppocr.modeling.transforms import build_transform
from ppocr.modeling.backbones import build_backbone
from ppocr.modeling.necks import build_neck
from ppocr.modeling.heads import build_head

__all__ = ['OmniParser']


class OmniParser(nn.Layer):
    """
    OmniParser: A Unified Framework for Text Spotting, Key Information Extraction and Table Recognition
    
    Paper: OmniParser: A Unified Framework for Text Spotting, Key Information Extraction and Table Recognition
    """
    
    def __init__(self, **kwargs):
        super(OmniParser, self).__init__()
        
        # Build backbone
        self.backbone = build_backbone(kwargs.get("Backbone"))
        in_channels = self.backbone.out_channels
        
        # Build neck (optional)
        neck_param = kwargs.get("Neck", None)
        if neck_param is not None:
            neck_param['in_channels'] = in_channels
            self.neck = build_neck(neck_param)
            in_channels = self.neck.out_channels
        else:
            self.neck = None
        
        # Build pixel branch head (text detection)
        if kwargs.get("PixelHead", None) is not None:
            kwargs["PixelHead"]["in_channels"] = in_channels
            self.pixel_head = build_head(kwargs["PixelHead"])
        else:
            self.pixel_head = None
        
        # Build table branch head (table recognition)
        if kwargs.get("TableHead", None) is not None:
            kwargs["TableHead"]["in_channels"] = in_channels
            self.table_head = build_head(kwargs["TableHead"])
        else:
            self.table_head = None
            
        # Build KIE branch head (key information extraction)
        if kwargs.get("KIEHead", None) is not None:
            kwargs["KIEHead"]["in_channels"] = in_channels
            self.kie_head = build_head(kwargs["KIEHead"])
        else:
            self.kie_head = None
        
        self.mode = kwargs.get('mode', 'all')  # 'all', 'text', 'table', 'kie'
        
    def forward(self, x, targets=None):
        """Forward pass of OmniParser
        
        Args:
            x (Tensor): Input images of shape [N, C, H, W]
            targets (dict, optional): Ground-truth for training
            
        Returns:
            dict: Dictionary containing predictions or losses
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Apply neck if available
        if self.neck is not None:
            features = self.neck(features)
        
        result = {}
        losses = {}
        
        # Apply pixel head for text detection
        if self.pixel_head is not None and (self.mode in ['all', 'text']):
            if self.training and targets is not None:
                pixel_losses = self.pixel_head(features, targets)
                losses.update(pixel_losses)
            else:
                pixel_results = self.pixel_head(features)
                result.update(pixel_results)
        
        # Apply table head for table recognition
        if self.table_head is not None and (self.mode in ['all', 'table']):
            if self.training and targets is not None:
                table_losses = self.table_head(features, targets)
                losses.update(table_losses)
            else:
                table_results = self.table_head(features)
                result.update(table_results)
        
        # Apply KIE head for key information extraction
        if self.kie_head is not None and (self.mode in ['all', 'kie']):
            if self.training and targets is not None:
                kie_losses = self.kie_head(features, targets)
                losses.update(kie_losses)
            else:
                kie_results = self.kie_head(features)
                result.update(kie_results)
        
        if self.training:
            return losses
        else:
            return result
