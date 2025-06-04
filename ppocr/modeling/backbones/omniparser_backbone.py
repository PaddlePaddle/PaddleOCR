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
OmniParser backbone: Enhanced backbone with high-resolution feature maps for text spotting,
table recognition, and key information extraction.
"""

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppocr.modeling.backbones.rec_resnet_vd import ResNet

__all__ = ['OmniParserBackbone']


class ConvBNLayer(nn.Layer):
    """Basic Conv-BN-ReLU structure"""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0,
                 dilation=1,
                 groups=1,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=False)
            
        self.bn = nn.BatchNorm2D(out_channels)
        
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'leaky_relu':
            self.act = nn.LeakyReLU(0.1)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class FPN(nn.Layer):
    """Feature Pyramid Network for multi-scale feature fusion"""
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        
        self.lateral_convs = nn.LayerList()
        self.fpn_convs = nn.LayerList()
        
        for i in range(len(in_channels_list)):
            lateral_conv = ConvBNLayer(
                in_channels=in_channels_list[i],
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                act='relu')
            fpn_conv = ConvBNLayer(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                act='relu')
            
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
    
    def forward(self, inputs):
        # Build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            # Resize using nearest neighbor interpolation followed by convolution
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], 
                scale_factor=2, 
                mode='nearest')
        
        # Apply convolution on merged features
        outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(len(laterals))
        ]
        
        return outs


class OmniParserBackbone(nn.Layer):
    """
    OmniParser backbone with ResNet and FPN for multi-scale feature extraction
    """
    def __init__(self, in_channels=3, **kwargs):
        super(OmniParserBackbone, self).__init__()
        
        # Base ResNet
        self.resnet = ResNet(in_channels=in_channels, layers=50, **kwargs)
        
        # Get output channels from each stage
        resnet_out_channels = self.resnet.out_channels
        
        # FPN feature fusion
        self.fpn = FPN(
            in_channels_list=resnet_out_channels,
            out_channels=kwargs.get('fpn_out_channels', 256)
        )
        
        # Enhanced feature extraction for tables 
        self.table_enhancer = nn.Sequential(
            ConvBNLayer(
                in_channels=kwargs.get('fpn_out_channels', 256),
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                act='relu'),
            ConvBNLayer(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
                act='relu')
        )
        
        # Output channels (for connecting to subsequent heads)
        self.out_channels = kwargs.get('fpn_out_channels', 256)
        
    def forward(self, x):
        """
        Forward pass for OmniParser backbone
        
        Args:
            x (Tensor): Input tensor of shape [N, C, H, W]
            
        Returns:
            dict: Dictionary containing multi-scale features
        """
        # Get ResNet features
        resnet_feats = self.resnet(x)
        
        # Apply FPN
        fpn_feats = self.fpn(resnet_feats)
        
        # Enhanced features for table recognition
        table_feats = self.table_enhancer(fpn_feats[-1])
        
        # Return features for different tasks
        features = {
            'fpn_feats': fpn_feats,      # For text detection
            'table_feats': table_feats,  # For table recognition
            'kie_feats': fpn_feats[-1]   # For key information extraction
        }
        
        return features
