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
This code implements the backbone network for MixTeX model
Implementation based on the paper: "MixTeX: An Efficient Multi-Modal LaTeX Recognition Model for Offline CPU Inference"
GitHub: https://github.com/RQLuo/MixTeX-Latex-OCR
Paper: https://arxiv.org/abs/2406.17148
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ConvBNLayer(nn.Layer):
    """Basic Conv-BN-ReLU structure"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, groups=1, act="relu"):
        super(ConvBNLayer, self).__init__()
        
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False)
            
        self.bn = nn.BatchNorm2D(out_channels)
        
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "hard_swish":
            self.act = nn.Hardswish()
        else:
            self.act = None
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class DepthwiseSeparable(nn.Layer):
    """Depthwise separable convolution for CPU efficiency"""
    
    def __init__(self, in_channels, out_channels, stride, act="relu"):
        super(DepthwiseSeparable, self).__init__()
        
        # Depthwise convolution
        self.depthwise_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            act=act)
            
        # Pointwise convolution
        self.pointwise_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act=act)
    
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class MixTeXBackbone(nn.Layer):
    """MixTeX backbone implementation with efficient convolutional blocks"""
    
    def __init__(self, in_channels=1, **kwargs):
        super(MixTeXBackbone, self).__init__()
        
        self.out_channels = 512  # Output channels for the backbone
        
        # Initial conv layer
        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1)
            
        # First stage with depthwise separable convolutions
        self.stage1_1 = DepthwiseSeparable(
            in_channels=32,
            out_channels=64,
            stride=1)
            
        self.stage1_2 = DepthwiseSeparable(
            in_channels=64,
            out_channels=128,
            stride=2)
            
        # Second stage with depthwise separable convolutions
        self.stage2_1 = DepthwiseSeparable(
            in_channels=128,
            out_channels=128,
            stride=1)
            
        self.stage2_2 = DepthwiseSeparable(
            in_channels=128,
            out_channels=256,
            stride=2)
            
        # Third stage with depthwise separable convolutions
        self.stage3_1 = DepthwiseSeparable(
            in_channels=256,
            out_channels=256,
            stride=1)
            
        self.stage3_2 = DepthwiseSeparable(
            in_channels=256,
            out_channels=512,
            stride=(2, 1))  # Reduce height only at this stage
            
        # Final stage with depthwise separable convolutions
        self.stage4_1 = DepthwiseSeparable(
            in_channels=512,
            out_channels=512,
            stride=1)
            
        self.stage4_2 = DepthwiseSeparable(
            in_channels=512,
            out_channels=512,
            stride=(2, 1))  # Reduce height only at this stage
            
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        """Forward pass for MixTeX backbone
        
        Args:
            x (Tensor): Input image tensor of shape [N, C, H, W]
            
        Returns:
            Tensor: Feature maps of shape [N, C, H, W] 
        """
        # First conv layer
        x = self.conv1(x)
        
        # First stage
        x = self.stage1_1(x)
        x = self.stage1_2(x)
        
        # Second stage
        x = self.stage2_1(x)
        x = self.stage2_2(x)
        
        # Third stage
        x = self.stage3_1(x)
        x = self.stage3_2(x)
        
        # Fourth stage
        x = self.stage4_1(x)
        x = self.stage4_2(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        return x
