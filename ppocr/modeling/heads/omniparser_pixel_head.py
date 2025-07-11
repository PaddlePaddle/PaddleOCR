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
Pixel Head for OmniParser: Responsible for text detection with pixel-level predictions
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppocr.modeling.backbones.det_mobilenet_v3 import ConvBNLayer

__all__ = ['OmniParserPixelHead']


class PixelDecoder(nn.Layer):
    """Pixel decoder for generating high-resolution segmentation maps"""
    def __init__(self, in_channels, feature_channels=128):
        super(PixelDecoder, self).__init__()
        
        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act='relu')
            
        self.conv2 = ConvBNLayer(
            in_channels=feature_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act='relu')
            
        self.conv3 = nn.Conv2D(
            in_channels=feature_channels,
            out_channels=1,  # Single channel output for text/non-text
            kernel_size=1,
            stride=1)
        
    def forward(self, x):
        """
        Args:
            x: Input feature map
            
        Returns:
            Tensor: Pixel-level predictions
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class OmniParserPixelHead(nn.Layer):
    """Pixel head for text detection in OmniParser"""
    def __init__(self, in_channels, hidden_dim=256, **kwargs):
        super(OmniParserPixelHead, self).__init__()
        
        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            act='relu')
            
        # Text region segmentation branch
        self.text_decoder = PixelDecoder(
            in_channels=hidden_dim,
            feature_channels=hidden_dim // 2)
            
        # Text center line segmentation branch
        self.center_decoder = PixelDecoder(
            in_channels=hidden_dim,
            feature_channels=hidden_dim // 2)
            
        # Text boundary/border segmentation branch
        self.border_decoder = PixelDecoder(
            in_channels=hidden_dim, 
            feature_channels=hidden_dim // 2)
            
        # Loss weights
        self.text_loss_weight = kwargs.get('text_loss_weight', 1.0)
        self.center_loss_weight = kwargs.get('center_loss_weight', 0.5)
        self.border_loss_weight = kwargs.get('border_loss_weight', 0.5)
        
        # Post-processing thresholds
        self.text_threshold = kwargs.get('text_threshold', 0.5)
        self.center_threshold = kwargs.get('center_threshold', 0.5)
        self.border_threshold = kwargs.get('border_threshold', 0.5)
    
    def _dice_loss(self, pred, gt, mask=None):
        """Dice loss for segmentation tasks"""
        pred = F.sigmoid(pred)
        
        if mask is not None:
            mask = mask.astype('float32')
            pred = pred * mask
            gt = gt * mask
            
        intersection = paddle.sum(pred * gt)
        union = paddle.sum(pred) + paddle.sum(gt)
        dice_loss = 1 - (2 * intersection + 1) / (union + 1)
        
        return dice_loss
    
    def _weighted_bce_loss(self, pred, gt, mask=None, weight=None):
        """Weighted binary cross-entropy loss"""
        pred = F.sigmoid(pred)
        
        if mask is not None:
            mask = mask.astype('float32')
            pred = pred * mask
            gt = gt * mask
            
        if weight is not None:
            weight = weight.astype('float32')
            bce_loss = F.binary_cross_entropy(
                pred, gt, weight=weight, reduction='mean')
        else:
            bce_loss = F.binary_cross_entropy(
                pred, gt, reduction='mean')
            
        return bce_loss
    
    def forward_train(self, features, targets):
        """Forward pass during training"""
        # Get feature map from backbone
        x = features['fpn_feats'][-1]
        
        # Apply initial convolution
        x = self.conv1(x)
        
        # Get predictions for each segmentation branch
        text_pred = self.text_decoder(x)
        center_pred = self.center_decoder(x)
        border_pred = self.border_decoder(x)
        
        # Get ground-truth
        text_gt = targets['text_mask']
        center_gt = targets['center_mask']
        border_gt = targets['border_mask']
        mask = targets.get('mask', None)
        
        # Calculate losses
        text_loss = self._dice_loss(text_pred, text_gt, mask) + \
                   self._weighted_bce_loss(text_pred, text_gt, mask)
                   
        center_loss = self._dice_loss(center_pred, center_gt, mask) + \
                     self._weighted_bce_loss(center_pred, center_gt, mask)
                     
        border_loss = self._dice_loss(border_pred, border_gt, mask) + \
                     self._weighted_bce_loss(border_pred, border_gt, mask)
        
        # Weighted sum of losses
        loss = self.text_loss_weight * text_loss + \
               self.center_loss_weight * center_loss + \
               self.border_loss_weight * border_loss
        
        return {
            'pixel_loss': loss,
            'text_loss': text_loss,
            'center_loss': center_loss,
            'border_loss': border_loss
        }
    
    def forward_test(self, features):
        """Forward pass during testing/inference"""
        # Get feature map from backbone
        x = features['fpn_feats'][-1]
        
        # Apply initial convolution
        x = self.conv1(x)
        
        # Get predictions for each segmentation branch
        text_pred = self.text_decoder(x)
        center_pred = self.center_decoder(x)
        border_pred = self.border_decoder(x)
        
        # Apply sigmoid to get probability maps
        text_prob = F.sigmoid(text_pred)
        center_prob = F.sigmoid(center_pred)
        border_prob = F.sigmoid(border_pred)
        
        # Return segmentation probability maps
        return {
            'text_prob': text_prob,
            'center_prob': center_prob,
            'border_prob': border_prob
        }
    
    def forward(self, features, targets=None):
        """Forward pass based on mode (training or testing)"""
        if self.training and targets is not None:
            return self.forward_train(features, targets)
        else:
            return self.forward_test(features)
