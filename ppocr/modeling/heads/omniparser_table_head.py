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
Table Head for OmniParser: Responsible for table structure recognition
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppocr.modeling.backbones.det_mobilenet_v3 import ConvBNLayer

__all__ = ['OmniParserTableHead']


class TableStructureDecoder(nn.Layer):
    """Decoder for table structure (rows, columns, cells)"""
    def __init__(self, in_channels, feature_channels=128, num_classes=3):
        super(TableStructureDecoder, self).__init__()
        
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
            
        # Output heads for different table components
        self.pred_conv = nn.Conv2D(
            in_channels=feature_channels,
            out_channels=num_classes,  # Usually classes are: background, row, column
            kernel_size=1,
            stride=1)
        
    def forward(self, x):
        """
        Args:
            x: Input feature map
            
        Returns:
            Tensor: Table structure predictions
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pred_conv(x)
        return x


class OmniParserTableHead(nn.Layer):
    """Table head for table recognition in OmniParser"""
    def __init__(self, in_channels, hidden_dim=256, **kwargs):
        super(OmniParserTableHead, self).__init__()
        
        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            act='relu')
            
        # Table structure segmentation branch (rows, columns, cells)
        self.table_structure_decoder = TableStructureDecoder(
            in_channels=hidden_dim,
            feature_channels=hidden_dim // 2,
            num_classes=3)  # Background, row line, column line
            
        # Table boundary detection branch
        self.table_boundary_decoder = TableStructureDecoder(
            in_channels=hidden_dim,
            feature_channels=hidden_dim // 2,
            num_classes=2)  # Background, table boundary
        
        # Loss weights
        self.structure_loss_weight = kwargs.get('structure_loss_weight', 1.0)
        self.boundary_loss_weight = kwargs.get('boundary_loss_weight', 0.5)
    
    def _cross_entropy_loss(self, pred, gt, mask=None):
        """Cross-entropy loss for multi-class segmentation"""
        if mask is not None:
            mask = mask.astype('float32')
            pred = pred * mask.unsqueeze(1)
            
        loss = F.cross_entropy(pred, gt, reduction='mean')
        return loss
    
    def _dice_loss_multiclass(self, pred, gt, num_classes, mask=None):
        """Dice loss for multi-class segmentation"""
        dice_loss = 0
        
        # Convert ground truth to one-hot encoding
        gt_one_hot = F.one_hot(gt, num_classes=num_classes)
        gt_one_hot = gt_one_hot.transpose([0, 3, 1, 2])
        
        # Calculate dice loss for each class
        for i in range(num_classes):
            pred_i = F.softmax(pred, axis=1)[:, i]
            gt_i = gt_one_hot[:, i]
            
            if mask is not None:
                mask = mask.astype('float32')
                pred_i = pred_i * mask
                gt_i = gt_i * mask
                
            intersection = paddle.sum(pred_i * gt_i)
            union = paddle.sum(pred_i) + paddle.sum(gt_i)
            dice_loss_i = 1 - (2 * intersection + 1) / (union + 1)
            dice_loss += dice_loss_i
            
        # Average across classes
        dice_loss = dice_loss / num_classes
        return dice_loss
    
    def forward_train(self, features, targets):
        """Forward pass during training"""
        # Get feature map from backbone designed for table recognition
        x = features['table_feats']
        
        # Apply initial convolution
        x = self.conv1(x)
        
        # Get predictions for table structure and boundary
        structure_pred = self.table_structure_decoder(x)
        boundary_pred = self.table_boundary_decoder(x)
        
        # Get ground-truth
        structure_gt = targets['structure_mask']
        boundary_gt = targets['boundary_mask']
        mask = targets.get('mask', None)
        
        # Calculate losses
        structure_ce_loss = self._cross_entropy_loss(structure_pred, structure_gt, mask)
        structure_dice_loss = self._dice_loss_multiclass(structure_pred, structure_gt, 3, mask)
        structure_loss = structure_ce_loss + structure_dice_loss
        
        boundary_ce_loss = self._cross_entropy_loss(boundary_pred, boundary_gt, mask)
        boundary_dice_loss = self._dice_loss_multiclass(boundary_pred, boundary_gt, 2, mask)
        boundary_loss = boundary_ce_loss + boundary_dice_loss
        
        # Weighted sum of losses
        loss = self.structure_loss_weight * structure_loss + \
               self.boundary_loss_weight * boundary_loss
        
        return {
            'table_loss': loss,
            'structure_loss': structure_loss,
            'boundary_loss': boundary_loss
        }
    
    def forward_test(self, features):
        """Forward pass during testing/inference"""
        # Get feature map from backbone designed for table recognition
        x = features['table_feats']
        
        # Apply initial convolution
        x = self.conv1(x)
        
        # Get predictions for table structure and boundary
        structure_pred = self.table_structure_decoder(x)
        boundary_pred = self.table_boundary_decoder(x)
        
        # Return segmentation predictions
        return {
            'structure_pred': structure_pred,
            'boundary_pred': boundary_pred
        }
    
    def forward(self, features, targets=None):
        """Forward pass based on mode (training or testing)"""
        if self.training and targets is not None:
            return self.forward_train(features, targets)
        else:
            return self.forward_test(features)
