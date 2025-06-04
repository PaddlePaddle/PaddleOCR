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
Multi-task loss for OmniParser unified framework
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

__all__ = ['MultiTaskLoss']


class MultiTaskLoss(nn.Layer):
    """
    Multi-task loss function for OmniParser, combining losses from different task heads:
    1. Text detection (pixel head)
    2. Table recognition (table head)
    3. Key Information Extraction (KIE head)
    """
    def __init__(self, weights=None, **kwargs):
        super(MultiTaskLoss, self).__init__()
        
        # Default weights for each task loss
        self.weights = weights or {
            'pixel_loss': 1.0,
            'table_loss': 1.0,
            'kie_loss': 1.0
        }
    
    def forward(self, predicts, batch):
        """
        Calculate the combined loss from all tasks
        
        Args:
            predicts (dict): Model predictions from all heads
            batch (dict): Batch data with ground truth
            
        Returns:
            dict: Loss values for each task and total loss
        """
        total_loss = 0.0
        losses = {}
        
        # Process pixel head loss (text detection)
        if 'pixel_loss' in predicts:
            pixel_loss = predicts['pixel_loss']
            pixel_loss_weighted = self.weights['pixel_loss'] * pixel_loss
            total_loss += pixel_loss_weighted
            losses['pixel_loss'] = pixel_loss
            losses['pixel_loss_weighted'] = pixel_loss_weighted
        
        # Process table head loss (table recognition)
        if 'table_loss' in predicts:
            table_loss = predicts['table_loss']
            table_loss_weighted = self.weights['table_loss'] * table_loss
            total_loss += table_loss_weighted
            losses['table_loss'] = table_loss
            losses['table_loss_weighted'] = table_loss_weighted
        
        # Process KIE head loss (key information extraction)
        if 'kie_loss' in predicts:
            kie_loss = predicts['kie_loss']
            kie_loss_weighted = self.weights['kie_loss'] * kie_loss
            total_loss += kie_loss_weighted
            losses['kie_loss'] = kie_loss
            losses['kie_loss_weighted'] = kie_loss_weighted
        
        # Calculate individual component losses
        # Text detection components
        if 'text_loss' in predicts:
            losses['text_loss'] = predicts['text_loss']
        if 'center_loss' in predicts:
            losses['center_loss'] = predicts['center_loss']
        if 'border_loss' in predicts:
            losses['border_loss'] = predicts['border_loss']
        
        # Table recognition components
        if 'structure_loss' in predicts:
            losses['structure_loss'] = predicts['structure_loss']
        if 'boundary_loss' in predicts:
            losses['boundary_loss'] = predicts['boundary_loss']
        
        # Set total loss
        losses['loss'] = total_loss
        
        return losses
