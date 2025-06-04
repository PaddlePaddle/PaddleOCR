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
This code implements the loss function for MixTeX model
Implementation based on the paper: "MixTeX: An Efficient Multi-Modal LaTeX Recognition Model for Offline CPU Inference"
GitHub: https://github.com/RQLuo/MixTeX-Latex-OCR
Paper: https://arxiv.org/abs/2406.17148
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class MixTeXLoss(nn.Layer):
    """
    MixTeX loss function implementation
    
    This loss combines cross-entropy loss for token prediction with label smoothing
    to improve model generalization and robustness.
    """
    def __init__(self, ignore_index=0, label_smoothing=0.1, **kwargs):
        super(MixTeXLoss, self).__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction='none'
        )
        
    def _compute_loss(self, logits, targets):
        """Compute loss with padding mask"""
        # Flatten the logits and targets for loss calculation
        batch_size, seq_len, vocab_size = logits.shape
        flat_logits = logits.reshape([-1, vocab_size])
        flat_targets = targets.reshape([-1])
        
        # Calculate raw loss
        losses = self.criterion(flat_logits, flat_targets)
        
        # Create padding mask (1 for tokens, 0 for padding)
        padding_mask = (flat_targets != self.ignore_index).astype('float32')
        
        # Apply mask and calculate average loss
        masked_loss = losses * padding_mask
        sum_loss = paddle.sum(masked_loss)
        sum_mask = paddle.sum(padding_mask) + 1e-10  # Add small epsilon to avoid division by zero
        avg_loss = sum_loss / sum_mask
        
        return avg_loss
        
    def forward(self, predicts, batch):
        """
        Calculate the loss for MixTeX model
        
        Args:
            predicts: dict containing model predictions, with key 'logits'
            batch: dict containing ground truth data, with key 'targets'
            
        Returns:
            dict: containing the calculated loss
        """
        # Extract logits and targets
        if isinstance(predicts, dict):
            logits = predicts.get('logits', None)
            targets = batch.get('target', None)
        else:
            logits, targets = predicts, batch
            
        # Check input validity
        if logits is None or targets is None:
            raise ValueError("Both logits and targets must be provided for loss calculation")
            
        # Calculate loss
        loss = self._compute_loss(logits, targets)
        
        return {'loss': loss}
