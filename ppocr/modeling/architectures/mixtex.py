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
This code implements the MixTeX architecture
Implementation based on the paper: "MixTeX: An Efficient Multi-Modal LaTeX Recognition Model for Offline CPU Inference"
GitHub: https://github.com/RQLuo/MixTeX-Latex-OCR
Paper: https://arxiv.org/abs/2406.17148
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .base_model import BaseModel
from ..backbones.mixtex_backbone import MixTeXBackbone
from ..heads.mixtex_head import MixTeXHead


class MixTeX(BaseModel):
    """MixTeX model architecture
    
    This architecture combines a convolutional backbone optimized for CPU inference
    with a transformer-based decoder head for LaTeX text generation.
    
    Args:
        config (dict): Configuration parameters
    """
    
    def __init__(self, config):
        super(MixTeX, self).__init__(config)
        
        # Extract configuration params with defaults
        in_channels = config.get('in_channels', 1)
        hidden_size = config.get('hidden_size', 512)
        out_channels = config.get('out_channels')
        
        if out_channels is None:
            raise ValueError("'out_channels' must be specified in config for MixTeX model")
        
        # Build backbone - convolutional feature extractor
        self.backbone = MixTeXBackbone(
            in_channels=in_channels
        )
        
        # Build head - transformer decoder for sequence generation
        self.head = MixTeXHead(
            in_channels=self.backbone.out_channels,
            out_channels=out_channels,
            hidden_size=hidden_size
        )
        
    def forward(self, x, data=None):
        """Forward pass for MixTeX model
        
        Args:
            x (Tensor): Input image tensor of shape [N, C, H, W]
            data (dict, optional): Additional data for training, including target tokens
        
        Returns:
            dict: Model output with keys:
                - 'logits': Output logits of shape [N, seq_len, vocab_size] during training
                - 'pred': Predicted token indices of shape [N, seq_len] during inference
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Process features with head
        if self.training and data is not None:
            # Training mode - use teacher forcing with target
            target = data.get('target', None)
            if target is None:
                raise ValueError("'target' not found in data during training")
            preds = self.head(features, target)
        else:
            # Inference mode - generate sequence autoregressively
            preds = self.head(features)
            
        return preds
