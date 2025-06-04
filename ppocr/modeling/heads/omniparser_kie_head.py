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
KIE Head for OmniParser: Responsible for key information extraction
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppocr.modeling.backbones.det_mobilenet_v3 import ConvBNLayer

__all__ = ['OmniParserKIEHead']


class RelationEncoder(nn.Layer):
    """Encoder for modeling entity relations in KIE"""
    def __init__(self, in_channels, hidden_dim=256, num_heads=8, dropout=0.1):
        super(RelationEncoder, self).__init__()
        
        self.self_attn = nn.MultiHeadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads,
            dropout=dropout)
            
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.relu
        
    def forward(self, src, src_mask=None):
        """
        Args:
            src: Input tensor
            src_mask: Optional mask for attention
            
        Returns:
            Tensor: Encoded features
        """
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class EntityClassifier(nn.Layer):
    """Classifier for entity types in KIE"""
    def __init__(self, hidden_dim, num_classes):
        super(EntityClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor
            
        Returns:
            Tensor: Classification logits
        """
        return self.classifier(x)


class OmniParserKIEHead(nn.Layer):
    """Key Information Extraction head for OmniParser"""
    def __init__(self, in_channels, hidden_dim=256, num_classes=10, **kwargs):
        super(OmniParserKIEHead, self).__init__()
        
        # Initial convolution to reduce channels
        self.conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            act='relu')
            
        # Global average pooling followed by projection
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Relation encoder (transformer-based)
        self.relation_encoder = RelationEncoder(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=kwargs.get('num_heads', 8),
            dropout=kwargs.get('dropout', 0.1))
            
        # Entity classifier
        self.entity_classifier = EntityClassifier(
            hidden_dim=hidden_dim,
            num_classes=num_classes)
            
        # Loss function
        self.loss_weight = kwargs.get('loss_weight', 1.0)
        
    def _cross_entropy_loss(self, pred, gt, weight=None):
        """Cross-entropy loss for classification"""
        if weight is not None:
            loss = F.cross_entropy(
                pred, gt, weight=weight, reduction='mean')
        else:
            loss = F.cross_entropy(
                pred, gt, reduction='mean')
                
        return loss
        
    def forward_train(self, features, targets):
        """Forward pass during training"""
        # Get feature map from backbone designed for KIE
        x = features['kie_feats']
        
        # Apply initial convolution
        x = self.conv(x)
        
        # Process text regions to get entity representations
        batch_size = x.shape[0]
        
        # Get text region positions and features
        regions = targets['regions']  # [batch_size, num_regions, 4] (x1, y1, x2, y2)
        region_labels = targets['region_labels']  # [batch_size, num_regions]
        
        # Extract region features using RoI pooling or similar technique
        # Note: This is simplified here; in practice, you would need a proper RoI extraction
        region_features = []
        for b in range(batch_size):
            regions_b = regions[b]  # [num_regions, 4]
            features_b = []
            
            for region in regions_b:
                x1, y1, x2, y2 = region
                # Extract region feature (simplified)
                # In practice, use RoI pooling or alignment
                roi_feat = x[b:b+1, :, y1:y2, x1:x2]
                roi_feat = self.avg_pool(roi_feat).squeeze(-1).squeeze(-1)
                features_b.append(roi_feat)
                
            # Stack region features for this batch item
            if features_b:
                features_b = paddle.stack(features_b)
            else:
                # Handle case with no regions
                features_b = paddle.zeros([0, x.shape[1]])
                
            region_features.append(features_b)
            
        # Apply projection
        region_feats = [self.proj(feat) for feat in region_features]
        
        # Apply relation encoder to model dependencies between entities
        enhanced_feats = []
        for feat in region_feats:
            if feat.shape[0] > 0:  # Check if there are any regions
                # Create attention mask if needed
                # enhanced = self.relation_encoder(feat)
                # Simple pass-through if no relation encoding
                enhanced = feat
                enhanced_feats.append(enhanced)
            else:
                enhanced_feats.append(feat)
        
        # Classifier to predict entity types
        logits = []
        for feat in enhanced_feats:
            if feat.shape[0] > 0:
                logit = self.entity_classifier(feat)
                logits.append(logit)
            else:
                # No regions case
                logits.append(paddle.zeros([0, self.num_classes]))
        
        # Calculate loss
        loss = 0.0
        batch_size = len(region_labels)
        valid_samples = 0
        
        for i in range(batch_size):
            if logits[i].shape[0] > 0 and region_labels[i].shape[0] > 0:
                loss_i = self._cross_entropy_loss(logits[i], region_labels[i])
                loss += loss_i
                valid_samples += 1
                
        if valid_samples > 0:
            loss = loss / valid_samples
            
        return {'kie_loss': loss * self.loss_weight}
    
    def forward_test(self, features):
        """Forward pass during testing/inference"""
        # Get feature map from backbone designed for KIE
        x = features['kie_feats']
        
        # Apply initial convolution
        x = self.conv(x)
        
        # Note: During inference, we need text regions detected by a text detector
        # Here we return processed features that can be used with detected regions
        return {'kie_features': x}
    
    def forward(self, features, targets=None):
        """Forward pass based on mode (training or testing)"""
        if self.training and targets is not None:
            return self.forward_train(features, targets)
        else:
            return self.forward_test(features)
