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
This code implements the head network for MixTeX model
Implementation based on the paper: "MixTeX: An Efficient Multi-Modal LaTeX Recognition Model for Offline CPU Inference"
GitHub: https://github.com/RQLuo/MixTeX-Latex-OCR
Paper: https://arxiv.org/abs/2406.17148
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import math


def _initialize_weights(layer):
    """Initialize weights for transformer layers"""
    if isinstance(layer, nn.Linear):
        nn.initializer.XavierNormal()(layer.weight)
        if layer.bias is not None:
            nn.initializer.Constant(value=0.0)(layer.bias)
    elif isinstance(layer, nn.LayerNorm):
        nn.initializer.Constant(value=1.0)(layer.weight)
        nn.initializer.Constant(value=0.0)(layer.bias)


class PositionalEncoding(nn.Layer):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute positional encodings
        pe = paddle.zeros([max_len, d_model])
        position = paddle.arange(0, max_len).unsqueeze(1).astype('float32')
        div_term = paddle.exp(
            paddle.arange(0, d_model, 2).astype('float32') *
            (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices
        pe[:, 0::2] = paddle.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = paddle.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter but should be saved and restored)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class MultiHeadAttention(nn.Layer):
    """Multi-head attention implementation"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        # Linear layers for query, key, value
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        
        # Linear projections and reshape for multi-head attention
        q = self.q_linear(q).reshape([batch_size, -1, self.num_heads, self.head_dim])
        k = self.k_linear(k).reshape([batch_size, -1, self.num_heads, self.head_dim])
        v = self.v_linear(v).reshape([batch_size, -1, self.num_heads, self.head_dim])
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose([0, 2, 1, 3])
        k = k.transpose([0, 2, 1, 3])
        v = v.transpose([0, 2, 1, 3])
        
        # Scaled dot-product attention
        scores = paddle.matmul(q, k.transpose([0, 1, 3, 2])) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask * -1e9
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of values
        context = paddle.matmul(attn_weights, v)
        
        # Transpose back and reshape
        context = context.transpose([0, 2, 1, 3]).reshape([batch_size, -1, self.d_model])
        
        # Final linear layer
        output = self.out(context)
        
        return output


class FeedForward(nn.Layer):
    """Feed-forward network in transformer"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class DecoderLayer(nn.Layer):
    """Transformer decoder layer"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt2, tgt2, tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        
        # Cross-attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(tgt2, memory, memory, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        
        # Feed-forward
        tgt2 = self.norm3(tgt)
        tgt2 = self.feed_forward(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        
        return tgt


class TransformerDecoder(nn.Layer):
    """Transformer decoder for sequence generation"""
    
    def __init__(self, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        self.layers = nn.LayerList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
            
        return self.norm(tgt)


class MixTeXHead(nn.Layer):
    """MixTeX head with transformer decoder"""
    
    def __init__(self, in_channels, out_channels, hidden_size=512, 
                 num_layers=4, num_heads=8, dropout=0.1, max_seq_len=512):
        super(MixTeXHead, self).__init__()
        
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.max_seq_len = max_seq_len
        
        # CNN feature projection
        self.proj = nn.Linear(in_channels, hidden_size)
        
        # Embed tokens
        self.embedding = nn.Embedding(out_channels, hidden_size)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_size, dropout)
        
        # Transformer decoder
        self.decoder = TransformerDecoder(
            d_model=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=hidden_size * 4,
            dropout=dropout
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, out_channels)
        
        # Special tokens
        self.bos_idx = 1  # Beginning of sequence
        self.eos_idx = 2  # End of sequence
        self.pad_idx = 0  # Padding
        
        # Initialize weights
        self.apply(_initialize_weights)
        
    def _generate_square_subsequent_mask(self, sz):
        """Generate square mask for self-attention"""
        mask = paddle.triu(paddle.ones([sz, sz]), diagonal=1)
        return mask.astype('bool')
    
    def encode_image_features(self, features):
        """Encode image features from backbone"""
        # Shape: [batch_size, channels, height, width]
        batch_size, channels, h, w = features.shape
        
        # Reshape to [batch_size, height*width, channels]
        features = features.transpose([0, 2, 3, 1]).reshape([batch_size, h * w, channels])
        
        # Project to hidden size
        features = self.proj(features)
        
        return features
    
    def forward_training(self, features, target):
        """Forward pass during training with teacher forcing"""
        # Encode image features
        memory = self.encode_image_features(features)
        
        # Get target shape
        batch_size = target.shape[0]
        tgt_len = target.shape[1]
        
        # Embed target tokens and add positional encoding
        tgt_embedding = self.embedding(target)
        tgt_embedding = self.pos_encoding(tgt_embedding)
        
        # Create target mask for transformer
        tgt_mask = self._generate_square_subsequent_mask(tgt_len)
        
        # Decode with transformer
        decoded = self.decoder(tgt_embedding, memory, tgt_mask=tgt_mask)
        
        # Project to vocabulary
        logits = self.output_proj(decoded)
        
        # Return logits for training
        return {'logits': logits}
    
    def forward_inference(self, features):
        """Forward pass during inference with greedy decoding"""
        # Encode image features
        memory = self.encode_image_features(features)
        
        # Get batch size
        batch_size = memory.shape[0]
        
        # Start with BOS token for each batch item
        ys = paddle.full([batch_size, 1], self.bos_idx, dtype='int64')
        
        # Greedy decoding
        for i in range(self.max_seq_len - 1):
            # Embed current sequence
            tgt = self.embedding(ys)
            tgt = self.pos_encoding(tgt)
            
            # Create target mask
            tgt_mask = self._generate_square_subsequent_mask(ys.shape[1])
            
            # Decode with transformer
            out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            
            # Project to vocabulary
            logits = self.output_proj(out)
            
            # Get prediction for next token (last position)
            next_token = paddle.argmax(logits[:, -1, :], axis=1, keepdim=True)
            
            # Append to sequence
            ys = paddle.concat([ys, next_token], axis=1)
            
            # Break if all sequences in the batch have EOS
            if paddle.all(next_token == self.eos_idx):
                break
        
        return {'pred': ys}
    
    def forward(self, features, target=None):
        """Forward pass based on training or inference mode"""
        if target is not None:
            # Training mode with teacher forcing
            return self.forward_training(features, target)
        else:
            # Inference mode with greedy decoding
            return self.forward_inference(features)
