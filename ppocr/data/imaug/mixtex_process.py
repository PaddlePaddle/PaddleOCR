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
This code implements the data processing functions for MixTeX model
Implementation based on the paper: "MixTeX: An Efficient Multi-Modal LaTeX Recognition Model for Offline CPU Inference"
GitHub: https://github.com/RQLuo/MixTeX-Latex-OCR
Paper: https://arxiv.org/abs/2406.17148
"""

import os
import cv2
import math
import numpy as np
import paddle
from paddle.vision import transforms


class VocabDict(object):
    """Dictionary class for vocabulary"""
    
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = {
            '<pad>': 0,  # Padding token
            '<bos>': 1,  # Beginning of sequence token
            '<eos>': 2,  # End of sequence token
            '<unk>': 3,  # Unknown token
        }
    
    def load_from_file(self, vocab_path):
        """Load vocabulary from file
        
        Args:
            vocab_path (str): Path to vocabulary file
                Each line contains a single token
        """
        # Initialize with special tokens
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            
        # Continue numbering from the last special token
        next_idx = len(self.special_tokens)
        
        # Load vocabulary from file
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and line not in self.token_to_id:
                        self.token_to_id[line] = next_idx
                        self.id_to_token[next_idx] = line
                        next_idx += 1
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading vocabulary from {vocab_path}: {e}")
            
        return len(self.token_to_id)


class ImageProcessor(object):
    """Image processor for MixTeX model"""
    
    def __init__(self, img_h, img_w):
        self.img_h = img_h
        self.img_w = img_w
        
    def resize_normalize(self, img):
        """Resize and normalize image
        
        Args:
            img (numpy.ndarray): Input image of shape [H, W, C]
            
        Returns:
            numpy.ndarray: Processed image of shape [C, H, W]
        """
        # Convert to grayscale if image is RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Add channel dimension if grayscale
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
            
        # Calculate aspect ratio and resize
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(self.img_h * ratio) > self.img_w:
            target_w = self.img_w
        else:
            target_w = int(math.ceil(self.img_h * ratio))
        
        # Resize to target height and calculated width
        img = cv2.resize(img, (target_w, self.img_h))
        
        # Pad with zeros to reach target width
        target_img = np.ones([self.img_h, self.img_w, 1], dtype=np.float32) * 255
        target_img[:, 0:target_w, :] = img if len(img.shape) == 3 else np.expand_dims(img, axis=2)
        
        # Normalize pixel values to [-1, 1]
        target_img = target_img / 255.0 * 2.0 - 1.0
        
        # Transpose to [C, H, W] format
        target_img = target_img.transpose((2, 0, 1))
        
        return target_img


class MixTeXDataProcess(object):
    """Data processing for MixTeX model"""
    
    def __init__(self, vocab_path, img_h=128, img_w=512, max_seq_len=512, **kwargs):
        self.vocab_path = vocab_path
        self.img_h = img_h
        self.img_w = img_w
        self.max_seq_len = max_seq_len
        
        # Initialize vocabulary
        self.vocab = VocabDict()
        self.vocab_size = self.vocab.load_from_file(vocab_path)
        
        # Initialize image processor
        self.img_processor = ImageProcessor(img_h, img_w)
        
        # Special token indices for easy access
        self.pad_idx = self.vocab.special_tokens['<pad>']
        self.bos_idx = self.vocab.special_tokens['<bos>']
        self.eos_idx = self.vocab.special_tokens['<eos>']
        self.unk_idx = self.vocab.special_tokens['<unk>']
        
    def __call__(self, data):
        """Process input data for MixTeX model
        
        Args:
            data: dict with keys 'image' and 'label'
                'image': numpy array of shape [H, W, C]
                'label': string of LaTeX markup
        
        Returns:
            dict: processed data with keys 'image' and 'target'
        """
        # Process image
        image = data['image']
        processed_img = self.img_processor.resize_normalize(image)
        data['image'] = processed_img
        
        # Process label if available
        if 'label' in data and data['label'] is not None:
            label = data['label']
            target = self.encode(label)
            data['target'] = np.array(target, dtype=np.int64)
        
        return data
    
    def encode(self, text):
        """Encode text to token ids
        
        Args:
            text (str): Input LaTeX text
            
        Returns:
            list: List of token ids with BOS and EOS tokens
        """
        # Break text into tokens based on whitespace and special characters
        tokens = []
        current_token = ""
        special_chars = ['\\', '{', '}', '_', '^', '&']
        
        # Simple tokenization for LaTeX
        for char in text:
            if char in special_chars or char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(char)
            else:
                current_token += char
                
        if current_token:
            tokens.append(current_token)
            
        # Convert tokens to ids
        token_ids = [self.bos_idx]  # Start with BOS token
        
        for token in tokens:
            if token in self.vocab.token_to_id:
                token_ids.append(self.vocab.token_to_id[token])
            else:
                token_ids.append(self.unk_idx)  # Unknown token
        
        token_ids.append(self.eos_idx)  # End with EOS token
        
        # Truncate if too long
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len - 1] + [self.eos_idx]
        
        # Pad if necessary
        padding_length = self.max_seq_len - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.pad_idx] * padding_length
            
        return token_ids
    
    def decode(self, token_ids):
        """Decode token ids to text
        
        Args:
            token_ids (list or numpy.ndarray): List of token ids
            
        Returns:
            str: Decoded LaTeX text
        """
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
            
        # Decode token ids to tokens
        tokens = []
        for idx in token_ids:
            # Stop decoding at EOS token
            if idx == self.eos_idx:
                break
                
            # Skip BOS and PAD tokens
            if idx in [self.bos_idx, self.pad_idx]:
                continue
                
            if idx in self.vocab.id_to_token:
                tokens.append(self.vocab.id_to_token[idx])
            else:
                tokens.append(self.vocab.id_to_token[self.unk_idx])  # Unknown token
        
        # Reconstruct LaTeX text
        text = ''.join(tokens)
        
        return text
