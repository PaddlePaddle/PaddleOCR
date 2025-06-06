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
This code implements the post-processing for MixTeX model
Implementation based on the paper: "MixTeX: An Efficient Multi-Modal LaTeX Recognition Model for Offline CPU Inference"
GitHub: https://github.com/RQLuo/MixTeX-Latex-OCR
Paper: https://arxiv.org/abs/2406.17148
"""

import numpy as np


class MixTeXPostProcess(object):
    """MixTeX post-processing class for converting model output to LaTeX text"""
    
    def __init__(self, vocab_path, **kwargs):
        """Initialize the MixTeX post-processor
        
        Args:
            vocab_path (str): Path to the vocabulary file
        """
        self.vocab_path = vocab_path
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = {
            '<pad>': 0,
            '<bos>': 1,
            '<eos>': 2,
            '<unk>': 3,
        }
        
        # Load vocabulary
        self._load_vocab()
        
    def _load_vocab(self):
        """Load vocabulary from file"""
        # Initialize with special tokens
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            
        # Continue numbering from the last special token
        next_idx = len(self.special_tokens)
        
        # Load vocabulary from file
        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and line not in self.token_to_id:
                        self.token_to_id[line] = next_idx
                        self.id_to_token[next_idx] = line
                        next_idx += 1
        except FileNotFoundError:
            raise FileNotFoundError(f"Vocabulary file not found at {self.vocab_path}")
        except Exception as e:
            raise ValueError(f"Error loading vocabulary from {self.vocab_path}: {e}")
    
    def __call__(self, preds, batch=None, *args, **kwargs):
        """Convert prediction tensors to LaTeX strings
        
        Args:
            preds (dict): Prediction dictionary from the model with key 'pred'
                containing token indices of shape [batch_size, seq_len]
            batch (dict, optional): Input batch data
            
        Returns:
            list: List of dictionaries with keys 'pred', 'conf', and 'pred_id'
                for each item in the batch
        """
        results = []
        if isinstance(preds, dict):
            # Get predicted token indices
            pred_ids = preds.get('pred', None)
            
            # If no predictions found, return empty results
            if pred_ids is None:
                return results
            
            # Convert predictions to LaTeX text for each item in batch
            for idx, pred_id in enumerate(pred_ids):
                # Skip special tokens and decode text
                text = self.decode(pred_id)
                
                # Confidence is not directly available from transformer models
                # Set to 1.0 as a placeholder
                confidence = 1.0
                
                results.append({
                    'pred': text,            # Decoded LaTeX text 
                    'conf': confidence,      # Confidence score
                    'pred_id': pred_id.tolist()  # Raw token IDs
                })
                
        return results
    
    def decode(self, token_ids):
        """Decode token ids to LaTeX text
        
        Args:
            token_ids (list or numpy.ndarray): List of token ids
            
        Returns:
            str: Decoded LaTeX text
        """
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
            
        # Decode token ids to tokens
        tokens = []
        eos_idx = self.token_to_id['<eos>']
        bos_idx = self.token_to_id['<bos>']
        pad_idx = self.token_to_id['<pad>']
        unk_idx = self.token_to_id['<unk>']
        
        for idx in token_ids:
            # Stop decoding at EOS token
            if idx == eos_idx:
                break
                
            # Skip BOS and PAD tokens
            if idx in [bos_idx, pad_idx]:
                continue
                
            # Convert token ID to token string
            if idx in self.id_to_token:
                tokens.append(self.id_to_token[idx])
            else:
                # Use unknown token for unknown IDs
                tokens.append(self.id_to_token[unk_idx])
        
        # Join tokens to form LaTeX text
        text = ''.join(tokens)
        
        return text
