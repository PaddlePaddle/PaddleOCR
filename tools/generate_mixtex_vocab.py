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
This script is used to generate vocabulary files for MixTeX models.
It processes annotation files containing LaTeX formulas and creates
a frequency-based vocabulary.
"""

import os
import re
import sys
import argparse
import collections
from tqdm import tqdm


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate vocabulary for MixTeX model')
    parser.add_argument('--annotation_file', required=True, type=str, 
                        help='Path to annotation file with format "image_path\\tlatex_formula"')
    parser.add_argument('--output_file', required=True, type=str, 
                        help='Path to save generated vocabulary file')
    parser.add_argument('--min_freq', type=int, default=2,
                        help='Minimum frequency required to include a token in vocabulary')
    parser.add_argument('--max_vocab_size', type=int, default=10000,
                        help='Maximum vocabulary size (excluding special tokens)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose information')
    return parser.parse_args()


def tokenize_latex(text):
    """Tokenize LaTeX formula into tokens
    
    Args:
        text (str): LaTeX formula
        
    Returns:
        list: List of tokens
    """
    # Break text into tokens based on whitespace and special characters
    tokens = []
    current_token = ""
    special_chars = ['\\', '{', '}', '_', '^', '&', '#', '%', '$', '~', ' ']
    
    for char in text:
        if char in special_chars:
            if current_token:
                tokens.append(current_token)
                current_token = ""
            tokens.append(char)
        else:
            current_token += char
            
    if current_token:
        tokens.append(current_token)
        
    return tokens


def process_annotations(annotation_file, min_freq, max_vocab_size, verbose):
    """Process annotation file and generate vocabulary
    
    Args:
        annotation_file (str): Path to annotation file
        min_freq (int): Minimum frequency required to include a token
        max_vocab_size (int): Maximum vocabulary size (excluding special tokens)
        verbose (bool): Whether to print verbose information
        
    Returns:
        list: List of tokens in vocabulary
    """
    special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
    counter = collections.Counter()
    
    # Read annotation file
    if verbose:
        print(f"Processing annotation file: {annotation_file}")
    
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading annotation file: {e}")
        sys.exit(1)
    
    # Count token frequencies
    for idx, line in enumerate(tqdm(lines, desc="Processing annotations")):
        try:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) != 2:
                if verbose:
                    print(f"Warning: Line {idx+1} doesn't contain tab separator, skipping")
                continue
                
            _, formula = parts
            formula = formula.strip()
            tokens = tokenize_latex(formula)
            counter.update(tokens)
            
        except Exception as e:
            if verbose:
                print(f"Error processing line {idx+1}: {e}")
            continue
    
    # Filter tokens by frequency and limit vocabulary size
    filtered_tokens = []
    for token, count in counter.most_common():
        if count < min_freq:
            break
            
        if token in special_tokens:
            continue
            
        filtered_tokens.append(token)
        
        if len(filtered_tokens) >= max_vocab_size:
            break
    
    # Add special tokens at the beginning
    final_vocab = special_tokens + filtered_tokens
    
    if verbose:
        print(f"Total unique tokens: {len(counter)}")
        print(f"Tokens after frequency filtering (>= {min_freq}): {len(filtered_tokens)}")
        print(f"Final vocabulary size (with special tokens): {len(final_vocab)}")
    
    return final_vocab


def save_vocabulary(vocab, output_file, verbose):
    """Save vocabulary to file
    
    Args:
        vocab (list): List of tokens in vocabulary
        output_file (str): Path to save vocabulary file
        verbose (bool): Whether to print verbose information
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for token in vocab:
                f.write(f"{token}\n")
                
        if verbose:
            print(f"Vocabulary saved to {output_file}")
            
    except Exception as e:
        print(f"Error saving vocabulary: {e}")
        sys.exit(1)


def main():
    """Main function for vocabulary generation"""
    args = parse_args()
    
    # Process annotation file
    vocab = process_annotations(
        args.annotation_file, 
        args.min_freq, 
        args.max_vocab_size,
        args.verbose
    )
    
    # Save vocabulary
    save_vocabulary(vocab, args.output_file, args.verbose)


if __name__ == '__main__':
    main()
