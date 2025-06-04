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
Data preprocessing for OmniParser
"""

import cv2
import math
import numpy as np
import paddle
import random
import PIL
from PIL import Image

from ppocr.data.imaug.iaa_augment import IaaAugment
from ppocr.data.imaug.text_image_aug.augment import tia_perspective, tia_stretch, tia_distort

__all__ = ['OmniParserDataProcess']


class OmniParserDataProcess(object):
    """
    Data processing class for OmniParser unified framework, handling multi-modal document inputs.
    """
    def __init__(self, 
                 image_shape=None,
                 augmentation=False,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 **kwargs):
        self.image_shape = image_shape
        self.augmentation = augmentation
        self.mean = np.array(mean).reshape((1, 1, 3))
        self.std = np.array(std).reshape((1, 1, 3))
        
        # Augmentation configuration
        if augmentation:
            self.iaa_aug = IaaAugment(kwargs.get('aug_config', []))
    
    def resize_image(self, img, target_size=None, keep_ratio=True):
        """Resize image with optional ratio preservation."""
        img_h, img_w = img.shape[:2]
        
        if target_size is None:
            target_size = self.image_shape
            
        if keep_ratio:
            # Calculate target height and width while maintaining aspect ratio
            scale = min(target_size[0] / img_h, target_size[1] / img_w)
            resize_h = int(img_h * scale)
            resize_w = int(img_w * scale)
            
            resize_img = cv2.resize(img, (resize_w, resize_h))
            
            # Create new empty image with target size
            new_img = np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
            new_img[:resize_h, :resize_w, :] = resize_img
            
            # Calculate ratio for annotation scaling
            ratio_h = resize_h / img_h
            ratio_w = resize_w / img_w
            
            return new_img, [ratio_h, ratio_w]
        else:
            # Direct resize to target size
            resize_img = cv2.resize(img, (target_size[1], target_size[0]))
            ratio_h = target_size[0] / img_h
            ratio_w = target_size[1] / img_w
            return resize_img, [ratio_h, ratio_w]
    
    def normalize(self, img):
        """Normalize image with mean and std."""
        img = img.astype(np.float32) / 255.0
        img -= self.mean
        img /= self.std
        return img
    
    def preprocess_mask(self, mask, target_size=None):
        """Process segmentation masks."""
        if target_size is None:
            target_size = self.image_shape
        
        # Resize mask to target size
        mask = cv2.resize(
            mask, (target_size[1], target_size[0]), 
            interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def process_text_regions(self, text_regions, ratio_h, ratio_w):
        """Process text region coordinates with resize ratios."""
        processed_regions = []
        for region in text_regions:
            x1, y1, x2, y2 = region
            # Scale coordinates
            x1 = int(x1 * ratio_w)
            y1 = int(y1 * ratio_h)
            x2 = int(x2 * ratio_w)
            y2 = int(y2 * ratio_h)
            processed_regions.append([x1, y1, x2, y2])
        
        return processed_regions
    
    def process_table_cells(self, cells, ratio_h, ratio_w):
        """Process table cell coordinates with resize ratios."""
        processed_cells = []
        for cell in cells:
            # Each cell might have format [row_start, row_end, col_start, col_end, x1, y1, x2, y2]
            row_start, row_end, col_start, col_end, x1, y1, x2, y2 = cell
            # Scale coordinates
            x1 = int(x1 * ratio_w)
            y1 = int(y1 * ratio_h)
            x2 = int(x2 * ratio_w)
            y2 = int(y2 * ratio_h)
            processed_cells.append([row_start, row_end, col_start, col_end, x1, y1, x2, y2])
        
        return processed_cells
    
    def __call__(self, data):
        """
        Process input data for OmniParser.
        
        Args:
            data (dict): Input data with image and annotations
            
        Returns:
            dict: Processed data ready for model
        """
        img = data['image']
        
        # Apply augmentation if enabled
        if self.augmentation and random.random() < 0.3:
            img = self.iaa_aug(img)
            
        # Resize image
        img, [ratio_h, ratio_w] = self.resize_image(img, self.image_shape)
        
        # Normalize image
        img = self.normalize(img)
        
        # Transpose from HWC to CHW format
        img = img.transpose(2, 0, 1)
        
        # Process masks if available
        if 'text_mask' in data:
            data['text_mask'] = self.preprocess_mask(data['text_mask'], self.image_shape)
        
        if 'center_mask' in data:
            data['center_mask'] = self.preprocess_mask(data['center_mask'], self.image_shape)
        
        if 'border_mask' in data:
            data['border_mask'] = self.preprocess_mask(data['border_mask'], self.image_shape)
        
        if 'structure_mask' in data:
            data['structure_mask'] = self.preprocess_mask(data['structure_mask'], self.image_shape)
        
        if 'boundary_mask' in data:
            data['boundary_mask'] = self.preprocess_mask(data['boundary_mask'], self.image_shape)
        
        # Process regions/boxes if available
        if 'text_regions' in data:
            data['text_regions'] = self.process_text_regions(data['text_regions'], ratio_h, ratio_w)
        
        if 'table_cells' in data:
            data['table_cells'] = self.process_table_cells(data['table_cells'], ratio_h, ratio_w)
        
        # Update with processed image
        data['image'] = img
        
        # Add resize ratios for postprocessing
        data['ratio_h'] = ratio_h
        data['ratio_w'] = ratio_w
        
        return data
