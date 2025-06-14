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
Post-processing module for OmniParser outputs
"""

import cv2
import numpy as np
import paddle
import scipy.spatial as spatial
from shapely.geometry import Polygon, Point
from skimage.draw import polygon as drawpoly

__all__ = ['OmniParserPostProcess']


class TextSegPostProcess(object):
    """Text segmentation post-processing for OmniParser"""
    def __init__(self, text_threshold=0.5, center_threshold=0.5, border_threshold=0.5, **kwargs):
        self.text_threshold = text_threshold
        self.center_threshold = center_threshold
        self.border_threshold = border_threshold
        
    def _get_contours(self, text_score, center_score, border_score):
        """Extract contours from segmentation maps"""
        # Binarize score maps
        text_mask = text_score > self.text_threshold
        center_mask = center_score > self.center_threshold
        border_mask = border_score > self.border_threshold
        
        # Combine masks
        final_mask = text_mask & center_mask & (~border_mask)
        
        # Find contours
        final_mask = final_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    def _contours_to_boxes(self, contours, min_size=3):
        """Convert contours to bounding boxes"""
        boxes = []
        for contour in contours:
            # Filter small contours
            if len(contour) < min_size:
                continue
                
            # Get bounding box
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Convert to [x1, y1, x2, y2] format
            x_min, y_min = np.min(box, axis=0)
            x_max, y_max = np.max(box, axis=0)
            boxes.append([x_min, y_min, x_max, y_max])
            
        return boxes
    
    def __call__(self, preds, ratio_h=1.0, ratio_w=1.0):
        """
        Post-process text segmentation outputs
        
        Args:
            preds (dict): Model predictions
            ratio_h (float): Height scaling ratio for original image
            ratio_w (float): Width scaling ratio for original image
            
        Returns:
            list: List of text boxes
        """
        text_score = preds['text_prob'][0, 0].numpy()
        center_score = preds['center_prob'][0, 0].numpy()
        border_score = preds['border_prob'][0, 0].numpy()
        
        # Get contours and boxes
        contours = self._get_contours(text_score, center_score, border_score)
        boxes = self._contours_to_boxes(contours)
        
        # Scale boxes back to original image size
        scaled_boxes = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            scaled_boxes.append([
                int(x_min / ratio_w),
                int(y_min / ratio_h),
                int(x_max / ratio_w),
                int(y_max / ratio_h)
            ])
            
        return scaled_boxes


class TablePostProcess(object):
    """Table structure post-processing for OmniParser"""
    def __init__(self, structure_thresh=0.5, boundary_thresh=0.5, **kwargs):
        self.structure_thresh = structure_thresh
        self.boundary_thresh = boundary_thresh
        
    def _get_table_boundary(self, boundary_pred):
        """Extract table boundary from prediction"""
        # Obtain probability map for boundary
        boundary_prob = paddle.nn.functional.softmax(boundary_pred, axis=1)[0, 1].numpy()
        
        # Binarize probability map
        boundary_mask = (boundary_prob > self.boundary_thresh).astype(np.uint8) * 255
        
        # Find contours for table boundary
        contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest contour as table boundary
        if not contours:
            return None
            
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return [x, y, x + w, y + h]
    
    def _get_table_structure(self, structure_pred, table_region):
        """Extract table rows and columns from prediction"""
        # Get table region
        if table_region is None:
            return [], []
            
        x1, y1, x2, y2 = table_region
        
        # Obtain class probability maps (background, row line, column line)
        structure_probs = paddle.nn.functional.softmax(structure_pred, axis=1)[0].numpy()
        
        # Extract row and column probability maps
        row_prob = structure_probs[1, y1:y2, x1:x2]  # Class 1 for row line
        col_prob = structure_probs[2, y1:y2, x1:x2]  # Class 2 for column line
        
        # Binarize probability maps
        row_mask = (row_prob > self.structure_thresh).astype(np.uint8) * 255
        col_mask = (col_prob > self.structure_thresh).astype(np.uint8) * 255
        
        # Get row positions
        row_positions = []
        row_projection = np.sum(row_mask, axis=1) / 255
        for i in range(1, len(row_projection) - 1):
            if row_projection[i] > row_projection[i-1] and row_projection[i] > row_projection[i+1]:
                if row_projection[i] > 0.3 * max(row_projection):  # Filter weak lines
                    row_positions.append(i + y1)
                    
        # Get column positions
        col_positions = []
        col_projection = np.sum(col_mask, axis=0) / 255
        for i in range(1, len(col_projection) - 1):
            if col_projection[i] > col_projection[i-1] and col_projection[i] > col_projection[i+1]:
                if col_projection[i] > 0.3 * max(col_projection):  # Filter weak lines
                    col_positions.append(i + x1)
        
        return row_positions, col_positions
        
    def __call__(self, preds, ratio_h=1.0, ratio_w=1.0):
        """
        Post-process table structure outputs
        
        Args:
            preds (dict): Model predictions
            ratio_h (float): Height scaling ratio for original image
            ratio_w (float): Width scaling ratio for original image
            
        Returns:
            dict: Table structure information
        """
        structure_pred = preds['structure_pred']
        boundary_pred = preds['boundary_pred']
        
        # Get table boundary
        table_region = self._get_table_boundary(boundary_pred)
        
        if table_region is None:
            return {
                'table_region': None,
                'row_positions': [],
                'col_positions': [],
                'cells': []
            }
        
        # Get table rows and columns
        row_positions, col_positions = self._get_table_structure(structure_pred, table_region)
        
        # Extract cells from rows and columns
        cells = []
        for i in range(len(row_positions) - 1):
            for j in range(len(col_positions) - 1):
                cells.append([
                    i, i+1, j, j+1,  # Row start, row end, col start, col end
                    col_positions[j], row_positions[i], col_positions[j+1], row_positions[i+1]  # Cell coordinates
                ])
        
        # Scale back to original image size
        x1, y1, x2, y2 = table_region
        scaled_table_region = [
            int(x1 / ratio_w),
            int(y1 / ratio_h),
            int(x2 / ratio_w),
            int(y2 / ratio_h)
        ]
        
        scaled_row_positions = [int(pos / ratio_h) for pos in row_positions]
        scaled_col_positions = [int(pos / ratio_w) for pos in col_positions]
        
        scaled_cells = []
        for cell in cells:
            row_s, row_e, col_s, col_e, cx1, cy1, cx2, cy2 = cell
            scaled_cells.append([
                row_s, row_e, col_s, col_e,
                int(cx1 / ratio_w),
                int(cy1 / ratio_h),
                int(cx2 / ratio_w),
                int(cy2 / ratio_h)
            ])
        
        return {
            'table_region': scaled_table_region,
            'row_positions': scaled_row_positions,
            'col_positions': scaled_col_positions,
            'cells': scaled_cells
        }


class KIEPostProcess(object):
    """KIE post-processing for OmniParser"""
    def __init__(self, classes=None, **kwargs):
        # Entity class names
        self.classes = classes or [
            "other",
            "company",
            "address",
            "date",
            "total",
            "name"
        ]
    
    def __call__(self, preds, text_boxes=None, ratio_h=1.0, ratio_w=1.0):
        """
        Post-process KIE outputs
        
        Args:
            preds (dict): Model predictions
            text_boxes (list): Text boxes from text detection
            ratio_h (float): Height scaling ratio for original image
            ratio_w (float): Width scaling ratio for original image
            
        Returns:
            list: List of entities with their types
        """
        # During inference, we need to combine with text detection results
        if text_boxes is None:
            return []
        
        # KIE features
        kie_features = preds['kie_features']
        
        # Extract features for each text box
        # In real deployment, you would need OCR model to get text content for these boxes
        entities = []
        
        # For demonstration, return empty list since we need OCR to complete KIE
        return entities


class OmniParserPostProcess(object):
    """Post-processing for OmniParser unified framework"""
    def __init__(self, mode='all', **kwargs):
        self.mode = mode  # 'all', 'text', 'table', 'kie'
        
        # Initialize component post-processors
        if mode in ['all', 'text']:
            self.text_postprocess = TextSegPostProcess(**kwargs)
        else:
            self.text_postprocess = None
            
        if mode in ['all', 'table']:
            self.table_postprocess = TablePostProcess(**kwargs)
        else:
            self.table_postprocess = None
            
        if mode in ['all', 'kie']:
            self.kie_postprocess = KIEPostProcess(**kwargs)
        else:
            self.kie_postprocess = None
    
    def __call__(self, preds, data=None):
        """
        Post-process OmniParser outputs for all tasks
        
        Args:
            preds (dict): Model predictions
            data (dict): Input data with metadata
            
        Returns:
            dict: Processed results with text boxes, table structure, and entities
        """
        results = {}
        ratio_h = data.get('ratio_h', 1.0) if data is not None else 1.0
        ratio_w = data.get('ratio_w', 1.0) if data is not None else 1.0
        
        # Process text detection if available
        if self.text_postprocess is not None and any(k in preds for k in ['text_prob', 'center_prob', 'border_prob']):
            text_boxes = self.text_postprocess(preds, ratio_h, ratio_w)
            results['text_boxes'] = text_boxes
        else:
            results['text_boxes'] = []
        
        # Process table recognition if available
        if self.table_postprocess is not None and all(k in preds for k in ['structure_pred', 'boundary_pred']):
            table_structure = self.table_postprocess(preds, ratio_h, ratio_w)
            results['table_structure'] = table_structure
        else:
            results['table_structure'] = {
                'table_region': None,
                'row_positions': [],
                'col_positions': [],
                'cells': []
            }
        
        # Process KIE if available
        if self.kie_postprocess is not None and 'kie_features' in preds:
            entities = self.kie_postprocess(preds, results.get('text_boxes', []), ratio_h, ratio_w)
            results['entities'] = entities
        else:
            results['entities'] = []
        
        return results
