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
Evaluation metrics for OmniParser unified framework
"""

import numpy as np
import paddle
from shapely.geometry import Polygon
import cv2

__all__ = ['MultiTaskMetric']


class TextDetectionMetric(object):
    """Metric for text detection evaluation"""
    def __init__(self, iou_threshold=0.5, **kwargs):
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset metrics"""
        self.true_positives = 0  # TP: detected boxes that match ground truth
        self.false_positives = 0  # FP: detected boxes that don't match ground truth
        self.false_negatives = 0  # FN: ground truth boxes not detected
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        # Convert to [x1, y1, x2, y2] format if not already
        if len(box1) > 4:  # Polygon points format
            box1 = self._polygon_to_rect(box1)
        if len(box2) > 4:  # Polygon points format
            box2 = self._polygon_to_rect(box2)
            
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate areas
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate IoU
        union = box1_area + box2_area - intersection
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def _polygon_to_rect(self, polygon):
        """Convert polygon points to rectangle [x1, y1, x2, y2]"""
        points = np.reshape(polygon, (-1, 2))
        min_x = np.min(points[:, 0])
        min_y = np.min(points[:, 1])
        max_x = np.max(points[:, 0])
        max_y = np.max(points[:, 1])
        return [min_x, min_y, max_x, max_y]
    
    def _match_boxes(self, gt_boxes, pred_boxes):
        """Match detected boxes to ground truth boxes"""
        if len(gt_boxes) == 0:
            return [], list(range(len(pred_boxes))), []
        
        if len(pred_boxes) == 0:
            return [], [], list(range(len(gt_boxes)))
            
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                iou_matrix[i, j] = self._calculate_iou(gt_box, pred_box)
        
        # Find matches using greedy algorithm
        matches = []  # (gt_idx, pred_idx)
        unmatched_pred = list(range(len(pred_boxes)))
        unmatched_gt = list(range(len(gt_boxes)))
        
        # Sort IoU in descending order
        for gt_idx, pred_idx in zip(*np.unravel_index(iou_matrix.flatten().argsort()[::-1], iou_matrix.shape)):
            # If IoU is below threshold or gt/pred already matched, skip
            if iou_matrix[gt_idx, pred_idx] < self.iou_threshold:
                break
                
            if gt_idx in unmatched_gt and pred_idx in unmatched_pred:
                matches.append((gt_idx, pred_idx))
                unmatched_gt.remove(gt_idx)
                unmatched_pred.remove(pred_idx)
        
        return matches, unmatched_pred, unmatched_gt
    
    def update(self, pred_boxes, gt_boxes):
        """Update metrics with batch results"""
        matches, unmatched_pred, unmatched_gt = self._match_boxes(gt_boxes, pred_boxes)
        
        # Update metrics
        self.true_positives += len(matches)
        self.false_positives += len(unmatched_pred)
        self.false_negatives += len(unmatched_gt)
    
    def compute_metrics(self):
        """Compute precision, recall, and F-score"""
        if self.true_positives + self.false_positives == 0:
            precision = 0
        else:
            precision = self.true_positives / (self.true_positives + self.false_positives)
            
        if self.true_positives + self.false_negatives == 0:
            recall = 0
        else:
            recall = self.true_positives / (self.true_positives + self.false_negatives)
            
        if precision + recall == 0:
            f_score = 0
        else:
            f_score = 2 * precision * recall / (precision + recall)
            
        return {
            'text_box_precision': precision,
            'text_box_recall': recall,
            'text_box_f_score': f_score
        }


class TableStructureMetric(object):
    """Metric for table structure recognition evaluation"""
    def __init__(self, **kwargs):
        self.reset()
    
    def reset(self):
        """Reset metrics"""
        self.true_positives = 0  # TP: correctly detected cells
        self.false_positives = 0  # FP: detected cells that don't match ground truth
        self.false_negatives = 0  # FN: ground truth cells not detected
    
    def _cell_match(self, pred_cell, gt_cell, row_match_threshold=0.5, col_match_threshold=0.5):
        """Check if predicted cell matches ground truth cell"""
        # Each cell has [row_start, row_end, col_start, col_end, x1, y1, x2, y2]
        # Check if cell indices match
        row_match = (pred_cell[0] == gt_cell[0] and pred_cell[1] == gt_cell[1])
        col_match = (pred_cell[2] == gt_cell[2] and pred_cell[3] == gt_cell[3])
        
        # If indices match, return True
        if row_match and col_match:
            return True
            
        # Otherwise, check spatial overlap
        pred_box = pred_cell[4:]  # [x1, y1, x2, y2]
        gt_box = gt_cell[4:]  # [x1, y1, x2, y2]
        
        # Calculate intersection
        x1 = max(pred_box[0], gt_box[0])
        y1 = max(pred_box[1], gt_box[1])
        x2 = min(pred_box[2], gt_box[2])
        y2 = min(pred_box[3], gt_box[3])
        
        if x2 <= x1 or y2 <= y1:
            return False
            
        intersection = (x2 - x1) * (y2 - y1)
        pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        
        # Calculate IoU
        union = pred_area + gt_area - intersection
        iou = intersection / union if union > 0 else 0
        
        return iou > 0.7  # High threshold for cell matching
    
    def _match_cells(self, gt_cells, pred_cells):
        """Match detected cells to ground truth cells"""
        if len(gt_cells) == 0:
            return [], list(range(len(pred_cells))), []
        
        if len(pred_cells) == 0:
            return [], [], list(range(len(gt_cells)))
        
        # Create match matrix
        match_matrix = np.zeros((len(gt_cells), len(pred_cells)), dtype=bool)
        for i, gt_cell in enumerate(gt_cells):
            for j, pred_cell in enumerate(pred_cells):
                match_matrix[i, j] = self._cell_match(pred_cell, gt_cell)
        
        # Find matches using greedy algorithm
        matches = []
        unmatched_pred = list(range(len(pred_cells)))
        unmatched_gt = list(range(len(gt_cells)))
        
        # For each ground truth cell, find best matching predicted cell
        for gt_idx in range(len(gt_cells)):
            if gt_idx not in unmatched_gt:
                continue
                
            best_match = None
            best_match_idx = -1
            
            for pred_idx in unmatched_pred:
                if match_matrix[gt_idx, pred_idx]:
                    best_match = pred_idx
                    best_match_idx = pred_idx
                    break
                    
            if best_match is not None:
                matches.append((gt_idx, best_match_idx))
                unmatched_gt.remove(gt_idx)
                unmatched_pred.remove(best_match_idx)
        
        return matches, unmatched_pred, unmatched_gt
    
    def update(self, pred_structure, gt_structure):
        """Update metrics with batch results"""
        pred_cells = pred_structure.get('cells', [])
        gt_cells = gt_structure.get('cells', [])
        
        matches, unmatched_pred, unmatched_gt = self._match_cells(gt_cells, pred_cells)
        
        # Update metrics
        self.true_positives += len(matches)
        self.false_positives += len(unmatched_pred)
        self.false_negatives += len(unmatched_gt)
    
    def compute_metrics(self):
        """Compute precision, recall, and F-score"""
        if self.true_positives + self.false_positives == 0:
            precision = 0
        else:
            precision = self.true_positives / (self.true_positives + self.false_positives)
            
        if self.true_positives + self.false_negatives == 0:
            recall = 0
        else:
            recall = self.true_positives / (self.true_positives + self.false_negatives)
            
        if precision + recall == 0:
            f_score = 0
        else:
            f_score = 2 * precision * recall / (precision + recall)
            
        return {
            'table_structure_precision': precision,
            'table_structure_recall': recall,
            'table_structure_f_score': f_score
        }


class KIEMetric(object):
    """Metric for KIE evaluation"""
    def __init__(self, **kwargs):
        self.reset()
    
    def reset(self):
        """Reset metrics"""
        self.correct = 0  # Correctly classified entities
        self.total_pred = 0  # Total predicted entities
        self.total_gt = 0  # Total ground truth entities
    
    def update(self, pred_entities, gt_entities):
        """Update metrics with batch results"""
        if not gt_entities:
            self.total_pred += len(pred_entities)
            return
        
        if not pred_entities:
            self.total_gt += len(gt_entities)
            return
        
        # Count predicted and ground truth entities
        self.total_pred += len(pred_entities)
        self.total_gt += len(gt_entities)
        
        # Count correctly classified entities
        for gt_entity in gt_entities:
            gt_text = gt_entity.get('text', '')
            gt_label = gt_entity.get('label', '')
            
            for pred_entity in pred_entities:
                pred_text = pred_entity.get('text', '')
                pred_label = pred_entity.get('label', '')
                
                # Text and label must match for correct classification
                if gt_text == pred_text and gt_label == pred_label:
                    self.correct += 1
                    break  # Found a match, move to next ground truth
    
    def compute_metrics(self):
        """Compute precision, recall, and F-score"""
        if self.total_pred == 0:
            precision = 0
        else:
            precision = self.correct / self.total_pred
            
        if self.total_gt == 0:
            recall = 0
        else:
            recall = self.correct / self.total_gt
            
        if precision + recall == 0:
            f_score = 0
        else:
            f_score = 2 * precision * recall / (precision + recall)
            
        return {
            'kie_precision': precision,
            'kie_recall': recall,
            'kie_f_score': f_score
        }


class MultiTaskMetric(object):
    """
    Unified metrics for OmniParser multi-task evaluation
    """
    def __init__(self, main_indicator='hmean', **kwargs):
        self.main_indicator = main_indicator
        self.reset()
        
        # Initialize task-specific metrics
        self.text_metric = TextDetectionMetric(**kwargs)
        self.table_metric = TableStructureMetric(**kwargs)
        self.kie_metric = KIEMetric(**kwargs)
        
        # Store task weights for harmonic mean calculation
        self.weights = {
            'text_box_f_score': kwargs.get('text_box_weight', 1.0),
            'table_structure_f_score': kwargs.get('table_structure_weight', 1.0),
            'kie_f_score': kwargs.get('kie_weight', 1.0)
        }
        
    def reset(self):
        """Reset all metrics"""
        self.text_metric = TextDetectionMetric()
        self.table_metric = TableStructureMetric()
        self.kie_metric = KIEMetric()
    
    def __call__(self, preds, batch):
        """
        Update metrics with batch predictions
        
        Args:
            preds (dict): Prediction results
            batch (dict): Batch data with ground truth
        """
        # Update text detection metrics
        if 'text_boxes' in preds:
            pred_boxes = preds['text_boxes']
            gt_boxes = batch.get('text_regions', [])
            self.text_metric.update(pred_boxes, gt_boxes)
        
        # Update table structure metrics
        if 'table_structure' in preds:
            pred_structure = preds['table_structure']
            gt_structure = {
                'table_region': batch.get('table_region', None),
                'row_positions': batch.get('row_positions', []),
                'col_positions': batch.get('col_positions', []),
                'cells': batch.get('table_cells', [])
            }
            self.table_metric.update(pred_structure, gt_structure)
        
        # Update KIE metrics
        if 'entities' in preds:
            pred_entities = preds['entities']
            gt_entities = batch.get('entities', [])
            self.kie_metric.update(pred_entities, gt_entities)
    
    def get_metric(self):
        """
        Compute and return all metrics
        
        Returns:
            dict: Dictionary with all metrics
        """
        # Calculate metrics for each task
        text_metrics = self.text_metric.compute_metrics()
        table_metrics = self.table_metric.compute_metrics()
        kie_metrics = self.kie_metric.compute_metrics()
        
        # Combine all metrics
        metrics = {}
        metrics.update(text_metrics)
        metrics.update(table_metrics)
        metrics.update(kie_metrics)
        
        # Calculate harmonic mean of F-scores as the main metric
        f_scores = [
            metrics['text_box_f_score'] * self.weights['text_box_f_score'],
            metrics['table_structure_f_score'] * self.weights['table_structure_f_score'],
            metrics['kie_f_score'] * self.weights['kie_f_score']
        ]
        
        # Filter out zero weights
        f_scores = [f for f, w in zip(f_scores, self.weights.values()) if w > 0]
        
        if len(f_scores) > 0:
            hmean = len(f_scores) / sum(1.0 / (f + 1e-10) for f in f_scores)
        else:
            hmean = 0
            
        metrics['hmean'] = hmean
        
        return metrics
