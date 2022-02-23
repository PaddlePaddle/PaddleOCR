# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import math
from functools import partial

import numpy as np
import paddle
from paddle.metric import Metric

__all__ = ['DetectionF1', 'CorrectionF1']


class DetectionF1(Metric):
    def __init__(self, pos_label=1, name='DetectionF1', *args, **kwargs):
        super(DetectionF1, self).__init__(*args, **kwargs)
        self.pos_label = pos_label
        self._name = name
        self.reset()

    def update(self, preds, labels, length, *args):
        # [B, T, 2]
        pred_labels = preds.argmax(axis=-1)
        for i, label_length in enumerate(length):
            pred_label = pred_labels[i][1:1 + label_length]
            label = labels[i][1:1 + label_length]
            # the sequence has errors
            if (label == self.pos_label).any():
                if (pred_label == label).all():
                    self.tp += 1
                else:
                    self.fn += 1
            else:
                if (label != pred_label).any():
                    self.fp += 1

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def accumulate(self):
        precision = np.nan
        if self.tp + self.fp > 0:
            precision = self.tp / (self.tp + self.fp)
        recall = np.nan
        if self.tp + self.fn > 0:
            recall = self.tp / (self.tp + self.fn)
        if self.tp == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return f1, precision, recall

    def name(self):
        """
        Returns name of the metric instance.

        Returns:
           str: The name of the metric instance.

        """
        return self._name


class CorrectionF1(DetectionF1):
    def __init__(self, pos_label=1, name='CorrectionF1', *args, **kwargs):
        super(CorrectionF1, self).__init__(pos_label, name, *args, **kwargs)

    def update(self, det_preds, det_labels, corr_preds, corr_labels, length,
               *args):
        # [B, T, 2]
        det_preds_labels = det_preds.argmax(axis=-1)
        corr_preds_labels = corr_preds.argmax(axis=-1)

        for i, label_length in enumerate(length):
            # Ignore [CLS] token, so calculate from position 1.
            det_preds_label = det_preds_labels[i][1:1 + label_length]
            det_label = det_labels[i][1:1 + label_length]
            corr_preds_label = corr_preds_labels[i][1:1 + label_length]
            corr_label = corr_labels[i][1:1 + label_length]

            # The sequence has any errors.
            if (det_label == self.pos_label).any():
                corr_pred_label = corr_preds_label * det_preds_label
                corr_label = det_label * corr_label
                if (corr_pred_label == corr_label).all():
                    self.tp += 1
                else:
                    self.fn += 1
            else:
                if (det_label != det_preds_label).any():
                    self.fp += 1
