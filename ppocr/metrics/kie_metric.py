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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

__all__ = ['KIEMetric']


class KIEMetric(object):
    def __init__(self, main_indicator='hmean', **kwargs):
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        nodes, _ = preds
        gts, tag = batch[4].squeeze(0), batch[5].tolist()[0]
        gts = gts[:tag[0], :1].reshape([-1])
        result = self.compute_f1_score(nodes, gts)
        self.results.append(result)

    def compute_f1_score(self, preds, gts):
        preds = preds.numpy()
        ignores = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25]
        C = preds.shape[1]
        classes = np.array(sorted(set(range(C)) - set(ignores)))
        hist = np.bincount(
            (gts * C).astype('int64') + preds.argmax(1), minlength=C
            **2).reshape([C, C]).astype('float32')
        diag = np.diag(hist)
        recalls = diag / hist.sum(1).clip(min=1)
        precisions = diag / hist.sum(0).clip(min=1)
        f1 = 2 * recalls * precisions / (recalls + precisions).clip(min=1e-8)
        return f1[classes]

    def combine_results(self, results):
        data = {'hmean': np.mean(results[0])}
        return data

    def get_metric(self):
        metircs = self.combine_results(self.results)
        self.reset()
        return metircs

    def reset(self):
        self.results = []  # clear results
