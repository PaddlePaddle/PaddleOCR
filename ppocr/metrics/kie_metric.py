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
# The code is refer from: https://github.com/open-mmlab/mmocr/blob/main/mmocr/core/evaluation/kie_metric.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle

__all__ = ["KIEMetric"]


class KIEMetric(object):
    def __init__(self, main_indicator="hmean", **kwargs):
        self.main_indicator = main_indicator
        self.reset()
        self.node = []
        self.gt = []

    def __call__(self, preds, batch, **kwargs):
        nodes, _ = preds
        gts, tag = batch[4].squeeze(0), batch[5].tolist()[0]
        gts = gts[: tag[0], :1].reshape([-1])
        self.node.append(nodes.numpy())
        self.gt.append(gts)
        # result = self.compute_f1_score(nodes, gts)
        # self.results.append(result)

    def compute_f1_score(self, preds, gts):
        ignores = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25]
        C = preds.shape[1]
        classes = np.array(sorted(set(range(C)) - set(ignores)))
        hist = (
            np.bincount((gts * C).astype("int64") + preds.argmax(1), minlength=C**2)
            .reshape([C, C])
            .astype("float32")
        )
        diag = np.diag(hist)
        recalls = diag / hist.sum(1).clip(min=1)
        precisions = diag / hist.sum(0).clip(min=1)
        f1 = 2 * recalls * precisions / (recalls + precisions).clip(min=1e-8)
        return f1[classes]

    def combine_results(self, results):
        node = np.concatenate(self.node, 0)
        gts = np.concatenate(self.gt, 0)
        results = self.compute_f1_score(node, gts)
        data = {"hmean": results.mean()}
        return data

    def get_metric(self):
        metrics = self.combine_results(self.results)
        self.reset()
        return metrics

    def reset(self):
        self.results = []  # clear results
        self.node = []
        self.gt = []
