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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import io
import numpy as np

from ppocr.utils.e2e_metric.Deteval import combine_results, get_score_C


class CTMetric(object):
    def __init__(self, gt_dir, main_indicator, **kwargs):
        self.main_indicator = main_indicator
        self.gt_dir = gt_dir
        self.input_ids = []
        self.preds = []
        self.global_sigma = []
        self.global_tau = []
        self.reset()

    def reset(self):
        self.results = []  # clear results

    def __call__(self, preds, batch, **kwargs):
        self.preds = preds
        result = get_score_C(self.gt_dir, self.preds['input_id'],
                             self.preds['bboxes'])

        self.results.append(result)

    def get_metric(self):
        """
        Input format: y0,x0, ..... yn,xn. Each detection is separated by the end of line token ('\n')'
        """
        metrics = combine_results(self.results, rec_flag=False)
        self.reset()
        return metrics
