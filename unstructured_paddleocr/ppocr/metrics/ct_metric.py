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

import os
from scipy import io
import numpy as np

from ppocr.utils.e2e_metric.Deteval import combine_results, get_score_C


class CTMetric(object):
    def __init__(self, main_indicator, delimiter='\t', **kwargs):
        self.delimiter = delimiter
        self.main_indicator = main_indicator
        self.reset()

    def reset(self):
        self.results = []  # clear results

    def __call__(self, preds, batch, **kwargs):
        # NOTE: only support bs=1 now, as the label length of different sample is Unequal 
        assert len(
            preds) == 1, "CentripetalText test now only suuport batch_size=1."
        label = batch[2]
        text = batch[3]
        pred = preds[0]['points']
        result = get_score_C(label, text, pred)

        self.results.append(result)

    def get_metric(self):
        """
        Input format: y0,x0, ..... yn,xn. Each detection is separated by the end of line token ('\n')'
        """
        metrics = combine_results(self.results, rec_flag=False)
        self.reset()
        return metrics
