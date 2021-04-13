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

__all__ = ['E2EMetric']

from ppocr.utils.e2e_metric.Deteval import get_socre, combine_results
from ppocr.utils.e2e_utils.extract_textpoint_slow import get_dict


class E2EMetric(object):
    def __init__(self,
                 gt_mat_dir,
                 character_dict_path,
                 main_indicator='f_score_e2e',
                 **kwargs):
        self.gt_mat_dir = gt_mat_dir
        self.label_list = get_dict(character_dict_path)
        self.max_index = len(self.label_list)
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        img_id = batch[5][0]
        e2e_info_list = [{
            'points': det_polyon,
            'text': pred_str
        } for det_polyon, pred_str in zip(preds['points'], preds['strs'])]
        result = get_socre(self.gt_mat_dir, img_id, e2e_info_list)
        self.results.append(result)

    def get_metric(self):
        metircs = combine_results(self.results)
        self.reset()
        return metircs

    def reset(self):
        self.results = []  # clear results
