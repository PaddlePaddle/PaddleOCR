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

__all__ = ['E2EMetric']

from ppocr.utils.e2e_metric.Deteval import *


class E2EMetric(object):
    def __init__(self, main_indicator='f_score_e2e', **kwargs):
        self.label_list = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
            'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
            'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        '''
       batch: a list produced by dataloaders.
           image: np.ndarray  of shape (N, C, H, W).
           ratio_list: np.ndarray  of shape(N,2)
           polygons: np.ndarray  of shape (N, K, 4, 2), the polygons of objective regions.
           ignore_tags: np.ndarray  of shape (N, K), indicates whether a region is ignorable or not.
       preds: a list of dict produced by post process
            points: np.ndarray of shape (N, K, 4, 2), the polygons of objective regions.
       '''

        gt_polyons_batch = batch[2]
        temp_gt_strs_batch = batch[3]
        ignore_tags_batch = batch[4]
        gt_strs_batch = []
        temp_gt_strs_batch = temp_gt_strs_batch[0].tolist()
        for temp_list in temp_gt_strs_batch:
            t = ""
            for index in temp_list:
                if index < 36:
                    t += self.label_list[index]
            gt_strs_batch.append(t)

        for pred, gt_polyons, gt_strs, ignore_tags in zip(
                preds, gt_polyons_batch, gt_strs_batch, ignore_tags_batch):
            # prepare gt
            gt_info_list = [{
                'points': gt_polyon,
                'text': gt_str,
                'ignore': ignore_tag
            } for gt_polyon, gt_str, ignore_tag in
                            zip(gt_polyons, gt_strs, ignore_tags)]
            # prepare det
            e2e_info_list = [{
                'points': det_polyon,
                'text': pred_str
            } for det_polyon, pred_str in zip(pred['points'], preds['strs'])]
            result = get_socre(gt_info_list, e2e_info_list)
            self.results.append(result)

    def get_metric(self):
        """
        return metrics {
                 'precision': 0,
                 'recall': 0,
                 'hmean': 0
            }
        """
        metircs = combine_results(self.results)
        self.reset()
        return metircs

    def reset(self):
        self.results = []  # clear results
