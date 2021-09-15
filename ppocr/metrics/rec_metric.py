# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import Levenshtein


class RecMetric(object):
    def __init__(self, main_indicator='acc', **kwargs):
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            pred = pred.replace(" ", "")
            target = target.replace(" ", "")
            norm_edit_dis += Levenshtein.distance(pred, target) / max(
                len(pred), len(target), 1)
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        return {
            'acc': correct_num / all_num,
            'norm_edit_dis': 1 - norm_edit_dis / all_num
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        acc = 1.0 * self.correct_num / self.all_num
        norm_edit_dis = 1 - self.norm_edit_dis / self.all_num
        self.reset()
        return {'acc': acc, 'norm_edit_dis': norm_edit_dis}

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
        
