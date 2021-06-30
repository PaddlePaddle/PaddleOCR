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
import numpy as np
class TableMetric(object):
    def __init__(self, main_indicator='acc', **kwargs):
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, pred, batch, *args, **kwargs):
        structure_probs = pred['structure_probs'].numpy()
        structure_labels = batch[1]
        correct_num = 0
        all_num = 0
        structure_probs = np.argmax(structure_probs, axis=2)
        structure_labels = structure_labels[:, 1:]
        batch_size = structure_probs.shape[0]
        for bno in range(batch_size):
            all_num += 1
            if (structure_probs[bno] == structure_labels[bno]).all():
                correct_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        return {
            'acc': correct_num * 1.0 / all_num,
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
            }
        """
        acc = 1.0 * self.correct_num / self.all_num
        self.reset()
        return {'acc': acc}

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
