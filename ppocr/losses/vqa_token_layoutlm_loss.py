# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import nn


class VQASerTokenLayoutLMLoss(nn.Layer):
    def __init__(self, num_classes):
        super().__init__()
        self.loss_class = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.ignore_index = self.loss_class.ignore_index

    def forward(self, predicts, batch):
        labels = batch[1]
        attention_mask = batch[4]
        if attention_mask is not None:
            active_loss = attention_mask.reshape([-1, ]) == 1
            active_outputs = predicts.reshape(
                [-1, self.num_classes])[active_loss]
            active_labels = labels.reshape([-1, ])[active_loss]
            loss = self.loss_class(active_outputs, active_labels)
        else:
            loss = self.loss_class(
                predicts.reshape([-1, self.num_classes]),
                labels.reshape([-1, ]))
        return {'loss': loss}
