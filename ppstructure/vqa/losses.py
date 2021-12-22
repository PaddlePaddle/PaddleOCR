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

from paddle import nn


class SERLoss(nn.Layer):
    def __init__(self, num_classes):
        super().__init__()
        self.loss_class = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.ignore_index = self.loss_class.ignore_index

    def forward(self, labels, outputs, attention_mask):
        if attention_mask is not None:
            active_loss = attention_mask.reshape([-1, ]) == 1
            active_outputs = outputs.reshape(
                [-1, self.num_classes])[active_loss]
            active_labels = labels.reshape([-1, ])[active_loss]
            loss = self.loss_class(active_outputs, active_labels)
        else:
            loss = self.loss_class(
                outputs.reshape([-1, self.num_classes]), labels.reshape([-1, ]))
        return loss
