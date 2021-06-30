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


class ClsLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(ClsLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, predicts, batch):
        label = batch[1]
        loss = self.loss_func(input=predicts, label=label)
        return {'loss': loss}
