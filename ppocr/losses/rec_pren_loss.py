# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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


class PRENLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(PRENLoss, self).__init__()
        # note: 0 is padding idx
        self.loss_func = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    def forward(self, predicts, batch):
        loss = self.loss_func(predicts, batch[1].astype('int64'))
        return {'loss': loss}
