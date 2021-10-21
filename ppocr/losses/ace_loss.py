# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
import paddle.nn as nn


class ACELoss(nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(
            weight=None,
            ignore_index=0,
            reduction='none',
            soft_label=True,
            axis=-1)

    def __call__(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
            
        B, N = predicts.shape[:2]
        div = paddle.to_tensor([N]).astype('float32')

        predicts = nn.functional.softmax(predicts, axis=-1)
        aggregation_preds = paddle.sum(predicts, axis=1)
        aggregation_preds = paddle.divide(aggregation_preds, div)

        length = batch[2].astype("float32")
        batch = batch[3].astype("float32")
        batch[:, 0] = paddle.subtract(div, length)
        batch = paddle.divide(batch, div)

        loss = self.loss_func(aggregation_preds, batch)
        return {"loss_ace": loss}
