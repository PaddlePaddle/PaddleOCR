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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn


class ParseQLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(ParseQLoss, self).__init__()

    def forward(self, predicts, targets):
        label = targets[1]  # label
        label_len = targets[2]
        max_step = paddle.max(label_len).cpu().numpy()[0] + 2
        tgt = label[:, :max_step]

        logits_list = predicts["logits_list"]
        pad_id = predicts["pad_id"]
        eos_id = predicts["eos_id"]

        tgt_out = tgt[:, 1:]
        loss = 0
        loss_numel = 0
        n = (tgt_out != pad_id).sum().item()

        for i, logits in enumerate(logits_list):
            loss += n * paddle.nn.functional.cross_entropy(
                input=logits, label=tgt_out.flatten(), ignore_index=pad_id
            )
            loss_numel += n
            if i == 1:
                tgt_out = paddle.where(condition=tgt_out == eos_id, x=pad_id, y=tgt_out)
                n = (tgt_out != pad_id).sum().item()
        loss /= loss_numel

        return {"loss": loss}
