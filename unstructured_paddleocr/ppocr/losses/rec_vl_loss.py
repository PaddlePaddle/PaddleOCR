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
"""
This code is refer from:
https://github.com/wangyuxin87/VisionLAN
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn


class VLLoss(nn.Layer):
    def __init__(self, mode="LF_1", weight_res=0.5, weight_mas=0.5, **kwargs):
        super(VLLoss, self).__init__()
        self.loss_func = paddle.nn.loss.CrossEntropyLoss(reduction="mean")
        assert mode in ["LF_1", "LF_2", "LA"]
        self.mode = mode
        self.weight_res = weight_res
        self.weight_mas = weight_mas

    def flatten_label(self, target):
        label_flatten = []
        label_length = []
        for i in range(0, target.shape[0]):
            cur_label = target[i].tolist()
            label_flatten += cur_label[: cur_label.index(0) + 1]
            label_length.append(cur_label.index(0) + 1)
        label_flatten = paddle.to_tensor(label_flatten, dtype="int64")
        label_length = paddle.to_tensor(label_length, dtype="int32")
        return (label_flatten, label_length)

    def _flatten(self, sources, lengths):
        return paddle.concat([t[:l] for t, l in zip(sources, lengths)])

    def forward(self, predicts, batch):
        text_pre = predicts[0]
        target = batch[1].astype("int64")
        label_flatten, length = self.flatten_label(target)
        text_pre = self._flatten(text_pre, length)
        if self.mode == "LF_1":
            loss = self.loss_func(text_pre, label_flatten)
        else:
            text_rem = predicts[1]
            text_mas = predicts[2]
            target_res = batch[2].astype("int64")
            target_sub = batch[3].astype("int64")
            label_flatten_res, length_res = self.flatten_label(target_res)
            label_flatten_sub, length_sub = self.flatten_label(target_sub)
            text_rem = self._flatten(text_rem, length_res)
            text_mas = self._flatten(text_mas, length_sub)
            loss_ori = self.loss_func(text_pre, label_flatten)
            loss_res = self.loss_func(text_rem, label_flatten_res)
            loss_mas = self.loss_func(text_mas, label_flatten_sub)
            loss = loss_ori + loss_res * self.weight_res + loss_mas * self.weight_mas
        return {"loss": loss}
