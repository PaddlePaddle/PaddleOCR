# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


class UniMERNetLoss(nn.Layer):
    def __init__(self, length_aware=True, vocab_size=50000):
        super(UniMERNetLoss, self).__init__()
        self.ignore_index = -100
        self.vocab_size = vocab_size
        self.pad_token_id = 1
        self.length_aware = length_aware
        self.cross = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.ignore_index
        )
        self.counting_loss_fct = nn.SmoothL1Loss()

    def _get_count_gt(self, labels):
        mask = (labels != self.pad_token_id).cast("float32")
        one_hot_labels = F.one_hot(
            labels, num_classes=self.vocab_size
        ) * mask.unsqueeze(-1)
        count_gt = paddle.sum(one_hot_labels, axis=1)
        return count_gt

    def forward(self, preds, batch):
        logits, count_pred, masked_label = preds
        labels = batch[1][:, 1:]
        word_loss = self.cross(
            paddle.reshape(logits, [-1, logits.shape[-1]]),
            paddle.reshape(masked_label[:, 1:], [-1]),
        )
        loss = word_loss
        if self.length_aware:
            count_gt = self._get_count_gt(labels)
            count_gt = paddle.log(count_gt.cast(paddle.float32) + 1)
            count_loss = self.counting_loss_fct(count_pred, count_gt)
            loss += 0.5 * count_loss
        return {"loss": loss, "word_loss": word_loss, "count_loss": count_loss}
