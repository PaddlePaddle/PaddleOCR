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


class PPFormulaNet_S_Loss(nn.Layer):
    """
    PP=FormulaNet-S adopt CrossEntropyLoss for network training.
    """

    def __init__(self, vocab_size=50000, parallel_step=1):
        super(PPFormulaNet_S_Loss, self).__init__()
        self.ignore_index = -100
        self.vocab_size = vocab_size
        self.parallel_step = int(parallel_step)
        self.pad_token_id = 1
        # ignore padding characters during training
        self.cross = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.ignore_index
        )

    def forward(self, preds, batch):
        logits, masked_label = preds

        word_loss = self.cross(
            paddle.reshape(logits, [-1, logits.shape[-1]]),
            paddle.reshape(masked_label[:, self.parallel_step :], [-1]),
        )
        loss = word_loss
        return {
            "loss": loss,
            "word_loss": word_loss,
        }


class PPFormulaNet_L_Loss(nn.Layer):
    """
    PPFormulaNet_L adopt CrossEntropyLoss for network training.
    """

    def __init__(self, vocab_size=50000):
        super(PPFormulaNet_L_Loss, self).__init__()
        self.ignore_index = -100
        self.vocab_size = vocab_size
        self.pad_token_id = 1
        # ignore padding characters during training
        self.cross = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.ignore_index
        )

    def forward(self, preds, batch):
        logits, masked_label = preds

        word_loss = self.cross(
            paddle.reshape(logits, [-1, logits.shape[-1]]),
            paddle.reshape(masked_label[:, 1:], [-1]),
        )
        loss = word_loss
        return {
            "loss": loss,
            "word_loss": word_loss,
        }
