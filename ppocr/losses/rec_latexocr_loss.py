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

"""
This code is refer from:
https://github.com/lucidrains/x-transformers/blob/main/x_transformers/autoregressive_wrapper.py
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


class LaTeXOCRLoss(nn.Layer):
    """
    LaTeXOCR adopt CrossEntropyLoss for network training.
    """

    def __init__(self):
        super(LaTeXOCRLoss, self).__init__()
        self.ignore_index = -100
        self.cross = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.ignore_index
        )

    def forward(self, preds, batch):
        word_probs = preds
        labels = batch[1][:, 1:]
        word_loss = self.cross(
            paddle.reshape(word_probs, [-1, word_probs.shape[-1]]),
            paddle.reshape(labels, [-1]),
        )

        loss = word_loss
        return {"loss": loss}
