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
from paddle import nn
import fasttext


class AsterLoss(nn.Layer):
    def __init__(self,
                 weight=None,
                 size_average=True,
                 ignore_index=-100,
                 sequence_normalize=False,
                 sample_normalize=True,
                 **kwargs):
        super(AsterLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.sequence_normalize = sequence_normalize
        self.sample_normalize = sample_normalize
        self.loss_func = paddle.nn.CosineSimilarity()

    def forward(self, predicts, batch):
        targets = batch[1].astype("int64")
        label_lengths = batch[2].astype('int64')
        # sem_target = batch[3].astype('float32')
        embedding_vectors = predicts['embedding_vectors']
        rec_pred = predicts['rec_pred']

        # semantic loss
        # print(embedding_vectors)
        # print(embedding_vectors.shape)
        # targets = fasttext[targets]
        # sem_loss = 1 - self.loss_func(embedding_vectors, targets)

        # rec loss
        batch_size, num_steps, num_classes = rec_pred.shape[0], rec_pred.shape[
            1], rec_pred.shape[2]
        assert len(targets.shape) == len(list(rec_pred.shape)) - 1, \
            "The target's shape and inputs's shape is [N, d] and [N, num_steps]"

        mask = paddle.zeros([batch_size, num_steps])
        for i in range(batch_size):
            mask[i, :label_lengths[i]] = 1
        mask = paddle.cast(mask, "float32")
        max_length = max(label_lengths)
        assert max_length == rec_pred.shape[1]
        targets = targets[:, :max_length]
        mask = mask[:, :max_length]
        rec_pred = paddle.reshape(rec_pred, [-1, rec_pred.shape[-1]])
        input = nn.functional.log_softmax(rec_pred, axis=1)
        targets = paddle.reshape(targets, [-1, 1])
        mask = paddle.reshape(mask, [-1, 1])
        # print("input:", input)
        output = -paddle.gather(input, index=targets, axis=1) * mask
        output = paddle.sum(output)
        if self.sequence_normalize:
            output = output / paddle.sum(mask)
        if self.sample_normalize:
            output = output / batch_size
        loss = output
        return {'loss': loss}  # , 'sem_loss':sem_loss}
