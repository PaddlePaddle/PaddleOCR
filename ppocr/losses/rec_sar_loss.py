from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn


class SARLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(SARLoss, self).__init__()
        self.loss_func = paddle.nn.loss.CrossEntropyLoss(
            reduction="mean", ignore_index=92)

    def forward(self, predicts, batch):
        predict = predicts[:, :
                           -1, :]  # ignore last index of outputs to be in same seq_len with targets
        label = batch[1].astype(
            "int64")[:, 1:]  # ignore first index of target in loss calculation
        batch_size, num_steps, num_classes = predict.shape[0], predict.shape[
            1], predict.shape[2]
        assert len(label.shape) == len(list(predict.shape)) - 1, \
            "The target's shape and inputs's shape is [N, d] and [N, num_steps]"

        inputs = paddle.reshape(predict, [-1, num_classes])
        targets = paddle.reshape(label, [-1])
        loss = self.loss_func(inputs, targets)
        return {'loss': loss}
