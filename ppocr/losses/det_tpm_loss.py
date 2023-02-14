# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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


class TextLoss(nn.Layer):
    def __init__(self, k):
        super().__init__()
        self.MSE_loss = nn.MSELoss(reduction='none')
        self.KL_loss = nn.KLDivLoss(reduction='none')
        self.k = 3.14159  #cfg.fuc_k

    def single_loss(self, pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        sum_loss = 0.
        pre_loss = pre_loss.reshape([batch_size, -1])
        loss_label = loss_label.reshape([batch_size, -1])
        eps = 0.001
        for i in range(batch_size):
            average_number = 0
            positive_pixel = len(pre_loss[i][(loss_label[i] >= eps)])
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = paddle.mean(pre_loss[i][(loss_label[i] >= eps)])
                sum_loss += posi_loss
                if len(pre_loss[i][(loss_label[i] < eps)]) < 3 * positive_pixel:
                    nega_loss = paddle.mean(pre_loss[i][(loss_label[i] < eps)])
                    average_number += len(pre_loss[i][(loss_label[i] < eps)])
                else:
                    nega_loss = paddle.mean(
                        paddle.topk(pre_loss[i][(loss_label[i] < eps)], 3 *
                                    positive_pixel)[0])
                    average_number += 3 * positive_pixel
                sum_loss += nega_loss
            else:
                nega_loss = paddle.mean(paddle.topk(pre_loss[i], 100)[0])
                average_number += 100
                sum_loss += nega_loss

        return sum_loss

    def smooth_l1_loss(self, inputs, target, sigma=9.0, reduction='mean'):
        try:
            diff = paddle.abs(inputs - target)
            less_one = paddle.cast(diff < 1.0 / sigma, "float32")
            loss = less_one * 0.5 * diff ** 2 * sigma \
                   + paddle.abs(paddle.to_tensor(1.0) - less_one) * (diff - 0.5 / sigma)
            loss = loss if loss.numel() > 0 else paddle.zeros_like(inputs)
        except Exception as e:
            print('paddle smooth L1 Exception:', e)
            loss = paddle.zeros_like(inputs)
        if reduction == 'sum':
            loss = paddle.sum(loss)
        elif reduction == 'mean':
            loss = paddle.mean(loss)
        else:
            loss = loss
        return loss

    def sigmoid_alpha(self, x, d):
        eps = paddle.to_tensor(0.0001)
        alpha = self.k
        dm = paddle.where(d >= eps, d, eps)
        betak = (1 + np.exp(-alpha)) / (1 - np.exp(-alpha))
        res = (2 * F.sigmoid(x * alpha / dm) - 1) * betak

        return F.relu(res)

    def forward(self, inputs, train_mask, tr_mask):
        """
          calculate textsnake loss
        """
        b, c, h, w = inputs.shape
        loss_sum = paddle.to_tensor(0.)
        for i in range(c):
            reg_loss = self.MSE_loss(
                F.sigmoid(inputs[:, i]), tr_mask[:, :, :, i])
            reg_loss = paddle.mul(reg_loss, paddle.cast(train_mask, "float32"))
            reg_loss = self.single_image_loss(reg_loss, tr_mask[:, :, :, i]) / b
            loss_sum = loss_sum + reg_loss

        return loss_sum
