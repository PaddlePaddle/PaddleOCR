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

import paddle
from paddle import nn
from paddle.nn import functional as F
import numpy as np
from ppocr.utils.iou import iou


class PSELoss(nn.Layer):
    def __init__(self,
                 alpha,
                 ohem_ratio=3,
                 kernel_sample_mask='pred',
                 reduction='sum',
                 **kwargs):
        """Implement PSE Loss.
        """
        super(PSELoss, self).__init__()
        assert reduction in ['sum', 'mean', 'none']
        self.alpha = alpha
        self.ohem_ratio = ohem_ratio
        self.kernel_sample_mask = kernel_sample_mask
        self.reduction = reduction

    def forward(self, outputs, labels):
        predicts = outputs['maps']
        predicts = F.interpolate(predicts, scale_factor=4)

        texts = predicts[:, 0, :, :]
        kernels = predicts[:, 1:, :, :]
        gt_texts, gt_kernels, training_masks = labels[1:]

        # text loss
        selected_masks = self.ohem_batch(texts, gt_texts, training_masks)

        loss_text = self.dice_loss(texts, gt_texts, selected_masks)
        iou_text = iou((texts > 0).astype('int64'),
                       gt_texts,
                       training_masks,
                       reduce=False)
        losses = dict(loss_text=loss_text, iou_text=iou_text)

        # kernel loss
        loss_kernels = []
        if self.kernel_sample_mask == 'gt':
            selected_masks = gt_texts * training_masks
        elif self.kernel_sample_mask == 'pred':
            selected_masks = (
                F.sigmoid(texts) > 0.5).astype('float32') * training_masks

        for i in range(kernels.shape[1]):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.dice_loss(kernel_i, gt_kernel_i,
                                           selected_masks)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = paddle.mean(paddle.stack(loss_kernels, axis=1), axis=1)
        iou_kernel = iou((kernels[:, -1, :, :] > 0).astype('int64'),
                         gt_kernels[:, -1, :, :],
                         training_masks * gt_texts,
                         reduce=False)
        losses.update(dict(loss_kernels=loss_kernels, iou_kernel=iou_kernel))
        loss = self.alpha * loss_text + (1 - self.alpha) * loss_kernels
        losses['loss'] = loss
        if self.reduction == 'sum':
            losses = {x: paddle.sum(v) for x, v in losses.items()}
        elif self.reduction == 'mean':
            losses = {x: paddle.mean(v) for x, v in losses.items()}
        return losses

    def dice_loss(self, input, target, mask):
        input = F.sigmoid(input)

        input = input.reshape([input.shape[0], -1])
        target = target.reshape([target.shape[0], -1])
        mask = mask.reshape([mask.shape[0], -1])

        input = input * mask
        target = target * mask

        a = paddle.sum(input * target, 1)
        b = paddle.sum(input * input, 1) + 0.001
        c = paddle.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)
        return 1 - d

    def ohem_single(self, score, gt_text, training_mask, ohem_ratio=3):
        pos_num = int(paddle.sum((gt_text > 0.5).astype('float32'))) - int(
            paddle.sum(
                paddle.logical_and((gt_text > 0.5), (training_mask <= 0.5))
                .astype('float32')))

        if pos_num == 0:
            # selected_mask = gt_text.copy() * 0 # may be not good
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(
                [1, selected_mask.shape[0], selected_mask.shape[1]]).astype(
                    'float32')
            return selected_mask

        neg_num = int(paddle.sum((gt_text <= 0.5).astype('float32')))
        neg_num = int(min(pos_num * ohem_ratio, neg_num))

        if neg_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.view(
                1, selected_mask.shape[0],
                selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_score = paddle.masked_select(score, gt_text <= 0.5)
        neg_score_sorted = paddle.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]

        selected_mask = paddle.logical_and(
            paddle.logical_or((score >= threshold), (gt_text > 0.5)),
            (training_mask > 0.5))
        selected_mask = selected_mask.reshape(
            [1, selected_mask.shape[0], selected_mask.shape[1]]).astype(
                'float32')
        return selected_mask

    def ohem_batch(self, scores, gt_texts, training_masks, ohem_ratio=3):
        selected_masks = []
        for i in range(scores.shape[0]):
            selected_masks.append(
                self.ohem_single(scores[i, :, :], gt_texts[i, :, :],
                                 training_masks[i, :, :], ohem_ratio))

        selected_masks = paddle.concat(selected_masks, 0).astype('float32')
        return selected_masks
