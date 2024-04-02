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
"""
This code is refer from:
https://github.com/shengtao96/CentripetalText/tree/main/models/loss
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import paddle.nn.functional as F
import numpy as np


def ohem_single(score, gt_text, training_mask):
    # online hard example mining

    pos_num = int(paddle.sum(gt_text > 0.5)) - int(
        paddle.sum((gt_text > 0.5) & (training_mask <= 0.5)))

    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = paddle.cast(
            selected_mask.reshape(
                (1, selected_mask.shape[0], selected_mask.shape[1])), "float32")
        return selected_mask

    neg_num = int(paddle.sum((gt_text <= 0.5) & (training_mask > 0.5)))
    neg_num = int(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = paddle.cast(
            selected_mask.reshape(
                (1, selected_mask.shape[0], selected_mask.shape[1])), "float32")
        return selected_mask

    # hard example
    neg_score = score[(gt_text <= 0.5) & (training_mask > 0.5)]
    neg_score_sorted = paddle.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) |
                     (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = paddle.cast(
        selected_mask.reshape(
            (1, selected_mask.shape[0], selected_mask.shape[1])), "float32")
    return selected_mask


def ohem_batch(scores, gt_texts, training_masks):
    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(
            ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[
                i, :, :]))

    selected_masks = paddle.cast(paddle.concat(selected_masks, 0), "float32")
    return selected_masks


def iou_single(a, b, mask, n_class):
    EPS = 1e-6
    valid = mask == 1
    a = a[valid]
    b = b[valid]
    miou = []

    # iou of each class
    for i in range(n_class):
        inter = paddle.cast(((a == i) & (b == i)), "float32")
        union = paddle.cast(((a == i) | (b == i)), "float32")

        miou.append(paddle.sum(inter) / (paddle.sum(union) + EPS))
    miou = sum(miou) / len(miou)
    return miou


def iou(a, b, mask, n_class=2, reduce=True):
    batch_size = a.shape[0]

    a = a.reshape((batch_size, -1))
    b = b.reshape((batch_size, -1))
    mask = mask.reshape((batch_size, -1))

    iou = paddle.zeros((batch_size, ), dtype="float32")
    for i in range(batch_size):
        iou[i] = iou_single(a[i], b[i], mask[i], n_class)

    if reduce:
        iou = paddle.mean(iou)
    return iou


class DiceLoss(nn.Layer):
    def __init__(self, loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, input, target, mask, reduce=True):
        batch_size = input.shape[0]
        input = F.sigmoid(input)  # scale to 0-1

        input = input.reshape((batch_size, -1))
        target = paddle.cast(target.reshape((batch_size, -1)), "float32")
        mask = paddle.cast(mask.reshape((batch_size, -1)), "float32")

        input = input * mask
        target = target * mask

        a = paddle.sum(input * target, axis=1)
        b = paddle.sum(input * input, axis=1) + 0.001
        c = paddle.sum(target * target, axis=1) + 0.001
        d = (2 * a) / (b + c)
        loss = 1 - d

        loss = self.loss_weight * loss

        if reduce:
            loss = paddle.mean(loss)

        return loss


class SmoothL1Loss(nn.Layer):
    def __init__(self, beta=1.0, loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight

        np_coord = np.zeros(shape=[640, 640, 2], dtype=np.int64)
        for i in range(640):
            for j in range(640):
                np_coord[i, j, 0] = j
                np_coord[i, j, 1] = i
        np_coord = np_coord.reshape((-1, 2))

        self.coord = self.create_parameter(
            shape=[640 * 640, 2],
            dtype="int32",  # NOTE: not support "int64" before paddle 2.3.1
            default_initializer=nn.initializer.Assign(value=np_coord))
        self.coord.stop_gradient = True

    def forward_single(self, input, target, mask, beta=1.0, eps=1e-6):
        batch_size = input.shape[0]

        diff = paddle.abs(input - target) * mask.unsqueeze(1)
        loss = paddle.where(diff < beta, 0.5 * diff * diff / beta,
                            diff - 0.5 * beta)
        loss = paddle.cast(loss.reshape((batch_size, -1)), "float32")
        mask = paddle.cast(mask.reshape((batch_size, -1)), "float32")
        loss = paddle.sum(loss, axis=-1)
        loss = loss / (mask.sum(axis=-1) + eps)

        return loss

    def select_single(self, distance, gt_instance, gt_kernel_instance,
                      training_mask):

        with paddle.no_grad():
            # paddle 2.3.1, paddle.slice not support:
            # distance[:, self.coord[:, 1], self.coord[:, 0]]
            select_distance_list = []
            for i in range(2):
                tmp1 = distance[i, :]
                tmp2 = tmp1[self.coord[:, 1], self.coord[:, 0]]
                select_distance_list.append(tmp2.unsqueeze(0))
            select_distance = paddle.concat(select_distance_list, axis=0)

            off_points = paddle.cast(
                self.coord, "float32") + 10 * select_distance.transpose((1, 0))

            off_points = paddle.cast(off_points, "int64")
            off_points = paddle.clip(off_points, 0, distance.shape[-1] - 1)

            selected_mask = (
                gt_instance[self.coord[:, 1], self.coord[:, 0]] !=
                gt_kernel_instance[off_points[:, 1], off_points[:, 0]])
            selected_mask = paddle.cast(
                selected_mask.reshape((1, -1, distance.shape[-1])), "int64")
            selected_training_mask = selected_mask * training_mask

            return selected_training_mask

    def forward(self,
                distances,
                gt_instances,
                gt_kernel_instances,
                training_masks,
                gt_distances,
                reduce=True):

        selected_training_masks = []
        for i in range(distances.shape[0]):
            selected_training_masks.append(
                self.select_single(distances[i, :, :, :], gt_instances[i, :, :],
                                   gt_kernel_instances[i, :, :], training_masks[
                                       i, :, :]))
        selected_training_masks = paddle.cast(
            paddle.concat(selected_training_masks, 0), "float32")

        loss = self.forward_single(distances, gt_distances,
                                   selected_training_masks, self.beta)
        loss = self.loss_weight * loss

        with paddle.no_grad():
            batch_size = distances.shape[0]
            false_num = selected_training_masks.reshape((batch_size, -1))
            false_num = false_num.sum(axis=-1)
            total_num = paddle.cast(
                training_masks.reshape((batch_size, -1)), "float32")
            total_num = total_num.sum(axis=-1)
            iou_text = (total_num - false_num) / (total_num + 1e-6)

        if reduce:
            loss = paddle.mean(loss)

        return loss, iou_text


class CTLoss(nn.Layer):
    def __init__(self):
        super(CTLoss, self).__init__()
        self.kernel_loss = DiceLoss()
        self.loc_loss = SmoothL1Loss(beta=0.1, loss_weight=0.05)

    def forward(self, preds, batch):
        imgs = batch[0]
        out = preds['maps']
        gt_kernels, training_masks, gt_instances, gt_kernel_instances, training_mask_distances, gt_distances = batch[
            1:]

        kernels = out[:, 0, :, :]
        distances = out[:, 1:, :, :]

        # kernel loss
        selected_masks = ohem_batch(kernels, gt_kernels, training_masks)

        loss_kernel = self.kernel_loss(
            kernels, gt_kernels, selected_masks, reduce=False)

        iou_kernel = iou(paddle.cast((kernels > 0), "int64"),
                         gt_kernels,
                         training_masks,
                         reduce=False)
        losses = dict(loss_kernels=loss_kernel, )

        # loc loss
        loss_loc, iou_text = self.loc_loss(
            distances,
            gt_instances,
            gt_kernel_instances,
            training_mask_distances,
            gt_distances,
            reduce=False)
        losses.update(dict(loss_loc=loss_loc, ))

        loss_all = loss_kernel + loss_loc
        losses = {'loss': loss_all}

        return losses
