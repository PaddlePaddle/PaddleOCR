# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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


def compute_mean_covariance(img):
    batch_size = img.shape[0]
    channel_num = img.shape[1]
    height = img.shape[2]
    width = img.shape[3]
    num_pixels = height * width

    # batch_size * channel_num * 1 * 1
    mu = img.mean(2, keepdim=True).mean(3, keepdim=True)

    # batch_size * channel_num * num_pixels
    img_hat = img - mu.expand_as(img)
    img_hat = img_hat.reshape([batch_size, channel_num, num_pixels])
    # batch_size * num_pixels * channel_num
    img_hat_transpose = img_hat.transpose([0, 2, 1])
    # batch_size * channel_num * channel_num
    covariance = paddle.bmm(img_hat, img_hat_transpose)
    covariance = covariance / num_pixels

    return mu, covariance


def dice_coefficient(y_true_cls, y_pred_cls, training_mask):
    eps = 1e-5
    intersection = paddle.sum(y_true_cls * y_pred_cls * training_mask)
    union = paddle.sum(y_true_cls * training_mask) + paddle.sum(
        y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    return loss
