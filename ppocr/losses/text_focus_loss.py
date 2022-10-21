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
https://github.com/FudanVI/FudanOCR/blob/main/scene-text-telescope/loss/text_focus_loss.py
"""

import paddle.nn as nn
import paddle
import numpy as np
import pickle as pkl

standard_alphebet = '-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
standard_dict = {}
for index in range(len(standard_alphebet)):
    standard_dict[standard_alphebet[index]] = index


def load_confuse_matrix(confuse_dict_path):
    f = open(confuse_dict_path, 'rb')
    data = pkl.load(f)
    f.close()
    number = data[:10]
    upper = data[10:36]
    lower = data[36:]
    end = np.ones((1, 62))
    pad = np.ones((63, 1))
    rearrange_data = np.concatenate((end, number, lower, upper), axis=0)
    rearrange_data = np.concatenate((pad, rearrange_data), axis=1)
    rearrange_data = 1 / rearrange_data
    rearrange_data[rearrange_data == np.inf] = 1
    rearrange_data = paddle.to_tensor(rearrange_data)

    lower_alpha = 'abcdefghijklmnopqrstuvwxyz'
    # upper_alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in range(63):
        for j in range(63):
            if i != j and standard_alphebet[j] in lower_alpha:
                rearrange_data[i][j] = max(rearrange_data[i][j], rearrange_data[i][j + 26])
    rearrange_data = rearrange_data[:37, :37]

    return rearrange_data


def weight_cross_entropy(pred, gt, weight_table):
    batch = gt.shape[0]
    weight = weight_table[gt]
    pred_exp = paddle.exp(pred)
    pred_exp_weight = weight * pred_exp
    loss = 0
    for i in range(len(gt)):
        loss -= paddle.log(pred_exp_weight[i][gt[i]] / paddle.sum(pred_exp_weight, 1)[i])
    return loss / batch


class TelescopeLoss(nn.Layer):
    def __init__(self, confuse_dict_path):
        super(TelescopeLoss, self).__init__()
        self.weight_table = load_confuse_matrix(confuse_dict_path)
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, data):
        sr_img = pred["sr_img"]
        hr_img = pred["hr_img"]
        sr_pred = pred["sr_pred"]
        text_gt = pred["text_gt"]

        word_attention_map_gt = pred["word_attention_map_gt"]
        word_attention_map_pred = pred["word_attention_map_pred"]
        mse_loss = self.mse_loss(sr_img, hr_img)
        attention_loss = self.l1_loss(word_attention_map_gt, word_attention_map_pred)
        recognition_loss = weight_cross_entropy(sr_pred, text_gt, self.weight_table)
        loss = mse_loss + attention_loss * 10 + recognition_loss * 0.0005
        return {
            "mse_loss": mse_loss,
            "attention_loss": attention_loss,
            "loss": loss
        }
