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
https://github.com/FudanVI/FudanOCR/blob/main/text-gestalt/loss/stroke_focus_loss.py
"""
import cv2
import sys
import time
import string
import random
import numpy as np
import paddle.nn as nn
import paddle


class StrokeFocusLoss(nn.Layer):
    def __init__(self, character_dict_path=None, **kwargs):
        super(StrokeFocusLoss, self).__init__(character_dict_path)
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.english_stroke_alphabet = "0123456789"
        self.english_stroke_dict = {}
        for index in range(len(self.english_stroke_alphabet)):
            self.english_stroke_dict[self.english_stroke_alphabet[index]] = index

        stroke_decompose_lines = open(character_dict_path, "r").readlines()
        self.dic = {}
        for line in stroke_decompose_lines:
            line = line.strip()
            character, sequence = line.split()
            self.dic[character] = sequence

    def forward(self, pred, data):
        sr_img = pred["sr_img"]
        hr_img = pred["hr_img"]

        mse_loss = self.mse_loss(sr_img, hr_img)
        word_attention_map_gt = pred["word_attention_map_gt"]
        word_attention_map_pred = pred["word_attention_map_pred"]

        hr_pred = pred["hr_pred"]
        sr_pred = pred["sr_pred"]

        attention_loss = paddle.nn.functional.l1_loss(
            word_attention_map_gt, word_attention_map_pred
        )

        loss = (mse_loss + attention_loss * 50) * 100

        return {"mse_loss": mse_loss, "attention_loss": attention_loss, "loss": loss}
