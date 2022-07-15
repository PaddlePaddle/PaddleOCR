import cv2
import sys
import time
import string
import random
import numpy as np
import paddle.nn as nn
import paddle

class StrokeFocusLoss(nn.Layer):
    def __init__(self,  **kwargs):
        super(StrokeFocusLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.english_stroke_alphabet = '0123456789'
        self.english_stroke_dict = {}
        for index in range(len(self.english_stroke_alphabet)):
            self.english_stroke_dict[self.english_stroke_alphabet[index]] = index

        stroke_decompose_lines = open('./train_data/mydata/english_decomposition.txt',
                     'r').readlines()
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

        attention_loss = paddle.nn.functional.l1_loss(word_attention_map_gt, word_attention_map_pred)

        loss = (mse_loss + attention_loss * 50)*100

        return {"mse_loss": mse_loss, "attention_loss":attention_loss, "loss":loss}