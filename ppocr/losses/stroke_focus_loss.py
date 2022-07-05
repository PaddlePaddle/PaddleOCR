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

        # print("sr_img:", np.sum(sr_img.detach().cpu().numpy()))
        # print("hr_img:", np.sum(hr_img.detach().cpu().numpy()))
        # select correct
        # hr_correct_list = pred["hr_correct_list"]
        # sr_correct_list = pred["sr_correct_list"]
        word_attention_map_gt = pred["word_attention_map_gt"]
        word_attention_map_pred = pred["word_attention_map_pred"]

        hr_pred = pred["hr_pred"]
        sr_pred = pred["sr_pred"]
        # print("hr_pred:", hr_pred)
        # print("hr_pred, cuda:{}, numpy:{}".format(paddle.sum(hr_pred), np.sum(hr_pred.numpy())))

        # print("word_attention_map_gt:",word_attention_map_gt)
        #print("word_attention_map_pred:",word_attention_map_pred)


        # np.save("gt.npy", word_attention_map_gt.numpy())
        # np.save("pred.npy", word_attention_map_pred.numpy())
        # print("max gt:{}, max pred:{}".format(np.max(word_attention_map_gt.numpy()), np.max(word_attention_map_pred.numpy())))
        # print("min gt:{}, min pred:{}".format(np.min(word_attention_map_gt.numpy()), np.min(word_attention_map_pred.numpy())))
        # print(word_attention_map_gt.shape)

        # cuda = paddle.sum(paddle.abs(word_attention_map_gt- word_attention_map_pred), dtype="float32")
        # numpy = np.sum(paddle.abs(word_attention_map_gt - word_attention_map_pred).numpy())

        # cuda = paddle.sum(paddle.abs(word_attention_map_gt- word_attention_map_pred), dtype="float32")
        # numpy = np.sum(paddle.abs(word_attention_map_gt - word_attention_map_pred).numpy())
        # print("attention map numpy:", numpy)
        # print("attention map cuda:", cuda)
        # print("sum gt pd:{}, np:{}".format(paddle.sum(paddle.abs(word_attention_map_gt)), np.sum(paddle.abs(word_attention_map_gt).numpy())))
        # print("sum gt pd:{}, np:{}".format(paddle.sum(paddle.abs(word_attention_map_pred)), np.sum(paddle.abs(word_attention_map_pred).numpy())))


        attention_loss = paddle.nn.functional.l1_loss(word_attention_map_gt, word_attention_map_pred)


        # print("word_attention_map_gt:", word_attention_map_gt)
        # print("mse_loss:", mse_loss.numpy())
        # print("attention loss:", attention_loss.numpy())
        # exit()
        loss = (mse_loss + attention_loss * 50)*100
        # exit()
        return {"mse_loss": mse_loss, "attention_loss":attention_loss, "loss":loss}