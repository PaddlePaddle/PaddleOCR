# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import cv2
import copy
import numpy as np
import math
import time
import sys

import paddle.fluid as fluid

import tools.infer.utility as utility
from ppocr.utils.utility import initial_logger
logger = initial_logger()
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.data.det.sast_process import SASTProcessTest
from ppocr.data.det.east_process import EASTProcessTest
from ppocr.data.det.db_process import DBProcessTest
from ppocr.postprocess.db_postprocess import DBPostProcess
from ppocr.postprocess.east_postprocess import EASTPostPocess
from ppocr.postprocess.sast_postprocess import SASTPostProcess

import tools.infer.predict_det as predict_det


# from urs/transform
def get_rotated_size(w, h, theta):
    # 以中心为原心为旋转
    half_w = int(w // 2 + 1)
    half_h = int(h // 2 + 1)
    # 对右下顶点进行旋转，最大的宽度和高度
    new_w1 = (math.cos(theta) * half_w - math.sin(theta) * half_h) * 2
    new_h1 = (math.sin(theta) * half_w + math.cos(theta) * half_h) * 2
    # print("image shape : {}, (new_w, new_h) : {},{}".format((w, h), new_w1, new_h1))

    # 对右上顶点进行旋转，最大的宽度和高度
    new_w2 = (math.cos(theta) * half_w - math.sin(theta) * -half_h) * 2
    new_h2 = (math.sin(theta) * half_w + math.cos(theta) * -half_h) * 2
    # print("image shape : {}, (new_w, new_h) : {},{}".format((w, h), new_w2, new_h2))

    # 求出最大的宽度和高度
    new_w = int(max(abs(new_w1), abs(new_w2)))
    new_h = int(max(abs(new_h1), abs(new_h2)))
    # print(new_w, new_h)
    return new_w, new_h


# from urs/transform
def rotate_image(img, theta):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), math.degrees(theta), 1.0)
    new_w, new_h = get_rotated_size(w, h, theta)
    # return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC)
    M[0, 2] += (new_w - w) // 2
    M[1, 2] += (new_h - h) // 2
    return cv2.warpAffine(img, M, (new_w, new_h))


# from urs/transform
# 假设最长的5个文本条的方向就是图像的方向
def get_rotated_radian(text_boxes):
    long_side_list = []
    theta_list = []
    for box in text_boxes:
        d12 = math.sqrt((box[0] - box[2]) ** 2 + (box[1] - box[3]) ** 2)
        d14 = math.sqrt((box[0] - box[6]) ** 2 + (box[1] - box[7]) ** 2)
        if d12 >= d14:
            long_side_list.append(d12)
            theta = (box[3] - box[1]) / (box[2] - box[0] + 0.001)
        else:
            long_side_list.append(d14)
            theta = (box[7] - box[1]) / (box[6] - box[0] + 0.001)
        theta_list.append(math.atan(theta))
    indies = np.argsort(long_side_list)[::-1][:5]
    top_theta_list = [theta_list[i] for i in indies]
    top_theta_indies = np.argsort(top_theta_list)
    return top_theta_list[top_theta_indies[len(top_theta_list)//2]]


def length(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_box_length(box):
    t = length(box[0, 0], box[0, 1], box[1, 0], box[1, 1])
    r = length(box[1, 0], box[1, 1], box[2, 0], box[2, 1])
    b = length(box[2, 0], box[2, 1], box[3, 0], box[3, 1])
    l = length(box[3, 0], box[3, 1], box[0, 0], box[0, 1])
    return t, r, b, l


def check_box_num(boxes, ratio=3.0):
    num = 0
    for box in boxes:
        # print(type(box), box)
        t, r, b, l = get_box_length(box)
        if (t + b) / (l + r) >= ratio or (l + r) / (t + b) >= ratio:
            num += 1
    return num


# 假设获取的文本框越长，检测的效果越好。
def check_box_score(boxes, ratio=2.0):
    score = 0
    for box in boxes:
        # print(type(box), box)
        t, r, b, l = get_box_length(box)
        if (t + b) / (l + r) >= ratio:
            score += (t + b) / (l + r)
        elif (l + r) / (t + b) >= ratio:
            score += (l + r) / (t + b)
    return score


def calibrate_img(text_detector, img, image_file, out_dir):
    total_time = 0
    # 添加4个旋转后图像
    imgs = [img]
    for theta in [math.pi / 8, math.pi / 4, math.pi * 3 / 8, math.pi / 2]:
        imgs.append(rotate_image(img, theta))

    # 对5个图像分别检测文本框，并以长文本框数量第一次计算最佳旋转角度
    dt_boxes_list = []
    max_valid_score = 0
    score_list = []
    for i in range(len(imgs)):
        dt_boxes, elapse = text_detector(imgs[i])
        total_time += elapse
        dt_boxes_list.append(dt_boxes)
        # elapse_list.append(elapse)
        score = check_box_score(dt_boxes, 3)
        if score > max_valid_score:
            max_valid_score = score
        print("Predict time of:{}-{}， boxes:{}-{}，time:{}".format(image_file, i, len(dt_boxes), int(score), elapse))
        score_list.append(score)
        '''
        src_im = utility.draw_text_det_res2(dt_boxes_list[i], imgs[i])
        new_filename = "det_res_{}-{}-{}.png".format(img_name_pure, i, int(score_list[i]))
        cv2.imwrite(os.path.join(draw_img_save, new_filename), src_im)
        '''

    index = score_list.index(max_valid_score)
    if len(dt_boxes_list[index]) < 1:
        return img, None, 0
    theta = get_rotated_radian(dt_boxes_list[index].reshape((-1, 8)).tolist())
    print("get_rotated_radian : {}".format(theta))
    theta += index * math.pi / 8
    rotate_angle = math.degrees(theta)
    print("判断图像的旋转方向 {} {}".format(theta, rotate_angle))

    h, w = img.shape[:2]
    # h, w = imgs[index].shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_angle, 1.0)
    print("M : {}".format(M))
    new_w, new_h = get_rotated_size(w, h, theta)
    M[0, 2] += (new_w - w) // 2
    M[1, 2] += (new_h - h) // 2
    print("image shape : {}, (new_w, new_h) : {},{}".format((h, w), new_w, new_h))
    # image = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC)
    image = cv2.warpAffine(img, M, (new_w, new_h))
    # image = cv2.warpAffine(imgs[index], M, (new_w, new_h))
    dt_boxes, elapse = text_detector(image)
    print("Predict time of:{}， boxes:{}，time:{}".format(image_file, len(dt_boxes), elapse))
    return image, dt_boxes, rotate_angle


if __name__ == "__main__":
    args = utility.parse_args()
    image_file_list = get_image_file_list(args.image_dir)
    text_detector = predict_det.TextDetector(args)
    count = 0
    total_time = 0
    draw_img_save = "./inference_results"
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        count += 1
        image, dt_boxes, rotate_angle = calibrate_img(text_detector, img, image_file, draw_img_save)
        if dt_boxes is not None and len(dt_boxes) > 0:
            image = utility.draw_text_det_res2(dt_boxes, image)
        img_name_pure = os.path.basename(image_file)
        new_filename = "res_{}-{}.png".format(img_name_pure, int(rotate_angle))
        cv2.imwrite(os.path.join(draw_img_save, new_filename), image)
        # break
    if count > 1:
        print("Avg Time:", total_time / (count - 1))
