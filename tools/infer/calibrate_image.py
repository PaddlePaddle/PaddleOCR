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
# from ppocr.utils.utility import initial_logger
# logger = initial_logger()
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.data.det.sast_process import SASTProcessTest
from ppocr.data.det.east_process import EASTProcessTest
from ppocr.data.det.db_process import DBProcessTest
from ppocr.postprocess.db_postprocess import DBPostProcess
from ppocr.postprocess.east_postprocess import EASTPostPocess
from ppocr.postprocess.sast_postprocess import SASTPostProcess

import tools.infer.predict_det as predict_det


import logging
def initial_logger():
    FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
    logging.basicConfig(level=logging.NOTSET, format=FORMAT)
    logger = logging.getLogger(__name__)
    return logger
logger = initial_logger()


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


def rotate_image(img, theta):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), math.degrees(theta), 1.0)
    new_w, new_h = get_rotated_size(w, h, theta)
    # return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC)
    M[0, 2] += (new_w - w) // 2
    M[1, 2] += (new_h - h) // 2
    return cv2.warpAffine(img, M, (new_w, new_h))


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


def rotate_first(text_detector, img):
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
        dt_boxes_list.append(dt_boxes)
        score = check_box_score(dt_boxes, 3)
        if score > max_valid_score:
            max_valid_score = score
        logger.debug("Predict time of:{}-{}， boxes:{}-{}".format(i, elapse, len(dt_boxes), int(score)))
        score_list.append(score)
    index = score_list.index(max_valid_score)
    if len(dt_boxes_list[index]) < 1:
        return img, None, 0
    return imgs[index], dt_boxes_list[index], index * math.pi / 8, max_valid_score


def order_points_clockwise(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost, rightMost = xSorted[:2, :], xSorted[2:, :]

    (tl, bl) = leftMost[np.argsort(leftMost[:, 1]), :]
    (tr, br) = rightMost[np.argsort(rightMost[:, 1]), :]

    return np.array([tl, tr, br, bl], dtype="float32")


import random
import line_seg
def rectify_img(img, boxes, debug=False):
    # 画文本框到图中。
    img_lines = img.copy()
    if debug:
        for box in boxes:
            print(type(box), box)
            color = [random.randint(0, 255) for _ in range(3)]
            cv2.fillConvexPoly(img_lines, box.astype(np.int), color)
        img_lines = cv2.addWeighted(img, 0.5, img_lines, 0.5, 0)

    # 找到上边线、下边线
    h, w = img.shape[:2]
    border_right = line_seg.LineSeg(w - 1, 0, w - 1, h - 1)
    border_left = line_seg.LineSeg(0, 0, 0, h - 1)
    middle_ver = line_seg.LineSeg(w // 2, 0, w // 2, h - 1)
    middle_min_y, middle_max_y = h - 1, 0
    line_hor_top, line_hor_bottom = None, None  # 横向上边线、横向下边线
    min_top_begin, min_top_end = None, None
    max_bottom_begin, max_bottom_end = None, None
    for box in boxes:
        box = box.reshape((8,)).tolist()
        top = line_seg.LineSeg(box[0], box[1], box[2], box[3])
        bottom = line_seg.LineSeg(box[6], box[7], box[4], box[5])
        top_begin = top.get_cross_point(border_left)
        top_end = top.get_cross_point(border_right)
        bottom_begin = bottom.get_cross_point(border_left)
        bottom_end = bottom.get_cross_point(border_right)
        middle_ver_begin = top.get_cross_point(middle_ver)
        middle_ver_end = bottom.get_cross_point(middle_ver)
        if middle_ver_begin[1] < middle_min_y:
            middle_min_y = middle_ver_begin[1]
            line_hor_top = top
            min_top_begin = top_begin
            min_top_end = top_end
        if middle_ver_end[1] > middle_max_y:
            middle_max_y = middle_ver_end[1]
            line_hor_bottom = bottom
            max_bottom_begin = bottom_begin
            max_bottom_end = bottom_end
        if debug:
            # cv2.line(img_lines, left_begin, left_end, (255, 0, 0), 2)
            cv2.line(img_lines, top_begin, top_end, (0, 255, 0), 2)
            # cv2.line(img_lines, right_begin, right_end, (0, 0, 255), 2)
            cv2.line(img_lines, bottom_begin, bottom_end, (0, 255, 255), 2)

    if debug:
        cv2.line(img_lines, min_top_begin, min_top_end, (0, 255, 0), 4)
        cv2.line(img_lines, max_bottom_begin, max_bottom_end, (0, 255, 255), 4)
    line_hor_middle_begin = ((min_top_begin[0] + max_bottom_begin[0])//2,
                            (min_top_begin[1] + max_bottom_begin[1])//2)
    line_hor_middle_end = ((min_top_end[0] + max_bottom_end[0])//2,
                           (min_top_end[1] + max_bottom_end[1])//2)
    # 横向中间线
    line_hor_middle = line_seg.LineSeg(line_hor_middle_begin[0], line_hor_middle_begin[1],
                                       line_hor_middle_end[0], line_hor_middle_end[1])
    if debug:
        cv2.line(img_lines, line_hor_middle_begin, line_hor_middle_end, (255, 0, 0), 4)
        cv2.line(img_lines, (h//2, middle_min_y), (h//2, middle_max_y), (255, 0, 0), 4)

    min_x, max_x = w-1, 0
    left_middle_point, right_middle_point = None, None
    for box in boxes:
        box = box.reshape((8,)).tolist()
        left = line_seg.LineSeg(box[0], box[1], box[6], box[7])
        right = line_seg.LineSeg(box[2], box[3], box[4], box[5])
        left_middle = left.get_cross_point(line_hor_middle)
        right_middle = right.get_cross_point(line_hor_middle)
        if left_middle[0] < min_x:
            min_x = left_middle[0]
            left_middle_point = left_middle
        if right_middle[0] > max_x:
            max_x = right_middle[0]
            right_middle_point = right_middle
    left_a, left_b, left_c = line_hor_middle.get_line_vertical(left_middle_point)
    right_a, right_b, right_c = line_hor_middle.get_line_vertical(right_middle_point)
    lt = line_seg.get_cross_point(left_a, left_b, left_c, line_hor_top.A, line_hor_top.B, line_hor_top.C)
    lb = line_seg.get_cross_point(left_a, left_b, left_c, line_hor_bottom.A, line_hor_bottom.B, line_hor_bottom.C)
    rt = line_seg.get_cross_point(right_a, right_b, right_c, line_hor_top.A, line_hor_top.B, line_hor_top.C)
    rb = line_seg.get_cross_point(right_a, right_b, right_c, line_hor_bottom.A, line_hor_bottom.B, line_hor_bottom.C)
    if debug:
        cv2.line(img_lines, lt, lb, (255, 0, 0), 4)
        cv2.line(img_lines, rt, rb, (0, 0, 255), 4)
        cv2.imwrite("tmp.png", img_lines)
    return np.array([lt, rt, rb, lb])
    # pts1 = np.array([lt, rt, rb, lb])
    # center, size, angle = cv2.minAreaRect(pts1)
    # return im


def calibrate_img(text_detector, img, image_file, out_dir, debug=False):
    # 1 通过多次旋转图片，初步找到旋转角度
    img1, boxes1, theta1, score1 = rotate_first(text_detector, img)
    if debug:
        rotate_angle1 = math.degrees(theta1)
        logger.debug("第一次旋转：文本框数量1：{}，旋转角度1：{}, score1: {}".format(len(boxes1), rotate_angle1, score1))
        img_name_pure = os.path.basename(image_file)
        new_filename = "{}.1-rotate.{}.{}.png".format(img_name_pure, int(rotate_angle1), int(score1))
        cv2.imwrite(os.path.join(out_dir, new_filename), utility.draw_text_det_res2(boxes1, img1))

    # 2 计算文本框旋转角度
    theta_delta = get_rotated_radian(boxes1.reshape((-1, 8)).tolist())
    theta2 = theta1 + theta_delta
    if debug:
        rotate_angle2 = math.degrees(theta2)
        logger.debug("判断图像的旋转方向 {} {} {} {}".format(theta1, theta_delta, theta2, rotate_angle2))

    img2 = rotate_image(img, theta2)
    boxes2, elapse2 = text_detector(img2)
    score2 = check_box_score(boxes2)
    if debug:
        logger.debug("第二次旋转:boxes2:{}, 旋转角度2：{}， score2:{}".format(len(boxes2), rotate_angle2, score2))
        new_filename = "{}.2-rotate.{}.{}.png".format(img_name_pure, int(rotate_angle2), int(score2))
        cv2.imwrite(os.path.join(draw_img_save, new_filename), utility.draw_text_det_res2(boxes2, img2))

    # 3 找到文本区域的四边形顶点
    # if score1 > score2:
    #     pts1 = rectify_img(img1, boxes1)
    #     img3 = img1.copy()
    # else:
    pts1 = rectify_img(img2, boxes2)
    img3 = img2.copy()

    size = (round(line_seg.line_length(pts1[0, 0], pts1[0, 1], pts1[1, 0], pts1[1, 1])),
            round(line_seg.line_length(pts1[0, 0], pts1[0, 1], pts1[1, 0], pts1[1, 1])))
    pts1 = pts1.astype(np.float32)
    pad1, pad2 = round(size[0] / 10), round(size[1] / 10)
    size = (size[0] + 2 * pad1, size[1] + 2 * pad2)
    pts2 = np.array([[pad1, pad2],
                     [size[0] - pad1 - 1, pad2],
                     [size[0] - pad1 - 1, size[1] - pad2 - 1],
                     [pad1, size[1] - pad2 - 1]]).astype(np.float32)
    print(pts1, pts1.shape, pts1.dtype)
    print(pts2, pts2.shape, pts2.dtype)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img4 = cv2.warpPerspective(img3, M, size)
    if debug:
        img3_show = img3.copy()
        cv2.polylines(img3_show, [pts1.astype(np.int)], True, color=(255, 0, 0), thickness=4)
        cv2.imwrite(os.path.join(draw_img_save, "{}.3-rectify.png".format(img_name_pure)), img3_show)
        cv2.imwrite(os.path.join(draw_img_save, "{}.4-rectify.png".format(img_name_pure)), img4)
    return img2, boxes2, theta2


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
        image, dt_boxes, rotate_angle = calibrate_img(text_detector, img, image_file, draw_img_save, debug=True)
        # if dt_boxes is not None and len(dt_boxes) > 0:
        #     image = utility.draw_text_det_res2(dt_boxes, image)
        # img_name_pure = os.path.basename(image_file)
        # new_filename = "res_{}-{}.png".format(img_name_pure, int(rotate_angle))
        # cv2.imwrite(os.path.join(draw_img_save, new_filename), image)
        # break
    if count > 1:
        print("Avg Time:", total_time / (count - 1))
