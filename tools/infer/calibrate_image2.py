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
from calibrate_image import get_rotated_size, check_box_score
# from calibrate_image import rotate_image, rotate_first


def rotate_image2(img, theta):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), math.degrees(theta), 1.0)
    new_w, new_h = get_rotated_size(w, h, theta)
    # return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    center = (w / 2, h / 2)
    lt = np.array([new_w/2, new_h/2]) + M[0:2, 0:2].dot(np.array((0, 0)) - np.array(center))
    rt = np.array([new_w/2, new_h/2]) + M[0:2, 0:2].dot(np.array((w - 1, 0)) - np.array(center))
    rb = np.array([new_w/2, new_h/2]) + M[0:2, 0:2].dot(np.array((w - 1, h - 1)) - np.array(center))
    lb = np.array([new_w/2, new_h/2]) + M[0:2, 0:2].dot(np.array((0, h - 1)) - np.array(center))
    # img_box = [round(x) for x in np.array([lt, rt, rb, lb]).reshape((8,)).tolist()]
    img_box = np.array([lt, rt, rb, lb]).astype(int)
    # print(theta, img_box)
    return cv2.warpAffine(img, M, (new_w, new_h)), img_box


def rotate_first2(text_detector, img):
    # 添加4个旋转后图像
    imgs = [img]
    h, w = img.shape[:2]
    img_box_list = [np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])]
    for theta in [math.pi / 8, math.pi / 4, math.pi * 3 / 8, math.pi / 2]:
        new_img, new_box = rotate_image2(img, theta)
        imgs.append(new_img)
        img_box_list.append(new_box)
    print(img_box_list)

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
    return imgs[index], dt_boxes_list[index], index * math.pi / 8, max_valid_score, img_box_list[index]


def calibrate_img2(text_detector, img, image_file, out_dir, debug=False):
    # 1 通过多次旋转图片，初步找到旋转角度
    img1, boxes1, theta1, score1, img_box1 = rotate_first2(text_detector, img)
    if debug:
        print('img_box1 : ', img_box1)
        rotate_angle1 = math.degrees(theta1)
        logger.debug("第一次旋转：文本框数量1：{}，旋转角度1：{}, score1: {}".format(len(boxes1), rotate_angle1, score1))
        new_filename = "{}.1-rotate.{}.{}.jpg".format(os.path.basename(image_file), int(rotate_angle1), int(score1))
        cv2.imwrite(os.path.join(out_dir, new_filename), utility.draw_text_det_res2(boxes1, img1))

    img_mask = np.zeros(img1.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(img_mask, img_box1, (255, ))

    img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img_gaus = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # Implementation has been removed due original code license issues in function 'LineSegmentDetectorImpl'
    # LSD = cv2.createLineSegmentDetector(0, _scale=1)
    # lines = LSD.detect(img_gray)
    fld = cv2.ximgproc.createFastLineDetector()
    lines = fld.detect(img_gaus)
    drawn_img = fld.drawSegments(img1, lines)
    print(lines)
    # image, dt_boxes, rotate_angle = calibrate_img(text_detector, img, image_file, draw_img_save, debug=True)
    # if dt_boxes is not None and len(dt_boxes) > 0:
    #     image = utility.draw_text_det_res2(dt_boxes, image)
    if debug:
        new_filename = "{}-mask.png".format(os.path.basename(image_file))
        cv2.imwrite(os.path.join(draw_img_save, new_filename), img_mask)

        new_filename = "{}-{}.png".format(os.path.basename(image_file), len(lines))
        cv2.imwrite(os.path.join(draw_img_save, new_filename), drawn_img)


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
        h, w = img.shape[:2]
        print(image_file, img.shape)
        if w > 2000 and w >= h:
            img = cv2.resize(img, (2000, round(2000 * h / w)))
        elif h > 2000 and h >= w:
            img = cv2.resize(img, (round(2000 * w / h), 2000))
        print(img.shape)
        calibrate_img2(text_detector, img, image_file, draw_img_save, debug=True)
        break
    if count > 1:
        print("Avg Time:", total_time / (count - 1))