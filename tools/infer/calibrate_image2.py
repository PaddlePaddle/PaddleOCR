import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
from line_seg import LineSeg
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
from calibrate_image import get_rotated_size, check_box_score, get_box_length, length
# from calibrate_image import rotate_image, rotate_first
_debug = True
_out_dir = './inference_results'
_img_base_name = ''


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
    '''
    return :
        旋转后图像
        文本框列表
        旋转角度
        评价分数
        有效区域
    '''
    # 添加4个旋转后图像
    imgs = [img]
    h, w = img.shape[:2]
    img_box_list = [np.array([[0, 0], [w, 0], [w, h], [0, h]])]
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
        return img, None, 0, 0, img_box_list[0]
    return imgs[index], dt_boxes_list[index], index * math.pi / 8, max_valid_score, img_box_list[index]


def get_img_mask(img, box, boxes):
    '''
    有效区域为box
    文本框区域为无效区域
    '''
    img_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    center, size, angle = cv2.minAreaRect(box)
    size = (size[0] - 10, size[1] - 10)
    print(center, size, angle)
    img_box = cv2.boxPoints((center, size, angle))
    print(img_box, type(img_box))
    cv2.fillConvexPoly(img_mask, img_box.astype(int), (255,))
    for box in boxes:
        cv2.fillConvexPoly(img_mask, box.astype(np.int), (0,))
    if _debug:
        new_filename = "{}.11-mask.png".format(_img_base_name)
        cv2.imwrite(os.path.join(_out_dir, new_filename), img_mask)
    return img_mask


def get_lines_by_box(boxes):
    box_height_list = []
    lines_valid_text_box = []
    for box in boxes:
        # print(box, type(box))
        t, r, b, l = get_box_length(box)
        box_width = (t + b) / 2
        box_height = (l + r) / 2
        # box_height_list.append(min(box_width, box_height))
        # 大于文本框宽高比大于2倍的才算作有效直线
        if box_width > box_height * 2:
            lines_valid_text_box.append([[box[0, 0], box[0, 1], box[1, 0], box[1, 1]]])
            lines_valid_text_box.append([[box[3, 0], box[3, 1], box[2, 0], box[2, 1]]])
            box_height_list.append(box_height)
        elif box_height > box_width * 2:
            lines_valid_text_box.append([[box[0, 0], box[0, 1], box[3, 0], box[3, 1]]])
            lines_valid_text_box.append([[box[1, 0], box[1, 1], box[2, 0], box[2, 1]]])
            box_height_list.append(box_width)
    # img_mask = cv2.erode(img_mask, np.ones((9, 9), np.uint8), iterations=1)
    box_height_avg = np.median(box_height_list)
    print('box_height_avg:', box_height_avg)
    print('lines-valid-textbox:', lines_valid_text_box, type(lines_valid_text_box))
    return lines_valid_text_box, box_height_avg


def get_lines_by_ld(img, img_mask, box_height_avg):
    # 找出非文本有效区域内检测到的线段
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaus = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # Implementation has been removed due original code license issues in function 'LineSegmentDetectorImpl'
    # LSD = cv2.createLineSegmentDetector(0, _scale=1)
    # lines = LSD.detect(img_gray)
    fld = cv2.ximgproc.createFastLineDetector()
    lines = fld.detect(img_gaus)
    lines_valid = []
    for i, line in enumerate(lines):
        line = line[0].astype(np.int)
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        # 大于文本高度1.5倍的才算有效的直线
        if img_mask[y1, x1] > 0 and img_mask[y2, x2] > 0 and length(x1, y1, x2, y2) > box_height_avg * 1.5:
            lines_valid.append(lines[i])
    lines_valid = np.array(lines_valid)
    print('lines(fld):')
    print(lines, type(lines), lines.shape)
    print('lines(fld-valid):')
    print(lines_valid, type(lines_valid), lines_valid.shape)
    if _debug:
        '''
        drawn_img = img.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(drawn_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        new_filename = "{}.22-lines_fld.png".format(_img_base_name)
        cv2.imwrite(os.path.join(_out_dir, new_filename), drawn_img)
        '''
        drawn_img = img.copy()
        for line in lines_valid:
            x1, y1, x2, y2 = line[0]
            cv2.line(drawn_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        new_filename = "{}.22-lines_fld_valid.png".format(_img_base_name)
        cv2.imwrite(os.path.join(_out_dir, new_filename), drawn_img)
    return lines_valid


def lines_split_by_direction(lines, lines_valid_hor, lines_valid_ver):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) > abs(y1 - y2) * 2:
            lines_valid_hor.append([[x1, y1, x2, y2]])
        elif abs(y1 - y2) > abs(x1 - x2) * 2:
            lines_valid_ver.append([[x1, y1, x2, y2]])
    return lines_valid_hor, lines_valid_ver


def line_infer(line, x):
    vx = line[0][0]
    vy = line[1][0]
    x0 = line[2][0]
    y0 = line[3][0]
    y = vy * (x - x0) / vx + y0
    return round(y)


def calibrate_img2(text_detector, img):
    """
    """
    # 1 通过多次旋转图片，初步找到旋转角度
    img1, boxes1, theta1, score1, img_box1 = rotate_first2(text_detector, img)
    if _debug:
        print('img_box1 : ', img_box1, type(img_box1))
        rotate_angle1 = math.degrees(theta1)
        logger.debug("第一次旋转：文本框数量1：{}，旋转角度1：{}, score1: {}".format(len(boxes1), rotate_angle1, score1))
        new_filename = "{}.10-rotate.{}.{}.jpg".format(_img_base_name, int(rotate_angle1), int(score1))
        cv2.imwrite(os.path.join(_out_dir, new_filename), utility.draw_text_det_res2(boxes1, img1))
    if len(boxes1) < 4:
        return img1, boxes1, theta1, score1, img_box1

    # 2 找出非文本有效区域
    img_mask = get_img_mask(img1, img_box1, boxes1)
    # 找出文本框的有效直线
    lines_valid_text_box, box_height_avg = get_lines_by_box(boxes1)
    if _debug:
        drawn_img = img1.copy()
        for line in lines_valid_text_box:
            print(line, type(line))
            cv2.line(drawn_img, tuple(line[0][:2]), tuple(line[0][2:4]), (255, 0, 0), 2)
        new_filename = "{}.21-lines_text-box.png".format(_img_base_name)
        cv2.imwrite(os.path.join(_out_dir, new_filename), drawn_img)
    # 找出fld的有效直线
    lines_valid_fld = get_lines_by_ld(img1, img_mask, box_height_avg)
    # 有效的横线、有效的竖线
    lines_valid_hor, lines_valid_ver = [], []
    lines_valid_hor, lines_valid_ver = lines_split_by_direction(lines_valid_fld, lines_valid_hor, lines_valid_ver)
    lines_valid_hor, lines_valid_ver = lines_split_by_direction(lines_valid_text_box, lines_valid_hor, lines_valid_ver)
    if _debug:
        print("lines_valid_hor : ", lines_valid_hor)
        print("lines_valid_ver : ", lines_valid_ver)
        drawn_img = img1.copy()
        for line in lines_valid_hor:
            cv2.line(drawn_img, tuple(line[0][:2]), tuple(line[0][2:4]), (255, 0, 0), 2)
        for line in lines_valid_ver:
            cv2.line(drawn_img, tuple(line[0][:2]), tuple(line[0][2:4]), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(_out_dir, _img_base_name + '.23.lines_hor_ver.png'), drawn_img)

    h, w = img1.shape[:2]
    lineseg_center_cross_x_min = w - 1
    lineseg_center_cross_x_max = 0
    lineseg_center_cross_y_min = h - 1
    lineseg_center_cross_y_max = 0
    slope_hor_list, slope_ver_list = [], []

    for line in lines_valid_hor:
        x1, y1, x2, y2 = line[0]
        # 横线与中竖线交点
        line_seg = LineSeg(x1, y1, x2, y2)
        cross_pt_hor = line_seg.get_cross_point(LineSeg(w / 2, 0, w / 2, h - 1))
        # theta_hor.append([cross_pt_hor[1], 100 * math.degrees(math.atan((y1 - y2) / (x1 - x2)))])
        slope_hor_list.append([cross_pt_hor[1], 10000 * (y2 - y1) / (x2 - x1)])
        if cross_pt_hor[1] > lineseg_center_cross_y_max:
            lineseg_center_cross_y_max = cross_pt_hor[1]
        if cross_pt_hor[1] < lineseg_center_cross_y_min:
            lineseg_center_cross_y_min = cross_pt_hor[1]
    for line in lines_valid_ver:
        x1, y1, x2, y2 = line[0]
        # 竖线与中横线交点
        cross_pt_ver = LineSeg(0, h / 2, w - 1, h / 2).get_cross_point(LineSeg(x1, y1, x2, y2))
        # theta_ver.append([cross_pt_ver[0], 100 * math.degrees(math.atan((x1 - x2) / (y1 - y2)))])
        slope_ver_list.append([cross_pt_ver[0], 10000 * (x2 - x1) / (y2 - y1)])
        if cross_pt_ver[0] > lineseg_center_cross_x_max:
            lineseg_center_cross_x_max = cross_pt_ver[0]
        if cross_pt_ver[0] < lineseg_center_cross_x_min:
            lineseg_center_cross_x_min = cross_pt_ver[0]
    print("斜率统计结果：")
    print(slope_hor_list)
    print(slope_ver_list)

    line_top, line_bottom, line_left, line_right = None, None, None, None
    line_top_left, line_top_right = None, None
    line_bot_left, line_bot_right = None, None
    line_left_top, line_left_bot = None, None
    line_right_top, line_right_bot = None, None
    if len(slope_hor_list) > 2:
        # theta_hor_line = cv2.fitLine(np.array(theta_hor), cv2.DIST_L1, 0, 0.01, 0.01)
        slope_hor_line = cv2.fitLine(np.array(slope_hor_list), cv2.DIST_L2, 0, 0.01, 0.01)
        # print('theta_hor_line:', theta_hor_line)
        print('slope_hor_line : ', slope_hor_line)
        line_top_pt = (w / 2, lineseg_center_cross_y_min)
        line_bottom_pt = (w / 2, lineseg_center_cross_y_max)
        print(line_top_pt, line_bottom_pt)
        # print(theta_top, theta_bot)
        slope_top = line_infer(slope_hor_line, line_top_pt[1]) / 10000
        slope_bot = line_infer(slope_hor_line, line_bottom_pt[1]) / 10000
        line_top = (-1.0 * slope_top, 1.0, w / 2 * slope_top - line_top_pt[1])
        line_bottom = (-1.0 * slope_bot, 1.0, w / 2 * slope_bot - line_bottom_pt[1])
        line_top_left = LineSeg(0, 0, 0, h - 1).get_cross_point_by_param(*line_top)
        line_top_right = LineSeg(w-1, 0, w-1, h-1).get_cross_point_by_param(*line_top)
        line_bot_left = LineSeg(0, 0, 0, h - 1).get_cross_point_by_param(*line_bottom)
        line_bot_right = LineSeg(w-1, 0, w-1, h-1).get_cross_point_by_param(*line_bottom)
    if len(slope_ver_list) > 2:
        # theta_ver_line = cv2.fitLine(np.array(theta_ver), cv2.DIST_L1, 0, 0.01, 0.01)
        slope_ver_line = cv2.fitLine(np.array(slope_ver_list), cv2.DIST_L2, 0, 0.01, 0.01)
        print('slope_ver_line : ', slope_ver_line)
        # print('theta_ver_line:', theta_ver_line)
        line_left_pt = (lineseg_center_cross_x_min, h / 2)
        line_right_pt = (lineseg_center_cross_x_max, h / 2)
        print(line_left_pt, line_right_pt)
        slope_left = line_infer(slope_ver_line, line_left_pt[0]) / 10000
        slope_right = line_infer(slope_ver_line, line_right_pt[0]) / 10000
        print(slope_left, slope_right)

        line_left = (1.0, -1.0 * slope_left, h / 2 * slope_left - line_left_pt[0])
        line_right = (1.0, -1.0 * slope_right, h / 2 * slope_right - line_right_pt[0])
        line_left_top = LineSeg(0, 0, w-1, 0).get_cross_point_by_param(*line_left)
        line_left_bot = LineSeg(0, h-1, w - 1, h - 1).get_cross_point_by_param(*line_left)
        line_right_top = LineSeg(0, 0, w-1, 0).get_cross_point_by_param(*line_right)
        line_right_bot = LineSeg(0, h-1, w-1, h - 1).get_cross_point_by_param(*line_right)
    if _debug:
        drawn_img = img1.copy()
        print('slope_top, slope_bot : ')
        print(slope_top, slope_bot)
        if line_top_left is not None and line_top_right is not None:
            cv2.line(drawn_img, line_top_left, line_top_right, (255, 0, 0), 2)
        if line_bot_left is not None and line_bot_right is not None:
            cv2.line(drawn_img, line_bot_left, line_bot_right, (0, 255, 0), 2)
        if line_left_top is not None and line_left_bot is not None:
            cv2.line(drawn_img, line_left_top, line_left_bot, (0, 0, 255), 2)
        if line_right_top is not None and line_right_bot is not None:
            cv2.line(drawn_img, line_right_top, line_right_bot, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(_out_dir, "{}.31.top_bot.png".format(_img_base_name)),
                    drawn_img)


if __name__ == "__main__":
    args = utility.parse_args()
    image_file_list = get_image_file_list(args.image_dir)
    text_detector = predict_det.TextDetector(args)
    count = 0
    total_time = 0
    if _debug and not os.path.exists(_out_dir):
        os.makedirs(_out_dir)
    for image_file in image_file_list:
        if _debug:
            _img_base_name = os.path.basename(image_file)
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
        calibrate_img2(text_detector, img)
        # break
    if count > 1:
        print("Avg Time:", total_time / (count - 1))