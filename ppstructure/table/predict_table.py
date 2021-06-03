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
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import cv2
import copy
import numpy as np
import time
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import ppstructure.table.predict_structure as predict_strture
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.utils.logging import get_logger
from ppstructure.table.matcher import distance, compute_iou

logger = get_logger()


def expand(pix, det_box, shape):
    x0, y0, x1, y1 = det_box
    #     print(shape)
    h, w, c = shape
    tmp_x0 = x0 - pix
    tmp_x1 = x1 + pix
    tmp_y0 = y0 - pix
    tmp_y1 = y1 + pix
    x0_ = tmp_x0 if tmp_x0 >= 0 else 0
    x1_ = tmp_x1 if tmp_x1 <= w else w
    y0_ = tmp_y0 if tmp_y0 >= 0 else 0
    y1_ = tmp_y1 if tmp_y1 <= h else h
    return x0_, y0_, x1_, y1_


class TableSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.table_structurer = predict_strture.TableStructurer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score

    def __call__(self, img):
        ori_im = img.copy()
        structure_res, elapse = self.table_structurer(copy.deepcopy(img))
        dt_boxes, elapse = self.text_detector(copy.deepcopy(img))
        dt_boxes = sorted_boxes(dt_boxes)

        r_boxes = []
        for box in dt_boxes:
            x_min = box[:, 0].min() - 1
            x_max = box[:, 0].max() + 1
            y_min = box[:, 1].min() - 1
            y_max = box[:, 1].max() + 1
            box = [x_min, y_min, x_max, y_max]
            r_boxes.append(box)
        dt_boxes = np.array(r_boxes)

        # logger.info("dt_boxes num : {}, elapse : {}".format(
        #     len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        for i in range(len(dt_boxes)):
            det_box = dt_boxes[i]
            x0, y0, x1, y1 = expand(2, det_box, ori_im.shape)
            text_rect = ori_im[int(y0):int(y1), int(x0):int(x1), :]
            img_crop_list.append(text_rect)
        rec_res, elapse = self.text_recognizer(img_crop_list)
        # logger.info("rec_res num  : {}, elapse : {}".format(
        #     len(rec_res), elapse))

        pred_html, pred = self.rebuild_table(structure_res, dt_boxes, rec_res)
        return pred_html

    def rebuild_table(self, structure_res, dt_boxes, rec_res):
        pred_structures, pred_bboxes = structure_res
        matched_index = self.match_result(dt_boxes, pred_bboxes)
        pred_html, pred = self.get_pred_html(pred_structures, matched_index, rec_res)
        return pred_html, pred

    def match_result(self, dt_boxes, pred_bboxes):
        matched = {}
        for i, gt_box in enumerate(dt_boxes):
            # gt_box = [np.min(gt_box[:, 0]), np.min(gt_box[:, 1]), np.max(gt_box[:, 0]), np.max(gt_box[:, 1])]
            distances = []
            for j, pred_box in enumerate(pred_bboxes):
                distances.append(
                    (distance(gt_box, pred_box), 1. - compute_iou(gt_box, pred_box)))  # 获取两两cell之间的L1距离和 1- IOU
            sorted_distances = distances.copy()
            # 根据距离和IOU挑选最"近"的cell
            sorted_distances = sorted(sorted_distances, key=lambda item: (item[1], item[0]))
            if distances.index(sorted_distances[0]) not in matched.keys():
                matched[distances.index(sorted_distances[0])] = [i]
            else:
                matched[distances.index(sorted_distances[0])].append(i)
        return matched

    def get_pred_html(self, pred_structures, matched_index, ocr_contents):
        end_html = []
        td_index = 0
        for tag in pred_structures:
            if '</td>' in tag:
                if td_index in matched_index.keys():
                    b_with = False
                    if '<b>' in ocr_contents[matched_index[td_index][0]] and len(matched_index[td_index]) > 1:
                        b_with = True
                        end_html.extend('<b>')
                    for i, td_index_index in enumerate(matched_index[td_index]):
                        content = ocr_contents[td_index_index][0]
                        if len(matched_index[td_index]) > 1:
                            if len(content) == 0:
                                continue
                            if content[0] == ' ':
                                content = content[1:]
                            if '<b>' in content:
                                content = content[3:]
                            if '</b>' in content:
                                content = content[:-4]
                            if len(content) == 0:
                                continue
                            if i != len(matched_index[td_index]) - 1 and ' ' != content[-1]:
                                content += ' '
                        end_html.extend(content)
                    if b_with:
                        end_html.extend('</b>')

                end_html.append(tag)
                td_index += 1
            else:
                end_html.append(tag)
        return ''.join(end_html), end_html


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

def to_excel(html_table, excel_path):
    from tablepyxl import tablepyxl
    tablepyxl.document_to_xl(html_table, excel_path)


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    excel_save_folder = 'output/table'
    os.makedirs(excel_save_folder, exist_ok=True)

    text_sys = TableSystem(args)
    img_num = len(image_file_list)
    for i, image_file in enumerate(image_file_list):
        logger.info("[{}/{}] {}".format(i, img_num, image_file))
        img, flag = check_and_read_gif(image_file)
        excel_path = os.path.join(excel_save_folder, os.path.basename(image_file).split('.')[0] + '.xlsx')
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        pred_html = text_sys(img)

        to_excel(pred_html, excel_path)
        logger.info('excel saved to {}'.format(excel_path))
        logger.info(pred_html)
        elapse = time.time() - starttime
        logger.info("Predict time : {:.3f}s".format(elapse))


if __name__ == "__main__":
    args = utility.parse_args()
    if args.use_mp:
        p_list = []
        total_process_num = args.total_process_num
        for process_id in range(total_process_num):
            cmd = [sys.executable, "-u"] + sys.argv + [
                "--process_id={}".format(process_id),
                "--use_mp={}".format(False)
            ]
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()
    else:
        main(args)
