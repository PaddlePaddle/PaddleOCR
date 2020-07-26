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
__dir__ = os.path.dirname(__file__)
sys.path.append(__dir__)
sys.path.append(os.path.join(__dir__, '../..'))

import tools.infer.utility as utility
from ppocr.utils.utility import initial_logger
logger = initial_logger()
from ppocr.utils.utility import get_image_file_list
import cv2
from ppocr.data.det.east_process import EASTProcessTest
from ppocr.data.det.db_process import DBProcessTest
from ppocr.postprocess.db_postprocess import DBPostProcess
from ppocr.postprocess.east_postprocess import EASTPostPocess
import copy
import numpy as np
import math
import time
import sys


class TextDetector(object):
    def __init__(self, args):
        max_side_len = args.det_max_side_len
        self.det_algorithm = args.det_algorithm
        preprocess_params = {'max_side_len': max_side_len}
        postprocess_params = {}
        if self.det_algorithm == "DB":
            self.preprocess_op = DBProcessTest(preprocess_params)
            postprocess_params["thresh"] = args.det_db_thresh
            postprocess_params["box_thresh"] = args.det_db_box_thresh
            postprocess_params["max_candidates"] = 1000
            postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
            self.postprocess_op = DBPostProcess(postprocess_params)
        elif self.det_algorithm == "EAST":
            self.preprocess_op = EASTProcessTest(preprocess_params)
            postprocess_params["score_thresh"] = args.det_east_score_thresh
            postprocess_params["cover_thresh"] = args.det_east_cover_thresh
            postprocess_params["nms_thresh"] = args.det_east_nms_thresh
            self.postprocess_op = EASTPostPocess(postprocess_params)
        else:
            logger.info("unknown det_algorithm:{}".format(self.det_algorithm))
            sys.exit(0)

        self.predictor, self.input_tensor, self.output_tensors =\
            utility.create_predictor(args, mode="det")

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(4):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 10 or rect_height <= 10:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        im, ratio_list = self.preprocess_op(img)
        if im is None:
            return None, 0
        im = im.copy()
        starttime = time.time()
        self.input_tensor.copy_from_cpu(im)
        self.predictor.zero_copy_run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
        outs_dict = {}
        if self.det_algorithm == "EAST":
            outs_dict['f_geo'] = outputs[0]
            outs_dict['f_score'] = outputs[1]
        else:
            outs_dict['maps'] = outputs[0]
        dt_boxes_list = self.postprocess_op(outs_dict, [ratio_list])
        dt_boxes = dt_boxes_list[0]
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        elapse = time.time() - starttime
        return dt_boxes, elapse


if __name__ == "__main__":
    args = utility.parse_args()
    image_file_list = get_image_file_list(args.image_dir)
    text_detector = TextDetector(args)
    count = 0
    total_time = 0
    draw_img_save = "./inference_results"
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    for image_file in image_file_list:
        img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        dt_boxes, elapse = text_detector(img)
        if count > 0:
            total_time += elapse
        count += 1
        print("Predict time of %s:" % image_file, elapse)
        src_im = utility.draw_text_det_res(dt_boxes, image_file)
        img_name_pure = image_file.split("/")[-1]
        cv2.imwrite(
            os.path.join(draw_img_save, "det_res_%s" % img_name_pure), src_im)
    if count > 1:
        print("Avg Time:", total_time / (count - 1))
