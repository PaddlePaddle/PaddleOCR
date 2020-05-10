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

import utility
from ppocr.utils.utility import initial_logger
logger = initial_logger()
import cv2
import predict_det
import predict_rec
import copy
import numpy as np
import math
import time


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)

    def get_rotate_crop_image(self, img, points):
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        img_crop_width = int(np.linalg.norm(points[0] - points[1]))
        img_crop_height = int(np.linalg.norm(points[0] - points[3]))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],\
            [img_crop_width, img_crop_height], [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img_crop,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            print(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        if dt_boxes is None:
            return None, None
        img_crop_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        rec_res, elapse = self.text_recognizer(img_crop_list)
        #         self.print_draw_crop_rec_res(img_crop_list, rec_res)
        return dt_boxes, rec_res


if __name__ == "__main__":
    args = utility.parse_args()
    image_file_list = utility.get_image_file_list(args.image_dir)
    text_sys = TextSystem(args)
    for image_file in image_file_list:
        img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        dt_boxes, rec_res = text_sys(img)
        elapse = time.time() - starttime
        print("Predict time of %s: %.3fs" % (image_file, elapse))
        dt_num = len(dt_boxes)
        dt_boxes_final = []
        for dno in range(dt_num):
            text, score = rec_res[dno]
            if score >= 0:
                text_str = "%s, %.3f" % (text, score)
                print(text_str)
                dt_boxes_final.append(dt_boxes[dno])
        utility.draw_text_det_res(dt_boxes_final, image_file)
