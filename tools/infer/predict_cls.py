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

import tools.infer.utility as utility
from ppocr.utils.utility import initial_logger

logger = initial_logger()
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
import cv2
import copy
import numpy as np
import math
import time
from paddle import fluid


class TextClassifier(object):
    def __init__(self, args):
        if args.use_pdserving is False:
            self.predictor, self.input_tensor, self.output_tensors = \
                utility.create_predictor(args, mode="cls")
            self.use_zero_copy_run = args.use_zero_copy_run
        self.cls_image_shape = [int(v) for v in args.cls_image_shape.split(",")]
        self.cls_batch_num = args.rec_batch_num
        self.label_list = args.label_list
        self.cls_thresh = args.cls_thresh

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.cls_image_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        if self.cls_image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_list = copy.deepcopy(img_list)
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the cls process
        indices = np.argsort(np.array(width_list))

        cls_res = [['', 0.0]] * img_num
        batch_num = self.cls_batch_num
        predict_time = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            starttime = time.time()

            if self.use_zero_copy_run:
                self.input_tensor.copy_from_cpu(norm_img_batch)
                self.predictor.zero_copy_run()
            else:
                norm_img_batch = fluid.core.PaddleTensor(norm_img_batch)
                self.predictor.run([norm_img_batch])

            prob_out = self.output_tensors[0].copy_to_cpu()
            label_out = self.output_tensors[1].copy_to_cpu()
            if len(label_out.shape) != 1:
                prob_out, label_out = label_out, prob_out
            elapse = time.time() - starttime
            predict_time += elapse
            for rno in range(len(label_out)):
                label_idx = label_out[rno]
                score = prob_out[rno][label_idx]
                label = self.label_list[label_idx]
                cls_res[indices[beg_img_no + rno]] = [label, score]
                if '180' in label and score > self.cls_thresh:
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(
                        img_list[indices[beg_img_no + rno]], 1)
        return img_list, cls_res, predict_time


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_classifier = TextClassifier(args)
    valid_image_file_list = []
    img_list = []
    for image_file in image_file_list[:10]:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    try:
        img_list, cls_res, predict_time = text_classifier(img_list)
    except Exception as e:
        print(e)
        exit()
    for ino in range(len(img_list)):
        print("Predicts of %s:%s" % (valid_image_file_list[ino], cls_res[ino]))
    print("Total predict time for %d images:%.3f" %
          (len(img_list), predict_time))


if __name__ == "__main__":
    main(utility.parse_args())
