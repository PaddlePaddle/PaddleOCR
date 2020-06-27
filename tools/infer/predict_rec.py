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
from ppocr.utils.utility import get_image_file_list
import cv2
import copy
import numpy as np
import math
import time
from ppocr.utils.character import CharacterOps


class TextRecognizer(object):
    def __init__(self, args):
        self.predictor, self.input_tensor, self.output_tensors =\
            utility.create_predictor(args, mode="rec")
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.character_type = args.rec_char_type
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm
        char_ops_params = {"character_type": args.rec_char_type,
                           "character_dict_path": args.rec_char_dict_path}
        if self.rec_algorithm != "RARE":
            char_ops_params['loss_type'] = 'ctc'
            self.loss_type = 'ctc'
        else:
            char_ops_params['loss_type'] = 'attention'
            self.loss_type = 'attention'
        self.char_ops = CharacterOps(char_ops_params)

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        if self.character_type == "ch":
            imgW = int(math.ceil(32 * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH), interpolation=cv2.INTER_CUBIC)
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        # rec_res = []
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        predict_time = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                # h, w = img_list[ino].shape[0:2]
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                # norm_img = self.resize_norm_img(img_list[ino], max_wh_ratio)
                norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            starttime = time.time()
            self.input_tensor.copy_from_cpu(norm_img_batch)
            self.predictor.zero_copy_run()

            if self.loss_type == "ctc":
                rec_idx_batch = self.output_tensors[0].copy_to_cpu()
                rec_idx_lod = self.output_tensors[0].lod()[0]
                predict_batch = self.output_tensors[1].copy_to_cpu()
                predict_lod = self.output_tensors[1].lod()[0]
                elapse = time.time() - starttime
                predict_time += elapse
                for rno in range(len(rec_idx_lod) - 1):
                    beg = rec_idx_lod[rno]
                    end = rec_idx_lod[rno + 1]
                    rec_idx_tmp = rec_idx_batch[beg:end, 0]
                    preds_text = self.char_ops.decode(rec_idx_tmp)
                    beg = predict_lod[rno]
                    end = predict_lod[rno + 1]
                    probs = predict_batch[beg:end, :]
                    ind = np.argmax(probs, axis=1)
                    blank = probs.shape[1]
                    valid_ind = np.where(ind != (blank - 1))[0]
                    score = np.mean(probs[valid_ind, ind[valid_ind]])
                    # rec_res.append([preds_text, score])
                    rec_res[indices[beg_img_no + rno]] = [preds_text, score]
            else:
                rec_idx_batch = self.output_tensors[0].copy_to_cpu()
                predict_batch = self.output_tensors[1].copy_to_cpu()
                elapse = time.time() - starttime
                predict_time += elapse
                for rno in range(len(rec_idx_batch)):
                    end_pos = np.where(rec_idx_batch[rno, :] == 1)[0]
                    if len(end_pos) <= 1:
                        preds = rec_idx_batch[rno, 1:]
                        score = np.mean(predict_batch[rno, 1:])
                    else:
                        preds = rec_idx_batch[rno, 1:end_pos[1]]
                        score = np.mean(predict_batch[rno, 1:end_pos[1]])
                    preds_text = self.char_ops.decode(preds)
                    # rec_res.append([preds_text, score])
                    rec_res[indices[beg_img_no + rno]] = [preds_text, score]

        return rec_res, predict_time


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_recognizer = TextRecognizer(args)
    valid_image_file_list = []
    img_list = []
    for image_file in image_file_list:
        img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    try:
        rec_res, predict_time = text_recognizer(img_list)
    except Exception as e:
        print(e)
        logger.info(
            "ERROR!!!! \n"
            "Please read the FAQ：https://github.com/PaddlePaddle/PaddleOCR#faq \n"
            "If your model has tps module:  "
            "TPS does not support variable shape.\n"
            "Please set --rec_image_shape='3,32,100' and --rec_char_type='en' ")
        exit()
    for ino in range(len(img_list)):
        print("Predicts of %s:%s" % (valid_image_file_list[ino], rec_res[ino]))
    print("Total predict time for %d images:%.3f" %
          (len(img_list), predict_time))


if __name__ == "__main__":
    main(utility.parse_args())
