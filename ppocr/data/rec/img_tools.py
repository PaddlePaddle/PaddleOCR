#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import math
import cv2
import numpy as np
from ppocr.utils.utility import initial_logger
logger = initial_logger()


def get_bounding_box_rect(pos):
    left = min(pos[0])
    right = max(pos[0])
    top = min(pos[1])
    bottom = max(pos[1])
    return [left, top, right, bottom]


def resize_norm_img(img, image_shape):
    imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


def resize_norm_img_chinese(img, image_shape):
    imgC, imgH, imgW = image_shape
    # todo: change to 0 and modified image shape
    max_wh_ratio = 0
    h, w = img.shape[0], img.shape[1]
    ratio = w * 1.0 / h
    max_wh_ratio = max(max_wh_ratio, ratio)
    imgW = int(32 * max_wh_ratio)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


def get_img_data(value):
    """get_img_data"""
    if not value:
        return None
    imgdata = np.frombuffer(value, dtype='uint8')
    if imgdata is None:
        return None
    imgori = cv2.imdecode(imgdata, 1)
    if imgori is None:
        return None
    return imgori


def process_image(img,
                  image_shape,
                  label=None,
                  char_ops=None,
                  loss_type=None,
                  max_text_length=None,
                  tps=None,
                  infer_mode=False):
    if infer_mode and char_ops.character_type == "ch" and not tps:
        norm_img = resize_norm_img_chinese(img, image_shape)
    else:
        norm_img = resize_norm_img(img, image_shape)

    norm_img = norm_img[np.newaxis, :]
    if label is not None:
        # char_num = char_ops.get_char_num()
        text = char_ops.encode(label)
        if len(text) == 0 or len(text) > max_text_length:
            logger.info(
                "Warning in ppocr/data/rec/img_tools.py:line106: Wrong data type."
                "Excepted string with length between 1 and {}, but "
                "got '{}'. Label is '{}'".format(max_text_length,
                                                 len(text), label))
            return None
        else:
            if loss_type == "ctc":
                text = text.reshape(-1, 1)
                return (norm_img, text)
            elif loss_type == "attention":
                beg_flag_idx = char_ops.get_beg_end_flag_idx("beg")
                end_flag_idx = char_ops.get_beg_end_flag_idx("end")
                beg_text = np.append(beg_flag_idx, text)
                end_text = np.append(text, end_flag_idx)
                beg_text = beg_text.reshape(-1, 1)
                end_text = end_text.reshape(-1, 1)
                return (norm_img, beg_text, end_text)
            else:
                assert False, "Unsupport loss_type %s in process_image"\
                    % loss_type
    return (norm_img)
