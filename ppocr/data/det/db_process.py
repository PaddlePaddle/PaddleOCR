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
import json
import sys
from ppocr.utils.utility import initial_logger
logger = initial_logger()

from .data_augment import AugmentData
from .random_crop_data import RandomCropData
from .make_shrink_map import MakeShrinkMap
from .make_border_map import MakeBorderMap


class DBProcessTrain(object):
    """
    DB pre-process for Train mode
    """

    def __init__(self, params):
        self.img_set_dir = params['img_set_dir']
        self.image_shape = params['image_shape']

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def make_data_dict(self, imgvalue, entry):
        boxes = []
        texts = []
        ignores = []
        for rect in entry:
            points = rect['points']
            transcription = rect['transcription']
            try:
                box = self.order_points_clockwise(
                    np.array(points).reshape(-1, 2))
                if cv2.contourArea(box) > 0:
                    boxes.append(box)
                    texts.append(transcription)
                    ignores.append(transcription in ['*', '###'])
            except:
                print('load label failed!')
        data = {
            'image': imgvalue,
            'shape': [imgvalue.shape[0], imgvalue.shape[1]],
            'polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores,
        }
        return data

    def NormalizeImage(self, data):
        im = data['image']
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        im = im.astype(np.float32, copy=False)
        im = im / 255
        im -= img_mean
        im /= img_std
        channel_swap = (2, 0, 1)
        im = im.transpose(channel_swap)
        data['image'] = im
        return data

    def FilterKeys(self, data):
        filter_keys = ['polys', 'texts', 'ignore_tags', 'shape']
        for key in filter_keys:
            if key in data:
                del data[key]
        return data

    def convert_label_infor(self, label_infor):
        label_infor = label_infor.decode()
        label_infor = label_infor.encode('utf-8').decode('utf-8-sig')
        substr = label_infor.strip("\n").split("\t")
        img_path = self.img_set_dir + substr[0]
        label = json.loads(substr[1])
        return img_path, label

    def __call__(self, label_infor):
        img_path, gt_label = self.convert_label_infor(label_infor)
        imgvalue = cv2.imread(img_path)
        if imgvalue is None:
            logger.info("{} does not exist!".format(img_path))
            return None
        if len(list(imgvalue.shape)) == 2 or imgvalue.shape[2] == 1:
            imgvalue = cv2.cvtColor(imgvalue, cv2.COLOR_GRAY2BGR)
        data = self.make_data_dict(imgvalue, gt_label)
        data = AugmentData(data)
        data = RandomCropData(data, self.image_shape[1:])
        data = MakeShrinkMap(data)
        data = MakeBorderMap(data)
        data = self.NormalizeImage(data)
        data = self.FilterKeys(data)
        return data['image'], data['shrink_map'], data['shrink_mask'], data[
            'threshold_map'], data['threshold_mask']


class DBProcessTest(object):
    """
    DB pre-process for Test mode
    """

    def __init__(self, params):
        super(DBProcessTest, self).__init__()
        self.resize_type = 0
        if 'det_image_shape' in params:
            self.image_shape = params['det_image_shape']
            # print(self.image_shape)
            self.resize_type = 1
        if 'max_side_len' in params:
            self.max_side_len = params['max_side_len']
        else:
            self.max_side_len = 2400

    def resize_image_type0(self, im):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        max_side_len = self.max_side_len
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # limit the max side
        if max(resize_h, resize_w) > max_side_len:
            if resize_h > resize_w:
                ratio = float(max_side_len) / resize_h
            else:
                ratio = float(max_side_len) / resize_w
        else:
            ratio = 1.
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)
        if resize_h % 32 == 0:
            resize_h = resize_h
        elif resize_h // 32 <= 1:
            resize_h = 32
        else:
            resize_h = (resize_h // 32 - 1) * 32
        if resize_w % 32 == 0:
            resize_w = resize_w
        elif resize_w // 32 <= 1:
            resize_w = 32
        else:
            resize_w = (resize_w // 32 - 1) * 32
        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            im = cv2.resize(im, (int(resize_w), int(resize_h)))
        except:
            print(im.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return im, (ratio_h, ratio_w)

    def resize_image_type1(self, im):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = im.shape[:2]  # (h, w, c)
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        return im, (ratio_h, ratio_w)

    def normalize(self, im):
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        im = im.astype(np.float32, copy=False)
        im = im / 255
        im -= img_mean
        im /= img_std
        channel_swap = (2, 0, 1)
        im = im.transpose(channel_swap)
        return im

    def __call__(self, im):
        if self.resize_type == 0:
            im, (ratio_h, ratio_w) = self.resize_image_type0(im)
        else:
            im, (ratio_h, ratio_w) = self.resize_image_type1(im)
        im = self.normalize(im)
        im = im[np.newaxis, :]
        return [im, (ratio_h, ratio_w)]
