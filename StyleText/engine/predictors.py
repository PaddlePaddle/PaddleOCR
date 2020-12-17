# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import cv2
import math
import paddle

from arch import style_text_rec
from utils.sys_funcs import check_gpu
from utils.logging import get_logger


class StyleTextRecPredictor(object):
    def __init__(self, config):
        algorithm = config['Predictor']['algorithm']
        assert algorithm in ["StyleTextRec"
                             ], "Generator {} not supported.".format(algorithm)
        use_gpu = config["Global"]['use_gpu']
        check_gpu(use_gpu)
        paddle.set_device('gpu' if use_gpu else 'cpu')
        self.logger = get_logger()
        self.generator = getattr(style_text_rec, algorithm)(config)
        self.height = config["Global"]["image_height"]
        self.width = config["Global"]["image_width"]
        self.scale = config["Predictor"]["scale"]
        self.mean = config["Predictor"]["mean"]
        self.std = config["Predictor"]["std"]
        self.expand_result = config["Predictor"]["expand_result"]

    def predict(self, style_input, text_input):
        style_input = self.rep_style_input(style_input, text_input)
        tensor_style_input = self.preprocess(style_input)
        tensor_text_input = self.preprocess(text_input)
        style_text_result = self.generator.forward(tensor_style_input,
                                                   tensor_text_input)
        fake_fusion = self.postprocess(style_text_result["fake_fusion"])
        fake_text = self.postprocess(style_text_result["fake_text"])
        fake_sk = self.postprocess(style_text_result["fake_sk"])
        fake_bg = self.postprocess(style_text_result["fake_bg"])
        bbox = self.get_text_boundary(fake_text)
        if bbox:
            left, right, top, bottom = bbox
            fake_fusion = fake_fusion[top:bottom, left:right, :]
            fake_text = fake_text[top:bottom, left:right, :]
            fake_sk = fake_sk[top:bottom, left:right, :]
            fake_bg = fake_bg[top:bottom, left:right, :]

        # fake_fusion = self.crop_by_text(img_fake_fusion, img_fake_text)
        return {
            "fake_fusion": fake_fusion,
            "fake_text": fake_text,
            "fake_sk": fake_sk,
            "fake_bg": fake_bg,
        }

    def preprocess(self, img):
        img = (img.astype('float32') * self.scale - self.mean) / self.std
        img_height, img_width, channel = img.shape
        assert channel == 3, "Please use an rgb image."
        ratio = img_width / float(img_height)
        if math.ceil(self.height * ratio) > self.width:
            resized_w = self.width
        else:
            resized_w = int(math.ceil(self.height * ratio))
        img = cv2.resize(img, (resized_w, self.height))

        new_img = np.zeros([self.height, self.width, 3]).astype('float32')
        new_img[:, 0:resized_w, :] = img
        img = new_img.transpose((2, 0, 1))
        img = img[np.newaxis, :, :, :]
        return paddle.to_tensor(img)

    def postprocess(self, tensor):
        img = tensor.numpy()[0]
        img = img.transpose((1, 2, 0))
        img = (img * self.std + self.mean) / self.scale
        img = np.maximum(img, 0.0)
        img = np.minimum(img, 255.0)
        img = img.astype('uint8')
        return img

    def rep_style_input(self, style_input, text_input):
        rep_num = int(1.2 * (text_input.shape[1] / text_input.shape[0]) /
                      (style_input.shape[1] / style_input.shape[0])) + 1
        style_input = np.tile(style_input, reps=[1, rep_num, 1])
        max_width = int(self.width / self.height * style_input.shape[0])
        style_input = style_input[:, :max_width, :]
        return style_input

    def get_text_boundary(self, text_img):
        img_height = text_img.shape[0]
        img_width = text_img.shape[1]
        bounder = 3
        text_canny_img = cv2.Canny(text_img, 10, 20)
        edge_num_h = text_canny_img.sum(axis=0)
        no_zero_list_h = np.where(edge_num_h > 0)[0]
        edge_num_w = text_canny_img.sum(axis=1)
        no_zero_list_w = np.where(edge_num_w > 0)[0]
        if len(no_zero_list_h) == 0 or len(no_zero_list_w) == 0:
            return None
        left = max(no_zero_list_h[0] - bounder, 0)
        right = min(no_zero_list_h[-1] + bounder, img_width)
        top = max(no_zero_list_w[0] - bounder, 0)
        bottom = min(no_zero_list_w[-1] + bounder, img_height)
        return [left, right, top, bottom]
