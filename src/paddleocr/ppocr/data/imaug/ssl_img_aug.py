# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import math
import cv2
import numpy as np
import random
from PIL import Image

from .rec_img_aug import resize_norm_img


class SSLRotateResize(object):
    def __init__(
        self, image_shape, padding=False, select_all=True, mode="train", **kwargs
    ):
        self.image_shape = image_shape
        self.padding = padding
        self.select_all = select_all
        self.mode = mode

    def __call__(self, data):
        img = data["image"]

        data["image_r90"] = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        data["image_r180"] = cv2.rotate(data["image_r90"], cv2.ROTATE_90_CLOCKWISE)
        data["image_r270"] = cv2.rotate(data["image_r180"], cv2.ROTATE_90_CLOCKWISE)

        images = []
        for key in ["image", "image_r90", "image_r180", "image_r270"]:
            images.append(
                resize_norm_img(
                    data.pop(key), image_shape=self.image_shape, padding=self.padding
                )[0]
            )
        data["image"] = np.stack(images, axis=0)
        data["label"] = np.array(list(range(4)))
        if not self.select_all:
            data["image"] = data["image"][0::2]  # just choose 0 and 180
            data["label"] = data["label"][0:2]  # label needs to be continuous
        if self.mode == "test":
            data["image"] = data["image"][0]
            data["label"] = data["label"][0]
        return data
