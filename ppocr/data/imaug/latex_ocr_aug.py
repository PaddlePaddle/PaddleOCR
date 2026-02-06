# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

"""
This code is refer from:
https://github.com/lukas-blecher/LaTeX-OCR/blob/main/pix2tex/dataset/transforms.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import math
import cv2
import numpy as np
import albumentations as A
from PIL import Image


class LatexTrainTransform:
    def __init__(self, bitmap_prob=0.04, **kwargs):
        # your init code
        self.bitmap_prob = bitmap_prob
        self.train_transform = A.Compose(
            [
                A.Compose(
                    [
                        A.ShiftScaleRotate(
                            shift_limit=0,
                            scale_limit=(-0.15, 0),
                            rotate_limit=1,
                            border_mode=0,
                            interpolation=3,
                            value=[255, 255, 255],
                            p=1,
                        ),
                        A.GridDistortion(
                            distort_limit=0.1,
                            border_mode=0,
                            interpolation=3,
                            value=[255, 255, 255],
                            p=0.5,
                        ),
                    ],
                    p=0.15,
                ),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
                A.GaussNoise(10, p=0.2),
                A.RandomBrightnessContrast(0.05, (-0.2, 0), True, p=0.2),
                A.ImageCompression(95, p=0.3),
                A.ToGray(always_apply=True),
            ]
        )

    def __call__(self, data):
        img = data["image"]
        if np.random.random() < self.bitmap_prob:
            img[img != 255] = 0
        img = self.train_transform(image=img)["image"]
        data["image"] = img
        return data


class LatexTestTransform:
    def __init__(self, **kwargs):
        # your init code
        self.test_transform = A.Compose(
            [
                A.ToGray(always_apply=True),
            ]
        )

    def __call__(self, data):
        img = data["image"]
        img = self.test_transform(image=img)["image"]
        data["image"] = img
        return data


class MinMaxResize:
    def __init__(self, min_dimensions=[32, 32], max_dimensions=[672, 192], **kwargs):
        # your init code
        self.min_dimensions = min_dimensions
        self.max_dimensions = max_dimensions
        # pass

    def pad_(self, img, divable=32):
        threshold = 128
        data = np.array(img.convert("LA"))
        if data[..., -1].var() == 0:
            data = (data[..., 0]).astype(np.uint8)
        else:
            data = (255 - data[..., -1]).astype(np.uint8)
        data = (data - data.min()) / (data.max() - data.min()) * 255
        if data.mean() > threshold:
            # To invert the text to white
            gray = 255 * (data < threshold).astype(np.uint8)
        else:
            gray = 255 * (data > threshold).astype(np.uint8)
            data = 255 - data

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        rect = data[b : b + h, a : a + w]
        im = Image.fromarray(rect).convert("L")
        dims = []
        for x in [w, h]:
            div, mod = divmod(x, divable)
            dims.append(divable * (div + (1 if mod > 0 else 0)))
        padded = Image.new("L", dims, 255)
        padded.paste(im, (0, 0, im.size[0], im.size[1]))
        return padded

    def minmax_size_(self, img, max_dimensions, min_dimensions):
        if max_dimensions is not None:
            ratios = [a / b for a, b in zip(img.size, max_dimensions)]
            if any([r > 1 for r in ratios]):
                size = np.array(img.size) // max(ratios)
                img = img.resize(tuple(size.astype(int)), Image.BILINEAR)
        if min_dimensions is not None:
            # hypothesis: there is a dim in img smaller than min_dimensions, and return a proper dim >= min_dimensions
            padded_size = [
                max(img_dim, min_dim)
                for img_dim, min_dim in zip(img.size, min_dimensions)
            ]
            if padded_size != list(img.size):  # assert hypothesis
                padded_im = Image.new("L", padded_size, 255)
                padded_im.paste(img, img.getbbox())
                img = padded_im
        return img

    def __call__(self, data):
        img = data["image"]
        h, w = img.shape[:2]
        if (
            self.min_dimensions[0] <= w <= self.max_dimensions[0]
            and self.min_dimensions[1] <= h <= self.max_dimensions[1]
        ):
            return data
        else:
            im = Image.fromarray(np.uint8(img))
            im = self.minmax_size_(
                self.pad_(im), self.max_dimensions, self.min_dimensions
            )
            im = np.array(im)
            im = np.dstack((im, im, im))
            data["image"] = im
            return data


class LatexImageFormat:
    def __init__(self, **kwargs):
        # your init code
        pass

    def __call__(self, data):
        img = data["image"]
        im_h, im_w = img.shape[:2]
        divide_h = math.ceil(im_h / 16) * 16
        divide_w = math.ceil(im_w / 16) * 16
        img = img[:, :, 0]
        img = np.pad(
            img, ((0, divide_h - im_h), (0, divide_w - im_w)), constant_values=(1, 1)
        )
        img_expanded = img[:, :, np.newaxis].transpose(2, 0, 1)
        data["image"] = img_expanded
        return data
