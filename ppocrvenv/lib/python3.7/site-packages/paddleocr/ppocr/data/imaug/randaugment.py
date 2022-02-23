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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
import six


class RawRandAugment(object):
    def __init__(self,
                 num_layers=2,
                 magnitude=5,
                 fillcolor=(128, 128, 128),
                 **kwargs):
        self.num_layers = num_layers
        self.magnitude = magnitude
        self.max_level = 10

        abso_level = self.magnitude / self.max_level
        self.level_map = {
            "shearX": 0.3 * abso_level,
            "shearY": 0.3 * abso_level,
            "translateX": 150.0 / 331 * abso_level,
            "translateY": 150.0 / 331 * abso_level,
            "rotate": 30 * abso_level,
            "color": 0.9 * abso_level,
            "posterize": int(4.0 * abso_level),
            "solarize": 256.0 * abso_level,
            "contrast": 0.9 * abso_level,
            "sharpness": 0.9 * abso_level,
            "brightness": 0.9 * abso_level,
            "autocontrast": 0,
            "equalize": 0,
            "invert": 0
        }

        # from https://stackoverflow.com/questions/5252170/
        # specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot,
                                   Image.new("RGBA", rot.size, (128, ) * 4),
                                   rot).convert(img.mode)

        rnd_ch_op = random.choice

        self.func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * rnd_ch_op([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * rnd_ch_op([-1, 1]), 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * img.size[0] * rnd_ch_op([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * img.size[1] * rnd_ch_op([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(
                1 + magnitude * rnd_ch_op([-1, 1])),
            "posterize": lambda img, magnitude:
            ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude:
            ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude:
            ImageEnhance.Contrast(img).enhance(
                1 + magnitude * rnd_ch_op([-1, 1])),
            "sharpness": lambda img, magnitude:
            ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * rnd_ch_op([-1, 1])),
            "brightness": lambda img, magnitude:
            ImageEnhance.Brightness(img).enhance(
                1 + magnitude * rnd_ch_op([-1, 1])),
            "autocontrast": lambda img, magnitude:
            ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

    def __call__(self, img):
        avaiable_op_names = list(self.level_map.keys())
        for layer_num in range(self.num_layers):
            op_name = np.random.choice(avaiable_op_names)
            img = self.func[op_name](img, self.level_map[op_name])
        return img


class RandAugment(RawRandAugment):
    """ RandAugment wrapper to auto fit different img types """

    def __init__(self, prob=0.5, *args, **kwargs):
        self.prob = prob
        if six.PY2:
            super(RandAugment, self).__init__(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs)

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data
        img = data['image']
        if not isinstance(img, Image.Image):
            img = np.ascontiguousarray(img)
            img = Image.fromarray(img)

        if six.PY2:
            img = super(RandAugment, self).__call__(img)
        else:
            img = super().__call__(img)

        if isinstance(img, Image.Image):
            img = np.asarray(img)
        data['image'] = img
        return data
