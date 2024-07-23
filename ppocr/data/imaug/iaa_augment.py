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
"""
This code is refer from:
https://github.com/WenmuZhou/DBNet.pytorch/blob/master/data_loader/modules/iaa_augment.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import numpy as np
import albumentations as A


class AugmenterBuilder(object):
    def __init__(self):
        pass

    def build(self, args, root=True):
        if args is None or len(args) == 0:
            return None
        elif isinstance(args, list):
            if root:
                sequence = [self.build(value, root=False) for value in args]
                return A.Compose(
                    sequence, keypoint_params=A.KeypointParams(format="xy")
                )
            else:
                return getattr(A, args[0])(
                    *[self.to_tuple_if_list(a) for a in args[1:]]
                )
        elif isinstance(args, dict):
            cls = getattr(A, args["type"])
            return cls(**{k: self.to_tuple_if_list(v) for k, v in args["args"].items()})
        else:
            raise RuntimeError("unknown augmenter arg: " + str(args))

    def to_tuple_if_list(self, obj):
        if isinstance(obj, list):
            return tuple(obj)
        return obj


class IaaAugment:
    def __init__(self, augmenter_args=None, **kwargs):
        if augmenter_args is None:
            augmenter_args = [
                {"type": "HorizontalFlip", "args": {"p": 0.5}},
                {"type": "Affine", "args": {"rotate": [-10, 10]}},
                {"type": "RandomScale", "args": {"scale_limit": (0.5, 3)}},
            ]
        self.augmenter = AugmenterBuilder().build(augmenter_args)

    def __call__(self, data):
        image = data["image"]

        if self.augmenter:
            # note: Affine could work incorrectly in ReplayMode if set_deterministic(True)
            # self.augmenter.set_deterministic(True)
            data["image"] = self.augmenter(image=image)["image"]
            data = self.may_augment_annotation(data)
        return data

    def may_augment_annotation(self, data):
        if self.augmenter is None:
            return data

        line_polys = []
        for poly in data["polys"]:
            new_poly = self.may_augment_poly(data["image"], poly)
            line_polys.append(new_poly)
        data["polys"] = np.array(line_polys)
        return data

    def may_augment_poly(self, image, poly):
        keypoints = [(p[0], p[1]) for p in poly]
        poly = self.augmenter(image=image, keypoints=keypoints)["keypoints"]
        return poly
