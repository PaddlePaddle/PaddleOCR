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

import numpy as np

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A


class AugmenterBuilder:
    def __init__(self):
        pass

    def build(self, args):
        if not args:
            return None
        elif isinstance(args, list):
            # Recursively build transforms from the list
            transforms = [self.build(value) for value in args if self.build(value)]
            return A.Compose(
                transforms,
                # Use KeypointParams to handle keypoints (polygons represented as keypoints)
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )
        elif isinstance(args, dict):
            # Get the transform type and its arguments
            transform_type = args.get("type")
            transform_args = args.get("args", {})
            # Map the transform type to the corresponding function
            transform_func = self._get_transform_function(
                transform_type, transform_args
            )
            if transform_func:
                return transform_func
            else:
                raise NotImplementedError(
                    f"Transform {transform_type} not implemented."
                )
        else:
            raise RuntimeError(f"Unknown augmenter arg: {args}")

    def _get_transform_function(self, transform_type, transform_args):
        # Define mapping from transform types to functions
        transform_mapping = {
            "Fliplr": self._build_horizontal_flip,
            "Affine": self._build_affine,
            "Resize": self._build_resize,
        }
        func = transform_mapping.get(transform_type)
        if func:
            return func(transform_args)
        else:
            return None

    def _build_horizontal_flip(self, transform_args):
        p = transform_args.get("p", 0.5)
        return A.HorizontalFlip(p=p)

    def _build_affine(self, transform_args):
        rotate = transform_args.get("rotate")
        shear = transform_args.get("shear")
        translate_percent = transform_args.get("translate_percent")
        affine_args = {"fit_output": True}
        if rotate is not None:
            affine_args["rotate"] = (
                tuple(rotate) if isinstance(rotate, list) else rotate
            )
        if shear is not None:
            affine_args["shear"] = shear
        if translate_percent is not None:
            affine_args["translate_percent"] = translate_percent
        return A.Affine(**affine_args)

    def _build_resize(self, transform_args):
        size = transform_args.get("size", [1.0, 1.0])
        if isinstance(size, list) and len(size) == 2:
            scale_factor = size[0]
            height = int(scale_factor * 100)
            width = int(scale_factor * 100)
            return A.Resize(height=height, width=width)
        elif isinstance(size, (int, float)):
            scale_factor = float(size)
            height = int(scale_factor * 100)
            width = int(scale_factor * 100)
            return A.Resize(height=height, width=width)
        else:
            raise ValueError("Invalid size parameter for Resize")


class IaaAugment:
    def __init__(self, augmenter_args=None, **kwargs):
        if augmenter_args is None:
            augmenter_args = [
                {"type": "Fliplr", "args": {"p": 0.5}},
                {"type": "Affine", "args": {"rotate": [-10, 10]}},
                {"type": "Resize", "args": {"size": [0.5, 3.0]}},
            ]
        self.augmenter = AugmenterBuilder().build(augmenter_args)

    def __call__(self, data):
        image = data["image"]
        polys = data["polys"]
        # Flatten polys to keypoints and keep track of groupings
        keypoints = []
        keypoint_groups = []
        for poly in polys:
            poly_keypoints = [tuple(point) for point in poly]
            keypoints.extend(poly_keypoints)
            keypoint_groups.append(len(poly_keypoints))
        if self.augmenter:
            transformed = self.augmenter(image=image, keypoints=keypoints)
            data["image"] = transformed["image"]
            # Reconstruct polys from transformed keypoints
            transformed_keypoints = transformed["keypoints"]
            new_polys = []
            idx = 0
            for group_length in keypoint_groups:
                new_poly = np.array(
                    transformed_keypoints[idx : idx + group_length], dtype=np.float32
                )
                new_polys.append(new_poly)
                idx += group_length
            data["polys"] = new_polys
        return data
