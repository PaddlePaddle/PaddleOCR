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
import os

# Prevent automatic updates in Albumentations for stability in augmentation behavior
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations.geometric import functional as fgeometric
from packaging import version

ALBU_VERSION = version.parse(A.__version__)
IS_ALBU_NEW_VERSION = ALBU_VERSION >= version.parse("1.4.15")


# Custom resize transformation mimicking Imgaug's behavior with scaling
class ImgaugLikeResize(DualTransform):
    def __init__(self, scale_range=(0.5, 3.0), interpolation=1, p=1.0):
        super(ImgaugLikeResize, self).__init__(p)
        self.scale_range = scale_range
        self.interpolation = interpolation

    # Resize the image based on a randomly chosen scale within the scale range
    def apply(self, img, scale=1.0, **params):
        height, width = img.shape[:2]
        new_height = int(height * scale)
        new_width = int(width * scale)

        if IS_ALBU_NEW_VERSION:
            return fgeometric.resize(
                img, (new_height, new_width), interpolation=self.interpolation
            )
        return fgeometric.resize(
            img, new_height, new_width, interpolation=self.interpolation
        )

    # Apply the same scaling transformation to keypoints (e.g., polygon points)
    def apply_to_keypoints(self, keypoints, scale=1.0, **params):
        return np.array(
            [(x * scale, y * scale) + tuple(rest) for x, y, *rest in keypoints]
        )

    # Get random scale parameter within the specified range
    def get_params(self):
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return {"scale": scale}


# Builder class to translate custom augmenter arguments into Albumentations-compatible format
class AugmenterBuilder(object):
    def __init__(self):
        # Map common Imgaug transformations to equivalent Albumentations transforms
        self.imgaug_to_albu = {
            "Fliplr": "HorizontalFlip",
            "Flipud": "VerticalFlip",
            "Affine": "Affine",
            # Additional mappings can be added here if needed
        }

    # Recursive method to construct augmentation pipeline based on provided arguments
    def build(self, args, root=True):
        if args is None or len(args) == 0:
            return None
        elif isinstance(args, list):
            # Build the full augmentation sequence if it's a root-level call
            if root:
                sequence = [self.build(value, root=False) for value in args]
                return A.Compose(
                    sequence,
                    keypoint_params=A.KeypointParams(
                        format="xy", remove_invisible=False
                    ),
                )
            else:
                # Build individual augmenters for nested arguments
                augmenter_type = args[0]
                augmenter_args = args[1] if len(args) > 1 else {}
                augmenter_args_mapped = self.map_arguments(
                    augmenter_type, augmenter_args
                )
                augmenter_type_mapped = self.imgaug_to_albu.get(
                    augmenter_type, augmenter_type
                )
                if augmenter_type_mapped == "Resize":
                    return ImgaugLikeResize(**augmenter_args_mapped)
                else:
                    cls = getattr(A, augmenter_type_mapped)
                    return cls(
                        **{
                            k: self.to_tuple_if_list(v)
                            for k, v in augmenter_args_mapped.items()
                        }
                    )
        elif isinstance(args, dict):
            # Process individual transformation specified as dictionary
            augmenter_type = args["type"]
            augmenter_args = args.get("args", {})
            augmenter_args_mapped = self.map_arguments(augmenter_type, augmenter_args)
            augmenter_type_mapped = self.imgaug_to_albu.get(
                augmenter_type, augmenter_type
            )
            if augmenter_type_mapped == "Resize":
                return ImgaugLikeResize(**augmenter_args_mapped)
            else:
                cls = getattr(A, augmenter_type_mapped)
                return cls(
                    **{
                        k: self.to_tuple_if_list(v)
                        for k, v in augmenter_args_mapped.items()
                    }
                )
        else:
            raise RuntimeError("Unknown augmenter arg: " + str(args))

    # Map arguments to expected format for each augmenter type
    def map_arguments(self, augmenter_type, augmenter_args):
        augmenter_args = augmenter_args.copy()  # Avoid modifying the original arguments
        if augmenter_type == "Resize":
            # Ensure size is a valid 2-element list or tuple
            size = augmenter_args.get("size")
            if size:
                if not isinstance(size, (list, tuple)) or len(size) != 2:
                    raise ValueError(
                        f"'size' must be a list or tuple of two numbers, but got {size}"
                    )
                min_scale, max_scale = size
                return {
                    "scale_range": (min_scale, max_scale),
                    "interpolation": 1,  # Linear interpolation
                    "p": 1.0,
                }
            else:
                return {"scale_range": (1.0, 1.0), "interpolation": 1, "p": 1.0}
        elif augmenter_type == "Affine":
            # Map rotation to a tuple and ensure p=1.0 to apply transformation
            rotate = augmenter_args.get("rotate", 0)
            if isinstance(rotate, list):
                rotate = tuple(rotate)
            elif isinstance(rotate, (int, float)):
                rotate = (float(rotate), float(rotate))
            augmenter_args["rotate"] = rotate
            augmenter_args["p"] = 1.0
            return augmenter_args
        else:
            # For other augmenters, ensure 'p' probability is specified
            p = augmenter_args.get("p", 1.0)
            augmenter_args["p"] = p
            return augmenter_args

    # Convert lists to tuples for Albumentations compatibility
    def to_tuple_if_list(self, obj):
        if isinstance(obj, list):
            return tuple(obj)
        return obj


# Wrapper class for image and polygon transformations using Imgaug-style augmentation
class IaaAugment:
    def __init__(self, augmenter_args=None, **kwargs):
        if augmenter_args is None:
            # Default augmenters if none are specified
            augmenter_args = [
                {"type": "Fliplr", "args": {"p": 0.5}},
                {"type": "Affine", "args": {"rotate": [-10, 10]}},
                {"type": "Resize", "args": {"size": [0.5, 3]}},
            ]
        self.augmenter = AugmenterBuilder().build(augmenter_args)

    # Apply the augmentations to image and polygon data
    def __call__(self, data):
        image = data["image"]

        if self.augmenter:
            # Flatten polygons to individual keypoints for transformation
            keypoints = []
            keypoints_lengths = []
            for poly in data["polys"]:
                keypoints.extend([tuple(point) for point in poly])
                keypoints_lengths.append(len(poly))

            # Apply the augmentation pipeline to image and keypoints
            transformed = self.augmenter(image=image, keypoints=keypoints)
            data["image"] = transformed["image"]

            # Extract transformed keypoints and reconstruct polygon structures
            transformed_keypoints = transformed["keypoints"]

            # Reassemble polygons from transformed keypoints
            new_polys = []
            idx = 0
            for length in keypoints_lengths:
                new_poly = transformed_keypoints[idx : idx + length]
                new_polys.append(np.array([kp[:2] for kp in new_poly]))
                idx += length
            data["polys"] = np.array(new_polys)
        return data
