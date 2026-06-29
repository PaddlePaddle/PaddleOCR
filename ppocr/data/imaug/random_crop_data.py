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
https://github.com/WenmuZhou/DBNet.pytorch/blob/master/data_loader/modules/random_crop_data.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
import random

from paddle import get_device

from shapely.geometry import Polygon, box as shapely_box
from shapely import intersection


def is_poly_in_rect(poly, x, y, w, h):
    poly = np.array(poly)
    if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
        return False
    if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
        return False
    return True


def is_poly_outside_rect(poly, x, y, w, h):
    poly = np.array(poly)
    if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
        return True
    if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
        return True
    return False


def split_regions(axis):
    regions = []
    min_axis = 0
    for i in range(1, axis.shape[0]):
        if axis[i] != axis[i - 1] + 1:
            region = axis[min_axis:i]
            min_axis = i
            regions.append(region)
    return regions


def random_select(axis, max_size):
    xx = np.random.choice(axis, size=2)
    xmin = np.min(xx)
    xmax = np.max(xx)
    xmin = np.clip(xmin, 0, max_size - 1)
    xmax = np.clip(xmax, 0, max_size - 1)
    return xmin, xmax


def region_wise_random_select(regions, max_size):
    selected_index = list(np.random.choice(len(regions), 2))
    selected_values = []
    for index in selected_index:
        axis = regions[index]
        xx = int(np.random.choice(axis, size=1))
        selected_values.append(xx)
    xmin = min(selected_values)
    xmax = max(selected_values)
    return xmin, xmax


def get_min_rotated_rect_side(poly):
    """
    Compute the minimum side length of the minimum bounding rotated rectangle of a polygon.
    """
    poly = np.array(poly).astype(np.float32)
    if len(poly) < 3:
        return 0
    rect = cv2.minAreaRect(poly)
    width, height = rect[1]
    return min(width, height)


def get_min_quad_side(quad):
    """
    Compute the minimum side length of a quadrilateral.
    """
    if len(quad) != 4:
        return 0
    quad = np.array(quad)
    sides = []
    for i in range(4):
        side = np.linalg.norm(quad[i] - quad[(i + 1) % 4])
        sides.append(side)
    return min(sides) if sides else 0


def clip_poly_to_rect(poly, x, y, w, h):
    """
    Clip a polygon to a rectangular region and return the clipped quadrilateral.

    Args:
        poly: Original polygon vertices [[x1, y1], [x2, y2], ...]
        x, y, w, h: Position and size of the clipping rectangle

    Returns:
        Clipped quadrilateral vertices, or None if the result is invalid.
    """
    try:
        # Create polygon and clipping rectangle
        poly_shape = Polygon(poly)
        crop_rect = shapely_box(x, y, x + w, y + h)

        # Compute intersection
        clipped = intersection(poly_shape, crop_rect)

        # No intersection or empty
        if clipped.is_empty:
            return None

        # Get intersection coordinates
        if clipped.geom_type == "Polygon":
            coords = list(clipped.exterior.coords[:-1])  # Remove duplicate last point
        elif clipped.geom_type == "MultiPolygon":
            # Multiple polygons, select the largest by area
            largest = max(clipped.geoms, key=lambda p: p.area)
            coords = list(largest.exterior.coords[:-1])
        elif clipped.geom_type == "GeometryCollection":
            # Extract polygons from geometry collection
            polygons = [g for g in clipped.geoms if g.geom_type == "Polygon"]
            if not polygons:
                return None
            largest = max(polygons, key=lambda p: p.area)
            coords = list(largest.exterior.coords[:-1])
        else:
            return None

        # Less than 3 points, invalid
        if len(coords) <= 3:
            return None

        # Convert to numpy array
        coords = np.array(coords)

        # Exactly 4 points, return directly
        if len(coords) == 4:
            return coords

        # More than 4 points, use Douglas-Peucker to simplify to quadrilateral
        # Output points are a subset of original coords (no out-of-bounds), IoU~0.99
        if len(coords) > 4:
            poly_cv = coords.reshape(-1, 1, 2).astype(np.float32)
            peri = cv2.arcLength(poly_cv, True)
            if peri < 1e-6:
                return None
            lo, hi = 0.0, 0.5
            best = None
            for _ in range(50):
                mid = (lo + hi) / 2
                approx = cv2.approxPolyDP(poly_cv, mid * peri, True)
                if len(approx) <= 4:
                    best = approx
                    hi = mid
                else:
                    lo = mid
            if best is not None and len(best) >= 3:
                return best.reshape(-1, 2)
            return None

        return coords
    except Exception as e:
        return None


class RandomCrop(object):
    def __init__(
        self,
        size=(640, 640),
        max_tries=10,
        min_crop_side_ratio=0.1,
        keep_ratio=True,
        **kwargs,
    ):
        self.size = size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio
        self.keep_ratio = keep_ratio

    def __call__(self, data):
        img = data["image"]
        text_polys = data["polys"]
        ignore_tags = data["ignore_tags"]
        texts = data["texts"]

        # Separate care and ignore text boxes
        care_indices = [i for i, tag in enumerate(ignore_tags) if not tag]
        all_care_polys = [text_polys[i] for i in care_indices]

        h, w, _ = img.shape

        # If no valid text boxes, still need to resize and pad the image
        if len(all_care_polys) == 0:
            # Use entire image as crop region, skip crop loop
            crop_x, crop_y, crop_w, crop_h = 0, 0, w, h
            valid_care_data = []
        else:
            # Pre-compute char heights (min side of min bounding rotated rect) for all care boxes
            char_heights = np.array(
                [get_min_rotated_rect_side(poly) for poly in all_care_polys]
            )

            # Try to find a suitable crop region
            valid_care_data = []
            for attempt in range(self.max_tries):
                # Randomly determine crop region width and height
                crop_w_min = min(int(w * self.min_crop_side_ratio), self.size[0])
                crop_w_max = int(self.size[0] * 3)
                crop_w = (
                    w
                    if crop_w_min >= crop_w_max
                    else min(random.randint(crop_w_min, crop_w_max), w)
                )

                crop_h_min = min(int(h * self.min_crop_side_ratio), self.size[1])
                crop_h_max = int(self.size[1] * 3)
                crop_h = (
                    h
                    if crop_h_min >= crop_h_max
                    else min(random.randint(crop_h_min, crop_h_max), h)
                )

                # Randomly determine crop region start position
                crop_x = 0 if crop_w >= w else random.randint(0, w - crop_w)
                crop_y = 0 if crop_h >= h else random.randint(0, h - crop_h)

                # Check each care text box, clip and validate simultaneously (computed once)
                valid_care_data = []
                for care_idx, (poly, char_height) in enumerate(
                    zip(all_care_polys, char_heights)
                ):
                    # Quick check: completely outside, skip
                    if is_poly_outside_rect(poly, crop_x, crop_y, crop_w, crop_h):
                        continue

                    # Completely inside, no clipping needed
                    if is_poly_in_rect(poly, crop_x, crop_y, crop_w, crop_h):
                        valid_care_data.append(
                            (care_idx, None)
                        )  # None means no clipping needed
                        continue

                    # Truncated box, clip and validate (executed once)
                    clipped_poly = clip_poly_to_rect(
                        poly, crop_x, crop_y, crop_w, crop_h
                    )
                    if clipped_poly is None:
                        continue

                    # Validate clipped polygon - area check
                    clipped_area = cv2.contourArea(clipped_poly.astype(np.float32))
                    if clipped_area < 80:
                        continue

                    # Validate - char height check
                    clipped_char_height = get_min_rotated_rect_side(clipped_poly)
                    if clipped_char_height < char_height * 0.35:
                        continue

                    # Validate - min side length check (quadrilaterals only)
                    if len(clipped_poly) == 4:
                        min_side = get_min_quad_side(clipped_poly)
                        if min_side < char_height * 0.35:
                            continue

                    # All validations passed, save the clipped polygon
                    valid_care_data.append((care_idx, clipped_poly))

                # At least one valid text box, use this crop region
                if len(valid_care_data) >= 1:
                    break
            else:
                # All attempts failed, use original region
                crop_x, crop_y, crop_w, crop_h = 0, 0, w, h
                valid_care_data = [(i, None) for i in range(len(all_care_polys))]

        # Crop and scale image
        # Only shrink when crop region is larger than target size, otherwise just pad
        need_resize = crop_w > self.size[0] or crop_h > self.size[1]

        if need_resize:
            # Crop region larger than target, need to shrink
            scale_w = self.size[0] / crop_w
            scale_h = self.size[1] / crop_h
            scale = min(scale_w, scale_h)
            h_resized = int(crop_h * scale)
            w_resized = int(crop_w * scale)
        else:
            # Crop region smaller than or equal to target, no upscaling
            scale = 1.0
            h_resized = crop_h
            w_resized = crop_w

        if self.keep_ratio:
            # Random padding - compute padding size
            pad_h = self.size[1] - h_resized
            pad_w = self.size[0] - w_resized

            # Randomly distribute padding to each side
            pad_top = random.randint(0, pad_h) if pad_h > 0 else 0
            pad_left = random.randint(0, pad_w) if pad_w > 0 else 0

            # Resize cropped image (only when shrinking is needed)
            cropped_img = img[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
            if need_resize:
                resized_img = cv2.resize(cropped_img, (w_resized, h_resized))
            else:
                resized_img = cropped_img

            # Create padded image
            padimg = np.zeros((self.size[1], self.size[0], img.shape[2]), img.dtype)
            padimg[pad_top : pad_top + h_resized, pad_left : pad_left + w_resized] = (
                resized_img
            )
            img = padimg
        else:
            img = cv2.resize(
                img[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w],
                tuple(self.size),
            )
            pad_left = 0
            pad_top = 0

        # Build fast lookup set of valid care indices
        valid_care_indices_set = {care_idx for care_idx, _ in valid_care_data}

        # Build mapping from care_idx to clipped polygon
        care_idx_to_clipped = {
            care_idx: clipped for care_idx, clipped in valid_care_data
        }

        # Build output text box list
        text_polys_crop = []
        ignore_tags_crop = []
        texts_crop = []

        for all_idx, (poly, text, tag) in enumerate(
            zip(text_polys, texts, ignore_tags)
        ):
            if tag:
                # Ignore text box, simple processing
                if not is_poly_outside_rect(poly, crop_x, crop_y, crop_w, crop_h):
                    adjusted_poly = (poly - (crop_x, crop_y)) * scale + (
                        pad_left,
                        pad_top,
                    )
                    adjusted_poly[:, 0] = np.clip(adjusted_poly[:, 0], 0, self.size[0])
                    adjusted_poly[:, 1] = np.clip(adjusted_poly[:, 1], 0, self.size[1])
                    text_polys_crop.append(adjusted_poly.tolist())
                    ignore_tags_crop.append(tag)
                    texts_crop.append(text)
            else:
                # Care text box, find corresponding care_idx
                try:
                    care_idx = care_indices.index(all_idx)
                except ValueError:
                    continue

                # Check if this is a valid text box
                if care_idx not in valid_care_indices_set:
                    continue

                # Get clipped polygon (if any)
                clipped_poly = care_idx_to_clipped[care_idx]

                if clipped_poly is None:
                    # Completely inside, use original polygon
                    adjusted_poly = (poly - (crop_x, crop_y)) * scale + (
                        pad_left,
                        pad_top,
                    )
                else:
                    # Use clipped polygon
                    adjusted_poly = (clipped_poly - (crop_x, crop_y)) * scale + (
                        pad_left,
                        pad_top,
                    )

                text_polys_crop.append(adjusted_poly.tolist())
                ignore_tags_crop.append(tag)
                texts_crop.append(text)
        data["image"] = img
        # Pad polygons to uniform point count to avoid inhomogeneous array error
        if text_polys_crop:
            max_points = max(len(p) for p in text_polys_crop)
            for i, poly in enumerate(text_polys_crop):
                if len(poly) < max_points:
                    text_polys_crop[i] = poly + [poly[-1]] * (max_points - len(poly))
        data["polys"] = np.array(text_polys_crop, dtype=np.float32)
        data["ignore_tags"] = ignore_tags_crop
        data["texts"] = texts_crop
        return data


def crop_area(im, text_polys, min_crop_side_ratio, max_tries):
    h, w, _ = im.shape
    h_array = np.zeros(h, dtype=np.int32)
    w_array = np.zeros(w, dtype=np.int32)
    for points in text_polys:
        points = np.round(points, decimals=0).astype(np.int32)
        minx = np.min(points[:, 0])
        maxx = np.max(points[:, 0])
        w_array[minx:maxx] = 1
        miny = np.min(points[:, 1])
        maxy = np.max(points[:, 1])
        h_array[miny:maxy] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]

    if len(h_axis) == 0 or len(w_axis) == 0:
        return 0, 0, w, h

    h_regions = split_regions(h_axis)
    w_regions = split_regions(w_axis)

    for i in range(max_tries):
        if len(w_regions) > 1:
            xmin, xmax = region_wise_random_select(w_regions, w)
        else:
            xmin, xmax = random_select(w_axis, w)
        if len(h_regions) > 1:
            ymin, ymax = region_wise_random_select(h_regions, h)
        else:
            ymin, ymax = random_select(h_axis, h)

        if (
            xmax - xmin < min_crop_side_ratio * w
            or ymax - ymin < min_crop_side_ratio * h
        ):
            # area too small
            continue
        num_poly_in_rect = 0
        for poly in text_polys:
            if not is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                num_poly_in_rect += 1
                break

        if num_poly_in_rect > 0:
            return xmin, ymin, xmax - xmin, ymax - ymin

    return 0, 0, w, h


class EastRandomCropData(object):
    def __init__(
        self,
        size=(640, 640),
        max_tries=10,
        min_crop_side_ratio=0.1,
        keep_ratio=True,
        **kwargs,
    ):
        self.size = size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio
        self.keep_ratio = keep_ratio

    def __call__(self, data):
        img = data["image"]
        text_polys = data["polys"]
        ignore_tags = data["ignore_tags"]
        texts = data["texts"]
        all_care_polys = [text_polys[i] for i, tag in enumerate(ignore_tags) if not tag]
        # 计算crop区域
        crop_x, crop_y, crop_w, crop_h = crop_area(
            img, all_care_polys, self.min_crop_side_ratio, self.max_tries
        )
        # crop 图片 保持比例填充
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        if self.keep_ratio:
            padimg = np.zeros((self.size[1], self.size[0], img.shape[2]), img.dtype)
            padimg[:h, :w] = cv2.resize(
                img[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w], (w, h)
            )
            img = padimg
        else:
            img = cv2.resize(
                img[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w],
                tuple(self.size),
            )
        # crop 文本框
        text_polys_crop = []
        ignore_tags_crop = []
        texts_crop = []
        for poly, text, tag in zip(text_polys, texts, ignore_tags):
            poly = ((poly - (crop_x, crop_y)) * scale).tolist()
            if not is_poly_outside_rect(poly, 0, 0, w, h):
                text_polys_crop.append(poly)
                ignore_tags_crop.append(tag)
                texts_crop.append(text)
        data["image"] = img
        data["polys"] = np.array(text_polys_crop)
        if "iluvatar_gpu" in get_device():
            data["polys"] = np.array(text_polys_crop).astype(np.float32)
        data["ignore_tags"] = ignore_tags_crop
        data["texts"] = texts_crop
        return data


class RandomCropImgMask(object):
    def __init__(self, size, main_key, crop_keys, p=3 / 8, **kwargs):
        self.size = size
        self.main_key = main_key
        self.crop_keys = crop_keys
        self.p = p

    def __call__(self, data):
        image = data["image"]

        h, w = image.shape[0:2]
        th, tw = self.size
        if w == tw and h == th:
            return data

        mask = data[self.main_key]
        if np.max(mask) > 0 and random.random() > self.p:
            # make sure to crop the text region
            tl = np.min(np.where(mask > 0), axis=1) - (th, tw)
            tl[tl < 0] = 0
            br = np.max(np.where(mask > 0), axis=1) - (th, tw)
            br[br < 0] = 0

            br[0] = min(br[0], h - th)
            br[1] = min(br[1], w - tw)

            i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
            j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
        else:
            i = random.randint(0, h - th) if h - th > 0 else 0
            j = random.randint(0, w - tw) if w - tw > 0 else 0

        # return i, j, th, tw
        for k in data:
            if k in self.crop_keys:
                if len(data[k].shape) == 3:
                    if np.argmin(data[k].shape) == 0:
                        img = data[k][:, i : i + th, j : j + tw]
                        if img.shape[1] != img.shape[2]:
                            a = 1
                    elif np.argmin(data[k].shape) == 2:
                        img = data[k][i : i + th, j : j + tw, :]
                        if img.shape[1] != img.shape[0]:
                            a = 1
                    else:
                        img = data[k]
                else:
                    img = data[k][i : i + th, j : j + tw]
                    if img.shape[0] != img.shape[1]:
                        a = 1
                data[k] = img
        return data
