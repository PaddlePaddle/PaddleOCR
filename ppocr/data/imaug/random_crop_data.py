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
    计算多边形的最小外接旋转矩形的最小边长
    """
    poly = np.array(poly).astype(np.float32)
    if len(poly) < 3:
        return 0
    rect = cv2.minAreaRect(poly)
    width, height = rect[1]
    return min(width, height)


def get_min_quad_side(quad):
    """
    计算四边形的最小边长
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
    将多边形裁剪到矩形区域内，返回裁剪后的四边形

    Args:
        poly: 原始多边形顶点 [[x1, y1], [x2, y2], ...]
        x, y, w, h: 裁剪矩形的位置和大小

    Returns:
        裁剪后的四边形顶点，如果裁剪后无效则返回None
    """
    try:
        # 创建多边形和裁剪矩形
        poly_shape = Polygon(poly)
        crop_rect = shapely_box(x, y, x + w, y + h)

        # 计算交集
        clipped = intersection(poly_shape, crop_rect)

        # 如果没有交集或交集为空
        if clipped.is_empty:
            return None

        # 获取交集的坐标
        if clipped.geom_type == "Polygon":
            coords = list(clipped.exterior.coords[:-1])  # 去掉重复的最后一个点
        elif clipped.geom_type == "MultiPolygon":
            # 如果是多个多边形，选择面积最大的
            largest = max(clipped.geoms, key=lambda p: p.area)
            coords = list(largest.exterior.coords[:-1])
        elif clipped.geom_type == "GeometryCollection":
            # 从几何集合中提取多边形
            polygons = [g for g in clipped.geoms if g.geom_type == "Polygon"]
            if not polygons:
                return None
            largest = max(polygons, key=lambda p: p.area)
            coords = list(largest.exterior.coords[:-1])
        else:
            return None

        # 如果点少于3个，无效
        if len(coords) <= 3:
            return None

        # 转换为numpy数组
        coords = np.array(coords)

        # 如果点数等于4，直接返回
        if len(coords) == 4:
            return coords

        # 如果点数大于4，选择原始顶点和边上的交点来构成四边形
        if len(coords) > 4:
            from itertools import combinations
            from shapely.geometry import LineString

            # 创建包含所有点的多边形
            all_polygon = Polygon(coords)
            # 使用凸包近似为四边形
            convex_hull = all_polygon.convex_hull
            # 获取凸包的顶点
            hull_coords = list(convex_hull.exterior.coords)
            hull_coords.pop()  # 删除最后一个重复的起点

            if len(hull_coords) > 4:
                # 如果凸包顶点超过4个，选择距离最远的四个点
                max_distance = 0
                best_quad = None
                for quad in combinations(hull_coords, 4):
                    distance = sum(
                        [
                            LineString([quad[i], quad[(i + 1) % 4]]).length
                            for i in range(4)
                        ]
                    )
                    if distance > max_distance:
                        max_distance = distance
                        best_quad = quad
                quadrilateral = best_quad
            else:
                quadrilateral = hull_coords

            return np.array(quadrilateral)

        return coords
    except Exception as e:
        return None


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

        # 分离 care 和 ignore 的文本框
        care_indices = [i for i, tag in enumerate(ignore_tags) if not tag]
        all_care_polys = [text_polys[i] for i in care_indices]

        h, w, _ = img.shape

        # 如果没有有效文本框，仍需要对图像进行 resize 和 padding
        if len(all_care_polys) == 0:
            # 使用整个图像作为裁剪区域，跳过裁剪循环直接进行 resize 和 padding
            crop_x, crop_y, crop_w, crop_h = 0, 0, w, h
            valid_care_data = []
        else:
            # 预先计算所有 care 文本框的字符高度（最小外接旋转矩形的最小边）
            char_heights = np.array(
                [get_min_rotated_rect_side(poly) for poly in all_care_polys]
            )

            # 尝试找到合适的裁剪区域
            valid_care_data = []
            for attempt in range(self.max_tries):
                # 随机确定裁剪区域的宽度和高度
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

                # 随机确定裁剪区域的起始位置
                crop_x = 0 if crop_w >= w else random.randint(0, w - crop_w)
                crop_y = 0 if crop_h >= h else random.randint(0, h - crop_h)

                # 检查每个 care 文本框，同时进行裁剪和验证（只计算一次）
                valid_care_data = []
                for care_idx, (poly, char_height) in enumerate(
                    zip(all_care_polys, char_heights)
                ):
                    # 快速判断：如果完全在外部，跳过
                    if is_poly_outside_rect(poly, crop_x, crop_y, crop_w, crop_h):
                        continue

                    # 如果完全在内部，无需裁剪
                    if is_poly_in_rect(poly, crop_x, crop_y, crop_w, crop_h):
                        valid_care_data.append((care_idx, None))  # None 表示不需要裁剪
                        continue

                    # 被截断的框，裁剪并验证（只执行一次）
                    clipped_poly = clip_poly_to_rect(
                        poly, crop_x, crop_y, crop_w, crop_h
                    )
                    if clipped_poly is None:
                        continue

                    # 验证裁剪后的多边形 - 面积检查
                    clipped_area = cv2.contourArea(clipped_poly.astype(np.float32))
                    if clipped_area < 80:
                        continue

                    # 验证 - 字符高度检查
                    clipped_char_height = get_min_rotated_rect_side(clipped_poly)
                    if clipped_char_height < char_height * 0.35:
                        continue

                    # 验证 - 最小边长检查（仅针对四边形）
                    if len(clipped_poly) == 4:
                        min_side = get_min_quad_side(clipped_poly)
                        if min_side < char_height * 0.35:
                            continue

                    # 所有验证通过，保存裁剪后的多边形
                    valid_care_data.append((care_idx, clipped_poly))

                # 如果至少有一个有效的文本框，使用这个裁剪区域
                if len(valid_care_data) >= 1:
                    break
            else:
                # 所有尝试都失败，使用原始区域
                crop_x, crop_y, crop_w, crop_h = 0, 0, w, h
                valid_care_data = [(i, None) for i in range(len(all_care_polys))]

        # 裁剪并缩放图像
        # 只有当 crop 区域大于 size 时才缩小，否则只做 padding
        need_resize = crop_w > self.size[0] or crop_h > self.size[1]

        if need_resize:
            # crop 区域大于 size，需要缩小
            scale_w = self.size[0] / crop_w
            scale_h = self.size[1] / crop_h
            scale = min(scale_w, scale_h)
            h_resized = int(crop_h * scale)
            w_resized = int(crop_w * scale)
        else:
            # crop 区域小于等于 size，不放大
            scale = 1.0
            h_resized = crop_h
            w_resized = crop_w

        if self.keep_ratio:
            # 随机 padding - 计算需要 pad 的大小
            pad_h = self.size[1] - h_resized
            pad_w = self.size[0] - w_resized

            # 随机分配 padding 到各边
            pad_top = random.randint(0, pad_h) if pad_h > 0 else 0
            pad_left = random.randint(0, pad_w) if pad_w > 0 else 0

            # Resize 裁剪后的图像（仅在需要缩小时）
            cropped_img = img[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
            if need_resize:
                resized_img = cv2.resize(cropped_img, (w_resized, h_resized))
            else:
                resized_img = cropped_img

            # 创建 padding 后的图像
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

        # 构建有效 care 索引的快速查找集合
        valid_care_indices_set = {care_idx for care_idx, _ in valid_care_data}

        # 构建 care_idx 到裁剪后多边形的映射
        care_idx_to_clipped = {
            care_idx: clipped for care_idx, clipped in valid_care_data
        }

        # 构建输出的文本框列表
        text_polys_crop = []
        ignore_tags_crop = []
        texts_crop = []

        for all_idx, (poly, text, tag) in enumerate(
            zip(text_polys, texts, ignore_tags)
        ):
            if tag:
                # ignore 文本框，简单处理
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
                # care 文本框，查找对应的 care_idx
                try:
                    care_idx = care_indices.index(all_idx)
                except ValueError:
                    continue

                # 检查是否是有效的文本框
                if care_idx not in valid_care_indices_set:
                    continue

                # 获取裁剪后的多边形（如果有）
                clipped_poly = care_idx_to_clipped[care_idx]

                if clipped_poly is None:
                    # 完全在内部，使用原始多边形
                    adjusted_poly = (poly - (crop_x, crop_y)) * scale + (
                        pad_left,
                        pad_top,
                    )
                else:
                    # 使用裁剪后的多边形
                    adjusted_poly = (clipped_poly - (crop_x, crop_y)) * scale + (
                        pad_left,
                        pad_top,
                    )

                text_polys_crop.append(adjusted_poly.tolist())
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
