# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
import random


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

        if xmax - xmin < min_crop_side_ratio * w or ymax - ymin < min_crop_side_ratio * h:
            # area too small
            continue
        num_poly_in_rect = 0
        for poly in text_polys:
            if not is_poly_outside_rect(poly, xmin, ymin, xmax - xmin,
                                        ymax - ymin):
                num_poly_in_rect += 1
                break

        if num_poly_in_rect > 0:
            return xmin, ymin, xmax - xmin, ymax - ymin

    return 0, 0, w, h


def RandomCropData(data, size):
    max_tries = 10
    min_crop_side_ratio = 0.1
    require_original_image = False
    keep_ratio = True

    im = data['image']
    text_polys = data['polys']
    ignore_tags = data['ignore_tags']
    texts = data['texts']
    all_care_polys = [
        text_polys[i] for i, tag in enumerate(ignore_tags) if not tag
    ]
    crop_x, crop_y, crop_w, crop_h = crop_area(im, all_care_polys,
                                               min_crop_side_ratio, max_tries)
    dh, dw = size
    scale_w = dw / crop_w
    scale_h = dh / crop_h
    scale = min(scale_w, scale_h)
    h = int(crop_h * scale)
    w = int(crop_w * scale)
    if keep_ratio:
        padimg = np.zeros((dh, dw, im.shape[2]), im.dtype)
        padimg[:h, :w] = cv2.resize(
            im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
        img = padimg
    else:
        img = cv2.resize(im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w],
                         (dw, dh))
    text_polys_crop = []
    ignore_tags_crop = []
    texts_crop = []
    for poly, text, tag in zip(text_polys, texts, ignore_tags):
        poly = ((poly - (crop_x, crop_y)) * scale).tolist()
        if not is_poly_outside_rect(poly, 0, 0, w, h):
            text_polys_crop.append(poly)
            ignore_tags_crop.append(tag)
            texts_crop.append(text)
    data['image'] = img
    data['polys'] = np.array(text_polys_crop)
    data['ignore_tags'] = ignore_tags_crop
    data['texts'] = texts_crop
    return data
