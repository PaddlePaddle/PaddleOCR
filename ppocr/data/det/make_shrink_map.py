# -*- coding:utf-8 -*- 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper


def validate_polygons(polygons, ignore_tags, h, w):
    '''
    polygons (numpy.array, required): of shape (num_instances, num_points, 2)
    '''
    if len(polygons) == 0:
        return polygons, ignore_tags
    assert len(polygons) == len(ignore_tags)
    for polygon in polygons:
        polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

    for i in range(len(polygons)):
        area = polygon_area(polygons[i])
        if abs(area) < 1:
            ignore_tags[i] = True
        if area > 0:
            polygons[i] = polygons[i][::-1, :]
    return polygons, ignore_tags


def polygon_area(polygon):
    edge = 0
    for i in range(polygon.shape[0]):
        next_index = (i + 1) % polygon.shape[0]
        edge += (polygon[next_index, 0] - polygon[i, 0]) * (
            polygon[next_index, 1] - polygon[i, 1])

    return edge / 2.


def MakeShrinkMap(data):
    min_text_size = 8
    shrink_ratio = 0.4

    image = data['image']
    text_polys = data['polys']
    ignore_tags = data['ignore_tags']

    h, w = image.shape[:2]
    text_polys, ignore_tags = validate_polygons(text_polys, ignore_tags, h, w)
    gt = np.zeros((h, w), dtype=np.float32)
    # gt = np.zeros((1, h, w), dtype=np.float32)
    mask = np.ones((h, w), dtype=np.float32)
    for i in range(len(text_polys)):
        polygon = text_polys[i]
        height = max(polygon[:, 1]) - min(polygon[:, 1])
        width = max(polygon[:, 0]) - min(polygon[:, 0])
        # height = min(np.linalg.norm(polygon[0] - polygon[3]),
        #             np.linalg.norm(polygon[1] - polygon[2]))
        # width = min(np.linalg.norm(polygon[0] - polygon[1]),
        #             np.linalg.norm(polygon[2] - polygon[3]))
        if ignore_tags[i] or min(height, width) < min_text_size:
            cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
            ignore_tags[i] = True
        else:
            polygon_shape = Polygon(polygon)
            distance = polygon_shape.area * (
                1 - np.power(shrink_ratio, 2)) / polygon_shape.length
            subject = [tuple(l) for l in text_polys[i]]
            padding = pyclipper.PyclipperOffset()
            padding.AddPath(subject, pyclipper.JT_ROUND,
                            pyclipper.ET_CLOSEDPOLYGON)
            shrinked = padding.Execute(-distance)
            if shrinked == []:
                cv2.fillPoly(mask,
                             polygon.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
                continue
            shrinked = np.array(shrinked[0]).reshape(-1, 2)
            cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
            # cv2.fillPoly(gt[0], [shrinked.astype(np.int32)], 1)

    data['shrink_map'] = gt
    data['shrink_mask'] = mask
    return data
