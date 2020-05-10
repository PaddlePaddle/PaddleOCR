# -*- coding:utf-8 -*- 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import random
import cv2
import math

import imgaug
import imgaug.augmenters as iaa


def AugmentData(data):
    img = data['image']
    shape = img.shape

    aug = iaa.Sequential(
        [iaa.Fliplr(0.5), iaa.Affine(rotate=(-10, 10)), iaa.Resize(
            (0.5, 3))]).to_deterministic()

    def may_augment_annotation(aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for poly in data['polys']:
            new_poly = may_augment_poly(aug, shape, poly)
            line_polys.append(new_poly)
        data['polys'] = np.array(line_polys)
        return data

    def may_augment_poly(aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(
                keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly

    img_aug = aug.augment_image(img)
    data['image'] = img_aug
    data = may_augment_annotation(aug, data, shape)
    return data
