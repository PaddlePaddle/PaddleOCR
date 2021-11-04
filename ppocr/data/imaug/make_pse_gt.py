# -*- coding:utf-8 -*- 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

__all__ = ['MakePseGt']


class MakePseGt(object):
    r'''
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''

    def __init__(self, kernel_num=7, size=640, min_shrink_ratio=0.4, **kwargs):
        self.kernel_num = kernel_num
        self.min_shrink_ratio = min_shrink_ratio
        self.size = size

    def __call__(self, data):

        image = data['image']
        text_polys = data['polys']
        ignore_tags = data['ignore_tags']

        h, w, _ = image.shape
        short_edge = min(h, w)
        if short_edge < self.size:
            # keep short_size >= self.size
            scale = self.size / short_edge
            image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
            text_polys *= scale

        gt_kernels = []
        for i in range(1, self.kernel_num + 1):
            # s1->sn, from big to small
            rate = 1.0 - (1.0 - self.min_shrink_ratio) / (self.kernel_num - 1
                                                          ) * i
            text_kernel, ignore_tags = self.generate_kernel(
                image.shape[0:2], rate, text_polys, ignore_tags)
            gt_kernels.append(text_kernel)

        training_mask = np.ones(image.shape[0:2], dtype='uint8')
        for i in range(text_polys.shape[0]):
            if ignore_tags[i]:
                cv2.fillPoly(training_mask,
                             text_polys[i].astype(np.int32)[np.newaxis, :, :],
                             0)

        gt_kernels = np.array(gt_kernels)
        gt_kernels[gt_kernels > 0] = 1

        data['image'] = image
        data['polys'] = text_polys
        data['gt_kernels'] = gt_kernels[0:]
        data['gt_text'] = gt_kernels[0]
        data['mask'] = training_mask.astype('float32')
        return data

    def generate_kernel(self,
                        img_size,
                        shrink_ratio,
                        text_polys,
                        ignore_tags=None):
        """
        Refer to part of the code:
        https://github.com/open-mmlab/mmocr/blob/main/mmocr/datasets/pipelines/textdet_targets/base_textdet_targets.py
        """

        h, w = img_size
        text_kernel = np.zeros((h, w), dtype=np.float32)
        for i, poly in enumerate(text_polys):
            polygon = Polygon(poly)
            distance = polygon.area * (1 - shrink_ratio * shrink_ratio) / (
                polygon.length + 1e-6)
            subject = [tuple(l) for l in poly]
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            shrinked = np.array(pco.Execute(-distance))

            if len(shrinked) == 0 or shrinked.size == 0:
                if ignore_tags is not None:
                    ignore_tags[i] = True
                continue
            try:
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
            except:
                if ignore_tags is not None:
                    ignore_tags[i] = True
                continue
            cv2.fillPoly(text_kernel, [shrinked.astype(np.int32)], i + 1)
        return text_kernel, ignore_tags
