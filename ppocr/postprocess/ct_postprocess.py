# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is refered from:
https://github.com/WenmuZhou/DBNet.pytorch/blob/master/post_processing/seg_detector_representer.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.nn.functional as F
import os
import os.path as osp
import numpy as np
import cv2
import paddle
from shapely.geometry import Polygon
import pyclipper


class ResultFormat(object):
    def __init__(self, data_type, result_path):
        self.data_type = data_type
        self.result_path = result_path

        if osp.isfile(result_path):
            os.remove(result_path)

        if result_path.endswith('.zip'):
            result_path = result_path.replace('.zip', '')

        if not osp.exists(result_path):
            os.makedirs(result_path)

    def write_result(self, img_name, outputs):
        if 'IC15' in self.data_type:
            self._write_result_ic15(img_name, outputs)
        elif 'TT' in self.data_type:
            self._write_result_tt(img_name, outputs)
        elif 'CTW' in self.data_type:
            self._write_result_ctw(img_name, outputs)
        elif 'MSRA' in self.data_type:
            self._write_result_msra(img_name, outputs)

    def _write_result_ic15(self, img_name, outputs):
        assert self.result_path.endswith(
            '.zip'), 'Error: ic15 result should be a zip file!'

        tmp_folder = self.result_path.replace('.zip', '')

        bboxes = outputs['bboxes']

        lines = []
        for i, bbox in enumerate(bboxes):
            values = [int(v) for v in bbox]
            line = "%d,%d,%d,%d,%d,%d,%d,%d\n" % tuple(values)
            lines.append(line)

        file_name = 'res_%s.txt' % img_name
        file_path = osp.join(tmp_folder, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)

        z = zipfile.ZipFile(self.result_path, 'a', zipfile.ZIP_DEFLATED)
        z.write(file_path, file_name)
        z.close()

    def _write_result_tt(self, image_name, outputs):
        bboxes = outputs['bboxes']

        lines = []
        for i, bbox in enumerate(bboxes):
            bbox = bbox.reshape(-1, 2)[:, ::-1].reshape(-1)
            values = [int(v) for v in bbox]
            line = "%d" % values[0]
            for v_id in range(1, len(values)):
                line += ",%d" % values[v_id]
            line += '\n'
            lines.append(line)

        file_name = '%s.txt' % image_name
        file_path = osp.join(self.result_path, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)

    def _write_result_ctw(self, image_name, outputs):
        bboxes = outputs['bboxes']

        lines = []
        for i, bbox in enumerate(bboxes):
            bbox = bbox.reshape(-1, 2)[:, ::-1].reshape(-1)
            values = [int(v) for v in bbox]
            line = "%d" % values[0]
            for v_id in range(1, len(values)):
                line += ",%d" % values[v_id]
            line += '\n'
            lines.append(line)

        file_name = '%s.txt' % image_name
        file_path = osp.join(self.result_path, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)

    def _write_result_msra(self, image_name, outputs):
        bboxes = outputs['bboxes']

        lines = []
        for b_idx, bbox in enumerate(bboxes):
            values = [int(v) for v in bbox]
            line = "%d" % values[0]
            for v_id in range(1, len(values)):
                line += ", %d" % values[v_id]
            line += '\n'
            lines.append(line)

        file_name = '%s.txt' % image_name
        file_path = osp.join(self.result_path, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)


class CTPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self, **kwargs):
        self.coord = np.zeros((2, 300, 300), dtype=np.int32)
        for i in range(300):
            for j in range(300):
                self.coord[0, i, j] = j
                self.coord[1, i, j] = i

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def __call__(self, outs, batch):
        imgs = batch[0]
        out = self._upsample(outs, imgs.shape, 4)

        img_meta = batch[1]
        img_path = batch[2][0]

        min_score = 0.88  #最好可配置
        min_area = 16
        bbox_type = 'poly'
        #print(list(img_meta.keys()))

        outputs = dict()

        score = F.sigmoid(out[:, 0, :, :])
        kernel = out[:, 0, :, :] > 0.2
        loc = paddle.cast(out[:, 1:, :, :], "float32")

        score = score.numpy()[0].astype(np.float32)
        kernel = kernel.numpy()[0].astype(np.uint8)
        loc = loc.numpy()[0].astype(np.float32)

        label_num, label_kernel = cv2.connectedComponents(
            kernel, connectivity=4)
        for i in range(1, label_num):
            ind = (label_kernel == i)
            if ind.sum() < 10:
                label_kernel[ind] = 0

        label = np.zeros_like(label_kernel)
        h, w = label_kernel.shape
        pixels = self.coord[:, :h, :w].reshape(2, -1)
        points = pixels.transpose([1, 0]).astype(np.float32)

        off_points = (
            points + 10. / 4. * loc[:, pixels[1], pixels[0]].T).astype(np.int32)
        off_points[:, 0] = np.clip(off_points[:, 0], 0, label.shape[1] - 1)
        off_points[:, 1] = np.clip(off_points[:, 1], 0, label.shape[0] - 1)
        label[pixels[1], pixels[0]] = label_kernel[off_points[:, 1],
                                                   off_points[:, 0]]
        label[label_kernel > 0] = label_kernel[label_kernel > 0]

        score_pocket = [0.0]
        for i in range(1, label_num):
            # ind = ((label_kernel == i) & (label == i))
            ind = (label_kernel == i)
            if ind.sum() == 0:
                score_pocket.append(0.0)
                continue
            score_i = np.mean(score[ind])
            score_pocket.append(score_i)

        # image size
        org_img_size = img_meta['org_img_size'][0]
        img_size = img_meta['img_size'][0]

        label_num = np.max(label) + 1
        label = cv2.resize(
            label, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)

        scale = (float(org_img_size[1]) / float(img_size[1]),
                 float(org_img_size[0]) / float(img_size[0]))

        bboxes = []
        scores = []
        for i in range(1, label_num):
            ind = (label == i)
            points = np.array(np.where(ind)).transpose((1, 0))

            if points.shape[0] < min_area:
                continue

            score_i = score_pocket[i]
            if score_i < min_score:
                continue

            if bbox_type == 'rect':
                rect = cv2.minAreaRect(points[:, ::-1])
                bbox = cv2.boxPoints(rect) * scale
                z = bbox.mean(0)
                bbox = z + (bbox - z) * 0.85
            elif bbox_type == 'poly':
                binary = np.zeros(label.shape, dtype='uint8')
                binary[ind] = 1
                try:
                    _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_SIMPLE)
                except BaseException:
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                # contour = contours[0]
                # epsilon = 0.01 * cv2.arcLength(contour, True)
                # approx = cv2.approxPolyDP(contour, epsilon, True)
                # bbox = approx * scale
                bbox = contours[0] * scale

            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))
            scores.append(score_i)

        outputs.update(dict(bboxes=bboxes, scores=scores))

        # self.vis(img_meta, score.copy(), kernel.copy(), label_kernel.copy(), label.copy(), outputs, loc.copy())
        # embed()

        rf = ResultFormat('CT_TT', 'outputs/submit_tt/')
        # save result
        image_name, _ = osp.splitext(osp.basename(img_path))
        rf.write_result(image_name, outputs)

        return outputs
