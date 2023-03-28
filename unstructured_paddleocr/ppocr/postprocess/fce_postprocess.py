# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
https://github.com/open-mmlab/mmocr/blob/v0.3.0/mmocr/models/textdet/postprocess/wrapper.py
"""

import cv2
import paddle
import numpy as np
from numpy.fft import ifft
from ppocr.utils.poly_nms import poly_nms, valid_boundary


def fill_hole(input_mask):
    h, w = input_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()

    mask = np.zeros((h + 4, w + 4), np.uint8)

    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)

    return ~canvas | input_mask


def fourier2poly(fourier_coeff, num_reconstr_points=50):
    """ Inverse Fourier transform
        Args:
            fourier_coeff (ndarray): Fourier coefficients shaped (n, 2k+1),
                with n and k being candidates number and Fourier degree
                respectively.
            num_reconstr_points (int): Number of reconstructed polygon points.
        Returns:
            Polygons (ndarray): The reconstructed polygons shaped (n, n')
        """

    a = np.zeros((len(fourier_coeff), num_reconstr_points), dtype='complex')
    k = (len(fourier_coeff[0]) - 1) // 2

    a[:, 0:k + 1] = fourier_coeff[:, k:]
    a[:, -k:] = fourier_coeff[:, :k]

    poly_complex = ifft(a) * num_reconstr_points
    polygon = np.zeros((len(fourier_coeff), num_reconstr_points, 2))
    polygon[:, :, 0] = poly_complex.real
    polygon[:, :, 1] = poly_complex.imag
    return polygon.astype('int32').reshape((len(fourier_coeff), -1))


class FCEPostProcess(object):
    """
    The post process for FCENet.
    """

    def __init__(self,
                 scales,
                 fourier_degree=5,
                 num_reconstr_points=50,
                 decoding_type='fcenet',
                 score_thr=0.3,
                 nms_thr=0.1,
                 alpha=1.0,
                 beta=1.0,
                 box_type='poly',
                 **kwargs):

        self.scales = scales
        self.fourier_degree = fourier_degree
        self.num_reconstr_points = num_reconstr_points
        self.decoding_type = decoding_type
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.alpha = alpha
        self.beta = beta
        self.box_type = box_type

    def __call__(self, preds, shape_list):
        score_maps = []
        for key, value in preds.items():
            if isinstance(value, paddle.Tensor):
                value = value.numpy()
            cls_res = value[:, :4, :, :]
            reg_res = value[:, 4:, :, :]
            score_maps.append([cls_res, reg_res])

        return self.get_boundary(score_maps, shape_list)

    def resize_boundary(self, boundaries, scale_factor):
        """Rescale boundaries via scale_factor.

        Args:
            boundaries (list[list[float]]): The boundary list. Each boundary
            with size 2k+1 with k>=4.
            scale_factor(ndarray): The scale factor of size (4,).

        Returns:
            boundaries (list[list[float]]): The scaled boundaries.
        """
        boxes = []
        scores = []
        for b in boundaries:
            sz = len(b)
            valid_boundary(b, True)
            scores.append(b[-1])
            b = (np.array(b[:sz - 1]) *
                 (np.tile(scale_factor[:2], int(
                     (sz - 1) / 2)).reshape(1, sz - 1))).flatten().tolist()
            boxes.append(np.array(b).reshape([-1, 2]))

        return np.array(boxes, dtype=np.float32), scores

    def get_boundary(self, score_maps, shape_list):
        assert len(score_maps) == len(self.scales)
        boundaries = []
        for idx, score_map in enumerate(score_maps):
            scale = self.scales[idx]
            boundaries = boundaries + self._get_boundary_single(score_map,
                                                                scale)

        # nms
        boundaries = poly_nms(boundaries, self.nms_thr)
        boundaries, scores = self.resize_boundary(
            boundaries, (1 / shape_list[0, 2:]).tolist()[::-1])

        boxes_batch = [dict(points=boundaries, scores=scores)]
        return boxes_batch

    def _get_boundary_single(self, score_map, scale):
        assert len(score_map) == 2
        assert score_map[1].shape[1] == 4 * self.fourier_degree + 2

        return self.fcenet_decode(
            preds=score_map,
            fourier_degree=self.fourier_degree,
            num_reconstr_points=self.num_reconstr_points,
            scale=scale,
            alpha=self.alpha,
            beta=self.beta,
            box_type=self.box_type,
            score_thr=self.score_thr,
            nms_thr=self.nms_thr)

    def fcenet_decode(self,
                      preds,
                      fourier_degree,
                      num_reconstr_points,
                      scale,
                      alpha=1.0,
                      beta=2.0,
                      box_type='poly',
                      score_thr=0.3,
                      nms_thr=0.1):
        """Decoding predictions of FCENet to instances.

        Args:
            preds (list(Tensor)): The head output tensors.
            fourier_degree (int): The maximum Fourier transform degree k.
            num_reconstr_points (int): The points number of the polygon
                reconstructed from predicted Fourier coefficients.
            scale (int): The down-sample scale of the prediction.
            alpha (float) : The parameter to calculate final scores. Score_{final}
                    = (Score_{text region} ^ alpha)
                    * (Score_{text center region}^ beta)
            beta (float) : The parameter to calculate final score.
            box_type (str):  Boundary encoding type 'poly' or 'quad'.
            score_thr (float) : The threshold used to filter out the final
                candidates.
            nms_thr (float) :  The threshold of nms.

        Returns:
            boundaries (list[list[float]]): The instance boundary and confidence
                list.
        """
        assert isinstance(preds, list)
        assert len(preds) == 2
        assert box_type in ['poly', 'quad']

        cls_pred = preds[0][0]
        tr_pred = cls_pred[0:2]
        tcl_pred = cls_pred[2:]

        reg_pred = preds[1][0].transpose([1, 2, 0])
        x_pred = reg_pred[:, :, :2 * fourier_degree + 1]
        y_pred = reg_pred[:, :, 2 * fourier_degree + 1:]

        score_pred = (tr_pred[1]**alpha) * (tcl_pred[1]**beta)
        tr_pred_mask = (score_pred) > score_thr
        tr_mask = fill_hole(tr_pred_mask)

        tr_contours, _ = cv2.findContours(
            tr_mask.astype(np.uint8), cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)  # opencv4

        mask = np.zeros_like(tr_mask)
        boundaries = []
        for cont in tr_contours:
            deal_map = mask.copy().astype(np.int8)
            cv2.drawContours(deal_map, [cont], -1, 1, -1)

            score_map = score_pred * deal_map
            score_mask = score_map > 0
            xy_text = np.argwhere(score_mask)
            dxy = xy_text[:, 1] + xy_text[:, 0] * 1j

            x, y = x_pred[score_mask], y_pred[score_mask]
            c = x + y * 1j
            c[:, fourier_degree] = c[:, fourier_degree] + dxy
            c *= scale

            polygons = fourier2poly(c, num_reconstr_points)
            score = score_map[score_mask].reshape(-1, 1)
            polygons = poly_nms(np.hstack((polygons, score)).tolist(), nms_thr)

            boundaries = boundaries + polygons

        boundaries = poly_nms(boundaries, nms_thr)

        if box_type == 'quad':
            new_boundaries = []
            for boundary in boundaries:
                poly = np.array(boundary[:-1]).reshape(-1, 2).astype(np.float32)
                score = boundary[-1]
                points = cv2.boxPoints(cv2.minAreaRect(poly))
                points = np.int0(points)
                new_boundaries.append(points.reshape(-1).tolist() + [score])
                boundaries = new_boundaries

        return boundaries
