from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import paddle
from numpy.fft import ifft
import Polygon as plg


def points2polygon(points):
    """Convert k points to 1 polygon.

    Args:
        points (ndarray or list): A ndarray or a list of shape (2k)
            that indicates k points.

    Returns:
        polygon (Polygon): A polygon object.
    """
    if isinstance(points, list):
        points = np.array(points)

    assert isinstance(points, np.ndarray)
    assert (points.size % 2 == 0) and (points.size >= 8)

    point_mat = points.reshape([-1, 2])
    return plg.Polygon(point_mat)


def poly_intersection(poly_det, poly_gt):
    """Calculate the intersection area between two polygon.

    Args:
        poly_det (Polygon): A polygon predicted by detector.
        poly_gt (Polygon): A gt polygon.

    Returns:
        intersection_area (float): The intersection area between two polygons.
    """
    assert isinstance(poly_det, plg.Polygon)
    assert isinstance(poly_gt, plg.Polygon)

    poly_inter = poly_det & poly_gt
    if len(poly_inter) == 0:
        return 0, poly_inter
    return poly_inter.area(), poly_inter


def poly_union(poly_det, poly_gt):
    """Calculate the union area between two polygon.

    Args:
        poly_det (Polygon): A polygon predicted by detector.
        poly_gt (Polygon): A gt polygon.

    Returns:
        union_area (float): The union area between two polygons.
    """
    assert isinstance(poly_det, plg.Polygon)
    assert isinstance(poly_gt, plg.Polygon)

    area_det = poly_det.area()
    area_gt = poly_gt.area()
    area_inters, _ = poly_intersection(poly_det, poly_gt)
    return area_det + area_gt - area_inters


def valid_boundary(x, with_score=True):
    num = len(x)
    if num < 8:
        return False
    if num % 2 == 0 and (not with_score):
        return True
    if num % 2 == 1 and with_score:
        return True

    return False


def boundary_iou(src, target):
    """Calculate the IOU between two boundaries.

    Args:
       src (list): Source boundary.
       target (list): Target boundary.

    Returns:
       iou (float): The iou between two boundaries.
    """
    assert valid_boundary(src, False)
    assert valid_boundary(target, False)
    src_poly = points2polygon(src)
    target_poly = points2polygon(target)

    return poly_iou(src_poly, target_poly)


def poly_iou(poly_det, poly_gt):
    """Calculate the IOU between two polygons.

    Args:
        poly_det (Polygon): A polygon predicted by detector.
        poly_gt (Polygon): A gt polygon.

    Returns:
        iou (float): The IOU between two polygons.
    """
    assert isinstance(poly_det, plg.Polygon)
    assert isinstance(poly_gt, plg.Polygon)
    area_inters, _ = poly_intersection(poly_det, poly_gt)
    area_union = poly_union(poly_det, poly_gt)
    if area_union == 0:
        return 0.0
    return area_inters / area_union


def poly_nms(polygons, threshold):
    assert isinstance(polygons, list)

    polygons = np.array(sorted(polygons, key=lambda x: x[-1]))

    keep_poly = []
    index = [i for i in range(polygons.shape[0])]

    while len(index) > 0:
        keep_poly.append(polygons[index[-1]].tolist())
        A = polygons[index[-1]][:-1]
        index = np.delete(index, -1)

        iou_list = np.zeros((len(index), ))
        for i in range(len(index)):
            B = polygons[index[i]][:-1]

            iou_list[i] = boundary_iou(A, B)
        remove_index = np.where(iou_list > threshold)
        index = np.delete(index, remove_index)

    return keep_poly


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


def fcenet_decode(preds,
                  fourier_degree,
                  num_reconstr_points,
                  scale,
                  alpha=1.0,
                  beta=2.0,
                  text_repr_type='poly',
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
        text_repr_type (str):  Boundary encoding type 'poly' or 'quad'.
        score_thr (float) : The threshold used to filter out the final
            candidates.
        nms_thr (float) :  The threshold of nms.

    Returns:
        boundaries (list[list[float]]): The instance boundary and confidence
            list.
    """
    assert isinstance(preds, list)
    assert len(preds) == 2
    assert text_repr_type in ['poly', 'quad']

    # import pdb;pdb.set_trace()
    cls_pred = preds[0][0]
    # tr_pred = F.softmax(cls_pred[0:2], axis=0).cpu().numpy()
    # tcl_pred = F.softmax(cls_pred[2:], axis=0).cpu().numpy()

    tr_pred = cls_pred[0:2]
    tcl_pred = cls_pred[2:]

    reg_pred = preds[1][0].transpose([1, 2, 0])  #.cpu().numpy()
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

    if text_repr_type == 'quad':
        new_boundaries = []
        for boundary in boundaries:
            poly = np.array(boundary[:-1]).reshape(-1, 2).astype(np.float32)
            score = boundary[-1]
            points = cv2.boxPoints(cv2.minAreaRect(poly))
            points = np.int0(points)
            new_boundaries.append(points.reshape(-1).tolist() + [score])

    return boundaries


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
                 text_repr_type='poly',
                 **kwargs):

        self.scales = scales
        self.fourier_degree = fourier_degree
        self.num_reconstr_points = num_reconstr_points
        self.decoding_type = decoding_type
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.alpha = alpha
        self.beta = beta
        self.text_repr_type = text_repr_type

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
        # assert check_argument.is_2dlist(boundaries)
        # assert isinstance(scale_factor, np.ndarray)
        # assert scale_factor.shape[0] == 4

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
        # import pdb;pdb.set_trace()
        boundaries = []
        for idx, score_map in enumerate(score_maps):
            scale = self.scales[idx]
            boundaries = boundaries + self._get_boundary_single(score_map,
                                                                scale)

        # nms
        boundaries = poly_nms(boundaries, self.nms_thr)
        # if rescale:
        # import pdb;pdb.set_trace()
        boundaries, scores = self.resize_boundary(
            boundaries, (1 / shape_list[0, 2:]).tolist()[::-1])

        boxes_batch = [dict(points=boundaries, scores=scores)]
        return boxes_batch

    def _get_boundary_single(self, score_map, scale):
        assert len(score_map) == 2
        assert score_map[1].shape[1] == 4 * self.fourier_degree + 2

        return fcenet_decode(
            preds=score_map,
            fourier_degree=self.fourier_degree,
            num_reconstr_points=self.num_reconstr_points,
            scale=scale,
            alpha=self.alpha,
            beta=self.beta,
            text_repr_type=self.text_repr_type,
            score_thr=self.score_thr,
            nms_thr=self.nms_thr)
