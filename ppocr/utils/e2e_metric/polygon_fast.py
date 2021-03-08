import numpy as np
from shapely.geometry import Polygon
#import Polygon
"""
:param det_x: [1, N] Xs of detection's vertices 
:param det_y: [1, N] Ys of detection's vertices
:param gt_x: [1, N] Xs of groundtruth's vertices
:param gt_y: [1, N] Ys of groundtruth's vertices

##############
All the calculation of 'AREA' in this script is handled by:
1) First generating a binary mask with the polygon area filled up with 1's
2) Summing up all the 1's
"""


def area(x, y):
    polygon = Polygon(np.stack([x, y], axis=1))
    return float(polygon.area)


def approx_area_of_intersection(det_x, det_y, gt_x, gt_y):
    """
    This helper determine if both polygons are intersecting with each others with an approximation method.
    Area of intersection represented by the minimum bounding rectangular [xmin, ymin, xmax, ymax]
    """
    det_ymax = np.max(det_y)
    det_xmax = np.max(det_x)
    det_ymin = np.min(det_y)
    det_xmin = np.min(det_x)

    gt_ymax = np.max(gt_y)
    gt_xmax = np.max(gt_x)
    gt_ymin = np.min(gt_y)
    gt_xmin = np.min(gt_x)

    all_min_ymax = np.minimum(det_ymax, gt_ymax)
    all_max_ymin = np.maximum(det_ymin, gt_ymin)

    intersect_heights = np.maximum(0.0, (all_min_ymax - all_max_ymin))

    all_min_xmax = np.minimum(det_xmax, gt_xmax)
    all_max_xmin = np.maximum(det_xmin, gt_xmin)
    intersect_widths = np.maximum(0.0, (all_min_xmax - all_max_xmin))

    return intersect_heights * intersect_widths


def area_of_intersection(det_x, det_y, gt_x, gt_y):
    p1 = Polygon(np.stack([det_x, det_y], axis=1)).buffer(0)
    p2 = Polygon(np.stack([gt_x, gt_y], axis=1)).buffer(0)
    return float(p1.intersection(p2).area)


def area_of_union(det_x, det_y, gt_x, gt_y):
    p1 = Polygon(np.stack([det_x, det_y], axis=1)).buffer(0)
    p2 = Polygon(np.stack([gt_x, gt_y], axis=1)).buffer(0)
    return float(p1.union(p2).area)


def iou(det_x, det_y, gt_x, gt_y):
    return area_of_intersection(det_x, det_y, gt_x, gt_y) / (
        area_of_union(det_x, det_y, gt_x, gt_y) + 1.0)


def iod(det_x, det_y, gt_x, gt_y):
    """
    This helper determine the fraction of intersection area over detection area
    """
    return area_of_intersection(det_x, det_y, gt_x, gt_y) / (
        area(det_x, det_y) + 1.0)
