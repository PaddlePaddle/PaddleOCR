"""
Locality aware nms.
This code is referred from: https://github.com/songdejia/EAST/blob/master/locality_aware_nms.py
"""

import numpy as np
from shapely.geometry import Polygon


def intersection(g, p):
    """
    Intersection.
    """
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    g = g.buffer(0)
    p = p.buffer(0)
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


def intersection_iog(g, p):
    """
    Intersection_iog.
    """
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    # union = g.area + p.area - inter
    union = p.area
    if union == 0:
        print("p_area is very small")
        return 0
    else:
        return inter / union


def weighted_merge(g, p):
    """
    Weighted merge.
    """
    g[:8] = (g[8] * g[:8] + p[8] * p[:8]) / (g[8] + p[8])
    g[8] = g[8] + p[8]
    return g


def standard_nms(S, thres):
    """
    Standard nms.
    """
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds + 1]

    return S[keep]


def standard_nms_inds(S, thres):
    """
    Standard nms, return inds.
    """
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds + 1]

    return keep


def nms(S, thres):
    """
    nms.
    """
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds + 1]

    return keep


def soft_nms(boxes_in, Nt_thres=0.3, threshold=0.8, sigma=0.5, method=2):
    """
    soft_nms
    :para boxes_in, N x 9 (coords + score)
    :para threshould, eliminate cases min score(0.001)
    :para Nt_thres, iou_threshi
    :para sigma, gaussian weght
    :method, linear or gaussian
    """
    boxes = boxes_in.copy()
    N = boxes.shape[0]
    if N is None or N < 1:
        return np.array([])
    pos, maxpos = 0, 0
    weight = 0.0
    inds = np.arange(N)
    tbox, sbox = boxes[0].copy(), boxes[0].copy()
    for i in range(N):
        maxscore = boxes[i, 8]
        maxpos = i
        tbox = boxes[i].copy()
        ti = inds[i]
        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 8]:
                maxscore = boxes[pos, 8]
                maxpos = pos
            pos = pos + 1
        # add max box as a detection
        boxes[i, :] = boxes[maxpos, :]
        inds[i] = inds[maxpos]
        # swap
        boxes[maxpos, :] = tbox
        inds[maxpos] = ti
        tbox = boxes[i].copy()
        pos = i + 1
        # NMS iteration
        while pos < N:
            sbox = boxes[pos].copy()
            ts_iou_val = intersection(tbox, sbox)
            if ts_iou_val > 0:
                if method == 1:
                    if ts_iou_val > Nt_thres:
                        weight = 1 - ts_iou_val
                    else:
                        weight = 1
                elif method == 2:
                    weight = np.exp(-1.0 * ts_iou_val**2 / sigma)
                else:
                    if ts_iou_val > Nt_thres:
                        weight = 0
                    else:
                        weight = 1
                boxes[pos, 8] = weight * boxes[pos, 8]
                # if box score falls below threshold, discard the box by
                # swapping last box update N
                if boxes[pos, 8] < threshold:
                    boxes[pos, :] = boxes[N - 1, :]
                    inds[pos] = inds[N - 1]
                    N = N - 1
                    pos = pos - 1
            pos = pos + 1

    return boxes[:N]


def nms_locality(polys, thres=0.3):
    """
    locality aware nms of EAST
    :param polys: a N*9 numpy array. first 8 coordinates, then prob
    :return: boxes after nms
    """
    S = []
    p = None
    for g in polys:
        if p is not None and intersection(g, p) > thres:
            p = weighted_merge(g, p)
        else:
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)

    if len(S) == 0:
        return np.array([])
    return standard_nms(np.array(S), thres)


if __name__ == "__main__":
    # 343,350,448,135,474,143,369,359
    print(Polygon(np.array([[343, 350], [448, 135], [474, 143], [369, 359]])).area)
