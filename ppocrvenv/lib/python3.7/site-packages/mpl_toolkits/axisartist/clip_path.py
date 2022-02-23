import numpy as np
from math import degrees
from matplotlib import _api
import math


_api.warn_deprecated("3.5", name=__name__, obj_type="module")


def atan2(dy, dx):
    if dx == 0 and dy == 0:
        _api.warn_external("dx and dy are 0")
        return 0
    else:
        return math.atan2(dy, dx)


# FIXME : The current algorithm seems to return incorrect angle when the line
# ends at the boundary.
def clip(xlines, ylines, x0, clip="right", xdir=True, ydir=True):

    clipped_xlines = []
    clipped_ylines = []

    _pos_angles = []

    xsign = 1 if xdir else -1
    ysign = 1 if ydir else -1

    for x, y in zip(xlines, ylines):

        if clip in ["up", "right"]:
            b = (x < x0).astype("i")
            db = b[1:] - b[:-1]
        else:
            b = (x > x0).astype("i")
            db = b[1:] - b[:-1]

        if b[0]:
            ns = 0
        else:
            ns = -1
        segx, segy = [], []
        for (i,) in np.argwhere(db):
            c = db[i]
            if c == -1:
                dx = (x0 - x[i])
                dy = (y[i+1] - y[i]) * (dx / (x[i+1] - x[i]))
                y0 = y[i] + dy
                clipped_xlines.append(np.concatenate([segx, x[ns:i+1], [x0]]))
                clipped_ylines.append(np.concatenate([segy, y[ns:i+1], [y0]]))
                ns = -1
                segx, segy = [], []

                if dx == 0. and dy == 0:
                    dx = x[i+1] - x[i]
                    dy = y[i+1] - y[i]

                a = degrees(atan2(ysign*dy, xsign*dx))
                _pos_angles.append((x0, y0, a))

            elif c == 1:
                dx = (x0 - x[i])
                dy = (y[i+1] - y[i]) * (dx / (x[i+1] - x[i]))
                y0 = y[i] + dy
                segx, segy = [x0], [y0]
                ns = i+1

                if dx == 0. and dy == 0:
                    dx = x[i+1] - x[i]
                    dy = y[i+1] - y[i]

                a = degrees(atan2(ysign*dy, xsign*dx))
                _pos_angles.append((x0, y0, a))

        if ns != -1:
            clipped_xlines.append(np.concatenate([segx, x[ns:]]))
            clipped_ylines.append(np.concatenate([segy, y[ns:]]))

    return clipped_xlines, clipped_ylines, _pos_angles


def clip_line_to_rect(xline, yline, bbox):

    x0, y0, x1, y1 = bbox.extents

    xdir = x1 > x0
    ydir = y1 > y0

    if x1 > x0:
        lx1, ly1, c_right_ = clip([xline], [yline], x1,
                                  clip="right", xdir=xdir, ydir=ydir)
        lx2, ly2, c_left_ = clip(lx1, ly1, x0,
                                 clip="left", xdir=xdir, ydir=ydir)
    else:
        lx1, ly1, c_right_ = clip([xline], [yline], x0,
                                  clip="right", xdir=xdir, ydir=ydir)
        lx2, ly2, c_left_ = clip(lx1, ly1, x1,
                                 clip="left", xdir=xdir, ydir=ydir)

    if y1 > y0:
        ly3, lx3, c_top_ = clip(ly2, lx2, y1,
                                clip="right", xdir=ydir, ydir=xdir)
        ly4, lx4, c_bottom_ = clip(ly3, lx3, y0,
                                   clip="left", xdir=ydir, ydir=xdir)
    else:
        ly3, lx3, c_top_ = clip(ly2, lx2, y0,
                                clip="right", xdir=ydir, ydir=xdir)
        ly4, lx4, c_bottom_ = clip(ly3, lx3, y1,
                                   clip="left", xdir=ydir, ydir=xdir)

    c_left = [((x, y), (a + 90) % 180 - 90) for x, y, a in c_left_
              if bbox.containsy(y)]
    c_bottom = [((x, y), (90 - a) % 180) for y, x, a in c_bottom_
                if bbox.containsx(x)]
    c_right = [((x, y), (a + 90) % 180 + 90) for x, y, a in c_right_
               if bbox.containsy(y)]
    c_top = [((x, y), (90 - a) % 180 + 180) for y, x, a in c_top_
             if bbox.containsx(x)]

    return list(zip(lx4, ly4)), [c_left, c_bottom, c_right, c_top]
