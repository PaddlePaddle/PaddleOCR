"""
Various transforms used for by the 3D code
"""

import numpy as np
import numpy.linalg as linalg


def _line2d_seg_dist(p1, p2, p0):
    """
    Return the distance(s) from line defined by p1 - p2 to point(s) p0.

    p0[0] = x(s)
    p0[1] = y(s)

    intersection point p = p1 + u*(p2-p1)
    and intersection point lies within segment if u is between 0 and 1
    """

    x21 = p2[0] - p1[0]
    y21 = p2[1] - p1[1]
    x01 = np.asarray(p0[0]) - p1[0]
    y01 = np.asarray(p0[1]) - p1[1]

    u = (x01*x21 + y01*y21) / (x21**2 + y21**2)
    u = np.clip(u, 0, 1)
    d = np.hypot(x01 - u*x21, y01 - u*y21)

    return d


def world_transformation(xmin, xmax,
                         ymin, ymax,
                         zmin, zmax, pb_aspect=None):
    """
    Produce a matrix that scales homogeneous coords in the specified ranges
    to [0, 1], or [0, pb_aspect[i]] if the plotbox aspect ratio is specified.
    """
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    if pb_aspect is not None:
        ax, ay, az = pb_aspect
        dx /= ax
        dy /= ay
        dz /= az

    return np.array([[1/dx, 0,    0,    -xmin/dx],
                     [0,    1/dy, 0,    -ymin/dy],
                     [0,    0,    1/dz, -zmin/dz],
                     [0,    0,    0,    1]])


def view_transformation(E, R, V):
    n = (E - R)
    ## new
#    n /= np.linalg.norm(n)
#    u = np.cross(V, n)
#    u /= np.linalg.norm(u)
#    v = np.cross(n, u)
#    Mr = np.diag([1.] * 4)
#    Mt = np.diag([1.] * 4)
#    Mr[:3,:3] = u, v, n
#    Mt[:3,-1] = -E
    ## end new

    ## old
    n = n / np.linalg.norm(n)
    u = np.cross(V, n)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    Mr = [[u[0], u[1], u[2], 0],
          [v[0], v[1], v[2], 0],
          [n[0], n[1], n[2], 0],
          [0,    0,    0,    1]]
    #
    Mt = [[1, 0, 0, -E[0]],
          [0, 1, 0, -E[1]],
          [0, 0, 1, -E[2]],
          [0, 0, 0, 1]]
    ## end old

    return np.dot(Mr, Mt)


def persp_transformation(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, a, b],
                     [0, 0, -1, 0]])


def ortho_transformation(zfront, zback):
    # note: w component in the resulting vector will be (zback-zfront), not 1
    a = -(zfront + zback)
    b = -(zfront - zback)
    return np.array([[2, 0, 0, 0],
                     [0, 2, 0, 0],
                     [0, 0, -2, 0],
                     [0, 0, a, b]])


def _proj_transform_vec(vec, M):
    vecw = np.dot(M, vec)
    w = vecw[3]
    # clip here..
    txs, tys, tzs = vecw[0]/w, vecw[1]/w, vecw[2]/w
    return txs, tys, tzs


def _proj_transform_vec_clip(vec, M):
    vecw = np.dot(M, vec)
    w = vecw[3]
    # clip here.
    txs, tys, tzs = vecw[0] / w, vecw[1] / w, vecw[2] / w
    tis = (0 <= vecw[0]) & (vecw[0] <= 1) & (0 <= vecw[1]) & (vecw[1] <= 1)
    if np.any(tis):
        tis = vecw[1] < 1
    return txs, tys, tzs, tis


def inv_transform(xs, ys, zs, M):
    iM = linalg.inv(M)
    vec = _vec_pad_ones(xs, ys, zs)
    vecr = np.dot(iM, vec)
    try:
        vecr = vecr / vecr[3]
    except OverflowError:
        pass
    return vecr[0], vecr[1], vecr[2]


def _vec_pad_ones(xs, ys, zs):
    return np.array([xs, ys, zs, np.ones_like(xs)])


def proj_transform(xs, ys, zs, M):
    """
    Transform the points by the projection matrix
    """
    vec = _vec_pad_ones(xs, ys, zs)
    return _proj_transform_vec(vec, M)


transform = proj_transform


def proj_transform_clip(xs, ys, zs, M):
    """
    Transform the points by the projection matrix
    and return the clipping result
    returns txs, tys, tzs, tis
    """
    vec = _vec_pad_ones(xs, ys, zs)
    return _proj_transform_vec_clip(vec, M)


def proj_points(points, M):
    return np.column_stack(proj_trans_points(points, M))


def proj_trans_points(points, M):
    xs, ys, zs = zip(*points)
    return proj_transform(xs, ys, zs, M)


def rot_x(V, alpha):
    cosa, sina = np.cos(alpha), np.sin(alpha)
    M1 = np.array([[1, 0, 0, 0],
                   [0, cosa, -sina, 0],
                   [0, sina, cosa, 0],
                   [0, 0, 0, 1]])
    return np.dot(M1, V)
