# coding: utf-8
"""Common tools to optical flow algorithms.

"""

import numpy as np
from scipy import ndimage as ndi

from ..transform import pyramid_reduce
from ..util.dtype import _convert


def get_warp_points(grid, flow):
    """Compute warp point coordinates.

    Parameters
    ----------
    grid : iterable
        The sparse grid to be warped (optained using
        ``np.meshgrid(..., sparse=True)).``)
    flow : ndarray
        The warping motion field.

    Returns
    -------
    out : ndarray
        The warp point coordinates.

    """
    out = flow.copy()
    for idx, g in enumerate(grid):
        out[idx, ...] += g
    return out


def resize_flow(flow, shape):
    """Rescale the values of the vector field (u, v) to the desired shape.

    The values of the output vector field are scaled to the new
    resolution.

    Parameters
    ----------
    flow : ndarray
        The motion field to be processed.
    shape : iterable
        Couple of integers representing the output shape.

    Returns
    -------
    rflow : ndarray
        The resized and rescaled motion field.

    """

    scale = [n / o for n, o in zip(shape, flow.shape[1:])]
    scale_factor = np.array(scale, dtype=flow.dtype)

    for _ in shape:
        scale_factor = scale_factor[..., np.newaxis]

    rflow = scale_factor*ndi.zoom(flow, [1] + scale, order=0,
                                  mode='nearest', prefilter=False)

    return rflow


def get_pyramid(I, downscale=2.0, nlevel=10, min_size=16):
    """Construct image pyramid.

    Parameters
    ----------
    I : ndarray
        The image to be preprocessed (Gray scale or RGB).
    downscale : float
        The pyramid downscale factor.
    nlevel : int
        The maximum number of pyramid levels.
    min_size : int
        The minimum size for any dimension of the pyramid levels.

    Returns
    -------
    pyramid : list[ndarray]
        The coarse to fine images pyramid.

    """

    pyramid = [I]
    size = min(I.shape)
    count = 1

    while (count < nlevel) and (size > downscale * min_size):
        J = pyramid_reduce(pyramid[-1], downscale, channel_axis=None)
        pyramid.append(J)
        size = min(J.shape)
        count += 1

    return pyramid[::-1]


def coarse_to_fine(I0, I1, solver, downscale=2, nlevel=10, min_size=16,
                   dtype=np.float32):
    """Generic coarse to fine solver.

    Parameters
    ----------
    I0 : ndarray
        The first gray scale image of the sequence.
    I1 : ndarray
        The second gray scale image of the sequence.
    solver : callable
        The solver applyed at each pyramid level.
    downscale : float
        The pyramid downscale factor.
    nlevel : int
        The maximum number of pyramid levels.
    min_size : int
        The minimum size for any dimension of the pyramid levels.
    dtype : dtype
        Output data type.

    Returns
    -------
    flow : ndarray
        The estimated optical flow components for each axis.

    """

    if I0.shape != I1.shape:
        raise ValueError("Input images should have the same shape")

    if np.dtype(dtype).char not in 'efdg':
        raise ValueError("Only floating point data type are valid"
                         " for optical flow")

    pyramid = list(zip(get_pyramid(_convert(I0, dtype),
                                   downscale, nlevel, min_size),
                       get_pyramid(_convert(I1, dtype),
                                   downscale, nlevel, min_size)))

    # Initialization to 0 at coarsest level.
    flow = np.zeros((pyramid[0][0].ndim, ) + pyramid[0][0].shape,
                    dtype=dtype)

    flow = solver(pyramid[0][0], pyramid[0][1], flow)

    for J0, J1 in pyramid[1:]:
        flow = solver(J0, J1, resize_flow(flow, J0.shape))

    return flow
