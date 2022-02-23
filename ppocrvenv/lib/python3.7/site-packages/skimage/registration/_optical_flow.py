# coding: utf-8
"""TV-L1 optical flow algorithm implementation.

"""

from functools import partial
from itertools import combinations_with_replacement

import numpy as np
from scipy import ndimage as ndi

from .._shared.filters import gaussian as gaussian_filter
from .._shared.utils import _supported_float_type
from ..transform import warp
from ._optical_flow_utils import coarse_to_fine, get_warp_points


def _tvl1(reference_image, moving_image, flow0, attachment, tightness,
          num_warp, num_iter, tol, prefilter):
    """TV-L1 solver for optical flow estimation.

    Parameters
    ----------
    reference_image : ndarray, shape (M, N[, P[, ...]])
        The first gray scale image of the sequence.
    moving_image : ndarray, shape (M, N[, P[, ...]])
        The second gray scale image of the sequence.
    flow0 : ndarray, shape (image0.ndim, M, N[, P[, ...]])
        Initialization for the vector field.
    attachment : float
        Attachment parameter. The smaller this parameter is,
        the smoother is the solutions.
    tightness : float
        Tightness parameter. It should have a small value in order to
        maintain attachement and regularization parts in
        correspondence.
    num_warp : int
        Number of times image1 is warped.
    num_iter : int
        Number of fixed point iteration.
    tol : float
        Tolerance used as stopping criterion based on the L² distance
        between two consecutive values of (u, v).
    prefilter : bool
        Whether to prefilter the estimated optical flow before each
        image warp.

    Returns
    -------
    flow : ndarray, shape ((image0.ndim, M, N[, P[, ...]])
        The estimated optical flow components for each axis.

    """

    dtype = reference_image.dtype
    grid = np.meshgrid(*[np.arange(n, dtype=dtype)
                         for n in reference_image.shape],
                       indexing='ij', sparse=True)

    dt = 0.5 / reference_image.ndim
    reg_num_iter = 2
    f0 = attachment * tightness
    f1 = dt / tightness
    tol *= reference_image.size

    flow_current = flow_previous = flow0

    g = np.zeros((reference_image.ndim,) + reference_image.shape, dtype=dtype)
    proj = np.zeros((reference_image.ndim, reference_image.ndim,)
                    + reference_image.shape, dtype=dtype)

    s_g = [slice(None), ] * g.ndim
    s_p = [slice(None), ] * proj.ndim
    s_d = [slice(None), ] * (proj.ndim-2)

    for _ in range(num_warp):
        if prefilter:
            flow_current = ndi.median_filter(flow_current,
                                             [1] + reference_image.ndim * [3])

        image1_warp = warp(moving_image, get_warp_points(grid, flow_current),
                           mode='edge')
        grad = np.array(np.gradient(image1_warp))
        NI = (grad*grad).sum(0)
        NI[NI == 0] = 1

        rho_0 = image1_warp - reference_image - (grad * flow_current).sum(0)

        for _ in range(num_iter):

            # Data term

            rho = rho_0 + (grad*flow_current).sum(0)

            idx = abs(rho) <= f0 * NI

            flow_auxiliary = flow_current

            flow_auxiliary[:, idx] -= rho[idx]*grad[:, idx]/NI[idx]

            idx = ~idx
            srho = f0 * np.sign(rho[idx])
            flow_auxiliary[:, idx] -= srho*grad[:, idx]

            # Regularization term
            flow_current = flow_auxiliary.copy()

            for idx in range(reference_image.ndim):
                s_p[0] = idx
                for _ in range(reg_num_iter):
                    for ax in range(reference_image.ndim):
                        s_g[0] = ax
                        s_g[ax+1] = slice(0, -1)
                        g[tuple(s_g)] = np.diff(flow_current[idx], axis=ax)
                        s_g[ax+1] = slice(None)

                    norm = np.sqrt((g ** 2).sum(0))[np.newaxis, ...]
                    norm *= f1
                    norm += 1.
                    proj[idx] -= dt * g
                    proj[idx] /= norm

                    # d will be the (negative) divergence of proj[idx]
                    d = -proj[idx].sum(0)
                    for ax in range(reference_image.ndim):
                        s_p[1] = ax
                        s_p[ax+2] = slice(0, -1)
                        s_d[ax] = slice(1, None)
                        d[tuple(s_d)] += proj[tuple(s_p)]
                        s_p[ax+2] = slice(None)
                        s_d[ax] = slice(None)

                    flow_current[idx] = flow_auxiliary[idx] + d

        flow_previous -= flow_current  # The difference as stopping criteria
        if (flow_previous*flow_previous).sum() < tol:
            break

        flow_previous = flow_current

    return flow_current


def optical_flow_tvl1(reference_image, moving_image,
                      *,
                      attachment=15, tightness=0.3, num_warp=5, num_iter=10,
                      tol=1e-4, prefilter=False, dtype=np.float32):
    r"""Coarse to fine optical flow estimator.

    The TV-L1 solver is applied at each level of the image
    pyramid. TV-L1 is a popular algorithm for optical flow estimation
    introduced by Zack et al. [1]_, improved in [2]_ and detailed in [3]_.

    Parameters
    ----------
    reference_image : ndarray, shape (M, N[, P[, ...]])
        The first gray scale image of the sequence.
    moving_image : ndarray, shape (M, N[, P[, ...]])
        The second gray scale image of the sequence.
    attachment : float, optional
        Attachment parameter (:math:`\lambda` in [1]_). The smaller
        this parameter is, the smoother the returned result will be.
    tightness : float, optional
        Tightness parameter (:math:`\tau` in [1]_). It should have
        a small value in order to maintain attachement and
        regularization parts in correspondence.
    num_warp : int, optional
        Number of times image1 is warped.
    num_iter : int, optional
        Number of fixed point iteration.
    tol : float, optional
        Tolerance used as stopping criterion based on the L² distance
        between two consecutive values of (u, v).
    prefilter : bool, optional
        Whether to prefilter the estimated optical flow before each
        image warp. When True, a median filter with window size 3
        along each axis is applied. This helps to remove potential
        outliers.
    dtype : dtype, optional
        Output data type: must be floating point. Single precision
        provides good results and saves memory usage and computation
        time compared to double precision.

    Returns
    -------
    flow : ndarray, shape ((image0.ndim, M, N[, P[, ...]])
        The estimated optical flow components for each axis.

    Notes
    -----
    Color images are not supported.

    References
    ----------
    .. [1] Zach, C., Pock, T., & Bischof, H. (2007, September). A
       duality based approach for realtime TV-L 1 optical flow. In Joint
       pattern recognition symposium (pp. 214-223). Springer, Berlin,
       Heidelberg. :DOI:`10.1007/978-3-540-74936-3_22`
    .. [2] Wedel, A., Pock, T., Zach, C., Bischof, H., & Cremers,
       D. (2009). An improved algorithm for TV-L 1 optical flow. In
       Statistical and geometrical approaches to visual motion analysis
       (pp. 23-45). Springer, Berlin, Heidelberg.
       :DOI:`10.1007/978-3-642-03061-1_2`
    .. [3] Pérez, J. S., Meinhardt-Llopis, E., & Facciolo,
       G. (2013). TV-L1 optical flow estimation. Image Processing On
       Line, 2013, 137-150. :DOI:`10.5201/ipol.2013.26`

    Examples
    --------
    >>> from skimage.color import rgb2gray
    >>> from skimage.data import stereo_motorcycle
    >>> from skimage.registration import optical_flow_tvl1
    >>> image0, image1, disp = stereo_motorcycle()
    >>> # --- Convert the images to gray level: color is not supported.
    >>> image0 = rgb2gray(image0)
    >>> image1 = rgb2gray(image1)
    >>> flow = optical_flow_tvl1(image1, image0)

    """

    solver = partial(_tvl1, attachment=attachment,
                     tightness=tightness, num_warp=num_warp, num_iter=num_iter,
                     tol=tol, prefilter=prefilter)

    if np.dtype(dtype) != _supported_float_type(dtype):
        msg = f"dtype={dtype} is not supported. Try 'float32' or 'float64.'"
        raise ValueError(msg)

    return coarse_to_fine(reference_image, moving_image, solver, dtype=dtype)


def _ilk(reference_image, moving_image, flow0, radius, num_warp, gaussian,
         prefilter):
    """Iterative Lucas-Kanade (iLK) solver for optical flow estimation.

    Parameters
    ----------
    reference_image : ndarray, shape (M, N[, P[, ...]])
        The first gray scale image of the sequence.
    moving_image : ndarray, shape (M, N[, P[, ...]])
        The second gray scale image of the sequence.
    flow0 : ndarray, shape (reference_image.ndim, M, N[, P[, ...]])
        Initialization for the vector field.
    radius : int
        Radius of the window considered around each pixel.
    num_warp : int
        Number of times moving_image is warped.
    gaussian : bool
        if True, a gaussian kernel is used for the local
        integration. Otherwise, a uniform kernel is used.
    prefilter : bool
        Whether to prefilter the estimated optical flow before each
        image warp. This helps to remove potential outliers.

    Returns
    -------
    flow : ndarray, shape ((reference_image.ndim, M, N[, P[, ...]])
        The estimated optical flow components for each axis.

    """
    dtype = reference_image.dtype
    ndim = reference_image.ndim
    size = 2 * radius + 1

    if gaussian:
        sigma = ndim * (size / 4, )
        filter_func = partial(gaussian_filter, sigma=sigma, mode='mirror')
    else:
        filter_func = partial(ndi.uniform_filter, size=ndim * (size, ),
                              mode='mirror')

    flow = flow0
    # For each pixel location (i, j), the optical flow X = flow[:, i, j]
    # is the solution of the ndim x ndim linear system
    # A[i, j] * X = b[i, j]
    A = np.zeros(reference_image.shape + (ndim, ndim), dtype=dtype)
    b = np.zeros(reference_image.shape + (ndim, ), dtype=dtype)

    grid = np.meshgrid(*[np.arange(n, dtype=dtype)
                         for n in reference_image.shape],
                       indexing='ij', sparse=True)

    for _ in range(num_warp):
        if prefilter:
            flow = ndi.median_filter(flow, (1, ) + ndim * (3, ))

        moving_image_warp = warp(moving_image, get_warp_points(grid, flow),
                                 mode='edge')
        grad = np.stack(np.gradient(moving_image_warp), axis=0)
        error_image = ((grad * flow).sum(axis=0)
                       + reference_image - moving_image_warp)

        # Local linear systems creation
        for i, j in combinations_with_replacement(range(ndim), 2):
            A[..., i, j] = A[..., j, i] = filter_func(grad[i] * grad[j])

        for i in range(ndim):
            b[..., i] = filter_func(grad[i] * error_image)

        # Don't consider badly conditioned linear systems
        idx = abs(np.linalg.det(A)) < 1e-14
        A[idx] = np.eye(ndim, dtype=dtype)
        b[idx] = 0

        # Solve the local linear systems
        flow = np.moveaxis(np.linalg.solve(A, b), ndim, 0)

    return flow


def optical_flow_ilk(reference_image, moving_image, *,
                     radius=7, num_warp=10, gaussian=False,
                     prefilter=False, dtype=np.float32):
    """Coarse to fine optical flow estimator.

    The iterative Lucas-Kanade (iLK) solver is applied at each level
    of the image pyramid. iLK [1]_ is a fast and robust alternative to
    TVL1 algorithm although less accurate for rendering flat surfaces
    and object boundaries (see [2]_).

    Parameters
    ----------
    reference_image : ndarray, shape (M, N[, P[, ...]])
        The first gray scale image of the sequence.
    moving_image : ndarray, shape (M, N[, P[, ...]])
        The second gray scale image of the sequence.
    radius : int, optional
        Radius of the window considered around each pixel.
    num_warp : int, optional
        Number of times moving_image is warped.
    gaussian : bool, optional
        If True, a Gaussian kernel is used for the local
        integration. Otherwise, a uniform kernel is used.
    prefilter : bool, optional
        Whether to prefilter the estimated optical flow before each
        image warp. When True, a median filter with window size 3
        along each axis is applied. This helps to remove potential
        outliers.
    dtype : dtype, optional
        Output data type: must be floating point. Single precision
        provides good results and saves memory usage and computation
        time compared to double precision.

    Returns
    -------
    flow : ndarray, shape ((reference_image.ndim, M, N[, P[, ...]])
        The estimated optical flow components for each axis.

    Notes
    -----
    - The implemented algorithm is described in **Table2** of [1]_.
    - Color images are not supported.

    References
    ----------
    .. [1] Le Besnerais, G., & Champagnat, F. (2005, September). Dense
       optical flow by iterative local window registration. In IEEE
       International Conference on Image Processing 2005 (Vol. 1,
       pp. I-137). IEEE. :DOI:`10.1109/ICIP.2005.1529706`
    .. [2] Plyer, A., Le Besnerais, G., & Champagnat,
       F. (2016). Massively parallel Lucas Kanade optical flow for
       real-time video processing applications. Journal of Real-Time
       Image Processing, 11(4), 713-730. :DOI:`10.1007/s11554-014-0423-0`

    Examples
    --------
    >>> from skimage.color import rgb2gray
    >>> from skimage.data import stereo_motorcycle
    >>> from skimage.registration import optical_flow_ilk
    >>> reference_image, moving_image, disp = stereo_motorcycle()
    >>> # --- Convert the images to gray level: color is not supported.
    >>> reference_image = rgb2gray(reference_image)
    >>> moving_image = rgb2gray(moving_image)
    >>> flow = optical_flow_ilk(moving_image, reference_image)

    """

    solver = partial(_ilk, radius=radius, num_warp=num_warp, gaussian=gaussian,
                     prefilter=prefilter)

    if np.dtype(dtype) != _supported_float_type(dtype):
        msg = f"dtype={dtype} is not supported. Try 'float32' or 'float64.'"
        raise ValueError(msg)

    return coarse_to_fine(reference_image, moving_image, solver, dtype=dtype)
