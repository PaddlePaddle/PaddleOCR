import math

import numpy as np
from numpy import arctan2, exp, pi, sqrt

from .. import draw
from ..util.dtype import img_as_float
from .._shared.filters import gaussian
from .._shared.utils import check_nD
from ..color import gray2rgb



def daisy(image, step=4, radius=15, rings=3, histograms=8, orientations=8,
          normalization='l1', sigmas=None, ring_radii=None, visualize=False):
    '''Extract DAISY feature descriptors densely for the given image.

    DAISY is a feature descriptor similar to SIFT formulated in a way that
    allows for fast dense extraction. Typically, this is practical for
    bag-of-features image representations.

    The implementation follows Tola et al. [1]_ but deviate on the following
    points:

      * Histogram bin contribution are smoothed with a circular Gaussian
        window over the tonal range (the angular range).
      * The sigma values of the spatial Gaussian smoothing in this code do not
        match the sigma values in the original code by Tola et al. [2]_. In
        their code, spatial smoothing is applied to both the input image and
        the center histogram. However, this smoothing is not documented in [1]_
        and, therefore, it is omitted.

    Parameters
    ----------
    image : (M, N) array
        Input image (grayscale).
    step : int, optional
        Distance between descriptor sampling points.
    radius : int, optional
        Radius (in pixels) of the outermost ring.
    rings : int, optional
        Number of rings.
    histograms : int, optional
        Number of histograms sampled per ring.
    orientations : int, optional
        Number of orientations (bins) per histogram.
    normalization : [ 'l1' | 'l2' | 'daisy' | 'off' ], optional
        How to normalize the descriptors

          * 'l1': L1-normalization of each descriptor.
          * 'l2': L2-normalization of each descriptor.
          * 'daisy': L2-normalization of individual histograms.
          * 'off': Disable normalization.

    sigmas : 1D array of float, optional
        Standard deviation of spatial Gaussian smoothing for the center
        histogram and for each ring of histograms. The array of sigmas should
        be sorted from the center and out. I.e. the first sigma value defines
        the spatial smoothing of the center histogram and the last sigma value
        defines the spatial smoothing of the outermost ring. Specifying sigmas
        overrides the following parameter.

            ``rings = len(sigmas) - 1``

    ring_radii : 1D array of int, optional
        Radius (in pixels) for each ring. Specifying ring_radii overrides the
        following two parameters.

            ``rings = len(ring_radii)``
            ``radius = ring_radii[-1]``

        If both sigmas and ring_radii are given, they must satisfy the
        following predicate since no radius is needed for the center
        histogram.

            ``len(ring_radii) == len(sigmas) + 1``

    visualize : bool, optional
        Generate a visualization of the DAISY descriptors

    Returns
    -------
    descs : array
        Grid of DAISY descriptors for the given image as an array
        dimensionality  (P, Q, R) where

            ``P = ceil((M - radius*2) / step)``
            ``Q = ceil((N - radius*2) / step)``
            ``R = (rings * histograms + 1) * orientations``

    descs_img : (M, N, 3) array (only if visualize==True)
        Visualization of the DAISY descriptors.

    References
    ----------
    .. [1] Tola et al. "Daisy: An efficient dense descriptor applied to wide-
           baseline stereo." Pattern Analysis and Machine Intelligence, IEEE
           Transactions on 32.5 (2010): 815-830.
    .. [2] http://cvlab.epfl.ch/software/daisy
    '''

    check_nD(image, 2, 'img')

    image = img_as_float(image)
    float_dtype = image.dtype

    # Validate parameters.
    if sigmas is not None and ring_radii is not None \
            and len(sigmas) - 1 != len(ring_radii):
        raise ValueError('`len(sigmas)-1 != len(ring_radii)`')
    if ring_radii is not None:
        rings = len(ring_radii)
        radius = ring_radii[-1]
    if sigmas is not None:
        rings = len(sigmas) - 1
    if sigmas is None:
        sigmas = [radius * (i + 1) / float(2 * rings) for i in range(rings)]
    if ring_radii is None:
        ring_radii = [radius * (i + 1) / float(rings) for i in range(rings)]
    if normalization not in ['l1', 'l2', 'daisy', 'off']:
        raise ValueError('Invalid normalization method.')

    # Compute image derivatives.
    dx = np.zeros(image.shape, dtype=float_dtype)
    dy = np.zeros(image.shape, dtype=float_dtype)
    dx[:, :-1] = np.diff(image, n=1, axis=1)
    dy[:-1, :] = np.diff(image, n=1, axis=0)

    # Compute gradient orientation and magnitude and their contribution
    # to the histograms.
    grad_mag = sqrt(dx ** 2 + dy ** 2)
    grad_ori = arctan2(dy, dx)
    orientation_kappa = orientations / pi
    orientation_angles = [2 * o * pi / orientations - pi
                          for o in range(orientations)]
    hist = np.empty((orientations,) + image.shape, dtype=float_dtype)
    for i, o in enumerate(orientation_angles):
        # Weigh bin contribution by the circular normal distribution
        hist[i, :, :] = exp(orientation_kappa * np.cos(grad_ori - o))
        # Weigh bin contribution by the gradient magnitude
        hist[i, :, :] = np.multiply(hist[i, :, :], grad_mag)

    # Smooth orientation histograms for the center and all rings.
    sigmas = [sigmas[0]] + sigmas
    hist_smooth = np.empty((rings + 1,) + hist.shape, dtype=float_dtype)
    for i in range(rings + 1):
        for j in range(orientations):
            hist_smooth[i, j, :, :] = gaussian(hist[j, :, :], sigma=sigmas[i],
                                               mode='reflect')

    # Assemble descriptor grid.
    theta = [2 * pi * j / histograms for j in range(histograms)]
    desc_dims = (rings * histograms + 1) * orientations
    descs = np.empty((desc_dims, image.shape[0] - 2 * radius,
                      image.shape[1] - 2 * radius),
                     dtype=float_dtype)
    descs[:orientations, :, :] = hist_smooth[0, :, radius:-radius,
                                             radius:-radius]
    idx = orientations
    for i in range(rings):
        for j in range(histograms):
            y_min = radius + int(round(ring_radii[i] * math.sin(theta[j])))
            y_max = descs.shape[1] + y_min
            x_min = radius + int(round(ring_radii[i] * math.cos(theta[j])))
            x_max = descs.shape[2] + x_min
            descs[idx:idx + orientations, :, :] = hist_smooth[i + 1, :,
                                                              y_min:y_max,
                                                              x_min:x_max]
            idx += orientations
    descs = descs[:, ::step, ::step]
    descs = descs.swapaxes(0, 1).swapaxes(1, 2)

    # Normalize descriptors.
    if normalization != 'off':
        descs += 1e-10
        if normalization == 'l1':
            descs /= np.sum(descs, axis=2)[:, :, np.newaxis]
        elif normalization == 'l2':
            descs /= sqrt(np.sum(descs ** 2, axis=2))[:, :, np.newaxis]
        elif normalization == 'daisy':
            for i in range(0, desc_dims, orientations):
                norms = sqrt(np.sum(descs[:, :, i:i + orientations] ** 2,
                                    axis=2))
                descs[:, :, i:i + orientations] /= norms[:, :, np.newaxis]

    if visualize:
        descs_img = gray2rgb(image)
        for i in range(descs.shape[0]):
            for j in range(descs.shape[1]):
                # Draw center histogram sigma
                color = [1, 0, 0]
                desc_y = i * step + radius
                desc_x = j * step + radius
                rows, cols, val = draw.circle_perimeter_aa(desc_y, desc_x, int(sigmas[0]))
                draw.set_color(descs_img, (rows, cols), color, alpha=val)
                max_bin = np.max(descs[i, j, :])
                for o_num, o in enumerate(orientation_angles):
                    # Draw center histogram bins
                    bin_size = descs[i, j, o_num] / max_bin
                    dy = sigmas[0] * bin_size * math.sin(o)
                    dx = sigmas[0] * bin_size * math.cos(o)
                    rows, cols, val = draw.line_aa(desc_y, desc_x, int(desc_y + dy),
                                                   int(desc_x + dx))
                    draw.set_color(descs_img, (rows, cols), color, alpha=val)
                for r_num, r in enumerate(ring_radii):
                    color_offset = float(1 + r_num) / rings
                    color = (1 - color_offset, 1, color_offset)
                    for t_num, t in enumerate(theta):
                        # Draw ring histogram sigmas
                        hist_y = desc_y + int(round(r * math.sin(t)))
                        hist_x = desc_x + int(round(r * math.cos(t)))
                        rows, cols, val = draw.circle_perimeter_aa(hist_y, hist_x,
                                                                   int(sigmas[r_num + 1]))
                        draw.set_color(descs_img, (rows, cols), color, alpha=val)
                        for o_num, o in enumerate(orientation_angles):
                            # Draw histogram bins
                            bin_size = descs[i, j, orientations + r_num *
                                             histograms * orientations +
                                             t_num * orientations + o_num]
                            bin_size /= max_bin
                            dy = sigmas[r_num + 1] * bin_size * math.sin(o)
                            dx = sigmas[r_num + 1] * bin_size * math.cos(o)
                            rows, cols, val = draw.line_aa(hist_y, hist_x,
                                                           int(hist_y + dy),
                                                           int(hist_x + dx))
                            draw.set_color(descs_img, (rows, cols), color, alpha=val)
        return descs, descs_img
    else:
        return descs
