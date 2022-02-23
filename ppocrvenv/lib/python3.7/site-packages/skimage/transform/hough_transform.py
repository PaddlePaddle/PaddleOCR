import numpy as np
from scipy.spatial import cKDTree
from ._hough_transform import (_hough_circle,
                               _hough_ellipse,
                               _hough_line,
                               _probabilistic_hough_line as _prob_hough_line)


def hough_line_peaks(hspace, angles, dists, min_distance=9, min_angle=10,
                     threshold=None, num_peaks=np.inf):
    """Return peaks in a straight line Hough transform.

    Identifies most prominent lines separated by a certain angle and distance
    in a Hough transform. Non-maximum suppression with different sizes is
    applied separately in the first (distances) and second (angles) dimension
    of the Hough space to identify peaks.

    Parameters
    ----------
    hspace : (N, M) array
        Hough space returned by the `hough_line` function.
    angles : (M,) array
        Angles returned by the `hough_line` function. Assumed to be continuous.
        (`angles[-1] - angles[0] == PI`).
    dists : (N, ) array
        Distances returned by the `hough_line` function.
    min_distance : int, optional
        Minimum distance separating lines (maximum filter size for first
        dimension of hough space).
    min_angle : int, optional
        Minimum angle separating lines (maximum filter size for second
        dimension of hough space).
    threshold : float, optional
        Minimum intensity of peaks. Default is `0.5 * max(hspace)`.
    num_peaks : int, optional
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` coordinates based on peak intensity.

    Returns
    -------
    accum, angles, dists : tuple of array
        Peak values in Hough space, angles and distances.

    Examples
    --------
    >>> from skimage.transform import hough_line, hough_line_peaks
    >>> from skimage.draw import line
    >>> img = np.zeros((15, 15), dtype=bool)
    >>> rr, cc = line(0, 0, 14, 14)
    >>> img[rr, cc] = 1
    >>> rr, cc = line(0, 14, 14, 0)
    >>> img[cc, rr] = 1
    >>> hspace, angles, dists = hough_line(img)
    >>> hspace, angles, dists = hough_line_peaks(hspace, angles, dists)
    >>> len(angles)
    2

    """
    from ..feature.peak import _prominent_peaks

    min_angle = min(min_angle, hspace.shape[1])
    h, a, d = _prominent_peaks(hspace, min_xdistance=min_angle,
                               min_ydistance=min_distance,
                               threshold=threshold,
                               num_peaks=num_peaks)
    if a.any():
        return (h, angles[a], dists[d])
    else:
        return (h, np.array([]), np.array([]))


def hough_circle(image, radius, normalize=True, full_output=False):
    """Perform a circular Hough transform.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image with nonzero values representing edges.
    radius : scalar or sequence of scalars
        Radii at which to compute the Hough transform.
        Floats are converted to integers.
    normalize : boolean, optional (default True)
        Normalize the accumulator with the number
        of pixels used to draw the radius.
    full_output : boolean, optional (default False)
        Extend the output size by twice the largest
        radius in order to detect centers outside the
        input picture.

    Returns
    -------
    H : 3D ndarray (radius index, (M + 2R, N + 2R) ndarray)
        Hough transform accumulator for each radius.
        R designates the larger radius if full_output is True.
        Otherwise, R = 0.

    Examples
    --------
    >>> from skimage.transform import hough_circle
    >>> from skimage.draw import circle_perimeter
    >>> img = np.zeros((100, 100), dtype=bool)
    >>> rr, cc = circle_perimeter(25, 35, 23)
    >>> img[rr, cc] = 1
    >>> try_radii = np.arange(5, 50)
    >>> res = hough_circle(img, try_radii)
    >>> ridx, r, c = np.unravel_index(np.argmax(res), res.shape)
    >>> r, c, try_radii[ridx]
    (25, 35, 23)

    """
    radius = np.atleast_1d(np.asarray(radius))
    return _hough_circle(image, radius.astype(np.intp),
                         normalize=normalize, full_output=full_output)


def hough_ellipse(image, threshold=4, accuracy=1, min_size=4, max_size=None):
    """Perform an elliptical Hough transform.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image with nonzero values representing edges.
    threshold : int, optional
        Accumulator threshold value.
    accuracy : double, optional
        Bin size on the minor axis used in the accumulator.
    min_size : int, optional
        Minimal major axis length.
    max_size : int, optional
        Maximal minor axis length.
        If None, the value is set to the half of the smaller
        image dimension.

    Returns
    -------
    result : ndarray with fields [(accumulator, yc, xc, a, b, orientation)].
          Where ``(yc, xc)`` is the center, ``(a, b)`` the major and minor
          axes, respectively. The `orientation` value follows
          `skimage.draw.ellipse_perimeter` convention.

    Examples
    --------
    >>> from skimage.transform import hough_ellipse
    >>> from skimage.draw import ellipse_perimeter
    >>> img = np.zeros((25, 25), dtype=np.uint8)
    >>> rr, cc = ellipse_perimeter(10, 10, 6, 8)
    >>> img[cc, rr] = 1
    >>> result = hough_ellipse(img, threshold=8)
    >>> result.tolist()
    [(10, 10.0, 10.0, 8.0, 6.0, 0.0)]

    Notes
    -----
    The accuracy must be chosen to produce a peak in the accumulator
    distribution. In other words, a flat accumulator distribution with low
    values may be caused by a too low bin size.

    References
    ----------
    .. [1] Xie, Yonghong, and Qiang Ji. "A new efficient ellipse detection
           method." Pattern Recognition, 2002. Proceedings. 16th International
           Conference on. Vol. 2. IEEE, 2002
    """
    return _hough_ellipse(image, threshold=threshold, accuracy=accuracy,
                          min_size=min_size, max_size=max_size)


def hough_line(image, theta=None):
    """Perform a straight line Hough transform.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image with nonzero values representing edges.
    theta : 1D ndarray of double, optional
        Angles at which to compute the transform, in radians.
        Defaults to a vector of 180 angles evenly spaced in the
        range [-pi/2, pi/2).

    Returns
    -------
    hspace : 2-D ndarray of uint64
        Hough transform accumulator.
    angles : ndarray
        Angles at which the transform is computed, in radians.
    distances : ndarray
        Distance values.

    Notes
    -----
    The origin is the top left corner of the original image.
    X and Y axis are horizontal and vertical edges respectively.
    The distance is the minimal algebraic distance from the origin
    to the detected line.
    The angle accuracy can be improved by decreasing the step size in
    the `theta` array.

    Examples
    --------
    Generate a test image:

    >>> img = np.zeros((100, 150), dtype=bool)
    >>> img[30, :] = 1
    >>> img[:, 65] = 1
    >>> img[35:45, 35:50] = 1
    >>> for i in range(90):
    ...     img[i, i] = 1
    >>> rng = np.random.default_rng()
    >>> img += rng.random(img.shape) > 0.95

    Apply the Hough transform:

    >>> out, angles, d = hough_line(img)

    .. plot:: hough_tf.py

    """
    if image.ndim != 2:
        raise ValueError('The input image `image` must be 2D.')

    if theta is None:
        # These values are approximations of pi/2
        theta = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)

    return _hough_line(image, theta=theta)


def probabilistic_hough_line(image, threshold=10, line_length=50, line_gap=10,
                             theta=None, seed=None):
    """Return lines from a progressive probabilistic line Hough transform.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image with nonzero values representing edges.
    threshold : int, optional
        Threshold
    line_length : int, optional
        Minimum accepted length of detected lines.
        Increase the parameter to extract longer lines.
    line_gap : int, optional
        Maximum gap between pixels to still form a line.
        Increase the parameter to merge broken lines more aggressively.
    theta : 1D ndarray, dtype=double, optional
        Angles at which to compute the transform, in radians.
        Defaults to a vector of 180 angles evenly spaced in the
        range [-pi/2, pi/2).
    seed : int, optional
        Seed to initialize the random number generator.

    Returns
    -------
    lines : list
      List of lines identified, lines in format ((x0, y0), (x1, y1)),
      indicating line start and end.

    References
    ----------
    .. [1] C. Galamhos, J. Matas and J. Kittler, "Progressive probabilistic
           Hough transform for line detection", in IEEE Computer Society
           Conference on Computer Vision and Pattern Recognition, 1999.
    """

    if image.ndim != 2:
        raise ValueError('The input image `image` must be 2D.')

    if theta is None:
        theta = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)

    return _prob_hough_line(image, threshold=threshold, line_length=line_length,
                            line_gap=line_gap, theta=theta, seed=seed)


def hough_circle_peaks(hspaces, radii, min_xdistance=1, min_ydistance=1,
                       threshold=None, num_peaks=np.inf,
                       total_num_peaks=np.inf, normalize=False):
    """Return peaks in a circle Hough transform.

    Identifies most prominent circles separated by certain distances in given
    Hough spaces. Non-maximum suppression with different sizes is applied
    separately in the first and second dimension of the Hough space to
    identify peaks. For circles with different radius but close in distance,
    only the one with highest peak is kept.

    Parameters
    ----------
    hspaces : (N, M) array
        Hough spaces returned by the `hough_circle` function.
    radii : (M,) array
        Radii corresponding to Hough spaces.
    min_xdistance : int, optional
        Minimum distance separating centers in the x dimension.
    min_ydistance : int, optional
        Minimum distance separating centers in the y dimension.
    threshold : float, optional
        Minimum intensity of peaks in each Hough space.
        Default is `0.5 * max(hspace)`.
    num_peaks : int, optional
        Maximum number of peaks in each Hough space. When the
        number of peaks exceeds `num_peaks`, only `num_peaks`
        coordinates based on peak intensity are considered for the
        corresponding radius.
    total_num_peaks : int, optional
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` coordinates based on peak intensity.
    normalize : bool, optional
        If True, normalize the accumulator by the radius to sort the prominent
        peaks.

    Returns
    -------
    accum, cx, cy, rad : tuple of array
        Peak values in Hough space, x and y center coordinates and radii.

    Examples
    --------
    >>> from skimage import transform, draw
    >>> img = np.zeros((120, 100), dtype=int)
    >>> radius, x_0, y_0 = (20, 99, 50)
    >>> y, x = draw.circle_perimeter(y_0, x_0, radius)
    >>> img[x, y] = 1
    >>> hspaces = transform.hough_circle(img, radius)
    >>> accum, cx, cy, rad = hough_circle_peaks(hspaces, [radius,])

    Notes
    -----
    Circles with bigger radius have higher peaks in Hough space. If larger
    circles are preferred over smaller ones, `normalize` should be False.
    Otherwise, circles will be returned in the order of decreasing voting
    number.
    """
    from ..feature.peak import _prominent_peaks

    r = []
    cx = []
    cy = []
    accum = []

    for rad, hp in zip(radii, hspaces):
        h_p, x_p, y_p = _prominent_peaks(hp,
                                         min_xdistance=min_xdistance,
                                         min_ydistance=min_ydistance,
                                         threshold=threshold,
                                         num_peaks=num_peaks)
        r.extend((rad,)*len(h_p))
        cx.extend(x_p)
        cy.extend(y_p)
        accum.extend(h_p)

    r = np.array(r)
    cx = np.array(cx)
    cy = np.array(cy)
    accum = np.array(accum)
    if normalize:
        s = np.argsort(accum / r)
    else:
        s = np.argsort(accum)
    accum_sorted, cx_sorted, cy_sorted, r_sorted = \
        accum[s][::-1], cx[s][::-1], cy[s][::-1], r[s][::-1]

    tnp = len(accum_sorted) if total_num_peaks == np.inf else total_num_peaks

    # Skip searching for neighboring circles
    # if default min_xdistance and min_ydistance are used
    # or if no peak was detected
    if (min_xdistance == 1 and min_ydistance == 1) or len(accum_sorted) == 0:
        return (accum_sorted[:tnp],
                cx_sorted[:tnp],
                cy_sorted[:tnp],
                r_sorted[:tnp])

    # For circles with centers too close, only keep the one with
    # the highest peak
    should_keep = label_distant_points(
        cx_sorted, cy_sorted, min_xdistance, min_ydistance, tnp
    )
    return (accum_sorted[should_keep],
            cx_sorted[should_keep],
            cy_sorted[should_keep],
            r_sorted[should_keep])


def label_distant_points(xs, ys, min_xdistance, min_ydistance, max_points):
    """Keep points that are separated by certain distance in each dimension.

    The first point is always accpeted and all subsequent points are selected
    so that they are distant from all their preceding ones.

    Parameters
    ----------
    xs : array
        X coordinates of points.
    ys : array
        Y coordinates of points.
    min_xdistance : int
        Minimum distance separating points in the x dimension.
    min_ydistance : int
        Minimum distance separating points in the y dimension.
    max_points : int
        Max number of distant points to keep.

    Returns
    -------
    should_keep : array of bool
        A mask array for distant points to keep.
    """
    is_neighbor = np.zeros(len(xs), dtype=bool)
    coordinates = np.stack([xs, ys], axis=1)
    # Use a KDTree to search for neighboring points effectively
    kd_tree = cKDTree(coordinates)
    n_pts = 0
    for i in range(len(xs)):
        if n_pts >= max_points:
            # Ignore the point if points to keep reaches maximum
            is_neighbor[i] = True
        elif not is_neighbor[i]:
            # Find a short list of candidates to remove
            # by searching within a circle
            neighbors_i = kd_tree.query_ball_point(
                (xs[i], ys[i]),
                np.hypot(min_xdistance, min_ydistance)
            )
            # Check distance in both dimensions and mark if close
            for ni in neighbors_i:
                x_close = abs(xs[ni] - xs[i]) <= min_xdistance
                y_close = abs(ys[ni] - ys[i]) <= min_ydistance
                if x_close and y_close and ni > i:
                    is_neighbor[ni] = True
            n_pts += 1
    should_keep = ~is_neighbor
    return should_keep
