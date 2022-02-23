import numpy as np

from .._shared._geometry import polygon_clip
from .._shared.version_requirements import require
from ._draw import (_coords_inside_image, _line, _line_aa,
                    _polygon, _ellipse_perimeter,
                    _circle_perimeter, _circle_perimeter_aa,
                    _bezier_curve)


def _ellipse_in_shape(shape, center, radii, rotation=0.):
    """Generate coordinates of points within ellipse bounded by shape.

    Parameters
    ----------
    shape :  iterable of ints
        Shape of the input image.  Must be at least length 2. Only the first
        two values are used to determine the extent of the input image.
    center : iterable of floats
        (row, column) position of center inside the given shape.
    radii : iterable of floats
        Size of two half axes (for row and column)
    rotation : float, optional
        Rotation of the ellipse defined by the above, in radians
        in range (-PI, PI), in contra clockwise direction,
        with respect to the column-axis.

    Returns
    -------
    rows : iterable of ints
        Row coordinates representing values within the ellipse.
    cols : iterable of ints
        Corresponding column coordinates representing values within the ellipse.
    """
    r_lim, c_lim = np.ogrid[0:float(shape[0]), 0:float(shape[1])]
    r_org, c_org = center
    r_rad, c_rad = radii
    rotation %= np.pi
    sin_alpha, cos_alpha = np.sin(rotation), np.cos(rotation)
    r, c = (r_lim - r_org), (c_lim - c_org)
    distances = ((r * cos_alpha + c * sin_alpha) / r_rad) ** 2 \
                + ((r * sin_alpha - c * cos_alpha) / c_rad) ** 2
    return np.nonzero(distances < 1)


def ellipse(r, c, r_radius, c_radius, shape=None, rotation=0.):
    """Generate coordinates of pixels within ellipse.

    Parameters
    ----------
    r, c : double
        Centre coordinate of ellipse.
    r_radius, c_radius : double
        Minor and major semi-axes. ``(r/r_radius)**2 + (c/c_radius)**2 = 1``.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output pixel
        coordinates. This is useful for ellipses which exceed the image size.
        By default the full extent of the ellipse are used. Must be at least
        length 2. Only the first two values are used to determine the extent.
    rotation : float, optional (default 0.)
        Set the ellipse rotation (rotation) in range (-PI, PI)
        in contra clock wise direction, so PI/2 degree means swap ellipse axis

    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of ellipse.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Examples
    --------
    >>> from skimage.draw import ellipse
    >>> img = np.zeros((10, 12), dtype=np.uint8)
    >>> rr, cc = ellipse(5, 6, 3, 5, rotation=np.deg2rad(30))
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    Notes
    -----
    The ellipse equation::

        ((x * cos(alpha) + y * sin(alpha)) / x_radius) ** 2 +
        ((x * sin(alpha) - y * cos(alpha)) / y_radius) ** 2 = 1


    Note that the positions of `ellipse` without specified `shape` can have
    also, negative values, as this is correct on the plane. On the other hand
    using these ellipse positions for an image afterwards may lead to appearing
    on the other side of image, because ``image[-1, -1] = image[end-1, end-1]``

    >>> rr, cc = ellipse(1, 2, 3, 6)
    >>> img = np.zeros((6, 12), dtype=np.uint8)
    >>> img[rr, cc] = 1
    >>> img
    array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]], dtype=uint8)
    """

    center = np.array([r, c])
    radii = np.array([r_radius, c_radius])
    # allow just rotation with in range +/- 180 degree
    rotation %= np.pi

    # compute rotated radii by given rotation
    r_radius_rot = abs(r_radius * np.cos(rotation)) \
                   + c_radius * np.sin(rotation)
    c_radius_rot = r_radius * np.sin(rotation) \
                   + abs(c_radius * np.cos(rotation))
    # The upper_left and lower_right corners of the smallest rectangle
    # containing the ellipse.
    radii_rot = np.array([r_radius_rot, c_radius_rot])
    upper_left = np.ceil(center - radii_rot).astype(int)
    lower_right = np.floor(center + radii_rot).astype(int)

    if shape is not None:
        # Constrain upper_left and lower_right by shape boundary.
        upper_left = np.maximum(upper_left, np.array([0, 0]))
        lower_right = np.minimum(lower_right, np.array(shape[:2]) - 1)

    shifted_center = center - upper_left
    bounding_shape = lower_right - upper_left + 1

    rr, cc = _ellipse_in_shape(bounding_shape, shifted_center, radii, rotation)
    rr.flags.writeable = True
    cc.flags.writeable = True
    rr += upper_left[0]
    cc += upper_left[1]
    return rr, cc


def disk(center, radius, *, shape=None):
    """Generate coordinates of pixels within circle.

    Parameters
    ----------
    center : tuple
        Center coordinate of disk.
    radius : double
        Radius of disk.
    shape : tuple, optional
        Image shape as a tuple of size 2. Determines the maximum
        extent of output pixel coordinates. This is useful for disks that
        exceed the image size. If None, the full extent of the disk is used.
        The  shape might result in negative coordinates and wraparound
        behaviour.

    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of disk.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.draw import disk
    >>> shape = (4, 4)
    >>> img = np.zeros(shape, dtype=np.uint8)
    >>> rr, cc = disk((0, 0), 2, shape=shape)
    >>> img[rr, cc] = 1
    >>> img
    array([[1, 1, 0, 0],
           [1, 1, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=uint8)
    >>> img = np.zeros(shape, dtype=np.uint8)
    >>> # Negative coordinates in rr and cc perform a wraparound
    >>> rr, cc = disk((0, 0), 2, shape=None)
    >>> img[rr, cc] = 1
    >>> img
    array([[1, 1, 0, 1],
           [1, 1, 0, 1],
           [0, 0, 0, 0],
           [1, 1, 0, 1]], dtype=uint8)
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = disk((4, 4), 5)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    r, c = center
    return ellipse(r, c, radius, radius, shape)


@require("matplotlib", ">=3.0.3")
def polygon_perimeter(r, c, shape=None, clip=False):
    """Generate polygon perimeter coordinates.

    Parameters
    ----------
    r : (N,) ndarray
        Row coordinates of vertices of polygon.
    c : (N,) ndarray
        Column coordinates of vertices of polygon.
    shape : tuple, optional
        Image shape which is used to determine maximum extents of output pixel
        coordinates. This is useful for polygons that exceed the image size.
        If None, the full extents of the polygon is used.  Must be at least
        length 2. Only the first two values are used to determine the extent of
        the input image.
    clip : bool, optional
        Whether to clip the polygon to the provided shape.  If this is set
        to True, the drawn figure will always be a closed polygon with all
        edges visible.

    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of polygon.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Examples
    --------
    >>> from skimage.draw import polygon_perimeter
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = polygon_perimeter([5, -1, 5, 10],
    ...                            [-1, 5, 11, 5],
    ...                            shape=img.shape, clip=True)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
           [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=uint8)

    """
    if clip:
        if shape is None:
            raise ValueError("Must specify clipping shape")
        clip_box = np.array([0, 0, shape[0] - 1, shape[1] - 1])
    else:
        clip_box = np.array([np.min(r), np.min(c),
                             np.max(r), np.max(c)])

    # Do the clipping irrespective of whether clip is set.  This
    # ensures that the returned polygon is closed and is an array.
    r, c = polygon_clip(r, c, *clip_box)

    r = np.round(r).astype(int)
    c = np.round(c).astype(int)

    # Construct line segments
    rr, cc = [], []
    for i in range(len(r) - 1):
        line_r, line_c = line(r[i], c[i], r[i + 1], c[i + 1])
        rr.extend(line_r)
        cc.extend(line_c)

    rr = np.asarray(rr)
    cc = np.asarray(cc)

    if shape is None:
        return rr, cc
    else:
        return _coords_inside_image(rr, cc, shape)


def set_color(image, coords, color, alpha=1):
    """Set pixel color in the image at the given coordinates.

    Note that this function modifies the color of the image in-place.
    Coordinates that exceed the shape of the image will be ignored.

    Parameters
    ----------
    image : (M, N, D) ndarray
        Image
    coords : tuple of ((P,) ndarray, (P,) ndarray)
        Row and column coordinates of pixels to be colored.
    color : (D,) ndarray
        Color to be assigned to coordinates in the image.
    alpha : scalar or (N,) ndarray
        Alpha values used to blend color with image.  0 is transparent,
        1 is opaque.

    Examples
    --------
    >>> from skimage.draw import line, set_color
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = line(1, 1, 20, 20)
    >>> set_color(img, (rr, cc), 1)
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=uint8)

    """
    rr, cc = coords

    if image.ndim == 2:
        image = image[..., np.newaxis]

    color = np.array(color, ndmin=1, copy=False)

    if image.shape[-1] != color.shape[-1]:
        raise ValueError(f'Color shape ({color.shape[0]}) must match last '
                          'image dimension ({image.shape[-1]}).')

    if np.isscalar(alpha):
        # Can be replaced by ``full_like`` when numpy 1.8 becomes
        # minimum dependency
        alpha = np.ones_like(rr) * alpha

    rr, cc, alpha = _coords_inside_image(rr, cc, image.shape, val=alpha)

    alpha = alpha[..., np.newaxis]

    color = color * alpha
    vals = image[rr, cc] * (1 - alpha)

    image[rr, cc] = vals + color


def line(r0, c0, r1, c1):
    """Generate line pixel coordinates.

    Parameters
    ----------
    r0, c0 : int
        Starting position (row, column).
    r1, c1 : int
        End position (row, column).

    Returns
    -------
    rr, cc : (N,) ndarray of int
        Indices of pixels that belong to the line.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Notes
    -----
    Anti-aliased line generator is available with `line_aa`.

    Examples
    --------
    >>> from skimage.draw import line
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = line(1, 1, 8, 8)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    return _line(r0, c0, r1, c1)


def line_aa(r0, c0, r1, c1):
    """Generate anti-aliased line pixel coordinates.

    Parameters
    ----------
    r0, c0 : int
        Starting position (row, column).
    r1, c1 : int
        End position (row, column).

    Returns
    -------
    rr, cc, val : (N,) ndarray (int, int, float)
        Indices of pixels (`rr`, `cc`) and intensity values (`val`).
        ``img[rr, cc] = val``.

    References
    ----------
    .. [1] A Rasterizing Algorithm for Drawing Curves, A. Zingl, 2012
           http://members.chello.at/easyfilter/Bresenham.pdf

    Examples
    --------
    >>> from skimage.draw import line_aa
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc, val = line_aa(1, 1, 8, 8)
    >>> img[rr, cc] = val * 255
    >>> img
    array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0, 255,  74,   0,   0,   0,   0,   0,   0,   0],
           [  0,  74, 255,  74,   0,   0,   0,   0,   0,   0],
           [  0,   0,  74, 255,  74,   0,   0,   0,   0,   0],
           [  0,   0,   0,  74, 255,  74,   0,   0,   0,   0],
           [  0,   0,   0,   0,  74, 255,  74,   0,   0,   0],
           [  0,   0,   0,   0,   0,  74, 255,  74,   0,   0],
           [  0,   0,   0,   0,   0,   0,  74, 255,  74,   0],
           [  0,   0,   0,   0,   0,   0,   0,  74, 255,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=uint8)
    """
    return _line_aa(r0, c0, r1, c1)


def polygon(r, c, shape=None):
    """Generate coordinates of pixels within polygon.

    Parameters
    ----------
    r : (N,) ndarray
        Row coordinates of vertices of polygon.
    c : (N,) ndarray
        Column coordinates of vertices of polygon.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for polygons that exceed the image
        size. If None, the full extent of the polygon is used.  Must be at
        least length 2. Only the first two values are used to determine the
        extent of the input image.

    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of polygon.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Examples
    --------
    >>> from skimage.draw import polygon
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> r = np.array([1, 2, 8])
    >>> c = np.array([1, 7, 4])
    >>> rr, cc = polygon(r, c)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    """
    return _polygon(r, c, shape)


def circle_perimeter(r, c, radius, method='bresenham', shape=None):
    """Generate circle perimeter coordinates.

    Parameters
    ----------
    r, c : int
        Centre coordinate of circle.
    radius : int
        Radius of circle.
    method : {'bresenham', 'andres'}, optional
        bresenham : Bresenham method (default)
        andres : Andres method
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for circles that exceed the image
        size. If None, the full extent of the circle is used.  Must be at least
        length 2. Only the first two values are used to determine the extent of
        the input image.

    Returns
    -------
    rr, cc : (N,) ndarray of int
        Bresenham and Andres' method:
        Indices of pixels that belong to the circle perimeter.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Notes
    -----
    Andres method presents the advantage that concentric
    circles create a disc whereas Bresenham can make holes. There
    is also less distortions when Andres circles are rotated.
    Bresenham method is also known as midpoint circle algorithm.
    Anti-aliased circle generator is available with `circle_perimeter_aa`.

    References
    ----------
    .. [1] J.E. Bresenham, "Algorithm for computer control of a digital
           plotter", IBM Systems journal, 4 (1965) 25-30.
    .. [2] E. Andres, "Discrete circles, rings and spheres", Computers &
           Graphics, 18 (1994) 695-706.

    Examples
    --------
    >>> from skimage.draw import circle_perimeter
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = circle_perimeter(4, 4, 3)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    return _circle_perimeter(r, c, radius, method, shape)


def circle_perimeter_aa(r, c, radius, shape=None):
    """Generate anti-aliased circle perimeter coordinates.

    Parameters
    ----------
    r, c : int
        Centre coordinate of circle.
    radius : int
        Radius of circle.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for circles that exceed the image
        size. If None, the full extent of the circle is used.  Must be at least
        length 2. Only the first two values are used to determine the extent of
        the input image.

    Returns
    -------
    rr, cc, val : (N,) ndarray (int, int, float)
        Indices of pixels (`rr`, `cc`) and intensity values (`val`).
        ``img[rr, cc] = val``.

    Notes
    -----
    Wu's method draws anti-aliased circle. This implementation doesn't use
    lookup table optimization.

    Use the function ``draw.set_color`` to apply ``circle_perimeter_aa``
    results to color images.

    References
    ----------
    .. [1] X. Wu, "An efficient antialiasing technique", In ACM SIGGRAPH
           Computer Graphics, 25 (1991) 143-152.

    Examples
    --------
    >>> from skimage.draw import circle_perimeter_aa
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc, val = circle_perimeter_aa(4, 4, 3)
    >>> img[rr, cc] = val * 255
    >>> img
    array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,  60, 211, 255, 211,  60,   0,   0,   0],
           [  0,  60, 194,  43,   0,  43, 194,  60,   0,   0],
           [  0, 211,  43,   0,   0,   0,  43, 211,   0,   0],
           [  0, 255,   0,   0,   0,   0,   0, 255,   0,   0],
           [  0, 211,  43,   0,   0,   0,  43, 211,   0,   0],
           [  0,  60, 194,  43,   0,  43, 194,  60,   0,   0],
           [  0,   0,  60, 211, 255, 211,  60,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=uint8)

    >>> from skimage import data, draw
    >>> image = data.chelsea()
    >>> rr, cc, val = draw.circle_perimeter_aa(r=100, c=100, radius=75)
    >>> draw.set_color(image, (rr, cc), [1, 0, 0], alpha=val)
    """
    return _circle_perimeter_aa(r, c, radius, shape)


def ellipse_perimeter(r, c, r_radius, c_radius, orientation=0, shape=None):
    """Generate ellipse perimeter coordinates.

    Parameters
    ----------
    r, c : int
        Centre coordinate of ellipse.
    r_radius, c_radius : int
        Minor and major semi-axes. ``(r/r_radius)**2 + (c/c_radius)**2 = 1``.
    orientation : double, optional
        Major axis orientation in clockwise direction as radians.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for ellipses that exceed the image
        size. If None, the full extent of the ellipse is used.  Must be at
        least length 2. Only the first two values are used to determine the
        extent of the input image.

    Returns
    -------
    rr, cc : (N,) ndarray of int
        Indices of pixels that belong to the ellipse perimeter.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    References
    ----------
    .. [1] A Rasterizing Algorithm for Drawing Curves, A. Zingl, 2012
           http://members.chello.at/easyfilter/Bresenham.pdf

    Examples
    --------
    >>> from skimage.draw import ellipse_perimeter
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = ellipse_perimeter(5, 5, 3, 4)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)


    Note that the positions of `ellipse` without specified `shape` can have
    also, negative values, as this is correct on the plane. On the other hand
    using these ellipse positions for an image afterwards may lead to appearing
    on the other side of image, because ``image[-1, -1] = image[end-1, end-1]``

    >>> rr, cc = ellipse_perimeter(2, 3, 4, 5)
    >>> img = np.zeros((9, 12), dtype=np.uint8)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    return _ellipse_perimeter(r, c, r_radius, c_radius, orientation, shape)


def bezier_curve(r0, c0, r1, c1, r2, c2, weight, shape=None):
    """Generate Bezier curve coordinates.

    Parameters
    ----------
    r0, c0 : int
        Coordinates of the first control point.
    r1, c1 : int
        Coordinates of the middle control point.
    r2, c2 : int
        Coordinates of the last control point.
    weight : double
        Middle control point weight, it describes the line tension.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for curves that exceed the image
        size. If None, the full extent of the curve is used.

    Returns
    -------
    rr, cc : (N,) ndarray of int
        Indices of pixels that belong to the Bezier curve.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Notes
    -----
    The algorithm is the rational quadratic algorithm presented in
    reference [1]_.

    References
    ----------
    .. [1] A Rasterizing Algorithm for Drawing Curves, A. Zingl, 2012
           http://members.chello.at/easyfilter/Bresenham.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.draw import bezier_curve
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = bezier_curve(1, 5, 5, -2, 8, 8, 2)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    return _bezier_curve(r0, c0, r1, c1, r2, c2, weight, shape)


def rectangle(start, end=None, extent=None, shape=None):
    """Generate coordinates of pixels within a rectangle.

    Parameters
    ----------
    start : tuple
        Origin point of the rectangle, e.g., ``([plane,] row, column)``.
    end : tuple
        End point of the rectangle ``([plane,] row, column)``.
        For a 2D matrix, the slice defined by the rectangle is
        ``[start:(end+1)]``.
        Either `end` or `extent` must be specified.
    extent : tuple
        The extent (size) of the drawn rectangle.  E.g.,
        ``([num_planes,] num_rows, num_cols)``.
        Either `end` or `extent` must be specified.
        A negative extent is valid, and will result in a rectangle
        going along the opposite direction. If extent is negative, the
        `start` point is not included.
    shape : tuple, optional
        Image shape used to determine the maximum bounds of the output
        coordinates. This is useful for clipping rectangles that exceed
        the image size. By default, no clipping is done.

    Returns
    -------
    coords : array of int, shape (Ndim, Npoints)
        The coordinates of all pixels in the rectangle.

    Notes
    -----
    This function can be applied to N-dimensional images, by passing `start` and
    `end` or `extent` as tuples of length N.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.draw import rectangle
    >>> img = np.zeros((5, 5), dtype=np.uint8)
    >>> start = (1, 1)
    >>> extent = (3, 3)
    >>> rr, cc = rectangle(start, extent=extent, shape=img.shape)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)


    >>> img = np.zeros((5, 5), dtype=np.uint8)
    >>> start = (0, 1)
    >>> end = (3, 3)
    >>> rr, cc = rectangle(start, end=end, shape=img.shape)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    >>> import numpy as np
    >>> from skimage.draw import rectangle
    >>> img = np.zeros((6, 6), dtype=np.uint8)
    >>> start = (3, 3)
    >>>
    >>> rr, cc = rectangle(start, extent=(2, 2))
    >>> img[rr, cc] = 1
    >>> rr, cc = rectangle(start, extent=(-2, 2))
    >>> img[rr, cc] = 2
    >>> rr, cc = rectangle(start, extent=(-2, -2))
    >>> img[rr, cc] = 3
    >>> rr, cc = rectangle(start, extent=(2, -2))
    >>> img[rr, cc] = 4
    >>> print(img)
    [[0 0 0 0 0 0]
     [0 3 3 2 2 0]
     [0 3 3 2 2 0]
     [0 4 4 1 1 0]
     [0 4 4 1 1 0]
     [0 0 0 0 0 0]]

    """
    tl, br = _rectangle_slice(start=start, end=end, extent=extent)

    if shape is not None:
        n_dim = len(start)
        br = np.minimum(shape[0:n_dim], br)
        tl = np.maximum(np.zeros_like(shape[0:n_dim]), tl)
    coords = np.meshgrid(*[np.arange(st, en) for st, en in zip(tuple(tl),
                                                               tuple(br))])
    return coords


@require("matplotlib", ">=3.0.3")
def rectangle_perimeter(start, end=None, extent=None, shape=None, clip=False):
    """Generate coordinates of pixels that are exactly around a rectangle.

    Parameters
    ----------
    start : tuple
        Origin point of the inner rectangle, e.g., ``(row, column)``.
    end : tuple
        End point of the inner rectangle ``(row, column)``.
        For a 2D matrix, the slice defined by inner the rectangle is
        ``[start:(end+1)]``.
        Either `end` or `extent` must be specified.
    extent : tuple
        The extent (size) of the inner rectangle.  E.g.,
        ``(num_rows, num_cols)``.
        Either `end` or `extent` must be specified.
        Negative extents are permitted. See `rectangle` to better
        understand how they behave.
    shape : tuple, optional
        Image shape used to determine the maximum bounds of the output
        coordinates. This is useful for clipping perimeters that exceed
        the image size. By default, no clipping is done.  Must be at least
        length 2. Only the first two values are used to determine the extent of
        the input image.
    clip : bool, optional
        Whether to clip the perimeter to the provided shape. If this is set
        to True, the drawn figure will always be a closed polygon with all
        edges visible.

    Returns
    -------
    coords : array of int, shape (2, Npoints)
        The coordinates of all pixels in the rectangle.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.draw import rectangle_perimeter
    >>> img = np.zeros((5, 6), dtype=np.uint8)
    >>> start = (2, 3)
    >>> end = (3, 4)
    >>> rr, cc = rectangle_perimeter(start, end=end, shape=img.shape)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1],
           [0, 0, 1, 0, 0, 1],
           [0, 0, 1, 0, 0, 1],
           [0, 0, 1, 1, 1, 1]], dtype=uint8)

    >>> img = np.zeros((5, 5), dtype=np.uint8)
    >>> r, c = rectangle_perimeter(start, (10, 10), shape=img.shape, clip=True)
    >>> img[r, c] = 1
    >>> img
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1],
           [0, 0, 1, 0, 1],
           [0, 0, 1, 0, 1],
           [0, 0, 1, 1, 1]], dtype=uint8)

    """
    top_left, bottom_right = _rectangle_slice(start=start,
                                              end=end,
                                              extent=extent)

    top_left -= 1
    r = [top_left[0], top_left[0], bottom_right[0], bottom_right[0],
         top_left[0]]
    c = [top_left[1], bottom_right[1], bottom_right[1], top_left[1],
         top_left[1]]
    return polygon_perimeter(r, c, shape=shape, clip=clip)


def _rectangle_slice(start, end=None, extent=None):
    """Return the slice ``(top_left, bottom_right)`` of the rectangle.

    Returns
    =======
    (top_left, bottomm_right)
        The slice you would need to select the region in the rectangle defined
        by the parameters.
        Select it like:

        ``rect[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]``
    """
    if end is None and extent is None:
        raise ValueError("Either `end` or `extent` must be given.")
    if end is not None and extent is not None:
        raise ValueError("Cannot provide both `end` and `extent`.")

    if extent is not None:
        end = np.asarray(start) + np.asarray(extent)
    top_left = np.minimum(start, end)
    bottom_right = np.maximum(start, end)

    if extent is None:
        bottom_right += 1

    return (top_left, bottom_right)
