import math
import numpy as np

from . import (polygon as draw_polygon, disk as draw_disk,
               ellipse as draw_ellipse)
from .._shared.utils import deprecate_multichannel_kwarg, warn


def _generate_rectangle_mask(point, image, shape, random):
    """Generate a mask for a filled rectangle shape.

    The height and width of the rectangle are generated randomly.

    Parameters
    ----------
    point : tuple
        The row and column of the top left corner of the rectangle.
    image : tuple
        The height, width and depth of the image into which the shape
        is placed.
    shape : tuple
        The minimum and maximum size of the shape to fit.
    random : `numpy.random.Generator`

        The random state to use for random sampling.

    Raises
    ------
    ArithmeticError
        When a shape cannot be fit into the image with the given starting
        coordinates. This usually means the image dimensions are too small or
        shape dimensions too large.

    Returns
    -------
    label : tuple
        A (category, ((r0, r1), (c0, c1))) tuple specifying the category and
        bounding box coordinates of the shape.
    indices : 2-D array
        A mask of indices that the shape fills.

    """
    available_width = min(image[1] - point[1], shape[1]) - shape[0]
    available_height = min(image[0] - point[0], shape[1]) - shape[0]

    # Pick random widths and heights.
    r = shape[0] + random.integers(max(1, available_height)) - 1
    c = shape[0] + random.integers(max(1, available_width)) - 1
    rectangle = draw_polygon([
        point[0],
        point[0] + r,
        point[0] + r,
        point[0],
    ], [
        point[1],
        point[1],
        point[1] + c,
        point[1] + c,
    ])
    label = ('rectangle', ((point[0], point[0] + r + 1),
                           (point[1], point[1] + c + 1)))

    return rectangle, label


def _generate_circle_mask(point, image, shape, random):
    """Generate a mask for a filled circle shape.

    The radius of the circle is generated randomly.

    Parameters
    ----------
    point : tuple
        The row and column of the top left corner of the rectangle.
    image : tuple
        The height, width and depth of the image into which the shape is placed.
    shape : tuple
        The minimum and maximum size and color of the shape to fit.
    random : `numpy.random.Generator`
        The random state to use for random sampling.

    Raises
    ------
    ArithmeticError
        When a shape cannot be fit into the image with the given starting
        coordinates. This usually means the image dimensions are too small or
        shape dimensions too large.

    Returns
    -------
    label : tuple
        A (category, ((r0, r1), (c0, c1))) tuple specifying the category and
        bounding box coordinates of the shape.
    indices : 2-D array
        A mask of indices that the shape fills.
    """
    if shape[0] == 1 or shape[1] == 1:
        raise ValueError('size must be > 1 for circles')
    min_radius = shape[0] // 2.0
    max_radius = shape[1] // 2.0
    left = point[1]
    right = image[1] - point[1]
    top = point[0]
    bottom = image[0] - point[0]
    available_radius = min(left, right, top,
                           bottom, max_radius) - min_radius
    if available_radius < 0:
        raise ArithmeticError('cannot fit shape to image')
    radius = int(min_radius + random.integers(max(1, available_radius)))
    # TODO: think about how to deprecate this
    # while draw_circle was deprecated in favor of draw_disk
    # switching to a label of 'disk' here
    # would be a breaking change for downstream libraries
    # See discussion on naming convention here
    # https://github.com/scikit-image/scikit-image/pull/4428
    disk = draw_disk((point[0], point[1]), radius)
    # Until a deprecation path is decided, always return `'circle'`
    label = ('circle', ((point[0] - radius + 1, point[0] + radius),
                        (point[1] - radius + 1, point[1] + radius)))

    return disk, label


def _generate_triangle_mask(point, image, shape, random):
    """Generate a mask for a filled equilateral triangle shape.

    The length of the sides of the triangle is generated randomly.

    Parameters
    ----------
    point : tuple
        The row and column of the top left corner of a up-pointing triangle.
    image : tuple
        The height, width and depth of the image into which the shape
        is placed.
    shape : tuple
        The minimum and maximum size and color of the shape to fit.
    random : `numpy.random.Generator`
        The random state to use for random sampling.

    Raises
    ------
    ArithmeticError
        When a shape cannot be fit into the image with the given starting
        coordinates. This usually means the image dimensions are too small or
        shape dimensions too large.

    Returns
    -------
    label : tuple
        A (category, ((r0, r1), (c0, c1))) tuple specifying the category and
        bounding box coordinates of the shape.
    indices : 2-D array
        A mask of indices that the shape fills.

    """
    if shape[0] == 1 or shape[1] == 1:
        raise ValueError('dimension must be > 1 for triangles')
    available_side = min(image[1] - point[1], point[0],
                         shape[1]) - shape[0]
    side = shape[0] + random.integers(max(1, available_side)) - 1
    triangle_height = int(np.ceil(np.sqrt(3 / 4.0) * side))
    triangle = draw_polygon([
        point[0],
        point[0] - triangle_height,
        point[0],
    ], [
        point[1],
        point[1] + side // 2,
        point[1] + side,
    ])
    label = ('triangle', ((point[0] - triangle_height, point[0] + 1),
                          (point[1], point[1] + side + 1)))

    return triangle, label


def _generate_ellipse_mask(point, image, shape, random):
    """Generate a mask for a filled ellipse shape.

    The rotation, major and minor semi-axes of the ellipse are generated
    randomly.

    Parameters
    ----------
    point : tuple
        The row and column of the top left corner of the rectangle.
    image : tuple
        The height, width and depth of the image into which the shape is
        placed.
    shape : tuple
        The minimum and maximum size and color of the shape to fit.
    random : `numpy.random.Generator`
        The random state to use for random sampling.

    Raises
    ------
    ArithmeticError
        When a shape cannot be fit into the image with the given starting
        coordinates. This usually means the image dimensions are too small or
        shape dimensions too large.

    Returns
    -------
    label : tuple
        A (category, ((r0, r1), (c0, c1))) tuple specifying the category and
        bounding box coordinates of the shape.
    indices : 2-D array
        A mask of indices that the shape fills.
    """
    if shape[0] == 1 or shape[1] == 1:
        raise ValueError('size must be > 1 for ellipses')
    min_radius = shape[0] / 2.0
    max_radius = shape[1] / 2.0
    left = point[1]
    right = image[1] - point[1]
    top = point[0]
    bottom = image[0] - point[0]
    available_radius = min(left, right, top, bottom, max_radius)
    if available_radius < min_radius:
        raise ArithmeticError('cannot fit shape to image')
    # NOTE: very conservative because we could take into account the fact that
    # we have 2 different radii, but this is a good first approximation.
    # Also, we can afford to have a uniform sampling because the ellipse will
    # be rotated.
    r_radius = random.uniform(min_radius, available_radius + 1)
    c_radius = random.uniform(min_radius, available_radius + 1)
    rotation = random.uniform(-np.pi, np.pi)
    ellipse = draw_ellipse(
        point[0],
        point[1],
        r_radius,
        c_radius,
        shape=image[:2],
        rotation=rotation,
    )
    max_radius = math.ceil(max(r_radius, c_radius))
    min_x = np.min(ellipse[0])
    max_x = np.max(ellipse[0]) + 1
    min_y = np.min(ellipse[1])
    max_y = np.max(ellipse[1]) + 1
    label = ('ellipse', ((min_x, max_x), (min_y, max_y)))

    return ellipse, label


# Allows lookup by key as well as random selection.
SHAPE_GENERATORS = dict(
    rectangle=_generate_rectangle_mask,
    circle=_generate_circle_mask,
    triangle=_generate_triangle_mask,
    ellipse=_generate_ellipse_mask)
SHAPE_CHOICES = list(SHAPE_GENERATORS.values())


def _generate_random_colors(num_colors, num_channels, intensity_range, random):
    """Generate an array of random colors.

    Parameters
    ----------
    num_colors : int
        Number of colors to generate.
    num_channels : int
        Number of elements representing color.
    intensity_range : {tuple of tuples of ints, tuple of ints}, optional
        The range of values to sample pixel values from. For grayscale images
        the format is (min, max). For multichannel - ((min, max),) if the
        ranges are equal across the channels, and
        ((min_0, max_0), ... (min_N, max_N)) if they differ.
    random : `numpy.random.Generator`
        The random state to use for random sampling.

    Raises
    ------
    ValueError
        When the `intensity_range` is not in the interval (0, 255).

    Returns
    -------
    colors : array
        An array of shape (num_colors, num_channels), where the values for
        each channel are drawn from the corresponding `intensity_range`.

    """
    if num_channels == 1:
        intensity_range = (intensity_range, )
    elif len(intensity_range) == 1:
        intensity_range = intensity_range * num_channels
    colors = [random.integers(r[0], r[1] + 1, size=num_colors)
              for r in intensity_range]
    return np.transpose(colors)


@deprecate_multichannel_kwarg(multichannel_position=5)
def random_shapes(image_shape,
                  max_shapes,
                  min_shapes=1,
                  min_size=2,
                  max_size=None,
                  multichannel=True,
                  num_channels=3,
                  shape=None,
                  intensity_range=None,
                  allow_overlap=False,
                  num_trials=100,
                  random_seed=None,
                  *,
                  channel_axis=-1):
    """Generate an image with random shapes, labeled with bounding boxes.

    The image is populated with random shapes with random sizes, random
    locations, and random colors, with or without overlap.

    Shapes have random (row, col) starting coordinates and random sizes bounded
    by `min_size` and `max_size`. It can occur that a randomly generated shape
    will not fit the image at all. In that case, the algorithm will try again
    with new starting coordinates a certain number of times. However, it also
    means that some shapes may be skipped altogether. In that case, this
    function will generate fewer shapes than requested.

    Parameters
    ----------
    image_shape : tuple
        The number of rows and columns of the image to generate.
    max_shapes : int
        The maximum number of shapes to (attempt to) fit into the shape.
    min_shapes : int, optional
        The minimum number of shapes to (attempt to) fit into the shape.
    min_size : int, optional
        The minimum dimension of each shape to fit into the image.
    max_size : int, optional
        The maximum dimension of each shape to fit into the image.
    multichannel : bool, optional
        If True, the generated image has ``num_channels`` color channels,
        otherwise generates grayscale image. This argument is deprecated:
        specify `channel_axis` instead.
    num_channels : int, optional
        Number of channels in the generated image. If 1, generate monochrome
        images, else color images with multiple channels. Ignored if
        ``multichannel`` is set to False.
    shape : {rectangle, circle, triangle, ellipse, None} str, optional
        The name of the shape to generate or `None` to pick random ones.
    intensity_range : {tuple of tuples of uint8, tuple of uint8}, optional
        The range of values to sample pixel values from. For grayscale
        images the format is (min, max). For multichannel - ((min, max),)
        if the ranges are equal across the channels, and
        ((min_0, max_0), ... (min_N, max_N)) if they differ. As the
        function supports generation of uint8 arrays only, the maximum
        range is (0, 255). If None, set to (0, 254) for each channel
        reserving color of intensity = 255 for background.
    allow_overlap : bool, optional
        If `True`, allow shapes to overlap.
    num_trials : int, optional
        How often to attempt to fit a shape into the image before skipping it.
    random_seed : {None, int, `numpy.random.Generator`}, optional
        If `random_seed` is None the `numpy.random.Generator` singleton is
        used.
        If `random_seed` is an int, a new ``Generator`` instance is used,
        seeded with `random_seed`.
        If `random_seed` is already a ``Generator`` instance then that instance
        is used.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    image : uint8 array
        An image with the fitted shapes.
    labels : list
        A list of labels, one per shape in the image. Each label is a
        (category, ((r0, r1), (c0, c1))) tuple specifying the category and
        bounding box coordinates of the shape.

    Examples
    --------
    >>> import skimage.draw
    >>> image, labels = skimage.draw.random_shapes((32, 32), max_shapes=3)
    >>> image # doctest: +SKIP
    array([
       [[255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        ...,
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]]], dtype=uint8)
    >>> labels # doctest: +SKIP
    [('circle', ((22, 18), (25, 21))),
     ('triangle', ((5, 6), (13, 13)))]
    """
    if min_size > image_shape[0] or min_size > image_shape[1]:
        raise ValueError('Minimum dimension must be less than ncols and nrows')
    max_size = max_size or max(image_shape[0], image_shape[1])

    if channel_axis is None:
        num_channels = 1

    if intensity_range is None:
        intensity_range = (0, 254) if num_channels == 1 else ((0, 254), )
    else:
        tmp = (intensity_range, ) if num_channels == 1 else intensity_range
        for intensity_pair in tmp:
            for intensity in intensity_pair:
                if not (0 <= intensity <= 255):
                    msg = 'Intensity range must lie within (0, 255) interval'
                    raise ValueError(msg)

    random = np.random.default_rng(random_seed)
    user_shape = shape
    image_shape = (image_shape[0], image_shape[1], num_channels)
    image = np.full(image_shape, 255, dtype=np.uint8)
    filled = np.zeros(image_shape, dtype=bool)
    labels = []

    num_shapes = random.integers(min_shapes, max_shapes + 1)
    colors = _generate_random_colors(num_shapes, num_channels,
                                     intensity_range, random)
    shape = (min_size, max_size)
    for shape_idx in range(num_shapes):
        if user_shape is None:
            shape_generator = random.choice(SHAPE_CHOICES)
        else:
            shape_generator = SHAPE_GENERATORS[user_shape]
        for _ in range(num_trials):
            # Pick start coordinates.
            column = random.integers(max(1, image_shape[1] - min_size))
            row = random.integers(max(1, image_shape[0] - min_size))
            point = (row, column)
            try:
                indices, label = shape_generator(point, image_shape, shape,
                                                 random)
            except ArithmeticError:
                # Couldn't fit the shape, skip it.
                indices = []
                continue
            # Check if there is an overlap where the mask is nonzero.
            if allow_overlap or not filled[indices].any():
                image[indices] = colors[shape_idx]
                filled[indices] = True
                labels.append(label)
                break
        else:
            warn('Could not fit any shapes to image, '
                 'consider reducing the minimum dimension')

    if channel_axis is None:
        image = np.squeeze(image, axis=2)
    else:
        image = np.moveaxis(image, -1, channel_axis)

    return image, labels
