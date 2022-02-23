
from itertools import chain
from operator import add

import numpy as np

from ._haar import haar_like_feature_coord_wrapper
from ._haar import haar_like_feature_wrapper
from ..color import gray2rgb
from ..draw import rectangle
from ..util import img_as_float

FEATURE_TYPE = ('type-2-x', 'type-2-y',
                'type-3-x', 'type-3-y',
                'type-4')


def _validate_feature_type(feature_type):
    """Transform feature type to an iterable and check that it exists."""
    if feature_type is None:
        feature_type_ = FEATURE_TYPE
    else:
        if isinstance(feature_type, str):
            feature_type_ = [feature_type]
        else:
            feature_type_ = feature_type
        for feat_t in feature_type_:
            if feat_t not in FEATURE_TYPE:
                raise ValueError(
                                f'The given feature type is unknown. Got {feat_t} instead of one '
                                f'of {FEATURE_TYPE}.')
    return feature_type_


def haar_like_feature_coord(width, height, feature_type=None):
    """Compute the coordinates of Haar-like features.

    Parameters
    ----------
    width : int
        Width of the detection window.
    height : int
        Height of the detection window.
    feature_type : str or list of str or None, optional
        The type of feature to consider:

        - 'type-2-x': 2 rectangles varying along the x axis;
        - 'type-2-y': 2 rectangles varying along the y axis;
        - 'type-3-x': 3 rectangles varying along the x axis;
        - 'type-3-y': 3 rectangles varying along the y axis;
        - 'type-4': 4 rectangles varying along x and y axis.

        By default all features are extracted.

    Returns
    -------
    feature_coord : (n_features, n_rectangles, 2, 2), ndarray of list of \
tuple coord
        Coordinates of the rectangles for each feature.
    feature_type : (n_features,), ndarray of str
        The corresponding type for each feature.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.transform import integral_image
    >>> from skimage.feature import haar_like_feature_coord
    >>> feat_coord, feat_type = haar_like_feature_coord(2, 2, 'type-4')
    >>> feat_coord # doctest: +SKIP
    array([ list([[(0, 0), (0, 0)], [(0, 1), (0, 1)],
                  [(1, 1), (1, 1)], [(1, 0), (1, 0)]])], dtype=object)
    >>> feat_type
    array(['type-4'], dtype=object)

    """
    feature_type_ = _validate_feature_type(feature_type)

    feat_coord, feat_type = zip(*[haar_like_feature_coord_wrapper(width,
                                                                  height,
                                                                  feat_t)
                                  for feat_t in feature_type_])

    return np.concatenate(feat_coord), np.hstack(feat_type)


def haar_like_feature(int_image, r, c, width, height, feature_type=None,
                      feature_coord=None):
    """Compute the Haar-like features for a region of interest (ROI) of an
    integral image.

    Haar-like features have been successfully used for image classification and
    object detection [1]_. It has been used for real-time face detection
    algorithm proposed in [2]_.

    Parameters
    ----------
    int_image : (M, N) ndarray
        Integral image for which the features need to be computed.
    r : int
        Row-coordinate of top left corner of the detection window.
    c : int
        Column-coordinate of top left corner of the detection window.
    width : int
        Width of the detection window.
    height : int
        Height of the detection window.
    feature_type : str or list of str or None, optional
        The type of feature to consider:

        - 'type-2-x': 2 rectangles varying along the x axis;
        - 'type-2-y': 2 rectangles varying along the y axis;
        - 'type-3-x': 3 rectangles varying along the x axis;
        - 'type-3-y': 3 rectangles varying along the y axis;
        - 'type-4': 4 rectangles varying along x and y axis.

        By default all features are extracted.

        If using with `feature_coord`, it should correspond to the feature
        type of each associated coordinate feature.
    feature_coord : ndarray of list of tuples or None, optional
        The array of coordinates to be extracted. This is useful when you want
        to recompute only a subset of features. In this case `feature_type`
        needs to be an array containing the type of each feature, as returned
        by :func:`haar_like_feature_coord`. By default, all coordinates are
        computed.

    Returns
    -------
    haar_features : (n_features,) ndarray of int or float
        Resulting Haar-like features. Each value is equal to the subtraction of
        sums of the positive and negative rectangles. The data type depends of
        the data type of `int_image`: `int` when the data type of `int_image`
        is `uint` or `int` and `float` when the data type of `int_image` is
        `float`.

    Notes
    -----
    When extracting those features in parallel, be aware that the choice of the
    backend (i.e. multiprocessing vs threading) will have an impact on the
    performance. The rule of thumb is as follows: use multiprocessing when
    extracting features for all possible ROI in an image; use threading when
    extracting the feature at specific location for a limited number of ROIs.
    Refer to the example
    :ref:`sphx_glr_auto_examples_applications_plot_haar_extraction_selection_classification.py`
    for more insights.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.transform import integral_image
    >>> from skimage.feature import haar_like_feature
    >>> img = np.ones((5, 5), dtype=np.uint8)
    >>> img_ii = integral_image(img)
    >>> feature = haar_like_feature(img_ii, 0, 0, 5, 5, 'type-3-x')
    >>> feature
    array([-1, -2, -3, -4, -5, -1, -2, -3, -4, -5, -1, -2, -3, -4, -5, -1, -2,
           -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -1, -2, -3, -1,
           -2, -3, -1, -2, -1, -2, -1, -2, -1, -1, -1])

    You can compute the feature for some pre-computed coordinates.

    >>> from skimage.feature import haar_like_feature_coord
    >>> feature_coord, feature_type = zip(
    ...     *[haar_like_feature_coord(5, 5, feat_t)
    ...       for feat_t in ('type-2-x', 'type-3-x')])
    >>> # only select one feature over two
    >>> feature_coord = np.concatenate([x[::2] for x in feature_coord])
    >>> feature_type = np.concatenate([x[::2] for x in feature_type])
    >>> feature = haar_like_feature(img_ii, 0, 0, 5, 5,
    ...                             feature_type=feature_type,
    ...                             feature_coord=feature_coord)
    >>> feature
    array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -3, -5, -2, -4, -1,
           -3, -5, -2, -4, -2, -4, -2, -4, -2, -1, -3, -2, -1, -1, -1, -1, -1])

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Haar-like_feature
    .. [2] Oren, M., Papageorgiou, C., Sinha, P., Osuna, E., & Poggio, T.
           (1997, June). Pedestrian detection using wavelet templates.
           In Computer Vision and Pattern Recognition, 1997. Proceedings.,
           1997 IEEE Computer Society Conference on (pp. 193-199). IEEE.
           http://tinyurl.com/y6ulxfta
           :DOI:`10.1109/CVPR.1997.609319`
    .. [3] Viola, Paul, and Michael J. Jones. "Robust real-time face
           detection." International journal of computer vision 57.2
           (2004): 137-154.
           https://www.merl.com/publications/docs/TR2004-043.pdf
           :DOI:`10.1109/CVPR.2001.990517`

    """
    if feature_coord is None:
        feature_type_ = _validate_feature_type(feature_type)

        return np.hstack(list(chain.from_iterable(
            haar_like_feature_wrapper(int_image, r, c, width, height, feat_t,
                                      feature_coord)
            for feat_t in feature_type_)))
    else:
        if feature_coord.shape[0] != feature_type.shape[0]:
            raise ValueError("Inconsistent size between feature coordinates"
                             "and feature types.")

        mask_feature = [feature_type == feat_t for feat_t in FEATURE_TYPE]
        haar_feature_idx, haar_feature = zip(
            *[(np.flatnonzero(mask),
               haar_like_feature_wrapper(int_image, r, c, width, height,
                                         feat_t, feature_coord[mask]))
              for mask, feat_t in zip(mask_feature, FEATURE_TYPE)
              if np.count_nonzero(mask)])

        haar_feature_idx = np.concatenate(haar_feature_idx)
        haar_feature = np.concatenate(haar_feature)

        haar_feature[haar_feature_idx] = haar_feature.copy()
        return haar_feature


def draw_haar_like_feature(image, r, c, width, height,
                           feature_coord,
                           color_positive_block=(1., 0., 0.),
                           color_negative_block=(0., 1., 0.),
                           alpha=0.5, max_n_features=None, random_state=None):
    """Visualization of Haar-like features.

    Parameters
    ----------
    image : (M, N) ndarray
        The region of an integral image for which the features need to be
        computed.
    r : int
        Row-coordinate of top left corner of the detection window.
    c : int
        Column-coordinate of top left corner of the detection window.
    width : int
        Width of the detection window.
    height : int
        Height of the detection window.
    feature_coord : ndarray of list of tuples or None, optional
        The array of coordinates to be extracted. This is useful when you want
        to recompute only a subset of features. In this case `feature_type`
        needs to be an array containing the type of each feature, as returned
        by :func:`haar_like_feature_coord`. By default, all coordinates are
        computed.
    color_positive_block : tuple of 3 floats
        Floats specifying the color for the positive block. Corresponding
        values define (R, G, B) values. Default value is red (1, 0, 0).
    color_negative_block : tuple of 3 floats
        Floats specifying the color for the negative block Corresponding values
        define (R, G, B) values. Default value is blue (0, 1, 0).
    alpha : float
        Value in the range [0, 1] that specifies opacity of visualization. 1 -
        fully transparent, 0 - opaque.
    max_n_features : int, default=None
        The maximum number of features to be returned.
        By default, all features are returned.
    random_state : {None, int, `numpy.random.Generator`}, optional
        If `random_state` is None the `numpy.random.Generator` singleton is
        used.
        If `random_state` is an int, a new ``Generator`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` instance then that
        instance is used.

        The random state is used when generating a set of
        features smaller than the total number of available features.

    Returns
    -------
    features : (M, N), ndarray
        An image in which the different features will be added.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.feature import haar_like_feature_coord
    >>> from skimage.feature import draw_haar_like_feature
    >>> feature_coord, _ = haar_like_feature_coord(2, 2, 'type-4')
    >>> image = draw_haar_like_feature(np.zeros((2, 2)),
    ...                                0, 0, 2, 2,
    ...                                feature_coord,
    ...                                max_n_features=1)
    >>> image
    array([[[0. , 0.5, 0. ],
            [0.5, 0. , 0. ]],
    <BLANKLINE>
           [[0.5, 0. , 0. ],
            [0. , 0.5, 0. ]]])

    """
    random_state = np.random.default_rng(random_state)
    color_positive_block = np.asarray(color_positive_block, dtype=np.float64)
    color_negative_block = np.asarray(color_negative_block, dtype=np.float64)

    if max_n_features is None:
        feature_coord_ = feature_coord
    else:
        feature_coord_ = random_state.choice(feature_coord,
                                             size=max_n_features,
                                             replace=False)

    output = np.copy(image)
    if len(image.shape) < 3:
        output = gray2rgb(image)
    output = img_as_float(output)

    for coord in feature_coord_:
        for idx_rect, rect in enumerate(coord):
            coord_start, coord_end = rect
            coord_start = tuple(map(add, coord_start, [r, c]))
            coord_end = tuple(map(add, coord_end, [r, c]))
            rr, cc = rectangle(coord_start, coord_end)

            if ((idx_rect + 1) % 2) == 0:
                new_value = ((1 - alpha) *
                             output[rr, cc] + alpha * color_positive_block)
            else:
                new_value = ((1 - alpha) *
                             output[rr, cc] + alpha * color_negative_block)
            output[rr, cc] = new_value

    return output
