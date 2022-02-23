import numpy as np
from .dtype import dtype_limits


def invert(image, signed_float=False):
    """Invert an image.

    Invert the intensity range of the input image, so that the dtype maximum
    is now the dtype minimum, and vice-versa. This operation is
    slightly different depending on the input dtype:

    - unsigned integers: subtract the image from the dtype maximum
    - signed integers: subtract the image from -1 (see Notes)
    - floats: subtract the image from 1 (if signed_float is False, so we
      assume the image is unsigned), or from 0 (if signed_float is True).

    See the examples for clarification.

    Parameters
    ----------
    image : ndarray
        Input image.
    signed_float : bool, optional
        If True and the image is of type float, the range is assumed to
        be [-1, 1]. If False and the image is of type float, the range is
        assumed to be [0, 1].

    Returns
    -------
    inverted : ndarray
        Inverted image.

    Notes
    -----
    Ideally, for signed integers we would simply multiply by -1. However,
    signed integer ranges are asymmetric. For example, for np.int8, the range
    of possible values is [-128, 127], so that -128 * -1 equals -128! By
    subtracting from -1, we correctly map the maximum dtype value to the
    minimum.

    Examples
    --------
    >>> img = np.array([[100,  0, 200],
    ...                 [  0, 50,   0],
    ...                 [ 30,  0, 255]], np.uint8)
    >>> invert(img)
    array([[155, 255,  55],
           [255, 205, 255],
           [225, 255,   0]], dtype=uint8)
    >>> img2 = np.array([[ -2, 0, -128],
    ...                  [127, 0,    5]], np.int8)
    >>> invert(img2)
    array([[   1,   -1,  127],
           [-128,   -1,   -6]], dtype=int8)
    >>> img3 = np.array([[ 0., 1., 0.5, 0.75]])
    >>> invert(img3)
    array([[1.  , 0.  , 0.5 , 0.25]])
    >>> img4 = np.array([[ 0., 1., -1., -0.25]])
    >>> invert(img4, signed_float=True)
    array([[-0.  , -1.  ,  1.  ,  0.25]])
    """
    if image.dtype == 'bool':
        inverted = ~image
    elif np.issubdtype(image.dtype, np.unsignedinteger):
        max_val = dtype_limits(image, clip_negative=False)[1]
        inverted = np.subtract(max_val, image, dtype=image.dtype)
    elif np.issubdtype(image.dtype, np.signedinteger):
        inverted = np.subtract(-1, image, dtype=image.dtype)
    else:  # float dtype
        if signed_float:
            inverted = -image
        else:
            inverted = np.subtract(1, image, dtype=image.dtype)
    return inverted
