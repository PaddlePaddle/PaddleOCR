"""

General Description
-------------------

These filters compute the local histogram at each pixel, using a sliding window
similar to the method described in [1]_. A histogram is built using a moving
window in order to limit redundant computation. The moving window follows a
snake-like path:

...------------------------↘
↙--------------------------↙
↘--------------------------...

The local histogram is updated at each pixel as the footprint window
moves by, i.e. only those pixels entering and leaving the footprint
update the local histogram. The histogram size is 8-bit (256 bins) for 8-bit
images and 2- to 16-bit for 16-bit images depending on the maximum value of the
image.

The filter is applied up to the image border, the neighborhood used is
adjusted accordingly. The user may provide a mask image (same size as input
image) where non zero values are the part of the image participating in the
histogram computation. By default the entire image is filtered.

This implementation outperforms :func:`skimage.morphology.dilation`
for large footprints.

Input images will be cast in unsigned 8-bit integer or unsigned 16-bit integer
if necessary. The number of histogram bins is then determined from the maximum
value present in the image. Eventually, the output image is cast in the input
dtype, or the `output_dtype` if set.

To do
-----

* add simple examples, adapt documentation on existing examples
* add/check existing doc
* adapting tests for each type of filter


References
----------

.. [1] Huang, T. ,Yang, G. ;  Tang, G.. "A fast two-dimensional
       median filtering algorithm", IEEE Transactions on Acoustics, Speech and
       Signal Processing, Feb 1979. Volume: 27 , Issue: 1, Page(s): 13 - 18.

"""

import numpy as np
from scipy import ndimage as ndi

from ..._shared.utils import check_nD, deprecate_kwarg, warn
from ...util import img_as_ubyte
from . import generic_cy


__all__ = ['autolevel', 'equalize', 'gradient', 'maximum', 'mean',
           'geometric_mean', 'subtract_mean', 'median', 'minimum', 'modal',
           'enhance_contrast', 'pop', 'threshold', 'noise_filter',
           'entropy', 'otsu']


def _preprocess_input(image, footprint=None, out=None, mask=None,
                      out_dtype=None, pixel_size=1):
    """Preprocess and verify input for filters.rank methods.

    Parameters
    ----------
    image : 2-D array (integer or float)
        Input image.
    footprint : 2-D array (integer or float), optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer or float), optional
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    out_dtype : data-type, optional
        Desired output data-type. Default is None, which means we cast output
        in input dtype.
    pixel_size : int, optional
        Dimension of each pixel. Default value is 1.

    Returns
    -------
    image : 2-D array (np.uint8 or np.uint16)
    footprint : 2-D array (np.uint8)
        The neighborhood expressed as a binary 2-D array.
    out : 3-D array (same dtype out_dtype or as input)
        Output array. The two first dimensions are the spatial ones, the third
        one is the pixel vector (length 1 by default).
    mask : 2-D array (np.uint8)
        Mask array that defines (>0) area of the image included in the local
        neighborhood.
    n_bins : int
        Number of histogram bins.

    """
    check_nD(image, 2)
    input_dtype = image.dtype
    if (input_dtype in (bool, bool) or out_dtype in (bool, bool)):
        raise ValueError('dtype cannot be bool.')
    if input_dtype not in (np.uint8, np.uint16):
        message = (f'Possible precision loss converting image of type '
                   f'{input_dtype} to uint8 as required by rank filters. '
                   f'Convert manually using skimage.util.img_as_ubyte to '
                   f'silence this warning.')
        warn(message, stacklevel=5)
        image = img_as_ubyte(image)

    footprint = np.ascontiguousarray(img_as_ubyte(footprint > 0))
    if footprint.ndim != image.ndim:
        raise ValueError('Image dimensions and neighborhood dimensions'
                         'do not match')

    image = np.ascontiguousarray(image)

    if mask is not None:
        mask = img_as_ubyte(mask)
        mask = np.ascontiguousarray(mask)

    if image is out:
        raise NotImplementedError("Cannot perform rank operation in place.")

    if out is None:
        if out_dtype is None:
            out_dtype = image.dtype
        out = np.empty(image.shape + (pixel_size,), dtype=out_dtype)
    else:
        if len(out.shape) == 2:
            out = out.reshape(out.shape + (pixel_size,))

    if image.dtype in (np.uint8, np.int8):
        n_bins = 256
    else:
        # Convert to a Python int to avoid the potential overflow when we add
        # 1 to the maximum of the image.
        n_bins = int(max(3, image.max())) + 1

    if n_bins > 2 ** 10:
        warn(f'Bad rank filter performance is expected due to a '
             f'large number of bins ({n_bins}), equivalent to an approximate '
             f'bitdepth of {np.log2(n_bins):.1f}.',
             stacklevel=2)

    return image, footprint, out, mask, n_bins


def _handle_input_3D(image, footprint=None, out=None, mask=None,
                     out_dtype=None, pixel_size=1):
    """Preprocess and verify input for filters.rank methods.

    Parameters
    ----------
    image : 3-D array (integer or float)
        Input image.
    footprint : 3-D array (integer or float), optional
        The neighborhood expressed as a 3-D array of 1's and 0's.
    out : 3-D array (integer or float), optional
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    out_dtype : data-type, optional
        Desired output data-type. Default is None, which means we cast output
        in input dtype.
    pixel_size : int, optional
        Dimension of each pixel. Default value is 1.

    Returns
    -------
    image : 3-D array (np.uint8 or np.uint16)
    footprint : 3-D array (np.uint8)
        The neighborhood expressed as a binary 3-D array.
    out : 3-D array (same dtype out_dtype or as input)
        Output array. The two first dimensions are the spatial ones, the third
        one is the pixel vector (length 1 by default).
    mask : 3-D array (np.uint8)
        Mask array that defines (>0) area of the image included in the local
        neighborhood.
    n_bins : int
        Number of histogram bins.

    """
    check_nD(image, 3)
    if image.dtype not in (np.uint8, np.uint16):
        message = (f'Possible precision loss converting image of type '
                   f'{image.dtype} to uint8 as required by rank filters. '
                   f'Convert manually using skimage.util.img_as_ubyte to '
                   f'silence this warning.')
        warn(message, stacklevel=2)
        image = img_as_ubyte(image)

    footprint = np.ascontiguousarray(img_as_ubyte(footprint > 0))
    if footprint.ndim != image.ndim:
        raise ValueError('Image dimensions and neighborhood dimensions'
                         'do not match')
    image = np.ascontiguousarray(image)

    if mask is None:
        mask = np.ones(image.shape, dtype=np.uint8)
    else:
        mask = img_as_ubyte(mask)
        mask = np.ascontiguousarray(mask)

    if image is out:
        raise NotImplementedError("Cannot perform rank operation in place.")

    if out is None:
        if out_dtype is None:
            out_dtype = image.dtype
        out = np.empty(image.shape + (pixel_size,), dtype=out_dtype)
    else:
        out = out.reshape(out.shape + (pixel_size,))

    is_8bit = image.dtype in (np.uint8, np.int8)

    if is_8bit:
        n_bins = 256
    else:
        # Convert to a Python int to avoid the potential overflow when we add
        # 1 to the maximum of the image.
        n_bins = int(max(3, image.max())) + 1

    if n_bins > 2**10:
        warn(f'Bad rank filter performance is expected due to a '
             f'large number of bins ({n_bins}), equivalent to an approximate '
             f'bitdepth of {np.log2(n_bins):.1f}.',
             stacklevel=2)

    return image, footprint, out, mask, n_bins


def _apply_scalar_per_pixel(func, image, footprint, out, mask, shift_x,
                            shift_y, out_dtype=None):
    """Process the specific cython function to the image.

    Parameters
    ----------
    func : function
        Cython function to apply.
    image : 2-D array (integer or float)
        Input image.
    footprint : 2-D array (integer or float)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer or float)
        If None, a new array is allocated.
    mask : ndarray (integer or float)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    out_dtype : data-type, optional
        Desired output data-type. Default is None, which means we cast output
        in input dtype.

    """
    # preprocess and verify the input
    image, footprint, out, mask, n_bins = _preprocess_input(image, footprint,
                                                            out, mask,
                                                            out_dtype)

    # apply cython function
    func(image, footprint, shift_x=shift_x, shift_y=shift_y, mask=mask,
         out=out, n_bins=n_bins)

    return np.squeeze(out, axis=-1)


def _apply_scalar_per_pixel_3D(func, image, footprint, out, mask, shift_x,
                               shift_y, shift_z, out_dtype=None):

    image, footprint, out, mask, n_bins = _handle_input_3D(
        image, footprint, out, mask, out_dtype
    )

    func(image, footprint, shift_x=shift_x, shift_y=shift_y, shift_z=shift_z,
         mask=mask, out=out, n_bins=n_bins)

    return out.reshape(out.shape[:3])


def _apply_vector_per_pixel(func, image, footprint, out, mask, shift_x,
                            shift_y, out_dtype=None, pixel_size=1):
    """

    Parameters
    ----------
    func : function
        Cython function to apply.
    image : 2-D array (integer or float)
        Input image.
    footprint : 2-D array (integer or float)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer or float)
        If None, a new array is allocated.
    mask : ndarray (integer or float)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    out_dtype : data-type, optional
        Desired output data-type. Default is None, which means we cast output
        in input dtype.
    pixel_size : int, optional
        Dimension of each pixel.

    Returns
    -------
    out : 3-D array with float dtype of dimensions (H,W,N), where (H,W) are
        the dimensions of the input image and N is n_bins or
        ``image.max() + 1`` if no value is provided as a parameter.
        Effectively, each pixel is a N-D feature vector that is the histogram.
        The sum of the elements in the feature vector will be 1, unless no
        pixels in the window were covered by both footprint and mask, in which
        case all elements will be 0.

    """
    # preprocess and verify the input
    image, footprint, out, mask, n_bins = _preprocess_input(image, footprint,
                                                            out, mask,
                                                            out_dtype,
                                                            pixel_size)

    # apply cython function
    func(image, footprint, shift_x=shift_x, shift_y=shift_y, mask=mask,
         out=out, n_bins=n_bins)

    return out


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def autolevel(image, footprint, out=None, mask=None,
              shift_x=False, shift_y=False, shift_z=False):
    """Auto-level image using local histogram.

    This filter locally stretches the histogram of gray values to cover the
    entire range of values from "white" to "black".

    Parameters
    ----------
    image : ([P,] M, N) ndarray (uint8, uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ([P,] M, N) array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import autolevel
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> auto = autolevel(img, disk(5))
    >>> auto_vol = autolevel(volume, ball(5))

    """

    np_image = np.asanyarray(image)
    if np_image.ndim == 2:
        return _apply_scalar_per_pixel(generic_cy._autolevel, image, footprint,
                                       out=out, mask=mask,
                                       shift_x=shift_x, shift_y=shift_y)
    else:
        return _apply_scalar_per_pixel_3D(generic_cy._autolevel_3D, image,
                                          footprint, out=out, mask=mask,
                                          shift_x=shift_x, shift_y=shift_y,
                                          shift_z=shift_z)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def equalize(image, footprint, out=None, mask=None,
             shift_x=False, shift_y=False, shift_z=False):
    """Equalize image using local histogram.

    Parameters
    ----------
    image : ([P,] M, N) ndarray (uint8, uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ([P,] M, N) array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import equalize
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> equ = equalize(img, disk(5))
    >>> equ_vol = equalize(volume, ball(5))

    """

    np_image = np.asanyarray(image)
    if np_image.ndim == 2:
        return _apply_scalar_per_pixel(generic_cy._equalize, image, footprint,
                                       out=out, mask=mask,
                                       shift_x=shift_x, shift_y=shift_y)
    else:
        return _apply_scalar_per_pixel_3D(generic_cy._equalize_3D, image,
                                          footprint, out=out, mask=mask,
                                          shift_x=shift_x, shift_y=shift_y,
                                          shift_z=shift_z)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def gradient(image, footprint, out=None, mask=None,
             shift_x=False, shift_y=False, shift_z=False):
    """Return local gradient of an image (i.e. local maximum - local minimum).

    Parameters
    ----------
    image : ([P,] M, N) ndarray (uint8, uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ([P,] M, N) array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import gradient
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> out = gradient(img, disk(5))
    >>> out_vol = gradient(volume, ball(5))

    """

    np_image = np.asanyarray(image)
    if np_image.ndim == 2:
        return _apply_scalar_per_pixel(generic_cy._gradient, image, footprint,
                                       out=out, mask=mask,
                                       shift_x=shift_x, shift_y=shift_y)
    else:
        return _apply_scalar_per_pixel_3D(generic_cy._gradient_3D, image,
                                          footprint, out=out, mask=mask,
                                          shift_x=shift_x, shift_y=shift_y,
                                          shift_z=shift_z)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def maximum(image, footprint, out=None, mask=None,
            shift_x=False, shift_y=False, shift_z=False):
    """Return local maximum of an image.

    Parameters
    ----------
    image : ([P,] M, N) ndarray (uint8, uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ([P,] M, N) array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray (same dtype as input image)
        Output image.

    See also
    --------
    skimage.morphology.dilation

    Notes
    -----
    The lower algorithm complexity makes `skimage.filters.rank.maximum`
    more efficient for larger images and footprints.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import maximum
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> out = maximum(img, disk(5))
    >>> out_vol = maximum(volume, ball(5))

    """

    np_image = np.asanyarray(image)
    if np_image.ndim == 2:
        return _apply_scalar_per_pixel(generic_cy._maximum, image, footprint,
                                       out=out, mask=mask,
                                       shift_x=shift_x, shift_y=shift_y)
    else:
        return _apply_scalar_per_pixel_3D(generic_cy._maximum_3D, image,
                                          footprint, out=out, mask=mask,
                                          shift_x=shift_x, shift_y=shift_y,
                                          shift_z=shift_z)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def mean(image, footprint, out=None, mask=None,
         shift_x=False, shift_y=False, shift_z=False):
    """Return local mean of an image.

    Parameters
    ----------
    image : ([P,] M, N) ndarray (uint8, uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ([P,] M, N) array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import mean
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> avg = mean(img, disk(5))
    >>> avg_vol = mean(volume, ball(5))

    """

    np_image = np.asanyarray(image)
    if np_image.ndim == 2:
        return _apply_scalar_per_pixel(generic_cy._mean, image, footprint,
                                       out=out, mask=mask,
                                       shift_x=shift_x, shift_y=shift_y)
    else:
        return _apply_scalar_per_pixel_3D(generic_cy._mean_3D, image,
                                          footprint, out=out, mask=mask,
                                          shift_x=shift_x, shift_y=shift_y,
                                          shift_z=shift_z)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def geometric_mean(image, footprint, out=None, mask=None,
                   shift_x=False, shift_y=False, shift_z=False):
    """Return local geometric mean of an image.

    Parameters
    ----------
    image : ([P,] M, N) ndarray (uint8, uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ([P,] M, N) array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import mean
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> avg = geometric_mean(img, disk(5))
    >>> avg_vol = geometric_mean(volume, ball(5))

    References
    ----------
    .. [1] Gonzalez, R. C. and Wood, R. E. "Digital Image Processing (3rd Edition)."
           Prentice-Hall Inc, 2006.

    """

    np_image = np.asanyarray(image)
    if np_image.ndim == 2:
        return _apply_scalar_per_pixel(generic_cy._geometric_mean, image,
                                       footprint, out=out, mask=mask,
                                       shift_x=shift_x, shift_y=shift_y)
    else:
        return _apply_scalar_per_pixel_3D(generic_cy._geometric_mean_3D, image,
                                          footprint, out=out, mask=mask,
                                          shift_x=shift_x, shift_y=shift_y,
                                          shift_z=shift_z)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def subtract_mean(image, footprint, out=None, mask=None,
                  shift_x=False, shift_y=False, shift_z=False):
    """Return image subtracted from its local mean.

    Parameters
    ----------
    image : ([P,] M, N) ndarray (uint8, uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ([P,] M, N) array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray (same dtype as input image)
        Output image.

    Notes
    -----
    Subtracting the mean value may introduce underflow. To compensate
    this potential underflow, the obtained difference is downscaled by
    a factor of 2 and shifted by `n_bins / 2 - 1`, the median value of
    the local histogram (`n_bins = max(3, image.max()) +1` for 16-bits
    images and 256 otherwise).

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import subtract_mean
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> out = subtract_mean(img, disk(5))
    >>> out_vol = subtract_mean(volume, ball(5))

    """

    np_image = np.asanyarray(image)
    if np_image.ndim == 2:
        return _apply_scalar_per_pixel(generic_cy._subtract_mean, image,
                                       footprint, out=out, mask=mask,
                                       shift_x=shift_x, shift_y=shift_y)
    else:
        return _apply_scalar_per_pixel_3D(generic_cy._subtract_mean_3D, image,
                                          footprint, out=out, mask=mask,
                                          shift_x=shift_x, shift_y=shift_y,
                                          shift_z=shift_z)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def median(image, footprint=None, out=None, mask=None,
           shift_x=False, shift_y=False, shift_z=False):
    """Return local median of an image.

    Parameters
    ----------
    image : ([P,] M, N) ndarray (uint8, uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's. If None, a
        full square of size 3 is used.
    out : ([P,] M, N) array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray (same dtype as input image)
        Output image.

    See also
    --------
    skimage.filters.median : Implementation of a median filtering which handles
        images with floating precision.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import median
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> med = median(img, disk(5))
    >>> med_vol = median(volume, ball(5))

    """

    np_image = np.asanyarray(image)
    if footprint is None:
        footprint = ndi.generate_binary_structure(image.ndim, image.ndim)
    if np_image.ndim == 2:
        return _apply_scalar_per_pixel(generic_cy._median, image, footprint,
                                       out=out, mask=mask,
                                       shift_x=shift_x, shift_y=shift_y)
    else:
        return _apply_scalar_per_pixel_3D(generic_cy._median_3D, image,
                                          footprint, out=out, mask=mask,
                                          shift_x=shift_x, shift_y=shift_y,
                                          shift_z=shift_z)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def minimum(image, footprint, out=None, mask=None,
            shift_x=False, shift_y=False, shift_z=False):
    """Return local minimum of an image.

    Parameters
    ----------
    image : ([P,] M, N) ndarray (uint8, uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ([P,] M, N) array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray (same dtype as input image)
        Output image.

    See also
    --------
    skimage.morphology.erosion

    Notes
    -----
    The lower algorithm complexity makes `skimage.filters.rank.minimum` more
    efficient for larger images and footprints.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import minimum
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> out = minimum(img, disk(5))
    >>> out_vol = minimum(volume, ball(5))

    """

    np_image = np.asanyarray(image)
    if np_image.ndim == 2:
        return _apply_scalar_per_pixel(generic_cy._minimum, image, footprint,
                                       out=out, mask=mask,
                                       shift_x=shift_x, shift_y=shift_y)
    else:
        return _apply_scalar_per_pixel_3D(generic_cy._minimum_3D, image,
                                          footprint, out=out, mask=mask,
                                          shift_x=shift_x, shift_y=shift_y,
                                          shift_z=shift_z)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def modal(image, footprint, out=None, mask=None,
          shift_x=False, shift_y=False, shift_z=False):
    """Return local mode of an image.

    The mode is the value that appears most often in the local histogram.

    Parameters
    ----------
    image : ([P,] M, N) ndarray (uint8, uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ([P,] M, N) array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import modal
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> out = modal(img, disk(5))
    >>> out_vol = modal(volume, ball(5))

    """

    np_image = np.asanyarray(image)
    if np_image.ndim == 2:
        return _apply_scalar_per_pixel(generic_cy._modal, image, footprint,
                                       out=out, mask=mask,
                                       shift_x=shift_x, shift_y=shift_y)
    else:
        return _apply_scalar_per_pixel_3D(generic_cy._modal_3D, image,
                                          footprint, out=out, mask=mask,
                                          shift_x=shift_x, shift_y=shift_y,
                                          shift_z=shift_z)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def enhance_contrast(image, footprint, out=None, mask=None,
                     shift_x=False, shift_y=False, shift_z=False):
    """Enhance contrast of an image.

    This replaces each pixel by the local maximum if the pixel gray value is
    closer to the local maximum than the local minimum. Otherwise it is
    replaced by the local minimum.

    Parameters
    ----------
    image : ([P,] M, N) ndarray (uint8, uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ([P,] M, N) array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray (same dtype as input image)
        Output image

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import enhance_contrast
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> out = enhance_contrast(img, disk(5))
    >>> out_vol = enhance_contrast(volume, ball(5))

    """

    np_image = np.asanyarray(image)
    if np_image.ndim == 2:
        return _apply_scalar_per_pixel(generic_cy._enhance_contrast, image,
                                       footprint, out=out, mask=mask,
                                       shift_x=shift_x, shift_y=shift_y)
    else:
        return _apply_scalar_per_pixel_3D(generic_cy._enhance_contrast_3D,
                                          image, footprint, out=out, mask=mask,
                                          shift_x=shift_x, shift_y=shift_y,
                                          shift_z=shift_z)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def pop(image, footprint, out=None, mask=None,
        shift_x=False, shift_y=False, shift_z=False):
    """Return the local number (population) of pixels.

    The number of pixels is defined as the number of pixels which are included
    in the footprint and the mask.

    Parameters
    ----------
    image : ([P,] M, N) ndarray (uint8, uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ([P,] M, N) array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage.morphology import square, cube # Need to add 3D example
    >>> import skimage.filters.rank as rank
    >>> img = 255 * np.array([[0, 0, 0, 0, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.pop(img, square(3))
    array([[4, 6, 6, 6, 4],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [4, 6, 6, 6, 4]], dtype=uint8)

    """

    np_image = np.asanyarray(image)
    if np_image.ndim == 2:
        return _apply_scalar_per_pixel(generic_cy._pop, image, footprint,
                                       out=out, mask=mask,
                                       shift_x=shift_x, shift_y=shift_y)
    else:
        return _apply_scalar_per_pixel_3D(generic_cy._pop_3D, image,
                                          footprint, out=out, mask=mask,
                                          shift_x=shift_x, shift_y=shift_y,
                                          shift_z=shift_z)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def sum(image, footprint, out=None, mask=None,
        shift_x=False, shift_y=False, shift_z=False):
    """Return the local sum of pixels.

    Note that the sum may overflow depending on the data type of the input
    array.

    Parameters
    ----------
    image : ([P,] M, N) ndarray (uint8, uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ([P,] M, N) array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage.morphology import square, cube # Need to add 3D example
    >>> import skimage.filters.rank as rank         # Cube seems to fail but
    >>> img = np.array([[0, 0, 0, 0, 0],            # Ball can pass
    ...                 [0, 1, 1, 1, 0],
    ...                 [0, 1, 1, 1, 0],
    ...                 [0, 1, 1, 1, 0],
    ...                 [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.sum(img, square(3))
    array([[1, 2, 3, 2, 1],
           [2, 4, 6, 4, 2],
           [3, 6, 9, 6, 3],
           [2, 4, 6, 4, 2],
           [1, 2, 3, 2, 1]], dtype=uint8)

    """

    np_image = np.asanyarray(image)
    if np_image.ndim == 2:
        return _apply_scalar_per_pixel(generic_cy._sum, image, footprint,
                                       out=out, mask=mask,
                                       shift_x=shift_x, shift_y=shift_y)
    else:
        return _apply_scalar_per_pixel_3D(generic_cy._sum_3D, image,
                                          footprint, out=out, mask=mask,
                                          shift_x=shift_x, shift_y=shift_y,
                                          shift_z=shift_z)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def threshold(image, footprint, out=None, mask=None,
              shift_x=False, shift_y=False, shift_z=False):
    """Local threshold of an image.

    The resulting binary mask is True if the gray value of the center pixel is
    greater than the local mean.

    Parameters
    ----------
    image : ([P,] M, N) ndarray (uint8, uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ([P,] M, N) array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage.morphology import square, cube # Need to add 3D example
    >>> from skimage.filters.rank import threshold
    >>> img = 255 * np.array([[0, 0, 0, 0, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> threshold(img, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 0, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    """

    np_image = np.asanyarray(image)
    if np_image.ndim == 2:
        return _apply_scalar_per_pixel(generic_cy._threshold, image, footprint,
                                       out=out, mask=mask,
                                       shift_x=shift_x, shift_y=shift_y)
    else:
        return _apply_scalar_per_pixel_3D(generic_cy._threshold_3D, image,
                                          footprint, out=out, mask=mask,
                                          shift_x=shift_x, shift_y=shift_y,
                                          shift_z=shift_z)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def noise_filter(image, footprint, out=None, mask=None,
                 shift_x=False, shift_y=False, shift_z=False):
    """Noise feature.

    Parameters
    ----------
    image : ([P,] M, N) ndarray (uint8, uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ([P,] M, N) array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    References
    ----------
    .. [1] N. Hashimoto et al. Referenceless image quality evaluation
                     for whole slide imaging. J Pathol Inform 2012;3:9.

    Returns
    -------
    out : ([P,] M, N) ndarray (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import noise_filter
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> out = noise_filter(img, disk(5))
    >>> out_vol = noise_filter(volume, ball(5))

    """

    np_image = np.asanyarray(image)
    if np_image.ndim == 2:
        # ensure that the central pixel in the footprint is empty
        centre_r = int(footprint.shape[0] / 2) + shift_y
        centre_c = int(footprint.shape[1] / 2) + shift_x
        # make a local copy
        footprint_cpy = footprint.copy()
        footprint_cpy[centre_r, centre_c] = 0

        return _apply_scalar_per_pixel(generic_cy._noise_filter, image,
                                       footprint_cpy, out=out, mask=mask,
                                       shift_x=shift_x, shift_y=shift_y)
    else:
        # ensure that the central pixel in the footprint is empty
        centre_r = int(footprint.shape[0] / 2) + shift_y
        centre_c = int(footprint.shape[1] / 2) + shift_x
        centre_z = int(footprint.shape[2] / 2) + shift_z
        # make a local copy
        footprint_cpy = footprint.copy()
        footprint_cpy[centre_r, centre_c, centre_z] = 0

        return _apply_scalar_per_pixel_3D(generic_cy._noise_filter_3D,
                                          image, footprint_cpy, out=out,
                                          mask=mask, shift_x=shift_x,
                                          shift_y=shift_y, shift_z=shift_z)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def entropy(image, footprint, out=None, mask=None,
            shift_x=False, shift_y=False, shift_z=False):
    """Local entropy.

    The entropy is computed using base 2 logarithm i.e. the filter returns the
    minimum number of bits needed to encode the local gray level
    distribution.

    Parameters
    ----------
    image : ([P,] M, N) ndarray (uint8, uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ([P,] M, N) array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray (float)
        Output image.

    References
    ----------
    .. [1] `https://en.wikipedia.org/wiki/Entropy_(information_theory) <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.filters.rank import entropy
    >>> from skimage.morphology import disk, ball
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> ent = entropy(img, disk(5))
    >>> ent_vol = entropy(volume, ball(5))

    """

    np_image = np.asanyarray(image)
    if np_image.ndim == 2:
        return _apply_scalar_per_pixel(generic_cy._entropy, image, footprint,
                                       out=out, mask=mask,
                                       shift_x=shift_x, shift_y=shift_y,
                                       out_dtype=np.double)
    else:
        return _apply_scalar_per_pixel_3D(generic_cy._entropy_3D, image,
                                          footprint, out=out, mask=mask,
                                          shift_x=shift_x, shift_y=shift_y,
                                          shift_z=shift_z, out_dtype=np.double)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def otsu(image, footprint, out=None, mask=None,
         shift_x=False, shift_y=False, shift_z=False):
    """Local Otsu's threshold value for each pixel.

    Parameters
    ----------
    image : ([P,] M, N) ndarray (uint8, uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ([P,] M, N) array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray (same dtype as input image)
        Output image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Otsu's_method

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.filters.rank import otsu
    >>> from skimage.morphology import disk, ball
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> local_otsu = otsu(img, disk(5))
    >>> thresh_image = img >= local_otsu
    >>> local_otsu_vol = otsu(volume, ball(5))
    >>> thresh_image_vol = volume >= local_otsu_vol

    """

    np_image = np.asanyarray(image)
    if np_image.ndim == 2:
        return _apply_scalar_per_pixel(generic_cy._otsu, image, footprint,
                                       out=out, mask=mask,
                                       shift_x=shift_x, shift_y=shift_y)
    else:
        return _apply_scalar_per_pixel_3D(generic_cy._otsu_3D, image,
                                          footprint, out=out, mask=mask,
                                          shift_x=shift_x, shift_y=shift_y,
                                          shift_z=shift_z)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def windowed_histogram(image, footprint, out=None, mask=None,
                       shift_x=False, shift_y=False, n_bins=None):
    """Normalized sliding window histogram

    Parameters
    ----------
    image : 2-D array (integer or float)
        Input image.
    footprint : 2-D array (integer or float)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer or float), optional
        If None, a new array is allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    n_bins : int or None
        The number of histogram bins. Will default to ``image.max() + 1``
        if None is passed.

    Returns
    -------
    out : 3-D array (float)
        Array of dimensions (H,W,N), where (H,W) are the dimensions of the
        input image and N is n_bins or ``image.max() + 1`` if no value is
        provided as a parameter. Effectively, each pixel is a N-D feature
        vector that is the histogram. The sum of the elements in the feature
        vector will be 1, unless no pixels in the window were covered by both
        footprint and mask, in which case all elements will be 0.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.filters.rank import windowed_histogram
    >>> from skimage.morphology import disk, ball
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> hist_img = windowed_histogram(img, disk(5))

    """

    if n_bins is None:
        n_bins = int(image.max()) + 1

    return _apply_vector_per_pixel(generic_cy._windowed_hist, image, footprint,
                                   out=out, mask=mask,
                                   shift_x=shift_x, shift_y=shift_y,
                                   out_dtype=np.double,
                                   pixel_size=n_bins)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def majority(image, footprint, *, out=None, mask=None,
             shift_x=False, shift_y=False, shift_z=False):
    """Assign to each pixel the most common value within its neighborhood.

    Parameters
    ----------
    image : ndarray
        Image array (uint8, uint16 array).
    footprint : 2-D array (integer or float)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (integer or float), optional
        If None, a new array will be allocated.
    mask : ndarray (integer or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.filters.rank import majority
    >>> from skimage.morphology import disk, ball
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> maj_img = majority(img, disk(5))
    >>> maj_img_vol = majority(volume, ball(5))

    """

    np_image = np.asanyarray(image)
    if np_image.ndim == 2:
        return _apply_scalar_per_pixel(generic_cy._majority, image,
                                       footprint, out=out, mask=mask,
                                       shift_x=shift_x, shift_y=shift_y)
    else:
        return _apply_scalar_per_pixel_3D(generic_cy._majority_3D,
                                          image, footprint, out=out, mask=mask,
                                          shift_x=shift_x, shift_y=shift_y,
                                          shift_z=shift_z)
