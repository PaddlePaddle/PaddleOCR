"""
Augmenters that perform simple arithmetic changes.

List of augmenters:

    * :class:`Add`
    * :class:`AddElementwise`
    * :class:`AdditiveGaussianNoise`
    * :class:`AdditiveLaplaceNoise`
    * :class:`AdditivePoissonNoise`
    * :class:`Multiply`
    * :class:`MultiplyElementwise`
    * :class:`Cutout`
    * :class:`Dropout`
    * :class:`CoarseDropout`
    * :class:`Dropout2d`
    * :class:`TotalDropout`
    * :class:`ReplaceElementwise`
    * :class:`ImpulseNoise`
    * :class:`SaltAndPepper`
    * :class:`CoarseSaltAndPepper`
    * :class:`Salt`
    * :class:`CoarseSalt`
    * :class:`Pepper`
    * :class:`CoarsePepper`
    * :class:`Invert`
    * :class:`Solarize`
    * :class:`ContrastNormalization`
    * :class:`JpegCompression`

"""
from __future__ import print_function, division, absolute_import

import tempfile

import imageio
import numpy as np

import imgaug as ia
from . import meta
from .. import parameters as iap
from .. import dtypes as iadt
from .. import random as iarandom


# fill modes for apply_cutout_() and Cutout augmenter
# contains roughly:
#     'str fill_mode_name => (str module_name, str function_name)'
# We could also assign the function to each fill mode name instead of its
# name, but that has the disadvantage that these aren't defined yet (they
# are defined further below) and that during unittesting they would be harder
# to mock. (mock.patch() seems to not automatically replace functions
# assigned in that way.)
_CUTOUT_FILL_MODES = {
    "constant": ("imgaug.augmenters.arithmetic", "_fill_rectangle_constant_"),
    "gaussian": ("imgaug.augmenters.arithmetic", "_fill_rectangle_gaussian_")
}


def add_scalar(image, value):
    """Add a single scalar value or one scalar value per channel to an image.

    This method ensures that ``uint8`` does not overflow during the addition.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: limited; tested (1)
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: limited; tested (1)
        * ``int16``: limited; tested (1)
        * ``int32``: no
        * ``int64``: no
        * ``float16``: limited; tested (1)
        * ``float32``: limited; tested (1)
        * ``float64``: no
        * ``float128``: no
        * ``bool``: limited; tested (1)

        - (1) Non-uint8 dtypes can overflow. For floats, this can result
              in +/-inf.

    Parameters
    ----------
    image : ndarray
        Image array of shape ``(H,W,[C])``.
        If `value` contains more than one value, the shape of the image is
        expected to be ``(H,W,C)``.

    value : number or ndarray
        The value to add to the image. Either a single value or an array
        containing exactly one component per channel, i.e. ``C`` components.

    Returns
    -------
    ndarray
        Image with value added to it.

    """
    if image.size == 0:
        return np.copy(image)

    iadt.gate_dtypes(
        image,
        allowed=["bool",
                 "uint8", "uint16",
                 "int8", "int16",
                 "float16", "float32"],
        disallowed=["uint32", "uint64", "uint128", "uint256",
                    "int32", "int64", "int128", "int256",
                    "float64", "float96", "float128",
                    "float256"],
        augmenter=None)

    if image.dtype.name == "uint8":
        return _add_scalar_to_uint8(image, value)
    return _add_scalar_to_non_uint8(image, value)


def _add_scalar_to_uint8(image, value):
    # Using this LUT approach is significantly faster than using
    # numpy-based adding with dtype checks (around 3-4x speedup) and is
    # still faster than the simple numpy image+sample approach without LUT
    # (about 10% at 64x64 and about 2x at 224x224 -- maybe dependent on
    # installed BLAS libraries?)

    # pylint: disable=no-else-return

    is_single_value = (
        ia.is_single_number(value)
        or ia.is_np_scalar(value)
        or (ia.is_np_array(value) and value.size == 1))
    is_channelwise = not is_single_value
    nb_channels = 1 if image.ndim == 2 else image.shape[-1]

    value = np.clip(np.round(value), -255, 255).astype(np.int16)
    value_range = np.arange(0, 256, dtype=np.int16)

    if is_channelwise:
        assert value.ndim == 1, (
            "Expected `value` to be 1-dimensional, got %d-dimensional "
            "data with shape %s." % (value.ndim, value.shape))
        assert image.ndim == 3, (
            "Expected `image` to be 3-dimensional when adding one value per "
            "channel, got %d-dimensional data with shape %s." % (
                image.ndim, image.shape))
        assert image.shape[-1] == value.size, (
            "Expected number of channels in `image` and number of components "
            "in `value` to be identical. Got %d vs. %d." % (
                image.shape[-1], value.size))

        # TODO check if tile() is here actually needed
        tables = np.tile(
            value_range[:, np.newaxis],
            (1, nb_channels)
        ) + value[np.newaxis, :]
    else:
        tables = value_range + value
    tables = np.clip(tables, 0, 255).astype(image.dtype)
    return ia.apply_lut(image, tables)


def _add_scalar_to_non_uint8(image, value):
    input_dtype = image.dtype

    is_single_value = (
        ia.is_single_number(value)
        or ia.is_np_scalar(value)
        or (ia.is_np_array(value) and value.size == 1))
    is_channelwise = not is_single_value
    nb_channels = 1 if image.ndim == 2 else image.shape[-1]

    shape = (1, 1, nb_channels if is_channelwise else 1)
    value = np.array(value).reshape(shape)

    # We limit here the value range of the value parameter to the
    # bytes in the image's dtype. This prevents overflow problems
    # and makes it less likely that the image has to be up-casted,
    # which again improves performance and saves memory. Note that
    # this also enables more dtypes for image inputs.
    # The downside is that the mul parameter is limited in its
    # value range.
    #
    # We need 2* the itemsize of the image here to allow to shift
    # the image's max value to the lowest possible value, e.g. for
    # uint8 it must allow for -255 to 255.
    itemsize = image.dtype.itemsize * 2
    dtype_target = np.dtype("%s%d" % (value.dtype.kind, itemsize))
    value = iadt.clip_to_dtype_value_range_(
        value, dtype_target, validate=True)

    # Itemsize is currently reduced from 2 to 1 due to clip no
    # longer supporting int64, which can cause issues with int32
    # samples (32*2 = 64bit).
    # TODO limit value ranges of samples to int16/uint16 for
    #      security
    image, value = iadt.promote_array_dtypes_(
        [image, value],
        dtypes=[image.dtype, dtype_target],
        increase_itemsize_factor=1)
    image = np.add(image, value, out=image, casting="no")

    return iadt.restore_dtypes_(image, input_dtype)


def add_elementwise(image, values):
    """Add an array of values to an image.

    This method ensures that ``uint8`` does not overflow during the addition.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: limited; tested (1)
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: limited; tested (1)
        * ``int16``: limited; tested (1)
        * ``int32``: no
        * ``int64``: no
        * ``float16``: limited; tested (1)
        * ``float32``: limited; tested (1)
        * ``float64``: no
        * ``float128``: no
        * ``bool``: limited; tested (1)

        - (1) Non-uint8 dtypes can overflow. For floats, this can result
              in +/-inf.

    Parameters
    ----------
    image : ndarray
        Image array of shape ``(H,W,[C])``.

    values : ndarray
        The values to add to the image. Expected to have the same height
        and width as `image` and either no channels or one channel or
        the same number of channels as `image`.

    Returns
    -------
    ndarray
        Image with values added to it.

    """
    iadt.gate_dtypes(
        image,
        allowed=["bool",
                 "uint8", "uint16",
                 "int8", "int16",
                 "float16", "float32"],
        disallowed=["uint32", "uint64", "uint128", "uint256",
                    "int32", "int64", "int128", "int256",
                    "float64", "float96", "float128",
                    "float256"],
        augmenter=None)

    if image.dtype.name == "uint8":
        return _add_elementwise_to_uint8(image, values)
    return _add_elementwise_to_non_uint8(image, values)


def _add_elementwise_to_uint8(image, values):
    # This special uint8 block is around 60-100% faster than the
    # corresponding non-uint8 function further below (more speedup
    # for smaller images).
    #
    # Also tested to instead compute min/max of image and value
    # and then only convert image/value dtype if actually
    # necessary, but that was like 20-30% slower, even for 224x224
    # images.
    #
    if values.dtype.kind == "f":
        values = np.round(values)

    image = image.astype(np.int16)
    values = np.clip(values, -255, 255).astype(np.int16)

    image_aug = image + values
    image_aug = np.clip(image_aug, 0, 255).astype(np.uint8)

    return image_aug


def _add_elementwise_to_non_uint8(image, values):
    # We limit here the value range of the value parameter to the
    # bytes in the image's dtype. This prevents overflow problems
    # and makes it less likely that the image has to be up-casted,
    # which again improves performance and saves memory. Note that
    # this also enables more dtypes for image inputs.
    # The downside is that the mul parameter is limited in its
    # value range.
    #
    # We need 2* the itemsize of the image here to allow to shift
    # the image's max value to the lowest possible value, e.g. for
    # uint8 it must allow for -255 to 255.
    input_shape = image.shape
    input_dtype = image.dtype

    if image.ndim == 2:
        image = image[..., np.newaxis]
    if values.ndim == 2:
        values = values[..., np.newaxis]
    nb_channels = image.shape[-1]

    itemsize = image.dtype.itemsize * 2
    dtype_target = np.dtype("%s%d" % (values.dtype.kind, itemsize))
    values = iadt.clip_to_dtype_value_range_(values, dtype_target,
                                             validate=100)

    if values.shape[2] == 1:
        values = np.tile(values, (1, 1, nb_channels))

    # Decreased itemsize from 2 to 1 here, see explanation in Add.
    image, values = iadt.promote_array_dtypes_(
        [image, values],
        dtypes=[image.dtype, dtype_target],
        increase_itemsize_factor=1)
    image = np.add(image, values, out=image, casting="no")
    image = iadt.restore_dtypes_(image, input_dtype)

    if len(input_shape) == 2:
        return image[..., 0]
    return image


def multiply_scalar(image, multiplier):
    """Multiply an image by a single scalar or one scalar per channel.

    This method ensures that ``uint8`` does not overflow during the
    multiplication.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: limited; tested (1)
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: limited; tested (1)
        * ``int16``: limited; tested (1)
        * ``int32``: no
        * ``int64``: no
        * ``float16``: limited; tested (1)
        * ``float32``: limited; tested (1)
        * ``float64``: no
        * ``float128``: no
        * ``bool``: limited; tested (1)

        - (1) Non-uint8 dtypes can overflow. For floats, this can result in
              +/-inf.

    note::

        Tests were only conducted for rather small multipliers, around
        ``-10.0`` to ``+10.0``.

        In general, the multipliers sampled from `multiplier` must be in a
        value range that corresponds to the input image's dtype. E.g. if the
        input image has dtype ``uint16`` and the samples generated from
        `multiplier` are ``float64``, this function will still force all
        samples to be within the value range of ``float16``, as it has the
        same number of bytes (two) as ``uint16``. This is done to make
        overflows less likely to occur.

    Parameters
    ----------
    image : ndarray
        Image array of shape ``(H,W,[C])``.
        If `value` contains more than one value, the shape of the image is
        expected to be ``(H,W,C)``.

    multiplier : number or ndarray
        The multiplier to use. Either a single value or an array
        containing exactly one component per channel, i.e. ``C`` components.

    Returns
    -------
    ndarray
        Image, multiplied by `multiplier`.

    """
    if image.size == 0:
        return np.copy(image)

    iadt.gate_dtypes(
        image,
        allowed=["bool",
                 "uint8", "uint16",
                 "int8", "int16",
                 "float16", "float32"],
        disallowed=["uint32", "uint64", "uint128", "uint256",
                    "int32", "int64", "int128", "int256",
                    "float64", "float96", "float128",
                    "float256"],
        augmenter=None)

    if image.dtype.name == "uint8":
        return _multiply_scalar_to_uint8(image, multiplier)
    return _multiply_scalar_to_non_uint8(image, multiplier)


def _multiply_scalar_to_uint8(image, multiplier):
    # Using this LUT approach is significantly faster than
    # else-block code (more than 10x speedup) and is still faster
    # than the simpler image*sample approach without LUT (1.5-3x
    # speedup, maybe dependent on installed BLAS libraries?)

    # pylint: disable=no-else-return

    is_single_value = (
        ia.is_single_number(multiplier)
        or ia.is_np_scalar(multiplier)
        or (ia.is_np_array(multiplier) and multiplier.size == 1))
    is_channelwise = not is_single_value
    nb_channels = 1 if image.ndim == 2 else image.shape[-1]

    multiplier = np.float32(multiplier)
    value_range = np.arange(0, 256, dtype=np.float32)

    if is_channelwise:
        assert multiplier.ndim == 1, (
            "Expected `multiplier` to be 1-dimensional, got %d-dimensional "
            "data with shape %s." % (multiplier.ndim, multiplier.shape))
        assert image.ndim == 3, (
            "Expected `image` to be 3-dimensional when multiplying by one "
            "value per channel, got %d-dimensional data with shape %s." % (
                image.ndim, image.shape))
        assert image.shape[-1] == multiplier.size, (
            "Expected number of channels in `image` and number of components "
            "in `multiplier` to be identical. Got %d vs. %d." % (
                image.shape[-1], multiplier.size))

        # TODO check if tile() is here actually needed
        tables = np.tile(
            value_range[:, np.newaxis],
            (1, nb_channels)
        ) * multiplier[np.newaxis, :]
    else:
        tables = value_range * multiplier
    tables = np.clip(tables, 0, 255).astype(image.dtype)
    return ia.apply_lut(image, tables)


def _multiply_scalar_to_non_uint8(image, multiplier):
    # TODO estimate via image min/max values whether a resolution
    #      increase is necessary
    input_dtype = image.dtype

    is_single_value = (
        ia.is_single_number(multiplier)
        or ia.is_np_scalar(multiplier)
        or (ia.is_np_array(multiplier) and multiplier.size == 1))
    is_channelwise = not is_single_value
    nb_channels = 1 if image.ndim == 2 else image.shape[-1]

    shape = (1, 1, nb_channels if is_channelwise else 1)
    multiplier = np.array(multiplier).reshape(shape)

    # deactivated itemsize increase due to clip causing problems
    # with int64, see Add
    # mul_min = np.min(mul)
    # mul_max = np.max(mul)
    # is_not_increasing_value_range = (
    #         (-1 <= mul_min <= 1)
    #         and (-1 <= mul_max <= 1))

    # We limit here the value range of the mul parameter to the
    # bytes in the image's dtype. This prevents overflow problems
    # and makes it less likely that the image has to be up-casted,
    # which again improves performance and saves memory. Note that
    # this also enables more dtypes for image inputs.
    # The downside is that the mul parameter is limited in its
    # value range.
    itemsize = max(
        image.dtype.itemsize,
        2 if multiplier.dtype.kind == "f" else 1
    )  # float min itemsize is 2 not 1
    dtype_target = np.dtype("%s%d" % (multiplier.dtype.kind, itemsize))
    multiplier = iadt.clip_to_dtype_value_range_(
        multiplier, dtype_target, validate=True)

    image, multiplier = iadt.promote_array_dtypes_(
        [image, multiplier],
        dtypes=[image.dtype, dtype_target],
        # increase_itemsize_factor=(
        #     1 if is_not_increasing_value_range else 2)
        increase_itemsize_factor=1
    )
    image = np.multiply(image, multiplier, out=image, casting="no")

    return iadt.restore_dtypes_(image, input_dtype)


def multiply_elementwise(image, multipliers):
    """Multiply an image with an array of values.

    This method ensures that ``uint8`` does not overflow during the addition.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: limited; tested (1)
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: limited; tested (1)
        * ``int16``: limited; tested (1)
        * ``int32``: no
        * ``int64``: no
        * ``float16``: limited; tested (1)
        * ``float32``: limited; tested (1)
        * ``float64``: no
        * ``float128``: no
        * ``bool``: limited; tested (1)

        - (1) Non-uint8 dtypes can overflow. For floats, this can result
              in +/-inf.

    note::

        Tests were only conducted for rather small multipliers, around
        ``-10.0`` to ``+10.0``.

        In general, the multipliers sampled from `multipliers` must be in a
        value range that corresponds to the input image's dtype. E.g. if the
        input image has dtype ``uint16`` and the samples generated from
        `multipliers` are ``float64``, this function will still force all
        samples to be within the value range of ``float16``, as it has the
        same number of bytes (two) as ``uint16``. This is done to make
        overflows less likely to occur.

    Parameters
    ----------
    image : ndarray
        Image array of shape ``(H,W,[C])``.

    multipliers : ndarray
        The multipliers with which to multiply the image. Expected to have
        the same height and width as `image` and either no channels or one
        channel or the same number of channels as `image`.

    Returns
    -------
    ndarray
        Image, multiplied by `multipliers`.

    """
    iadt.gate_dtypes(
        image,
        allowed=["bool",
                 "uint8", "uint16",
                 "int8", "int16",
                 "float16", "float32"],
        disallowed=["uint32", "uint64", "uint128", "uint256",
                    "int32", "int64", "int128", "int256",
                    "float64", "float96", "float128", "float256"],
        augmenter=None)

    if multipliers.dtype.kind == "b":
        # TODO extend this with some shape checks
        image *= multipliers
        return image
    if image.dtype.name == "uint8":
        return _multiply_elementwise_to_uint8(image, multipliers)
    return _multiply_elementwise_to_non_uint8(image, multipliers)


def _multiply_elementwise_to_uint8(image, multipliers):
    # This special uint8 block is around 60-100% faster than the
    # non-uint8 block further below (more speedup for larger images).
    if multipliers.dtype.kind == "f":
        # interestingly, float32 is here significantly faster than
        # float16
        # TODO is that system dependent?
        # TODO does that affect int8-int32 too?
        multipliers = multipliers.astype(np.float32, copy=False)
        image_aug = image.astype(np.float32)
    else:
        multipliers = multipliers.astype(np.int16, copy=False)
        image_aug = image.astype(np.int16)

    image_aug = np.multiply(image_aug, multipliers, casting="no", out=image_aug)
    return iadt.restore_dtypes_(image_aug, np.uint8, round=False)


def _multiply_elementwise_to_non_uint8(image, multipliers):
    input_dtype = image.dtype

    # TODO maybe introduce to stochastic parameters some way to
    #      get the possible min/max values, could make things
    #      faster for dropout to get 0/1 min/max from the binomial
    # itemsize decrease is currently deactivated due to issues
    # with clip and int64, see Add
    mul_min = np.min(multipliers)
    mul_max = np.max(multipliers)
    # is_not_increasing_value_range = (
    #     (-1 <= mul_min <= 1) and (-1 <= mul_max <= 1))

    # We limit here the value range of the mul parameter to the
    # bytes in the image's dtype. This prevents overflow problems
    # and makes it less likely that the image has to be up-casted,
    # which again improves performance and saves memory. Note that
    # this also enables more dtypes for image inputs.
    # The downside is that the mul parameter is limited in its
    # value range.
    itemsize = max(
        image.dtype.itemsize,
        2 if multipliers.dtype.kind == "f" else 1
    )  # float min itemsize is 2
    dtype_target = np.dtype("%s%d" % (multipliers.dtype.kind, itemsize))
    multipliers = iadt.clip_to_dtype_value_range_(
        multipliers, dtype_target,
        validate=True, validate_values=(mul_min, mul_max))

    if multipliers.shape[2] == 1:
        # TODO check if tile() is here actually needed
        nb_channels = image.shape[-1]
        multipliers = np.tile(multipliers, (1, 1, nb_channels))

    image, multipliers = iadt.promote_array_dtypes_(
        [image, multipliers],
        dtypes=[image, dtype_target],
        increase_itemsize_factor=1
        # increase_itemsize_factor=(
        #     1 if is_not_increasing_value_range else 2)
    )
    image = np.multiply(image, multipliers, out=image, casting="no")
    return iadt.restore_dtypes_(image, input_dtype)


def cutout(image, x1, y1, x2, y2,
           fill_mode="constant", cval=0, fill_per_channel=False,
           seed=None):
    """Fill a single area within an image using a fill mode.

    This cutout method uses the top-left and bottom-right corner coordinates
    of the cutout region given as absolute pixel values.

    .. note::

        Gaussian fill mode will assume that float input images contain values
        in the interval ``[0.0, 1.0]`` and hence sample values from a
        gaussian within that interval, i.e. from ``N(0.5, std=0.5/3)``.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.arithmetic.cutout_`.

    Added in 0.4.0.

    Parameters
    ----------
    image : ndarray
        Image to modify.

    x1 : number
        See :func:`~imgaug.augmenters.arithmetic.cutout_`.

    y1 : number
        See :func:`~imgaug.augmenters.arithmetic.cutout_`.

    x2 : number
        See :func:`~imgaug.augmenters.arithmetic.cutout_`.

    y2 : number
        See :func:`~imgaug.augmenters.arithmetic.cutout_`.

    fill_mode : {'constant', 'gaussian'}, optional
        See :func:`~imgaug.augmenters.arithmetic.cutout_`.

    cval : number or tuple of number, optional
        See :func:`~imgaug.augmenters.arithmetic.cutout_`.

    fill_per_channel : number or bool, optional
        See :func:`~imgaug.augmenters.arithmetic.cutout_`.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.arithmetic.cutout_`.

    Returns
    -------
    ndarray
        Image with area filled in.

    """
    return cutout_(np.copy(image),
                   x1, y1, x2, y2,
                   fill_mode, cval, fill_per_channel, seed)


def cutout_(image, x1, y1, x2, y2,
            fill_mode="constant", cval=0, fill_per_channel=False,
            seed=None):
    """Fill a single area within an image using a fill mode (in-place).

    This cutout method uses the top-left and bottom-right corner coordinates
    of the cutout region given as absolute pixel values.

    .. note::

        Gaussian fill mode will assume that float input images contain values
        in the interval ``[0.0, 1.0]`` and hence sample values from a
        gaussian within that interval, i.e. from ``N(0.5, std=0.5/3)``.

    Added in 0.4.0.

    **Supported dtypes**:

    minimum of (
        :func:`~imgaug.augmenters.arithmetic._fill_rectangle_gaussian_`,
        :func:`~imgaug.augmenters.arithmetic._fill_rectangle_constant_`
    )

    Parameters
    ----------
    image : ndarray
        Image to modify. Might be modified in-place.

    x1 : number
        X-coordinate of the top-left corner of the cutout region.

    y1 : number
        Y-coordinate of the top-left corner of the cutout region.

    x2 : number
        X-coordinate of the bottom-right corner of the cutout region.

    y2 : number
        Y-coordinate of the bottom-right corner of the cutout region.

    fill_mode : {'constant', 'gaussian'}, optional
        Fill mode to use.

    cval : number or tuple of number, optional
        The constant value to use when filling with mode ``constant``.
        May be an intensity value or color tuple.

    fill_per_channel : number or bool, optional
        Whether to fill in a channelwise fashion.
        If number then a value ``>=0.5`` will be interpreted as ``True``.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        A random number generator to sample random values from.
        Usually an integer seed value or an ``RNG`` instance.
        See :class:`imgaug.random.RNG` for details.

    Returns
    -------
    ndarray
        Image with area filled in.
        The input image might have been modified in-place.

    """
    import importlib

    height, width = image.shape[0:2]
    x1 = min(max(int(x1), 0), width)
    y1 = min(max(int(y1), 0), height)
    x2 = min(max(int(x2), 0), width)
    y2 = min(max(int(y2), 0), height)

    if x2 > x1 and y2 > y1:
        assert fill_mode in _CUTOUT_FILL_MODES, (
            "Expected one of the following fill modes: %s. "
            "Got: %s." % (
                str(list(_CUTOUT_FILL_MODES.keys())), fill_mode))

        module_name, fname = _CUTOUT_FILL_MODES[fill_mode]
        module = importlib.import_module(module_name)
        func = getattr(module, fname)
        image = func(
            image,
            x1=x1, y1=y1, x2=x2, y2=y2,
            cval=cval,
            per_channel=(fill_per_channel >= 0.5),
            random_state=(
                iarandom.RNG(seed)
                if not isinstance(seed, iarandom.RNG)
                else seed)  # only RNG(.) without "if" is ~8x slower
        )
    return image


def _fill_rectangle_gaussian_(image, x1, y1, x2, y2, cval, per_channel,
                              random_state):
    """Fill a rectangular image area with samples from a gaussian.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: limited; tested (1)
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: limited; tested (1)
        * ``float16``: yes; tested (2)
        * ``float32``: yes; tested (2)
        * ``float64``: yes; tested (2)
        * ``float128``: limited; tested (1) (2)
        * ``bool``: yes; tested

        - (1) Possible loss of resolution due to gaussian values being sampled
              as ``float64`` s.
        - (2) Float input arrays are assumed to be in interval ``[0.0, 1.0]``
              and all gaussian samples are within that interval too.

    """
    # for float we assume value range [0.0, 1.0]
    # that matches the common use case and also makes the tests way easier
    # we also set bool here manually as the center value returned by
    # get_value_range_for_dtype() is None
    kind = image.dtype.kind
    if kind in ["f", "b"]:
        min_value = 0.0
        center_value = 0.5
        max_value = 1.0
    else:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(
            image.dtype)

    # set standard deviation to 1/3 of value range to get 99.7% of values
    # within [min v.r., max v.r.]
    # we also divide by 2 because we want to spread towards the
    # "left"/"right" of the center value by half of the value range
    stddev = (float(max_value) - float(min_value)) / 2.0 / 3.0

    height = y2 - y1
    width = x2 - x1
    shape = (height, width)
    if per_channel and image.ndim == 3:
        shape = shape + (image.shape[2],)
    rect = random_state.normal(center_value, stddev, size=shape)
    if image.dtype.kind == "b":
        rect_vr = (rect > 0.5)
    else:
        rect_vr = np.clip(rect, min_value, max_value).astype(image.dtype)

    if image.ndim == 3:
        image[y1:y2, x1:x2, :] = np.atleast_3d(rect_vr)
    else:
        image[y1:y2, x1:x2] = rect_vr

    return image


def _fill_rectangle_constant_(image, x1, y1, x2, y2, cval, per_channel,
                              random_state):
    """Fill a rectangular area within an image with constant value(s).

    `cval` may be a single value or one per channel. If the number of items
    in `cval` does not match the number of channels in `image`, it may
    be tiled up to the number of channels.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    """
    if ia.is_iterable(cval):
        if per_channel:
            nb_channels = None if image.ndim == 2 else image.shape[-1]
            if nb_channels is None:
                cval = cval[0]
            elif len(cval) < nb_channels:
                mul = int(np.ceil(nb_channels / len(cval)))
                cval = np.tile(cval, (mul,))[0:nb_channels]
            elif len(cval) > nb_channels:
                cval = cval[0:nb_channels]
        else:
            cval = cval[0]

    # without the array(), uint64 max value is assigned as 0
    image[y1:y2, x1:x2, ...] = np.array(cval, dtype=image.dtype)

    return image


def replace_elementwise_(image, mask, replacements):
    """Replace components in an image array with new values.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: no (1)
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: no (2)
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: no
        * ``bool``: yes; tested

        - (1) ``uint64`` is currently not supported, because
              :func:`~imgaug.dtypes.clip_to_dtype_value_range_()` does not
              support it, which again is because numpy.clip() seems to not
              support it.
        - (2) `int64` is disallowed due to being converted to `float64`
              by :func:`numpy.clip` since 1.17 (possibly also before?).

    Parameters
    ----------
    image : ndarray
        Image array of shape ``(H,W,[C])``.

    mask : ndarray
        Mask of shape ``(H,W,[C])`` denoting which components to replace.
        If ``C`` is provided, it must be ``1`` or match the ``C`` of `image`.
        May contain floats in the interval ``[0.0, 1.0]``.

    replacements : iterable
        Replacements to place in `image` at the locations defined by `mask`.
        This 1-dimensional iterable must contain exactly as many values
        as there are replaced components in `image`.

    Returns
    -------
    ndarray
        Image with replaced components.

    """
    iadt.gate_dtypes(
        image,
        allowed=["bool",
                 "uint8", "uint16", "uint32",
                 "int8", "int16", "int32",
                 "float16", "float32", "float64"],
        disallowed=["uint64", "uint128", "uint256",
                    "int64", "int128", "int256",
                    "float96", "float128", "float256"],
        augmenter=None)

    # This is slightly faster (~20%) for masks that are True at many
    # locations, but slower (~50%) for masks with few Trues, which is
    # probably the more common use-case:
    #
    # replacement_samples = self.replacement.draw_samples(
    #     sampling_shape, random_state=rs_replacement)
    #
    # # round, this makes 0.2 e.g. become 0 in case of boolean
    # # image (otherwise replacing values with 0.2 would
    # # lead to True instead of False).
    # if (image.dtype.kind in ["i", "u", "b"]
    #         and replacement_samples.dtype.kind == "f"):
    #     replacement_samples = np.round(replacement_samples)
    #
    # replacement_samples = iadt.clip_to_dtype_value_range_(
    #     replacement_samples, image.dtype, validate=False)
    # replacement_samples = replacement_samples.astype(
    #     image.dtype, copy=False)
    #
    # if sampling_shape[2] == 1:
    #     mask_samples = np.tile(mask_samples, (1, 1, nb_channels))
    #     replacement_samples = np.tile(
    #         replacement_samples, (1, 1, nb_channels))
    # mask_thresh = mask_samples > 0.5
    # image[mask_thresh] = replacement_samples[mask_thresh]
    input_shape = image.shape
    if image.ndim == 2:
        image = image[..., np.newaxis]
    if mask.ndim == 2:
        mask = mask[..., np.newaxis]

    mask_thresh = mask > 0.5
    if mask.shape[2] == 1:
        nb_channels = image.shape[-1]
        # TODO verify if tile() is here really necessary
        mask_thresh = np.tile(mask_thresh, (1, 1, nb_channels))

    # round, this makes 0.2 e.g. become 0 in case of boolean
    # image (otherwise replacing values with 0.2 would lead to True
    # instead of False).
    if image.dtype.kind in ["i", "u", "b"] and replacements.dtype.kind == "f":
        replacements = np.round(replacements)

    replacement_samples = iadt.clip_to_dtype_value_range_(
        replacements, image.dtype, validate=False)
    replacement_samples = replacement_samples.astype(image.dtype, copy=False)

    image[mask_thresh] = replacement_samples
    if len(input_shape) == 2:
        return image[..., 0]
    return image


def invert(image, min_value=None, max_value=None, threshold=None,
           invert_above_threshold=True):
    """Invert an array.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.arithmetic.invert_`.

    Parameters
    ----------
    image : ndarray
        See :func:`invert_`.

    min_value : None or number, optional
        See :func:`invert_`.

    max_value : None or number, optional
        See :func:`invert_`.

    threshold : None or number, optional
        See :func:`invert_`.

    invert_above_threshold : bool, optional
        See :func:`invert_`.

    Returns
    -------
    ndarray
        Inverted image.

    """
    return invert_(np.copy(image), min_value=min_value, max_value=max_value,
                   threshold=threshold,
                   invert_above_threshold=invert_above_threshold)


def invert_(image, min_value=None, max_value=None, threshold=None,
            invert_above_threshold=True):
    """Invert an array in-place.

    Added in 0.4.0.

    **Supported dtypes**:

    if (min_value=None and max_value=None):

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    if (min_value!=None or max_value!=None):

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: no (1)
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: no (2)
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: no (2)
        * ``float128``: no (3)
        * ``bool``: no (4)

        - (1) Not allowed due to numpy's clip converting from ``uint64`` to
              ``float64``.
        - (2) Not allowed as int/float have to be increased in resolution
              when using min/max values.
        - (3) Not tested.
        - (4) Makes no sense when using min/max values.

    Parameters
    ----------
    image : ndarray
        Image array of shape ``(H,W,[C])``.
        The array *might* be modified in-place.

    min_value : None or number, optional
        Minimum of the value range of input images, e.g. ``0`` for ``uint8``
        images. If set to ``None``, the value will be automatically derived
        from the image's dtype.

    max_value : None or number, optional
        Maximum of the value range of input images, e.g. ``255`` for ``uint8``
        images. If set to ``None``, the value will be automatically derived
        from the image's dtype.

    threshold : None or number, optional
        A threshold to use in order to invert only numbers above or below
        the threshold. If ``None`` no thresholding will be used.

    invert_above_threshold : bool, optional
        If ``True``, only values ``>=threshold`` will be inverted.
        Otherwise, only values ``<threshold`` will be inverted.
        If `threshold` is ``None`` this parameter has no effect.

    Returns
    -------
    ndarray
        Inverted image. This *can* be the same array as input in `image`,
        modified in-place.

    """
    # when no custom min/max are chosen, all bool, uint, int and float dtypes
    # should be invertable (float tested only up to 64bit)
    # when chosing custom min/max:
    # - bool makes no sense, not allowed
    # - int and float must be increased in resolution if custom min/max values
    #   are chosen, hence they are limited to 32 bit and below
    # - uint64 is converted by numpy's clip to float64, hence loss of accuracy
    # - float16 seems to not be perfectly accurate, but still ok-ish -- was
    #   off by 10 for center value of range (float 16 min, 16), where float
    #   16 min is around -65500
    allow_dtypes_custom_minmax = {"uint8", "uint16", "uint32",
                                  "int8", "int16", "int32",
                                  "float16", "float32"}

    min_value_dt, _, max_value_dt = \
        iadt.get_value_range_of_dtype(image.dtype)
    min_value = (min_value_dt
                 if min_value is None else min_value)
    max_value = (max_value_dt
                 if max_value is None else max_value)
    assert min_value >= min_value_dt, (
        "Expected min_value to be above or equal to dtype's min "
        "value, got %s (vs. min possible %s for %s)" % (
            str(min_value), str(min_value_dt), image.dtype.name)
    )
    assert max_value <= max_value_dt, (
        "Expected max_value to be below or equal to dtype's max "
        "value, got %s (vs. max possible %s for %s)" % (
            str(max_value), str(max_value_dt), image.dtype.name)
    )
    assert min_value < max_value, (
        "Expected min_value to be below max_value, got %s "
        "and %s" % (
            str(min_value), str(max_value))
    )

    if min_value != min_value_dt or max_value != max_value_dt:
        assert image.dtype.name in allow_dtypes_custom_minmax, (
            "Can use custom min/max values only with the following "
            "dtypes: %s. Got: %s." % (
                ", ".join(allow_dtypes_custom_minmax), image.dtype.name))

    if image.dtype.name == "uint8":
        return _invert_uint8_(image, min_value, max_value, threshold,
                              invert_above_threshold)

    dtype_kind_to_invert_func = {
        "b": _invert_bool,
        "u": _invert_uint16_or_larger_,  # uint8 handled above
        "i": _invert_int_,
        "f": _invert_float
    }

    func = dtype_kind_to_invert_func[image.dtype.kind]

    if threshold is None:
        return func(image, min_value, max_value)

    arr_inv = func(np.copy(image), min_value, max_value)
    if invert_above_threshold:
        mask = (image >= threshold)
    else:
        mask = (image < threshold)
    image[mask] = arr_inv[mask]
    return image


def _invert_bool(arr, min_value, max_value):
    assert min_value == 0 and max_value == 1, (
        "min_value and max_value must be 0 and 1 for bool arrays. "
        "Got %.4f and %.4f." % (min_value, max_value))
    return ~arr


# Added in 0.4.0.
def _invert_uint8_(arr, min_value, max_value, threshold,
                   invert_above_threshold):
    table = _generate_table_for_invert_uint8(
        min_value, max_value, threshold, invert_above_threshold)
    arr = ia.apply_lut_(arr, table)
    return arr


# Added in 0.4.0.
def _invert_uint16_or_larger_(arr, min_value, max_value):
    min_max_is_vr = (min_value == 0
                     and max_value == np.iinfo(arr.dtype).max)
    if min_max_is_vr:
        return max_value - arr
    return _invert_by_distance(
        np.clip(arr, min_value, max_value),
        min_value, max_value
    )


# Added in 0.4.0.
def _invert_int_(arr, min_value, max_value):
    # note that for int dtypes the max value is
    #   (-1) * min_value - 1
    # e.g. -128 and 127 (min/max) for int8
    # mapping example:
    #  [-4, -3, -2, -1,  0,  1,  2,  3]
    # will be mapped to
    #  [ 3,  2,  1,  0, -1, -2, -3, -4]
    # hence we can not simply compute the inverse as:
    #  after = (-1) * before
    # but instead need
    #  after = (-1) * before - 1
    # however, this exceeds the value range for the minimum value, e.g.
    # for int8: -128 -> 128 -> 127, where 128 exceeds it. Hence, we must
    # compute the inverse via a mask (extra step for the minimum)
    # or we have to increase the resolution of the array. Here, a
    # two-step approach is used.

    if min_value == (-1) * max_value - 1:
        arr_inv = np.copy(arr)
        mask = (arr_inv == min_value)

        # there is probably a one-liner here to do this, but
        #  ((-1) * (arr_inv * ~mask) - 1) + mask * max_value
        # has the disadvantage of inverting min_value to max_value - 1
        # while
        #  ((-1) * (arr_inv * ~mask) - 1) + mask * (max_value+1)
        #  ((-1) * (arr_inv * ~mask) - 1) + mask * max_value + mask
        # both sometimes increase the dtype resolution (e.g. int32 to int64)
        arr_inv[mask] = max_value
        arr_inv[~mask] = (-1) * arr_inv[~mask] - 1

        return arr_inv

    return _invert_by_distance(
        np.clip(arr, min_value, max_value),
        min_value, max_value
    )


def _invert_float(arr, min_value, max_value):
    if np.isclose(max_value, (-1)*min_value, rtol=0):
        return (-1) * arr
    return _invert_by_distance(
        np.clip(arr, min_value, max_value),
        min_value, max_value
    )


def _invert_by_distance(arr, min_value, max_value):
    arr_inv = arr
    if arr.dtype.kind in ["i", "f"]:
        arr_inv = iadt.increase_array_resolutions_([np.copy(arr)], 2)[0]
    distance_from_min = np.abs(arr_inv - min_value)  # d=abs(v-min)
    arr_inv = max_value - distance_from_min  # v'=MAX-d
    # due to floating point inaccuracies, we might exceed the min/max
    # values for floats here, hence clip this happens especially for
    # values close to the float dtype's maxima
    if arr.dtype.kind == "f":
        arr_inv = np.clip(arr_inv, min_value, max_value)
    if arr.dtype.kind in ["i", "f"]:
        arr_inv = iadt.restore_dtypes_(
            arr_inv, arr.dtype, clip=False)
    return arr_inv


# Added in 0.4.0.
def _generate_table_for_invert_uint8(min_value, max_value, threshold,
                                     invert_above_threshold):
    table = np.arange(256).astype(np.int32)
    full_value_range = (min_value == 0 and max_value == 255)
    if full_value_range:
        table_inv = table[::-1]
    else:
        distance_from_min = np.abs(table - min_value)
        table_inv = max_value - distance_from_min
    table_inv = np.clip(table_inv, min_value, max_value).astype(np.uint8)

    if threshold is not None:
        table = table.astype(np.uint8)
        if invert_above_threshold:
            table_inv = np.concatenate([
                table[0:int(threshold)],
                table_inv[int(threshold):]
            ], axis=0)
        else:
            table_inv = np.concatenate([
                table_inv[0:int(threshold)],
                table[int(threshold):]
            ], axis=0)

    return table_inv


def solarize(image, threshold=128):
    """Invert pixel values above a threshold.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.arithmetic.solarize_`.

    Parameters
    ----------
    image : ndarray
        See :func:`solarize_`.

    threshold : None or number, optional
        See :func:`solarize_`.

    Returns
    -------
    ndarray
        Inverted image.

    """
    return solarize_(np.copy(image), threshold=threshold)


def solarize_(image, threshold=128):
    """Invert pixel values above a threshold in-place.

    This function is a wrapper around :func:`invert`.

    This function performs the same transformation as
    :func:`PIL.ImageOps.solarize`.

    Added in 0.4.0.

    **Supported dtypes**:

    See ``~imgaug.augmenters.arithmetic.invert_(min_value=None and max_value=None)``.

    Parameters
    ----------
    image : ndarray
        See :func:`invert_`.

    threshold : None or number, optional
        See :func:`invert_`.
        Note: The default threshold is optimized for ``uint8`` images.


    Returns
    -------
    ndarray
        Inverted image. This *can* be the same array as input in `image`,
        modified in-place.

    """
    return invert_(image, threshold=threshold)


def compress_jpeg(image, compression):
    """Compress an image using jpeg compression.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: ?
        * ``uint32``: ?
        * ``uint64``: ?
        * ``int8``: ?
        * ``int16``: ?
        * ``int32``: ?
        * ``int64``: ?
        * ``float16``: ?
        * ``float32``: ?
        * ``float64``: ?
        * ``float128``: ?
        * ``bool``: ?

    Parameters
    ----------
    image : ndarray
        Image of dtype ``uint8`` and shape ``(H,W,[C])``. If ``C`` is provided,
        it must be ``1`` or ``3``.

    compression : int
        Strength of the compression in the interval ``[0, 100]``.

    Returns
    -------
    ndarray
        Input image after applying jpeg compression to it and reloading
        the result into a new array. Same shape and dtype as the input.

    """
    import PIL.Image

    if image.size == 0:
        return np.copy(image)

    # The value range 1 to 95 is suggested by PIL's save() documentation
    # Values above 95 seem to not make sense (no improvement in visual
    # quality, but large file size).
    # A value of 100 would mostly deactivate jpeg compression.
    # A value of 0 would lead to no compression (instead of maximum
    # compression).
    # We use range 1 to 100 here, because this augmenter is about
    # generating images for training and not for saving, hence we do not
    # care about large file sizes.
    maximum_quality = 100
    minimum_quality = 1

    assert image.dtype.name == "uint8", (
        "Jpeg compression can only be applied to uint8 images. "
        "Got dtype %s." % (image.dtype.name,))
    assert 0 <= compression <= 100, (
        "Expected compression to be in the interval [0, 100], "
        "got %.4f." % (compression,))

    has_no_channels = (image.ndim == 2)
    is_single_channel = (image.ndim == 3 and image.shape[-1] == 1)
    if is_single_channel:
        image = image[..., 0]

    assert has_no_channels or is_single_channel or image.shape[-1] == 3, (
        "Expected either a grayscale image of shape (H,W) or (H,W,1) or an "
        "RGB image of shape (H,W,3). Got shape %s." % (image.shape,))

    # Map from compression to quality used by PIL
    # We have valid compressions from 0 to 100, i.e. 101 possible
    # values
    quality = int(
        np.clip(
            np.round(
                minimum_quality
                + (maximum_quality - minimum_quality)
                * (1.0 - (compression / 101))
            ),
            minimum_quality,
            maximum_quality
        )
    )

    image_pil = PIL.Image.fromarray(image)
    with tempfile.NamedTemporaryFile(mode="wb+", suffix=".jpg") as f:
        image_pil.save(f, quality=quality)

        # Read back from file.
        # We dont read from f.name, because that leads to PermissionDenied
        # errors on Windows. We add f.seek(0) here, because otherwise we get
        # `SyntaxError: index out of range` in PIL.
        f.seek(0)
        pilmode = "RGB"
        if has_no_channels or is_single_channel:
            pilmode = "L"
        image = imageio.imread(f, pilmode=pilmode, format="jpeg")
    if is_single_channel:
        image = image[..., np.newaxis]
    return image


class Add(meta.Augmenter):
    """
    Add a value to all pixels in an image.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.arithmetic.add_scalar`.

    Parameters
    ----------
    value : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Value to add to all pixels.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Add(10)

    Always adds a value of 10 to all channels of all pixels of all input
    images.

    >>> aug = iaa.Add((-10, 10))

    Adds a value from the discrete interval ``[-10..10]`` to all pixels of
    input images. The exact value is sampled per image.

    >>> aug = iaa.Add((-10, 10), per_channel=True)

    Adds a value from the discrete interval ``[-10..10]`` to all pixels of
    input images. The exact value is sampled per image *and* channel,
    i.e. to a red-channel it might add 5 while subtracting 7 from the
    blue channel of the same image.

    >>> aug = iaa.Add((-10, 10), per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """

    def __init__(self, value=(-20, 20), per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Add, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.value = iap.handle_continuous_param(
            value, "value", value_range=None, tuple_to_uniform=True,
            list_to_choice=True)
        self.per_channel = iap.handle_probability_param(
            per_channel, "per_channel")

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images
        nb_images = len(images)
        nb_channels_max = meta.estimate_max_number_of_channels(images)
        rss = random_state.duplicate(2)

        per_channel_samples = self.per_channel.draw_samples(
            (nb_images,), random_state=rss[0])
        value_samples = self.value.draw_samples(
            (nb_images, nb_channels_max), random_state=rss[1])

        gen = enumerate(zip(images, value_samples, per_channel_samples))
        for i, (image, value_samples_i, per_channel_samples_i) in gen:
            nb_channels = image.shape[2]

            # Example code to directly add images via image+sample (uint8 only)
            # if per_channel_samples_i > 0.5:
            #     result = []
            #     image = image.astype(np.int16)
            #     value_samples_i = value_samples_i.astype(np.int16)
            #     for c, value in enumerate(value_samples_i[0:nb_channels]):
            #         result.append(
            #             np.clip(
            #                 image[..., c:c+1] + value, 0, 255
            #             ).astype(np.uint8))
            #     images[i] = np.concatenate(result, axis=2)
            # else:
            #     images[i] = np.clip(
            #         image.astype(np.int16)
            #         + value_samples_i[0].astype(np.int16),
            #         0, 255
            #     ).astype(np.uint8)

            if per_channel_samples_i > 0.5:
                value = value_samples_i[0:nb_channels]
            else:
                # the if/else here catches the case of the channel axis being 0
                value = value_samples_i[0] if value_samples_i.size > 0 else []

            batch.images[i] = add_scalar(image, value)

        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.value, self.per_channel]


# TODO merge this with Add
class AddElementwise(meta.Augmenter):
    """
    Add to the pixels of images values that are pixelwise randomly sampled.

    While the ``Add`` Augmenter samples one value to add *per image* (and
    optionally per channel), this augmenter samples different values per image
    and *per pixel* (and optionally per channel), i.e. intensities of
    neighbouring pixels may be increased/decreased by different amounts.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.arithmetic.add_elementwise`.

    Parameters
    ----------
    value : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Value to add to the pixels.

            * If an int, exactly that value will always be used.
            * If a tuple ``(a, b)``, then values from the discrete interval
              ``[a..b]`` will be sampled per image and pixel.
            * If a list of integers, a random value will be sampled from the
              list per image and pixel.
            * If a ``StochasticParameter``, then values will be sampled per
              image and pixel from that parameter.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.AddElementwise(10)

    Always adds a value of 10 to all channels of all pixels of all input
    images.

    >>> aug = iaa.AddElementwise((-10, 10))

    Samples per image and pixel a value from the discrete interval
    ``[-10..10]`` and adds that value to the respective pixel.

    >>> aug = iaa.AddElementwise((-10, 10), per_channel=True)

    Samples per image, pixel *and also channel* a value from the discrete
    interval ``[-10..10]`` and adds it to the respective pixel's channel value.
    Therefore, added values may differ between channels of the same pixel.

    >>> aug = iaa.AddElementwise((-10, 10), per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """

    def __init__(self, value=(-20, 20), per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(AddElementwise, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.value = iap.handle_continuous_param(
            value, "value", value_range=None, tuple_to_uniform=True,
            list_to_choice=True)
        self.per_channel = iap.handle_probability_param(
            per_channel, "per_channel")

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images
        nb_images = len(images)
        rss = random_state.duplicate(1+nb_images)
        per_channel_samples = self.per_channel.draw_samples(
            (nb_images,), random_state=rss[0])

        gen = enumerate(zip(images, per_channel_samples, rss[1:]))
        for i, (image, per_channel_samples_i, rs) in gen:
            height, width, nb_channels = image.shape
            sample_shape = (height,
                            width,
                            nb_channels if per_channel_samples_i > 0.5 else 1)
            values = self.value.draw_samples(sample_shape, random_state=rs)

            batch.images[i] = add_elementwise(image, values)

        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.value, self.per_channel]


# TODO rename to AddGaussianNoise?
# TODO examples say that iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)) samples
#      the scale from the uniform dist. per image, but is that still the case?
#      AddElementwise seems to now sample once for all images, which should
#      lead to a single scale value.
class AdditiveGaussianNoise(AddElementwise):
    """
    Add noise sampled from gaussian distributions elementwise to images.

    This augmenter samples and adds noise elementwise, i.e. it can add
    different noise values to neighbouring pixels and is comparable
    to ``AddElementwise``.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.arithmetic.AddElementwise`.

    Parameters
    ----------
    loc : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Mean of the normal distribution from which the noise is sampled.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list per
              image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

    scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Standard deviation of the normal distribution that generates the noise.
        Must be ``>=0``. If ``0`` then `loc` will simply be added to all
        pixels.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list per
              image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.AdditiveGaussianNoise(scale=0.1*255)

    Adds gaussian noise from the distribution ``N(0, 0.1*255)`` to images.
    The samples are drawn per image and pixel.

    >>> aug = iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))

    Adds gaussian noise from the distribution ``N(0, s)`` to images,
    where ``s`` is sampled per image from the interval ``[0, 0.1*255]``.

    >>> aug = iaa.AdditiveGaussianNoise(scale=0.1*255, per_channel=True)

    Adds gaussian noise from the distribution ``N(0, 0.1*255)`` to images,
    where the noise value is different per image and pixel *and* channel (e.g.
    a different one for red, green and blue channels of the same pixel).
    This leads to "colorful" noise.

    >>> aug = iaa.AdditiveGaussianNoise(scale=0.1*255, per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """
    def __init__(self, loc=0, scale=(0, 15), per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        loc2 = iap.handle_continuous_param(
            loc, "loc", value_range=None, tuple_to_uniform=True,
            list_to_choice=True)
        scale2 = iap.handle_continuous_param(
            scale, "scale", value_range=(0, None), tuple_to_uniform=True,
            list_to_choice=True)

        value = iap.Normal(loc=loc2, scale=scale2)

        super(AdditiveGaussianNoise, self).__init__(
            value, per_channel=per_channel,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


# TODO add tests
# TODO rename to AddLaplaceNoise?
class AdditiveLaplaceNoise(AddElementwise):
    """
    Add noise sampled from laplace distributions elementwise to images.

    The laplace distribution is similar to the gaussian distribution, but
    puts more weight on the long tail. Hence, this noise will add more
    outliers (very high/low values). It is somewhere between gaussian noise and
    salt and pepper noise.

    Values of around ``255 * 0.05`` for `scale` lead to visible noise (for
    ``uint8``).
    Values of around ``255 * 0.10`` for `scale` lead to very visible
    noise (for ``uint8``).
    It is recommended to usually set `per_channel` to ``True``.

    This augmenter samples and adds noise elementwise, i.e. it can add
    different noise values to neighbouring pixels and is comparable
    to ``AddElementwise``.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.arithmetic.AddElementwise`.

    Parameters
    ----------
    loc : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Mean of the laplace distribution that generates the noise.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list per
              image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

    scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Standard deviation of the laplace distribution that generates the noise.
        Must be ``>=0``. If ``0`` then only `loc` will be used.
        Recommended to be around ``255*0.05``.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list per
              image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.AdditiveLaplaceNoise(scale=0.1*255)

    Adds laplace noise from the distribution ``Laplace(0, 0.1*255)`` to images.
    The samples are drawn per image and pixel.

    >>> aug = iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255))

    Adds laplace noise from the distribution ``Laplace(0, s)`` to images,
    where ``s`` is sampled per image from the interval ``[0, 0.1*255]``.

    >>> aug = iaa.AdditiveLaplaceNoise(scale=0.1*255, per_channel=True)

    Adds laplace noise from the distribution ``Laplace(0, 0.1*255)`` to images,
    where the noise value is different per image and pixel *and* channel (e.g.
    a different one for the red, green and blue channels of the same pixel).
    This leads to "colorful" noise.

    >>> aug = iaa.AdditiveLaplaceNoise(scale=0.1*255, per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """
    def __init__(self, loc=0, scale=(0, 15), per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        loc2 = iap.handle_continuous_param(
            loc, "loc", value_range=None, tuple_to_uniform=True,
            list_to_choice=True)
        scale2 = iap.handle_continuous_param(
            scale, "scale", value_range=(0, None), tuple_to_uniform=True,
            list_to_choice=True)

        value = iap.Laplace(loc=loc2, scale=scale2)

        super(AdditiveLaplaceNoise, self).__init__(
            value,
            per_channel=per_channel,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


# TODO add tests
# TODO rename to AddPoissonNoise?
class AdditivePoissonNoise(AddElementwise):
    """
    Add noise sampled from poisson distributions elementwise to images.

    Poisson noise is comparable to gaussian noise, as e.g. generated via
    ``AdditiveGaussianNoise``. As poisson distributions produce only positive
    numbers, the sign of the sampled values are here randomly flipped.

    Values of around ``10.0`` for `lam` lead to visible noise (for ``uint8``).
    Values of around ``20.0`` for `lam` lead to very visible noise (for
    ``uint8``).
    It is recommended to usually set `per_channel` to ``True``.

    This augmenter samples and adds noise elementwise, i.e. it can add
    different noise values to neighbouring pixels and is comparable
    to ``AddElementwise``.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.arithmetic.AddElementwise`.

    Parameters
    ----------
    lam : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Lambda parameter of the poisson distribution. Must be ``>=0``.
        Recommended values are around ``0.0`` to ``10.0``.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.AdditivePoissonNoise(lam=5.0)

    Adds poisson noise sampled from a poisson distribution with a ``lambda``
    parameter of ``5.0`` to images.
    The samples are drawn per image and pixel.

    >>> aug = iaa.AdditivePoissonNoise(lam=(0.0, 15.0))

    Adds poisson noise sampled from ``Poisson(x)`` to images, where ``x`` is
    randomly sampled per image from the interval ``[0.0, 15.0]``.

    >>> aug = iaa.AdditivePoissonNoise(lam=5.0, per_channel=True)

    Adds poisson noise sampled from ``Poisson(5.0)`` to images,
    where the values are different per image and pixel *and* channel (e.g. a
    different one for red, green and blue channels for the same pixel).

    >>> aug = iaa.AdditivePoissonNoise(lam=(0.0, 15.0), per_channel=True)

    Adds poisson noise sampled from ``Poisson(x)`` to images,
    with ``x`` being sampled from ``uniform(0.0, 15.0)`` per image and
    channel. This is the *recommended* configuration.

    >>> aug = iaa.AdditivePoissonNoise(lam=(0.0, 15.0), per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """
    def __init__(self, lam=(0.0, 15.0), per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        lam2 = iap.handle_continuous_param(
            lam, "lam",
            value_range=(0, None), tuple_to_uniform=True, list_to_choice=True)

        value = iap.RandomSign(iap.Poisson(lam=lam2))

        super(AdditivePoissonNoise, self).__init__(
            value,
            per_channel=per_channel,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Multiply(meta.Augmenter):
    """
    Multiply all pixels in an image with a random value sampled once per image.

    This augmenter can be used to make images lighter or darker.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.arithmetic.multiply_scalar`.

    Parameters
    ----------
    mul : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        The value with which to multiply the pixel values in each image.

            * If a number, then that value will always be used.
            * If a tuple ``(a, b)``, then a value from the interval ``[a, b]``
              will be sampled per image and used for all pixels.
            * If a list, then a random value will be sampled from that list per
              image.
            * If a ``StochasticParameter``, then that parameter will be used to
              sample a new value per image.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Multiply(2.0)

    Multiplies all images by a factor of ``2``, making the images significantly
    brighter.

    >>> aug = iaa.Multiply((0.5, 1.5))

    Multiplies images by a random value sampled uniformly from the interval
    ``[0.5, 1.5]``, making some images darker and others brighter.

    >>> aug = iaa.Multiply((0.5, 1.5), per_channel=True)

    Identical to the previous example, but the sampled multipliers differ by
    image *and* channel, instead of only by image.

    >>> aug = iaa.Multiply((0.5, 1.5), per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """

    def __init__(self, mul=(0.8, 1.2), per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Multiply, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.mul = iap.handle_continuous_param(
            mul, "mul", value_range=None, tuple_to_uniform=True,
            list_to_choice=True)
        self.per_channel = iap.handle_probability_param(
            per_channel, "per_channel")

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images
        nb_images = len(images)
        nb_channels_max = meta.estimate_max_number_of_channels(images)
        rss = random_state.duplicate(2)
        per_channel_samples = self.per_channel.draw_samples(
            (nb_images,), random_state=rss[0])
        mul_samples = self.mul.draw_samples(
            (nb_images, nb_channels_max), random_state=rss[1])

        gen = enumerate(zip(images, mul_samples, per_channel_samples))
        for i, (image, mul_samples_i, per_channel_samples_i) in gen:
            nb_channels = image.shape[2]

            # Example code to directly multiply images via image*sample
            # (uint8 only) -- apparently slower than LUT
            # if per_channel_samples_i > 0.5:
            #     result = []
            #     image = image.astype(np.float32)
            #     mul_samples_i = mul_samples_i.astype(np.float32)
            #     for c, mul in enumerate(mul_samples_i[0:nb_channels]):
            #         result.append(
            #             np.clip(
            #                 image[..., c:c+1] * mul, 0, 255
            #             ).astype(np.uint8))
            #     images[i] = np.concatenate(result, axis=2)
            # else:
            #     images[i] = np.clip(
            #         image.astype(np.float32)
            #         * mul_samples_i[0].astype(np.float32),
            #         0, 255
            #     ).astype(np.uint8)

            if per_channel_samples_i > 0.5:
                mul = mul_samples_i[0:nb_channels]
            else:
                # the if/else here catches the case of the channel axis being 0
                mul = mul_samples_i[0] if mul_samples_i.size > 0 else []
            batch.images[i] = multiply_scalar(image, mul)

        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.mul, self.per_channel]


# TODO merge with Multiply
class MultiplyElementwise(meta.Augmenter):
    """
    Multiply image pixels with values that are pixelwise randomly sampled.

    While the ``Multiply`` Augmenter uses a constant multiplier *per
    image* (and optionally channel), this augmenter samples the multipliers
    to use per image and *per pixel* (and optionally per channel).

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.arithmetic.multiply_elementwise`.

    Parameters
    ----------
    mul : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        The value with which to multiply pixel values in the image.

            * If a number, then that value will always be used.
            * If a tuple ``(a, b)``, then a value from the interval ``[a, b]``
              will be sampled per image and pixel.
            * If a list, then a random value will be sampled from that list
              per image and pixel.
            * If a ``StochasticParameter``, then that parameter will be used to
              sample a new value per image and pixel.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.MultiplyElementwise(2.0)

    Multiply all images by a factor of ``2.0``, making them significantly
    bighter.

    >>> aug = iaa.MultiplyElementwise((0.5, 1.5))

    Samples per image and pixel uniformly a value from the interval
    ``[0.5, 1.5]`` and multiplies the pixel with that value.

    >>> aug = iaa.MultiplyElementwise((0.5, 1.5), per_channel=True)

    Samples per image and pixel *and channel* uniformly a value from the
    interval ``[0.5, 1.5]`` and multiplies the pixel with that value. Therefore,
    used multipliers may differ between channels of the same pixel.

    >>> aug = iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """

    def __init__(self, mul=(0.8, 1.2), per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(MultiplyElementwise, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.mul = iap.handle_continuous_param(
            mul, "mul",
            value_range=None, tuple_to_uniform=True, list_to_choice=True)
        self.per_channel = iap.handle_probability_param(per_channel,
                                                        "per_channel")

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images
        nb_images = len(images)
        rss = random_state.duplicate(1+nb_images)
        per_channel_samples = self.per_channel.draw_samples(
            (nb_images,), random_state=rss[0])
        is_mul_binomial = isinstance(self.mul, iap.Binomial) or (
            isinstance(self.mul, iap.FromLowerResolution)
            and isinstance(self.mul.other_param, iap.Binomial)
        )

        gen = enumerate(zip(images, per_channel_samples, rss[1:]))
        for i, (image, per_channel_samples_i, rs) in gen:
            height, width, nb_channels = image.shape
            sample_shape = (height,
                            width,
                            nb_channels if per_channel_samples_i > 0.5 else 1)
            mul = self.mul.draw_samples(sample_shape, random_state=rs)
            # TODO let Binomial return boolean mask directly instead of [0, 1]
            #      integers?

            # hack to improve performance for Dropout and CoarseDropout
            # converts mul samples to mask if mul is binomial
            if mul.dtype.kind != "b" and is_mul_binomial:
                mul = mul.astype(bool, copy=False)

            batch.images[i] = multiply_elementwise(image, mul)

        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.mul, self.per_channel]


# Added in 0.4.0.
class _CutoutSamples(object):
    # Added in 0.4.0.
    def __init__(self, nb_iterations, pos_x, pos_y, size_h, size_w, squared,
                 fill_mode, cval, fill_per_channel):
        self.nb_iterations = nb_iterations
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.size_h = size_h
        self.size_w = size_w
        self.squared = squared
        self.fill_mode = fill_mode
        self.cval = cval
        self.fill_per_channel = fill_per_channel


class Cutout(meta.Augmenter):
    """Fill one or more rectangular areas in an image using a fill mode.

    See paper "Improved Regularization of Convolutional Neural Networks with
    Cutout" by DeVries and Taylor.

    In contrast to the paper, this implementation also supports replacing
    image sub-areas with gaussian noise, random intensities or random RGB
    colors. It also supports non-squared areas. While the paper uses
    absolute pixel values for the size and position, this implementation
    uses relative values, which seems more appropriate for mixed-size
    datasets. The position parameter furthermore allows more flexibility, e.g.
    gaussian distributions around the center.

    .. note::

        This augmenter affects only image data. Other datatypes (e.g.
        segmentation map pixels or keypoints within the filled areas)
        are not affected.

    .. note::

        Gaussian fill mode will assume that float input images contain values
        in the interval ``[0.0, 1.0]`` and hence sample values from a
        gaussian within that interval, i.e. from ``N(0.5, std=0.5/3)``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.arithmetic.cutout_`.

    Parameters
    ----------
    nb_iterations : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        How many rectangular areas to fill.

            * If ``int``: Exactly that many areas will be filled on all images.
            * If ``tuple`` ``(a, b)``: A value from the interval ``[a, b]``
              will be sampled per image.
            * If ``list``: A random value will be sampled from that ``list``
              per image.
            * If ``StochasticParameter``: That parameter will be used to
              sample ``(B,)`` values per batch of ``B`` images.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        Defines the position of each area to fill.
        Analogous to the definition in e.g.
        :class:`~imgaug.augmenters.size.CropToFixedSize`.
        Usually, ``uniform`` (anywhere in the image) or ``normal`` (anywhere
        in the image with preference around the center) are sane values.

    size : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        The size of the rectangle to fill as a fraction of the corresponding
        image size, i.e. with value range ``[0.0, 1.0]``. The size is sampled
        independently per image axis.

            * If ``number``: Exactly that size is always used.
            * If ``tuple`` ``(a, b)``: A value from the interval ``[a, b]``
              will be sampled per area and axis.
            * If ``list``: A random value will be sampled from that ``list``
              per area and axis.
            * If ``StochasticParameter``: That parameter will be used to
              sample ``(N, 2)`` values per batch, where ``N`` is the total
              number of areas to fill within the whole batch.

    squared : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to generate only squared areas cutout areas or allow
        rectangular ones. If this evaluates to a true-like value, the
        first value from `size` will be converted to absolute pixels and used
        for both axes.

        If this value is a float ``p``, then for ``p`` percent of all areas
        to be filled `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    fill_mode : str or list of str or imgaug.parameters.StochasticParameter, optional
        Mode to use in order to fill areas. Corresponds to ``mode`` parameter
        in some other augmenters. Valid strings for the mode are:

            * ``contant``: Fill each area with a single value.
            * ``gaussian``: Fill each area with gaussian noise.

        Valid datatypes are:

            * If ``str``: Exactly that mode will alaways be used.
            * If ``list``: A random value will be sampled from that ``list``
              per area.
            * If ``StochasticParameter``: That parameter will be used to
              sample ``(N,)`` values per batch, where ``N`` is the total number
              of areas to fill within the whole batch.

    cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        The value to use (i.e. the color) to fill areas if `fill_mode` is
        ```constant``.

            * If ``number``: Exactly that value is used for all areas
              and channels.
            * If ``tuple`` ``(a, b)``: A value from the interval ``[a, b]``
              will be sampled per area (and channel if ``per_channel=True``).
            * If ``list``: A random value will be sampled from that ``list``
              per area (and channel if ``per_channel=True``).
            * If ``StochasticParameter``: That parameter will be used to
              sample ``(N, Cmax)`` values per batch, where ``N`` is the total
              number of areas to fill within the whole batch and ``Cmax``
              is the maximum number of channels in any image (usually ``3``).
              If ``per_channel=False``, only the first value of the second
              axis is used.

    fill_per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to fill each area in a channelwise fashion (``True``) or
        not (``False``).
        The behaviour per fill mode is:

            * ``constant``: Whether to fill all channels with the same value
              (i.e, grayscale) or different values (i.e. usually RGB color).
            * ``gaussian``: Whether to sample once from a gaussian and use the
              values for all channels (i.e. grayscale) or to sample
              channelwise (i.e. RGB colors)

        If this value is a float ``p``, then for ``p`` percent of all areas
        to be filled `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Cutout(nb_iterations=2)

    Fill per image two random areas, by default with grayish pixels.

    >>> aug = iaa.Cutout(nb_iterations=(1, 5), size=0.2, squared=False)

    Fill per image between one and five areas, each having ``20%``
    of the corresponding size of the height and width (for non-square
    images this results in non-square areas to be filled).

    >>> aug = iaa.Cutout(fill_mode="constant", cval=255)

    Fill all areas with white pixels.

    >>> aug = iaa.Cutout(fill_mode="constant", cval=(0, 255),
    >>>                  fill_per_channel=0.5)

    Fill ``50%`` of all areas with a random intensity value between
    ``0`` and ``256``. Fill the other ``50%`` of all areas with
    random colors.

    >>> aug = iaa.Cutout(fill_mode="gaussian", fill_per_channel=True)

    Fill areas with gaussian channelwise noise (i.e. usually RGB).

    """

    # Added in 0.4.0.
    def __init__(self,
                 nb_iterations=1,
                 position="uniform",
                 size=0.2,
                 squared=True,
                 fill_mode="constant",
                 cval=128,
                 fill_per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        from .size import _handle_position_parameter  # TODO move to iap
        from .geometric import _handle_cval_arg  # TODO move to iap

        super(Cutout, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        self.nb_iterations = iap.handle_discrete_param(
            nb_iterations, "nb_iterations", value_range=(0, None),
            tuple_to_uniform=True, list_to_choice=True, allow_floats=False)
        self.position = _handle_position_parameter(position)
        self.size = iap.handle_continuous_param(
            size, "size", value_range=(0.0, 1.0+1e-4),
            tuple_to_uniform=True, list_to_choice=True)
        self.squared = iap.handle_probability_param(squared, "squared")
        self.fill_mode = self._handle_fill_mode_param(fill_mode)
        self.cval = _handle_cval_arg(cval)
        self.fill_per_channel = iap.handle_probability_param(
            fill_per_channel, "fill_per_channel")

    # Added in 0.4.0.
    @classmethod
    def _handle_fill_mode_param(cls, fill_mode):
        if ia.is_string(fill_mode):
            assert fill_mode in _CUTOUT_FILL_MODES, (
                "Expected 'fill_mode' to be one of: %s. Got %s." % (
                    str(list(_CUTOUT_FILL_MODES.keys())), fill_mode))
            return iap.Deterministic(fill_mode)
        if isinstance(fill_mode, iap.StochasticParameter):
            return fill_mode
        assert ia.is_iterable(fill_mode), (
            "Expected 'fill_mode' to be a string, "
            "StochasticParameter or list of strings. Got type %s." % (
                type(fill_mode).__name__))
        return iap.Choice(fill_mode)

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        samples = self._draw_samples(batch.images, random_state)

        # map from xyhw to xyxy (both relative coords)
        cutout_height_half = samples.size_h / 2
        cutout_width_half = samples.size_w / 2
        x1_rel = samples.pos_x - cutout_width_half
        y1_rel = samples.pos_y - cutout_height_half
        x2_rel = samples.pos_x + cutout_width_half
        y2_rel = samples.pos_y + cutout_height_half

        nb_iterations_sum = 0
        gen = enumerate(zip(batch.images, samples.nb_iterations))
        for i, (image, nb_iterations) in gen:
            start = nb_iterations_sum
            end = start + nb_iterations

            height, width = image.shape[0:2]

            # map from relative xyxy to absolute xyxy coords
            batch.images[i] = self._augment_image_by_samples(
                image,
                x1_rel[start:end] * width,
                y1_rel[start:end] * height,
                x2_rel[start:end] * width,
                y2_rel[start:end] * height,
                samples.squared[start:end],
                samples.fill_mode[start:end],
                samples.cval[start:end],
                samples.fill_per_channel[start:end],
                random_state)

            nb_iterations_sum += nb_iterations

        return batch

    # Added in 0.4.0.
    def _draw_samples(self, images, random_state):
        rngs = random_state.duplicate(8)
        nb_rows = len(images)
        nb_channels_max = meta.estimate_max_number_of_channels(images)

        nb_iterations = self.nb_iterations.draw_samples(
            (nb_rows,), random_state=rngs[0])
        nb_dropped_areas = int(np.sum(nb_iterations))

        if isinstance(self.position, tuple):
            pos_x = self.position[0].draw_samples((nb_dropped_areas,),
                                                  random_state=rngs[1])
            pos_y = self.position[1].draw_samples((nb_dropped_areas,),
                                                  random_state=rngs[2])
        else:
            pos = self.position.draw_samples((nb_dropped_areas, 2),
                                             random_state=rngs[1])
            pos_x = pos[:, 0]
            pos_y = pos[:, 1]

        size = self.size.draw_samples((nb_dropped_areas, 2),
                                      random_state=rngs[3])
        squared = self.squared.draw_samples((nb_dropped_areas,),
                                            random_state=rngs[4])
        fill_mode = self.fill_mode.draw_samples(
            (nb_dropped_areas,), random_state=rngs[5])

        cval = self.cval.draw_samples((nb_dropped_areas, nb_channels_max),
                                      random_state=rngs[6])

        fill_per_channel = self.fill_per_channel.draw_samples(
            (nb_dropped_areas,), random_state=rngs[7])

        return _CutoutSamples(
            nb_iterations=nb_iterations,
            pos_x=pos_x,
            pos_y=pos_y,
            size_h=size[:, 0],
            size_w=size[:, 1],
            squared=squared,
            fill_mode=fill_mode,
            cval=cval,
            fill_per_channel=fill_per_channel
        )

    # Added in 0.4.0.
    @classmethod
    def _augment_image_by_samples(cls, image, x1, y1, x2, y2, squared,
                                  fill_mode, cval, fill_per_channel,
                                  random_state):
        for i, x1_i in enumerate(x1):
            x2_i = x2[i]
            if squared[i] >= 0.5:
                height_h = (y2[i] - y1[i]) / 2
                x_center = x1_i + (x2_i - x1_i) / 2
                x1_i = x_center - height_h
                x2_i = x_center + height_h

            image = cutout_(
                image,
                x1=x1_i,
                y1=y1[i],
                x2=x2_i,
                y2=y2[i],
                fill_mode=fill_mode[i],
                cval=cval[i],
                fill_per_channel=fill_per_channel[i],
                seed=random_state)
        return image

    # Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.nb_iterations, self.position, self.size, self.squared,
                self.fill_mode, self.cval, self.fill_per_channel]


# TODO verify that (a, b) still leads to a p being sampled per image and not
#      per batch
class Dropout(MultiplyElementwise):
    """
    Set a fraction of pixels in images to zero.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.arithmetic.MultiplyElementwise`.

    Parameters
    ----------
    p : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The probability of any pixel being dropped (i.e. to set it to zero).

            * If a float, then that value will be used for all images. A value
              of ``1.0`` would mean that all pixels will be dropped
              and ``0.0`` that no pixels will be dropped. A value of ``0.05``
              corresponds to ``5`` percent of all pixels being dropped.
            * If a tuple ``(a, b)``, then a value ``p`` will be sampled from
              the interval ``[a, b]`` per image and be used as the pixel's
              dropout probability.
            * If a list, then a value will be sampled from that list per
              batch and used as the probability.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine per pixel whether it should be *kept* (sampled value
              of ``>0.5``) or shouldn't be kept (sampled value of ``<=0.5``).
              If you instead want to provide the probability as a stochastic
              parameter, you can usually do ``imgaug.parameters.Binomial(1-p)``
              to convert parameter `p` to a 0/1 representation.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Dropout(0.02)

    Drops ``2`` percent of all pixels.

    >>> aug = iaa.Dropout((0.0, 0.05))

    Drops in each image a random fraction of all pixels, where the fraction
    is uniformly sampled from the interval ``[0.0, 0.05]``.

    >>> aug = iaa.Dropout(0.02, per_channel=True)

    Drops ``2`` percent of all pixels in a channelwise fashion, i.e. it is
    unlikely for any pixel to have all channels set to zero (black pixels).

    >>> aug = iaa.Dropout(0.02, per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for ``50`` percent of all images.

    """
    def __init__(self, p=(0.0, 0.05), per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        p_param = _handle_dropout_probability_param(p, "p")

        super(Dropout, self).__init__(
            p_param,
            per_channel=per_channel,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


# Added in 0.4.0.
def _handle_dropout_probability_param(p, name):
    if ia.is_single_number(p):
        p_param = iap.Binomial(1 - p)
    elif isinstance(p, tuple):
        assert len(p) == 2, (
            "Expected `%s` to be given as a tuple containing exactly 2 values, "
            "got %d values." % (name, len(p),))
        assert p[0] < p[1], (
            "Expected `%s` to be given as a tuple containing exactly 2 values "
            "(a, b) with a < b. Got %.4f and %.4f." % (name, p[0], p[1]))
        assert 0 <= p[0] <= 1.0 and 0 <= p[1] <= 1.0, (
            "Expected `%s` given as tuple to only contain values in the "
            "interval [0.0, 1.0], got %.4f and %.4f." % (name, p[0], p[1]))

        p_param = iap.Binomial(iap.Uniform(1 - p[1], 1 - p[0]))
    elif ia.is_iterable(p):
        assert all([ia.is_single_number(v) for v in p]), (
            "Expected iterable parameter '%s' to only contain numbers, "
            "got %s." % (name, [type(v) for v in p],))
        assert all([0 <= p_i <= 1.0 for p_i in p]), (
            "Expected iterable parameter '%s' to only contain probabilities "
            "in the interval [0.0, 1.0], got values %s." % (
                name, ", ".join(["%.4f" % (p_i,) for p_i in p])))
        p_param = iap.Binomial(1 - iap.Choice(p))
    elif isinstance(p, iap.StochasticParameter):
        p_param = p
    else:
        raise Exception(
            "Expected `%s` to be float or int or tuple (<number>, <number>) "
            "or StochasticParameter, got type '%s'." % (
                name, type(p).__name__,))

    return p_param


# TODO invert size_px and size_percent so that larger values denote larger
#      areas being dropped instead of the opposite way around
class CoarseDropout(MultiplyElementwise):
    """
    Set rectangular areas within images to zero.

    In contrast to ``Dropout``, these areas can have larger sizes.
    (E.g. you might end up with three large black rectangles in an image.)
    Note that the current implementation leads to correlated sizes,
    so if e.g. there is any thin and high rectangle that is dropped, there is
    a high likelihood that all other dropped areas are also thin and high.

    This method is implemented by generating the dropout mask at a
    lower resolution (than the image has) and then upsampling the mask
    before dropping the pixels.

    This augmenter is similar to Cutout. Usually, cutout is defined as an
    operation that drops exactly one rectangle from an image, while here
    ``CoarseDropout`` can drop multiple rectangles (with some correlation
    between the sizes of these rectangles).

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.arithmetic.MultiplyElementwise`.

    Parameters
    ----------
    p : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The probability of any pixel being dropped (i.e. set to zero) in
        the lower-resolution dropout mask.

            * If a float, then that value will be used for all pixels. A value
              of ``1.0`` would mean, that all pixels will be dropped. A value
              of ``0.0`` would lead to no pixels being dropped.
            * If a tuple ``(a, b)``, then a value ``p`` will be sampled from
              the interval ``[a, b]`` per image and be used as the dropout
              probability.
            * If a list, then a value will be sampled from that list per
              batch and used as the probability.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine per pixel whether it should be *kept* (sampled value
              of ``>0.5``) or shouldn't be kept (sampled value of ``<=0.5``).
              If you instead want to provide the probability as a stochastic
              parameter, you can usually do ``imgaug.parameters.Binomial(1-p)``
              to convert parameter `p` to a 0/1 representation.

    size_px : None or int or tuple of int or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the dropout
        mask in absolute pixel dimensions.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being dropped (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_percent` must be set.
            * If an integer, then that size will always be used for both height
              and width. E.g. a value of ``3`` would lead to a ``3x3`` mask,
              which is then upsampled to ``HxW``, where ``H`` is the image size
              and ``W`` the image width.
            * If a tuple ``(a, b)``, then two values ``M``, ``N`` will be
              sampled from the discrete interval ``[a..b]``. The dropout mask
              will then be generated at size ``MxN`` and upsampled to ``HxW``.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : None or float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the dropout
        mask *in percent* of the input image.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being dropped (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_px` must be set.
            * If a float, then that value will always be used as the percentage
              of the height and width (relative to the original size). E.g. for
              value ``p``, the mask will be sampled from ``(p*H)x(p*W)`` and
              later upsampled to ``HxW``.
            * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be
              sampled from the interval ``(a, b)`` and used as the size
              fractions, i.e the mask size will be ``(m*H)x(n*W)``.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    min_size : int, optional
        Minimum height and width of the low resolution mask. If
        `size_percent` or `size_px` leads to a lower value than this,
        `min_size` will be used instead. This should never have a value of
        less than ``2``, otherwise one may end up with a ``1x1`` low resolution
        mask, leading easily to the whole image being dropped.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.CoarseDropout(0.02, size_percent=0.5)

    Drops ``2`` percent of all pixels on a lower-resolution image that has
    ``50`` percent of the original image's size, leading to dropped areas that
    have roughly ``2x2`` pixels size.

    >>> aug = iaa.CoarseDropout((0.0, 0.05), size_percent=(0.05, 0.5))

    Generates a dropout mask at ``5`` to ``50`` percent of each input image's
    size. In that mask, ``0`` to ``5`` percent of all pixels are marked as
    being dropped. The mask is afterwards projected to the input image's
    size to apply the actual dropout operation.

    >>> aug = iaa.CoarseDropout((0.0, 0.05), size_px=(2, 16))

    Same as the previous example, but the lower resolution image has ``2`` to
    ``16`` pixels size. On images of e.g. ``224x224` pixels in size this would
    lead to fairly large areas being dropped (height/width of ``224/2`` to
    ``224/16``).

    >>> aug = iaa.CoarseDropout(0.02, size_percent=0.5, per_channel=True)

    Drops ``2`` percent of all pixels at ``50`` percent resolution (``2x2``
    sizes) in a channel-wise fashion, i.e. it is unlikely for any pixel to
    have all channels set to zero (black pixels).

    >>> aug = iaa.CoarseDropout(0.02, size_percent=0.5, per_channel=0.5)

    Same as the previous example, but the `per_channel` feature is only active
    for ``50`` percent of all images.

    """
    def __init__(self, p=(0.02, 0.1), size_px=None, size_percent=None,
                 per_channel=False, min_size=3,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        p_param = _handle_dropout_probability_param(p, "p")

        if size_px is not None:
            p_param = iap.FromLowerResolution(other_param=p_param,
                                              size_px=size_px,
                                              min_size=min_size)
        elif size_percent is not None:
            p_param = iap.FromLowerResolution(other_param=p_param,
                                              size_percent=size_percent,
                                              min_size=min_size)
        else:
            # default if neither size_px nor size_percent is provided
            # is size_px=(3, 8)
            p_param = iap.FromLowerResolution(other_param=p_param,
                                              size_px=(3, 8),
                                              min_size=min_size)

        super(CoarseDropout, self).__init__(
            p_param,
            per_channel=per_channel,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Dropout2d(meta.Augmenter):
    """Drop random channels from images.

    For image data, dropped channels will be filled with zeros.

    .. note::

        This augmenter may also set the arrays of heatmaps and segmentation
        maps to zero and remove all coordinate-based data (e.g. it removes
        all bounding boxes on images that were filled with zeros).
        It does so if and only if *all* channels of an image are dropped.
        If ``nb_keep_channels >= 1`` then that never happens.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    p : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The probability of any channel to be dropped (i.e. set to zero).

            * If a ``float``, then that value will be used for all channels.
              A value of ``1.0`` would mean, that all channels will be dropped.
              A value of ``0.0`` would lead to no channels being dropped.
            * If a tuple ``(a, b)``, then a value ``p`` will be sampled from
              the interval ``[a, b)`` per batch and be used as the dropout
              probability.
            * If a list, then a value will be sampled from that list per
              batch and used as the probability.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine per channel whether it should be *kept* (sampled value
              of ``>=0.5``) or shouldn't be kept (sampled value of ``<0.5``).
              If you instead want to provide the probability as a stochastic
              parameter, you can usually do ``imgaug.parameters.Binomial(1-p)``
              to convert parameter `p` to a 0/1 representation.

    nb_keep_channels : int
        Minimum number of channels to keep unaltered in all images.
        E.g. a value of ``1`` means that at least one channel in every image
        will not be dropped, even if ``p=1.0``. Set to ``0`` to allow dropping
        all channels.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Dropout2d(p=0.5)

    Create a dropout augmenter that drops on average half of all image
    channels. Dropped channels will be filled with zeros. At least one
    channel is kept unaltered in each image (default setting).

    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Dropout2d(p=0.5, nb_keep_channels=0)

    Create a dropout augmenter that drops on average half of all image
    channels *and* may drop *all* channels in an image (i.e. images may
    contain nothing but zeros).

    """

    # Added in 0.4.0.
    def __init__(self, p=0.1, nb_keep_channels=1,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Dropout2d, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        self.p = _handle_dropout_probability_param(p, "p")
        self.nb_keep_channels = max(nb_keep_channels, 0)

        self._drop_images = True
        self._drop_heatmaps = True
        self._drop_segmentation_maps = True
        self._drop_keypoints = True
        self._drop_bounding_boxes = True
        self._drop_polygons = True
        self._drop_line_strings = True

        self._heatmaps_cval = 0.0
        self._segmentation_maps_cval = 0

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        imagewise_drop_channel_ids, all_dropped_ids = self._draw_samples(
            batch, random_state)

        if batch.images is not None:
            for image, drop_ids in zip(batch.images,
                                       imagewise_drop_channel_ids):
                image[:, :, drop_ids] = 0

        # Skip the non-image data steps below if we won't modify non-image
        # anyways. Minor performance improvement.
        if len(all_dropped_ids) == 0:
            return batch

        if batch.heatmaps is not None and self._drop_heatmaps:
            cval = self._heatmaps_cval
            for drop_idx in all_dropped_ids:
                batch.heatmaps[drop_idx].arr_0to1[...] = cval

        if batch.segmentation_maps is not None and self._drop_segmentation_maps:
            cval = self._segmentation_maps_cval
            for drop_idx in all_dropped_ids:
                batch.segmentation_maps[drop_idx].arr[...] = cval

        for attr_name in ["keypoints", "bounding_boxes", "polygons",
                          "line_strings"]:
            do_drop = getattr(self, "_drop_%s" % (attr_name,))
            attr_value = getattr(batch, attr_name)
            if attr_value is not None and do_drop:
                for drop_idx in all_dropped_ids:
                    # same as e.g.:
                    #     batch.bounding_boxes[drop_idx].bounding_boxes = []
                    setattr(attr_value[drop_idx], attr_name, [])

        return batch

    # Added in 0.4.0.
    def _draw_samples(self, batch, random_state):
        # maybe noteworthy here that the channel axis can have size 0,
        # e.g. (5, 5, 0)
        shapes = batch.get_rowwise_shapes()
        shapes = [shape
                  if len(shape) >= 2
                  else tuple(list(shape) + [1])
                  for shape in shapes]
        imagewise_channels = np.array([
            shape[2] for shape in shapes
        ], dtype=np.int32)

        # channelwise drop value over all images (float <0.5 = drop channel)
        p_samples = self.p.draw_samples((int(np.sum(imagewise_channels)),),
                                        random_state=random_state)

        # We map the flat p_samples array to an imagewise one,
        # convert the mask to channel-ids to drop and remove channel ids if
        # there are more to be dropped than are allowed to be dropped (see
        # nb_keep_channels).
        # We also track all_dropped_ids, which contains the ids of examples
        # (not channel ids!) where all channels were dropped.
        imagewise_channels_to_drop = []
        all_dropped_ids = []
        channel_idx = 0
        for i, nb_channels in enumerate(imagewise_channels):
            p_samples_i = p_samples[channel_idx:channel_idx+nb_channels]

            drop_ids = np.nonzero(p_samples_i < 0.5)[0]
            nb_dropable = max(nb_channels - self.nb_keep_channels, 0)
            if len(drop_ids) > nb_dropable:
                random_state.shuffle(drop_ids)
                drop_ids = drop_ids[:nb_dropable]
            imagewise_channels_to_drop.append(drop_ids)

            all_dropped = (len(drop_ids) == nb_channels)
            if all_dropped:
                all_dropped_ids.append(i)

            channel_idx += nb_channels

        return imagewise_channels_to_drop, all_dropped_ids

    # Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p, self.nb_keep_channels]


class TotalDropout(meta.Augmenter):
    """Drop all channels of a defined fraction of all images.

    For image data, all components of dropped images will be filled with zeros.

    .. note::

        This augmenter also sets the arrays of heatmaps and segmentation
        maps to zero and removes all coordinate-based data (e.g. it removes
        all bounding boxes on images that were filled with zeros).

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    p : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The probability of an image to be filled with zeros.

            * If ``float``: The value will be used for all images.
              A value of ``1.0`` would mean that all images will be set to zero.
              A value of ``0.0`` would lead to no images being set to zero.
            * If ``tuple`` ``(a, b)``: A value ``p`` will be sampled from
              the interval ``[a, b)`` per batch and be used as the dropout
              probability.
            * If a list, then a value will be sampled from that list per
              batch and used as the probability.
            * If ``StochasticParameter``: The parameter will be used to
              determine per image whether it should be *kept* (sampled value
              of ``>=0.5``) or shouldn't be kept (sampled value of ``<0.5``).
              If you instead want to provide the probability as a stochastic
              parameter, you can usually do ``imgaug.parameters.Binomial(1-p)``
              to convert parameter `p` to a 0/1 representation.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.TotalDropout(1.0)

    Create an augmenter that sets *all* components of all images to zero.

    >>> aug = iaa.TotalDropout(0.5)

    Create an augmenter that sets *all* components of ``50%`` of all images to
    zero.

    """

    # Added in 0.4.0.
    def __init__(self, p=1,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(TotalDropout, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        self.p = _handle_dropout_probability_param(p, "p")

        self._drop_images = True
        self._drop_heatmaps = True
        self._drop_segmentation_maps = True
        self._drop_keypoints = True
        self._drop_bounding_boxes = True
        self._drop_polygons = True
        self._drop_line_strings = True

        self._heatmaps_cval = 0.0
        self._segmentation_maps_cval = 0

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        drop_mask = self._draw_samples(batch, random_state)
        drop_ids = None

        if batch.images is not None and self._drop_images:
            if ia.is_np_array(batch.images):
                batch.images[drop_mask, ...] = 0
            else:
                drop_ids = self._generate_drop_ids_once(drop_mask, drop_ids)
                for drop_idx in drop_ids:
                    batch.images[drop_idx][...] = 0

        if batch.heatmaps is not None and self._drop_heatmaps:
            drop_ids = self._generate_drop_ids_once(drop_mask, drop_ids)
            cval = self._heatmaps_cval
            for drop_idx in drop_ids:
                batch.heatmaps[drop_idx].arr_0to1[...] = cval

        if batch.segmentation_maps is not None and self._drop_segmentation_maps:
            drop_ids = self._generate_drop_ids_once(drop_mask, drop_ids)
            cval = self._segmentation_maps_cval
            for drop_idx in drop_ids:
                batch.segmentation_maps[drop_idx].arr[...] = cval

        for attr_name in ["keypoints", "bounding_boxes", "polygons",
                          "line_strings"]:
            do_drop = getattr(self, "_drop_%s" % (attr_name,))
            attr_value = getattr(batch, attr_name)
            if attr_value is not None and do_drop:
                drop_ids = self._generate_drop_ids_once(drop_mask, drop_ids)
                for drop_idx in drop_ids:
                    # same as e.g.:
                    #     batch.bounding_boxes[drop_idx].bounding_boxes = []
                    setattr(attr_value[drop_idx], attr_name, [])

        return batch

    # Added in 0.4.0.
    def _draw_samples(self, batch, random_state):
        p = self.p.draw_samples((batch.nb_rows,), random_state=random_state)
        drop_mask = (p < 0.5)
        return drop_mask

    # Added in 0.4.0.
    @classmethod
    def _generate_drop_ids_once(cls, drop_mask, drop_ids):
        if drop_ids is None:
            drop_ids = np.nonzero(drop_mask)[0]
        return drop_ids

    # Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p]


class ReplaceElementwise(meta.Augmenter):
    """
    Replace pixels in an image with new values.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.arithmetic.replace_elementwise_`.

    Parameters
    ----------
    mask : float or tuple of float or list of float or imgaug.parameters.StochasticParameter
        Mask that indicates the pixels that are supposed to be replaced.
        The mask will be binarized using a threshold of ``0.5``. A value
        of ``1`` then indicates a pixel that is supposed to be replaced.

            * If this is a float, then that value will be used as the
              probability of being a ``1`` in the mask (sampled per image and
              pixel) and hence being replaced.
            * If a tuple ``(a, b)``, then the probability will be uniformly
              sampled per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image and pixel.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample a mask per image.

    replacement : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        The replacement to use at all locations that are marked as ``1`` in
        the mask.

            * If this is a number, then that value will always be used as the
              replacement.
            * If a tuple ``(a, b)``, then the replacement will be sampled
              uniformly per image and pixel from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image and pixel.
            * If a ``StochasticParameter``, then this parameter will be used
              sample replacement values per image and pixel.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = ReplaceElementwise(0.05, [0, 255])

    Replaces ``5`` percent of all pixels in each image by either ``0``
    or ``255``.

    >>> import imgaug.augmenters as iaa
    >>> aug = ReplaceElementwise(0.1, [0, 255], per_channel=0.5)

    For ``50%`` of all images, replace ``10%`` of all pixels with either the
    value ``0`` or the value ``255`` (same as in the previous example). For
    the other ``50%`` of all images, replace *channelwise* ``10%`` of all
    pixels with either the value ``0`` or the value ``255``. So, it will be
    very rare for each pixel to have all channels replaced by ``255`` or
    ``0``.

    >>> import imgaug.augmenters as iaa
    >>> import imgaug.parameters as iap
    >>> aug = ReplaceElementwise(0.1, iap.Normal(128, 0.4*128), per_channel=0.5)

    Replace ``10%`` of all pixels by gaussian noise centered around ``128``.
    Both the replacement mask and the gaussian noise are sampled channelwise
    for ``50%`` of all images.

    >>> import imgaug.augmenters as iaa
    >>> import imgaug.parameters as iap
    >>> aug = ReplaceElementwise(
    >>>     iap.FromLowerResolution(iap.Binomial(0.1), size_px=8),
    >>>     iap.Normal(128, 0.4*128),
    >>>     per_channel=0.5)

    Replace ``10%`` of all pixels by gaussian noise centered around ``128``.
    Sample the replacement mask at a lower resolution (``8x8`` pixels) and
    upscale it to the image size, resulting in coarse areas being replaced by
    gaussian noise.

    """

    def __init__(self, mask, replacement, per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(ReplaceElementwise, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.mask = iap.handle_probability_param(
            mask, "mask", tuple_to_uniform=True, list_to_choice=True)
        self.replacement = iap.handle_continuous_param(replacement,
                                                       "replacement")
        self.per_channel = iap.handle_probability_param(per_channel,
                                                        "per_channel")

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images
        nb_images = len(images)
        rss = random_state.duplicate(1+2*nb_images)
        per_channel_samples = self.per_channel.draw_samples(
            (nb_images,), random_state=rss[0])

        gen = enumerate(zip(images, per_channel_samples, rss[1::2], rss[2::2]))
        for i, (image, per_channel_i, rs_mask, rs_replacement) in gen:
            height, width, nb_channels = image.shape
            sampling_shape = (height,
                              width,
                              nb_channels if per_channel_i > 0.5 else 1)
            mask_samples = self.mask.draw_samples(sampling_shape,
                                                  random_state=rs_mask)

            # TODO add separate per_channels for mask and replacement
            # TODO add test that replacement with per_channel=False is not
            #      sampled per channel
            if per_channel_i <= 0.5:
                nb_channels = image.shape[-1]
                replacement_samples = self.replacement.draw_samples(
                    (int(np.sum(mask_samples[:, :, 0])),),
                    random_state=rs_replacement)
                # important here to use repeat instead of tile. repeat
                # converts e.g. [0, 1, 2] to [0, 0, 1, 1, 2, 2], while tile
                # leads to [0, 1, 2, 0, 1, 2]. The assignment below iterates
                # over each channel and pixel simultaneously, *not* first
                # over all pixels of channel 0, then all pixels in
                # channel 1, ...
                replacement_samples = np.repeat(replacement_samples,
                                                nb_channels)
            else:
                replacement_samples = self.replacement.draw_samples(
                    (int(np.sum(mask_samples)),), random_state=rs_replacement)

            batch.images[i] = replace_elementwise_(image, mask_samples,
                                                   replacement_samples)

        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.mask, self.replacement, self.per_channel]


class SaltAndPepper(ReplaceElementwise):
    """
    Replace pixels in images with salt/pepper noise (white/black-ish colors).

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.arithmetic.ReplaceElementwise`.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
        Probability of replacing a pixel to salt/pepper noise.

            * If a float, then that value will always be used as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled
              uniformly per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a image-sized mask will be
              sampled from that parameter per image. Any value ``>0.5`` in
              that mask will be replaced with salt and pepper noise.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.SaltAndPepper(0.05)

    Replace ``5%`` of all pixels with salt and pepper noise.

    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.SaltAndPepper(0.05, per_channel=True)

    Replace *channelwise* ``5%`` of all pixels with salt and pepper
    noise.

    """
    def __init__(self, p=(0.0, 0.03), per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(SaltAndPepper, self).__init__(
            mask=p,
            replacement=iap.Beta(0.5, 0.5) * 255,
            per_channel=per_channel,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class ImpulseNoise(SaltAndPepper):
    """
    Add impulse noise to images.

    This is identical to ``SaltAndPepper``, except that `per_channel` is
    always set to ``True``.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.arithmetic.SaltAndPepper`.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
        Probability of replacing a pixel to impulse noise.

            * If a float, then that value will always be used as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled
              uniformly per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a image-sized mask will be
              sampled from that parameter per image. Any value ``>0.5`` in
              that mask will be replaced with impulse noise noise.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.ImpulseNoise(0.1)

    Replace ``10%`` of all pixels with impulse noise.

    """

    def __init__(self, p=(0.0, 0.03),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(ImpulseNoise, self).__init__(
            p=p,
            per_channel=True,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class CoarseSaltAndPepper(ReplaceElementwise):
    """
    Replace rectangular areas in images with white/black-ish pixel noise.

    This adds salt and pepper noise (noisy white-ish and black-ish pixels) to
    rectangular areas within the image. Note that this means that within these
    rectangular areas the color varies instead of each rectangle having only
    one color.

    See also the similar ``CoarseDropout``.

    TODO replace dtype support with uint8 only, because replacement is
         geared towards that value range

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.arithmetic.ReplaceElementwise`.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
        Probability of changing a pixel to salt/pepper noise.

            * If a float, then that value will always be used as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled
              uniformly per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a lower-resolution mask will
              be sampled from that parameter per image. Any value ``>0.5`` in
              that mask will denote a spatial location that is to be replaced
              by salt and pepper noise.

    size_px : int or tuple of int or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the
        replacement mask in absolute pixel dimensions.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being replaced (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_percent` must be set.
            * If an integer, then that size will always be used for both height
              and width. E.g. a value of ``3`` would lead to a ``3x3`` mask,
              which is then upsampled to ``HxW``, where ``H`` is the image size
              and ``W`` the image width.
            * If a tuple ``(a, b)``, then two values ``M``, ``N`` will be
              sampled from the discrete interval ``[a..b]``. The mask
              will then be generated at size ``MxN`` and upsampled to ``HxW``.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the
        replacement mask *in percent* of the input image.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being replaced (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_px` must be set.
            * If a float, then that value will always be used as the percentage
              of the height and width (relative to the original size). E.g. for
              value ``p``, the mask will be sampled from ``(p*H)x(p*W)`` and
              later upsampled to ``HxW``.
            * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be
              sampled from the interval ``(a, b)`` and used as the size
              fractions, i.e the mask size will be ``(m*H)x(n*W)``.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    min_size : int, optional
        Minimum height and width of the low resolution mask. If
        `size_percent` or `size_px` leads to a lower value than this,
        `min_size` will be used instead. This should never have a value of
        less than ``2``, otherwise one may end up with a ``1x1`` low resolution
        mask, leading easily to the whole image being replaced.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1))

    Marks ``5%`` of all pixels in a mask to be replaced by salt/pepper
    noise. The mask has ``1%`` to ``10%`` the size of the input image.
    The mask is then upscaled to the input image size, leading to large
    rectangular areas being marked as to be replaced. These areas are then
    replaced in the input image by salt/pepper noise.

    >>> aug = iaa.CoarseSaltAndPepper(0.05, size_px=(4, 16))

    Same as in the previous example, but the replacement mask before upscaling
    has a size between ``4x4`` and ``16x16`` pixels (the axis sizes are sampled
    independently, i.e. the mask may be rectangular).

    >>> aug = iaa.CoarseSaltAndPepper(
    >>>    0.05, size_percent=(0.01, 0.1), per_channel=True)

    Same as in the first example, but mask and replacement are each sampled
    independently per image channel.

    """

    def __init__(self, p=(0.02, 0.1), size_px=None, size_percent=None,
                 per_channel=False, min_size=3,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        mask = iap.handle_probability_param(
            p, "p", tuple_to_uniform=True, list_to_choice=True)

        if size_px is not None:
            mask_low = iap.FromLowerResolution(
                other_param=mask, size_px=size_px, min_size=min_size)
        elif size_percent is not None:
            mask_low = iap.FromLowerResolution(
                other_param=mask, size_percent=size_percent, min_size=min_size)
        else:
            mask_low = iap.FromLowerResolution(
                other_param=mask, size_px=(3, 8), min_size=min_size)

        replacement = iap.Beta(0.5, 0.5) * 255

        super(CoarseSaltAndPepper, self).__init__(
            mask=mask_low,
            replacement=replacement,
            per_channel=per_channel,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Salt(ReplaceElementwise):
    """
    Replace pixels in images with salt noise, i.e. white-ish pixels.

    This augmenter is similar to ``SaltAndPepper``, but adds no pepper noise to
    images.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.arithmetic.ReplaceElementwise`.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
        Probability of replacing a pixel with salt noise.

            * If a float, then that value will always be used as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled
              uniformly per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a image-sized mask will be
              sampled from that parameter per image. Any value ``>0.5`` in
              that mask will be replaced with salt noise.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Salt(0.05)

    Replace ``5%`` of all pixels with salt noise (white-ish colors).

    """

    def __init__(self, p=(0.0, 0.03), per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        replacement01 = iap.ForceSign(
            iap.Beta(0.5, 0.5) - 0.5,
            positive=True,
            mode="invert"
        ) + 0.5
        # FIXME max replacement seems to essentially never exceed 254
        replacement = replacement01 * 255

        super(Salt, self).__init__(
            mask=p,
            replacement=replacement,
            per_channel=per_channel,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class CoarseSalt(ReplaceElementwise):
    """
    Replace rectangular areas in images with white-ish pixel noise.

    See also the similar ``CoarseSaltAndPepper``.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.arithmetic.ReplaceElementwise`.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
        Probability of changing a pixel to salt noise.

            * If a float, then that value will always be used as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled
              uniformly per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a lower-resolution mask will
              be sampled from that parameter per image. Any value ``>0.5`` in
              that mask will denote a spatial location that is to be replaced
              by salt noise.

    size_px : int or tuple of int or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the
        replacement mask in absolute pixel dimensions.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being replaced (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_percent` must be set.
            * If an integer, then that size will always be used for both height
              and width. E.g. a value of ``3`` would lead to a ``3x3`` mask,
              which is then upsampled to ``HxW``, where ``H`` is the image size
              and ``W`` the image width.
            * If a tuple ``(a, b)``, then two values ``M``, ``N`` will be
              sampled from the discrete interval ``[a..b]``. The mask
              will then be generated at size ``MxN`` and upsampled to ``HxW``.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the
        replacement mask *in percent* of the input image.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being replaced (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_px` must be set.
            * If a float, then that value will always be used as the percentage
              of the height and width (relative to the original size). E.g. for
              value ``p``, the mask will be sampled from ``(p*H)x(p*W)`` and
              later upsampled to ``HxW``.
            * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be
              sampled from the interval ``(a, b)`` and used as the size
              fractions, i.e the mask size will be ``(m*H)x(n*W)``.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    min_size : int, optional
        Minimum height and width of the low resolution mask. If
        `size_percent` or `size_px` leads to a lower value than this,
        `min_size` will be used instead. This should never have a value of
        less than ``2``, otherwise one may end up with a ``1x1`` low resolution
        mask, leading easily to the whole image being replaced.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.CoarseSalt(0.05, size_percent=(0.01, 0.1))

    Mark ``5%`` of all pixels in a mask to be replaced by salt
    noise. The mask has ``1%`` to ``10%`` the size of the input image.
    The mask is then upscaled to the input image size, leading to large
    rectangular areas being marked as to be replaced. These areas are then
    replaced in the input image by salt noise.

    """

    def __init__(self, p=(0.02, 0.1), size_px=None, size_percent=None,
                 per_channel=False, min_size=3,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        mask = iap.handle_probability_param(
            p, "p", tuple_to_uniform=True, list_to_choice=True)

        if size_px is not None:
            mask_low = iap.FromLowerResolution(
                other_param=mask, size_px=size_px, min_size=min_size)
        elif size_percent is not None:
            mask_low = iap.FromLowerResolution(
                other_param=mask, size_percent=size_percent, min_size=min_size)
        else:
            mask_low = iap.FromLowerResolution(
                other_param=mask, size_px=(3, 8), min_size=min_size)

        replacement01 = iap.ForceSign(
            iap.Beta(0.5, 0.5) - 0.5,
            positive=True,
            mode="invert"
        ) + 0.5
        replacement = replacement01 * 255

        super(CoarseSalt, self).__init__(
            mask=mask_low,
            replacement=replacement,
            per_channel=per_channel,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Pepper(ReplaceElementwise):
    """
    Replace pixels in images with pepper noise, i.e. black-ish pixels.

    This augmenter is similar to ``SaltAndPepper``, but adds no salt noise to
    images.

    This augmenter is similar to ``Dropout``, but slower and the black pixels
    are not uniformly black.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.arithmetic.ReplaceElementwise`.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
        Probability of replacing a pixel with pepper noise.

            * If a float, then that value will always be used as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled
              uniformly per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a image-sized mask will be
              sampled from that parameter per image. Any value ``>0.5`` in
              that mask will be replaced with pepper noise.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Pepper(0.05)

    Replace ``5%`` of all pixels with pepper noise (black-ish colors).

    """

    def __init__(self, p=(0.0, 0.05), per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        replacement01 = iap.ForceSign(
            iap.Beta(0.5, 0.5) - 0.5,
            positive=False,
            mode="invert"
        ) + 0.5
        replacement = replacement01 * 255

        super(Pepper, self).__init__(
            mask=p,
            replacement=replacement,
            per_channel=per_channel,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class CoarsePepper(ReplaceElementwise):
    """
    Replace rectangular areas in images with black-ish pixel noise.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.arithmetic.ReplaceElementwise`.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
        Probability of changing a pixel to pepper noise.

            * If a float, then that value will always be used as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled
              uniformly per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a lower-resolution mask will
              be sampled from that parameter per image. Any value ``>0.5`` in
              that mask will denote a spatial location that is to be replaced
              by pepper noise.

    size_px : int or tuple of int or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the
        replacement mask in absolute pixel dimensions.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being replaced (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_percent` must be set.
            * If an integer, then that size will always be used for both height
              and width. E.g. a value of ``3`` would lead to a ``3x3`` mask,
              which is then upsampled to ``HxW``, where ``H`` is the image size
              and ``W`` the image width.
            * If a tuple ``(a, b)``, then two values ``M``, ``N`` will be
              sampled from the discrete interval ``[a..b]``. The mask
              will then be generated at size ``MxN`` and upsampled to ``HxW``.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the
        replacement mask *in percent* of the input image.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being replaced (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_px` must be set.
            * If a float, then that value will always be used as the percentage
              of the height and width (relative to the original size). E.g. for
              value ``p``, the mask will be sampled from ``(p*H)x(p*W)`` and
              later upsampled to ``HxW``.
            * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be
              sampled from the interval ``(a, b)`` and used as the size
              fractions, i.e the mask size will be ``(m*H)x(n*W)``.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    min_size : int, optional
        Minimum size of the low resolution mask, both width and height. If
        `size_percent` or `size_px` leads to a lower value than this, `min_size`
        will be used instead. This should never have a value of less than 2,
        otherwise one may end up with a ``1x1`` low resolution mask, leading
        easily to the whole image being replaced.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.CoarsePepper(0.05, size_percent=(0.01, 0.1))

    Mark ``5%`` of all pixels in a mask to be replaced by pepper
    noise. The mask has ``1%`` to ``10%`` the size of the input image.
    The mask is then upscaled to the input image size, leading to large
    rectangular areas being marked as to be replaced. These areas are then
    replaced in the input image by pepper noise.

    """

    def __init__(self, p=(0.02, 0.1), size_px=None, size_percent=None,
                 per_channel=False, min_size=3,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        mask = iap.handle_probability_param(
            p, "p", tuple_to_uniform=True, list_to_choice=True)

        if size_px is not None:
            mask_low = iap.FromLowerResolution(
                other_param=mask, size_px=size_px, min_size=min_size)
        elif size_percent is not None:
            mask_low = iap.FromLowerResolution(
                other_param=mask, size_percent=size_percent, min_size=min_size)
        else:
            mask_low = iap.FromLowerResolution(
                other_param=mask, size_px=(3, 8), min_size=min_size)

        replacement01 = iap.ForceSign(
            iap.Beta(0.5, 0.5) - 0.5,
            positive=False,
            mode="invert"
        ) + 0.5
        replacement = replacement01 * 255

        super(CoarsePepper, self).__init__(
            mask=mask_low,
            replacement=replacement,
            per_channel=per_channel,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Invert(meta.Augmenter):
    """
    Invert all values in images, e.g. turn ``5`` into ``255-5=250``.

    For the standard value range of 0-255 it converts ``0`` to ``255``,
    ``255`` to ``0`` and ``10`` to ``(255-10)=245``.
    Let ``M`` be the maximum value possible, ``m`` the minimum value possible,
    ``v`` a value. Then the distance of ``v`` to ``m`` is ``d=abs(v-m)`` and
    the new value is given by ``v'=M-d``.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.arithmetic.invert_`.

    Parameters
    ----------
    p : float or imgaug.parameters.StochasticParameter, optional
        The probability of an image to be inverted.

            * If a float, then that probability will be used for all images,
              i.e. `p` percent of all images will be inverted.
            * If a ``StochasticParameter``, then that parameter will be queried
              per image and is expected to return values in the interval
              ``[0.0, 1.0]``, where values ``>0.5`` mean that the image
              is supposed to be inverted. Recommended to be some form of
              ``imgaug.parameters.Binomial``.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    min_value : None or number, optional
        Minimum of the value range of input images, e.g. ``0`` for ``uint8``
        images. If set to ``None``, the value will be automatically derived
        from the image's dtype.

    max_value : None or number, optional
        Maximum of the value range of input images, e.g. ``255`` for ``uint8``
        images. If set to ``None``, the value will be automatically derived
        from the image's dtype.

    threshold : None or number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        A threshold to use in order to invert only numbers above or below
        the threshold. If ``None`` no thresholding will be used.

            * If ``None``: No thresholding will be used.
            * If ``number``: The value will be used for all images.
            * If ``tuple`` ``(a, b)``: A value will be uniformly sampled per
              image from the interval ``[a, b)``.
            * If ``list``: A random value will be picked from the list per
              image.
            * If ``StochasticParameter``: Per batch of size ``N``, the
              parameter will be queried once to return ``(N,)`` samples.

    invert_above_threshold : bool or float or imgaug.parameters.StochasticParameter, optional
        If ``True``, only values ``>=threshold`` will be inverted.
        Otherwise, only values ``<threshold`` will be inverted.
        If a ``number``, then expected to be in the interval ``[0.0, 1.0]`` and
        denoting an imagewise probability. If a ``StochasticParameter`` then
        ``(N,)`` values will be sampled from the parameter per batch of size
        ``N`` and interpreted as ``True`` if ``>0.5``.
        If `threshold` is ``None`` this parameter has no effect.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Invert(0.1)

    Inverts the colors in ``10`` percent of all images.

    >>> aug = iaa.Invert(0.1, per_channel=True)

    Inverts the colors in ``10`` percent of all image channels. This may or
    may not lead to multiple channels in an image being inverted.

    >>> aug = iaa.Invert(0.1, per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """
    # when no custom min/max are chosen, all bool, uint, int and float dtypes
    # should be invertable (float tested only up to 64bit)
    # when chosing custom min/max:
    # - bool makes no sense, not allowed
    # - int and float must be increased in resolution if custom min/max values
    #   are chosen, hence they are limited to 32 bit and below
    # - uint64 is converted by numpy's clip to float64, hence loss of accuracy
    # - float16 seems to not be perfectly accurate, but still ok-ish -- was
    #   off by 10 for center value of range (float 16 min, 16), where float
    #   16 min is around -65500
    ALLOW_DTYPES_CUSTOM_MINMAX = [
        np.dtype(dt) for dt in [
            np.uint8, np.uint16, np.uint32,
            np.int8, np.int16, np.int32,
            np.float16, np.float32
        ]
    ]

    def __init__(self, p=1, per_channel=False, min_value=None, max_value=None,
                 threshold=None, invert_above_threshold=0.5,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Invert, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        # TODO allow list and tuple for p
        self.p = iap.handle_probability_param(p, "p")
        self.per_channel = iap.handle_probability_param(per_channel,
                                                        "per_channel")
        self.min_value = min_value
        self.max_value = max_value

        if threshold is None:
            self.threshold = None
        else:
            self.threshold = iap.handle_continuous_param(
                threshold, "threshold", value_range=None, tuple_to_uniform=True,
                list_to_choice=True)
        self.invert_above_threshold = iap.handle_probability_param(
            invert_above_threshold, "invert_above_threshold")

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        samples = self._draw_samples(batch, random_state)

        for i, image in enumerate(batch.images):
            if 0 in image.shape:
                continue

            kwargs = {
                "min_value": samples.min_value[i],
                "max_value": samples.max_value[i],
                "threshold": samples.threshold[i],
                "invert_above_threshold": samples.invert_above_threshold[i]
            }

            if samples.per_channel[i]:
                nb_channels = image.shape[2]
                mask = samples.p[i, :nb_channels]
                image[..., mask] = invert_(image[..., mask], **kwargs)
            else:
                if samples.p[i, 0]:
                    image[:, :, :] = invert_(image, **kwargs)

        return batch

    # Added in 0.4.0.
    def _draw_samples(self, batch, random_state):
        nb_images = batch.nb_rows
        nb_channels = meta.estimate_max_number_of_channels(batch.images)
        p = self.p.draw_samples((nb_images, nb_channels),
                                random_state=random_state)
        p = (p > 0.5)
        per_channel = self.per_channel.draw_samples((nb_images,),
                                                    random_state=random_state)
        per_channel = (per_channel > 0.5)
        min_value = [self.min_value] * nb_images
        max_value = [self.max_value] * nb_images

        if self.threshold is None:
            threshold = [None] * nb_images
        else:
            threshold = self.threshold.draw_samples(
                (nb_images,), random_state=random_state)

        invert_above_threshold = self.invert_above_threshold.draw_samples(
            (nb_images,), random_state=random_state)
        invert_above_threshold = (invert_above_threshold > 0.5)

        return _InvertSamples(
            p=p,
            per_channel=per_channel,
            min_value=min_value,
            max_value=max_value,
            threshold=threshold,
            invert_above_threshold=invert_above_threshold
        )

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p, self.per_channel, self.min_value, self.max_value,
                self.threshold, self.invert_above_threshold]


# Added in 0.4.0.
class _InvertSamples(object):
    # Added in 0.4.0.
    def __init__(self, p, per_channel, min_value, max_value,
                 threshold, invert_above_threshold):
        self.p = p
        self.per_channel = per_channel
        self.min_value = min_value
        self.max_value = max_value
        self.threshold = threshold
        self.invert_above_threshold = invert_above_threshold


class Solarize(Invert):
    """Invert all pixel values above a threshold.

    This is the same as :class:`Invert`, but sets a default threshold around
    ``128`` (+/- 64, decided per image) and default `invert_above_threshold`
    to ``True`` (i.e. only values above the threshold will be inverted).

    See :class:`Invert` for more details.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.arithmetic.Invert`.

    Parameters
    ----------
    p : float or imgaug.parameters.StochasticParameter
        See :class:`Invert`.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        See :class:`Invert`.

    min_value : None or number, optional
        See :class:`Invert`.

    max_value : None or number, optional
        See :class:`Invert`.

    threshold : None or number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :class:`Invert`.

    invert_above_threshold : bool or float or imgaug.parameters.StochasticParameter, optional
        See :class:`Invert`.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Solarize(0.5, threshold=(32, 128))

    Invert the colors in ``50`` percent of all images for pixels with a
    value between ``32`` and ``128`` or more. The threshold is sampled once
    per image. The thresholding operation happens per channel.

    """
    def __init__(self, p=1, per_channel=False, min_value=None, max_value=None,
                 threshold=(128-64, 128+64), invert_above_threshold=True,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Solarize, self).__init__(
            p=p, per_channel=per_channel,
            min_value=min_value, max_value=max_value,
            threshold=threshold, invert_above_threshold=invert_above_threshold,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


# TODO remove from examples
@ia.deprecated("imgaug.contrast.LinearContrast")
def ContrastNormalization(alpha=1.0, per_channel=False,
                          seed=None, name=None,
                          random_state="deprecated",
                          deterministic="deprecated"):
    """
    Change the contrast of images.

    dtype support:

        See ``imgaug.augmenters.contrast.LinearContrast``.

    Deprecated since 0.3.0.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Strength of the contrast normalization. Higher values than 1.0
        lead to higher contrast, lower values decrease the contrast.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be sampled per image
              uniformly from the interval ``[a, b]`` and be used as the alpha
              value.
            * If a list, then a random value will be picked per image from
              that list.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample the alpha value per image.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> iaa.ContrastNormalization((0.5, 1.5))

    Decreases oder improves contrast per image by a random factor between
    ``0.5`` and ``1.5``. The factor ``0.5`` means that any difference from
    the center value (i.e. 128) will be halved, leading to less contrast.

    >>> iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)

    Same as before, but for 50 percent of all images the normalization is done
    independently per channel (i.e. factors can vary per channel for the same
    image). In the other 50 percent of all images, the factor is the same for
    all channels.

    """
    # pylint: disable=invalid-name
    # placed here to avoid cyclic dependency
    from . import contrast as contrast_lib
    return contrast_lib.LinearContrast(
        alpha=alpha, per_channel=per_channel,
        seed=seed, name=name, random_state=random_state, deterministic=deterministic)


# TODO try adding per channel somehow
class JpegCompression(meta.Augmenter):
    """
    Degrade the quality of images by JPEG-compressing them.

    During JPEG compression, high frequency components (e.g. edges) are removed.
    With low compression (strength) only the highest frequency components are
    removed, while very high compression (strength) will lead to only the
    lowest frequency components "surviving". This lowers the image quality.
    For more details, see https://en.wikipedia.org/wiki/Compression_artifact.

    Note that this augmenter still returns images as numpy arrays (i.e. saves
    the images with JPEG compression and then reloads them into arrays). It
    does not return the raw JPEG file content.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.arithmetic.compress_jpeg`.

    Parameters
    ----------
    compression : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Degree of compression used during JPEG compression within value range
        ``[0, 100]``. Higher values denote stronger compression and will cause
        low-frequency components to disappear. Note that JPEG's compression
        strength is also often set as a *quality*, which is the inverse of this
        parameter. Common choices for the *quality* setting are around 80 to 95,
        depending on the image. This translates here to a *compression*
        parameter of around 20 to 5.

            * If a single number, then that value always will be used as the
              compression.
            * If a tuple ``(a, b)``, then the compression will be
              a value sampled uniformly from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image and used as the compression.
            * If a ``StochasticParameter``, then ``N`` samples will be drawn
              from that parameter per ``N`` input images, each representing the
              compression for the ``n``-th image.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.JpegCompression(compression=(70, 99))

    Remove high frequency components in images via JPEG compression with
    a *compression strength* between ``70`` and ``99`` (randomly and
    uniformly sampled per image). This corresponds to a (very low) *quality*
    setting of ``1`` to ``30``.

    """

    def __init__(self, compression=(0, 100),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(JpegCompression, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        # will be converted to int during augmentation, which is why we allow
        # floats here
        self.compression = iap.handle_continuous_param(
            compression, "compression",
            value_range=(0, 100), tuple_to_uniform=True, list_to_choice=True)

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images
        nb_images = len(images)
        samples = self.compression.draw_samples((nb_images,),
                                                random_state=random_state)

        for i, (image, sample) in enumerate(zip(images, samples)):
            batch.images[i] = compress_jpeg(image, int(sample))

        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.compression]
