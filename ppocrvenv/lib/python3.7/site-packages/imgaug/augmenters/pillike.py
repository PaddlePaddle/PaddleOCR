"""
Augmenters that have identical outputs to well-known PIL functions.

The ``like`` in ``pillike`` indicates that the augmenters in this module
have identical outputs and mostly identical inputs to corresponding PIL
functions, but do not *have to* wrap these functions internally. They may
use internally different (e.g. faster) techniques to produce these outputs.

Some of the augmenters in this module may also exist in other modules
under similar name. These other augmenters may currently have the same
outputs as the corresponding PIL functions, but that is not guaranteed
for the future. Use the augmenters in this module if identical outputs
to PIL are required.

List of augmenters:

    * :class:`Solarize`
    * :class:`Posterize`
    * :class:`Equalize`
    * :class:`Autocontrast`
    * :class:`EnhanceColor`
    * :class:`EnhanceContrast`
    * :class:`EnhanceBrightness`
    * :class:`EnhanceSharpness`
    * :class:`FilterBlur`
    * :class:`FilterSmooth`
    * :class:`FilterSmoothMore`
    * :class:`FilterEdgeEnhance`
    * :class:`FilterEdgeEnhanceMore`
    * :class:`FilterFindEdges`
    * :class:`FilterContour`
    * :class:`FilterEmboss`
    * :class:`FilterSharpen`
    * :class:`FilterDetail`
    * :class:`Affine`

Standard usage of these augmenters follows roughly the schema::

    import numpy as np
    import imgaug.augmenters as iaa

    aug = iaa.pillike.Affine(translate_px={"x": (-5, 5)})
    image = np.full((32, 32, 3), 255, dtype=np.uint8)

    images_aug = aug(images=[image, image, image])

Added in 0.4.0.

"""
from __future__ import print_function, division, absolute_import

import six.moves as sm
import numpy as np
import cv2
import PIL.Image
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageFilter

import imgaug as ia
from imgaug.imgaug import _normalize_cv2_input_arr_
from . import meta
from . import arithmetic
from . import color as colorlib
from . import contrast as contrastlib
from . import geometric
from . import size as sizelib
from .. import parameters as iap


# TODO some of the augmenters in this module broke on numpy arrays as
#      image inputs (as opposed to lists of arrays) without any test failing
#      add appropriate tests for that

_EQUALIZE_USE_PIL_BELOW = 64*64  # H*W


# Added in 0.4.0.
def _ensure_valid_shape(image, func_name):
    is_hw1 = image.ndim == 3 and image.shape[-1] == 1
    if is_hw1:
        image = image[:, :, 0]
    assert (
        image.ndim == 2
        or (image.ndim == 3 and image.shape[-1] in [3, 4])
    ), (
        "Can apply %s only to images of "
        "shape (H, W) or (H, W, 1) or (H, W, 3) or (H, W, 4). "
        "Got shape %s." % (func_name, image.shape,))
    return image, is_hw1


def solarize_(image, threshold=128):
    """Invert all array components above a threshold in-place.

    This function has identical outputs to ``PIL.ImageOps.solarize``.
    It does however work in-place.

    Added in 0.4.0.

    **Supported dtypes**:

    See ``~imgaug.augmenters.arithmetic.invert_(min_value=None and max_value=None)``.

    Parameters
    ----------
    image : ndarray
        Image array of shape ``(H,W,[C])``.
        The array *might* be modified in-place.

    threshold : int, optional
        A threshold to use in order to invert only numbers above or below
        the threshold.

    Returns
    -------
    ndarray
        Inverted image.
        This *can* be the same array as input in `image`, modified in-place.

    """
    return arithmetic.invert_(image, threshold=threshold)


def solarize(image, threshold=128):
    """Invert all array components above a threshold.

    This function has identical outputs to ``PIL.ImageOps.solarize``.

    Added in 0.4.0.

    **Supported dtypes**:

    See ``~imgaug.augmenters.arithmetic.invert_(min_value=None and max_value=None)``.

    Parameters
    ----------
    image : ndarray
        Image array of shape ``(H,W,[C])``.

    threshold : int, optional
        A threshold to use in order to invert only numbers above or below
        the threshold.

    Returns
    -------
    ndarray
        Inverted image.

    """
    return arithmetic.invert(image, threshold=threshold)


def posterize_(image, bits):
    """Reduce the number of bits for each color channel in-place.

    This function has identical outputs to ``PIL.ImageOps.posterize``.
    It does however work in-place.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.color.quantize_uniform_to_n_bits_`.

    Parameters
    ----------
    image : ndarray
        Image array of shape ``(H,W,[C])``.

    bits : int
        The number of bits to keep per component.
        Values in the interval ``[1, 8]`` are valid.

    Returns
    -------
    ndarray
        Posterized image.
        This *can* be the same array as input in `image`, modified in-place.

    """
    return colorlib.posterize(image, bits)


def posterize(image, bits):
    """Reduce the number of bits for each color channel.

    This function has identical outputs to ``PIL.ImageOps.posterize``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.color.quantize_uniform_to_n_bits`.

    Parameters
    ----------
    image : ndarray
        Image array of shape ``(H,W,[C])``.

    bits : int
        The number of bits to keep per component.
        Values in the interval ``[1, 8]`` are valid.

    Returns
    -------
    ndarray
        Posterized image.

    """
    return colorlib.posterize(image, bits)


def equalize(image, mask=None):
    """Equalize the image histogram.

    See :func:`~imgaug.augmenters.pillike.equalize_` for details.

    This function is identical in inputs and outputs to
    ``PIL.ImageOps.equalize``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.pillike.equalize_`.

    Parameters
    ----------
    image : ndarray
        ``uint8`` ``(H,W,[C])`` image to equalize.

    mask : None or ndarray, optional
        An optional mask. If given, only the pixels selected by the mask are
        included in the analysis.

    Returns
    -------
    ndarray
        Equalized image.

    """
    # internally used method works in-place by default and hence needs a copy
    size = image.size
    if size == 0:
        return np.copy(image)
    if size >= _EQUALIZE_USE_PIL_BELOW:
        image = np.copy(image)
    return equalize_(image, mask)


def equalize_(image, mask=None):
    """Equalize the image histogram in-place.

    This function applies a non-linear mapping to the input image, in order
    to create a uniform distribution of grayscale values in the output image.

    This function has identical outputs to ``PIL.ImageOps.equalize``.
    It does however work in-place.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        ``uint8`` ``(H,W,[C])`` image to equalize.

    mask : None or ndarray, optional
        An optional mask. If given, only the pixels selected by the mask are
        included in the analysis.

    Returns
    -------
    ndarray
        Equalized image. *Might* have been modified in-place.

    """
    nb_channels = 1 if image.ndim == 2 else image.shape[-1]
    if nb_channels not in [1, 3]:
        result = [equalize_(image[:, :, c])
                  for c in np.arange(nb_channels)]
        return np.stack(result, axis=-1)

    assert image.dtype.name == "uint8", (
        "Expected image of dtype uint8, got dtype %s." % (image.dtype.name,))
    if mask is not None:
        assert mask.ndim == 2, (
            "Expected 2-dimensional mask, got shape %s." % (mask.shape,))
        assert mask.dtype.name == "uint8", (
            "Expected mask of dtype uint8, got dtype %s." % (mask.dtype.name,))

    size = image.size
    if size == 0:
        return image
    if nb_channels == 3 and size < _EQUALIZE_USE_PIL_BELOW:
        return _equalize_pil_(image, mask)
    return _equalize_no_pil_(image, mask)


# note that this is supposed to be a non-PIL reimplementation of PIL's
# equalize, which produces slightly different results from cv2.equalizeHist()
# Added in 0.4.0.
def _equalize_no_pil_(image, mask=None):
    nb_channels = 1 if image.ndim == 2 else image.shape[-1]
    # TODO remove the first axis, no longer needed
    lut = np.empty((1, 256, nb_channels), dtype=np.int32)

    for c_idx in range(nb_channels):
        if image.ndim == 2:
            image_c = image[:, :, np.newaxis]
        else:
            image_c = image[:, :, c_idx:c_idx+1]
        histo = cv2.calcHist(
            [_normalize_cv2_input_arr_(image_c)], [0], mask, [256], [0, 256])
        if len(histo.nonzero()[0]) <= 1:
            lut[0, :, c_idx] = np.arange(256).astype(np.int32)
            continue

        step = np.sum(histo[:-1]) // 255
        if not step:
            lut[0, :, c_idx] = np.arange(256).astype(np.int32)
            continue

        n = step // 2
        cumsum = np.cumsum(histo)
        lut[0, 0, c_idx] = n
        lut[0, 1:, c_idx] = n + cumsum[0:-1]
        lut[0, :, c_idx] //= int(step)
    lut = np.clip(lut, None, 255, out=lut).astype(np.uint8)
    image = ia.apply_lut_(image, lut)
    return image


# Added in 0.4.0.
def _equalize_pil_(image, mask=None):
    if mask is not None:
        mask = PIL.Image.fromarray(mask).convert("L")

    # don't return np.asarray(...) directly as its results are read-only
    image[...] = np.asarray(
        PIL.ImageOps.equalize(
            PIL.Image.fromarray(image),
            mask=mask
        )
    )
    return image


def autocontrast(image, cutoff=0, ignore=None):
    """Maximize (normalize) image contrast.

    This function calculates a histogram of the input image, removes
    **cutoff** percent of the lightest and darkest pixels from the histogram,
    and remaps the image so that the darkest pixel becomes black (``0``), and
    the lightest becomes white (``255``).

    This function has identical outputs to ``PIL.ImageOps.autocontrast``.
    The speed is almost identical.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image for which to enhance the contrast.

    cutoff : number
        How many percent to cut off at the low and high end of the
        histogram. E.g. ``20`` will cut off the lowest and highest ``20%``
        of values. Expected value range is ``[0, 100]``.

    ignore : None or int or iterable of int
        Intensity values to ignore, i.e. to treat as background. If ``None``,
        no pixels will be ignored. Otherwise exactly the given intensity
        value(s) will be ignored.

    Returns
    -------
    ndarray
        Contrast-enhanced image.

    """
    assert image.dtype.name == "uint8", (
        "Can apply autocontrast only to uint8 images, got dtype %s." % (
            image.dtype.name,))

    if 0 in image.shape:
        return np.copy(image)

    standard_channels = (image.ndim == 2 or image.shape[2] == 3)

    if cutoff and standard_channels:
        return _autocontrast_pil(image, cutoff, ignore)
    return _autocontrast_no_pil(image, cutoff, ignore)


# Added in 0.4.0.
def _autocontrast_pil(image, cutoff, ignore):
    # don't return np.asarray(...) as its results are read-only
    return np.array(
        PIL.ImageOps.autocontrast(
            PIL.Image.fromarray(image),
            cutoff=cutoff, ignore=ignore
        )
    )


# This function is only faster than the corresponding PIL function if no
# cutoff is used.
# C901 is "<functionname> is too complex"
# Added in 0.4.0.
def _autocontrast_no_pil(image, cutoff, ignore):  # noqa: C901
    # pylint: disable=invalid-name
    if ignore is not None and not ia.is_iterable(ignore):
        ignore = [ignore]

    result = np.empty_like(image)
    if result.ndim == 2:
        result = result[..., np.newaxis]
    nb_channels = image.shape[2] if image.ndim >= 3 else 1
    for c_idx in sm.xrange(nb_channels):
        # using [0] instead of [int(c_idx)] allows this to work with >4
        # channels
        if image.ndim == 2:
            image_c = image[:, :, np.newaxis]
        else:
            image_c = image[:, :, c_idx:c_idx+1]
        h = cv2.calcHist(
            [_normalize_cv2_input_arr_(image_c)], [0], None, [256], [0, 256])
        if ignore is not None:
            h[ignore] = 0

        if cutoff:
            cs = np.cumsum(h)
            n = cs[-1]
            cut = n * cutoff // 100

            # remove cutoff% pixels from the low end
            lo_cut = cut - cs
            lo_cut_nz = np.nonzero(lo_cut <= 0.0)[0]
            if len(lo_cut_nz) == 0:
                lo = 255
            else:
                lo = lo_cut_nz[0]
            if lo > 0:
                h[:lo] = 0
            h[lo] = lo_cut[lo]

            # remove cutoff% samples from the hi end
            cs_rev = np.cumsum(h[::-1])
            hi_cut = cs_rev - cut
            hi_cut_nz = np.nonzero(hi_cut > 0.0)[0]
            if len(hi_cut_nz) == 0:
                hi = -1
            else:
                hi = 255 - hi_cut_nz[0]
            h[hi+1:] = 0
            if hi > -1:
                h[hi] = hi_cut[255-hi]

        # find lowest/highest samples after preprocessing
        for lo, lo_val in enumerate(h):
            if lo_val:
                break
        for hi in range(255, -1, -1):
            if h[hi]:
                break
        if hi <= lo:
            # don't bother
            lut = np.arange(256)
        else:
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            ix = np.arange(256).astype(np.float64) * scale + offset
            ix = np.clip(ix, 0, 255).astype(np.uint8)
            lut = ix
        lut = np.array(lut, dtype=np.uint8)

        # Vectorized implementation of above block.
        # This is overall slower.
        # h_nz = np.nonzero(h)[0]
        # if len(h_nz) <= 1:
        #     lut = np.arange(256).astype(np.uint8)
        # else:
        #     lo = h_nz[0]
        #     hi = h_nz[-1]
        #
        #     scale = 255.0 / (hi - lo)
        #     offset = -lo * scale
        #     ix = np.arange(256).astype(np.float64) * scale + offset
        #     ix = np.clip(ix, 0, 255).astype(np.uint8)
        #     lut = ix

        # TODO change to a single call instead of one per channel
        image_c_aug = ia.apply_lut(image_c, lut)
        result[:, :, c_idx:c_idx+1] = image_c_aug
    if image.ndim == 2:
        return result[..., 0]
    return result


# Added in 0.4.0.
def _apply_enhance_func(image, cls, factor):
    assert image.dtype.name == "uint8", (
        "Can apply PIL image enhancement only to uint8 images, "
        "got dtype %s." % (image.dtype.name,))

    if 0 in image.shape:
        return np.copy(image)

    image, is_hw1 = _ensure_valid_shape(
        image, "imgaug.augmenters.pillike.enhance_*()")

    # don't return np.asarray(...) as its results are read-only
    result = np.array(
        cls(
            PIL.Image.fromarray(image)
        ).enhance(factor)
    )
    if is_hw1:
        result = result[:, :, np.newaxis]
    return result


def enhance_color(image, factor):
    """Change the strength of colors in an image.

    This function has identical outputs to
    ``PIL.ImageEnhance.Color``.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image to modify.

    factor : number
        Colorfulness of the output image. Values close to ``0.0`` lead
        to grayscale images, values above ``1.0`` increase the strength of
        colors. Sane values are roughly in ``[0.0, 3.0]``.

    Returns
    -------
    ndarray
        Color-modified image.

    """
    return _apply_enhance_func(image, PIL.ImageEnhance.Color, factor)


def enhance_contrast(image, factor):
    """Change the contrast of an image.

    This function has identical outputs to
    ``PIL.ImageEnhance.Contrast``.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image to modify.

    factor : number
        Strength of contrast in the image. Values below ``1.0`` decrease the
        contrast, leading to a gray image around ``0.0``. Values
        above ``1.0`` increase the contrast. Sane values are roughly in
        ``[0.5, 1.5]``.

    Returns
    -------
    ndarray
        Contrast-modified image.

    """
    return _apply_enhance_func(image, PIL.ImageEnhance.Contrast, factor)


def enhance_brightness(image, factor):
    """Change the brightness of images.

    This function has identical outputs to
    ``PIL.ImageEnhance.Brightness``.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image to modify.

    factor : number
        Brightness of the image. Values below ``1.0`` decrease the brightness,
        leading to a black image around ``0.0``. Values above ``1.0`` increase
        the brightness. Sane values are roughly in ``[0.5, 1.5]``.

    Returns
    -------
    ndarray
        Brightness-modified image.

    """
    return _apply_enhance_func(image, PIL.ImageEnhance.Brightness, factor)


def enhance_sharpness(image, factor):
    """Change the sharpness of an image.

    This function has identical outputs to
    ``PIL.ImageEnhance.Sharpness``.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image to modify.

    factor : number
        Sharpness of the image. Values below ``1.0`` decrease the sharpness,
        values above ``1.0`` increase it. Sane values are roughly in
        ``[0.0, 2.0]``.

    Returns
    -------
    ndarray
        Sharpness-modified image.

    """
    return _apply_enhance_func(image, PIL.ImageEnhance.Sharpness, factor)


# Added in 0.4.0.
def _filter_by_kernel(image, kernel):
    assert image.dtype.name == "uint8", (
        "Can apply PIL filters only to uint8 images, "
        "got dtype %s." % (image.dtype.name,))

    if 0 in image.shape:
        return np.copy(image)

    image, is_hw1 = _ensure_valid_shape(
        image, "imgaug.augmenters.pillike.filter_*()")

    image_pil = PIL.Image.fromarray(image)

    image_filtered = image_pil.filter(kernel)

    # don't return np.asarray(...) as its results are read-only
    result = np.array(image_filtered)
    if is_hw1:
        result = result[:, :, np.newaxis]
    return result


def filter_blur(image):
    """Apply a blur filter kernel to the image.

    This is the same as using PIL's ``PIL.ImageFilter.BLUR`` kernel.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image to modify.

    Returns
    -------
    ndarray
        Blurred image.

    """
    return _filter_by_kernel(image, PIL.ImageFilter.BLUR)


def filter_smooth(image):
    """Apply a smoothness filter kernel to the image.

    This is the same as using PIL's ``PIL.ImageFilter.SMOOTH`` kernel.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image to modify.

    Returns
    -------
    ndarray
        Smoothened image.

    """
    return _filter_by_kernel(image, PIL.ImageFilter.SMOOTH)


def filter_smooth_more(image):
    """Apply a strong smoothness filter kernel to the image.

    This is the same as using PIL's ``PIL.ImageFilter.SMOOTH_MORE`` kernel.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image to modify.

    Returns
    -------
    ndarray
        Smoothened image.

    """
    return _filter_by_kernel(image, PIL.ImageFilter.SMOOTH_MORE)


def filter_edge_enhance(image):
    """Apply an edge enhancement filter kernel to the image.

    This is the same as using PIL's ``PIL.ImageFilter.EDGE_ENHANCE`` kernel.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image to modify.

    Returns
    -------
    ndarray
        Image with enhanced edges.

    """
    return _filter_by_kernel(image, PIL.ImageFilter.EDGE_ENHANCE)


def filter_edge_enhance_more(image):
    """Apply a stronger edge enhancement filter kernel to the image.

    This is the same as using PIL's ``PIL.ImageFilter.EDGE_ENHANCE_MORE``
    kernel.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image to modify.

    Returns
    -------
    ndarray
        Smoothened image.

    """
    return _filter_by_kernel(image, PIL.ImageFilter.EDGE_ENHANCE_MORE)


def filter_find_edges(image):
    """Apply an edge detection filter kernel to the image.

    This is the same as using PIL's ``PIL.ImageFilter.FIND_EDGES`` kernel.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image to modify.

    Returns
    -------
    ndarray
        Image with detected edges.

    """
    return _filter_by_kernel(image, PIL.ImageFilter.FIND_EDGES)


def filter_contour(image):
    """Apply a contour filter kernel to the image.

    This is the same as using PIL's ``PIL.ImageFilter.CONTOUR`` kernel.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image to modify.

    Returns
    -------
    ndarray
        Image with pronounced contours.

    """
    return _filter_by_kernel(image, PIL.ImageFilter.CONTOUR)


def filter_emboss(image):
    """Apply an emboss filter kernel to the image.

    This is the same as using PIL's ``PIL.ImageFilter.EMBOSS`` kernel.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image to modify.

    Returns
    -------
    ndarray
        Embossed image.

    """
    return _filter_by_kernel(image, PIL.ImageFilter.EMBOSS)


def filter_sharpen(image):
    """Apply a sharpening filter kernel to the image.

    This is the same as using PIL's ``PIL.ImageFilter.SHARPEN`` kernel.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image to modify.

    Returns
    -------
    ndarray
        Sharpened image.

    """
    return _filter_by_kernel(image, PIL.ImageFilter.SHARPEN)


def filter_detail(image):
    """Apply a detail enhancement filter kernel to the image.

    This is the same as using PIL's ``PIL.ImageFilter.DETAIL`` kernel.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image to modify.

    Returns
    -------
    ndarray
        Image with enhanced details.

    """
    return _filter_by_kernel(image, PIL.ImageFilter.DETAIL)


# TODO unify this with the matrix generation for Affine,
#      there is probably no need to keep these separate
# Added in 0.4.0.
def _create_affine_matrix(scale_x=1.0, scale_y=1.0,
                          translate_x_px=0, translate_y_px=0,
                          rotate_deg=0,
                          shear_x_deg=0, shear_y_deg=0,
                          center_px=(0, 0)):
    scale_x = max(scale_x, 0.0001)
    scale_y = max(scale_y, 0.0001)

    rotate_rad, shear_x_rad, shear_y_rad = np.deg2rad([rotate_deg,
                                                       shear_x_deg,
                                                       shear_y_deg])
    rotate_rad = (-1) * rotate_rad

    matrix_centerize = np.array([
        [1, 0, (-1) * center_px[0]],
        [0, 1, (-1) * center_px[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    matrix_scale = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    matrix_translate = np.array([
        [1, 0, translate_x_px],
        [0, 1, translate_y_px],
        [0, 0, 1]
    ], dtype=np.float32)

    matrix_shear = np.array([
        [1, np.tanh(shear_x_rad), 0],
        [np.tanh(shear_y_rad), 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    matrix_rotate = np.array([
        [np.cos(rotate_rad), np.sin(rotate_rad), 0],
        [-np.sin(rotate_rad), np.cos(rotate_rad), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    matrix_decenterize = np.array([
        [1, 0, center_px[0]],
        [0, 1, center_px[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    for other_matrix in [matrix_centerize,
                         matrix_rotate, matrix_shear,
                         matrix_scale, matrix_translate,
                         matrix_decenterize]:
        matrix = np.matmul(other_matrix, matrix)

    matrix = np.linalg.inv(matrix)

    return matrix


def warp_affine(image,
                scale_x=1.0, scale_y=1.0,
                translate_x_px=0, translate_y_px=0,
                rotate_deg=0,
                shear_x_deg=0, shear_y_deg=0,
                fillcolor=None,
                center=(0.5, 0.5)):
    """Apply an affine transformation to an image.

    This function has identical outputs to
    ``PIL.Image.transform`` with ``method=PIL.Image.AFFINE``.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image to modify. Expected to be ``uint8`` with shape ``(H,W)``
        or ``(H,W,C)`` with ``C`` being ``3`` or ``4``.

    scale_x : number, optional
        Affine scale factor along the x-axis, where ``1.0`` denotes an
        identity transform and ``2.0`` is a strong zoom-in effect.

    scale_y : number, optional
        Affine scale factor along the y-axis, where ``1.0`` denotes an
        identity transform and ``2.0`` is a strong zoom-in effect.

    translate_x_px : number, optional
        Affine translation along the x-axis in pixels.
        Positive values translate the image towards the right.

    translate_y_px : number, optional
        Affine translation along the y-axis in pixels.
        Positive values translate the image towards the bottom.

    rotate_deg : number, optional
        Affine rotation in degrees *around the top left* of the image.

    shear_x_deg : number, optional
        Affine shearing in degrees along the x-axis with center point
        being the top-left of the image.

    shear_y_deg : number, optional
        Affine shearing in degrees along the y-axis with center point
        being the top-left of the image.

    fillcolor : None or int or tuple of int, optional
        Color tuple or intensity value to use when filling up newly
        created pixels. ``None`` fills with zeros. ``int`` will only fill
        the ``0`` th channel with that intensity value and all other channels
        with ``0`` (this is the default behaviour of PIL, use a tuple to
        fill all channels).

    center : tuple of number, optional
        Center xy-coordinate of the affine transformation, given as *relative*
        values, i.e. ``(0.0, 0.0)`` sets the transformation center to the
        top-left image corner, ``(1.0, 0.0)`` sets it to the the top-right
        image corner and ``(0.5, 0.5)`` sets it to the image center.
        The transformation center is relevant e.g. for rotations ("rotate
        around this center point"). PIL uses the image top-left corner
        as the transformation center if no centerization is included in the
        affine transformation matrix.

    Returns
    -------
    ndarray
        Image after affine transformation.

    """
    assert image.dtype.name == "uint8", (
        "Can apply PIL affine transformation only to uint8 images, "
        "got dtype %s." % (image.dtype.name,))

    if 0 in image.shape:
        return np.copy(image)

    fillcolor = fillcolor if fillcolor is not None else 0
    if ia.is_iterable(fillcolor):
        # make sure that iterable fillcolor contains only ints
        # otherwise we get a deprecation warning in py3.8
        fillcolor = tuple(map(int, fillcolor))

    image, is_hw1 = _ensure_valid_shape(
        image, "imgaug.augmenters.pillike.warp_affine()")

    image_pil = PIL.Image.fromarray(image)

    height, width = image.shape[0:2]
    center_px = (width * center[0], height * center[1])
    matrix = _create_affine_matrix(scale_x=scale_x,
                                   scale_y=scale_y,
                                   translate_x_px=translate_x_px,
                                   translate_y_px=translate_y_px,
                                   rotate_deg=rotate_deg,
                                   shear_x_deg=shear_x_deg,
                                   shear_y_deg=shear_y_deg,
                                   center_px=center_px)
    matrix = matrix[:2, :].flat

    # don't return np.asarray(...) as its results are read-only
    result = np.array(
        image_pil.transform(image_pil.size, PIL.Image.AFFINE, matrix,
                            fillcolor=fillcolor)
    )

    if is_hw1:
        result = result[:, :, np.newaxis]
    return result


# we don't use pil_solarize() here. but instead just subclass Invert,
# which is easier and comes down to the same
class Solarize(arithmetic.Invert):
    """Augmenter with identical outputs to PIL's ``solarize()`` function.

    This augmenter inverts all pixel values above a threshold.

    The outputs are identical to PIL's ``solarize()``.

    Added in 0.4.0.

    **Supported dtypes**:

    See ``~imgaug.augmenters.arithmetic.invert_(min_value=None and max_value=None)``.

    Parameters
    ----------
    p : float or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.arithmetic.Invert`.

    threshold : None or number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.arithmetic.Invert`.

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

    def __init__(self, p=1.0, threshold=128,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Solarize, self).__init__(
            p=p, per_channel=False,
            min_value=None, max_value=None,
            threshold=threshold, invert_above_threshold=True,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Posterize(colorlib.Posterize):
    """Augmenter with identical outputs to PIL's ``posterize()`` function.

    This augmenter quantizes each array component to ``N`` bits.

    This class is currently an alias for
    :class:`~imgaug.augmenters.color.Posterize`, which again is an alias
    for :class:`~imgaug.augmenters.color.UniformColorQuantizationToNBits`,
    i.e. all three classes are right now guarantueed to have the same
    outputs as PIL's function.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.color.Posterize`.

    """


class Equalize(meta.Augmenter):
    """Equalize the image histogram.

    This augmenter has identical outputs to ``PIL.ImageOps.equalize``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.pillike.equalize_`.

    Parameters
    ----------
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
    >>> aug = iaa.pillike.Equalize()

    Equalize the histograms of all input images.

    """

    # Added in 0.4.0.
    def __init__(self,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Equalize, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        # pylint: disable=no-self-use
        if batch.images is not None:
            for image in batch.images:
                image[...] = equalize_(image)
        return batch

    # Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return []


class Autocontrast(contrastlib._ContrastFuncWrapper):
    """Adjust contrast by cutting off ``p%`` of lowest/highest histogram values.

    This augmenter has identical outputs to ``PIL.ImageOps.autocontrast``.

    See :func:`~imgaug.augmenters.pillike.autocontrast` for more details.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.pillike.autocontrast`.

    Parameters
    ----------
    cutoff : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Percentage of values to cut off from the low and high end of each
        image's histogram, before stretching it to ``[0, 255]``.

            * If ``int``: The value will be used for all images.
            * If ``tuple`` ``(a, b)``: A value will be uniformly sampled from
              the discrete interval ``[a..b]`` per image.
            * If ``list``: A random value will be sampled from the list
              per image.
            * If ``StochasticParameter``: A value will be sampled from that
              parameter per image.

    per_channel :  bool or float, optional
        Whether to use the same value for all channels (``False``) or to
        sample a new value for each channel (``True``). If this value is a
        float ``p``, then for ``p`` percent of all images `per_channel` will
        be treated as ``True``, otherwise as ``False``.

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
    >>> aug = iaa.pillike.Autocontrast()

    Modify the contrast of images by cutting off the ``0`` to ``20%`` lowest
    and highest values from the histogram, then stretching it to full length.

    >>> aug = iaa.pillike.Autocontrast((10, 20), per_channel=True)

    Modify the contrast of images by cutting off the ``10`` to ``20%`` lowest
    and highest values from the histogram, then stretching it to full length.
    The cutoff value is sampled per *channel* instead of per *image*.

    """
    # pylint: disable=protected-access

    # Added in 0.4.0.
    def __init__(self, cutoff=(0, 20), per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        params1d = [
            iap.handle_discrete_param(
                cutoff, "cutoff", value_range=(0, 49), tuple_to_uniform=True,
                list_to_choice=True)
        ]
        func = autocontrast

        super(Autocontrast, self).__init__(
            func, params1d, per_channel,
            dtypes_allowed=["uint8"],
            dtypes_disallowed=["uint16", "uint32", "uint64",
                               "int8", "int16", "int32", "int64",
                               "float16", "float32", "float64",
                               "float16", "float32", "float64", "float96",
                               "float128", "float256", "bool"],
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


# Added in 0.4.0.
class _EnhanceBase(meta.Augmenter):
    # Added in 0.4.0.
    def __init__(self, func, factor, factor_value_range,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(_EnhanceBase, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        self.func = func
        self.factor = iap.handle_continuous_param(
            factor, "factor", value_range=factor_value_range,
            tuple_to_uniform=True, list_to_choice=True)

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        factors = self._draw_samples(len(batch.images), random_state)
        for image, factor in zip(batch.images, factors):
            image[...] = self.func(image, factor)
        return batch

    # Added in 0.4.0.
    def _draw_samples(self, nb_rows, random_state):
        return self.factor.draw_samples((nb_rows,), random_state=random_state)

    # Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.factor]


class EnhanceColor(_EnhanceBase):
    """Convert images to grayscale.

    This augmenter has identical outputs to ``PIL.ImageEnhance.Color``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.pillike.enhance_color`.

    Parameters
    ----------
    factor : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Colorfulness of the output image. Values close to ``0.0`` lead
        to grayscale images, values above ``1.0`` increase the strength of
        colors. Sane values are roughly in ``[0.0, 3.0]``.

            * If ``number``: The value will be used for all images.
            * If ``tuple`` ``(a, b)``: A value will be uniformly sampled per
              image from the interval ``[a, b)``.
            * If ``list``: A random value will be picked from the list per
              image.
            * If ``StochasticParameter``: Per batch of size ``N``, the
              parameter will be queried once to return ``(N,)`` samples.

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
    >>> aug = iaa.pillike.EnhanceColor()

    Create an augmenter to remove a random fraction of color from
    input images.

    """

    # Added in 0.4.0.
    def __init__(self, factor=(0.0, 3.0),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(EnhanceColor, self).__init__(
            func=enhance_color,
            factor=factor,
            factor_value_range=(0.0, None),
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class EnhanceContrast(_EnhanceBase):
    """Change the contrast of images.

    This augmenter has identical outputs to ``PIL.ImageEnhance.Contrast``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.pillike.enhance_contrast`.

    Parameters
    ----------
    factor : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Strength of contrast in the image. Values below ``1.0`` decrease the
        contrast, leading to a gray image around ``0.0``. Values
        above ``1.0`` increase the contrast. Sane values are roughly in
        ``[0.5, 1.5]``.

            * If ``number``: The value will be used for all images.
            * If ``tuple`` ``(a, b)``: A value will be uniformly sampled per
              image from the interval ``[a, b)``.
            * If ``list``: A random value will be picked from the list per
              image.
            * If ``StochasticParameter``: Per batch of size ``N``, the
              parameter will be queried once to return ``(N,)`` samples.

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
    >>> aug = iaa.pillike.EnhanceContrast()

    Create an augmenter that worsens the contrast of an image by a random
    factor.

    """

    # Added in 0.4.0.
    def __init__(self, factor=(0.5, 1.5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(EnhanceContrast, self).__init__(
            func=enhance_contrast,
            factor=factor,
            factor_value_range=(0.0, None),
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class EnhanceBrightness(_EnhanceBase):
    """Change the brightness of images.

    This augmenter has identical outputs to
    ``PIL.ImageEnhance.Brightness``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.pillike.enhance_brightness`.

    Parameters
    ----------
    factor : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Brightness of the image. Values below ``1.0`` decrease the brightness,
        leading to a black image around ``0.0``. Values above ``1.0`` increase
        the brightness. Sane values are roughly in ``[0.5, 1.5]``.

            * If ``number``: The value will be used for all images.
            * If ``tuple`` ``(a, b)``: A value will be uniformly sampled per
              image from the interval ``[a, b)``.
            * If ``list``: A random value will be picked from the list per
              image.
            * If ``StochasticParameter``: Per batch of size ``N``, the
              parameter will be queried once to return ``(N,)`` samples.

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
    >>> aug = iaa.pillike.EnhanceBrightness()

    Create an augmenter that worsens the brightness of an image by a random
    factor.

    """

    # Added in 0.4.0.
    def __init__(self, factor=(0.5, 1.5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(EnhanceBrightness, self).__init__(
            func=enhance_brightness,
            factor=factor,
            factor_value_range=(0.0, None),
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class EnhanceSharpness(_EnhanceBase):
    """Change the sharpness of images.

    This augmenter has identical outputs to
    ``PIL.ImageEnhance.Sharpness``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.pillike.enhance_sharpness`.

    Parameters
    ----------
    factor : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Sharpness of the image. Values below ``1.0`` decrease the sharpness,
        values above ``1.0`` increase it. Sane values are roughly in
        ``[0.0, 2.0]``.

            * If ``number``: The value will be used for all images.
            * If ``tuple`` ``(a, b)``: A value will be uniformly sampled per
              image from the interval ``[a, b)``.
            * If ``list``: A random value will be picked from the list per
              image.
            * If ``StochasticParameter``: Per batch of size ``N``, the
              parameter will be queried once to return ``(N,)`` samples.

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
    >>> aug = iaa.pillike.EnhanceSharpness()

    Create an augmenter that randomly decreases or increases the sharpness
    of an image.

    """

    # Added in 0.4.0.
    def __init__(self, factor=(0.0, 2.0),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(EnhanceSharpness, self).__init__(
            func=enhance_sharpness,
            factor=factor,
            factor_value_range=(0.0, None),
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


# Added in 0.4.0.
class _FilterBase(meta.Augmenter):
    # Added in 0.4.0.
    def __init__(self, func,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(_FilterBase, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        self.func = func

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is not None:
            for image in batch.images:
                image[...] = self.func(image)
        return batch

    # Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return []


class FilterBlur(_FilterBase):
    """Apply a blur filter kernel to images.

    This augmenter has identical outputs to
    calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.BLUR``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.pillike.filter_blur`.

    Parameters
    ----------
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
    >>> aug = iaa.pillike.FilterBlur()

    Create an augmenter that applies a blur filter kernel to images.

    """

    # Added in 0.4.0.
    def __init__(self,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(FilterBlur, self).__init__(
            func=filter_blur,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class FilterSmooth(_FilterBase):
    """Apply a smoothening filter kernel to images.

    This augmenter has identical outputs to
    calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.SMOOTH``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.pillike.filter_smooth`.

    Parameters
    ----------
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
    >>> aug = iaa.pillike.FilterSmooth()

    Create an augmenter that applies a smoothening filter kernel to images.

    """

    # Added in 0.4.0.
    def __init__(self,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(FilterSmooth, self).__init__(
            func=filter_smooth,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class FilterSmoothMore(_FilterBase):
    """Apply a strong smoothening filter kernel to images.

    This augmenter has identical outputs to
    calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.BLUR``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.pillike.filter_smooth_more`.

    Parameters
    ----------
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
    >>> aug = iaa.pillike.FilterSmoothMore()

    Create an augmenter that applies a strong smoothening filter kernel to
    images.

    """

    # Added in 0.4.0.
    def __init__(self,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(FilterSmoothMore, self).__init__(
            func=filter_smooth_more,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class FilterEdgeEnhance(_FilterBase):
    """Apply an edge enhance filter kernel to images.

    This augmenter has identical outputs to
    calling ``PIL.Image.filter`` with kernel
    ``PIL.ImageFilter.EDGE_ENHANCE``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.pillike.filter_edge_enhance`.

    Parameters
    ----------
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
    >>> aug = iaa.pillike.FilterEdgeEnhance()

    Create an augmenter that applies a edge enhancement filter kernel to
    images.

    """

    # Added in 0.4.0.
    def __init__(self,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(FilterEdgeEnhance, self).__init__(
            func=filter_edge_enhance,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class FilterEdgeEnhanceMore(_FilterBase):
    """Apply a strong edge enhancement filter kernel to images.

    This augmenter has identical outputs to
    calling ``PIL.Image.filter`` with kernel
    ``PIL.ImageFilter.EDGE_ENHANCE_MORE``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.pillike.filter_edge_enhance_more`.

    Parameters
    ----------
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
    >>> aug = iaa.pillike.FilterEdgeEnhanceMore()

    Create an augmenter that applies a strong edge enhancement filter kernel
    to images.

    """

    # Added in 0.4.0.
    def __init__(self,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(FilterEdgeEnhanceMore, self).__init__(
            func=filter_edge_enhance_more,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class FilterFindEdges(_FilterBase):
    """Apply a edge detection kernel to images.

    This augmenter has identical outputs to
    calling ``PIL.Image.filter`` with kernel
    ``PIL.ImageFilter.FIND_EDGES``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.pillike.filter_find_edges`.

    Parameters
    ----------
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
    >>> aug = iaa.pillike.FilterFindEdges()

    Create an augmenter that applies an edge detection filter kernel to images.

    """

    # Added in 0.4.0.
    def __init__(self,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(FilterFindEdges, self).__init__(
            func=filter_find_edges,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class FilterContour(_FilterBase):
    """Apply a contour detection filter kernel to images.

    This augmenter has identical outputs to
    calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.CONTOUR``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.pillike.filter_contour`.

    Parameters
    ----------
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
    >>> aug = iaa.pillike.FilterContour()

    Create an augmenter that applies a contour detection filter kernel to
    images.

    """

    # Added in 0.4.0.
    def __init__(self,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(FilterContour, self).__init__(
            func=filter_contour,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class FilterEmboss(_FilterBase):
    """Apply an emboss filter kernel to images.

    This augmenter has identical outputs to
    calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.EMBOSS``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.pillike.filter_emboss`.

    Parameters
    ----------
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
    >>> aug = iaa.pillike.FilterEmboss()

    Create an augmenter that applies an emboss filter kernel to images.

    """

    # Added in 0.4.0.
    def __init__(self,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(FilterEmboss, self).__init__(
            func=filter_emboss,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class FilterSharpen(_FilterBase):
    """Apply a sharpening filter kernel to images.

    This augmenter has identical outputs to
    calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.SHARPEN``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.pillike.filter_sharpen`.

    Parameters
    ----------
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
    >>> aug = iaa.pillike.FilterSharpen()

    Create an augmenter that applies a sharpening filter kernel to images.

    """

    # Added in 0.4.0.
    def __init__(self,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(FilterSharpen, self).__init__(
            func=filter_sharpen,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class FilterDetail(_FilterBase):
    """Apply a detail enhancement filter kernel to images.

    This augmenter has identical outputs to
    calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.DETAIL``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.pillike.filter_detail`.

    Parameters
    ----------
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
    >>> aug = iaa.pillike.FilterDetail()

    Create an augmenter that applies a detail enhancement filter kernel to
    images.

    """

    # Added in 0.4.0.
    def __init__(self,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(FilterDetail, self).__init__(
            func=filter_detail,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Affine(geometric.Affine):
    """Apply PIL-like affine transformations to images.

    This augmenter has identical outputs to
    ``PIL.Image.transform`` with parameter ``method=PIL.Image.AFFINE``.

    .. warning::

        This augmenter can currently only transform image-data.
        Batches containing heatmaps, segmentation maps and
        coordinate-based augmentables will be rejected with an error.
        Use :class:`~imgaug.augmenters.geometric.Affine` if you have to
        transform such inputs.

    .. note::

        This augmenter uses the image center as the transformation center.
        This has to be explicitly enforced in PIL using corresponding
        translation matrices. Without such translation, PIL uses the image
        top left corner as the transformation center. To mirror that
        behaviour, use ``center=(0.0, 0.0)``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.pillike.warp_affine`.

    Parameters
    ----------
    scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter or dict {"x": number/tuple/list/StochasticParameter, "y": number/tuple/list/StochasticParameter}, optional
        See :class:`~imgaug.augmenters.geometric.Affine`.

    translate_percent : None or number or tuple of number or list of number or imgaug.parameters.StochasticParameter or dict {"x": number/tuple/list/StochasticParameter, "y": number/tuple/list/StochasticParameter}, optional
        See :class:`~imgaug.augmenters.geometric.Affine`.

    translate_px : None or int or tuple of int or list of int or imgaug.parameters.StochasticParameter or dict {"x": int/tuple/list/StochasticParameter, "y": int/tuple/list/StochasticParameter}, optional
        See :class:`~imgaug.augmenters.geometric.Affine`.

    rotate : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.geometric.Affine`.

    shear : number or tuple of number or list of number or imgaug.parameters.StochasticParameter or dict {"x": int/tuple/list/StochasticParameter, "y": int/tuple/list/StochasticParameter}, optional
        See :class:`~imgaug.augmenters.geometric.Affine`.

    fillcolor : number or tuple of number or list of number or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See parameter ``cval`` in :class:`~imgaug.augmenters.geometric.Affine`.

    center : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        The center point of the affine transformation, given as relative
        xy-coordinates.
        Set this to ``(0.0, 0.0)`` or ``left-top`` to use the top left image
        corner as the transformation center.
        Set this to ``(0.5, 0.5)`` or ``center-center`` to use the image
        center as the transformation center.
        See also paramerer ``position`` in
        :class:`~imgaug.augmenters.size.PadToFixedSize` for details
        about valid datatypes of this parameter.

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
    >>> aug = iaa.pillike.Affine(scale={"x": (0.8, 1.2), "y": (0.5, 1.5)})

    Create an augmenter that applies affine scaling (zoom in/out) to images.
    Along the x-axis they are scaled to 80-120% of their size, along
    the y-axis to 50-150% (both values randomly and uniformly chosen per
    image).

    >>> aug = iaa.pillike.Affine(translate_px={"x": 0, "y": [-10, 10]},
    >>>                          fillcolor=128)

    Create an augmenter that translates images along the y-axis by either
    ``-10px`` or ``10px``. Newly created pixels are always filled with
    the value ``128`` (along all channels).

    >>> aug = iaa.pillike.Affine(rotate=(-20, 20), fillcolor=(0, 256))

    Rotate an image by ``-20`` to ``20`` degress and fill up all newly
    created pixels with a random RGB color.

    See the similar augmenter :class:`~imgaug.augmenters.geometric.Affine`
    for more examples.

    """

    # Added in 0.4.0.
    def __init__(self, scale=1.0, translate_percent=None, translate_px=None,
                 rotate=0.0, shear=0.0, fillcolor=0, center=(0.5, 0.5),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Affine, self).__init__(
            scale=scale,
            translate_percent=translate_percent,
            translate_px=translate_px,
            rotate=rotate,
            shear=shear,
            order=1,
            cval=fillcolor,
            mode="constant",
            fit_output=False,
            backend="auto",
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        # TODO move that func to iap
        self.center = sizelib._handle_position_parameter(center)

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        cols = batch.get_column_names()
        assert len(cols) == 0 or (len(cols) == 1 and "images" in cols), (
            "pillike.Affine can currently only process image data. Got a "
            "batch containing: %s. Use imgaug.augmenters.geometric.Affine for "
            "batches containing non-image data." % (", ".join(cols),))

        return super(Affine, self)._augment_batch_(
            batch, random_state, parents, hooks)

    # Added in 0.4.0.
    def _augment_images_by_samples(self, images, samples,
                                   image_shapes=None,
                                   return_matrices=False):
        assert return_matrices is False, (
            "Got unexpectedly return_matrices=True. pillike.Affine does not "
            "yet produce that output.")

        for i, image in enumerate(images):
            image_shape = (image.shape if image_shapes is None
                           else image_shapes[i])

            params = samples.get_affine_parameters(
                i, arr_shape=image_shape, image_shape=image_shape)

            image[...] = warp_affine(
                image,
                scale_x=params["scale_x"],
                scale_y=params["scale_y"],
                translate_x_px=params["translate_x_px"],
                translate_y_px=params["translate_y_px"],
                rotate_deg=params["rotate_deg"],
                shear_x_deg=params["shear_x_deg"],
                shear_y_deg=params["shear_y_deg"],
                fillcolor=tuple(samples.cval[i]),
                center=(samples.center_x[i], samples.center_y[i])
            )

        return images

    # Added in 0.4.0.
    def _draw_samples(self, nb_samples, random_state):
        # standard affine samples
        samples = super(Affine, self)._draw_samples(nb_samples,
                                                    random_state)

        # add samples for 'center' parameter, which is not yet a part of
        # Affine
        if isinstance(self.center, tuple):
            xx = self.center[0].draw_samples(nb_samples,
                                             random_state=random_state)
            yy = self.center[1].draw_samples(nb_samples,
                                             random_state=random_state)
        else:
            xy = self.center.draw_samples((nb_samples, 2),
                                          random_state=random_state)
            xx = xy[:, 0]
            yy = xy[:, 1]

        samples.center_x = xx
        samples.center_y = yy
        return samples

    # Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [
            self.scale, self.translate, self.rotate, self.shear, self.cval,
            self.center]
