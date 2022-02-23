"""
Augmenters that affect image colors or image colorspaces.

List of augmenters:

    * :class:`InColorspace` (deprecated)
    * :class:`WithColorspace`
    * :class:`WithBrightnessChannels`
    * :class:`MultiplyAndAddToBrightness`
    * :class:`MultiplyBrightness`
    * :class:`AddToBrightness`
    * :class:`WithHueAndSaturation`
    * :class:`MultiplyHueAndSaturation`
    * :class:`MultiplyHue`
    * :class:`MultiplySaturation`
    * :class:`RemoveSaturation`
    * :class:`AddToHueAndSaturation`
    * :class:`AddToHue`
    * :class:`AddToSaturation`
    * :class:`ChangeColorspace`
    * :class:`Grayscale`
    * :class:`ChangeColorTemperature`
    * :class:`KMeansColorQuantization`
    * :class:`UniformColorQuantization`
    * :class:`Posterize`

"""
from __future__ import print_function, division, absolute_import

from abc import ABCMeta, abstractmethod

import numpy as np
import cv2
import six
import six.moves as sm

import imgaug as ia
from imgaug.imgaug import _normalize_cv2_input_arr_
from . import meta
from . import blend
from . import arithmetic
from .. import parameters as iap
from .. import dtypes as iadt
from .. import random as iarandom


# pylint: disable=invalid-name
CSPACE_RGB = "RGB"
CSPACE_BGR = "BGR"
CSPACE_GRAY = "GRAY"
CSPACE_YCrCb = "YCrCb"
CSPACE_HSV = "HSV"
CSPACE_HLS = "HLS"
CSPACE_Lab = "Lab"  # aka CIELAB
# TODO add Luv to various color/contrast augmenters as random default choice?
CSPACE_Luv = "Luv"  # aka CIE 1976, aka CIELUV
CSPACE_YUV = "YUV"  # aka CIE 1960
CSPACE_CIE = "CIE"  # aka CIE 1931, aka XYZ in OpenCV
CSPACE_ALL = {CSPACE_RGB, CSPACE_BGR, CSPACE_GRAY, CSPACE_YCrCb,
              CSPACE_HSV, CSPACE_HLS, CSPACE_Lab, CSPACE_Luv,
              CSPACE_YUV, CSPACE_CIE}
# pylint: enable=invalid-name


def _get_opencv_attr(attr_names):
    for attr_name in attr_names:
        if hasattr(cv2, attr_name):
            return getattr(cv2, attr_name)
    ia.warn("Could not find any of the following attributes in cv2: %s. "
            "This can cause issues with colorspace transformations." % (
                attr_names))
    return None


_CSPACE_OPENCV_CONV_VARS = {
    # RGB
    (CSPACE_RGB, CSPACE_BGR): cv2.COLOR_RGB2BGR,
    (CSPACE_RGB, CSPACE_GRAY): cv2.COLOR_RGB2GRAY,
    (CSPACE_RGB, CSPACE_YCrCb): _get_opencv_attr(["COLOR_RGB2YCR_CB"]),
    (CSPACE_RGB, CSPACE_HSV): cv2.COLOR_RGB2HSV,
    (CSPACE_RGB, CSPACE_HLS): cv2.COLOR_RGB2HLS,
    (CSPACE_RGB, CSPACE_Lab): _get_opencv_attr(["COLOR_RGB2LAB",
                                                "COLOR_RGB2Lab"]),
    (CSPACE_RGB, CSPACE_Luv): cv2.COLOR_RGB2LUV,
    (CSPACE_RGB, CSPACE_YUV): cv2.COLOR_RGB2YUV,
    (CSPACE_RGB, CSPACE_CIE): cv2.COLOR_RGB2XYZ,
    # BGR
    (CSPACE_BGR, CSPACE_RGB): cv2.COLOR_BGR2RGB,
    (CSPACE_BGR, CSPACE_GRAY): cv2.COLOR_BGR2GRAY,
    (CSPACE_BGR, CSPACE_YCrCb): _get_opencv_attr(["COLOR_BGR2YCR_CB"]),
    (CSPACE_BGR, CSPACE_HSV): cv2.COLOR_BGR2HSV,
    (CSPACE_BGR, CSPACE_HLS): cv2.COLOR_BGR2HLS,
    (CSPACE_BGR, CSPACE_Lab): _get_opencv_attr(["COLOR_BGR2LAB",
                                                "COLOR_BGR2Lab"]),
    (CSPACE_BGR, CSPACE_Luv): cv2.COLOR_BGR2LUV,
    (CSPACE_BGR, CSPACE_YUV): cv2.COLOR_BGR2YUV,
    (CSPACE_BGR, CSPACE_CIE): cv2.COLOR_BGR2XYZ,
    # GRAY
    # YCrCb
    (CSPACE_YCrCb, CSPACE_RGB): _get_opencv_attr(["COLOR_YCrCb2RGB",
                                                  "COLOR_YCR_CB2RGB"]),
    (CSPACE_YCrCb, CSPACE_BGR): _get_opencv_attr(["COLOR_YCrCb2BGR",
                                                  "COLOR_YCR_CB2BGR"]),
    # HSV
    (CSPACE_HSV, CSPACE_RGB): cv2.COLOR_HSV2RGB,
    (CSPACE_HSV, CSPACE_BGR): cv2.COLOR_HSV2BGR,
    # HLS
    (CSPACE_HLS, CSPACE_RGB): cv2.COLOR_HLS2RGB,
    (CSPACE_HLS, CSPACE_BGR): cv2.COLOR_HLS2BGR,
    # Lab
    (CSPACE_Lab, CSPACE_RGB): _get_opencv_attr(["COLOR_Lab2RGB",
                                                "COLOR_LAB2RGB"]),
    (CSPACE_Lab, CSPACE_BGR): _get_opencv_attr(["COLOR_Lab2BGR",
                                                "COLOR_LAB2BGR"]),
    # Luv
    (CSPACE_Luv, CSPACE_RGB): _get_opencv_attr(["COLOR_Luv2RGB",
                                                "COLOR_LUV2RGB"]),
    (CSPACE_Luv, CSPACE_BGR): _get_opencv_attr(["COLOR_Luv2BGR",
                                                "COLOR_LUV2BGR"]),
    # YUV
    (CSPACE_YUV, CSPACE_RGB): cv2.COLOR_YUV2RGB,
    (CSPACE_YUV, CSPACE_BGR): cv2.COLOR_YUV2BGR,
    # CIE
    (CSPACE_CIE, CSPACE_RGB): cv2.COLOR_XYZ2RGB,
    (CSPACE_CIE, CSPACE_BGR): cv2.COLOR_XYZ2BGR,
}

# This defines which colorspace pairs will be converted in-place in
# change_colorspace_(). Currently, all colorspaces seem to work fine with
# in-place transformations, which is why they are all set to True.
_CHANGE_COLORSPACE_INPLACE = {
    # RGB
    (CSPACE_RGB, CSPACE_BGR): True,
    (CSPACE_RGB, CSPACE_GRAY): True,
    (CSPACE_RGB, CSPACE_YCrCb): True,
    (CSPACE_RGB, CSPACE_HSV): True,
    (CSPACE_RGB, CSPACE_HLS): True,
    (CSPACE_RGB, CSPACE_Lab): True,
    (CSPACE_RGB, CSPACE_Luv): True,
    (CSPACE_RGB, CSPACE_YUV): True,
    (CSPACE_RGB, CSPACE_CIE): True,
    # BGR
    (CSPACE_BGR, CSPACE_RGB): True,
    (CSPACE_BGR, CSPACE_GRAY): True,
    (CSPACE_BGR, CSPACE_YCrCb): True,
    (CSPACE_BGR, CSPACE_HSV): True,
    (CSPACE_BGR, CSPACE_HLS): True,
    (CSPACE_BGR, CSPACE_Lab): True,
    (CSPACE_BGR, CSPACE_Luv): True,
    (CSPACE_BGR, CSPACE_YUV): True,
    (CSPACE_BGR, CSPACE_CIE): True,
    # GRAY
    # YCrCb
    (CSPACE_YCrCb, CSPACE_RGB): True,
    (CSPACE_YCrCb, CSPACE_BGR): True,
    # HSV
    (CSPACE_HSV, CSPACE_RGB): True,
    (CSPACE_HSV, CSPACE_BGR): True,
    # HLS
    (CSPACE_HLS, CSPACE_RGB): True,
    (CSPACE_HLS, CSPACE_BGR): True,
    # Lab
    (CSPACE_Lab, CSPACE_RGB): True,
    (CSPACE_Lab, CSPACE_BGR): True,
    # Luv
    (CSPACE_Luv, CSPACE_RGB): True,
    (CSPACE_Luv, CSPACE_BGR): True,
    # YUV
    (CSPACE_YUV, CSPACE_RGB): True,
    (CSPACE_YUV, CSPACE_BGR): True,
    # CIE
    (CSPACE_CIE, CSPACE_RGB): True,
    (CSPACE_CIE, CSPACE_BGR): True,
}


def change_colorspace_(image, to_colorspace, from_colorspace=CSPACE_RGB):
    """Change the colorspace of an image inplace.

    .. note::

        All outputs of this function are `uint8`. For some colorspaces this
        may not be optimal.

    .. note::

        Output grayscale images will still have three channels.

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
        The image to convert from one colorspace into another.
        Usually expected to have shape ``(H,W,3)``.

    to_colorspace : str
        The target colorspace. See the ``CSPACE`` constants,
        e.g. ``imgaug.augmenters.color.CSPACE_RGB``.

    from_colorspace : str, optional
        The source colorspace. Analogous to `to_colorspace`. Defaults
        to ``RGB``.

    Returns
    -------
    ndarray
        Image with target colorspace. *Can* be the same array instance as was
        originally provided (i.e. changed inplace). Grayscale images will
        still have three channels.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> import numpy as np
    >>> # fake RGB image
    >>> image_rgb = np.arange(4*4*3).astype(np.uint8).reshape((4, 4, 3))
    >>> image_bgr = iaa.change_colorspace_(np.copy(image_rgb), iaa.CSPACE_BGR)

    """
    # some colorspaces here should use image/255.0 according to
    # the docs, but at least for conversion to grayscale that
    # results in errors, ie uint8 is expected

    # this was once used to accomodate for image .flags -- still necessary?
    def _get_dst(image_, from_to_cspace):
        if _CHANGE_COLORSPACE_INPLACE[from_to_cspace]:
            return image_
        return None

    # cv2 does not support height/width 0
    # we don't check here if the channel axis is zero-sized as for colorspace
    # transformations it should never be 0
    if 0 in image.shape[0:2]:
        return image

    iadt.gate_dtypes(
        image,
        allowed=["uint8"],
        disallowed=[
            "bool",
            "uint16", "uint32", "uint64", "uint128", "uint256",
            "int32", "int64", "int128", "int256",
            "float16", "float32", "float64", "float96", "float128",
            "float256"],
        augmenter=None)

    for arg_name in ["to_colorspace", "from_colorspace"]:
        assert locals()[arg_name] in CSPACE_ALL, (
            "Expected `%s` to be one of: %s. Got: %s." % (
                arg_name, CSPACE_ALL, locals()[arg_name]))

    assert from_colorspace != CSPACE_GRAY, (
        "Cannot convert from grayscale to another colorspace as colors "
        "cannot be recovered.")

    assert image.ndim == 3, (
        "Expected image shape to be three-dimensional, i.e. (H,W,C), "
        "got %d dimensions with shape %s." % (image.ndim, image.shape))
    assert image.shape[2] == 3, (
        "Expected number of channels to be three, "
        "got %d channels (shape %s)." % (image.shape[2], image.shape,))

    if from_colorspace == to_colorspace:
        return image

    from_to_direct = (from_colorspace, to_colorspace)
    from_to_indirect = [
        (from_colorspace, CSPACE_RGB),
        (CSPACE_RGB, to_colorspace)
    ]

    image = _normalize_cv2_input_arr_(image)
    image_aug = image
    if from_to_direct in _CSPACE_OPENCV_CONV_VARS:
        from2to_var = _CSPACE_OPENCV_CONV_VARS[from_to_direct]
        dst = _get_dst(image_aug, from_to_direct)
        image_aug = cv2.cvtColor(image_aug, from2to_var, dst=dst)
    else:
        from2rgb_var = _CSPACE_OPENCV_CONV_VARS[from_to_indirect[0]]
        rgb2to_var = _CSPACE_OPENCV_CONV_VARS[from_to_indirect[1]]

        dst1 = _get_dst(image_aug, from_to_indirect[0])
        dst2 = _get_dst(image_aug, from_to_indirect[1])

        image_aug = cv2.cvtColor(image_aug, from2rgb_var, dst=dst1)
        image_aug = cv2.cvtColor(image_aug, rgb2to_var, dst=dst2)

    assert image_aug.dtype.name == "uint8"

    # for grayscale: covnert from (H, W) to (H, W, 3)
    if len(image_aug.shape) == 2:
        image_aug = image_aug[:, :, np.newaxis]
        image_aug = np.tile(image_aug, (1, 1, 3))

    return image_aug


def change_colorspaces_(images, to_colorspaces, from_colorspaces=CSPACE_RGB):
    """Change the colorspaces of a batch of images inplace.

    .. note::

        All outputs of this function are `uint8`. For some colorspaces this
        may not be optimal.

    .. note::

        Output grayscale images will still have three channels.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.color.change_colorspace_`.

    Parameters
    ----------
    images : ndarray or list of ndarray
        The images to convert from one colorspace into another.
        Either a list of ``(H,W,3)`` arrays or a single ``(N,H,W,3)`` array.

    to_colorspaces : str or iterable of str
        The target colorspaces. Either a single string (all images will be
        converted to the same colorspace) or an iterable of strings (one per
        image). See the ``CSPACE`` constants, e.g.
        ``imgaug.augmenters.color.CSPACE_RGB``.

    from_colorspaces : str or list of str, optional
        The source colorspace. Analogous to `to_colorspace`. Defaults
        to ``RGB``.

    Returns
    -------
    ndarray or list of ndarray
        Images with target colorspaces. *Can* contain the same array instances
        as were originally provided (i.e. changed inplace). Grayscale images
        will still have three channels.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> import numpy as np
    >>> # fake RGB image
    >>> image_rgb = np.arange(4*4*3).astype(np.uint8).reshape((4, 4, 3))
    >>> images_rgb = [image_rgb, image_rgb, image_rgb]
    >>> images_rgb_copy = [np.copy(image_rgb) for image_rgb in images_rgb]
    >>> images_bgr = iaa.change_colorspaces_(images_rgb_copy, iaa.CSPACE_BGR)

    Create three example ``RGB`` images and convert them to ``BGR`` colorspace.

    >>> images_rgb_copy = [np.copy(image_rgb) for image_rgb in images_rgb]
    >>> images_various = iaa.change_colorspaces_(
    >>>     images_rgb_copy, [iaa.CSPACE_BGR, iaa.CSPACE_HSV, iaa.CSPACE_GRAY])

    Chnage the colorspace of the first image to ``BGR``, the one of the second
    image to ``HSV`` and the one of the third image to ``grayscale`` (note
    that in the latter case the image will still have shape ``(H,W,3)``,
    not ``(H,W,1)``).

    """
    def _validate(arg, arg_name):
        if ia.is_string(arg):
            arg = [arg] * len(images)
        else:
            assert ia.is_iterable(arg), (
                "Expected `%s` to be either an iterable of strings or a single "
                "string. Got type: %s." % (arg_name, type(arg).__name__)
            )
            assert len(arg) == len(images), (
                "If `%s` is provided as a list it must have the same length "
                "as `images`. Got length %d, expected %d." % (
                    arg_name, len(arg), len(images)))

        return arg

    to_colorspaces = _validate(to_colorspaces, "to_colorspaces")
    from_colorspaces = _validate(from_colorspaces, "from_colorspaces")

    gen = zip(images, to_colorspaces, from_colorspaces)
    for i, (image, to_colorspace, from_colorspace) in enumerate(gen):
        images[i] = change_colorspace_(image, to_colorspace, from_colorspace)
    return images


# Added in 0.4.0.
class _KelvinToRGBTableSingleton(object):
    _INSTANCE = None

    # Added in 0.4.0.
    @classmethod
    def get_instance(cls):
        if cls._INSTANCE is None:
            cls._INSTANCE = _KelvinToRGBTable()
        return cls._INSTANCE


# Added in 0.4.0.
class _KelvinToRGBTable(object):
    # Added in 0.4.0.
    def __init__(self):
        self.table = self.create_table()

    def transform_kelvins_to_rgb_multipliers(self, kelvins):
        """Transform kelvin values to corresponding multipliers for RGB images.

        A single returned multiplier denotes the channelwise multipliers
        in the range ``[0.0, 1.0]`` to apply to an image to change its kelvin
        value to the desired one.

        Added in 0.4.0.

        Parameters
        ----------
        kelvins : iterable of number
            Imagewise temperatures in kelvin.

        Returns
        -------
        ndarray
            ``float32 (N, 3) ndarrays``, one per kelvin.

        """
        kelvins = np.clip(kelvins, 1000, 40000)

        tbl_indices = kelvins / 100 - (1000//100)
        tbl_indices_floored = np.floor(tbl_indices)
        tbl_indices_ceiled = np.ceil(tbl_indices)
        interpolation_factors = tbl_indices - tbl_indices_floored

        tbl_indices_floored_int = tbl_indices_floored.astype(np.int32)
        tbl_indices_ceiled_int = tbl_indices_ceiled.astype(np.int32)

        multipliers_floored = self.table[tbl_indices_floored_int, :]
        multipliers_ceiled = self.table[tbl_indices_ceiled_int, :]
        multipliers = (
            multipliers_floored
            + interpolation_factors
            * (multipliers_ceiled - multipliers_floored)
        )

        return multipliers

    # Added in 0.4.0.
    @classmethod
    def create_table(cls):
        table = np.float32([
            [255, 56, 0],  # K=1000
            [255, 71, 0],  # K=1100
            [255, 83, 0],  # K=1200
            [255, 93, 0],  # K=1300
            [255, 101, 0],  # K=1400
            [255, 109, 0],  # K=1500
            [255, 115, 0],  # K=1600
            [255, 121, 0],  # K=1700
            [255, 126, 0],  # K=1800
            [255, 131, 0],  # K=1900
            [255, 137, 18],  # K=2000
            [255, 142, 33],  # K=2100
            [255, 147, 44],  # K=2200
            [255, 152, 54],  # K=2300
            [255, 157, 63],  # K=2400
            [255, 161, 72],  # K=2500
            [255, 165, 79],  # K=2600
            [255, 169, 87],  # K=2700
            [255, 173, 94],  # K=2800
            [255, 177, 101],  # K=2900
            [255, 180, 107],  # K=3000
            [255, 184, 114],  # K=3100
            [255, 187, 120],  # K=3200
            [255, 190, 126],  # K=3300
            [255, 193, 132],  # K=3400
            [255, 196, 137],  # K=3500
            [255, 199, 143],  # K=3600
            [255, 201, 148],  # K=3700
            [255, 204, 153],  # K=3800
            [255, 206, 159],  # K=3900
            [255, 209, 163],  # K=4000
            [255, 211, 168],  # K=4100
            [255, 213, 173],  # K=4200
            [255, 215, 177],  # K=4300
            [255, 217, 182],  # K=4400
            [255, 219, 186],  # K=4500
            [255, 221, 190],  # K=4600
            [255, 223, 194],  # K=4700
            [255, 225, 198],  # K=4800
            [255, 227, 202],  # K=4900
            [255, 228, 206],  # K=5000
            [255, 230, 210],  # K=5100
            [255, 232, 213],  # K=5200
            [255, 233, 217],  # K=5300
            [255, 235, 220],  # K=5400
            [255, 236, 224],  # K=5500
            [255, 238, 227],  # K=5600
            [255, 239, 230],  # K=5700
            [255, 240, 233],  # K=5800
            [255, 242, 236],  # K=5900
            [255, 243, 239],  # K=6000
            [255, 244, 242],  # K=6100
            [255, 245, 245],  # K=6200
            [255, 246, 248],  # K=6300
            [255, 248, 251],  # K=6400
            [255, 249, 253],  # K=6500
            [254, 249, 255],  # K=6600
            [252, 247, 255],  # K=6700
            [249, 246, 255],  # K=6800
            [247, 245, 255],  # K=6900
            [245, 243, 255],  # K=7000
            [243, 242, 255],  # K=7100
            [240, 241, 255],  # K=7200
            [239, 240, 255],  # K=7300
            [237, 239, 255],  # K=7400
            [235, 238, 255],  # K=7500
            [233, 237, 255],  # K=7600
            [231, 236, 255],  # K=7700
            [230, 235, 255],  # K=7800
            [228, 234, 255],  # K=7900
            [227, 233, 255],  # K=8000
            [225, 232, 255],  # K=8100
            [224, 231, 255],  # K=8200
            [222, 230, 255],  # K=8300
            [221, 230, 255],  # K=8400
            [220, 229, 255],  # K=8500
            [218, 228, 255],  # K=8600
            [217, 227, 255],  # K=8700
            [216, 227, 255],  # K=8800
            [215, 226, 255],  # K=8900
            [214, 225, 255],  # K=9000
            [212, 225, 255],  # K=9100
            [211, 224, 255],  # K=9200
            [210, 223, 255],  # K=9300
            [209, 223, 255],  # K=9400
            [208, 222, 255],  # K=9500
            [207, 221, 255],  # K=9600
            [207, 221, 255],  # K=9700
            [206, 220, 255],  # K=9800
            [205, 220, 255],  # K=9900
            [204, 219, 255],  # K=10000
            [203, 219, 255],  # K=10100
            [202, 218, 255],  # K=10200
            [201, 218, 255],  # K=10300
            [201, 217, 255],  # K=10400
            [200, 217, 255],  # K=10500
            [199, 216, 255],  # K=10600
            [199, 216, 255],  # K=10700
            [198, 216, 255],  # K=10800
            [197, 215, 255],  # K=10900
            [196, 215, 255],  # K=11000
            [196, 214, 255],  # K=11100
            [195, 214, 255],  # K=11200
            [195, 214, 255],  # K=11300
            [194, 213, 255],  # K=11400
            [193, 213, 255],  # K=11500
            [193, 212, 255],  # K=11600
            [192, 212, 255],  # K=11700
            [192, 212, 255],  # K=11800
            [191, 211, 255],  # K=11900
            [191, 211, 255],  # K=12000
            [190, 211, 255],  # K=12100
            [190, 210, 255],  # K=12200
            [189, 210, 255],  # K=12300
            [189, 210, 255],  # K=12400
            [188, 210, 255],  # K=12500
            [188, 209, 255],  # K=12600
            [187, 209, 255],  # K=12700
            [187, 209, 255],  # K=12800
            [186, 208, 255],  # K=12900
            [186, 208, 255],  # K=13000
            [185, 208, 255],  # K=13100
            [185, 208, 255],  # K=13200
            [185, 207, 255],  # K=13300
            [184, 207, 255],  # K=13400
            [184, 207, 255],  # K=13500
            [183, 207, 255],  # K=13600
            [183, 206, 255],  # K=13700
            [183, 206, 255],  # K=13800
            [182, 206, 255],  # K=13900
            [182, 206, 255],  # K=14000
            [182, 205, 255],  # K=14100
            [181, 205, 255],  # K=14200
            [181, 205, 255],  # K=14300
            [181, 205, 255],  # K=14400
            [180, 205, 255],  # K=14500
            [180, 204, 255],  # K=14600
            [180, 204, 255],  # K=14700
            [179, 204, 255],  # K=14800
            [179, 204, 255],  # K=14900
            [179, 204, 255],  # K=15000
            [178, 203, 255],  # K=15100
            [178, 203, 255],  # K=15200
            [178, 203, 255],  # K=15300
            [178, 203, 255],  # K=15400
            [177, 203, 255],  # K=15500
            [177, 202, 255],  # K=15600
            [177, 202, 255],  # K=15700
            [177, 202, 255],  # K=15800
            [176, 202, 255],  # K=15900
            [176, 202, 255],  # K=16000
            [176, 202, 255],  # K=16100
            [175, 201, 255],  # K=16200
            [175, 201, 255],  # K=16300
            [175, 201, 255],  # K=16400
            [175, 201, 255],  # K=16500
            [175, 201, 255],  # K=16600
            [174, 201, 255],  # K=16700
            [174, 201, 255],  # K=16800
            [174, 200, 255],  # K=16900
            [174, 200, 255],  # K=17000
            [173, 200, 255],  # K=17100
            [173, 200, 255],  # K=17200
            [173, 200, 255],  # K=17300
            [173, 200, 255],  # K=17400
            [173, 200, 255],  # K=17500
            [172, 199, 255],  # K=17600
            [172, 199, 255],  # K=17700
            [172, 199, 255],  # K=17800
            [172, 199, 255],  # K=17900
            [172, 199, 255],  # K=18000
            [171, 199, 255],  # K=18100
            [171, 199, 255],  # K=18200
            [171, 199, 255],  # K=18300
            [171, 198, 255],  # K=18400
            [171, 198, 255],  # K=18500
            [170, 198, 255],  # K=18600
            [170, 198, 255],  # K=18700
            [170, 198, 255],  # K=18800
            [170, 198, 255],  # K=18900
            [170, 198, 255],  # K=19000
            [170, 198, 255],  # K=19100
            [169, 198, 255],  # K=19200
            [169, 197, 255],  # K=19300
            [169, 197, 255],  # K=19400
            [169, 197, 255],  # K=19500
            [169, 197, 255],  # K=19600
            [169, 197, 255],  # K=19700
            [169, 197, 255],  # K=19800
            [168, 197, 255],  # K=19900
            [168, 197, 255],  # K=20000
            [168, 197, 255],  # K=20100
            [168, 197, 255],  # K=20200
            [168, 196, 255],  # K=20300
            [168, 196, 255],  # K=20400
            [168, 196, 255],  # K=20500
            [167, 196, 255],  # K=20600
            [167, 196, 255],  # K=20700
            [167, 196, 255],  # K=20800
            [167, 196, 255],  # K=20900
            [167, 196, 255],  # K=21000
            [167, 196, 255],  # K=21100
            [167, 196, 255],  # K=21200
            [166, 196, 255],  # K=21300
            [166, 195, 255],  # K=21400
            [166, 195, 255],  # K=21500
            [166, 195, 255],  # K=21600
            [166, 195, 255],  # K=21700
            [166, 195, 255],  # K=21800
            [166, 195, 255],  # K=21900
            [166, 195, 255],  # K=22000
            [165, 195, 255],  # K=22100
            [165, 195, 255],  # K=22200
            [165, 195, 255],  # K=22300
            [165, 195, 255],  # K=22400
            [165, 195, 255],  # K=22500
            [165, 195, 255],  # K=22600
            [165, 194, 255],  # K=22700
            [165, 194, 255],  # K=22800
            [165, 194, 255],  # K=22900
            [164, 194, 255],  # K=23000
            [164, 194, 255],  # K=23100
            [164, 194, 255],  # K=23200
            [164, 194, 255],  # K=23300
            [164, 194, 255],  # K=23400
            [164, 194, 255],  # K=23500
            [164, 194, 255],  # K=23600
            [164, 194, 255],  # K=23700
            [164, 194, 255],  # K=23800
            [164, 194, 255],  # K=23900
            [163, 194, 255],  # K=24000
            [163, 194, 255],  # K=24100
            [163, 193, 255],  # K=24200
            [163, 193, 255],  # K=24300
            [163, 193, 255],  # K=24400
            [163, 193, 255],  # K=24500
            [163, 193, 255],  # K=24600
            [163, 193, 255],  # K=24700
            [163, 193, 255],  # K=24800
            [163, 193, 255],  # K=24900
            [163, 193, 255],  # K=25000
            [162, 193, 255],  # K=25100
            [162, 193, 255],  # K=25200
            [162, 193, 255],  # K=25300
            [162, 193, 255],  # K=25400
            [162, 193, 255],  # K=25500
            [162, 193, 255],  # K=25600
            [162, 193, 255],  # K=25700
            [162, 193, 255],  # K=25800
            [162, 192, 255],  # K=25900
            [162, 192, 255],  # K=26000
            [162, 192, 255],  # K=26100
            [162, 192, 255],  # K=26200
            [162, 192, 255],  # K=26300
            [161, 192, 255],  # K=26400
            [161, 192, 255],  # K=26500
            [161, 192, 255],  # K=26600
            [161, 192, 255],  # K=26700
            [161, 192, 255],  # K=26800
            [161, 192, 255],  # K=26900
            [161, 192, 255],  # K=27000
            [161, 192, 255],  # K=27100
            [161, 192, 255],  # K=27200
            [161, 192, 255],  # K=27300
            [161, 192, 255],  # K=27400
            [161, 192, 255],  # K=27500
            [161, 192, 255],  # K=27600
            [161, 192, 255],  # K=27700
            [160, 192, 255],  # K=27800
            [160, 192, 255],  # K=27900
            [160, 191, 255],  # K=28000
            [160, 191, 255],  # K=28100
            [160, 191, 255],  # K=28200
            [160, 191, 255],  # K=28300
            [160, 191, 255],  # K=28400
            [160, 191, 255],  # K=28500
            [160, 191, 255],  # K=28600
            [160, 191, 255],  # K=28700
            [160, 191, 255],  # K=28800
            [160, 191, 255],  # K=28900
            [160, 191, 255],  # K=29000
            [160, 191, 255],  # K=29100
            [160, 191, 255],  # K=29200
            [159, 191, 255],  # K=29300
            [159, 191, 255],  # K=29400
            [159, 191, 255],  # K=29500
            [159, 191, 255],  # K=29600
            [159, 191, 255],  # K=29700
            [159, 191, 255],  # K=29800
            [159, 191, 255],  # K=29900
            [159, 191, 255],  # K=30000
            [159, 191, 255],  # K=30100
            [159, 191, 255],  # K=30200
            [159, 191, 255],  # K=30300
            [159, 190, 255],  # K=30400
            [159, 190, 255],  # K=30500
            [159, 190, 255],  # K=30600
            [159, 190, 255],  # K=30700
            [159, 190, 255],  # K=30800
            [159, 190, 255],  # K=30900
            [159, 190, 255],  # K=31000
            [158, 190, 255],  # K=31100
            [158, 190, 255],  # K=31200
            [158, 190, 255],  # K=31300
            [158, 190, 255],  # K=31400
            [158, 190, 255],  # K=31500
            [158, 190, 255],  # K=31600
            [158, 190, 255],  # K=31700
            [158, 190, 255],  # K=31800
            [158, 190, 255],  # K=31900
            [158, 190, 255],  # K=32000
            [158, 190, 255],  # K=32100
            [158, 190, 255],  # K=32200
            [158, 190, 255],  # K=32300
            [158, 190, 255],  # K=32400
            [158, 190, 255],  # K=32500
            [158, 190, 255],  # K=32600
            [158, 190, 255],  # K=32700
            [158, 190, 255],  # K=32800
            [158, 190, 255],  # K=32900
            [158, 190, 255],  # K=33000
            [158, 190, 255],  # K=33100
            [157, 190, 255],  # K=33200
            [157, 190, 255],  # K=33300
            [157, 189, 255],  # K=33400
            [157, 189, 255],  # K=33500
            [157, 189, 255],  # K=33600
            [157, 189, 255],  # K=33700
            [157, 189, 255],  # K=33800
            [157, 189, 255],  # K=33900
            [157, 189, 255],  # K=34000
            [157, 189, 255],  # K=34100
            [157, 189, 255],  # K=34200
            [157, 189, 255],  # K=34300
            [157, 189, 255],  # K=34400
            [157, 189, 255],  # K=34500
            [157, 189, 255],  # K=34600
            [157, 189, 255],  # K=34700
            [157, 189, 255],  # K=34800
            [157, 189, 255],  # K=34900
            [157, 189, 255],  # K=35000
            [157, 189, 255],  # K=35100
            [157, 189, 255],  # K=35200
            [157, 189, 255],  # K=35300
            [157, 189, 255],  # K=35400
            [157, 189, 255],  # K=35500
            [156, 189, 255],  # K=35600
            [156, 189, 255],  # K=35700
            [156, 189, 255],  # K=35800
            [156, 189, 255],  # K=35900
            [156, 189, 255],  # K=36000
            [156, 189, 255],  # K=36100
            [156, 189, 255],  # K=36200
            [156, 189, 255],  # K=36300
            [156, 189, 255],  # K=36400
            [156, 189, 255],  # K=36500
            [156, 189, 255],  # K=36600
            [156, 189, 255],  # K=36700
            [156, 189, 255],  # K=36800
            [156, 189, 255],  # K=36900
            [156, 189, 255],  # K=37000
            [156, 189, 255],  # K=37100
            [156, 188, 255],  # K=37200
            [156, 188, 255],  # K=37300
            [156, 188, 255],  # K=37400
            [156, 188, 255],  # K=37500
            [156, 188, 255],  # K=37600
            [156, 188, 255],  # K=37700
            [156, 188, 255],  # K=37800
            [156, 188, 255],  # K=37900
            [156, 188, 255],  # K=38000
            [156, 188, 255],  # K=38100
            [156, 188, 255],  # K=38200
            [156, 188, 255],  # K=38300
            [155, 188, 255],  # K=38400
            [155, 188, 255],  # K=38500
            [155, 188, 255],  # K=38600
            [155, 188, 255],  # K=38700
            [155, 188, 255],  # K=38800
            [155, 188, 255],  # K=38900
            [155, 188, 255],  # K=39000
            [155, 188, 255],  # K=39100
            [155, 188, 255],  # K=39200
            [155, 188, 255],  # K=39300
            [155, 188, 255],  # K=39400
            [155, 188, 255],  # K=39500
            [155, 188, 255],  # K=39600
            [155, 188, 255],  # K=39700
            [155, 188, 255],  # K=39800
            [155, 188, 255],  # K=39900
            [155, 188, 255],  # K=40000
        ]) / 255.0
        _KelvinToRGBTable._TABLE = table
        return table


def change_color_temperatures_(images, kelvins, from_colorspaces=CSPACE_RGB):
    """Change in-place the temperature of images to given values in Kelvin.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.color.change_colorspace_`.

    Parameters
    ----------
    images : ndarray or list of ndarray
        The images which's color temperature is supposed to be changed.
        Either a list of ``(H,W,3)`` arrays or a single ``(N,H,W,3)`` array.

    kelvins : iterable of number
        Temperatures in Kelvin. One per image. Expected value range is in
        the interval ``(1000, 4000)``.

    from_colorspaces : str or list of str, optional
        The source colorspace.
        See :func:`~imgaug.augmenters.color.change_colorspaces_`.
        Defaults to ``RGB``.

    Returns
    -------
    ndarray or list of ndarray
        Images with target color temperatures.
        The input array(s) might have been changed in-place.

    """
    # we return here early, because we validate below the first kelvin value
    if len(images) == 0:
        return images

    # TODO this is very similar to the validation in change_colorspaces_().
    #      Make DRY.
    def _validate(arg, arg_name, datatype):
        if ia.is_iterable(arg) and not ia.is_string(arg):
            assert len(arg) == len(images), (
                "If `%s` is provided as an iterable it must have the same "
                "length as `images`. Got length %d, expected %d." % (
                    arg_name, len(arg), len(images)))
        elif datatype == "str":
            assert ia.is_string(arg), (
                "Expected `%s` to be either an iterable of strings or a single "
                "string. Got type %s." % (arg_name, type(arg).__name__))
            arg = [arg] * len(images)
        else:
            assert ia.is_single_number(arg), (
                "Expected `%s` to be either an iterable of numbers or a single "
                "number. Got type %s." % (arg_name, type(arg).__name__))
            arg = np.tile(np.float32([arg]), (len(images),))
        return arg

    kelvins = _validate(kelvins, "kelvins", "number")
    from_colorspaces = _validate(from_colorspaces, "from_colorspaces", "str")

    # list `kelvins` inputs are not yet converted to ndarray by _validate()
    kelvins = np.array(kelvins, dtype=np.float32)

    # Validate only one kelvin value for performance reasons.
    # If values are outside that range, the kelvin table simply clips them.
    # If there are no images (and hence no kelvin values), we already returned
    # above.
    assert 1000 <= kelvins[0] <= 40000, (
        "Expected Kelvin values in the interval [1000, 40000]. "
        "Got interval [%.8f, %.8f]." % (np.min(kelvins), np.max(kelvins)))

    table = _KelvinToRGBTableSingleton.get_instance()
    rgb_multipliers = table.transform_kelvins_to_rgb_multipliers(kelvins)
    rgb_multipliers_nhwc = rgb_multipliers.reshape((-1, 1, 1, 3))

    gen = enumerate(zip(images, rgb_multipliers_nhwc, from_colorspaces))
    for i, (image, rgb_multiplier_hwc, from_colorspace) in gen:
        image_rgb = change_colorspace_(image,
                                       to_colorspace=CSPACE_RGB,
                                       from_colorspace=from_colorspace)

        # we should always have uint8 here as only that is accepted by
        # convert_colorspace
        assert image_rgb.dtype.name == "uint8", (
            "Expected dtype uint8, got %s." % (image_rgb.dtype.name,))

        # all multipliers are in the range [0.0, 1.0], hence we can afford to
        # not clip here
        image_temp_adj = np.round(
            image_rgb.astype(np.float32) * rgb_multiplier_hwc
        ).astype(np.uint8)

        image_orig_cspace = change_colorspace_(image_temp_adj,
                                               to_colorspace=from_colorspace,
                                               from_colorspace=CSPACE_RGB)
        images[i] = image_orig_cspace
    return images


def change_color_temperature(image, kelvin, from_colorspace=CSPACE_RGB):
    """Change the temperature of an image to a given value in Kelvin.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.color.change_color_temperatures_`.

    Parameters
    ----------
    image : ndarray
        The image which's color temperature is supposed to be changed.
        Expected to be of shape ``(H,W,3)`` array.

    kelvin : number
        The temperature in Kelvin. Expected value range is in
        the interval ``(1000, 4000)``.

    from_colorspace : str, optional
        The source colorspace.
        See :func:`~imgaug.augmenters.color.change_colorspaces_`.
        Defaults to ``RGB``.

    Returns
    -------
    ndarray
        Image with target color temperature.

    """
    return change_color_temperatures_(image[np.newaxis, ...],
                                      [kelvin],
                                      from_colorspaces=[from_colorspace])[0]


@ia.deprecated(alt_func="WithColorspace")
def InColorspace(to_colorspace, from_colorspace="RGB", children=None,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
    """Convert images to another colorspace."""
    # pylint: disable=invalid-name
    return WithColorspace(to_colorspace, from_colorspace, children,
                          seed=seed, name=name,
                          random_state=random_state,
                          deterministic=deterministic)


# TODO add tests
class WithColorspace(meta.Augmenter):
    """
    Apply child augmenters within a specific colorspace.

    This augumenter takes a source colorspace A and a target colorspace B
    as well as children C. It changes images from A to B, then applies the
    child augmenters C and finally changes the colorspace back from B to A.
    See also ChangeColorspace() for more.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.color.change_colorspaces_`.

    Parameters
    ----------
    to_colorspace : str
        See :func:`~imgaug.augmenters.color.change_colorspace_`.

    from_colorspace : str, optional
        See :func:`~imgaug.augmenters.color.change_colorspace_`.

    children : imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter or None, optional
        One or more augmenters to apply to converted images.

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
    >>> aug = iaa.WithColorspace(
    >>>     to_colorspace=iaa.CSPACE_HSV,
    >>>     from_colorspace=iaa.CSPACE_RGB,
    >>>     children=iaa.WithChannels(
    >>>         0,
    >>>         iaa.Add((0, 50))
    >>>     )
    >>> )

    Convert to ``HSV`` colorspace, add a value between ``0`` and ``50``
    (uniformly sampled per image) to the Hue channel, then convert back to the
    input colorspace (``RGB``).

    """

    def __init__(self, to_colorspace, from_colorspace=CSPACE_RGB, children=None,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(WithColorspace, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.to_colorspace = to_colorspace
        self.from_colorspace = from_colorspace
        self.children = meta.handle_children_list(children, self.name, "then")

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        with batch.propagation_hooks_ctx(self, hooks, parents):
            # TODO this did not fail in the tests when there was only one
            #      `if` with all three steps in it
            if batch.images is not None:
                batch.images = change_colorspaces_(
                    batch.images,
                    to_colorspaces=self.to_colorspace,
                    from_colorspaces=self.from_colorspace)

            batch = self.children.augment_batch_(
                batch,
                parents=parents + [self],
                hooks=hooks
            )

            if batch.images is not None:
                batch.images = change_colorspaces_(
                    batch.images,
                    to_colorspaces=self.from_colorspace,
                    from_colorspaces=self.to_colorspace)
        return batch

    def _to_deterministic(self):
        aug = self.copy()
        aug.children = aug.children.to_deterministic()
        aug.deterministic = True
        aug.random_state = self.random_state.derive_rng_()
        return aug

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.channels]

    def get_children_lists(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_children_lists`."""
        return [self.children]

    def __str__(self):
        return (
            "WithColorspace(from_colorspace=%s, "
            "to_colorspace=%s, name=%s, children=[%s], deterministic=%s)" % (
                self.from_colorspace, self.to_colorspace, self.name,
                self.children, self.deterministic)
        )


class WithBrightnessChannels(meta.Augmenter):
    """Augmenter to apply child augmenters to brightness-related image channels.

    This augmenter first converts an image to a random colorspace containing a
    brightness-related channel (e.g. ``V`` in ``HSV``), then extracts that
    channel and applies its child augmenters to this one channel. Afterwards,
    it reintegrates the augmented channel into the full image and converts
    back to the input colorspace.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.color.change_colorspaces_`.

    Parameters
    ----------
    children : imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter or None, optional
        One or more augmenters to apply to the brightness channels.
        They receive images with a single channel and have to modify these.

    to_colorspace : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        Colorspace in which to extract the brightness-related channels.
        Currently, ``imgaug.augmenters.color.CSPACE_YCrCb``, ``CSPACE_HSV``,
        ``CSPACE_HLS``, ``CSPACE_Lab``, ``CSPACE_Luv``, ``CSPACE_YUV``,
        ``CSPACE_CIE`` are supported.

            * If ``imgaug.ALL``: Will pick imagewise a random colorspace from
              all supported colorspaces.
            * If ``str``: Will always use this colorspace.
            * If ``list`` or ``str``: Will pick imagewise a random colorspace
              from this list.
            * If :class:`~imgaug.parameters.StochasticParameter`:
              A parameter that will be queried once per batch to generate
              all target colorspaces. Expected to return strings matching the
              ``CSPACE_*`` constants.

    from_colorspace : str, optional
        See :func:`~imgaug.augmenters.color.change_colorspace_`.

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
    >>> aug = iaa.WithBrightnessChannels(iaa.Add((-50, 50)))

    Add ``-50`` to ``50`` to the brightness-related channels of each image.

    >>> aug = iaa.WithBrightnessChannels(
    >>>     iaa.Add((-50, 50)), to_colorspace=[iaa.CSPACE_Lab, iaa.CSPACE_HSV])

    Add ``-50`` to ``50`` to the brightness-related channels of each image,
    but pick those brightness-related channels only from ``Lab`` (``L``) and
    ``HSV`` (``V``) colorspaces.

    >>> aug = iaa.WithBrightnessChannels(
    >>>     iaa.Add((-50, 50)), from_colorspace=iaa.CSPACE_BGR)

    Add ``-50`` to ``50`` to the brightness-related channels of each image,
    where the images are provided in ``BGR`` colorspace instead of the
    standard ``RGB``.

    """

    # Usually one would think that CSPACE_CIE (=XYZ) would also work, as
    # wikipedia says that Y denotes luminance, but this resulted in strong
    # color changes (tried also the other channels).
    _CSPACE_TO_CHANNEL_ID = {
        CSPACE_YCrCb: 0,
        CSPACE_HSV: 2,
        CSPACE_HLS: 1,
        CSPACE_Lab: 0,
        CSPACE_Luv: 0,
        CSPACE_YUV: 0
    }

    _VALID_COLORSPACES = set(_CSPACE_TO_CHANNEL_ID.keys())

    # Added in 0.4.0.
    def __init__(self, children=None,
                 to_colorspace=[
                     CSPACE_YCrCb,
                     CSPACE_HSV,
                     CSPACE_HLS,
                     CSPACE_Lab,
                     CSPACE_Luv,
                     CSPACE_YUV],
                 from_colorspace="RGB",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        # pylint: disable=dangerous-default-value
        super(WithBrightnessChannels, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.children = meta.handle_children_list(children, self.name, "then")
        self.to_colorspace = iap.handle_categorical_string_param(
            to_colorspace, "to_colorspace",
            valid_values=self._VALID_COLORSPACES)
        self.from_colorspace = from_colorspace

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        with batch.propagation_hooks_ctx(self, hooks, parents):
            images_cvt = None
            to_colorspaces = None

            if batch.images is not None:
                to_colorspaces = self.to_colorspace.draw_samples(
                    (len(batch.images),), random_state)
                images_cvt = change_colorspaces_(
                    batch.images,
                    from_colorspaces=self.from_colorspace,
                    to_colorspaces=to_colorspaces,)
                brightness_channels = self._extract_brightness_channels(
                    images_cvt, to_colorspaces)

                batch.images = brightness_channels

            batch = self.children.augment_batch_(
                batch, parents=parents + [self], hooks=hooks)

            if batch.images is not None:
                batch.images = self._invert_extract_brightness_channels(
                    batch.images, images_cvt, to_colorspaces)

                batch.images = change_colorspaces_(
                    batch.images,
                    from_colorspaces=to_colorspaces,
                    to_colorspaces=self.from_colorspace)

        return batch

    # Added in 0.4.0.
    def _extract_brightness_channels(self, images, colorspaces):
        result = []
        for image, colorspace in zip(images, colorspaces):
            channel_id = self._CSPACE_TO_CHANNEL_ID[colorspace]
            # Note that augmenters expect (H,W,C) and not (H,W), so cannot
            # just use image[:, :, channel_id] here.
            channel = image[:, :, channel_id:channel_id+1]
            result.append(channel)
        return result

    # Added in 0.4.0.
    def _invert_extract_brightness_channels(self, channels, images,
                                            colorspaces):
        for channel, image, colorspace in zip(channels, images, colorspaces):
            channel_id = self._CSPACE_TO_CHANNEL_ID[colorspace]
            image[:, :, channel_id:channel_id+1] = channel
        return images

    # Added in 0.4.0.
    def _to_deterministic(self):
        aug = self.copy()
        aug.children = aug.children.to_deterministic()
        aug.deterministic = True
        aug.random_state = self.random_state.derive_rng_()
        return aug

    # Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.to_colorspace, self.from_colorspace]

    # Added in 0.4.0.
    def get_children_lists(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_children_lists`."""
        return [self.children]

    # Added in 0.4.0.
    def __str__(self):
        return (
            "WithBrightnessChannels("
            "to_colorspace=%s, "
            "from_colorspace=%s, "
            "name=%s, "
            "children=%s, "
            "deterministic=%s)" % (
                self.to_colorspace,
                self.from_colorspace,
                self.name,
                self.children,
                self.deterministic)
        )


class MultiplyAndAddToBrightness(WithBrightnessChannels):
    """Multiply and add to the brightness channels of input images.

    This is a wrapper around :class:`WithBrightnessChannels` and hence
    performs internally the same projection to random colorspaces.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.color.WithBrightnessChannels`.

    Parameters
    ----------
    mul : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.airthmetic.Multiply`.

    add : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.airthmetic.Add`.

    to_colorspace : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.color.WithBrightnessChannels`.

    from_colorspace : str, optional
        See :class:`~imgaug.augmenters.color.WithBrightnessChannels`.

    random_order : bool, optional
        Whether to apply the add and multiply operations in random
        order (``True``). If ``False``, this augmenter will always first
        multiply and then add.

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
    >>> aug = iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))

    Convert each image to a colorspace with a brightness-related channel,
    extract that channel, multiply it by a factor between ``0.5`` and ``1.5``,
    add a value between ``-30`` and ``30`` and convert back to the original
    colorspace.

    """

    # Added in 0.4.0.
    def __init__(self, mul=(0.7, 1.3), add=(-30, 30),
                 to_colorspace=[
                     CSPACE_YCrCb,
                     CSPACE_HSV,
                     CSPACE_HLS,
                     CSPACE_Lab,
                     CSPACE_Luv,
                     CSPACE_YUV],
                 from_colorspace="RGB",
                 random_order=True,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        # pylint: disable=dangerous-default-value
        mul = (
            meta.Identity()
            if ia.is_single_number(mul) and np.isclose(mul, 1.0)
            else arithmetic.Multiply(mul))
        add = meta.Identity() if add == 0 else arithmetic.Add(add)

        super(MultiplyAndAddToBrightness, self).__init__(
            children=meta.Sequential(
                [mul, add],
                random_order=random_order
            ),
            to_colorspace=to_colorspace,
            from_colorspace=from_colorspace,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

    # Added in 0.4.0.
    def __str__(self):
        return (
            "MultiplyAndAddToBrightness("
            "mul=%s, "
            "add=%s, "
            "to_colorspace=%s, "
            "from_colorspace=%s, "
            "random_order=%s, "
            "name=%s, "
            "deterministic=%s)" % (
                str(self.children[0]),
                str(self.children[1]),
                self.to_colorspace,
                self.from_colorspace,
                self.children.random_order,
                self.name,
                self.deterministic)
        )


class MultiplyBrightness(MultiplyAndAddToBrightness):
    """Multiply the brightness channels of input images.

    This is a wrapper around :class:`WithBrightnessChannels` and hence
    performs internally the same projection to random colorspaces.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.color.MultiplyAndAddToBrightness`.

    Parameters
    ----------
    mul : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.airthmetic.Multiply`.

    to_colorspace : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.color.WithBrightnessChannels`.

    from_colorspace : str, optional
        See :class:`~imgaug.augmenters.color.WithBrightnessChannels`.

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
    >>> aug = iaa.MultiplyBrightness((0.5, 1.5))

    Convert each image to a colorspace with a brightness-related channel,
    extract that channel, multiply it by a factor between ``0.5`` and ``1.5``,
    and convert back to the original colorspace.

    """

    # Added in 0.4.0.
    def __init__(self, mul=(0.7, 1.3),
                 to_colorspace=[
                     CSPACE_YCrCb,
                     CSPACE_HSV,
                     CSPACE_HLS,
                     CSPACE_Lab,
                     CSPACE_Luv,
                     CSPACE_YUV],
                 from_colorspace="RGB",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        # pylint: disable=dangerous-default-value
        super(MultiplyBrightness, self).__init__(
            mul=mul,
            add=0,
            to_colorspace=to_colorspace,
            from_colorspace=from_colorspace,
            random_order=False,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class AddToBrightness(MultiplyAndAddToBrightness):
    """Add to the brightness channels of input images.

    This is a wrapper around :class:`WithBrightnessChannels` and hence
    performs internally the same projection to random colorspaces.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.color.MultiplyAndAddToBrightness`.

    Parameters
    ----------
    add : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.airthmetic.Add`.

    to_colorspace : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.color.WithBrightnessChannels`.

    from_colorspace : str, optional
        See :class:`~imgaug.augmenters.color.WithBrightnessChannels`.

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
    >>> aug = iaa.AddToBrightness((-30, 30))

    Convert each image to a colorspace with a brightness-related channel,
    extract that channel, add between ``-30`` and ``30`` and convert back
    to the original colorspace.

    """

    # Added in 0.4.0.
    def __init__(self, add=(-30, 30),
                 to_colorspace=[
                     CSPACE_YCrCb,
                     CSPACE_HSV,
                     CSPACE_HLS,
                     CSPACE_Lab,
                     CSPACE_Luv,
                     CSPACE_YUV],
                 from_colorspace="RGB",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        # pylint: disable=dangerous-default-value
        super(AddToBrightness, self).__init__(
            mul=1.0,
            add=add,
            to_colorspace=to_colorspace,
            from_colorspace=from_colorspace,
            random_order=False,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


# TODO Merge this into WithColorspace? A bit problematic due to int16
#      conversion that would make WithColorspace less flexible.
# TODO add option to choose overflow behaviour for hue and saturation channels,
#      e.g. clip, modulo or wrap
class WithHueAndSaturation(meta.Augmenter):
    """
    Apply child augmenters to hue and saturation channels.

    This augumenter takes an image in a source colorspace, converts
    it to HSV, extracts the H (hue) and S (saturation) channels,
    applies the provided child augmenters to these channels
    and finally converts back to the original colorspace.

    The image array generated by this augmenter and provided to its children
    is in ``int16`` (**sic!** only augmenters that can handle ``int16`` arrays
    can be children!). The hue channel is mapped to the value
    range ``[0, 255]``. Before converting back to the source colorspace, the
    saturation channel's values are clipped to ``[0, 255]``. A modulo operation
    is applied to the hue channel's values, followed by a mapping from
    ``[0, 255]`` to ``[0, 180]`` (and finally the colorspace conversion).

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.color.change_colorspaces_`.

    Parameters
    ----------
    from_colorspace : str, optional
        See :func:`~imgaug.augmenters.color.change_colorspace_`.

    children : imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter or None, optional
        One or more augmenters to apply to converted images.
        They receive ``int16`` images with two channels (hue, saturation)
        and have to modify these.

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
    >>> aug = iaa.WithHueAndSaturation(
    >>>     iaa.WithChannels(0, iaa.Add((0, 50)))
    >>> )

    Create an augmenter that will add a random value between ``0`` and ``50``
    (uniformly sampled per image) hue channel in HSV colorspace. It
    automatically accounts for the hue being in angular representation, i.e.
    if the angle goes beyond 360 degrees, it will start again at 0 degrees.
    The colorspace is finally converted back to ``RGB`` (default setting).

    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.WithHueAndSaturation([
    >>>     iaa.WithChannels(0, iaa.Add((-30, 10))),
    >>>     iaa.WithChannels(1, [
    >>>         iaa.Multiply((0.5, 1.5)),
    >>>         iaa.LinearContrast((0.75, 1.25))
    >>>     ])
    >>> ])

    Create an augmenter that adds a random value sampled uniformly
    from the range ``[-30, 10]`` to the hue and multiplies the saturation
    by a random factor sampled uniformly from ``[0.5, 1.5]``. It also
    modifies the contrast of the saturation channel. After these steps,
    the ``HSV`` image is converted back to ``RGB``.

    """

    def __init__(self, children=None, from_colorspace="RGB",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(WithHueAndSaturation, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.children = meta.handle_children_list(children, self.name, "then")
        self.from_colorspace = from_colorspace

        # this dtype needs to be able to go beyond [0, 255] to e.g. accomodate
        # for Add or Multiply
        self._internal_dtype = np.int16

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        with batch.propagation_hooks_ctx(self, hooks, parents):
            images_hs, images_hsv = self._images_to_hsv_(batch.images)
            batch.images = images_hs

            batch = self.children.augment_batch_(
                batch, parents=parents + [self], hooks=hooks)

            batch.images = self._hs_to_images_(batch.images, images_hsv)

        return batch

    # Added in 0.4.0.
    def _images_to_hsv_(self, images):
        if images is None:
            return None, None

        # RGB (or other source colorspace) -> HSV
        images_hsv = change_colorspaces_(
            images, CSPACE_HSV, self.from_colorspace)

        # HSV -> HS
        images_hs = []
        for image_hsv in images_hsv:
            image_hsv = image_hsv.astype(np.int16)
            # project hue from [0,180] to [0,255] so that child augmenters
            # can assume the same value range for all channels
            hue = (
                (image_hsv[:, :, 0].astype(np.float32) / 180.0) * 255.0
            ).astype(self._internal_dtype)
            saturation = image_hsv[:, :, 1]
            images_hs.append(np.stack([hue, saturation], axis=-1))
        if ia.is_np_array(images_hsv):
            images_hs = np.stack(images_hs, axis=0)
        return images_hs, images_hsv

    # Added in 0.4.0.
    def _hs_to_images_(self, images_hs, images_hsv):
        if images_hs is None:
            return None
        # postprocess augmented HS int16 data
        # hue: modulo to [0, 255] then project to [0, 360/2]
        # saturation: clip to [0, 255]
        # + convert to uint8
        # + re-attach V channel to HS
        hue_and_sat_proj = []
        for i, hs_aug in enumerate(images_hs):
            hue_aug = hs_aug[:, :, 0]
            sat_aug = hs_aug[:, :, 1]
            hue_aug = (
                (np.mod(hue_aug, 255).astype(np.float32) / 255.0)
                * (360/2)
            ).astype(np.uint8)
            sat_aug = iadt.clip_(sat_aug, 0, 255).astype(np.uint8)
            hue_and_sat_proj.append(
                np.stack([hue_aug, sat_aug, images_hsv[i][:, :, 2]],
                         axis=-1)
            )
        if ia.is_np_array(images_hs):
            hue_and_sat_proj = np.uint8(hue_and_sat_proj)

        # HSV -> RGB (or whatever the source colorspace was)
        images_rgb = change_colorspaces_(
            hue_and_sat_proj,
            to_colorspaces=self.from_colorspace,
            from_colorspaces=CSPACE_HSV)
        return images_rgb

    def _to_deterministic(self):
        aug = self.copy()
        aug.children = aug.children.to_deterministic()
        aug.deterministic = True
        aug.random_state = self.random_state.derive_rng_()
        return aug

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.from_colorspace]

    def get_children_lists(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_children_lists`."""
        return [self.children]

    def __str__(self):
        return (
            "WithHueAndSaturation(from_colorspace=%s, "
            "name=%s, children=[%s], deterministic=%s)" % (
                self.from_colorspace, self.name,
                self.children, self.deterministic)
        )


class MultiplyHueAndSaturation(WithHueAndSaturation):
    """
    Multipy hue and saturation by random values.

    The augmenter first transforms images to HSV colorspace, then multiplies
    the pixel values in the H and S channels and afterwards converts back to
    RGB.

    This augmenter is a wrapper around ``WithHueAndSaturation``.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.color.WithHueAndSaturation`.

    Parameters
    ----------
    mul : None or number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Multiplier with which to multiply all hue *and* saturation values of
        all pixels.
        It is expected to be in the range ``-10.0`` to ``+10.0``.
        Note that values of ``0.0`` or lower will remove all saturation.

            * If this is ``None``, `mul_hue` and/or `mul_saturation`
              may be set to values other than ``None``.
            * If a number, then that multiplier will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous
              range ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then a value will be sampled from that
              parameter per image.

    mul_hue : None or number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Multiplier with which to multiply all hue values.
        This is expected to be in the range ``-10.0`` to ``+10.0`` and will
        automatically be projected to an angular representation using
        ``(hue/255) * (360/2)`` (OpenCV's hue representation is in the
        range ``[0, 180]`` instead of ``[0, 360]``).
        Only this or `mul` may be set, not both.

            * If this and `mul_saturation` are both ``None``, `mul` may
              be set to a non-``None`` value.
            * If a number, then that multiplier will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous
              range ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then a value will be sampled from that
              parameter per image.

    mul_saturation : None or number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Multiplier with which to multiply all saturation values.
        It is expected to be in the range ``0.0`` to ``+10.0``.
        Only this or `mul` may be set, not both.

            * If this and `mul_hue` are both ``None``, `mul` may
              be set to a non-``None`` value.
            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous
              range ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then a value will be sampled from that
              parameter per image.

    per_channel : bool or float, optional
        Whether to sample per image only one value from `mul` and use it for
        both hue and saturation (``False``) or to sample independently one
        value for hue and one for saturation (``True``).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``, otherwise as ``False``.

        This parameter has no effect if `mul_hue` and/or `mul_saturation`
        are used instead of `mul`.

    from_colorspace : str, optional
        See :func:`~imgaug.augmenters.color.change_colorspace_`.

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
    >>> aug = iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)

    Multiply hue and saturation by random values between ``0.5`` and ``1.5``
    (independently per channel and the same value for all pixels within
    that channel). The hue will be automatically projected to an angular
    representation.

    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.MultiplyHueAndSaturation(mul_hue=(0.5, 1.5))

    Multiply only the hue by random values between ``0.5`` and ``1.5``.

    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.MultiplyHueAndSaturation(mul_saturation=(0.5, 1.5))

    Multiply only the saturation by random values between ``0.5`` and ``1.5``.

    """

    def __init__(self, mul=None, mul_hue=None, mul_saturation=None,
                 per_channel=False, from_colorspace="RGB",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        if mul is None and mul_hue is None and mul_saturation is None:
            mul_hue = (0.5, 1.5)
            mul_saturation = (0.0, 1.7)

        if mul is not None:
            assert mul_hue is None, (
                "`mul_hue` may not be set if `mul` is set. "
                "It is set to: %s (type: %s)." % (
                    str(mul_hue), type(mul_hue)))
            assert mul_saturation is None, (
                "`mul_saturation` may not be set if `mul` is set. "
                "It is set to: %s (type: %s)." % (
                    str(mul_saturation), type(mul_saturation)))
            mul = iap.handle_continuous_param(
                mul, "mul", value_range=(-10.0, 10.0), tuple_to_uniform=True,
                list_to_choice=True)
        else:
            if mul_hue is not None:
                mul_hue = iap.handle_continuous_param(
                    mul_hue, "mul_hue", value_range=(-10.0, 10.0),
                    tuple_to_uniform=True, list_to_choice=True)
            if mul_saturation is not None:
                mul_saturation = iap.handle_continuous_param(
                    mul_saturation, "mul_saturation", value_range=(0.0, 10.0),
                    tuple_to_uniform=True, list_to_choice=True)

        if random_state != "deprecated":
            seed = random_state
            random_state = "deprecated"

        if seed is None:
            rss = [None] * 5
        else:
            rss = iarandom.RNG(seed).derive_rngs_(5)

        children = []
        if mul is not None:
            children.append(
                arithmetic.Multiply(
                    mul,
                    per_channel=per_channel,
                    seed=rss[0],
                    name="%s-Multiply" % (name,),
                    random_state=random_state,
                    deterministic=deterministic
                )
            )
        else:
            if mul_hue is not None:
                children.append(
                    meta.WithChannels(
                        0,
                        arithmetic.Multiply(
                            mul_hue,
                            seed=rss[0],
                            name="%s-MultiplyHue" % (name,),
                            random_state=random_state,
                            deterministic=deterministic
                        ),
                        seed=rss[1],
                        name="%s-WithChannelsHue" % (name,),
                        random_state=random_state,
                        deterministic=deterministic
                    )
                )
            if mul_saturation is not None:
                children.append(
                    meta.WithChannels(
                        1,
                        arithmetic.Multiply(
                            mul_saturation,
                            seed=rss[2],
                            name="%s-MultiplySaturation" % (name,),
                            random_state=random_state,
                            deterministic=deterministic
                        ),
                        seed=rss[3],
                        name="%s-WithChannelsSaturation" % (name,),
                        random_state=random_state,
                        deterministic=deterministic
                    )
                )

        super(MultiplyHueAndSaturation, self).__init__(
            children,
            from_colorspace=from_colorspace,
            seed=rss[4],
            name=name,
            random_state=random_state, deterministic=deterministic
        )


class MultiplyHue(MultiplyHueAndSaturation):
    """
    Multiply the hue of images by random values.

    The augmenter first transforms images to HSV colorspace, then multiplies
    the pixel values in the H channel and afterwards converts back to
    RGB.

    This augmenter is a shortcut for ``MultiplyHueAndSaturation(mul_hue=...)``.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.color.MultiplyHueAndSaturation`.

    Parameters
    ----------
    mul : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Multiplier with which to multiply all hue values.
        This is expected to be in the range ``-10.0`` to ``+10.0`` and will
        automatically be projected to an angular representation using
        ``(hue/255) * (360/2)`` (OpenCV's hue representation is in the
        range ``[0, 180]`` instead of ``[0, 360]``).
        Only this or `mul` may be set, not both.

            * If a number, then that multiplier will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous
              range ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then a value will be sampled from that
              parameter per image.

    from_colorspace : str, optional
        See :func:`~imgaug.augmenters.color.change_colorspace_`.

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
    >>> aug = iaa.MultiplyHue((0.5, 1.5))

    Multiply the hue channel of images using random values between ``0.5``
    and ``1.5``.

    """

    def __init__(self, mul=(-3.0, 3.0), from_colorspace="RGB",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(MultiplyHue, self).__init__(
            mul_hue=mul,
            from_colorspace=from_colorspace,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class MultiplySaturation(MultiplyHueAndSaturation):
    """
    Multiply the saturation of images by random values.

    The augmenter first transforms images to HSV colorspace, then multiplies
    the pixel values in the H channel and afterwards converts back to
    RGB.

    This augmenter is a shortcut for
    ``MultiplyHueAndSaturation(mul_saturation=...)``.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.color.MultiplyHueAndSaturation`.

    Parameters
    ----------
    mul : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Multiplier with which to multiply all saturation values.
        It is expected to be in the range ``0.0`` to ``+10.0``.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous
              range ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then a value will be sampled from that
              parameter per image.

    from_colorspace : str, optional
        See :func:`~imgaug.augmenters.color.change_colorspace_`.

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
    >>> aug = iaa.MultiplySaturation((0.5, 1.5))

    Multiply the saturation channel of images using random values between
    ``0.5`` and ``1.5``.

    """

    def __init__(self, mul=(0.0, 3.0), from_colorspace="RGB",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(MultiplySaturation, self).__init__(
            mul_saturation=mul,
            from_colorspace=from_colorspace,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class RemoveSaturation(MultiplySaturation):
    """Decrease the saturation of images by varying degrees.

    This creates images looking similar to :class:`Grayscale`.

    This augmenter is the same as ``MultiplySaturation((0.0, 1.0))``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.color.MultiplySaturation`.

    Parameters
    ----------
    mul : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        *Inverse* multiplier to use for the saturation values.
        High values denote stronger color removal. E.g. ``1.0`` will remove
        all saturation, ``0.0`` will remove nothing.
        Expected value range is ``[0.0, 1.0]``.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous
              range ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then a value will be sampled from that
              parameter per image.

    from_colorspace : str, optional
        See :func:`~imgaug.augmenters.color.change_colorspace_`.

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
    >>> aug = iaa.RemoveSaturation((0.0, 1.0))

    Create an augmenter that decreases saturation by varying degrees.

    >>> aug = iaa.RemoveSaturation(1.0)

    Create an augmenter that removes all saturation from input images.
    This is similar to :class:`Grayscale`.

    >>> aug = iaa.RemoveSaturation(from_colorspace=iaa.CSPACE_BGR)

    Create an augmenter that decreases saturation of images in ``BGR``
    colorspace by varying degrees.

    """

    # Added in 0.4.0.
    def __init__(self, mul=1, from_colorspace=CSPACE_RGB,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        mul = iap.Subtract(
            1.0,
            iap.handle_continuous_param(mul, "mul",
                                        value_range=(0.0, 1.0),
                                        tuple_to_uniform=True,
                                        list_to_choice=True),
            elementwise=True
        )
        super(RemoveSaturation, self).__init__(
            mul, from_colorspace=from_colorspace,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


# TODO removed deterministic and random_state here as parameters, because this
# function creates multiple child augmenters. not sure if this is sensible
# (give them all the same random state instead?)
# TODO this is for now deactivated, because HSV images returned by opencv have
#      value range 0-180 for the hue channel
#      and are supposed to be angular representations, i.e. if values go below
#      0 or above 180 they are supposed to overflow
#      to 180 and 0
# pylint: disable=pointless-string-statement
"""
def AddToHueAndSaturation(value=0, per_channel=False, from_colorspace="RGB",
                          channels=[0, 1], name=None):  # pylint: disable=locally-disabled, dangerous-default-value, line-too-long
    ""
    Augmenter that transforms images into HSV space, selects the H and S
    channels and then adds a given range of values to these.

    Parameters
    ----------
    value : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.arithmetic.Add.__init__()`.

    per_channel : bool or float, optional
        See :func:`~imgaug.augmenters.arithmetic.Add.__init__()`.

    from_colorspace : str, optional
        See :func:`~imgaug.augmenters.color.change_colorspace_`.

    channels : int or list of int or None, optional
        See :func:`~imgaug.augmenters.meta.WithChannels.__init__()`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = AddToHueAndSaturation((-20, 20), per_channel=True)

    Adds random values between -20 and 20 to the hue and saturation
    (independently per channel and the same value for all pixels within
    that channel).

    ""
    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return WithColorspace(
        to_colorspace="HSV",
        from_colorspace=from_colorspace,
        children=meta.WithChannels(
            channels=channels,
            children=arithmetic.Add(value=value, per_channel=per_channel)
        ),
        name=name
    )
"""
# pylint: enable=pointless-string-statement


class AddToHueAndSaturation(meta.Augmenter):
    """
    Increases or decreases hue and saturation by random values.

    The augmenter first transforms images to HSV colorspace, then adds random
    values to the H and S channels and afterwards converts back to RGB.

    This augmenter is faster than using ``WithHueAndSaturation`` in combination
    with ``Add``.

    TODO add float support

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.color.change_colorspace_`.

    Parameters
    ----------
    value : None or int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Value to add to the hue *and* saturation of all pixels.
        It is expected to be in the range ``-255`` to ``+255``.

            * If this is ``None``, `value_hue` and/or `value_saturation`
              may be set to values other than ``None``.
            * If an integer, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the discrete
              range ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then a value will be sampled from that
              parameter per image.

    value_hue : None or int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Value to add to the hue of all pixels.
        This is expected to be in the range ``-255`` to ``+255`` and will
        automatically be projected to an angular representation using
        ``(hue/255) * (360/2)`` (OpenCV's hue representation is in the
        range ``[0, 180]`` instead of ``[0, 360]``).
        Only this or `value` may be set, not both.

            * If this and `value_saturation` are both ``None``, `value` may
              be set to a non-``None`` value.
            * If an integer, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the discrete
              range ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then a value will be sampled from that
              parameter per image.

    value_saturation : None or int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Value to add to the saturation of all pixels.
        It is expected to be in the range ``-255`` to ``+255``.
        Only this or `value` may be set, not both.

            * If this and `value_hue` are both ``None``, `value` may
              be set to a non-``None`` value.
            * If an integer, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the discrete
              range ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then a value will be sampled from that
              parameter per image.

    per_channel : bool or float, optional
        Whether to sample per image only one value from `value` and use it for
        both hue and saturation (``False``) or to sample independently one
        value for hue and one for saturation (``True``).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``, otherwise as ``False``.

        This parameter has no effect is `value_hue` and/or `value_saturation`
        are used instead of `value`.

    from_colorspace : str, optional
        See :func:`~imgaug.augmenters.color.change_colorspace_`.

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
    >>> aug = iaa.AddToHueAndSaturation((-50, 50), per_channel=True)

    Add random values between ``-50`` and ``50`` to the hue and saturation
    (independently per channel and the same value for all pixels within
    that channel).

    """

    _LUT_CACHE = None

    def __init__(self, value=None, value_hue=None, value_saturation=None,
                 per_channel=False, from_colorspace="RGB",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(AddToHueAndSaturation, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        if value is None and value_hue is None and value_saturation is None:
            value_hue = (-40, 40)
            value_saturation = (-40, 40)

        self.value = self._handle_value_arg(value, value_hue, value_saturation)
        self.value_hue = self._handle_value_hue_arg(value_hue)
        self.value_saturation = self._handle_value_saturation_arg(
            value_saturation)
        self.per_channel = iap.handle_probability_param(per_channel,
                                                        "per_channel")
        self.from_colorspace = from_colorspace
        self.backend = "cv2"

        # precompute tables for cv2.LUT
        if self.backend == "cv2" and AddToHueAndSaturation._LUT_CACHE is None:
            AddToHueAndSaturation._LUT_CACHE = self._generate_lut_table()

    def _draw_samples(self, augmentables, random_state):
        nb_images = len(augmentables)
        rss = random_state.duplicate(2)

        if self.value is not None:
            per_channel = self.per_channel.draw_samples(
                (nb_images,), random_state=rss[0])
            per_channel = (per_channel > 0.5)

            samples = self.value.draw_samples(
                (nb_images, 2), random_state=rss[1]).astype(np.int32)
            assert -255 <= samples[0, 0] <= 255, (
                "Expected values sampled from `value` in "
                "AddToHueAndSaturation to be in range [-255, 255], "
                "but got %.8f." % (samples[0, 0]))

            samples_hue = samples[:, 0]
            samples_saturation = np.copy(samples[:, 0])
            samples_saturation[per_channel] = samples[per_channel, 1]
        else:
            if self.value_hue is not None:
                samples_hue = self.value_hue.draw_samples(
                    (nb_images,), random_state=rss[0]).astype(np.int32)
            else:
                samples_hue = np.zeros((nb_images,), dtype=np.int32)

            if self.value_saturation is not None:
                samples_saturation = self.value_saturation.draw_samples(
                    (nb_images,), random_state=rss[1]).astype(np.int32)
            else:
                samples_saturation = np.zeros((nb_images,), dtype=np.int32)

        # project hue to angular representation
        # OpenCV uses range [0, 180] for the hue
        samples_hue = (
            (samples_hue.astype(np.float32) / 255.0) * (360/2)
        ).astype(np.int32)

        return samples_hue, samples_saturation

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images
        input_dtypes = iadt.copy_dtypes_for_restore(images, force_list=True)

        # surprisingly, placing this here seems to be slightly slower than
        # placing it inside the loop
        # if isinstance(images_hsv, list):
        #    images_hsv = [img.astype(np.int32) for img in images_hsv]
        # else:
        #    images_hsv = images_hsv.astype(np.int32)

        images_hsv = change_colorspaces_(
            images, CSPACE_HSV, self.from_colorspace)
        samples = self._draw_samples(images, random_state)
        hues = samples[0]
        saturations = samples[1]

        # this is needed if no cache for LUT is used:
        # value_range = np.arange(0, 256, dtype=np.int16)

        gen = enumerate(zip(images_hsv, hues, saturations))
        for i, (image_hsv, hue_i, saturation_i) in gen:
            if image_hsv.size == 0:
                continue

            if self.backend == "cv2":
                image_hsv = self._transform_image_cv2(
                    image_hsv, hue_i, saturation_i)
            else:
                image_hsv = self._transform_image_numpy(
                    image_hsv, hue_i, saturation_i)

            image_hsv = image_hsv.astype(input_dtypes[i])
            image_rgb = change_colorspace_(
                image_hsv,
                to_colorspace=self.from_colorspace,
                from_colorspace=CSPACE_HSV)
            batch.images[i] = image_rgb

        return batch

    @classmethod
    def _transform_image_cv2(cls, image_hsv, hue, saturation):
        # this has roughly the same speed as the numpy backend
        # for 64x64 and is about 25% faster for 224x224

        # code without using cache:
        # table_hue = np.mod(value_range + sample_hue, 180)
        # table_saturation = np.clip(value_range + sample_saturation, 0, 255)

        # table_hue = table_hue.astype(np.uint8, copy=False)
        # table_saturation = table_saturation.astype(np.uint8, copy=False)

        # image_hsv[..., 0] = cv2.LUT(image_hsv[..., 0], table_hue)
        # image_hsv[..., 1] = cv2.LUT(image_hsv[..., 1], table_saturation)

        # code with using cache (at best maybe 10% faster for 64x64):
        table_hue = cls._LUT_CACHE[0]
        table_saturation = cls._LUT_CACHE[1]
        tables = [
            table_hue[255+int(hue)],
            table_saturation[255+int(saturation)]
        ]

        image_hsv[..., [0, 1]] = ia.apply_lut(image_hsv[..., [0, 1]],
                                              tables)

        return image_hsv

    @classmethod
    def _transform_image_numpy(cls, image_hsv, hue, saturation):
        # int16 seems to be slightly faster than int32
        image_hsv = image_hsv.astype(np.int16)
        # np.mod() works also as required here for negative values
        image_hsv[..., 0] = np.mod(image_hsv[..., 0] + hue, 180)
        image_hsv[..., 1] = np.clip(
            image_hsv[..., 1] + saturation, 0, 255)
        return image_hsv

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.value, self.value_hue, self.value_saturation,
                self.per_channel, self.from_colorspace]

    @classmethod
    def _handle_value_arg(cls, value, value_hue, value_saturation):
        if value is not None:
            assert value_hue is None, (
                "`value_hue` may not be set if `value` is set. "
                "It is set to: %s (type: %s)." % (
                    str(value_hue), type(value_hue)))
            assert value_saturation is None, (
                "`value_saturation` may not be set if `value` is set. "
                "It is set to: %s (type: %s)." % (
                    str(value_saturation), type(value_saturation)))
            return iap.handle_discrete_param(
                value, "value", value_range=(-255, 255), tuple_to_uniform=True,
                list_to_choice=True, allow_floats=False)

        return None

    @classmethod
    def _handle_value_hue_arg(cls, value_hue):
        if value_hue is not None:
            # we don't have to verify here that value is None, as the
            # exclusivity was already ensured in _handle_value_arg()
            return iap.handle_discrete_param(
                value_hue, "value_hue", value_range=(-255, 255),
                tuple_to_uniform=True, list_to_choice=True, allow_floats=False)

        return None

    @classmethod
    def _handle_value_saturation_arg(cls, value_saturation):
        if value_saturation is not None:
            # we don't have to verify here that value is None, as the
            # exclusivity was already ensured in _handle_value_arg()
            return iap.handle_discrete_param(
                value_saturation, "value_saturation", value_range=(-255, 255),
                tuple_to_uniform=True, list_to_choice=True, allow_floats=False)
        return None

    @classmethod
    def _generate_lut_table(cls):
        # TODO Changing the dtype here to int8 makes gen test for this method
        #      fail, but all other tests still succeed. How can this be?
        #      The dtype was verified to remain int8, having min & max at
        #      -128 & 127.
        dtype = np.uint8
        table = (np.zeros((256*2, 256), dtype=dtype),
                 np.zeros((256*2, 256), dtype=dtype))
        value_range = np.arange(0, 256, dtype=np.int16)
        # this could be done slightly faster by vectorizing the loop
        for i in sm.xrange(-255, 255+1):
            table_hue = np.mod(value_range + i, 180)
            table_saturation = np.clip(value_range + i, 0, 255)
            table[0][255+i, :] = table_hue
            table[1][255+i, :] = table_saturation
        return table


class AddToHue(AddToHueAndSaturation):
    """
    Add random values to the hue of images.

    The augmenter first transforms images to HSV colorspace, then adds random
    values to the H channel and afterwards converts back to RGB.

    If you want to change both the hue and the saturation, it is recommended
    to use ``AddToHueAndSaturation`` as otherwise the image will be
    converted twice to HSV and back to RGB.

    This augmenter is a shortcut for ``AddToHueAndSaturation(value_hue=...)``.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.color.AddToHueAndSaturation`.

    Parameters
    ----------
    value : None or int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Value to add to the hue of all pixels.
        This is expected to be in the range ``-255`` to ``+255`` and will
        automatically be projected to an angular representation using
        ``(hue/255) * (360/2)`` (OpenCV's hue representation is in the
        range ``[0, 180]`` instead of ``[0, 360]``).

            * If an integer, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the discrete
              range ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then a value will be sampled from that
              parameter per image.

    from_colorspace : str, optional
        See :func:`~imgaug.augmenters.color.change_colorspace_`.

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
    >>> aug = iaa.AddToHue((-50, 50))

    Sample random values from the discrete uniform range ``[-50..50]``,
    convert them to angular representation and add them to the hue, i.e.
    to the ``H`` channel in ``HSV`` colorspace.

    """

    def __init__(self, value=(-255, 255), from_colorspace=CSPACE_RGB,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(AddToHue, self).__init__(
            value_hue=value,
            from_colorspace=from_colorspace,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class AddToSaturation(AddToHueAndSaturation):
    """
    Add random values to the saturation of images.

    The augmenter first transforms images to HSV colorspace, then adds random
    values to the S channel and afterwards converts back to RGB.

    If you want to change both the hue and the saturation, it is recommended
    to use ``AddToHueAndSaturation`` as otherwise the image will be
    converted twice to HSV and back to RGB.

    This augmenter is a shortcut for
    ``AddToHueAndSaturation(value_saturation=...)``.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.color.AddToHueAndSaturation`.

    Parameters
    ----------
    value : None or int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Value to add to the saturation of all pixels.
        It is expected to be in the range ``-255`` to ``+255``.

            * If an integer, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the discrete
              range ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then a value will be sampled from that
              parameter per image.

    from_colorspace : str, optional
        See :func:`~imgaug.augmenters.color.change_colorspace_`.

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
    >>> aug = iaa.AddToSaturation((-50, 50))

    Sample random values from the discrete uniform range ``[-50..50]``,
    and add them to the saturation, i.e. to the ``S`` channel in ``HSV``
    colorspace.

    """

    def __init__(self, value=(-75, 75), from_colorspace="RGB",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(AddToSaturation, self).__init__(
            value_saturation=value,
            from_colorspace=from_colorspace,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


# TODO tests
# TODO rename to ChangeColorspace3D and then introduce ChangeColorspace, which
#      does not enforce 3d images?
class ChangeColorspace(meta.Augmenter):
    """
    Augmenter to change the colorspace of images.

    .. note::

        This augmenter is not tested. Some colorspaces might work, others
        might not.

    ..note::

        This augmenter tries to project the colorspace value range on
        0-255. It outputs dtype=uint8 images.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.color.change_colorspace_`.

    Parameters
    ----------
    to_colorspace : str or list of str or imgaug.parameters.StochasticParameter
        The target colorspace.
        Allowed strings are: ``RGB``, ``BGR``, ``GRAY``, ``CIE``, ``YCrCb``,
        ``HSV``, ``HLS``, ``Lab``, ``Luv``.
        These are also accessible via
        ``imgaug.augmenters.color.CSPACE_<NAME>``,
        e.g. ``imgaug.augmenters.CSPACE_YCrCb``.

            * If a string, it must be among the allowed colorspaces.
            * If a list, it is expected to be a list of strings, each one
              being an allowed colorspace. A random element from the list
              will be chosen per image.
            * If a StochasticParameter, it is expected to return string. A new
              sample will be drawn per image.

    from_colorspace : str, optional
        The source colorspace (of the input images).
        See `to_colorspace`. Only a single string is allowed.

    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        The alpha value of the new colorspace when overlayed over the
        old one. A value close to 1.0 means that mostly the new
        colorspace is visible. A value close to 0.0 means, that mostly the
        old image is visible.

            * If an int or float, exactly that value will be used.
            * If a tuple ``(a, b)``, a random value from the range
              ``a <= x <= b`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, a value will be sampled from the
              parameter per image.

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

    """

    # TODO mark these as deprecated
    RGB = CSPACE_RGB
    BGR = CSPACE_BGR
    GRAY = CSPACE_GRAY
    CIE = CSPACE_CIE
    YCrCb = CSPACE_YCrCb
    HSV = CSPACE_HSV
    HLS = CSPACE_HLS
    Lab = CSPACE_Lab
    Luv = CSPACE_Luv
    COLORSPACES = {RGB, BGR, GRAY, CIE, YCrCb, HSV, HLS, Lab, Luv}
    # TODO access cv2 COLOR_ variables directly instead of indirectly via
    #      dictionary mapping
    CV_VARS = {
        # RGB
        "RGB2BGR": cv2.COLOR_RGB2BGR,
        "RGB2GRAY": cv2.COLOR_RGB2GRAY,
        "RGB2CIE": cv2.COLOR_RGB2XYZ,
        "RGB2YCrCb": cv2.COLOR_RGB2YCR_CB,
        "RGB2HSV": cv2.COLOR_RGB2HSV,
        "RGB2HLS": cv2.COLOR_RGB2HLS,
        "RGB2Lab": cv2.COLOR_RGB2LAB,
        "RGB2Luv": cv2.COLOR_RGB2LUV,
        # BGR
        "BGR2RGB": cv2.COLOR_BGR2RGB,
        "BGR2GRAY": cv2.COLOR_BGR2GRAY,
        "BGR2CIE": cv2.COLOR_BGR2XYZ,
        "BGR2YCrCb": cv2.COLOR_BGR2YCR_CB,
        "BGR2HSV": cv2.COLOR_BGR2HSV,
        "BGR2HLS": cv2.COLOR_BGR2HLS,
        "BGR2Lab": cv2.COLOR_BGR2LAB,
        "BGR2Luv": cv2.COLOR_BGR2LUV,
        # HSV
        "HSV2RGB": cv2.COLOR_HSV2RGB,
        "HSV2BGR": cv2.COLOR_HSV2BGR,
        # HLS
        "HLS2RGB": cv2.COLOR_HLS2RGB,
        "HLS2BGR": cv2.COLOR_HLS2BGR,
        # Lab
        "Lab2RGB": (
            cv2.COLOR_Lab2RGB
            if hasattr(cv2, "COLOR_Lab2RGB") else cv2.COLOR_LAB2RGB),
        "Lab2BGR": (
            cv2.COLOR_Lab2BGR
            if hasattr(cv2, "COLOR_Lab2BGR") else cv2.COLOR_LAB2BGR)
    }

    def __init__(self, to_colorspace, from_colorspace=CSPACE_RGB, alpha=1.0,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(ChangeColorspace, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        # TODO somehow merge this with Alpha augmenter?
        self.alpha = iap.handle_continuous_param(
            alpha, "alpha", value_range=(0, 1.0), tuple_to_uniform=True,
            list_to_choice=True)

        if ia.is_string(to_colorspace):
            assert to_colorspace in CSPACE_ALL, (
                "Expected 'to_colorspace' to be one of %s. Got %s." % (
                    CSPACE_ALL, to_colorspace))
            self.to_colorspace = iap.Deterministic(to_colorspace)
        elif ia.is_iterable(to_colorspace):
            all_strings = all(
                [ia.is_string(colorspace) for colorspace in to_colorspace])
            assert all_strings, (
                "Expected list of 'to_colorspace' to only contain strings. "
                "Got types %s." % (
                    ", ".join([str(type(v)) for v in to_colorspace])))
            all_valid = all(
                [(colorspace in CSPACE_ALL)
                 for colorspace in to_colorspace])
            assert all_valid, (
                "Expected list of 'to_colorspace' to only contain strings "
                "that are in %s. Got strings %s." % (
                    CSPACE_ALL, to_colorspace))
            self.to_colorspace = iap.Choice(to_colorspace)
        elif isinstance(to_colorspace, iap.StochasticParameter):
            self.to_colorspace = to_colorspace
        else:
            raise Exception("Expected to_colorspace to be string, list of "
                            "strings or StochasticParameter, got %s." % (
                                type(to_colorspace),))

        assert ia.is_string(from_colorspace), (
            "Expected from_colorspace to be a single string, "
            "got type %s." % (type(from_colorspace),))
        assert from_colorspace in CSPACE_ALL, (
            "Expected from_colorspace to be one of: %s. Got: %s." % (
                ", ".join(CSPACE_ALL), from_colorspace))
        assert from_colorspace != CSPACE_GRAY, (
            "Cannot convert from grayscale images to other colorspaces.")
        self.from_colorspace = from_colorspace

        # epsilon value to check if alpha is close to 1.0 or 0.0
        self.eps = 0.001

    def _draw_samples(self, n_augmentables, random_state):
        rss = random_state.duplicate(2)
        alphas = self.alpha.draw_samples(
            (n_augmentables,), random_state=rss[0])
        to_colorspaces = self.to_colorspace.draw_samples(
            (n_augmentables,), random_state=rss[1])
        return alphas, to_colorspaces

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images
        nb_images = len(images)
        alphas, to_colorspaces = self._draw_samples(nb_images, random_state)
        for i, image in enumerate(images):
            alpha = alphas[i]
            to_colorspace = to_colorspaces[i]

            assert to_colorspace in CSPACE_ALL, (
                "Expected 'to_colorspace' to be one of %s. Got %s." % (
                    CSPACE_ALL, to_colorspace))

            if alpha <= self.eps or self.from_colorspace == to_colorspace:
                pass  # no change necessary
            else:
                image_aug = change_colorspace_(image, to_colorspace,
                                               self.from_colorspace)
                batch.images[i] = blend.blend_alpha(image_aug, image, alpha,
                                                    self.eps)

        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.to_colorspace, self.alpha]


# TODO This should rather do the blending in RGB or BGR space.
#      Currently, if the input image is in e.g. HSV space, it will blend in
#      that space.
# TODO rename to Grayscale3D and add Grayscale that keeps the image at 1D?
class Grayscale(ChangeColorspace):
    """Augmenter to convert images to their grayscale versions.

    .. note::

        Number of output channels is still ``3``, i.e. this augmenter just
        "removes" color.

    TODO check dtype support

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.color.change_colorspace_`.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        The alpha value of the grayscale image when overlayed over the
        old image. A value close to 1.0 means, that mostly the new grayscale
        image is visible. A value close to 0.0 means, that mostly the
        old image is visible.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value from the range
              ``a <= x <= b`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, a value will be sampled from the
              parameter per image.

    from_colorspace : str, optional
        The source colorspace (of the input images).
        See :func:`~imgaug.augmenters.color.change_colorspace_`.

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
    >>> aug = iaa.Grayscale(alpha=1.0)

    Creates an augmenter that turns images to their grayscale versions.

    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Grayscale(alpha=(0.0, 1.0))

    Creates an augmenter that turns images to their grayscale versions with
    an alpha value in the range ``0 <= alpha <= 1``. An alpha value of 0.5 would
    mean, that the output image is 50 percent of the input image and 50
    percent of the grayscale image (i.e. 50 percent of color removed).

    """

    def __init__(self, alpha=1, from_colorspace=CSPACE_RGB,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Grayscale, self).__init__(
            to_colorspace=CSPACE_GRAY,
            alpha=alpha,
            from_colorspace=from_colorspace,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class ChangeColorTemperature(meta.Augmenter):
    """Change the temperature to a provided Kelvin value.

    Low Kelvin values around ``1000`` to ``4000`` will result in red, yellow
    or orange images. Kelvin values around ``10000`` to ``40000`` will result
    in progressively darker blue tones.

    Color temperatures taken from
    `<http://www.vendian.org/mncharity/dir3/blackbody/UnstableURLs/bbr_color.html>`_

    Basic method to change color temperatures taken from
    `<https://stackoverflow.com/a/11888449>`_

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.color.change_color_temperatures_`.

    Parameters
    ----------
    kelvin : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Temperature in Kelvin. The temperatures of images will be modified to
        this value. Must be in the interval ``[1000, 40000]``.

            * If a number, exactly that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the
              interval ``[a, b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
            ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.ChangeColorTemperature((1100, 10000))

    Create an augmenter that changes the color temperature of images to
    a random value between ``1100`` and ``10000`` Kelvin.

    """

    # Added in 0.4.0.
    def __init__(self, kelvin=(1000, 11000), from_colorspace=CSPACE_RGB,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(ChangeColorTemperature, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.kelvin = iap.handle_continuous_param(
            kelvin, "kelvin", value_range=(1000, 40000), tuple_to_uniform=True,
            list_to_choice=True)
        self.from_colorspace = from_colorspace

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is not None:
            nb_rows = batch.nb_rows
            kelvins = self.kelvin.draw_samples((nb_rows,),
                                               random_state=random_state)

            batch.images = change_color_temperatures_(
                batch.images, kelvins, from_colorspaces=self.from_colorspace)

        return batch

    # Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.kelvin, self.from_colorspace]


@six.add_metaclass(ABCMeta)
class _AbstractColorQuantization(meta.Augmenter):
    def __init__(self,
                 counts=(2, 16),  # number of bits or colors
                 counts_value_range=(2, None),
                 from_colorspace=CSPACE_RGB,
                 to_colorspace=[CSPACE_RGB, CSPACE_Lab],
                 max_size=128,
                 interpolation="linear",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        # pylint: disable=dangerous-default-value
        super(_AbstractColorQuantization, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.counts_value_range = counts_value_range
        self.counts = iap.handle_discrete_param(
            counts, "counts", value_range=counts_value_range,
            tuple_to_uniform=True, list_to_choice=True, allow_floats=False)
        self.from_colorspace = from_colorspace
        self.to_colorspace = to_colorspace
        self.max_size = max_size
        self.interpolation = interpolation

    def _draw_samples(self, n_augmentables, random_state):
        counts = self.counts.draw_samples((n_augmentables,), random_state)
        counts = np.round(counts).astype(np.int32)

        # Note that we can get values outside of the value range for counts
        # here if a StochasticParameter was provided, e.g.
        # Deterministic(1) is currently not verified.
        counts = np.clip(counts,
                         self.counts_value_range[0],
                         self.counts_value_range[1])

        return counts

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images
        rss = random_state.duplicate(1 + len(images))
        counts = self._draw_samples(len(images), rss[-1])

        for i, image in enumerate(images):
            batch.images[i] = self._augment_single_image(image, counts[i],
                                                         rss[i])
        return batch

    def _augment_single_image(self, image, counts, random_state):
        # pylint: disable=protected-access, invalid-name
        assert image.shape[-1] in [1, 3, 4], (
            "Expected image with 1, 3 or 4 channels, "
            "got %d (shape: %s)." % (image.shape[-1], image.shape))

        orig_shape = image.shape
        image = self._ensure_max_size(
            image, self.max_size, self.interpolation)

        if image.shape[-1] == 1:
            # 2D image
            image_aug = self._quantize(image, counts)
        else:
            # 3D image with 3 or 4 channels
            alpha_channel = None
            if image.shape[-1] == 4:
                alpha_channel = image[:, :, 3:4]
                image = image[:, :, 0:3]

            if self.to_colorspace is None:
                cs = meta.Identity()
                cs_inv = meta.Identity()
            else:
                # TODO quite hacky to recover the sampled to_colorspace here
                #      by accessing _draw_samples(). Would be better to have
                #      an inverse augmentation method in ChangeColorspace.

                # We use random_state.copy() in this method, but that is not
                # expected to cause unchanged an random_state, because
                # _augment_batch_() uses an un-copied one for _draw_samples()
                cs = ChangeColorspace(
                    from_colorspace=self.from_colorspace,
                    to_colorspace=self.to_colorspace,
                    random_state=random_state.copy(),)
                _, to_colorspaces = cs._draw_samples(
                    1, random_state.copy())
                cs_inv = ChangeColorspace(
                    from_colorspace=to_colorspaces[0],
                    to_colorspace=self.from_colorspace,
                    random_state=random_state.copy())

            image_tf = cs.augment_image(image)
            image_tf_aug = self._quantize(image_tf, counts)
            image_aug = cs_inv.augment_image(image_tf_aug)

            if alpha_channel is not None:
                image_aug = np.concatenate([image_aug, alpha_channel], axis=2)

        if orig_shape != image_aug.shape:
            image_aug = ia.imresize_single_image(
                image_aug,
                orig_shape[0:2],
                interpolation=self.interpolation)

        return image_aug

    @abstractmethod
    def _quantize(self, image, counts):
        """Apply the augmenter-specific quantization function to an image."""

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.counts,
                self.from_colorspace,
                self.to_colorspace,
                self.max_size,
                self.interpolation]

    # TODO this is the same function as in Superpixels._ensure_max_size
    #      make DRY
    @classmethod
    def _ensure_max_size(cls, image, max_size, interpolation):
        if max_size is not None:
            size = max(image.shape[0], image.shape[1])
            if size > max_size:
                resize_factor = max_size / size
                new_height = int(image.shape[0] * resize_factor)
                new_width = int(image.shape[1] * resize_factor)
                image = ia.imresize_single_image(
                    image,
                    (new_height, new_width),
                    interpolation=interpolation)
        return image


class KMeansColorQuantization(_AbstractColorQuantization):
    """
    Quantize colors using k-Means clustering.

    This "collects" the colors from the input image, groups them into
    ``k`` clusters using k-Means clustering and replaces the colors in the
    input image using the cluster centroids.

    This is slower than ``UniformColorQuantization``, but adapts dynamically
    to the color range in the input image.

    .. note::

        This augmenter expects input images to be either grayscale
        or to have 3 or 4 channels and use colorspace `from_colorspace`. If
        images have 4 channels, it is assumed that the 4th channel is an alpha
        channel and it will not be quantized.

    **Supported dtypes**:

    if (image size <= max_size):

        minimum of (
            :class:`~imgaug.augmenters.color.ChangeColorspace`,
            :func:`~imgaug.augmenters.color.quantize_kmeans`
        )

    if (image size > max_size):

        minimum of (
            :class:`~imgaug.augmenters.color.ChangeColorspace`,
            :func:`~imgaug.augmenters.color.quantize_kmeans`,
            :func:`~imgaug.imgaug.imresize_single_image`
        )

    Parameters
    ----------
    n_colors : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Target number of colors in the generated output image.
        This corresponds to the number of clusters in k-Means, i.e. ``k``.
        Sampled values below ``2`` will always be clipped to ``2``.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    to_colorspace : None or str or list of str or imgaug.parameters.StochasticParameter
        The colorspace in which to perform the quantization.
        See :func:`~imgaug.augmenters.color.change_colorspace_` for valid values.
        This will be ignored for grayscale input images.

            * If ``None`` the colorspace of input images will not be changed.
            * If a string, it must be among the allowed colorspaces.
            * If a list, it is expected to be a list of strings, each one
              being an allowed colorspace. A random element from the list
              will be chosen per image.
            * If a StochasticParameter, it is expected to return string. A new
              sample will be drawn per image.

    from_colorspace : str, optional
        The colorspace of the input images.
        See `to_colorspace`. Only a single string is allowed.

    max_size : int or None, optional
        Maximum image size at which to perform the augmentation.
        If the width or height of an image exceeds this value, it will be
        downscaled before running the augmentation so that the longest side
        matches `max_size`.
        This is done to speed up the augmentation. The final output image has
        the same size as the input image. Use ``None`` to apply no downscaling.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is
        exceeded. Valid methods are the same as in
        :func:`~imgaug.imgaug.imresize_single_image`.

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
    >>> aug = iaa.KMeansColorQuantization()

    Create an augmenter to apply k-Means color quantization to images using a
    random amount of colors, sampled uniformly from the interval ``[2..16]``.
    It assumes the input image colorspace to be ``RGB`` and clusters colors
    randomly in ``RGB`` or ``Lab`` colorspace.

    >>> aug = iaa.KMeansColorQuantization(n_colors=8)

    Create an augmenter that quantizes images to (up to) eight colors.

    >>> aug = iaa.KMeansColorQuantization(n_colors=(4, 16))

    Create an augmenter that quantizes images to (up to) ``n`` colors,
    where ``n`` is randomly and uniformly sampled from the discrete interval
    ``[4..16]``.

    >>> aug = iaa.KMeansColorQuantization(
    >>>     from_colorspace=iaa.CSPACE_BGR)

    Create an augmenter that quantizes input images that are in
    ``BGR`` colorspace. The quantization happens in ``RGB`` or ``Lab``
    colorspace, into which the images are temporarily converted.

    >>> aug = iaa.KMeansColorQuantization(
    >>>     to_colorspace=[iaa.CSPACE_RGB, iaa.CSPACE_HSV])

    Create an augmenter that quantizes images by clustering colors randomly
    in either ``RGB`` or ``HSV`` colorspace. The assumed input colorspace
    of images is ``RGB``.

    """

    def __init__(self, n_colors=(2, 16), from_colorspace=CSPACE_RGB,
                 to_colorspace=[CSPACE_RGB, CSPACE_Lab],
                 max_size=128, interpolation="linear",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        # pylint: disable=dangerous-default-value
        super(KMeansColorQuantization, self).__init__(
            counts=n_colors,
            from_colorspace=from_colorspace,
            to_colorspace=to_colorspace,
            max_size=max_size,
            interpolation=interpolation,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

    @property
    def n_colors(self):
        """Alias for property ``counts``.

        Added in 0.4.0.

        """
        return self.counts

    def _quantize(self, image, counts):
        return quantize_kmeans(image, counts)


@ia.deprecated("imgaug.augmenters.colors.quantize_kmeans")
def quantize_colors_kmeans(image, n_colors, n_max_iter=10, eps=1.0):
    """Outdated name of :func:`quantize_kmeans`.

    Deprecated since 0.4.0.

    """
    return quantize_kmeans(arr=image, nb_clusters=n_colors,
                           nb_max_iter=n_max_iter, eps=eps)


def quantize_kmeans(arr, nb_clusters, nb_max_iter=10, eps=1.0):
    """Quantize an array into N bins using k-means clustering.

    If the input is an image, this method returns in an image with a maximum
    of ``N`` colors. Similar colors are grouped to their mean. The k-means
    clustering happens across channels and not channelwise.

    Code similar to https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/
    py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html

    .. warning::

        This function currently changes the RNG state of both OpenCV's
        internal RNG and imgaug's global RNG. This is necessary in order
        to ensure that the k-means clustering happens deterministically.

    Added in 0.4.0. (Previously called ``quantize_colors_kmeans()``.)

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
    arr : ndarray
        Array to quantize. Expected to be of shape ``(H,W)`` or ``(H,W,C)``
        with ``C`` usually being ``1`` or ``3``.

    nb_clusters : int
        Number of clusters to quantize into, i.e. ``k`` in k-means clustering.
        This corresponds to the maximum number of colors in an output image.

    nb_max_iter : int, optional
        Maximum number of iterations that the k-means clustering algorithm
        is run.

    eps : float, optional
        Minimum change of all clusters per k-means iteration. If all clusters
        change by less than this amount in an iteration, the clustering is
        stopped.

    Returns
    -------
    ndarray
        Image with quantized colors.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> import numpy as np
    >>> image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape((4, 4, 3))
    >>> image_quantized = iaa.quantize_kmeans(image, 6)

    Generates a ``4x4`` image with ``3`` channels, containing consecutive
    values from ``0`` to ``4*4*3``, leading to an equal number of colors.
    These colors are then quantized so that only ``6`` are remaining. Note
    that the six remaining colors do have to appear in the input image.

    """
    assert arr.ndim in [2, 3], (
        "Expected two- or three-dimensional array shape, "
        "got shape %s." % (arr.shape,))
    assert arr.dtype.name == "uint8", "Expected uint8 array, got %s." % (
        arr.dtype.name,)
    assert 2 <= nb_clusters <= 256, (
        "Expected nb_clusters to be in the discrete interval [2..256]. "
        "Got a value of %d instead." % (nb_clusters,))

    # without this check, kmeans throws an exception
    n_pixels = np.prod(arr.shape[0:2])
    if nb_clusters >= n_pixels:
        return np.copy(arr)

    nb_channels = 1 if arr.ndim == 2 else arr.shape[-1]
    pixel_vectors = arr.reshape((-1, nb_channels)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                nb_max_iter, eps)
    attempts = 1

    # We want our quantization function to be deterministic (so that the
    # augmenter using it can also be executed deterministically). Hence we
    # set the RGN seed here.
    # This is fairly ugly, but in cv2 there seems to be no other way to
    # achieve determinism. Using cv2.KMEANS_PP_CENTERS does not help, as it
    # is non-deterministic (tested). In C++ the function has an rgn argument,
    # but not in python. In python there also seems to be no way to read out
    # cv2's RNG state, so we can't set it back after executing this function.
    # TODO this is quite hacky
    cv2.setRNGSeed(1)
    _compactness, labels, centers = cv2.kmeans(
        pixel_vectors, nb_clusters, None, criteria, attempts,
        cv2.KMEANS_RANDOM_CENTERS)
    # TODO replace by sample_seed function
    # cv2 seems to be able to handle SEED_MAX_VALUE (tested) but not floats
    cv2.setRNGSeed(iarandom.get_global_rng().generate_seed_())

    # Convert back to uint8 (or whatever the image dtype was) and to input
    # image shape
    centers_uint8 = np.array(centers, dtype=arr.dtype)
    quantized_flat = centers_uint8[labels.flatten()]
    return quantized_flat.reshape(arr.shape)


class UniformColorQuantization(_AbstractColorQuantization):
    """Quantize colors into N bins with regular distance.

    For ``uint8`` images the equation is ``floor(v/q)*q + q/2`` with
    ``q = 256/N``, where ``v`` is a pixel intensity value and ``N`` is
    the target number of colors after quantization.

    This augmenter is faster than ``KMeansColorQuantization``, but the
    set of possible output colors is constant (i.e. independent of the
    input images). It may produce unsatisfying outputs for input images
    that are made up of very similar colors.

    .. note::

        This augmenter expects input images to be either grayscale
        or to have 3 or 4 channels and use colorspace `from_colorspace`. If
        images have 4 channels, it is assumed that the 4th channel is an alpha
        channel and it will not be quantized.

    **Supported dtypes**:

    if (image size <= max_size):

        minimum of (
            :class:`~imgaug.augmenters.color.ChangeColorspace`,
            :func:`~imgaug.augmenters.color.quantize_uniform_`
        )

    if (image size > max_size):

        minimum of (
            :class:`~imgaug.augmenters.color.ChangeColorspace`,
            :func:`~imgaug.augmenters.color.quantize_uniform_`,
            :func:`~imgaug.imgaug.imresize_single_image`
        )

    Parameters
    ----------
    n_colors : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Target number of colors to use in the generated output image.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    to_colorspace : None or str or list of str or imgaug.parameters.StochasticParameter
        The colorspace in which to perform the quantization.
        See :func:`~imgaug.augmenters.color.change_colorspace_` for valid values.
        This will be ignored for grayscale input images.

            * If ``None`` the colorspace of input images will not be changed.
            * If a string, it must be among the allowed colorspaces.
            * If a list, it is expected to be a list of strings, each one
              being an allowed colorspace. A random element from the list
              will be chosen per image.
            * If a StochasticParameter, it is expected to return string. A new
              sample will be drawn per image.

    from_colorspace : str, optional
        The colorspace of the input images.
        See `to_colorspace`. Only a single string is allowed.

    max_size : None or int, optional
        Maximum image size at which to perform the augmentation.
        If the width or height of an image exceeds this value, it will be
        downscaled before running the augmentation so that the longest side
        matches `max_size`.
        This is done to speed up the augmentation. The final output image has
        the same size as the input image. Use ``None`` to apply no downscaling.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is
        exceeded. Valid methods are the same as in
        :func:`~imgaug.imgaug.imresize_single_image`.

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
    >>> aug = iaa.UniformColorQuantization()

    Create an augmenter to apply uniform color quantization to images using a
    random amount of colors, sampled uniformly from the discrete interval
    ``[2..16]``.

    >>> aug = iaa.UniformColorQuantization(n_colors=8)

    Create an augmenter that quantizes images to (up to) eight colors.

    >>> aug = iaa.UniformColorQuantization(n_colors=(4, 16))

    Create an augmenter that quantizes images to (up to) ``n`` colors,
    where ``n`` is randomly and uniformly sampled from the discrete interval
    ``[4..16]``.

    >>> aug = iaa.UniformColorQuantization(
    >>>     from_colorspace=iaa.CSPACE_BGR,
    >>>     to_colorspace=[iaa.CSPACE_RGB, iaa.CSPACE_HSV])

    Create an augmenter that uniformly quantizes images in either ``RGB``
    or ``HSV`` colorspace (randomly picked per image). The input colorspace
    of all images has to be ``BGR``.

    """

    def __init__(self,
                 n_colors=(2, 16),
                 from_colorspace=CSPACE_RGB,
                 to_colorspace=None,
                 max_size=None,
                 interpolation="linear",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        # pylint: disable=dangerous-default-value
        super(UniformColorQuantization, self).__init__(
            counts=n_colors,
            from_colorspace=from_colorspace,
            to_colorspace=to_colorspace,
            max_size=max_size,
            interpolation=interpolation,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

    @property
    def n_colors(self):
        """Alias for property ``counts``.

        Added in 0.4.0.

        """
        return self.counts

    def _quantize(self, image, counts):
        return quantize_uniform_(image, counts)


class UniformColorQuantizationToNBits(_AbstractColorQuantization):
    """Quantize images by setting ``8-B`` bits of each component to zero.

    This augmenter sets the ``8-B`` highest frequency (rightmost) bits of
    each array component to zero. For ``B`` bits this is equivalent to
    changing each component's intensity value ``v`` to
    ``v' = v & (2**(8-B) - 1)``, e.g. for ``B=3`` this results in
    ``v' = c & ~(2**(3-1) - 1) = c & ~3 = c & ~0000 0011 = c & 1111 1100``.

    This augmenter behaves for ``B`` similarly to
    ``UniformColorQuantization(2**B)``, but quantizes each bin with interval
    ``(a, b)`` to ``a`` instead of to ``a + (b-a)/2``.

    This augmenter is comparable to :func:`PIL.ImageOps.posterize`.

    .. note::

        This augmenter expects input images to be either grayscale
        or to have 3 or 4 channels and use colorspace `from_colorspace`. If
        images have 4 channels, it is assumed that the 4th channel is an alpha
        channel and it will not be quantized.

    Added in 0.4.0.

    **Supported dtypes**:

    if (image size <= max_size):

        minimum of (
            :class:`~imgaug.augmenters.color.ChangeColorspace`,
            :func:`~imgaug.augmenters.color.quantize_uniform`
        )

    if (image size > max_size):

        minimum of (
            :class:`~imgaug.augmenters.color.ChangeColorspace`,
            :func:`~imgaug.augmenters.color.quantize_uniform`,
            :func:`~imgaug.imgaug.imresize_single_image`
        )

    Parameters
    ----------
    nb_bits : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of bits to keep in each image's array component.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    to_colorspace : None or str or list of str or imgaug.parameters.StochasticParameter
        The colorspace in which to perform the quantization.
        See :func:`~imgaug.augmenters.color.change_colorspace_` for valid values.
        This will be ignored for grayscale input images.

            * If ``None`` the colorspace of input images will not be changed.
            * If a string, it must be among the allowed colorspaces.
            * If a list, it is expected to be a list of strings, each one
              being an allowed colorspace. A random element from the list
              will be chosen per image.
            * If a StochasticParameter, it is expected to return string. A new
              sample will be drawn per image.

    from_colorspace : str, optional
        The colorspace of the input images.
        See `to_colorspace`. Only a single string is allowed.

    max_size : None or int, optional
        Maximum image size at which to perform the augmentation.
        If the width or height of an image exceeds this value, it will be
        downscaled before running the augmentation so that the longest side
        matches `max_size`.
        This is done to speed up the augmentation. The final output image has
        the same size as the input image. Use ``None`` to apply no downscaling.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is
        exceeded. Valid methods are the same as in
        :func:`~imgaug.imgaug.imresize_single_image`.

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
    >>> aug = iaa.UniformColorQuantizationToNBits()

    Create an augmenter to apply uniform color quantization to images using a
    random amount of bits to remove, sampled uniformly from the discrete
    interval ``[1..8]``.

    >>> aug = iaa.UniformColorQuantizationToNBits(nb_bits=(2, 8))

    Create an augmenter that quantizes images by removing ``8-B`` rightmost
    bits from each component, where ``B`` is uniformly sampled from the
    discrete interval ``[2..8]``.

    >>> aug = iaa.UniformColorQuantizationToNBits(
    >>>     from_colorspace=iaa.CSPACE_BGR,
    >>>     to_colorspace=[iaa.CSPACE_RGB, iaa.CSPACE_HSV])

    Create an augmenter that uniformly quantizes images in either ``RGB``
    or ``HSV`` colorspace (randomly picked per image). The input colorspace
    of all images has to be ``BGR``.

    """

    # Added in 0.4.0.
    def __init__(self,
                 nb_bits=(1, 8),
                 from_colorspace=CSPACE_RGB,
                 to_colorspace=None,
                 max_size=None,
                 interpolation="linear",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        # pylint: disable=dangerous-default-value

        # wrt value range: for discrete params, (1, 8) results in
        # DiscreteUniform with interval [1, 8]
        super(UniformColorQuantizationToNBits, self).__init__(
            counts=nb_bits,
            counts_value_range=(1, 8),
            from_colorspace=from_colorspace,
            to_colorspace=to_colorspace,
            max_size=max_size,
            interpolation=interpolation,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

    # Added in 0.4.0.
    def _quantize(self, image, counts):
        return quantize_uniform_to_n_bits_(image, counts)


class Posterize(UniformColorQuantizationToNBits):
    """Alias for :class:`UniformColorQuantizationToNBits`.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.color.UniformColorQuantizationToNBits`.

    """


@ia.deprecated("imgaug.augmenters.colors.quantize_uniform")
def quantize_colors_uniform(image, n_colors):
    """Outdated name for :func:`quantize_uniform`.

    Deprecated since 0.4.0.

    """
    return quantize_uniform(arr=image, nb_bins=n_colors)


def quantize_uniform(arr, nb_bins, to_bin_centers=True):
    """Quantize an array into N equally-sized bins.

    See :func:`quantize_uniform_` for details.

    Added in 0.4.0. (Previously called ``quantize_colors_uniform()``.)

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.color.quantize_uniform_`.

    Parameters
    ----------
    arr : ndarray
        See :func:`quantize_uniform_`.

    nb_bins : int
        See :func:`quantize_uniform_`.

    to_bin_centers : bool
        See :func:`quantize_uniform_`.

    Returns
    -------
    ndarray
        Array with quantized components.

    """
    return quantize_uniform_(np.copy(arr),
                             nb_bins=nb_bins,
                             to_bin_centers=to_bin_centers)


def quantize_uniform_(arr, nb_bins, to_bin_centers=True):
    """Quantize an array into N equally-sized bins in-place.

    This can be used to quantize/posterize an image into N colors.

    For ``uint8`` arrays the equation is ``floor(v/q)*q + q/2`` with
    ``q = 256/N``, where ``v`` is a pixel intensity value and ``N`` is
    the target number of bins (roughly matches number of colors) after
    quantization.

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
    arr : ndarray
        Array to quantize, usually an image. Expected to be of shape ``(H,W)``
        or ``(H,W,C)`` with ``C`` usually being ``1`` or ``3``.
        This array *may* be changed in-place.

    nb_bins : int
        Number of equally-sized bins to quantize into. This corresponds to
        the maximum number of colors in an output image.

    to_bin_centers : bool
        Whether to quantize each bin ``(a, b)`` to ``a + (b-a)/2`` (center
        of bin, ``True``) or to ``a`` (lower boundary, ``False``).

    Returns
    -------
    ndarray
        Array with quantized components. This *may* be the input array with
        components changed in-place.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> import numpy as np
    >>> image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape((4, 4, 3))
    >>> image_quantized = iaa.quantize_uniform_(np.copy(image), 6)

    Generates a ``4x4`` image with ``3`` channels, containing consecutive
    values from ``0`` to ``4*4*3``, leading to an equal number of colors.
    Each component is then quantized into one of ``6`` bins that regularly
    split up the value range of ``[0..255]``, i.e. the resolution w.r.t. to
    the value range is reduced.

    """
    if nb_bins == 256 or 0 in arr.shape:
        return arr

    # TODO remove dtype check here? apply_lut_() does that already
    assert arr.dtype.name == "uint8", "Expected uint8 image, got %s." % (
        arr.dtype.name,)
    assert 2 <= nb_bins <= 256, (
        "Expected nb_bins to be in the discrete interval [2..256]. "
        "Got a value of %d instead." % (nb_bins,))

    table_class = (_QuantizeUniformCenterizedLUTTableSingleton
                   if to_bin_centers
                   else _QuantizeUniformNotCenterizedLUTTableSingleton)
    table = (table_class
             .get_instance()
             .get_for_nb_bins(nb_bins))
    arr = ia.apply_lut_(arr, table)
    return arr


# Added in 0.4.0.
class _QuantizeUniformCenterizedLUTTableSingleton(object):
    _INSTANCE = None

    @classmethod
    def get_instance(cls):
        """Get singleton instance of :class:`_QuantizeUniformLUTTable`.

        Added in 0.4.0.

        Returns
        -------
        _QuantizeUniformLUTTable
            The global instance of :class:`_QuantizeUniformLUTTable`.

        """
        if cls._INSTANCE is None:
            cls._INSTANCE = _QuantizeUniformLUTTable(centerize=True)
        return cls._INSTANCE


# Added in 0.4.0.
class _QuantizeUniformNotCenterizedLUTTableSingleton(object):
    """Table for :func:`quantize_uniform` with ``to_bin_centers=False``."""
    _INSTANCE = None

    @classmethod
    def get_instance(cls):
        """Get singleton instance of :class:`_QuantizeUniformLUTTable`.

        Added in 0.4.0.

        Returns
        -------
        _QuantizeUniformLUTTable
            The global instance of :class:`_QuantizeUniformLUTTable`.

        """
        if cls._INSTANCE is None:
            cls._INSTANCE = _QuantizeUniformLUTTable(centerize=False)
        return cls._INSTANCE


# Added in 0.4.0.
class _QuantizeUniformLUTTable(object):
    def __init__(self, centerize):
        self.table = self._generate_quantize_uniform_table(centerize)

    def get_for_nb_bins(self, nb_bins):
        """Get LUT ndarray for a provided number of bins.

        Added in 0.4.0.

        """
        return self.table[nb_bins, :]

    # Added in 0.4.0.
    @classmethod
    def _generate_quantize_uniform_table(cls, centerize):
        # For simplicity, we generate here the tables for nb_bins=0 (results
        # in all zeros) and nb_bins=256 too, even though these should usually
        # not be requested.
        table = np.arange(0, 256).astype(np.float32)
        table_all_nb_bins = np.zeros((256, 256), dtype=np.float32)

        # This loop could be done a little bit faster by vectorizing it.
        # It is expected to be run exactly once per run of a whole script,
        # making the difference negligible.
        for nb_bins in np.arange(1, 255).astype(np.uint8):
            binsize = 256 / nb_bins
            table_q_f32 = np.floor(table / binsize) * binsize
            if centerize:
                table_q_f32 = table_q_f32 + binsize/2
            table_all_nb_bins[nb_bins] = table_q_f32
        table_all_nb_bins = np.clip(
            np.round(table_all_nb_bins), 0, 255).astype(np.uint8)
        return table_all_nb_bins


def quantize_uniform_to_n_bits(arr, nb_bits):
    """Reduce each component in an array to a maximum number of bits.

    See :func:`quantize_uniform_to_n_bits` for details.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.color.quantize_uniform_to_n_bits_`.

    Parameters
    ----------
    arr : ndarray
        See :func:`quantize_uniform_to_n_bits`.

    nb_bits : int
        See :func:`quantize_uniform_to_n_bits`.

    Returns
    -------
    ndarray
        Array with quantized components.

    """
    return quantize_uniform_to_n_bits_(np.copy(arr), nb_bits=nb_bits)


def quantize_uniform_to_n_bits_(arr, nb_bits):
    """Reduce each component in an array to a maximum number of bits in-place.

    This operation sets the ``8-B`` highest frequency (rightmost) bits to zero.
    For ``B`` bits this is equivalent to changing each component's intensity
    value ``v`` to ``v' = v & (2**(8-B) - 1)``, e.g. for ``B=3`` this results
    in ``v' = c & ~(2**(3-1) - 1) = c & ~3 = c & ~0000 0011 = c & 1111 1100``.

    This is identical to :func:`quantize_uniform` with ``nb_bins=2**nb_bits``
    and ``to_bin_centers=False``.

    This function produces the same outputs as :func:`PIL.ImageOps.posterize`,
    but is significantly faster.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.color.quantize_uniform_`.

    Parameters
    ----------
    arr : ndarray
        Array to quantize, usually an image. Expected to be of shape ``(H,W)``
        or ``(H,W,C)`` with ``C`` usually being ``1`` or ``3``.
        This array *may* be changed in-place.

    nb_bits : int
        Number of bits to keep in each array component.

    Returns
    -------
    ndarray
        Array with quantized components. This *may* be the input array with
        components changed in-place.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> import numpy as np
    >>> image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape((4, 4, 3))
    >>> image_quantized = iaa.quantize_uniform_to_n_bits_(np.copy(image), 6)

    Generates a ``4x4`` image with ``3`` channels, containing consecutive
    values from ``0`` to ``4*4*3``, leading to an equal number of colors.
    These colors are then quantized so that each component's ``8-6=2``
    rightmost bits are set to zero.

    """
    assert 1 <= nb_bits <= 8, (
        "Expected nb_bits to be in the discrete interval [1..8]. "
        "Got a value of %d instead." % (nb_bits,))
    return quantize_uniform_(arr, nb_bins=2**nb_bits, to_bin_centers=False)


def posterize(arr, nb_bits):
    """Alias for :func:`quantize_uniform_to_n_bits`.

    This function is an alias for :func:`quantize_uniform_to_n_bits` and was
    added for users familiar with the same function in PIL.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.color.quantize_uniform_to_n_bits`.

    Parameters
    ----------
    arr : ndarray
        See :func:`quantize_uniform_to_n_bits`.

    nb_bits : int
        See :func:`quantize_uniform_to_n_bits`.

    Returns
    -------
    ndarray
        Array with quantized components.

    """
    return quantize_uniform_to_n_bits(arr, nb_bits)
