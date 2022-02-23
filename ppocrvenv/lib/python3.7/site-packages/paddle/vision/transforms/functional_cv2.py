# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division

import sys
import math
import numbers
import warnings
import collections

import numpy as np
from numpy import sin, cos, tan

import paddle
from paddle.utils import try_import

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

__all__ = []


def to_tensor(pic, data_format='CHW'):
    """Converts a ``numpy.ndarray`` to paddle.Tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (np.ndarray): Image to be converted to tensor.
        data_format (str, optional): Data format of output tensor, should be 'HWC' or 
            'CHW'. Default: 'CHW'.

    Returns:
        Tensor: Converted image.

    """

    if data_format not in ['CHW', 'HWC']:
        raise ValueError('data_format should be CHW or HWC. Got {}'.format(
            data_format))

    if pic.ndim == 2:
        pic = pic[:, :, None]

    if data_format == 'CHW':
        img = paddle.to_tensor(pic.transpose((2, 0, 1)))
    else:
        img = paddle.to_tensor(pic)

    if paddle.fluid.data_feeder.convert_dtype(img.dtype) == 'uint8':
        return paddle.cast(img, np.float32) / 255.
    else:
        return img


def resize(img, size, interpolation='bilinear'):
    """
    Resizes the image to given size

    Args:
        input (np.ndarray): Image to be resized.
        size (int|list|tuple): Target size of input data, with (height, width) shape.
        interpolation (int|str, optional): Interpolation method. when use cv2 backend, 
            support method are as following: 
            - "nearest": cv2.INTER_NEAREST, 
            - "bilinear": cv2.INTER_LINEAR, 
            - "area": cv2.INTER_AREA, 
            - "bicubic": cv2.INTER_CUBIC, 
            - "lanczos": cv2.INTER_LANCZOS4

    Returns:
        np.array: Resized image.

    """
    cv2 = try_import('cv2')
    _cv2_interp_from_str = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'area': cv2.INTER_AREA,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }

    if not (isinstance(size, int) or
            (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    h, w = img.shape[:2]

    if isinstance(size, int):
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            output = cv2.resize(
                img,
                dsize=(ow, oh),
                interpolation=_cv2_interp_from_str[interpolation])
        else:
            oh = size
            ow = int(size * w / h)
            output = cv2.resize(
                img,
                dsize=(ow, oh),
                interpolation=_cv2_interp_from_str[interpolation])
    else:
        output = cv2.resize(
            img,
            dsize=(size[1], size[0]),
            interpolation=_cv2_interp_from_str[interpolation])
    if len(img.shape) == 3 and img.shape[2] == 1:
        return output[:, :, np.newaxis]
    else:
        return output


def pad(img, padding, fill=0, padding_mode='constant'):
    """
    Pads the given numpy.array on all sides with specified padding mode and fill value.

    Args:
        img (np.array): Image to be padded.
        padding (int|list|tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If list/tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a list/tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (float, optional): Pixel fill value for constant fill. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant. Default: 0. 
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default: 'constant'.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value on the edge of the image

            - reflect: pads with reflection of image (without repeating the last value on the edge)

                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image (repeating the last value on the edge)

                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        np.array: Padded image.

    """
    cv2 = try_import('cv2')
    _cv2_pad_from_str = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }

    if not isinstance(padding, (numbers.Number, list, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, list, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')

    if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
        raise ValueError(
            "Padding must be an int or a 2, or 4 element tuple, not a " +
            "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, list):
        padding = tuple(padding)
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    if len(img.shape) == 3 and img.shape[2] == 1:
        return cv2.copyMakeBorder(
            img,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=_cv2_pad_from_str[padding_mode],
            value=fill)[:, :, np.newaxis]
    else:
        return cv2.copyMakeBorder(
            img,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=_cv2_pad_from_str[padding_mode],
            value=fill)


def crop(img, top, left, height, width):
    """Crops the given image.

    Args:
        img (np.array): Image to be cropped. (0,0) denotes the top left 
            corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        np.array: Cropped image.

    """

    return img[top:top + height, left:left + width, :]


def center_crop(img, output_size):
    """Crops the given image and resize it to desired size.

        Args:
            img (np.array): Image to be cropped. (0,0) denotes the top left corner of the image.
            output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions
            backend (str, optional): The image proccess backend type. Options are `pil`, `cv2`. Default: 'pil'. 
        
        Returns:
            np.array: Cropped image.

        """

    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))

    h, w = img.shape[0:2]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)


def hflip(img):
    """Horizontally flips the given image.

    Args:
        img (np.array): Image to be flipped.

    Returns:
        np.array:  Horizontall flipped image.

    """
    cv2 = try_import('cv2')

    return cv2.flip(img, 1)


def vflip(img):
    """Vertically flips the given np.array.

    Args:
        img (np.array): Image to be flipped.

    Returns:
        np.array:  Vertically flipped image.

    """
    cv2 = try_import('cv2')

    if len(img.shape) == 3 and img.shape[2] == 1:
        return cv2.flip(img, 0)[:, :, np.newaxis]
    else:
        return cv2.flip(img, 0)


def adjust_brightness(img, brightness_factor):
    """Adjusts brightness of an image.

    Args:
        img (np.array): Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        np.array: Brightness adjusted image.

    """
    cv2 = try_import('cv2')

    table = np.array([i * brightness_factor
                      for i in range(0, 256)]).clip(0, 255).astype('uint8')

    if len(img.shape) == 3 and img.shape[2] == 1:
        return cv2.LUT(img, table)[:, :, np.newaxis]
    else:
        return cv2.LUT(img, table)


def adjust_contrast(img, contrast_factor):
    """Adjusts contrast of an image.

    Args:
        img (np.array): Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        np.array: Contrast adjusted image.

    """
    cv2 = try_import('cv2')

    table = np.array([(i - 74) * contrast_factor + 74
                      for i in range(0, 256)]).clip(0, 255).astype('uint8')
    if len(img.shape) == 3 and img.shape[2] == 1:
        return cv2.LUT(img, table)[:, :, np.newaxis]
    else:
        return cv2.LUT(img, table)


def adjust_saturation(img, saturation_factor):
    """Adjusts color saturation of an image.

    Args:
        img (np.array): Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        np.array: Saturation adjusted image.

    """
    cv2 = try_import('cv2')

    dtype = img.dtype
    img = img.astype(np.float32)
    alpha = np.random.uniform(
        max(0, 1 - saturation_factor), 1 + saturation_factor)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img[..., np.newaxis]
    img = img * alpha + gray_img * (1 - alpha)
    return img.clip(0, 255).astype(dtype)


def adjust_hue(img, hue_factor):
    """Adjusts hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    Args:
        img (np.array): Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        np.array: Hue adjusted image.

    """
    cv2 = try_import('cv2')

    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor:{} is not in [-0.5, 0.5].'.format(
            hue_factor))

    dtype = img.dtype
    img = img.astype(np.uint8)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    h, s, v = cv2.split(hsv_img)

    alpha = np.random.uniform(hue_factor, hue_factor)
    h = h.astype(np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over="ignore"):
        h += np.uint8(alpha * 255)
    hsv_img = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR_FULL).astype(dtype)


def rotate(img,
           angle,
           interpolation='nearest',
           expand=False,
           center=None,
           fill=0):
    """Rotates the image by angle.

    Args:
        img (np.array): Image to be rotated.
        angle (float or int): In degrees degrees counter clockwise order.
        interpolation (int|str, optional): Interpolation method. If omitted, or if the 
            image has only one channel, it is set to cv2.INTER_NEAREST.
            when use cv2 backend, support method are as following: 
            - "nearest": cv2.INTER_NEAREST, 
            - "bilinear": cv2.INTER_LINEAR, 
            - "bicubic": cv2.INTER_CUBIC
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.

    Returns:
        np.array: Rotated image.

    """
    cv2 = try_import('cv2')
    _cv2_interp_from_str = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'area': cv2.INTER_AREA,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }

    h, w = img.shape[0:2]
    if center is None:
        center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1)

    if expand:

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        # calculate output size
        xx = []
        yy = []

        angle = -math.radians(angle)
        expand_matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        post_trans = (0, 0)
        expand_matrix[2], expand_matrix[5] = transform(
            -center[0] - post_trans[0], -center[1] - post_trans[1],
            expand_matrix)
        expand_matrix[2] += center[0]
        expand_matrix[5] += center[1]

        for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
            x, y = transform(x, y, expand_matrix)
            xx.append(x)
            yy.append(y)
        nw = math.ceil(max(xx)) - math.floor(min(xx))
        nh = math.ceil(max(yy)) - math.floor(min(yy))

        M[0, 2] += (nw - w) * 0.5
        M[1, 2] += (nh - h) * 0.5

        w, h = int(nw), int(nh)

    if len(img.shape) == 3 and img.shape[2] == 1:
        return cv2.warpAffine(
            img,
            M, (w, h),
            flags=_cv2_interp_from_str[interpolation],
            borderValue=fill)[:, :, np.newaxis]
    else:
        return cv2.warpAffine(
            img,
            M, (w, h),
            flags=_cv2_interp_from_str[interpolation],
            borderValue=fill)


def to_grayscale(img, num_output_channels=1):
    """Converts image to grayscale version of image.

    Args:
        img (np.array): Image to be converted to grayscale.

    Returns:
        np.array: Grayscale version of the image.
            if num_output_channels = 1 : returned image is single channel

            if num_output_channels = 3 : returned image is 3 channel with r = g = b

    """
    cv2 = try_import('cv2')

    if num_output_channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    elif num_output_channels == 3:
        # much faster than doing cvtColor to go back to gray
        img = np.broadcast_to(
            cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis], img.shape)
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img


def normalize(img, mean, std, data_format='CHW', to_rgb=False):
    """Normalizes a ndarray imge or image with mean and standard deviation.

    Args:
        img (np.array): input data to be normalized.
        mean (list|tuple): Sequence of means for each channel.
        std (list|tuple): Sequence of standard deviations for each channel.
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.
        to_rgb (bool, optional): Whether to convert to rgb. Default: False.

    Returns:
        np.array: Normalized mage.

    """

    if data_format == 'CHW':
        mean = np.float32(np.array(mean).reshape(-1, 1, 1))
        std = np.float32(np.array(std).reshape(-1, 1, 1))
    else:
        mean = np.float32(np.array(mean).reshape(1, 1, -1))
        std = np.float32(np.array(std).reshape(1, 1, -1))
    if to_rgb:
        # inplace
        img = img[..., ::-1]

    img = (img - mean) / std
    return img
