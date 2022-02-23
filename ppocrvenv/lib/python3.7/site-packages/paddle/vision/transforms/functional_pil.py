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
from PIL import Image, ImageOps, ImageEnhance

import numpy as np
from numpy import sin, cos, tan
import paddle

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

_pil_interp_from_str = {
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'box': Image.BOX,
    'lanczos': Image.LANCZOS,
    'hamming': Image.HAMMING
}

__all__ = []


def to_tensor(pic, data_format='CHW'):
    """Converts a ``PIL.Image`` to paddle.Tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL.Image): Image to be converted to tensor.
        data_format (str, optional): Data format of output tensor, should be 'HWC' or 
            'CHW'. Default: 'CHW'.

    Returns:
        Tensor: Converted image.

    """

    if data_format not in ['CHW', 'HWC']:
        raise ValueError('data_format should be CHW or HWC. Got {}'.format(
            data_format))

    # PIL Image
    if pic.mode == 'I':
        img = paddle.to_tensor(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        # cast and reshape not support int16
        img = paddle.to_tensor(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'F':
        img = paddle.to_tensor(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * paddle.to_tensor(np.array(pic, np.uint8, copy=False))
    else:
        img = paddle.to_tensor(np.array(pic, copy=False))

    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)

    dtype = paddle.fluid.data_feeder.convert_dtype(img.dtype)
    if dtype == 'uint8':
        img = paddle.cast(img, np.float32) / 255.

    img = img.reshape([pic.size[1], pic.size[0], nchannel])

    if data_format == 'CHW':
        img = img.transpose([2, 0, 1])

    return img


def resize(img, size, interpolation='bilinear'):
    """
    Resizes the image to given size

    Args:
        input (PIL.Image): Image to be resized.
        size (int|list|tuple): Target size of input data, with (height, width) shape.
        interpolation (int|str, optional): Interpolation method. when use pil backend, 
            support method are as following: 
            - "nearest": Image.NEAREST, 
            - "bilinear": Image.BILINEAR, 
            - "bicubic": Image.BICUBIC, 
            - "box": Image.BOX, 
            - "lanczos": Image.LANCZOS, 
            - "hamming": Image.HAMMING

    Returns:
        PIL.Image: Resized image.

    """

    if not (isinstance(size, int) or
            (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), _pil_interp_from_str[interpolation])
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), _pil_interp_from_str[interpolation])
    else:
        return img.resize(size[::-1], _pil_interp_from_str[interpolation])


def pad(img, padding, fill=0, padding_mode='constant'):
    """
    Pads the given PIL.Image on all sides with specified padding mode and fill value.

    Args:
        img (PIL.Image): Image to be padded.
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
        PIL.Image: Padded image.

    """

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

    if padding_mode == 'constant':
        if img.mode == 'P':
            palette = img.getpalette()
            image = ImageOps.expand(img, border=padding, fill=fill)
            image.putpalette(palette)
            return image

        return ImageOps.expand(img, border=padding, fill=fill)
    else:
        if img.mode == 'P':
            palette = img.getpalette()
            img = np.asarray(img)
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)),
                         padding_mode)
            img = Image.fromarray(img)
            img.putpalette(palette)
            return img

        img = np.asarray(img)
        # RGB image
        if len(img.shape) == 3:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right),
                               (0, 0)), padding_mode)
        # Grayscale image
        if len(img.shape) == 2:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)),
                         padding_mode)

        return Image.fromarray(img)


def crop(img, top, left, height, width):
    """Crops the given PIL Image.

    Args:
        img (PIL.Image): Image to be cropped. (0,0) denotes the top left 
            corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        PIL.Image: Cropped image.

    """
    return img.crop((left, top, left + width, top + height))


def center_crop(img, output_size):
    """Crops the given PIL Image and resize it to desired size.

        Args:
            img (PIL.Image): Image to be cropped. (0,0) denotes the top left corner of the image.
            output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions
            backend (str, optional): The image proccess backend type. Options are `pil`, `cv2`. Default: 'pil'. 
        
        Returns:
            PIL.Image: Cropped image.

        """

    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))

    image_width, image_height = img.size
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return crop(img, crop_top, crop_left, crop_height, crop_width)


def hflip(img):
    """Horizontally flips the given PIL Image.

    Args:
        img (PIL.Image): Image to be flipped.

    Returns:
        PIL.Image:  Horizontall flipped image.

    """

    return img.transpose(Image.FLIP_LEFT_RIGHT)


def vflip(img):
    """Vertically flips the given PIL Image.

    Args:
        img (PIL.Image): Image to be flipped.

    Returns:
        PIL.Image:  Vertically flipped image.

    """

    return img.transpose(Image.FLIP_TOP_BOTTOM)


def adjust_brightness(img, brightness_factor):
    """Adjusts brightness of an Image.

    Args:
        img (PIL.Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL.Image: Brightness adjusted image.

    """

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    """Adjusts contrast of an Image.

    Args:
        img (PIL.Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        PIL.Image: Contrast adjusted image.

    """

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    """Adjusts color saturation of an image.

    Args:
        img (PIL.Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL.Image: Saturation adjusted image.

    """

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):
    """Adjusts hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    Args:
        img (PIL.Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL.Image: Hue adjusted image.

    """
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor:{} is not in [-0.5, 0.5].'.format(
            hue_factor))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


def rotate(img,
           angle,
           interpolation="nearest",
           expand=False,
           center=None,
           fill=0):
    """Rotates the image by angle.

    Args:
        img (PIL.Image): Image to be rotated.
        angle (float or int): In degrees degrees counter clockwise order.
        interpolation (str, optional): Interpolation method. If omitted, or if the 
            image has only one channel, it is set to PIL.Image.NEAREST . when use pil backend, 
            support method are as following: 
            - "nearest": Image.NEAREST, 
            - "bilinear": Image.BILINEAR, 
            - "bicubic": Image.BICUBIC
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
        PIL.Image: Rotated image.

    """

    if isinstance(fill, int):
        fill = tuple([fill] * 3)

    return img.rotate(
        angle,
        _pil_interp_from_str[interpolation],
        expand,
        center,
        fillcolor=fill)


def to_grayscale(img, num_output_channels=1):
    """Converts image to grayscale version of image.

    Args:
        img (PIL.Image): Image to be converted to grayscale.
        backend (str, optional): The image proccess backend type. Options are `pil`, 
                    `cv2`. Default: 'pil'. 

    Returns:
        PIL.Image: Grayscale version of the image.
            if num_output_channels = 1 : returned image is single channel

            if num_output_channels = 3 : returned image is 3 channel with r = g = b

    """

    if num_output_channels == 1:
        img = img.convert('L')
    elif num_output_channels == 3:
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img
