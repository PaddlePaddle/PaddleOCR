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

import math
import numbers

import paddle
import paddle.nn.functional as F

import sys
import collections

__all__ = []


def _assert_image_tensor(img, data_format):
    if not isinstance(
            img, paddle.Tensor) or img.ndim != 3 or not data_format.lower() in (
                'chw', 'hwc'):
        raise RuntimeError(
            'not support [type={}, ndim={}, data_format={}] paddle image'.
            format(type(img), img.ndim, data_format))


def _get_image_h_axis(data_format):
    if data_format.lower() == 'chw':
        return -2
    elif data_format.lower() == 'hwc':
        return -3


def _get_image_w_axis(data_format):
    if data_format.lower() == 'chw':
        return -1
    elif data_format.lower() == 'hwc':
        return -2


def _get_image_c_axis(data_format):
    if data_format.lower() == 'chw':
        return -3
    elif data_format.lower() == 'hwc':
        return -1


def _get_image_n_axis(data_format):
    if len(data_format) == 3:
        return None
    elif len(data_format) == 4:
        return 0


def _is_channel_last(data_format):
    return _get_image_c_axis(data_format) == -1


def _is_channel_first(data_format):
    return _get_image_c_axis(data_format) == -3


def _get_image_num_batches(img, data_format):
    if _get_image_n_axis(data_format):
        return img.shape[_get_image_n_axis(data_format)]
    return None


def _get_image_num_channels(img, data_format):
    return img.shape[_get_image_c_axis(data_format)]


def _get_image_size(img, data_format):
    return img.shape[_get_image_w_axis(data_format)], img.shape[
        _get_image_h_axis(data_format)]


def normalize(img, mean, std, data_format='CHW'):
    """Normalizes a tensor image given mean and standard deviation.

    Args:
        img (paddle.Tensor): input data to be normalized.
        mean (list|tuple): Sequence of means for each channel.
        std (list|tuple): Sequence of standard deviations for each channel.
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.

    Returns:
        Tensor: Normalized mage.

    """
    _assert_image_tensor(img, data_format)

    mean = paddle.to_tensor(mean, place=img.place)
    std = paddle.to_tensor(std, place=img.place)

    if _is_channel_first(data_format):
        mean = mean.reshape([-1, 1, 1])
        std = std.reshape([-1, 1, 1])

    return (img - mean) / std


def to_grayscale(img, num_output_channels=1, data_format='CHW'):
    """Converts image to grayscale version of image.

    Args:
        img (paddel.Tensor): Image to be converted to grayscale.
        num_output_channels (int, optionl[1, 3]):
            if num_output_channels = 1 : returned image is single channel
            if num_output_channels = 3 : returned image is 3 channel 
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.

    Returns:
        paddle.Tensor: Grayscale version of the image.
    """
    _assert_image_tensor(img, data_format)

    if num_output_channels not in (1, 3):
        raise ValueError('num_output_channels should be either 1 or 3')

    rgb_weights = paddle.to_tensor(
        [0.2989, 0.5870, 0.1140], place=img.place).astype(img.dtype)

    if _is_channel_first(data_format):
        rgb_weights = rgb_weights.reshape((-1, 1, 1))

    _c_index = _get_image_c_axis(data_format)

    img = (img * rgb_weights).sum(axis=_c_index, keepdim=True)
    _shape = img.shape
    _shape[_c_index] = num_output_channels

    return img.expand(_shape)


def _affine_grid(theta, w, h, ow, oh):
    d = 0.5
    base_grid = paddle.ones((1, oh, ow, 3), dtype=theta.dtype)

    x_grid = paddle.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, ow)
    base_grid[..., 0] = x_grid
    y_grid = paddle.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, oh).unsqueeze_(-1)
    base_grid[..., 1] = y_grid

    scaled_theta = theta.transpose(
        (0, 2, 1)) / paddle.to_tensor([0.5 * w, 0.5 * h])
    output_grid = base_grid.reshape((1, oh * ow, 3)).bmm(scaled_theta)

    return output_grid.reshape((1, oh, ow, 2))


def _grid_transform(img, grid, mode, fill):
    if img.shape[0] > 1:
        grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2],
                           grid.shape[3])

    if fill is not None:
        dummy = paddle.ones(
            (img.shape[0], 1, img.shape[2], img.shape[3]), dtype=img.dtype)
        img = paddle.concat((img, dummy), axis=1)

    img = F.grid_sample(
        img, grid, mode=mode, padding_mode="zeros", align_corners=False)

    # Fill with required color
    if fill is not None:
        mask = img[:, -1:, :, :]  # n 1 h w
        img = img[:, :-1, :, :]  # n c h w
        mask = mask.expand_as(img)
        len_fill = len(fill) if isinstance(fill, (tuple, list)) else 1
        fill_img = paddle.to_tensor(fill).reshape(
            (1, len_fill, 1, 1)).expand_as(img)

        if mode == 'nearest':
            mask = paddle.cast(mask < 0.5, img.dtype)
            img = img * (1. - mask) + mask * fill_img
        else:  # 'bilinear'
            img = img * mask + (1.0 - mask) * fill_img

    return img


def rotate(img,
           angle,
           interpolation='nearest',
           expand=False,
           center=None,
           fill=None,
           data_format='CHW'):
    """Rotates the image by angle.

    Args:
        img (paddle.Tensor): Image to be rotated.
        angle (float or int): In degrees degrees counter clockwise order.
        interpolation (str, optional): Interpolation method. If omitted, or if the 
            image has only one channel, it is set NEAREST . when use pil backend, 
            support method are as following: 
            - "nearest" 
            - "bilinear"
            - "bicubic"
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
        paddle.Tensor: Rotated image.

    """

    angle = -angle % 360
    img = img.unsqueeze(0)

    # n, c, h, w = img.shape
    w, h = _get_image_size(img, data_format=data_format)

    img = img if data_format.lower() == 'chw' else img.transpose((0, 3, 1, 2))

    post_trans = [0, 0]

    if center is None:
        rotn_center = [0, 0]
    else:
        rotn_center = [(p - s * 0.5) for p, s in zip(center, [w, h])]

    angle = math.radians(angle)
    matrix = [
        math.cos(angle),
        math.sin(angle),
        0.0,
        -math.sin(angle),
        math.cos(angle),
        0.0,
    ]

    matrix[2] += matrix[0] * (-rotn_center[0] - post_trans[0]) + matrix[1] * (
        -rotn_center[1] - post_trans[1])
    matrix[5] += matrix[3] * (-rotn_center[0] - post_trans[0]) + matrix[4] * (
        -rotn_center[1] - post_trans[1])

    matrix[2] += rotn_center[0]
    matrix[5] += rotn_center[1]

    matrix = paddle.to_tensor(matrix, place=img.place)
    matrix = matrix.reshape((1, 2, 3))

    if expand:
        # calculate output size
        corners = paddle.to_tensor(
            [[-0.5 * w, -0.5 * h, 1.0], [-0.5 * w, 0.5 * h, 1.0],
             [0.5 * w, 0.5 * h, 1.0], [0.5 * w, -0.5 * h, 1.0]],
            place=matrix.place).astype(matrix.dtype)

        _pos = corners.reshape(
            (1, -1, 3)).bmm(matrix.transpose((0, 2, 1))).reshape((1, -1, 2))
        _min = _pos.min(axis=-2).floor()
        _max = _pos.max(axis=-2).ceil()

        npos = _max - _min
        nw = npos[0][0]
        nh = npos[0][1]

        ow, oh = int(nw.numpy()[0]), int(nh.numpy()[0])

    else:
        ow, oh = w, h

    grid = _affine_grid(matrix, w, h, ow, oh)

    out = _grid_transform(img, grid, mode=interpolation, fill=fill)

    out = out if data_format.lower() == 'chw' else out.transpose((0, 2, 3, 1))

    return out.squeeze(0)


def vflip(img, data_format='CHW'):
    """Vertically flips the given paddle tensor.

    Args:
        img (paddle.Tensor): Image to be flipped.
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.

    Returns:
        paddle.Tensor:  Vertically flipped image.

    """
    _assert_image_tensor(img, data_format)

    h_axis = _get_image_h_axis(data_format)

    return img.flip(axis=[h_axis])


def hflip(img, data_format='CHW'):
    """Horizontally flips the given paddle.Tensor Image.

    Args:
        img (paddle.Tensor): Image to be flipped.
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.

    Returns:
        paddle.Tensor:  Horizontall flipped image.

    """
    _assert_image_tensor(img, data_format)

    w_axis = _get_image_w_axis(data_format)

    return img.flip(axis=[w_axis])


def crop(img, top, left, height, width, data_format='CHW'):
    """Crops the given paddle.Tensor Image.

    Args:
        img (paddle.Tensor): Image to be cropped. (0,0) denotes the top left 
            corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.
    Returns:
        paddle.Tensor: Cropped image.

    """
    _assert_image_tensor(img, data_format)

    if _is_channel_first(data_format):
        return img[:, top:top + height, left:left + width]
    else:
        return img[top:top + height, left:left + width, :]


def center_crop(img, output_size, data_format='CHW'):
    """Crops the given paddle.Tensor Image and resize it to desired size.

        Args:
            img (paddle.Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
            output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions   
            data_format (str, optional): Data format of img, should be 'HWC' or 
                'CHW'. Default: 'CHW'.     
        Returns:
            paddle.Tensor: Cropped image.

        """
    _assert_image_tensor(img, data_format)

    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))

    image_width, image_height = _get_image_size(img, data_format)
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return crop(
        img,
        crop_top,
        crop_left,
        crop_height,
        crop_width,
        data_format=data_format)


def pad(img, padding, fill=0, padding_mode='constant', data_format='CHW'):
    """
    Pads the given paddle.Tensor on all sides with specified padding mode and fill value.

    Args:
        img (paddle.Tensor): Image to be padded.
        padding (int|list|tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
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
        paddle.Tensor: Padded image.

    """
    _assert_image_tensor(img, data_format)

    if not isinstance(padding, (numbers.Number, list, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, list, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')

    if isinstance(padding, (list, tuple)) and len(padding) not in [2, 4]:
        raise ValueError(
            "Padding must be an int or a 2, or 4 element tuple, not a " +
            "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    else:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    padding = [pad_left, pad_right, pad_top, pad_bottom]

    if padding_mode == 'edge':
        padding_mode = 'replicate'
    elif padding_mode == 'symmetric':
        raise ValueError('Do not support symmetric mdoe')

    img = img.unsqueeze(0)
    #  'constant', 'reflect', 'replicate', 'circular'
    img = F.pad(img,
                pad=padding,
                mode=padding_mode,
                value=float(fill),
                data_format='N' + data_format)

    return img.squeeze(0)


def resize(img, size, interpolation='bilinear', data_format='CHW'):
    """
    Resizes the image to given size

    Args:
        input (paddle.Tensor): Image to be resized.
        size (int|list|tuple): Target size of input data, with (height, width) shape.
        interpolation (int|str, optional): Interpolation method. when use paddle backend, 
            support method are as following: 
            - "nearest"  
            - "bilinear"
            - "bicubic"
            - "trilinear"
            - "area"
            - "linear"
        data_format (str, optional): paddle.Tensor format
            - 'CHW'
            - 'HWC'
    Returns:
        paddle.Tensor: Resized image.

    """
    _assert_image_tensor(img, data_format)

    if not (isinstance(size, int) or
            (isinstance(size, (tuple, list)) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = _get_image_size(img, data_format)
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
    else:
        oh, ow = size

    img = img.unsqueeze(0)
    img = F.interpolate(
        img,
        size=(oh, ow),
        mode=interpolation.lower(),
        data_format='N' + data_format.upper())

    return img.squeeze(0)
