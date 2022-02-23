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
from PIL import Image
from numpy import sin, cos, tan
import paddle

from . import functional_pil as F_pil
from . import functional_cv2 as F_cv2
from . import functional_tensor as F_t

__all__ = []


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return isinstance(img, paddle.Tensor)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_tensor(pic, data_format='CHW'):
    """Converts a ``PIL.Image`` or ``numpy.ndarray`` to paddle.Tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL.Image|np.ndarray): Image to be converted to tensor.
        data_format (str, optional): Data format of output tensor, should be 'HWC' or 
            'CHW'. Default: 'CHW'.

    Returns:
        Tensor: Converted image. Data type is same as input img.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            tensor = F.to_tensor(fake_img)
            print(tensor.shape)

    """
    if not (_is_pil_image(pic) or _is_numpy_image(pic) or
            _is_tensor_image(pic)):
        raise TypeError(
            'pic should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}'.
            format(type(pic)))

    if _is_pil_image(pic):
        return F_pil.to_tensor(pic, data_format)
    elif _is_numpy_image(pic):
        return F_cv2.to_tensor(pic, data_format)
    else:
        return pic if data_format.lower() == 'chw' else pic.transpose((1, 2, 0))


def resize(img, size, interpolation='bilinear'):
    """
    Resizes the image to given size

    Args:
        input (PIL.Image|np.ndarray): Image to be resized.
        size (int|list|tuple): Target size of input data, with (height, width) shape.
        interpolation (int|str, optional): Interpolation method. when use pil backend, 
            support method are as following: 
            - "nearest": Image.NEAREST, 
            - "bilinear": Image.BILINEAR, 
            - "bicubic": Image.BICUBIC, 
            - "box": Image.BOX, 
            - "lanczos": Image.LANCZOS, 
            - "hamming": Image.HAMMING
            when use cv2 backend, support method are as following: 
            - "nearest": cv2.INTER_NEAREST, 
            - "bilinear": cv2.INTER_LINEAR, 
            - "area": cv2.INTER_AREA, 
            - "bicubic": cv2.INTER_CUBIC, 
            - "lanczos": cv2.INTER_LANCZOS4

    Returns:
        PIL.Image or np.array: Resized image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            converted_img = F.resize(fake_img, 224)
            print(converted_img.size)

            converted_img = F.resize(fake_img, (200, 150))
            print(converted_img.size)
    """
    if not (_is_pil_image(img) or _is_numpy_image(img) or
            _is_tensor_image(img)):
        raise TypeError(
            'img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}'.
            format(type(img)))

    if _is_pil_image(img):
        return F_pil.resize(img, size, interpolation)
    elif _is_tensor_image(img):
        return F_t.resize(img, size, interpolation)
    else:
        return F_cv2.resize(img, size, interpolation)


def pad(img, padding, fill=0, padding_mode='constant'):
    """
    Pads the given PIL.Image or numpy.array on all sides with specified padding mode and fill value.

    Args:
        img (PIL.Image|np.array): Image to be padded.
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
        PIL.Image or np.array: Padded image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            padded_img = F.pad(fake_img, padding=1)
            print(padded_img.size)

            padded_img = F.pad(fake_img, padding=(2, 1))
            print(padded_img.size)
    """
    if not (_is_pil_image(img) or _is_numpy_image(img) or
            _is_tensor_image(img)):
        raise TypeError(
            'img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}'.
            format(type(img)))

    if _is_pil_image(img):
        return F_pil.pad(img, padding, fill, padding_mode)
    elif _is_tensor_image(img):
        return F_t.pad(img, padding, fill, padding_mode)
    else:
        return F_cv2.pad(img, padding, fill, padding_mode)


def crop(img, top, left, height, width):
    """Crops the given Image.

    Args:
        img (PIL.Image|np.array): Image to be cropped. (0,0) denotes the top left 
            corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        PIL.Image or np.array: Cropped image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            cropped_img = F.crop(fake_img, 56, 150, 200, 100)
            print(cropped_img.size)

    """
    if not (_is_pil_image(img) or _is_numpy_image(img) or
            _is_tensor_image(img)):
        raise TypeError(
            'img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}'.
            format(type(img)))

    if _is_pil_image(img):
        return F_pil.crop(img, top, left, height, width)
    elif _is_tensor_image(img):
        return F_t.crop(img, top, left, height, width)
    else:
        return F_cv2.crop(img, top, left, height, width)


def center_crop(img, output_size):
    """Crops the given Image and resize it to desired size.

        Args:
            img (PIL.Image|np.array): Image to be cropped. (0,0) denotes the top left corner of the image.
            output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions
        
        Returns:
            PIL.Image or np.array: Cropped image.

        Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            cropped_img = F.center_crop(fake_img, (150, 100))
            print(cropped_img.size)
        """
    if not (_is_pil_image(img) or _is_numpy_image(img) or
            _is_tensor_image(img)):
        raise TypeError(
            'img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}'.
            format(type(img)))

    if _is_pil_image(img):
        return F_pil.center_crop(img, output_size)
    elif _is_tensor_image(img):
        return F_t.center_crop(img, output_size)
    else:
        return F_cv2.center_crop(img, output_size)


def hflip(img):
    """Horizontally flips the given Image or np.array.

    Args:
        img (PIL.Image|np.array): Image to be flipped.

    Returns:
        PIL.Image or np.array:  Horizontall flipped image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            flpped_img = F.hflip(fake_img)
            print(flpped_img.size)

    """
    if not (_is_pil_image(img) or _is_numpy_image(img) or
            _is_tensor_image(img)):
        raise TypeError(
            'img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}'.
            format(type(img)))

    if _is_pil_image(img):
        return F_pil.hflip(img)
    elif _is_tensor_image(img):
        return F_t.hflip(img)
    else:
        return F_cv2.hflip(img)


def vflip(img):
    """Vertically flips the given Image or np.array.

    Args:
        img (PIL.Image|np.array): Image to be flipped.

    Returns:
        PIL.Image or np.array:  Vertically flipped image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            flpped_img = F.vflip(fake_img)
            print(flpped_img.size)

    """
    if not (_is_pil_image(img) or _is_numpy_image(img) or
            _is_tensor_image(img)):
        raise TypeError(
            'img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}'.
            format(type(img)))

    if _is_pil_image(img):
        return F_pil.vflip(img)
    elif _is_tensor_image(img):
        return F_t.vflip(img)
    else:
        return F_cv2.vflip(img)


def adjust_brightness(img, brightness_factor):
    """Adjusts brightness of an Image.

    Args:
        img (PIL.Image|np.array): Image to be adjusted.
        brightness_factor (float): How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL.Image or np.array: Brightness adjusted image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            converted_img = F.adjust_brightness(fake_img, 0.4)
            print(converted_img.size)
    """
    if not (_is_pil_image(img) or _is_numpy_image(img)):
        raise TypeError(
            'img should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.
            format(type(img)))

    if _is_pil_image(img):
        return F_pil.adjust_brightness(img, brightness_factor)
    else:
        return F_cv2.adjust_brightness(img, brightness_factor)


def adjust_contrast(img, contrast_factor):
    """Adjusts contrast of an Image.

    Args:
        img (PIL.Image|np.array): Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        PIL.Image or np.array: Contrast adjusted image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            converted_img = F.adjust_contrast(fake_img, 0.4)
            print(converted_img.size)
    """
    if not (_is_pil_image(img) or _is_numpy_image(img)):
        raise TypeError(
            'img should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.
            format(type(img)))

    if _is_pil_image(img):
        return F_pil.adjust_contrast(img, contrast_factor)
    else:
        return F_cv2.adjust_contrast(img, contrast_factor)


def adjust_saturation(img, saturation_factor):
    """Adjusts color saturation of an image.

    Args:
        img (PIL.Image|np.array): Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL.Image or np.array: Saturation adjusted image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            converted_img = F.adjust_saturation(fake_img, 0.4)
            print(converted_img.size)

    """
    if not (_is_pil_image(img) or _is_numpy_image(img)):
        raise TypeError(
            'img should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.
            format(type(img)))

    if _is_pil_image(img):
        return F_pil.adjust_saturation(img, saturation_factor)
    else:
        return F_cv2.adjust_saturation(img, saturation_factor)


def adjust_hue(img, hue_factor):
    """Adjusts hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    Args:
        img (PIL.Image|np.array): Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL.Image or np.array: Hue adjusted image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            converted_img = F.adjust_hue(fake_img, 0.4)
            print(converted_img.size)

    """
    if not (_is_pil_image(img) or _is_numpy_image(img)):
        raise TypeError(
            'img should be PIL Image or ndarray with dim=[2 or 3]. Got {}'.
            format(type(img)))

    if _is_pil_image(img):
        return F_pil.adjust_hue(img, hue_factor)
    else:
        return F_cv2.adjust_hue(img, hue_factor)


def rotate(img,
           angle,
           interpolation="nearest",
           expand=False,
           center=None,
           fill=0):
    """Rotates the image by angle.


    Args:
        img (PIL.Image|np.array): Image to be rotated.
        angle (float or int): In degrees degrees counter clockwise order.
        interpolation (str, optional): Interpolation method. If omitted, or if the 
            image has only one channel, it is set to PIL.Image.NEAREST or cv2.INTER_NEAREST 
            according the backend. when use pil backend, support method are as following: 
            - "nearest": Image.NEAREST, 
            - "bilinear": Image.BILINEAR, 
            - "bicubic": Image.BICUBIC
            when use cv2 backend, support method are as following: 
            - "nearest": cv2.INTER_NEAREST, 
            - "bilinear": cv2.INTER_LINEAR, 
            - "bicubic": cv2.INTER_CUBIC
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-list|2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (3-list|3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.


    Returns:
        PIL.Image or np.array: Rotated image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            rotated_img = F.rotate(fake_img, 90)
            print(rotated_img.size)

    """
    if not (_is_pil_image(img) or _is_numpy_image(img) or
            _is_tensor_image(img)):
        raise TypeError(
            'img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}'.
            format(type(img)))

    if isinstance(center, list):
        center = tuple(center)
    if isinstance(fill, list):
        fill = tuple(fill)

    if _is_pil_image(img):
        return F_pil.rotate(img, angle, interpolation, expand, center, fill)
    elif _is_tensor_image(img):
        return F_t.rotate(img, angle, interpolation, expand, center, fill)
    else:
        return F_cv2.rotate(img, angle, interpolation, expand, center, fill)


def to_grayscale(img, num_output_channels=1):
    """Converts image to grayscale version of image.

    Args:
        img (PIL.Image|np.array): Image to be converted to grayscale.

    Returns:
        PIL.Image or np.array: Grayscale version of the image.
            if num_output_channels = 1 : returned image is single channel

            if num_output_channels = 3 : returned image is 3 channel with r = g = b
    
    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            gray_img = F.to_grayscale(fake_img)
            print(gray_img.size)

    """
    if not (_is_pil_image(img) or _is_numpy_image(img) or
            _is_tensor_image(img)):
        raise TypeError(
            'img should be PIL Image or Tensor Image or ndarray with dim=[2 or 3]. Got {}'.
            format(type(img)))

    if _is_pil_image(img):
        return F_pil.to_grayscale(img, num_output_channels)
    elif _is_tensor_image(img):
        return F_t.to_grayscale(img, num_output_channels)
    else:
        return F_cv2.to_grayscale(img, num_output_channels)


def normalize(img, mean, std, data_format='CHW', to_rgb=False):
    """Normalizes a tensor or image with mean and standard deviation.

    Args:
        img (PIL.Image|np.array|paddle.Tensor): input data to be normalized.
        mean (list|tuple): Sequence of means for each channel.
        std (list|tuple): Sequence of standard deviations for each channel.
        data_format (str, optional): Data format of input img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.
        to_rgb (bool, optional): Whether to convert to rgb. If input is tensor, 
            this option will be igored. Default: False.

    Returns:
        np.ndarray or Tensor: Normalized mage. Data format is same as input img.
    
    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            mean = [127.5, 127.5, 127.5]
            std = [127.5, 127.5, 127.5]

            normalized_img = F.normalize(fake_img, mean, std, data_format='HWC')
            print(normalized_img.max(), normalized_img.min())

    """

    if _is_tensor_image(img):
        return F_t.normalize(img, mean, std, data_format)
    else:
        if _is_pil_image(img):
            img = np.array(img).astype(np.float32)

        return F_cv2.normalize(img, mean, std, data_format, to_rgb)
