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
import sys
import random

import numpy as np
import numbers
import types
import collections
import warnings
import traceback

from paddle.utils import try_import
from . import functional as F

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

__all__ = []


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif F._is_numpy_image(img):
        return img.shape[:2][::-1]
    elif F._is_tensor_image(img):
        return img.shape[1:][::-1]  # chw
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


def _check_input(value,
                 name,
                 center=1,
                 bound=(0, float('inf')),
                 clip_first_on_zero=True):
    if isinstance(value, numbers.Number):
        if value < 0:
            raise ValueError(
                "If {} is a single number, it must be non negative.".format(
                    name))
        value = [center - value, center + value]
        if clip_first_on_zero:
            value[0] = max(value[0], 0)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError("{} values should be between {}".format(name,
                                                                     bound))
    else:
        raise TypeError(
            "{} should be a single number or a list/tuple with lenght 2.".
            format(name))

    if value[0] == value[1] == center:
        value = None
    return value


class Compose(object):
    """
    Composes several transforms together use for composing list of transforms
    together for a dataset transform.

    Args:
        transforms (list|tuple): List/Tuple of transforms to compose.

    Returns:
        A compose object which is callable, __call__ for this Compose
        object will call each given :attr:`transforms` sequencely.

    Examples:
    
        .. code-block:: python

            from paddle.vision.datasets import Flowers
            from paddle.vision.transforms import Compose, ColorJitter, Resize

            transform = Compose([ColorJitter(), Resize(size=608)])
            flowers = Flowers(mode='test', transform=transform)

            for i in range(10):
                sample = flowers[i]
                print(sample[0].size, sample[1])

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for f in self.transforms:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                print("fail to perform transform [{}] with error: "
                      "{} and stack:\n{}".format(f, e, str(stack_info)))
                raise e
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class BaseTransform(object):
    """
    Base class of all transforms used in computer vision.

    calling logic: 

        if keys is None:
            _get_params -> _apply_image()
        else:
            _get_params -> _apply_*() for * in keys 

    If you want to implement a self-defined transform method for image,
    rewrite _apply_* method in subclass.

    Args:
        keys (list[str]|tuple[str], optional): Input type. Input is a tuple contains different structures,
            key is used to specify the type of input. For example, if your input
            is image type, then the key can be None or ("image"). if your input
            is (image, image) type, then the keys should be ("image", "image"). 
            if your input is (image, boxes), then the keys should be ("image", "boxes").

            Current available strings & data type are describe below:

            - "image": input image, with shape of (H, W, C) 
            - "coords": coordinates, with shape of (N, 2) 
            - "boxes": bounding boxes, with shape of (N, 4), "xyxy" format, 
            
                       the 1st "xy" represents top left point of a box, 
                       the 2nd "xy" represents right bottom point.

            - "mask": map used for segmentation, with shape of (H, W, 1)
            
            You can also customize your data types only if you implement the corresponding
            _apply_*() methods, otherwise ``NotImplementedError`` will be raised.
    
    Examples:
    
        .. code-block:: python

            import numpy as np
            from PIL import Image
            import paddle.vision.transforms.functional as F
            from paddle.vision.transforms import BaseTransform

            def _get_image_size(img):
                if F._is_pil_image(img):
                    return img.size
                elif F._is_numpy_image(img):
                    return img.shape[:2][::-1]
                else:
                    raise TypeError("Unexpected type {}".format(type(img)))

            class CustomRandomFlip(BaseTransform):
                def __init__(self, prob=0.5, keys=None):
                    super(CustomRandomFlip, self).__init__(keys)
                    self.prob = prob

                def _get_params(self, inputs):
                    image = inputs[self.keys.index('image')]
                    params = {}
                    params['flip'] = np.random.random() < self.prob
                    params['size'] = _get_image_size(image)
                    return params

                def _apply_image(self, image):
                    if self.params['flip']:
                        return F.hflip(image)
                    return image

                # if you only want to transform image, do not need to rewrite this function
                def _apply_coords(self, coords):
                    if self.params['flip']:
                        w = self.params['size'][0]
                        coords[:, 0] = w - coords[:, 0]
                    return coords

                # if you only want to transform image, do not need to rewrite this function
                def _apply_boxes(self, boxes):
                    idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
                    coords = np.asarray(boxes).reshape(-1, 4)[:, idxs].reshape(-1, 2)
                    coords = self._apply_coords(coords).reshape((-1, 4, 2))
                    minxy = coords.min(axis=1)
                    maxxy = coords.max(axis=1)
                    trans_boxes = np.concatenate((minxy, maxxy), axis=1)
                    return trans_boxes
                    
                # if you only want to transform image, do not need to rewrite this function
                def _apply_mask(self, mask):
                    if self.params['flip']:
                        return F.hflip(mask)
                    return mask

            # create fake inputs
            fake_img = Image.fromarray((np.random.rand(400, 500, 3) * 255.).astype('uint8'))
            fake_boxes = np.array([[2, 3, 200, 300], [50, 60, 80, 100]])
            fake_mask = fake_img.convert('L')

            # only transform for image:
            flip_transform = CustomRandomFlip(1.0)
            converted_img = flip_transform(fake_img)

            # transform for image, boxes and mask
            flip_transform = CustomRandomFlip(1.0, keys=('image', 'boxes', 'mask'))
            (converted_img, converted_boxes, converted_mask) = flip_transform((fake_img, fake_boxes, fake_mask))
            print('converted boxes', converted_boxes)

    """

    def __init__(self, keys=None):
        if keys is None:
            keys = ("image", )
        elif not isinstance(keys, Sequence):
            raise ValueError(
                "keys should be a sequence, but got keys={}".format(keys))
        for k in keys:
            if self._get_apply(k) is None:
                raise NotImplementedError(
                    "{} is unsupported data structure".format(k))
        self.keys = keys

        # storage some params get from function get_params()
        self.params = None

    def _get_params(self, inputs):
        pass

    def __call__(self, inputs):
        """Apply transform on single input data"""
        if not isinstance(inputs, tuple):
            inputs = (inputs, )

        self.params = self._get_params(inputs)

        outputs = []
        for i in range(min(len(inputs), len(self.keys))):
            apply_func = self._get_apply(self.keys[i])
            if apply_func is None:
                outputs.append(inputs[i])
            else:
                outputs.append(apply_func(inputs[i]))
        if len(inputs) > len(self.keys):
            outputs.extend(inputs[len(self.keys):])

        if len(outputs) == 1:
            outputs = outputs[0]
        else:
            outputs = tuple(outputs)
        return outputs

    def _get_apply(self, key):
        return getattr(self, "_apply_{}".format(key), None)

    def _apply_image(self, image):
        raise NotImplementedError

    def _apply_boxes(self, boxes):
        raise NotImplementedError

    def _apply_mask(self, mask):
        raise NotImplementedError


class ToTensor(BaseTransform):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to ``paddle.Tensor``.

    Converts a PIL.Image or numpy.ndarray (H x W x C) to a paddle.Tensor of shape (C x H x W).

    If input is a grayscale image (H x W), it will be converted to a image of shape (H x W x 1). 
    And the shape of output tensor will be (1 x H x W).

    If you want to keep the shape of output tensor as (H x W x C), you can set data_format = ``HWC`` .

    Converts a PIL.Image or numpy.ndarray in the range [0, 255] to a paddle.Tensor in the 
    range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, 
    RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8. 

    In the other cases, tensors are returned without scaling.

    Args:
        data_format (str, optional): Data format of output tensor, should be 'HWC' or 
            'CHW'. Default: 'CHW'.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.
    
    Shape:
        - img(PIL.Image|np.ndarray): The input image with shape (H x W x C).
        - output(np.ndarray): A tensor with shape (C x H x W) or (H x W x C) according option data_format.

    Returns:
        A callable object of ToTensor.

    Examples:
    
        .. code-block:: python

            import numpy as np
            from PIL import Image

            import paddle.vision.transforms as T
            import paddle.vision.transforms.functional as F

            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

            transform = T.ToTensor()

            tensor = transform(fake_img)

    """

    def __init__(self, data_format='CHW', keys=None):
        super(ToTensor, self).__init__(keys)
        self.data_format = data_format

    def _apply_image(self, img):
        """
        Args:
            img (PIL.Image|np.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(img, self.data_format)


class Resize(BaseTransform):
    """Resize the input Image to the given size.

    Args:
        size (int|list|tuple): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int|str, optional): Interpolation method. Default: 'bilinear'. 
            when use pil backend, support method are as following: 
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
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A resized image.

    Returns:
        A callable object of Resize.

    Examples:
    
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import Resize

            transform = Resize(size=224)

            fake_img = Image.fromarray((np.random.rand(100, 120, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)
            print(fake_img.size)
    """

    def __init__(self, size, interpolation='bilinear', keys=None):
        super(Resize, self).__init__(keys)
        assert isinstance(size, int) or (isinstance(size, Iterable) and
                                         len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def _apply_image(self, img):
        return F.resize(img, self.size, self.interpolation)


class RandomResizedCrop(BaseTransform):
    """Crop the input data to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 1.33) of the original aspect ratio is made.
    After applying crop transfrom, the input data will be resized to given size.

    Args:
        size (int|list|tuple): Target size of output image, with (height, width) shape.
        scale (list|tuple): Scale range of the cropped image before resizing, relatively to the origin 
            image. Default: (0.08, 1.0)
        ratio (list|tuple): Range of aspect ratio of the origin aspect ratio cropped. Default: (0.75, 1.33)
        interpolation (int|str, optional): Interpolation method. Default: 'bilinear'. when use pil backend, 
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
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A cropped image.

    Returns:
        A callable object of RandomResizedCrop.

    Examples:
    
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import RandomResizedCrop

            transform = RandomResizedCrop(224)

            fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)
            print(fake_img.size)

    """

    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4, 4. / 3),
                 interpolation='bilinear',
                 keys=None):
        super(RandomResizedCrop, self).__init__(keys)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        assert (scale[0] <= scale[1]), "scale should be of kind (min, max)"
        assert (ratio[0] <= ratio[1]), "ratio should be of kind (min, max)"
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def _get_param(self, image, attempts=10):
        width, height = _get_image_size(image)
        area = height * width

        for _ in range(attempts):
            target_area = np.random.uniform(*self.scale) * area
            log_ratio = tuple(math.log(x) for x in self.ratio)
            aspect_ratio = math.exp(np.random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:
            # return whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def _apply_image(self, img):
        i, j, h, w = self._get_param(img)

        cropped_img = F.crop(img, i, j, h, w)
        return F.resize(cropped_img, self.size, self.interpolation)


class CenterCrop(BaseTransform):
    """Crops the given the input data at the center.

    Args:
        size (int|list|tuple): Target size of output image, with (height, width) shape.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A cropped image.

    Returns:
        A callable object of CenterCrop.

    Examples:
    
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import CenterCrop

            transform = CenterCrop(224)

            fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)
            print(fake_img.size)
    """

    def __init__(self, size, keys=None):
        super(CenterCrop, self).__init__(keys)
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def _apply_image(self, img):
        return F.center_crop(img, self.size)


class RandomHorizontalFlip(BaseTransform):
    """Horizontally flip the input data randomly with a given probability.

    Args:
        prob (float, optional): Probability of the input data being flipped. Should be in [0, 1]. Default: 0.5
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A horiziotal flipped image.

    Returns:
        A callable object of RandomHorizontalFlip.

    Examples:
    
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import RandomHorizontalFlip

            transform = RandomHorizontalFlip(0.5)

            fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)
            print(fake_img.size)
    """

    def __init__(self, prob=0.5, keys=None):
        super(RandomHorizontalFlip, self).__init__(keys)
        assert 0 <= prob <= 1, "probability must be between 0 and 1"
        self.prob = prob

    def _apply_image(self, img):
        if random.random() < self.prob:
            return F.hflip(img)
        return img


class RandomVerticalFlip(BaseTransform):
    """Vertically flip the input data randomly with a given probability.

    Args:
        prob (float, optional): Probability of the input data being flipped. Default: 0.5
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A vertical flipped image.

    Returns:
        A callable object of RandomVerticalFlip.

    Examples:
    
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import RandomVerticalFlip

            transform = RandomVerticalFlip()

            fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)
            print(fake_img.size)

    """

    def __init__(self, prob=0.5, keys=None):
        super(RandomVerticalFlip, self).__init__(keys)
        assert 0 <= prob <= 1, "probability must be between 0 and 1"
        self.prob = prob

    def _apply_image(self, img):
        if random.random() < self.prob:
            return F.vflip(img)
        return img


class Normalize(BaseTransform):
    """Normalize the input data with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input data.
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (int|float|list|tuple): Sequence of means for each channel.
        std (int|float|list|tuple): Sequence of standard deviations for each channel.
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.
        to_rgb (bool, optional): Whether to convert to rgb. Default: False.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A normalized array or tensor.

    Returns:
        A callable object of Normalize.

    Examples:
    
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import Normalize

            normalize = Normalize(mean=[127.5, 127.5, 127.5], 
                                  std=[127.5, 127.5, 127.5],
                                  data_format='HWC')

            fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

            fake_img = normalize(fake_img)
            print(fake_img.shape)
            print(fake_img.max, fake_img.max)
    
    """

    def __init__(self,
                 mean=0.0,
                 std=1.0,
                 data_format='CHW',
                 to_rgb=False,
                 keys=None):
        super(Normalize, self).__init__(keys)
        if isinstance(mean, numbers.Number):
            mean = [mean, mean, mean]

        if isinstance(std, numbers.Number):
            std = [std, std, std]

        self.mean = mean
        self.std = std
        self.data_format = data_format
        self.to_rgb = to_rgb

    def _apply_image(self, img):
        return F.normalize(img, self.mean, self.std, self.data_format,
                           self.to_rgb)


class Transpose(BaseTransform):
    """Transpose input data to a target format.
    For example, most transforms use HWC mode image,
    while the Neural Network might use CHW mode input tensor.
    output image will be an instance of numpy.ndarray. 

    Args:
        order (list|tuple, optional): Target order of input data. Default: (2, 0, 1).
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.
    
    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(np.ndarray|Paddle.Tensor): A transposed array or tensor. If input 
            is a PIL.Image, output will be converted to np.ndarray automatically.

    Returns:
        A callable object of Transpose.

    Examples:
    
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import Transpose

            transform = Transpose()

            fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)
            print(fake_img.shape)
    
    """

    def __init__(self, order=(2, 0, 1), keys=None):
        super(Transpose, self).__init__(keys)
        self.order = order

    def _apply_image(self, img):
        if F._is_tensor_image(img):
            return img.transpose(self.order)

        if F._is_pil_image(img):
            img = np.asarray(img)

        if len(img.shape) == 2:
            img = img[..., np.newaxis]
        return img.transpose(self.order)


class BrightnessTransform(BaseTransform):
    """Adjust brightness of the image.

    Args:
        value (float): How much to adjust the brightness. Can be any
            non negative number. 0 gives the original image
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): An image with a transform in brghtness.

    Returns:
        A callable object of BrightnessTransform.

    Examples:
    
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import BrightnessTransform

            transform = BrightnessTransform(0.4)

            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)
            
    """

    def __init__(self, value, keys=None):
        super(BrightnessTransform, self).__init__(keys)
        self.value = _check_input(value, 'brightness')

    def _apply_image(self, img):
        if self.value is None:
            return img

        brightness_factor = random.uniform(self.value[0], self.value[1])
        return F.adjust_brightness(img, brightness_factor)


class ContrastTransform(BaseTransform):
    """Adjust contrast of the image.

    Args:
        value (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives the original image
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): An image with a transform in contrast.

    Returns:
        A callable object of ContrastTransform.

    Examples:
    
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import ContrastTransform

            transform = ContrastTransform(0.4)

            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)

    """

    def __init__(self, value, keys=None):
        super(ContrastTransform, self).__init__(keys)
        if value < 0:
            raise ValueError("contrast value should be non-negative")
        self.value = _check_input(value, 'contrast')

    def _apply_image(self, img):
        if self.value is None:
            return img

        contrast_factor = random.uniform(self.value[0], self.value[1])
        return F.adjust_contrast(img, contrast_factor)


class SaturationTransform(BaseTransform):
    """Adjust saturation of the image.

    Args:
        value (float): How much to adjust the saturation. Can be any
            non negative number. 0 gives the original image
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): An image with a transform in saturation.

    Returns:
        A callable object of SaturationTransform.

    Examples:
    
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import SaturationTransform

            transform = SaturationTransform(0.4)

            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))
        
            fake_img = transform(fake_img)

    """

    def __init__(self, value, keys=None):
        super(SaturationTransform, self).__init__(keys)
        self.value = _check_input(value, 'saturation')

    def _apply_image(self, img):
        if self.value is None:
            return img

        saturation_factor = random.uniform(self.value[0], self.value[1])
        return F.adjust_saturation(img, saturation_factor)


class HueTransform(BaseTransform):
    """Adjust hue of the image.

    Args:
        value (float): How much to adjust the hue. Can be any number
            between 0 and 0.5, 0 gives the original image
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): An image with a transform in hue.

    Returns:
        A callable object of HueTransform.

    Examples:
    
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import HueTransform

            transform = HueTransform(0.4)

            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)

    """

    def __init__(self, value, keys=None):
        super(HueTransform, self).__init__(keys)
        self.value = _check_input(
            value, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def _apply_image(self, img):
        if self.value is None:
            return img

        hue_factor = random.uniform(self.value[0], self.value[1])
        return F.adjust_hue(img, hue_factor)


class ColorJitter(BaseTransform):
    """Randomly change the brightness, contrast, saturation and hue of an image.

    Args:
        brightness (float): How much to jitter brightness.
            Chosen uniformly from [max(0, 1 - brightness), 1 + brightness]. Should be non negative numbers.
        contrast (float): How much to jitter contrast.
            Chosen uniformly from [max(0, 1 - contrast), 1 + contrast]. Should be non negative numbers.
        saturation (float): How much to jitter saturation.
            Chosen uniformly from [max(0, 1 - saturation), 1 + saturation]. Should be non negative numbers.
        hue (float): How much to jitter hue.
            Chosen uniformly from [-hue, hue]. Should have 0<= hue <= 0.5.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A color jittered image.

    Returns:
        A callable object of ColorJitter.

    Examples:
    
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import ColorJitter

            transform = ColorJitter(0.4, 0.4, 0.4, 0.4)

            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)

    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0,
                 keys=None):
        super(ColorJitter, self).__init__(keys)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def _get_param(self, brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            transforms.append(BrightnessTransform(brightness, self.keys))

        if contrast is not None:
            transforms.append(ContrastTransform(contrast, self.keys))

        if saturation is not None:
            transforms.append(SaturationTransform(saturation, self.keys))

        if hue is not None:
            transforms.append(HueTransform(hue, self.keys))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def _apply_image(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self._get_param(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)


class RandomCrop(BaseTransform):
    """Crops the given CV Image at a random location.

    Args:
        size (sequence|int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int|sequence|optional): Optional padding on each border
            of the image. If a sequence of length 4 is provided, it is used to pad left, 
            top, right, bottom borders respectively. Default: 0.
        pad_if_needed (boolean|optional): It will pad the image if smaller than the
            desired size to avoid raising an exception. Default: False.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.
    
    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A random cropped image.

    Returns:
        A callable object of RandomCrop.

    Examples:
    
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import RandomCrop

            transform = RandomCrop(224)

            fake_img = Image.fromarray((np.random.rand(324, 300, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)
            print(fake_img.size)
    """

    def __init__(self,
                 size,
                 padding=None,
                 pad_if_needed=False,
                 fill=0,
                 padding_mode='constant',
                 keys=None):
        super(RandomCrop, self).__init__(keys)
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def _get_param(self, img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def _apply_image(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        w, h = _get_image_size(img)

        # pad the width if needed
        if self.pad_if_needed and w < self.size[1]:
            img = F.pad(img, (self.size[1] - w, 0), self.fill,
                        self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and h < self.size[0]:
            img = F.pad(img, (0, self.size[0] - h), self.fill,
                        self.padding_mode)

        i, j, h, w = self._get_param(img, self.size)

        return F.crop(img, i, j, h, w)


class Pad(BaseTransform):
    """Pads the given CV Image on all sides with the given "pad" value.

    Args:
        padding (int|list|tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If list/tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a list/tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int|list|tuple): Pixel fill value for constant fill. Default is 0. If a list/tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            ``constant`` means pads with a constant value, this value is specified with fill. 
            ``edge`` means pads with the last value at the edge of the image. 
            ``reflect`` means pads with reflection of image (without repeating the last value on the edge) 
            padding ``[1, 2, 3, 4]`` with 2 elements on both sides in reflect mode 
            will result in ``[3, 2, 1, 2, 3, 4, 3, 2]``.
            ``symmetric`` menas pads with reflection of image (repeating the last value on the edge)
            padding ``[1, 2, 3, 4]`` with 2 elements on both sides in symmetric mode 
            will result in ``[2, 1, 1, 2, 3, 4, 4, 3]``.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.
    
    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A paded image.

    Returns:
        A callable object of Pad.

    Examples:
    
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import Pad

            transform = Pad(2)

            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)
            print(fake_img.size)
    """

    def __init__(self, padding, fill=0, padding_mode='constant', keys=None):
        assert isinstance(padding, (numbers.Number, list, tuple))
        assert isinstance(fill, (numbers.Number, str, list, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        if isinstance(padding, list):
            padding = tuple(padding)
        if isinstance(fill, list):
            fill = tuple(fill)

        if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
            raise ValueError(
                "Padding must be an int or a 2, or 4 element tuple, not a " +
                "{} element tuple".format(len(padding)))

        super(Pad, self).__init__(keys)
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def _apply_image(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img, self.padding, self.fill, self.padding_mode)


class RandomRotation(BaseTransform):
    """Rotates the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees) clockwise order.
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
        expand (bool|optional): Optional expansion flag. Default: False.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple|optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.
    
    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A rotated image.

    Returns:
        A callable object of RandomRotation.

    Examples:
    
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import RandomRotation

            transform = RandomRotation(90)

            fake_img = Image.fromarray((np.random.rand(200, 150, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)
            print(fake_img.size)
    """

    def __init__(self,
                 degrees,
                 interpolation='nearest',
                 expand=False,
                 center=None,
                 fill=0,
                 keys=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        super(RandomRotation, self).__init__(keys)
        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill

    def _get_param(self, degrees):
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def _apply_image(self, img):
        """
        Args:
            img (PIL.Image|np.array): Image to be rotated.

        Returns:
            PIL.Image or np.array: Rotated image.
        """

        angle = self._get_param(self.degrees)

        return F.rotate(img, angle, self.interpolation, self.expand,
                        self.center, self.fill)


class Grayscale(BaseTransform):
    """Converts image to grayscale.

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): Grayscale version of the input image. 
            - If output_channels == 1 : returned image is single channel
            - If output_channels == 3 : returned image is 3 channel with r == g == b

    Returns:
        A callable object of Grayscale.

    Examples:
    
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import Grayscale

            transform = Grayscale()

            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)
            print(np.array(fake_img).shape)
    """

    def __init__(self, num_output_channels=1, keys=None):
        super(Grayscale, self).__init__(keys)
        self.num_output_channels = num_output_channels

    def _apply_image(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        return F.to_grayscale(img, self.num_output_channels)
