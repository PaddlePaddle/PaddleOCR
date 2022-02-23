# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import numpy as np
from PIL import Image
from six.moves import cStringIO as StringIO

__all__ = []


def resize_image(img, target_size):
    """
    Resize an image so that the shorter edge has length target_size.
    img: the input image to be resized.
    target_size: the target resized image size.
    """
    percent = (target_size / float(min(img.size[0], img.size[1])))
    resized_size = int(round(img.size[0] * percent)), int(
        round(img.size[1] * percent))
    img = img.resize(resized_size, Image.ANTIALIAS)
    return img


def flip(im):
    """
    Return the flipped image.
    Flip an image along the horizontal direction.
    im: input image, (K x H x W) ndarrays
    """
    if len(im.shape) == 3:
        return im[:, :, ::-1]
    else:
        return im[:, ::-1]


def crop_img(im, inner_size, color=True, test=True):
    """
    Return cropped image.
    The size of the cropped image is inner_size * inner_size.
    im: (K x H x W) ndarrays
    inner_size: the cropped image size.
    color: whether it is color image.
    test: whether in test mode.
      If False, does random cropping and flipping.
      If True, crop the center of images.
    """
    if color:
        height, width = max(inner_size, im.shape[1]), max(inner_size,
                                                          im.shape[2])
        padded_im = np.zeros((3, height, width))
        startY = (height - im.shape[1]) / 2
        startX = (width - im.shape[2]) / 2
        endY, endX = startY + im.shape[1], startX + im.shape[2]
        padded_im[:, startY:endY, startX:endX] = im
    else:
        im = im.astype('float32')
        height, width = max(inner_size, im.shape[0]), max(inner_size,
                                                          im.shape[1])
        padded_im = np.zeros((height, width))
        startY = (height - im.shape[0]) / 2
        startX = (width - im.shape[1]) / 2
        endY, endX = startY + im.shape[0], startX + im.shape[1]
        padded_im[startY:endY, startX:endX] = im
    if test:
        startY = (height - inner_size) / 2
        startX = (width - inner_size) / 2
    else:
        startY = np.random.randint(0, height - inner_size + 1)
        startX = np.random.randint(0, width - inner_size + 1)
    endY, endX = startY + inner_size, startX + inner_size
    if color:
        pic = padded_im[:, startY:endY, startX:endX]
    else:
        pic = padded_im[startY:endY, startX:endX]
    if (not test) and (np.random.randint(2) == 0):
        pic = flip(pic)
    return pic


def decode_jpeg(jpeg_string):
    np_array = np.array(Image.open(StringIO(jpeg_string)))
    if len(np_array.shape) == 3:
        np_array = np.transpose(np_array, (2, 0, 1))
    return np_array


def preprocess_img(im, img_mean, crop_size, is_train, color=True):
    """
    Does data augmentation for images.
    If is_train is false, cropping the center region from the image.
    If is_train is true, randomly crop a region from the image,
    and random does flipping.
    im: (K x H x W) ndarrays
    """
    im = im.astype('float32')
    test = not is_train
    pic = crop_img(im, crop_size, color, test)
    pic -= img_mean
    return pic.flatten()


def load_meta(meta_path, mean_img_size, crop_size, color=True):
    """
    Return the loaded meta file.
    Load the meta image, which is the mean of the images in the dataset.
    The mean image is subtracted from every input image so that the expected mean
    of each input image is zero.
    """
    mean = np.load(meta_path)['data_mean']
    border = (mean_img_size - crop_size) / 2
    if color:
        assert (mean_img_size * mean_img_size * 3 == mean.shape[0])
        mean = mean.reshape(3, mean_img_size, mean_img_size)
        mean = mean[:, border:border + crop_size, border:border +
                    crop_size].astype('float32')
    else:
        assert (mean_img_size * mean_img_size == mean.shape[0])
        mean = mean.reshape(mean_img_size, mean_img_size)
        mean = mean[border:border + crop_size, border:border +
                    crop_size].astype('float32')
    return mean


def load_image(img_path, is_color=True):
    """
    Load image and return.
    img_path: image path.
    is_color: is color image or not.
    """
    img = Image.open(img_path)
    img.load()
    return img


def oversample(img, crop_dims):
    """
    image : iterable of (H x W x K) ndarrays
    crop_dims: (height, width) tuple for the crops.
    Returned data contains ten crops of input image, namely,
    four corner patches and the center patch as well as their
    horizontal reflections.
    """
    # Dimensions and center.
    im_shape = np.array(img[0].shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates
    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate(
        [-crop_dims / 2.0, crop_dims / 2.0])
    crops_ix = np.tile(crops_ix, (2, 1))

    # Extract crops
    crops = np.empty(
        (10 * len(img), crop_dims[0], crop_dims[1], im_shape[-1]),
        dtype=np.float32)
    ix = 0
    for im in img:
        for crop in crops_ix:
            crops[ix] = im[crop[0]:crop[2], crop[1]:crop[3], :]
            ix += 1
        crops[ix - 5:ix] = crops[ix - 5:ix, :, ::-1, :]  # flip for mirrors
    return crops


class ImageTransformer:
    def __init__(self,
                 transpose=None,
                 channel_swap=None,
                 mean=None,
                 is_color=True):
        self.is_color = is_color
        self.set_transpose(transpose)
        self.set_channel_swap(channel_swap)
        self.set_mean(mean)

    def set_transpose(self, order):
        if order is not None:
            if self.is_color:
                assert 3 == len(order)
        self.transpose = order

    def set_channel_swap(self, order):
        if order is not None:
            if self.is_color:
                assert 3 == len(order)
        self.channel_swap = order

    def set_mean(self, mean):
        if mean is not None:
            # mean value, may be one value per channel
            if mean.ndim == 1:
                mean = mean[:, np.newaxis, np.newaxis]
            else:
                # elementwise mean
                if self.is_color:
                    assert len(mean.shape) == 3
        self.mean = mean

    def transformer(self, data):
        if self.transpose is not None:
            data = data.transpose(self.transpose)
        if self.channel_swap is not None:
            data = data[self.channel_swap, :, :]
        if self.mean is not None:
            data -= self.mean
        return data
