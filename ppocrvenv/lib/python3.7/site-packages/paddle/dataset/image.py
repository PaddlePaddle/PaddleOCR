#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""
This file contains some common interfaces for image preprocess.
Many users are confused about the image layout. We introduce
the image layout as follows.

- CHW Layout

  - The abbreviations: C=channel, H=Height, W=Width
  - The default layout of image opened by cv2 or PIL is HWC.
    PaddlePaddle only supports the CHW layout. And CHW is simply
    a transpose of HWC. It must transpose the input image.

- Color format: RGB or BGR

  OpenCV use BGR color format. PIL use RGB color format. Both
  formats can be used for training. Noted that, the format should
  be keep consistent between the training and inference period.
"""

from __future__ import print_function

import six
import numpy as np
# FIXME(minqiyang): this is an ugly fix for the numpy bug reported here
# https://github.com/numpy/numpy/issues/12497
if six.PY3:
    import subprocess
    import sys
    import os
    interpreter = sys.executable
    # Note(zhouwei): if use Python/C 'PyRun_SimpleString', 'sys.executable'
    # will be the C++ execubable on Windows
    if sys.platform == 'win32' and 'python.exe' not in interpreter:
        interpreter = sys.exec_prefix + os.sep + 'python.exe'
    import_cv2_proc = subprocess.Popen(
        [interpreter, "-c", "import cv2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    out, err = import_cv2_proc.communicate()
    retcode = import_cv2_proc.poll()
    if retcode != 0:
        cv2 = None
    else:
        import cv2
else:
    try:
        import cv2
    except ImportError:
        cv2 = None
import os
import tarfile
import six.moves.cPickle as pickle

__all__ = []


def _check_cv2():
    if cv2 is None:
        import sys
        sys.stderr.write(
            '''Warning with paddle image module: opencv-python should be imported,
         or paddle image module could NOT work; please install opencv-python first.'''
        )
        return False
    else:
        return True


def batch_images_from_tar(data_file,
                          dataset_name,
                          img2label,
                          num_per_batch=1024):
    """
    Read images from tar file and batch them into batch file.

    :param data_file: path of image tar file
    :type data_file: string
    :param dataset_name: 'train','test' or 'valid'
    :type dataset_name: string
    :param img2label: a dic with image file name as key
                    and image's label as value
    :type img2label: dic
    :param num_per_batch: image number per batch file
    :type num_per_batch: int
    :return: path of list file containing paths of batch file
    :rtype: string
    """
    batch_dir = data_file + "_batch"
    out_path = "%s/%s_%s" % (batch_dir, dataset_name, os.getpid())
    meta_file = "%s/%s_%s.txt" % (batch_dir, dataset_name, os.getpid())

    if os.path.exists(out_path):
        return meta_file
    else:
        os.makedirs(out_path)

    tf = tarfile.open(data_file)
    mems = tf.getmembers()
    data = []
    labels = []
    file_id = 0
    for mem in mems:
        if mem.name in img2label:
            data.append(tf.extractfile(mem).read())
            labels.append(img2label[mem.name])
            if len(data) == num_per_batch:
                output = {}
                output['label'] = labels
                output['data'] = data
                pickle.dump(
                    output,
                    open('%s/batch_%d' % (out_path, file_id), 'wb'),
                    protocol=2)
                file_id += 1
                data = []
                labels = []
    if len(data) > 0:
        output = {}
        output['label'] = labels
        output['data'] = data
        pickle.dump(
            output, open('%s/batch_%d' % (out_path, file_id), 'wb'), protocol=2)

    with open(meta_file, 'a') as meta:
        for file in os.listdir(out_path):
            meta.write(os.path.abspath("%s/%s" % (out_path, file)) + "\n")
    return meta_file


def load_image_bytes(bytes, is_color=True):
    """
    Load an color or gray image from bytes array.

    Example usage:

    .. code-block:: python

        with open('cat.jpg') as f:
            im = load_image_bytes(f.read())

    :param bytes: the input image bytes array.
    :type bytes: str
    :param is_color: If set is_color True, it will load and
                     return a color image. Otherwise, it will
                     load and return a gray image.
    :type is_color: bool
    """
    assert _check_cv2() is True

    flag = 1 if is_color else 0
    file_bytes = np.asarray(bytearray(bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, flag)
    return img


def load_image(file, is_color=True):
    """
    Load an color or gray image from the file path.

    Example usage:

    .. code-block:: python

        im = load_image('cat.jpg')

    :param file: the input image path.
    :type file: string
    :param is_color: If set is_color True, it will load and
                     return a color image. Otherwise, it will
                     load and return a gray image.
    :type is_color: bool
    """
    assert _check_cv2() is True

    # cv2.IMAGE_COLOR for OpenCV3
    # cv2.CV_LOAD_IMAGE_COLOR for older OpenCV Version
    # cv2.IMAGE_GRAYSCALE for OpenCV3
    # cv2.CV_LOAD_IMAGE_GRAYSCALE for older OpenCV Version
    # Here, use constant 1 and 0
    # 1: COLOR, 0: GRAYSCALE
    flag = 1 if is_color else 0
    im = cv2.imread(file, flag)
    return im


def resize_short(im, size):
    """
    Resize an image so that the length of shorter edge is size.

    Example usage:

    .. code-block:: python

        im = load_image('cat.jpg')
        im = resize_short(im, 256)

    :param im: the input image with HWC layout.
    :type im: ndarray
    :param size: the shorter edge size of image after resizing.
    :type size: int
    """
    assert _check_cv2() is True

    h, w = im.shape[:2]
    h_new, w_new = size, size
    if h > w:
        h_new = size * h // w
    else:
        w_new = size * w // h
    im = cv2.resize(im, (w_new, h_new), interpolation=cv2.INTER_CUBIC)
    return im


def to_chw(im, order=(2, 0, 1)):
    """
    Transpose the input image order. The image layout is HWC format
    opened by cv2 or PIL. Transpose the input image to CHW layout
    according the order (2,0,1).

    Example usage:

    .. code-block:: python

        im = load_image('cat.jpg')
        im = resize_short(im, 256)
        im = to_chw(im)

    :param im: the input image with HWC layout.
    :type im: ndarray
    :param order: the transposed order.
    :type order: tuple|list
    """
    assert len(im.shape) == len(order)
    im = im.transpose(order)
    return im


def center_crop(im, size, is_color=True):
    """
    Crop the center of image with size.

    Example usage:

    .. code-block:: python

        im = center_crop(im, 224)

    :param im: the input image with HWC layout.
    :type im: ndarray
    :param size: the cropping size.
    :type size: int
    :param is_color: whether the image is color or not.
    :type is_color: bool
    """
    h, w = im.shape[:2]
    h_start = (h - size) // 2
    w_start = (w - size) // 2
    h_end, w_end = h_start + size, w_start + size
    if is_color:
        im = im[h_start:h_end, w_start:w_end, :]
    else:
        im = im[h_start:h_end, w_start:w_end]
    return im


def random_crop(im, size, is_color=True):
    """
    Randomly crop input image with size.

    Example usage:

    .. code-block:: python

        im = random_crop(im, 224)

    :param im: the input image with HWC layout.
    :type im: ndarray
    :param size: the cropping size.
    :type size: int
    :param is_color: whether the image is color or not.
    :type is_color: bool
    """
    h, w = im.shape[:2]
    h_start = np.random.randint(0, h - size + 1)
    w_start = np.random.randint(0, w - size + 1)
    h_end, w_end = h_start + size, w_start + size
    if is_color:
        im = im[h_start:h_end, w_start:w_end, :]
    else:
        im = im[h_start:h_end, w_start:w_end]
    return im


def left_right_flip(im, is_color=True):
    """
    Flip an image along the horizontal direction.
    Return the flipped image.

    Example usage:

    .. code-block:: python

        im = left_right_flip(im)

    :param im: input image with HWC layout or HW layout for gray image
    :type im: ndarray
    :param is_color: whether input image is color or not
    :type is_color: bool
    """
    if len(im.shape) == 3 and is_color:
        return im[:, ::-1, :]
    else:
        return im[:, ::-1]


def simple_transform(im,
                     resize_size,
                     crop_size,
                     is_train,
                     is_color=True,
                     mean=None):
    """
    Simply data argumentation for training. These operations include
    resizing, croping and flipping.

    Example usage:

    .. code-block:: python

        im = simple_transform(im, 256, 224, True)

    :param im: The input image with HWC layout.
    :type im: ndarray
    :param resize_size: The shorter edge length of the resized image.
    :type resize_size: int
    :param crop_size: The cropping size.
    :type crop_size: int
    :param is_train: Whether it is training or not.
    :type is_train: bool
    :param is_color: whether the image is color or not.
    :type is_color: bool
    :param mean: the mean values, which can be element-wise mean values or
                 mean values per channel.
    :type mean: numpy array | list
    """
    im = resize_short(im, resize_size)
    if is_train:
        im = random_crop(im, crop_size, is_color=is_color)
        if np.random.randint(2) == 0:
            im = left_right_flip(im, is_color)
    else:
        im = center_crop(im, crop_size, is_color=is_color)
    if len(im.shape) == 3:
        im = to_chw(im)

    im = im.astype('float32')
    if mean is not None:
        mean = np.array(mean, dtype=np.float32)
        # mean value, may be one value per channel
        if mean.ndim == 1 and is_color:
            mean = mean[:, np.newaxis, np.newaxis]
        elif mean.ndim == 1:
            mean = mean
        else:
            # elementwise mean
            assert len(mean.shape) == len(im)
        im -= mean

    return im


def load_and_transform(filename,
                       resize_size,
                       crop_size,
                       is_train,
                       is_color=True,
                       mean=None):
    """
    Load image from the input file `filename` and transform image for
    data argumentation. Please refer to the `simple_transform` interface
    for the transform operations.

    Example usage:

    .. code-block:: python

        im = load_and_transform('cat.jpg', 256, 224, True)

    :param filename: The file name of input image.
    :type filename: string
    :param resize_size: The shorter edge length of the resized image.
    :type resize_size: int
    :param crop_size: The cropping size.
    :type crop_size: int
    :param is_train: Whether it is training or not.
    :type is_train: bool
    :param is_color: whether the image is color or not.
    :type is_color: bool
    :param mean: the mean values, which can be element-wise mean values or
                 mean values per channel.
    :type mean: numpy array | list
    """
    im = load_image(filename, is_color)
    im = simple_transform(im, resize_size, crop_size, is_train, is_color, mean)
    return im
