# Copyright (c) 2020 VisualDL Authors. All Rights Reserve.
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
# =======================================================================
import math
from functools import reduce

import numpy as np

from visualdl.component.base_component import convert_to_HWC


def padding_image(img, height, width):
    height_old, width_old, _ = img.shape
    height_before = math.floor((height - height_old) / 2)
    height_after = height - height_old - height_before

    width_before = math.floor((width - width_old) / 2)
    width_after = width - width_old - width_before

    return np.pad(array=img,
                  pad_width=((height_before, height_after), (width_before, width_after), (0, 0)),
                  mode="constant")


def merge_images(imgs, dataformats, scale=1.0, rows=-1):
    assert rows <= len(imgs), "rows should not greater than numbers of pictures"
    channel = imgs[0].shape[2]
    # convert format of each image to `hwc`
    for i, img in enumerate(imgs):
        imgs[i] = convert_to_HWC(img, dataformats)

    height = -1
    width = -1

    for img in imgs:
        height = height if height > img.shape[0] else img.shape[0]
        width = width if width > img.shape[1] else img.shape[1]

    # padding every sub-image with height and width
    for i, img in enumerate(imgs):
        imgs[i] = padding_image(img, height, width)

    # get row and col
    len_imgs = len(imgs)
    if -1 == rows:
        rows = cols = math.floor(math.sqrt(len_imgs))
        while rows*cols < len_imgs:
            if rows <= cols:
                rows += 1
            else:
                cols += 1
    else:
        cols = math.ceil(len_imgs/rows)

    # add white sub-image
    for i in range(rows*cols-len_imgs):
        imgs = np.concatenate((imgs, np.zeros((height, width, channel), dtype=np.uint8)[None, :]))

    imgs = reduce(lambda x, y: np.concatenate((x, y)), [
        reduce(lambda x, y: np.concatenate((x, y), 1),
               imgs[i * cols: (i + 1) * cols]) for i in range(rows)])

    # choose bigger number of rows and cols

    scale = 1.0/scale * rows if rows > cols else 1.0/scale * cols

    dsize = tuple(map(lambda x: math.floor(x/scale), imgs.shape))[-2::-1]

    try:
        import cv2

        imgs = cv2.resize(src=imgs, dsize=dsize)
    except ImportError:
        from PIL import Image

        imgs = Image.fromarray(imgs)
        imgs.resize(dsize)
        imgs = np.array(imgs)

    return imgs
