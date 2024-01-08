# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
from paddle.nn import functional as F


def to_tensor(img: np.ndarray):
    """
    Converts a numpy array image (HWC) to a Paddle tensor (NCHW).

    Args:
        img (numpy.ndarray): The input image as a numpy array.

    Returns:
        out (paddle.Tensor): The output tensor.
    """
    img = img[:, :, ::-1]
    img = img.astype("float32") / 255.0
    img = img.transpose(2, 0, 1)
    out: paddle.Tensor = paddle.to_tensor(img)
    out = paddle.unsqueeze(out, axis=0)

    return out


def to_image(x: paddle.Tensor):
    """
    Converts a Paddle tensor (NCHW) to a numpy array image (HWC).

    Args:
        x (paddle.Tensor): The input tensor.

    Returns:
        out (numpy.ndarray): The output image as a numpy array.
    """
    out: np.ndarray = x.squeeze().numpy()
    out = out.transpose(1, 2, 0)
    out = out * 255.0
    out = out.astype("uint8")
    out = out[:, :, ::-1]

    return out


def unwarp(img, bm, bm_data_format="NCHW"):
    """
    Unwarp an image using a flow field.

    Args:
        img (paddle.Tensor): The input image.
        bm (paddle.Tensor): The flow field.

    Returns:
        out (paddle.Tensor): The output image.
    """
    _, _, h, w = img.shape

    if bm_data_format == "NHWC":
        bm = bm.transpose([0, 3, 1, 2])

    # NCHW
    bm = F.upsample(bm, size=(h, w), mode="bilinear", align_corners=True)
    # NHWC
    bm = bm.transpose([0, 2, 3, 1])
    # NCHW
    out = F.grid_sample(img, bm)

    return out
