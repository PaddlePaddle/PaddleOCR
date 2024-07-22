# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import numbers
import numpy as np
from collections import defaultdict


class DictCollator(object):
    """
    data batch
    """

    def __call__(self, batch):
        # todo：support batch operators
        data_dict = defaultdict(list)
        to_tensor_keys = []
        for sample in batch:
            for k, v in sample.items():
                if isinstance(v, (np.ndarray, paddle.Tensor, numbers.Number)):
                    if k not in to_tensor_keys:
                        to_tensor_keys.append(k)
                data_dict[k].append(v)
        for k in to_tensor_keys:
            data_dict[k] = paddle.to_tensor(data_dict[k])
        return data_dict


class ListCollator(object):
    """
    data batch
    """

    def __call__(self, batch):
        # todo：support batch operators
        data_dict = defaultdict(list)
        to_tensor_idxs = []
        for sample in batch:
            for idx, v in enumerate(sample):
                if isinstance(v, (np.ndarray, paddle.Tensor, numbers.Number)):
                    if idx not in to_tensor_idxs:
                        to_tensor_idxs.append(idx)
                data_dict[idx].append(v)
        for idx in to_tensor_idxs:
            data_dict[idx] = paddle.to_tensor(data_dict[idx])
        return list(data_dict.values())


class SSLRotateCollate(object):
    """
    bach: [
        [(4*3xH*W), (4,)]
        [(4*3xH*W), (4,)]
        ...
    ]
    """

    def __call__(self, batch):
        output = [np.concatenate(d, axis=0) for d in zip(*batch)]
        return output


class DyMaskCollator(object):
    """
    batch: [
        image [batch_size, channel, maxHinbatch, maxWinbatch]
        image_mask [batch_size, channel, maxHinbatch, maxWinbatch]
        label [batch_size, maxLabelLen]
        label_mask [batch_size, maxLabelLen]
        ...
    ]
    """

    def __call__(self, batch):
        max_width, max_height, max_length = 0, 0, 0
        bs, channel = len(batch), batch[0][0].shape[0]
        proper_items = []
        for item in batch:
            if (
                item[0].shape[1] * max_width > 1600 * 320
                or item[0].shape[2] * max_height > 1600 * 320
            ):
                continue
            max_height = (
                item[0].shape[1] if item[0].shape[1] > max_height else max_height
            )
            max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
            max_length = len(item[1]) if len(item[1]) > max_length else max_length
            proper_items.append(item)

        images, image_masks = np.zeros(
            (len(proper_items), channel, max_height, max_width), dtype="float32"
        ), np.zeros((len(proper_items), 1, max_height, max_width), dtype="float32")
        labels, label_masks = np.zeros(
            (len(proper_items), max_length), dtype="int64"
        ), np.zeros((len(proper_items), max_length), dtype="int64")

        for i in range(len(proper_items)):
            _, h, w = proper_items[i][0].shape
            images[i][:, :h, :w] = proper_items[i][0]
            image_masks[i][:, :h, :w] = 1
            l = len(proper_items[i][1])
            labels[i][:l] = proper_items[i][1]
            label_masks[i][:l] = 1

        return images, image_masks, labels, label_masks


class LaTeXOCRCollator(object):
    """
    batch: [
        image [batch_size, channel, maxHinbatch, maxWinbatch]
        label [batch_size, maxLabelLen]
        label_mask [batch_size, maxLabelLen]
        ...
    ]
    """

    def __call__(self, batch):
        images, labels, attention_mask = batch[0]
        return images, labels, attention_mask
