#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from ..framework import in_dygraph_mode
from .. import core, layers

try:
    from collections.abc import Sequence, Mapping
except:
    from collections import Sequence, Mapping


def default_collate_fn(batch):
    """
    Default batch collating function for :code:`paddle.io.DataLoader`,
    get input data as a list of sample datas, each element in list
    if the data of a sample, and sample data should composed of list,
    dictionary, string, number, numpy array and paddle.Tensor, this
    function will parse input data recursively and stack number,
    numpy array and paddle.Tensor datas as batch datas. e.g. for
    following input data:

    [{'image': np.array(shape=[3, 224, 224]), 'label': 1},
     {'image': np.array(shape=[3, 224, 224]), 'label': 3},
     {'image': np.array(shape=[3, 224, 224]), 'label': 4},
     {'image': np.array(shape=[3, 224, 224]), 'label': 5},]
    
    
    This default collate function zipped each number and numpy array
    field together and stack each field as the batch field as follows:

    {'image': np.array(shape=[4, 3, 224, 224]), 'label': np.array([1, 3, 4, 5])}


    Args:  
        batch(list of sample data): batch should be a list of sample data.
    
    Returns:
        Batched data: batched each number, numpy array and paddle.Tensor
                      in input data.
    """
    sample = batch[0]
    if isinstance(sample, np.ndarray):
        batch = np.stack(batch, axis=0)
        return batch
    elif isinstance(sample, paddle.Tensor):
        return layers.stack(batch, axis=0)
    elif isinstance(sample, numbers.Number):
        batch = np.array(batch)
        return batch
    elif isinstance(sample, (str, bytes)):
        return batch
    elif isinstance(sample, Mapping):
        return {
            key: default_collate_fn([d[key] for d in batch])
            for key in sample
        }
    elif isinstance(sample, Sequence):
        sample_fields_num = len(sample)
        if not all(len(sample) == sample_fields_num for sample in iter(batch)):
            raise RuntimeError(
                "fileds number not same among samples in a batch")
        return [default_collate_fn(fields) for fields in zip(*batch)]

    raise TypeError("batch data con only contains: tensor, numpy.ndarray, "
                    "dict, list, number, but got {}".format(type(sample)))


def default_convert_fn(batch):
    """
    Default batch converting function for :code:`paddle.io.DataLoader`.
    get input data as a list of sample datas, each element in list
    if the data of a sample, and sample data should composed of list,
    dictionary, string, number, numpy array and paddle.Tensor.

    .. note::
        This function is default :attr:`collate_fn` in **Distable
        automatic batching** mode, for **Distable automatic batching**
        mode, please ses :attr:`paddle.io.DataLoader`

    Args:  
        batch(list of sample data): batch should be a list of sample data.
    
    Returns:
        Batched data: batched each number, numpy array and paddle.Tensor
                      in input data.
    """
    if isinstance(batch, (paddle.Tensor, np.ndarray)):
        return batch
    elif isinstance(batch, (str, bytes)):
        return batch
    elif isinstance(batch, Mapping):
        return {key: default_convert_fn(batch[key]) for key in batch}
    elif isinstance(batch, Sequence):
        return [default_convert_fn(d) for d in batch]
    else:
        return batch
