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

import numpy as np

__all__ = ['Stack', 'Pad', 'Tuple', 'Dict']


class Stack(object):
    """
    Stacks the input data samples to construct the batch. The N input samples
    must have the same shape/length and will be stacked to construct a batch.

    Args:
        axis (int, optional): The axis in the result data along which the input
            data are stacked. Default: 0.
        dtype (str|numpy.dtype, optional): The value type of the output. If it
            is set to None, the type of input data is used. Default: None.
    """

    def __init__(self, axis=0, dtype=None):
        self._axis = axis
        self._dtype = dtype

    def __call__(self, data):
        """
        Batchifies the input data by stacking.

        Args:
            data (list[numpy.ndarray]): The input data samples. It is a list. 
                Each element is a numpy.ndarray or list.

        Returns:
            numpy.ndarray: Stacked batch data.


        Example:
            .. code-block:: python

                from paddlenlp.data import Stack
                a = [1, 2, 3, 4]
                b = [3, 4, 5, 6]
                c = [5, 6, 7, 8]
                result = Stack()([a, b, c])
                '''
                [[1, 2, 3, 4],
                 [3, 4, 5, 6],
                 [5, 6, 7, 8]]
                '''
        """
        data = np.stack(
            data,
            axis=self._axis).astype(self._dtype) if self._dtype else np.stack(
                data, axis=self._axis)
        return data


class Pad(object):
    """
    Pads the input data samples to the largest length at `axis`.

    Args:
        pad_val (float|int, optional): The padding value. Default: 0.
        axis (int, optional): The axis to pad the arrays. The arrays will be
            padded to the largest length at `axis`. For example, assume the 
            input arrays have shape (10, 8, 5), (6, 8, 5), (3, 8, 5) and the 
            axis is 0. Each input will be padded into (10, 8, 5) and then 
            stacked to form the final output, which has shape (3, 10, 8, 5). 
            Default: 0.
        ret_length (bool|numpy.dtype, optional): If it is bool, indicate whether
            to return the valid length in the output, and the data type of
            returned length is int32 if True. If it is numpy.dtype, indicate the
            data type of returned length. Default: None.
        dtype (numpy.dtype, optional): The value type of the output. If it is
            set to None, the input data type is used. Default: None.
        pad_right (bool, optional): Whether the padding direction is right-side. 
            If True, it indicates we pad to the right side, while False indicates 
            we pad to the left side. Default: True.
     """

    def __init__(self,
                 pad_val=0,
                 axis=0,
                 ret_length=None,
                 dtype=None,
                 pad_right=True):
        self._pad_val = pad_val
        self._axis = axis
        self._ret_length = ret_length
        self._dtype = dtype
        self._pad_right = pad_right

    def __call__(self, data):
        """
        Batchifies the input data by padding. The input will be padded to the 
        largest dimension at `axis` and then stacked to form the final output. 
        In addition, the function will output the original dimensions at the 
        `axis` if `ret_length` is not None or False.

        Args:
            data (list[numpy.ndarray|list]): The input data samples. It is a 
                list. Each element is a numpy.ndarray or list.

        Returns:
            numpy.ndarray|tuple[numpy.ndarray]: If `ret_length` is False, it 
            is a numpy.ndarray representing the padded batch data and the 
            shape is (N, …). Otherwise, it is a tuple, besides the padded batch 
            data, the tuple also includes a numpy.ndarray representing original 
            length at `axis` of all input samples, which shaped `(N,)`. 

        Example:
            .. code-block:: python

                from paddlenlp.data import Pad
                a = [1, 2, 3, 4]
                b = [5, 6, 7]
                c = [8, 9]
                result = Pad(pad_val=0)([a, b, c])
                '''
                [[1, 2, 3, 4],
                 [5, 6, 7, 0],
                 [8, 9, 0, 0]]
                '''
        """
        arrs = [np.asarray(ele) for ele in data]
        original_length = [ele.shape[self._axis] for ele in arrs]
        max_size = max(original_length)
        ret_shape = list(arrs[0].shape)
        ret_shape[self._axis] = max_size
        ret_shape = (len(arrs), ) + tuple(ret_shape)
        ret = np.full(
            shape=ret_shape,
            fill_value=self._pad_val,
            dtype=arrs[0].dtype if self._dtype is None else self._dtype)
        for i, arr in enumerate(arrs):
            if arr.shape[self._axis] == max_size:
                ret[i] = arr
            else:
                slices = [slice(None) for _ in range(arr.ndim)]
                if self._pad_right:
                    slices[self._axis] = slice(0, arr.shape[self._axis])
                else:
                    slices[self._axis] = slice(max_size - arr.shape[self._axis],
                                               max_size)

                if slices[self._axis].start != slices[self._axis].stop:
                    slices = [slice(i, i + 1)] + slices
                    ret[tuple(slices)] = arr
        if self._ret_length:
            return ret, np.asarray(
                original_length,
                dtype="int32") if self._ret_length == True else np.asarray(
                    original_length, self._ret_length)
        else:
            return ret


class Tuple(object):
    """
    Wraps multiple batchify functions together. The input functions will be applied
    to the corresponding input fields.
    
    Each sample should be a list or tuple containing multiple fields. The i'th
    batchify function stored in Tuple will be applied on the i'th field. 
    
    For example, when data sample is (nd_data, label), you can wrap two batchify
    functions using `Tuple(DataBatchify, LabelBatchify)` to batchify nd_data and
    label correspondingly.

    Args:
        fn (callable|list[callable]|tuple[callable]): The batchify functions to 
            wrap. It is a callable function or a list/tuple of callable functions.
        args (tuple[callable]): The additional batchify functions to wrap.
    """

    def __init__(self, fn, *args):
        if isinstance(fn, (list, tuple)):
            assert len(args) == 0, 'Input pattern not understood. The input of Tuple can be ' \
                                   'Tuple(A, B, C) or Tuple([A, B, C]) or Tuple((A, B, C)). ' \
                                   'Received fn=%s, args=%s' % (str(fn), str(args))
            self._fn = fn
        else:
            self._fn = (fn, ) + args
        for i, ele_fn in enumerate(self._fn):
            assert callable(
                ele_fn
            ), 'Batchify functions must be callable! type(fn[%d]) = %s' % (
                i, str(type(ele_fn)))

    def __call__(self, data):
        """
        Batchifies data samples by applying each function on the corresponding 
        data field, and each data field is produced by stacking the field data 
        of samples.

        Args:
            data (list|tuple): The samples to batchfy. Each sample in list/tuple
                should contain `N` fields.

        Returns:
            tuple: A tuple composed of results from all including batchifying 
            functions.

        Example:
            .. code-block:: python
                
                from paddlenlp.data import Stack, Pad, Tuple
                data = [
                        [[1, 2, 3, 4], [1]],
                        [[5, 6, 7], [0]],
                        [[8, 9], [1]],
                       ]
                batchify_fn = Tuple(Pad(pad_val=0), Stack())
                ids, label = batchify_fn(data)
                '''
                ids:
                [[1, 2, 3, 4],
                [5, 6, 7, 0],
                [8, 9, 0, 0]]
                label: [[1], [0], [1]]
                '''
        """

        assert len(data[0]) == len(self._fn),\
            'The number of attributes in each data sample should contain' \
            ' {} elements'.format(len(self._fn))
        ret = []
        for i, ele_fn in enumerate(self._fn):
            result = ele_fn([ele[i] for ele in data])
            if isinstance(result, (tuple, list)):
                ret.extend(result)
            else:
                ret.append(result)
        return tuple(ret)


class Dict(object):
    """
    Wraps multiple batchify functions together. The input functions will be 
    applied to the corresponding input fields.
    
    Each sample should be a dict containing multiple fields. Each batchify 
    function with key stored in `Dict` will be applied on the field which has 
    the same key. 
    
    For example, when data sample is {'tokens': tokens, 'labels': labels}, you 
    can wrap two batchify functions using 
    `Dict({'tokens': DataBatchify, 'labels': LabelBatchify})` to batchify tokens 
    and labels correspondingly.

    Args:
        fn (dict): The batchify functions to wrap. It is a dict, which values is 
            callable functions.
    """

    def __init__(self, fn):
        assert isinstance(fn, (dict)), 'Input pattern not understood. The input of Dict must be a dict with key of input column name and value of collate_fn ' \
                                   'Received fn=%s' % (str(fn))

        self._fn = fn

        for col_name, ele_fn in self._fn.items():
            assert callable(
                ele_fn
            ), 'Batchify functions must be callable! type(fn[%d]) = %s' % (
                col_name, str(type(ele_fn)))

    def __call__(self, data):
        """
        Batchifies data samples by applying each function on the corresponding 
        data field, and each data field is produced by stacking the field data 
        with the same key as batchify functions of all samples.

        Args:
            data (list[dict]|tuple[dict]): The samples to batchfy. Each sample 
                in list/tuple is a dict with `N` key-values.
                
        Returns:
            tuple: A tuple composed of results from all including batchifying 
            functions.
            
        Example:
            .. code-block:: python

                from paddlenlp.data import Stack, Pad, Dict
                data = [
                        {'labels':[1], 'token_ids':[1, 2, 3, 4]},
                        {'labels':[0], 'token_ids':[5, 6, 7]},
                        {'labels':[1], 'token_ids':[8, 9]},
                       ]
                batchify_fn = Dict({'token_ids':Pad(pad_val=0), 'labels':Stack()})
                ids, label = batchify_fn(data)
                '''
                ids:
                [[1, 2, 3, 4],
                [5, 6, 7, 0],
                [8, 9, 0, 0]]
                label: [[1], [0], [1]]
                '''
        """

        ret = []
        for col_name, ele_fn in self._fn.items():
            result = ele_fn([ele[col_name] for ele in data])
            if isinstance(result, (tuple, list)):
                ret.extend(result)
            else:
                ret.append(result)
        return tuple(ret)
