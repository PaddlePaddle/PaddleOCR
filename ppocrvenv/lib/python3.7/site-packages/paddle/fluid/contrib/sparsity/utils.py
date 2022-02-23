# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
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
Utilities of Auto SParsity (ASP).
"""

from __future__ import print_function

import sys
import math
import collections
import numpy as np
from enum import Enum
from itertools import permutations
import threading

__all__ = [
    'calculate_density', 'check_mask_1d', 'get_mask_1d', 'check_mask_2d',
    'get_mask_2d_greedy', 'get_mask_2d_best', 'create_mask', 'check_sparsity',
    'MaskAlgo', 'CheckMethod'
]


class MaskAlgo(Enum):
    r"""
    A collection of all mask generating algorithms.
    There currently are three algorithms, `MASK_1D`, `MASK_2D_GREEDY` and `MASK_2D_BEST`
    """
    MASK_1D = 'get_mask_1d'
    MASK_2D_GREEDY = 'get_mask_2d_greedy'
    MASK_2D_BEST = 'get_mask_2d_best'


class CheckMethod(Enum):
    r"""
    A collection of all sparsity checking approaches.
    There currently are two methods, `CHECK_1D` and `CHECK_2D`
    """
    CHECK_1D = 'check_mask_1d'
    CHECK_2D = 'check_mask_2d'

    @staticmethod
    def get_checking_method(mask_algo):
        r"""
        Get sparsity checking method by mask generating algorithm.

        Args:
            mask_algo (MaskAlgo): The algorithm of mask generating.
        Returns:
            CheckMethod: The corresponded sparsity checking method.
        Examples:
            .. code-block:: python

            import numpy as np
            from paddle.static.sparsity import MaskAlgo
            from paddle.fluid.contrib.sparsity import CheckMethod

            CheckMethod.get_checking_method(MaskAlgo.MASK_1D)
            # CheckMethod.CHECK_1D

            CheckMethod.get_checking_method(MaskAlgo.MASK_2D_GREEDY)
            # CheckMethod.CHECK_2D

            CheckMethod.get_checking_method(MaskAlgo.MASK_2D_BEST)
            # CheckMethod.CHECK_2D
        """
        assert isinstance(mask_algo, MaskAlgo), \
               "mask_algo should be MaskAlgo type"
        if mask_algo == MaskAlgo.MASK_1D:
            return CheckMethod.CHECK_1D
        else:
            return CheckMethod.CHECK_2D


def calculate_density(x):
    r"""
    Return the density of the input tensor.

    Args:
        x (nparray): The input tensor.
    Returns:
        float: The density of :attr:`x`.
    Examples:
        .. code-block:: python

          import numpy as np
          import paddle.static.sparsity as sparsity

          x = np.array([[0, 1, 3, 0],
                        [1, 1, 0, 1]])
          sparsity.calculate_density(x) # 0.625
    """
    x_flattened = x.flatten()
    return float(np.nonzero(x_flattened)[0].size) / x_flattened.size


def _reshape_1d(mat, m):
    r"""
    Reshape the input 2D matrix to shape (-1, m).
    If the second dimension of :attr:`mat` is not a multiples of :attr:`m`, 
    then this function would pad the remainder with 0 before reshaping.

    .. math::

        remainder = mat.shape[1] % m

    Args:
        mat (nparray): The input 2D matrix.
        m (int): The second dimension of reshaped matrix.
    Returns:
        tuple: A pair of the reshaped and padded matrix and the shape of padded matrix (non-reshaping).
    """
    assert len(mat.shape) == 2, "The input mat should be a 2D matrix!"

    remainder = mat.shape[1] % m
    if mat.shape[1] % m > 0:
        mat_padded = np.zeros((mat.shape[0], mat.shape[1] + (m - remainder)))
        mat_padded[:, :mat.shape[1]] = mat
        shape = mat_padded.shape
        return mat_padded.reshape(-1, m), shape
    else:
        return mat.reshape(-1, m), mat.shape


def check_mask_1d(mat, n, m):
    r"""
    Check if every row of the input matrix :attr:`mat` is in 1D `n:m` sparse pattern.
    This function would pad the second dimension of :attr:`mat` by zero 
    to be a multiples of :attr:`m` if necessary.

    1D `n:m` sparse pattern: At least :attr:`n` zeros in every :math:`1 \times m` block.

    Args:
        mat (nparray): The input matrix.
        n (int): n of `n:m` sparse pattern.
        m (int): m of `n:m` sparse pattern.
    Returns:
        bool: True if every row of :attr:`mat` is in 1D n:m sparse pattern, else False.
    Examples:
        .. code-block:: python

          import numpy as np
          import paddle.fluid.contrib.sparsity as sparsity

          x = np.array([[0, 1, 3, 0],
                        [1, 0, 0, 1]])
          sparsity.check_mask_1d(x, 2, 4) # True

          x = np.array([[0, 1, 5, 4],
                        [1, 0, 0, 1]])
          sparsity.check_mask_1d(x, 2, 4) # False

          # x would be padded to shape (2, 8)
          x = np.array([[0, 1, 0, 4, 6],
                        [1, 0, 0, 1, 7]])
          sparsity.check_mask_1d(x, 2, 4) # True
    """
    if len(mat.shape) <= 1:
        mat_flattern, shape = _reshape_1d(mat.reshape(1, mat.shape[0]), m)
    else:
        mat_flattern, shape = _reshape_1d(mat, m)

    for sub_mat in mat_flattern:
        if np.nonzero(sub_mat)[0].size > (m - n):
            return False
    return True


def get_mask_1d(mat, n, m):
    r"""
    Generate 1D `n:m` sparse pattern mask of the input matrix :attr:`mat` 
    in row-directory. This function would pad the second dimension of :attr:`mat` 
    by zero to be a multiples of :attr:`m` before mask generation.

    1D `n:m` sparse pattern: At least :attr:`n` zeros in every :math:`1 \times m` block.

    Args:
        mat (nparray): The input matrix.
        n (int): n of `n:m` sparse pattern.
        m (int): m of `n:m` sparse pattern.
    Returns:
        nparray: The 1D `n:m` sparse mask of :attr:`mat`.
    Examples:
        .. code-block:: python

          import numpy as np
          import paddle.fluid.contrib.sparsity as sparsity

          mat = np.array([[0, 1, 5, 4],
                          [2, 7, 3, 6]])
          mask = sparsity.get_mask_1d(mat, 2, 4)
          # nparray([[0, 0, 1, 1],
          #          [0, 1, 0, 1]])
          sparsity.check_mask_1d(mask, 2, 4) # True
    """
    mat_flattern, shape = _reshape_1d(mat, m)

    mask_flattern = np.ones_like(mat_flattern)
    mask = np.ones_like(mat)
    for i in range(mat_flattern.shape[0]):
        sub_mat = mat_flattern[i]
        min_order_indices = np.argsort(np.absolute(sub_mat))
        mask_flattern[i, min_order_indices[:n].tolist()] = 0
    mask_flattern = mask_flattern.reshape(shape)
    mask[:, :] = mask_flattern[:, :mat.shape[1]]
    return mask


def _reshape_2d(mat, m):
    r"""
    Reshape the input 2D matrix to shape (-1, :math:`m \times m`).
    In each dimension of :attr:`mat`, if it is not a multiples of :attr:`m`, 
    then this function would pad the remainder with 0 before reshaping.

    .. math::

        remainder_0 = mat.shape[0] % m \\
        remainder_1 = mat.shape[1] % m

    Args:
        mat (nparray): The input 2D matrix.
        m (int): The square root of second dimension of reshaped matrix.
    Returns:
        tuple: A pair of the reshaped and padded matrix and the shape of padded matrix (non-reshaping).
    """
    assert len(mat.shape) == 2, "The input mat should be a 2D matrix!"

    remainder_0 = mat.shape[0] % m
    remainder_1 = mat.shape[1] % m

    new_shape = (mat.shape[0] if remainder_0 == 0 \
                 else mat.shape[0] + (m - remainder_0),
                 mat.shape[1] if remainder_1 == 0 \
                 else mat.shape[1] + (m - remainder_1))
    mat_padded = np.zeros(new_shape)
    mat_padded[:mat.shape[0], :mat.shape[1]] = mat

    mat_flattern = np.empty(new_shape).reshape(-1, m * m)
    curr_idx = 0
    for row_start in range(0, mat_padded.shape[0], m):
        row_end = row_start + m
        for col_start in range(0, mat_padded.shape[1], m):
            col_end = col_start + m
            sub_mat = np.squeeze(mat_padded[row_start:row_end, \
                                            col_start:col_end] \
                                            .reshape(-1))
            mat_flattern[curr_idx] = sub_mat
            curr_idx += 1
    return mat_flattern, mat_padded.shape


def check_mask_2d(mat, n, m):
    r"""
    Check if every :math:`m \times m` block of the input matrix :attr:`mat` is in 2D `n:m` sparse pattern.
    This function would pad each dimension of :attr:`mat` by zero to be a multiples of 
    :attr:`m` if necessary.

    2D `n:m` sparse pattern: At least :math:`n \times n` zeros in every :math:`m \times m` block 
    under the constraint of at least :attr:`n` zeros for each row and column.

    Args:
        mat (nparray): The input matrix.
        n (int): n of `n:m` sparse pattern.
        m (int): m of `n:m` sparse pattern.
    Returns:
        bool: True if  every :math:`m \times m` block of the input matrix :attr:`mat` is in 2D `n:m` sparse pattern, else False.
    Examples:
        .. code-block:: python

          import numpy as np
          import paddle.fluid.contrib.sparsity as sparsity

          x = np.array([[0, 8, 9, 0],
                        [9, 0, 0, 10],
                        [5, 0, 0, 6],
                        [0, 4, 6, 0]])
          sparsity.check_mask_2d(x, 2, 4) # True

          x = np.array([[0, 8, 0, 9],
                        [9, 0, 0, 10],
                        [0, 5, 0, 6],
                        [0, 4, 6, 0]])
          sparsity.check_mask_2d(x, 2, 4) # False

          # x would be padded to shape (8, 8)
          x = np.array([[0, 8, 0, 9],
                        [9, 0, 7, 0],
                        [0, 5, 0, 6],
                        [3, 0, 6, 0],
                        [1, 1, 0, 1]])
          sparsity.check_mask_2d(x, 2, 4) # True
    """
    mat_padded, shape = _reshape_2d(mat, m)
    for sub_mat in mat_padded:
        sub_mask = np.absolute(np.squeeze(sub_mat.reshape(m, m))) > 0
        if (np.sum(np.sum(sub_mask, axis=1) > (m-n)) != 0) and \
            (np.sum(np.sum(sub_mask, axis=0) > (m-n)) != 0):
            return False
    return True


def get_mask_2d_greedy(mat, n, m):
    r"""
    Greedily generate 2D `n:m` sparse pattern mask of the input matrix :attr:`mat`. 
    This function would pad each dimension of :attr:`mat` by zero to be a multiples of :attr:`m` before mask generation.

    2D `n:m` sparse pattern: At least :math:`n \times n` zeros in every :math:`m \times m` block 
    under the constraint of at least :attr:`n` zeros for each row and column.
    Greedily generating: For each :math:`m \times m` block, selecting values to keep in descent order.

    Args:
        mat (nparray): The input matrix.
        n (int): n of `n:m` sparse pattern.
        m (int): m of `n:m` sparse pattern.
    Returns:
        nparray: The 2D `n:m` sparse mask of :attr:`mat`.
    Examples:
        .. code-block:: python

          import numpy as np
          import paddle.fluid.contrib.sparsity as sparsity

          mat = np.array([[9, 8, 3, 7],
                          [9, 2, 1, 10],
                          [5, 1, 3, 6],
                          [2, 4, 6, 1]])
          mask = sparsity.get_mask_2d_greedy(mat, 2, 4)
          # nparray([[1. 1. 0. 0.]
          #          [1. 0. 0. 1.]
          #          [0. 0. 1. 1.]
          #          [0. 1. 1. 0.]])
          sparsity.check_mask_2d(mask, 2, 4) # True
    """
    mat_padded, shape = _reshape_2d(mat, m)
    mask_padded = np.zeros_like(mat_padded).reshape(-1, m, m)

    for idx in range(len(mat_padded)):
        sub_mat = np.absolute(np.squeeze(mat_padded[idx]))
        sub_mask = np.squeeze(mask_padded[idx])

        min_order_1d_indices = np.argsort(sub_mat)
        min_order_2d_indices = [(int(x / m), x % m)
                                for x in min_order_1d_indices]
        row_counter = collections.Counter()
        col_counter = collections.Counter()

        for i in range(len(min_order_1d_indices) - 1, -1, -1):
            matrix_entry = min_order_2d_indices[i]
            if (row_counter[matrix_entry[0]] == n) or \
               (col_counter[matrix_entry[1]] == n):
                continue

            sub_mask[matrix_entry[0], matrix_entry[1]] = 1.0
            row_counter[matrix_entry[0]] += 1
            col_counter[matrix_entry[1]] += 1

    mask = np.empty(shape)
    curr_idx = 0
    for row_start in range(0, shape[0], m):
        row_end = row_start + m
        for col_start in range(0, shape[1], m):
            col_end = col_start + m
            mask[row_start:row_end, col_start:col_end] = mask_padded[curr_idx]
            curr_idx += 1
    return mask[:mat.shape[0], :mat.shape[1]]


_valid_2d_patterns_lock = threading.Lock()
_valid_2d_patterns = {}


def _compute_valid_2d_patterns(n, m):
    r"""
    Compute all vaild 2D `n:m` sparse patterns.

    2D `n:m` sparse pattern: At least :math:`n \times n` zeros in every :math:`m \times m` block 
    under the constraint of at least :attr:`n` zeros for each row and column.

    Args:
        n (int): n of `n:m` sparse pattern.
        m (int): m of `n:m` sparse pattern.
    Returns:
        dictionary: A dictionary with key: *m_n* (string) and value: all vaild 2D `n:m` sparse patterns.
    """
    global _valid_2d_patterns_lock
    global _valid_2d_patterns

    valid_key = '{}_{}'.format(m, n)
    if valid_key in _valid_2d_patterns:
        return _valid_2d_patterns[valid_key]
    else:
        patterns = np.zeros(m)
        patterns[:n] = 1
        patterns = list(set(permutations(patterns.tolist())))
        patterns = patterns + patterns
        patterns = np.asarray(list(set(permutations(patterns, m))))

        valid = ((patterns.sum(axis=1) <= n).sum(axis=1) == m
                 ).nonzero()[0].reshape(-1)
        valid_patterns = np.empty((valid.shape[0], m, m))
        valid_patterns[:] = patterns[valid[:]]

        _valid_2d_patterns_lock.acquire()
        _valid_2d_patterns[valid_key] = valid_patterns
        _valid_2d_patterns_lock.release()

        return valid_patterns


def get_mask_2d_best(mat, n, m):
    r"""
    Generate 2D `n:m` sparse pattern mask of the input matrix :attr:`mat` 
    to form sparse matrix with maximun L1 norm .This function would pad each 
    dimension of :attr:`mat` by zero to be a multiples of :attr:`m` before mask generation.

    2D `n:m` sparse pattern: At least :math:`n \times n` zeros in every :math:`m \times m` block 
    under the constraint of at least :attr:`n` zeros for each row and column.

    *Note*: L1 norm of sparse matrix from `Best` API is greater than or equal to the one from `Greedy`.

    Args:
        mat (nparray): The input matrix.
        n (int): n of `n:m` sparse pattern.
        m (int): m of `n:m` sparse pattern.
    Returns:
        nparray: The 1D `n:m` sparse mask of :attr:`mat`.
    Examples:
        .. code-block:: python

          import numpy as np
          import paddle.fluid.contrib.sparsity as sparsity

          mat = np.array([[2, 8, 9, 9],
                          [9, 1, 3, 9],
                          [5, 6, 3, 9],
                          [2, 4, 6, 9]])
          mask_greedy = sparsity.get_mask_2d_greedy(mat, 2, 4)
          mask_best = sparsity.get_mask_2d_best(mat, 2, 4)
          print("L1 norm of `greedy` sparse matrix", np.multiply(mat, mask_greedy).sum()) # 56
          print("L1 norm of `best` sparse matrix", np.multiply(mat, mask_best).sum()) # 61
    """
    patterns = _compute_valid_2d_patterns(n, m)

    mat_flattern, shape = _reshape_2d(mat, m)
    mask_flattern = np.ones_like(mat_flattern).reshape(-1, m, m)
    pmax = np.argmax(
        np.matmul(mat_flattern, patterns.reshape(patterns.shape[0], m * m).T),
        axis=1)

    mask_flattern[:] = patterns[pmax[:]]
    mask = np.empty(shape)

    curr_idx = 0
    for row_start in range(0, shape[0], m):
        row_end = row_start + m
        for col_start in range(0, shape[1], m):
            col_end = col_start + m
            mask[row_start:row_end, col_start:col_end] = mask_flattern[curr_idx]
            curr_idx += 1
    return mask[:mat.shape[0], :mat.shape[1]]


def create_mask(tensor, func_name=MaskAlgo.MASK_1D, n=2, m=4):
    r"""
    Create `n:m` sparse pattern mask of the input tensor via function given by :attr:`func_name`.
    Currently only support tensor with dimension less than or equal to 4.

    Args:
        tensor (nparray): The input tensor.
        func_name (MaskAlgo, optional): The function name to generate spase mask. Default is `MaskAlgo.MASK_1D`. All options please refer to `MaskAlgo`.
        n (int, optional): n of `n:m` sparse pattern. Default is 2.
        m (int, optional): m of `n:m` sparse pattern. Default is 4.
    Returns:
        nparray: The `n:m` sparse mask of :attr:`tensor` generated by :attr:`func_name`.
    Examples:
        .. code-block:: python

          import numpy as np
          import paddle.fluid.contrib.sparsity as sparsity

          tensor = np.array([[2, 8, 9, 9],
                             [9, 1, 3, 9],
                             [5, 6, 3, 9],
                             [2, 4, 6, 9]])
          mask_1d = sparsity.create_mask(tensor, func_name=sparsity.MaskAlgo.MASK_1D)
          # nparray([[0 0 1 1],
          #          [1 0 0 1],
          #          [0 1 0 1],
          #          [0 0 1 1]])
          mask_2d = sparsity.create_mask(tensor, func_name=sparsity.MaskAlgo.MASK_2D_BEST)
          # nparray([[0 1 1 0],
          #          [1 0 0 1],
          #          [1 1 0 0],
          #          [0 0 1 1]])
    """
    shape = tensor.shape
    dtype = tensor.dtype
    t = tensor.astype(float)

    assert isinstance(func_name, MaskAlgo), \
           "func_name argumet of create_mask is only accepted as type MaskAlgo. " \
           "But got {}".format(type(func_name))
    func = getattr(sys.modules[__name__], func_name.value, None)
    if len(shape) == 1:
        t = t.reshape(1, shape[0])
    elif len(shape) == 2:
        t = t.reshape(shape[0], shape[1])
    elif len(shape) == 3:
        t = t.reshape(shape[0] * shape[1], shape[2])
    # 4d-tensor conv (out, in, h, w) -> (out, in*h*w) in GemmConvKernel Op
    elif len(shape) == 4:
        t = t.reshape(shape[0], shape[1] * shape[2] * shape[3])
    else:
        raise ValueError("The dimension of input tensor is not supported in create_mask, " \
                         "Only dimension < 4 is supported but got {}".format(len(shape)))

    mask = func(t, n=n, m=m)
    return mask.reshape(shape).astype(dtype)


def check_sparsity(tensor, func_name=CheckMethod.CHECK_1D, n=2, m=4):
    r"""
    Check if input tensor is in `n:m` sparse pattern via function given by :attr:`func_name`.
    Currently only support tensor with dimension less than or equal to 4.

    Args:
        tensor (nparray): The input tensor.
        func_name (CheckMethod, optional): The function name to generate spase mask. Default is `CheckMethod.CHECK_1D`. All options please refer to `CheckMethod`.
        n (int, optional): n of `n:m` sparse pattern. Default is 2.
        m (int, optional): m of `n:m` sparse pattern. Default is 4.
    Returns:
        bool: True if tensor pass checking of function given by :attr:`func_name`, else False.
    Examples:
        .. code-block:: python

          import numpy as np
          import paddle.fluid.contrib.sparsity as sparsity

          tensor = np.array([[2, 8, 9, 9],
                             [9, 1, 3, 9],
                             [5, 6, 3, 9],
                             [2, 4, 6, 9]])
          mask_1d = sparsity.create_mask(tensor, func_name=sparsity.MaskAlgo.MASK_1D)
          # nparray([[0 0 1 1],
          #          [1 0 0 1],
          #          [0 1 0 1],
          #          [0 0 1 1]])
          sparsity.check_sparsity(mask_1d, func_name=sparsity.CheckMethod.CHECK_1D) # True
          sparsity.check_sparsity(mask_1d, func_name=sparsity.CheckMethod.CHECK_2D) # False
    """
    shape = tensor.shape
    t = tensor.astype(float)

    assert type(func_name) == CheckMethod, \
           "func_name argumet of check_sparsity is only accepted as type CheckMethod. " \
           "But got {}".format(type(func_name))
    func = getattr(sys.modules[__name__], func_name.value, None)
    if len(shape) == 1:
        t = t.reshape(1, shape[0])
    elif len(shape) == 2:
        t = t.reshape(shape[0], shape[1])
    elif len(shape) == 3:
        t = t.reshape(shape[0] * shape[1], shape[2])
    # 4d-tensor conv (out, in, h, w) -> (out, in*h*w) in GemmConvKernel Op
    elif len(shape) == 4:
        t = t.reshape(shape[0], shape[1] * shape[2] * shape[3])
    else:
        raise ValueError("The dimension of input tensor is not supported in create_mask, " \
                         "Only dimension < 4 is supported but got {}".format(len(shape)))

    return func(t, n=n, m=m)
