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

from paddle.fluid.layer_helper import LayerHelper, in_dygraph_mode
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle import _C_ops

__all__ = []


def segment_sum(data, segment_ids, name=None):
    r"""
    Segment Sum Operator.

    This operator sums the elements of input `data` which with
    the same index in `segment_ids`.
    It computes a tensor such that $out_i = \\sum_{j} data_{j}$
    where sum is over j such that `segment_ids[j] == i`.

    Args:
        data (Tensor): A tensor, available data type float32, float64.
        segment_ids (Tensor): A 1-D tensor, which have the same size
                            with the first dimension of input data. 
                            Available data type is int32, int64.
        name (str, optional): Name for the operation (optional, default is None). 
                            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
       output (Tensor): the reduced result.

    Examples:

        .. code-block:: python

            import paddle
            data = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype='float32')
            segment_ids = paddle.to_tensor([0, 0, 1], dtype='int32')
            out = paddle.incubate.segment_sum(data, segment_ids)
            #Outputs: [[4., 4., 4.], [4., 5., 6.]]

    """
    if in_dygraph_mode():
        out, tmp = _C_ops.segment_pool(data, segment_ids, 'pooltype', "SUM")
        return out

    check_variable_and_dtype(data, "X", ("float32", "float64"), "segment_pool")
    check_variable_and_dtype(segment_ids, "SegmentIds", ("int32", "int64"),
                             "segment_pool")

    helper = LayerHelper("segment_sum", **locals())
    out = helper.create_variable_for_type_inference(dtype=data.dtype)
    summed_ids = helper.create_variable_for_type_inference(dtype=data.dtype)
    helper.append_op(
        type="segment_pool",
        inputs={"X": data,
                "SegmentIds": segment_ids},
        outputs={"Out": out,
                 "SummedIds": summed_ids},
        attrs={"pooltype": "SUM"})
    return out


def segment_mean(data, segment_ids, name=None):
    r"""
    Segment mean Operator.

    Ihis operator calculate the mean value of input `data` which
    with the same index in `segment_ids`.
    It computes a tensor such that $out_i = \\frac{1}{n_i}  \\sum_{j} data[j]$
    where sum is over j such that 'segment_ids[j] == i' and $n_i$ is the number
    of all index 'segment_ids[j] == i'.

    Args:
        data (tensor): a tensor, available data type float32, float64.
        segment_ids (tensor): a 1-d tensor, which have the same size 
                            with the first dimension of input data. 
                            available data type is int32, int64.
        name (str, optional): Name for the operation (optional, default is None). 
                            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
       output (Tensor): the reduced result.

    Examples:

        .. code-block:: python

            import paddle
            data = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype='float32')
            segment_ids = paddle.to_tensor([0, 0, 1], dtype='int32')
            out = paddle.incubate.segment_mean(data, segment_ids)
            #Outputs: [[2., 2., 2.], [4., 5., 6.]]

    """
    if in_dygraph_mode():
        out, tmp = _C_ops.segment_pool(data, segment_ids, 'pooltype', "MEAN")
        return out

    check_variable_and_dtype(data, "X", ("float32", "float64"), "segment_pool")
    check_variable_and_dtype(segment_ids, "SegmentIds", ("int32", "int64"),
                             "segment_pool")

    helper = LayerHelper("segment_mean", **locals())
    out = helper.create_variable_for_type_inference(dtype=data.dtype)
    summed_ids = helper.create_variable_for_type_inference(dtype=data.dtype)
    helper.append_op(
        type="segment_pool",
        inputs={"X": data,
                "SegmentIds": segment_ids},
        outputs={"Out": out,
                 "SummedIds": summed_ids},
        attrs={"pooltype": "MEAN"})
    return out


def segment_min(data, segment_ids, name=None):
    r"""
    Segment min operator.

    This operator calculate the minimum elements of input `data` which with
    the same index in `segment_ids`.
    It computes a tensor such that $out_i = \\min_{j} data_{j}$
    where min is over j such that `segment_ids[j] == i`.

    Args:
        data (tensor): a tensor, available data type float32, float64.
        segment_ids (tensor): a 1-d tensor, which have the same size
                            with the first dimension of input data. 
                            available data type is int32, int64.
        name (str, optional): Name for the operation (optional, default is None). 
                            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
       output (Tensor): the reduced result.

    Examples:

        .. code-block:: python

            import paddle
            data = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype='float32')
            segment_ids = paddle.to_tensor([0, 0, 1], dtype='int32')
            out = paddle.incubate.segment_min(data, segment_ids)
            #Outputs:  [[1., 2., 1.], [4., 5., 6.]]

    """
    if in_dygraph_mode():
        out, tmp = _C_ops.segment_pool(data, segment_ids, 'pooltype', "MIN")
        return out

    check_variable_and_dtype(data, "X", ("float32", "float64"), "segment_pool")
    check_variable_and_dtype(segment_ids, "SegmentIds", ("int32", "int64"),
                             "segment_pool")

    helper = LayerHelper("segment_min", **locals())
    out = helper.create_variable_for_type_inference(dtype=data.dtype)
    summed_ids = helper.create_variable_for_type_inference(dtype=data.dtype)
    helper.append_op(
        type="segment_pool",
        inputs={"X": data,
                "SegmentIds": segment_ids},
        outputs={"Out": out,
                 "SummedIds": summed_ids},
        attrs={"pooltype": "MIN"})
    return out


def segment_max(data, segment_ids, name=None):
    r"""
    Segment max operator.

    This operator calculate the maximum elements of input `data` which with
    the same index in `segment_ids`.
    It computes a tensor such that $out_i = \\max_{j} data_{j}$
    where max is over j such that `segment_ids[j] == i`.

    Args:
        data (tensor): a tensor, available data type float32, float64.
        segment_ids (tensor): a 1-d tensor, which have the same size
                            with the first dimension of input data. 
                            available data type is int32, int64.
        name (str, optional): Name for the operation (optional, default is None). 
                            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
       output (Tensor): the reduced result.

    Examples:

        .. code-block:: python

            import paddle
            data = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype='float32')
            segment_ids = paddle.to_tensor([0, 0, 1], dtype='int32')
            out = paddle.incubate.segment_max(data, segment_ids)
            #Outputs: [[3., 2., 3.], [4., 5., 6.]]

    """
    if in_dygraph_mode():
        out, tmp = _C_ops.segment_pool(data, segment_ids, 'pooltype', "MAX")
        return out

    check_variable_and_dtype(data, "X", ("float32", "float64"), "segment_pool")
    check_variable_and_dtype(segment_ids, "SegmentIds", ("int32", "int64"),
                             "segment_pool")

    helper = LayerHelper("segment_max", **locals())
    out = helper.create_variable_for_type_inference(dtype=data.dtype)
    summed_ids = helper.create_variable_for_type_inference(dtype=data.dtype)
    helper.append_op(
        type="segment_pool",
        inputs={"X": data,
                "SegmentIds": segment_ids},
        outputs={"Out": out,
                 "SummedIds": summed_ids},
        attrs={"pooltype": "MAX"})
    return out
