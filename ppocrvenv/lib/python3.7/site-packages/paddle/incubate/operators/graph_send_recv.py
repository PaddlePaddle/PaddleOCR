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

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle.fluid import core


def graph_send_recv(x, src_index, dst_index, pool_type="sum", name=None):
    r"""

    Graph Learning Send_Recv combine operator.

    This operator is mainly used in Graph Learning domain, and the main purpose is to reduce intermediate memory 
    consumption in the process of message passing. Take `x` as the input tensor, we first use `src_index`
    to gather the corresponding data, and then use `dst_index` to update the corresponding position of output tensor 
    in different pooling types, like sum, mean, max, or min.

    .. code-block:: text

           Given:

           X = [[0, 2, 3],
                [1, 4, 5],
                [2, 6, 7]]

           src_index = [0, 1, 2, 0]

           dst_index = [1, 2, 1, 0]

           pool_type = "sum"

           Then:

           Out = [[0, 2, 3],
                  [2, 8, 10],
                  [1, 4, 5]]

    Args:
        x (Tensor): The input tensor, and the available data type is float32, float64, int32, int64.
        src_index (Tensor): An 1-D tensor, and the available data type is int32, int64.
        dst_index (Tensor): An 1-D tensor, and should have the same shape as `src_index`. 
                            The available data type is int32, int64. 
        pool_type (str): The pooling type of graph_send_recv, including `sum`, `mean`, `max`, `min`.
                         Default value is `sum`.
        name (str, optional): Name for the operation (optional, default is None).
                              For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): The output tensor, should have the same shape and same dtype as input tensor `x`.

    Examples:

        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
            indexes = paddle.to_tensor([[0, 1], [1, 2], [2, 1], [0, 0]], dtype="int32")
            src_index = indexes[:, 0]
            dst_index = indexes[:, 1]
            out = paddle.incubate.graph_send_recv(x, src_index, dst_index, pool_type="sum")
            # Outputs: [[0., 2., 3.], [2., 8., 10.], [1., 4., 5.]]

    """

    if pool_type not in ["sum", "mean", "max", "min"]:
        raise ValueError(
            "pool_type should be `sum`, `mean`, `max` or `min`, but received %s"
            % pool_type)

    if in_dygraph_mode():
        out, tmp = core.ops.graph_send_recv(x, src_index, dst_index,
                                            'pool_type', pool_type.upper())
        return out

    check_variable_and_dtype(x, "X", ("float32", "float64", "int32", "int64"),
                             "graph_send_recv")
    check_variable_and_dtype(src_index, "Src_index", ("int32", "int64"),
                             "graph_send_recv")
    check_variable_and_dtype(dst_index, "Dst_index", ("int32", "int64"),
                             "graph_send_recv")

    helper = LayerHelper("graph_send_recv", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    dst_count = helper.create_variable_for_type_inference(
        dtype="int32", stop_gradient=True)
    helper.append_op(
        type="graph_send_recv",
        inputs={"X": x,
                "Src_index": src_index,
                "Dst_index": dst_index},
        outputs={"Out": out,
                 "Dst_count": dst_count},
        attrs={"pool_type": pool_type.upper()})
    return out
