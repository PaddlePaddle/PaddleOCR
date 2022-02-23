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
import paddle.nn as nn
from paddle.fluid.framework import in_dygraph_mode
from paddle.distributed.fleet import fleet

__all__ = [
    'guard',
    'ParallelEmbedding',
    'ColumnParallelLiner',
    'RowParallelLiner',
]


def guard(device):
    def decorator(Layer):
        class WrapperClass(Layer):
            def __init__(self, *args, **kw):
                with paddle.static.device_guard(device):
                    print("Init {} on {}".format(Layer.__name__, device))
                    super().__init__(*args, **kw)

            def forward(self, *args, **kw):
                with paddle.static.device_guard(device):
                    print("Forward {} on {}".format(Layer.__name__, device))
                    return super().forward(*args, **kw)

        return WrapperClass

    return decorator


class ParallelEmbedding(nn.Layer):
    """
    Parallel Embedding.

    Args:
        num_embeddings (int):
            The size of embedding dictionary which dictates the maximum value of the input id.
        embedding_dim (int):
            The dimensions of each embedding vector.
        rank (int):
            The rank of the current part, which determines the start index of the vocab.
        world_size (int):
            The number of trainers.
        weight_attr (Tensor, optional):
            Specify the weight parameter property, including the initialization method.
            Defaults to None which means the default weight parameter property will be used.
        name (str, optional):
            Normally there is no need for user to set this property.
            Defaults to None.
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 rank,
                 world_size,
                 weight_attr=None,
                 name=None):
        super(ParallelEmbedding, self).__init__()
        self.rank = rank
        self.world_size = world_size
        self.num_embeddings = num_embeddings
        self.is_mp = (self.world_size > 1)

        assert num_embeddings % self.world_size == 0, \
            "The length of the vocabulary must be divisible by the parallelism degree of MP"

        per_part_size = num_embeddings // self.world_size

        self.vocab_start_index = self.rank * per_part_size
        self._dtype = self._helper.get_default_dtype()
        self._size = [per_part_size, embedding_dim]
        self._weight_attr = weight_attr
        self._name = name

        self.weight = self.create_parameter(
            attr=self._weight_attr,
            shape=self._size,
            dtype=self._dtype,
            is_bias=False)
        self.weight.is_distributed = True

        startup_block = paddle.static.default_startup_program().global_block()
        main_block = paddle.static.default_main_program().global_block()
        startup_block.vars[self.weight.name].is_distributed = True
        main_block.vars[self.weight.name].is_distributed = True

    def forward(self, x):
        """
        Args:
            x (Tensor):
                A Tensor contains the id information.
                Its data type should be int32 or int64, and the value of the input id should be in [0, weight.shape[0]] .

        Returns:
            Tensor: Returns the embedding Tensor mapped by x.
        """
        if self.is_mp:
            output_parallel = paddle.distributed.collective._c_lookup_table(
                self.weight,
                x,
                start_index=self.vocab_start_index,
                name=self._name)
            output = paddle.distributed.collective._mp_allreduce(
                output_parallel,
                group=None,
                use_calc_stream=True,
                use_model_parallel=True)
        else:
            output = paddle.nn.functional.embedding(
                x,
                weight=self.weight,
                padding_idx=None,
                sparse=False,
                name=self._name)
        return output


class ColumnParallelLiner(nn.Layer):
    """
    Parallel Linear, axis=1.

    Args:
        size (int):
            The size of embedding vector.
        num_partitions (int, optional):
            The number of parts within a model parallel group. Defaults to 1.
        gather_out (bool, optional):
            Whether to gather the output tensor. Defaults to True.
        param_attr (Tensor, optional):
            Specify the parameter property, including the initialization method.
            Defaults to None which means the default parameter property will be used.
        bias_attr (Tensor, optional):
            Specify the bias property.
            Defaults to None which means the default parameter property will be used.
        name (str, optional):
            Normally there is no need for user to set this property.
            Defaults to None.

    """

    def __init__(self,
                 size,
                 num_partitions=1,
                 gather_out=True,
                 param_attr=None,
                 bias_attr=None,
                 name=None):
        super().__init__()

        if in_dygraph_mode():
            rank = paddle.distributed.get_rank()
            nranks = paddle.distributed.get_world_size()
        else:
            assert fleet._role_maker, ("To use paddle.distributed.split, "
                                       "you must call fleet.init() firstly.")
            rank = fleet.worker_index()
            nranks = fleet.worker_num()

        # rank within a model parallel group
        inner_rank = rank % num_partitions
        self.gather_out = gather_out

        assert size[1] % num_partitions == 0, (
            "Number of column of the weight for linear ({}) must be"
            " divisible by num_partitions ({})".format(size[1], num_partitions))
        self.per_part_size = size[1] // num_partitions
        linear_size = (size[0], self.per_part_size)

        num_rows, num_cols = linear_size

        if not name:
            name = "fc_by_col_rank_%d" % inner_rank
        else:
            name = name + "_by_col_rank_%d" % inner_rank

        self.linear = paddle.nn.Linear(
            num_rows,
            num_cols,
            weight_attr=param_attr,
            bias_attr=bias_attr,
            name=name)

        weight = self.linear.weight
        weight.is_distributed = True
        # alias for weight tensor
        self.weight = self.linear.weight

        startup_block = paddle.static.default_startup_program().global_block()
        main_block = paddle.static.default_main_program().global_block()
        startup_block.vars[weight.name].is_distributed = True
        main_block.vars[weight.name].is_distributed = True
        # set is_distributed for splited bias
        # if a linear layer is splited by col, the bias would also be split into each rank as its weight
        if self.linear._bias_attr != False:
            startup_block.vars[self.linear.bias.name].is_distributed = True
            main_block.vars[self.linear.bias.name].is_distributed = True
            self.bias = self.linear.bias

    def forward(self, x):
        """
        Args:
            x (Tensor):
                The input tensor. Its data type can be int or float.

        Returns:
            Tensor: Returns the embedding Tensor mapped by x.
        """
        group = None
        x = paddle.distributed.collective._c_identity(x, group=group)
        output_parallel = self.linear(x)
        if self.gather_out is False:
            return output_parallel

        return paddle.distributed.collective._concat(
            output_parallel, group=group)


class RowParallelLiner(nn.Layer):
    """
    Parallel Linear, axis=0.

    Args:
        size (int):
            The size of embedding vector.
        num_partitions (int, optional):
            The number of parts within a model parallel group. Defaults to 1.
        input_is_parallel (bool, optional):
            Whether the input is parallel. Defaults to `False`.
        param_attr (Tensor, optional):
            Specify the parameter property, including the initialization method.
            Defaults to None which means the default parameter property will be used.
        bias_attr (Tensor, optional):
            Specify the bias property.
            Defaults to None which means the default parameter property will be used.
        name (str, optional):
            Normally there is no need for user to set this property.
            Defaults to None.

    """

    def __init__(self,
                 size,
                 num_partitions=1,
                 input_is_parallel=False,
                 param_attr=None,
                 bias_attr=None,
                 name=None):
        super().__init__()

        if in_dygraph_mode():
            rank = paddle.distributed.get_rank()
            nranks = paddle.distributed.get_world_size()
        else:
            assert fleet._role_maker, ("To use paddle.distributed.split, "
                                       "you must call fleet.init() firstly.")
            rank = fleet.worker_index()
            nranks = fleet.worker_num()

        # rank within a model parallel group
        inner_rank = rank % num_partitions
        self.input_is_parallel = input_is_parallel

        assert size[0] % num_partitions == 0, (
            "Number of rows of the weight for linear ({}) must be"
            " divisible by num_partitions ({})".format(size[0], num_partitions))
        self.per_part_size = size[0] // num_partitions
        linear_size = (self.per_part_size, size[1])

        num_rows, num_cols = linear_size

        if not name:
            name = "fc_by_row_rank_%d" % inner_rank
        else:
            name = name + "_by_row_rank_%d" % inner_rank
        self.linear = paddle.nn.Linear(
            num_rows,
            num_cols,
            weight_attr=param_attr,
            # NOTE(wangxi): row split, bias need add after allreduce
            bias_attr=False,
            name=name)

        weight = self.linear.weight
        weight.is_distributed = True
        # alias for weight tensor
        self.weight = self.linear.weight
        self.bias = self.linear.bias

        startup_block = paddle.static.default_startup_program().global_block()
        main_block = paddle.static.default_main_program().global_block()
        startup_block.vars[weight.name].is_distributed = True
        main_block.vars[weight.name].is_distributed = True
        # set is_distributed for splited bias
        # if a linear layer is splited by row, each rank would hold a complete bias

        if bias_attr is not False:
            self.bias = self.create_parameter(
                shape=[num_cols],
                attr=bias_attr,
                dtype=self._dtype,
                is_bias=True)
        else:
            self.bias = None

    def forward(self, x):
        """
        Args:
            x (Tensor):
                The input tensor. Its data type can be int or float.

        Returns:
            Tensor: Returns the embedding Tensor mapped by x.
        """
        group = None
        if self.input_is_parallel:
            assert x.shape[-1] == self.per_part_size, (
                "The width ({}) of the input "
                "x must be equal to the height ({}) of the weight. Maybe you "
                "should split the input x using paddle.split.".format(
                    x.shape[-1], self.per_part_size))
        else:
            # split last dim
            x = paddle.distributed.collective._c_split(x, group=group)
        output_parallel = self.linear(x)
        output = paddle.distributed.collective._mp_allreduce(
            output_parallel,
            group=group,
            use_calc_stream=True,
            use_model_parallel=True)
        output = output + self.bias if self.bias is not None else output
        return output
