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

######
from functools import reduce

import paddle
from paddle import framework
from ...utils.log_util import logger


def _is_trainable(param: paddle.Tensor) -> bool:
    return not param.stop_gradient


class DygraphShardingOptimizer(object):
    """
    A wrapper for Sharding Optimizer in Dygraph. 

    .. warning: DygraphShardingOptimizer is experimental and subject to change.

    .. ZeRO: https://arxiv.org/abs/1910.02054

    """

    # TODO (JZ-LIANG) 
    # TO support following featrues in future:
    # 1. fused update parameter sync
    # 2. parameters_groups
    # 3. dynamic trainable params, which is the case bewteen pretraining and finetuning
    # 4. option to choose fuse comm (more GPU MEM need) or un-fuse comm

    def __init__(
            self,
            hcg,
            user_defined_strategy,
            params,
            inner_optimizer_class,
            **inner_optimizer_kargs, ):

        if not isinstance(params, list):
            raise TypeError(
                "`parameters` argument given to the DygraphShardingOptimizer should be "
                "an iterable of paddle Tensors, but got argument type is `{}`.".
                format(type(params)))
        self._parameter_list = params
        self._reference_is_trainable_params = list(
            map(_is_trainable, self._parameter_list))

        self._inner_optimizer_class = inner_optimizer_class
        self._inner_optimizer_kargs = inner_optimizer_kargs

        # sharding parallel information
        # TODO better way to get the hcg & user_defined_strategy
        self._hcg = hcg
        self._user_defined_strategy = user_defined_strategy
        self._sharding_world_size = self._hcg.get_sharding_parallel_world_size()
        self._sharding_rank = self._hcg.get_sharding_parallel_rank()

        # logic partitioning
        self._build_sharding_mapping()

        # actually create opt ops
        self._buid_inner_optimizer()

    def clear_grad(self):
        """
        should clear grad for all parameters in model
        """
        for p in self._parameter_list:
            if not p.stop_gradient:
                p.clear_gradient()

    def _build_sharding_mapping(self):

        self._rank2params = self._partition_parameters()
        self._param2rank = self._map_param_to_rank()

    def _partition_parameters(self):
        """
        Partitions parameters among sharding ranks.

        Return:
        Dict[int, List] 
        """
        # TODO(JZ-LIANG) support multiple partition methods
        # method1: greedy even but unorder
        # method2: roughly even with oreder

        mapping = {}
        for rank_ in range(self._sharding_world_size):
            mapping[rank_] = []
        sizes = [0] * self._sharding_world_size
        for param in self._parameter_list:
            rank = sizes.index(min(sizes))
            mapping[rank].append(param)
            numel = reduce(lambda x, y: x * y, param.shape)
            assert numel > 0, "param [{}] should larger than 0, but it is [{}]".format(
                param.name, numel)
            sizes[rank] += numel

        return mapping

    def _map_param_to_rank(self):
        """
        mapping parameters to the shard which holds it.

        Return:
        Dict[str, int] 
        """
        mapping = {}
        for rank, params in self._rank2params.items():
            for param in params:
                mapping[param.name] = rank
        return mapping

    def _buid_inner_optimizer(self):
        # we rely on the inner opt to determine whether a parameter is stop_gradient or not:
        # create moment
        # update related ops: clip, regular, opt  
        self._inner_optimizer = self._inner_optimizer_class(
            parameters=self._rank2params[self._sharding_rank],
            **self._inner_optimizer_kargs)

    def _sharding_sync_parameters(self):
        """
        sync parameter across sharding group
        """
        # TODO speed up this functional

        logger.debug("sharding start sync parameters")
        with framework.no_grad():
            # TODO detach not need (?)
            for rank, params in self._rank2params.items():
                for param in params:
                    paddle.distributed.broadcast(
                        param,
                        # the collective API need src rank to be the global rank id 
                        # instead of the relative logic rank id within group 
                        src=self._hcg.get_sharding_parallel_group().ranks[rank],
                        group=self._hcg.get_sharding_parallel_group(),
                        use_calc_stream=True)

    def _update_trainable(self):
        """
        allow user to update trainable parameters list during training
        """
        raise NotImplementedError

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameters=None,
                 no_grad_set=None):

        # NOTE in dygraph mode, the only different between step and minimize is that minimize 
        # allow user to customize the parameters for updating on each step

        input_param_names = set([param.name for param in parameters])
        parameters = list(
            filter(lambda x: x.name in input_param_names, self._rank2params[
                self._sharding_rank]))
        result = self._inner_optimizer.minimize(loss, startup_program,
                                                parameters, no_grad_set)

        # sync parameters accross sharding ranks
        self._sharding_sync_parameters()

        return result

    def step(self):
        # TODO Check whether the model trainable param changed and update state accordingly

        # actually updating
        self._inner_optimizer.step()

        # sync parameters accross sharding ranks
        self._sharding_sync_parameters()

    # TODO is it a good way to make _grad_clip a property
    @property
    def _grad_clip(self):
        assert self._inner_optimizer is not None, "inner opt of sharding is not initiliazed."
        return self._inner_optimizer._grad_clip

    def __getattr__(self, item):
        return getattr(self._inner_optimizer, item)
