# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except jin compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import six
import numpy as np
import warnings
from collections import OrderedDict
import itertools
import warnings
from contextlib import contextmanager

import paddle
from paddle.fluid import core
from paddle.fluid import framework
from paddle.fluid.dygraph import layers
from paddle.fluid.dygraph import parallel_helper
from paddle.fluid.dygraph import to_variable, no_grad
from paddle.utils import deprecated
from ..layers import collective
from paddle.fluid.dygraph import base as imperative_base
from paddle.fluid.framework import ParamBase

__all__ = ["prepare_context", "ParallelEnv", "DataParallel"]

ParallelStrategy = core.ParallelStrategy


@deprecated(since="2.0.0", update_to="paddle.distributed.init_parallel_env")
def prepare_context(strategy=None):
    '''
    :api_attr: imperative
    '''
    if strategy is None:
        strategy = ParallelStrategy()
        strategy.nranks = Env().nranks
        strategy.local_rank = Env().local_rank
        strategy.trainer_endpoints = Env().trainer_endpoints
        strategy.current_endpoint = Env().current_endpoint
    if strategy.nranks < 2:
        return
    assert framework.in_dygraph_mode() is True, \
        "dygraph.prepare_context should be used with dygraph mode."
    place = framework._current_expected_place()
    assert place is not None, \
        "dygraph.prepare_context should be used in fluid.dygraph.guard(place) guard."
    if not parallel_helper._is_parallel_ctx_initialized():
        if isinstance(place, core.CUDAPlace):
            parallel_helper._set_parallel_ctx(
                core.NCCLParallelContext(strategy, place))
        elif isinstance(place, core.XPUPlace):
            parallel_helper._set_parallel_ctx(
                core.BKCLParallelContext(strategy, place))
        else:
            # TODO(Yancey1989): add Gloo Parallel Context to support CPU parallel computation
            assert ("Only support CUDAPlace or XPUPlace for now.")
        parallel_helper._init_parallel_ctx()
    return strategy


class ParallelEnv(object):
    """
    .. note::
        This API is not recommended, if you need to get rank and world_size, 
        it is recommended to use ``paddle.distributed.get_rank()`` and 
        ``paddle.distributed.get_world_size()`` .

    This class is used to obtain the environment variables required for 
    the parallel execution of ``paddle.nn.Layer`` in dynamic mode.

    The parallel execution in dynamic mode needs to be started using ``paddle.distributed.launch``
    or ``paddle.distributed.spawn`` .

    Examples:
      .. code-block:: python

        import paddle
        import paddle.distributed as dist

        def train():
            # 1. initialize parallel environment
            dist.init_parallel_env()

            # 2. get current ParallelEnv
            parallel_env = dist.ParallelEnv()
            print("rank: ", parallel_env.rank)
            print("world_size: ", parallel_env.world_size)

            # print result in process 1:
            # rank: 1
            # world_size: 2
            # print result in process 2:
            # rank: 2
            # world_size: 2

        if __name__ == '__main__':
            # 1. start by ``paddle.distributed.spawn`` (default)
            dist.spawn(train, nprocs=2)
            # 2. start by ``paddle.distributed.launch``
            # train()
    """

    def __init__(self):
        self._rank = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        self._world_size = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))

        # imperative only support one gpu or xpu
        if core.is_compiled_with_cuda():
            selected_gpus = os.getenv("FLAGS_selected_gpus", "0").split(",")
            self._device_id = int(selected_gpus[0])
        elif core.is_compiled_with_xpu():
            selected_xpus = os.getenv("FLAGS_selected_xpus", "0").split(",")
            self._device_id = int(selected_xpus[0])

        self._trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS",
                                            "").split(",")
        self._current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT", "")
        self._nrings = int(os.getenv("FLAGS_nccl_nrings", "1"))
        assert self._nrings > 0, \
            "nccl_nrings must be an integer greater than 0."
        assert self._nrings < 9, \
            "nccl_nrings should be less than 9, which is enough in most scenarios."

    @property
    def rank(self):
        """
        Rank of current trainer.

        Its value is equal to the value of the environment variable ``PADDLE_TRAINER_ID`` . The default value is 0.

        Examples:
          .. code-block:: python

            # execute this command in terminal: export PADDLE_TRAINER_ID=0
            import paddle.distributed as dist
            
            env = dist.ParallelEnv()
            print("The rank is %d" % env.rank)
            # The rank is 0
        """
        return self._rank

    @property
    def world_size(self):
        """
        The number of trainers (number of processes participating in current job).

        Its value is equal to the value of the environment variable ``PADDLE_TRAINERS_NUM`` . The default value is 1.

        Examples:
          .. code-block:: python

            # execute this command in terminal: export PADDLE_TRAINERS_NUM=4
            import paddle.distributed as dist
            
            env = dist.ParallelEnv()
            print("The world_size is %d" % env.world_size)
            # The world_size is 4
        """
        return self._world_size

    @property
    def device_id(self):
        """
        The ID of selected GPU card for parallel training.

        Its value is equal to the value of the environment variable ``FLAGS_selected_gpus`` . The default value is 0.

        Examples:
          .. code-block:: python

            # execute this command in terminal: export FLAGS_selected_gpus=1
            import paddle.distributed as dist
            
            env = dist.ParallelEnv()
            print("The device id are %d" % env.device_id)
            # The device id are 1
        """
        return self._device_id

    @property
    def current_endpoint(self):
        """
        The endpoint of current trainer, it is in the form of (node IP + port).

        Its value is equal to the value of the environment variable ``PADDLE_CURRENT_ENDPOINT`` . The default value is "".

        Examples:
          .. code-block:: python
            
            # execute this command in terminal: export PADDLE_CURRENT_ENDPOINT=127.0.0.1:6170
            import paddle.distributed as dist
            
            env = dist.ParallelEnv()
            print("The current endpoint are %s" % env.current_endpoint)
            # The current endpoint are 127.0.0.1:6170
        """
        return self._current_endpoint

    @property
    def trainer_endpoints(self):
        """
        The endpoints of all trainer nodes in the task, 
        which are used to broadcast the NCCL ID when NCCL2 is initialized.

        Its value is equal to the value of the environment variable ``PADDLE_TRAINER_ENDPOINTS`` . The default value is "".

        Examples:
          .. code-block:: python

            # execute this command in terminal: export PADDLE_TRAINER_ENDPOINTS=127.0.0.1:6170,127.0.0.1:6171
            import paddle.distributed as dist
            
            env = dist.ParallelEnv()
            print("The trainer endpoints are %s" % env.trainer_endpoints)
            # The trainer endpoints are ['127.0.0.1:6170', '127.0.0.1:6171']
        """
        return self._trainer_endpoints

    @property
    def nrings(self):
        """
        Nrings of current trainer.

        Its value is equal to the value of the environment variable ``FLAGS_nccl_nrings`` . The default value is 1.

        Examples:
          .. code-block:: python

            # execute this command in terminal: export FLAGS_nccl_nrings=1
            import paddle.distributed as dist
            
            env = dist.ParallelEnv()
            print("The nrings is %d" % env.nrings)
            # the number of ring is 1
        """
        return self._nrings

    # [aliases] Compatible with old method names
    local_rank = rank
    nranks = world_size
    dev_id = device_id


# NOTE: [ Compatible ] Originally this class name is `Env`. The semantics of the old class names
# are inaccurate and may confuse users, so replace it with `ParallelEnv`, but to be compatible
# with the old examples, here still need to keep this name.
Env = ParallelEnv


def _build_default_parallel_strategy():
    strategy = ParallelStrategy()
    strategy.nranks = ParallelEnv().nranks
    strategy.local_rank = ParallelEnv().local_rank
    strategy.trainer_endpoints = ParallelEnv().trainer_endpoints
    strategy.current_endpoint = ParallelEnv().current_endpoint
    return strategy


def _coalesce_tensors(var_groups):
    from ..layers import nn
    coalesced_grads_and_grad_vars = []
    for group_id, grad_vars in var_groups.items():
        flattened_vars = []
        g_var_shapes = []
        for g_var in grad_vars:
            g_var_shapes.append(g_var.shape)
            flattened_vars.append(
                nn.reshape(
                    x=g_var, shape=[np.prod(g_var.shape)]))
        coalesced_grad = nn.concat(flattened_vars)
        coalesced_grads_and_grad_vars.append(
            [coalesced_grad, grad_vars, g_var_shapes])
    return coalesced_grads_and_grad_vars


@framework.dygraph_only
def _reshape_inplace(x, shape):
    x_shape = framework._varbase_creator(dtype=x.dtype)
    framework._dygraph_tracer().trace_op(
        type="reshape2",
        inputs={'X': x},
        outputs={'Out': x,
                 'XShape': x_shape},
        attrs={'shape': shape})


@framework.dygraph_only
def _split_tensors(coalesced_grads_and_grad_vars):
    for coalesced_grad, origin_grad_vars, grad_shapes in coalesced_grads_and_grad_vars:
        grad_var_len = [np.prod(g_shape) for g_shape in grad_shapes]
        framework._dygraph_tracer().trace_op(
            type='split',
            inputs={'X': coalesced_grad},
            outputs={'Out': origin_grad_vars},
            attrs={'sections': grad_var_len,
                   'axis': 0})
        for g_var, g_shape in zip(origin_grad_vars, grad_shapes):
            _reshape_inplace(x=g_var, shape=g_shape)
            assert g_var.shape == g_shape


def scale_loss(loss):
    # TODO(liuyuhui) Currently only for xpu. Will be removed in the future.
    if not ParallelEnv().world_size > 1:
        return loss

    loss_scale = to_variable(
        np.array([ParallelEnv().world_size]).astype("float32"))
    loss_scale.stop_gradient = True
    scaled_loss = loss / loss_scale
    return scaled_loss


@imperative_base.no_grad
@framework.dygraph_only
def build_groups(vars, group_size):
    group_idx = 0
    memory_counter = 0
    var_groups = OrderedDict()
    dtype = vars[0].dtype

    for var in vars:
        bytes = np.prod(var.shape) * core.size_of_dtype(var.dtype)
        if memory_counter < group_size and dtype == var.dtype:
            memory_counter += bytes
        else:
            memory_counter = bytes
            dtype = var.dtype
            group_idx += 1
        var_groups.setdefault(group_idx, []).append(var)
    return _coalesce_tensors(var_groups)


@imperative_base.no_grad
@framework.dygraph_only
def sync_params_buffers(model,
                        comm_group=None,
                        src_rank=0,
                        is_model_parallel=False):
    model_vars = []
    for _, param in model.state_dict().items():
        if not isinstance(param, core.VarBase):
            raise TypeError("The data type of '%s' must be Varbase" %
                            param.name)

        # is_distributed param not need to sync when in mp mode
        if isinstance(param, ParamBase):
            if is_model_parallel and param.is_distributed:
                continue

            # NOTE(shenliang03): Support situations that do not require synchronization parameters, 
            # such as moe's expert parameters
            if getattr(param, "no_sync", False):
                continue

        if param.type == core.VarDesc.VarType.VOCAB:
            continue

        model_vars.append(param.detach())
    if len(model_vars) == 0:
        return

    # group size is 128M
    coalesced_vars = build_groups(model_vars, 128 * 1024 * 1024)

    for coalesced_var, _, _ in coalesced_vars:
        paddle.distributed.broadcast(
            coalesced_var, src=src_rank, group=comm_group, use_calc_stream=True)

    for coalesced_var, origin_vars, var_shapes in coalesced_vars:
        var_len = [np.prod(v_shape) for v_shape in var_shapes]
        paddle.fluid.framework._dygraph_tracer().trace_op(
            type='split',
            inputs={'X': coalesced_var},
            outputs={'Out': origin_vars},
            attrs={'sections': var_len,
                   'axis': 0})


class DataParallel(layers.Layer):
    """
    Run the dygraph module with data parallelism.

    Currently, DataParallel class only supports to run the dynamic graph
    with multi-process. 
    
    Now supports two ways to start training:

    1. start by ``paddle.distributed.spawn`` method, for example:

        ``python demo.py`` (spawn need to be called in ``__main__`` method)
    
    2. start by ``paddle.distributed.launch`` module, for example:
    
        ``python -m paddle.distributed.launch --gpus=0,1 demo.py`` .

    And the content of `demo.py` is the code of examples.

    Args:
        layers(Layer): The module that should be executed by data parallel.
        strategy(ParallelStrategy, optional): (deprecated) The strategy of data parallelism, 
            contains environment configuration related to parallel execution. Default: None.
        comm_buffer_size(int, optional):  It limits the memory size(MB) of one buffer  
                                          parameters' gradient which is the input of communication 
                                          calling(e.g NCCLAllReduce). Default: 25.
        last_comm_buffer_size(float, optional): It limits memory size(MB) of last buffer in communication
                                         calling. Making the last communication buffer size small is useful to 
                                         improve performance. Default: 1.
        find_unused_parameters(bool, optional): Whether to traverse the entire backward graph from the
                                                all tensors in the return value of the wrapped model's 
                                                forward function. For parameters not involved in loss 
                                                calculation, their gradients will be marked as ready in 
                                                advance to prepare reduce. Please note that all forward 
                                                outputs derived from the wrapped model parameters must 
                                                participate in the calculation of loss and subsequent 
                                                gradient calculations. If not, serious error will occur.
                                                Note that setting the find_unused_parameters to True 
                                                will affect computing performance. Therefore, if all parameters
                                                are sure to participate in the loss calculation and the 
                                                autograd graph construction, please set it False. Default: False.
            
    Returns:
        Layer: The data paralleled module.

    Examples:

        .. code-block:: python
            :name: dp-example

            # required: distributed
            import paddle
            import paddle.nn as nn
            import paddle.optimizer as opt
            import paddle.distributed as dist

            class LinearNet(nn.Layer):
                def __init__(self):
                    super(LinearNet, self).__init__()
                    self._linear1 = nn.Linear(10, 10)
                    self._linear2 = nn.Linear(10, 1)
                    
                def forward(self, x):
                    return self._linear2(self._linear1(x))

            def train():
                # 1. initialize parallel environment
                dist.init_parallel_env()

                # 2. create data parallel layer & optimizer
                layer = LinearNet()
                dp_layer = paddle.DataParallel(layer)

                loss_fn = nn.MSELoss()
                adam = opt.Adam(
                    learning_rate=0.001, parameters=dp_layer.parameters())

                # 3. run layer
                inputs = paddle.randn([10, 10], 'float32')
                outputs = dp_layer(inputs)
                labels = paddle.randn([10, 1], 'float32')
                loss = loss_fn(outputs, labels)
                
                loss.backward()

                adam.step()
                adam.clear_grad()

            if __name__ == '__main__':
                # 1. start by ``paddle.distributed.spawn`` (default)
                dist.spawn(train, nprocs=2)
                # 2. start by ``paddle.distributed.launch``
                # train()


    .. note::
        ``PyLayer`` is not supported in DataParallel. To solve problems of this kind, 
        it's recommended to skip gradient synchronization among multiple cards by 'no_sync', 
        and manually implement 'all_reduce' before model optimization. There is an example 
        showing specific implemetation processing.

    Examples:

        .. code-block:: python
            :name: dp-pylayer-example

            # required: distributed
            import numpy
            import paddle
            import paddle.distributed as dist
            from paddle.autograd import PyLayer
            from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

            class cus_tanh(PyLayer):
                @staticmethod
                def forward(ctx, x):
                    y = paddle.tanh(x)
                    ctx.save_for_backward(y)
                    return y

                @staticmethod
                def backward(ctx, dy):
                    y, = ctx.saved_tensor()
                    grad = dy * (1 - paddle.square(y))
                    return grad

            class SimpleNet(paddle.nn.Layer):
                def __init__(self):
                    super(SimpleNet, self).__init__()
                    self.linear = paddle.nn.Linear(2, 2)

                def forward(self, inputs):
                    inputs = cus_tanh.apply(inputs)
                    return self.linear(inputs)

            if __name__ == '__main__':
                dist.init_parallel_env()

                model = SimpleNet()
                model = paddle.DataParallel(model)
                opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())

                for step in range(10):
                    x_data = numpy.random.randn(2, 2).astype(numpy.float32)
                    x = paddle.to_tensor(x_data)
                    x.stop_gradient = False

                    # step 1 : skip gradient synchronization by 'no_sync'
                    with model.no_sync():
                        y_pred = model(x)
                        loss = y_pred.mean()
                        loss.backward()

                    # step 2 : fuse + allreduce manually before optimization
                    fused_allreduce_gradients(list(model.parameters()), None)

                    opt.step()
                    opt.clear_grad()

    """

    def __init__(self,
                 layers,
                 strategy=None,
                 comm_buffer_size=25,
                 last_comm_buffer_size=1,
                 find_unused_parameters=False):
        super(DataParallel,
              self).__init__(layers.full_name() + "_data_parallel")

        self._layers = layers
        self.find_unused_parameters = find_unused_parameters
        self.grad_need_sync = True

        # NOTE(chenweihang): The ParallelStrategy here is not strictly a strategy. 
        # It just stores some environment variables, which can be constructed by 
        # ParallelEnv. Here it is set as an optional argument.
        # This parameter is not removed because of compatibility with 1.x writing.
        if strategy is not None:
            self._strategy = strategy
        else:
            self._strategy = _build_default_parallel_strategy()

        if self._strategy.nranks > 1:
            # check the environment
            assert parallel_helper.__parallel_ctx__clz__ is not None, \
            "ParallelContext must be initialized before. You should use init_parallel_env() before" \
            "constructing the DataParallel."

            # sync buffer and params
            # TODO(liuyuhui) Currently not support xpu. xpu is 
            # still broadcasting parameters when calling layer
            if not paddle.is_compiled_with_xpu():
                sync_params_buffers(self._layers)

            self.comm_buffer_size = int(comm_buffer_size * 1024 * 1024)
            # NOTE(shenliang03): We can set environment variables to control 
            # the size of the group, Default: 1MB. The role of this small group is: 
            # when the last group allreduce, the overlap cannot work. Making the 
            # the last group small is useful to improve performance.
            self.last_comm_buffer_size = int(last_comm_buffer_size * 1024 *
                                             1024)
            self.init_reducer()
        else:
            warnings.warn("The program will return to single-card operation. "
                          "Please check 1, whether you use spawn or fleetrun "
                          "to start the program. 2, Whether it is a multi-card "
                          "program. 3, Is the current environment multi-card.")

    def init_reducer(self):
        layers_param = []
        params_set = set()
        for sublayer in self.sublayers():
            for _, param in sublayer.named_parameters(include_sublayers=False):
                if param is None or param in params_set:
                    continue
                params_set.add(param)
                if not isinstance(param, core.VarBase):
                    raise TypeError("The data type of '%s' must be Varbase" %
                                    param.name)
                if param.trainable:
                    layers_param.append((sublayer, param))

        trainable_parameters = [param for _, param in layers_param]

        assert len(trainable_parameters) > 0, \
            "This model does not have any parameters to train, and " \
            "does not need to use DataParallel"

        # NOTE(shenliang03): Here we can only use the attributes to judge whether
        # parameter is sparse(or SelectedRows). The reason is that the sparse message
        # can't be obtained when bp hasn't happened yet. So if layer supports sparse parameter,
        # we should add the layer here like "paddle.nn.layer.common.Embedding".
        def check_layer_sparse(sublayer):
            if isinstance(sublayer, paddle.nn.layer.common.Embedding):
                return sublayer._sparse
            # NOTE(shenliang03):This is for compatibility. If paddle.fluid.dygraph.Embedding 
            # is removed in the future, the check will also be removed here.
            if isinstance(sublayer, paddle.fluid.dygraph.Embedding):
                return sublayer._is_sparse
            return False

        is_sparse_gradient = [
            check_layer_sparse(sublayer) for sublayer, _ in layers_param
        ]

        self.group_indices = core.assign_group_by_size(
            trainable_parameters, is_sparse_gradient,
            [self.last_comm_buffer_size, self.comm_buffer_size])

        self._reducer = core.Reducer(
            trainable_parameters,
            list(reversed(self.group_indices)), is_sparse_gradient,
            parallel_helper.__parallel_ctx__clz__,
            [self.last_comm_buffer_size, self.comm_buffer_size],
            self.find_unused_parameters)

    def _find_varbase(self, obj):
        if isinstance(obj, core.VarBase):
            return [obj]
        if isinstance(obj, (list, tuple)):
            return itertools.chain(*map(self._find_varbase, obj))
        if isinstance(obj, dict):
            return itertools.chain(*map(self._find_varbase, obj.values()))
        return []

    @contextmanager
    def no_sync(self):
        """
        A context manager to stop gradient synchronization. Within no_sync(), 
        gradients of parameters will only be accumulated on model and not 
        synchronized util the first forward-backward out of this context.

        Examples:
            .. code-block:: python

                # required: distributed
                import paddle
                import paddle.nn as nn
                import paddle.distributed as dist

                class SimpleNet(nn.Layer):
                    def __init__(self):
                        super(SimpleNet, self).__init__()
                        self._linear = nn.Linear(10, 1)
                        
                    def forward(self, x):
                        return self._linear(x)

                dist.init_parallel_env()
                model = SimpleNet()
                dp_model = paddle.DataParallel(model)

                inputs_1 = paddle.randn([10, 10], 'float32')
                inputs_2 = paddle.ones([10, 10], 'float32')

                with dp_model.no_sync():
                    # gradients will not be synchronized
                    dp_model(inputs_1).backward()

                # synchronization happens here
                dp_model(inputs_2).backward()

        """
        tmp_grad_need_sync = self.grad_need_sync
        self.grad_need_sync = False
        try:
            yield
        finally:
            self.grad_need_sync = tmp_grad_need_sync

    def forward(self, *inputs, **kwargs):
        outputs = self._layers(*inputs, **kwargs)
        if self._strategy.nranks > 1 and framework._dygraph_tracer(
        )._has_grad and self.grad_need_sync:
            self._reducer.prepare_for_backward(
                list(self._find_varbase(outputs)))
        return outputs

    @deprecated(
        since="2.0.0", reason="This method does not need to be called anymore.")
    def scale_loss(self, loss):
        """
        Deprecated method, now ``scale_loss`` is an empty method,  
        keep this method just for compatibility.
        """
        return loss

    @deprecated(
        since="2.0.0", reason="This method does not need to be called anymore.")
    def apply_collective_grads(self):
        """
        Deprecated method, now ``apply_collective_grads`` is an empty method, 
        keep this method just for compatibility.
        """
        return

    def state_dict(self,
                   destination=None,
                   include_sublayers=True,
                   structured_name_prefix=""):
        '''
        Get all parameters and persistable buffers of current layer and its sub-layers. And set them into a dict

        Parameters:
            destination(dict, optional) : If provide, all the parameters and persistable buffers will be set to this dict . Default: None
            include_sublayers(bool, optional) : If true, also include the parameters and persistable buffers from sublayers. Default: True

        Retruns:
            dict: a dict contains all the parameters and persistable buffers.

        Examples:
            .. code-block:: python

                import paddle
                import paddle.distributed as dist

                dist.init_parallel_env()

                emb = fluid.dygraph.Embedding([10, 10])
                emb = fluid.dygraph.DataParallel(emb)

                state_dict = emb.state_dict()
                paddle.save(state_dict, "paddle_dy.pdparams")

        '''

        return self._layers.state_dict(
            destination=destination,
            include_sublayers=include_sublayers,
            structured_name_prefix=structured_name_prefix)

    @framework.deprecate_stat_dict
    def set_state_dict(self, state_dict, use_structured_name=True):
        '''
        Set parameters and persistable buffers from state_dict. All the parameters and buffers will be reset by the tensor in the state_dict

        Parameters:
            state_dict(dict) : Dict contains all the parameters and persistable buffers.
            use_structured_name(bool, optional) : If true, use structured name as key, otherwise, use parameter or buffer name as key. 
                                                  Default: True
        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle
                import paddle.distributed as dist

                dist.init_parallel_env()

                emb = paddle.nn.Embedding(10, 10)
                emb = fluid.dygraph.DataParallel(emb)

                state_dict = emb.state_dict()
                paddle.save(state_dict, "paddle_dy.pdparams")

                para_state_dict = paddle.load("paddle_dy.pdparams")
                emb.set_state_dict(para_state_dict)

        '''

        self._layers.set_state_dict(
            state_dict, use_structured_name=use_structured_name)

    # [aliases] Compatible with old method names
    set_dict = set_state_dict
    load_dict = set_state_dict
