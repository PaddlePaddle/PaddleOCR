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

import contextlib

import paddle
from paddle.fluid import core
from paddle import _C_ops
from paddle.autograd import PyLayer
from paddle.fluid import framework
from ...utils.recompute import check_recompute_necessary, detach_variable
from ..parallel_layers.random import get_rng_state_tracker

__all__ = []

FLOAT_TYPE_DICT = {
    paddle.float16: "float16",
    paddle.float32: "float32",
    paddle.float64: "float64",
}

PADDLE_TO_NUMBER = {
    paddle.float16: 0,
    paddle.float32: 1,
    paddle.float64: 2,
    paddle.int32: 3,
    paddle.int64: 4
}

NUMBER_TO_DTYPE = {
    0: "float16",
    1: "float32",
    2: "float64",
    3: "int32",
    4: "int64"
}


def is_float_tensor(tensor):
    """Is a float tensor"""
    return tensor.dtype in FLOAT_TYPE_DICT.keys()


def get_tensor_dtype(dtype):
    assert dtype in FLOAT_TYPE_DICT.keys()
    return FLOAT_TYPE_DICT[dtype]


def paddle_2_number(dtype):
    assert dtype in PADDLE_TO_NUMBER.keys()
    return PADDLE_TO_NUMBER[dtype]


def number_2_dtype(number):
    assert number in NUMBER_TO_DTYPE.keys()
    return NUMBER_TO_DTYPE[number]


def get_tensor_bytes(tensor):
    """Get the bytes a tensor occupied."""
    elem_size = None
    if tensor.dtype == paddle.float32:
        elem_size = 4
    elif tensor.dtype == paddle.float64:
        elem_size = 8
    elif tensor.dtype == paddle.int64:
        elem_size = 8
    elif tensor.dtype == paddle.int32:
        elem_size = 4
    elif tensor.dtype == paddle.float16:
        elem_size = 2
    elif tensor.dtype == paddle.int8:
        elem_size = 1
    else:
        raise ValueError("unknown data type: {}".format(tensor.dtype))
    return tensor.numel() * elem_size


_hcg = None
_recompute_offload = False
_recompute_partition = False


def _initialize_recompute_setting(is_offload, is_partition):
    global _recompute_offload, _recompute_partition

    _recompute_offload = is_offload
    _recompute_partition = is_partition


def _initialize_recompute_hcg(hcg):
    global _hcg
    _hcg = hcg


def _all_gather(tensor, group=None, use_calc_stream=True):
    """
    The main difference with paddle.distributed.all_gather: 
    no need to pass in tensor_list, the returned tensor is spliced
    """
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id
    nranks = paddle.distributed.collective._get_global_group(
    ).nranks if group is None else group.nranks
    return _C_ops.c_allgather(tensor, 'use_calc_stream', use_calc_stream,
                              'ring_id', ring_id, 'nranks', nranks)


def _split_activation(tensor):
    global _hcg

    mp_degree = _hcg.get_model_parallel_world_size()
    mp_rank = _hcg.get_model_parallel_rank()
    if mp_degree < 2:
        return tensor

    tensor_numel = paddle.numel(tensor)
    assert tensor_numel != 0, "can't recompute zero element"
    assert tensor_numel % mp_degree == 0, "The capacity of the activation () cannot be divisible by mp_degree()".format(
        tensor_numel, mp_degree)

    # use inplace operation to save memory
    data = tensor.flatten_()

    part_size = tensor_numel // mp_degree
    start = part_size * mp_rank
    end = start + part_size
    return data[start:end]


def _merge_activation(tensor):
    global _hcg
    mp_degree = _hcg.get_model_parallel_world_size()
    mp_rank = _hcg.get_model_parallel_rank()
    mp_group = _hcg.get_model_parallel_group()
    if mp_degree < 2:
        return tensor
    return _all_gather(tensor, group=mp_group)


@contextlib.contextmanager
def _swith_rng_state_tracker(rng_state, tracker):
    orig_cuda_rng_state = paddle.get_cuda_rng_state()
    orig_cuda_rng_tracker = get_rng_state_tracker().get_states_tracker()

    paddle.set_cuda_rng_state(rng_state)
    get_rng_state_tracker().set_states_tracker(tracker)
    try:
        yield
    finally:
        paddle.set_cuda_rng_state(orig_cuda_rng_state)
        get_rng_state_tracker().set_states_tracker(orig_cuda_rng_tracker)


class _HPRecomputeFunction(PyLayer):
    """
    Compared with paddle.distributed.fleet.utils.recompute, there are the following differences:
    1. In order to support PipeLineParallel, the input of recompute is modified to ensure that the input can be tuple type.
    2. Offload support for activation
    3. Support MP segmentation of activation to further reduce cuda memory
    4. Adapt to the random state of MP
    """

    @staticmethod
    def forward(ctx, run_function, all_outputs, *args):
        check_recompute_necessary(args)

        # store for recomputing 
        ctx.run_function = run_function

        # store the rng states
        ctx.fwd_cuda_rng_state = paddle.get_cuda_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_rng_state_tracker(
        ).get_states_tracker()

        # save input for backward
        ctx.inputs = []
        ctx.tensor_indices = []
        ctx.tensor_shapes = []
        tensor_inputs = []

        cur_device = paddle.get_device()
        assert 'gpu:' in paddle.get_device(
        ), "Recompute with RNG is not support current device: {}.".format(
            cur_device)

        # TODO support AMP
        tracer = framework._dygraph_tracer()
        ctx.is_fw_autocast = False if tracer._amp_level == core.AmpLevel.O0 else True
        if tracer._amp_level == core.AmpLevel.O2:
            ctx.amp_level = 'O2'
        elif tracer._amp_level in (core.AmpLevel.O1, core.AmpLevel.O0):
            ctx.amp_level = 'O1'
        else:
            raise ValueError("unsupported amp level: {}".format(
                tracer._amp_level))
        ctx.amp_white_list, ctx.amp_black_list = tracer._get_amp_op_list()

        with paddle.no_grad():
            outputs = run_function(*args)

        for i, arg in enumerate(args):
            if paddle.is_tensor(arg):
                state = arg.stop_gradient
                if _recompute_partition:
                    ctx.tensor_shapes.append(arg.shape)
                    partition = _split_activation(arg.detach()).clone()
                    # TODO(shenliang03) not use calculate stream to D2H to speed
                    arg = partition.cpu() if _recompute_offload else partition
                else:
                    arg = arg.cpu() if _recompute_offload else arg
                arg.stop_gradient = state
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        if paddle.is_tensor(outputs):
            all_outputs += [outputs]
            return outputs
        else:
            all_outputs += outputs
            return tuple(outputs)

    @staticmethod
    def backward(ctx, *args):
        with paddle.fluid.dygraph.guard():
            # Restore inputs
            inputs = list(ctx.inputs)
            tensor_indices = ctx.tensor_indices
            tensor_shapes = ctx.tensor_shapes
            tensors = list(ctx.saved_tensor())

            device_id = paddle.distributed.ParallelEnv().device_id
            for i, idx in enumerate(tensor_indices):
                if _recompute_partition:
                    state = tensors[i].stop_gradient
                    tensors[i] = _merge_activation(tensors[i]).detach(
                    ).reshape_(tensor_shapes[i])
                    tensors[i].stop_gradient = state
                inputs[idx] = tensors[i].cuda(
                    device_id) if _recompute_offload else tensors[i]

            tracer = framework._dygraph_tracer()
            tracer._has_grad = True

            # need restore auto_cast state as well as w/b list
            with _swith_rng_state_tracker(ctx.fwd_cuda_rng_state,
                                          ctx.fwd_cuda_rng_state_tracker):
                with paddle.amp.auto_cast(
                        enable=ctx.is_fw_autocast,
                        custom_white_list=ctx.amp_white_list,
                        custom_black_list=ctx.amp_black_list,
                        level=ctx.amp_level):
                    detached_inputs = detach_variable(tuple(inputs))
                    outputs = ctx.run_function(*detached_inputs)

            if isinstance(outputs, core.VarBase):
                outputs = (outputs, )
            assert len(outputs) == len(args)

            forward_outputs_with_grad = []
            backward_inputs = []

            for i in range(len(outputs)):
                if isinstance(outputs[i],
                              core.VarBase) and not outputs[i].stop_gradient:
                    forward_outputs_with_grad.append(outputs[i])
                    backward_inputs.append(args[i])

            if len(forward_outputs_with_grad) == 0:
                raise RuntimeError(
                    "none of output has stop_gradient=False, this recompute() is not necessary"
                )

            # actually backward            
            paddle.autograd.backward(forward_outputs_with_grad, backward_inputs)
            grads = list(inp._grad_ivar() for inp in detached_inputs
                         if isinstance(inp, core.VarBase))
            return grads


def _hp_recompute(function, *args):
    # NODTE(shenliang03)The current hybrid parallel recompute has limitations. 
    # It cannot handle the following situations:
    # 1. The calculation output of recompute, there are tensors that do not require gradients.
    # 2. The forward output tensor has no gradient. This problem can be solved temporarily by detach().
    # 3. Here, we only use float dtype to distinguish whether a gradient is needed in output tensor

    all_outputs = []
    _HPRecomputeFunction.apply(function, all_outputs, *args)

    if len(all_outputs) == 1:
        return all_outputs[0]
    else:
        for output in all_outputs:
            if paddle.is_tensor(output) and not is_float_tensor(output):
                output.stop_gradient = True

        return tuple(all_outputs)
