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
from paddle.fluid import core
from paddle.autograd import PyLayer
from paddle.fluid import framework
import contextlib

import logging
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

__all__ = []


def detach_variable(inputs):
    out = []
    for inp in inputs:
        if not isinstance(inp, core.VarBase):
            out.append(inp)
            continue

        x = inp.detach()
        x.stop_gradient = inp.stop_gradient
        out.append(x)
    return tuple(out)


def check_recompute_necessary(inputs):
    if not any(input_.stop_gradient == False for input_ in inputs
               if isinstance(input_, paddle.Tensor)):
        logger.warn(
            "[Recompute]: None of the inputs to current recompute block need grad, "
            "therefore there is NO need to recompute this block in backward !")


@contextlib.contextmanager
def swith_rng_state(rng_state):
    orig_cuda_rng_state = paddle.get_cuda_rng_state()
    paddle.set_cuda_rng_state(rng_state)
    try:
        yield
    finally:
        paddle.set_cuda_rng_state(orig_cuda_rng_state)


class RecomputeFunction(PyLayer):
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_recompute_necessary(args)

        # store for recomputing 
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state

        # NOTE the number of outputs of backward() should be equal to the number of tensors in forward()'s input
        # the order of tensors in backward()'s output should be the same as tensors in forward()'s input
        # None tensor inputs will be filtered in backward inputs.

        # save input for backward
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if paddle.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)
        ctx.save_for_backward(*tensor_inputs)

        # NOTE recompute with restore RNG only support one senario where one process for one cuda gpu.
        # one process with multiple gpu and mix-gpu-cpu senarios are not support
        if ctx.preserve_rng_state:
            cur_device = paddle.get_device()
            if 'gpu:' not in cur_device:
                raise RuntimeError(
                    "Recompute with RNG perserve is not support current device: {}.".
                    format(cur_device))
            ctx.fw_cuda_rng_state = paddle.get_cuda_rng_state()

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
        return outputs

    @staticmethod
    def backward(ctx, *args):
        with paddle.fluid.dygraph.guard():
            # TODO need to check the recompute calling is vaild or not

            # Restore inputs
            inputs = list(ctx.inputs)
            tensor_indices = ctx.tensor_indices
            tensors = ctx.saved_tensor()
            for i, idx in enumerate(tensor_indices):
                inputs[idx] = tensors[i]

            # paddle.enable_grad()
            tracer = framework._dygraph_tracer()
            tracer._has_grad = True

            # NOTE support AMP
            # need restore auto_cast state as well as w/b list
            if ctx.preserve_rng_state:
                with swith_rng_state(ctx.fw_cuda_rng_state):
                    with paddle.amp.auto_cast(
                            enable=ctx.is_fw_autocast,
                            custom_white_list=ctx.amp_white_list,
                            custom_black_list=ctx.amp_black_list,
                            level=ctx.amp_level):
                        detached_inputs = detach_variable(tuple(inputs))
                        outputs = ctx.run_function(*detached_inputs)
            else:
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

            # run backward() with only tensor that requires grad
            forward_outputs_with_grad = []
            # NOTE In Transformer-like network, if user put the attention mask into the recompute segment output,
            # pylayer will force the stop_gradient of attention mask to be False, which will make the number of 
            # tensor that need grad does not match.
            # the following backward_inputs_with_grad is used to avoid this case.
            backward_inputs_with_grad = []
            for i in range(len(outputs)):
                if isinstance(outputs[i],
                              core.VarBase) and not outputs[i].stop_gradient:
                    forward_outputs_with_grad.append(outputs[i])
                    backward_inputs_with_grad.append(args[i])

            if len(forward_outputs_with_grad) == 0:
                raise RuntimeError(
                    "none of output has requires_grad=True, this recompute() is not necessary"
                )

            # actually backward            
            paddle.autograd.backward(forward_outputs_with_grad,
                                     backward_inputs_with_grad)

            grads = list(inp._grad_ivar() for inp in detached_inputs
                         if isinstance(inp, core.VarBase))
            return grads


def recompute(function, *args, **kwargs):
    """
    recompute intermediate activations to save then memory.

    Args:
        function: layer of sequence of layers that describes part of forward pass of the model whose 
        intermediate activations will be released to save memory in forward stage and will be recomputed 
        in backward stage for gradient calculation.
        preserve_rng_state(bool, optional):  if preserve the RNG state of forward and restore it in backward. 
        args: inputs to the function

    Returns:
        Output of function on args
    """
    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(
            arg for arg in kwargs))

    return RecomputeFunction.apply(function, preserve, *args)
