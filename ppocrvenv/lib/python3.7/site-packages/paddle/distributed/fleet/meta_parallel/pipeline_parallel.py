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

import paddle
import paddle.fluid as fluid
from .meta_parallel_base import MetaParallelBase
from .pp_utils.utils import is_float_tensor, _initialize_recompute_hcg
from .parallel_layers.pp_layers import PipelineLayer

from ..utils.hybrid_parallel_util import broadcast_mp_parameters
from ..utils.hybrid_parallel_util import broadcast_dp_parameters
from ..utils.hybrid_parallel_util import broadcast_sharding_parameters
from ..utils.log_util import logger
from ..meta_optimizers.dygraph_optimizer import HybridParallelOptimizer, HybridParallelGradScaler
from .pp_utils import p2p_communication as p2p

__all__ = []


class PipelineParallel(MetaParallelBase):
    def __init__(self, layers, hcg, strategy):
        if not isinstance(layers, PipelineLayer):
            raise TypeError(
                "The Layer should be a derived class of PipelineLayer.")
        super(PipelineParallel, self).__init__(layers, hcg, strategy)
        self.use_data_parallel = self._hcg.get_data_parallel_world_size() > 1
        self.use_model_parallel = self._hcg.get_model_parallel_world_size() > 1
        self.use_sharding_parallel = self._hcg.get_sharding_parallel_world_size(
        ) > 1

        self.total_loss = None

        self.micro_batch_size = self._strategy.pipeline_configs[
            'micro_batch_size']
        self.accumulate_steps = self._strategy.pipeline_configs[
            'accumulate_steps']

        self._using_cache = self._strategy.pipeline_configs['p2p_cache_shape']

        self.num_stages = self._hcg.get_pipe_parallel_world_size()
        self.stage_id = self._hcg.get_stage_id()
        self.pp_group = self._hcg.get_pipe_parallel_group()

        p2p.initialize_p2p_groups(hcg, self._using_cache)

        _initialize_recompute_hcg(hcg)

        self.is_first_stage = self.stage_id == 0
        self.is_last_stage = (self.stage_id == (self.num_stages - 1))
        self.global_rank = self._hcg.get_global_rank()
        self.micro_batch_id = 0

        self._compute_loss = True

        logger.info("Pipeline Info -- num_stages: {}, stage_id: {}".format(
            self.num_stages, self.stage_id))

        if self.use_model_parallel:
            logger.info("start broadcast mp parameters")
            broadcast_mp_parameters(self._layers, self._hcg)

        if self.use_sharding_parallel:
            logger.info("start broadcast sharding parameters")
            broadcast_sharding_parameters(self._layers, self._hcg)

        if self.use_data_parallel:
            logger.info("start broadcast dp parameters")
            broadcast_dp_parameters(self._layers, self._hcg)

    def forward_backward_pipeline(self, data, scaler=None):
        # use the 1f1b scheduling strategy.
        # this strategy is inspired by:
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/schedules.py

        self.scaler = scaler

        # store data for train
        self.data = data

        # store total loss of entire batch
        self.total_loss = None

        # store data id for micro_batch
        self.micro_batch_id = 0

        startup_steps = (self.num_stages - self.stage_id - 1)
        startup_steps = min(startup_steps, self.accumulate_steps)
        steady_steps = self.accumulate_steps - startup_steps

        input_buffers = []
        output_buffers = []

        for step_id in range(startup_steps):
            input_tensor = p2p.recv_forward()

            output_tensor = self._forward_step(input_tensor)
            p2p.send_forward(output_tensor)

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor)

        if steady_steps > 0:
            input_tensor = p2p.recv_forward()

        for i in range(steady_steps):
            last_iter = (i == (steady_steps - 1))

            output_tensor = self._forward_step(input_tensor)

            output_tensor_grad = p2p.send_forward_recv_backward(output_tensor)

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor)

            input_tensor, output_tensor = input_buffers.pop(
                0), output_buffers.pop(0)

            input_tensor_grad = self._backward_step(input_tensor, output_tensor,
                                                    output_tensor_grad)

            if last_iter:
                input_tensor = None
                p2p.send_backward(input_tensor_grad)
            else:
                input_tensor = p2p.send_backward_recv_forward(input_tensor_grad)

        for i in range(startup_steps):
            input_tensor = input_buffers.pop(0)
            output_tensor = output_buffers.pop(0)

            output_tensor_grad = p2p.recv_backward()

            input_tensor_grad = self._backward_step(input_tensor, output_tensor,
                                                    output_tensor_grad)
            p2p.send_backward(input_tensor_grad)

        self._layers.allreduce_shared_weight_gradients()
        with paddle.amp.auto_cast(enable=False):
            train_loss = self._broadcast_final_loss()
        return train_loss

    def train_batch(self, data, optimizer, lr_scheduler=None, scaler=None):
        assert isinstance(optimizer, HybridParallelOptimizer), (
            'optimizer should be HybridParallelOptimizer subclass.')

        assert fluid.framework._dygraph_tracer()._has_grad, (
            'Please enable the generation of gradients.')

        if self.is_first_stage or self.is_last_stage:
            assert data is not None, (
                "For the first and the last stage, the data must be set.")
        else:
            data = None

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self._layers.train()

        # 1f1b for pipeline
        train_loss = self.forward_backward_pipeline(data, scaler)

        # optimizer
        with paddle.amp.auto_cast(enable=False):
            self._optimizer_step()

        return train_loss

    def eval_batch(self, data, compute_loss=False):
        self._layers.eval()
        self._compute_loss = compute_loss

        # save data for eval
        self.data = data
        # store data id for micro_batch
        self.micro_batch_id = 0

        # store total loss of entire batch
        self.total_loss = None

        startup_steps = (self.num_stages - self.stage_id - 1)
        startup_steps = min(startup_steps, self.accumulate_steps)
        steady_steps = self.accumulate_steps - startup_steps

        input_buffers = []
        output_buffers = []

        for step_id in range(startup_steps):
            input_tensor = p2p.recv_forward()

            output_tensor = self._forward_step(input_tensor)
            p2p.send_forward(output_tensor)

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor)

        if steady_steps > 0:
            input_tensor = p2p.recv_forward()

        for i in range(steady_steps):
            last_iter = (i == (steady_steps - 1))

            output_tensor = self._forward_step(input_tensor)
            p2p.send_forward(output_tensor)

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor)

            if not last_iter:
                input_tensor = p2p.recv_forward()

        if self._compute_loss:
            self.train_loss = self._broadcast_final_loss()
        else:
            self.train_loss = output_buffers

        return self.train_loss

    def _forward_step(self, input_tensor):
        if self.stage_id == 0:
            input_tensor = self._load_micro_batch(self.micro_batch_id)

        output_tensor = self._layers.forward(input_tensor)

        if self.is_last_stage:
            # train calculate loss for train
            if self._compute_loss:
                assert self._layers._loss_fn is not None, "loss function should exist to compute loss"
                labels = self._load_micro_batch(self.micro_batch_id)
                output_tensor = self._layers._loss_fn(output_tensor, labels)
                assert isinstance(
                    output_tensor, paddle.Tensor
                ), "Currently, loss_fn should obtain Paddle.Tensor dtype"

                with paddle.amp.auto_cast(enable=False):
                    if self.accumulate_steps > 1:
                        output_tensor = output_tensor / self.accumulate_steps

                    if self.total_loss is None:
                        self.total_loss = paddle.zeros_like(output_tensor)
                    self.total_loss += output_tensor.detach()

        self.micro_batch_id += 1
        return output_tensor

    def _backward_step(self, input_tensor, output_tensor, output_tensor_grad):
        if self.is_last_stage:
            assert output_tensor_grad is None
            if self.scaler:
                paddle.autograd.backward(self.scaler.scale(output_tensor))
            else:
                paddle.autograd.backward(output_tensor)
        else:
            if isinstance(output_tensor, tuple):
                outputs = [t for t in output_tensor if not t.stop_gradient]
                assert len(outputs) == len(output_tensor_grad)
                paddle.autograd.backward(
                    tensors=outputs,
                    grad_tensors=[t for t in output_tensor_grad])
            else:
                paddle.autograd.backward(
                    tensors=[output_tensor], grad_tensors=[output_tensor_grad])

        input_tensor_grad = None
        if input_tensor is not None:
            if isinstance(input_tensor, tuple):
                input_tensor_grad = tuple(
                    [t.grad for t in input_tensor if not t.stop_gradient])
            else:
                input_tensor_grad = input_tensor.grad
        return input_tensor_grad

    def _load_micro_batch(self, cache_id):
        inputs = self.data
        begin = cache_id * self.micro_batch_size
        end = begin + self.micro_batch_size

        if self.is_first_stage:
            assert len(inputs) == 2, "length of input should be 2"
            if isinstance(inputs[0], tuple):
                assert len(
                    inputs[0]
                ) > 1, "If you use tuple for input data, it should have at least two inputs."
                batch_size = inputs[0][0].shape[0]
                assert self.micro_batch_size * self.accumulate_steps == batch_size, (
                    "batch_size needs to be divisible by micro_batch_size. Currently, "
                    "batch_size = %d, micro_batch_size = %d, accumulate_steps = %d."
                    %
                    (batch_size, self.micro_batch_size, self.accumulate_steps))
                data = [input[begin:end, :].detach() for input in inputs[0]]
                return tuple(data)
            else:
                batch_size = inputs[0].shape[0]
                assert self.micro_batch_size * self.accumulate_steps == batch_size
                return inputs[0][begin:end, :].detach()
        elif self.is_last_stage:
            assert len(inputs) == 2, "length of input should be 2"
            if isinstance(inputs[1], tuple):
                batch_size = inputs[1][0].shape[0]
                assert self.micro_batch_size * self.accumulate_steps == batch_size
                data = [input[begin:end, :].detach() for input in inputs[1]]
                return tuple(data)
            else:
                batch_size = inputs[1].shape[0]
                assert self.micro_batch_size * self.accumulate_steps == batch_size
                return inputs[1][begin:end, :].detach()
        else:
            # No data input is required for other stages
            inputs = None

    def _broadcast_final_loss(self):
        if self.is_last_stage:
            assert self.total_loss is not None, "train_batch() in last stage should obtain vaild loss"
            loss = self.total_loss.detach()
            is_fp32 = paddle.to_tensor(
                1) if loss.dtype == paddle.float32 else paddle.to_tensor(0)
            paddle.distributed.broadcast(
                is_fp32,
                src=self.global_rank,
                use_calc_stream=True,
                group=self.pp_group)
            paddle.distributed.broadcast(
                loss,
                src=self.global_rank,
                use_calc_stream=True,
                group=self.pp_group)
        else:
            is_fp32 = paddle.to_tensor(1)
            paddle.distributed.broadcast(
                is_fp32,
                src=self._hcg.get_rank_from_stage(self.num_stages - 1),
                use_calc_stream=True,
                group=self.pp_group)
            loss = paddle.zeros(
                shape=[1],
                dtype="float32") if is_fp32.numpy()[0] else paddle.zeros(
                    shape=[1], dtype="float16")
            paddle.distributed.broadcast(
                loss,
                src=self._hcg.get_rank_from_stage(self.num_stages - 1),
                use_calc_stream=True,
                group=self.pp_group)
        return loss

    def _optimizer_step(self):
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.clear_grad()
        if self.lr_scheduler:
            self.lr_scheduler.step()
