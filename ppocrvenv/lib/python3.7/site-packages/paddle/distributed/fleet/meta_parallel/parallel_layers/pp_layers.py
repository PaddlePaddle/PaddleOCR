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

import math
import re
import glob
import os
import numpy as np
import random
from functools import partial

import paddle
from paddle.fluid.dygraph.layers import Layer
from ...utils.log_util import logger, layer_to_str
from ..pp_utils.utils import _hp_recompute, _initialize_recompute_setting

__all__ = []


class LayerDesc(object):
    def __init__(self, layer_func, *inputs, **kwargs):
        self.layer_func = layer_func
        self.inputs = inputs
        self.kwargs = kwargs

        if not issubclass(layer_func, Layer):
            raise TypeError(
                "The input(layer_func) should be a derived class of Layer.")

    def build_layer(self):
        return self.layer_func(*self.inputs, **self.kwargs)

    def __repr__(self):
        return layer_to_str(self.layer_func.__name__, *self.inputs,
                            **self.kwargs)


class SharedLayerDesc(LayerDesc):
    def __init__(self,
                 key,
                 layer_func,
                 forward_func=None,
                 shared_weight_attr='weight',
                 *inputs,
                 **kwargs):
        super(SharedLayerDesc, self).__init__(layer_func, *inputs, **kwargs)
        self.layer_name = key
        self.forward_func = forward_func
        self.shared_weight_attr = shared_weight_attr


class SegmentLayers(object):
    def __init__(self, layers_desc, num_parts, method="uniform"):
        self._layers_desc = layers_desc
        self.method = method
        self.num_parts = num_parts
        self.num_items = len(layers_desc)
        assert self.num_items >= self.num_parts, "layer number should be greater than number of segments"

    def do_segment(self):
        if self.method == "uniform":
            return self.uniform(self.num_items, self.num_parts)

        elif self.method.startswith('layer:'):
            # Divide equally according to the specified layer
            layername = self.method.split(':')[1]
            weights = [0] * len(self._layers_desc)
            weight_idxs = self._gen_layer_weight(layername)
            for idx in weight_idxs:
                weights[idx] = 1

            assert sum(
                weights
            ) % self.num_parts == 0, "number of layers ({}) should be divided by part number({})".format(
                sum(weights), self.num_parts)
            part_size = sum(weights) // self.num_parts
            result = [0 for _ in range(self.num_parts + 1)]

            memory_counter = 0
            result_idx = 1
            for idx, weight in enumerate(weights):
                memory_counter += weight
                if memory_counter == part_size:
                    result[result_idx] = idx + 1
                    result_idx += 1
                    memory_counter = 0
            result[self.num_parts] = len(weights)
            return result

    def _gen_layer_weight(self, layername):
        weight_idxs = []
        regex = re.compile(layername, re.IGNORECASE)
        for idx, layer in enumerate(self._layers_desc):
            name = None
            if isinstance(layer, Layer):
                name = layer.__class__.__name__
            elif isinstance(layer, LayerDesc):
                name = layer.layer_func.__name__
            else:
                try:
                    name = layer.__name__
                except AttributeError:
                    # it is not error
                    continue
            if regex.search(name):
                weight_idxs.append(idx)

        assert len(
            weight_idxs) > 0, "weight_idxs' length should be greater than 0"
        return weight_idxs

    def uniform(self, num_items, num_parts):
        result = [0 for _ in range(num_parts + 1)]
        part_size = math.floor(num_items / num_parts)
        for i in range(num_parts):
            result[i] = int(min(part_size * i, num_items))
        result[num_parts] = num_items
        return result


class PipelineLayer(Layer):
    def __init__(self,
                 layers,
                 num_stages=None,
                 topology=None,
                 loss_fn=None,
                 seg_method="uniform",
                 recompute_interval=0,
                 recompute_offload=False,
                 recompute_partition=False):
        super(PipelineLayer, self).__init__()
        if num_stages is None and topology is None:
            raise ValueError("should provide num_stages or topology")

        # lazy import
        import paddle.distributed as dist
        from paddle.distributed import fleet

        self.device_id = dist.ParallelEnv().device_id
        self.layers = layers
        self._loss_fn = loss_fn
        self._topo = topology
        self._recompute_interval = recompute_interval
        self._recompute_offload = recompute_offload
        self._recompute_partition = recompute_partition

        if recompute_interval > 0:
            logger.info(
                "Start Recompute for PipeLineParallel. recompute_offload: {}, recompute_partition: {}".
                format(recompute_offload, recompute_partition))
        _initialize_recompute_setting(recompute_offload, recompute_partition)

        world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()

        if self._topo:
            self._stage_id = self._topo.get_coord(self.global_rank).pipe
            self._num_stages = self._topo.get_dim_size("pipe")
            if num_stages:
                assert self._num_stages == num_stages, "num_stages should be equal to be %d" % (
                    self._num_stages)
        else:
            # construct default topology
            if world_size % num_stages != 0:
                raise ValueError("should provide correct num_stages({}) "
                                 "which can be divided by world_size({})".
                                 format(num_stages, world_size))
            dp_num = world_size // num_stages
            self._topo = fleet.CommunicateTopology(["data", "pipe", "model"],
                                                   [dp_num, num_stages, 1])
            self._stage_id = self._topo.get_coord(self.global_rank).pipe
            self._num_stages = self._topo.get_dim_size("pipe")

        # initialize segment
        self._layers_desc = list(self.layers)
        self._num_layers = len(self._layers_desc)
        self._start_pos = 0
        self._end_pos = self._num_layers - 1
        self._segment_network(seg_method)
        self.shared_layers = paddle.nn.LayerDict()
        self.shared_weight_attrs = {}

        # construct layer
        self.run_function = []
        self._build_layer()

        self.shared_comm = self._construct_shared_comm()
        self._synchronize_shared_weights()

    def get_stage_from_index(self, layer_idx):
        assert 0 <= layer_idx < self._num_layers, "layer_idx is out of bound"
        for stage in range(self._topo.get_dim('pipe')):
            if self.segment_parts[stage] <= layer_idx < self.segment_parts[stage
                                                                           + 1]:
                return stage

    def _construct_shared_comm(self):
        shared_comm = {}
        if self._topo.get_dim("pipe") == 1:
            return

        layers_desc = self._layers_desc
        shared_layer_names = set(
            s.layer_name for s in layers_desc if isinstance(s, SharedLayerDesc))
        for key in shared_layer_names:
            shared_layers = []
            for idx, layer in enumerate(layers_desc):
                if isinstance(layer,
                              SharedLayerDesc) and layer.layer_name == key:
                    shared_layers.append(idx)

            shared_stages = set(
                self.get_stage_from_index(idx) for idx in shared_layers)
            self._dp_degree = self._topo.get_dim('data')
            self._mp_degree = self._topo.get_dim('model')
            self._sharding_degree = self._topo.get_dim('sharding')

            shared_ranks = []
            for dp in range(self._dp_degree):
                for sharding in range(self._sharding_degree):
                    for mp in range(self._mp_degree):
                        shared_ranks = []
                        for s in sorted(shared_stages):
                            shared_ranks.append(
                                self._topo.get_rank_from_stage(
                                    self.global_rank,
                                    pipe=s,
                                    data=dp,
                                    sharding=sharding,
                                    model=mp))

                        group = paddle.distributed.new_group(ranks=shared_ranks)
                        if self.global_rank in shared_ranks:
                            assert key in self.shared_layers
                            if key in self.shared_layers:
                                shared_comm[key] = {
                                    'ranks': shared_ranks,
                                    'group': group,
                                    'weight_attr':
                                    self.shared_weight_attrs[key],
                                    'layer': self.shared_layers[key],
                                }
        return shared_comm

    def _synchronize_shared_weights(self):
        for key, comm in self.shared_comm.items():
            with paddle.framework.no_grad():
                paddle.distributed.broadcast(
                    getattr(comm['layer'], comm['weight_attr']),
                    src=min(comm['ranks']),
                    group=comm['group'])

            for param in comm['layer'].parameters():
                if self.global_rank != min(comm['ranks']):
                    setattr(param, 'is_firstly_shared', False)

    def allreduce_shared_weight_gradients(self):
        for key, comm in self.shared_comm.items():
            param = getattr(self.shared_layers[key], comm['weight_attr'])
            # need use trace_op to allreduce weight
            with paddle.framework.no_grad():
                paddle.fluid.framework._dygraph_tracer().trace_op(
                    type="c_allreduce_sum",
                    inputs={'X': param._grad_ivar()},
                    outputs={'Out': param._grad_ivar()},
                    attrs={
                        'ring_id': comm['group'].id,
                        'use_calc_stream': True
                    })

    def _segment_network(self, seg_method):
        logger.info("start segment network..")
        seg = SegmentLayers(
            self._layers_desc, num_parts=self._num_stages, method=seg_method)
        self.segment_parts = seg.do_segment()

        logger.info("segment result:" + ", ".join(
            str(arg) for arg in self.segment_parts))

        self._start_pos = self.segment_parts[self._stage_id]
        self._end_pos = self.segment_parts[self._stage_id + 1]

        # print information for debug
        for stage in range(self._num_stages):
            start = self.segment_parts[stage]
            end = self.segment_parts[stage + 1]
            logger.info("stage={}, global_rank={} ,layer_number={}".format(
                stage, self.global_rank, end - start))

            for index, layer in enumerate(self._layers_desc[start:end]):
                logger.info("{}: {}".format(index + start, str(layer)))

        if self._loss_fn:
            try:
                logger.info("loss: {}".format(self._loss_fn.__name__))
            except AttributeError:
                logger.info("loss: {}".format(self._loss_fn.__class__.__name__))

    def _build_layer(self):
        start = self._start_pos
        end = self._end_pos
        for index, layer in enumerate(self._layers_desc[start:end]):
            layer_index = start + index
            if isinstance(layer, Layer):
                self.run_function.append(layer)
                self.add_sublayer(str(layer_index), layer)
            elif isinstance(layer, SharedLayerDesc):
                if layer.layer_name not in self.shared_layers:
                    self.shared_layers[layer.layer_name] = layer.build_layer()
                    self.shared_weight_attrs[
                        layer.layer_name] = layer.shared_weight_attr
                    for param in self.shared_layers[
                            layer.layer_name].parameters():
                        setattr(param, "is_firstly_shared", True)

                if layer.forward_func is None:
                    self.run_function.append(self.shared_layers[
                        layer.layer_name])

                else:
                    self.run_function.append(
                        partial(layer.forward_func, self.shared_layers[
                            layer.layer_name]))

            elif isinstance(layer, LayerDesc):
                model = layer.build_layer()
                self.run_function.append(model)
                self.add_sublayer(str(layer_index), model)
            else:
                self.run_function.append(layer)

    def forward_function(self, start, end):
        def execute_func(*x):
            if len(x) == 1:
                x = x[0]
            for idx, layer in enumerate(self.run_function[start:end]):
                x = layer(x)
            return x

        return execute_func

    def forward(self, input):
        if self._recompute_interval == 0:
            input = self.forward_function(0, len(self.run_function))(input)
        else:
            num_layers = len(self.run_function)
            for start_idx in range(0, num_layers, self._recompute_interval):
                end_idx = min(start_idx + self._recompute_interval, num_layers)
                funcs = self.run_function[start_idx:end_idx]

                if not isinstance(input, tuple):
                    input = (input, )

                if self._need_recompute(funcs, input):
                    input = _hp_recompute(
                        self.forward_function(start_idx, end_idx), *input)
                else:
                    input = self.forward_function(start_idx, end_idx)(*input)

        return input

    def _need_recompute(self, funcs, inputs):
        if not any(input_.stop_gradient == False for input_ in inputs
                   if isinstance(input_, paddle.Tensor)):
            return False

        params = [f.parameters() for f in funcs if isinstance(f, Layer)]
        return any(len(list(p)) > 0 for p in params)

    def save_state_dict(self, path):
        if self._topo.get_coord(self.global_rank).data != 0:
            return

        def _offset_dirname(ckpt_dir, local_layer_idx):
            idx = local_layer_idx + self._start_pos
            model_rank = self._topo.get_coord(self.global_rank).model
            rank_message = "-tensor_" + "{:0>2d}".format(model_rank)
            layer_save_path = os.path.join(ckpt_dir,
                                           'layer_{:0>2d}'.format(idx))
            layer_save_path = layer_save_path + rank_message + '-model_states.pdparams'
            return layer_save_path

        os.makedirs(path, exist_ok=True)
        for idx, layer in enumerate(self.run_function):
            model_save_path = _offset_dirname(path, idx)
            if not hasattr(layer, 'state_dict'):
                continue
            paddle.save(layer.state_dict(), model_save_path)

        logger.info("save model state successfully...")

    def set_state_dir(self, path):
        assert os.path.exists(
            path), "{} not found, please check the path".format(path)

        for idx, layer in enumerate(self.run_function):
            if not hasattr(layer, 'set_state_dict'):
                continue
            layer_idx = idx + self._start_pos
            layer_save_path = os.path.join(path,
                                           'layer_{0:0>2d}'.format(layer_idx))
            model_files = glob.glob(layer_save_path + "*model_states.pdparams")
            model_files.sort()
            mp_rank = self._topo.get_coord(self.global_rank).model
            mp_world_size = self._topo.get_dim('model')
            num_files = len(model_files)

            load_param_path = model_files[mp_rank * num_files // mp_world_size]
            model_state_dict = paddle.load(load_param_path)
            layer.set_state_dict(model_state_dict)

        self._synchronize_shared_weights()
        logger.info("load model state successfully...")
