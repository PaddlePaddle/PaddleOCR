#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import multiprocessing
import os
import six
import sys
from .. import compat as cpt
from . import framework
from .framework import _get_paddle_place, _get_paddle_place_list
from .framework import cuda_places, cpu_places, xpu_places
from . import core

__all__ = ['CompiledProgram', 'ExecutionStrategy', 'BuildStrategy']

ExecutionStrategy = core.ParallelExecutor.ExecutionStrategy
BuildStrategy = core.ParallelExecutor.BuildStrategy
InferNativeConfig = core.NativeConfig
InferAnalysisConfig = core.AnalysisConfig
DeviceType = core.DeviceType


def _place_obj(place):
    p = core.Place()
    p.set_place(place)
    return p


def _is_pserver_mode(main_program):
    main = main_program if main_program \
        else framework.default_main_program()
    for op in main.global_block().ops:
        if op.type in ["send", "recv"]:
            return True
    return False


def _has_backward_op(graph):
    for node in graph.nodes():
        if node.is_op() and node.op() is not None and \
                node.op().type().endswith("_grad"):
            return True
    return False


def _prune_feed_ops(program):
    # prune the feed ops in the program.
    pop_idx = []
    for i, op in enumerate(program.global_block().ops):
        if op.type == "feed": pop_idx.append(i)
    for index in pop_idx[::-1]:
        program.global_block()._remove_op(index)


def _has_optimize_op(block):
    for op in block.ops:
        op_maker = core.op_proto_and_checker_maker
        optimize = core.op_proto_and_checker_maker.OpRole.Optimize
        if op_maker.kOpRoleVarAttrName() in op.attr_names and \
                int(op.all_attrs()[op_maker.kOpRoleAttrName()]) == int(optimize):
            return True
    return False


def _has_optimizer_in_control_flow(program):
    if not program:
        program = framework.default_main_program()
    for op in program.global_block().ops:
        if op.type == "conditional_block_grad":
            sub_block = program.block(op._block_attr_id("sub_block"))
            if _has_optimize_op(sub_block):
                return True

    return False


class CompiledProgram(object):
    """
    :api_attr: Static Graph
    
    The CompiledProgram is used to transform a program or graph for
    various optimizations according to the configuration of build_strategy,
    for example, the operators' fusion in the computation graph, memory
    optimization during the execution of the computation graph, etc.
    For more information about build_strategy, please refer to
    :code:`paddle.static.BuildStrategy`.

    Args:
        program_or_graph (Graph|Program): This argument is the Program or Graph
            being executed.
        build_strategy(BuildStrategy): This argument is used to compile the
            program or graph with the specified options, such as operators' fusion
            in the computational graph and memory optimization during the execution
            of the computational graph. For more information about build_strategy,
            please refer to :code:`paddle.static.BuildStrategy`. The default is None.

    Returns:
        CompiledProgram

    Example:
        .. code-block:: python

            import numpy
            import paddle
            import paddle.static as static

            paddle.enable_static()

            place = paddle.CUDAPlace(0) # paddle.CPUPlace()
            exe = static.Executor(place)

            data = static.data(name='X', shape=[None, 1], dtype='float32')
            hidden = static.nn.fc(x=data, size=10)
            loss = paddle.mean(hidden)
            paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

            exe.run(static.default_startup_program())
            compiled_prog = static.CompiledProgram(
                static.default_main_program())

            x = numpy.random.random(size=(10, 1)).astype('float32')
            loss_data, = exe.run(compiled_prog,
                                feed={"X": x},
                                fetch_list=[loss.name])
    """

    def __init__(self, program_or_graph, build_strategy=None):
        if isinstance(program_or_graph, core.Graph):
            self._graph = program_or_graph
            # don't not create a new program here.
            self._program = None
        elif isinstance(program_or_graph, framework.Program):
            _prune_feed_ops(program_or_graph)
            self._graph = core.Graph(program_or_graph.desc)
            self._program = program_or_graph
        else:
            raise TypeError(
                "The type of program_to_graph parameter is wrong, expected Graph or Program, but received %s"
                % type(program_or_graph))

        self._scope = None
        self._place = None
        self._executor = None
        self._compiled = False
        self._is_data_parallel = False
        self._is_inference = False
        self._loss_name = None
        self._share_vars_from = None
        self._places = None
        self._build_strategy = build_strategy
        self._exec_strategy = None

    def with_data_parallel(self,
                           loss_name=None,
                           build_strategy=None,
                           exec_strategy=None,
                           share_vars_from=None,
                           places=None):
        """
        This interface is used to transform the input Program or Graph to a multi-graph
        to run the model in data parallel mode. Users can use the build_strategy and
        exec_strategy to set some optimizations that can be applied during the construction
        and computation of the Graph, such as reducing the number of AllReduce operations,
        specifying the size of the thread pool used in the computation Graph running the model,
        and so on. 
        
        .. note::
            If build_strategy is specified when building CompiledProgram and calling 
            with_data_parallel, build_strategy in CompiledProgram will be overwritten, therefore, 
            if it is data parallel training, it is recommended to set build_strategy when calling 
            with_data_parallel interface.

        Args:
            loss_name (str): This parameter is the name of the loss Tensor of the model.
                **Note: If it is model training, you must set loss_name, otherwise the
                result may be problematic**. The default is None.
            build_strategy(BuildStrategy): This parameter is used to compile the
                program or graph with the specified options, such as operators' fusion
                in the computational graph and memory optimization during the execution
                of the computational graph. For more information about build_strategy,
                please refer to :code:`fluid.BuildStrategy`. The default is None.
            exec_strategy(ExecutionStrategy): exec_strategy specifies the options that can
                be changed when running the current model, such as the thread pool size.
                For more information about exec_strategy, please refer to :code:`fluid.ExecutionStrategy`.
                The default is None.
            share_vars_from(CompiledProgram): If share_vars_from is set, the current
                CompiledProgram will share the parameter value with the CompiledProgram
                specified by share_vars_from. This parameter needs to be set when model testing
                is required during model training, and the data parallel mode is used for
                training and testing. Since CompiledProgram will only distribute parameter
                Tensors to other devices when it is first executed, the CompiledProgram
                specified by share_vars_from must be run before the current CompiledProgram.
                The default is None.
            places(list(CUDAPlace)|list(CPUPlace)|list(str)|None): This parameter specifies the device
                on which the model is running. If you want to run on GPU0 and GPU1, places are
                [fluid.CUDAPlace(0), fluid.CUDAPlace(1)]; if you want to run with 2 CPUs, places are
                [fluid.CPUPlace()] * 2. If the parameter is not set, i.e. the parameter is None,
                the available device will be obtained from the environment variable when the model
                is executed: If the GPU is used, the currently available device ID is obtained
                from the environment variable FLAGS_selected_gpus or CUDA_VISIBLE_DEVICES when
                the model is executed; CPU, when the model is executed, the currently available
                CPU number is obtained from the environment variable CPU_NUM. For example,
                export CPU_NUM=4, if the environment variable is not set, the executor will
                add the variable to the environment variable and set its value to 1.
                The default is None. If ``places`` is the list of string, the string in the list
                can be ``cpu``, ``gpu:x``, where ``x`` is the index of the GPUs. 

        Returns:
            CompiledProgram

        Example:
            .. code-block:: python

                import numpy
                import os
                import paddle
                import paddle.static as static

                paddle.enable_static()

                use_cuda = True
                place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
                parallel_places = [paddle.CUDAPlace(0), paddle.CUDAPlace(1)] if use_cuda else [paddle.CPUPlace()] * 2

                # NOTE: If you use CPU to run the program, you need
                # to specify the CPU_NUM, otherwise, paddle will use
                # all the number of the logic core as the CPU_NUM,
                # in that case, the batch size of the input should be
                # greater than CPU_NUM, if not, the process will be
                # failed by an exception.
                if not use_cuda:
                    os.environ['CPU_NUM'] = str(2)

                exe = static.Executor(place)

                data = static.data(name='X', shape=[None, 1], dtype='float32')
                hidden = static.nn.fc(x=data, size=10)
                loss = paddle.mean(hidden)

                test_program = static.default_main_program().clone(for_test=True)
                paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

                exe.run(static.default_startup_program())
                compiled_train_prog = static.CompiledProgram(
                    static.default_main_program()).with_data_parallel(
                            loss_name=loss.name, places=parallel_places)
                # NOTE: if not set share_vars_from=compiled_train_prog,
                # the parameters used in test process are different with 
                # the parameters used by train process
                compiled_test_prog = static.CompiledProgram(
                    test_program).with_data_parallel(
                            share_vars_from=compiled_train_prog,
                            places=parallel_places)

                train_data = numpy.random.random(size=(10, 1)).astype('float32')
                loss_data, = exe.run(compiled_train_prog,
                                feed={"X": train_data},
                                fetch_list=[loss.name])
                test_data = numpy.random.random(size=(10, 1)).astype('float32')
                loss_data, = exe.run(compiled_test_prog,
                                feed={"X": test_data},
                                fetch_list=[loss.name])
        """
        assert not self._is_data_parallel, "Already compiled with parallel, cannot be recompiled."
        assert not self._is_inference, "Cannot compile with both data parallel and inference."
        self._is_data_parallel = True
        # FIXME(zcd): Currently, the build_strategy can be set during creating
        # CompiledProgram or calling with_data_parallel, and it may be confusing,
        # but in the long run, we should set up build_strategy only when creating
        # CompiledProgram, and exec_strategy should be deprecated.
        if build_strategy is not None: self._build_strategy = build_strategy
        self._exec_strategy = exec_strategy
        self._loss_name = loss_name
        self._share_vars_from = share_vars_from
        if isinstance(places, (list, tuple)):
            self._places = _get_paddle_place_list(places)
        else:
            self._places = _get_paddle_place(places)

        if _has_backward_op(self._graph):
            assert self._loss_name is not None, "The loss name of CompiledProgram is None. The loss name should be set if CompiledProgram contains backward part."

        if self._places is not None:
            if not isinstance(self._places, (list, tuple)):
                self._places = [self._places]

        return self

    def _with_inference_optimize(self, config):
        """ Add inference optimize

        Args:
            config: instance of `NativeConfig` or `AnalysisConfig` to create predictor
        Returns:
            self
        """
        assert not self._is_data_parallel, "Cannot compile with both data parallel and inference"
        assert not self._is_inference, "Already compiled with inference, cannot be recompiled."

        assert any([
            isinstance(config, InferNativeConfig),
            isinstance(config, InferAnalysisConfig)
        ])
        self._is_inference = True
        self._infer_config = config
        return self

    def _with_distributed(self):
        raise NotImplementedError(
            "Subclass of CompiledProgram should implement _with_distributed method."
        )

    def _compile_data_parallel(self, places, use_device, scope=None):
        if self._share_vars_from:
            if scope:
                sys.stderr.write("share_vars_from is set, scope is ignored.\n")
            if not self._share_vars_from._is_data_parallel:
                raise ValueError(
                    "The shared Program is not data parallel, cannot "
                    "share variables from it.")
            if self._share_vars_from._executor is None:
                raise ValueError(
                    "The shared Program is not compiled and executed, so there is no "
                    "variables to share.")
            self._local_scopes = self._share_vars_from._executor.local_scopes()
        else:
            assert scope is not None, ""
            self._local_scopes = []

        assert isinstance(places, tuple) or isinstance(places, list), \
            "Currently , The places type can only be list or tuple, but the input type is {}.".format(type(places))

        if self._build_strategy is None:
            self._build_strategy = BuildStrategy()
        self._build_strategy.is_distribution = _is_pserver_mode(self._program)

        if self._exec_strategy is None:
            self._exec_strategy = ExecutionStrategy()
        self._exec_strategy._use_device = use_device

        if self._exec_strategy.num_threads == 0:
            if self._exec_strategy._use_device == DeviceType.CUDA:
                # Experiments on se-resnext shows that too many threads hurt
                # performance. Worth tunning for other models in the future.
                self._exec_strategy.num_threads = len(places) * 4
            elif self._exec_strategy._use_device == DeviceType.XPU:
                # Currently only single thread is supported in Kunlun XPU.
                self._exec_strategy.num_threads = 1
            else:
                self._exec_strategy.num_threads = len(places) * 2

        if self._build_strategy.num_trainers > 1:
            assert self._is_data_parallel, \
                "If you use multi-trainer to train the model, you should use "\
                "the data parallel model, i.e. calling with_data_parallel function."

        # TODO(wuyi): trainer endpoings should be passed in through
        # build_strategy, not program.xxx.
        # TODO(gongwb): let user to set them once.
        if self._program and self._build_strategy.num_trainers > 1 and \
                self._program._trainers_endpoints:
            tps = self._program._trainers_endpoints

            assert self._build_strategy.num_trainers == len(
                tps), "The trainer numbers is not equal to endpoint numbers."
            self._build_strategy.trainers_endpoints = tps

        if self._program:
            self._build_strategy.nccl_comm_num = self._program._nccl_comm_num
            self._build_strategy.use_hierarchical_allreduce = self._program._use_hierarchical_allreduce
            self._build_strategy.hierarchical_allreduce_inter_nranks = self._program._hierarchical_allreduce_inter_nranks

        if self._build_strategy.sync_batch_norm:
            self._build_strategy.enable_sequential_execution = True

        if self._program is not None and self._program._enable_dgc:
            assert self._exec_strategy._use_device == DeviceType.CUDA, "DGC only used under CUDA environment."
            assert self._build_strategy.num_trainers * len(
                places) > 1, "DGC is not avaliable for single card training."
            assert self._build_strategy.reduce_strategy == BuildStrategy.ReduceStrategy.AllReduce, "DGC \
                only can be used for AllReduce BuildStrategy."

            # DGC doesn't support fuse for now, close fuse.
            self._build_strategy.fuse_all_reduce_ops = False

        self._persistable_vars = []
        for node in self._graph.nodes():
            if node.is_var() and node.var() is not None and node.var().persistable() and \
                    node.var().type() != core.VarDesc.VarType.RAW:
                self._persistable_vars.append(cpt.to_text(node.name()))

        places = list(map(_place_obj, places))

        # ParallelExecutor would broadcast all the parameters during initializing.
        # The parameters of each process should be in the same ordered for the data-parallelism
        # distributed training to keep the broadcast correct.
        self._persistable_vars = list(set(self._persistable_vars))
        self._persistable_vars.sort()

        return core.ParallelExecutor(
            places, self._persistable_vars,
            cpt.to_text(self._loss_name)
            if self._loss_name else six.u(''), self._scope, self._local_scopes,
            self._exec_strategy, self._build_strategy, self._graph)

    def _compile_inference(self):
        return core.create_paddle_predictor(self._infer_config)

    def _compile(self, scope, place):
        """Compile the program based on the configs.

        Args:
            scope: The variables (resources) that are associated with
               this compiled program.
            place: The location that the compiled program will be run on.

        Returns:
            self
        """
        if self._compiled:
            if scope and self._scope != scope:
                raise ValueError("Cannot compile program with different scope.")
            if place and not self._place._equals(place):
                raise ValueError("Cannot compile program with different place.")
            return self
        self._compiled = True

        self._scope = scope
        self._place = place

        if self._is_inference:
            self._executor = self._compile_inference()
        else:
            if self._is_data_parallel:
                self._places = self._get_places(self._place, self._places)
            else:
                self._places = [self._place]

            # Todo(liym27):If optimizer is used in control flow,
            #  training on multi-places is not supported now, will
            #  be supported later.
            if len(self._places) > 1 and \
                    _has_optimizer_in_control_flow(self._program):
                raise NotImplementedError(
                    "If optimizer is used in control flow, "
                    "training on multi-places is not supported now.")
            if isinstance(self._place, core.CUDAPlace):
                use_device = DeviceType.CUDA
            elif isinstance(self._place, core.XPUPlace):
                use_device = DeviceType.XPU
            else:
                use_device = DeviceType.CPU
            self._executor = self._compile_data_parallel(
                use_device=use_device, scope=self._scope, places=self._places)
        return self

    def _get_places(self, place, place_list):
        has_set_place = (place_list is not None)
        if has_set_place:
            for p in place_list:
                assert p._type() == place._type(), \
                    "Place type not match. You may set wrong type of places."
        else:
            if isinstance(place, core.CUDAPlace):
                place_list = cuda_places()
            elif isinstance(place, core.XPUPlace):
                place_list = xpu_places()
            else:
                place_list = cpu_places()
        assert place_list, "No places for execution."
        return place_list
