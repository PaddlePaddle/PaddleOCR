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

from __future__ import print_function
from . import core
from . import framework
from . import executor
from . import compiler
from .data_feeder import check_type
import sys

__all__ = ['ParallelExecutor']

ExecutionStrategy = core.ParallelExecutor.ExecutionStrategy
BuildStrategy = core.ParallelExecutor.BuildStrategy


class ParallelExecutor(object):
    """
	:api_attr: Static Graph

    The ParallelExecutor is an upgraded version of :code:`paddle.static.Executor` that supports multi-node model
    training and testing based on the data-parallel mode. In data-parallel mode,
    ParallelExecutor will broadcast the parameters from Node0 to other nodes during
    construction and copy the input Program to other nodes from Node0 to make sure
    that the initial state on each node is the same. Each node runs the model independently
    and the parameters' gradient is aggregated between those nodes during backward
    computation, and then each node independently updates its parameters. If you use
    the GPU to run the model, i.e. use_cuda=True, the node refers to the GPU,
    ParallelExecutor will automatically get the GPU resources available on the
    current machine, users can also set the available GPU resources in the environment
    variable, for example: want to use GPU0, GPU1, export CUDA_VISIBLEDEVICES=0,1;
    If the operation is performed on the CPU, i.e. use_cuda=False, the node refers to the CPU.
    **Note: At this time, the user needs to manually add CPU_NUM to the environment variable
    and set the number of CPU devices. For example, export CPU_NUM=4, if the environment
    variable is not set, the executor will add the variable to the environment variable
    and set it to 1.**


    Args:
        use_cuda (bool): Whether to use CUDA or not.
        loss_name (str): This parameter is the name of the loss Tensor of the
            model. **Note: If it is data-parallel model training, you must set loss_name,
            otherwise, the results may be wrong**. The default is None.
        main_program (Program): This parameter represents the Program to be executed.
            If this parameter is not provided, that parameter is None, the program will
            be set to :code:`paddle.static.default_main_program()`. The default is None.
        share_vars_from(ParallelExecutor): If share_vars_from is set, the current
            ParallelExecutor will share the parameters with the ParallelExecutor
            specified by share_vars_from. This parameter needs to be set when model testing
            is required during model training, and the data parallel mode is used for
            training and testing. Since ParallelExecutor will only distribute parameter
            variables to other devices when it is first executed, the ParallelExecutor
            specified by share_vars_from must be run before the current ParallelExecutor.
            The default is None.
        exec_strategy(ExecutionStrategy): exec_strategy specifies the options that can
            be changed when running the current model, such as the thread pool size.
            For more information about exec_strategy, please refer to :code:`paddle.static.ExecutionStrategy`.
            The default is None.
        build_strategy(BuildStrategy): By configuring build_strategy, we can
            optimize the computational graph, such as operators' fusion in the
            computational graph and memory optimization during the execution
            of the computational graph. For more information about build_strategy,
            please refer to :code:`paddle.static.BuildStrategy`.  The default is None.
        num_trainers(int): This parameter needs to be set in GPU distributed training.
            If the parameter value is greater than 1, NCCL will be initialized by multi-level
            nodes. Each node should have the same number of GPUs. The default is 1.
        trainer_id(int): This parameter needs to be set when performing GPU distributed
            training. This parameter must be used with the num_trainers parameter.
            Trainer_id indicates the "rank" of the current node. The trainer_id starts
            counting from 0. The default is 0.
        scope(Scope): Specifies the scope in which the program is executed.
            The default is paddle.static.global_scope().

    Returns:
        ParallelExecutor: The initialized ParallelExecutor object.

    Raises:
        TypeError: If share_vars_from is provided, but not ParallelExecutor object.

    NOTES:

        1. If you only use ParallelExecutor to do multi-card test, you don't need to set loss_name
           and share_vars_from.

        2. If you need to train and test the model with ParallelExecutor, the share_vars_from
           must be set when building the ParallelExecutor corresponding to the model test.
           Otherwise, the parameters used in the model test and the model training are inconsistent.

    Examples:
        .. code-block:: python

          import paddle
          import numpy
          import os

          use_cuda = True
          paddle.enable_static()
          place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

          # NOTE: If you use CPU to run the program, you need
          # to specify the CPU_NUM, otherwise, PaddlePaddle will use
          # all the number of the logic core as the CPU_NUM,
          # in that case, the batch size of the input should be
          # greater than CPU_NUM, if not, the process will be
          # failed by an exception.
          if not use_cuda:
              os.environ['CPU_NUM'] = str(2)

          exe = paddle.static.Executor(place)

          train_program = paddle.static.Program()
          startup_program = paddle.static.Program()
          with paddle.static.program_guard(train_program, startup_program):
              data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
              hidden = paddle.static.nn.fc(data, 10)
              loss = paddle.mean(hidden)
              test_program = paddle.static.default_main_program().clone(for_test=True)
              paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

          exe.run(startup_program)

          train_exe = paddle.static.ParallelExecutor(use_cuda=use_cuda,
                                                     main_program=train_program,
                                                     loss_name=loss.name)
          # Note: if share_vars_from is not set here, the test parameter is different to the train one
          test_exe = paddle.static.ParallelExecutor(use_cuda=use_cuda,
                                                    main_program=test_program,
                                                    share_vars_from=train_exe)

          x = numpy.random.random(size=(10, 1)).astype('float32')
          loss_data, = train_exe.run(feed={"X": x},
                                     fetch_list=[loss.name])

          loss_data, = test_exe.run(feed={"X": x},
                                    fetch_list=[loss.name])

    """

    def __init__(self,
                 use_cuda,
                 loss_name=None,
                 main_program=None,
                 share_vars_from=None,
                 exec_strategy=None,
                 build_strategy=None,
                 num_trainers=1,
                 trainer_id=0,
                 scope=None):
        if build_strategy is None:
            build_strategy = BuildStrategy()

        # TODO(paddle-dev): trainer_id and num_trainers should be removed from parameter list.
        if num_trainers != 1 and build_strategy.num_trainers != num_trainers:
            sys.stderr.write(
                'The value of build_strategy.num_trainers[%d] is overwritten '
                'by the passed num_trainers[%d].\n' %
                (build_strategy.num_trainers, num_trainers))
            build_strategy.num_trainers = num_trainers
        if trainer_id != 0 and build_strategy.trainer_id != trainer_id:
            sys.stderr.write(
                'The value of build_strategy.trainer_id[%d] is overwritten '
                'by the passed trainer_id[%d].\n' %
                (build_strategy.trainer_id, trainer_id))
            build_strategy.trainer_id = trainer_id

        self._places = framework.cuda_places(
        ) if use_cuda else framework.cpu_places()
        self._scope = scope if scope is not None else executor.global_scope()

        main_program = main_program if main_program is not None \
            else framework.default_main_program()

        self._compiled_program = compiler.CompiledProgram(main_program)
        if share_vars_from:
            assert isinstance(
                share_vars_from, ParallelExecutor
            ), "The share_vars_from should be ParallelExecutor."

        self._compiled_program.with_data_parallel(
            loss_name=loss_name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy,
            share_vars_from=share_vars_from._compiled_program
            if share_vars_from else None)

        self._place = core.CUDAPlace(0) if use_cuda else core.CPUPlace()
        self._exe = executor.Executor(self._place)

    def run(self, fetch_list, feed=None, feed_dict=None, return_numpy=True):
        """
        This interface is used to run the current model. It should be noted
        that the executor will execute all the operators in the Program,
        and will not prune some operators in the Program according to the
        fetch_list.

        Args:
            fetch_list(list): This parameter represents the Tensors that need to be returned
                after the model runs. The default is None.
            feed(list|dict): This parameter represents the input Tensors of the model.
                If it is single card training, the feed is dict type, and if it is multi-card
                training, the parameter feed can be dict or list of Tensor. If the
                parameter type is dict, the data in the feed will be split and sent to
                multiple devices (CPU/GPU), that is to say, the input data will be evenly
                sent to different devices, so you should make sure the number of samples of
                the current mini-batch must be greater than the number of places;
                if the parameter type is list, those data are copied directly to each device,
                so the length of this list should be equal to the number of places.
                The default is None.
            feed_dict: Alias for feed parameter, for backward compatibility.
                This parameter has been deprecated. Default None.
            return_numpy(bool): This parameter indicates whether convert the fetched Tensors
                (the Tensor specified in the fetch list) to numpy.ndarray. if it is False,
                the type of the return value is a list of :code:`LoDTensor`. The default is True.

        Returns:
            List: The fetched result list.

        Raises:
            ValueError: If the feed is a list, but its length is not equal the
                length of active places, or its element's is not dict.

        NOTES:
            1. If the feed parameter is dict type, the input data will be evenly distributed
               to different cards. For example, using two GPUs to run the model, the input
               sample number is 3, that is, [0, 1, 2], the sample number on GPU0 is 1,
               that is, [0], and the sample number on GPU1 is 2, that is, [1, 2].
               If the number of samples is less than the number of devices, the program will
               throw an exception, so when running the model, you should make sure that the
               number of samples of the last batch of the data set should be greater than the
               number of CPU cores or GPU cards, if it is less than, it is recommended that
               the batch be discarded.
            2. If the number of CPU cores or GPU cards available is greater than 1, the fetch
               results are spliced together in dimension 0 for the same Tensor values
               (Tensors in fetch_list) on different devices.


        Examples:
            .. code-block:: python

              import paddle
              import numpy
              import os

              use_cuda = True
              paddle.enable_static()
              place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

              # NOTE: If you use CPU to run the program, you need
              # to specify the CPU_NUM, otherwise, PaddlePaddle will use
              # all the number of the logic core as the CPU_NUM,
              # in that case, the batch size of the input should be
              # greater than CPU_NUM, if not, the process will be
              # failed by an exception.
              if not use_cuda:
                  os.environ['CPU_NUM'] = str(2)

              exe = paddle.static.Executor(place)

              train_program = paddle.static.Program()
              startup_program = paddle.static.Program()
              with paddle.static.program_guard(train_program, startup_program):
                  data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
                  hidden = paddle.static.nn.fc(data, 10)
                  loss = paddle.mean(hidden)
                  paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

              exe.run(startup_program)

              train_exe = paddle.static.ParallelExecutor(use_cuda=use_cuda,
                                                         main_program=train_program,
                                                         loss_name=loss.name)

              # If the feed is a dict:
              # the image will be split into devices. If there is two devices
              # each device will process an image with shape (5, 1)
              x = numpy.random.random(size=(10, 1)).astype('float32')
              loss_data, = train_exe.run(feed={"X": x},
                                         fetch_list=[loss.name])

              # If the feed is a list:
              # each device will process each element in the list.
              # the 1st device will process an image with shape (10, 1)
              # the 2nd device will process an image with shape (9, 1)
              #
              # you can use exe.device_count to get the device number.
              x2 = numpy.random.random(size=(9, 1)).astype('float32')
              loss_data, = train_exe.run(feed=[{"X": x}, {"X": x2}],
                                         fetch_list=[loss.name])

        """
        return self._exe.run(program=self._compiled_program,
                             scope=self._scope,
                             feed=feed,
                             fetch_list=fetch_list,
                             return_numpy=return_numpy)

    @property
    def device_count(self):
        return len(self._places)

    def drop_local_exe_scopes(self):
        """
        Drop the local execution scopes immediately. In order to avoid frequently
        application and release of temporary variables, the strategy adopted by
        ParallelExecutor is to drop the local execution scopes after several iterations.
        ParallelExecutor provides the num_iteration_per_drop_scope option in
        :code:`paddle.static.ExecutionStrategy`, which indicates how many iterations are intervened to
        drop the local execution scopes. If the num_iteration_per_drop_scope value
        is 100, but you want to drop the local execution scopes after 50 iterations,
        you can call the interface manually.

        Returns:
            None

        Examples:
            .. code-block:: python

              import paddle
              import numpy
              import os

              use_cuda = True
              # NOTE: If you use CPU to run the program, you need
              # to specify the CPU_NUM, otherwise, PaddlePaddle will use
              # all the number of the logic core as the CPU_NUM,
              # in that case, the batch size of the input should be
              # greater than CPU_NUM, if not, the process will be
              # failed by an exception.
              if not use_cuda:
                  os.environ['CPU_NUM'] = str(2)

              paddle.enable_static()
              train_program = paddle.static.Program()
              startup_program = paddle.static.Program()
              with paddle.static.program_guard(train_program, startup_program):
                  data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
                  hidden = paddle.static.nn.fc(data, 10)
                  loss = paddle.mean(hidden)

              place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
              exe = paddle.static.Executor(place)
              exe.run(startup_program)

              parallel_exe = paddle.static.ParallelExecutor(use_cuda=use_cuda,
                                                            main_program=train_program,
                                                            loss_name=loss.name)

              x = numpy.random.random(size=(10, 1)).astype('float32')
              loss_data, = parallel_exe.run(feed={"X": x},
                                            fetch_list=[loss.name])

              parallel_exe.drop_local_exe_scopes()

        """
        check_type(self._compiled_program._executor,
                   "the Executor of compiled program", core.ParallelExecutor,
                   "ParallelExecutor.drop_local_exe_scopes")
        self._compiled_program._executor.drop_local_exe_scopes()

    # This API is used to check whether DropLocalExeScopes can work.
    def _need_create_local_exe_scopes(self):
        check_type(self._compiled_program._executor,
                   "the Executor of compiled program", core.ParallelExecutor,
                   "ParallelExecutor._need_create_local_exe_scopes")
        return self._compiled_program._executor._need_create_local_exe_scopes()
