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
import multiprocessing
import os
import six
import sys
import threading

from ..data_feeder import DataFeeder
from .control_flow import BlockGuard
from .layer_function_generator import templatedoc
from .. import core
from ..executor import global_scope
from ..framework import convert_np_dtype_to_dtype_, default_main_program, \
    default_startup_program, program_guard, Program, Variable
from ..layer_helper import LayerHelper
from ..unique_name import generate as unique_name

import logging
from ..data_feeder import check_dtype, check_type
from paddle.fluid.framework import static_only
from ..framework import _get_paddle_place, _current_expected_place, _set_expected_place

__all__ = [
    'data', 'read_file', 'double_buffer', 'py_reader',
    'create_py_reader_by_data', 'load'
]


@static_only
def data(name,
         shape,
         append_batch_size=True,
         dtype='float32',
         lod_level=0,
         type=core.VarDesc.VarType.LOD_TENSOR,
         stop_gradient=True):
    """
    **Data Layer**

    This operator creates the global variable. The global variables can be
    accessed by all the following operators in the graph.

    Note: 
        :code:`paddle.fluid.layers.data` is deprecated as it will be removed in 
        a later version. Please use :code:`paddle.fluid.data` .

        This :code:`paddle.fluid.layers.data` set shape and dtype at compile
        time but does NOT check the shape or the dtype of fed data, the
        :code:`paddle.fluid.data` checks the shape and the dtype of data fed 
        by Executor or ParallelExecutor during run time.

        To feed variable size inputs, users can feed variable size inputs
        directly to this :code:`paddle.fluid.layers.data` and PaddlePaddle will
        fit the size accordingly. Or set -1 on the variable dimension when using
        :code:`paddle.fluid.data` .

        The default :code:`stop_gradient` attribute of the Variable created by
        this API is true, which means the gradient won't be passed backward
        through the data Varaible. Set :code:`var.stop_gradient = False` If
        user would like to pass backward gradient.

    Args:
       name(str): The name/alias of the variable, see :ref:`api_guide_Name`
            for more details.
       shape(list|tuple): Tuple declaring the shape. If :code:`append_batch_size` is
            True and there is no -1 inside :code:`shape`, it should be 
            considered as the shape of the each sample. Otherwise, it should
            be considered as the shape of the batched data.  
       append_batch_size(bool):
          1. If true, it prepends -1 to the shape.
            For example if shape=[1], the resulting shape is [-1, 1]. This will 
            be useful to set different batch size at run time.
          2. If shape contains -1, such as shape=[1, -1].
            append_batch_size will be enforced to be be False (ineffective)
            because PaddlePaddle cannot set more than 1 unknown number on the
            shape.
       dtype(np.dtype|VarType|str): The type of the data. Supported dtype: bool,
            float16, float32, float64, int8, int16, int32, int64, uint8.
       type(VarType): The output type. Supported dtype: VarType.LOD_TENSOR,
            VarType.SELECTED_ROWS, VarType.NCCL_ID. Default: VarType.LOD_TENSOR. 
       lod_level(int): The LoD Level. 0 means the input data is not a sequence.
            Default: 0.
       stop_gradient(bool): A boolean that mentions whether gradient should flow.
            Default: True. 

    Returns:
        The global variable that gives access to the data.

    Return Type:
        Variable

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.data(name='x', shape=[784], dtype='float32')
    """
    helper = LayerHelper('data', **locals())

    check_type(name, 'name', (six.binary_type, six.text_type), 'data')
    check_type(shape, 'shape', (list, tuple), 'data')

    shape = list(shape)
    for i in six.moves.range(len(shape)):
        if shape[i] is None:
            shape[i] = -1
            append_batch_size = False
        elif shape[i] < 0:
            append_batch_size = False

    if append_batch_size:
        shape = [-1] + shape  # append batch size as -1

    data_var = helper.create_global_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        type=type,
        stop_gradient=stop_gradient,
        lod_level=lod_level,
        is_data=True)
    return data_var


class BlockGuardServ(BlockGuard):
    """
    BlockGuardServ class.

    BlockGuardServ class is used to create an op with a block in a program.
    """

    def __init__(self, server):
        if not (isinstance(server, ListenAndServ)):
            raise TypeError("BlockGuardServ takes a ListenAndServ")
        super(BlockGuardServ, self).__init__(server.helper.main_program)
        self.server = server

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False

        self.server.complete_op()
        return super(BlockGuardServ, self).__exit__(exc_type, exc_val, exc_tb)


class ListenAndServ(object):
    """
    **ListenAndServ Layer**

    ListenAndServ is used to create a rpc server bind and listen
    on specific TCP port, this server will run the sub-block when
    received variables from clients.

    Args:
        endpoint(string): IP:port string which the server will listen on.
        inputs(list): a list of variables that the server will get from clients.
        fan_in(int): how many client are expected to report to this server, default: 1.
        optimizer_mode(bool): whether to run the server as a parameter server, default: True.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            with fluid.program_guard(main):
                serv = layers.ListenAndServ(
                    "127.0.0.1:6170", ["X"], optimizer_mode=False)
                with serv.do():
                    x = layers.data(
                        shape=[32, 32],
                        dtype='float32',
                        name="X",
                        append_batch_size=False)
                    fluid.initializer.Constant(value=1.0)(x, main.global_block())
                    layers.scale(x=x, scale=10.0, out=out_var)

            exe = fluid.Executor(place)
            exe.run(main)
    """

    def __init__(self, endpoint, inputs, fan_in=1, optimizer_mode=True):
        self.helper = LayerHelper("listen_and_serv")
        self.inputs = inputs
        self.outputs = []
        self.endpoint = endpoint
        self.fan_in = fan_in
        # FIXME(typhoonzero): add optimizer_mode is stupid, should make it more
        # general.
        self.optimizer_mode = optimizer_mode

    def do(self):
        return BlockGuardServ(self)

    def get_params_and_grads(self):
        main_program = self.helper.main_program
        current_block = main_program.current_block()
        parent_block = self.parent_block()
        # params and grads in the same order.
        params = list()
        grads = list()
        for op in current_block.ops:
            # FIXME(typhoonzero): op.inputs is None if it's cloned.
            if self.optimizer_mode:
                if "Grad" in op.inputs and "Param" in op.inputs:
                    params.append(op.inputs["Param"].name)
                    grads.append(op.inputs["Grad"].name)
            else:
                # simple recv mode, recv operators inputs.
                for iname in op.input_names:
                    for in_var_name in op.input(iname):
                        params.append(parent_block.var(in_var_name))
                        grads.append(parent_block.var(in_var_name))

        return params, grads

    def parent_block(self):
        prog = self.helper.main_program
        parent_idx = prog.current_block().parent_idx
        assert parent_idx >= 0
        parent_block = prog.block(parent_idx)
        return parent_block

    def complete_op(self):
        from ..incubate.fleet.parameter_server.mode import DistributedMode

        main_program = self.helper.main_program
        current_block = main_program.current_block()
        parent_block = self.parent_block()

        parent_block.append_op(
            type='listen_and_serv',
            inputs={"X": self.inputs},
            outputs={},
            attrs={
                'endpoint': self.endpoint,
                'Fanin': self.fan_in,
                'optimize_blocks': [
                    current_block
                ],  # did not support multiple optimize blocks in layers
                'distributed_mode':
                DistributedMode.SYNC,  # did not support async now in layers
                'grad_to_block_id': [""]
            })


def Send(endpoints, send_vars, dummy_output=None, sync=True):
    """
    Send variables to the server side, and get vars from server
    side when server have finished running server side program.

    Args:
        endpoints (str): comma separated IP:PORT pairs in the order
                   of send_vars to send
        send_vars (list): variables to send to server
        sync (bool): whether to wait the request finish

    """
    assert (type(send_vars) == list)

    if dummy_output is None:
        dummy_output = []
    elif isinstance(dummy_output, Variable):
        dummy_output = [dummy_output]

    assert (type(dummy_output) == list)

    epmap = endpoints.split(",")
    endpoints = list(set(epmap))

    helper = LayerHelper("Send", **locals())
    rpc_op_role_name = core.op_proto_and_checker_maker.kOpRoleAttrName()

    helper.append_op(
        type="send",
        inputs={"X": send_vars},
        outputs={"Out": dummy_output},
        attrs={
            "endpoints": endpoints,
            "epmap": epmap,
            rpc_op_role_name: core.op_proto_and_checker_maker.OpRole.RPC
        })
    if sync:
        helper.append_op(
            type="send_barrier",
            inputs={"X": dummy_output},
            outputs={"Out": []},
            attrs={"endpoints": endpoints})


def Recv(endpoints, get_vars, dummy_input=None, sync=True):
    """
    Receive variables from server side

    Args:
        endpoints (str): comma separated IP:PORT pairs in the order
                   of send_vars to send
        get_vars (list): vars to get from server after send completes.
        sync (bool): whether to wait the request finish

    Returns:
        list: list of received variables
    """
    assert (type(get_vars) == list)

    if dummy_input is None:
        dummy_input = []
    elif isinstance(dummy_input, Variable):
        dummy_input = [dummy_input]

    assert (type(dummy_input) == list)

    epmap = endpoints.split(",")
    endpoints = list(set(epmap))

    helper = LayerHelper("Recv", **locals())
    helper.append_op(
        type="recv",
        inputs={"X": dummy_input},
        outputs={"Out": get_vars},
        attrs={"endpoints": endpoints,
               "epmap": epmap})
    if sync:
        helper.append_op(
            type="fetch_barrier",
            outputs={"Out": get_vars},
            attrs={"endpoints": endpoints})
    return get_vars


def monkey_patch_reader_methods(reader):
    def __get_reader__():
        scope = global_scope()
        var = scope.find_var(reader.name)
        return var.get_reader()

    def reset():
        return __get_reader__().reset()

    reader.reset = reset
    reader.stop_gradient = True
    reader.persistable = True
    return reader


def _copy_reader_var_(block, var):
    new_var = block.create_var(name=var.name, type=core.VarDesc.VarType.READER)
    new_var.desc.set_shapes(var.desc.shapes())
    new_var.desc.set_dtypes(var.desc.dtypes())
    new_var.desc.set_lod_levels(var.desc.lod_levels())
    new_var.persistable = True
    return new_var


def _copy_reader_create_op_(block, op):
    input_param_names = op.input_names
    new_input_map = {}
    for param_name in input_param_names:
        new_input_map[param_name] = []
        arg_names = op.input(param_name)
        for arg_name in arg_names:
            new_input_map[param_name].append(block.var(arg_name))

    output_param_names = op.output_names
    new_output_map = {}
    for param_name in output_param_names:
        new_output_map[param_name] = []
        arg_names = op.output(param_name)
        for arg_name in arg_names:
            new_output_map[param_name].append(block.var(arg_name))

    new_op = block.append_op(
        type=op.type,
        inputs=new_input_map,
        outputs=new_output_map,
        attrs=op.all_attrs())
    return new_op


def _py_reader(capacity,
               shapes,
               dtypes,
               lod_levels=None,
               name=None,
               use_double_buffer=True,
               feed_list=None):
    if feed_list is not None:
        if not isinstance(feed_list, list):
            raise TypeError("feed_list should be a list of Variable"
                            " instead of " + str(type(feed_list)))
        lod_levels = []
        dtypes = []
        shape_concat = []
        ranks = []
        shapes = []
        need_check_feed = []

        for feed_data in feed_list:
            dtypes.append(feed_data.dtype)
            shape_concat.extend(feed_data.shape)
            ranks.append(len(feed_data.shape))
            shapes.append(feed_data.shape)
            lod_levels.append(feed_data.lod_level)
            need_check_feed.append(int(feed_data.desc.need_check_feed()))
    else:
        dtypes = [convert_np_dtype_to_dtype_(dt) for dt in dtypes]
        need_check_feed = [0 for dt in dtypes]
        shape_concat = []
        ranks = []

        for shape in shapes:
            shape_concat.extend(shape)
            ranks.append(len(shape))

        if lod_levels is None:
            lod_levels = [0] * len(shapes)
    dtype_int = [int(t) for t in dtypes]
    if name is None:
        queue_name = unique_name('lod_tensor_blocking_queue')
        reader_name = unique_name('create_py_reader')
        double_buffer_name = unique_name('double_buffer')
    else:
        queue_name = "_".join([name, "queue"])
        reader_name = "_".join([name, "reader"])
        double_buffer_name = "_".join([name, "double_buffer"])

    var = global_scope().var(queue_name)
    feed_queue = core.init_lod_tensor_blocking_queue(var, capacity, False)

    startup_blk = default_startup_program().current_block()
    startup_var = startup_blk.create_var(name=reader_name)
    startup_blk.append_op(
        type='create_py_reader',
        inputs={'blocking_queue': [queue_name]},
        outputs={'Out': [startup_var]},
        attrs={
            'shape_concat': shape_concat,
            'lod_levels': lod_levels,
            'dtypes': dtype_int,
            'need_check_feed': need_check_feed,
            'ranks': ranks
        })

    startup_var.desc.set_dtypes(dtypes)
    startup_var.persistable = True

    main_prog_var = _copy_reader_var_(default_main_program().current_block(),
                                      startup_var)

    reader = monkey_patch_reader_methods(main_prog_var)
    if use_double_buffer:
        double_buffer_reader = double_buffer(reader, name=double_buffer_name)
        # we return a double buffer reader. However, the reset method comes from
        # py_reader.
        double_buffer_reader.reset = reader.reset
        reader = double_buffer_reader

    # monkey patch py_reader special methods
    reader.queue = feed_queue
    current_reset_method = reader.reset
    reader.thread = None
    reader.tensor_provider = None
    reader.exited = False

    def start_provide_thread(func):
        def __provider_thread__(legacy_expected_place):
            try:
                # See _DataLoaderIterSingleProcess._thread_loop() for why set expected place here.
                _set_expected_place(legacy_expected_place)

                for tensors in func():
                    array = core.LoDTensorArray()
                    for item in tensors:
                        if not isinstance(item, core.LoDTensor):
                            tmp = core.LoDTensor()
                            tmp.set(item, core.CPUPlace())
                            item = tmp

                        array.append(item)

                    if reader.exited:
                        break
                    feed_queue.push(array)
                    if reader.exited:
                        break
                feed_queue.close()
            except Exception as ex:
                feed_queue.kill()
                logging.warn('Your decorated reader has raised an exception!')
                six.reraise(*sys.exc_info())

        reader.thread = threading.Thread(
            target=__provider_thread__, args=(_current_expected_place(), ))
        reader.thread.daemon = True
        reader.thread.start()

    def __set_tensor_provider__(func):
        reader.tensor_provider = func

    def __set_paddle_reader__(paddle_reader):
        with program_guard(Program(), Program()):
            actual_feed_list = feed_list
            if actual_feed_list is None:
                actual_feed_list = []
                counter = 0
                for dtype, shape, lod_level in zip(dtypes, shapes, lod_levels):
                    name = str(counter)
                    actual_feed_list.append(
                        data(
                            name=name,
                            dtype=dtype,
                            shape=shape,
                            lod_level=lod_level))
                    counter += 1

            data_names = [feed_data.name for feed_data in actual_feed_list]
            feeder = DataFeeder(
                feed_list=actual_feed_list, place=core.CPUPlace())
            paddle_reader = feeder.decorate_reader(
                paddle_reader, multi_devices=False)

        def __tensor_provider__():
            for slots in paddle_reader():
                yield [slots[data_name] for data_name in data_names]

        __set_tensor_provider__(__tensor_provider__)

    def __reset__():
        current_reset_method()
        if reader.thread is not None and reader.tensor_provider is not None:
            reader.exited = True
            reader.thread.join()
            reader.exited = False

    def __start__():
        start_provide_thread(reader.tensor_provider)

    reader.reset = __reset__
    reader.decorate_tensor_provider = __set_tensor_provider__
    reader.decorate_paddle_reader = __set_paddle_reader__

    reader.decorate_batch_generator = __set_tensor_provider__
    reader.decorate_sample_list_generator = __set_paddle_reader__
    reader.start = __start__

    return reader


def py_reader(capacity,
              shapes,
              dtypes,
              lod_levels=None,
              name=None,
              use_double_buffer=True):
    """
	:api_attr: Static Graph

    Create a Python reader for data feeding in Python

    This operator returns a Reader Variable.
    The Reader provides :code:`decorate_paddle_reader()` and
    :code:`decorate_tensor_provider()` to set a Python generator as the data
    source and feed the data from the data source to the Reader Variable. 
    When :code:`Executor::Run()` is invoked in C++ side, the data from the 
    generator would be read automatically. Unlike :code:`DataFeeder.feed()`,
    the data reading process and :code:`Executor::Run()` process can run in 
    parallel using :code:`py_reader`. The :code:`start()` method of the Reader
    should be called when each pass begins, while the :code:`reset()` method 
    should be called when the pass ends and :code:`fluid.core.EOFException` raises.

    Note:
       :code:`Program.clone()` method cannot clone :code:`py_reader`. You can 
       refer to :ref:`api_fluid_Program` for more details.
       
       The :code:`read_file` call needs to be in the program block of :code:`py_reader`.
       You can refer to :ref:`api_fluid_layers_read_file` for more details.

    Args:
       capacity(int): The buffer capacity maintained by :code:`py_reader`.
       shapes(list|tuple): List of tuples which declaring data shapes. shapes[i] 
            represents the i-th data shape.
       dtypes(list|tuple): List of strings which declaring data type. Supported dtype:
            bool, float16, float32, float64, int8, int16, int32, int64, uint8.
       lod_levels(list|tuple): List of ints which declaring data lod_level.
       name(basestring): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.
       use_double_buffer(bool): Whether use double buffer or not. The double buffer is 
            for pre-reading the data of the next batch and copy the data asynchronously 
            from CPU to GPU. Default is True.

    Returns:
       A Reader from which we can get feeding data.

    Return Type:
       Variable

    Examples:
       1. The basic usage of :code:`py_reader` is as follows:
       
       .. code-block:: python
    
         import paddle
         import paddle.fluid as fluid
         import paddle.dataset.mnist as mnist

         def network(image, label):
             # user defined network, here a softmax regession example
             predict = fluid.layers.fc(input=image, size=10, act='softmax')
             return fluid.layers.cross_entropy(input=predict, label=label)

         reader = fluid.layers.py_reader(capacity=64,
                                         shapes=[(-1, 1, 28, 28), (-1, 1)],
                                         dtypes=['float32', 'int64'])
         reader.decorate_paddle_reader(
             paddle.reader.shuffle(paddle.batch(mnist.train(), batch_size=5),
                                   buf_size=1000))

         img, label = fluid.layers.read_file(reader)
         loss = network(img, label)

         fluid.Executor(fluid.CUDAPlace(0)).run(fluid.default_startup_program())
         exe = fluid.ParallelExecutor(use_cuda=True)
         for epoch_id in range(10):
             reader.start()
             try:
                 while True:
                     exe.run(fetch_list=[loss.name])
             except fluid.core.EOFException:
                 reader.reset()

         fluid.io.save_inference_model(dirname='./model',
                                       feeded_var_names=[img.name, label.name],
                                       target_vars=[loss],
                                       executor=fluid.Executor(fluid.CUDAPlace(0)))

       2. When training and testing are both performed, two different
       :code:`py_reader` should be created with different names, e.g.:

       .. code-block:: python
    
         import paddle
         import paddle.fluid as fluid
         import paddle.dataset.mnist as mnist

         def network(reader):
             img, label = fluid.layers.read_file(reader)
             # User defined network. Here a simple regression as example
             predict = fluid.layers.fc(input=img, size=10, act='softmax')
             loss = fluid.layers.cross_entropy(input=predict, label=label)
             return fluid.layers.mean(loss)

         # Create train_main_prog and train_startup_prog
         train_main_prog = fluid.Program()
         train_startup_prog = fluid.Program()
         with fluid.program_guard(train_main_prog, train_startup_prog):
             # Use fluid.unique_name.guard() to share parameters with test program
             with fluid.unique_name.guard():
                 train_reader = fluid.layers.py_reader(capacity=64,
                                                       shapes=[(-1, 1, 28, 28),
                                                               (-1, 1)],
                                                       dtypes=['float32', 'int64'],
                                                       name='train_reader')
                 train_reader.decorate_paddle_reader(
                     paddle.reader.shuffle(paddle.batch(mnist.train(), batch_size=5),
                                           buf_size=500))
                 train_loss = network(train_reader)  # some network definition
                 adam = fluid.optimizer.Adam(learning_rate=0.01)
                 adam.minimize(train_loss)

         # Create test_main_prog and test_startup_prog
         test_main_prog = fluid.Program()
         test_startup_prog = fluid.Program()
         with fluid.program_guard(test_main_prog, test_startup_prog):
             # Use fluid.unique_name.guard() to share parameters with train program
             with fluid.unique_name.guard():
                 test_reader = fluid.layers.py_reader(capacity=32,
                                                      shapes=[(-1, 1, 28, 28), (-1, 1)],
                                                      dtypes=['float32', 'int64'],
                                                      name='test_reader')
                 test_reader.decorate_paddle_reader(paddle.batch(mnist.test(), 512))
                 test_loss = network(test_reader)

         fluid.Executor(fluid.CUDAPlace(0)).run(train_startup_prog)
         fluid.Executor(fluid.CUDAPlace(0)).run(test_startup_prog)

         train_exe = fluid.ParallelExecutor(use_cuda=True,
                                            loss_name=train_loss.name,
                                            main_program=train_main_prog)
         test_exe = fluid.ParallelExecutor(use_cuda=True,
                                           loss_name=test_loss.name,
                                           main_program=test_main_prog)
         for epoch_id in range(10):
             train_reader.start()
             try:
                 while True:
                    train_exe.run(fetch_list=[train_loss.name])
             except fluid.core.EOFException:
                 train_reader.reset()

         test_reader.start()
         try:
             while True:
                 test_exe.run(fetch_list=[test_loss.name])
         except fluid.core.EOFException:
             test_reader.reset()
    """
    logging.warn(
        'paddle.fluid.layers.py_reader() may be deprecated in the near future. '
        'Please use paddle.fluid.io.DataLoader.from_generator() instead.')
    return _py_reader(
        capacity=capacity,
        shapes=shapes,
        dtypes=dtypes,
        lod_levels=lod_levels,
        name=name,
        use_double_buffer=use_double_buffer)


def create_py_reader_by_data(capacity,
                             feed_list,
                             name=None,
                             use_double_buffer=True):
    """
	:api_attr: Static Graph

    The OP creates a Python reader for data feeding in Python, it is similar
    to :ref:`api_fluid_layers_py_reader` except that it can read data from
    the list of feed variables.

    Parameters:
        capacity (int): The buffer capacity maintained by :code:`py_reader`. Its unit
            is batch number. Set larger :attr:`capacity` if the reader is fast.
        feed_list (list(Variable)): The feed variables, are usually created by
            :code:`fluid.data()`.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`. Default: None.
        use_double_buffer (bool, optional): Whether use double buffer. If it's True,
            the OP would prefetch next batch data asynchronously. Default: True.

    Returns:
        Reader: A Reader for data feeding. The data types of read data are the same as the data types of variables of :attr:`feed_list`.

    Examples:
        .. code-block:: python

          import paddle
          import paddle.fluid as fluid
          import paddle.dataset.mnist as mnist

          def network(img, label):
              # User defined network. Here a simple regression as example
              predict = fluid.layers.fc(input=img, size=10, act='softmax')
              loss = fluid.layers.cross_entropy(input=predict, label=label)
              return fluid.layers.mean(loss)

          MEMORY_OPT = False
          USE_CUDA = False

          image = fluid.data(name='image', shape=[None, 1, 28, 28], dtype='float32')
          label = fluid.data(name='label', shape=[None, 1], dtype='int64')
          reader = fluid.layers.create_py_reader_by_data(capacity=64,
                                                         feed_list=[image, label])
          reader.decorate_paddle_reader(
              paddle.reader.shuffle(paddle.batch(mnist.train(), batch_size=5), buf_size=500))
          img, label = fluid.layers.read_file(reader)
          loss = network(img, label) # The definition of custom network and the loss function

          place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()
          exe = fluid.Executor(place)
          exe.run(fluid.default_startup_program())

          build_strategy = fluid.BuildStrategy()
          build_strategy.memory_optimize = True if MEMORY_OPT else False
          exec_strategy = fluid.ExecutionStrategy()
          compiled_prog = fluid.compiler.CompiledProgram(
          fluid.default_main_program()).with_data_parallel(
              loss_name=loss.name,
              build_strategy=build_strategy,
              exec_strategy=exec_strategy)

          for epoch_id in range(2):
          reader.start()
          try:
              while True:
                  exe.run(compiled_prog, fetch_list=[loss.name])
          except fluid.core.EOFException:
              reader.reset()
    """
    logging.warn(
        'paddle.fluid.layers.create_py_reader_by_data() may be deprecated in the near future. '
        'Please use paddle.fluid.io.DataLoader.from_generator() instead.')
    return _py_reader(
        capacity=capacity,
        shapes=None,
        dtypes=None,
        lod_levels=None,
        name=name,
        use_double_buffer=use_double_buffer,
        feed_list=feed_list)


def __create_shared_decorated_reader__(op_type, reader, attrs):
    var_name = unique_name(op_type)
    startup_blk = default_startup_program().current_block()
    startup_var = startup_blk.create_var(name=var_name)
    startop_op = startup_blk.append_op(
        type=op_type,
        inputs={'UnderlyingReader': reader},
        outputs={'Out': [startup_var]},
        attrs=attrs)
    startup_var.persistable = True
    main_prog_block = default_main_program().current_block()
    main_prog_var = _copy_reader_var_(main_prog_block, startup_var)
    _copy_reader_create_op_(main_prog_block, startop_op)
    return monkey_patch_reader_methods(main_prog_var)


def __create_unshared_decorated_reader__(op_type, reader, attrs, name=None):
    new_reader_name = name if name is not None else unique_name(op_type)
    main_blk = default_main_program().current_block()
    new_reader = main_blk.create_var(name=new_reader_name)
    main_blk.append_op(
        type=op_type,
        inputs={'UnderlyingReader': reader},
        outputs={'Out': [new_reader]},
        attrs=attrs)
    return monkey_patch_reader_methods(new_reader)


def double_buffer(reader, place=None, name=None):
    """
    Wrap a double buffer reader. The class Reader contains DecoratedReader and FileReader. Moreover, the DecoratedReader is inherited by CustomReader and BufferedReader. This function is related to BufferedReader. The data will copy to target place with a double buffer queue. If the target place is None, the place that executor perform on will be used.


    Args:
        reader (Variable): The Reader Variable need to be wrapped.
        place (Place|str, optional): The place of target data, such as CPU, GPU, and if use GPU, it's necessary to point out which card is involved. Default is the sample place of executor perform.
            if ``place`` is string, It can be ``cpu``, ``gpu:x``, where ``x`` is the ndex of the GPUs. 
        name (str, optional): Variable name. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`. Default is None. 

    Returns:
        Variable(Reader): wrapped reader with double buffer.

    Examples:
        ..  code-block:: python
          
            import paddle.fluid as fluid
            reader = fluid.layers.py_reader(capacity=64,
                                            shapes=[(-1, 1, 28, 28), (-1, 1)],
                                            dtypes=['float32', 'int64'],
                                            use_double_buffer=False)
            reader = fluid.layers.double_buffer(reader)
            image, label = fluid.layers.read_file(reader)
    """
    attrs = dict()
    if place is not None:
        attrs['place'] = str(_get_paddle_place(place)).upper()

    return __create_unshared_decorated_reader__(
        'create_double_buffer_reader', reader, attrs, name=name)


def read_file(reader):
    """
	:api_attr: Static Graph

    Execute the given reader and get data via it.

    A reader is also a Variable. It can be a raw reader generated by
    `fluid.layers.open_files()` or a decorated one generated by
    `fluid.layers.double_buffer()` .

    Args:

        reader(Variable): The reader to execute.

    Returns:
        Tuple[Variable]: Data read from the given reader.

    Examples:
        .. code-block:: python
          
           import paddle.fluid as fluid
           reader = fluid.layers.py_reader(capacity=64,
                                           shapes=[(-1, 1, 28, 28), (-1, 1)],
                                           dtypes=['float32', 'int64'])
           image, label = fluid.layers.read_file(reader)
    """
    helper = LayerHelper('read_file')
    out = [
        helper.create_variable_for_type_inference(
            stop_gradient=True, dtype='float32')
        for _ in range(len(reader.desc.shapes()))
    ]
    helper.append_op(
        type='read', inputs={'Reader': [reader]}, outputs={'Out': out})
    if len(out) == 1:
        return out[0]
    else:
        return out


def load(out, file_path, load_as_fp16=None):
    """
    Load operator will load a LoDTensor / SelectedRows variable from disk file.

    Args:
        out(Variable): The LoDTensor / SelectedRows need to be loaded..

        file_path(STRING): Variable will be loaded from "file_path".

        load_as_fp16(BOOLEAN): If true, the tensor will be first loaded and then converted to float16 data type. Otherwise, the tensor will be directly loaded without data type conversion. Default is false..
    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            tmp_tensor = fluid.layers.create_tensor(dtype='float32')
            fluid.layers.load(tmp_tensor, "./tmp_tensor.bin")
    """
    helper = LayerHelper("load", **locals())
    attrs = {"file_path": file_path}
    if load_as_fp16 is not None:
        attrs['load_as_fp16'] = load_as_fp16
    helper.append_op(type="load", inputs={}, outputs={"Out": out}, attrs=attrs)
