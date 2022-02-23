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

import errno
import inspect
import logging
import os
import warnings
import six
import numpy as np

import paddle
from paddle.fluid import (
    core,
    Variable,
    CompiledProgram,
    default_main_program,
    Program,
    layers,
    unique_name,
    program_guard, )
from paddle.fluid.io import prepend_feed_ops, append_fetch_ops
from paddle.fluid.framework import static_only, Parameter
from paddle.fluid.executor import Executor, global_scope
from paddle.fluid.log_helper import get_logger

__all__ = []

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


def _check_args(caller, args, supported_args=None, deprecated_args=None):
    supported_args = [] if supported_args is None else supported_args
    deprecated_args = [] if deprecated_args is None else deprecated_args
    for arg in args:
        if arg in deprecated_args:
            raise ValueError(
                "argument '{}' in function '{}' is deprecated, only {} are supported.".
                format(arg, caller, supported_args))
        elif arg not in supported_args:
            raise ValueError(
                "function '{}' doesn't support argument '{}',\n only {} are supported.".
                format(caller, arg, supported_args))


def _check_vars(name, var_list):
    if not isinstance(var_list, list):
        var_list = [var_list]
    if not var_list or not all([isinstance(var, Variable) for var in var_list]):
        raise ValueError(
            "'{}' should be a Variable or a list of Variable.".format(name))


def _normalize_path_prefix(path_prefix):
    """
    convert path_prefix to absolute path.
    """
    if not isinstance(path_prefix, six.string_types):
        raise ValueError("'path_prefix' should be a string.")
    if path_prefix.endswith("/"):
        raise ValueError("'path_prefix' should not be a directory")
    path_prefix = os.path.normpath(path_prefix)
    path_prefix = os.path.abspath(path_prefix)
    return path_prefix


def _get_valid_program(program=None):
    """
    return default main program if program is None.
    """
    if program is None:
        program = default_main_program()
    elif isinstance(program, CompiledProgram):
        program = program._program
        if program is None:
            raise TypeError(
                "The type of input program is invalid, expected tyep is Program, but received None"
            )
        warnings.warn(
            "The input is a CompiledProgram, this is not recommended.")
    if not isinstance(program, Program):
        raise TypeError(
            "The type of input program is invalid, expected type is fluid.Program, but received %s"
            % type(program))
    return program


def _clone_var_in_block(block, var):
    assert isinstance(var, Variable)
    if var.desc.type() == core.VarDesc.VarType.LOD_TENSOR:
        return block.create_var(
            name=var.name,
            shape=var.shape,
            dtype=var.dtype,
            type=var.type,
            lod_level=var.lod_level,
            persistable=True)
    else:
        return block.create_var(
            name=var.name,
            shape=var.shape,
            dtype=var.dtype,
            type=var.type,
            persistable=True)


def normalize_program(program, feed_vars, fetch_vars):
    """
    :api_attr: Static Graph

    Normalize/Optimize a program according to feed_vars and fetch_vars.

    Args:
        program(Program): Specify a program you want to optimize.
        feed_vars(Variable | list[Variable]): Variables needed by inference.
        fetch_vars(Variable | list[Variable]): Variables returned by inference.

    Returns:
        Program: Normalized/Optimized program.

    Raises:
        TypeError: If `program` is not a Program, an exception is thrown.
        TypeError: If `feed_vars` is not a Variable or a list of Variable, an exception is thrown.
        TypeError: If `fetch_vars` is not a Variable or a list of Variable, an exception is thrown.

    Examples:
        .. code-block:: python

            import paddle

            paddle.enable_static()

            path_prefix = "./infer_model"

            # User defined network, here a softmax regession example
            image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
            label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
            predict = paddle.static.nn.fc(image, 10, activation='softmax')

            loss = paddle.nn.functional.cross_entropy(predict, label)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(paddle.static.default_startup_program())

            # normalize main program.
            program = paddle.static.default_main_program()
            normalized_program = paddle.static.normalize_program(program, [image], [predict])

    """
    if not isinstance(program, Program):
        raise TypeError(
            "program type must be `fluid.Program`, but received `%s`" %
            type(program))
    if not isinstance(feed_vars, list):
        feed_vars = [feed_vars]
    if not all(isinstance(v, Variable) for v in feed_vars):
        raise TypeError(
            "feed_vars type must be a Variable or a list of Variable.")
    if not isinstance(fetch_vars, list):
        fetch_vars = [fetch_vars]
    if not all(isinstance(v, Variable) for v in fetch_vars):
        raise TypeError(
            "fetch_vars type must be a Variable or a list of Variable.")

    # remind users to set auc_states to 0 if auc op were found.
    for op in program.global_block().ops:
        # clear device of Op
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        op._set_attr(device_attr_name, "")
        if op.type == 'auc':
            warnings.warn("Be sure that you have set auc states to 0 "
                          "before saving inference model.")
            break

    # fix the bug that the activation op's output as target will be pruned.
    # will affect the inference performance.
    # TODO(Superjomn) add an IR pass to remove 1-scale op.
    with program_guard(program):
        uniq_fetch_vars = []
        for i, var in enumerate(fetch_vars):
            if var.dtype != paddle.bool:
                var = layers.scale(
                    var, 1., name="save_infer_model/scale_{}".format(i))
            uniq_fetch_vars.append(var)
        fetch_vars = uniq_fetch_vars

    # serialize program
    copy_program = program.clone()
    global_block = copy_program.global_block()
    remove_op_idx = []
    for i, op in enumerate(global_block.ops):
        op.desc.set_is_target(False)
        if op.type == "feed" or op.type == "fetch":
            remove_op_idx.append(i)
    for idx in remove_op_idx[::-1]:
        global_block._remove_op(idx)
    copy_program.desc.flush()

    feed_var_names = [var.name for var in feed_vars]
    copy_program = copy_program._prune_with_input(
        feeded_var_names=feed_var_names, targets=fetch_vars)
    copy_program = copy_program._inference_optimize(prune_read_op=True)
    fetch_var_names = [var.name for var in fetch_vars]
    prepend_feed_ops(copy_program, feed_var_names)
    append_fetch_ops(copy_program, fetch_var_names)
    copy_program.desc._set_version()
    return copy_program


def is_persistable(var):
    """
    Check whether the given variable is persistable.

    Args:
        var(Variable): The variable to be checked.

    Returns:
        bool: True if the given `var` is persistable
        False if not.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid

            paddle.enable_static()
            param = fluid.default_main_program().global_block().var('fc.b')
            res = fluid.io.is_persistable(param)
    """
    if var.desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
                    var.desc.type() == core.VarDesc.VarType.FETCH_LIST or \
                    var.desc.type() == core.VarDesc.VarType.READER:
        return False
    return var.persistable


@static_only
def serialize_program(feed_vars, fetch_vars, **kwargs):
    """
    :api_attr: Static Graph

    Serialize default main program according to feed_vars and fetch_vars.

    Args:
        feed_vars(Variable | list[Variable]): Variables needed by inference.
        fetch_vars(Variable | list[Variable]): Variables returned by inference.
        kwargs: Supported keys including 'program'.Attention please, kwargs is used for backward compatibility mainly.
          - program(Program): specify a program if you don't want to use default main program.

    Returns:
        bytes: serialized program.

    Raises:
        ValueError: If `feed_vars` is not a Variable or a list of Variable, an exception is thrown.
        ValueError: If `fetch_vars` is not a Variable or a list of Variable, an exception is thrown.

    Examples:
        .. code-block:: python

            import paddle

            paddle.enable_static()

            path_prefix = "./infer_model"

            # User defined network, here a softmax regession example
            image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
            label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
            predict = paddle.static.nn.fc(image, 10, activation='softmax')

            loss = paddle.nn.functional.cross_entropy(predict, label)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(paddle.static.default_startup_program())

            # serialize the default main program to bytes.
            serialized_program = paddle.static.serialize_program([image], [predict])

            # deserialize bytes to program
            deserialized_program = paddle.static.deserialize_program(serialized_program)

    """
    # verify feed_vars
    _check_vars('feed_vars', feed_vars)
    # verify fetch_vars
    _check_vars('fetch_vars', fetch_vars)

    program = _get_valid_program(kwargs.get('program', None))
    program = normalize_program(program, feed_vars, fetch_vars)
    return _serialize_program(program)


def _serialize_program(program):
    """
    serialize given program to bytes.
    """
    return program.desc.serialize_to_string()


@static_only
def serialize_persistables(feed_vars, fetch_vars, executor, **kwargs):
    """
    :api_attr: Static Graph

    Serialize parameters using given executor and default main program according to feed_vars and fetch_vars.

    Args:
        feed_vars(Variable | list[Variable]): Variables needed by inference.
        fetch_vars(Variable | list[Variable]): Variables returned by inference.
        kwargs: Supported keys including 'program'.Attention please, kwargs is used for backward compatibility mainly.
          - program(Program): specify a program if you don't want to use default main program.

    Returns:
        bytes: serialized program.

    Raises:
        ValueError: If `feed_vars` is not a Variable or a list of Variable, an exception is thrown.
        ValueError: If `fetch_vars` is not a Variable or a list of Variable, an exception is thrown.

    Examples:
        .. code-block:: python

            import paddle

            paddle.enable_static()

            path_prefix = "./infer_model"

            # User defined network, here a softmax regession example
            image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
            label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
            predict = paddle.static.nn.fc(image, 10, activation='softmax')

            loss = paddle.nn.functional.cross_entropy(predict, label)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(paddle.static.default_startup_program())

            # serialize parameters to bytes.
            serialized_params = paddle.static.serialize_persistables([image], [predict], exe)

            # deserialize bytes to parameters.
            main_program = paddle.static.default_main_program()
            deserialized_params = paddle.static.deserialize_persistables(main_program, serialized_params, exe)

    """
    # verify feed_vars
    _check_vars('feed_vars', feed_vars)
    # verify fetch_vars
    _check_vars('fetch_vars', fetch_vars)

    program = _get_valid_program(kwargs.get('program', None))
    program = normalize_program(program, feed_vars, fetch_vars)
    return _serialize_persistables(program, executor)


def _serialize_persistables(program, executor):
    """
    Serialize parameters using given program and executor.
    """
    vars_ = list(filter(is_persistable, program.list_vars()))
    # warn if no variable found in model
    if len(vars_) == 0:
        warnings.warn("no variable in your model, please ensure there are any "
                      "variables in your model to save")
        return None
    # create a new program and clone persitable vars to it
    save_program = Program()
    save_block = save_program.global_block()
    save_var_map = {}
    for var in vars_:
        if var.type != core.VarDesc.VarType.RAW:
            var_copy = _clone_var_in_block(save_block, var)
            save_var_map[var_copy.name] = var

    # create in_vars and out_var, then append a save_combine op to save_program
    in_vars = []
    for name in sorted(save_var_map.keys()):
        in_vars.append(save_var_map[name])

    out_var_name = unique_name.generate("out_var")
    out_var = save_block.create_var(
        type=core.VarDesc.VarType.RAW, name=out_var_name)
    out_var.desc.set_persistable(True)
    save_block.append_op(
        type='save_combine',
        inputs={'X': in_vars},
        outputs={'Y': out_var},
        attrs={'file_path': '',
               'save_to_memory': True})
    # run save_program to save vars
    # NOTE(zhiqiu): save op will add variable kLookupTablePath to save_program.desc,
    # which leads to diff between save_program and its desc. Call _sync_with_cpp
    # to keep consistency.
    save_program._sync_with_cpp()
    executor.run(save_program)
    # return serialized bytes in out_var
    return global_scope().find_var(out_var_name).get_bytes()


def save_to_file(path, content):
    """
    Save content to given path.
    Args:
        path(str): Path to write content to.
        content(bytes): Content to write.
    Returns:
        None
    """

    if not isinstance(content, bytes):
        raise ValueError("'content' type should be bytes.")
    with open(path, "wb") as f:
        f.write(content)


@static_only
def save_inference_model(path_prefix, feed_vars, fetch_vars, executor,
                         **kwargs):
    """
    :api_attr: Static Graph

    Save current model and its parameters to given path. i.e.
    Given path_prefix = "/path/to/modelname", after invoking
    save_inference_model(path_prefix, feed_vars, fetch_vars, executor),
    you will find two files named modelname.pdmodel and modelname.pdiparams
    under "/path/to", which represent your model and parameters respectively.

    Args:
        path_prefix(str): Directory path to save model + model name without suffix.
        feed_vars(Variable | list[Variable]): Variables needed by inference.
        fetch_vars(Variable | list[Variable]): Variables returned by inference.
        executor(Executor): The executor that saves the inference model. You can refer
                            to :ref:`api_guide_executor_en` for more details.
        kwargs: Supported keys including 'program' and "clip_extra". Attention please, kwargs is used for backward compatibility mainly.
          - program(Program): specify a program if you don't want to use default main program.
          - clip_extra(bool): set to True if you want to clip extra information for every operator.
    Returns:
        None

    Raises:
        ValueError: If `feed_vars` is not a Variable or a list of Variable, an exception is thrown.
        ValueError: If `fetch_vars` is not a Variable or a list of Variable, an exception is thrown.

    Examples:
        .. code-block:: python

            import paddle

            paddle.enable_static()

            path_prefix = "./infer_model"

            # User defined network, here a softmax regession example
            image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
            label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
            predict = paddle.static.nn.fc(image, 10, activation='softmax')

            loss = paddle.nn.functional.cross_entropy(predict, label)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(paddle.static.default_startup_program())

            # Feed data and train process

            # Save inference model. Note we don't save label and loss in this example
            paddle.static.save_inference_model(path_prefix, [image], [predict], exe)

            # In this example, the save_inference_mode inference will prune the default
            # main program according to the network's input node (img) and output node(predict).
            # The pruned inference program is going to be saved in file "./infer_model.pdmodel"
            # and parameters are going to be saved in file "./infer_model.pdiparams".

    """

    # check path_prefix, set model_path and params_path
    path_prefix = _normalize_path_prefix(path_prefix)
    try:
        # mkdir may conflict if pserver and trainer are running on the same machine
        dirname = os.path.dirname(path_prefix)
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    model_path = path_prefix + ".pdmodel"
    params_path = path_prefix + ".pdiparams"
    if os.path.isdir(model_path):
        raise ValueError("'{}' is an existing directory.".format(model_path))
    if os.path.isdir(params_path):
        raise ValueError("'{}' is an existing directory.".format(params_path))

    # verify feed_vars
    _check_vars('feed_vars', feed_vars)
    # verify fetch_vars
    _check_vars('fetch_vars', fetch_vars)

    program = _get_valid_program(kwargs.get('program', None))
    clip_extra = kwargs.get('clip_extra', False)
    program = normalize_program(program, feed_vars, fetch_vars)
    # serialize and save program
    program_bytes = _serialize_program(
        program._remove_training_info(clip_extra=clip_extra))
    save_to_file(model_path, program_bytes)
    # serialize and save params
    params_bytes = _serialize_persistables(program, executor)
    save_to_file(params_path, params_bytes)


@static_only
def deserialize_program(data):
    """
    :api_attr: Static Graph

    Deserialize given data to a program.

    Args:
        data(bytes): serialized program.

    Returns:
        Program: deserialized program.

    Examples:
        .. code-block:: python

            import paddle

            paddle.enable_static()

            path_prefix = "./infer_model"

            # User defined network, here a softmax regession example
            image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
            label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
            predict = paddle.static.nn.fc(image, 10, activation='softmax')

            loss = paddle.nn.functional.cross_entropy(predict, label)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(paddle.static.default_startup_program())

            # serialize the default main program to bytes.
            serialized_program = paddle.static.serialize_program([image], [predict])

            # deserialize bytes to program
            deserialized_program = paddle.static.deserialize_program(serialized_program)

    """
    program = Program.parse_from_string(data)
    if not core._is_program_version_supported(program._version()):
        raise ValueError("Unsupported program version: %d\n" %
                         program._version())
    return program


@static_only
def deserialize_persistables(program, data, executor):
    """
    :api_attr: Static Graph

    Deserialize given data to parameters according to given program and executor.

    Args:
        program(Program): program that contains parameter names (to deserialize).
        data(bytes): serialized parameters.
        executor(Executor): executor used to run load op.

    Returns:
        Program: deserialized program.

    Examples:
        .. code-block:: python

            import paddle

            paddle.enable_static()

            path_prefix = "./infer_model"

            # User defined network, here a softmax regession example
            image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
            label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
            predict = paddle.static.nn.fc(image, 10, activation='softmax')

            loss = paddle.nn.functional.cross_entropy(predict, label)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(paddle.static.default_startup_program())

            # serialize parameters to bytes.
            serialized_params = paddle.static.serialize_persistables([image], [predict], exe)

            # deserialize bytes to parameters.
            main_program = paddle.static.default_main_program()
            deserialized_params = paddle.static.deserialize_persistables(main_program, serialized_params, exe)


    """
    if not isinstance(program, Program):
        raise TypeError(
            "program type must be `fluid.Program`, but received `%s`" %
            type(program))
    # load params to a tmp program
    load_program = Program()
    load_block = load_program.global_block()
    vars_ = list(filter(is_persistable, program.list_vars()))

    origin_shape_map = {}
    load_var_map = {}
    check_vars = []
    sparse_vars = []
    for var in vars_:
        assert isinstance(var, Variable)
        if var.type == core.VarDesc.VarType.RAW:
            continue
        if isinstance(var, Parameter):
            origin_shape_map[var.name] = tuple(var.desc.get_shape())
        if var.type == core.VarDesc.VarType.SELECTED_ROWS:
            sparse_vars.append(var)
            continue
        var_copy = _clone_var_in_block(load_block, var)
        check_vars.append(var)
        load_var_map[var_copy.name] = var_copy

    # append load_combine op to load parameters,
    load_var_list = []
    for name in sorted(load_var_map.keys()):
        load_var_list.append(load_var_map[name])
    load_block.append_op(
        type='load_combine',
        inputs={},
        outputs={"Out": load_var_list},
        # if load from memory, file_path is data
        attrs={'file_path': data,
               'model_from_memory': True})
    executor.run(load_program)
    # check var shape
    for var in check_vars:
        if not isinstance(var, Parameter):
            continue
        var_tmp = paddle.fluid.global_scope().find_var(var.name)
        assert var_tmp != None, "can't not find var: " + var.name
        new_shape = (np.array(var_tmp.get_tensor())).shape
        assert var.name in origin_shape_map, var.name + " MUST in var list."
        origin_shape = origin_shape_map.get(var.name)
        if new_shape != origin_shape:
            raise RuntimeError(
                "Shape mismatch, program needs a parameter with shape ({}), "
                "but the loaded parameter ('{}') has a shape of ({}).".format(
                    origin_shape, var.name, new_shape))


def load_from_file(path):
    """
    Load file in binary mode.
    Args:
        path(str): Path of an existed file.
    Returns:
        bytes: Content of file.
    """
    with open(path, 'rb') as f:
        data = f.read()
    return data


@static_only
def load_inference_model(path_prefix, executor, **kwargs):
    """
    :api_attr: Static Graph

    Load inference model from a given path. By this API, you can get the model
    structure(Inference Program) and model parameters.

    Args:
        path_prefix(str | None): One of the following:
          - Directory path to save model + model name without suffix.
          - Set to None when reading the model from memory.
        executor(Executor): The executor to run for loading inference model.
                            See :ref:`api_guide_executor_en` for more details about it.
        kwargs: Supported keys including 'model_filename', 'params_filename'.Attention please, kwargs is used for backward compatibility mainly.
          - model_filename(str): specify model_filename if you don't want to use default name.
          - params_filename(str): specify params_filename if you don't want to use default name.

    Returns:
        list: The return of this API is a list with three elements:
        (program, feed_target_names, fetch_targets). The `program` is a
        ``Program`` (refer to :ref:`api_guide_Program_en`), which is used for inference.
        The `feed_target_names` is a list of ``str``, which contains names of variables
        that need to feed data in the inference program. The `fetch_targets` is a list of
        ``Variable`` (refer to :ref:`api_guide_Program_en`). It contains variables from which
        we can get inference results.

    Raises:
        ValueError: If `path_prefix.pdmodel` or `path_prefix.pdiparams`  doesn't exist.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            paddle.enable_static()

            # Build the model
            startup_prog = paddle.static.default_startup_program()
            main_prog = paddle.static.default_main_program()
            with paddle.static.program_guard(main_prog, startup_prog):
                image = paddle.static.data(name="img", shape=[64, 784])
                w = paddle.create_parameter(shape=[784, 200], dtype='float32')
                b = paddle.create_parameter(shape=[200], dtype='float32')
                hidden_w = paddle.matmul(x=image, y=w)
                hidden_b = paddle.add(hidden_w, b)
            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup_prog)

            # Save the inference model
            path_prefix = "./infer_model"
            paddle.static.save_inference_model(path_prefix, [image], [hidden_b], exe)

            [inference_program, feed_target_names, fetch_targets] = (
                paddle.static.load_inference_model(path_prefix, exe))
            tensor_img = np.array(np.random.random((64, 784)), dtype=np.float32)
            results = exe.run(inference_program,
                          feed={feed_target_names[0]: tensor_img},
                          fetch_list=fetch_targets)

            # In this example, the inference program was saved in file
            # "./infer_model.pdmodel" and parameters were saved in file
            # " ./infer_model.pdiparams".
            # By the inference program, feed_target_names and
            # fetch_targets, we can use an executor to run the inference
            # program to get the inference result.
    """
    # check kwargs
    supported_args = ('model_filename', 'params_filename')
    deprecated_args = ('pserver_endpoints', )
    caller = inspect.currentframe().f_code.co_name
    _check_args(caller, kwargs, supported_args, deprecated_args)

    # load from memory
    if path_prefix is None:
        _logger.warning("Load inference model from memory is deprecated.")
        model_filename = kwargs.get('model_filename', None)
        params_filename = kwargs.get('params_filename', None)
        if params_filename is None:
            raise ValueError(
                "params_filename cannot be None when path_prefix is None.")
        load_dirname = ''
        program_bytes = model_filename
        params_bytes = params_filename
    # load from file
    else:
        # check and norm path_prefix
        path_prefix = _normalize_path_prefix(path_prefix)

        # set model_path and params_path in new way,
        # path_prefix represents a file path without suffix in this case.
        if not kwargs:
            model_path = path_prefix + ".pdmodel"
            params_path = path_prefix + ".pdiparams"
        # set model_path and params_path in old way for compatible,
        # path_prefix represents a directory path.
        else:
            model_filename = kwargs.get('model_filename', None)
            params_filename = kwargs.get('params_filename', None)
            # set model_path
            if model_filename is None:
                model_path = os.path.join(path_prefix, "__model__")
            else:
                model_path = os.path.join(path_prefix,
                                          model_filename + ".pdmodel")
                if not os.path.exists(model_path):
                    model_path = os.path.join(path_prefix, model_filename)
            # set params_path
            if params_filename is None:
                params_path = os.path.join(path_prefix, "")
            else:
                params_path = os.path.join(path_prefix,
                                           params_filename + ".pdiparams")
                if not os.path.exists(params_path):
                    params_path = os.path.join(path_prefix, params_filename)
            _logger.warning("The old way to load inference model is deprecated."
                            " model path: {}, params path: {}".format(
                                model_path, params_path))
        program_bytes = load_from_file(model_path)
        load_dirname = os.path.dirname(params_path)
        params_filename = os.path.basename(params_path)
        # load params data
        params_path = os.path.join(load_dirname, params_filename)
        params_bytes = load_from_file(params_path)

    # deserialize bytes to program
    program = deserialize_program(program_bytes)
    # deserialize bytes to params
    deserialize_persistables(program, params_bytes, executor)

    feed_target_names = program.desc.get_feed_target_names()
    fetch_target_names = program.desc.get_fetch_target_names()
    fetch_targets = [
        program.global_block().var(name) for name in fetch_target_names
    ]

    return [program, feed_target_names, fetch_targets]
