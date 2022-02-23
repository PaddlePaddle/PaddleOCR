# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import pickle
import warnings
import functools
from collections import OrderedDict
import inspect

import six
import paddle
from paddle.fluid import core
from paddle.fluid.compiler import BuildStrategy, CompiledProgram, ExecutionStrategy
from paddle.fluid.data_feeder import check_type
from paddle.fluid.layers.utils import flatten, pack_sequence_as
from paddle.fluid.dygraph.base import program_desc_tracing_guard, switch_to_static_graph
from paddle.fluid.dygraph.dygraph_to_static import logging_utils
from paddle.fluid.dygraph.dygraph_to_static.convert_call_func import ConversionOptions, CONVERSION_OPTIONS
from paddle.fluid.dygraph.dygraph_to_static.logging_utils import set_code_level, set_verbosity
from paddle.fluid.dygraph.dygraph_to_static.program_translator import ProgramTranslator, StaticFunction, unwrap_decorators
from paddle.fluid.dygraph.io import TranslatedLayer, INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX, INFER_PARAMS_INFO_SUFFIX
from paddle.fluid.dygraph.layers import Layer
from paddle.fluid.executor import Executor, scope_guard
from paddle.fluid.framework import Block, ParamBase, Program, Variable, Parameter
from paddle.fluid.framework import _current_expected_place, _dygraph_guard, _dygraph_tracer
from paddle.fluid.framework import dygraph_only, in_dygraph_mode
from paddle.fluid.wrapped_decorator import wrap_decorator

__all__ = [
    'TracedLayer', 'declarative', 'dygraph_to_static_func', 'set_code_level',
    'set_verbosity', 'save', 'load', 'not_to_static'
]


def create_program_from_desc(program_desc):
    program = Program()
    program.desc = program_desc
    program.blocks = [Block(program, 0)]
    program._sync_with_cpp()
    return program


def _extract_vars(inputs, result_list, err_tag='inputs'):
    if isinstance(inputs, Variable):
        result_list.append(inputs)
    elif isinstance(inputs, (list, tuple)):
        for var in inputs:
            _extract_vars(var, result_list, err_tag)
    else:
        raise TypeError(
            "The type of 'each element of {}' in fluid.dygraph.jit.TracedLayer.trace must be fluid.Variable, but received {}.".
            format(err_tag, type(inputs)))


def extract_vars(inputs, err_tag='inputs'):
    result_list = []
    _extract_vars(inputs, result_list, err_tag)
    return result_list


def _dygraph_to_static_func_(dygraph_func):
    """
    Converts imperative dygraph APIs into declarative function APIs. Decorator
    @dygraph_to_static_func only converts imperative dygraph APIs into
    declarative net-building APIs, which means it doesn't return immediate
    digital result as imperative mode. Users should handle Program and Executor
    by themselves.

    Note:
    This decorator is NOT our recommended way to transform imperative function
    to declarative function. We will remove this decorator after we finalize
    cleaning up code.

    Args:
        dygraph_func (callable): callable imperative function.

    Returns:
        Callable: converting imperative dygraph APIs into declarative
        net-building APIs.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np
          from paddle.fluid.dygraph.jit import dygraph_to_static_func

          @dygraph_to_static_func
          def func(x):
              if fluid.layers.mean(x) < 0:
                  x_v = x - 1
              else:
                  x_v = x + 1

               return x_v

          x = fluid.layers.fill_constant(shape=[3, 3], value=0, dtype='float64')

          x_v = func(x)
          exe = fluid.Executor(fluid.CPUPlace())
          out = exe.run(fetch_list=[x_v])
          print(out[0])
          # [[1. 1. 1.]
          #  [1. 1. 1.]
          #  [1. 1. 1.]]

    """

    # TODO: remove this decorator after we finalize training API
    def __impl__(*args, **kwargs):
        program_translator = ProgramTranslator()
        if in_dygraph_mode() or not program_translator.enable_to_static:
            logging_utils.warn(
                "The decorator 'dygraph_to_static_func' doesn't work in "
                "dygraph mode or set ProgramTranslator.enable to False. "
                "We will just return dygraph output.")
            return dygraph_func(*args, **kwargs)
        static_func = program_translator.get_func(dygraph_func)
        return static_func(*args, **kwargs)

    return __impl__


dygraph_to_static_func = wrap_decorator(_dygraph_to_static_func_)


def copy_decorator_attrs(original_func, decorated_obj):
    """
    Copies some necessary attributes from original function into decorated function.

    Args:
        original_func(callable): the original decorated function.
        decorated_obj(StaticFunction): the target decorated StaticFunction object.
    """
    decorator_name = "declarative"

    decorated_obj.__name__ = original_func.__name__
    decorated_obj._decorator_name = decorator_name
    decorated_obj.__wrapped__ = original_func
    decorated_obj.__doc__ = original_func.__doc__
    if hasattr(original_func, "__module__"):
        decorated_obj.__module__ = original_func.__module__

    return decorated_obj


def declarative(function=None, input_spec=None, build_strategy=None):
    """
    Converts imperative dygraph APIs into declarative function APIs. Decorator
    @declarative handles the Program and Executor of static mode and returns
    the result as dygraph Tensor(s). Users could use the returned dygraph
    Tensor(s) to do imperative training, inference, or other operations. If the
    decorated function calls other imperative function, the called one will be
    converted into declarative function as well.

    Args:
        function (callable): callable imperative function.
        input_spec(list[InputSpec]|tuple[InputSpec]): list/tuple of InputSpec to specific the shape/dtype/name
            information of each input Tensor.
        build_strategy(BuildStrategy|None): This argument is used to compile the
            converted program with the specified options, such as operators' fusion
            in the computational graph and memory optimization during the execution
            of the computational graph. For more information about build_strategy,
            please refer to :code:`paddle.static.BuildStrategy`. The default is None.


    Returns:
        Tensor(s): containing the numerical result.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.jit import to_static

            @to_static
            def func(x):
                if paddle.mean(x) < 0:
                    x_v = x - 1
                else:
                    x_v = x + 1
                return x_v

            x = paddle.ones([1, 2], dtype='float32')
            x_v = func(x)
            print(x_v) # [[2. 2.]]

    """

    def decorated(python_func):
        """
        Decorates a python function into a StaticFunction object.
        """
        # Step 1. unwrap the function if it is already decorated.
        _, python_func = unwrap_decorators(python_func)

        # Step 2. copy some attributes from original python function.
        static_layer = copy_decorator_attrs(
            original_func=python_func,
            decorated_obj=StaticFunction(
                function=python_func,
                input_spec=input_spec,
                build_strategy=build_strategy))

        return static_layer

    build_strategy = build_strategy or BuildStrategy()
    if not isinstance(build_strategy, BuildStrategy):
        raise TypeError(
            "Required type(build_strategy) shall be `paddle.static.BuildStrategy`, but received {}".
            format(type(build_strategy).__name__))

    # for usage: `declarative(foo, ...)`
    if function is not None:
        if isinstance(function, Layer):
            if isinstance(function.forward, StaticFunction):
                class_name = function.__class__.__name__
                logging_utils.warn(
                    "`{}.forward` has already been decorated somewhere. It will be redecorated to replace previous one.".
                    format(class_name))
            function.forward = decorated(function.forward)
            return function
        else:
            return decorated(function)

    # for usage: `@declarative`
    return decorated


def not_to_static(func=None):
    """
    A Decorator to suppresses the convertion of a function.

    Args:
        func(callable): The function to decorate.

    Returns:
        callable: A function which won't be converted in Dynamic-to-Static.

    Examples:
        .. code-block:: python

            import paddle

            @paddle.jit.not_to_static
            def func_not_to_static(x):
                res = x - 1
                return res

            @paddle.jit.to_static
            def func(x):
                if paddle.mean(x) < 0:
                    out = func_not_to_static(x)
                else:
                    out = x + 1
                return out

            x = paddle.ones([1, 2], dtype='float32')
            out = func(x)
            print(out) # [[2. 2.]]
    """
    if func is None:
        return not_to_static

    options = ConversionOptions(not_convert=True)
    setattr(func, CONVERSION_OPTIONS, options)
    return func


class _SaveLoadConfig(object):
    def __init__(self):
        self._output_spec = None
        self._model_filename = None
        self._params_filename = None
        self._separate_params = False
        # used for `paddle.load`
        self._keep_name_table = False

        # NOTE: Users rarely use following configs, so these configs are not open to users,
        # reducing user learning costs, but we retain the configuration capabilities

        # If True, programs are modified to only support direct inference deployment.
        # Otherwise,more information will be stored for flexible optimization and re-training.
        # Currently, only True is supported
        self._export_for_deployment = True

        # If True, It will save inference program only, and do not save params of Program
        self._program_only = False

    @property
    def output_spec(self):
        return self._output_spec

    @output_spec.setter
    def output_spec(self, spec):
        if spec is None:
            return
        if not isinstance(spec, list):
            raise TypeError(
                "The config `output_spec` should be 'list', but received input type is %s."
                % type(input))
            for var in spec:
                if not isinstance(var, core.VarBase):
                    raise TypeError(
                        "The element in config `output_spec` list should be 'Variable', but received element's type is %s."
                        % type(var))
        self._output_spec = spec

    @property
    def model_filename(self):
        return self._model_filename

    @model_filename.setter
    def model_filename(self, filename):
        if filename is None:
            return
        if not isinstance(filename, six.string_types):
            raise TypeError(
                "The config `model_filename` should be str, but received input's type is %s."
                % type(filename))
        if len(filename) == 0:
            raise ValueError("The config `model_filename` is empty string.")
        self._model_filename = filename

    @property
    def params_filename(self):
        return self._params_filename

    @params_filename.setter
    def params_filename(self, filename):
        if filename is None:
            return
        if not isinstance(filename, six.string_types):
            raise TypeError(
                "The config `params_filename` should be str, but received input's type is %s."
                % type(filename))
        if len(filename) == 0:
            raise ValueError("The config `params_filename` is empty string.")
        self._params_filename = filename

    @property
    def keep_name_table(self):
        return self._keep_name_table

    @keep_name_table.setter
    def keep_name_table(self, value):
        if value is None:
            return
        if not isinstance(value, bool):
            raise TypeError(
                "The config `keep_name_table` should be bool value, but received input's type is %s."
                % type(value))
        self._keep_name_table = value


def _parse_save_configs(configs):
    supported_configs = ['output_spec']

    # input check
    for key in configs:
        if key not in supported_configs:
            raise ValueError(
                "The additional config (%s) of `paddle.jit.save` is not supported."
                % (key))

    # construct inner config
    inner_config = _SaveLoadConfig()
    inner_config.output_spec = configs.get('output_spec', None)

    return inner_config


def _parse_load_config(configs):
    supported_configs = ['model_filename', 'params_filename']

    # input check
    for key in configs:
        if key not in supported_configs:
            raise ValueError(
                "The additional config (%s) of `paddle.jit.load` is not supported."
                % (key))

    # construct inner config
    inner_config = _SaveLoadConfig()
    inner_config.model_filename = configs.get('model_filename', None)
    inner_config.params_filename = configs.get('params_filename', None)

    return inner_config


def _get_input_var_names(inputs, input_spec):
    name_none_error = "The %s's name is None. " \
        "When using jit.save, please set InputSepc's name in " \
        "to_static(input_spec=[]) and jit.save(input_spec=[]) " \
        "and make sure they are consistent."
    name_no_exists_error = "The tensor `%s` does not exists. " \
        "Please make sure the name of InputSpec or example Tensor " \
        "in input_spec is the same as the name of InputSpec in " \
        "`to_static` decorated on the Layer.forward method."
    result_list = []
    input_var_names = [
        var.name for var in flatten(inputs) if isinstance(var, Variable)
    ]
    if input_spec is None:
        # no prune
        return input_var_names
    else:
        # fileter out non-tensor type spec infos.
        input_spec = [
            spec for spec in input_spec
            if isinstance(spec, paddle.static.InputSpec)
        ]

    if len(input_spec) == len(input_var_names):
        # no prune
        result_list = input_var_names
        # if input spec name not in input_var_names, only raise warning
        for spec in input_spec:
            if spec.name is None:
                warnings.warn(name_none_error % spec)
            elif spec.name not in input_var_names:
                warnings.warn(name_no_exists_error % spec.name)
            else:
                # do nothing
                pass
    else:
        # prune
        for spec in input_spec:
            if spec.name is None:
                # name is None, the input_spec only can be InputSpec
                raise ValueError(name_none_error % spec)
            elif spec.name not in input_var_names:
                # the input_spec can be `InputSpec` or `VarBase`
                raise ValueError(name_no_exists_error % spec.name)
            else:
                result_list.append(spec.name)

    return result_list


def _get_output_vars(outputs, output_spec):
    name_no_exists_error = "The tensor `%s` does not exists. " \
        "Please make sure the name of example Tensor " \
        "in configs.output_spec is the output tensor of " \
        "Layer.forward method."
    result_list = []
    output_vars_dict = OrderedDict()
    for var in flatten(outputs):
        if isinstance(var, Variable):
            output_vars_dict[var.name] = var
    if output_spec is None:
        result_list = output_vars_dict.values()
    elif output_spec is not None and len(output_spec) == len(output_vars_dict):
        result_list = output_vars_dict.values()
        for var in output_spec:
            if var.name not in output_vars_dict:
                warnings.warn(name_no_exists_error % var.name)
    else:
        for var in output_spec:
            if var.name not in output_vars_dict:
                raise ValueError(name_no_exists_error % var.name)
            else:
                result_list.append(output_vars_dict[var.name])
    return result_list


# NOTE(chenweihang): [ Handling of use cases of API paddle.jit.load ]
# `paddle.jit.load` may be used to load saved results of:
# 1. Expected cases:
#   - paddle.jit.save
#   - paddle.static.save_inference_model
#   - paddle.fluid.io.save_inference_model
# 2. Error cases:
#   - paddle.save: no .pdmodel for prefix
#   - paddle.static.save: no .pdiparams but .pdparams exists
#   - paddle.fluid.io.save_params/save_persistables: no __model__
# TODO(chenweihang): polish error message in above error cases
def _build_load_path_and_config(path, config):
    # NOTE(chenweihang): If both [prefix save format] and [directory save format] exist,
    # raise error, avoid confusing behavior
    prefix_format_path = path + INFER_MODEL_SUFFIX
    prefix_format_exist = os.path.exists(prefix_format_path)
    directory_format_exist = os.path.isdir(path)
    if prefix_format_exist and directory_format_exist:
        raise ValueError(
            "The %s.pdmodel and %s directory exist at the same time, "
            "don't know which one to load, please make sure that the specified target "
            "of ``path`` is unique." % (path, path))
    elif not prefix_format_exist and not directory_format_exist:
        raise ValueError("The ``path`` (%s) to load model not exists." % path)
    else:
        if prefix_format_exist:
            file_prefix = os.path.basename(path)
            model_path = os.path.dirname(path)
            if config.model_filename is not None:
                warnings.warn(
                    "When loading the result saved with the "
                    "specified file prefix, the ``model_filename`` config does "
                    "not take effect.")
            config.model_filename = file_prefix + INFER_MODEL_SUFFIX
            if config.params_filename is not None:
                warnings.warn(
                    "When loading the result saved with the "
                    "specified file prefix, the ``params_filename`` config does "
                    "not take effect.")
            config.params_filename = file_prefix + INFER_PARAMS_SUFFIX
        else:
            # Compatible with the old save_inference_model format
            model_path = path

    return model_path, config


@switch_to_static_graph
def save(layer, path, input_spec=None, **configs):
    """
    Saves input Layer or function as ``paddle.jit.TranslatedLayer``
    format model, which can be used for inference or fine-tuning after loading.

    It will save the translated program and all related persistable
    variables of input Layer to given ``path`` .

    ``path`` is the prefix of saved objects, and the saved translated program file
    suffix is ``.pdmodel`` , the saved persistable variables file suffix is ``.pdiparams`` ,
    and here also saved some additional variable description information to a file,
    its suffix is ``.pdiparams.info``, these additional information is used in fine-tuning.

    The saved model can be loaded by follow APIs:
      - ``paddle.jit.load``
      - ``paddle.static.load_inference_model``
      - Other C++ inference APIs

    .. note::
        When using ``paddle.jit.save`` to save a function, parameters will not be saved. If you have to 
        save the parameter, please pass the Layer containing function and parameter to ``paddle.jit.save``.

    Args:
        layer (Layer|function): The Layer or function to be saved.
        path (str): The path prefix to save model. The format is ``dirname/file_prefix`` or ``file_prefix``.
        input_spec (list or tuple[InputSpec|Tensor|Python built-in variable], optional): Describes the input of the saved model's forward
            method, which can be described by InputSpec or example Tensor. Moreover, we support to specify non-tensor type argument,
            such as int, float, string, or list/dict of them.If None, all input variables of
            the original Layer's forward method would be the inputs of the saved model. Default None.
        **configs (dict, optional): Other save configuration options for compatibility. We do not
            recommend using these configurations, they may be removed in the future. If not necessary,
            DO NOT use them. Default None.
            The following options are currently supported:
            (1) output_spec (list[Tensor]): Selects the output targets of the saved model.
            By default, all return variables of original Layer's forward method are kept as the
            output of the saved model. If the provided ``output_spec`` list is not all output variables,
            the saved model will be pruned according to the given ``output_spec`` list.

    Returns:
        None

    Examples:
        .. code-block:: python

            # example 1: save layer
            import numpy as np
            import paddle
            import paddle.nn as nn
            import paddle.optimizer as opt

            BATCH_SIZE = 16
            BATCH_NUM = 4
            EPOCH_NUM = 4

            IMAGE_SIZE = 784
            CLASS_NUM = 10

            # define a random dataset
            class RandomDataset(paddle.io.Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples

                def __getitem__(self, idx):
                    image = np.random.random([IMAGE_SIZE]).astype('float32')
                    label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
                    return image, label

                def __len__(self):
                    return self.num_samples

            class LinearNet(nn.Layer):
                def __init__(self):
                    super(LinearNet, self).__init__()
                    self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

                @paddle.jit.to_static
                def forward(self, x):
                    return self._linear(x)

            def train(layer, loader, loss_fn, opt):
                for epoch_id in range(EPOCH_NUM):
                    for batch_id, (image, label) in enumerate(loader()):
                        out = layer(image)
                        loss = loss_fn(out, label)
                        loss.backward()
                        opt.step()
                        opt.clear_grad()
                        print("Epoch {} batch {}: loss = {}".format(
                            epoch_id, batch_id, np.mean(loss.numpy())))

            # 1. train & save model.

            # create network
            layer = LinearNet()
            loss_fn = nn.CrossEntropyLoss()
            adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

            # create data loader
            dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
            loader = paddle.io.DataLoader(dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                num_workers=2)

            # train
            train(layer, loader, loss_fn, adam)

            # save
            path = "example_model/linear"
            paddle.jit.save(layer, path)

            # example 2: save function
            import paddle
            from paddle.static import InputSpec


            def save_function():
                @paddle.jit.to_static
                def fun(inputs):
                    return paddle.tanh(inputs)

                path = 'test_jit_save_load_function_1/func'
                inps = paddle.rand([3, 6])
                origin = fun(inps)

                paddle.jit.save(fun, path)
                load_func = paddle.jit.load(path)

                load_result = load_func(inps)
                print((load_result - origin).abs().max() < 1e-10)
                
            save_function()
    """

    # 1. input build & check
    prog_translator = ProgramTranslator()
    if not prog_translator.enable_to_static:
        raise RuntimeError(
            "The paddle.jit.save doesn't work when setting ProgramTranslator.enable to False."
        )

    if not (isinstance(layer, Layer) or inspect.isfunction(layer) or isinstance(
            layer, StaticFunction)):
        raise TypeError(
            "The input of paddle.jit.save should be 'Layer' or 'Function', but received input type is %s."
            % type(layer))
    elif inspect.isfunction(layer) or isinstance(layer, StaticFunction):
        warnings.warn(
            'What you save is a function, and `jit.save` will generate the name of the model file according to `path` you specify. When loading these files with `jit.load`, you get a `TranslatedLayer` whose inference result is the same as the inference result of the function you saved.'
        )

    # NOTE(chenweihang): If the input layer be wrapped by DataParallel,
    # the args and kwargs of forward method will can't be parsed by
    # function_spec, so here we save DataParallel._layers instead
    # DataParallel it self
    # NOTE(chenweihang): using inner_layer, do not change input layer
    if isinstance(layer, paddle.DataParallel):
        inner_layer = layer._layers
    else:
        inner_layer = layer

    # path check
    file_prefix = os.path.basename(path)
    if file_prefix == "":
        raise ValueError(
            "The input path MUST be format of dirname/file_prefix "
            "[dirname\\file_prefix in Windows system], but received "
            "file_prefix is empty string.")

    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

    # avoid change user given input_spec
    inner_input_spec = None
    if input_spec is not None:
        if isinstance(layer, Layer):
            for attr_func in dir(inner_layer):
                static_func = getattr(inner_layer, attr_func, None)
                if isinstance(static_func,
                              StaticFunction) and 'forward' != attr_func:
                    raise ValueError(
                        "If there are static functions other than 'forward' that need to be saved, the input 'input_spec' should be None, but received the type of 'input_spec' is %s."
                        % type(input_spec))

        if not isinstance(input_spec, (list, tuple)):
            raise TypeError(
                "The input input_spec should be 'list', but received input_spec's type is %s."
                % type(input_spec))
        inner_input_spec = []
        for var in flatten(input_spec):
            if isinstance(var, paddle.static.InputSpec):
                inner_input_spec.append(var)
            elif isinstance(var, (core.VarBase, Variable)):
                inner_input_spec.append(
                    paddle.static.InputSpec.from_tensor(var))
            else:
                # NOTE(Aurelius84): Support non-Tensor type in `input_spec`.
                inner_input_spec.append(var)

    # parse configs
    configs = _parse_save_configs(configs)
    scope = core.Scope()
    extra_var_info = dict()
    if isinstance(layer, Layer):
        functions = dir(inner_layer)
    else:
        # layer is function
        functions = [layer, ]
    for attr_func in functions:
        if isinstance(layer, Layer):
            static_func = getattr(inner_layer, attr_func, None)
            if isinstance(static_func, StaticFunction):
                concrete_program = static_func.concrete_program_specify_input_spec(
                    inner_input_spec)
            elif 'forward' == attr_func:
                # transform in jit.save, if input_spec is incomplete, declarative will throw error
                # inner_input_spec is list[InputSpec], it should be packed with same structure
                # as original input_spec here.
                if inner_input_spec:
                    inner_input_spec = pack_sequence_as(input_spec,
                                                        inner_input_spec)
                static_forward = declarative(
                    inner_layer.forward, input_spec=inner_input_spec)
                concrete_program = static_forward.concrete_program
                # the input_spec has been used in declarative, which is equal to
                # @declarative with input_spec and jit.save without input_spec,
                # avoid needless warning
                inner_input_spec = None
            else:
                continue

        else:
            # When layer is a function
            if isinstance(attr_func, StaticFunction):
                concrete_program = attr_func.concrete_program_specify_input_spec(
                    inner_input_spec)
            else:
                if inner_input_spec:
                    inner_input_spec = pack_sequence_as(input_spec,
                                                        inner_input_spec)
                static_function = declarative(
                    attr_func, input_spec=inner_input_spec)
                concrete_program = static_function.concrete_program

                if static_function._class_instance is None:
                    warnings.warn(
                        '`jit.save` will only save the `Program`, not the parameters. If you have to save the parameters, please make sure that {} is a member function of `paddle.nn.Layer` and the saved parameters are in `state_dict`'.
                        format(layer))

        dygraph_state_dict = None
        if isinstance(inner_layer, Layer):
            dygraph_state_dict = inner_layer.to_static_state_dict()
        elif isinstance(attr_func, StaticFunction):
            if attr_func._class_instance:
                dygraph_state_dict = attr_func._class_instance.to_static_state_dict(
                )

        if dygraph_state_dict:
            # NOTE(chenweihang): we maintain the mapping of variable name to
            # structured name, the buffer variable (non-persistable)
            # saved to inference program may not need by dygraph Layer,
            # we only record the state_dict variable's structured name
            state_names_dict = dict()
            state_var_dict = dict()
            for structured_name, var in six.iteritems(dygraph_state_dict):
                state_names_dict[var.name] = structured_name
                state_var_dict[var.name] = var

            # 3. share parameters from Layer to scope & record var info
            for param_or_buffer in concrete_program.parameters:
                # share to scope
                if param_or_buffer.type == core.VarDesc.VarType.VOCAB:
                    scr_tensor = param_or_buffer.value().get_map_tensor()
                    tgt_var = scope.var(param_or_buffer.name)
                    tgt_var.set_vocab(scr_tensor)
                else:
                    param_or_buffer_tensor = scope.var(
                        param_or_buffer.name).get_tensor()
                    #src_tensor = param_or_buffer.value().get_tensor()
                    src_tensor = state_var_dict[param_or_buffer.name].value(
                    ).get_tensor()
                    param_or_buffer_tensor._share_data_with(src_tensor)
                # record var info
                if param_or_buffer.name not in extra_var_info:
                    extra_info_dict = dict()
                    if param_or_buffer.name in state_names_dict:
                        extra_info_dict['structured_name'] = state_names_dict[
                            param_or_buffer.name]
                    extra_info_dict[
                        'stop_gradient'] = param_or_buffer.stop_gradient
                    if isinstance(param_or_buffer, ParamBase):
                        extra_info_dict['trainable'] = param_or_buffer.trainable
                    extra_var_info[param_or_buffer.name] = extra_info_dict

        # 4. build input & output of save_infernece_model
        # NOTE(chenweihang): [ Get input variables name ]
        # There are two cases, whether to prune the inputs or not
        # - not prune inputs (recommend):
        #   - the len(input_spec) == len((concrete_program.inputs) - 1
        #   - here can use concrete_program.inputs directly
        # - prune inputs:
        #   - the input_spec length < len((concrete_program.inputs) - 1
        #   - the input_spec's name should be in concrete_program.inputs
        input_var_names = _get_input_var_names(concrete_program.inputs,
                                               inner_input_spec)

        # NOTE(chenweihang): [ Get output variables ]
        # the rule is like [ Get input variables name ]. For output var,
        # we only support VarBase spec, and actually, we only need the
        # var name of output, and we don't recommended to use output_spec
        output_vars = _get_output_vars(concrete_program.outputs,
                                       configs.output_spec)

        # 5. save inference model
        from paddle.fluid.io import save_inference_model

        # construct new save_inference_model arguments
        model_path = dirname
        # NOTE(chenweihang): because prefix contains model and params filename,
        # so we don't support set model_filename & params_filename
        if 'forward' == attr_func or not isinstance(layer, Layer):
            model_filename = file_prefix + INFER_MODEL_SUFFIX
            params_filename = file_prefix + INFER_PARAMS_SUFFIX
        else:
            model_filename = file_prefix + '.' + attr_func + INFER_MODEL_SUFFIX
            params_filename = file_prefix + '.' + attr_func + INFER_PARAMS_SUFFIX

        with scope_guard(scope):
            save_inference_model(
                dirname=model_path,
                feeded_var_names=input_var_names,
                target_vars=output_vars,
                executor=Executor(_current_expected_place()),
                main_program=concrete_program.main_program.clone(),
                model_filename=model_filename,
                params_filename=params_filename,
                export_for_deployment=configs._export_for_deployment,
                program_only=configs._program_only,
                clip_extra=False)

    # NOTE(chenweihang): [ Save extra variable info ]
    # save_inference_model will lose some important variable information, including:
    #   - Variable name and correspondence (when saved variables as one file)
    #   - Variable.stop_gradient information
    #   - Which persistent variable are parameter and which are not
    #   - Parameter.trainable information
    #
    # The lost information cannot be recovered when it is loaded again,
    # so if we want to perform fine-tune after loading, we may need to
    # configure redundant information to proceed.
    #
    # Due to compatibility issues, we cannot change the original storage structure,
    # but we can save these information in `jit.save` without changing the original
    # storage to improve user experience. So we save extra information into
    # file `***.pdiparams.info`

    # "layer" can only be Layer or function or StaticFunction.

    contain_parameter = False
    for var in concrete_program.main_program.list_vars():
        contain_parameter |= isinstance(var, Parameter)

    if (isinstance(layer, Layer) or contain_parameter) and extra_var_info:
        with scope_guard(scope):
            extra_var_info_path = path + INFER_PARAMS_INFO_SUFFIX
            with open(extra_var_info_path, 'wb') as f:
                pickle.dump(extra_var_info, f, protocol=2)


@dygraph_only
def load(path, **configs):
    """
    :api_attr: imperative

    Load model saved by ``paddle.jit.save`` or ``paddle.static.save_inference_model`` or
    paddle 1.x API ``paddle.fluid.io.save_inference_model`` as ``paddle.jit.TranslatedLayer``,
    then performing inference or fine-tune training.

    .. note::
        If you load model saved by ``paddle.static.save_inference_model`` ,
        there will be the following limitations when using it in fine-tuning:
        1. Imperative mode do not support LoDTensor. All original model's feed targets or parametars that depend on LoD are temporarily unavailable.
        2. All saved model's feed targets need to be passed into TranslatedLayer's forward function.
        3. The variable's ``stop_gradient`` information is lost and can not be recovered.
        4. The parameter's ``trainable`` information is lost and can not be recovered.

    Args:
        path (str): The path prefix to load model. The format is ``dirname/file_prefix`` or ``file_prefix`` .
        **configs (dict, optional): Other load configuration options for compatibility. We do not
            recommend using these configurations, they may be removed in the future. If not necessary,
            DO NOT use them. Default None.
            The following options are currently supported:
            (1) model_filename (str): The inference model file name of the paddle 1.x
            ``save_inference_model`` save format. Default file name is :code:`__model__` .
            (2) params_filename (str): The persistable variables file name of the paddle 1.x
            ``save_inference_model`` save format. No default file name, save variables separately
            by default.


    Returns:
        TranslatedLayer: A Layer object can run saved translated model.

    Examples:
        1. Load model saved by ``paddle.jit.save`` then performing inference and fine-tune training.

        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.nn as nn
            import paddle.optimizer as opt

            BATCH_SIZE = 16
            BATCH_NUM = 4
            EPOCH_NUM = 4

            IMAGE_SIZE = 784
            CLASS_NUM = 10

            # define a random dataset
            class RandomDataset(paddle.io.Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples

                def __getitem__(self, idx):
                    image = np.random.random([IMAGE_SIZE]).astype('float32')
                    label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
                    return image, label

                def __len__(self):
                    return self.num_samples

            class LinearNet(nn.Layer):
                def __init__(self):
                    super(LinearNet, self).__init__()
                    self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

                @paddle.jit.to_static
                def forward(self, x):
                    return self._linear(x)

            def train(layer, loader, loss_fn, opt):
                for epoch_id in range(EPOCH_NUM):
                    for batch_id, (image, label) in enumerate(loader()):
                        out = layer(image)
                        loss = loss_fn(out, label)
                        loss.backward()
                        opt.step()
                        opt.clear_grad()
                        print("Epoch {} batch {}: loss = {}".format(
                            epoch_id, batch_id, np.mean(loss.numpy())))

            # 1. train & save model.

            # create network
            layer = LinearNet()
            loss_fn = nn.CrossEntropyLoss()
            adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

            # create data loader
            dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
            loader = paddle.io.DataLoader(dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                num_workers=2)

            # train
            train(layer, loader, loss_fn, adam)

            # save
            path = "example_model/linear"
            paddle.jit.save(layer, path)

            # 2. load model

            # load
            loaded_layer = paddle.jit.load(path)

            # inference
            loaded_layer.eval()
            x = paddle.randn([1, IMAGE_SIZE], 'float32')
            pred = loaded_layer(x)

            # fine-tune
            loaded_layer.train()
            adam = opt.Adam(learning_rate=0.001, parameters=loaded_layer.parameters())
            train(loaded_layer, loader, loss_fn, adam)


        2. Load model saved by ``paddle.fluid.io.save_inference_model`` then performing and fine-tune training.

        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.static as static
            import paddle.nn as nn
            import paddle.optimizer as opt
            import paddle.nn.functional as F

            BATCH_SIZE = 16
            BATCH_NUM = 4
            EPOCH_NUM = 4

            IMAGE_SIZE = 784
            CLASS_NUM = 10

            # define a random dataset
            class RandomDataset(paddle.io.Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples

                def __getitem__(self, idx):
                    image = np.random.random([IMAGE_SIZE]).astype('float32')
                    label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
                    return image, label

                def __len__(self):
                    return self.num_samples

            paddle.enable_static()

            image = static.data(name='image', shape=[None, 784], dtype='float32')
            label = static.data(name='label', shape=[None, 1], dtype='int64')
            pred = static.nn.fc(x=image, size=10, activation='softmax')
            loss = F.cross_entropy(input=pred, label=label)
            avg_loss = paddle.mean(loss)

            optimizer = paddle.optimizer.SGD(learning_rate=0.001)
            optimizer.minimize(avg_loss)

            place = paddle.CPUPlace()
            exe = static.Executor(place)
            exe.run(static.default_startup_program())

            # create data loader
            dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
            loader = paddle.io.DataLoader(dataset,
                feed_list=[image, label],
                places=place,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                return_list=False,
                num_workers=2)

            # 1. train and save inference model
            for data in loader():
                exe.run(
                    static.default_main_program(),
                    feed=data,
                    fetch_list=[avg_loss])

            model_path = "fc.example.model"
            paddle.fluid.io.save_inference_model(
                model_path, ["image"], [pred], exe)

            # 2. load model

            # enable dygraph mode
            paddle.disable_static(place)

            # load
            fc = paddle.jit.load(model_path)

            # inference
            fc.eval()
            x = paddle.randn([1, IMAGE_SIZE], 'float32')
            pred = fc(x)

            # fine-tune
            fc.train()
            loss_fn = nn.CrossEntropyLoss()
            adam = opt.Adam(learning_rate=0.001, parameters=fc.parameters())
            loader = paddle.io.DataLoader(dataset,
                places=place,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                num_workers=2)
            for epoch_id in range(EPOCH_NUM):
                for batch_id, (image, label) in enumerate(loader()):
                    out = fc(image)
                    loss = loss_fn(out, label)
                    loss.backward()
                    adam.step()
                    adam.clear_grad()
                    print("Epoch {} batch {}: loss = {}".format(
                        epoch_id, batch_id, np.mean(loss.numpy())))
    """
    # 1. construct correct config
    config = _parse_load_config(configs)
    model_path, config = _build_load_path_and_config(path, config)

    return TranslatedLayer._construct(model_path, config)


@dygraph_only
def _trace(layer,
           inputs,
           feed_prefix='feed_',
           fetch_prefix='fetch_',
           tmp_prefix='t_'):
    assert isinstance(layer, Layer)

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    tracer = _dygraph_tracer()._get_program_desc_tracer()

    var_list = extract_vars(inputs)

    with program_desc_tracing_guard(True):
        original_outputs = layer(*inputs)
        if not isinstance(original_outputs, (list, tuple)):
            outputs = [original_outputs]
        else:
            outputs = original_outputs
        out_vars = extract_vars(outputs, err_tag='outputs')

        program_desc, feed_names, fetch_names, parameters = tracer.create_program_desc(
            var_list, feed_prefix, out_vars, fetch_prefix, tmp_prefix)
        tracer.reset()

    with _dygraph_guard(None):
        program = create_program_from_desc(program_desc)

    return original_outputs, program, feed_names, fetch_names, parameters


class TracedLayer(object):
    """
    :api_attr: imperative

    TracedLayer is used to convert a forward dygraph model to a static
    graph model. This is mainly used to save the dygraph model for online
    inference using C++. Besides, users can also do inference in Python
    using the converted static graph model, which usually has better
    performance than the original dygraph model.

    TracedLayer would run the static graph model using :code:`Executor`
    and :code:`CompiledProgram` . The static graph model would share
    parameters with the dygraph model.

    All TracedLayer objects should not be created by constructor and should
    be created by static method :code:`TracedLayer.trace(layer, inputs)` .

    The TracedLayer can only be used to convert the data-independent dygraph
    model into the static graph model, which means the dygraph model should
    be independent with the tensor data and shape.
    """

    def __init__(self, program, parameters, feed_names, fetch_names):
        self._program = program
        self._feed_names = feed_names
        self._fetch_names = fetch_names
        self._params = parameters

        self._place = _current_expected_place()

        self._scope = core.Scope()
        for p in parameters:
            src_tensor = p.value().get_tensor()
            dst_tensor = self._scope.var(p.name).get_tensor()
            dst_tensor._share_data_with(src_tensor)

        self._exe = Executor(self._place)
        self._compiled_program = None
        self._build_strategy = None
        self._exec_strategy = None

    @property
    def program(self):
        return self._program

    def _switch(self, is_test=True):
        for block_id in range(self._program.num_blocks):
            block = self._program.block(block_id)
            for op in block.ops:
                if op.has_attr("is_test"):
                    op._set_attr("is_test", is_test)

    @staticmethod
    @dygraph_only
    def trace(layer, inputs):
        """
        This method is the only allowed method to create TracedLayer object.
        It would call the :code:`layer(*inputs)` method to run the dygraph
        model and convert it into a static graph model.

        Args:
            layer (paddle.nn.Layer): the layer object to be traced.
            inputs (list(Tensor)|tuple(Tensor)|Tensor): the input tensors of
                the layer object.

        Returns:
            tuple: A tuple of 2 items, whose the first item is the output of
                :code:`layer(*inputs)` , and the second item is the created
                TracedLayer object.

        Examples:
            .. code-block:: python:

                import paddle

                class ExampleLayer(paddle.nn.Layer):
                    def __init__(self):
                        super(ExampleLayer, self).__init__()
                        self._fc = paddle.nn.Linear(3, 10)

                    def forward(self, input):
                        return self._fc(input)


                layer = ExampleLayer()
                in_var = paddle.uniform(shape=[2, 3], dtype='float32')
                out_dygraph, static_layer = paddle.jit.TracedLayer.trace(layer, inputs=[in_var])

                # run the static graph model using Executor inside
                out_static_graph = static_layer([in_var])

                print(len(out_static_graph)) # 1
                print(out_static_graph[0].shape) # (2, 10)

                # save the static graph model for inference
                static_layer.save_inference_model(dirname='./saved_infer_model')

        """
        assert isinstance(
            layer, Layer
        ), "The type of 'layer' in fluid.dygraph.jit.TracedLayer.trace must be fluid.dygraph.Layer, but received {}.".format(
            type(layer))
        outs, prog, feed, fetch, parameters = _trace(layer, inputs)
        traced = TracedLayer(prog, parameters, feed, fetch)
        return outs, traced

    def set_strategy(self, build_strategy=None, exec_strategy=None):
        """
        Set the strategies when running static graph model.

        Args:
            build_strategy (BuildStrategy, optional): build strategy of
                :code:`CompiledProgram` inside TracedLayer. Default None.
            exec_strategy (ExecutionStrategy, optional): execution strategy of
                :code:`CompiledProgram` inside TracedLayer. Default None.

        Returns:
            None

        Examples:
            .. code-block:: python:

                import paddle

                class ExampleLayer(paddle.nn.Layer):
                    def __init__(self):
                        super(ExampleLayer, self).__init__()
                        self._fc = paddle.nn.Linear(3, 10)

                    def forward(self, input):
                        return self._fc(input)

                layer = ExampleLayer()
                in_var = paddle.uniform(shape=[2, 3], dtype='float32')

                out_dygraph, static_layer = paddle.jit.TracedLayer.trace(layer, inputs=[in_var])

                build_strategy = paddle.static.BuildStrategy()
                build_strategy.enable_inplace = True

                exec_strategy = paddle.static.ExecutionStrategy()
                exec_strategy.num_threads = 2

                static_layer.set_strategy(build_strategy=build_strategy, exec_strategy=exec_strategy)
                out_static_graph = static_layer([in_var])

        """
        assert self._compiled_program is None, "Cannot set strategy after run"
        assert isinstance(
            build_strategy, (type(None), BuildStrategy)
        ), "The type of 'build_strategy' in fluid.dygraph.jit.TracedLayer.set_strategy must be fluid.BuildStrategy, but received {}.".format(
            type(build_strategy))
        assert isinstance(
            exec_strategy, (type(None), ExecutionStrategy)
        ), "The type of 'exec_strategy' in fluid.dygraph.jit.TracedLayer.set_strategy must be fluid.ExecutionStrategy, but received {}.".format(
            type(exec_strategy))
        self._build_strategy = build_strategy
        self._exec_strategy = exec_strategy

    @switch_to_static_graph
    def _compile(self):
        self._compiled_program = CompiledProgram(
            self._program).with_data_parallel(
                build_strategy=self._build_strategy,
                exec_strategy=self._exec_strategy,
                places=self._place)

    def _build_feed(self, inputs):
        assert isinstance(inputs, (list, tuple)), \
            "Inputs should be a list or tuple of variables"
        assert len(inputs) == len(self._feed_names)
        feed_dict = {}
        if in_dygraph_mode():
            for x, name in zip(inputs, self._feed_names):
                feed_dict[name] = x.value().get_tensor()
        else:
            for x, name in zip(inputs, self._feed_names):
                feed_dict[name] = x

        return feed_dict

    @switch_to_static_graph
    def _run(self, feed):
        return self._exe.run(self._compiled_program,
                             feed=feed,
                             fetch_list=self._fetch_names)

    def __call__(self, inputs):
        with scope_guard(self._scope):
            if self._compiled_program is None:
                self._compile()

            return self._run(self._build_feed(inputs))

    @switch_to_static_graph
    def save_inference_model(self, path, feed=None, fetch=None, **kwargs):
        """
        Save the TracedLayer to a model for inference. The saved
        inference model can be loaded by C++ inference APIs.

        ``path`` is the prefix of saved objects, and the saved translated program file
        suffix is ``.pdmodel`` , the saved persistable variables file suffix is ``.pdiparams`` .

        Args:
            path(str): The path prefix to save model. The format is ``dirname/file_prefix`` or ``file_prefix``.
            feed (list[int], optional): the input variable indices of the saved
                inference model. If None, all input variables of the
                TracedLayer object would be the inputs of the saved inference
                model. Default None.
            fetch (list[int], optional): the output variable indices of the
                saved inference model. If None, all output variables of the
                TracedLayer object would be the outputs of the saved inference
                model. Default None.
            kwargs: Supported keys including 'clip_extra'.set to True if you want to clip extra information for every operator.

        Returns:
            None

        Examples:
            .. code-block:: python:

                import numpy as np
                import paddle

                class ExampleLayer(paddle.nn.Layer):
                    def __init__(self):
                        super(ExampleLayer, self).__init__()
                        self._fc = paddle.nn.Linear(3, 10)

                    def forward(self, input):
                        return self._fc(input)

                save_dirname = './saved_infer_model'
                in_np = np.random.random([2, 3]).astype('float32')
                in_var = paddle.to_tensor(in_np)
                layer = ExampleLayer()

                out_dygraph, static_layer = paddle.jit.TracedLayer.trace(layer, inputs=[in_var])
                static_layer.save_inference_model(save_dirname, feed=[0], fetch=[0])

                paddle.enable_static()
                place = paddle.CPUPlace()
                exe = paddle.static.Executor(place)
                program, feed_vars, fetch_vars = paddle.static.load_inference_model(save_dirname,
                                                    exe)

                fetch, = exe.run(program, feed={feed_vars[0]: in_np}, fetch_list=fetch_vars)
                print(fetch.shape) # (2, 10)
        """
        check_type(path, "path", str,
                   "fluid.dygraph.jit.TracedLayer.save_inference_model")
        check_type(feed, "feed", (type(None), list),
                   "fluid.dygraph.jit.TracedLayer.save_inference_model")
        if isinstance(feed, list):
            for f in feed:
                check_type(f, "each element of feed", int,
                           "fluid.dygraph.jit.TracedLayer.save_inference_model")
        check_type(fetch, "fetch", (type(None), list),
                   "fluid.dygraph.jit.TracedLayer.save_inference_model")
        if isinstance(fetch, list):
            for f in fetch:
                check_type(f, "each element of fetch", int,
                           "fluid.dygraph.jit.TracedLayer.save_inference_model")
        clip_extra = kwargs.get('clip_extra', False)
        # path check
        file_prefix = os.path.basename(path)
        if file_prefix == "":
            raise ValueError(
                "The input path MUST be format of dirname/file_prefix "
                "[dirname\\file_prefix in Windows system], but received "
                "file_prefix is empty string.")

        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

        from paddle.fluid.io import save_inference_model

        def get_feed_fetch(all_vars, partial_vars):
            if partial_vars is None:
                return all_vars

            return [all_vars[idx] for idx in partial_vars]

        with scope_guard(self._scope):
            feeded_var_names = get_feed_fetch(self._feed_names, feed)
            target_var_names = get_feed_fetch(self._fetch_names, fetch)
            target_vars = []
            for name in target_var_names:
                target_var = self._program.global_block().vars.get(name, None)
                assert target_var is not None, "{} cannot be found".format(name)
                target_vars.append(target_var)

            model_filename = file_prefix + INFER_MODEL_SUFFIX
            params_filename = file_prefix + INFER_PARAMS_SUFFIX

            save_inference_model(
                dirname=dirname,
                feeded_var_names=feeded_var_names,
                target_vars=target_vars,
                executor=self._exe,
                main_program=self._program.clone(),
                model_filename=model_filename,
                params_filename=params_filename,
                clip_extra=clip_extra)
