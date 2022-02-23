# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import copy
import functools
import logging
import inspect
import pdb
import re
import types

import numpy
import six

from paddle.fluid.dygraph.container import Sequential
from paddle.fluid.dygraph.dygraph_to_static.convert_operators import convert_len, convert_zip
from paddle.fluid.dygraph.dygraph_to_static.logging_utils import TranslatorLogger
from paddle.fluid.dygraph.dygraph_to_static.program_translator import StaticFunction
from paddle.fluid.dygraph.dygraph_to_static.program_translator import convert_to_static
from paddle.fluid.dygraph.dygraph_to_static.program_translator import unwrap_decorators
from paddle.fluid.dygraph.dygraph_to_static.utils import is_paddle_func
from paddle.fluid.dygraph.layers import Layer

__all__ = ["convert_call"]

# TODO(liym27): A better way to do this.
BUILTIN_LIKELY_MODULES = [
    collections, pdb, copy, inspect, re, six, numpy, logging
]
# The api(s) should be considered as plain function and convert
# them into static layer code.
PADDLE_NEED_CONVERT_APIS = [Sequential]

translator_logger = TranslatorLogger()

CONVERSION_OPTIONS = "An attribute for a function that indicates conversion flags of the function in dynamic-to-static."


class ConversionOptions(object):
    """
    A container for conversion flags of a function in dynamic-to-static.

    Attributes:
        not_convert(bool): An attribute indicates that the function won't be converted in dynamic-to-static.

    NOTE(liym27): More attributes and methods can be added in this class.
    """

    def __init__(self, not_convert=False):
        self.not_convert = not_convert


def is_builtin(func):
    if isinstance(func, types.BuiltinFunctionType):
        return True
    elif func in six.moves.builtins.__dict__.values():
        return True
    else:
        return False


def is_builtin_len(func):
    if isinstance(func, types.BuiltinFunctionType) and func.__name__ == 'len':
        return True
    return False


def is_builtin_zip(func):
    return is_builtin(func) and func.__name__ == 'zip'


def is_unsupported(func):
    """
    Checks whether the func is supported by dygraph to static graph.
    """

    for m in BUILTIN_LIKELY_MODULES:
        for v in m.__dict__.values():
            func_in_dict = func == v
            if isinstance(func_in_dict, (list, numpy.ndarray)):
                func_in_dict = numpy.array(func_in_dict).any()
            if func_in_dict:
                translator_logger.log(
                    2,
                    "Whitelist: {} is part of built-in module and does not have to be transformed.".
                    format(func))
                return True

    # NOTE: should be placed before `is_paddle_func`
    if type(func) in PADDLE_NEED_CONVERT_APIS:
        return False

    if is_paddle_func(func):
        translator_logger.log(
            2,
            "Whitelist: {} is part of Paddle module and does not have to be transformed.".
            format(func))
        return True


def convert_call(func):
    """
    Converts a function call which needs to be transformed to static function.

    Args:
        func (callable): A callable function or method to convert.

    Returns:
        Callable: A converted function.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.jit.dy2static import convert_call

            paddle.enable_static()
            def dyfunc(x):
                if paddle.mean(x) < 0:
                    x_v = x - 1
                else:
                    x_v = x + 1
                return x_v

            new_func = convert_call(dyfunc)
            x = paddle.tensor.manipulation.fill_constant(shape=[3, 3], value=0, dtype='float64')
            x_v = new_func(x)

            exe = paddle.static.Executor(paddle.CPUPlace())
            out = exe.run(fetch_list=[x_v])
            print(out[0])
            # [[1. 1. 1.]
            #  [1. 1. 1.]
            #  [1. 1. 1.]]

    """
    translator_logger.log(1,
                          "Convert callable object: convert {}.".format(func))
    func_self = None
    converted_call = None

    # Function in convert_call may be decorated by another `@to_static`,
    # in this case, unwraps it into a raw method or function.
    _, func = unwrap_decorators(func)

    options = getattr(func, CONVERSION_OPTIONS, None)
    if options is not None and options.not_convert:
        translator_logger.log(
            2,
            "{} is not converted when it is decorated by 'paddle.jit.not_to_static'.".
            format(func))
        return func

    if is_builtin_len(func):
        return convert_len

    if is_builtin_zip(func):
        return convert_zip

    if is_builtin(func) or is_unsupported(func):
        return func

    if inspect.isgeneratorfunction(func):
        # NOTE(xiongkun03): inspect.isfunction() will return True even though func is a generator function. 
        # If we don't deal generatorfunction here, we will regard it as normal function and get errors in some
        # occasion.
        number_of_stars = 30
        translator_logger.warn(
            "\n\n" + "*" * number_of_stars +
            "\nYour function:`{}` doesn't support to transform to static function because it is a generator function, it will be run as-is."
            .format(func.__name__) + "\n" + "*" * number_of_stars + "\n\n")
        return func

    if inspect.isfunction(func):
        # TODO(liym27): If func is a lambda function, special conversion is needed.
        if func.__name__ == '<lambda>':
            return func
        try:
            # Note(Aurelius84): Because `@declarative` returns a class instance instead of
            # a function. This will modify the value referring to itself in `__globals__`.

            # For example:
            #
            #      @declarative
            #      def foo(x):
            #          return x
            #
            # `foo` will be converted into a wrapper class, suppose as `StaticFunction`.
            # And `foo.__globals__['foo']` will still return this `StaticFunction` instead of
            # `foo` function. So `isinstance(fn, StaticFunction)` is added here.
            global_functions = set()
            for fn in func.__globals__.values():
                if inspect.isfunction(fn):
                    global_functions.add(fn)
                elif isinstance(fn, StaticFunction):
                    _, fn = unwrap_decorators(fn)
                    global_functions.add(fn)

            if func in global_functions:
                converted_call = convert_to_static(func)
                func_self = getattr(func, '__self__', None)
            else:
                # NOTE:
                # If func is not in __globals__, it does not need to be transformed
                # because it has been transformed before.
                translator_logger.warn(
                    "{} doesn't have to be transformed to static function because it has been transformed before, it will be run as-is."
                    .format(func))
                converted_call = func
        except AttributeError:
            # NOTE:
            # If func is not in __globals__, it does not need to be transformed
            # because it has been transformed before.
            converted_call = None
        except (IOError, OSError):
            # NOTE:
            # If func has been decorated, its source code can not be get
            # so that it can not be transformed to static function.
            converted_call = None
    elif inspect.ismethod(func):
        try:
            converted_call = convert_to_static(func)
            func_self = getattr(func, '__self__', None)
        except (IOError, OSError):
            # NOTE: func may have been decorated.
            converted_call = None

    elif hasattr(func, '__class__') and hasattr(func.__class__, '__call__'):
        if hasattr(func, 'forward') and isinstance(func, Layer):
            try:
                _, forward_func = unwrap_decorators(func.forward)
                forward_func = convert_to_static(forward_func)
                # Bound mothod will be convert into plain function after `convert_to_static`.
                # So descriptor mechanism is used to bound `self` instance on function to
                # keep it as bound method.
                setattr(func, 'forward', forward_func.__get__(func))
            except (IOError, OSError, TypeError):
                # NOTE: func.forward may have been decorated.
                func_self = None if func_self else func_self
            converted_call = func
        else:
            try:
                call_func = func.__class__.__call__
                converted_call = convert_to_static(call_func)
                func_self = func
            except (IOError, OSError, TypeError):
                # NOTE:
                # If `func` is a class which is being initialized, for example `convert_call(Foo)()`,
                # it doesn't need to be transformed
                func_self = None if func_self else func_self
    else:
        raise NotImplementedError(
            "Callable {} can not be transformed at present.".format(func))

    if converted_call is None:
        translator_logger.warn(
            "{} doesn't have to be transformed to static function, and it will be run as-is."
            .format(func))
        return func

    if func_self:
        converted_call = functools.partial(converted_call, func_self)
    return converted_call
