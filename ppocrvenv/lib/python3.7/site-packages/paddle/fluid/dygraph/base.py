# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from ..wrapped_decorator import signature_safe_contextmanager, wrap_decorator
import decorator
import contextlib
import functools
import inspect
import sys
import numpy as np
from paddle.fluid import core
from paddle.fluid import framework
from paddle.fluid.multiprocess_utils import CleanupFuncRegistrar
from .tracer import Tracer
import logging
from ..data_feeder import convert_dtype
import warnings
from ..framework import _get_paddle_place
import paddle

__all__ = [
    'no_grad', 'no_grad_', 'grad', 'guard', 'enable_dygraph', 'disable_dygraph',
    'enabled', 'to_variable'
]

# Flag that indicates whether running code under `@declarative`
_in_declarative_mode_ = False


def in_declarative_mode():
    """
    Return a bool value that indicates whether running code under `@declarative`

    """
    return _in_declarative_mode_


def _switch_to_static_graph_(func):
    def __impl__(*args, **kwargs):
        with framework._dygraph_guard(None):
            return func(*args, **kwargs)

    return __impl__


switch_to_static_graph = wrap_decorator(_switch_to_static_graph_)


@signature_safe_contextmanager
def _switch_declarative_mode_guard_(is_declarative=True):

    global _in_declarative_mode_
    original_val = _in_declarative_mode_
    _in_declarative_mode_ = is_declarative
    yield
    _in_declarative_mode_ = original_val


@signature_safe_contextmanager
def program_desc_tracing_guard(enable):
    tracer = framework._dygraph_tracer()
    if tracer:
        original_val = tracer._enable_program_desc_tracing
        tracer._enable_program_desc_tracing = enable
    try:
        yield
    finally:
        if tracer:
            tracer._enable_program_desc_tracing = original_val


_functional_dygraph_context_manager = None


@signature_safe_contextmanager
def param_guard(parameters):
    # Note: parameters is a reference of self._parameters or self._buffers
    if in_declarative_mode() and not framework.in_dygraph_mode() and parameters:
        origin_parameters = parameters.copy()
        for name, var_base in parameters.items():
            if isinstance(var_base, list):
                new_var = [_convert_into_variable(var) for var in var_base]
            else:
                new_var = _convert_into_variable(var_base)
            parameters[name] = new_var
        yield
        parameters.update(origin_parameters)
    else:
        yield


def _convert_into_variable(var_base):
    """
    Convert Varbase into Variable.
    """
    if isinstance(var_base, core.VarBase):
        # Check whether has been created before.
        new_var = var_base.block._find_var_recursive(var_base.name)
        if new_var is not None:
            assert isinstance(new_var, framework.Variable)
        # Convert ParamBase into Parameter with same attributes in dy2stat.
        elif isinstance(var_base, framework.ParamBase):
            new_var = var_base._to_static_var(to_parameter=True)
        else:
            # Note(Aurelius84): Convert VarBase in self._buffers into Variable with
            # same attributes and set persistable=True to allow saving this var.
            # Because users can create a VarBase in `__init__`  like a
            # `mask` Tensor or `hidden_0` in RNN layers, which is equivalent to a Parameter
            # and necessary for inferring. It will be pruned if it's not necessary for inferring.

            # But if its shape is empty while created from `create_variable()`, we consider this buffer
            # non-persistable. See case of `drop_state` in lstm api.
            is_persistable = len(var_base.shape) > 0

            new_var = var_base._to_static_var(
                to_parameter=False, persistable=is_persistable)
        return new_var
    else:
        return var_base


def enabled():
    """
    This function checks whether the program runs in dynamic graph mode or not.
    You can enter dynamic graph mode with :ref:`api_fluid_dygraph_guard` api,
    or enable and disable dynamic graph mode with :ref:`api_fluid_dygraph_enable_dygraph`
    and :ref:`api_fluid_dygraph_disable_dygraph` api .

    **Note**:
        ``fluid.dygraph.enabled`` is the alias of ``fluid.in_dygraph_mode``, and
        ``fluid.in_dygraph_mode`` is recommended to use.

    Returns:
        bool: Whether the program is running in dynamic graph mode.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            fluid.enable_dygraph()  # Now we are in dygragh mode
            print(fluid.dygraph.enabled())  # True
            fluid.disable_dygraph()
            print(fluid.dygraph.enabled())  # False
    """
    return framework.in_dygraph_mode()


def enable_dygraph(place=None):
    """

    .. note::
        Dynamic graph mode is turn ON by default since paddle 2.0.0

    This API turn OFF static graph mode. You can turn ON static graph mode by `enable_static <./disable_dygraph_en.html>`_ .

    Parameters:
        place(paddle.CPUPlace|paddle.CUDAPlace|str, optional): Place to run dynamic graph. Default: None. Which means that the running place will be 
            determined according to the way of paddle compilation. If ``place`` is string, It can be ``cpu``, and ``gpu:x``, where ``x`` is the
            index of the GPUs.

    return:
        None

    Examples:
        .. code-block:: python

            import paddle
            print(paddle.in_dynamic_mode())  # True, dynamic mode is turn ON by default since paddle 2.0.0

            paddle.enable_static()
            print(paddle.in_dynamic_mode())  # False, Now we are in static mode

            paddle.disable_static()
            print(paddle.in_dynamic_mode())  # True, Now we are in dynamic mode

    """
    global _functional_dygraph_context_manager
    if _functional_dygraph_context_manager is None:
        _functional_dygraph_context_manager = guard(
            place=_get_paddle_place(place))
        _functional_dygraph_context_manager.__enter__()

        # call disable_dygraph when Python exit
        CleanupFuncRegistrar.register(disable_dygraph)


def disable_dygraph():
    """

    .. note::
        Dynamic graph mode is turn ON by default since paddle 2.0.0

    This API turn ON static graph mode. You can turn ON static graph mode by `disable_static <./enable_dygraph_en.html>`_ .

    return:
        None

    Examples:
        .. code-block:: python

            import paddle
            print(paddle.in_dynamic_mode())  # True, dynamic mode is turn ON by default since paddle 2.0.0

            paddle.enable_static()
            print(paddle.in_dynamic_mode())  # False, Now we are in static mode

            paddle.disable_static()
            print(paddle.in_dynamic_mode())  # True, Now we are in dynamic mode

    """
    global _functional_dygraph_context_manager
    if _functional_dygraph_context_manager is not None:
        _functional_dygraph_context_manager.__exit__(*sys.exc_info())
        _functional_dygraph_context_manager = None


@signature_safe_contextmanager
def _switch_tracer_mode_guard_(is_train=True):
    tracer = framework._dygraph_tracer()
    if tracer:
        has_grad = tracer._has_grad
        tracer._has_grad = is_train
        try:
            yield
        finally:
            tracer._has_grad = has_grad
    else:
        yield


def no_grad(func=None):
    """
    :api_attr: imperative

    Create a context which disables dygraph gradient calculation.
    In this mode, the result of every computation will have `stop_gradient=True`.

    Also functions as a decorator. (Make sure to instantiate without parenthesis.)

    Examples:

     .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        # use as generator

        data = np.array([[2, 3], [4, 5]]).astype('float32')
        with fluid.dygraph.guard():
            l0 = fluid.Linear(2, 2)  # l0.weight.gradient() is None
            l1 = fluid.Linear(2, 2)
            with fluid.dygraph.no_grad():
                # l1.weight.stop_gradient is False
                tmp = l1.weight * 2  # tmp.stop_gradient is True
            x = fluid.dygraph.to_variable(data)
            y = l0(x) + tmp
            o = l1(y)
            o.backward()
            print(tmp.gradient() is None)  # True
            print(l0.weight.gradient() is None)  # False

        # use as decorator

        @fluid.dygraph.no_grad
        def test_layer():
            with fluid.dygraph.guard():
                inp = np.ones([3, 1024], dtype='float32')
                t = fluid.dygraph.base.to_variable(inp)
                linear1 = fluid.Linear(1024, 4, bias_attr=False)
                linear2 = fluid.Linear(4, 4)
                ret = linear1(t)
                dy_ret = linear2(ret)

        test_layer()

    """
    if func is None:
        return _switch_tracer_mode_guard_(is_train=False)
    else:

        @decorator.decorator
        def __impl__(func, *args, **kwargs):
            with _switch_tracer_mode_guard_(is_train=False):
                return func(*args, **kwargs)

        return __impl__(func)


class no_grad_:
    """
    :api_attr: imperative

    Create a context which disables dygraph gradient calculation.
    In this mode, the result of every computation will have `stop_gradient` set
    to `True`.

    Also functions as a decorator. (Make sure to use an instance.)

    Examples:

     .. code-block:: python

        import numpy as np
        import paddle

        # use as generator

        data = np.array([[2, 3], [4, 5]]).astype('float32')
        l0 = paddle.nn.Linear(2, 2)  # l0.weight.gradient() is None
        l1 = paddle.nn.Linear(2, 2)
        with paddle.no_grad():
            # l1.weight.stop_gradient is False
            tmp = l1.weight * 2  # tmp.stop_gradient is True
        x = paddle.to_tensor(data)
        y = l0(x) + tmp
        o = l1(y)
        o.backward()
        print(tmp.gradient() is None)  # True
        print(l0.weight.gradient() is None)  # False

        # use as decorator

        @paddle.no_grad()
        def test_layer():
            inp = np.ones([3, 1024], dtype='float32')
            t = paddle.to_tensor(inp)
            linear1 = paddle.nn.Linear(1024, 4, bias_attr=False)
            linear2 = paddle.nn.Linear(4, 4)
            ret = linear1(t)
            dy_ret = linear2(ret)

        test_layer()
    """

    def __call__(self, func):
        @decorator.decorator
        def _decorate_function(func, *args, **kwargs):
            with self:
                return func(*args, **kwargs)

        @decorator.decorator
        def _decorate_generator(func, *args, **kwargs):
            gen = func(*args, **kwargs)
            with self:
                for x in gen:
                    yield x

        if inspect.isgeneratorfunction(func):
            return _decorate_generator(func)
        else:
            return _decorate_function(func)

    def __enter__(self):
        tracer = framework._dygraph_tracer()
        if tracer:
            self.orig = tracer._has_grad
            tracer._has_grad = False

    def __exit__(self, *args):
        tracer = framework._dygraph_tracer()
        if tracer:
            tracer._has_grad = self.orig


@signature_safe_contextmanager
def guard(place=None):
    """
    :api_attr: imperative

    This context will create a dygraph context for dygraph to run, using python ``with`` statement.

    Parameters:
        place(fluid.CPUPlace| fluid.CUDAPlace|str, optional): Place to execute dygraph. 
            If None, the running place will be determined according to the way of paddle compilation.
            If ``place`` is string, It can be ``cpu``, ``gpu:x`` and ``xpu:x``, where ``x`` is the
            index of the GPUs or XPUs. Default: None

    return:
        None

    Examples:

     .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        with fluid.dygraph.guard():
            inp = np.ones([3, 1024], dtype='float32')
            t = fluid.dygraph.base.to_variable(inp)
            linear1 = fluid.Linear(1024, 4, bias_attr=False)
            linear2 = fluid.Linear(4, 4)
            ret = linear1(t)
            dy_ret = linear2(ret)

    """
    train = framework.Program()
    startup = framework.Program()
    tracer = Tracer()
    VarBase = core.VarBase

    if place is not None:
        expected_place = _get_paddle_place(place)
    else:
        expected_place = framework._current_expected_place()

    with framework.program_guard(train, startup):
        with framework.unique_name.guard():
            with framework._dygraph_guard(tracer):
                with framework._dygraph_place_guard(expected_place):
                    yield


@framework.dygraph_only
def grad(outputs,
         inputs,
         grad_outputs=None,
         retain_graph=None,
         create_graph=False,
         only_inputs=True,
         allow_unused=False,
         no_grad_vars=None):
    ''' 
    .. note::
        **This API is ONLY available in Dygraph mode.**

    This API computes the sum of gradients of `outputs` with respect to each `inputs` .

    Parameters:
        outputs (Tensor|list(Tensor)|tuple(Tensor)): the output Tensor or 
            Tensor list/tuple of the graph to compute gradients.
        inputs (Tensor|list(Tensor)|tuple(Tensor)): the input Tensor or 
            Tensor list/tuple of the graph to compute gradients. The returned
            values of this API are the gradients of `inputs` . 
        grad_outputs (Tensor|list(Tensor|None)|tuple(Tensor|None), optional): 
            initial gradient values of `outputs` . If `grad_outputs` is None, 
            the initial gradient values of `outputs` would be Tensors filled with 1; 
            if `grad_outputs` is not None, it must have the same length as `outputs` , 
            and in this case, the initial gradient value of the i-th `outputs` would
            be: (1) a Tensor filled with 1 when the i-th element of `grad_outputs` 
            is None; (2) the i-th element of `grad_outputs` when the i-th element of
            `grad_outputs` is a Tensor. Default None.
        retain_graph (bool, optional): whether to retain the forward graph which 
            is used to calculate the gradient. When it is True, the graph would 
            be retained, in which way users can calculate backward twice for the 
            same graph. When it is False, the graph would be freed. Default None,
            which means it is equal to `create_graph` . 
        create_graph (bool, optional): whether to create the gradient graphs of
            the computing process. When it is True, higher order derivatives are
            supported to compute; when it is False, the gradient graphs of the
            computing process would be discarded. Default False.
        only_inputs (bool, optional): whether to only compute the gradients of
            `inputs` . If it is False, the gradients of all remaining leaf 
            Tensors in the graph would be also computed and accumulated. 
            If it is True, only the gradients of `inputs` would be computed.
            Default True. only_inputs=False is under development, and it is
            not supported yet.    
        allow_unused (bool, optional): whether to raise error or return None if some 
            Tensors of `inputs` are unreachable in the graph. If some Tensors of 
            `inputs` are unreachable in the graph (i.e., their gradients are None),  
            error would be raised if allow_unused=False, or None would be returned as
            their gradients if allow_unused=True. Default False.
        no_grad_vars (Tensor|list(Tensor)|tuple(Tensor)|set(Tensor), optional): 
            the Tensors whose gradients are not needed to compute. Default None.

    Returns:
        tuple: a tuple of Tensors, whose length is the same as the Tensor number 
        inside `inputs`, and the i-th returned Tensor is the sum of gradients of 
        `outputs` with respect to the i-th `inputs`.

    Examples 1:
        .. code-block:: python

            import paddle

            def test_dygraph_grad(create_graph):
                x = paddle.ones(shape=[1], dtype='float32')
                x.stop_gradient = False
                y = x * x

                # Since y = x * x, dx = 2 * x
                dx = paddle.grad(
                        outputs=[y],
                        inputs=[x],
                        create_graph=create_graph,
                        retain_graph=True)[0]

                z = y + dx

                # If create_graph = False, the gradient of dx
                # would not be backpropagated. Therefore,
                # z = x * x + dx, and x.gradient() = 2 * x = 2.0

                # If create_graph = True, the gradient of dx
                # would be backpropagated. Therefore,
                # z = x * x + dx = x * x + 2 * x, and
                # x.gradient() = 2 * x + 2 = 4.0

                z.backward()
                return x.gradient()

            print(test_dygraph_grad(create_graph=False)) # [2.]
            print(test_dygraph_grad(create_graph=True)) # [4.]

    Examples 2:
        .. code-block:: python

            import paddle

            def test_dygraph_grad(grad_outputs=None):
                x = paddle.to_tensor(2.0)
                x.stop_gradient = False

                y1 = x * x
                y2 = x * 3 

                # If grad_outputs=None, dy1 = [1], dy2 = [1].
                # If grad_outputs=[g1, g2], then:
                #    - dy1 = [1] if g1 is None else g1
                #    - dy2 = [1] if g2 is None else g2

                # Since y1 = x * x, dx = 2 * x * dy1.
                # Since y2 = x * 3, dx = 3 * dy2.
                # Therefore, the final result would be:
                # dx = 2 * x * dy1 + 3 * dy2 = 4 * dy1 + 3 * dy2.

                dx = paddle.grad(
                    outputs=[y1, y2], 
                    inputs=[x],
                    grad_outputs=grad_outputs)[0]

                return dx.numpy()

            grad_value = paddle.to_tensor(4.0)
            # dy1 = [1], dy2 = [1]
            print(test_dygraph_grad(None)) # [7.]

            # dy1 = [1], dy2 = [4]
            print(test_dygraph_grad([None, grad_value])) # [16.]

            # dy1 = [4], dy2 = [1]
            print(test_dygraph_grad([grad_value, None])) # [19.]

            # dy1 = [3], dy2 = [4]
            grad_y1 = paddle.to_tensor(3.0)
            print(test_dygraph_grad([grad_y1, grad_value])) # [24.]
	'''

    def check_in_out(in_out_list, name):
        assert in_out_list is not None, "{} should not be None".format(name)

        if isinstance(in_out_list, (list, tuple)):
            assert len(in_out_list) > 0, "{} cannot be empty".format(name)
            for each_var in in_out_list:
                assert isinstance(
                    each_var,
                    core.VarBase), "Elements of {} must be Variable".format(
                        name)
            return in_out_list
        else:
            assert isinstance(
                in_out_list,
                core.VarBase), "{} must be Variable or list of Variable".format(
                    name)
            return [in_out_list]

    outputs = check_in_out(outputs, 'outputs')
    inputs = check_in_out(inputs, 'inputs')

    if grad_outputs is not None:
        if not isinstance(grad_outputs, (list, tuple)):
            grad_outputs = [grad_outputs]

        for each_var in grad_outputs:
            if each_var is not None:
                assert isinstance(
                    each_var, core.VarBase
                ), "grad_outputs must be None, a Variable or a list containing None or Variables"
    else:
        grad_outputs = []

    if len(grad_outputs) > 0:
        assert len(grad_outputs) == len(
            outputs), "The length of grad_outputs must be equal to outputs"

    if no_grad_vars is None:
        no_grad_vars = []
    elif isinstance(no_grad_vars, core.VarBase):
        no_grad_vars = [no_grad_vars]
    elif isinstance(no_grad_vars, (list, tuple, set)):
        no_grad_vars = list(no_grad_vars)
        for var in no_grad_vars:
            assert isinstance(
                var, core.VarBase), "no_grad_vars can only contains Variable"
    else:
        raise AssertionError(
            "no_grad_vars must be None, Variable or list/tuple/set of Variables")

    assert isinstance(create_graph, bool), "create_graph must be True or False"

    if retain_graph is None:
        retain_graph = create_graph

    assert isinstance(retain_graph,
                      bool), "retain_graph must be None, True or False"

    assert isinstance(allow_unused, bool), "allow_unused must be True or False"

    assert isinstance(only_inputs, bool), "only_inputs must be True or False"
    assert only_inputs, "only_inputs=False is not supported yet"

    place = core.Place()
    place.set_place(framework._current_expected_place())
    return core.dygraph_partial_grad(inputs, outputs, grad_outputs,
                                     no_grad_vars, place, create_graph,
                                     retain_graph, allow_unused, only_inputs)


@framework.dygraph_only
def to_variable(value, name=None, zero_copy=None, dtype=None):
    r"""
    :api_attr: imperative

    The API will create a ``Variable`` object from 
    tuple, list, numpy\.ndarray or Variable object.

    Parameters:
        value(tuple|list|ndarray|Variable|Tensor): Initial data. 
            Can be a list, tuple, NumPy ndarray, Variable, Tensor.
            The shape can be multi-dimensional. The data type is one of 
            numpy\.{float16, float32, float64, int16, int32, int64, 
            uint8, uint16, complex64, complex128}.
        name(str, optional): The default value is None. Normally there is no 
            need for user to set this property. For more information, please 
            refer to :ref:`api_guide_Name` . 
        zero_copy(bool, optional): Whether to share memory with the input numpy 
            array. This parameter only works with CPUPlace and will be set to 
            True when it is None. Default: None. (Note: zero_copy is discarded temporally for some reason.)
        dtype(str, optional): The desired data type of returned ``Variable`` .
            Can be 'bool' , 'float16' , 'float32' , 'float64' , 'int8' , 'int16' , 
            'int32' , 'int64' , 'uint8' . Default: None.

    Returns:
        Variable : If ``value`` is a tuple/list/numpy\.ndarray object, 
            return ``Tensor`` created from the corresponding numpy\.ndarray object, which has 
            same data type and shape with ``value``. 


    Examples:

     .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        with fluid.dygraph.guard(fluid.CPUPlace()):
            x = np.ones([2, 2], np.float32)
            y = fluid.dygraph.to_variable(x, zero_copy=False)
            x[0][0] = -1
            y[0][0].numpy()  # array([1.], dtype=float32)
            y = fluid.dygraph.to_variable(x)
            x[0][0] = 0
            y[0][0].numpy()  # array([0.], dtype=float32)
            c = np.array([2+1j, 2])
            z = fluid.dygraph.to_variable(c)
            z.numpy() # array([2.+1.j, 2.+0.j])
            z.dtype # 'complex128'

            y = fluid.dygraph.to_variable([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
            y.shape     # [3L, 2L]

            y = fluid.dygraph.to_variable(((0.1, 1.2), (2.2, 3.1), (4.9, 5.2)), dtype='int32')
            y.shape     # [3L, 2L]

    """
    support_type = (list, tuple, np.ndarray, core.VarBase, framework.Variable,
                    core.Tensor, core.LoDTensor)
    if not isinstance(value, support_type):
        raise TypeError(
            "The type of 'value' in fluid.dygraph.to_variable must be %s, but received %s."
            % (support_type, type(value)))
    if isinstance(value, (core.VarBase, framework.Variable)):
        return value
    elif isinstance(value, (core.Tensor, core.LoDTensor)):
        return core.VarBase(value)
    else:
        if isinstance(framework._current_expected_place(),
                      framework.core.CPUPlace):
            #TODO(zhiqiu): we found two problems when enable zero_copy on CPUPlace.
            # (1): eigen requires 16-bytes alignments, but the data of numpy array may not statisfy.
            # Details: https://eigen.tuxfamily.org/dox/group__TopicUnalignedArrayAssert.html
            # (2): when used in flask framework, it may result in hang.
            # Details: https://github.com/PaddlePaddle/Paddle/issues/26635
            # So, we temporally diable the zero_copy strategy.
            if zero_copy == True:
                warnings.warn(
                    "Currently, zero_copy is not supported, and it will be discarded."
                )
                zero_copy = False
        else:
            assert not zero_copy, "zero_copy mode can only be used with CPUPlace"

        if not isinstance(value, np.ndarray):
            value = np.array(value)

        if dtype is not None:
            dtype = convert_dtype(dtype)
            if value.dtype != dtype:
                value = value.astype(dtype)

        py_var = core.VarBase(
            value=value,
            place=framework._current_expected_place(),
            persistable=False,
            zero_copy=zero_copy,
            name=name if name else '')
        return py_var
