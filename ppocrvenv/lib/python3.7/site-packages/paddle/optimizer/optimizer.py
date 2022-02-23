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

import numpy as np
import six
import logging
from collections import defaultdict

import paddle
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table
from paddle.fluid.framework import Program, Variable, name_scope, default_main_program, default_startup_program, device_guard

from ..fluid import framework
from ..fluid import layers
from ..fluid import unique_name
from ..fluid.backward import append_backward, _some_in_set_, _append_grad_suffix_, _get_no_grad_set_name
from ..fluid.clip import GradientClipBase, GradientClipByNorm, error_clip_callback, append_gradient_clip_ops
from ..fluid.framework import program_guard, Parameter
from ..fluid.initializer import Constant
from ..fluid.layer_helper import LayerHelper
from ..fluid.layers import ops
from ..fluid.dygraph import base as imperative_base
from ..fluid.dygraph import no_grad
from paddle.fluid import core
from paddle.fluid.layers import tensor
from functools import reduce
from ..fluid.wrapped_decorator import signature_safe_contextmanager
from .. import compat as cpt
from .lr import LRScheduler
import copy
from paddle import _C_ops

__all__ = []


class Optimizer(object):
    r"""Optimizer Base class.

    Define the common interface of an optimizer.
    User should not use this class directly,
    but need to use one of it's implementation.

    Args:
        learning_rate (float|LRScheduler): The learning rate used to update ``Parameter``.
            It can be a float value or any subclass of ``LRScheduler`` .
        parameters (list|tuple, optional): List/Tuple of ``Tensor`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. And you can specify different options for \
            different parameter groups such as the learning rate, weight decay, etc, \
            then the parameters are list of dict. Note that the learning_rate in paramter groups \
            represents the scale of base learning_rate. \
            The default value is None in static mode, at this time all parameters will be updated.
        weight_decay (float|WeightDecayRegularizer, optional): The strategy of regularization. \
            It canbe a float value as coeff of L2 regularization or \
            :ref:`api_fluid_regularizer_L1Decay`, :ref:`api_fluid_regularizer_L2Decay`.
            If a parameter has set regularizer using :ref:`api_fluid_ParamAttr` already, \
            the regularization setting here in optimizer will be ignored for this parameter. \
            Otherwise, the regularization setting here in optimizer will take effect. \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of \
            some derived class of ``GradientClipBase`` . There are three cliping strategies \
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , \
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    Returns:
       Base class for optimizer. 
    
    Examples:
        .. code-block:: python

            #Take the subclass adam as an example
            import paddle
            linear = paddle.nn.Linear(10, 10)
            inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
            out = linear(inp)
            loss = paddle.mean(out)
            adam = paddle.optimizer.Adam(learning_rate=0.1,
                    parameters=linear.parameters())
            out.backward()
            adam.step()
            adam.clear_grad()

            #Take the subclass sgd as an example
            #optimize parameters in linear_1 and linear2 in different options. 
            #Note that the learning_rate of linear_2 is 0.01.
            linear_1 = paddle.nn.Linear(10, 10)
            linear_2 = paddle.nn.Linear(10, 10)
            inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
            out = linear_1(inp)
            out = linear_2(out)
            loss = paddle.mean(out)
            sgd = paddle.optimizer.SGD(
                learning_rate=0.1,
                parameters=[{
                    'params': linear_1.parameters()
                }, {
                    'params': linear_2.parameters(),
                    'weight_decay': 0.001,
                    'learning_rate': 0.1
                }],
                weight_decay=0.01)                   
            out.backward()
            sgd.step()
            sgd.clear_grad()

    """

    @imperative_base.no_grad
    def __init__(self,
                 learning_rate,
                 parameters=None,
                 weight_decay=None,
                 grad_clip=None,
                 name=None):

        if parameters is not None:
            # paddle.Tensor is also iterable, so here we don't check whether
            # the input is iterable, if the input is paddle.Tensor, the
            # list(paddle.Tensor) will be a error value
            if isinstance(parameters, paddle.Tensor):
                raise TypeError(
                    "`parameters` argument given to the optimizer should be "
                    "an iterable of paddle Tensors, but got argument type is `{}`.".
                    format(type(parameters)))
            if isinstance(parameters, dict):
                raise TypeError(
                    "`parameters` argument should not get dict type, "
                    "if parameter groups is needed, please set `parameters`"
                    " as list of dict")
            self._parameter_list = list(parameters)
        else:
            self._parameter_list = None

        self._name = name
        if framework.in_dygraph_mode():
            if self._parameter_list is None:
                raise AttributeError(
                    "parameters argument given to the Optimizer should not be None in dygraph mode."
                )
            if weight_decay is not None:
                if not isinstance(self._parameter_list[0], dict):
                    for param in self._parameter_list:
                        if hasattr(
                                param,
                                'regularizer') and param.regularizer is not None:
                            logging.info(
                                "If regularizer of a Parameter has been set by 'paddle.ParamAttr' or 'static.WeightNormParamAttr' already. "
                                "The weight_decay[%s] in Optimizer will not take effect, and it will only be applied to other Parameters!"
                                % weight_decay.__str__())
                            break

        if not isinstance(learning_rate, (float, LRScheduler)):
            raise TypeError(
                "learning rate should be float or LRScheduler, got %s here" %
                type(learning_rate))
        if grad_clip is not None:
            if not isinstance(grad_clip, GradientClipBase):
                raise TypeError(
                    "'grad_clip' should be an instance of GradientClipBase's derived class"
                )
        if isinstance(weight_decay, float):
            from ..fluid.regularizer import L2Decay
            self.regularization = L2Decay(weight_decay)
        else:
            self.regularization = weight_decay
        self._grad_clip = grad_clip
        self._learning_rate = learning_rate

        self._dtype = None
        # Infer the dtype form parameter
        if self._parameter_list:
            if isinstance(self._parameter_list[0], dict):
                for param_group in self._parameter_list:
                    assert 'params' in param_group, \
                        'params should be set in parameters if parameter groups are optimized in different options'
                self._dtype = self._parameter_list[0]['params'][0].dtype
            else:
                self._dtype = self._parameter_list[0].dtype

        # each program should have a independent learning rate
        # program -> tensor(learning_rate)
        self._learning_rate_map = dict()
        # Dictionary of accumulators. Some optimizer subclasses need to
        # allocate and manage extra tensors associated with the parameters
        # to train. These tensors are called accumulators.
        # {accum_name : { paramter_name : accumulator_for_parameter, ...}, ...}
        self._accumulators = defaultdict(lambda: dict())
        self.helper = None
        self._opti_name_list = []
        self._accumulators_holder = {}
        self._param_device_map = dict()
        self.clear_gradients = self.clear_grad
        self._default_dict = {
            'weight_decay': self.regularization,
            'grad_clip': self._grad_clip
        }

        self._param_groups = []
        if self._parameter_list and isinstance(self._parameter_list[0], dict):
            for param_group in self._parameter_list:
                self._add_param_group(param_group.copy())
        else:
            self._param_groups = self._parameter_list

    @framework.dygraph_only
    def state_dict(self):
        '''
        Get state dict information from optimizer. It contain all the tensor used by optimizer. For Adam optimizer, contains beta1, beta2, momentum etc. If LRScheduler have been used, global_step will be include in state dict.
        If the optimizer never be called(minimize function), the state_dict is empty.

        Args: 
            None

        Returns:
            state_dict(dict) : dict contains all the Tensor used by optimizer
        
        Examples:
            .. code-block:: python

                import paddle
                emb = paddle.nn.Embedding(10, 10)

                adam = paddle.optimizer.Adam(0.001, parameters=emb.parameters())
                state_dict = adam.state_dict()

        '''
        state_dict = {}
        for k, v in self._accumulators.items():
            for para_name, var_tmp in v.items():
                state_dict[var_tmp.name] = var_tmp
        # global step if use lr decay
        if isinstance(self._learning_rate, LRScheduler):
            state_dict["LR_Scheduler"] = self._learning_rate.state_dict()
        return state_dict

    @framework.dygraph_only
    def set_state_dict(self, state_dict):
        '''
        Load optimizer state dict. For Adam optimizer, contains beta1, beta2, momentum etc. If LRScheduler have been used, global_step will be changed.

        Args: 
            state_dict(dict) : Dict contains all the Tensor needed by optimizer
        Return:
            None
        
        Examples:
            .. code-block:: python

                import paddle

                emb = paddle.nn.Embedding(10, 10)

                layer_state_dict = emb.state_dict()
                paddle.save(layer_state_dict, "emb.pdparams")

                scheduler = paddle.optimizer.lr.NoamDecay(	
                    d_model=0.01, warmup_steps=100, verbose=True)
                adam = paddle.optimizer.Adam(
                    learning_rate=scheduler,
                    parameters=emb.parameters())
                opt_state_dict = adam.state_dict()
                paddle.save(opt_state_dict, "adam.pdopt")

                opti_state_dict = paddle.load("adam.pdopt")
                adam.set_state_dict(opti_state_dict)

        '''
        if isinstance(self._learning_rate, LRScheduler):
            self._learning_rate.set_dict(state_dict["LR_Scheduler"])

        if isinstance(self._learning_rate, LRScheduler):
            self._learning_rate.set_state_dict(state_dict["LR_Scheduler"])

        # NOTE: exclude learning rate scheduler's state from 
        # _accumulators_holder.
        state_dict = state_dict.copy()
        if "LR_Scheduler" in state_dict:
            state_dict.pop("LR_Scheduler")
        self._accumulators_holder = state_dict
        for k, v in self._accumulators.items():
            for para_name, var_tmp in v.items():
                assert var_tmp.name in state_dict, \
                        "optimizer Tensor {} not found".format( var_tmp.name )
                var = var_tmp.value()
                tensor = var.get_tensor()
                model_np = np.array(tensor)

                load_para = state_dict[var_tmp.name]

                if isinstance(load_para, Variable):
                    load_para_np = load_para.numpy()
                elif isinstance(load_para, core.VarBase):
                    load_para_np = load_para.numpy()
                elif isinstance(load_para, np.ndarray):
                    load_para_np = load_para
                else:
                    raise RuntimeError("State dict type {} not supprt".format(
                        str(type(load_para))))

                assert model_np.shape == load_para_np.shape,  \
                                          "Parameter shape not match, Dygraph Parameter [ {} ] need tensor with shape {} but load tensor with shape {}".format(
                                                 model_np.name, model_np.shape, load_para_np.shape)

                assert model_np.dtype == load_para_np.dtype, \
                                          "Parameter dtype not match, Dygraph Parameter [ {} ] need tensor with dtype {}  but load tensor with dtype {}".format(
                                                model_np.name, model_np.dtype, load_para_np.dtype)

                tensor.set(load_para_np, framework._current_expected_place())

    def get_opti_var_name_list(self):
        return self._opti_name_list

    def _create_global_learning_rate(self):
        if isinstance(self._learning_rate, LRScheduler):
            lr_var = self._global_learning_rate()
            # only create global lr_var once
            if not isinstance(lr_var, framework.Variable):
                lr_name = unique_name.generate('learning_rate')
                self._learning_rate._var_name = lr_name
                lr_var = self.helper.create_global_variable(
                    name=lr_name,
                    shape=[1],
                    persistable=True,
                    stop_gradient=True,
                    dtype=paddle.get_default_dtype()
                    if self._dtype is None else self._dtype)
                main_prog = framework.default_main_program()
                main_prog.lr_sheduler = self._learning_rate
                main_prog.lr_var = lr_var

                self._learning_rate_map[framework.default_main_program(
                )] = lr_var

            lr_value = float(self._learning_rate())
            self.helper.set_variable_initializer(
                lr_var, initializer=Constant(value=lr_value))
        elif isinstance(self._learning_rate, float):
            # only create global lr_var once
            lr = self._global_learning_rate()
            if isinstance(lr, framework.Variable):
                return
            else:
                self._learning_rate_map[framework.default_main_program(
                )] = layers.create_global_var(
                    name=unique_name.generate("learning_rate"),
                    shape=[1],
                    value=float(self._learning_rate),
                    dtype=paddle.get_default_dtype()
                    if self._dtype is None else self._dtype,
                    persistable=True)

    @framework.dygraph_only
    def set_lr(self, value):
        """
        :api_attr: imperative
        
        Set the value of the learning rate manually in the optimizer. If the optimizer use LRScheduler,
        this API cannot be invoked, because it will lead to conflict.

        Args:
            value (float): the value of learning rate

        Returns:
            None
          
        Examples:
            .. code-block:: python

                import paddle
                linear = paddle.nn.Linear(10, 10)

                adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())

                # set learning rate manually by python float value
                lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
                for i in range(5):
                    adam.set_lr(lr_list[i])
                    lr = adam.get_lr()
                    print("current lr is {}".format(lr))
                # Print:
                #    current lr is 0.2
                #    current lr is 0.3
                #    current lr is 0.4
                #    current lr is 0.5
                #    current lr is 0.6

        """
        if not isinstance(value, (int, float)):
            raise TypeError(
                "The type of 'value' in optimizer.set_lr must be float, but received %s."
                % (type(value)))
        if isinstance(self._learning_rate, LRScheduler):
            raise RuntimeError(
                "optimizer's learning rate can't be LRScheduler when invoke this API, because this will lead to conflict."
            )
        self._learning_rate = float(value)
        current_lr = self._global_learning_rate()
        if current_lr is not None:
            global_block = framework.default_main_program().global_block()
            global_block.append_op(
                type='fill_constant',
                outputs={'Out': [current_lr]},
                attrs={
                    'dtype': current_lr.dtype,
                    'shape': list(current_lr.shape),
                    'value': float(value)
                },
                stop_gradient=True)

    def get_lr(self):
        """
        Get current learning rate of optimizer. 
        If 'LRScheduler' is not used, the return value is all the same.
        If 'LRScheduler' is used, the return value is the current scheduled learing rete.

        Returns:
            float: The current learning rate of optimizer.

        Examples:
            .. code-block:: python

                # train on default dynamic graph mode
                import paddle
                import numpy as np
                emb = paddle.nn.Embedding(10, 3)

                ## example1: LRScheduler is not used, return the same value is all the same
                adam = paddle.optimizer.Adam(0.01, parameters = emb.parameters())
                for batch in range(10):
                    input = paddle.randint(low=0, high=5, shape=[5])
                    out = emb(input)
                    out.backward()
                    print("Learning rate of step{}: {}".format(batch, adam.get_lr())) # 0.01
                    adam.step()

                ## example2: StepDecay is used, return the scheduled learning rate
                scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
                adam = paddle.optimizer.Adam(scheduler, parameters = emb.parameters())
                for batch in range(10):
                    input = paddle.randint(low=0, high=5, shape=[5])
                    out = emb(input)
                    out.backward()
                    print("Learning rate of step{}: {}".format(batch, adam.get_lr())) # 0.5->0.05...
                    adam.step()
                    scheduler.step()

                # train on static graph mode
                paddle.enable_static()
                main_prog = paddle.static.Program()
                start_prog = paddle.static.Program()
                with paddle.static.program_guard(main_prog, start_prog):
                    x = paddle.static.data(name='x', shape=[None, 10])
                    z = paddle.static.nn.fc(x, 100)
                    loss = paddle.mean(z)
                    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
                    adam = paddle.optimizer.Adam(learning_rate=scheduler)
                    adam.minimize(loss)

                exe = paddle.static.Executor()
                exe.run(start_prog)
                for batch in range(10):
                    print("Learning rate of step{}: {}", adam.get_lr())     # 0.5->0.05->0.005...
                    out = exe.run(main_prog, feed={'x': np.random.randn(3, 10).astype('float32')})
                    scheduler.step()

        """
        if isinstance(self._learning_rate, float):
            return self._learning_rate
        else:
            return self._learning_rate()

    def _global_learning_rate(self, program=None):
        """
        get global decayed learning rate
        :return:
        """
        if program is None:
            program = framework.default_main_program()
        return self._learning_rate_map.get(program, None)

    def _append_optimize_op(self, block, param_and_grad):
        """ append optimize operator to block and return all the added optimize_op
        """
        raise NotImplementedError(
            "Class \"Optimizer\" connot be used directly as an optimizer, please use its subclasses such as \"Adam\""
        )

    def _create_param_lr(self, param_and_grad):
        # create learning rate tensor for every parameter
        param = param_and_grad[0]
        if hasattr(param, 'optimize_attr'):
            param_lr = param.optimize_attr['learning_rate']
            if type(param_lr) == Variable:
                return param_lr
            else:
                if param_lr == 1.0:
                    return self._global_learning_rate()
                else:
                    with default_main_program()._lr_schedule_guard(
                            is_with_opt=True), framework.name_scope(
                                'scale_with_param_lr'):
                        return self._global_learning_rate() * param_lr
        else:
            return self._global_learning_rate()

    def _create_accumulators(self, block, parameters):
        """Create all accumulators needed by the parameters

        Args:
            block: the block in which the loss tensor is present
            parameters: list of parameter tensors for the optimizer
        """
        pass

    def _finish_update(self, block, parameters_and_grads):
        """Finish any custom updates needed
           before completing an optimization step

        Args:
            block: the block in which the loss tensor is present
            parameters: list of parameter tensors for the optimizer

        Returns:
            None
        """
        pass

    def _add_accumulator(self,
                         name,
                         param,
                         dtype=None,
                         fill_value=0.0,
                         shape=None,
                         type=None,
                         device=None):
        """Utility function to add an accumulator for a parameter

        Args:
            block: the block in which the loss tensor is present
            name: name of the accumulator
            param: parameter tensor for which accumulator is to be added
            dtype: data type of the accumulator tensor
            fill_value: value to initialize the accumulator tensor
        """
        if self._name is not None:
            name = self._name + "_" + name
        if (name in self._accumulators and
                param.name in self._accumulators[name]):
            if framework.in_dygraph_mode():
                return self._accumulators[name][param.name]
            raise Exception("Accumulator {} already exists for parameter {}".
                            format(name, param.name))
        if shape == None:
            shape = param.shape
        assert isinstance(self.helper, LayerHelper)

        var_name = param.name + "_" + name
        var_name = unique_name.generate(var_name)
        self._opti_name_list.append(var_name)

        var = self.helper.create_global_variable(
            name=var_name,
            persistable=True,
            dtype=dtype or param.dtype,
            type=param.type if type is None else type,
            shape=shape,
            belong_to_optimizer=True)
        if device is None:
            device = self._get_device_for_param(param.name)
        with device_guard(device):
            self.helper.set_variable_initializer(
                var, initializer=Constant(value=float(fill_value)))

        if framework.in_dygraph_mode():
            if len(self._accumulators_holder) > 0:
                assert var_name in self._accumulators_holder, \
                        "Optimizer set error, {} should in state dict".format( var_name )
                var.set_value(self._accumulators_holder[var_name])

        self._accumulators[name][param.name] = var
        return var

    def _get_accumulator(self, name, param):
        """Utility function to fetch an accumulator for a parameter

        Args:
            name: name of the accumulator
            param: parameter tensor for which accumulator is to be fetched

        Returns:
            accumulator tensor for the parameter
        """
        if self._name is not None:
            name = self._name + "_" + name
        if (name not in self._accumulators or
                param.name not in self._accumulators[name]):
            raise Exception("Accumulator {} does not exist for parameter {}".
                            format(name, param.name))
        return self._accumulators[name][param.name]

    def _update_param_device_map(self, parameters_and_grads, target_block):
        for param_and_grad in parameters_and_grads:
            if param_and_grad[0].stop_gradient is False:
                param_name = param_and_grad[0].name
                ops = target_block.ops
                device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName(
                )
                for op in ops:
                    input_arg_names = op.input_arg_names
                    if param_name in input_arg_names:
                        self._param_device_map[param_name] = op.attr(
                            device_attr_name)
                        break

    def _get_device_for_param(self, param_name):
        device = None
        if param_name in self._param_device_map:
            device = self._param_device_map[param_name]
        return device

    def _create_optimization_pass(self, parameters_and_grads):
        """Add optimization operators to update gradients to tensors.

        Args:
          parameters_and_grads(list(tuple(Tensor, Tensor))):
            a list of (tensor, gradient) pair to update.

        Returns:
          return_op_list: a list of operators that will complete one step of
            optimization. This will include parameter update ops, global step
            update ops and any other custom ops required by subclasses to manage
            their internal state.
        """
        # This is a default implementation of create_optimization_pass that
        # can be shared by most optimizers. This implementation assumes that
        # the subclass will implement the _append_optimize_op method and the
        #  _initialize_tensors method. The subclass can extend the
        # _create_accumulators method if it needs to create accumulators
        # for parameters and extend _finish_update method to add custom ops.

        # Allways called under program_guard use global block as loss block
        # But if current block is in control flow, append optimize op in the
        # grad block of current block

        global_block = framework.default_main_program().global_block()
        target_block = global_block
        current_block = framework.default_main_program().current_block()
        if current_block.idx != global_block.idx:
            assert current_block.backward_block_idx != -1, \
                "current block is not global_block, but it doesn't have backward block."
            target_block = framework.default_main_program().blocks[
                current_block.backward_block_idx]

        start = len(target_block.ops)
        self.helper = LayerHelper(self.__class__.__name__)
        params_grads_device_map = parameters_and_grads['params'] if isinstance(
            parameters_and_grads, dict) else parameters_and_grads
        self._update_param_device_map(params_grads_device_map, target_block)
        if isinstance(parameters_and_grads, list):
            self._create_accumulators(
                target_block,
                [p[0] for p in parameters_and_grads if not p[0].stop_gradient])

        else:
            params_acc_dict = parameters_and_grads.copy()
            params_acc_dict['params'] = [
                p[0] for p in params_acc_dict['params']
                if not p[0].stop_gradient
            ]
            self._create_accumulators(target_block, params_acc_dict)

        self._create_global_learning_rate()

        if framework.in_dygraph_mode():

            if isinstance(parameters_and_grads, list):
                for param_and_grad in parameters_and_grads:
                    if param_and_grad[1] is None:
                        continue
                    if param_and_grad[0].stop_gradient is False:
                        self._append_optimize_op(target_block, param_and_grad)
            else:
                for param_and_grad in parameters_and_grads['params']:
                    if param_and_grad[1] is None:
                        continue
                    if param_and_grad[0].stop_gradient is False:
                        param_grad_dict = dict()
                        param_grad_dict['params'] = param_and_grad
                        param_grad_dict.update({
                            k: v
                            for k, v in parameters_and_grads.items()
                            if k != 'params'
                        })
                        self._append_optimize_op(target_block, param_grad_dict)
        else:
            for param_and_grad in parameters_and_grads:
                if param_and_grad[1] is None:
                    continue
                with param_and_grad[0].block.program._optimized_guard(
                        param_and_grad), name_scope("optimizer"):
                    if param_and_grad[0].stop_gradient is False:
                        device = self._get_device_for_param(param_and_grad[0]
                                                            .name)
                        with device_guard(device):
                            optimize_op = self._append_optimize_op(
                                target_block, param_and_grad)

        # Get custom finish ops for subclasses
        # FIXME: Need to fix this once we figure out how to handle dependencies
        self._finish_update(target_block, parameters_and_grads)

        end = len(target_block.ops)
        return target_block._slice_ops(start, end)

    def _append_dgc_ops(self, param_and_grad):
        pass

    def backward(self,
                 loss,
                 startup_program=None,
                 parameters=None,
                 no_grad_set=None,
                 callbacks=None):
        """
        The first part of ``minimize``, do auto-diff to append backward operations for
        the current program.

        Args:
            loss (Tensor): ``loss`` tensor to run optimizations.
            startup_program (Program, optional): :ref:`api_fluid_Program` for
                initializing parameters in ``parameters``. The default value
                is None, at this time :ref:`api_fluid_default_startup_program` will be used.
            parameters (list, optional): List of ``Tensor`` or ``Tensor.name`` to update
                to minimize ``loss``. The default value is None, at this time all parameters
                will be updated.
            no_grad_set (set, optional): Set of ``Tensor``  or ``Tensor.name`` that don't need
                to be updated. The default value is None.
            callbacks (list, optional): list of callable objects to run when appending backward
                operator for one parameter. The default value is None.

        Return:
            list: list of (param, grad) tensor pairs, param is ``Parameter``,
                grad is the gradient value corresponding to the parameter.

        Examples:
            .. code-block:: python

                import paddle
                import numpy as np
                value = np.arange(26).reshape(2, 13).astype("float32")
                a = paddle.to_tensor(value)
                linear = paddle.nn.Linear(13, 5)
                # This can be any optimizer supported by dygraph.
                adam = paddle.optimizer.Adam(learning_rate = 0.01, 
                                            parameters = linear.parameters())
                out = linear(a)
                out.backward()
                adam.step()
                adam.clear_grad()
        """
        act_no_grad_set = None
        if framework.in_dygraph_mode():
            pass
        else:
            act_no_grad_set = self._get_no_grad_set(loss, no_grad_set)

        # Infer dtype by loss if None
        if self._dtype is None:
            self._dtype = loss.dtype

        if framework.in_dygraph_mode():
            parameter_list = parameters if parameters \
                else self._parameter_list

            params_grads = []
            for param in parameter_list:
                if param.stop_gradient:
                    continue
                if param._grad_ivar() is not None:
                    # create gradient tensor
                    grad_var = param._grad_ivar()
                    params_grads.append((param, grad_var))
        else:
            if callbacks is None:
                callbacks = [error_clip_callback]
            else:
                assert (isinstance(callbacks, list))
            program = loss.block.program
            assert len(loss.shape) == 1 and loss.shape[0] == 1, \
                "The loss.shape should be (1L,), but the current loss.shape is {}. " \
                "Maybe that you should call paddle.mean to process the current loss.".format(
                    loss.shape)
            parameter_list = parameters if parameters \
                else self._parameter_list
            with program_guard(program, startup_program):
                params_grads = append_backward(loss, parameter_list,
                                               act_no_grad_set, callbacks)
                # Note: since we can't use all_reduce_op now,
                #  dgc_op should be the last op of one grad.
                self._append_dgc_ops(params_grads)
        return params_grads

    def apply_gradients(self, params_grads):
        """
        Second part of `minimize`, appending optimization operators for
        given `params_grads` pairs.

        Args:
            params_grads (list): list of (param, grad) pair to do optimization.

        Returns:
            list: A list of operators appended to the current program.

        Examples:
            .. code-block:: python

                import paddle
                import numpy as np

                inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
                linear = paddle.nn.Linear(10, 10)
                inp = paddle.to_tensor(inp)
                out = linear(inp)
                loss = paddle.mean(out)
                optimizer = paddle.optimizer.Adam(learning_rate=0.1,
                        parameters=linear.parameters())
                params_grads = optimizer.backward(loss)
                optimizer.apply_gradients(params_grads)

        """

        params_grads = sorted(params_grads, key=lambda x: x[0].name)

        # 'optimizer(grad_clip)' or 'set_gradient_clip'
        if self._grad_clip is not None:
            params_grads = self._grad_clip(params_grads)
        else:

            params_grads = append_gradient_clip_ops(params_grads)

        # Add regularization if any
        params_grads = self.append_regularization_ops(params_grads,
                                                      self.regularization)

        optimize_ops = self._create_optimization_pass(params_grads)
        return optimize_ops

    def _apply_optimize(self, loss, startup_program, params_grads):
        """
        Second part of `minimize`, appending optimization operators for
        given `params_grads` pairs.
        Args:
            loss (Tensor): loss tensor to run optimizations.
            startup_program (Program): startup_program for initializing parameters
                in `parameters`.
            params_grads (list): list of (param, grad) pair to do optimization.
        Returns:
            list: A list of operators appended to the current program.
        """
        if framework.in_dygraph_mode():
            with program_guard(framework.default_main_program(),
                               framework.default_startup_program()):
                if isinstance(params_grads, list):
                    if self._grad_clip is not None:
                        params_grads = self._grad_clip(params_grads)
                    params_grads = self.append_regularization_ops(
                        params_grads, self.regularization)
                else:
                    grad_clip = params_grads['grad_clip']
                    if grad_clip is not None:
                        params_grads['params'] = grad_clip(params_grads[
                            'params'])

                    params_grads['params'] = self.append_regularization_ops(
                        params_grads['params'], self.regularization)
                optimize_ops = self._create_optimization_pass(params_grads)
        else:
            program = loss.block.program
            with program_guard(program, startup_program):
                optimize_ops = self.apply_gradients(params_grads)
        return optimize_ops

    def _create_regularization_of_grad(self, param, grad, regularization=None):
        """ Create and add backward regularization Operators
    
        Function helper of append_regularization_ops.
        """
        # If no gradient or no regularization is specified,  then we don't need to do anything
        if grad is None or ((not hasattr(param, 'regularizer') or
                             (hasattr(param, 'regularizer') and
                              param.regularizer is None)) and
                            regularization is None):
            return grad
        regularization_term = None
        if hasattr(param, 'regularizer') and param.regularizer is not None:
            # Add variable for regularization term in grad block
            regularization_term = param.regularizer(param, grad, grad.block)
        elif regularization is not None:
            regularization_term = regularization(param, grad, grad.block)

        assert regularization_term is not None

        if framework.in_dygraph_mode():
            return _C_ops.sum([grad, regularization_term])

        new_grad = grad
        if grad.type == core.VarDesc.VarType.SELECTED_ROWS:
            # FIXME(zcd): If the grad is SELECTED_ROWS, after regularization,
            # the grad's type and name will be changed. But the gradient's name
            # is used in ParallelExecutor Reduce mode, so I add a flag for
            # the new_grad here.
            new_grad = grad.block.create_var(
                name=grad.name + core.kNewGradSuffix(),
                dtype=param.dtype,
                shape=param.shape,
                lod_level=param.lod_level,
                type=core.VarDesc.VarType.LOD_TENSOR)

        inputs = {"X": [grad, regularization_term]}
        outputs = {"Out": [new_grad]}
        grad.block.append_op(type='sum', inputs=inputs, outputs=outputs)

        return new_grad

    def append_regularization_ops(self,
                                  parameters_and_grads,
                                  regularization=None):
        r"""Create and add backward regularization Operators
    
        Creates and adds backward regularization operators in the BlockDesc.
        This will add gradients of the regularizer function to the gradients
        of the parameters and return these modified gradients. This is the
        same as implementing weight decay in optimizers for regularization.
    
        Args:
            parameters_and_grads: A list of (parameters, gradients) pairs
                                  that need to be regularized.
            regularization: A global regularizer. If the parameter is not
                            set. It will be applied with regularizer.
    
        Returns:
            list[(Variable, Variable)]: list of (parameters, gradients) \
            pair with the regularized gradient
    
        Raises:
            Exception: Unknown regularization type
        """
        params_and_grads = []
        if framework.in_dygraph_mode():
            for param, grad in parameters_and_grads:
                new_grad = self._create_regularization_of_grad(param, grad,
                                                               regularization)
                params_and_grads.append((param, new_grad))
        else:
            repeate_regularizer = False
            with framework.name_scope('regularization'):
                for param, grad in parameters_and_grads:
                    if not repeate_regularizer and param.regularizer is not None and regularization is not None:
                        repeate_regularizer = True
                        logging.info(
                            "If regularizer of a Parameter has been set by 'fluid.ParamAttr' or 'fluid.WeightNormParamAttr' already. "
                            "The Regularization[%s] in Optimizer will not take effect, and it will only be applied to other Parameters!"
                            % regularization.__str__())
                    with param.block.program._optimized_guard([param, grad]):
                        new_grad = self._create_regularization_of_grad(
                            param, grad, regularization)
                        params_and_grads.append((param, new_grad))
        return params_and_grads

    def _get_no_grad_set(self, loss, no_grad_set=None):
        no_grad_set = _get_no_grad_set_name(no_grad_set)
        parameters = loss.block.program.global_block().all_parameters()
        param_no_trainable = set([
            param.name for param in parameters if param.stop_gradient is True
        ])
        # If the parameter is no trainable, it should not have a gradient.
        no_grad_set.update(param_no_trainable)

        return no_grad_set

    @framework.dygraph_only
    def clear_grad(self):
        """
        Clear the gradients of all optimized parameters for model.

        If not, new gradient will accumulat on previous gradient.
        
        Returns:
            None
        
        Examples:
            .. code-block:: python

                import numpy as np
                import paddle

                value = np.arange(26).reshape(2, 13).astype("float32")
                a = paddle.to_tensor(value)
                linear = paddle.nn.Linear(13, 5)
                # This can be any optimizer supported by dygraph.
                adam = paddle.optimizer.Adam(learning_rate = 0.01, 
                                            parameters = linear.parameters())
                out = linear(a)
                out.backward()
                adam.step()
                adam.clear_grad()

        """
        if self._parameter_list is None or not isinstance(
                self._parameter_list[0], dict):
            for p in self._parameter_list:
                if not p.stop_gradient:
                    p.clear_gradient()
        else:
            for param_group in self._param_groups:
                for p in param_group['params']:
                    if not p.stop_gradient:
                        p.clear_gradient()

    @imperative_base.no_grad
    def minimize(self,
                 loss,
                 startup_program=None,
                 parameters=None,
                 no_grad_set=None):
        """
        Add operations to minimize ``loss`` by updating ``parameters``.

        Args:
            loss (Tensor): A ``Tensor`` containing the value to minimize.
            startup_program (Program, optional): :ref:`api_fluid_Program` for
                initializing parameters in ``parameters``. The default value
                is None, at this time :ref:`api_fluid_default_startup_program` will be used.
            parameters (list, optional): List of ``Tensor`` or ``Tensor.name`` to update
                to minimize ``loss``. The default value is None, at this time all parameters
                will be updated.
            no_grad_set (set, optional): Set of ``Tensor``  or ``Tensor.name`` that don't need
                to be updated. The default value is None.

        Returns:
            tuple: tuple (optimize_ops, params_grads), A list of operators appended
            by minimize and a list of (param, grad) tensor pairs, param is
            ``Parameter``, grad is the gradient value corresponding to the parameter.
            In static graph mode, the returned tuple can be passed to ``fetch_list`` in ``Executor.run()`` to 
            indicate program pruning. If so, the program will be pruned by ``feed`` and 
            ``fetch_list`` before run, see details in ``Executor``.

        Examples:
            .. code-block:: python
 
                import paddle
                linear = paddle.nn.Linear(10, 10)
                input = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
                out = linear(input)
                loss = paddle.mean(out)

                beta1 = paddle.to_tensor([0.9], dtype="float32")
                beta2 = paddle.to_tensor([0.99], dtype="float32")

                adam = paddle.optimizer.Adam(learning_rate=0.1,
                        parameters=linear.parameters(),
                        weight_decay=0.01)
                out.backward()
                adam.minimize(loss)
                adam.clear_grad()

        """
        assert isinstance(loss, Variable), "The loss should be an Tensor."

        parameter_list = parameters if parameters \
            else self._parameter_list

        params_grads = self.backward(
            loss,
            startup_program=startup_program,
            parameters=parameter_list,
            no_grad_set=no_grad_set)

        optimize_ops = self._apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads)

        return optimize_ops, params_grads

    @imperative_base.no_grad
    @framework.dygraph_only
    def step(self):
        """
        Execute the optimizer and update parameters once.
        
        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle
                import numpy as np

                value = np.arange(26).reshape(2, 13).astype("float32")
                a = paddle.to_tensor(value)
                linear = paddle.nn.Linear(13, 5)
                # This can be any optimizer supported by dygraph.
                adam = paddle.optimizer.Adam(learning_rate = 0.01, 
                                            parameters = linear.parameters())
                out = linear(a)
                out.backward()
                adam.step()
                adam.clear_grad()
        """

        if not isinstance(self._param_groups[0], dict):
            params_grads = []
            for param in self._param_groups:
                if param.stop_gradient:
                    continue
                if param._grad_ivar() is not None:
                    grad_var = param._grad_ivar()
                    params_grads.append((param, grad_var))

            self._apply_optimize(
                loss=None, startup_program=None, params_grads=params_grads)

        else:
            # optimize parameters in groups
            for param_group in self._param_groups:
                params_grads = defaultdict(lambda: list())
                for param in param_group['params']:
                    if param.stop_gradient:
                        continue
                    if param._grad_ivar() is not None:
                        grad_var = param._grad_ivar()
                        params_grads['params'].append((param, grad_var))
                params_grads.update(
                    {k: v
                     for k, v in param_group.items() if k != 'params'})
                self._apply_optimize(
                    loss=None, startup_program=None, params_grads=params_grads)

    def _add_param_group(self, param_group):
        """
        Add a param group to parameter_list.

        Args:
            param_group (dict): The group of Tensors to be optimzed with
            different optimization options.
        """
        params = param_group['params']
        if isinstance(params, Parameter):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError(
                "optimizer parameters should be in ordered collections,"
                "but received set, please use list instead.")
        else:
            param_group['params'] = list(params)

        # Update optimization options for each groups
        for k, v in self._default_dict.items():
            param_group.setdefault(k, v)

        param_set = set()
        for group in self._param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError(
                "some parameters appear in more than one parameter group")

        for param in param_group['params']:
            weight_decay = param_group['weight_decay']
            if isinstance(weight_decay, float):
                from ..fluid.regularizer import L2Decay
                regularization = L2Decay(weight_decay)
            else:
                regularization = weight_decay
            param.regularizer = regularization
            param.optimize_attr['learning_rate'] = param_group.get(
                'learning_rate', 1.)

        self._param_groups.append(param_group)

    def _update_param_group(self, parameters):
        """
        Update the param group with new entry
        Args:
            parameters (dict): The extra group of Tensors to be optimzed with
            different optimization options. Only used in child class.
        """
        pass
