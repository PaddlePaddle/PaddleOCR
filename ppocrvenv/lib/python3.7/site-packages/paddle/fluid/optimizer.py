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
import os
import logging
from collections import defaultdict

import paddle
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table
from paddle.fluid.framework import Program, Variable, Parameter, name_scope, default_main_program, default_startup_program, device_guard

from . import framework
from . import layers
from . import unique_name
from .backward import append_backward, _some_in_set_, _append_grad_suffix_, _get_no_grad_set_name
from .clip import GradientClipBase, GradientClipByNorm, error_clip_callback, append_gradient_clip_ops, ClipGradByGlobalNorm
from .framework import program_guard
from .initializer import Constant
from .layer_helper import LayerHelper
from .layers import ops
from .dygraph import base as imperative_base
from .dygraph import no_grad
from .dygraph.learning_rate_scheduler import LearningRateDecay, _LearningRateEpochDecay
from paddle.fluid import core
from paddle.fluid.layers import tensor
from functools import reduce
from functools import cmp_to_key
from .wrapped_decorator import signature_safe_contextmanager
from .. import compat as cpt
import warnings
from paddle import _C_ops

__all__ = [
    'SGD', 'Momentum', 'Adagrad', 'Adam', 'Adamax', 'Dpsgd', 'DecayedAdagrad',
    'Ftrl', 'SGDOptimizer', 'MomentumOptimizer', 'AdagradOptimizer',
    'AdamOptimizer', 'AdamaxOptimizer', 'DpsgdOptimizer',
    'DecayedAdagradOptimizer', 'RMSPropOptimizer', 'FtrlOptimizer', 'Adadelta',
    'AdadeltaOptimizer', 'ModelAverage', 'LarsMomentum',
    'LarsMomentumOptimizer', 'LambOptimizer', 'ExponentialMovingAverage',
    'PipelineOptimizer', 'LookaheadOptimizer', 'RecomputeOptimizer'
]


class Optimizer(object):
    """Optimizer Base class.

    Define the common interface of an optimizer.
    User should not use this class directly,
    but need to use one of it's implementation.
    """

    @imperative_base.no_grad
    def __init__(self,
                 learning_rate,
                 parameter_list=None,
                 regularization=None,
                 grad_clip=None,
                 flatten_param_grads=False,
                 align_size=-1,
                 name=None):
        """
        Args:
            flatten_param_grads (bool, optional): Whether to flatten all the parameters and grads. 
                If true, the parameters and gradients will be coalesce to contiguous mempry, 
                and the grad_clip ops / optimizer ops will be fuse to one operator.
        """
        # Because of the loop import, so place it in the function body
        from paddle.optimizer.lr import LRScheduler
        self._parameter_list = list(
            parameter_list) if parameter_list is not None else None
        self._name = name
        if framework.in_dygraph_mode():
            if not isinstance(learning_rate,
                              (float, LearningRateDecay, LRScheduler)):
                raise TypeError(
                    "learning rate should be float or LRScheduler, got %s here"
                    % type(learning_rate))
            if self._parameter_list is None:
                raise AttributeError(
                    "parameter_list argument given to the Optimizer should not be None in dygraph mode."
                )
            if regularization is not None:
                for param in self._parameter_list:
                    if param.regularizer is not None:
                        logging.info(
                            "If regularizer of a Parameter has been set by 'fluid.ParamAttr' or 'fluid.WeightNormParamAttr' already. "
                            "The Regularization[%s] in Optimizer will not take effect, and it will only be applied to other Parameters!"
                            % regularization.__str__())
                        break
        else:
            if not isinstance(learning_rate,
                              (float, framework.Variable, LRScheduler)):
                raise TypeError(
                    "learning rate should be float or LRScheduler, got %s here"
                    % type(learning_rate))

        if grad_clip is not None:
            if not isinstance(grad_clip, GradientClipBase):
                raise TypeError(
                    "'grad_clip' should be an instance of GradientClipBase's derived class"
                )
        self.regularization = regularization
        self._grad_clip = grad_clip
        self._learning_rate = learning_rate
        self._flatten_param_grads = flatten_param_grads
        self._align_size = align_size

        self._dtype = None
        # Infer the dtype form parameter
        if self._parameter_list:
            self._dtype = self._parameter_list[0].dtype

        # each program should have a independent learning rate
        # program -> Variable(learning_rate)
        self._learning_rate_map = dict()
        if isinstance(self._learning_rate, framework.Variable):
            self._learning_rate_map[framework.default_main_program(
            )] = self._learning_rate
        # Dictionary of accumulators. Some optimizer subclasses need to
        # allocate and manage extra variables associated with the parameters
        # to train. These variables are called accumulators.
        # {accum_name : { paramter_name : accumulator_for_parameter, ...}, ...}
        self._accumulators = defaultdict(lambda: dict())
        # global_accumulator dict, {accum_name : acc_variable, ...}
        self._global_accumulators = {}
        self.helper = LayerHelper(self.__class__.__name__)
        self._opti_name_list = []
        self._accumulators_holder = {}
        self._param_device_map = dict()
        # NOTE(zhiqiu): sometimes we want to add some variables(Tenosr) to the optimizer for a specific optimization,
        # for example, we want to pass 'found_inf' to adam optimizer so it can skip update when found_inf is True.
        # And these variables should not be the parameters of Optimizer's construnctor (because not commonly used). 
        # Use _auxiliary_vars together with _set_auxiliary_var/_get_auxiliary_var to achieve that.
        self._auxiliary_vars = dict()

    @framework.dygraph_only
    def state_dict(self):
        '''
        Get state dict information from optimizer. It contain all the variable used by optimizer. For Adam optimizer, contains beta1, beta2, momentum etc. If LearningRateDecay have been used, global_step will be include in state dict.
        If the optimizer never be called(minimize function), the state_dict is empty.

        Args: None
        Return:
            state_dict(dict) : dict contains all the variable used by optimizer
        
        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                with fluid.dygraph.guard():
                    emb = fluid.dygraph.Embedding([10, 10])

                    adam = fluid.optimizer.Adam(0.001, parameter_list=emb.parameters())
                    state_dict = adam.state_dict()

        '''
        from paddle.optimizer.lr import LRScheduler
        state_dict = {}
        for k, v in self._accumulators.items():
            for para_name, var_tmp in v.items():
                state_dict[var_tmp.name] = var_tmp
        for k, v in self._global_accumulators.items():
            state_dict[v.name] = v
        # global step if use lr decay
        if isinstance(self._learning_rate, LRScheduler):
            state_dict["LR_Scheduler"] = self._learning_rate.state_dict()
            return state_dict
        if isinstance(self._learning_rate, LearningRateDecay):
            state_dict["LR_Scheduler"] = self._learning_rate.state_dict()

            if not isinstance(self._learning_rate, _LearningRateEpochDecay):
                var_tmp = None
                var_temp = framework._varbase_creator(
                    None, name='global_step', dtype='int32')

                tensor.fill_constant(
                    [1], "int32", self._learning_rate.step_num, out=var_temp)

                state_dict['global_step'] = var_temp
        return state_dict

    @framework.dygraph_only
    def set_state_dict(self, state_dict):
        '''
        Load optimizer state dict. For Adam optimizer, contains beta1, beta2, momentum etc. If LearningRateDecay have been used, global_step will be changed.

        Args: 
            state_dict(dict) : Dict contains all the Variable needed by optimizer
        Return:
            None
        
        Examples:
            .. code-block:: python

                import paddle
                import paddle.fluid as fluid

                paddle.disable_static()

                emb = paddle.nn.Embedding(10, 10)

                state_dict = emb.state_dict()
                fluid.save_dygraph(state_dict, "paddle_dy")

                scheduler = paddle.optimizer.lr.NoamDecay(	
                    d_model=0.01, warmup_steps=100, verbose=True)
                adam = paddle.optimizer.Adam(
                    learning_rate=scheduler,
                    parameters=emb.parameters())
                state_dict = adam.state_dict()
                fluid.save_dygraph(state_dict, "paddle_dy")

                para_state_dict, opti_state_dict = fluid.load_dygraph("paddle_dy")
        '''
        from paddle.optimizer.lr import LRScheduler
        if isinstance(self._learning_rate, LRScheduler):
            self._learning_rate.set_dict(state_dict["LR_Scheduler"])

        if isinstance(self._learning_rate, LearningRateDecay):
            self._learning_rate.set_dict(state_dict["LR_Scheduler"])

            if not isinstance(self._learning_rate, _LearningRateEpochDecay):
                assert 'global_step' in state_dict, \
                        'Global step not in state dict, Dygraph use LearningRateDecay, global_step must in state_dict'
                global_step = state_dict['global_step']

                if isinstance(global_step, Variable):
                    step_np = global_step
                    step_np = np.array(step_np.value().get_tensor())
                    assert step_np.shape == (1,),  \
                            "global step shape is (1,), the shape is {}".format( step_np.shape )

                    self._learning_rate.step_num = int(step_np[0])
                elif isinstance(global_step, np.ndarray):
                    assert global_step.shape == (1,),  \
                            "global step shape is (1,), the shape is {}".format( global_step.shape )
                    self._learning_rate.step_num = global_step[0]
                else:
                    raise RuntimeError(
                        "Type not supprt, value in state dict must be [VarBase, Variable, numpy], the type is ",
                        type(global_step))

        def _load_state_para(state_dict, param):
            var = param.value()
            tensor = var.get_tensor()
            model_np = np.array(tensor)
            load_para = state_dict[param.name]
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
                                                param.name, model_np.shape, load_para_np.shape)

            assert model_np.dtype == load_para_np.dtype, \
                                        "Parameter dtype not match, Dygraph Parameter [ {} ] need tensor with dtype {}  but load tensor with dtype {}".format(
                                            param.name, model_np.dtype, load_para_np.dtype)

            tensor.set(load_para_np, framework._current_expected_place())

        self._accumulators_holder = state_dict
        for k, v in self._accumulators.items():
            for para_name, var_tmp in v.items():
                assert var_tmp.name in state_dict, \
                        "optimizer variable {} not found".format( var_tmp.name )
                _load_state_para(state_dict, var_tmp)

        for k, v in self._global_accumulators.items():
            assert v.name in state_dict, \
                        "optimizer variable {} not found".format( v.name )
            _load_state_para(state_dict, v)

    # [aliases] Compatible with old method names
    set_dict = set_state_dict

    def get_opti_var_name_list(self):
        return self._opti_name_list

    def _set_auxiliary_var(self, key, val):
        self._auxiliary_vars[key] = val

    def _get_auxiliary_var(self, key):
        if key in self._auxiliary_vars:
            return self._auxiliary_vars[key]
        else:
            return None

    def _create_global_learning_rate(self):
        from paddle.optimizer.lr import LRScheduler
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
                    dtype='float32' if self._dtype is None else self._dtype)
                main_prog = framework.default_main_program()
                main_prog.lr_sheduler = self._learning_rate
                main_prog.lr_var = lr_var
                self._learning_rate_map[framework.default_main_program(
                )] = lr_var

            lr_value = float(self._learning_rate())
            self.helper.set_variable_initializer(
                lr_var, initializer=Constant(value=lr_value))
            return

        if imperative_base.enabled():
            # create learning rate Variable
            if isinstance(self._learning_rate, float):
                lr = self._global_learning_rate()

                if isinstance(lr, framework.Variable):
                    return
                else:
                    self._learning_rate_map[framework.default_main_program(
                    )] = layers.create_global_var(
                        name=unique_name.generate("learning_rate"),
                        shape=[1],
                        value=float(self._learning_rate),
                        dtype='float32' if self._dtype is None else self._dtype,
                        persistable=True)
            # get learning rate Variable from LearningRateDecay
            elif isinstance(self._learning_rate, LearningRateDecay):
                self._learning_rate_map[framework.default_main_program(
                )] = self._learning_rate()
            else:
                raise TypeError(
                    "optimizer's learning rate must be float or LearningRateDecay"
                )
        else:
            lr = self._global_learning_rate()

            if isinstance(lr, framework.Variable):
                return
            else:
                if not isinstance(self._learning_rate, float):
                    raise TypeError(
                        "learning rate variable is create outside optimizer,"
                        "can not create new learning rate variable for new program"
                    )

            # create learning rate in the current main program
            self._learning_rate_map[framework.default_main_program(
            )] = layers.create_global_var(
                name=unique_name.generate("learning_rate"),
                shape=[1],
                value=float(self._learning_rate),
                dtype='float32' if self._dtype is None else self._dtype,
                persistable=True)

    @framework.dygraph_only
    def set_lr(self, value):
        """
        :api_attr: imperative
        
        Set the value of the learning rate manually in the optimizer. If the optimizer use LearningRateDecay,
        this API cannot be invoked, because it will lead to conflict.

        Args:
            value (float|Variable): the value of learning rate

        Returns:
            None
          
        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                        
                with fluid.dygraph.guard():
                    linear = fluid.dygraph.nn.Linear(10, 10)

                    adam = fluid.optimizer.Adam(0.1, parameter_list=linear.parameters())

                    # set learning rate manually by python float value
                    lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
                    for i in range(5):
                        adam.set_lr(lr_list[i])
                        lr = adam.current_step_lr()
                        print("current lr is {}".format(lr))
                    # Print:
                    #    current lr is 0.2
                    #    current lr is 0.3
                    #    current lr is 0.4
                    #    current lr is 0.5
                    #    current lr is 0.6


                    # set learning rate manually by framework Variable
                    lr_var = fluid.layers.create_global_var(
                        shape=[1], value=0.7, dtype='float32')
                    adam.set_lr(lr_var)
                    lr = adam.current_step_lr()
                    print("current lr is {}".format(lr))
                    # Print:
                    #    current lr is 0.7



        """
        if not isinstance(value, (framework.Variable, float)):
            raise TypeError(
                "The type of 'value' in optimizer.set_lr must be (float, Variable), but received %s."
                % (type(value)))
        if isinstance(self._learning_rate, LearningRateDecay):
            raise RuntimeError(
                "optimizer's learning rate can't be LearningRateDecay when invoke this API, because this will lead to conflict."
            )
        if isinstance(value, float):
            self._learning_rate = value
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
        else:
            assert len(value.shape) == 1 and value.shape[
                0] == 1, "optimizer's learning rate must be 1-D Tensor with shape[1]"
            self._learning_rate_map[framework.default_main_program()] = value

    @framework.dygraph_only
    def current_step_lr(self):
        """
        :api_attr: imperative
        
        Get current step learning rate. The return value is all the same When LearningRateDecay is not used,
        otherwise return the step learning rate.

        Returns:
            float: The learning rate of the current step.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                # example1: LearningRateDecay is not used, return value is all the same
                with fluid.dygraph.guard():
                    emb = fluid.dygraph.Embedding([10, 10])
                    adam = fluid.optimizer.Adam(0.001, parameter_list = emb.parameters())
                    lr = adam.current_step_lr()
                    print(lr) # 0.001

                # example2: PiecewiseDecay is used, return the step learning rate
                with fluid.dygraph.guard():
                    inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
                    linear = fluid.dygraph.nn.Linear(10, 10)
                    inp = fluid.dygraph.to_variable(inp)
                    out = linear(inp)
                    loss = fluid.layers.reduce_mean(out)
                    
                    bd = [2, 4, 6, 8]
                    value = [0.2, 0.4, 0.6, 0.8, 1.0]
                    adam = fluid.optimizer.Adam(fluid.dygraph.PiecewiseDecay(bd, value, 0),
                                           parameter_list=linear.parameters())

                    # first step: learning rate is 0.2
                    np.allclose(adam.current_step_lr(), 0.2, rtol=1e-06, atol=0.0) # True

                    # learning rate for different steps
                    ret = [0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0]
                    for i in range(12):
                        adam.minimize(loss)
                        lr = adam.current_step_lr()
                        np.allclose(lr, ret[i], rtol=1e-06, atol=0.0) # True

        """
        current_lr = self._global_learning_rate()
        if isinstance(current_lr, framework.Variable):
            return self._global_learning_rate().numpy()[0]

        if isinstance(self._learning_rate, float):
            return self._learning_rate
        elif isinstance(self._learning_rate, _LearningRateEpochDecay):
            step_lr = self._learning_rate()
            return step_lr.numpy()[0]
        else:
            step_lr = self._learning_rate.step()
            if isinstance(step_lr, (float, int)):
                return step_lr
            else:
                return step_lr.numpy()[0]

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
        raise NotImplementedError()

    def _create_param_lr(self, param_and_grad):
        # create learning rate variable for every parameter
        param = param_and_grad[0]
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

    def _create_accumulators(self, block, parameters):
        """Create all accumulators needed by the parameters

        Args:
            block: the block in which the loss variable is present
            parameters: list of parameter variables for the optimizer
        """
        pass

    def _finish_update(self, block, parameters_and_grads):
        """Finish any custom updates needed
           before completing an optimization step

        Args:
            block: the block in which the loss variable is present
            parameters: list of parameter variables for the optimizer

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
            block: the block in which the loss variable is present
            name: name of the accumulator
            param: parameter variable for which accumulator is to be added
            dtype: data type of the accumulator variable
            fill_value: value to initialize the accumulator variable
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

    def _add_global_accumulator(self,
                                name,
                                dtype=None,
                                fill_value=0.0,
                                shape=None,
                                type=None,
                                device=None):
        """Utility function to add a global accumulator for all parameters in the model

        Args:
            block: the block in which the loss variable is present
            name: name of the accumulator
            dtype: data type of the accumulator variable
            fill_value: value to initialize the accumulator variable
            shape: the shape of the accumulator
            type: the variable type of the accumulator
            device: the target place of the accumulator
        """
        if self._name is not None:
            name = self._name + "_" + name
        if (name in self._global_accumulators):
            if framework.in_dygraph_mode():
                return self._global_accumulators[name]
            raise Exception("Global accumulator {} already exists".format(name))
        if shape == None:
            shape = [1]  # most case, global accumulator is of shape [1]
        assert isinstance(self.helper, LayerHelper)

        var_name = name
        var_name = unique_name.generate(var_name)
        self._opti_name_list.append(var_name)

        var = self.helper.create_global_variable(
            name=var_name,
            persistable=True,
            dtype=dtype if dtype else self._dtype,
            type=type,
            shape=shape,
            belong_to_optimizer=True)
        if device is None:
            device = 'cpu'
        with device_guard(device):
            self.helper.set_variable_initializer(
                var, initializer=Constant(value=float(fill_value)))

        if framework.in_dygraph_mode():
            if len(self._accumulators_holder) > 0:
                assert var_name in self._accumulators_holder, \
                        "Optimizer set error, {} should in state dict".format( var_name )
                var.set_value(self._accumulators_holder[var_name])

        self._global_accumulators[name] = var
        return var

    def _get_accumulator(self, name, param):
        """Utility function to fetch an accumulator for a parameter

        Args:
            name: name of the accumulator
            param: parameter variable for which accumulator is to be fetched

        Returns:
            accumulator variable
        """
        if self._name is not None:
            name = self._name + "_" + name
        if (name not in self._accumulators or
                param.name not in self._accumulators[name]):
            raise Exception("Accumulator {} does not exist for parameter {}".
                            format(name, param.name))
        return self._accumulators[name][param.name]

    def _get_global_accumulator(self, name):
        """Utility function to fetch a global accumulator

        Args:
            name: name of the accumulator

        Returns:
            accumulator variable
        """
        if self._name is not None:
            name = self._name + "_" + name
        if (name not in self._global_accumulators):
            raise Exception("Global accumulator {} does not exist".format(name))
        return self._global_accumulators[name]

    def _update_param_device_map(self, parameters_and_grads, target_block):
        for param_and_grad in parameters_and_grads:
            if param_and_grad[0].trainable is True:
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
        """Add optimization operators to update gradients to variables.

        Args:
          parameters_and_grads(list(tuple(Variable, Variable))):
            a list of (variable, gradient) pair to update.

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

        self._update_param_device_map(parameters_and_grads, target_block)
        self._create_accumulators(
            target_block,
            [p[0] for p in parameters_and_grads if p[0].trainable])
        self._create_global_learning_rate()

        if framework.in_dygraph_mode():
            for param_and_grad in parameters_and_grads:
                if param_and_grad[1] is None:
                    continue
                if param_and_grad[0].trainable is True:
                    self._append_optimize_op(target_block, param_and_grad)
        else:
            for param_and_grad in parameters_and_grads:
                if param_and_grad[1] is None:
                    continue
                with param_and_grad[0].block.program._optimized_guard(
                        param_and_grad), name_scope("optimizer"):
                    if param_and_grad[0].trainable is True:
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

    def _process_distribute_lookuptable(self, param_grads):
        """
        Because distribute lookup table only support SGD optimizer for now, not support
        other optimizer and regularization, so we should find the table parameter out,
        and avoid to add regularization and other op for it, and add sgd optimize op
        for it independently.
        :param param_grads(list((Var, Var))): list of (param, grad) pair.
        :param loss: the loss variable.
        :param startup_program: the startup program
        """
        program = framework.default_main_program()
        global_block = framework.default_main_program().global_block()
        table_name = find_distributed_lookup_table(program)
        table_param = None
        table_grad = None
        new_param_grads = []
        for p, g in param_grads:
            if p.name == table_name:
                if table_param is not None:
                    raise RuntimeError(
                        "multi dist table var found, only support one now!")
                table_param = p
                table_grad = g
            else:
                new_param_grads.append((p, g))
        sgd_op = None
        if table_param is not None:
            param_and_grad = [table_param, table_grad]
            with table_param.block.program._optimized_guard(param_and_grad), \
                    framework.name_scope("optimizer"):
                self._create_global_learning_rate()
                # create the optimize op
                sgd_op = global_block.append_op(
                    type='sgd',
                    inputs={
                        "Param": table_param,
                        "Grad": table_grad,
                        "LearningRate": self._create_param_lr(param_and_grad)
                    },
                    outputs={"ParamOut": param_and_grad[0]})
        return new_param_grads, (table_param, table_grad), sgd_op

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        """
        The first part of ``minimize``, do auto-diff to append backward operations for
        the current program.

        Args:
            loss (Variable): ``loss`` variable to run optimizations.
            startup_program (Program, optional): :ref:`api_fluid_Program` for
                initializing parameters in ``parameter_list``. The default value
                is None, at this time :ref:`api_fluid_default_startup_program` will be used.
            parameter_list (Iterable, optional): Iterable of ``Variable`` or ``Variable.name`` to update
                to minimize ``loss``. The default value is None, at this time all parameters
                will be updated.
            no_grad_set (set, optional): Set of ``Variable``  or ``Variable.name`` that don't need
                to be updated. The default value is None.
            callbacks (list, optional): list of callable objects to run when appending backward
                operator for one parameter. The default value is None.

        Return:
            list: list of (param, grad) variable pairs, param is ``Parameter``,
                grad is the gradient value corresponding to the parameter.

        Examples:
            See examples in ``apply_gradients``.
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
            parameter_list = parameter_list if parameter_list \
                else self._parameter_list

            params_grads = []
            for param in parameter_list:
                if not param.trainable:
                    continue
                if param._grad_ivar() is not None:
                    # create gradient variable
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
                "Maybe that you should call fluid.layers.mean to process the current loss.".format(
                    loss.shape)
            parameter_list = parameter_list if parameter_list \
                else self._parameter_list
            with program_guard(program, startup_program):
                params_grads = append_backward(loss, parameter_list,
                                               act_no_grad_set, callbacks)
        return params_grads

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
                    if not repeate_regularizer and getattr(
                            param, 'regularizer',
                            None) is not None and regularization is not None:
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

    def flatten_param_grads(self, params_grads):
        need_flatten_params = []
        need_flatten_grads = []
        for p, g in params_grads:
            if g is None:
                continue
            g.persistable = True
            if getattr(p, 'need_clip', True) is False or getattr(
                    p, 'regularizer', None) is not None:
                warnings.warn(
                    "flatten_param_grads=True will be discarded since paramter '{}''s need_clip is False or "
                    "the regularizer is set".format(p.name))
                self._flatten_param_grads = False
                return params_grads

            need_flatten_params.append(p)
            need_flatten_grads.append(g)

        shape = [np.prod(p.shape) for p in need_flatten_params]
        block = need_flatten_params[0].block

        flatten_param = self.helper.create_global_variable(
            name='flatten_param',
            persistable=True,
            dtype=need_flatten_params[0].dtype,
            shape=[np.sum(shape)],
            belong_to_optimizer=True)

        flatten_param.trainable = True
        flatten_param.optimize_attr = need_flatten_params[0].optimize_attr
        flatten_param.regularizer = need_flatten_params[0].regularizer

        flatten_grad = self.helper.create_global_variable(
            name='flatten_grad',
            persistable=True,
            dtype=need_flatten_grads[0].dtype,
            shape=[np.sum(shape)],
            belong_to_optimizer=True)

        with program_guard(default_main_program()):
            block.append_op(
                type="coalesce_tensor",
                inputs={"Input": need_flatten_params},
                outputs={
                    "Output": need_flatten_params,
                    "FusedOutput": flatten_param
                },
                attrs={
                    "copy_data": True,
                    "use_align": True,
                    "align_size": self._align_size,
                    "dtype": need_flatten_params[0].dtype
                })

            block.append_op(
                type="coalesce_tensor",
                inputs={"Input": need_flatten_grads},
                outputs={
                    "Output": need_flatten_grads,
                    "FusedOutput": flatten_grad
                },
                attrs={
                    "copy_data": True,
                    "use_align": True,
                    "align_size": self._align_size,
                    "dtype": need_flatten_grads[0].dtype
                })

        #NOTE(zhiqiu): the initializer should be set after coalesce_tensor op,
        # so the shape of flatten_param and flatten_grad will be inferred.
        self.helper.set_variable_initializer(
            flatten_param, initializer=Constant(0.0))
        self.helper.set_variable_initializer(
            flatten_grad, initializer=Constant(0.0))

        return [(flatten_param, flatten_grad)]

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

                import paddle.fluid as fluid
                loss = network()
                optimizer = fluid.optimizer.SGD(learning_rate=0.1)
                params_grads = optimizer.backward(loss)
                # you may append operations for params_grads here
                # ...
                optimizer.apply_gradients(params_grads)
        """
        params_grads = sorted(params_grads, key=lambda x: x[0].name)

        # NOTE(zhiqiu): currently, only support ClipGradByGlobalNorm and without regularization.
        if self._flatten_param_grads and self.regularization is None:
            if self._grad_clip == None or isinstance(self._grad_clip,
                                                     ClipGradByGlobalNorm):
                params_grads = self.flatten_param_grads(params_grads)

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

    def apply_optimize(self, loss, startup_program, params_grads):
        """
        Second part of `minimize`, appending optimization operators for
        given `params_grads` pairs.
        Args:
            loss (Variable): loss variable to run optimizations.
            startup_program (Program): startup_program for initializing parameters
                in `parameter_list`.
            params_grads (list): list of (param, grad) pair to do optimization.
        Returns:
            list: A list of operators appended to the current program.
        """
        if framework.in_dygraph_mode():
            with program_guard(framework.default_main_program(),
                               framework.default_startup_program()):
                if self._grad_clip is not None:
                    params_grads = self._grad_clip(params_grads)
                params_grads = self.append_regularization_ops(
                    params_grads, self.regularization)
                optimize_ops = self._create_optimization_pass(params_grads)
        else:
            program = loss.block.program
            with program_guard(program, startup_program):
                optimize_ops = self.apply_gradients(params_grads)
        return optimize_ops

    def _get_no_grad_set(self, loss, no_grad_set=None):
        no_grad_set = _get_no_grad_set_name(no_grad_set)
        parameters = loss.block.program.global_block().all_parameters()
        param_no_trainable = set(
            [param.name for param in parameters if param.trainable is False])
        # If the parameter is no trainable, it should not have a gradient.
        no_grad_set.update(param_no_trainable)

        return no_grad_set

    @framework.dygraph_only
    def clear_gradients(self):
        """
        Clear the gradients of all optimized parameters for model.

        If not, new gradient will accumulat on previous gradient.
        
        Returns:
            None
        
        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                with fluid.dygraph.guard():
                    value = np.arange(26).reshape(2, 13).astype("float32")
                    a = fluid.dygraph.to_variable(value)
                    linear = fluid.Linear(13, 5, dtype="float32")
                    # This can be any optimizer supported by dygraph.
                    adam = fluid.optimizer.Adam(learning_rate = 0.01, 
                                                parameter_list = linear.parameters())
                    out = linear(a)
                    out.backward()
                    adam.minimize(out)
                    adam.clear_gradients()

        """
        for p in self._parameter_list:
            if p.trainable:
                p.clear_gradient()

    @imperative_base.no_grad
    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        """
        Add operations to minimize ``loss`` by updating ``parameter_list``.

        Args:
            loss (Variable): A ``Variable`` containing the value to minimize.
            startup_program (Program, optional): :ref:`api_fluid_Program` for
                initializing parameters in ``parameter_list``. The default value
                is None, at this time :ref:`api_fluid_default_startup_program` will be used.
            parameter_list (Iterable, optional): Iterable of ``Variable`` or ``Variable.name`` to update
                to minimize ``loss``. The default value is None, at this time all parameters
                will be updated.
            no_grad_set (set, optional): Set of ``Variable``  or ``Variable.name`` that don't need
                to be updated. The default value is None.

        Returns:
            tuple: tuple (optimize_ops, params_grads), A list of operators appended
            by minimize and a list of (param, grad) variable pairs, param is
            ``Parameter``, grad is the gradient value corresponding to the parameter.
            The returned tuple can be passed to ``fetch_list`` in ``Executor.run()`` to 
            indicate program pruning. If so, the program will be pruned by ``feed`` and 
            ``fetch_list`` before run, see details in ``Executor``.

        Examples:
            Please refer to the example of current Optimizer.
        """
        assert isinstance(loss, Variable), "The loss should be an Variable."

        parameter_list = parameter_list if parameter_list \
            else self._parameter_list

        params_grads = self.backward(
            loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set)

        optimize_ops = self.apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads)

        return optimize_ops, params_grads


class SGDOptimizer(Optimizer):
    r"""
    Optimizer of the stochastic gradient descent algorithm.

    .. math::

        param\_out = param - learning\_rate * grad

    Parameters:
        learning_rate (float|Variable): The learning rate used to update parameters. \
            Can be a float value or a Variable with one float value as data element.
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_fluid_regularizer_L1Decay` , :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of 
            some derived class of ``GradientClipBase`` . There are three cliping strategies 
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , 
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            place = fluid.CPUPlace()
            main = fluid.Program()
            with fluid.program_guard(main):
                x = fluid.layers.data(name='x', shape=[13], dtype='float32')
                y = fluid.layers.data(name='y', shape=[1], dtype='float32')
                y_predict = fluid.layers.fc(input=x, size=1, act=None)
                cost = fluid.layers.square_error_cost(input=y_predict, label=y)
                avg_cost = fluid.layers.mean(cost)

                sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
                sgd_optimizer.minimize(avg_cost)

                fetch_list = [avg_cost]
                train_reader = paddle.batch(
                    paddle.dataset.uci_housing.train(), batch_size=1)
                feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                for data in train_reader():
                    exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)

    """

    def __init__(self,
                 learning_rate,
                 parameter_list=None,
                 regularization=None,
                 grad_clip=None,
                 name=None):
        assert learning_rate is not None
        super(SGDOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            grad_clip=grad_clip,
            name=name)
        self.type = "sgd"

    @no_grad
    def _append_optimize_op(self, block, param_and_grad):
        lr = self._create_param_lr(param_and_grad)
        if framework.in_dygraph_mode():
            _C_ops.sgd(param_and_grad[0], lr, param_and_grad[1],
                       param_and_grad[0])
            return None

        assert isinstance(block, framework.Block)
        # create the optimize op
        sgd_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": lr
            },
            outputs={"ParamOut": param_and_grad[0]},
            stop_gradient=True)

        return sgd_op


class MomentumOptimizer(Optimizer):
    r"""

    Simple Momentum optimizer with velocity state

    This optimizer has a flag for Nestrov Momentum.

    The update equations are as follows:

    .. math::

        & velocity = mu * velocity + gradient

        & if (use\_nesterov):

        &\quad   param = param - (gradient + mu * velocity) * learning\_rate

        & else:

        &\quad   param = param - learning\_rate * velocity

    Parameters:
        learning_rate (float|Variable): The learning rate used to update parameters. \
            Can be a float value or a Variable with one float value as data element.
        momentum (float): Momentum factor
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        use_nesterov (bool, optional): Enables Nesterov momentum, default is false.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_fluid_regularizer_L1Decay` , :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of 
            some derived class of ``GradientClipBase`` . There are three cliping strategies 
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , 
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            place = fluid.CPUPlace()
            main = fluid.Program()
            with fluid.program_guard(main):
                x = fluid.layers.data(name='x', shape=[13], dtype='float32')
                y = fluid.layers.data(name='y', shape=[1], dtype='float32')
                y_predict = fluid.layers.fc(input=x, size=1, act=None)
                cost = fluid.layers.square_error_cost(input=y_predict, label=y)
                avg_cost = fluid.layers.mean(cost)

                moment_optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
                moment_optimizer.minimize(avg_cost)

                fetch_list = [avg_cost]
                train_reader = paddle.batch(
                    paddle.dataset.uci_housing.train(), batch_size=1)
                feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                for data in train_reader():
                    exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)

    """
    _velocity_acc_str = "velocity"

    def __init__(self,
                 learning_rate,
                 momentum,
                 parameter_list=None,
                 use_nesterov=False,
                 regularization=None,
                 grad_clip=None,
                 name=None):
        assert learning_rate is not None
        assert momentum is not None
        super(MomentumOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            grad_clip=grad_clip,
            name=name)
        self.type = "momentum"
        self._momentum = momentum
        self._use_nesterov = bool(use_nesterov)

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        for p in parameters:
            self._add_accumulator(self._velocity_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        velocity_acc = self._get_accumulator(self._velocity_acc_str,
                                             param_and_grad[0])
        lr = self._create_param_lr(param_and_grad)
        master_weight = None
        if framework.in_dygraph_mode():
            _, _, _ = _C_ops.momentum(
                param_and_grad[0], param_and_grad[1], velocity_acc, lr,
                master_weight, param_and_grad[0], velocity_acc, master_weight,
                'mu', self._momentum, 'use_nesterov', self._use_nesterov)
            return None

        attrs = {"mu": self._momentum, "use_nesterov": self._use_nesterov}
        inputs = {
            "Param": [param_and_grad[0]],
            "Grad": [param_and_grad[1]],
            "Velocity": [velocity_acc],
            "LearningRate": [lr]
        }

        outputs = {
            "ParamOut": [param_and_grad[0]],
            "VelocityOut": [velocity_acc]
        }
        # create the momentum optimize op
        momentum_op = block.append_op(
            type=self.type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=True)

        return momentum_op


class DGCMomentumOptimizer(Optimizer):
    r"""
	:api_attr: Static Graph

    DGC (Deep Gradient Compression) Momentum Optimizer. Original paper is https://arxiv.org/abs/1712.01887

    DGC reduces the communication bandwidth by sending only the important gradients (sparse update):\
        only gradients larger than a threshold are transmitted.

    To avoid losing information, DGC accumulates the rest of the gradients locally.

    Eventually, these gradients become large enough to be transmitted.

    Thus, DGC sends the large gradients immediately but eventually sends all of the gradients over time.

    To ensure no loss of accuracy, DGC employs momentum correction and local gradient clipping on top of the gradient sparsification to maintain model performance.

    DGC also uses momentum factor masking and warmup training to overcome the staleness problem caused by reduced communication.

    This optimizer will do two things:

        1. Compress the gradient by get TopK import value from tensor \
            and use it for allreduce to reduce network bandwidth.

        2. Call momentum to optimize the cost.

    Args:
        learning_rate (float|Variable): The learning rate used to update parameters. \
            It can be a float value or a Variable with one float value as a data element.
        momentum (float): Momentum factor.
        rampup_begin_step (int): The beginning step from which gradient compression is implemented.
        rampup_step (int): Time steps used in sparsity warm-up periods. Default is 1.
            For example, if the sparsity is [0.75, 0.9375, 0.984375, 0.996, 0.999], and the rampup_step is 100, \
                it will use 0.75 at 0~19 steps, and 0.9375 at 20~39 steps, and so on. \
                And when reach sparsity array ends, it will use 0.999 then and after.
        sparsity (list[float]): Get top important element from gradient tensor, the ratio is (1 - current sparsity). \
            Default is [0.999]. For example, if the sparsity is [0.99, 0.999], \
                the top [1%, 0.1%] important element will be transmitted.
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        use_nesterov (bool): Enables Nesterov momentum. True means use Nesterov. Default is False.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_fluid_regularizer_L1Decay` , :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipByNorm, optional): Gradient cliping strategy. ``DGCMomentumOptimizer`` only support 
            :ref:`api_fluid_clip_GradientClipByNorm` , and if not, it will raise TypeError. Default None, 
            meaning there is no gradient clipping.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            optimizer = fluid.optimizer.DGCMomentumOptimizer(
                        learning_rate=0.0001,
                        momentum=0.9,
                        rampup_step=1000,
                        rampup_begin_step=1252,
                        sparsity=[0.999, 0.999])

    """
    _u_velocity_acc_str = "_dgc_u_"
    _v_velocity_acc_str = "_dgc_v_"

    def __init__(self,
                 learning_rate,
                 momentum,
                 rampup_begin_step,
                 rampup_step=1,
                 sparsity=[0.999],
                 parameter_list=None,
                 use_nesterov=False,
                 num_trainers=None,
                 regularization=None,
                 grad_clip=None,
                 name=None):
        if framework.in_dygraph_mode():
            raise Exception("In dygraph, don't support DGCMomentumOptimizer.")

        assert core.is_compiled_with_cuda(), \
            "Paddle is not compiled with CUDA. DGC is only support GPU for now."

        assert learning_rate is not None
        assert momentum is not None
        super(DGCMomentumOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            grad_clip=grad_clip,
            name=name)
        self.type = "dgc_momentum"
        self._momentum = momentum
        self._use_nesterov = bool(use_nesterov)

        assert rampup_begin_step >= 0, "rampup_begin_step must >= 0"
        self._rampup_begin_step = rampup_begin_step
        self._rampup_step = rampup_step
        self._sparsity = sparsity

        self._rampup_begin_step_var = None
        self._global_step_var = None

        self._dgc_clip_norm = None
        if grad_clip is not None:
            if not isinstance(grad_clip, GradientClipByNorm):
                raise TypeError(
                    "The type of grad_clip should be 'GradientClipByNorm', because DGCMomentumOptimizer only support GradientClipByNorm"
                )
            assert isinstance(
                num_trainers, int
            ), "The type of num_trainers should be 'int', but received %s" % type(
                num_trainers)
            assert num_trainers > 0, "The value of num_trainers should be greater than 0!"

            self._num_trainers = num_trainers
            self._dgc_clip_norm = grad_clip.clip_norm * (num_trainers**-0.5)

        self.regular_type, self.regular_coeff = self._get_regularization_param(
            self.regularization)

    def _get_regularization_param(self, regularization):
        regular_type = 0
        regular_coeff = 0.0

        if regularization is not None:
            regular_coeff = regularization._regularization_coeff
            from .regularizer import L1Decay, L2Decay
            if isinstance(regularization, L1Decay):
                regular_type = 1
            elif isinstance(regularization, L2Decay):
                regular_type = 2
            else:
                assert False, 'regularization must be None|L1Decay|L2Deacy'
        return regular_type, regular_coeff

    def _is_use_dgc(self, param_var, grad_var):
        var_numel = abs(reduce(lambda x, y: x * y, param_var.shape))
        if var_numel < 16384 or \
           param_var.type == core.VarDesc.VarType.SELECTED_ROWS  or \
           grad_var.type == core.VarDesc.VarType.SELECTED_ROWS  or  \
               param_var.dtype != core.VarDesc.VarType.FP32 :
            return False
        return True

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)
        velocity_acc = self._get_accumulator(self._u_velocity_acc_str,
                                             param_and_grad[0])
        assert velocity_acc is not None

        inputs = {
            "Param": param_and_grad[0],
            "Grad": param_and_grad[1],
            "Velocity": velocity_acc,
            "LearningRate": self._create_param_lr(param_and_grad),
        }
        outputs = {
            "ParamOut": param_and_grad[0],
            "VelocityOut": velocity_acc,
        }
        attrs = {"mu": self._momentum, "use_nesterov": self._use_nesterov}

        if not self._is_use_dgc(param_and_grad[0], param_and_grad[1]):
            type = "momentum"
        else:
            type = "dgc_momentum"
            inputs.update({
                "current_step": self._global_step_var,
                "nranks": self._nranks_var
            })
            outputs.update({'Grad_out': param_and_grad[1]})
            attrs.update({"rampup_begin_step": float(self._rampup_begin_step)})

        # create the dgc momentum optimize op
        dgc_momentum_op = block.append_op(
            type=type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=True)
        return dgc_momentum_op

    def _add_auto_increment_var(self, counter_name, begin, step=1):
        helper = LayerHelper('global_step_counter')
        counter, is_new_var = helper.create_or_get_global_variable(
            name=counter_name, dtype='float32', shape=[1], persistable=True)
        if is_new_var:
            helper.set_variable_initializer(
                counter,
                initializer=Constant(
                    value=float(begin - 1), force_cpu=True))
            helper.main_program.global_block()._prepend_op(
                type='increment',
                inputs={'X': [counter]},
                outputs={'Out': [counter]},
                attrs={'step': float(step)},
                stop_gradient=True)
            counter.stop_gradient = True

        return counter

    def _add_nranks_var(self, name, value=-1):
        helper = LayerHelper('global_step_counter')
        counter, is_new_var = helper.create_or_get_global_variable(
            name=name, dtype='float32', shape=[1], persistable=True)
        if is_new_var:
            helper.set_variable_initializer(
                counter,
                initializer=Constant(
                    value=float(value), force_cpu=True))
            counter.stop_gradient = True

        return counter

    def _append_dgc_ops(self, param_and_grads):
        main_program = default_main_program()
        main_program._enable_dgc = True

        # step counter
        self._global_step_var = self._add_auto_increment_var(
            counter_name=core.dgc.kDGCCounterName(), begin=0)

        self._nranks_var = self._add_nranks_var(
            name=core.dgc.kDGCNRanksName(), value=-1)

        # rampup begin step var for all_reduce_op_handle
        self._rampup_begin_step_var = tensor.create_global_var(
            shape=[1],
            dtype=core.VarDesc.VarType.FP32,
            persistable=True,
            name=core.dgc.kDGCRampUpBeginStepName(),
            value=self._rampup_begin_step * 1.0,
            force_cpu=True)

        self.helper = LayerHelper(self.__class__.__name__)

        for param_var, grad_var in param_and_grads:
            # reuse velocity in dgc_op and dgc_momentum_op
            u_var = self._add_accumulator(self._u_velocity_acc_str, param_var)

            if not self._is_use_dgc(param_var, grad_var):
                continue

            v_var = self._add_accumulator(self._v_velocity_acc_str, param_var)

            k_var = tensor.create_global_var(
                shape=[1],
                dtype=param_var.dtype,
                persistable=True,
                name=param_var.name + core.dgc.kDGCKName(),
                value=0.0,
                force_cpu=True)

            encoded_var = tensor.create_global_var(
                shape=[1],
                dtype=param_var.dtype,
                persistable=True,
                name=param_var.name + core.dgc.kDGCEncodedName(),
                value=0.0,
                force_cpu=False)

            gather_var = tensor.create_global_var(
                shape=[1],
                dtype=param_var.dtype,
                persistable=True,
                name=param_var.name + core.dgc.kDGCGatherName(),
                value=0.0,
                force_cpu=False)

            # del back oprolevarname
            op_maker = core.op_proto_and_checker_maker
            backward = core.op_proto_and_checker_maker.OpRole.Backward
            for op in main_program.global_block().ops:
                if not self._is_the_backward_op(op):
                    continue

                var_attr = op.all_attrs()[op_maker.kOpRoleVarAttrName()]
                if param_var.name not in var_attr:
                    continue

                var_attr.remove(param_var.name)
                var_attr.remove(grad_var.name)
                if len(var_attr) > 1:
                    op._set_attr(op_maker.kOpRoleVarAttrName(), var_attr)
                else:
                    op._remove_attr(op_maker.kOpRoleVarAttrName())

            clip_var = grad_var
            if self._dgc_clip_norm is not None:
                clip_var = self._append_clip_norm(grad_var, self._dgc_clip_norm)
            self._dgc_op(param_var, clip_var, grad_var, u_var, v_var, k_var,
                         encoded_var, gather_var)

    def _is_the_backward_op(self, op):
        op_maker = core.op_proto_and_checker_maker
        backward = core.op_proto_and_checker_maker.OpRole.Backward
        if op_maker.kOpRoleVarAttrName() in op.attr_names and \
                int(op.all_attrs()[op_maker.kOpRoleAttrName()]) == int(backward):
            return True
        return False

    def _clip_by_norm(self, x, max_norm, name=None):
        args = {'x': x, 'max_norm': max_norm, 'name': name}

        helper = LayerHelper("dgc_clip_by_norm_op", **args)

        if name is None:
            name = unique_name.generate_with_ignorable_key(".".join(
                [helper.name, 'tmp']))

        out = helper.create_variable(
            type=x.type, name=name, dtype=x.dtype, persistable=False)

        helper.append_op(
            type="dgc_clip_by_norm",
            inputs={"X": x,
                    "current_step": self._global_step_var},
            attrs={
                "max_norm": max_norm,
                "rampup_begin_step": float(self._rampup_begin_step)
            },
            outputs={"Out": out})
        return out

    def _append_clip_norm(self, grad_var, clip_norm):
        with grad_var.block.program._backward_role_guard():
            return self._clip_by_norm(
                x=grad_var, max_norm=clip_norm, name=grad_var.name)

    def _dgc_op(self, param_var, clip_var, grad_var, u_var, v_var, k_var,
                encoded_var, gather_var):
        block = framework.default_main_program().global_block()
        op_maker = core.op_proto_and_checker_maker

        regular_type = self.regular_type
        regular_coeff = self.regular_coeff
        # The regularizer of the Parameters have higher priority
        if param_var.regularizer is not None:
            regular_type, regular_coeff = self._get_regularization_param(
                param_var.regularizer)

        dgc_op = block.append_op(
            type="dgc",
            inputs={
                "U": u_var,
                "V": v_var,
                "Grad": clip_var,
                "Param": param_var,
                "current_step": self._global_step_var,
                "nranks": self._nranks_var,
            },
            outputs={
                "U_out": u_var,
                "V_out": v_var,
                "EncodeGrad": encoded_var,
                "k": k_var,
                "Grad_out": grad_var,
                "GatherBuff": gather_var,
            },
            attrs={
                "m": self._momentum,
                "sparsity": self._sparsity,
                "use_nesterov": self._use_nesterov,
                "rampup_begin_step": float(self._rampup_begin_step),
                "rampup_step": float(self._rampup_step),
                "regular_coeff": float(regular_coeff),
                "regular_type": int(regular_type),
            },
            stop_gradient=True)

        backward = op_maker.OpRole.Backward
        dgc_op._set_attr(op_maker.kOpRoleAttrName(), backward)
        dgc_op._set_attr(op_maker.kOpRoleVarAttrName(),
                         [param_var.name, grad_var.name])

    @imperative_base.no_grad
    def apply_gradients(self, params_grads):
        # Note: since we can't use all_reduce_op now,
        # dgc_op should be the last op of one grad.
        # Maybe need a grad allreduce pass.
        self._append_dgc_ops(params_grads)

        params_grads = sorted(params_grads, key=lambda x: x[0].name)
        params_grads, table_param_and_grad, table_optimize_op = \
            self._process_distribute_lookuptable(params_grads)

        not_dgc_params_grads = []
        dgc_params_grads = []
        # DGC clip and regularization in optimizer.backward
        for param, grad in params_grads:
            if not self._is_use_dgc(param, grad):
                not_dgc_params_grads.append((param, grad))
            else:
                dgc_params_grads.append((param, grad))

        # 'optimizer(grad_clip)' or 'set_gradient_clip'
        if self._grad_clip is not None:
            not_dgc_params_grads = self._grad_clip(not_dgc_params_grads)
        else:
            not_dgc_params_grads = append_gradient_clip_ops(
                not_dgc_params_grads)

        not_dgc_params_grads = self.append_regularization_ops(
            not_dgc_params_grads, self.regularization)

        params_grads = not_dgc_params_grads + dgc_params_grads
        params_grads = sorted(params_grads, key=lambda x: x[0].name)

        optimize_ops = self._create_optimization_pass(params_grads)
        if table_optimize_op is not None:
            optimize_ops.append(table_optimize_op)
            params_grads.append(table_param_and_grad)

        return optimize_ops


class LarsMomentumOptimizer(Optimizer):
    r"""
    Momentum optimizer with LARS support

    The update equations are as follows:

    .. math::

        & local\_learning\_rate = learning\_rate * lars\_coeff * \\
          \\frac{||param||}{||gradient|| + lars\_weight\_decay * ||param||}

        & velocity = mu * velocity + local\_learning\_rate * (gradient + lars\_weight\_decay * param + epsilon)

        & param = param - velocity

    Parameters:
        learning_rate (float|Variable): The learning rate used to update parameters. \
            Can be a float value or a Variable with one float value as data element. \
            momentum (float): momentum factor
        lars_coeff (float): Defines how much we trust the layer to change its weights.
        lars_weight_decay (float): Weight decay coefficient for decaying using LARS.
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_fluid_regularizer_L1Decay` , :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of 
            some derived class of ``GradientClipBase`` . There are three cliping strategies 
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , 
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.
        exclude_from_weight_decay (list[str], optional): Name string of layers which will be exclude from lars weight decay. Default is None.
        epsilon (float, optional): Epsilon to avoid Division by Zero when calculate local lr. Default is 0.
        multi_precision (bool, optional): Whether to use multi-precision during weight updating.
        rescale_grad (float, optional): Multiply the gradient with `rescale_grad` \
            before updating. Often choose to be `1.0/batch_size`.
        
    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(inp, size=3)
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.LarsMomentumOptimizer(learning_rate=0.001, momentum=0.9)
            optimizer.minimize(out)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            exe.run(
                feed={"inp": np_inp},
                fetch_list=[out.name])
    """
    _velocity_acc_str = "velocity"

    def __init__(self,
                 learning_rate,
                 momentum,
                 lars_coeff=0.001,
                 lars_weight_decay=0.0005,
                 parameter_list=None,
                 regularization=None,
                 grad_clip=None,
                 name=None,
                 exclude_from_weight_decay=None,
                 epsilon=0,
                 multi_precision=False,
                 rescale_grad=1.0):
        assert learning_rate is not None
        assert momentum is not None
        super(LarsMomentumOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            grad_clip=grad_clip,
            name=name)
        self.type = "lars_momentum"
        self._momentum = momentum
        self._lars_coeff = float(lars_coeff)
        self._lars_weight_decay = float(lars_weight_decay)
        self._epsilon = float(epsilon)
        if exclude_from_weight_decay is None:
            self._exclude_from_weight_decay = []
        else:
            self._exclude_from_weight_decay = exclude_from_weight_decay
        self._multi_precision = multi_precision
        self._rescale_grad = float(rescale_grad)
        self._master_weights = {}

    def _create_master_weight(self, param):
        if param.name in self._master_weights:
            var = self._master_weights[param.name]
        else:
            assert isinstance(self.helper, LayerHelper)

            var_name = param.name + '_fp32_master'
            var_name = unique_name.generate(var_name)
            var = layers.create_global_var(
                name=var_name,
                shape=param.shape,
                value=0,
                dtype='float32',
                persistable=True)
            block = self.helper.startup_program.global_block()
            block.append_op(
                type="cast",
                inputs={"X": [param]},
                outputs={"Out": [var]},
                attrs={
                    "in_dtype": param.dtype,
                    "out_dtype": core.VarDesc.VarType.FP32
                })
            self._master_weights[param.name] = var
        return var

    def _get_accumulator(self, name, param):
        """Utility function to fetch an accumulator for a parameter
        Args:
            name: name of the accumulator
            param: parameter variable for which accumulator is to be fetched
        Returns:
            accumulator variable for the parameter
        """
        if self._name is not None:
            name = self._name + "_" + name
        find_master = self._multi_precision and param.dtype == core.VarDesc.VarType.FP16
        target_param = self._master_weights[
            param.name] if find_master else param
        target_name = target_param.name
        if (name not in self._accumulators or
                target_name not in self._accumulators[name]):
            raise Exception("Accumulator {} does not exist for parameter {}".
                            format(name, target_name))
        return self._accumulators[name][target_name]

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        for p in parameters:
            if self._multi_precision and p.dtype == core.VarDesc.VarType.FP16:
                master_p = self._create_master_weight(p)
                self._add_accumulator(self._velocity_acc_str, master_p)
                continue
            if p.dtype == core.VarDesc.VarType.FP16 and not self._multi_precision:
                warnings.warn(
                    "Accumulating with FP16 in optimizer can lead to poor accuracy or slow convergence."
                    "Consider using multi_precision=True option of the Lars optimizer."
                )
            self._add_accumulator(self._velocity_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)
        _lars_weight_decay = self._lars_weight_decay
        param_name = param_and_grad[0].name
        if len(self._exclude_from_weight_decay) > 0:
            for name in self._exclude_from_weight_decay:
                if name in param_name:
                    _lars_weight_decay = 0.0
                    break

        velocity_acc = self._get_accumulator(self._velocity_acc_str,
                                             param_and_grad[0])
        lr = self._create_param_lr(param_and_grad)

        find_master = self._multi_precision and param_and_grad[
            0].dtype == core.VarDesc.VarType.FP16
        master_weight = (self._master_weights[param_and_grad[0].name]
                         if find_master else None)

        attrs = {
            "mu": self._momentum,
            "lars_coeff": self._lars_coeff,
            "lars_weight_decay": [_lars_weight_decay],
            "multi_precision": find_master,
            "epsilon": self._epsilon,
            "rescale_grad": self._rescale_grad
        }

        inputs = {
            "Param": param_and_grad[0],
            "Grad": param_and_grad[1],
            "Velocity": velocity_acc,
            "LearningRate": lr
        }

        outputs = {"ParamOut": param_and_grad[0], "VelocityOut": velocity_acc}

        if find_master:
            inputs["MasterParam"] = master_weight
            outputs["MasterParamOut"] = master_weight

        # create the momentum optimize op
        momentum_op = block.append_op(
            type=self.type if _lars_weight_decay != 0.0 else 'momentum',
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=True)

        return momentum_op


class AdagradOptimizer(Optimizer):
    r"""
    The Adaptive Gradient optimizer (Adagrad for short) can adaptively assign
    different learning rates to individual parameters.

    The parameter ``param_out`` update rule with gradient ``grad``:

    .. math::

        moment\_out &= moment + grad * grad

        param\_out &= param - \\frac{learning\_rate * grad}{\sqrt{moment\_out} + \epsilon}

    Related paper: `Adaptive Subgradient Methods for Online Learning and
    Stochastic Optimization <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_.

    The original paper does not have the ``epsilon`` attribute. It is added here
    in our implementation as also proposed `Per-parameter adaptive learning rate
    methods <http://cs231n.github.io/neural-networks-3/#ada>`_
    for numerical stability to avoid the division by zero error.

    Args:
        learning_rate (float|Variable): The learning rate used to update ``Parameter``.
            It can be a float value or a ``Variable`` with a float type.
        epsilon (float, optional): A small float value for numerical stability.
            The default value is 1e-06.
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_fluid_regularizer_L1Decay` , :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of 
            some derived class of ``GradientClipBase`` . There are three cliping strategies 
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , 
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.
        initial_accumulator_value (float, optional): Initial value for moment accumulator.
            The default value is 0.0.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid

            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.data(name="inp", shape=[2, 2])
            out = fluid.layers.fc(inp, size=3)
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.2)
            optimizer.minimize(out)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            exe.run(
                feed={"inp": np_inp},
                fetch_list=[out.name])
    """
    _moment_acc_str = "moment"

    def __init__(self,
                 learning_rate,
                 epsilon=1.0e-6,
                 parameter_list=None,
                 regularization=None,
                 grad_clip=None,
                 name=None,
                 initial_accumulator_value=0.0):
        assert learning_rate is not None
        assert epsilon is not None
        super(AdagradOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            grad_clip=grad_clip,
            name=name)
        self.type = "adagrad"
        self._epsilon = epsilon
        self.initial_accumulator_value = initial_accumulator_value

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        for p in parameters:
            self._add_accumulator(
                self._moment_acc_str,
                p,
                fill_value=self.initial_accumulator_value)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        moment_acc = self._get_accumulator(self._moment_acc_str,
                                           param_and_grad[0])
        # Create the adagrad optimizer op
        adagrad_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "Moment": moment_acc,
                "LearningRate": self._create_param_lr(param_and_grad)
            },
            outputs={"ParamOut": param_and_grad[0],
                     "MomentOut": moment_acc},
            attrs={"epsilon": self._epsilon},
            stop_gradient=True)

        return adagrad_op


class AdamOptimizer(Optimizer):
    r"""
    The Adam optimizer uses an optimization described at the end
    of section 2 of `Adam paper <https://arxiv.org/abs/1412.6980>`_ ,
    it can dynamically adjusts the learning rate of each parameter using
    the 1st moment estimates and the 2nd moment estimates of the gradient.
    
    The parameter ``param_out`` update rule with gradient ``grad``:

    .. math::

        t & = t + 1

        moment\_1\_out & = {\\beta}_1 * moment\_1 + (1 - {\\beta}_1) * grad

        moment\_2\_out & = {\\beta}_2 * moment\_2 + (1 - {\\beta}_2) * grad * grad

        learning\_rate & = learning\_rate * \\
                          \\frac{\sqrt{1 - {\\beta}_2^t}}{1 - {\\beta}_1^t}

        param\_out & = param - learning\_rate * \\frac{moment\_1}{\sqrt{moment\_2} + \epsilon}

    Related paper: `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_

    Args:
        learning_rate (float|Variable, optional): The learning rate used to update ``Parameter``.
            It can be a float value or a ``Variable`` with a float type. The default value is 0.001.
        beta1 (float|Variable, optional): The exponential decay rate for the 1st moment estimates.
            It should be a float number or a Variable with shape [1] and data type as float32.
            The default value is 0.9.
        beta2 (float|Variable, optional): The exponential decay rate for the 2nd moment estimates.
            It should be a float number or a Variable with shape [1] and data type as float32.
            The default value is 0.999.
        epsilon (float|Tensor, optional): A small float value for numerical stability.
            It should be a float number or a Variable with shape [1] and data type as float32.
            The default value is 1e-08.
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_fluid_regularizer_L1Decay` , :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of 
            some derived class of ``GradientClipBase`` . There are three cliping strategies 
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , 
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.
        lazy_mode (bool, optional): The official Adam algorithm has two moving-average accumulators.
            The accumulators are updated at every step. Every element of the two moving-average
            is updated in both dense mode and sparse mode. If the size of parameter is very large,
            then the update may be very slow. The lazy mode only update the element that has
            gradient in current mini-batch, so it will be much more faster. But this mode has
            different semantics with the original Adam algorithm and may lead to different result.
            The default value is False.
        use_global_beta_pow (bool, optional): Whether to use global beta_pow. If true, Adam will use global beta_pow 
            for whole model instead of creating beta_pow for each parameter. Default is false.
        flatten_param_grads (bool, optional): Whether to flatten all parameters and gradients. Default is false.
        align_size (int, optional): The alignment size when flatten parameters and gradients. Default is -1, which means
            use same align_size as allocator. 

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid

            place = fluid.CPUPlace()
            main = fluid.Program()
            with fluid.program_guard(main):
                x = fluid.data(name='x', shape=[None, 13], dtype='float32')
                y = fluid.data(name='y', shape=[None, 1], dtype='float32')
                y_predict = fluid.layers.fc(input=x, size=1, act=None)
                cost = fluid.layers.square_error_cost(input=y_predict, label=y)
                avg_cost = fluid.layers.mean(cost)

                adam_optimizer = fluid.optimizer.AdamOptimizer(0.01)
                adam_optimizer.minimize(avg_cost)

                fetch_list = [avg_cost]
                train_reader = paddle.batch(
                    paddle.dataset.uci_housing.train(), batch_size=1)
                feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                for data in train_reader():
                    exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)

        .. code-block:: python

            # Adam with beta1/beta2 as Variable
            import paddle
            import paddle.fluid as fluid
            import paddle.fluid.layers.learning_rate_scheduler as lr_scheduler

            place = fluid.CPUPlace()
            main = fluid.Program()
            with fluid.program_guard(main):
                x = fluid.data(name='x', shape=[None, 13], dtype='float32')
                y = fluid.data(name='y', shape=[None, 1], dtype='float32')
                y_predict = fluid.layers.fc(input=x, size=1, act=None)
                cost = fluid.layers.square_error_cost(input=y_predict, label=y)
                avg_cost = fluid.layers.mean(cost)

                # define beta decay variable
                def get_decayed_betas(beta1_init, beta2_init, decay_steps, decay_rate, epsilon_init):
                    global_step = lr_scheduler._decay_step_counter()

                    beta1 = fluid.layers.create_global_var(
                        shape=[1],
                        value=float(beta1_init),
                        dtype='float32',
                        # set persistable for save checkpoints and resume
                        persistable=True,
                        name="beta1")
                    beta2 = fluid.layers.create_global_var(
                        shape=[1],
                        value=float(beta2_init),
                        dtype='float32',
                        # set persistable for save checkpoints and resume
                        persistable=True,
                        name="beta2")
                    epsilon = fluid.layers.create_global_var(
                        shape=[1],
                        value=float(epsilon_init),
                        dtype='float32',
                        # set persistable for save checkpoints and resume
                        persistable=True,
                        name="epsilon")

                    div_res = global_step / decay_steps
                    decayed_beta1 = beta1_init * (decay_rate**div_res)
                    decayed_beta2 = beta2_init * (decay_rate**div_res)
                    fluid.layers.assign(decayed_beta1, beta1)
                    fluid.layers.assign(decayed_beta2, beta2)

                    return beta1, beta2, epsilon

                beta1, beta2, epsilon = get_decayed_betas(0.9, 0.99, 1e5, 0.9, 1e-8)
                adam_optimizer = fluid.optimizer.AdamOptimizer(
                                                    learning_rate=0.01,
                                                    beta1=beta1,
                                                    beta2=beta2,
                                                    epsilon=epsilon)
                adam_optimizer.minimize(avg_cost)

                fetch_list = [avg_cost]
                train_reader = paddle.batch(
                    paddle.dataset.uci_housing.train(), batch_size=1)
                feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                for data in train_reader():
                    exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)
    """
    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"
    _beta1_pow_acc_str = "beta1_pow_acc"
    _beta2_pow_acc_str = "beta2_pow_acc"

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 parameter_list=None,
                 regularization=None,
                 grad_clip=None,
                 name=None,
                 lazy_mode=False,
                 use_global_beta_pow=False,
                 flatten_param_grads=False,
                 align_size=-1):
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        super(AdamOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            grad_clip=grad_clip,
            flatten_param_grads=flatten_param_grads,
            align_size=align_size,
            name=name)
        self.type = "adam"
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._lazy_mode = lazy_mode
        self._use_global_beta_pow = use_global_beta_pow

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        # Create accumulator tensors for first and second moments
        for p in parameters:
            self._add_accumulator(self._moment1_acc_str, p)
            self._add_accumulator(self._moment2_acc_str, p)
            if not self._use_global_beta_pow:
                self._add_accumulator(
                    name=self._beta1_pow_acc_str,
                    param=p,
                    fill_value=0.9 if isinstance(self._beta1, Variable) \
                            else self._beta1,
                    shape=[1],
                    type=core.VarDesc.VarType.LOD_TENSOR, device='cpu')
                self._add_accumulator(
                    name=self._beta2_pow_acc_str,
                    param=p,
                    fill_value=0.999 if isinstance(self._beta2, Variable) \
                            else self._beta2,
                    shape=[1],
                    type=core.VarDesc.VarType.LOD_TENSOR, device='cpu')
        if self._use_global_beta_pow:
            self._add_global_accumulator(
                name=self._beta1_pow_acc_str,
                fill_value=0.9 if isinstance(self._beta1, Variable) \
                        else self._beta1,
                shape=[1],
                type=core.VarDesc.VarType.LOD_TENSOR, device='cpu')
            self._add_global_accumulator(
                name=self._beta2_pow_acc_str,
                fill_value=0.999 if isinstance(self._beta2, Variable) \
                        else self._beta2,
                shape=[1],
                type=core.VarDesc.VarType.LOD_TENSOR, device='cpu')

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        moment1 = self._get_accumulator(self._moment1_acc_str,
                                        param_and_grad[0])
        moment2 = self._get_accumulator(self._moment2_acc_str,
                                        param_and_grad[0])
        if self._use_global_beta_pow:
            beta1_pow_acc = self._get_global_accumulator(
                self._beta1_pow_acc_str)
            beta2_pow_acc = self._get_global_accumulator(
                self._beta2_pow_acc_str)
        else:
            beta1_pow_acc = self._get_accumulator(self._beta1_pow_acc_str,
                                                  param_and_grad[0])
            beta2_pow_acc = self._get_accumulator(self._beta2_pow_acc_str,
                                                  param_and_grad[0])
        lr = self._create_param_lr(param_and_grad)
        # create the adam optimize op

        if framework.in_dygraph_mode():
            _beta1 = self._beta1 if not isinstance(
                self._beta1, Variable) else self._beta1.numpy().item(0)
            _beta2 = self._beta2 if not isinstance(
                self._beta2, Variable) else self._beta2.numpy().item(0)
            master_weight = None
            _, _, _, _, _, _ = _C_ops.adam(
                param_and_grad[0], param_and_grad[1], lr, moment1, moment2,
                beta1_pow_acc, beta2_pow_acc, master_weight, param_and_grad[0],
                moment1, moment2, beta1_pow_acc, beta2_pow_acc, master_weight,
                'epsilon', self._epsilon, 'lazy_mode', self._lazy_mode,
                'min_row_size_to_use_multithread', 1000, 'beta1', _beta1,
                'beta2', _beta2, 'use_global_beta_pow',
                self._use_global_beta_pow)

            return None

        inputs = {
            "Param": [param_and_grad[0]],
            "Grad": [param_and_grad[1]],
            "LearningRate": [lr],
            "Moment1": [moment1],
            "Moment2": [moment2],
            "Beta1Pow": [beta1_pow_acc],
            "Beta2Pow": [beta2_pow_acc]
        }

        # Pass found_inf to adam, to skip update for not only param, but also momentum and beta_pow
        found_inf = self._get_auxiliary_var('found_inf')

        if found_inf:
            inputs['SkipUpdate'] = found_inf

        outputs = {
            "ParamOut": [param_and_grad[0]],
            "Moment1Out": [moment1],
            "Moment2Out": [moment2],
            "Beta1PowOut": [beta1_pow_acc],
            "Beta2PowOut": [beta2_pow_acc],
        }
        attrs = {
            "lazy_mode": self._lazy_mode,
            "min_row_size_to_use_multithread": 1000,
            'use_global_beta_pow': self._use_global_beta_pow
        }

        if isinstance(self._beta1, Variable):
            inputs['Beta1Tensor'] = self._beta1
        else:
            attrs['beta1'] = self._beta1
        if isinstance(self._beta2, Variable):
            inputs['Beta2Tensor'] = self._beta2
        else:
            attrs['beta2'] = self._beta2
        if isinstance(self._epsilon, Variable):
            inputs['EpsilonTensor'] = self._epsilon
        else:
            attrs['epsilon'] = self._epsilon

        adam_op = block.append_op(
            type=self.type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=True)

        return adam_op

    def _finish_update(self, block, parameters_and_grads):
        r"""Update beta1_pow and beta2_pow accumulator
        """
        assert isinstance(block, framework.Block)
        if self._use_global_beta_pow:
            beta1_pow_acc = self._get_global_accumulator(
                self._beta1_pow_acc_str)
            beta2_pow_acc = self._get_global_accumulator(
                self._beta2_pow_acc_str)

            with block.program._optimized_guard([]):
                inputs = {"X": beta1_pow_acc}
                outputs = {"Out": beta1_pow_acc}
                attrs = {}
                if isinstance(self._beta1, Variable):
                    inputs["Y"] = self._beta1
                    # use elementwise_mul for better performance
                    block.append_op(
                        type="elementwise_mul",
                        inputs=inputs,
                        outputs=outputs,
                        attrs=attrs,
                        stop_gradient=True)
                else:
                    attrs['scale'] = self._beta1
                    block.append_op(
                        type="scale",
                        inputs=inputs,
                        outputs=outputs,
                        attrs=attrs,
                        stop_gradient=True)

                inputs = {"X": beta2_pow_acc}
                outputs = {"Out": beta2_pow_acc}
                attrs = {}
                if isinstance(self._beta2, Variable):
                    inputs["Y"] = self._beta2
                    # use elementwise_mul for better performance
                    block.append_op(
                        type="elementwise_mul",
                        inputs=inputs,
                        outputs=outputs,
                        attrs=attrs,
                        stop_gradient=True)
                else:
                    attrs['scale'] = self._beta2
                    block.append_op(
                        type="scale",
                        inputs=inputs,
                        outputs=outputs,
                        attrs=attrs,
                        stop_gradient=True)


class AdamaxOptimizer(Optimizer):
    r"""
    The Adamax optimizer is implemented based on the Adamax Optimization 
    in Section 7 of `Adam paper <https://arxiv.org/abs/1412.6980>`_.
    The Adamax algorithm is a variant of the Adam algorithm based on the infinite norm,
    which makes the learning rate update algorithm more stable and simple.

    The parameter ``param_out`` update rule with gradient ``grad``:

    .. math::

        t & = t + 1

        moment\_out & = {\\beta}_1 * moment + (1 - {\\beta}_1) * grad

        inf\_norm\_out & = max({\\beta}_2 * inf\_norm + \epsilon, |grad|)

        learning\_rate & = \\frac{learning\_rate}{1 - {\\beta}_1^t}

        param\_out & = param - learning\_rate * \\frac{moment\_out}{inf\_norm\_out}

    Related paper: `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_

    The original paper does not have an ``epsilon`` attribute,
    it is added here for numerical stability to prevent the division by 0 error.

    Args:
        learning_rate (float|Variable, optional): The learning rate used to update ``Parameter``.
            It can be a float value or a ``Variable`` with a float type. The default value is 0.001.
        beta1 (float, optional): The exponential decay rate for the 1st moment estimates.
            The default value is 0.9.
        beta2 (float, optional): The exponential decay rate for the 2nd moment estimates.
            The default value is 0.999.
        epsilon (float, optional): A small float value for numerical stability.
            The default value is 1e-08.
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_fluid_regularizer_L1Decay` , :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of 
            some derived class of ``GradientClipBase`` . There are three cliping strategies 
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , 
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    **Notes**:
        **Currently, AdamaxOptimizer doesn't support sparse parameter optimization.**

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy

          # First create the Executor.
          place = fluid.CPUPlace() # fluid.CUDAPlace(0)
          exe = fluid.Executor(place)

          train_program = fluid.Program()
          startup_program = fluid.Program()
          with fluid.program_guard(train_program, startup_program):
              data = fluid.data(name='X', shape=[None, 1], dtype='float32')
              hidden = fluid.layers.fc(input=data, size=10)
              loss = fluid.layers.mean(hidden)
              adam = fluid.optimizer.AdamaxOptimizer(learning_rate=0.2)
              adam.minimize(loss)

          # Run the startup program once and only once.
          exe.run(startup_program)

          x = numpy.random.random(size=(10, 1)).astype('float32')
          outs = exe.run(program=train_program,
                        feed={'X': x},
                         fetch_list=[loss.name])
    """
    _moment_acc_str = "moment"
    _inf_norm_acc_str = "inf_norm"
    _beta1_pow_acc_str = "beta1_pow_acc"

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 parameter_list=None,
                 regularization=None,
                 grad_clip=None,
                 name=None):
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        super(AdamaxOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            grad_clip=grad_clip,
            name=name)
        self.type = "adamax"
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

    def _create_accumulators(self, block, parameters):
        # Create accumulator tensors for first moment and infinity norm
        for p in parameters:
            self._add_accumulator(self._moment_acc_str, p)
            self._add_accumulator(self._inf_norm_acc_str, p)
            self._add_accumulator(
                name=self._beta1_pow_acc_str,
                param=p,
                fill_value=self._beta1,
                shape=[1])

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        moment = self._get_accumulator(self._moment_acc_str, param_and_grad[0])
        inf_norm = self._get_accumulator(self._inf_norm_acc_str,
                                         param_and_grad[0])
        beta1_pow_acc = self._get_accumulator(self._beta1_pow_acc_str,
                                              param_and_grad[0])
        # create the adamax optimize op
        adamax_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": self._create_param_lr(param_and_grad),
                "Moment": moment,
                "InfNorm": inf_norm,
                "Beta1Pow": beta1_pow_acc
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "MomentOut": moment,
                "InfNormOut": inf_norm
            },
            attrs={
                "beta1": self._beta1,
                "beta2": self._beta2,
                "epsilon": self._epsilon
            },
            stop_gradient=True)

        return adamax_op

    def _finish_update(self, block, parameters_and_grads):
        """Update Beta1 Power accumulator
        """
        assert isinstance(block, framework.Block)
        for param, grad in parameters_and_grads:
            if grad is None or param.trainable is False:
                continue
            with param.block.program._optimized_guard(
                [param, grad]), name_scope('adamx'):
                beta1_pow_acc = self._get_accumulator(self._beta1_pow_acc_str,
                                                      param)
                block.append_op(
                    type="scale",
                    inputs={"X": beta1_pow_acc},
                    outputs={"Out": beta1_pow_acc},
                    attrs={"scale": self._beta1},
                    stop_gradient=True)


class DpsgdOptimizer(Optimizer):
    r"""
    We implement the Dpsgd optimizer according to CCS16 paper -
    Deep Learning with Differential Privacy.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy

          # First create the Executor.
          place = fluid.CPUPlace() # fluid.CUDAPlace(0)
          exe = fluid.Executor(place)

          train_program = fluid.Program()
          startup_program = fluid.Program()
          with fluid.program_guard(train_program, startup_program):
              data = fluid.layers.data(name='X', shape=[1], dtype='float32')
              hidden = fluid.layers.fc(input=data, size=10)
              loss = fluid.layers.mean(hidden)
              optimizer = fluid.optimizer.Dpsgd(learning_rate=0.01, clip=10.0, batch_size=16.0, sigma=1.0)
              optimizer.minimize(loss)

          # Run the startup program once and only once.
          exe.run(startup_program)

          x = numpy.random.random(size=(10, 1)).astype('float32')
          outs = exe.run(program=train_program,
                        feed={'X': x},
                         fetch_list=[loss.name])

    Args:
        learning_rate (float|Variable): the learning rate used to update parameters. \
        Can be a float value or a Variable with one float value as data element.
        clip (float): clipping threshold
        batch_size (float): batch size.
        sigma (float): for gaussian noise.
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
    Notes:
       Currently, DpsgdOptimizer doesn't support sparse parameter optimization.
    """

    def __init__(self,
                 learning_rate=0.001,
                 clip=0.9,
                 batch_size=0.999,
                 sigma=1e-8,
                 parameter_list=None):
        assert learning_rate is not None
        assert clip is not None
        assert batch_size is not None
        assert sigma is not None
        super(DpsgdOptimizer, self).__init__(
            learning_rate=learning_rate, parameter_list=parameter_list)
        self.type = "dpsgd"
        self._clip = clip
        self._batch_size = batch_size
        self._sigma = sigma
        '''
        Note(wangzhongpu):
        This property is only used for debugging, do not need to set it!
        Dpsgd operator use time(NULL) as random seed to generate random number.
        However, during debugging, we need determinated result, so we will set self._seed to a fixed number.
        '''
        self._seed = None

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        # create the dpsgd optimize op
        if self._seed == None:
            self._seed = 0

        dpsgd_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": self._create_param_lr(param_and_grad)
            },
            outputs={"ParamOut": param_and_grad[0]},
            attrs={
                "clip": self._clip,
                "batch_size": self._batch_size,
                "sigma": self._sigma,
                "seed": self._seed
            },
            stop_gradient=True)

        return dpsgd_op


class DecayedAdagradOptimizer(Optimizer):
    r"""
    The Decayed Adagrad optimizer can be seen as an Adagrad algorithm that introduces
    the decay rate to solve the problem of a sharp drop in the learning rate
    during model training when using the AdagradOptimizer.

    The parameter ``param_out`` update rule with gradient ``grad``:

    .. math::

        moment\_out & = decay * moment + (1 - decay) * grad * grad

        param\_out & = param - \\frac{learning\_rate * grad}{\sqrt{moment\_out} + \epsilon}

    Related paper: `Adaptive Subgradient Methods for Online Learning and Stochastic
    Optimization <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_.

    The original paper does not have an ``epsilon`` attribute. It is added here for numerical
    stability to avoid the division by zero error.

    Args:
        learning_rate (float|Variable): The learning rate used to update ``Parameter``.
            It can be a float value or a ``Variable`` with a float type.
        decay (float, optional): The decay rate. The default value is 0.95.
        epsilon (float, optional): A small float value for numerical stability.
            The default value is 1e-06.
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_fluid_regularizer_L1Decay` , :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of 
            some derived class of ``GradientClipBase`` . There are three cliping strategies 
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , 
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    **Notes**:
        **Currently, DecayedAdagradOptimizer doesn't support sparse parameter optimization.**

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            x = fluid.data( name='x', shape=[None, 10], dtype='float32' )
            trans = fluid.layers.fc( x, 100 )
            cost = fluid.layers.reduce_mean( trans )
            optimizer = fluid.optimizer.DecayedAdagradOptimizer(learning_rate=0.2)
            optimizer.minimize(cost)
    """
    _moment_acc_str = "moment"

    def __init__(self,
                 learning_rate,
                 decay=0.95,
                 epsilon=1.0e-6,
                 parameter_list=None,
                 regularization=None,
                 grad_clip=None,
                 name=None):
        assert learning_rate is not None
        assert decay is not None
        assert epsilon is not None

        super(DecayedAdagradOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            grad_clip=grad_clip,
            name=name)
        self.type = "decayed_adagrad"
        self._decay = decay
        self._epsilon = epsilon

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        for p in parameters:
            self._add_accumulator(self._moment_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        moment_acc = self._get_accumulator(self._moment_acc_str,
                                           param_and_grad[0])

        # Create the decayed adagrad optimizer op
        decayed_adagrad_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "Moment": moment_acc,
                "LearningRate": self._create_param_lr(param_and_grad)
            },
            outputs={"ParamOut": param_and_grad[0],
                     "MomentOut": moment_acc},
            attrs={"epsilon": self._epsilon,
                   "decay": self._decay},
            stop_gradient=True)

        return decayed_adagrad_op


class AdadeltaOptimizer(Optimizer):
    r"""
    **Notes: This API does not support sparse parameter optimization.**

    Adadelta Optimizer. Please refer to this for details:
    `ADADELTA: AN ADAPTIVE LEARNING RATE METHOD <https://arxiv.org/abs/1212.5701>`_.

    The update is done as follows:

    .. math::

        E(g_t^2) &= \\rho * E(g_{t-1}^2) + (1-\\rho) * g^2

        learning\_rate &= \sqrt{ ( E(dx_{t-1}^2) + \\epsilon ) / ( E(g_t^2) + \\epsilon ) }

        E(dx_t^2) &= \\rho * E(dx_{t-1}^2) + (1-\\rho) * (-g*learning\_rate)^2

    Args:
        learning_rate (float|Variable): global learning rate.
        epsilon (float): a small float number for numeric stability. Default 1.0e-6.
        rho (float): a floating point value indicating the decay rate. Default 0.95.
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_fluid_regularizer_L1Decay` , :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of 
            some derived class of ``GradientClipBase`` . There are three cliping strategies 
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , 
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): The default value is None. Normally there is no need for user
                to set this property. For more information, please refer to
                :ref:`api_guide_Name` .

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            image = fluid.data(name='image', shape=[None, 28], dtype='float32')
            fc = fluid.layers.fc(image, size=10)
            cost = fluid.layers.reduce_mean(fc)
            optimizer = fluid.optimizer.Adadelta(
                learning_rate=0.0003, epsilon=1.0e-6, rho=0.95)

            # optimizer_ops is a list of optimizer operators to update parameters
            # params_grads is a list of (param, param_grad), where param is each
            # parameter and param_grad is the gradient variable of param.
            optimizer_ops, params_grads = optimizer.minimize(cost)
    """

    _avg_squared_grad_acc_str = "_avg_squared_grad"
    _avg_squared_update_acc_str = "_avg_squared_update"

    def __init__(self,
                 learning_rate,
                 epsilon=1.0e-6,
                 rho=0.95,
                 parameter_list=None,
                 regularization=None,
                 grad_clip=None,
                 name=None):
        if learning_rate is None:
            raise ValueError("learning_rate is not set.")
        if epsilon is None:
            raise ValueError("epsilon is not set.")
        if rho is None:
            raise ValueError("rho is not set.")
        super(AdadeltaOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            grad_clip=grad_clip,
            name=name)
        self.type = "adadelta"
        self._epsilon = epsilon
        self._rho = rho

    def _create_accumulators(self, block, parameters):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        for p in parameters:
            self._add_accumulator(self._avg_squared_grad_acc_str, p)
            self._add_accumulator(self._avg_squared_update_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        avg_squared_grad_acc = self._get_accumulator(
            self._avg_squared_grad_acc_str, param_and_grad[0])
        avg_squared_update_acc = self._get_accumulator(
            self._avg_squared_update_acc_str, param_and_grad[0])

        # Create the adadelta optimizer op
        adadelta_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "AvgSquaredGrad": avg_squared_grad_acc,
                "AvgSquaredUpdate": avg_squared_update_acc
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "AvgSquaredGradOut": avg_squared_grad_acc,
                "AvgSquaredUpdateOut": avg_squared_update_acc
            },
            attrs={"epsilon": self._epsilon,
                   "rho": self._rho},
            stop_gradient=True)

        return adadelta_op


class RMSPropOptimizer(Optimizer):
    r"""
    Root Mean Squared Propagation (RMSProp) is an unpublished, adaptive learning
    rate method. The original slides proposed RMSProp: Slide 29 of
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf .

    The original equation is as follows:

    ..  math::

        r(w, t) & = \\rho r(w, t-1) + (1 - \\rho)(\\nabla Q_{i}(w))^2

        w & = w - \\frac{\\eta} {\\sqrt{r(w,t) + \\epsilon}} \\nabla Q_{i}(w)

    The first equation calculates moving average of the squared gradient for
    each weight. Then dividing the gradient by :math:`sqrt{v(w,t)}`.

    In some cases, adding a momentum term :math: `\\beta` is beneficial.
    In our implementation, Nesterov momentum is used:

    ..  math::

        r(w, t) & = \\rho r(w, t-1) + (1 - \\rho)(\\nabla Q_{i}(w))^2

        v(w, t) & = \\beta v(w, t-1) + \\frac{\\eta} {\\sqrt{r(w,t) +
            \\epsilon}} \\nabla Q_{i}(w)

        w & = w - v(w, t)

    if centered is True:

    ..  math::

        r(w, t) & = \\rho r(w, t-1) + (1 - \\rho)(\\nabla Q_{i}(w))^2

        g(w, t) & = \\rho g(w, t-1) + (1 - \\rho)\\nabla Q_{i}(w)

        v(w, t) & = \\beta v(w, t-1) + \\frac{\\eta} {\\sqrt{r(w,t) - (g(w, t))^2 +
            \\epsilon}} \\nabla Q_{i}(w)

        w & = w - v(w, t)

    where, :math:`\\rho` is a hyperparameter and typical values are 0.9, 0.95
    and so on. :math: `beta` is the momentum term. :math: `\\epsilon` is a
    smoothing term to avoid division by zero, usually set somewhere in range
    from 1e-4 to 1e-8.


    Parameters:
        learning_rate(float): Global learning rate.
        rho(float): rho is :math: `\\rho` in equation, default is 0.95.
        epsilon(float): :math: `\\epsilon` in equation is smoothing term to
            avoid division by zero, default is 1e-6.
        momentum(float): :math:`\\beta` in equation is the momentum term,
            default is 0.0.
        centered(bool): If True, gradients are normalized by the estimated variance of
            the gradient; if False, by the uncentered second moment. Setting this to
            True may help with training, but is slightly more expensive in terms of
            computation and memory. Defaults to False.
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_fluid_regularizer_L1Decay` , :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of 
            some derived class of ``GradientClipBase`` . There are three cliping strategies 
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , 
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.

    Raises:
        ValueError: If learning_rate, rho, epsilon, momentum are None.

    Examples:
          .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            place = fluid.CPUPlace()
            main = fluid.Program()
            with fluid.program_guard(main):
                x = fluid.layers.data(name='x', shape=[13], dtype='float32')
                y = fluid.layers.data(name='y', shape=[1], dtype='float32')
                y_predict = fluid.layers.fc(input=x, size=1, act=None)
                cost = fluid.layers.square_error_cost(input=y_predict, label=y)
                avg_cost = fluid.layers.mean(cost)

                rms_optimizer = fluid.optimizer.RMSProp(learning_rate=0.1)
                rms_optimizer.minimize(avg_cost)

                fetch_list = [avg_cost]
                train_reader = paddle.batch(
                    paddle.dataset.uci_housing.train(), batch_size=1)
                feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                for data in train_reader():
                    exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)

    """

    _momentum_acc_str = "momentum"
    _mean_square_acc_str = "mean_square"
    _mean_grad_acc_str = "mean_grad"

    def __init__(self,
                 learning_rate,
                 rho=0.95,
                 epsilon=1.0e-6,
                 momentum=0.0,
                 centered=False,
                 parameter_list=None,
                 regularization=None,
                 grad_clip=None,
                 name=None):
        super(RMSPropOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            grad_clip=grad_clip,
            name=name)
        if learning_rate is None:
            raise ValueError("learning_rate is not set.")
        if rho is None:
            raise ValueError("rho is not set.")
        if epsilon is None:
            raise ValueError("epsilon is not set.")
        if momentum is None:
            raise ValueError("momentum is not set.")

        self.type = "rmsprop"
        self._rho = rho
        self._epsilon = epsilon
        self._momentum = momentum
        self._centered = centered

    def _create_accumulators(self, block, parameters):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        for p in parameters:
            self._add_accumulator(self._momentum_acc_str, p)
            self._add_accumulator(self._mean_square_acc_str, p)
            self._add_accumulator(self._mean_grad_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        momentum_acc = self._get_accumulator(self._momentum_acc_str,
                                             param_and_grad[0])
        mean_square_acc = self._get_accumulator(self._mean_square_acc_str,
                                                param_and_grad[0])
        mean_grad_acc = self._get_accumulator(self._mean_grad_acc_str,
                                              param_and_grad[0])
        rmsprop_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "Moment": momentum_acc,
                "MeanSquare": mean_square_acc,
                "MeanGrad": mean_grad_acc,
                "LearningRate": self._create_param_lr(param_and_grad),
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "MomentOut": momentum_acc,
                "MeanSquareOut": mean_square_acc,
                "MeanGradOut": mean_grad_acc
            },
            attrs={
                "epsilon": self._epsilon,
                "decay": self._rho,
                "momentum": self._momentum,
                "centered": self._centered
            },
            stop_gradient=True)

        return rmsprop_op


class FtrlOptimizer(Optimizer):
    r"""
    FTRL (Follow The Regularized Leader) Optimizer.

    The paper that proposed Follow The Regularized Leader (FTRL):
    (https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf)

    ..  math::

        &new\_accum = squared\_accum + grad^2

        &if (lr\_power == -0.5):

        &\quad  linear\_accum += grad - \\frac{\\sqrt{new\_accum} - \\sqrt{squared\_accum}}{learning\_rate * param}

        &else:

        &\quad   linear\_accum += grad - \\frac{new\_accum^{-lr\_power} - accum^{-lr\_power}}{learning\_rate * param}


        &x = l1 * sign(linear\_accum) - linear\_accum

        &if (lr\_power == -0.5):

        &\quad   y = \\frac{\\sqrt{new\_accum}}{learning\_rate} + (2 * l2)

        &\quad   pre\_shrink = \\frac{x}{y}

        &\quad   param = (abs(linear\_accum) > l1).select(pre\_shrink, 0.0)

        &else:

        &\quad   y = \\frac{new\_accum^{-lr\_power}}{learning\_rate} + (2 * l2)

        &\quad   pre\_shrink = \\frac{x}{y}

        &\quad   param = (abs(linear\_accum) > l1).select(pre\_shrink, 0.0)

        &squared\_accum += grad^2

    Parameters:
        learning_rate (float|Variable): Global learning rate.
        l1 (float): L1 regularization strength, default is 0.0.
        l2 (float): L2 regularization strength, default is 0.0.
        lr_power (float): Learning Rate Power, default is -0.5.
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_fluid_regularizer_L1Decay` , :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of 
            some derived class of ``GradientClipBase`` . There are three cliping strategies 
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , 
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.

    Raises:
        ValueError: If learning_rate, rho, epsilon, momentum are None.

    Examples:
          .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            place = fluid.CPUPlace()
            main = fluid.Program()
            with fluid.program_guard(main):
                x = fluid.layers.data(name='x', shape=[13], dtype='float32')
                y = fluid.layers.data(name='y', shape=[1], dtype='float32')
                y_predict = fluid.layers.fc(input=x, size=1, act=None)
                cost = fluid.layers.square_error_cost(input=y_predict, label=y)
                avg_cost = fluid.layers.mean(cost)

                ftrl_optimizer = fluid.optimizer.Ftrl(learning_rate=0.1)
                ftrl_optimizer.minimize(avg_cost)

                fetch_list = [avg_cost]
                train_reader = paddle.batch(
                    paddle.dataset.uci_housing.train(), batch_size=1)
                feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                for data in train_reader():
                    exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)

    NOTE:
       Currently, FtrlOptimizer doesn't support sparse parameter optimization.
    """

    _squared_acc_str = "squared"
    _linear_acc_str = "linear"

    def __init__(self,
                 learning_rate,
                 l1=0.0,
                 l2=0.0,
                 lr_power=-0.5,
                 parameter_list=None,
                 regularization=None,
                 grad_clip=None,
                 name=None):
        super(FtrlOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            grad_clip=grad_clip,
            name=name)
        if learning_rate is None:
            raise ValueError("learning_rate is not set.")

        self.type = "ftrl"
        self._l1 = l1
        self._l2 = l2
        self._lr_power = lr_power

    def _create_accumulators(self, block, parameters):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        for p in parameters:
            self._add_accumulator(self._squared_acc_str, p)
            self._add_accumulator(self._linear_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        squared_acc = self._get_accumulator(self._squared_acc_str,
                                            param_and_grad[0])
        linear_acc = self._get_accumulator(self._linear_acc_str,
                                           param_and_grad[0])
        ftrl_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "SquaredAccumulator": squared_acc,
                "LinearAccumulator": linear_acc,
                "LearningRate": self._create_param_lr(param_and_grad),
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "SquaredAccumOut": squared_acc,
                "LinearAccumOut": linear_acc
            },
            attrs={"l1": self._l1,
                   "l2": self._l2,
                   "lr_power": self._lr_power},
            stop_gradient=True)

        return ftrl_op


class LambOptimizer(AdamOptimizer):
    r"""
    LAMB (Layer-wise Adaptive Moments optimizer for Batching training) Optimizer.

    LAMB Optimizer is designed to scale up the batch size of training without losing 
    accuracy, which supports adaptive element-wise updating and accurate layer-wise 
    correction. For more information, please refer to `Large Batch Optimization for 
    Deep Learning: Training BERT in 76 minutes <https://arxiv.org/abs/1904.00962>`_ .

    The updating of parameters follows:

    ..  math::

        m_t &= \\beta_1 m_{t - 1}+ (1 - \\beta_1)g_t 

        v_t &= \\beta_2 v_{t - 1}  + (1 - \\beta_2)g_t^2

        m_t &= \\frac{m_t}{\\beta_1^t}

        v_t &= \\frac{v_t}{\\beta_2^t}

        r_t &= \\frac{m_t}{\\sqrt{v_t}+\\epsilon}

        w_t &= w_{t-1} -\\eta_t \\frac{\\left \| w_{t-1}\\right \|}{\\left \| r_t + \\lambda w_{t-1}\\right \|} (r_t + \\lambda w_{t-1})


    where :math:`m` is the 1st moment, and :math:`v` the 2nd moment, :math:`\\eta` the 
    learning rate, :math:`\\lambda` the LAMB weight decay rate.

    Args:
        learning_rate (float|Variable, optional): the learning rate used to update parameters. \
            Can be a float value or a Variable with data type float32. Default 0.001.
        lamb_weight_decay (float, optional): The LAMB weight decay rate. Default 0.01.
        beta1 (float, optional): The exponential decay rate for the 1st moment estimates.
            Default 0.9.
        beta2 (float, optional): The exponential decay rate for the 2nd moment estimates.
            Default 0.999.
        epsilon (float, optional): A small float value for numerical stability. Default 1e-6.
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_fluid_regularizer_L1Decay` , :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of 
            some derived class of ``GradientClipBase`` . There are three cliping strategies 
            ( :ref:`api_paddle_fluid_clip_ClipGradByGlobalNorm` , :ref:`api_paddle_fluid_clip_ClipGradByNorm` ,
            :ref:`api_paddle_fluid_clip_ClipGradByValue` ). If you want better convergence, it is recommended
            to use :ref:`api_paddle_fluid_clip_ClipGradByGlobalNorm` . Default None, meaning there is no gradient clipping.
        exclude_from_weight_decay_fn (function|None): Exclude a parameter from weight 
            decay when **exclude_from_weight_decay_fn(parameter)** returns true. 
            Default None.
        name(str|None): For detailed information, please refer to 
            :ref:`api_guide_Name` . Usually name is no need to set and None by default.

    Examples:
        .. code-block:: python
            
            import paddle.fluid as fluid 

            data = fluid.data(name='x', shape=[-1, 5], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            cost = fluid.layers.mean(hidden)

            def exclude_fn(param):
                return param.name.endswith('.b_0')

            optimizer = fluid.optimizer.Lamb(learning_rate=0.002,
                                             exclude_from_weight_decay_fn=exclude_fn)
            optimizer.minimize(cost)
    """
    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"
    _beta1_pow_acc_str = "beta1_pow_acc"
    _beta2_pow_acc_str = "beta2_pow_acc"

    def __init__(self,
                 learning_rate=0.001,
                 lamb_weight_decay=0.01,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-6,
                 parameter_list=None,
                 regularization=None,
                 grad_clip=None,
                 exclude_from_weight_decay_fn=None,
                 name=None):
        assert learning_rate is not None
        assert lamb_weight_decay is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        super(LambOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            grad_clip=grad_clip,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            name=name)
        self.type = "lamb"
        self._weight_decay = lamb_weight_decay
        self._exclude_from_weight_decay_fn = exclude_from_weight_decay_fn

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)
        block.program._use_lamb = True

        moment1 = self._get_accumulator(self._moment1_acc_str,
                                        param_and_grad[0])
        moment2 = self._get_accumulator(self._moment2_acc_str,
                                        param_and_grad[0])
        beta1_pow_acc = self._get_accumulator(self._beta1_pow_acc_str,
                                              param_and_grad[0])
        beta2_pow_acc = self._get_accumulator(self._beta2_pow_acc_str,
                                              param_and_grad[0])

        if self._exclude_from_weight_decay_fn is not None \
            and self._exclude_from_weight_decay_fn(param_and_grad[0]):
            weight_decay = 0.0
        else:
            weight_decay = self._weight_decay
        lr = self._create_param_lr(param_and_grad)

        if framework.in_dygraph_mode():
            _, _, _, _, _ = _C_ops.lamb(
                param_and_grad[0], param_and_grad[1], lr, moment1, moment2,
                beta1_pow_acc, beta2_pow_acc, param_and_grad[0], moment1,
                moment2, beta1_pow_acc, beta2_pow_acc, 'beta1', self._beta1,
                'beta2', self._beta2, 'epsilon', self._epsilon, 'weight_decay',
                weight_decay)
            return None

        # create the lamb optimize op
        lamb_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": lr,
                "Moment1": moment1,
                "Moment2": moment2,
                "Beta1Pow": beta1_pow_acc,
                "Beta2Pow": beta2_pow_acc
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "Moment1Out": moment1,
                "Moment2Out": moment2,
                "Beta1PowOut": beta1_pow_acc,
                "Beta2PowOut": beta2_pow_acc
            },
            attrs={
                "beta1": self._beta1,
                "beta2": self._beta2,
                "epsilon": self._epsilon,
                "weight_decay": weight_decay
            },
            stop_gradient=True)

        return lamb_op


# We short the class name, since users will use the optimizer with the package
# name. The sample code:
#
# import paddle.fluid as fluid
#
# sgd = fluid.optimizer.SGD(...)
#
# It is no need to add an `Optimizer` as the class suffix
SGD = SGDOptimizer
Momentum = MomentumOptimizer
Adagrad = AdagradOptimizer
Adam = AdamOptimizer
Adamax = AdamaxOptimizer
Dpsgd = DpsgdOptimizer
DecayedAdagrad = DecayedAdagradOptimizer
Adadelta = AdadeltaOptimizer
RMSProp = RMSPropOptimizer
Ftrl = FtrlOptimizer
LarsMomentum = LarsMomentumOptimizer
Lamb = LambOptimizer


class ModelAverage(Optimizer):
    r"""
	:api_attr: Static Graph

    The ModelAverage optimizer accumulates specific continuous historical parameters
    during training. The accumulated historical range can be controlled by the passed
    ``average_window_rate`` argument. The averaged ``Parameter`` are used in the prediction,
    which usually can improve the accuracy of the prediction.

    Accumulate the average of the ``Parameter`` in the sliding window, the result will be saved
    in a temporary variable, can be applied to the current model's ``Parameter`` by calling
    the ``apply()`` method, and the current model ``Parameter`` can be restored by calling
    the ``restore()`` method.

    The window size for calculating the average is determined by ``average_window_rate``,
    ``min_average_window``, ``max_average_window`` and the current ``Parameter`` update times (num_updates).

    When the cumulative times (num_accumulates) is greater than the specific window
    threshold (average_window), the accumulated ``Parameter`` temporary variable is set to 0.0.
    The following example will help to understand the role of these arguments:

    ::

        if num_accumulates >= min_average_window and num_accumulates >= min(max_average_window, num_updates * average_window_rate):
            num_accumulates = 0

    In the above conditional judgment statement, ``num_accumulates`` indicates the current
    accumulated number, which can be abstractly understood as the length of the cumulative window.
    The length of the window must be at least the length set by the ``min_average_window`` argument,
    and cannot exceed the length specified by the ``max_average_window`` argument or
    ``num_updates * average_window_rate``, where ``num_updates`` indicates the current ``Parameter``
    update times, ``average_window_rate`` is a coefficient that calculates the length of the window.

    Args:
        average_window_rate (float): The calculate ratio of the window length relative to ``Parameter`` update times.
        min_average_window (int, optional): the minimum size of average window length. The default value is 10000.
        max_average_window (int, optional): The maximum size of average window length. The default value is 10000.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_fluid_regularizer_L1Decay` , :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    Examples:

      .. code-block:: python

        import paddle.fluid as fluid
        import numpy

        # First create the Executor.
        place = fluid.CPUPlace()  # fluid.CUDAPlace(0)
        exe = fluid.Executor(place)

        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            # build net
            data = fluid.data(name='X', shape=[None, 1], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            loss = fluid.layers.mean(hidden)
            optimizer = fluid.optimizer.Momentum(learning_rate=0.2, momentum=0.1)
            optimizer.minimize(loss)

            # build ModelAverage optimizer
            model_average = fluid.optimizer.ModelAverage(0.15,
                                                         min_average_window=10000,
                                                         max_average_window=12500)

            exe.run(startup_program)
            for i in range(12500):
                x = numpy.random.random(size=(10, 1)).astype('float32')
                outs = exe.run(program=train_program,
                               feed={'X': x},
                               fetch_list=[loss.name])

            # apply ModelAverage
            with model_average.apply(exe):
                x = numpy.random.random(size=(10, 1)).astype('float32')
                exe.run(program=train_program,
                        feed={'X': x},
                        fetch_list=[loss.name])
    """

    def __init__(self,
                 average_window_rate,
                 min_average_window=10000,
                 max_average_window=10000,
                 regularization=None,
                 name=None):
        if framework.in_dygraph_mode():
            raise Exception("In dygraph, don't support ModelAverage.")
        super(ModelAverage, self).__init__(
            0.0, regularization=regularization, name=name)
        self.average_window = average_window_rate
        self.min_average_window = min_average_window
        self.max_average_window = max_average_window

        self.params_grads = []
        for param in framework.default_main_program().global_block(
        ).all_parameters():
            if param.do_model_average != False:
                grad = param.block.create_var(
                    name=unique_name.generate_with_ignorable_key(".".join(
                        [param.name, 'tmp'])),
                    dtype=param.dtype,
                    persistable=False,
                    stop_gradient=True)
                self.params_grads.append((param, grad))

        for param, grad in self.params_grads:
            if grad is None:
                continue
            with param.block.program._optimized_guard(
                [param, grad]), name_scope('move_average'):
                self._append_average_accumulate_op(param)

        self.apply_program = Program()
        block = self.apply_program.global_block()
        with program_guard(main_program=self.apply_program):
            for param_grad in self.params_grads:
                self._add_average_apply_op(block, param_grad)

        self.restore_program = Program()
        block = self.restore_program.global_block()
        with program_guard(main_program=self.restore_program):
            for param_grad in self.params_grads:
                self._add_average_restore_op(block, param_grad)

    def _add_average_apply_op(self, block, param_grad):
        param = block._clone_variable(param_grad[0])
        grad = block._clone_variable(param_grad[1])
        sum_1 = block._clone_variable(self._get_accumulator('sum_1', param))
        sum_2 = block._clone_variable(self._get_accumulator('sum_2', param))
        sum_3 = block._clone_variable(self._get_accumulator('sum_3', param))
        num_accumulates = block._clone_variable(
            self._get_accumulator('num_accumulates', param))
        old_num_accumulates = block._clone_variable(
            self._get_accumulator('old_num_accumulates', param))
        num_updates = block._clone_variable(
            self._get_accumulator('num_updates', param))
        # backup param value to grad
        layers.assign(input=param, output=grad)
        # param = (sum_1 + sum_2 + sum_3) / (num_accumulates + old_num_accumulates)
        tmp = layers.sum(x=[num_accumulates, old_num_accumulates])
        sum = layers.sum(x=[sum_1, sum_2, sum_3])
        tmp = layers.cast(
            x=tmp, dtype='float32' if self._dtype == None else self._dtype)
        sum = layers.cast(
            x=sum, dtype='float32' if self._dtype == None else self._dtype)
        ops._elementwise_div(x=sum, y=tmp, out=param)

    def _add_average_restore_op(self, block, param_grad):
        param = block._clone_variable(param_grad[0])
        grad = block._clone_variable(param_grad[1])
        layers.assign(input=grad, output=param)

    def _append_average_accumulate_op(self, param):
        self.helper = LayerHelper("average_accumulate")
        sum_1 = self._add_accumulator('sum_1', param)
        sum_2 = self._add_accumulator('sum_2', param)
        sum_3 = self._add_accumulator('sum_3', param)
        num_accumulates = self._add_accumulator(
            'num_accumulates', param, dtype='int64', shape=[1])
        old_num_accumulates = self._add_accumulator(
            'old_num_accumulates', param, dtype='int64', shape=[1])
        num_updates = self._add_accumulator(
            'num_updates', param, dtype='int64', shape=[1])

        self.helper.append_op(
            type='average_accumulates',
            inputs={
                "param": param,
                "in_sum_1": sum_1,
                "in_sum_2": sum_2,
                "in_sum_3": sum_3,
                "in_num_accumulates": num_accumulates,
                "in_old_num_accumulates": old_num_accumulates,
                "in_num_updates": num_updates
            },
            outputs={
                "out_sum_1": sum_1,
                "out_sum_2": sum_2,
                "out_sum_3": sum_3,
                "out_num_accumulates": num_accumulates,
                "out_old_num_accumulates": old_num_accumulates,
                "out_num_updates": num_updates,
            },
            attrs={
                "average_window": self.average_window,
                "min_average_window": self.min_average_window,
                "max_average_window": self.max_average_window,
            },
            stop_gradient=True)

    @signature_safe_contextmanager
    def apply(self, executor, need_restore=True):
        """
        Apply the average of the cumulative ``Parameter`` to the parameters of the current model.

        Args:
            executor(fluid.Executor): The current network executor.
            need_restore(bool): Restore flag variable, if set to True, the network will restore
                the parameters of the network to the default value, if set to False,
                it will not be restored. The default value is True.

        Examples:

          .. code-block:: python

            import paddle.fluid as fluid
            import numpy

            # First create the Executor.
            place = fluid.CPUPlace()  # fluid.CUDAPlace(0)
            exe = fluid.Executor(place)

            train_program = fluid.Program()
            startup_program = fluid.Program()
            with fluid.program_guard(train_program, startup_program):
                # build net
                data = fluid.data(name='X', shape=[None, 1], dtype='float32')
                hidden = fluid.layers.fc(input=data, size=10)
                loss = fluid.layers.mean(hidden)
                optimizer = fluid.optimizer.Momentum(learning_rate=0.2, momentum=0.1)
                optimizer.minimize(loss)

                # build ModelAverage optimizer
                model_average = fluid.optimizer.ModelAverage(0.15,
                                                            min_average_window=10000,
                                                            max_average_window=12500)

                exe.run(startup_program)
                for i in range(12500):
                    x = numpy.random.random(size=(10, 1)).astype('float32')
                    outs = exe.run(program=train_program,
                                feed={'X': x},
                                fetch_list=[loss.name])

                # apply ModelAverage
                with model_average.apply(exe):
                    x = numpy.random.random(size=(10, 1)).astype('float32')
                    exe.run(program=train_program,
                            feed={'X': x},
                            fetch_list=[loss.name])
        """
        executor.run(self.apply_program)
        try:
            yield
        finally:
            if need_restore:
                self.restore(executor)

    def restore(self, executor):
        """
        Restore ``Parameter`` values of current model.
        
        Args:
            executor(fluid.Executor): The current network executor.

        Examples:

          .. code-block:: python

            import paddle.fluid as fluid
            import numpy

            # First create the Executor.
            place = fluid.CPUPlace()  # fluid.CUDAPlace(0)
            exe = fluid.Executor(place)

            train_program = fluid.Program()
            startup_program = fluid.Program()
            with fluid.program_guard(train_program, startup_program):
                # build net
                data = fluid.data(name='X', shape=[None, 1], dtype='float32')
                hidden = fluid.layers.fc(input=data, size=10)
                loss = fluid.layers.mean(hidden)
                optimizer = fluid.optimizer.Momentum(learning_rate=0.2, momentum=0.1)
                optimizer.minimize(loss)

                # build ModelAverage optimizer
                model_average = fluid.optimizer.ModelAverage(0.15,
                                                            min_average_window=10000,
                                                            max_average_window=12500)

                exe.run(startup_program)
                for i in range(12500):
                    x = numpy.random.random(size=(10, 1)).astype('float32')
                    outs = exe.run(program=train_program,
                                feed={'X': x},
                                fetch_list=[loss.name])

                # apply ModelAverage
                with model_average.apply(exe, False):
                    x = numpy.random.random(size=(10, 1)).astype('float32')
                    exe.run(program=train_program,
                            feed={'X': x},
                            fetch_list=[loss.name])

                # restore Parameters
                model_average.restore(exe)
        """
        executor.run(self.restore_program)


class ExponentialMovingAverage(object):
    r"""
	:api_attr: Static Graph

    Compute the moving average of parameters with exponential decay.
    Given a parameter :math:`\\theta`, its exponential moving average (EMA)
    will be

    ..  math::

        \\text{EMA}_0 & = 0

	\\text{EMA}_t & = \\text{decay} * \\text{EMA}_{t-1} + (1 - \\text{decay}) * \\theta_t

    The average results calculated by **update()** method will be saved in 
    temporary variables which are created and maintained by the object, and can 
    be applied to parameters of current model by calling **apply()** method. And 
    the **restore()** method is used to restore the parameters.

    **Bias correction**. All EMAs are initialized to :math:`0` and hence they will be 
    zero biased, which can be corrected by divided by a factor 
    :math:`(1 - \\text{decay}^t)` , i.e., the actual EMAs applied to parameters 
    when calling **apply()** method would be 

    ..  math::
    
        \\widehat{\\text{EMA}}_t = \\frac{\\text{EMA}_t}{1 - \\text{decay}^t}

    **Decay rate scheduling**. A large decay rate very close to 1 would result 
    in that the averages move very slowly. And a better strategy is to set a 
    relative smaller decay rate in the very beginning. The argument **thres_steps**
    allows users to pass a Variable to schedule the decay rate, in this case, 
    the actual decay rate becomes
     
    ..  math::
    
        \\min(\\text{decay}, \\frac{1 + \\text{thres_steps}}{10 + \\text{thres_steps}})

    Usually **thres_steps** can be the global training steps.


    Args:
        decay (float, optional): The exponential decay rate, usually close to 1, such as 0.999, 0.9999, ... . Default 0.999.
        thres_steps (Variable|None, optional): If not `None`, schedule the decay rate. Default None.
        name (str|None, optional): For detailed information, please refer to :ref:`api_guide_Name`. Usually name is no need to set and None by default.


    Examples:

        .. code-block:: python

            import numpy
            import paddle
            import paddle.static as static
            from paddle.static import ExponentialMovingAverage

            paddle.enable_static()

            data = static.data(name='x', shape=[-1, 5], dtype='float32')
            hidden = static.nn.fc(x=data, size=10)
            cost = paddle.mean(hidden)

            test_program = static.default_main_program().clone(for_test=True)
            optimizer = paddle.optimizer.Adam(learning_rate=0.001)
            optimizer.minimize(cost)

            ema = ExponentialMovingAverage(0.999)
            ema.update()

            place = paddle.CPUPlace()
            exe = static.Executor(place)
            exe.run(static.default_startup_program())

            for pass_id in range(3):
                for batch_id in range(6):
                    data = numpy.random.random(size=(10, 5)).astype('float32')
                    exe.run(program=static.default_main_program(),
                    feed={'x': data}, 
                    fetch_list=[cost.name])

                # usage 1
                with ema.apply(exe):
                    data = numpy.random.random(size=(10, 5)).astype('float32')
                    exe.run(program=test_program,
                        feed={'x': data}, 
                        fetch_list=[hidden.name])

                # usage 2
                with ema.apply(exe, need_restore=False):
                    data = numpy.random.random(size=(10, 5)).astype('float32')
                    exe.run(program=test_program,
                        feed={'x': data}, 
                        fetch_list=[hidden.name])
                ema.restore(exe)

    """

    def __init__(self, decay=0.999, thres_steps=None, name=None):
        if framework.in_dygraph_mode():
            raise Exception(
                "In dygraph, don't support ExponentialMovingAverage.")
        self._decay = decay
        self._thres_steps = thres_steps
        self._name = name if name is not None else ''
        self._decay_var = self._get_ema_decay()

        self._step_counter_name = "@EMA_STEP_COUNTER@"
        self._params_tmps = []
        for param in default_main_program().global_block().all_parameters():
            if param.do_model_average != False:
                tmp = param.block.create_var(
                    name=unique_name.generate(".".join(
                        [self._name + param.name, 'ema_tmp'])),
                    dtype=param.dtype,
                    persistable=False,
                    stop_gradient=True)
                self._params_tmps.append((param, tmp))

        self._ema_vars = {}
        for param, tmp in self._params_tmps:
            with param.block.program._optimized_guard(
                [param, tmp]), name_scope('moving_average'):
                self._ema_vars[param.name] = self._create_ema_vars(param)

        self.apply_program = Program()
        block = self.apply_program.global_block()
        with program_guard(main_program=self.apply_program):
            decay_pow, global_step = self._get_decay_pow(block)
            for param, tmp in self._params_tmps:
                param = block._clone_variable(param)
                tmp = block._clone_variable(tmp)
                ema = block._clone_variable(self._ema_vars[param.name])
                layers.assign(input=param, output=tmp)
                # bias correction
                with layers.control_flow.Switch() as switch:
                    with switch.case(global_step > 0):
                        layers.assign(
                            output=param, input=ema / (1.0 - decay_pow))
                    with switch.default():
                        layers.assign(output=param, input=ema)

        self.restore_program = Program()
        block = self.restore_program.global_block()
        with program_guard(main_program=self.restore_program):
            for param, tmp in self._params_tmps:
                tmp = block._clone_variable(tmp)
                param = block._clone_variable(param)
                layers.assign(input=tmp, output=param)

    def _get_ema_decay(self):
        with default_main_program()._lr_schedule_guard():
            decay_var = layers.tensor.create_global_var(
                shape=[1],
                value=self._decay,
                dtype='float32',
                persistable=True,
                name="scheduled_ema_decay_rate")

            if self._thres_steps is not None:
                decay_t = (self._thres_steps + 1.0) / (self._thres_steps + 10.0)
                with layers.control_flow.Switch() as switch:
                    with switch.case(decay_t < self._decay):
                        layers.tensor.assign(decay_t, decay_var)
                    with switch.default():
                        layers.tensor.assign(
                            np.array(
                                [self._decay], dtype=np.float32),
                            decay_var)
        return decay_var

    def _get_decay_pow(self, block):
        global_step = layers.create_global_var(
            name=self._step_counter_name,
            shape=[1],
            value=0,
            dtype='int64',
            persistable=True)
        global_step = layers.cast(global_step, "float32")
        decay_var = block._clone_variable(self._decay_var)
        decay_pow_acc = layers.elementwise_pow(decay_var, global_step)
        return decay_pow_acc, global_step

    def _create_ema_vars(self, param):
        param_ema = layers.create_global_var(
            name=unique_name.generate(self._name + param.name + '_ema'),
            shape=param.shape,
            value=0.0,
            dtype=param.dtype,
            persistable=True)

        return param_ema

    def update(self):
        """ 
        Update Exponential Moving Average. Should only call this method in 
        train program.
        """
        global_step = layers.autoincreased_step_counter(
            counter_name=self._step_counter_name)
        param_master_emas = []
        for param, tmp in self._params_tmps:
            with param.block.program._optimized_guard(
                [param, tmp]), name_scope('moving_average'):
                param_ema = self._ema_vars[param.name]
                if param.name + '.master' in self._ema_vars:
                    master_ema = self._ema_vars[param.name + '.master']
                    param_master_emas.append([param_ema, master_ema])
                else:
                    ema_t = param_ema * self._decay_var + param * (
                        1 - self._decay_var)
                    layers.assign(input=ema_t, output=param_ema)

        # for fp16 params
        for param_ema, master_ema in param_master_emas:
            default_main_program().global_block().append_op(
                type="cast",
                inputs={"X": master_ema},
                outputs={"Out": param_ema},
                attrs={
                    "in_dtype": master_ema.dtype,
                    "out_dtype": param_ema.dtype
                })

    @signature_safe_contextmanager
    def apply(self, executor, need_restore=True):
        """
        Apply moving average to parameters for evaluation.
        
        Args:
            executor (Executor): The Executor to execute applying.
            need_restore (bool, optional): Whether to restore parameters after 
                applying. Default True.
        """
        executor.run(self.apply_program)
        try:
            yield
        finally:
            if need_restore:
                self.restore(executor)

    def restore(self, executor):
        """Restore parameters.
        
        Args:
            executor (Executor): The Executor to execute restoring.
        """
        executor.run(self.restore_program)


class PipelineOptimizer(object):
    """
	:api_attr: Static Graph

    Pipeline Optimizer: Make a program to run as pipeline, that is splitting a
    program into multiple sections (sub-programs) and each section run on a
    device to enable the training of large scale models and the use of
    heterogeneous devices. Meanwhile, all sections run in the stype of pipeline.

    Args:
        optimizer (Optimizer): The optimizer to use, such as SGD.
        num_microbatches (int): Number of microbatches. [Optional. Default:1].
        start_cpu_core_id (int): The first cpu core id to use. [Optional. Default:0].
    
    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers

            with fluid.device_guard("gpu:0"):
                x = fluid.layers.data(name='x', shape=[1], dtype='int64', lod_level=0)
                y = fluid.layers.data(name='y', shape=[1], dtype='int64', lod_level=0)
                data_loader = fluid.io.DataLoader.from_generator(
                    feed_list=[x, y],
                    capacity=64,
                    use_double_buffer=True,
                    iterable=False)

                emb_x = layers.embedding(input=x, param_attr=fluid.ParamAttr(name="embx"), size=[10,2], is_sparse=False)
                emb_y = layers.embedding(input=y, param_attr=fluid.ParamAttr(name="emby",learning_rate=0.9), size=[10,2], is_sparse=False)

            with fluid.device_guard("gpu:1"):
                concat = layers.concat([emb_x, emb_y], axis=1)
                fc = layers.fc(input=concat, name="fc", size=1, num_flatten_dims=1, bias_attr=False)
                loss = layers.reduce_mean(fc)
            optimizer = fluid.optimizer.SGD(learning_rate=0.5)
            optimizer = fluid.optimizer.PipelineOptimizer(optimizer)
            optimizer.minimize(loss)

            def train_reader():
                for _ in range(4):
                    x = np.random.random(size=[1]).astype('int64')
                    y = np.random.random(size=[1]).astype('int64')
                    yield x, y
            data_loader.set_sample_generator(train_reader, batch_size=1)

            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            batch_size = 1
            data_loader.start()
            exe.train_from_dataset(
                    fluid.default_main_program())
            data_loader.reset()
    """

    def __init__(self, optimizer, num_microbatches=1, start_cpu_core_id=0):
        self._device = 'cpu'
        if core.is_compiled_with_npu():
            self._device = "npu"
        elif core.is_compiled_with_cuda():
            self._device = "gpu"
        if framework.in_dygraph_mode():
            raise Exception("In dygraph, don't support PipelineOptimizer.")
        valid_optimizers = (Optimizer, paddle.optimizer.Optimizer,
                            paddle.fluid.contrib.mixed_precision.decorator.
                            OptimizerWithMixedPrecision)
        if not isinstance(optimizer, valid_optimizers):
            raise ValueError("The 'optimizer' parameter for "
                             "PipelineOptimizer must be an instance of "
                             "{}, but the given type is {}.".format(
                                 valid_optimizers, type(optimizer)))
        self._optimizer = optimizer

        # Get the original optimizer defined by users, such as SGD
        self._origin_optimizer = self._optimizer
        while hasattr(self._origin_optimizer, "inner_opt"):
            self._origin_optimizer = self._origin_optimizer.inner_opt

        assert num_microbatches >= 1, (
            "num_microbatches must be a positive value.")
        self._num_microbatches = num_microbatches
        assert start_cpu_core_id >= 0, (
            "start_cpu_core_id must be a non-negative integer.")
        self._start_cpu_core_id = start_cpu_core_id
        self._place_list = None
        op_maker = core.op_proto_and_checker_maker
        self._op_role = op_maker.OpRole
        self._op_role_key = op_maker.kOpRoleAttrName()
        self._op_role_var_key = op_maker.kOpRoleVarAttrName()
        self._op_device_key = op_maker.kOpDeviceAttrName()
        self._param_device_map = None
        self._pipeline_pair = []
        self._pp_ring_map = dict()
        self.output_var_to_op = None
        self.input_var_to_op = None

    # insert allreduce op to sync global information for global
    # gradient clip and amp
    def _insert_allreduce_op(self, op_idx, block):
        """
        Insert allreduce op to sync global information for global
        gradient clip and amp.
        """
        op = block.ops[op_idx]
        out_name = op.desc.output_arg_names()[0]
        out_var = block.var(out_name)
        offset = 0
        if op.type == "reduce_any":
            # cast the bool var to int32 to use allreduce_max op
            temp_var_name = unique_name.generate(out_name + "_cast_int32")
            temp_var = block.create_var(
                name=temp_var_name, shape=[1], dtype="int32")
            block._insert_op(
                op_idx + 1 + offset,
                type='cast',
                inputs={'X': out_var},
                outputs={'Out': temp_var},
                attrs={
                    'in_dtype': out_var.dtype,
                    'out_dtype': temp_var.dtype,
                    self._op_role_key: self._op_role.Optimize
                })
            offset += 1
        block._insert_op(
            op_idx + 1 + offset,
            type='c_allreduce_max'
            if op.type == "reduce_any" else 'c_allreduce_sum',
            inputs={'X': temp_var if op.type == "reduce_any" else out_var},
            outputs={'Out': temp_var if op.type == "reduce_any" else out_var},
            attrs={
                'ring_id': self.global_ring_id,
                self._op_role_key: self._op_role.Optimize,
                'use_calc_stream': True
            })
        offset += 1
        if op.type == "reduce_any":
            block._insert_op(
                op_idx + 1 + offset,
                type='cast',
                inputs={'X': temp_var},
                outputs={'Out': out_var},
                attrs={
                    'in_dtype': temp_var.dtype,
                    'out_dtype': out_var.dtype,
                    self._op_role_key: self._op_role.Optimize
                })
            offset += 1
        return offset

    def _create_vars(self, block, ori_block):
        # Create vars for block, copied from ori_block
        used_var_set = set()
        added_op_num = 0
        op_idx = 0
        op_size = block.desc.op_size()
        while op_idx < op_size + added_op_num:
            # Whether to insert allreduce_sum or allreduce_max op.
            # For amp and global gradient clip strategies, we should
            # get the global information, so allreduce op is needed.
            should_insert = False
            op = block.ops[op_idx]
            # For op process vars on all devices, remove its input 
            # vars not in this block
            reserved_x = []
            if op.type == 'reduce_any' and self._is_optimize_op(op):
                should_insert = True
            elif op.type == 'concat' and self._is_optimize_op(op):
                for input_name in op.desc.input("X"):
                    if block._find_var_recursive(input_name):
                        reserved_x.append(input_name)
                op.desc.set_input('X', reserved_x)
            elif op.type == 'update_loss_scaling':
                for input_name in op.desc.input("X"):
                    if block._find_var_recursive(input_name):
                        reserved_x.append(input_name)
                op.desc.set_input('X', reserved_x)
                op.desc.set_output('Out', reserved_x)
            elif op.type == 'check_finite_and_unscale':
                for input_name in op.desc.input("X"):
                    if block._find_var_recursive(input_name):
                        reserved_x.append(input_name)
                op.desc.set_input('X', reserved_x)
                op.desc.set_output('Out', reserved_x)
                if len(reserved_x) == 0:
                    block._remove_op(op_idx)
                    op_size -= 1
                    continue
            elif op.type == 'sum' and self._is_gradient_clip_op(op):
                for input_name in op.desc.input("X"):
                    if block._find_var_recursive(input_name):
                        reserved_x.append(input_name)
                op.desc.set_input('X', reserved_x)
                should_insert = True

            vars = op.desc.input_arg_names() + op.desc.output_arg_names()
            for var in vars:
                # a var whose name contains "blocking_queue" 
                # only exists in startup program 
                if var in used_var_set or "_blocking_queue" in var:
                    continue
                used_var_set.add(var)
                if block._find_var_recursive(str(var)): continue
                source_var = ori_block._var_recursive(str(var))
                if source_var.type == core.VarDesc.VarType.READER:
                    dest_var = block.create_var(
                        name=var,
                        type=core.VarDesc.VarType.READER,
                        persistable=source_var.persistable)
                else:
                    dest_var = block._clone_variable(source_var, False)
                self._clone_var_attr(dest_var, source_var)
            # When use with sharding, allreduce_sum and allreduce_max
            # used for global gradient clip and amp will be added by sharding.
            op_idx += 1
            if self.use_sharding or not should_insert: continue
            inserted_ops = self._insert_allreduce_op(op_idx - 1, block)
            added_op_num += inserted_ops
            op_idx += inserted_ops
        block._sync_with_cpp()

    def _is_loss_grad_op(self, op):
        assert self._op_role_key in op.attr_names
        op_role = int(op.attr(self._op_role_key))
        return op_role & int(self._op_role.Backward) and op_role & int(
            self._op_role.Loss)

    def _is_forward_op(self, op):
        return self._op_role_key in op.attr_names and (
            int(op.attr(self._op_role_key)) == int(self._op_role.Forward))

    def _is_backward_op(self, op):
        return self._op_role_key in op.attr_names and (
            int(op.attr(self._op_role_key)) & int(self._op_role.Backward))

    def _is_loss_op(self, op):
        assert self._op_role_key in op.attr_names
        return int(op.attr(self._op_role_key)) == int(self._op_role.Loss)

    def _is_optimize_op(self, op):
        return self._op_role_key in op.attr_names and (
            int(op.attr(self._op_role_key)) & int(self._op_role.Optimize))

    def _is_update_op(self, op):
        return 'Param' in op.input_names and 'Grad' in op.input_names and (
            "LearningRate" in op.input_names)

    def _split_program(self, main_program, devices):
        """
        Split a program into sections according to devices that ops run on.
        The op whose op_device attr is "gpu:all" is copied to all sections.

        Args:
            main_program (Program): the main program
            devices: all used devices
        """
        # Map from device to its corresponding section program info
        device_program_map = defaultdict(Program)

        block = main_program.block(0)
        for op in block.ops:
            device = op.attr(self._op_device_key)
            # Copy ops whose op_device set to "gpu:all" to all sections.
            if device == f"{self._device}:all":
                for device in devices:
                    program = device_program_map[device]
                    op_desc = op.desc
                    ap_op = program.global_block().desc.append_op()
                    ap_op.copy_from(op_desc)
                    ap_op._set_attr(self._op_device_key, "")
            else:
                program = device_program_map[device]
                op_desc = op.desc
                ap_op = program.global_block().desc.append_op()
                ap_op.copy_from(op_desc)
                ap_op._set_attr(self._op_device_key, "")

        program_list = []
        for key in devices:
            program = device_program_map[key]
            program._sync_with_cpp()
            program_list.append(program)

        return program_list

    def _get_op_device_for_startup_program(self, var_name):
        """
        For adam optimizer, it will add accumulators and initialize them
        with fill_constant, and force the op device to cpu. Hence, we should
        get the real op_device attribute of the fill_constant as the device
        where the corresponding parameters on.
        """
        assert "beta1_pow_acc" in var_name or "beta2_pow_acc" in var_name, \
            'For accumulators for Adam, the name must contain beta1_pow_acc ' \
            'or beta2_pow_acc.'
        param_name = var_name[0:var_name.index('_beta')]
        device = self._param_device_map[param_name]
        return device

    def _split_startup_program(self, startup_program, device_id):
        block = startup_program.global_block()
        new_startup_program = Program()
        for op in block.ops:
            device = op.attr(self._op_device_key)
            if device == "cpu":
                assert op.type == "fill_constant", (
                    "For ops in startup program with the op_device attribute "
                    "of cpu, they must be of type fill_constant.")
                output_var = op.output_arg_names[0]
                device = self._get_op_device_for_startup_program(output_var)

            if device:
                device_index = int(device.split(':')[1])
            else:
                # LR related ops
                device = None
            if device and device_index != device_id: continue
            op_desc = op.desc
            ap_op = new_startup_program.global_block().desc.append_op()
            ap_op.copy_from(op_desc)
            ap_op._set_attr(self._op_device_key, "")
        new_startup_program._sync_with_cpp()
        self._create_vars(new_startup_program.global_block(), block)
        return new_startup_program

    def _find_post_op(self, index, var_name):
        """
        Find the post op that has variable named var_name as input.
        """
        # bugfix for uniform hybrid parallelism
        if '.cast_fp32' in var_name:
            var_name = var_name.replace('.cast_fp32', '')
        if '.cast_fp16' in var_name:
            var_name = var_name.replace('.cast_fp16', '')

        post_ops = self.input_var_to_op[var_name]
        if post_ops == None: return None
        result_op = None
        for post_op, post_idx in reversed(post_ops):
            if post_idx > index:
                result_op = post_op
                break
        return result_op

    def _find_prev_op(self, index, var_name):
        """
        Find the previous op of op with index that outputs
        variable named var_name.
        """
        prev_ops = self.output_var_to_op[var_name]
        if prev_ops == None: return None
        result_op = None
        for prev_op, prev_idx in reversed(prev_ops):
            if prev_idx < index:
                result_op = prev_op
                break
        return result_op

    def _rename_arg(self, op, old_name, new_name):
        op._rename_input(old_name, new_name)
        op._rename_output(old_name, new_name)

    def _create_var(self, block, ref_var, name, dtype=None):
        """
        Create a new var for block, which has the same type,
        shape and dtype as ref_var, then rename it with the
        name `name`.
        """
        new_var = block.create_var(
            name=name,
            shape=ref_var.shape,
            dtype=ref_var.dtype if dtype is None else dtype,
            type=ref_var.type,
            lod_level=ref_var.lod_level,
            persistable=ref_var.persistable,
            is_data=ref_var.is_data,
            need_check_feed=ref_var.desc.need_check_feed())
        self._clone_var_attr(new_var, ref_var)
        return new_var

    def _clone_var_attr(self, dest, src):
        dest.stop_gradient = src.stop_gradient
        if hasattr(src, 'is_distributed'):
            dest.is_distributed = src.is_distributed

    def _strip_grad_suffix(self, name):
        """
        Strip the grad suffix from the given variable name
        """
        pos = name.find(core.grad_var_suffix())
        return name[:pos] if pos != -1 else name

    def _append_grad_suffix(self, name):
        """
        Append grad suffix to the given variable name
        """
        return name + core.grad_var_suffix()

    def _get_op_device_attr(self, op):
        """
        Get the op_device attribute of a op.
        """
        device = op.attr(self._op_device_key) \
            if op.has_attr(self._op_device_key) else None
        if device:
            assert device[0:3] == 'gpu' or device[0:3] == 'npu', "Now, only gpu and npu devices are " \
                "supported in pipeline parallemism."
        return device

    def _add_op_device_attr_for_op(self, op, idx, block):
        """
        Add op_device attrribute for ops that have not that attribute set.
        We use "gpu:all" to represent the op should be put on all
        sub-programs, such as lr-related ops. Note that: "gpu:all"
        is only used by pipeline as an indicator.
        """
        lrsched_role = int(self._op_role.LRSched)
        if op.attr(self._op_role_key) == lrsched_role:
            # For LRSched ops, we should put them on all sub-programs to
            # make sure each sub-program update the lr correctly
            op._set_attr(self._op_device_key, f"{self._device}:all")
        # bugfix in hybrid parallelism
        elif op.type == "sum" and self._is_backward_op(op):
            # For sum ops that compute the sum of @RENAMED@ vars
            for name in op.desc.input_arg_names():
                assert '@RENAME@' in name, \
                    "The op must be sum used to accumulate renamed vars."
            assert len(op.desc.output_arg_names()) == 1
            out_name = op.desc.output_arg_names()[0]
            post_op = self._find_post_op(idx, out_name)
            assert post_op.has_attr(
                'op_device'), "{} has no op_device attr for var {}".format(
                    post_op.type, out_name)
            device = post_op.attr(self._op_device_key)
            assert device, "The post op must have op_device set."
            op._set_attr(self._op_device_key, device)
        elif (op.type == "cast" or
              op.type == "scale") and self._is_backward_op(op):
            prev_op = self._find_prev_op(idx, op.desc.input("X")[0])
            op._set_attr(self._op_device_key, prev_op.attr(self._op_device_key))
        elif op.type == "memcpy" and not self._is_optimize_op(op):
            # for checkpoint offloading
            assert len(op.input_arg_names) == 1 and len(
                op.output_arg_names) == 1
            input_name = op.input_arg_names[0]
            output_name = op.output_arg_names[0]
            if '@Fetch' in output_name:
                post_op = self._find_post_op(idx, output_name)
                op._set_attr(self._op_device_key,
                             post_op.attr(self._op_device_key))
            else:
                prev_op = self._find_prev_op(idx, op.desc.input("X")[0])
                op._set_attr(self._op_device_key,
                             prev_op.attr(self._op_device_key))
        elif self._is_loss_op(op):
            # For loss * loss_scaling op added by AMP
            offset = 1
            while (not block.ops[idx + offset].has_attr(self._op_device_key) or
                   not block.ops[idx + offset].attr(self._op_device_key)):
                offset += 1
            device = block.ops[idx + offset].attr(self._op_device_key)
            assert device, "Please put you program within device_guard scope."
            for i in range(offset):
                block.ops[idx + i]._set_attr(self._op_device_key, device)
        elif self._is_optimize_op(op) and op.type == "cast":
            # For fp16-->fp32 cast added by AMP
            grad_name = op.output('Out')
            assert len(grad_name) == 1
            param_name = self._strip_grad_suffix(grad_name[0])
            device = self._param_device_map[param_name]
            op._set_attr(self._op_device_key, device)
        elif self._is_gradient_clip_op(op) or self._is_regularization_op(op):
            # For gradient clip and regularization ops, we set their op_device
            # attribute to the device where their corresponding parameters on.
            assert self._op_role_var_key in op.attr_names, "gradient_clip " \
                "and regularization ops must have op_role_var attribute."
            op_role_var = op.attr(self._op_role_var_key)
            assert len(op_role_var) == 2, "op_role_var for gradient_clip " \
                "regularization ops must have two elements."
            param_name = op_role_var[0]
            device = self._param_device_map[param_name]
            # For sum op added by global gradient clip, it must be 
            # put on all devices
            if (op.type == 'sum' or op.type == 'sqrt' or
                    op.type == 'fill_constant' or
                    op.type == 'elementwise_max' or
                    op.type == 'elementwise_div'):
                device = f"{self._device}:all"
            op._set_attr(self._op_device_key, device)
        elif op.type == "alloc_float_status" or op.type == "clear_float_status":
            op._set_attr(self._op_device_key, f"{self._device}:all")
            # NOTE(wangxi): NPU should only clear the float status
            # once at each batch step
            op._set_attr(self._op_role_key, self._op_role.LRSched)

            float_status_name = op.output_arg_names[0]
            float_status_var = block.var(float_status_name)
            # FIXME(wangxi): pipeline lr schedule will exec on sub_scope(0)
            # while update will exec on sub_scope(last_micro_step), should
            # set persistable to use global scope
            float_status_var.persistable = True
        else:
            other_known_ops = [
                'update_loss_scaling', 'reduce_any', 'concat', 'sum',
                'check_finite_and_unscale', 'memcpy'
            ]
            assert op.type in other_known_ops, "For other ops without " \
                "op_device set, they must be one of {}, but it " \
                "is {}".format(other_known_ops, op.type)
            assert self._is_optimize_op(op)
            op._set_attr(self._op_device_key, f"{self._device}:all")

    def _add_op_device_attr(self, block):
        """
        Add op_device attrribute for ops in block that have 
        not that attribute set.
        """
        for idx, op in enumerate(list(block.ops)):
            if (op.type == "create_py_reader" or op.type == "read" or
                    op.type == "create_double_buffer_reader"):
                # Copy read related ops to all section to make them exit 
                # after each epoch.
                # We use "gpu:all" to represent the op should be put on all
                # sub-programs, such as lr-related ops. Note that: "gpu:all"
                # is only used by pipeline as an indicator.
                op._set_attr(self._op_device_key, f"{self._device}:all")
                continue
            # op_device attribute has been set
            if self._get_op_device_attr(op): continue
            self._add_op_device_attr_for_op(op, idx, block)

    def _check_validation(self, block):
        """
        Check whether ops in a block have both the op_device and the 
        op_role attributes set.
        Then, return all devices in order.
        """
        device_list = []
        # Section worker only supports the following op_role
        valid_op_role_value = [
            int(self._op_role.LRSched),
            int(self._op_role.Forward),
            int(self._op_role.Backward),
            int(self._op_role.Loss),
            int(self._op_role.Optimize),
            int(self._op_role.Backward) | int(self._op_role.Loss),
        ]
        for op in block.ops:
            if not op._has_kernel(op.type):
                assert op.type == "conditional_block" and (
                    op.attr(self._op_role_key) == int(self._op_role.LRSched)), (
                        "Now, the only supported op without kernel is "
                        "conditional_block, and its op role must be LRSched.")
            assert op.has_attr(self._op_role_key), (
                "op ({}) has no {} attribute.".format(op.type,
                                                      self._op_role_key))
            op_role = op.attr(self._op_role_key)
            assert int(op_role) in valid_op_role_value, \
                "op_role {} for op {} must be one of {}".format(
                    op_role,
                    op.type,
                    valid_op_role_value)

            assert op.has_attr(self._op_device_key), (
                "op ({}) has no {} attribute.".format(op.type,
                                                      self._op_device_key))

            device = op.attr(self._op_device_key)
            assert device, ("op_device attribute for op "
                            "{} has not been set.".format(op.type))
            if device == f"{self._device}:all": continue

            dev_type = device.split(':')[0]
            assert dev_type == "gpu" or dev_type == 'npu', (
                "Now only gpu and npu devices are supported "
                "for pipeline parallelism.")

            if device not in device_list:
                device_list.append(device)

        return device_list

    def _insert_sendrecv_ops_for_boundaries(self, block):
        """
        Insert a pair of send and recv ops for every two
        consecutive ops on different devices.
        """
        # A map from var to device where op takes it as input,
        # avoiding multiple send and recv ops.
        input_var_to_device = dict()
        # bugfix hybrid parallelism
        first_optimize_index = None
        for index, op in enumerate(list(block.ops)):
            if self._is_optimize_op(op):
                first_optimize_index = index
                break
        extra_index_info = {
            'index': 0,
            'first_optimize_index': first_optimize_index
        }

        for index, op in enumerate(list(block.ops)):
            cur_device = op.attr(self._op_device_key)
            if cur_device == f"{self._device}:all": continue
            for var_name in op.input_arg_names:
                var = block.var(var_name)
                # skip data var
                if var.is_data: continue
                prev_device = None

                prev_op = self._find_prev_op(index, var_name)
                if prev_op is None:
                    if var_name not in self._param_device_map:
                        continue
                    prev_device = self._param_device_map[var_name]

                if not prev_device:
                    prev_device = prev_op.attr(self._op_device_key) \
                        if prev_op else None

                if prev_device is None or prev_device == f"{self._device}:all":
                    continue

                if prev_device == cur_device: continue

                if var_name not in input_var_to_device:
                    input_var_to_device[var_name] = []
                if (cur_device, prev_device) in input_var_to_device[var_name]:
                    continue

                device_type = cur_device.split(':')[0] + ':'

                def _check_stage(cur_id, prev_id):
                    # check send/recv stage valid
                    is_forward = self._is_forward_op(op)
                    is_backward = self._is_backward_op(op)
                    assert is_forward or is_backward, \
                        'send/recv in pipeline should only be inserted in forward or backward,' \
                        'please check the op_role of op={}'.format(op)

                    if is_forward:
                        assert prev_id < cur_id, \
                            "In forward, send/recv can only be passed forward, but now " \
                            "prev_stage={} great than cur_stage={}, please check op_device of op={}".format(
                                prev_id, cur_id, op)
                    elif is_backward:
                        assert prev_id > cur_id, \
                            "In backward, send/recv can only be passed backward, but now " \
                            "prev_stage={} less than cur_stage={}, please check op_device of op={}".format(
                                prev_id, cur_id, op)

                def _insert_send_recv(cur_id, prev_id):
                    cur_dev = device_type + str(cur_id)
                    prev_dev = device_type + str(prev_id)
                    if (cur_dev, prev_dev) in input_var_to_device[var_name]:
                        return

                    if cur_id - prev_id > 1:
                        _insert_send_recv(cur_id - 1, prev_id)
                        _insert_send_recv(cur_id, cur_id - 1)
                        input_var_to_device[var_name].append(
                            (cur_dev, prev_dev))
                        return
                    elif cur_id - prev_id < -1:
                        _insert_send_recv(cur_id + 1, prev_id)
                        _insert_send_recv(cur_id, cur_id + 1)
                        input_var_to_device[var_name].append(
                            (cur_dev, prev_dev))
                        return

                    assert abs(cur_id - prev_id) == 1
                    input_var_to_device[var_name].append((cur_dev, prev_dev))

                    op_role = op.attr(self._op_role_key)
                    var = block.vars[var_name]
                    pair = (prev_id, cur_id)
                    # 1000 is just a magic number
                    pair_key = prev_id * 1000 + cur_id
                    if pair not in self._pipeline_pair:
                        self._pipeline_pair.append(pair)
                        self._pp_ring_map[pair_key] = self.ring_id
                        ring_id = self.ring_id
                        self.ring_id += 1
                    else:
                        ring_id = self._pp_ring_map[pair_key]

                    if self.schedule_mode == 'F-then-B':  # F-then-B
                        block._insert_op_without_sync(
                            index=index + extra_index_info['index'],
                            type='send_v2',
                            inputs={'X': var},
                            attrs={
                                self._op_device_key: prev_dev,
                                self._op_role_key: op_role,
                                'use_calc_stream': True,
                                'peer': 1,
                                'ring_id': ring_id
                            })
                        extra_index_info['index'] += 1
                        var_shape = list(var.shape)
                        var_shape[0] = self.micro_batch_size if var_shape[
                            0] < 0 else var_shape[0]
                        block._insert_op_without_sync(
                            index=index + extra_index_info['index'],
                            type='recv_v2',
                            outputs={'Out': [var]},
                            attrs={
                                'out_shape': var_shape,
                                'dtype': var.dtype,
                                self._op_device_key: cur_dev,
                                self._op_role_key: op_role,
                                'use_calc_stream': True,
                                'peer': 0,
                                'ring_id': ring_id
                            })
                        extra_index_info['index'] += 1
                    elif self.schedule_mode == '1F1B':  # 1F1B
                        var_shape = list(var.shape)
                        var_shape[0] = self.micro_batch_size if var_shape[
                            0] < 0 else var_shape[0]

                        numel = np.prod(var_shape)
                        use_mp = (self.mp_degree > 1) and (
                            numel % self.mp_degree == 0)

                        if 'subprog' in var.name:
                            # For recompute, if the checkpoints var is layer_norm_6.tmp_2
                            # this var will be sent twice, layer_norm_6.tmp_2 for forward pass,
                            # layer_norm_6.tmp_2.subprog_* for recompute pass.
                            # We can store the first sent var and copy the value to the
                            # second one to reduce one send/recv op.
                            # The origin_ckpt_name is layer_norm_6.tmp_2, which will be used
                            # to find the stored var for the forward pass.
                            origin_name = var.name.split('subprog')[0][0:-1]
                            associate_var = block.var(origin_name)
                            block._insert_op_without_sync(
                                index=index + extra_index_info['index'],
                                type='assign',
                                inputs={'X': [associate_var]},
                                outputs={'Out': [var]},
                                attrs={
                                    'out_shape': var_shape,
                                    'dtype': var.dtype,
                                    self._op_device_key: cur_dev,
                                    self._op_role_key: op_role,
                                    'use_calc_stream': True,
                                })
                            extra_index_info['index'] += 1
                            return

                        _check_stage(cur_id, prev_id)

                        block._insert_op_without_sync(
                            index=index + extra_index_info['index'],
                            type='c_sync_calc_stream',
                            inputs={'X': [var]},
                            outputs={'Out': [var]},
                            attrs={
                                self._op_device_key: prev_dev,
                                self._op_role_key: op_role,
                            })
                        extra_index_info['index'] += 1
                        prefix_name = var.name.split('@')[0]
                        prefix_var = block.var(prefix_name)
                        is_param = True if isinstance(prefix_var,
                                                      Parameter) else False
                        block._insert_op_without_sync(
                            index=index + extra_index_info['index'],
                            type='send_v2'
                            if not use_mp or is_param else 'partial_send',
                            inputs={'X': var},
                            attrs={
                                self._op_device_key: prev_dev,
                                self._op_role_key: op_role,
                                'use_calc_stream': False,
                                'ring_id': ring_id,
                                'peer': 1,
                                # if send_v2, num&id attr is not in op_attrs, will not insert
                                'num': self.mp_degree,
                                'id': self.mp_rank,
                            })
                        extra_index_info['index'] += 1
                        insert_index = None
                        if int(op_role) == int(self._op_role.Backward):
                            insert_index = extra_index_info[
                                'first_optimize_index']
                            new_op_role = self._op_role.Optimize
                        else:
                            insert_index = index
                            new_op_role = self._op_role.Backward
                        sync_comm_op = block._insert_op_without_sync(
                            index=insert_index + extra_index_info['index'],
                            type='c_sync_comm_stream',
                            inputs={'X': [var]},
                            outputs={'Out': [var]},
                            attrs={
                                self._op_device_key: prev_dev,
                                self._op_role_key: new_op_role,
                                'ring_id': ring_id,
                            })
                        if int(op_role) == int(self._op_role.Forward):
                            sync_comm_op._set_attr('pipeline_flag', '')
                            extra_index_info['index'] += 1
                        block._insert_op_without_sync(
                            index=index + extra_index_info['index'],
                            type='recv_v2'
                            if not use_mp or is_param else 'partial_recv',
                            outputs={'Out': [var]},
                            attrs={
                                'out_shape': var_shape,
                                'dtype': var.dtype,
                                self._op_device_key: cur_dev,
                                self._op_role_key: op_role,
                                'use_calc_stream': True,
                                'peer': 0,
                                'ring_id': ring_id,
                                # if recv_v2, num&id attr is not in op_attrs, will not insert
                                'num': self.mp_degree,
                                'id': self.mp_rank,
                            })
                        extra_index_info['index'] += 1
                        if use_mp and not is_param:
                            block._insert_op_without_sync(
                                index=index + extra_index_info['index'],
                                type='partial_allgather',
                                inputs={'X': [var]},
                                outputs={'Out': [var]},
                                attrs={
                                    self._op_device_key: cur_dev,
                                    self._op_role_key: op_role,
                                    'use_calc_stream': True,
                                    'ring_id': 0,
                                    # if recv_v2, num&id attr is not in op_attrs, will not insert
                                    'nranks': self.mp_degree,
                                    'rank': self.mp_rank,
                                })
                            extra_index_info['index'] += 1
                    else:
                        raise ValueError(
                            "Now only 'F-then-B' and '1F1B' are supported."
                            "The given value is {}.".format(self.schedule_mode))

                _insert_send_recv(
                    int(cur_device.split(':')[1]),
                    int(prev_device.split(':')[1]))
        block._sync_with_cpp()

    def _insert_loss_scale(self, block):
        """
        Scale the loss corresponding to number of micro-batches.
        """
        if self._num_microbatches == 1: return
        for index, op in reversed(tuple(enumerate(list(block.ops)))):
            if self._is_loss_grad_op(op):
                assert op.type == 'fill_constant', \
                    "loss_grad_op must be fill_constant op, " \
                    "but this op is {}".format(op.type)
                assert op.has_attr('value')
                loss_scale = float(op.attr('value'))
                loss_scale = loss_scale / self._num_microbatches
                op._set_attr('value', loss_scale)
                break

    def _rename_gradient_var_name(self, block):
        for index, op in enumerate(block.ops):
            if not self._is_optimize_op(op): continue
            input_names = op.input_arg_names
            output_names = op.output_arg_names
            in_out_names = input_names + output_names
            if op.type == 'cast' or op.type == "c_sync_comm_stream": continue
            # append "MERGED" to the names of parameter gradients,
            # and mofify the op_role_var attribute (by rename_arg func).
            for name in in_out_names:
                if not core.grad_var_suffix() in name: continue
                param_name = name.strip(core.grad_var_suffix())
                new_grad_name = name + "@MERGED"
                self._rename_arg(op, name, new_grad_name)

    def _accumulate_gradients(self,
                              block,
                              pp_allreduce_in_optimize=False,
                              strategy=None,
                              shard=None):
        """
        Create a new merged gradient for each parameter and accumulate the
        corresponding gradient to it.
        """
        fp16_allreduce = strategy.fp16_allreduce if strategy else False
        if strategy and strategy.fuse_grad_merge:
            fused_gradient_names = self._accumulate_gradients_with_fuse(
                block, fp16_allreduce, strategy.fuse_grad_size_in_MB, shard)
            return fused_gradient_names

        merged_gradient_names = []
        first_opt_op_idx = None

        merged_suffix = '@MERGED@FP16' if fp16_allreduce else '@MERGED'
        dtype = paddle.float16 if fp16_allreduce else None

        for index, op in reversed(tuple(enumerate(list(block.ops)))):
            # remove the cast op of fp16 grad to fp32 grad
            if self._is_optimize_op(op) and op.type == 'cast':
                in_name = op.input_arg_names[0]
                out_name = op.output_arg_names[0]
                if out_name.strip('@GRAD') in self._param_device_map:
                    assert in_name.replace('.cast_fp16', '') == out_name
                    block._remove_op(index)
                    continue

            if self._is_backward_op(op) and first_opt_op_idx is None:
                first_opt_op_idx = index + 1
                # maybe have no optimize
                # if first_opt_op_idx == len(block.ops): return

            if self._is_backward_op(op) and (
                    self._op_role_var_key in op.attr_names):
                op_role_var = op.attr(self._op_role_var_key)
                if len(op_role_var) == 0: continue
                assert len(op_role_var) % 2 == 0
                for i in range(0, len(op_role_var), 2):
                    offset = 0
                    param_name = op_role_var[i]
                    if not block.has_var(param_name): continue
                    if '@BroadCast' in param_name: continue

                    param_grad_name = param_name + core.grad_var_suffix()
                    merged_param_grad_name = param_grad_name + merged_suffix
                    if not block.has_var(merged_param_grad_name):
                        self._create_var(block, block.vars[param_name],
                                         merged_param_grad_name, dtype)
                    assert block.has_var(merged_param_grad_name)

                    param_grad_var = block.var(param_grad_name)
                    merged_param_grad_var = block.var(merged_param_grad_name)
                    merged_param_grad_var.persistable = True
                    block._insert_op(
                        index=first_opt_op_idx + offset,
                        type='fill_constant',
                        inputs={},
                        outputs={'Out': [merged_param_grad_var]},
                        attrs={
                            'shape': merged_param_grad_var.shape,
                            'dtype': merged_param_grad_var.dtype,
                            'value': float(0),
                            # a trick to run this op once per mini-batch
                            self._op_role_key: self._op_role.Optimize.LRSched,
                        })
                    offset += 1
                    grad_name = op_role_var[i + 1]
                    grad_var = block.vars[grad_name]

                    is_fp16_grad = 'cast_fp16' in grad_name
                    need_cast = (is_fp16_grad is not fp16_allreduce)

                    if need_cast:
                        # if fp16_allreduce:
                        #     cast grad to fp16 to accumulate to merged gradient
                        # else:
                        #     cast grad to fp32 to accumulate to merged gradient
                        cast_grad_var_name = param_grad_name + '@TMP'
                        cast_grad_var = self._create_var(
                            block, param_grad_var, cast_grad_var_name, dtype)
                        cast_grad_var.persistable = False
                        block._insert_op(
                            index=first_opt_op_idx + offset,
                            type='cast',
                            inputs={'X': grad_var},
                            outputs={'Out': cast_grad_var},
                            attrs={
                                'in_dtype': grad_var.dtype,
                                'out_dtype': cast_grad_var.dtype,
                                self._op_role_key: self._op_role.Backward,
                            })
                        offset += 1
                        grad_var = cast_grad_var

                    block._insert_op(
                        index=first_opt_op_idx + offset,
                        type='sum',
                        inputs={'X': [merged_param_grad_var, grad_var]},
                        outputs={'Out': merged_param_grad_var},
                        attrs={self._op_role_key: self._op_role.Backward, })
                    offset += 1
                    merged_gradient_names.append(merged_param_grad_name)

        if not fp16_allreduce: return merged_gradient_names

        first_opt_op_idx = None
        for index, op in reversed(tuple(enumerate(list(block.ops)))):
            if self._is_backward_op(op) and first_opt_op_idx is None:
                first_opt_op_idx = index + 1
                break
        assert first_opt_op_idx is not None

        # insert cast op from fp16->fp32
        # FIXME(wangxi): maybe put in sharding is better, for some grad
        #                is not in sharding device.
        for fp16_grad_name in merged_gradient_names:
            grad_name = fp16_grad_name.replace('@FP16', '')
            param_name = fp16_grad_name.replace('@GRAD@MERGED@FP16', '')

            if not block.has_var(grad_name):
                self._create_var(block, block.vars[param_name], grad_name)
            assert block.has_var(grad_name)

            fp16_grad_var = block.var(fp16_grad_name)
            grad_var = block.var(grad_name)
            grad_var.persistable = False

            block._insert_op(
                index=first_opt_op_idx,
                type='cast',
                inputs={'X': fp16_grad_var},
                outputs={'Out': grad_var},
                attrs={
                    'in_dtype': fp16_grad_var.dtype,
                    'out_dtype': grad_var.dtype,
                    self._op_role_key: self._op_role.Optimize,
                })

        return merged_gradient_names

    def _insert_accumulate_gradients_with_fuse(self, main_block, fp16,
                                               fused_size, grad_param_pairs,
                                               first_opt_op_idx):
        grad_param_pairs = self._sort_grad_param_by_dtype(main_block,
                                                          grad_param_pairs)

        grad_param_segments = []
        merged_suffix = '@MERGED@FP16' if fp16 else '@MERGED'
        dtype = paddle.float16 if fp16 else paddle.float32
        cur_size = 0.
        last_dtype = None
        # split the grad based on dtype and fused size
        for grad, param in grad_param_pairs:
            real_grad = main_block.var(grad)
            # create the gradient merged var for each grad
            merged_grad_var = main_block.create_var(
                name=param + core.grad_var_suffix() + merged_suffix,
                dtype=dtype,
                shape=real_grad.shape,
                persistable=True,
                stop_gradient=False)
            real_param = main_block.var(param)
            if hasattr(real_param, 'is_distributed'):
                merged_grad_var.is_distributed = real_param.is_distributed
            tmp_size = self._get_var_size(real_grad)
            # two strategies for splitting the grad
            # 1. the current segment's size reach the user defined grad_size_in_MB
            # 2. the upcoming grad holds different dtype compared with grads in current segment
            if len(grad_param_segments) == 0 \
                    or cur_size + tmp_size > fused_size \
                    or real_grad.dtype != last_dtype:
                grad_param_segments.append(
                    ([real_grad], [real_param], [merged_grad_var]))
                last_dtype = real_grad.dtype
                cur_size = 0.
            else:
                grad_param_segments[-1][0].append(real_grad)
                grad_param_segments[-1][1].append(real_param)
                grad_param_segments[-1][2].append(merged_grad_var)
                cur_size += tmp_size

        fused_gradients = []
        fused_merged_gradients = []
        # create fused vars for grad and param
        for grad_param_segment in grad_param_segments:
            grad_segment = grad_param_segment[0]
            merged_grad_segment = grad_param_segment[2]
            fused_grad = main_block.create_var(
                name='FusedGrad_{}'.format(grad_segment[0].name),
                dtype=grad_segment[0].dtype,
                persistable=False,
                stop_gradient=False)
            # keep the '.cast_fp16' info in the fuse var name
            fused_merged_grad_name_prefix = 'FusedMergedGrad.cast_fp16.' if \
                merged_grad_segment[0].dtype == paddle.float16 else 'FusedMergedGrad'
            fused_merged_grad_name = fused_merged_grad_name_prefix + '_{}'.format(
                merged_grad_segment[0].name)
            fused_merged_grad = main_block.create_var(
                name=fused_merged_grad_name,
                dtype=merged_grad_segment[0].dtype,
                persistable=True,
                stop_gradient=False)
            fused_gradients.append(fused_grad)
            fused_merged_gradients.append(fused_merged_grad)

        assert len(fused_gradients) == len(grad_param_segments)
        assert len(fused_merged_gradients) == len(grad_param_segments)

        # insert coalesce op at the start of the backward pass
        # use param as the coalesce input to make sure the two Fused vars are in same shape
        first_back_op_idx = None
        for index, op in enumerate(main_block.ops):
            if self._is_backward_op(op) and first_back_op_idx is None:
                first_back_op_idx = index
                break
        assert first_back_op_idx is not None
        offset = 0
        for i in range(len(grad_param_segments)):
            fused_grad = fused_gradients[i]
            fused_merged_grad = fused_merged_gradients[i]
            grads = grad_param_segments[i][0]
            params = grad_param_segments[i][1]
            merged_grads = grad_param_segments[i][2]
            main_block._insert_op_without_sync(
                first_back_op_idx + offset,
                type="coalesce_tensor",
                inputs={"Input": params},
                outputs={"Output": grads,
                         "FusedOutput": fused_grad},
                attrs={
                    # Explanation of user_defined_size_of_dtype:
                    # In coalesce op, the align size is 256 bytes
                    # the float takes 4 bytes while fp16 takes 2 bytes.
                    # To meet the requirement, 128 fp16 or 64 float will be aligned
                    # Think the total shape of the input tensors if [64],
                    # if the dtype is float, then the shape of the fuse var is [64]
                    # however if the dytpe if fp16, the shape of the fuse var is [128],
                    # which will cause the fused vars' shape vary between each other.
                    # To make sure the shape of the fused vars are identical,
                    # we set the dtype of float and fp16 both to 2.
                    # Under this way, the fused vars' shape for float and fp16 are all [128]
                    "user_defined_size_of_dtype": 2,
                    "copy_data": False,
                    "use_align": True,
                    "dtype": grads[0].dtype,
                    self._op_role_key: self._op_role.Backward,
                    # On npu, the nan/inf check login is different with gpu.
                    # If there are some not initialized sections in the fused var,
                    # and the value in those sections are nan/inf, it will trigger the nan/inf check.
                    # To avoid these problematic triggers, set constant is needed for npu
                    "set_constant": core.is_compiled_with_npu(),
                    "constant": float(0.0),
                })
            offset += 1
            # For the gradient_merged_fused_var, given a init value during the coalesce op
            # this will remove a problematic fill_constant op. This op role of this coalesce
            # is set to be LRSched to make this coalesce (with init) only run once
            main_block._insert_op_without_sync(
                first_back_op_idx + offset,
                type="coalesce_tensor",
                inputs={"Input": params},
                outputs={
                    "Output": merged_grads,
                    "FusedOutput": fused_merged_grad
                },
                attrs={
                    "user_defined_size_of_dtype": 2,
                    "set_constant": True,
                    "constant": float(0.0),
                    "copy_data": False,
                    "use_align": True,
                    "dtype": merged_grads[0].dtype,
                    self._op_role_key: self._op_role.Optimize.LRSched
                })
            offset += 1

        # insert gradient merge relating ops
        first_opt_op_idx += offset
        offset = 0
        for i in range(len(fused_gradients)):
            fused_grad = fused_gradients[i]
            fused_merged_grad = fused_merged_gradients[i]
            is_fp16_grad = 'cast_fp16' in fused_grad.name
            need_cast = (is_fp16_grad is not fp16)
            if need_cast:
                # for fp16 allreduce, cast fp32 grad to fp16
                # for fp32 allreduce, cast fp16 grad to fp32
                cast_grad_var_name = fused_grad.name + '@TMP'
                cast_grad_var = main_block.create_var(
                    name=cast_grad_var_name,
                    dtype=dtype,
                    persistable=False,
                    stop_gradient=False)
                main_block._insert_op(
                    index=first_opt_op_idx + offset,
                    type='cast',
                    inputs={'X': fused_grad},
                    outputs={'Out': cast_grad_var},
                    attrs={
                        'in_dtype': fused_grad.dtype,
                        'out_dtype': cast_grad_var.dtype,
                        self._op_role_key: self._op_role.Backward,
                    })
                offset += 1
                fused_grad = cast_grad_var
            main_block._insert_op(
                index=first_opt_op_idx + offset,
                type='sum',
                inputs={'X': [fused_merged_grad, fused_grad]},
                outputs={'Out': fused_merged_grad},
                attrs={self._op_role_key: self._op_role.Backward})
            offset += 1

        if fp16:
            # if using fp16 allreduce, the optimizer needs fp32 grads, cast them back to fp32
            for grad, param in grad_param_pairs:
                real_grad = main_block.var(grad)
                fp16_grad_name = param + core.grad_var_suffix() + '@MERGED@FP16'
                assert main_block.has_var(fp16_grad_name)
                fp16_grad = main_block.var(fp16_grad_name)
                fp32_grad_name = param + core.grad_var_suffix() + '@MERGED'
                fp32_grad = main_block.create_var(
                    name=fp32_grad_name,
                    dtype=paddle.float32,
                    shape=real_grad.shape,
                    persistable=False,
                    stop_gradient=False)
                main_block._insert_op(
                    index=first_opt_op_idx + offset,
                    type='cast',
                    inputs={'X': fp16_grad},
                    outputs={'Out': fp32_grad},
                    attrs={
                        'in_dtype': paddle.float16,
                        'out_dtype': paddle.float32,
                        self._op_role_key: self._op_role.Optimize,
                    })
                offset += 1

        # replace the var with it's name, which will be used for inserting allreduce
        for i in range(len(fused_merged_gradients)):
            fused_merged_gradients[i] = fused_merged_gradients[i].name

        return fused_merged_gradients, first_opt_op_idx

    def _accumulate_gradients_with_fuse(self,
                                        main_block,
                                        fp16,
                                        fused_size,
                                        shard=None):
        first_opt_op_idx = None
        grad_param_pairs = []
        # obtain all param/grad pairs that needed to be fused
        for index, op in reversed(tuple(enumerate(list(main_block.ops)))):
            # remove the cast op of fp16 grad to fp32 grad
            if self._is_optimize_op(op) and op.type == 'cast':
                in_name = op.input_arg_names[0]
                out_name = op.output_arg_names[0]
                if out_name.strip('@GRAD') in self._param_device_map:
                    assert in_name.replace('.cast_fp16', '') == out_name
                    main_block._remove_op(index)
                    continue

            if self._is_backward_op(op) and first_opt_op_idx is None:
                first_opt_op_idx = index + 1
                # no optimize phase
                if first_opt_op_idx == len(main_block.ops):
                    return

            if self._is_backward_op(op) and (
                    self._op_role_var_key in op.attr_names):
                op_role_var = op.attr(self._op_role_var_key)
                if len(op_role_var) == 0:
                    continue
                assert len(op_role_var) % 2 == 0
                for i in range(0, len(op_role_var), 2):
                    param_name = op_role_var[i]
                    if not main_block.has_var(param_name):
                        continue
                    if '@BroadCast' in param_name:
                        continue
                    grad_param_pairs.append(
                        (op_role_var[i + 1], op_role_var[i]))

        if len(grad_param_pairs) == 0:
            return

        nranks = shard.worker_num if shard else 1
        device_to_pairs = [[] for _ in range(nranks)]
        for pair in grad_param_pairs:
            root_id = shard.device(pair[1]) if shard else 0
            assert 0 <= root_id < nranks
            device_to_pairs[root_id].append(pair)

        all_fused_merged_gradients = []
        for pairs in device_to_pairs:
            fused_merged_gradients, first_opt_op_idx = \
                self._insert_accumulate_gradients_with_fuse(
                    main_block, fp16, fused_size, pairs, first_opt_op_idx)
            all_fused_merged_gradients += fused_merged_gradients

        main_block._sync_with_cpp()
        return all_fused_merged_gradients

    def _sort_grad_param_by_dtype(self, main_block, grad_param_pairs):
        # sort the grad param paris by the dtype
        fp16_pairs = []
        fp32_pairs = []
        other_pairs = []
        for pairs in grad_param_pairs:
            dtype = main_block.var(pairs[0]).dtype
            if dtype == paddle.float32:
                fp32_pairs.append(pairs)
            elif dtype == paddle.float16:
                fp16_pairs.append(pairs)
            else:
                other_pairs.append(pairs)
        sorted_pairs = fp16_pairs
        sorted_pairs.extend(fp32_pairs)
        sorted_pairs.extend(other_pairs)
        return sorted_pairs

    def _get_var_size(self, var):
        dtype_to_size = {
            core.VarDesc.VarType.FP16: 2,
            core.VarDesc.VarType.FP32: 4,
            core.VarDesc.VarType.FP64: 8,
            core.VarDesc.VarType.INT16: 2,
            core.VarDesc.VarType.INT32: 4,
            core.VarDesc.VarType.INT64: 8,
            core.VarDesc.VarType.BOOL: 1,
            core.VarDesc.VarType.UINT8: 1,
        }
        assert -1 not in var.shape
        return reduce(lambda x, y: x * y,
                      var.shape) * dtype_to_size[var.dtype] / 1024.0 / 1024.0

    def _add_sub_blocks(self, main_block, program_list):
        main_program = main_block.program
        for prog in program_list:
            for op in prog.block(0).ops:
                if not op.has_attr('sub_block'):
                    continue
                origin_sub_block_id = op.attr('sub_block').id
                origin_sub_block = main_program.block(origin_sub_block_id)
                new_sub_block = prog._create_block(parent_idx=0)
                for sub_op in origin_sub_block.ops:
                    op_desc = sub_op.desc
                    ap_op = new_sub_block.desc.append_op()
                    ap_op.copy_from(op_desc)
                new_sub_block._sync_with_cpp()
                self._create_vars(new_sub_block, origin_sub_block)
                op._set_attr('sub_block', new_sub_block)

    def _get_device_info(self, block):
        for op in block.ops:
            if not op._has_kernel(op.type): continue
            op_device = op.attr(self._op_device_key)
            return op_device

    def _process_persistable_vars_in_multi_sections(self, main_program,
                                                    startup_prog, program_list):
        """
        Special Case: process persistable vars that exist in
        multiple sections, e.g., shared weight
        """
        # var_info = {var_name: [program1, program2...]},
        # persistable var only
        var_info = dict()
        for prog in program_list:
            block = prog.block(0)
            for var_name in block.vars:
                if var_name == "double_buffer_0": continue
                var = block.var(var_name)
                if not var.persistable: continue
                if not var_name in var_info:
                    var_info[var_name] = []
                if not prog in var_info[var_name]:
                    var_info[var_name].append(prog)
        for var_name in list(var_info.keys()):
            if len(var_info[var_name]) == 1:
                var_info.pop(var_name)

        # write_info = {var_name: program}, where program is the only program
        # in which the var named var_name is written.
        write_info = dict()
        for var_name in var_info.keys():
            for prog in var_info[var_name]:
                block = prog.block(0)
                for op in block.ops:
                    if op.type == "recv_v2" or op.type == "create_py_reader" or \
                        op.type == "read" or op.type == "update_loss_scaling":
                        continue
                    # We have processed lr related vars
                    if op.attr(self._op_role_key) == int(
                            self._op_role.Optimize.LRSched):
                        continue
                    if var_name in op.desc.output_arg_names():
                        assert var_name not in write_info, (
                            "two sections write the same var({}): second "
                            "op {}.".format(var_name, op))
                        write_info[var_name] = prog
                        break

        for var_name in var_info.keys():
            # Case 1: read only variables, no special process
            if not var_name in write_info: continue

            # Case 2: one write multiple reads
            write_prog = write_info[var_name]
            write_block = write_prog.block(0)
            write_device = self._get_device_info(write_block)
            write_dev_index = int(write_device.split(':')[1])
            all_progs = var_info[var_name]
            for prog in all_progs:
                if prog == write_prog: continue
                read_block = prog.block(0)
                read_device = self._get_device_info(read_block)
                read_dev_index = int(read_device.split(':')[1])
                pair = (write_dev_index, read_dev_index)
                pair_key = write_dev_index * 1000 + read_dev_index
                if pair not in self._pipeline_pair:
                    self._pipeline_pair.append(pair)
                    self._pp_ring_map[pair_key] = self.ring_id
                    ring_id = self.ring_id
                    self.ring_id += 1
                else:
                    ring_id = self._pp_ring_map[pair_key]

                write_block._insert_op(
                    index=0,
                    type='send_v2',
                    inputs={'X': write_block.var(var_name), },
                    attrs={
                        self._op_device_key: write_device,
                        'use_calc_stream': False,
                        # A trick to make the role LRSched to avoid copy every
                        # microbatch
                        self._op_role_key: self._op_role.LRSched,
                        'peer': read_dev_index,
                        'ring_id': ring_id
                    })
                read_block._insert_op(
                    index=0,
                    type='recv_v2',
                    outputs={'Out': [read_block.var(var_name)]},
                    attrs={
                        'out_shape': read_block.var(var_name).shape,
                        'dtype': read_block.var(var_name).dtype,
                        self._op_device_key: read_device,
                        'use_calc_stream': False,
                        # A trick to make the role LRSched to avoid copy every
                        # microbatch
                        self._op_role_key: self._op_role.LRSched,
                        'peer': write_dev_index,
                        'ring_id': ring_id
                    })
                read_block._insert_op(
                    index=1,
                    type='c_sync_comm_stream',
                    inputs={'X': [read_block.var(var_name)]},
                    outputs={'Out': [read_block.var(var_name)]},
                    attrs={
                        self._op_device_key: read_device,
                        # A trick to make the role LRSched to avoid copy every
                        # microbatch
                        self._op_role_key: self._op_role.LRSched,
                        'ring_id': ring_id
                    })

    def _is_gradient_clip_op(self, op):
        return op.desc.has_attr("op_namescope") \
            and op.desc.attr("op_namescope").startswith("/gradient_clip")

    def _is_regularization_op(self, op):
        return op.desc.has_attr("op_namescope") \
            and op.desc.attr("op_namescope").startswith("/regularization")

    def _is_weight_decay_op(self, op):
        # in AdamW namescope is /optimizer_*/weight decay/
        return op.desc.has_attr("op_namescope") \
            and 'weight decay' in op.desc.attr("op_namescope")

    def _get_input_output_info(self, block):
        '''
        Get info of op input and output.
        '''
        # A map from output var to op which generate it.
        output_var_to_op = defaultdict(list)
        # A map from var to op which takes it as input.
        input_var_to_op = defaultdict(list)

        for index, op in enumerate(block.ops):
            for var_name in op.input_arg_names:
                input_var_to_op[var_name].append([op, index])
            for var_name in op.output_arg_names:
                output_var_to_op[var_name].append([op, index])

        return output_var_to_op, input_var_to_op

    def _optimize_forward_send_sync(self, program):
        """
        optimize forward send's sync_comm_stream schedule
        """
        if self.schedule_mode != '1F1B': return

        block = program.block(0)

        recv_type = 'recv_v2' if self.mp_degree == 1 else 'partial_recv'
        backward_recv_index = None
        for index, op in enumerate(block.ops):
            if op.type == recv_type and self._is_backward_op(op):
                backward_recv_index = index
                break

        # last pipeline stage
        if backward_recv_index is None: return

        offset = 0
        for index, op in enumerate(list(block.ops)):
            if index >= backward_recv_index: break
            if op.type == 'c_sync_comm_stream' and op.has_attr('pipeline_flag'):
                var_name = op.input_arg_names[0]
                var = block.var(var_name)
                block._remove_op(index + offset, sync=False)
                offset -= 1
                # NOTE:
                # 1. When the backward recv is completed, it indicates
                # that the forward send is completed too. So we only need
                # to use the NOP op to prevent memory release.
                # 2. Because we removed sync_comm_op,
                # we will insert NOP after recv_op.
                block._insert_op_without_sync(
                    index=backward_recv_index,
                    type='nop',
                    inputs={'X': [var]},
                    outputs={'Out': [var]},
                    attrs={self._op_role_key: self._op_role.Backward})
        block._sync_with_cpp()

    def _mv_head_recv(self, program):
        """
        A pass to move the recv op to the beginning of
        the forward/backward phase
        """
        forward_insert_index = 0
        backward_insert_index = None
        block = program.global_block()
        num_ops = len(program.global_block().ops)
        for i in range(num_ops):
            insert_index = None
            op = program.global_block().ops[i]
            op_role = int(op.attr(self._op_role_key))
            if op_role == int(
                    self._op_role.Backward) and backward_insert_index is None:
                backward_insert_index = i
            if op.type != "partial_recv" and op.type != "partial_allgather" and op.type != "nop" and op.type != "recv_v2":
                continue
            if op_role == int(self._op_role.Forward):
                if i == forward_insert_index:
                    forward_insert_index += 1
                    continue
                insert_index = forward_insert_index
            elif op_role == int(self._op_role.Backward):
                if i == backward_insert_index:
                    backward_insert_index += 1
                    continue
                insert_index = backward_insert_index
            else:
                raise ValueError("Unknown op_role: {}".format(op_role))
            op_inputs = dict()
            for name in op.input_names:
                op_inputs[name] = op.input(name)
            op_outputs = dict()
            for name in op.output_names:
                op_outputs[name] = op.output(name)
            block._insert_op_without_sync(
                index=insert_index,
                type=op.type,
                inputs=op_inputs,
                outputs=op_outputs,
                attrs=op.all_attrs())
            block._remove_op(i + 1)
            if op_role == int(self._op_role.Forward):
                forward_insert_index += 1
            elif op_role == int(self._op_role.Backward):
                backward_insert_index += 1
        block._sync_with_cpp()

    def _check_pipeline_persist_var(self, program):
        """
        Pipeline may need multiple forward before
        """
        block = program.global_block()

        persist_output = set()
        used_in_backward = set()
        for op in block.ops:
            if self._is_forward_op(op):
                for var_name in op.output_arg_names:
                    var = block.vars[var_name]
                    if var.persistable:
                        persist_output.add(var_name)
            elif self._is_backward_op(op):
                for var_name in op.input_arg_names:
                    if var_name in persist_output:
                        used_in_backward.add(var_name)
        if len(used_in_backward) == 0:
            return
        warnings.warn(
            "The pipeline requires multiple forward calculations before backward, "
            "so when the persistable var is changed in the forward, it may cause "
            "errors in the backward calculation who using this persistable var. "
            "However, some backward op don't need this var(NoNeedBufferVars), "
            "there will be no error at this time.\n"
            "So please check these persistable vars which changed in "
            "forward and used in backward:\n{}".format(used_in_backward))

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        main_block = loss.block
        self.origin_main_block = main_block
        main_program = main_block.program
        if startup_program is None:
            startup_program = default_startup_program()

        pipeline_opt = main_program._pipeline_opt
        assert pipeline_opt, 'Please use pipeline with fleet.'
        required_keys = [
            'local_rank',
            'schedule_mode',
            'micro_batch_size',
            'ring_id',
            'global_ring_id',
            'use_sharding',
            'mp_degree',
            'mp_rank',
        ]
        for key in required_keys:
            assert key in pipeline_opt, \
                'Please use pipeline with fleet to use {}.'.format(key)
        self.local_rank = pipeline_opt['local_rank']
        self.schedule_mode = pipeline_opt['schedule_mode']
        self.micro_batch_size = pipeline_opt['micro_batch_size']
        self.use_sharding = pipeline_opt['use_sharding']
        self.ring_id = pipeline_opt['ring_id']
        self.global_ring_id = pipeline_opt['global_ring_id']
        self.mp_degree = pipeline_opt['mp_degree']
        self.mp_rank = pipeline_opt['mp_rank']
        assert self.mp_degree >= 1
        assert 0 <= self.mp_rank < self.mp_degree

        optimize_ops, params_grads = self._optimizer.minimize(
            loss, startup_program, parameter_list, no_grad_set)
        self._param_device_map = self._origin_optimizer._param_device_map

        self.output_var_to_op, self.input_var_to_op = \
            self._get_input_output_info(main_block)
        # Step1: add default op_device attribute for ops.
        self._add_op_device_attr(main_block)
        device_list = self._check_validation(main_block)

        def device_cmp(device1, device2):
            dev1_id = int(device1.split(':')[1])
            dev2_id = int(device2.split(':')[1])
            if dev1_id < dev2_id:
                return -1
            elif dev1_id > dev2_id:
                return 1
            else:
                return 0

        sorted_device_list = sorted(device_list, key=cmp_to_key(device_cmp))
        assert sorted_device_list == device_list, (
            "With pipeline parallelism, you must use gpu devices one after "
            "another in the order of their ids.")
        # Step2: add send and recv ops between section boundaries
        self._insert_sendrecv_ops_for_boundaries(main_block)

        # Step3: split program into sections and add pairs of
        # send and recv ops for data var.
        main_program = main_block.program
        program_list = self._split_program(main_program, device_list)
        for p in program_list:
            self._create_vars(p.global_block(), main_block)

        self.local_rank %= len(device_list)
        # Step3.5: optimize forward send sync_comm to overlap send and recv
        self._optimize_forward_send_sync(program_list[self.local_rank])

        # Step4: Special Case: process persistable vars that exist in
        # multiple sections
        # FIXME 
        # self._process_persistable_vars_in_multi_sections(
        #     main_program, startup_program, program_list)

        # Step5: Add sub blocks for section programs
        self._add_sub_blocks(main_block, program_list)

        place_list = []
        for dev in device_list:
            dev_index = int(dev.split(":")[1])
            if core.is_compiled_with_cuda():
                place_list.append(core.CUDAPlace(dev_index % 1))
            elif core.is_compiled_with_npu():
                place_list.append(core.NPUPlace(dev_index % 1))

        # Step6: Split startup program
        new_startup_program = self._split_startup_program(startup_program,
                                                          self.local_rank)

        startup_program._pipeline_opt = {
            "startup_program": new_startup_program,
        }
        real_block = program_list[self.local_rank].global_block()
        self._insert_loss_scale(real_block)
        if not self.use_sharding:
            # Step7: clear gradients before each mini-batch and 
            # accumulate gradients during backward
            self._rename_gradient_var_name(real_block)
            real_block._sync_with_cpp()
            self._accumulate_gradients(real_block)
            real_block._sync_with_cpp()

        if core.is_compiled_with_cuda():
            place_id = int(os.getenv("FLAGS_selected_gpus", "0"))
        elif core.is_compiled_with_npu():
            place_id = int(os.getenv("FLAGS_selected_npus", "0"))
        # A pass to move the recv op to the beginning of
        # the forward/backward phase
        self._mv_head_recv(program_list[self.local_rank])

        # A pass to check pipeline persist var which changed in
        # forward and used in backward
        self._check_pipeline_persist_var(program_list[self.local_rank])

        main_program._pipeline_opt = {
            "trainer": "PipelineTrainer",
            "device_worker": "Section",
            "pipeline_stage": self.local_rank,
            "num_pipeline_stages": len(device_list),
            "schedule_mode": self.schedule_mode,
            "inner_parallelism": len(device_list),
            "section_program": program_list[self.local_rank],
            "place": place_list[self.local_rank],
            "place_id": place_id,
            "sync_steps": -1,
            "num_microbatches": self._num_microbatches,
            "start_cpu_core_id": self._start_cpu_core_id,
        }
        return optimize_ops, params_grads, program_list, self._pipeline_pair, self._pp_ring_map


class RecomputeOptimizer(Optimizer):
    """
	:api_attr: Static Graph

    Recompute Optimizer Wrapper

    Normally, a training step contains three sub-steps: first, run forward
    Operators to calculate the loss; second, run backward Operators to 
    calculate gradient of the parameters; third, apply optimization method
    to update the value of the parameters.

    In the forward computation process, all variables that are needed by 
    backward computation process will be kept in memory, which occupy a great
    amount of memory when the network becomes very deep.

    Recompute split the network to k segments. In each segment, It will 
    recompute the forward Operators, before running backward operators. It is
    very helpful for saving memory.
 
    The Variables that separate a network to segments are called as checkpoints,
    and users should set it manually. The usage is very simple:

    Args:
        optimizer (Optimizer): The optimizer that is applied to parameters.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            def gen_data():
                return {"x": np.random.random(size=(32, 32)).astype('float32'),
                "y": np.random.randint(2, size=(32, 1)).astype('int64')}
            def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                print(input_x)
                fc_1 = fluid.layers.fc(input=input_x, size=hid_dim)
                prediction = fluid.layers.fc(input=[fc_1], size=label_dim, act='softmax')
                cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
                sum_cost = fluid.layers.reduce_mean(cost)
                return sum_cost, fc_1, prediction
            input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
            input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')
            cost, fc_1, pred = mlp(input_x, input_y)

            sgd = fluid.optimizer.Adam(learning_rate=0.01)
            sgd = fluid.optimizer.RecomputeOptimizer(sgd)
            sgd._set_checkpoints([fc_1, pred])
            sgd.minimize(cost)

            print("Finished optimize")
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            step = 10

            for i in range(step):
                cost_val = exe.run(feed=gen_data(),
                       program=fluid.default_main_program(),
                       fetch_list=[cost.name])
                print("step=%d cost=%f" % (i, cost_val[0]))

    """

    def __init__(self, optimizer):
        if framework.in_dygraph_mode():
            raise Exception("In dygraph, don't support RecomputeOptimizer.")
        self._optimizer = optimizer
        self._checkpoints = None
        self._learning_rate = self._optimizer._learning_rate
        self._learning_rate_map = self._optimizer._learning_rate_map
        self.enable_offload = False

    def _set_checkpoints(self, checkpoints):
        """
        Args:
            checkpoints (list): List of Variable or string    
        """
        assert isinstance(
            checkpoints, list
        ), "_checkpoints should be a list of Variable or a list of String"
        for ckpt in checkpoints:
            assert (
                isinstance(ckpt, six.string_types) or isinstance(ckpt, Variable)
            ), "_checkpoints should be a list of Variable or a list of String"
        self._checkpoints = checkpoints

    # should enable offload before calling backward 
    def _enable_offload(self):
        self.enable_offload = True

    @framework.deprecate_stat_dict
    def load(self, state_dict):
        """
	    :api_attr: Static Graph

        load function is not supported by Recompute Optimizer for now.
        :return: None

        Args:
            state_dict: the dict load by load_persistable method

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import paddle.compat as cpt
                
                def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                    fc_1 = fluid.layers.fc(input=input_x, size=hid_dim)
                    prediction = fluid.layers.fc(input=[fc_1], size=label_dim, act='softmax')
                    cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
                    sum_cost = fluid.layers.reduce_mean(cost)
                    return sum_cost, fc_1, prediction
                
                input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
                input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')
                cost, fc_1, pred = mlp(input_x, input_y)
                print("Finished FF")
                
                sgd = fluid.optimizer.Adam(learning_rate=0.01)
                sgd = fluid.optimizer.RecomputeOptimizer(sgd)
                sgd._set_checkpoints([fc_1, pred])
                try:
                    state_dict = {}
                    sgd.load(state_dict)
                except NotImplementedError as e:
                    print(cpt.get_exception_message(e))
        """
        raise NotImplementedError(
            "load function is not supported by Recompute Optimizer for now")

    def apply_gradients(self, params_grads):
        """
        call apply_gradients function of self._optimizer.

        Args:
            params_grads (list): list of (param, grad) pair to do optimization.

        Returns:
            list: A list of operators appended to the current program.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import paddle.fluid.framework as framework

                def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                    fc_1 = fluid.layers.fc(input=input_x, size=hid_dim)
                    prediction = fluid.layers.fc(input=[fc_1], size=label_dim, act='softmax')
                    cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
                    sum_cost = fluid.layers.reduce_mean(cost)
                    return sum_cost, fc_1, prediction


                input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
                input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')
                cost, fc_1, pred = mlp(input_x, input_y)
                print("Finished FF")

                sgd = fluid.optimizer.Adam(learning_rate=0.01)
                sgd = fluid.optimizer.RecomputeOptimizer(sgd)
                sgd._set_checkpoints([fc_1, pred])
                params_grads = sgd.backward(
                    cost,
                    startup_program=None,
                    parameter_list=None,
                    no_grad_set=None)

                program = cost.block.program
                with framework.program_guard(program, None):
                    optimize_ops = sgd.apply_gradients(params_grads)

                print("Finished apply gradients")
        """

        return self._optimizer.apply_gradients(params_grads=params_grads)

    def _creat_vars(self, varname):
        pinned_var_name = unique_name.generate(varname + "@Pinned")
        fetched_var_name = unique_name.generate(varname + "@Fetch")

        pinned_var = self._main_program.global_block().create_var(
            name=pinned_var_name,
            shape=self.checkpoint_shape,
            dtype=self._main_program.global_block().var(varname).dtype,
            persistable=False,
            stop_gradient=True)

        fetch_var = self._main_program.global_block().create_var(
            name=fetched_var_name,
            shape=self.checkpoint_shape,
            dtype=self._main_program.global_block().var(varname).dtype,
            persistable=False,
            stop_gradient=False)

        return pinned_var_name, fetched_var_name

    def _append_fill_constant_ops(self, startup_program):
        """
        add fill_constant_ops to the end of the prog

        we should fill the pinned vars before runing the main_prog
        to instantiate their tensor hold_, which could tell us whether 
        the host memory could hold all the checkpoints from all the 
        GPU devices in this node. 
        """
        op_role = 0
        block = startup_program.global_block()
        fill_constant_vars = self.checkpoint_name2pinned_name.values()
        OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
        for varname in fill_constant_vars:
            var = self._main_program.global_block().var(varname)
            # NOTE (JZ-LIANG) to pre-allocate the CUDAPinned MEM
            pinned_var = block.create_var(
                name=varname,
                shape=self.checkpoint_shape,
                dtype=self._main_program.global_block().var(var.name).dtype,
                persistable=False,
                stop_gradient=True)
            block.append_op(
                type='fill_constant',
                outputs={'Out': varname},
                attrs={
                    "shape": var.shape,
                    "dtype": var.dtype,
                    "value": 0.0,
                    "place_type": 2,
                    OP_ROLE_KEY: op_role,
                })

        return

    def _insert_async_memcpy_op(self, insert_idx, src_varname, dst_varname,
                                op_role, dst_place_type):
        OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
        self.block._insert_op_without_sync(
            insert_idx,
            type='memcpy',
            inputs={'X': [self._main_program.global_block().var(src_varname)]},
            outputs={
                'Out': [self._main_program.global_block().var(dst_varname)]
            },
            attrs={
                "dst_place_type": int(dst_place_type),
                OP_ROLE_KEY: op_role
            })

    def _insert_fetch_op(self, idx, varname):
        assert varname in self.checkpoint_name2pinned_name, "Try to fetch {} from Pinned Memory, but it is NOT a checkpoint".format(
            varname)

        pinned_varname = self.checkpoint_name2pinned_name[varname]
        fetch_varname = self.checkpoint_name2fetch_name[varname]
        self._insert_async_memcpy_op(idx, pinned_varname, fetch_varname, 1, 1)

    def _insert_offload_op(self, idx, varname):
        assert varname in self.checkpoint_name2pinned_name, "Try to offload {} to Pinned Memory, but it is NOT a checkpoint".format(
            varname)
        pinned_varname = self.checkpoint_name2pinned_name[varname]
        self._insert_async_memcpy_op(idx, varname, pinned_varname, 0, 2)

    def _insert_sync_op(self, op_idx, checkpoint_name):
        # single stream offload no need sync 
        pass

    def _record_fetch_op(self, idx):
        assert len(self.un_fetch_checkpoint_names
                   ) > 0, "Could NOT found checkpoint to fetch"
        checkpoint_name = self.un_fetch_checkpoint_names.pop(-1)
        logging.debug("Record fetch [{}]".format(checkpoint_name))
        self.idx2insertions[idx] = ("fetch", checkpoint_name)

        return checkpoint_name

    def _record_offload_op(self, idx, checkpoint_name):
        expected_checkpoint_name = self.un_offload_checkpoint_names.pop(0)
        assert checkpoint_name == expected_checkpoint_name, "expected to offload [{}] but got [{}]".format(
            expected_checkpoint_name, checkpoint_name)
        logging.debug("Record offload [{}]".format(checkpoint_name))
        self.idx2insertions[idx] = ("offload", checkpoint_name)

    def _record_sync_op(self, idx, checkpoint_name):
        assert checkpoint_name not in self.synced_checkpoints, "Try to sync the checkpoint [{}] twice".format(
            checkpoint_name)
        self.synced_checkpoints.add(checkpoint_name)
        logging.debug("Record offload sync [{}]".format(checkpoint_name))
        self.idx2insertions[idx] = ("sync", checkpoint_name)

    def _parse_backward(self):

        self.idx2insertions = {}
        # don't offload the last checkpoints, to favor throughput        
        self.un_fetch_checkpoint_names = self.sorted_checkpoint_names[:]
        self.un_fetch_checkpoint_names.pop(-1)
        need_fetch_checkpoint_names = self.un_fetch_checkpoint_names[:]
        self.checkpoint_usage_count = {}
        for checkpoint_name in self.un_fetch_checkpoint_names:
            self.checkpoint_usage_count[checkpoint_name] = 0

        self.bw_strart_op_idx = len(self.block.ops)
        for idx, op in enumerate(self.block.ops):
            if int(op.desc.attr("op_role")) == 1:
                self.bw_strart_op_idx = idx
                break

        assert self.bw_strart_op_idx < len(
            self.block.ops), "Could NOT found backword op in prog"

        # fetch second to last checkpoint at the beginning of BW
        fetched_checkpoint_varname = self._record_fetch_op(
            self.bw_strart_op_idx)
        last_last_fetch_checkpoint = None

        for i, op in enumerate(self.block.ops[self.bw_strart_op_idx:]):
            idx = self.bw_strart_op_idx + i
            input_vars = op.desc.input_arg_names()

            for input_var in input_vars:
                if input_var in need_fetch_checkpoint_names:
                    if input_var not in self.un_fetch_checkpoint_names:
                        # fetch the  offloade checkpoint when the first usage of its previous one
                        if self.checkpoint_usage_count[input_var] == 0:
                            # TODO (JZ-LIANG) sync memcpy_stream if extra stream for memcpy
                            second_to_last_fetch_checkpoint = fetched_checkpoint_varname
                            # there is NO fetch ahead the first checkpoint 
                            if input_var != self.sorted_checkpoint_names[0]:
                                fetched_checkpoint_varname = self._record_fetch_op(
                                    idx)

                        # should check the current used checkpoint is ths last fetch one 
                        assert second_to_last_fetch_checkpoint == input_var, "Current recompute segment should use [{}] BUT got [{}]".format(
                            second_to_last_fetch_checkpoint, input_var)
                        # rename
                        self.block.ops[idx]._rename_input(
                            input_var,
                            self.checkpoint_name2fetch_name[input_var])
                        self.checkpoint_usage_count[input_var] += 1
                    else:
                        raise ValueError(
                            "use checkpoint [{}] before fetch in BW".format(
                                input_var))

        assert len(self.un_fetch_checkpoint_names
                   ) == 0, "{} checkpoints have NOT been Recorded".format(
                       self.un_fetch_checkpoint_names)

    def _update_backward(self):
        if len(self.idx2insertions) == 0:
            return
        total_op = len(self.block.ops)
        for op_idx in reversed(range(self.bw_strart_op_idx, total_op)):
            if op_idx in self.idx2insertions:
                operation, checkpoint_name = self.idx2insertions[op_idx]
                if operation == "fetch":
                    self._insert_fetch_op(op_idx, checkpoint_name)
                    logging.debug("Insert [{}] fetch op.".format(
                        checkpoint_name))
                    del self.idx2insertions[op_idx]
                elif operation == "sync":
                    self._insert_sync_op(op_idx, checkpoint_name)
                    logging.debug("Sync [{}] fetch op.".format(checkpoint_name))
        self.block._sync_with_cpp()
        assert len(
            self.idx2insertions) == 0, "{} checkpoints left un-Fecthed".format(
                [ele[1] for ele in self.idx2insertions.values()])

    def _parse_forward(self):

        self.idx2insertions = {}
        # don't offload the last checkpoints, faster, less memory saving       
        self.un_offload_checkpoint_names = self.sorted_checkpoint_names[:]
        last_checkpoint = self.un_offload_checkpoint_names.pop(-1)
        need_offload_checkpoint_names = self.un_offload_checkpoint_names[:]
        self.checkpoint_usage_count_and_idx = {}
        for checkpoint_name in self.un_offload_checkpoint_names:
            self.checkpoint_usage_count_and_idx[checkpoint_name] = {
                'count': 0,
                'idx': -1
            }
        self.synced_checkpoints = set()
        self.fw_strart_op_idx = len(self.block.ops)
        for idx, op in enumerate(self.block.ops):
            if int(op.desc.attr("op_role")) == 0:
                self.fw_strart_op_idx = idx
                break

        assert self.fw_strart_op_idx < len(
            self.block.ops), "Could NOT found Forward op in prog"
        last_offload_checkpoint = None

        for i, op in enumerate(self.block.ops[self.fw_strart_op_idx:
                                              self.bw_strart_op_idx]):

            idx = self.fw_strart_op_idx + i
            output_vars = op.desc.output_arg_names()
            input_vars = op.desc.input_arg_names()

            for output_var in output_vars:
                if output_var in need_offload_checkpoint_names:
                    assert len(
                        output_vars
                    ) == 1, "chekpoint should be the only Output of a certain op, but [{}] is from [{}]".format(
                        output_var, op)

                    if output_var in self.un_offload_checkpoint_names:
                        # insert sync op if last checkpoint has not been sync
                        if last_offload_checkpoint != None:
                            if self.checkpoint_usage_count_and_idx[
                                    last_offload_checkpoint]['count'] == 0:
                                self._record_sync_op(idx,
                                                     last_offload_checkpoint)
                            else:
                                last_usage_idx = self.checkpoint_usage_count_and_idx[
                                    last_offload_checkpoint]['idx']
                                assert last_usage_idx > 0, "last_usage_idx of checkpoint [{}] should large than 0".format(
                                    last_offload_checkpoint)
                                self._record_sync_op(last_usage_idx + 1,
                                                     last_offload_checkpoint)
                        # insert offload op after the checkpoint's generation op
                        self._record_offload_op(idx + 1, output_var)
                        last_offload_checkpoint = output_var
                    else:
                        raise ValueError(
                            "There should be just ONE op that output checkpoint [{}]".
                            format(output_var))
                # need to sync the last need to offload checkpoint before the last checkpoint as output op
                if output_var == last_checkpoint:
                    assert len(
                        output_vars
                    ) == 1, "chekpoint should be the only Output of a certain op, but [{}] is from [{}]".format(
                        output_var, op)
                    assert last_offload_checkpoint == self.sorted_checkpoint_names[
                        -2], "the last offload chekpoint before [{}] is suppose to be [{}], but got [{}]".format(
                            last_checkpoint, self.sorted_checkpoint_names[-2],
                            last_offload_checkpoint)
                    # sync if last checkpoint has not been sync
                    if self.checkpoint_usage_count_and_idx[
                            last_offload_checkpoint]['idx'] == 0:
                        self._record_sync_op(idx, last_offload_checkpoint)
                    else:
                        last_usage_idx = self.checkpoint_usage_count_and_idx[
                            last_offload_checkpoint]['idx']
                        assert last_usage_idx > 0, "last_usage_idx of checkpoint [{}] should large than 0".format(
                            last_offload_checkpoint)
                        self._record_sync_op(last_usage_idx + 1,
                                             last_offload_checkpoint)
            # record checkpoint usage  
            for input_var in input_vars:
                if input_var in need_offload_checkpoint_names:
                    assert input_var not in self.synced_checkpoints, "checkpoint [{}] used after sync".format(
                        input_var)
                    self.checkpoint_usage_count_and_idx[input_var]['count'] += 1
                    self.checkpoint_usage_count_and_idx[input_var]['idx'] = idx

        assert len(self.un_offload_checkpoint_names
                   ) == 0, "{} checkpoints have NOT been Recorded".format(
                       self.un_fetch_checkpoint_names)
        assert len(self.synced_checkpoints) == len(
            need_offload_checkpoint_names
        ), "{} checkpoints have NOT been Recorded".format(
            set(need_offload_checkpoint_names) - set(self.synced_checkpoints))

    def _update_forward(self):
        if len(self.idx2insertions) == 0:
            return
        for op_idx in reversed(
                range(self.fw_strart_op_idx, self.bw_strart_op_idx)):
            if op_idx in self.idx2insertions:
                operation, checkpoint_name = self.idx2insertions[op_idx]
                if operation == "offload":
                    self._insert_offload_op(op_idx, checkpoint_name)
                    logging.debug("Insert [{}] offload op.".format(
                        checkpoint_name))
                    del self.idx2insertions[op_idx]
                elif operation == "sync":
                    self._insert_sync_op(op_idx, checkpoint_name)
                    logging.debug("Insert [{}] offload_sync op.".format(
                        checkpoint_name))
                    del self.idx2insertions[op_idx]

        self.block._sync_with_cpp()
        assert len(self.idx2insertions
                   ) == 0, "{} checkpoints left un-Offloaded".format(
                       [ele[1] for ele in self.idx2insertions.values()])

    def _check_offload_fetch(self):
        # TODO(JZ-LIANG) the single stream offload need no sync
        pass

    def _offload(self, loss, startup_program=None):
        """
        core steps for recompute offload
        1. create pinned vars and temp vars 
        2. parse & update Forward pass: offload, sync
        3. parse & update Backward pass: rename, fetch, sync
        4. verify the correctness
        """
        self._main_program = loss.block.program
        self.block = loss.block
        if startup_program == None:
            startup_program = paddle.static.default_startup_program()

        with program_guard(self._main_program, startup_program):
            assert len(self.checkpoint_shape) > 0, (
                "checkpoints shape {} should be an non empty list like: [12, 512, 1024]".
                format(self.checkpoint_shape))
            assert all([ele > 0 for ele in self.checkpoint_shape]), (
                "all ele in checkpoints shape {} should be a determined integer larger than 0".
                format(self.checkpoint_shape))
            self.checkpoint_name2pinned_name = dict()
            self.checkpoint_name2fetch_name = dict()
            for checkpoint_varname in self.sorted_checkpoint_names:
                pinned_var_name, fetch_var_name = self._creat_vars(
                    checkpoint_varname)
                self.checkpoint_name2pinned_name[
                    checkpoint_varname] = pinned_var_name
                self.checkpoint_name2fetch_name[
                    checkpoint_varname] = fetch_var_name
            self._append_fill_constant_ops(startup_program)
            # TODO (JZ-LIANG) to provide two offload stragtegy in future
            # step 2. parse & update FW: rename, offload, sync
            self._parse_backward()
            self._update_backward()
            # step 3. parse & update BW: rename, offload, sync
            self._parse_forward()
            self._update_forward()
            # step 4. verify the correctness
            self._check_offload_fetch()

        return

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        """
        call append_backward with checkpoints.

        Args:
            loss (Variable): loss variable to run optimizations.
            startup_program (Program): startup_program for initializing parameters
                in `parameter_list`.
            parameter_list (list): list of Variables or Variable.names to update.
            no_grad_set (set|None): set of Variables or Variables.names should be ignored.
            callbacks (list|None): list of callables to run when appending backward
                operator for one parameter.
            checkpoints (list): list of Variables as checkpoints

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
    
                def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                    fc_1 = fluid.layers.fc(input=input_x, size=hid_dim)
                    prediction = fluid.layers.fc(input=[fc_1], size=label_dim, act='softmax')
                    cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
                    sum_cost = fluid.layers.reduce_mean(cost)
                    return sum_cost, fc_1, prediction
    
    
                input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
                input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')
                cost, fc_1, pred = mlp(input_x, input_y)
                print("Finished FF")
    
                sgd = fluid.optimizer.Adam(learning_rate=0.01)
                sgd = fluid.optimizer.RecomputeOptimizer(sgd)
                sgd._set_checkpoints([fc_1, pred])
                params_grads = sgd.backward(
                    cost,
                    startup_program=None,
                    parameter_list=None,
                    no_grad_set=None)
                print("Finished backward")
        """
        assert (self._checkpoints is not None
                ), "You should call _set_checkpoints first"

        if framework.in_dygraph_mode():
            raise NotImplementedError(
                "DyGraph current does not support recompute")

        self._dtype = loss.dtype
        program = loss.block.program
        with program_guard(program, startup_program):
            checkpoint_vars = []
            for ckpt in self._checkpoints:
                if isinstance(ckpt, Variable):
                    checkpoint_vars.append(ckpt)
                else:
                    checkpoint_vars.append(loss.block.var(ckpt))

            # allow return to non-recompute when checkpoints is empty
            if len(checkpoint_vars) > 0:
                params_grads, sorted_checkpoint_names = append_backward(
                    loss,
                    parameter_list,
                    no_grad_set,
                    checkpoints=checkpoint_vars)
            else:
                params_grads = append_backward(
                    loss,
                    parameter_list,
                    no_grad_set,
                    checkpoints=checkpoint_vars)

        if self.enable_offload:
            self.sorted_checkpoint_names = sorted_checkpoint_names
            self._offload(loss, startup_program=startup_program)

        return params_grads

    def apply_optimize(self, loss, startup_program, params_grads):
        """
        call the apply_optimize function of self._optimizer
        Args:
            loss (Variable): loss variable to run optimizations.
            startup_program (Program): startup_program for initializing parameters
                in `parameter_list`.
            params_grads (list): list of (param, grad) pair to do optimization.
        Examples:
            .. code-block:: python
                import paddle.fluid as fluid
                
                def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                    fc_1 = fluid.layers.fc(input=input_x, size=hid_dim)
                    prediction = fluid.layers.fc(input=[fc_1], size=label_dim, act='softmax')
                    cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
                    sum_cost = fluid.layers.reduce_mean(cost)
                    return sum_cost, fc_1, prediction                
                
                input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
                input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')
                cost, fc_1, pred = mlp(input_x, input_y)
                print("Finished FF")
                
                sgd = fluid.optimizer.Adam(learning_rate=0.01)
                sgd = fluid.optimizer.RecomputeOptimizer(sgd)
                sgd._set_checkpoints([fc_1, pred])
                params_grads = sgd.backward(
                    cost,
                    startup_program=None,
                    parameter_list=None,
                    no_grad_set=None)
                
                optimize_ops = sgd.apply_optimize(
                    cost, startup_program=None, params_grads=params_grads)
                
                print("Finished apply_optimize")
        """

        return self._optimizer.apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads)

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        assert isinstance(loss, Variable), "The loss should be an Variable."
        assert (self._checkpoints is not None
                ), "You should call _set_checkpoints first"
        if framework.in_dygraph_mode():
            raise NotImplementedError(
                "DyGraph current does not support recompute")
        params_grads = self.backward(
            loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set)

        optimize_ops = self.apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads)

        return optimize_ops, params_grads


class LookaheadOptimizer(object):
    r"""
	:api_attr: Static Graph

    This implements the Lookahead optimizer of the
    paper : https://arxiv.org/abs/1907.08610.

    Lookahead keeps two sets of params: the fast_params and
    the slow_params. inner_optimizer update fast_params every 
    training step. Lookahead updates the slow_params and fast_params 
    every k training steps as follows:

    .. math::
        
        slow\_param_t &= slow\_param_{t-1} + \\alpha * (fast\_param_{t-1} - slow\_param_{t-1})
	
	fast\_param_t &=  slow\_param_t

    Args:
        inner_optimizer (Optimizer): The optimizer that update fast params step by step. 
        alpha (float): The learning rate of Lookahead.
        k (int): The slow params is updated every k steps.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np
            import numpy.random as random

            paddle.enable_static()
        
            x = fluid.layers.data(name='x', shape=[2], dtype='float32')
            label = fluid.layers.data(name="label", shape=[1], dtype="int64")
            y = fluid.layers.fc(input=[x], size=2, act="softmax")
            loss = fluid.layers.cross_entropy(input=y, label=label)
            loss = fluid.layers.mean(x=loss)
            sgd = fluid.optimizer.SGD(learning_rate=0.01)
            optimizer = fluid.optimizer.LookaheadOptimizer(sgd,
                                                alpha=0.5,
                                                k=5)
            optimizer.minimize(loss)
            main_program = fluid.default_main_program()
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            def train_reader(limit=5):
                for i in range(limit):
                    yield random.random([2]).astype('float32'), random.random([1]).astype('int64')
            
            feeder = fluid.DataFeeder(feed_list=[x, label], place=place)
            reader = paddle.batch(paddle.reader.shuffle(train_reader, buf_size=50000),batch_size=1)
            
            for batch_data in reader():
                exe.run(fluid.default_main_program(),
                feed=feeder.feed(batch_data))

    """

    def __init__(self, inner_optimizer, alpha=0.5, k=5):

        if framework.in_dygraph_mode():
            raise Exception("In dygraph, don't support LookaheadOptimizer.")
        assert (inner_optimizer is not None), "inner optimizer can not be None"
        assert (
            0.0 <= alpha <= 1.0
        ), "alpha should be larger or equal to 0.0, and less or equal than 1.0"
        assert (isinstance(k, int) and k > 0), "k should be a positive integer"

        self.inner_optimizer = inner_optimizer
        self.alpha = alpha
        self.k = k
        self.type = "lookahead"

    def minimize(self, loss, startup_program=None):

        # Apply inner optimizer to the main_program
        mini_out = self.inner_optimizer.minimize(
            loss, startup_program=startup_program)

        # Get startup_program and main_program
        if startup_program is None:
            startup_program = default_startup_program()
        main_block = loss.block

        # add some vars to the main_program
        params = [param.name for param in main_block.all_parameters()]
        param_to_slow = {}
        for param in params:
            fast_var = main_block.var(param)
            assert (fast_var is not None)
            slow_var = main_block.create_var(
                name=param + "@SLOW",
                shape=fast_var.shape,
                dtype=fast_var.dtype,
                persistable=True)
            param_to_slow[param] = slow_var

        # add some vars to the startup_program
        startup_block = startup_program.global_block()
        for param in params:
            fast_var = startup_block.var(param)
            assert (fast_var is not None)
            slow_var = startup_block.create_var(
                name=param + "@SLOW",
                shape=fast_var.shape,
                dtype=fast_var.dtype,
                persistable=True)

            startup_block.append_op(
                type="assign",
                inputs={"X": fast_var},
                outputs={"Out": slow_var})

        with framework.program_guard(main_block.program, startup_program):
            # Add Var k to main prog and startup prog
            k = layers.create_global_var(
                name="lookahead_k",
                shape=[1],
                value=int(self.k),
                dtype='int32',
                persistable=True)

            # Add Var alpha to main prog and startup prog
            alpha = layers.create_global_var(
                name="lookahead_alpha",
                shape=[1],
                value=float(self.alpha),
                dtype='float32',
                persistable=True)

            # Add Var step
            step = layers.create_global_var(
                name="lookahead_step",
                shape=[1],
                value=int(0),
                dtype='int32',
                persistable=True)
            layers.increment(x=step, value=1.0, in_place=True)

            # lookahead
            zero_var = layers.fill_constant(
                shape=[1], dtype='float32', value=0.0)

            one_var = layers.fill_constant(
                shape=[1], dtype='float32', value=1.0)

            mod = layers.elementwise_mod(step, k)
            with layers.control_flow.Switch() as switch:
                with switch.case(step == one_var):
                    for param_name in params:
                        fast_var = main_block.var(param_name)
                        slow_var = param_to_slow[param_name]
                        layers.assign(input=fast_var, output=slow_var)
                with switch.case(mod == zero_var):
                    for param_name in params:
                        fast_var = main_block.var(param_name)
                        slow_var = param_to_slow[param_name]
                        tmp_var = layers.elementwise_add(
                            layers.elementwise_mul(fast_var, alpha),
                            layers.elementwise_mul(
                                slow_var,
                                layers.elementwise_sub(one_var, alpha)))
                        layers.assign(input=tmp_var, output=slow_var)
                        layers.assign(input=tmp_var, output=fast_var)
                with switch.default():
                    pass
        return mini_out


class GradientMergeOptimizer(object):
    """
    Gradient Merge, also called as Gradient Accumulation,
    is a training strategy for larger batches. With this strategy,
    the parameter will not be updated until specific steps.

    For each step, the forward network and the backward network
    will run to calculate the gradient of the parameters.

    For every k step, the optimization network will run,
    applying a specific optimization method (such as SGD, Adam)
    to the parameters.

    Args:
        inner_optimizer (Optimizer): The specific optimization (such as SGD, Adam)
            which update the parameters
        k_steps (int): the update period of the parameters
        avg (bool): whether to average the gradients of each mini-batch,
            the default value is `True`

    Examples:
        .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        def gen_data(batch_size):
            return {"x": np.random.random(size=(batch_size, 32)).astype('float32'),
                    "y": np.random.random(size=(batch_size, 1)).astype('int64')}

        def mlp(input_x, input_y, hid_dim=128, label_dim=2):
            fc_1 = fluid.layers.fc(input=input_x, size=hid_dim)
            prediction = fluid.layers.fc(input=[fc_1], size=label_dim, act='softmax')
            cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
            sum_cost = fluid.layers.reduce_mean(cost)
            return sum_cost, fc_1, prediction

        input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
        input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')
        cost, fc_1, pred = mlp(input_x, input_y)
        sgd = fluid.optimizer.Adam(learning_rate=0.01)
        sgd = fluid.optimizer.GradientMergeOptimizer(sgd, k_steps=4, avg=True)
        sgd.minimize(cost)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        for i in range(10):
            cost_val = exe.run(feed=gen_data(32),
                       program=fluid.default_main_program(),
                       fetch_list=[cost.name])
            print("step=%d, cost=%f" % (i, cost_val[0]))
    """

    GRAD_MERGE_COND_NAME = "grad_merge_cond_name"

    def __init__(self, inner_optimizer, k_steps=1, avg=True):
        if framework.in_dygraph_mode():
            raise Exception(
                "In dygraph, we don't support GradientMergeOptimizer."
                "You can do Gradient merge by yourself with k-times forward + backward, "
                "and one-time optimizer.minimize()")

        assert (inner_optimizer is not None), "inner optimizer can not be None"
        assert (isinstance(k_steps, int) and
                k_steps > 0), "k_steps should be a positive integer"

        self.inner_optimizer = inner_optimizer
        self.k_steps = k_steps
        self.type = "gradient_merge"
        self.avg = avg
        self._optimize_ops = None

    def _set_k_steps(self, k_steps):
        self.k_steps = k_steps

    def _set_avg(self, avg):
        self.avg = avg

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        assert isinstance(loss, Variable), "The loss should be an Variable."
        assert (
            parameter_list is None
        ), "The parameter_list should be None when using GradientMergeOptimizer"
        assert (
            no_grad_set is None
        ), "The no_grad_set should be None when using GradientMergeOptimizer"

        params_grads = self.inner_optimizer.backward(
            loss, startup_program=startup_program)
        return params_grads

    def apply_optimize(self, loss, startup_program, params_grads):
        program = loss.block.program
        with program_guard(program, startup_program):
            optimize_ops = self.apply_gradients(params_grads)
        return optimize_ops

    def _is_the_backward_op(self, op):
        op_maker = core.op_proto_and_checker_maker
        backward = core.op_proto_and_checker_maker.OpRole.Backward
        if op_maker.kOpRoleVarAttrName() in op.attr_names and \
                int(op.all_attrs()[op_maker.kOpRoleAttrName()]) == int(backward):
            return True
        return False

    def _remove_op_role_var(self, param, grad):
        op_maker = core.op_proto_and_checker_maker
        op = grad.op
        assert self._is_the_backward_op(op), \
            'grad.op={} is not the backward op which produces the grad={}' \
            .format(op, grad.name)

        block = grad.block
        var_attr = op.all_attrs()[op_maker.kOpRoleVarAttrName()]
        assert param.name in var_attr, \
            'when using GradientMergeOptimizer, param={} must be in var_attr={}' \
            .format(param.name, var_attr)
        assert grad.name in var_attr, \
            'when using GradientMergeOptimizer, grad={} must be in var_attr={}' \
            .format(param.name, var_attr)

        # remove (param, grad) from op_role_var
        var_attr.remove(param.name)
        var_attr.remove(grad.name)
        if len(var_attr) > 1:
            op._set_attr(op_maker.kOpRoleVarAttrName(), var_attr)
        else:
            op._remove_attr(op_maker.kOpRoleVarAttrName())

    def _add_gm_op_role_var(self, op, param, grad, cond):
        grad.op = op
        op_maker = core.op_proto_and_checker_maker
        backward = op_maker.OpRole.Backward

        # NOTE(wangxi). When distributed, we will insert grad_merge_all_reduce_op_handle
        # in multi_devices_graph_pass, which will allreduce(grad) if cond is True, else
        # do nothing.
        # In this way, the gradient can be merged first, and then communicate when the
        # condition is met, reducing the number of communications to increase the
        # speed.
        op._set_attr(self.GRAD_MERGE_COND_NAME, cond.name)
        op._set_attr(op_maker.kOpRoleAttrName(), backward)
        op._set_attr(op_maker.kOpRoleVarAttrName(), [param.name, grad.name])

    def _get_gm_cond_var(self, main_block):
        # Add const var
        k_step_var = layers.create_global_var(
            name="gradient_merge_k",
            shape=[1],
            value=int(self.k_steps),
            dtype='int32',
            persistable=True,
            force_cpu=True)

        zero_var = layers.create_global_var(
            name="gradient_merge_zero",
            shape=[1],
            value=int(0),
            dtype='int32',
            persistable=True,
            force_cpu=True)

        # Add step var & cond var
        step_var = layers.create_global_var(
            name="gradient_merge_step",
            shape=[1],
            value=int(0),
            dtype='int32',
            persistable=True,
            force_cpu=True)

        cond_var = layers.create_global_var(
            name="gradient_merge_cond",
            shape=[1],
            value=bool(0),
            dtype='bool',
            persistable=False,
            force_cpu=True)

        with device_guard("cpu"):
            # step_var = (step_var + 1) % k_step
            layers.increment(x=step_var, value=1.0, in_place=True)
            main_block.append_op(
                type='elementwise_mod',
                inputs={'X': step_var,
                        'Y': k_step_var},
                outputs={'Out': step_var},
                attrs={'axis': -1,
                       'use_mkldnn': False})

            # cond_var = (step_var == 0)
            main_block.append_op(
                type='equal',
                inputs={'X': step_var,
                        'Y': zero_var},
                outputs={'Out': cond_var})

        return cond_var

    def apply_gradients(self, params_grads):
        main_program = default_main_program()
        startup_program = default_startup_program()
        main_block = main_program.global_block()
        startup_block = startup_program.global_block()

        cond = self._get_gm_cond_var(main_block)

        #TODO(mapingshuo) support sparse embedding
        # step1: remove grad.op's op_role_var
        for param, grad in params_grads:
            assert (
                param.type != core.VarDesc.VarType.SELECTED_ROWS
            ), "SELECTED_ROWS is not supported in GradientMergeOptimizer for now"

            self._remove_op_role_var(param, grad)

        param_to_grad = {k.name: v for (k, v) in params_grads}
        param_names = param_to_grad.keys()
        param_to_gradient_merge = {}

        new_params_grads = []
        # step2: create gradient_merge var and init with 0
        # and update op_role_var
        for param, grad in params_grads:
            param_name = param.name
            param_var = main_block.var(param_name)
            assert (param_var is not None)
            gradient_merge_var = main_block.create_var(
                name=param_name + "@GRAD@GradientMerge",
                shape=param_var.shape,
                dtype=param_var.dtype,
                persistable=True)
            param_to_gradient_merge[param_name] = gradient_merge_var

            startup_gradient_merge_var = startup_block.create_var(
                name=param_name + "@GRAD@GradientMerge",
                shape=param_var.shape,
                dtype=param_var.dtype,
                persistable=True)
            startup_block.append_op(
                type="fill_constant",
                outputs={"Out": startup_gradient_merge_var},
                attrs={
                    "shape": param_var.shape,
                    "dtype": param_var.dtype,
                    "value": float(0),
                })

            # grad_merge += grad
            new_grad_op = main_block.append_op(
                type="elementwise_add",
                inputs={'X': grad,
                        'Y': gradient_merge_var},
                outputs={'Out': gradient_merge_var},
                attrs={'axis': -1,
                       'use_mkldnn': False})
            self._add_gm_op_role_var(new_grad_op, param, gradient_merge_var,
                                     cond)
            new_params_grads.append([param, gradient_merge_var])

        def true_apply_gradient():
            cur_block_idx = main_program.current_block_idx
            cur_block = main_program.current_block()

            # cur_block's forward_block & backward_block is itself
            cur_block._set_forward_block_idx(cur_block_idx)
            op_maker = core.op_proto_and_checker_maker

            if self.avg:
                for param, new_grad in new_params_grads:
                    # grad /= k_steps
                    cur_block.append_op(
                        type='scale',
                        inputs={'X': new_grad},
                        outputs={'Out': new_grad},
                        attrs={
                            'scale': 1.0 / self.k_steps,
                            'bias': 0.0,
                            'bias_after_scale': False
                        })
                    new_grad.op._set_attr(op_maker.kOpRoleAttrName(),
                                          op_maker.OpRole.Backward)

            for param, new_grad in new_params_grads:
                # NOTE. regularization will append ops to grad.block,
                # while new_grad's real block is global_block,
                # but we want append regularization ops to cur_block,
                # so we set new_grad.block = cur_block
                new_grad.block = cur_block

            self._optimize_ops = self.inner_optimizer.apply_gradients(
                new_params_grads)

            # clear gradient_merge_vars
            for param, new_grad in new_params_grads:
                layers.fill_constant(
                    shape=new_grad.shape,
                    dtype=new_grad.dtype,
                    value=0.0,
                    out=new_grad)
                new_grad.op._set_attr(op_maker.kOpRoleAttrName(),
                                      op_maker.OpRole.Optimize)

        # step3. apply gradient
        layers.cond(cond, true_fn=true_apply_gradient, false_fn=None)

        return self._optimize_ops

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        assert isinstance(loss, Variable), "The loss should be an Variable."

        params_grads = self.backward(
            loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set)

        optimize_ops = self.apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads)

        return optimize_ops, params_grads
