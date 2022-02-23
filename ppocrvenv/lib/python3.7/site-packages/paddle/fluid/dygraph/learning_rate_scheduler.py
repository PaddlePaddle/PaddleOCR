# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import math
import warnings

from .. import unique_name
from ..framework import Variable
from ..data_feeder import check_type

__all__ = [
    'NoamDecay', 'PiecewiseDecay', 'NaturalExpDecay', 'ExponentialDecay',
    'InverseTimeDecay', 'PolynomialDecay', 'CosineDecay', 'LinearLrWarmup',
    'ReduceLROnPlateau', 'StepDecay', 'MultiStepDecay', 'LambdaDecay'
]


class LearningRateDecay(object):
    """
    Base class of learning rate decay
    
    Define the common interface of an LearningRateDecay.
    User should not use this class directly,
    but need to use one of it's implementation.
    """

    def __init__(self, begin=0, step=1, dtype='float32'):
        self.step_num = begin
        self.step_size = step
        self.dtype = dtype

    def __call__(self):
        lr = self.step()
        if isinstance(lr, float):
            lr = self.create_lr_var(lr)
        self.step_num += self.step_size
        return lr

    def create_lr_var(self, lr):
        """
        convert lr from float to variable

        Args: 
            lr: learning rate
        Returns:
            learning rate variable
        """
        from .. import layers
        lr = layers.create_global_var(
            name=unique_name.generate("learning_rate"),
            shape=[1],
            value=float(lr),
            dtype=self.dtype,
            persistable=False)
        return lr

    # Note: If you want to change what optimizer.state_dict stores, just overwrite this functions, 
    # "self.step_num" will be stored by default.
    def state_dict(self):
        """
        Returns the state of the scheduler as a :class:`dict`.

        It is a subset of self.__dict__ .
        """
        self._state_keys()
        state_dict = {}
        for key in self.keys:
            if key not in self.__dict__:
                continue
            value = self.__dict__[key]
            if isinstance(value, Variable):
                assert value.shape == [
                    1
                ], "shape of Variable in state_dict must be [1] {}".format(
                    value.shape)
                value = value.numpy()[0]
            state_dict[key] = value

        return state_dict

    def _state_keys(self):
        """
        set the keys in self.__dict__ that are needed to be saved.
        """
        self.keys = ['step_num']

    def set_state_dict(self, state_dict):
        """
        Loads the schedulers state.
        """
        self._state_keys()
        for key in self.keys:
            if key in state_dict:
                self.__dict__[key] = state_dict[key]
            else:
                raise RuntimeError(
                    "Please check whether state_dict is correct for optimizer. Can't find [ {} ] in state_dict".
                    format(key))
        if len(state_dict) > len(self.keys):
            warnings.warn(
                "There are some unused values in state_dict. Maybe the optimizer have different 'LearningRateDecay' when invoking state_dict and set_dict"
            )

    # [aliases] Compatible with old method names
    set_dict = set_state_dict

    def step(self):
        raise NotImplementedError()


class PiecewiseDecay(LearningRateDecay):
    """
    :api_attr: imperative
    
    Piecewise decay scheduler.

    The algorithm can be described as the code below.

    .. code-block:: text

        boundaries = [10000, 20000]
        values = [1.0, 0.5, 0.1]
        if global_step < 10000:
            learning_rate = 1.0
        elif 10000 <= global_step < 20000:
            learning_rate = 0.5
        else:
            learning_rate = 0.1

    Parameters:
        boundaries(list): A list of steps numbers. The type of element in the list is python int. 
        values(list): A list of learning rate values that will be picked during
            different step boundaries. The type of element in the list is python float.
        begin(int): The begin step to initialize the global_step in the description above.
        step(int, optional): The step size used to calculate the new global_step in the description above.
            The default value is 1.
        dtype(str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. The default value is 'float32'.

    Returns:
        None.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          boundaries = [10000, 20000]
          values = [1.0, 0.5, 0.1]
          with fluid.dygraph.guard():
              emb = fluid.dygraph.Embedding( [10, 10] )
              optimizer = fluid.optimizer.SGD(
                 learning_rate=fluid.dygraph.PiecewiseDecay(boundaries, values, 0),
                 parameter_list = emb.parameters() )
    """

    def __init__(self, boundaries, values, begin, step=1, dtype='float32'):
        super(PiecewiseDecay, self).__init__(begin, step, dtype)
        self.boundaries = boundaries
        self.values = values

        self.vars = []
        for value in values:
            self.vars.append(value)

    def step(self):
        for i in range(len(self.boundaries)):
            if self.step_num < self.boundaries[i]:
                return self.vars[i]
        return self.create_lr_var(self.vars[len(self.values) - 1])


class NaturalExpDecay(LearningRateDecay):
    r"""
    :api_attr: imperative

    Applies natural exponential decay to the initial learning rate.
    
    The algorithm can be described as following.

    .. math::

        decayed\_learning\_rate = learning\_rate * e^{y} 

    If staircase is set to False, then:

    .. math::

        y = - decay\_rate * \\frac{global\_step}{decay\_steps}

    If staircase is set to True, then:

    .. math::

        y = - decay\_rate * math.floor(\\frac{global\_step}{decay\_steps}) 

    Parameters:
        learning_rate(Variable|float): The initial learning rate. If the type 
            is Variable, it's a tensor with shape [1], the data type can be  
            float32 or float64. It also can be set to python int number.
        decay_steps(int): The decay step size. It determines the decay cycle.
        decay_rate(int): The decay rate.
        staircase(bool, optional): If set to True, decay the learning rate at discrete intervals. The 
            default value is False.
        begin(int, optional): The begin step. The initial value of global_step described above. The default value is 0.
        step(int, optional): The step size used to calculate the new global_step in the description above.
            The default value is 1.
        dtype(str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. The default value is 'float32'.

    Returns:
        None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            base_lr = 0.1
            with fluid.dygraph.guard():
                emb = fluid.dygraph.Embedding([10, 10])
                sgd_optimizer = fluid.optimizer.SGD(
                        learning_rate=fluid.dygraph.NaturalExpDecay(
                            learning_rate=base_lr,
                            decay_steps=10000,
                            decay_rate=0.5,
                            staircase=True),
                        parameter_list=emb.parameters())

    """

    def __init__(self,
                 learning_rate,
                 decay_steps,
                 decay_rate,
                 staircase=False,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(NaturalExpDecay, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def step(self):
        from .. import layers
        div_res = self.create_lr_var(self.step_num / self.decay_steps)
        if self.staircase:
            div_res = layers.floor(div_res)
        decayed_lr = self.learning_rate * layers.exp(-1 * self.decay_rate *
                                                     div_res)

        return decayed_lr


class ExponentialDecay(LearningRateDecay):
    r"""
    :api_attr: imperative

    Applies exponential decay to the learning rate.

    The algorithm can be described as following.
    
    .. math::

        decayed\_learning\_rate = learning\_rate * decay\_rate ^ y 

    If staircase is set to False, then:

    .. math::

        y = \\frac{global\_step}{decay\_steps} 

    If staircase is set to True, then:

    .. math::

        y = math.floor(\\frac{global\_step}{decay\_steps})


    Parameters:
        learning_rate(Variable|float): The initial learning rate. If the type 
            is Variable, it's a tensor with shape [1], the data type can be  
            float32 or float64. It also can be set to python int number.
        decay_steps(int): The decay step size. It determines the decay cycle.
        decay_rate(float): The decay rate.
        staircase(bool, optional): If set to True, decay the learning rate at discrete intervals. The 
            default value is False.
        begin(int, optional): The begin step. The initial value of global_step described above. The default value is 0.
        step(int, optional): The step size used to calculate the new global_step in the description above.
            The default value is 1.
        dtype(str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. The default value is 'float32'.

    Returns:
        None.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          base_lr = 0.1
          with fluid.dygraph.guard():
              sgd_optimizer = fluid.optimizer.SGD(
    	            learning_rate=fluid.dygraph.ExponentialDecay(
		        learning_rate=base_lr,
    		        decay_steps=10000,
		        decay_rate=0.5,
		        staircase=True))

    """

    def __init__(self,
                 learning_rate,
                 decay_steps,
                 decay_rate,
                 staircase=False,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(ExponentialDecay, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def step(self):
        from .. import layers
        div_res = self.create_lr_var(self.step_num / self.decay_steps)
        if self.staircase:
            div_res = layers.floor(div_res)

        decayed_lr = self.learning_rate * (self.decay_rate**div_res)

        return decayed_lr


class InverseTimeDecay(LearningRateDecay):
    r"""
    :api_attr: imperative

    Applies inverse time decay to the initial learning rate.

    The algorithm can be described as following.
    If staircase is set to False, then:

    .. math::

        decayed\_learning\_rate = \\frac{learning\_rate}{1 + decay\_rate * \\frac{global\_step}{decay\_step}}  

    If staircase is set to True, then:

    .. math::

        decayed\_learning\_rate = \\frac{learning\_rate}{1 + decay\_rate * math.floor(\\frac{global\_step}{decay\_step})}

    Parameters:
        learning_rate(Variable|float): The initial learning rate. If the type 
            is Variable, it's a tensor with shape [1], the data type can be  
            float32 or float64. It also can be set to python int number.
        decay_steps(int): The decay step size. It determines the decay cycle.
        decay_rate(float): The decay rate.
        staircase(bool, optional): If set to True, decay the learning rate at discrete intervals. The 
            default value is False.
        begin(int, optional): The begin step. The initial value of global_step described above. The default value is 0.
        step(int, optional): The step size used to calculate the new global_step in the description above.
            The default value is 1.
        dtype(str, optional): The data type used to create the learning rate variable. The data type can be 
            'float32', 'float64'. The default value is 'float32'.

    Returns:
        None.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          base_lr = 0.1
          with fluid.dygraph.guard():
              emb = fluid.dygraph.Embedding([10, 10])
              sgd_optimizer = fluid.optimizer.SGD(
	          learning_rate=fluid.dygraph.InverseTimeDecay(
		        learning_rate=base_lr,
		        decay_steps=10000,
		        decay_rate=0.5,
		        staircase=True),
                  parameter_list = emb.parameters())

    """

    def __init__(self,
                 learning_rate,
                 decay_steps,
                 decay_rate,
                 staircase=False,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(InverseTimeDecay, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def step(self):
        from .. import layers
        div_res = self.create_lr_var(self.step_num / self.decay_steps)
        if self.staircase:
            div_res = layers.floor(div_res)

        decayed_lr = self.learning_rate / (1 + self.decay_rate * div_res)

        return decayed_lr


class PolynomialDecay(LearningRateDecay):
    r"""
    :api_attr: imperative

    Applies polynomial decay to the initial learning rate.

    The algorithm can be described as following.

    If cycle is set to True, then:

    .. math::

        decay\_steps & = decay\_steps * math.ceil(\\frac{global\_step}{decay\_steps}) 

        decayed\_learning\_rate & = (learning\_rate-end\_learning\_rate)*(1-\\frac{global\_step}{decay\_steps})^{power}+end\_learning\_rate

    If cycle is set to False, then:

    .. math::

        global\_step & = min(global\_step, decay\_steps) 

        decayed\_learning\_rate & = (learning\_rate-end\_learning\_rate)*(1-\\frac{global\_step}{decay\_steps})^{power}+end\_learning\_rate

    Parameters:
        learning_rate(Variable|float): The initial learning rate. If the type 
            is Variable, it's a tensor with shape [1], the data type can be  
            float32 or float64. It also can be set to python int number.
        decay_steps(int): The decay step size. It determines the decay cycle.
        end_learning_rate(float, optional): The minimum final learning rate. The default value is 0.0001.
        power(float, optional): Power of polynomial. The default value is 1.0.
        cycle(bool, optional): If set true, decay the learning rate every decay_steps. The default value is False.
        begin(int, optional): The begin step. The initial value of global_step described above. The default value is 0.
        step(int, optional): The step size used to calculate the new global_step in the description above.
            The default value is 1.
        dtype(str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. The default value is 'float32'.

    Returns:
        None.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          start_lr = 0.01
          total_step = 5000
          end_lr = 0
          with fluid.dygraph.guard():
              emb = fluid.dygraph.Embedding( [10, 10])
              optimizer  = fluid.optimizer.SGD(
                  learning_rate = fluid.dygraph.PolynomialDecay(
                  start_lr, total_step, end_lr, power=1.0),
                  parameter_list = emb.parameters())

    """

    def __init__(self,
                 learning_rate,
                 decay_steps,
                 end_learning_rate=0.0001,
                 power=1.0,
                 cycle=False,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(PolynomialDecay, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle

    def step(self):
        from .. import layers
        tmp_step_num = self.step_num
        tmp_decay_steps = self.decay_steps
        if self.cycle:
            div_res = layers.ceil(
                self.create_lr_var(tmp_step_num / float(self.decay_steps)))

            if tmp_step_num == 0:
                div_res = self.create_lr_var(1.0)
            tmp_decay_steps = self.decay_steps * div_res
        else:
            tmp_step_num = self.create_lr_var(tmp_step_num
                                              if tmp_step_num < self.decay_steps
                                              else self.decay_steps)

        decayed_lr = (self.learning_rate - self.end_learning_rate) * \
            ((1 - tmp_step_num / tmp_decay_steps) ** self.power) + self.end_learning_rate
        return decayed_lr


class CosineDecay(LearningRateDecay):
    r"""
    :api_attr: imperative

    Applies cosine decay to the learning rate.

    The algorithm can be described as following.

    .. math::

        decayed\_learning\_rate = learning\_rate * 0.5 * (math.cos(global\_step * \\frac{math.pi}{step\_each\_epoch} ) + 1)
    
    Parameters:
        learning_rate(Variable|float): The initial learning rate. If the type 
            is Variable, it's a tensor with shape [1], the data type can be  
            float32 or float64. It also can be set to python int number.
        step_each_epoch(int): The number of steps in an epoch.
        epochs(int): The number of epochs.
        begin(int, optional): The begin step. The initial value of global_step described above. The default value is 0.
        step(int, optional): The step size used to calculate the new global_step in the description above.
            The default value is 1.
        dtype(str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. The default value is 'float32'.

    Returns:
        None.

    Examples:
	.. code-block:: python

  	    base_lr = 0.1
            with fluid.dygraph.guard():
                optimizer  = fluid.optimizer.SGD(
        	    learning_rate = fluid.dygraph.CosineDecay(
	                    base_lr, 10000, 120) )
    """

    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(CosineDecay, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.step_each_epoch = step_each_epoch
        self.epochs = epochs

    def step(self):
        from .. import layers
        cur_epoch = layers.floor(
            self.create_lr_var(self.step_num / self.step_each_epoch))
        decayed_lr = self.learning_rate * 0.5 * (
            layers.cos(cur_epoch * math.pi / self.epochs) + 1)
        return decayed_lr


class NoamDecay(LearningRateDecay):
    r"""
    :api_attr: imperative

    Applies Noam decay to the initial learning rate. 

    The algorithm can be described as following.

    .. math::

        decayed\_learning\_rate = learning\_rate * d_{model}^{-0.5} * min(global\_step^{-0.5}, global\_step * warmup\_steps^{-1.5})

    Please reference `attention is all you need <https://arxiv.org/pdf/1706.03762.pdf>`_ 

    Parameters:
        d$_{model}$(Variable|int): The dimensionality of input and output feature vector of model. If type is Variable, 
            it's a tensor with shape [1] and the data type can be int32 or int64. The type can also be python int.
        warmup_steps(Variable|int): The number of warmup steps. A super parameter. If type is Variable, 
            it's a tensor with shape [1] and the data type can be int32 or int64. The type can also be python int.
        begin(int, optional): The begin step. The initial value of global_step described above. The default value is 0.
        step(int, optional): The step size used to calculate the new global_step in the description above.
            The default value is 1.
        dtype(str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. The default value is 'float32'.
        learning_rate(Variable|float|int): The initial learning rate. If the type
            is Variable, it's a tensor with shape [1], the data type can be
            float32 or float64. It also can be set to python int number. Default 1.0

    Returns:
        None.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          warmup_steps = 100
          learning_rate = 0.01
          with fluid.dygraph.guard():
              emb = fluid.dygraph.Embedding([10, 10])
              optimizer  = fluid.optimizer.SGD(
                  learning_rate = fluid.dygraph.NoamDecay(
                         1/(warmup_steps *(learning_rate ** 2)),
                         warmup_steps),
                  parameter_list = emb.parameters())
    """

    def __init__(self,
                 d_model,
                 warmup_steps,
                 begin=1,
                 step=1,
                 dtype='float32',
                 learning_rate=1.0):
        super(NoamDecay, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def step(self):
        from .. import layers
        a = self.create_lr_var(self.step_num**-0.5)
        b = self.create_lr_var((self.warmup_steps**-1.5) * self.step_num)
        lr_value = self.learning_rate * (self.d_model
                                         **-0.5) * layers.elementwise_min(a, b)
        return lr_value


class LinearLrWarmup(LearningRateDecay):
    """
    :api_attr: imperative

    This operator use the linear learning rate warm up strategy to adjust the learning rate preliminarily before the normal learning rate scheduling.
    For more information, please refer to `Bag of Tricks for Image Classification with Convolutional Neural Networks <https://arxiv.org/abs/1812.01187>`_
    
    When global_step < warmup_steps, learning rate is updated as:
    
    .. code-block:: text
    
            linear_step = end_lr - start_lr
            lr = start_lr + linear_step * (global_step / warmup_steps)
    
    where start_lr is the initial learning rate, and end_lr is the final learning rate;
    
    When global_step >= warmup_steps, learning rate is updated as:
    
    .. code-block:: text
    
            lr = learning_rate
    
    where lr is the learning_rate after warm-up.
    
    Args:
        learning_rate (Variable|float): Learning_rate after warm-up, it could be 1D-Tensor or single value with the data type of float32.
        warmup_steps (int): Steps for warm up.
        start_lr (float): Initial learning rate of warm up.
        end_lr (float): Final learning rate of warm up.
        begin(int, optional): The begin step. The initial value of global_step described above. The default value is 0.
        step(int, optional): The step size used to calculate the new global_step in the description above.
            The default value is 1.
        dtype(str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. The default value is 'float32'.
    
    Returns:
        Variable: Warm-up learning rate with the same data type as learning_rate.
    
    
    Examples:
    
    .. code-block:: python
    
        import paddle.fluid as fluid
    
        learning_rate = 0.1 
        warmup_steps = 50
        start_lr = 0
        end_lr = 0.1

        with fluid.dygraph.guard(): 
            lr_decay = fluid.dygraph.LinearLrWarmup( learning_rate, warmup_steps, start_lr, end_lr)
    
       
    """

    def __init__(self,
                 learning_rate,
                 warmup_steps,
                 start_lr,
                 end_lr,
                 begin=1,
                 step=1,
                 dtype='float32'):
        super(LinearLrWarmup, self).__init__(begin, step, dtype)
        type_check = isinstance(learning_rate, float) or isinstance(
            learning_rate, int) or isinstance(learning_rate, LearningRateDecay)
        if not type_check:
            raise TypeError(
                "the type of learning_rate should be [int, float or LearningRateDecay], the current type is {}".
                format(learning_rate))
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        assert end_lr > start_lr, "end_lr {} must be greater than start_lr {}".format(
            end_lr, start_lr)
        self.lr_ratio_before_warmup = (
            float(end_lr) - float(start_lr)) / float(warmup_steps)

    def step(self):
        base_lr = self.learning_rate
        if isinstance(self.learning_rate, LearningRateDecay):
            base_lr = base_lr()

        from .. import layers
        if self.step_num < self.warmup_steps:
            return self.lr_ratio_before_warmup * self.step_num + self.start_lr
        else:
            return base_lr


class ReduceLROnPlateau(LearningRateDecay):
    """
    :api_attr: imperative

    Reduce learning rate when ``loss`` has stopped descending. Models often benefit from reducing the learning rate 
    by 2 to 10 times once model performance has no longer improvement.

    The ``loss`` is the one which has been pass into ``step`` , it must be 1-D Tensor with shape [1]. When ``loss`` 
    stop descending for a ``patience`` number of epochs, the learning rate will be reduced to ``learning_rate * decay_rate`` . 
    (Specially, ``mode`` can also be set to ``'max`` , in this case, when ``loss`` stop ascending for a ``patience`` number 
    of epochs, the learning rate will be reduced.)

    In addition, After each reduction, it will wait a ``cooldown`` number of epochs before resuming normal operation.

    Args:
        learning_rate (Variable|float|int): The initial learning rate. It can be set to python float or int number.
            If the type is Variable, it should be 1-D Tensor with shape [1], the data type can be 'float32' or 'float64'.
        mode (str, optional): ``'min'`` or ``'max'`` can be selected. Normally, it is ``'min'`` , which means that the 
            learning rate will reduce when ``loss`` stops descending. Specially, if it's set to ``'max'`` ,  the learning 
            rate will reduce when ``loss`` stops ascending. Default: ``'min'`` .
        decay_rate (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * decay_rate`` . 
            It should be less than 1.0. Default: 0.1.
        patience (int, optional): When ``loss`` doesn't improve for this number of epochs, learing rate will be reduced. 
            Default: 10.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False``.
        threshold (float, optional): ``threshold`` and ``threshold_mode`` will determine the minimum change of ``loss`` . 
            This make tiny changes of ``loss`` will be ignored. Default: 1e-4.
        threshold_mode (str, optional): ``'rel'`` or ``'abs'`` can be selected. In ``'rel'`` mode, the minimum change of ``loss``
            is ``last_loss * threshold`` , where ``last_loss`` is ``loss`` in last epoch. In ``'abs'`` mode, the minimum 
            change of ``loss`` is ``threshold`` . Default: ``'rel'`` .
        cooldown (int, optional): The number of epochs to wait before resuming normal operation. Default: 0.
        min_lr (float, optional): The lower bound of the learning rate after reduction. Default: 0.
        eps (float, optional): Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        dtype (str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. Default: 'float32'. 
    
    Returns:
        Reduced learning rate.

    Examples:
    
    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        with fluid.dygraph.guard():
            x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
            linear = fluid.dygraph.Linear(10, 10)
            input = fluid.dygraph.to_variable(x)

            reduce_lr = fluid.dygraph.ReduceLROnPlateau(
                                    learning_rate = 1.0,
                                    decay_rate = 0.5,
                                    patience = 5,
                                    verbose = True, 
                                    cooldown = 3)
            adam = fluid.optimizer.Adam(
                learning_rate = reduce_lr,
                parameter_list = linear.parameters())

            for epoch in range(10):
                total_loss = 0
                for bath_id in range(5):
                    out = linear(input)
                    loss = fluid.layers.reduce_mean(out)
                    total_loss += loss
                    adam.minimize(loss)
                
                avg_loss = total_loss/5

                # adjust learning rate according to avg_loss
                reduce_lr.step(avg_loss)
                lr = adam.current_step_lr()
                print("current avg_loss is %s, current lr is %s" % (avg_loss.numpy()[0], lr))

    """

    def __init__(self,
                 learning_rate,
                 mode='min',
                 decay_rate=0.1,
                 patience=10,
                 verbose=False,
                 threshold=1e-4,
                 threshold_mode='rel',
                 cooldown=0,
                 min_lr=0,
                 eps=1e-8,
                 dtype='float32'):
        super(ReduceLROnPlateau, self).__init__(dtype=dtype)
        mode = mode.lower()
        if mode not in ['min', 'max']:
            raise ValueError('mode ' + mode + ' is unknown!')
        self.mode = mode

        if decay_rate >= 1.0:
            raise ValueError(
                'new_lr = origin_lr * decay_rate and decay_rate should be < 1.0.'
            )
        self.decay_rate = self.create_lr_var(decay_rate)

        threshold_mode = threshold_mode.lower()
        if threshold_mode not in ['rel', 'abs']:
            raise ValueError('threshold mode ' + threshold_mode +
                             ' is unknown!')
        self.threshold_mode = threshold_mode
        check_type(learning_rate, 'learning_rate', (float, int, Variable),
                   'ReduceLROnPlateau')
        if not isinstance(learning_rate, (float, int, Variable)):
            raise TypeError(
                "The type of 'learning_rate' in 'ReduceLROnPlateau' must be 'float, int, Variable', but received %s."
                % type(learning_rate))

        self.learning_rate = learning_rate
        self.verbose = verbose
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = self.create_lr_var(min_lr)
        self.eps = eps

        self.cooldown_counter = 0
        self.best_loss = None
        self.num_bad_epochs = 0
        self.epoch_num = 0

    # "cooldown_counter / best_loss / num_bad_epochs / epoch_num / learning_rate" will be stored.
    def _state_keys(self):
        self.keys = [
            'cooldown_counter', 'best_loss', 'num_bad_epochs', 'epoch_num',
            'learning_rate'
        ]

    def __call__(self):
        if not isinstance(self.learning_rate, Variable):
            self.learning_rate = self.create_lr_var(self.learning_rate)
        return self.learning_rate

    def step(self, loss):
        """
        It should be invoked on each epoch. Update the learning rate in optimizer according to ``loss`` .  
        The new learning rate will take effect on next call to ``optimizer.minimize`` .

        Args:
            loss (Variable): A ``Variable`` that will be monitored to determine whether the learning rate will reduce. 
                If it stop descending for a ``patience`` number of epochs, the learning rate will reduce. It should 
                be 1-D Tensor with shape [1]. 
                Specially, if ``mode`` has been set to ``'max'`` ,  the learning rate will reduce when it stops ascending.
        Returns:
            None
        
        Examples:
            Please refer to the example of current LearningRateDecay.
        """

        # loss must be 1-D Tensor with shape [1]
        check_type(loss, 'loss', Variable, 'ReduceLROnPlateau.step')
        assert len(loss.shape) == 1 and loss.shape[0] == 1, "the loss.shape " \
            "should be (1L,), but the current loss.shape is {}. Maybe that "  \
            "you should call fluid.layers.mean to process it first.".format(loss.shape)

        self.epoch_num += 1
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        else:
            if self.best_loss is None or self._is_better(loss, self.best_loss):
                self.best_loss = loss
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs > self.patience:
                from .. import layers
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0
                new_lr = layers.elementwise_max(self.learning_rate *
                                                self.decay_rate, self.min_lr)
                if self.learning_rate - new_lr > self.eps:
                    if self.verbose:
                        old_lr = self.learning_rate.numpy()[0] if isinstance(
                            self.learning_rate,
                            Variable) else self.learning_rate
                        print('Epoch {}: reducing learning rate from {} to {}.'.
                              format(self.epoch_num, old_lr, new_lr.numpy()[0]))
                    self.learning_rate = new_lr

    def _is_better(self, current, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            return current < best - best * self.threshold

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return current < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            return current > best + best * self.threshold

        else:
            return current > best + self.threshold


class _LearningRateEpochDecay(LearningRateDecay):
    """
    :api_attr: imperative

    Base class of learning rate decay, which is updated each epoch.
    
    Define the common interface of an _LearningRateEpochDecay.
    User should not use this class directly,
    but need to use one of it's implementation. And invoke method: `epoch()` each epoch.
    """

    def __init__(self, learning_rate, dtype=None):
        if not isinstance(learning_rate, (float, int)):
            raise TypeError(
                "The type of 'learning_rate' must be 'float, int', but received %s."
                % type(learning_rate))
        if learning_rate < 0:
            raise ValueError("Invalid learning rate: {}".format(learning_rate))

        self.base_lr = float(learning_rate)

        self.epoch_num = -1
        self.dtype = dtype
        if dtype is None:
            self.dtype = "float32"
        self.learning_rate = self.create_lr_var(self.base_lr)

        self.epoch()

    # For those subclass who overload _LearningRateEpochDecay, "self.epoch_num/learning_rate" will be stored by default.
    # you can change it for your subclass.
    def _state_keys(self):
        self.keys = ['epoch_num', 'learning_rate']

    def __call__(self):
        """ 
        Return last computed learning rate on current epoch.
        """
        if not isinstance(self.learning_rate, Variable):
            self.learning_rate = self.create_lr_var(self.learning_rate)
        return self.learning_rate

    def epoch(self, epoch=None):
        """
        compueted learning_rate and update it when invoked.
        """
        if epoch is None:
            self.epoch_num += 1
        else:
            self.epoch_num = epoch

        self.learning_rate = self.get_lr()

    def get_lr(self):
        raise NotImplementedError


class StepDecay(_LearningRateEpochDecay):
    """
    :api_attr: imperative

    Decays the learning rate of ``optimizer`` by ``decay_rate`` every ``step_size`` number of epoch.

    The algorithm can be described as the code below. 

    .. code-block:: text

        learning_rate = 0.5
        step_size = 30
        decay_rate = 0.1

        learning_rate = 0.5     if epoch < 30
        learning_rate = 0.05    if 30 <= epoch < 60
        learning_rate = 0.005   if 60 <= epoch < 90
        ...

    Parameters:
        learning_rate (float|int): The initial learning rate. It can be set to python float or int number.
        step_size (int): Period of learning rate decay.
        decay_rate (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * decay_rate`` . 
            It should be less than 1.0. Default: 0.1.

    Returns:
        None.

    Examples:
        .. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np
            with fluid.dygraph.guard():
                x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
                linear = fluid.dygraph.Linear(10, 10)
                input = fluid.dygraph.to_variable(x)
                scheduler = fluid.dygraph.StepDecay(0.5, step_size=3)
                adam = fluid.optimizer.Adam(learning_rate = scheduler, parameter_list = linear.parameters())

                for epoch in range(9):
                    for batch_id in range(5):
                        out = linear(input)
                        loss = fluid.layers.reduce_mean(out)
                        adam.minimize(loss)  
                    scheduler.epoch()

                    print("epoch:{}, current lr is {}" .format(epoch, adam.current_step_lr()))
                    # epoch:0, current lr is 0.5
                    # epoch:1, current lr is 0.5
                    # epoch:2, current lr is 0.5
                    # epoch:3, current lr is 0.05
                    # epoch:4, current lr is 0.05
                    # epoch:5, current lr is 0.05
                    # epoch:6, current lr is 0.005
                    # epoch:7, current lr is 0.005
                    # epoch:8, current lr is 0.005

    """

    def __init__(self, learning_rate, step_size, decay_rate=0.1):
        if not isinstance(step_size, int):
            raise TypeError(
                "The type of 'step_size' must be 'int', but received %s." %
                type(step_size))
        if decay_rate >= 1.0:
            raise ValueError('decay_rate should be < 1.0.')

        self.step_size = step_size
        self.decay_rate = decay_rate
        super(StepDecay, self).__init__(learning_rate)

    def get_lr(self):
        decay_rate = self.create_lr_var(self.decay_rate)
        i = self.epoch_num // self.step_size
        return self.base_lr * (decay_rate**i)


class MultiStepDecay(_LearningRateEpochDecay):
    """
    :api_attr: imperative

    Decays the learning rate of ``optimizer`` by ``decay_rate`` once ``epoch`` reaches one of the milestones.

    The algorithm can be described as the code below. 

    .. code-block:: text

        learning_rate = 0.5
        milestones = [30, 50]
        decay_rate = 0.1
        if epoch < 30:
            learning_rate = 0.5
        elif epoch < 50:
            learning_rate = 0.05
        else:
            learning_rate = 0.005

    Parameters:
        learning_rate (float|int): The initial learning rate. It can be set to python float or int number.
        milestones (tuple|list): List or tuple of each boundaries. Must be increasing.
        decay_rate (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * decay_rate`` . 
            It should be less than 1.0. Default: 0.1.

    Returns:
        None.

    Examples:
        .. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np
            with fluid.dygraph.guard():
                x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
                linear = fluid.dygraph.Linear(10, 10)
                input = fluid.dygraph.to_variable(x)
                scheduler = fluid.dygraph.MultiStepDecay(0.5, milestones=[3, 5])
                adam = fluid.optimizer.Adam(learning_rate = scheduler, parameter_list = linear.parameters())

                for epoch in range(6):
                    for batch_id in range(5):
                        out = linear(input)
                        loss = fluid.layers.reduce_mean(out)
                        adam.minimize(loss)
                    scheduler.epoch()

                    print("epoch:{}, current lr is {}" .format(epoch, adam.current_step_lr()))
                    # epoch:0, current lr is 0.5
                    # epoch:1, current lr is 0.5
                    # epoch:2, current lr is 0.5
                    # epoch:3, current lr is 0.05
                    # epoch:4, current lr is 0.05
                    # epoch:5, current lr is 0.005

    """

    def __init__(self, learning_rate, milestones, decay_rate=0.1):
        if not isinstance(milestones, (tuple, list)):
            raise TypeError(
                "The type of 'milestones' in 'MultiStepDecay' must be 'tuple, list', but received %s."
                % type(milestones))

        if not all([
                milestones[i] < milestones[i + 1]
                for i in range(len(milestones) - 1)
        ]):
            raise ValueError('The elements of milestones must be incremented')
        if decay_rate >= 1.0:
            raise ValueError('decay_rate should be < 1.0.')

        self.milestones = milestones
        self.decay_rate = decay_rate
        super(MultiStepDecay, self).__init__(learning_rate)

    def get_lr(self):
        decay_rate = self.create_lr_var(self.decay_rate)
        for i in range(len(self.milestones)):
            if self.epoch_num < self.milestones[i]:
                return self.base_lr * (decay_rate**i)

        return self.base_lr * (decay_rate**len(self.milestones))


class LambdaDecay(_LearningRateEpochDecay):
    """
    :api_attr: imperative

    Sets the learning rate of ``optimizer`` to the initial lr times a multiplicative factor, and this multiplicative
    factor is computed by function ``lr_lambda`` . ``lr_lambda`` is funciton which receives ``epoch`` .

    The algorithm can be described as the code below. 

    .. code-block:: text

        learning_rate = 0.5        # init learning_rate
        lr_lambda = lambda epoch: 0.95 ** epoch

        learning_rate = 0.5        # epoch 0
        learning_rate = 0.475      # epoch 1
        learning_rate = 0.45125    # epoch 2

    Parameters:
        learning_rate (float|int): The initial learning rate. It can be set to python float or int number.
        lr_lambda (function): A function which computes a multiplicative factor given an integer parameter ``epoch`` , and 
            then multiply the initial learning rate by this multiplicative factor.
    
    Returns:
        None.

    Examples:
        .. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np
            with fluid.dygraph.guard():
                x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
                linear = fluid.dygraph.Linear(10, 10)
                input = fluid.dygraph.to_variable(x)
                scheduler = fluid.dygraph.LambdaDecay(0.5, lr_lambda=lambda x: 0.95**x)
                adam = fluid.optimizer.Adam(learning_rate = scheduler, parameter_list = linear.parameters())

                for epoch in range(6):
                    for batch_id in range(5):
                        out = linear(input)
                        loss = fluid.layers.reduce_mean(out)
                        adam.minimize(loss)
                    scheduler.epoch()

                    print("epoch:%d, current lr is %f" .format(epoch, adam.current_step_lr()))
                    # epoch:0, current lr is 0.5
                    # epoch:1, current lr is 0.475
                    # epoch:2, current lr is 0.45125

    """

    def __init__(self, learning_rate, lr_lambda):
        if not callable(lr_lambda):
            raise TypeError(
                "The type of 'lr_lambda' in 'LambdaDecay' must be 'function', but received %s."
                % type(lr_lambda))

        self.lr_lambda = lr_lambda
        super(LambdaDecay, self).__init__(learning_rate)

    def get_lr(self):
        base_lr = self.create_lr_var(self.base_lr)

        return self.base_lr * self.lr_lambda(self.epoch_num)
