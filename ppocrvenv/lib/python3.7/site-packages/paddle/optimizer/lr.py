# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

import math
import numpy
import warnings
from paddle import Tensor

__all__ = [ #noqa
    'LRScheduler',
    'NoamDecay',
    'PiecewiseDecay',
    'NaturalExpDecay',
    'InverseTimeDecay',
    'PolynomialDecay',
    'LinearWarmup',
    'ExponentialDecay',
    'MultiStepDecay',
    'StepDecay',
    'LambdaDecay',
    'ReduceOnPlateau',
    'CosineAnnealingDecay'
]


class LRScheduler(object):
    """

    LRScheduler Base class. Define the common interface of a learning rate scheduler.

    User can import it by ``from paddle.optimizer.lr import LRScheduler`` ,

    then overload it for your subclass and have a custom implementation of ``get_lr()`` .

    Otherwise, an ``NotImplementedError`` exception will be thrown.

    Args:
        learning_rate (float): The initial learning rate. It is a python float number.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .

    Returns:
        instance to schedule learning rate.

    Examples:
        Here is an example of a simple ``StepDecay`` implementation. 
        
        .. code-block:: python
            
            import paddle
            from paddle.optimizer.lr import LRScheduler

            class StepDecay(LRScheduler):
                def __init__(self,
                            learning_rate,
                            step_size,
                            gamma=0.1,
                            last_epoch=-1,
                            verbose=False):
                    if not isinstance(step_size, int):
                        raise TypeError(
                            "The type of 'step_size' must be 'int', but received %s." %
                            type(step_size))
                    if gamma >= 1.0:
                        raise ValueError('gamma should be < 1.0.')

                    self.step_size = step_size
                    self.gamma = gamma
                    super(StepDecay, self).__init__(learning_rate, last_epoch, verbose)

                def get_lr(self):
                    i = self.last_epoch // self.step_size
                    return self.base_lr * (self.gamma**i)

    """

    def __init__(self, learning_rate=0.1, last_epoch=-1, verbose=False):
        if not isinstance(learning_rate, (float, int)):
            raise TypeError(
                "The type of learning rate must be float, but received {}".
                format(type(learning_rate)))
        self.base_lr = float(learning_rate)
        self.last_lr = float(learning_rate)
        self.last_epoch = last_epoch
        self.verbose = verbose
        self._var_name = None

        self.step()

    def __call__(self):
        """ 
        Return lastest computed learning rate on current epoch.
        """
        return self.last_lr

    def step(self, epoch=None):
        """

        ``step`` should be called after ``optimizer.step`` . It will update the learning rate in optimizer according to current ``epoch`` .  
        The new learning rate will take effect on next ``optimizer.step`` .

        Args:
            epoch (int, None): specify current epoch. Default: None. Auto-increment from last_epoch=-1.

        Returns:
            None

        """
        if epoch is None:
            self.last_epoch += 1
            self.last_lr = self.get_lr()
        else:
            self.last_epoch = epoch
            if hasattr(self, "_get_closed_form_lr"):
                self.last_lr = self._get_closed_form_lr()
            else:
                self.last_lr = self.get_lr()

        if self.verbose:
            print('Epoch {}: {} set learning rate to {}.'.format(
                self.last_epoch, self.__class__.__name__, self.last_lr))

    def state_dict(self):
        """

        Returns the state of the scheduler as a :class:`dict`.

        It is a subset of ``self.__dict__`` .
        """
        self.state_keys()
        state_dict = {}
        for key in self.keys:
            if key not in self.__dict__:
                continue
            value = self.__dict__[key]
            if isinstance(value, Tensor):
                assert value.shape == [
                    1
                ], "shape of Tensor in state_dict must be [1] {}".format(
                    value.shape)
                value = value.numpy()[0]
            state_dict[key] = value

        return state_dict

    # For those subclass who overload LRScheduler, "last_epoch, last_lr" will be saved by default.
    # (Note): you can change it for your subclass.
    def state_keys(self):
        """

        For those subclass who overload ``LRScheduler`` (Base Class). Acquiescently, "last_epoch, last_lr" will be saved by ``self.keys = ['last_epoch', 'last_lr']`` .

        ``last_epoch`` is the current epoch num, and ``last_lr`` is the current learning rate.

        If you want to change the default behavior, you should have a custom implementation of ``_state_keys()`` to redefine ``self.keys`` .

        """
        self.keys = ['last_epoch', 'last_lr']

    def set_state_dict(self, state_dict):
        """

        Loads the schedulers state.
        """
        self.state_keys()
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

    # alias for set_state_dict
    set_dict = set_state_dict

    def get_lr(self):
        """
        
        For those subclass who overload ``LRScheduler`` (Base Class), User should have a custom implementation of ``get_lr()`` .

        Otherwise, an ``NotImplementedError`` exception will be thrown.
        """
        # calculate by python float
        raise NotImplementedError


class NoamDecay(LRScheduler):
    r"""

    Applies Noam Decay to the initial learning rate. 

    The algorithm can be described as following.

    .. math::

        new\_learning\_rate = learning\_rate * d_{model}^{-0.5} * min(epoch^{-0.5}, epoch * warmup\_steps^{-1.5})

    Please reference `attention is all you need <https://arxiv.org/pdf/1706.03762.pdf>`_ 


    Args:
        d$_{model}$(int): The dimensionality of input and output feature vector of model. It is a python int number.
        warmup_steps(int): The number of warmup steps. A super parameter. It is a python int number
        learning_rate (float): The initial learning rate. It is a python float number. Default: 1.0.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .

    Returns:
        ``NoamDecay`` instance to schedule learning rate.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            # train on default dynamic graph mode
            linear = paddle.nn.Linear(10, 10)
            scheduler = paddle.optimizer.lr.NoamDecay(d_model=0.01, warmup_steps=100, verbose=True)
            sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
            for epoch in range(20):
                for batch_id in range(5):
                    x = paddle.uniform([10, 10])
                    out = linear(x)
                    loss = paddle.mean(out)
                    loss.backward()
                    sgd.step()
                    sgd.clear_gradients()
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch

            # train on static graph mode
            paddle.enable_static()
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[None, 4, 5])
                y = paddle.static.data(name='y', shape=[None, 4, 5])
                z = paddle.static.nn.fc(x, 100)
                loss = paddle.mean(z)
                scheduler = paddle.optimizer.lr.NoamDecay(d_model=0.01, warmup_steps=100, verbose=True)
                sgd = paddle.optimizer.SGD(learning_rate=scheduler)
                sgd.minimize(loss)

            exe = paddle.static.Executor()
            exe.run(start_prog)
            for epoch in range(20):
                for batch_id in range(5):
                    out = exe.run(
                        main_prog,
                        feed={
                            'x': np.random.randn(3, 4, 5).astype('float32'),
                            'y': np.random.randn(3, 4, 5).astype('float32')
                        },
                        fetch_list=loss.name)
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch

    """

    def __init__(self,
                 d_model,
                 warmup_steps,
                 learning_rate=1.0,
                 last_epoch=-1,
                 verbose=False):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(NoamDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            a = 1
        else:
            a = self.last_epoch**-0.5
        b = self.warmup_steps**-1.5 * self.last_epoch
        return self.base_lr * (self.d_model**-0.5) * min(a, b)


class PiecewiseDecay(LRScheduler):
    """

    Piecewise learning rate scheduler.

    The algorithm can be described as the code below:

    .. code-block:: text

        boundaries = [100, 200]
        values = [1.0, 0.5, 0.1]
        if epoch < 100:
            learning_rate = 1.0
        elif 100 <= global_step < 200:
            learning_rate = 0.5
        else:
            learning_rate = 0.1

    Args:
        boundaries(list|tuple): A list/tuple of steps numbers. The type of element in the list is python int. 
        values(list|tuple): A list/tuple of learning rate values that will be picked during different epoch boundaries. 
            The type of element in the list is python float.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .

    Returns:
        ``PiecewiseDecay`` instance to schedule learning rate.

    Examples:
        
        .. code-block:: python

            import paddle
            import numpy as np

            # train on default dynamic graph mode
            linear = paddle.nn.Linear(10, 10)
            scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[3, 6, 9], values=[0.1, 0.2, 0.3, 0.4], verbose=True)
            sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
            for epoch in range(20):
                for batch_id in range(5):
                    x = paddle.uniform([10, 10])
                    out = linear(x)
                    loss = paddle.mean(out)
                    loss.backward()
                    sgd.step()
                    sgd.clear_gradients()
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch

            # train on static graph mode
            paddle.enable_static()
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[None, 4, 5])
                y = paddle.static.data(name='y', shape=[None, 4, 5])
                z = paddle.static.nn.fc(x, 100)
                loss = paddle.mean(z)
                scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[3, 6, 9], values=[0.1, 0.2, 0.3, 0.4], verbose=True)
                sgd = paddle.optimizer.SGD(learning_rate=scheduler)
                sgd.minimize(loss)

            exe = paddle.static.Executor()
            exe.run(start_prog)
            for epoch in range(20):
                for batch_id in range(5):
                    out = exe.run(
                        main_prog,
                        feed={
                            'x': np.random.randn(3, 4, 5).astype('float32'),
                            'y': np.random.randn(3, 4, 5).astype('float32')
                        },
                        fetch_list=loss.name)
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch
    """

    def __init__(self, boundaries, values, last_epoch=-1, verbose=False):
        self.boundaries = boundaries
        self.values = values
        super(PiecewiseDecay, self).__init__(
            last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        for i in range(len(self.boundaries)):
            if self.last_epoch < self.boundaries[i]:
                return self.values[i]
        return self.values[len(self.values) - 1]


class NaturalExpDecay(LRScheduler):
    r"""

    Applies natural exponential decay to the initial learning rate.
    
    The algorithm can be described as following:

    .. math::

        new\_learning\_rate = learning\_rate * e^{- gamma * epoch}

    Args:
        learning_rate (float): The initial learning rate. It is a python float number.
        gamma (float, optional): A Ratio to update the learning rate. Default: 0.1.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .

    Returns:
        ``NaturalExpDecay`` instance to schedule learning rate.

    Examples:
        
        .. code-block:: python

            import paddle
            import numpy as np

            # train on default dynamic graph mode
            linear = paddle.nn.Linear(10, 10)
            scheduler = paddle.optimizer.lr.NaturalExpDecay(learning_rate=0.5, gamma=0.1, verbose=True)
            sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
            for epoch in range(20):
                for batch_id in range(5):
                    x = paddle.uniform([10, 10])
                    out = linear(x)
                    loss = paddle.mean(out)
                    loss.backward()
                    sgd.step()
                    sgd.clear_gradients()
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch

            # train on static graph mode
            paddle.enable_static()
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[None, 4, 5])
                y = paddle.static.data(name='y', shape=[None, 4, 5])
                z = paddle.static.nn.fc(x, 100)
                loss = paddle.mean(z)
                scheduler = paddle.optimizer.lr.NaturalExpDecay(learning_rate=0.5, gamma=0.1, verbose=True)
                sgd = paddle.optimizer.SGD(learning_rate=scheduler)
                sgd.minimize(loss)

            exe = paddle.static.Executor()
            exe.run(start_prog)
            for epoch in range(20):
                for batch_id in range(5):
                    out = exe.run(
                        main_prog,
                        feed={
                            'x': np.random.randn(3, 4, 5).astype('float32'),
                            'y': np.random.randn(3, 4, 5).astype('float32')
                        },
                        fetch_list=loss.name)
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch
    """

    def __init__(self, learning_rate, gamma, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super(NaturalExpDecay, self).__init__(learning_rate, last_epoch,
                                              verbose)

    def get_lr(self):
        return self.base_lr * math.exp(-1 * self.gamma * self.last_epoch)


class InverseTimeDecay(LRScheduler):
    r"""

    Applies inverse time decay to the initial learning rate.

    The algorithm can be described as following:

    .. math::

        new\_learning\_rate = \frac{learning\_rate}{1 + gamma * epoch}

    Args:
        learning_rate (float): The initial learning rate. It is a python float number.
        gamma (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * gamma`` . 
            It should be less than 1.0. Default: 0.1.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .

    Returns:
        ``InverseTimeDecay`` instance to schedule learning rate.

    Examples:
        
        .. code-block:: python

            import paddle
            import numpy as np

            # train on default dynamic graph mode
            linear = paddle.nn.Linear(10, 10)
            scheduler = paddle.optimizer.lr.InverseTimeDecay(learning_rate=0.5, gamma=0.1, verbose=True)
            sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
            for epoch in range(20):
                for batch_id in range(5):
                    x = paddle.uniform([10, 10])
                    out = linear(x)
                    loss = paddle.mean(out)
                    loss.backward()
                    sgd.step()
                    sgd.clear_gradients()
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch

            # train on static graph mode
            paddle.enable_static()
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[None, 4, 5])
                y = paddle.static.data(name='y', shape=[None, 4, 5])
                z = paddle.static.nn.fc(x, 100)
                loss = paddle.mean(z)
                scheduler = paddle.optimizer.lr.InverseTimeDecay(learning_rate=0.5, gamma=0.1, verbose=True)
                sgd = paddle.optimizer.SGD(learning_rate=scheduler)
                sgd.minimize(loss)

            exe = paddle.static.Executor()
            exe.run(start_prog)
            for epoch in range(20):
                for batch_id in range(5):
                    out = exe.run(
                        main_prog,
                        feed={
                            'x': np.random.randn(3, 4, 5).astype('float32'),
                            'y': np.random.randn(3, 4, 5).astype('float32')
                        },
                        fetch_list=loss.name)
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch

    """

    def __init__(self, learning_rate, gamma, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super(InverseTimeDecay, self).__init__(learning_rate, last_epoch,
                                               verbose)

    def get_lr(self):
        return self.base_lr / (1 + self.gamma * self.last_epoch)


class PolynomialDecay(LRScheduler):
    r"""

    Applies polynomial decay to the initial learning rate.

    The algorithm can be described as following.

    If cycle is set to True, then:

    .. math::

        decay\_steps & = decay\_steps * math.ceil(\frac{epoch}{decay\_steps}) 

        new\_learning\_rate & = (learning\_rate-end\_lr)*(1-\frac{epoch}{decay\_steps})^{power}+end\_lr

    If cycle is set to False, then:

    .. math::

        epoch & = min(epoch, decay\_steps) 

        new\_learning\_rate & = (learning\_rate-end\_lr)*(1-\frac{epoch}{decay\_steps})^{power}+end\_lr


    Args:
        learning_rate (float): The initial learning rate. It is a python float number.
        decay_steps(int): The decay step size. It determines the decay cycle. It must be a positive integer.
        end_lr(float, optional): The minimum final learning rate. Default: 0.0001.
        power(float, optional): Power of polynomial. Default: 1.0.
        cycle(bool, optional): Whether the learning rate rises again. If True, then the learning rate will rise when it decrease 
            to ``end_lr`` .  If False, the learning rate is monotone decreasing. Default: False.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .

    Returns:
        ``PolynomialDecay`` instance to schedule learning rate.

    Examples:
        
        .. code-block:: python

            import paddle
            import numpy as np

            # train on default dynamic graph mode
            linear = paddle.nn.Linear(10, 10)
            scheduler = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.5, decay_steps=20, verbose=True)
            sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
            for epoch in range(20):
                for batch_id in range(5):
                    x = paddle.uniform([10, 10])
                    out = linear(x)
                    loss = paddle.mean(out)
                    loss.backward()
                    sgd.step()
                    sgd.clear_gradients()
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch

            # train on static graph mode
            paddle.enable_static()
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[None, 4, 5])
                y = paddle.static.data(name='y', shape=[None, 4, 5])
                z = paddle.static.nn.fc(x, 100)
                loss = paddle.mean(z)
                scheduler = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.5, decay_steps=20, verbose=True)
                sgd = paddle.optimizer.SGD(learning_rate=scheduler)
                sgd.minimize(loss)

            exe = paddle.static.Executor()
            exe.run(start_prog)
            for epoch in range(20):
                for batch_id in range(5):
                    out = exe.run(
                        main_prog,
                        feed={
                            'x': np.random.randn(3, 4, 5).astype('float32'),
                            'y': np.random.randn(3, 4, 5).astype('float32')
                        },
                        fetch_list=loss.name)
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch
    """

    def __init__(self,
                 learning_rate,
                 decay_steps,
                 end_lr=0.0001,
                 power=1.0,
                 cycle=False,
                 last_epoch=-1,
                 verbose=False):
        assert decay_steps > 0 and isinstance(
            decay_steps, int), " 'decay_steps' must be a positive integer."
        self.decay_steps = decay_steps
        self.end_lr = end_lr
        self.power = power
        self.cycle = cycle
        super(PolynomialDecay, self).__init__(learning_rate, last_epoch,
                                              verbose)

    def get_lr(self):
        tmp_epoch_num = self.last_epoch
        tmp_decay_steps = self.decay_steps
        if self.cycle:
            div_res = math.ceil(
                float(self.last_epoch) / float(self.decay_steps))

            if self.last_epoch == 0:
                div_res = 1
            tmp_decay_steps = self.decay_steps * div_res
        else:
            tmp_epoch_num = min(self.last_epoch, self.decay_steps)

        return (self.base_lr - self.end_lr) * (
            (1 - float(tmp_epoch_num) / float(tmp_decay_steps)
             )**self.power) + self.end_lr


class LinearWarmup(LRScheduler):
    r"""

    Linear learning rate warm up strategy. Update the learning rate preliminarily before the normal learning rate scheduler.
    For more information, please refer to `Bag of Tricks for Image Classification with Convolutional Neural Networks <https://arxiv.org/abs/1812.01187>`_
    
    When epoch < warmup_steps, learning rate is updated as:
    
    .. math::
    
            lr = start\_lr + (end\_lr - start\_lr) * \frac{epoch}{warmup\_steps}
    
    where start_lr is the initial learning rate, and end_lr is the final learning rate;
    
    When epoch >= warmup_steps, learning rate is updated as:
    
    .. math::
    
            lr = learning_rate
    
    where ``learning_rate`` is float or any subclass of ``LRScheduler`` .

    Args:
        learning_rate (float|LRScheduler): The learning rate after warm-up. It is a python float number or any subclass of ``LRScheduler`` .
        warmup_steps (int): total steps of warm up. It must be a positive integer.
        start_lr (float): Initial learning rate of warm up.
        end_lr (float): Final learning rate of warm up.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .

    Returns:
        ``LinearWarmup`` instance to schedule learning rate.

    Examples:
        
        .. code-block:: python

            import paddle
            import numpy as np

            # train on default dynamic graph mode
            linear = paddle.nn.Linear(10, 10)
            scheduler = paddle.optimizer.lr.LinearWarmup(
                    learning_rate=0.5, warmup_steps=20, start_lr=0, end_lr=0.5, verbose=True)
            sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
            for epoch in range(20):
                for batch_id in range(5):
                    x = paddle.uniform([10, 10])
                    out = linear(x)
                    loss = paddle.mean(out)
                    loss.backward()
                    sgd.step()
                    sgd.clear_gradients()
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch

            # train on static graph mode
            paddle.enable_static()
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[None, 4, 5])
                y = paddle.static.data(name='y', shape=[None, 4, 5])
                z = paddle.static.nn.fc(x, 100)
                loss = paddle.mean(z)
                scheduler = paddle.optimizer.lr.LinearWarmup(
                    learning_rate=0.5, warmup_steps=20, start_lr=0, end_lr=0.5, verbose=True)
                sgd = paddle.optimizer.SGD(learning_rate=scheduler)
                sgd.minimize(loss)

            exe = paddle.static.Executor()
            exe.run(start_prog)
            for epoch in range(20):
                for batch_id in range(5):
                    out = exe.run(
                        main_prog,
                        feed={
                            'x': np.random.randn(3, 4, 5).astype('float32'),
                            'y': np.random.randn(3, 4, 5).astype('float32')
                        },
                        fetch_list=loss.name)
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch
    """

    def __init__(self,
                 learning_rate,
                 warmup_steps,
                 start_lr,
                 end_lr,
                 last_epoch=-1,
                 verbose=False):
        type_check = isinstance(learning_rate, float) or isinstance(
            learning_rate, int) or isinstance(learning_rate, LRScheduler)
        if not type_check:
            raise TypeError(
                "the type of learning_rate should be [int, float or LRScheduler], the current type is {}".
                format(learning_rate))
        self.learning_rate = learning_rate
        assert warmup_steps > 0 and isinstance(
            warmup_steps, int), " 'warmup_steps' must be a positive integer."
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.end_lr = end_lr
        assert end_lr > start_lr, "end_lr {} must be greater than start_lr {}".format(
            end_lr, start_lr)
        super(LinearWarmup, self).__init__(start_lr, last_epoch, verbose)

    def state_dict(self):
        """
        Returns the state of the LinearWarmup scheduler as a :class:`dict`.

        It is a subset of ``self.__dict__`` .
        """
        state_dict = super(LinearWarmup, self).state_dict()
        if isinstance(self.learning_rate, LRScheduler):
            state_dict["LinearWarmup_LR"] = self.learning_rate.state_dict()
        return state_dict

    def set_state_dict(self, state_dict):
        """
        Loads state_dict for LinearWarmup scheduler.
        """
        super(LinearWarmup, self).set_state_dict(state_dict)
        if isinstance(self.learning_rate, LRScheduler):
            self.learning_rate.set_state_dict(state_dict["LinearWarmup_LR"])

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return (self.end_lr - self.start_lr) * float(
                self.last_epoch) / float(self.warmup_steps) + self.start_lr
        else:
            if isinstance(self.learning_rate, LRScheduler):
                self.learning_rate.step(self.last_epoch - self.warmup_steps)
                return self.learning_rate()

            return self.learning_rate


class ExponentialDecay(LRScheduler):
    r"""

    Update learning rate by `gamma` each epoch.

    The algorithm can be described as following.
    
    .. math::

        new\_learning\_rate = last\_learning\_rate * gamma

    Args:
        learning_rate (float): The initial learning rate. It is a python float number.
        gamma (float): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * gamma`` . 
            It should be less than 1.0.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .

    Returns:
        ``ExponentialDecay`` instance to schedule learning rate.

    Examples:
        
        .. code-block:: python

            import paddle
            import numpy as np

            # train on default dynamic graph mode
            linear = paddle.nn.Linear(10, 10)
            scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.5, gamma=0.9, verbose=True)
            sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
            for epoch in range(20):
                for batch_id in range(5):
                    x = paddle.uniform([10, 10])
                    out = linear(x)
                    loss = paddle.mean(out)
                    loss.backward()
                    sgd.step()
                    sgd.clear_gradients()
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch

            # train on static graph mode
            paddle.enable_static()
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[None, 4, 5])
                y = paddle.static.data(name='y', shape=[None, 4, 5])
                z = paddle.static.nn.fc(x, 100)
                loss = paddle.mean(z)
                scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.5, gamma=0.9, verbose=True)
                sgd = paddle.optimizer.SGD(learning_rate=scheduler)
                sgd.minimize(loss)

            exe = paddle.static.Executor()
            exe.run(start_prog)
            for epoch in range(20):
                for batch_id in range(5):
                    out = exe.run(
                        main_prog,
                        feed={
                            'x': np.random.randn(3, 4, 5).astype('float32'),
                            'y': np.random.randn(3, 4, 5).astype('float32')
                        },
                        fetch_list=loss.name)
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch
    """

    def __init__(self, learning_rate, gamma, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super(ExponentialDecay, self).__init__(learning_rate, last_epoch,
                                               verbose)

    def get_lr(self):
        return self.base_lr * (self.gamma**self.last_epoch)


class MultiStepDecay(LRScheduler):
    """
    Update the learning rate by ``gamma`` once ``epoch`` reaches one of the milestones.

    The algorithm can be described as the code below. 

    .. code-block:: text

        learning_rate = 0.5
        milestones = [30, 50]
        gamma = 0.1
        if epoch < 30:
            learning_rate = 0.5
        elif epoch < 50:
            learning_rate = 0.05
        else:
            learning_rate = 0.005

    Args:
        learning_rate (float): The initial learning rate. It is a python float number.
        milestones (tuple|list): List or tuple of each boundaries. Must be increasing.
        gamma (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * gamma`` . 
            It should be less than 1.0. Default: 0.1.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .
        

    Returns:
        ``MultiStepDecay`` instance to schedule learning rate.

    Examples:
        
        .. code-block:: python

            import paddle
            import numpy as np

            # train on default dynamic graph mode
            linear = paddle.nn.Linear(10, 10)
            scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.5, milestones=[2, 4, 6], gamma=0.8, verbose=True)
            sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
            for epoch in range(20):
                for batch_id in range(5):
                    x = paddle.uniform([10, 10])
                    out = linear(x)
                    loss = paddle.mean(out)
                    loss.backward()
                    sgd.step()
                    sgd.clear_gradients()
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch

            # train on static graph mode
            paddle.enable_static()
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[None, 4, 5])
                y = paddle.static.data(name='y', shape=[None, 4, 5])
                z = paddle.static.nn.fc(x, 100)
                loss = paddle.mean(z)
                scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.5, milestones=[2, 4, 6], gamma=0.8, verbose=True)
                sgd = paddle.optimizer.SGD(learning_rate=scheduler)
                sgd.minimize(loss)

            exe = paddle.static.Executor()
            exe.run(start_prog)
            for epoch in range(20):
                for batch_id in range(5):
                    out = exe.run(
                        main_prog,
                        feed={
                            'x': np.random.randn(3, 4, 5).astype('float32'),
                            'y': np.random.randn(3, 4, 5).astype('float32')
                        },
                        fetch_list=loss.name)
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch
    """

    def __init__(self,
                 learning_rate,
                 milestones,
                 gamma=0.1,
                 last_epoch=-1,
                 verbose=False):
        if not isinstance(milestones, (tuple, list)):
            raise TypeError(
                "The type of 'milestones' in 'MultiStepDecay' must be 'tuple, list', but received %s."
                % type(milestones))

        if not all([
                milestones[i] < milestones[i + 1]
                for i in range(len(milestones) - 1)
        ]):
            raise ValueError('The elements of milestones must be incremented')
        if gamma >= 1.0:
            raise ValueError('gamma should be < 1.0.')

        self.milestones = milestones
        self.gamma = gamma
        super(MultiStepDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        for i in range(len(self.milestones)):
            if self.last_epoch < self.milestones[i]:
                return self.base_lr * (self.gamma**i)
        return self.base_lr * (self.gamma**len(self.milestones))


class StepDecay(LRScheduler):
    """
    Update the learning rate of ``optimizer`` by ``gamma`` every ``step_size`` number of epoch.

    The algorithm can be described as the code below. 

    .. code-block:: text

        learning_rate = 0.5
        step_size = 30
        gamma = 0.1

        learning_rate = 0.5     if epoch < 30
        learning_rate = 0.05    if 30 <= epoch < 60
        learning_rate = 0.005   if 60 <= epoch < 90
        ...

    Args:
        learning_rate (float): The initial learning rate. It is a python float number.
        step_size (int): the interval to update. It must be a positive integer.
        gamma (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * gamma`` . 
            It should be less than 1.0. Default: 0.1.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .

    Returns:
        ``StepDecay`` instance to schedule learning rate.


    Examples:
        
        .. code-block:: python

            import paddle
            import numpy as np

            # train on default dynamic graph mode
            linear = paddle.nn.Linear(10, 10)
            scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=5, gamma=0.8, verbose=True)
            sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
            for epoch in range(20):
                for batch_id in range(5):
                    x = paddle.uniform([10, 10])
                    out = linear(x)
                    loss = paddle.mean(out)
                    loss.backward()
                    sgd.step()
                    sgd.clear_gradients()
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch

            # train on static graph mode
            paddle.enable_static()
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[None, 4, 5])
                y = paddle.static.data(name='y', shape=[None, 4, 5])
                z = paddle.static.nn.fc(x, 100)
                loss = paddle.mean(z)
                scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=5, gamma=0.8, verbose=True)
                sgd = paddle.optimizer.SGD(learning_rate=scheduler)
                sgd.minimize(loss)

            exe = paddle.static.Executor()
            exe.run(start_prog)
            for epoch in range(20):
                for batch_id in range(5):
                    out = exe.run(
                        main_prog,
                        feed={
                            'x': np.random.randn(3, 4, 5).astype('float32'),
                            'y': np.random.randn(3, 4, 5).astype('float32')
                        },
                        fetch_list=loss.name)
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch
    """

    def __init__(self,
                 learning_rate,
                 step_size,
                 gamma=0.1,
                 last_epoch=-1,
                 verbose=False):
        if not isinstance(step_size, int):
            raise TypeError(
                "The type of 'step_size' must be 'int', but received %s." %
                type(step_size))
        if gamma >= 1.0:
            raise ValueError('gamma should be < 1.0.')

        assert step_size > 0 and isinstance(
            step_size, int), " 'step_size' must be a positive integer."
        self.step_size = step_size
        self.gamma = gamma
        super(StepDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        i = self.last_epoch // self.step_size
        return self.base_lr * (self.gamma**i)


class LambdaDecay(LRScheduler):
    """
    Sets the learning rate of ``optimizer`` by function ``lr_lambda`` . ``lr_lambda`` is funciton which receives ``epoch`` .

    The algorithm can be described as the code below. 

    .. code-block:: text

        learning_rate = 0.5        # init learning_rate
        lr_lambda = lambda epoch: 0.95 ** epoch

        learning_rate = 0.5        # epoch 0, 0.5*0.95**0
        learning_rate = 0.475      # epoch 1, 0.5*0.95**1
        learning_rate = 0.45125    # epoch 2, 0.5*0.95**2

    Args:
        learning_rate (float): The initial learning rate. It is a python float number.
        lr_lambda (function): A function which computes a factor by ``epoch`` , and then multiply the initial learning rate by this factor.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .
    
    Returns:
        ``LambdaDecay`` instance to schedule learning rate.

    Examples:
        
        .. code-block:: python

            import paddle
            import numpy as np

            # train on default dynamic graph mode
            linear = paddle.nn.Linear(10, 10)
            scheduler = paddle.optimizer.lr.LambdaDecay(learning_rate=0.5, lr_lambda=lambda x:0.95**x, verbose=True)
            sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
            for epoch in range(20):
                for batch_id in range(5):
                    x = paddle.uniform([10, 10])
                    out = linear(x)
                    loss = paddle.mean(out)
                    loss.backward()
                    sgd.step()
                    sgd.clear_gradients()
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch

            # train on static graph mode
            paddle.enable_static()
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[None, 4, 5])
                y = paddle.static.data(name='y', shape=[None, 4, 5])
                z = paddle.static.nn.fc(x, 100)
                loss = paddle.mean(z)
                scheduler = paddle.optimizer.lr.LambdaDecay(learning_rate=0.5, lr_lambda=lambda x:0.95**x, verbose=True)
                sgd = paddle.optimizer.SGD(learning_rate=scheduler)
                sgd.minimize(loss)

            exe = paddle.static.Executor()
            exe.run(start_prog)
            for epoch in range(20):
                for batch_id in range(5):
                    out = exe.run(
                        main_prog,
                        feed={
                            'x': np.random.randn(3, 4, 5).astype('float32'),
                            'y': np.random.randn(3, 4, 5).astype('float32')
                        },
                        fetch_list=loss.name)
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch

    """

    def __init__(self, learning_rate, lr_lambda, last_epoch=-1, verbose=False):
        if not callable(lr_lambda):
            raise TypeError(
                "The type of 'lr_lambda' in 'LambdaDecay' must be 'function', but received %s."
                % type(lr_lambda))

        self.lr_lambda = lr_lambda
        super(LambdaDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        return self.base_lr * self.lr_lambda(self.last_epoch)


class ReduceOnPlateau(LRScheduler):
    """
    Reduce learning rate when ``metrics`` has stopped descending. Models often benefit from reducing the learning rate 
    by 2 to 10 times once model performance has no longer improvement.

    The ``metrics`` is the one which has been pass into ``step`` , it must be 1-D Tensor with shape [1]. When ``metrics`` 
    stop descending for a ``patience`` number of epochs, the learning rate will be reduced to ``learning_rate * factor`` . 
    (Specially, ``mode`` can also be set to ``'max`` , in this case, when ``metrics`` stop ascending for a ``patience`` 
    number of epochs, the learning rate will be reduced.)

    In addition, After each reduction, it will wait a ``cooldown`` number of epochs before resuming above operation.

    Args:
        learning_rate (float): The initial learning rate. It is a python float number.
        mode (str, optional): ``'min'`` or ``'max'`` can be selected. Normally, it is ``'min'`` , which means that the 
            learning rate will reduce when ``loss`` stops descending. Specially, if it's set to ``'max'`` ,  the learning 
            rate will reduce when ``loss`` stops ascending. Default: ``'min'`` .
        factor (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * factor`` . 
            It should be less than 1.0. Default: 0.1.
        patience (int, optional): When ``loss`` doesn't improve for this number of epochs, learing rate will be reduced. 
            Default: 10.
        threshold (float, optional): ``threshold`` and ``threshold_mode`` will determine the minimum change of ``loss`` . 
            This make tiny changes of ``loss`` will be ignored. Default: 1e-4.
        threshold_mode (str, optional): ``'rel'`` or ``'abs'`` can be selected. In ``'rel'`` mode, the minimum change of ``loss``
            is ``last_loss * threshold`` , where ``last_loss`` is ``loss`` in last epoch. In ``'abs'`` mode, the minimum 
            change of ``loss`` is ``threshold`` . Default: ``'rel'`` .
        cooldown (int, optional): The number of epochs to wait before resuming normal operation. Default: 0.
        min_lr (float, optional): The lower bound of the learning rate after reduction. Default: 0.
        epsilon (float, optional): Minimal decay applied to lr. If the difference between new and old lr is smaller than epsilon, 
            the update is ignored. Default: 1e-8.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False``.

    
    Returns:
        ``ReduceOnPlateau`` instance to schedule learning rate.


    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            # train on default dynamic graph mode
            linear = paddle.nn.Linear(10, 10)
            scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=1.0, factor=0.5, patience=5, verbose=True)
            sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
            for epoch in range(20):
                for batch_id in range(5):
                    x = paddle.uniform([10, 10])
                    out = linear(x)
                    loss = paddle.mean(out)
                    loss.backward()
                    sgd.step()
                    sgd.clear_gradients()
                    scheduler.step(loss)    # If you update learning rate each step
              # scheduler.step(loss)        # If you update learning rate each epoch

            # train on static graph mode
            paddle.enable_static()
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[None, 4, 5])
                y = paddle.static.data(name='y', shape=[None, 4, 5])
                z = paddle.static.nn.fc(x, 100)
                loss = paddle.mean(z)
                scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=1.0, factor=0.5, patience=5, verbose=True)
                sgd = paddle.optimizer.SGD(learning_rate=scheduler)
                sgd.minimize(loss)

            exe = paddle.static.Executor()
            exe.run(start_prog)
            for epoch in range(20):
                for batch_id in range(5):
                    out = exe.run(
                        main_prog,
                        feed={
                            'x': np.random.randn(3, 4, 5).astype('float32'),
                            'y': np.random.randn(3, 4, 5).astype('float32')
                        },
                        fetch_list=loss.name)
                    scheduler.step(out[0])    # If you update learning rate each step
              # scheduler.step(out[0])        # If you update learning rate each epoch

    """

    def __init__(self,
                 learning_rate,
                 mode='min',
                 factor=0.1,
                 patience=10,
                 threshold=1e-4,
                 threshold_mode='rel',
                 cooldown=0,
                 min_lr=0,
                 epsilon=1e-8,
                 verbose=False):
        mode = mode.lower()
        if mode not in ['min', 'max']:
            raise ValueError('mode: ' + mode + ' is unknown!')
        self.mode = mode

        if factor >= 1.0:
            raise ValueError(
                'new_lr = origin_lr * gamma and gamma should be < 1.0.')
        self.factor = factor

        threshold_mode = threshold_mode.lower()
        if threshold_mode not in ['rel', 'abs']:
            raise ValueError('threshold mode: ' + threshold_mode +
                             ' is unknown!')
        self.threshold_mode = threshold_mode
        if not isinstance(learning_rate, (float, int)):
            raise TypeError(
                "The type of 'learning_rate' in 'ReduceOnPlateau' must be 'float', but received %s."
                % type(learning_rate))

        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.epsilon = epsilon

        self.cooldown_counter = 0
        self.best = None
        self.num_bad_epochs = 0

        # Can not call Parent __init__, so implement here.
        self.base_lr = float(learning_rate)
        self.last_lr = float(learning_rate)
        self.last_epoch = 0
        self.verbose = verbose
        self._var_name = None

    # "cooldown_counter / best / num_bad_epochs / last_epoch / last_lr" will be stored.
    def state_keys(self):
        self.keys = [
            'cooldown_counter', 'best', 'num_bad_epochs', 'last_epoch',
            'last_lr'
        ]

    def step(self, metrics, epoch=None):
        """
        step should be called after `optimizer.step()` . It will update the learning rate in optimizer according to ``metrics`` .  
        The new learning rate will take effect on next epoch.

        Args:
            metrics (Tensor|numpy.ndarray|float): Which will be monitored to determine whether the learning rate will reduce. 
                If it stop descending for a ``patience`` number of epochs, the learning rate will reduce. If it's 'Tensor' or
                'numpy.ndarray', its shape must be [1].
            epoch (int, None): specify current epoch. Default: None. Auto-increment from last_epoch=-1.

        Returns:
            None
        
        Examples:
            Please refer to the example of current LRScheduler.
        """
        if epoch is None:
            self.last_epoch = self.last_epoch + 1
        else:
            self.last_epoch = epoch

        # loss must be float, numpy.ndarray or 1-D Tensor with shape [1]
        if isinstance(metrics, (Tensor, numpy.ndarray)):
            assert len(metrics.shape) == 1 and metrics.shape[0] == 1, "the metrics.shape " \
                "should be (1L,), but the current metrics.shape is {}. Maybe that "  \
                "you should call paddle.mean to process it first.".format(metrics.shape)
        elif not isinstance(metrics,
                            (int, float, numpy.float32, numpy.float64)):
            raise TypeError(
                "metrics must be 'int', 'float', 'np.float', 'numpy.ndarray' or 'paddle.Tensor', but receive {}".
                format(type(metrics)))

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        else:
            if self.best is None or self._is_better(metrics, self.best):
                self.best = metrics
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs > self.patience:
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0
                new_lr = max(self.last_lr * self.factor, self.min_lr)
                if self.last_lr - new_lr > self.epsilon:
                    self.last_lr = new_lr
                    if self.verbose:
                        print('Epoch {}: {} set learning rate to {}.'.format(
                            self.last_epoch, self.__class__.__name__,
                            self.last_lr))

    def _is_better(self, current, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            return current < best - best * self.threshold

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return current < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            return current > best + best * self.threshold

        else:
            return current > best + self.threshold


class CosineAnnealingDecay(LRScheduler):
    r"""

    Set the learning rate using a cosine annealing schedule, where :math:`\eta_{max}` is set to 
    the initial learning_rate. :math:`T_{cur}` is the number of epochs since the last restart in 
    SGDR.

    The algorithm can be described as following.

    .. math::

        \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
        + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
        & T_{cur} \neq (2k+1)T_{max}; 

        \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
        \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
        & T_{cur} = (2k+1)T_{max}.
    
    It has been proposed in `SGDR: Stochastic Gradient Descent with Warm Restarts <https://arxiv.org/abs/1608.03983>`_. 
    Note that this only implements the cosine annealing part of SGDR, and not the restarts.
    
    Args:
        learning_rate (float): The initial learning rate, that is :math:`\eta_{max}` . It can be set to python float or int number.
        T_max (int): Maximum number of iterations. It is half of the decay cycle of learning rate. It must be a positive integer.
        eta_min (float|int, optional): Minimum learning rate, that is :math:`\eta_{min}` . Default: 0.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .

    Returns:
        ``CosineAnnealingDecay`` instance to schedule learning rate.

    Examples:
        
        .. code-block:: python

            import paddle
            import numpy as np

            # train on default dynamic graph mode
            linear = paddle.nn.Linear(10, 10)
            scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.5, T_max=10, verbose=True)
            sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
            for epoch in range(20):
                for batch_id in range(5):
                    x = paddle.uniform([10, 10])
                    out = linear(x)
                    loss = paddle.mean(out)
                    loss.backward()
                    sgd.step()
                    sgd.clear_gradients()
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch

            # train on static graph mode
            paddle.enable_static()
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[None, 4, 5])
                y = paddle.static.data(name='y', shape=[None, 4, 5])
                z = paddle.static.nn.fc(x, 100)
                loss = paddle.mean(z)
                scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.5, T_max=10, verbose=True)
                sgd = paddle.optimizer.SGD(learning_rate=scheduler)
                sgd.minimize(loss)

            exe = paddle.static.Executor()
            exe.run(start_prog)
            for epoch in range(20):
                for batch_id in range(5):
                    out = exe.run(
                        main_prog,
                        feed={
                            'x': np.random.randn(3, 4, 5).astype('float32'),
                            'y': np.random.randn(3, 4, 5).astype('float32')
                        },
                        fetch_list=loss.name)
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch
    """

    def __init__(self,
                 learning_rate,
                 T_max,
                 eta_min=0,
                 last_epoch=-1,
                 verbose=False):
        if not isinstance(T_max, int):
            raise TypeError(
                "The type of 'T_max' in 'CosineAnnealingDecay' must be 'int', but received %s."
                % type(T_max))
        if not isinstance(eta_min, (float, int)):
            raise TypeError(
                "The type of 'eta_min' in 'CosineAnnealingDecay' must be 'float, int', but received %s."
                % type(eta_min))
        assert T_max > 0 and isinstance(
            T_max, int), " 'T_max' must be a positive integer."
        self.T_max = T_max
        self.eta_min = float(eta_min)
        super(CosineAnnealingDecay, self).__init__(learning_rate, last_epoch,
                                                   verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lr
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return self.last_lr + (self.base_lr - self.eta_min) * (1 - math.cos(
                math.pi / self.T_max)) / 2

        return (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / (
            1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) * (
                self.last_lr - self.eta_min) + self.eta_min

    def _get_closed_form_lr(self):
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(
            math.pi * self.last_epoch / self.T_max)) / 2
