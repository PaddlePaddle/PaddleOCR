import sys
import math
from paddle.optimizer.lr import LinearWarmup
from paddle.optimizer.lr import PiecewiseDecay
from paddle.optimizer.lr import CosineAnnealingDecay
from paddle.optimizer.lr import ExponentialDecay
import paddle
import paddle.regularizer as regularizer
from copy import deepcopy


class Cosine(CosineAnnealingDecay):
    """
    Cosine learning rate decay
    lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
    """

    def __init__(self, lr, step_each_epoch, epochs, **kwargs):
        super(Cosine, self).__init__(
            learning_rate=lr,
            T_max=step_each_epoch * epochs, )

        self.update_specified = False


class Piecewise(PiecewiseDecay):
    """
    Piecewise learning rate decay
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        decay_epochs(list): piecewise decay epochs
        gamma(float): decay factor
    """

    def __init__(self, lr, step_each_epoch, decay_epochs, gamma=0.1, **kwargs):
        boundaries = [step_each_epoch * e for e in decay_epochs]
        lr_values = [lr * (gamma**i) for i in range(len(boundaries) + 1)]
        super(Piecewise, self).__init__(boundaries=boundaries, values=lr_values)

        self.update_specified = False


class CosineWarmup(LinearWarmup):
    """
    Cosine learning rate decay with warmup
    [0, warmup_epoch): linear warmup
    [warmup_epoch, epochs): cosine decay
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        warmup_epoch(int): epoch num of warmup
    """

    def __init__(self, lr, step_each_epoch, epochs, warmup_epoch=5, **kwargs):
        assert epochs > warmup_epoch, "total epoch({}) should be larger than warmup_epoch({}) in CosineWarmup.".format(
            epochs, warmup_epoch)
        warmup_step = warmup_epoch * step_each_epoch
        start_lr = 0.0
        end_lr = lr
        lr_sch = Cosine(lr, step_each_epoch, epochs - warmup_epoch)

        super(CosineWarmup, self).__init__(
            learning_rate=lr_sch,
            warmup_steps=warmup_step,
            start_lr=start_lr,
            end_lr=end_lr)

        self.update_specified = False


class ExponentialWarmup(LinearWarmup):
    """
    Exponential learning rate decay with warmup
    [0, warmup_epoch): linear warmup
    [warmup_epoch, epochs): Exponential decay
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        decay_epochs(float): decay epochs
        decay_rate(float): decay rate
        warmup_epoch(int): epoch num of warmup
    """

    def __init__(self,
                 lr,
                 step_each_epoch,
                 decay_epochs=2.4,
                 decay_rate=0.97,
                 warmup_epoch=5,
                 **kwargs):
        warmup_step = warmup_epoch * step_each_epoch
        start_lr = 0.0
        end_lr = lr
        lr_sch = ExponentialDecay(lr, decay_rate)

        super(ExponentialWarmup, self).__init__(
            learning_rate=lr_sch,
            warmup_steps=warmup_step,
            start_lr=start_lr,
            end_lr=end_lr)

        # NOTE: hac method to update exponential lr scheduler
        self.update_specified = True
        self.update_start_step = warmup_step
        self.update_step_interval = int(decay_epochs * step_each_epoch)
        self.step_each_epoch = step_each_epoch


class LearningRateBuilder():
    """
    Build learning rate variable
    https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/layers_cn.html
    Args:
        function(str): class name of learning rate
        params(dict): parameters used for init the class
    """

    def __init__(self,
                 function='Linear',
                 params={'lr': 0.1,
                         'steps': 100,
                         'end_lr': 0.0}):
        self.function = function
        self.params = params

    def __call__(self):
        mod = sys.modules[__name__]
        lr = getattr(mod, self.function)(**self.params)
        return lr


class L1Decay(object):
    """
    L1 Weight Decay Regularization, which encourages the weights to be sparse.
    Args:
        factor(float): regularization coeff. Default:0.0.
    """

    def __init__(self, factor=0.0):
        super(L1Decay, self).__init__()
        self.factor = factor

    def __call__(self):
        reg = regularizer.L1Decay(self.factor)
        return reg


class L2Decay(object):
    """
    L2 Weight Decay Regularization, which encourages the weights to be sparse.
    Args:
        factor(float): regularization coeff. Default:0.0.
    """

    def __init__(self, factor=0.0):
        super(L2Decay, self).__init__()
        self.factor = factor

    def __call__(self):
        reg = regularizer.L2Decay(self.factor)
        return reg


class Momentum(object):
    """
    Simple Momentum optimizer with velocity state.
    Args:
        learning_rate (float|Variable) - The learning rate used to update parameters.
            Can be a float value or a Variable with one float value as data element.
        momentum (float) - Momentum factor.
        regularization (WeightDecayRegularizer, optional) - The strategy of regularization.
    """

    def __init__(self,
                 learning_rate,
                 momentum,
                 parameter_list=None,
                 regularization=None,
                 **args):
        super(Momentum, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.parameter_list = parameter_list
        self.regularization = regularization

    def __call__(self):
        opt = paddle.optimizer.Momentum(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            parameters=self.parameter_list,
            weight_decay=self.regularization)
        return opt


class RMSProp(object):
    """
    Root Mean Squared Propagation (RMSProp) is an unpublished, adaptive learning rate method.
    Args:
        learning_rate (float|Variable) - The learning rate used to update parameters.
            Can be a float value or a Variable with one float value as data element.
        momentum (float) - Momentum factor.
        rho (float) - rho value in equation.
        epsilon (float) - avoid division by zero, default is 1e-6.
        regularization (WeightDecayRegularizer, optional) - The strategy of regularization.
    """

    def __init__(self,
                 learning_rate,
                 momentum,
                 rho=0.95,
                 epsilon=1e-6,
                 parameter_list=None,
                 regularization=None,
                 **args):
        super(RMSProp, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rho = rho
        self.epsilon = epsilon
        self.parameter_list = parameter_list
        self.regularization = regularization

    def __call__(self):
        opt = paddle.optimizer.RMSProp(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            rho=self.rho,
            epsilon=self.epsilon,
            parameters=self.parameter_list,
            weight_decay=self.regularization)
        return opt


class OptimizerBuilder(object):
    """
    Build optimizer
    Args:
        function(str): optimizer name of learning rate
        params(dict): parameters used for init the class
        regularizer (dict): parameters used for create regularization
    """

    def __init__(self,
                 function='Momentum',
                 params={'momentum': 0.9},
                 regularizer=None):
        self.function = function
        self.params = params
        # create regularizer
        if regularizer is not None:
            mod = sys.modules[__name__]
            reg_func = regularizer['function'] + 'Decay'
            del regularizer['function']
            reg = getattr(mod, reg_func)(**regularizer)()
            self.params['regularization'] = reg

    def __call__(self, learning_rate, parameter_list=None):
        mod = sys.modules[__name__]
        opt = getattr(mod, self.function)
        return opt(learning_rate=learning_rate,
                   parameter_list=parameter_list,
                   **self.params)()


def create_optimizer(config, parameter_list=None):
    """
    Create an optimizer using config, usually including
    learning rate and regularization.

    Args:
        config(dict):  such as
        {
            'LEARNING_RATE':
                {'function': 'Cosine',
                 'params': {'lr': 0.1}
                },
            'OPTIMIZER':
                {'function': 'Momentum',
                 'params':{'momentum': 0.9},
                 'regularizer':
                    {'function': 'L2', 'factor': 0.0001}
                }
        }

    Returns:
        an optimizer instance
    """
    # create learning_rate instance
    lr_config = config['LEARNING_RATE']
    lr_config['params'].update({
        'epochs': config['epoch'],
        'step_each_epoch':
        config['total_images'] // config['TRAIN']['batch_size'],
    })
    lr = LearningRateBuilder(**lr_config)()

    # create optimizer instance
    opt_config = deepcopy(config['OPTIMIZER'])

    opt = OptimizerBuilder(**opt_config)
    return opt(lr, parameter_list), lr


def create_multi_optimizer(config, parameter_list=None):
    """
    """
    # create learning_rate instance
    lr_config = config['LEARNING_RATE']
    lr_config['params'].update({
        'epochs': config['epoch'],
        'step_each_epoch':
        config['total_images'] // config['TRAIN']['batch_size'],
    })
    lr = LearningRateBuilder(**lr_config)()

    # create optimizer instance
    opt_config = deepcopy.copy(config['OPTIMIZER'])
    opt = OptimizerBuilder(**opt_config)
    return opt(lr, parameter_list), lr
