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

import six
import warnings
import sys

from .initializer import Initializer, Xavier, Constant
from .regularizer import WeightDecayRegularizer
from paddle.fluid.data_feeder import check_type

__all__ = [
    'ParamAttr',
    'WeightNormParamAttr',
]


class ParamAttr(object):
    """
    Create a object to represent the attribute of parameter. The attributes are:
    name, initializer, learning rate, regularizer, trainable, gradient clip,
    and model average.
    
    Note:
        ``gradient_clip`` of ``ParamAttr`` HAS BEEN DEPRECATED since 2.0. 
        Please use ``need_clip`` in ``ParamAttr`` to speficiy the clip scope.
        There are three clipping strategies: :ref:`api_paddle_nn_ClipGradByGlobalNorm` , 
        :ref:`api_paddle_nn_ClipGradByNorm` , :ref:`api_paddle_nn_ClipGradByValue` .

    Parameters:
        name (str, optional): The parameter's name. Default None, meaning that the name
                would be created automatically.
        initializer (Initializer, optional): The method to initial this parameter. Default
                None, meaning that the weight parameter is initialized by Xavier initializer,
                and the bias parameter is initialized by 0.
        learning_rate (float, optional): The parameter's learning rate. The learning rate when
                optimize is the global learning rates times the parameter's learning rate times
                the factor of learning rate scheduler. Default 1.0.
        regularizer (WeightDecayRegularizer, optional): Regularization strategy. There are two method: 
                :ref:`api_paddle_regularizer_L1Decay` , :ref:`api_paddle_regularizer_L2Decay` . If 
                regularizer is also set in ``optimizer`` (such as :ref:`api_paddle_optimizer_SGD` ), 
                that regularizer setting in optimizer will be ignored. Default None, meaning there is 
                no regularization.
        trainable (bool, optional): Whether this parameter is trainable. Default True.
        do_model_average (bool, optional): Whether this parameter should do model average
                when model average is enabled. Only used in ExponentialMovingAverage. Default True.
        need_clip (bool, optional): Whether the parameter gradient need to be cliped in optimizer. Default is True.

    Returns:
       ParamAttr Object.

    Examples:
        .. code-block:: python

            import paddle

            weight_attr = paddle.ParamAttr(name="weight",
                                           learning_rate=0.5,
                                           regularizer=paddle.regularizer.L2Decay(1.0),
                                           trainable=True)
            print(weight_attr.name) # "weight"
            paddle.nn.Linear(3, 4, weight_attr=weight_attr)
    """

    def __init__(self,
                 name=None,
                 initializer=None,
                 learning_rate=1.0,
                 regularizer=None,
                 trainable=True,
                 do_model_average=True,
                 need_clip=True):

        if sys.version_info.major == 2:
            check_type(name, "name", (str, type(None), unicode), "ParamAttr")
        else:
            check_type(name, "name", (str, type(None)), "ParamAttr")
        check_type(learning_rate, "learning_rate", (float, int), "ParamAttr")
        check_type(trainable, "trainable", (bool), "ParamAttr")
        check_type(do_model_average, "do_model_average", (bool), "ParamAttr")
        check_type(need_clip, "need_clip", (bool), "ParamAttr")
        check_type(initializer, "initializer", (Initializer, type(None)),
                   "ParamAttr")
        check_type(regularizer, "regularizer",
                   (WeightDecayRegularizer, type(None)), "ParamAttr")

        self.name = name
        if self.name == "":
            raise ValueError("name of ParamAttr can not be empty str")

        self.initializer = initializer
        self.learning_rate = learning_rate
        self.regularizer = regularizer
        self.trainable = trainable
        self.do_model_average = do_model_average
        self.need_clip = need_clip

    def _set_default_initializer(self, initializer):
        """
        Set the default initializer, the initializer should be Constant,
        Uniform, Normal, Xavier, MSRA.

        Args:
            initializer(Initializer): the initializer to set.

        Returns:
            None
        """
        if initializer is None:
            if self.initializer is None:
                raise ValueError("ParamAttr.initializer is not set")
            return

        if self.initializer is not None:
            return

        self.initializer = initializer

    def _set_default_param_initializer(self):
        """
        Set the default initializer for the parameter with Xavier.

        Args:
            None.

        Returns:
            None.
        """
        self._set_default_initializer(Xavier())

    def _set_default_bias_initializer(self):
        """
        Set the default initializer for the bias with Constant(0.0).

        Args:
            None.

        Returns:
            None.
        """
        self._set_default_initializer(Constant(0.0))

    @staticmethod
    def _to_attr(arg):
        """
        Create ParamAttr[s].

        Args:
            arg: Arguments to initialize ParamAttr[s]. arg's type can be
                str, Initializer, float, WeightDecayRegularizer, BaseGradientClipAttr,
                bool, ParamAttr, or a list of above type.

        Returns:
            ParamAttr[s]: ParamAttr[s] initialized with arg.

        Raises:
            arg can not initialize a ParamAttr.
        """
        if arg is None:
            return ParamAttr()
        elif isinstance(arg, list) or isinstance(arg, tuple):
            return [ParamAttr._to_attr(a) for a in arg]
        elif isinstance(arg, ParamAttr):
            return arg
        elif isinstance(arg, six.string_types):
            return ParamAttr(name=arg)
        elif isinstance(arg, Initializer):
            return ParamAttr(initializer=arg)
        elif isinstance(arg, WeightDecayRegularizer):
            return ParamAttr(regularizer=arg)
        elif isinstance(arg, bool):
            return ParamAttr._to_attr(None) if arg else False
        else:
            raise TypeError("{0} cast to ParamAttr".format(type(arg)))

    def _to_kwargs(self, with_initializer=False):
        """
        Returns the attributes of this parameter.

        Args:
            with_initializer(bool): Whether to add initializer attr.

        Returns:
            Parameter attributes(map): The attributes of this parameter.
        """
        kwargs = {
            'name': self.name,
            'optimize_attr': {
                'learning_rate': self.learning_rate
            },
            'regularizer': self.regularizer,
            'trainable': self.trainable,
            'do_model_average': self.do_model_average,
            'need_clip': self.need_clip
        }
        if with_initializer:
            kwargs['initializer'] = self.initializer
        return kwargs


class WeightNormParamAttr(ParamAttr):
    r"""
	:api_attr: Static Graph

    Note:
        Please use 'paddle.nn.utils.weight_norm' in dygraph mode.

    Parameter of weight Norm. Weight Norm is a reparameterization of the weight vectors
    in a neural network that decouples the magnitude of those weight vectors from
    their direction. Weight Norm has been implemented as discussed in this
    paper: `Weight Normalization: A Simple Reparameterization to Accelerate
    Training of Deep Neural Networks
    <https://arxiv.org/pdf/1602.07868.pdf>`_.
      
    Note:
        ``gradient_clip`` of ``ParamAttr`` HAS BEEN DEPRECATED since 2.0. 
        Please use ``need_clip`` in ``ParamAttr`` to speficiy the clip scope.
        There are three clipping strategies: :ref:`api_paddle_nn_ClipGradByGlobalNorm` , 
        :ref:`api_paddle_nn_ClipGradByNorm` , :ref:`api_paddle_nn_ClipGradByValue` .
        

    Args:
        dim(int, optional): Dimension over which to compute the norm. Dim is a non-negative
            number which is less than the rank of weight Tensor. For Example, dim can
            be chosen from 0, 1, 2, 3 for convolution whose weight shape is [cout, cin, kh, kw]
            and rank is 4. Default None, meaning that all elements will be normalized.
        name(str, optional): The parameter's name. Default None, meaning that the name would
            be created automatically. Please refer to :ref:`api_guide_Name` for more details.
        initializer(Initializer, optional): The method to initialize this parameter, such as
            ``initializer = paddle.nn.initializer.Constant(1.0)``. Default None,
            meaning that the weight parameter is initialized by Xavier initializer, and
            the bias parameter is initialized by 0.
        learning_rate(float32, optional): The parameter's learning rate when
            optimizer is :math:`global\_lr * parameter\_lr * scheduler\_factor`.
            Default 1.0.
        regularizer (WeightDecayRegularizer, optional): Regularization strategy. There are
            two method: :ref:`api_paddle_regularizer_L1Decay` ,
            :ref:`api_paddle_regularizer_L2Decay`.
            If regularizer isralso set in ``optimizer``
            (such as :ref:`api_paddle_optimizer_SGD` ), that regularizer setting in
            optimizer will be ignored. Default None, meaning there is no regularization.
        trainable(bool, optional): Whether this parameter is trainable. Default True.
        do_model_average(bool, optional): Whether this parameter should do model average.
            Default False.
        need_clip (bool, optional): Whether the parameter gradient need to be cliped in optimizer. Default is True.

    Examples:
        .. code-block:: python
            
            import paddle

            paddle.enable_static()

            data = paddle.static.data(name="data", shape=[3, 32, 32], dtype="float32")

            fc = paddle.static.nn.fc(x=data,
                                     size=1000,
                                     weight_attr=paddle.static.WeightNormParamAttr(
                                         dim=None,
                                         name='weight_norm_param',
                                         initializer=paddle.nn.initializer.Constant(1.0),
                                         learning_rate=1.0,
                                         regularizer=paddle.regularizer.L2Decay(0.1),
                                         trainable=True,
                                         do_model_average=False,
                                         need_clip=True))

    """
    # List to record the parameters reparameterized by weight normalization.
    # If these parameters are treated as Variable rather than Parameter,
    # it can be used to discriminate these parameters and help to serialize
    # these paramters for inference.
    params_with_weight_norm = []

    def __init__(self,
                 dim=None,
                 name=None,
                 initializer=None,
                 learning_rate=1.0,
                 regularizer=None,
                 trainable=True,
                 do_model_average=False,
                 need_clip=True):
        super(WeightNormParamAttr, self).__init__(
            name=name,
            initializer=initializer,
            learning_rate=learning_rate,
            regularizer=regularizer,
            trainable=trainable,
            do_model_average=do_model_average,
            need_clip=need_clip)
        self.dim = dim
