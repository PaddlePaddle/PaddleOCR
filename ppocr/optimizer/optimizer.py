# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from paddle import optimizer as optim


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
                 weight_decay=None,
                 grad_clip=None,
                 **args):
        super(Momentum, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def __call__(self, model):
        train_params = [
            param for param in model.parameters() if param.trainable is True
        ]
        opt = optim.Momentum(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            parameters=train_params)
        return opt


class Adam(object):
    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-08,
                 parameter_list=None,
                 weight_decay=None,
                 grad_clip=None,
                 name=None,
                 lazy_mode=False,
                 **kwargs):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.parameter_list = parameter_list
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.name = name
        self.lazy_mode = lazy_mode
        self.group_lr = kwargs.get('group_lr', False)
        self.training_step = kwargs.get('training_step', None)

    def __call__(self, model):
        if self.group_lr:
            if self.training_step == 'LF_2':
                import paddle
                if isinstance(model, paddle.fluid.dygraph.parallel.
                              DataParallel):  # multi gpu
                    mlm = model._layers.head.MLM_VRM.MLM.parameters()
                    pre_mlm_pp = model._layers.head.MLM_VRM.Prediction.pp_share.parameters(
                    )
                    pre_mlm_w = model._layers.head.MLM_VRM.Prediction.w_share.parameters(
                    )
                else:  # single gpu
                    mlm = model.head.MLM_VRM.MLM.parameters()
                    pre_mlm_pp = model.head.MLM_VRM.Prediction.pp_share.parameters(
                    )
                    pre_mlm_w = model.head.MLM_VRM.Prediction.w_share.parameters(
                    )

                total = []
                for param in mlm:
                    total.append(id(param))
                for param in pre_mlm_pp:
                    total.append(id(param))
                for param in pre_mlm_w:
                    total.append(id(param))

                group_base_params = [
                    param for param in model.parameters() if id(param) in total
                ]
                group_small_params = [
                    param for param in model.parameters()
                    if id(param) not in total
                ]
                train_params = [{
                    'params': group_base_params
                }, {
                    'params': group_small_params,
                    'learning_rate': self.learning_rate.values[0] * 0.1
                }]

            else:
                print(
                    'group lr currently only support VisionLAN in LF_2 training step'
                )
                train_params = [
                    param for param in model.parameters()
                    if param.trainable is True
                ]
        else:
            train_params = [
                param for param in model.parameters() if param.trainable is True
            ]

        opt = optim.Adam(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            name=self.name,
            lazy_mode=self.lazy_mode,
            parameters=train_params)
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
                 momentum=0.0,
                 rho=0.95,
                 epsilon=1e-6,
                 weight_decay=None,
                 grad_clip=None,
                 **args):
        super(RMSProp, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def __call__(self, model):
        train_params = [
            param for param in model.parameters() if param.trainable is True
        ]
        opt = optim.RMSProp(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            rho=self.rho,
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            parameters=train_params)
        return opt


class Adadelta(object):
    def __init__(self,
                 learning_rate=0.001,
                 epsilon=1e-08,
                 rho=0.95,
                 parameter_list=None,
                 weight_decay=None,
                 grad_clip=None,
                 name=None,
                 **kwargs):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.rho = rho
        self.parameter_list = parameter_list
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.name = name

    def __call__(self, model):
        train_params = [
            param for param in model.parameters() if param.trainable is True
        ]
        opt = optim.Adadelta(
            learning_rate=self.learning_rate,
            epsilon=self.epsilon,
            rho=self.rho,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            name=self.name,
            parameters=train_params)
        return opt


class AdamW(object):
    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 weight_decay=0.01,
                 multi_precision=False,
                 grad_clip=None,
                 no_weight_decay_name=None,
                 one_dim_param_no_weight_decay=False,
                 name=None,
                 lazy_mode=False,
                 **args):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.grad_clip = grad_clip
        self.weight_decay = 0.01 if weight_decay is None else weight_decay
        self.grad_clip = grad_clip
        self.name = name
        self.lazy_mode = lazy_mode
        self.multi_precision = multi_precision
        self.no_weight_decay_name_list = no_weight_decay_name.split(
        ) if no_weight_decay_name else []
        self.one_dim_param_no_weight_decay = one_dim_param_no_weight_decay

    def __call__(self, model):
        parameters = [
            param for param in model.parameters() if param.trainable is True
        ]

        self.no_weight_decay_param_name_list = [
            p.name for n, p in model.named_parameters()
            if any(nd in n for nd in self.no_weight_decay_name_list)
        ]

        if self.one_dim_param_no_weight_decay:
            self.no_weight_decay_param_name_list += [
                p.name for n, p in model.named_parameters() if len(p.shape) == 1
            ]

        opt = optim.AdamW(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            parameters=parameters,
            weight_decay=self.weight_decay,
            multi_precision=self.multi_precision,
            grad_clip=self.grad_clip,
            name=self.name,
            lazy_mode=self.lazy_mode,
            apply_decay_param_fun=self._apply_decay_param_fun)
        return opt

    def _apply_decay_param_fun(self, name):
        return name not in self.no_weight_decay_param_name_list
