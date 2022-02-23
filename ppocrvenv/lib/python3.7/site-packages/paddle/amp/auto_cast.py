#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.dygraph.amp import amp_guard
from paddle.fluid.dygraph.amp import amp_decorate

__all__ = []


def auto_cast(enable=True,
              custom_white_list=None,
              custom_black_list=None,
              level='O1'):
    """
    Create a context which enables auto-mixed-precision(AMP) of operators executed in dynamic graph mode.
    If enabled, the input data type (float32 or float16) of each operator is decided 
    by autocast algorithm for better performance. 
    
    Commonly, it is used together with `GradScaler` to achieve Auto-Mixed-Precision in 
    imperative mode. It is used together with `decorator` to achieve Pure fp16 in imperative mode.

    Args:
        enable(bool, optional): Enable auto-mixed-precision or not. Default is True.
        custom_white_list(set|list|tuple, optional): The custom white_list. It's the set of ops that support
             fp16 calculation and are considered numerically-safe and performance-critical. These ops 
             will be converted to fp16.
        custom_black_list(set|list|tuple, optional): The custom black_list. The set of ops that support fp16
             calculation and are considered numerically-dangerous and whose effects may also be 
             observed in downstream ops. These ops will not be converted to fp16.
        level(str, optional): Auto mixed precision level. Accepted values are "O1" and "O2": O1 represent mixed precision, the input data type of each operator will be casted by white_list and black_list; 
             O2 represent Pure fp16, all operators parameters and input data will be casted to fp16, except operators in black_list, don't support fp16 kernel and batchnorm. Default is O1(amp)
        
    Examples:

     .. code-block:: python

        import paddle

        conv2d = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
        data = paddle.rand([10, 3, 32, 32])

        with paddle.amp.auto_cast():
            conv = conv2d(data)
            print(conv.dtype) # FP16

        with paddle.amp.auto_cast(enable=False):
            conv = conv2d(data)
            print(conv.dtype) # FP32

        with paddle.amp.auto_cast(custom_black_list={'conv2d'}):
            conv = conv2d(data)
            print(conv.dtype) # FP32

        a = paddle.rand([2,3])
        b = paddle.rand([2,3])
        with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}):
            c = a + b
            print(c.dtype) # FP16
        
        with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}, level='O2'):
            d = a + b
            print(d.dtype) # FP16

    """
    return amp_guard(enable, custom_white_list, custom_black_list, level)


def decorate(models,
             optimizers=None,
             level='O1',
             master_weight=None,
             save_dtype=None):
    """
    Decorate models and optimizers for auto-mixed-precision. When level is O1(amp), the decorate will do nothing. 
    When level is O2(pure fp16), the decorate will cast all parameters of models to FP16, except BatchNorm and LayerNorm.
    
    Commonly, it is used together with `auto_cast` to achieve Pure fp16 in imperative mode.

    Args:
        models(Layer|list of Layer, optional): The defined models by user, models must be either a single model or a list of models. Default is None.
        optimizers(Optimizer|list of Optimizer, optional): The defined optimizers by user, optimizers must be either a single optimizer or a list of optimizers. Default is None.
        level(str, optional): Auto mixed precision level. Accepted values are "O1" and "O2": O1 represent mixed precision, the decorator will do nothing; 
             O2 represent Pure fp16, the decorator will cast all parameters of models to FP16, except BatchNorm and LayerNorm. Default is O1(amp)
        master_weight(bool, optinal): For level='O2', whether to use multi-precision during weight updating. If master_weight is None, in O2 level optimizer will use multi-precision. Default is None.
        save_dtype(float, optional): The save model parameter dtype when use `paddle.save` or `paddle.jit.save`,it should be float16, float32, float64 or None.
             The save_dtype will not change model parameters dtype, it just change the state_dict dtype. When save_dtype is None, the save dtype is same as model dtype. Default is None.

    Examples:

     .. code-block:: python   

        # required: gpu
        # Demo1: single model and optimizer:
        import paddle

        model = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
        optimzier = paddle.optimizer.SGD(parameters=model.parameters())

        model, optimizer = paddle.amp.decorate(models=model, optimizers=optimzier, level='O2')

        data = paddle.rand([10, 3, 32, 32])

        with paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
            output = model(data)
            print(output.dtype) # FP16
            
        # required: gpu
        # Demo2: multi models and optimizers:
        model2 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
        optimizer2 = paddle.optimizer.Adam(parameters=model2.parameters())

        models, optimizers = paddle.amp.decorate(models=[model, model2], optimizers=[optimzier, optimizer2], level='O2')

        data = paddle.rand([10, 3, 32, 32])

        with paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
            output = models[0](data)
            output2 = models[1](data)
            print(output.dtype) # FP16
            print(output2.dtype) # FP16
    """
    return amp_decorate(models, optimizers, level, master_weight, save_dtype)
