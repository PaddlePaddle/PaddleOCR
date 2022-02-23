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

import os
from paddle.utils import try_import

__all__ = []


def export(layer, path, input_spec=None, opset_version=9, **configs):
    """
    Export Layer to ONNX format, which can use for inference via onnxruntime or other backends.
    For more details, Please refer to `paddle2onnx <https://github.com/PaddlePaddle/paddle2onnx>`_ .

    Args:
        layer (Layer): The Layer to be exported.
        path (str): The path prefix to export model. The format is ``dirname/file_prefix`` or ``file_prefix`` ,
            and the exported ONNX file suffix is ``.onnx`` . 
        input_spec (list[InputSpec|Tensor], optional): Describes the input of the exported model's forward 
            method, which can be described by InputSpec or example Tensor. If None, all input variables of 
            the original Layer's forward method would be the inputs of the exported ``ONNX`` model. Default: None.
        opset_version(int, optional): Opset version of exported ONNX model.
            Now, stable supported opset version include 9, 10, 11. Default: 9.
        **configs (dict, optional): Other export configuration options for compatibility. We do not 
            recommend using these configurations, they may be removed in the future. If not necessary, 
            DO NOT use them. Default None.
            The following options are currently supported:
            (1) output_spec (list[Tensor]): Selects the output targets of the exported model.
            By default, all return variables of original Layer's forward method are kept as the 
            output of the exported model. If the provided ``output_spec`` list is not all output variables, 
            the exported model will be pruned according to the given ``output_spec`` list. 
    Returns:
        None
    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            class LinearNet(paddle.nn.Layer):
                def __init__(self):
                    super(LinearNet, self).__init__()
                    self._linear = paddle.nn.Linear(128, 10)

                def forward(self, x):
                    return self._linear(x)

            # Export model with 'InputSpec' to support dynamic input shape.
            def export_linear_net():
                model = LinearNet()
                x_spec = paddle.static.InputSpec(shape=[None, 128], dtype='float32')
                paddle.onnx.export(model, 'linear_net', input_spec=[x_spec])

            export_linear_net()

            class Logic(paddle.nn.Layer):
                def __init__(self):
                    super(Logic, self).__init__()

                def forward(self, x, y, z):
                    if z:
                        return x
                    else:
                        return y

            # Export model with 'Tensor' to support pruned model by set 'output_spec'.
            def export_logic():
                model = Logic()
                x = paddle.to_tensor(np.array([1]))
                y = paddle.to_tensor(np.array([2]))
                # Static and run model.
                paddle.jit.to_static(model)
                out = model(x, y, z=True)
                paddle.onnx.export(model, 'pruned', input_spec=[x], output_spec=[out])

            export_logic()
    """

    p2o = try_import('paddle2onnx')

    file_prefix = os.path.basename(path)
    if file_prefix == "":
        raise ValueError("The input path MUST be format of dirname/file_prefix "
                         "[dirname\\file_prefix in Windows system], but "
                         "the file_prefix is empty in received path: {}".format(
                             path))
    save_file = path + '.onnx'

    p2o.dygraph2onnx(
        layer,
        save_file,
        input_spec=input_spec,
        opset_version=opset_version,
        **configs)
