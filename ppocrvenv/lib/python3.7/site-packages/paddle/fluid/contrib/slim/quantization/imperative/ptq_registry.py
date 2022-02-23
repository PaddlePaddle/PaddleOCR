#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

__all__ = ['PTQRegistry']


class LayerInfo(object):
    """
    Store the argnames of the inputs and outputs.
    """

    def __init__(self, layer, input_names, weight_names, output_names):
        super(LayerInfo, self).__init__()
        self.layer = layer
        self.input_names = input_names
        self.weight_names = weight_names
        self.output_names = output_names


PTQ_LAYERS_INFO = [
    LayerInfo(paddle.nn.Conv2D, ['Input'], ['Filter'], ['Output']),
    LayerInfo(paddle.nn.Linear, ['X'], ['Y'], ['Out']),
    LayerInfo(paddle.nn.BatchNorm2D, ['X'], [], ['Y']),
    LayerInfo(paddle.nn.AdaptiveMaxPool2D, ['X'], [], ['Out']),
    LayerInfo(paddle.nn.AdaptiveAvgPool2D, ['X'], [], ['Out']),
    LayerInfo(paddle.nn.AvgPool2D, ['X'], [], ['Out']),
    LayerInfo(paddle.nn.MaxPool2D, ['X'], [], ['Out']),
    LayerInfo(paddle.nn.ReLU, ['X'], [], ['Out']),
    LayerInfo(paddle.nn.ReLU6, ['X'], [], ['Out']),
    LayerInfo(paddle.nn.Hardswish, ['X'], [], ['Out']),
    LayerInfo(paddle.nn.Sigmoid, ['X'], [], ['Out']),
    LayerInfo(paddle.nn.Softmax, ['X'], [], ['Out']),
    LayerInfo(paddle.nn.Tanh, ['X'], [], ['Out']),
    LayerInfo(paddle.nn.quant.add, ['X', 'Y'], [], ['Out']),
]

QUANT_LAYERS_INFO = [
    LayerInfo(paddle.nn.quant.quant_layers.QuantizedConv2D, ['Input'],
              ['Filter'], ['Output']),
    LayerInfo(paddle.nn.quant.quant_layers.QuantizedLinear, ['X'], ['Y'],
              ['Out']),
]

SIMULATED_LAYERS = [paddle.nn.Conv2D, paddle.nn.Linear]


class PTQRegistry(object):
    """
    Register the supported layers for PTQ and provide layers info.
    """
    supported_layers_map = {}
    registered_layers_map = {}
    is_inited = False

    def __init__(self):
        super(PTQRegistry, self).__init__()

    @classmethod
    def _init(cls):
        if not cls.is_inited:
            for layer_info in PTQ_LAYERS_INFO:
                cls.supported_layers_map[layer_info.layer] = layer_info

            all_layers_info = PTQ_LAYERS_INFO + QUANT_LAYERS_INFO
            for layer_info in all_layers_info:
                cls.registered_layers_map[layer_info.layer] = layer_info
        cls.is_inited = True

    @classmethod
    def is_supported_layer(cls, layer):
        """
        Analyze whether the layer supports quantization.
        Args:
            layer(Layer): The input layer can be a python class or an instance.
        Returns:
            flag(bool): Whther the layer is supported.
        """
        cls._init()
        return layer in cls.supported_layers_map or \
            isinstance(layer, tuple(cls.supported_layers_map.keys()))

    @classmethod
    def is_registered_layer(cls, layer):
        """
        Analyze whether the layer is register layer_info.
        Args:
            layer(Layer): The input layer can be a python class or an instance.
        Returns:
            flag(bool): Wether the layer is register layer_info.
        """
        cls._init()
        return layer in cls.registered_layers_map or \
            isinstance(layer, tuple(cls.registered_layers_map.keys()))

    @classmethod
    def is_simulated_quant_layer(cls, layer):
        """
        Analyze whether the layer is simulated quant layer.
        Args:
            layer(Layer): The input layer can be a python class or an instance.
        Returns:
            flag(bool): Whther the layer is supported.
        """
        return layer in SIMULATED_LAYERS or \
            isinstance(layer, tuple(SIMULATED_LAYERS))

    @classmethod
    def layer_info(cls, layer):
        """
        Get the infomation for the layer.
        Args:
            layer(Layer): The input layer can be a python class or an instance.
        Returns:
            layer_info(LayerInfo): The layer info of the input layer.
        """
        assert cls.is_registered_layer(layer), \
            "The input layer is not register."

        for layer_key, layer_info in cls.registered_layers_map.items():
            if layer == layer_key or isinstance(layer, layer_key):
                return layer_info
