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

import paddle
import paddle.nn as nn

from paddle.utils.download import get_weights_path_from_url

__all__ = []

model_urls = {
    'vgg16': ('https://paddle-hapi.bj.bcebos.com/models/vgg16.pdparams',
              '89bbffc0f87d260be9b8cdc169c991c4'),
    'vgg19': ('https://paddle-hapi.bj.bcebos.com/models/vgg19.pdparams',
              '23b18bb13d8894f60f54e642be79a0dd')
}


class VGG(nn.Layer):
    """VGG model from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        features (nn.Layer): Vgg features create by function make_layers.
        num_classes (int): Output dim of last fc layer. If num_classes <=0, last fc layer 
                            will not be defined. Default: 1000.
        with_pool (bool): Use pool before the last three fc layer or not. Default: True.

    Examples:
        .. code-block:: python

            from paddle.vision.models import VGG
            from paddle.vision.models.vgg import make_layers

            vgg11_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

            features = make_layers(vgg11_cfg)

            vgg11 = VGG(features)

    """

    def __init__(self, features, num_classes=1000, with_pool=True):
        super(VGG, self).__init__()
        self.features = features
        self.num_classes = num_classes
        self.with_pool = with_pool

        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((7, 7))

        if num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096, num_classes), )

    def forward(self, x):
        x = self.features(x)

        if self.with_pool:
            x = self.avgpool(x)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.classifier(x)

        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2D(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2D(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2D(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B':
    [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512,
        512, 512, 'M'
    ],
    'E': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512,
        'M', 512, 512, 512, 512, 'M'
    ],
}


def _vgg(arch, cfg, batch_norm, pretrained, **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])

        param = paddle.load(weight_path)
        model.load_dict(param)

    return model


def vgg11(pretrained=False, batch_norm=False, **kwargs):
    """VGG 11-layer model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        batch_norm (bool): If True, returns a model with batch_norm layer. Default: False.

    Examples:
        .. code-block:: python

            from paddle.vision.models import vgg11

            # build model
            model = vgg11()

            # build vgg11 model with batch_norm
            model = vgg11(batch_norm=True)
    """
    model_name = 'vgg11'
    if batch_norm:
        model_name += ('_bn')
    return _vgg(model_name, 'A', batch_norm, pretrained, **kwargs)


def vgg13(pretrained=False, batch_norm=False, **kwargs):
    """VGG 13-layer model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        batch_norm (bool): If True, returns a model with batch_norm layer. Default: False.

    Examples:
        .. code-block:: python

            from paddle.vision.models import vgg13

            # build model
            model = vgg13()

            # build vgg13 model with batch_norm
            model = vgg13(batch_norm=True)
    """
    model_name = 'vgg13'
    if batch_norm:
        model_name += ('_bn')
    return _vgg(model_name, 'B', batch_norm, pretrained, **kwargs)


def vgg16(pretrained=False, batch_norm=False, **kwargs):
    """VGG 16-layer model 
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        batch_norm (bool): If True, returns a model with batch_norm layer. Default: False.

    Examples:
        .. code-block:: python

            from paddle.vision.models import vgg16

            # build model
            model = vgg16()

            # build vgg16 model with batch_norm
            model = vgg16(batch_norm=True)
    """
    model_name = 'vgg16'
    if batch_norm:
        model_name += ('_bn')
    return _vgg(model_name, 'D', batch_norm, pretrained, **kwargs)


def vgg19(pretrained=False, batch_norm=False, **kwargs):
    """VGG 19-layer model 
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        batch_norm (bool): If True, returns a model with batch_norm layer. Default: False.

    Examples:
        .. code-block:: python

            from paddle.vision.models import vgg19

            # build model
            model = vgg19()

            # build vgg19 model with batch_norm
            model = vgg19(batch_norm=True)
    """
    model_name = 'vgg19'
    if batch_norm:
        model_name += ('_bn')
    return _vgg(model_name, 'E', batch_norm, pretrained, **kwargs)
