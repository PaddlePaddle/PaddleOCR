#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from .resnet import ResNet  # noqa: F401
from .resnet import resnet18  # noqa: F401
from .resnet import resnet34  # noqa: F401
from .resnet import resnet50  # noqa: F401
from .resnet import resnet101  # noqa: F401
from .resnet import resnet152  # noqa: F401
from .mobilenetv1 import MobileNetV1  # noqa: F401
from .mobilenetv1 import mobilenet_v1  # noqa: F401
from .mobilenetv2 import MobileNetV2  # noqa: F401
from .mobilenetv2 import mobilenet_v2  # noqa: F401
from .vgg import VGG  # noqa: F401
from .vgg import vgg11  # noqa: F401
from .vgg import vgg13  # noqa: F401
from .vgg import vgg16  # noqa: F401
from .vgg import vgg19  # noqa: F401
from .lenet import LeNet  # noqa: F401

__all__ = [ #noqa
    'ResNet',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'VGG',
    'vgg11',
    'vgg13',
    'vgg16',
    'vgg19',
    'MobileNetV1',
    'mobilenet_v1',
    'MobileNetV2',
    'mobilenet_v2',
    'LeNet'
]
