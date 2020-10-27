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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid

from ppocr.utils.utility import create_module
from ppocr.utils.utility import initial_logger

logger = initial_logger()
from copy import deepcopy


class ClsModel(object):
    def __init__(self, params):
        super(ClsModel, self).__init__()
        global_params = params['Global']
        self.infer_img = global_params['infer_img']

        backbone_params = deepcopy(params["Backbone"])
        backbone_params.update(global_params)
        self.backbone = create_module(backbone_params['function']) \
            (params=backbone_params)

        head_params = deepcopy(params["Head"])
        head_params.update(global_params)
        self.head = create_module(head_params['function']) \
            (params=head_params)

        loss_params = deepcopy(params["Loss"])
        loss_params.update(global_params)
        self.loss = create_module(loss_params['function']) \
            (params=loss_params)

        self.image_shape = global_params['image_shape']

    def create_feed(self, mode):
        image_shape = deepcopy(self.image_shape)
        image_shape.insert(0, -1)
        if mode == "train":
            image = fluid.data(name='image', shape=image_shape, dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            feed_list = [image, label]
            labels = {'label': label}
            loader = fluid.io.DataLoader.from_generator(
                feed_list=feed_list,
                capacity=64,
                use_double_buffer=True,
                iterable=False)
        else:
            labels = None
            loader = None
            image = fluid.data(name='image', shape=image_shape, dtype='float32')
        image.stop_gradient = False
        return image, labels, loader

    def __call__(self, mode):
        image, labels, loader = self.create_feed(mode)
        inputs = image
        conv_feas = self.backbone(inputs)
        predicts = self.head(conv_feas, labels, mode)
        if mode == "train":
            loss = self.loss(predicts, labels)
            label = labels['label']
            acc = fluid.layers.accuracy(predicts['predict'], label, k=1)
            outputs = {'total_loss': loss, 'decoded_out': \
                predicts['decoded_out'], 'label': label, 'acc': acc}
            return loader, outputs
        elif mode == "export":
            return [image, predicts]
        else:
            return loader, predicts
