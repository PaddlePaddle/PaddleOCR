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


class DetModel(object):
    def __init__(self, params):
        """
        Detection module for OCR text detection.
        args:
            params (dict): the super parameters for detection module.
        """
        global_params = params['Global']
        self.algorithm = global_params['algorithm']

        backbone_params = deepcopy(params["Backbone"])
        backbone_params.update(global_params)
        self.backbone = create_module(backbone_params['function'])\
                (params=backbone_params)

        head_params = deepcopy(params["Head"])
        head_params.update(global_params)
        self.head = create_module(head_params['function'])\
                (params=head_params)

        loss_params = deepcopy(params["Loss"])
        loss_params.update(global_params)
        self.loss = create_module(loss_params['function'])\
                (params=loss_params)

        self.image_shape = global_params['image_shape']

    def create_feed(self, mode):
        """
        create Dataloader feeds
        args:
            mode (str): 'train' for training  or else for evaluation
        return: (image, corresponding label, dataloader)
        """
        image_shape = deepcopy(self.image_shape)
        image = fluid.layers.data(
            name='image', shape=image_shape, dtype='float32')
        if mode == "train":
            if self.algorithm == "EAST":
                score = fluid.layers.data(
                    name='score', shape=[1, 128, 128], dtype='float32')
                geo = fluid.layers.data(
                    name='geo', shape=[9, 128, 128], dtype='float32')
                mask = fluid.layers.data(
                    name='mask', shape=[1, 128, 128], dtype='float32')
                feed_list = [image, score, geo, mask]
                labels = {'score': score, 'geo': geo, 'mask': mask}
            elif self.algorithm == "DB":
                shrink_map = fluid.layers.data(
                    name='shrink_map', shape=image_shape[1:], dtype='float32')
                shrink_mask = fluid.layers.data(
                    name='shrink_mask', shape=image_shape[1:], dtype='float32')
                threshold_map = fluid.layers.data(
                    name='threshold_map',
                    shape=image_shape[1:],
                    dtype='float32')
                threshold_mask = fluid.layers.data(
                    name='threshold_mask',
                    shape=image_shape[1:],
                    dtype='float32')
                feed_list=[image, shrink_map, shrink_mask,\
                    threshold_map, threshold_mask]
                labels = {'shrink_map':shrink_map,\
                    'shrink_mask':shrink_mask,\
                    'threshold_map':threshold_map,\
                    'threshold_mask':threshold_mask}
            loader = fluid.io.DataLoader.from_generator(
                feed_list=feed_list,
                capacity=64,
                use_double_buffer=True,
                iterable=False)
        else:
            labels = None
            loader = None
        return image, labels, loader

    def __call__(self, mode):
        """
        run forward of defined module
        args:
            mode (str): 'train' for training; 'export'  for inference,
                others for evaluation]
        """
        image, labels, loader = self.create_feed(mode)
        conv_feas = self.backbone(image)
        if self.algorithm == "DB":
            predicts = self.head(conv_feas, mode)
        else:
            predicts = self.head(conv_feas)
        if mode == "train":
            losses = self.loss(predicts, labels)
            return loader, losses
        elif mode == "export":
            return [image, predicts]
        else:
            return loader, predicts
