# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import errno
import os
import pickle
import six

import paddle

from ppocr.utils.logging import get_logger

__all__ = ['load_model']


def _mkdir_if_not_exist(path, logger):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    'be happy if some process has already created {}'.format(
                        path))
            else:
                raise OSError('Failed to mkdir {}'.format(path))


def load_model(config, model, optimizer=None, model_type='det'):
    """
    load model from checkpoint or pretrained_model
    """
    logger = get_logger()
    global_config = config['Global']
    checkpoints = global_config.get('checkpoints')
    pretrained_model = global_config.get('pretrained_model')
    best_model_dict = {}

    if model_type == 'vqa':
        checkpoints = config['Architecture']['Backbone']['checkpoints']
        # load vqa method metric
        if checkpoints:
            if os.path.exists(os.path.join(checkpoints, 'metric.states')):
                with open(os.path.join(checkpoints, 'metric.states'),
                          'rb') as f:
                    states_dict = pickle.load(f) if six.PY2 else pickle.load(
                        f, encoding='latin1')
                best_model_dict = states_dict.get('best_model_dict', {})
                if 'epoch' in states_dict:
                    best_model_dict['start_epoch'] = states_dict['epoch'] + 1
            logger.info("resume from {}".format(checkpoints))

            if optimizer is not None:
                if checkpoints[-1] in ['/', '\\']:
                    checkpoints = checkpoints[:-1]
                if os.path.exists(checkpoints + '.pdopt'):
                    optim_dict = paddle.load(checkpoints + '.pdopt')
                    optimizer.set_state_dict(optim_dict)
                else:
                    logger.warning(
                        "{}.pdopt is not exists, params of optimizer is not loaded".
                        format(checkpoints))
        return best_model_dict

    if checkpoints:
        if checkpoints.endswith('.pdparams'):
            checkpoints = checkpoints.replace('.pdparams', '')
        assert os.path.exists(checkpoints + ".pdparams"), \
            "The {}.pdparams does not exists!".format(checkpoints)

        # load params from trained model
        params = paddle.load(checkpoints + '.pdparams')
        state_dict = model.state_dict()
        new_state_dict = {}
        for key, value in state_dict.items():
            if key not in params:
                logger.warning("{} not in loaded params {} !".format(
                    key, params.keys()))
                continue
            pre_value = params[key]
            if list(value.shape) == list(pre_value.shape):
                new_state_dict[key] = pre_value
            else:
                logger.warning(
                    "The shape of model params {} {} not matched with loaded params shape {} !".
                    format(key, value.shape, pre_value.shape))
        model.set_state_dict(new_state_dict)

        if optimizer is not None:
            if os.path.exists(checkpoints + '.pdopt'):
                optim_dict = paddle.load(checkpoints + '.pdopt')
                optimizer.set_state_dict(optim_dict)
            else:
                logger.warning(
                    "{}.pdopt is not exists, params of optimizer is not loaded".
                    format(checkpoints))

        if os.path.exists(checkpoints + '.states'):
            with open(checkpoints + '.states', 'rb') as f:
                states_dict = pickle.load(f) if six.PY2 else pickle.load(
                    f, encoding='latin1')
            best_model_dict = states_dict.get('best_model_dict', {})
            if 'epoch' in states_dict:
                best_model_dict['start_epoch'] = states_dict['epoch'] + 1
        logger.info("resume from {}".format(checkpoints))
    elif pretrained_model:
        load_pretrained_params(model, pretrained_model)
    else:
        logger.info('train from scratch')
    return best_model_dict


def load_pretrained_params(model, path):
    logger = get_logger()
    if path.endswith('.pdparams'):
        path = path.replace('.pdparams', '')
    assert os.path.exists(path + ".pdparams"), \
        "The {}.pdparams does not exists!".format(path)

    params = paddle.load(path + '.pdparams')
    state_dict = model.state_dict()
    new_state_dict = {}
    for k1 in params.keys():
        if k1 not in state_dict.keys():
            logger.warning("The pretrained params {} not in model".format(k1))
        else:
            if list(state_dict[k1].shape) == list(params[k1].shape):
                new_state_dict[k1] = params[k1]
            else:
                logger.warning(
                    "The shape of model params {} {} not matched with loaded params {} {} !".
                    format(k1, state_dict[k1].shape, k1, params[k1].shape))
    model.set_state_dict(new_state_dict)
    logger.info("load pretrain successful from {}".format(path))
    return model


def save_model(model,
               optimizer,
               model_path,
               logger,
               config,
               is_best=False,
               prefix='ppocr',
               **kwargs):
    """
    save model to the target path
    """
    _mkdir_if_not_exist(model_path, logger)
    model_prefix = os.path.join(model_path, prefix)
    paddle.save(optimizer.state_dict(), model_prefix + '.pdopt')
    if config['Architecture']["model_type"] != 'vqa':
        paddle.save(model.state_dict(), model_prefix + '.pdparams')
        metric_prefix = model_prefix
    else:
        if config['Global']['distributed']:
            model._layers.backbone.model.save_pretrained(model_prefix)
        else:
            model.backbone.model.save_pretrained(model_prefix)
        metric_prefix = os.path.join(model_prefix, 'metric')
    # save metric and config
    with open(metric_prefix + '.states', 'wb') as f:
        pickle.dump(kwargs, f, protocol=2)
    if is_best:
        logger.info('save best model is to {}'.format(model_prefix))
    else:
        logger.info("save model in {}".format(model_prefix))
