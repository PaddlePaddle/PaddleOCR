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

import os
import sys
import logging
import functools
import paddle.distributed as dist

logger_initialized = {}


def print_dict(d, logger, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            print_dict(v, logger, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            for value in v:
                print_dict(value, logger, delimiter + 4)
        else:
            logger.info("{}{} : {}".format(delimiter * " ", k, v))


@functools.lru_cache()
def get_logger(name="root", log_file=None, log_level=logging.DEBUG):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified a FileHandler will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_file is not None and dist.get_rank() == 0:
        log_file_folder = os.path.split(log_file)[0]
        os.makedirs(log_file_folder, exist_ok=True)
        file_handler = logging.FileHandler(log_file, "a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if dist.get_rank() == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)
    logger_initialized[name] = True
    return logger


def load_model(config, model, optimizer=None):
    """
    load model from checkpoint or pretrained_model
    """
    logger = get_logger()
    checkpoints = config.get("checkpoints")
    pretrained_model = config.get("pretrained_model")
    best_model_dict = {}
    if checkpoints:
        if checkpoints.endswith(".pdparams"):
            checkpoints = checkpoints.replace(".pdparams", "")
        assert os.path.exists(
            checkpoints + ".pdparams"
        ), "The {}.pdparams does not exists!".format(checkpoints)

        # load params from trained model
        params = paddle.load(checkpoints + ".pdparams")
        state_dict = model.state_dict()
        new_state_dict = {}
        for key, value in state_dict.items():
            if key not in params:
                logger.warning(
                    "{} not in loaded params {} !".format(key, params.keys())
                )
                continue
            pre_value = params[key]
            if list(value.shape) == list(pre_value.shape):
                new_state_dict[key] = pre_value
            else:
                logger.warning(
                    "The shape of model params {} {} not matched with loaded params shape {} !".format(
                        key, value.shape, pre_value.shape
                    )
                )
        model.set_state_dict(new_state_dict)

        if optimizer is not None:
            if os.path.exists(checkpoints + ".pdopt"):
                optim_dict = paddle.load(checkpoints + ".pdopt")
                optimizer.set_state_dict(optim_dict)
            else:
                logger.warning(
                    "{}.pdopt is not exists, params of optimizer is not loaded".format(
                        checkpoints
                    )
                )

        if os.path.exists(checkpoints + ".states"):
            with open(checkpoints + ".states", "rb") as f:
                states_dict = (
                    pickle.load(f) if six.PY2 else pickle.load(f, encoding="latin1")
                )
            best_model_dict = states_dict.get("best_model_dict", {})
            if "epoch" in states_dict:
                best_model_dict["start_epoch"] = states_dict["epoch"] + 1
        logger.info("resume from {}".format(checkpoints))
    elif pretrained_model:
        load_pretrained_params(model, pretrained_model)
    else:
        logger.info("train from scratch")
    return best_model_dict


def load_pretrained_params(model, path):
    logger = get_logger()
    if path.endswith(".pdparams"):
        path = path.replace(".pdparams", "")
    assert os.path.exists(
        path + ".pdparams"
    ), "The {}.pdparams does not exists!".format(path)

    params = paddle.load(path + ".pdparams")
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
                    "The shape of model params {} {} not matched with loaded params {} {} !".format(
                        k1, state_dict[k1].shape, k1, params[k1].shape
                    )
                )
    model.set_state_dict(new_state_dict)
    logger.info("load pretrain successful from {}".format(path))
    return model
