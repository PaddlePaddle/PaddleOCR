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
import json

import paddle

from ppocr.utils.logging import get_logger
from ppocr.utils.network import maybe_download_params

try:
    import encryption  # Attempt to import the encryption module for AIStudio's encryption model

    encrypted = encryption.is_encryption_needed()
except ImportError:
    get_logger().warning("Skipping import of the encryption module.")
    encrypted = False  # Encryption is not needed if the module cannot be imported

__all__ = ["load_model"]


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
                    "be happy if some process has already created {}".format(path)
                )
            else:
                raise OSError("Failed to mkdir {}".format(path))


def load_model(config, model, optimizer=None, model_type="det"):
    """
    load model from checkpoint or pretrained_model
    """
    logger = get_logger()
    global_config = config["Global"]
    checkpoints = global_config.get("checkpoints")
    pretrained_model = global_config.get("pretrained_model")
    best_model_dict = {}
    is_float16 = False
    is_nlp_model = model_type == "kie" and config["Architecture"]["algorithm"] not in [
        "SDMGR"
    ]

    if is_nlp_model is True:
        # NOTE: for kie model dsitillation, resume training is not supported now
        if config["Architecture"]["algorithm"] in ["Distillation"]:
            return best_model_dict
        checkpoints = config["Architecture"]["Backbone"]["checkpoints"]
        # load kie method metric
        if checkpoints:
            if os.path.exists(os.path.join(checkpoints, "metric.states")):
                with open(os.path.join(checkpoints, "metric.states"), "rb") as f:
                    states_dict = pickle.load(f, encoding="latin1")
                best_model_dict = states_dict.get("best_model_dict", {})
                if "epoch" in states_dict:
                    best_model_dict["start_epoch"] = states_dict["epoch"] + 1
            logger.info("resume from {}".format(checkpoints))

            if optimizer is not None:
                if checkpoints[-1] in ["/", "\\"]:
                    checkpoints = checkpoints[:-1]
                if os.path.exists(checkpoints + ".pdopt"):
                    optim_dict = paddle.load(checkpoints + ".pdopt")
                    optimizer.set_state_dict(optim_dict)
                else:
                    logger.warning(
                        "{}.pdopt is not exists, params of optimizer is not loaded".format(
                            checkpoints
                        )
                    )

        return best_model_dict

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
            if pre_value.dtype == paddle.float16:
                is_float16 = True
            if pre_value.dtype != value.dtype:
                pre_value = pre_value.astype(value.dtype)
            if list(value.shape) == list(pre_value.shape):
                new_state_dict[key] = pre_value
            else:
                logger.warning(
                    "The shape of model params {} {} not matched with loaded params shape {} !".format(
                        key, value.shape, pre_value.shape
                    )
                )
        model.set_state_dict(new_state_dict)
        if is_float16:
            logger.info(
                "The parameter type is float16, which is converted to float32 when loading"
            )
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
                states_dict = pickle.load(f, encoding="latin1")
            best_model_dict = states_dict.get("best_model_dict", {})
            best_model_dict["acc"] = 0.0
            if "epoch" in states_dict:
                best_model_dict["start_epoch"] = states_dict["epoch"] + 1
        logger.info("resume from {}".format(checkpoints))
    elif pretrained_model:
        is_float16 = load_pretrained_params(model, pretrained_model)
    else:
        logger.info("train from scratch")
    best_model_dict["is_float16"] = is_float16
    return best_model_dict


def load_pretrained_params(model, path):
    logger = get_logger()
    path = maybe_download_params(path)
    if path.endswith(".pdparams"):
        path = path.replace(".pdparams", "")
    assert os.path.exists(
        path + ".pdparams"
    ), "The {}.pdparams does not exists!".format(path)

    params = paddle.load(path + ".pdparams")

    state_dict = model.state_dict()

    new_state_dict = {}
    is_float16 = False

    for k1 in params.keys():
        if k1 not in state_dict.keys():
            logger.warning("The pretrained params {} not in model".format(k1))
        else:
            if params[k1].dtype == paddle.float16:
                is_float16 = True
            if params[k1].dtype != state_dict[k1].dtype:
                params[k1] = params[k1].astype(state_dict[k1].dtype)
            if list(state_dict[k1].shape) == list(params[k1].shape):
                new_state_dict[k1] = params[k1]
            else:
                logger.warning(
                    "The shape of model params {} {} not matched with loaded params {} {} !".format(
                        k1, state_dict[k1].shape, k1, params[k1].shape
                    )
                )

    model.set_state_dict(new_state_dict)
    if is_float16:
        logger.info(
            "The parameter type is float16, which is converted to float32 when loading"
        )
    logger.info("load pretrain successful from {}".format(path))
    return is_float16


def save_model(
    model,
    optimizer,
    model_path,
    logger,
    config,
    is_best=False,
    prefix="ppocr",
    **kwargs,
):
    """
    save model to the target path
    """
    _mkdir_if_not_exist(model_path, logger)
    model_prefix = os.path.join(model_path, prefix)

    if prefix == "best_accuracy":
        best_model_path = os.path.join(model_path, "best_model")
        _mkdir_if_not_exist(best_model_path, logger)

    paddle.save(optimizer.state_dict(), model_prefix + ".pdopt")
    if prefix == "best_accuracy":
        paddle.save(
            optimizer.state_dict(), os.path.join(best_model_path, "model.pdopt")
        )

    is_nlp_model = config["Architecture"]["model_type"] == "kie" and config[
        "Architecture"
    ]["algorithm"] not in ["SDMGR"]
    if is_nlp_model is not True:
        paddle.save(model.state_dict(), model_prefix + ".pdparams")
        metric_prefix = model_prefix

        if prefix == "best_accuracy":
            paddle.save(
                model.state_dict(), os.path.join(best_model_path, "model.pdparams")
            )

    else:  # for kie system, we follow the save/load rules in NLP
        if config["Global"]["distributed"]:
            arch = model._layers
        else:
            arch = model
        if config["Architecture"]["algorithm"] in ["Distillation"]:
            arch = arch.Student
        arch.backbone.model.save_pretrained(model_prefix)
        metric_prefix = os.path.join(model_prefix, "metric")

        if prefix == "best_accuracy":
            arch.backbone.model.save_pretrained(best_model_path)

    save_model_info = kwargs.pop("save_model_info", False)
    if save_model_info:
        with open(os.path.join(model_path, f"{prefix}.info.json"), "w") as f:
            json.dump(kwargs, f)
        logger.info("Already save model info in {}".format(model_path))
        if prefix != "latest":
            done_flag = kwargs.pop("done_flag", False)
            update_train_results(config, prefix, save_model_info, done_flag=done_flag)

    # save metric and config
    with open(metric_prefix + ".states", "wb") as f:
        pickle.dump(kwargs, f, protocol=2)
    if is_best:
        logger.info("save best model is to {}".format(model_prefix))
    else:
        logger.info("save model in {}".format(model_prefix))


def update_train_results(config, prefix, metric_info, done_flag=False, last_num=5):
    if paddle.distributed.get_rank() != 0:
        return

    assert last_num >= 1
    train_results_path = os.path.join(
        config["Global"]["save_model_dir"], "train_result.json"
    )
    save_model_tag = ["pdparams", "pdopt", "pdstates"]
    save_inference_tag = ["inference_config", "pdmodel", "pdiparams", "pdiparams.info"]
    if os.path.exists(train_results_path):
        with open(train_results_path, "r") as fp:
            train_results = json.load(fp)
    else:
        train_results = {}
        train_results["model_name"] = config["Global"]["pdx_model_name"]
        label_dict_path = config["Global"].get("character_dict_path", "")
        if label_dict_path != "":
            label_dict_path = os.path.abspath(label_dict_path)
            if not os.path.exists(label_dict_path):
                label_dict_path = ""
        train_results["label_dict"] = label_dict_path
        train_results["train_log"] = "train.log"
        train_results["visualdl_log"] = ""
        train_results["config"] = "config.yaml"
        train_results["models"] = {}
        for i in range(1, last_num + 1):
            train_results["models"][f"last_{i}"] = {}
        train_results["models"]["best"] = {}
    train_results["done_flag"] = done_flag
    if "best" in prefix:
        if "acc" in metric_info["metric"]:
            metric_score = metric_info["metric"]["acc"]
        elif "precision" in metric_info["metric"]:
            metric_score = metric_info["metric"]["precision"]
        elif "exp_rate" in metric_info["metric"]:
            metric_score = metric_info["metric"]["exp_rate"]
        else:
            raise ValueError("No metric score found.")
        train_results["models"]["best"]["score"] = metric_score
        for tag in save_model_tag:
            if tag == "pdparams" and encrypted:
                train_results["models"]["best"][tag] = os.path.join(
                    prefix,
                    (
                        f"{prefix}.encrypted.{tag}"
                        if tag != "pdstates"
                        else f"{prefix}.states"
                    ),
                )
            else:
                train_results["models"]["best"][tag] = os.path.join(
                    prefix,
                    f"{prefix}.{tag}" if tag != "pdstates" else f"{prefix}.states",
                )
        for tag in save_inference_tag:
            train_results["models"]["best"][tag] = os.path.join(
                prefix,
                "inference",
                f"inference.{tag}" if tag != "inference_config" else "inference.yml",
            )
    else:
        for i in range(last_num - 1, 0, -1):
            train_results["models"][f"last_{i + 1}"] = train_results["models"][
                f"last_{i}"
            ].copy()
        if "acc" in metric_info["metric"]:
            metric_score = metric_info["metric"]["acc"]
        elif "precision" in metric_info["metric"]:
            metric_score = metric_info["metric"]["precision"]
        elif "exp_rate" in metric_info["metric"]:
            metric_score = metric_info["metric"]["exp_rate"]
        else:
            metric_score = 0
        train_results["models"][f"last_{1}"]["score"] = metric_score
        for tag in save_model_tag:
            if tag == "pdparams" and encrypted:
                train_results["models"][f"last_{1}"][tag] = os.path.join(
                    prefix,
                    (
                        f"{prefix}.encrypted.{tag}"
                        if tag != "pdstates"
                        else f"{prefix}.states"
                    ),
                )
            else:
                train_results["models"][f"last_{1}"][tag] = os.path.join(
                    prefix,
                    f"{prefix}.{tag}" if tag != "pdstates" else f"{prefix}.states",
                )
        for tag in save_inference_tag:
            train_results["models"][f"last_{1}"][tag] = os.path.join(
                prefix,
                "inference",
                f"inference.{tag}" if tag != "inference_config" else "inference.yml",
            )

    with open(train_results_path, "w") as fp:
        json.dump(train_results, fp)
