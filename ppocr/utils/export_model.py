# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import yaml
import json
import copy
import paddle
import paddle.nn as nn
from paddle.jit import to_static

from collections import OrderedDict
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.logging import get_logger


def represent_dictionary_order(self, dict_data):
    return self.represent_mapping("tag:yaml.org,2002:map", dict_data.items())


def setup_orderdict():
    yaml.add_representer(OrderedDict, represent_dictionary_order)


def dump_infer_config(config, path, logger):
    setup_orderdict()
    infer_cfg = OrderedDict()
    if config["Global"].get("hpi_config_path", None):
        hpi_config = yaml.safe_load(open(config["Global"]["hpi_config_path"], "r"))
        rec_resize_img_dict = next(
            (
                item
                for item in config["Eval"]["dataset"]["transforms"]
                if "RecResizeImg" in item
            ),
            None,
        )
        if rec_resize_img_dict:
            dynamic_shapes = [1] + rec_resize_img_dict["RecResizeImg"]["image_shape"]
            if hpi_config["Hpi"]["backend_config"].get("paddle_tensorrt", None):
                hpi_config["Hpi"]["backend_config"]["paddle_tensorrt"][
                    "dynamic_shapes"
                ]["x"] = [dynamic_shapes for i in range(3)]
                hpi_config["Hpi"]["backend_config"]["paddle_tensorrt"][
                    "max_batch_size"
                ] = 1
            if hpi_config["Hpi"]["backend_config"].get("tensorrt", None):
                hpi_config["Hpi"]["backend_config"]["tensorrt"]["dynamic_shapes"][
                    "x"
                ] = [dynamic_shapes for i in range(3)]
                hpi_config["Hpi"]["backend_config"]["tensorrt"]["max_batch_size"] = 1
        else:
            if hpi_config["Hpi"]["backend_config"].get("paddle_tensorrt", None):
                hpi_config["Hpi"]["supported_backends"]["gpu"].remove("paddle_tensorrt")
                del hpi_config["Hpi"]["backend_config"]["paddle_tensorrt"]
            if hpi_config["Hpi"]["backend_config"].get("tensorrt", None):
                hpi_config["Hpi"]["supported_backends"]["gpu"].remove("tensorrt")
                del hpi_config["Hpi"]["backend_config"]["tensorrt"]
            hpi_config["Hpi"]["selected_backends"]["gpu"] = "paddle_infer"
        infer_cfg["Hpi"] = hpi_config["Hpi"]
    if config["Global"].get("pdx_model_name", None):
        infer_cfg["Global"] = {}
        infer_cfg["Global"]["model_name"] = config["Global"]["pdx_model_name"]

    infer_cfg["PreProcess"] = {"transform_ops": config["Eval"]["dataset"]["transforms"]}
    postprocess = OrderedDict()
    for k, v in config["PostProcess"].items():
        postprocess[k] = v

    if config["Architecture"].get("algorithm") in ["LaTeXOCR"]:
        tokenizer_file = config["Global"].get("rec_char_dict_path")
        if tokenizer_file is not None:
            with open(tokenizer_file, encoding="utf-8") as tokenizer_config_handle:
                character_dict = json.load(tokenizer_config_handle)
                postprocess["character_dict"] = character_dict
    else:
        if config["Global"].get("character_dict_path") is not None:
            with open(config["Global"]["character_dict_path"], encoding="utf-8") as f:
                lines = f.readlines()
                character_dict = [line.strip("\n") for line in lines]
            postprocess["character_dict"] = character_dict

    infer_cfg["PostProcess"] = postprocess

    with open(path, "w") as f:
        yaml.dump(
            infer_cfg, f, default_flow_style=False, encoding="utf-8", allow_unicode=True
        )
    logger.info("Export inference config file to {}".format(os.path.join(path)))


def export_single_model(
    model, arch_config, save_path, logger, input_shape=None, quanter=None
):
    if arch_config["algorithm"] == "SRN":
        max_text_length = arch_config["Head"]["max_text_length"]
        other_shape = [
            paddle.static.InputSpec(shape=[None, 1, 64, 256], dtype="float32"),
            [
                paddle.static.InputSpec(shape=[None, 256, 1], dtype="int64"),
                paddle.static.InputSpec(
                    shape=[None, max_text_length, 1], dtype="int64"
                ),
                paddle.static.InputSpec(
                    shape=[None, 8, max_text_length, max_text_length], dtype="int64"
                ),
                paddle.static.InputSpec(
                    shape=[None, 8, max_text_length, max_text_length], dtype="int64"
                ),
            ],
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "SAR":
        other_shape = [
            paddle.static.InputSpec(shape=[None, 3, 48, 160], dtype="float32"),
            [paddle.static.InputSpec(shape=[None], dtype="float32")],
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] in ["SVTR_LCNet", "SVTR_HGNet"]:
        other_shape = [
            paddle.static.InputSpec(shape=[None, 3, 48, -1], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] in ["SVTR", "CPPD"]:
        other_shape = [
            paddle.static.InputSpec(shape=[None] + input_shape, dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "PREN":
        other_shape = [
            paddle.static.InputSpec(shape=[None, 3, 64, 256], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["model_type"] == "sr":
        other_shape = [
            paddle.static.InputSpec(shape=[None, 3, 16, 64], dtype="float32")
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "ViTSTR":
        other_shape = [
            paddle.static.InputSpec(shape=[None, 1, 224, 224], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "ABINet":
        if not input_shape:
            input_shape = [3, 32, 128]
        other_shape = [
            paddle.static.InputSpec(shape=[None] + input_shape, dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] in ["NRTR", "SPIN", "RFL"]:
        other_shape = [
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] in ["SATRN"]:
        other_shape = [
            paddle.static.InputSpec(shape=[None, 3, 32, 100], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "VisionLAN":
        other_shape = [
            paddle.static.InputSpec(shape=[None, 3, 64, 256], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "RobustScanner":
        max_text_length = arch_config["Head"]["max_text_length"]
        other_shape = [
            paddle.static.InputSpec(shape=[None, 3, 48, 160], dtype="float32"),
            [
                paddle.static.InputSpec(
                    shape=[
                        None,
                    ],
                    dtype="float32",
                ),
                paddle.static.InputSpec(shape=[None, max_text_length], dtype="int64"),
            ],
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "CAN":
        other_shape = [
            [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype="float32"),
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype="float32"),
                paddle.static.InputSpec(
                    shape=[None, arch_config["Head"]["max_text_length"]], dtype="int64"
                ),
            ]
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "LaTeXOCR":
        other_shape = [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] in ["LayoutLM", "LayoutLMv2", "LayoutXLM"]:
        input_spec = [
            paddle.static.InputSpec(shape=[None, 512], dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, 512, 4], dtype="int64"),  # bbox
            paddle.static.InputSpec(shape=[None, 512], dtype="int64"),  # attention_mask
            paddle.static.InputSpec(shape=[None, 512], dtype="int64"),  # token_type_ids
            paddle.static.InputSpec(shape=[None, 3, 224, 224], dtype="int64"),  # image
        ]
        if "Re" in arch_config["Backbone"]["name"]:
            input_spec.extend(
                [
                    paddle.static.InputSpec(
                        shape=[None, 512, 3], dtype="int64"
                    ),  # entities
                    paddle.static.InputSpec(
                        shape=[None, None, 2], dtype="int64"
                    ),  # relations
                ]
            )
        if model.backbone.use_visual_backbone is False:
            input_spec.pop(4)
        model = to_static(model, input_spec=[input_spec])
    else:
        infer_shape = [3, -1, -1]
        if arch_config["model_type"] == "rec":
            infer_shape = [3, 32, -1]  # for rec model, H must be 32
            if (
                "Transform" in arch_config
                and arch_config["Transform"] is not None
                and arch_config["Transform"]["name"] == "TPS"
            ):
                logger.info(
                    "When there is tps in the network, variable length input is not supported, and the input size needs to be the same as during training"
                )
                infer_shape[-1] = 100
        elif arch_config["model_type"] == "table":
            infer_shape = [3, 488, 488]
            if arch_config["algorithm"] == "TableMaster":
                infer_shape = [3, 480, 480]
            if arch_config["algorithm"] == "SLANet":
                infer_shape = [3, -1, -1]
        model = to_static(
            model,
            input_spec=[
                paddle.static.InputSpec(shape=[None] + infer_shape, dtype="float32")
            ],
        )

    if (
        arch_config["model_type"] != "sr"
        and arch_config["Backbone"]["name"] == "PPLCNetV3"
    ):
        # for rep lcnetv3
        for layer in model.sublayers():
            if hasattr(layer, "rep") and not getattr(layer, "is_repped"):
                layer.rep()

    if quanter is None:
        paddle.jit.save(model, save_path)
    else:
        quanter.save_quantized_model(model, save_path)
    logger.info("inference model is saved to {}".format(save_path))
    return


def export(config, base_model=None, save_path=None):
    if paddle.distributed.get_rank() != 0:
        return
    logger = get_logger()
    # build post process
    post_process_class = build_post_process(config["PostProcess"], config["Global"])

    # build model
    # for rec algorithm
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in [
            "Distillation",
        ]:  # distillation model
            for key in config["Architecture"]["Models"]:
                if (
                    config["Architecture"]["Models"][key]["Head"]["name"] == "MultiHead"
                ):  # multi head
                    out_channels_list = {}
                    if config["PostProcess"]["name"] == "DistillationSARLabelDecode":
                        char_num = char_num - 2
                    if config["PostProcess"]["name"] == "DistillationNRTRLabelDecode":
                        char_num = char_num - 3
                    out_channels_list["CTCLabelDecode"] = char_num
                    out_channels_list["SARLabelDecode"] = char_num + 2
                    out_channels_list["NRTRLabelDecode"] = char_num + 3
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels_list"
                    ] = out_channels_list
                else:
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels"
                    ] = char_num
                # just one final tensor needs to exported for inference
                config["Architecture"]["Models"][key]["return_all_feats"] = False
        elif config["Architecture"]["Head"]["name"] == "MultiHead":  # multi head
            out_channels_list = {}
            char_num = len(getattr(post_process_class, "character"))
            if config["PostProcess"]["name"] == "SARLabelDecode":
                char_num = char_num - 2
            if config["PostProcess"]["name"] == "NRTRLabelDecode":
                char_num = char_num - 3
            out_channels_list["CTCLabelDecode"] = char_num
            out_channels_list["SARLabelDecode"] = char_num + 2
            out_channels_list["NRTRLabelDecode"] = char_num + 3
            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        else:  # base rec model
            config["Architecture"]["Head"]["out_channels"] = char_num

    # for sr algorithm
    if config["Architecture"]["model_type"] == "sr":
        config["Architecture"]["Transform"]["infer_mode"] = True

    # for latexocr algorithm
    if config["Architecture"].get("algorithm") in ["LaTeXOCR"]:
        config["Architecture"]["Backbone"]["is_predict"] = True
        config["Architecture"]["Backbone"]["is_export"] = True
        config["Architecture"]["Head"]["is_export"] = True
    if base_model is not None:
        model = base_model
        if isinstance(model, paddle.DataParallel):
            model = copy.deepcopy(model._layers)
        else:
            model = copy.deepcopy(model)
    else:
        model = build_model(config["Architecture"])
        load_model(config, model, model_type=config["Architecture"]["model_type"])
    model.eval()

    if not save_path:
        save_path = config["Global"]["save_inference_dir"]
    yaml_path = os.path.join(save_path, "inference.yml")

    arch_config = config["Architecture"]

    if (
        arch_config["algorithm"] in ["SVTR", "CPPD"]
        and arch_config["Head"]["name"] != "MultiHead"
    ):
        input_shape = config["Eval"]["dataset"]["transforms"][-2]["SVTRRecResizeImg"][
            "image_shape"
        ]
    elif arch_config["algorithm"].lower() == "ABINet".lower():
        rec_rs = [
            c
            for c in config["Eval"]["dataset"]["transforms"]
            if "ABINetRecResizeImg" in c
        ]
        input_shape = rec_rs[0]["ABINetRecResizeImg"]["image_shape"] if rec_rs else None
    else:
        input_shape = None

    if arch_config["algorithm"] in [
        "Distillation",
    ]:  # distillation model
        archs = list(arch_config["Models"].values())
        for idx, name in enumerate(model.model_name_list):
            sub_model_save_path = os.path.join(save_path, name, "inference")
            export_single_model(
                model.model_list[idx], archs[idx], sub_model_save_path, logger
            )
    else:
        save_path = os.path.join(save_path, "inference")
        export_single_model(
            model, arch_config, save_path, logger, input_shape=input_shape
        )
    dump_infer_config(config, yaml_path, logger)
