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
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import argparse

import paddle
from paddle.jit import to_static

from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.logging import get_logger
from tools.program import load_config, merge_config, ArgsParser


def export_single_model(model, arch_config, save_path, logger, quanter=None):
    if arch_config["algorithm"] == "SRN":
        max_text_length = arch_config["Head"]["max_text_length"]
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 1, 64, 256], dtype="float32"), [
                    paddle.static.InputSpec(
                        shape=[None, 256, 1],
                        dtype="int64"), paddle.static.InputSpec(
                            shape=[None, max_text_length, 1], dtype="int64"),
                    paddle.static.InputSpec(
                        shape=[None, 8, max_text_length, max_text_length],
                        dtype="int64"), paddle.static.InputSpec(
                            shape=[None, 8, max_text_length, max_text_length],
                            dtype="int64")
                ]
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "SAR":
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 48, 160], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "SVTR":
        if arch_config["Head"]["name"] == 'MultiHead':
            other_shape = [
                paddle.static.InputSpec(
                    shape=[None, 3, 48, -1], dtype="float32"),
            ]
        else:
            other_shape = [
                paddle.static.InputSpec(
                    shape=[None, 3, 64, 256], dtype="float32"),
            ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "PREN":
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 64, 512], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    else:
        infer_shape = [3, -1, -1]
        if arch_config["model_type"] == "rec":
            infer_shape = [3, 48, -1]  # for rec model, H must be 32
            if "Transform" in arch_config and arch_config[
                    "Transform"] is not None and arch_config["Transform"][
                        "name"] == "TPS":
                logger.info(
                    "When there is tps in the network, variable length input is not supported, and the input size needs to be the same as during training"
                )
                infer_shape[-1] = 100
            if arch_config["algorithm"] == "NRTR":
                infer_shape = [1, 32, 100]
        elif arch_config["model_type"] == "table":
            infer_shape = [3, 488, 488]
        model = to_static(
            model,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None] + infer_shape, dtype="float32")
            ])

    if quanter is None:
        paddle.jit.save(model, save_path)
    else:
        quanter.save_quantized_model(model, save_path)
    logger.info("inference model is saved to {}".format(save_path))
    return


def main():
    FLAGS = ArgsParser().parse_args()
    config = load_config(FLAGS.config)
    config = merge_config(config, FLAGS.opt)
    logger = get_logger()
    # build post process

    post_process_class = build_post_process(config["PostProcess"],
                                            config["Global"])

    # build model
    # for rec algorithm
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in ["Distillation",
                                                   ]:  # distillation model
            for key in config["Architecture"]["Models"]:
                if config["Architecture"]["Models"][key]["Head"][
                        "name"] == 'MultiHead':  # multi head
                    out_channels_list = {}
                    if config['PostProcess'][
                            'name'] == 'DistillationSARLabelDecode':
                        char_num = char_num - 2
                    out_channels_list['CTCLabelDecode'] = char_num
                    out_channels_list['SARLabelDecode'] = char_num + 2
                    config['Architecture']['Models'][key]['Head'][
                        'out_channels_list'] = out_channels_list
                else:
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels"] = char_num
                # just one final tensor needs to exported for inference
                config["Architecture"]["Models"][key][
                    "return_all_feats"] = False
        elif config['Architecture']['Head'][
                'name'] == 'MultiHead':  # multi head
            out_channels_list = {}
            char_num = len(getattr(post_process_class, 'character'))
            if config['PostProcess']['name'] == 'SARLabelDecode':
                char_num = char_num - 2
            out_channels_list['CTCLabelDecode'] = char_num
            out_channels_list['SARLabelDecode'] = char_num + 2
            config['Architecture']['Head'][
                'out_channels_list'] = out_channels_list
        else:  # base rec model
            config["Architecture"]["Head"]["out_channels"] = char_num

    model = build_model(config["Architecture"])
    load_model(config, model)
    model.eval()

    save_path = config["Global"]["save_inference_dir"]

    arch_config = config["Architecture"]

    if arch_config["algorithm"] in ["Distillation", ]:  # distillation model
        archs = list(arch_config["Models"].values())
        for idx, name in enumerate(model.model_name_list):
            sub_model_save_path = os.path.join(save_path, name, "inference")
            export_single_model(model.model_list[idx], archs[idx],
                                sub_model_save_path, logger)
    else:
        save_path = os.path.join(save_path, "inference")
        export_single_model(model, arch_config, save_path, logger)


if __name__ == "__main__":
    main()
