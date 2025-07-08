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
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..", "..", "..", "tools")))

import argparse

import paddle
from paddle.jit import to_static

from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.logging import get_logger
from tools.program import load_config, merge_config, ArgsParser
from ppocr.metrics import build_metric
import tools.program as program
from paddleslim.dygraph.quant import QAT
from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.utils.export_model import export_single_model


def main():
    ############################################################################################################
    # 1. quantization configs
    ############################################################################################################
    quant_config = {
        # weight preprocess type, default is None and no preprocessing is performed.
        "weight_preprocess_type": None,
        # activation preprocess type, default is None and no preprocessing is performed.
        "activation_preprocess_type": None,
        # weight quantize type, default is 'channel_wise_abs_max'
        "weight_quantize_type": "channel_wise_abs_max",
        # activation quantize type, default is 'moving_average_abs_max'
        "activation_quantize_type": "moving_average_abs_max",
        # weight quantize bit num, default is 8
        "weight_bits": 8,
        # activation quantize bit num, default is 8
        "activation_bits": 8,
        # data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'
        "dtype": "int8",
        # window size for 'range_abs_max' quantization. default is 10000
        "window_size": 10000,
        # The decay coefficient of moving average, default is 0.9
        "moving_rate": 0.9,
        # for dygraph quantization, layers of type in quantizable_layer_type will be quantized
        "quantizable_layer_type": ["Conv2D", "Linear"],
    }
    FLAGS = ArgsParser().parse_args()
    config = load_config(FLAGS.config)
    config = merge_config(config, FLAGS.opt)
    logger = get_logger()

    # build dataloader
    set_signal_handlers()
    valid_dataloader = build_dataloader(config, "Eval", device, logger)

    # build post process
    post_process_class = build_post_process(config["PostProcess"], config["Global"])

    # build model
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in [
            "Distillation",
        ]:  # distillation model
            for key in config["Architecture"]["Models"]:
                if (
                    config["Architecture"]["Models"][key]["Head"]["name"] == "MultiHead"
                ):  # for multi head
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
        elif config["Architecture"]["Head"]["name"] == "MultiHead":  # for multi head
            out_channels_list = {}
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

    model = build_model(config["Architecture"])
    extra_input_models = [
        "SRN",
        "NRTR",
        "SAR",
        "SEED",
        "SVTR",
        "SVTR_LCNet",
        "VisionLAN",
        "RobustScanner",
        "SVTR_HGNet",
    ]
    extra_input = False
    if config["Architecture"]["algorithm"] == "Distillation":
        for key in config["Architecture"]["Models"]:
            extra_input = (
                extra_input
                or config["Architecture"]["Models"][key]["algorithm"]
                in extra_input_models
            )
    else:
        extra_input = config["Architecture"]["algorithm"] in extra_input_models
    if "model_type" in config["Architecture"].keys():
        if config["Architecture"]["algorithm"] == "CAN":
            model_type = "can"
        elif config["Architecture"]["algorithm"] == "LaTeXOCR":
            model_type = "latexocr"
            config["Metric"]["cal_bleu_score"] = True
        elif config["Architecture"]["algorithm"] == "UniMERNet":
            model_type = "unimernet"
            config["Metric"]["cal_bleu_score"] = True
        elif config["Architecture"]["algorithm"] in [
            "PP-FormulaNet-S",
            "PP-FormulaNet-L",
        ]:
            model_type = "pp_formulanet"
            config["Metric"]["cal_bleu_score"] = True
        else:
            model_type = config["Architecture"]["model_type"]
    else:
        model_type = None

    # get QAT model
    quanter = QAT(config=quant_config)
    quanter.quantize(model)

    # build metric
    eval_class = build_metric(config["Metric"])
    # amp
    use_amp = config["Global"].get("use_amp", False)
    amp_level = config["Global"].get("amp_level", "O2")
    amp_custom_black_list = config["Global"].get("amp_custom_black_list", [])
    if use_amp:
        AMP_RELATED_FLAGS_SETTING = {
            "FLAGS_cudnn_batchnorm_spatial_persistent": 1,
        }
        paddle.set_flags(AMP_RELATED_FLAGS_SETTING)
        scale_loss = config["Global"].get("scale_loss", 1.0)
        use_dynamic_loss_scaling = config["Global"].get(
            "use_dynamic_loss_scaling", False
        )
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=scale_loss,
            use_dynamic_loss_scaling=use_dynamic_loss_scaling,
        )
        if amp_level == "O2":
            model = paddle.amp.decorate(
                models=model, level=amp_level, master_weight=True
            )
    else:
        scaler = None

    best_model_dict = load_model(
        config, model, model_type=config["Architecture"]["model_type"]
    )
    if len(best_model_dict):
        logger.info("metric in ckpt ***************")
        for k, v in best_model_dict.items():
            logger.info("{}:{}".format(k, v))

    # start eval
    metric = program.eval(
        model,
        valid_dataloader,
        post_process_class,
        eval_class,
        model_type,
        extra_input,
        scaler,
        amp_level,
        amp_custom_black_list,
    )
    model.eval()

    logger.info("metric eval ***************")
    for k, v in metric.items():
        logger.info("{}:{}".format(k, v))

    save_path = config["Global"]["save_inference_dir"]

    arch_config = config["Architecture"]

    if (
        arch_config["algorithm"] == "SVTR"
        and arch_config["Head"]["name"] != "MultiHead"
    ):
        input_shape = config["Eval"]["dataset"]["transforms"][-2]["SVTRRecResizeImg"][
            "image_shape"
        ]
    else:
        input_shape = None

    if arch_config["algorithm"] in [
        "Distillation",
    ]:  # distillation model
        archs = list(arch_config["Models"].values())
        for idx, name in enumerate(model.model_name_list):
            sub_model_save_path = os.path.join(save_path, name, "inference")
            export_single_model(
                model.model_list[idx],
                archs[idx],
                sub_model_save_path,
                logger,
                None,
                config,
                input_shape,
                quanter,
            )
    else:
        save_path = os.path.join(save_path, "inference")
        export_single_model(
            model,
            arch_config,
            save_path,
            logger,
            None,
            config,
            input_shape,
            quanter,
        )


if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess()
    main()
