# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddlex.inference import PaddlePredictorOption
from paddlex.utils.device import get_default_device, parse_device

from ._constants import (
    DEFAULT_CPU_THREADS,
    DEFAULT_DEVICE,
    DEFAULT_ENABLE_MKLDNN,
    DEFAULT_MKLDNN_CACHE_CAPACITY,
    DEFAULT_PRECISION,
    DEFAULT_USE_TENSORRT,
    SUPPORTED_PRECISION_LIST,
    DEFAULT_USE_CINN,
)
from ._utils.cli import str2bool


def parse_common_args(kwargs, *, default_enable_hpi):
    default_vals = {
        "device": DEFAULT_DEVICE,
        "enable_hpi": default_enable_hpi,
        "use_tensorrt": DEFAULT_USE_TENSORRT,
        "precision": DEFAULT_PRECISION,
        "enable_mkldnn": DEFAULT_ENABLE_MKLDNN,
        "mkldnn_cache_capacity": DEFAULT_MKLDNN_CACHE_CAPACITY,
        "cpu_threads": DEFAULT_CPU_THREADS,
        "enable_cinn": DEFAULT_USE_CINN,
    }

    unknown_names = kwargs.keys() - default_vals.keys()
    for name in unknown_names:
        raise ValueError(f"Unknown argument: {name}")

    kwargs = {**default_vals, **kwargs}

    if kwargs["precision"] not in SUPPORTED_PRECISION_LIST:
        raise ValueError(
            f"Invalid precision: {kwargs['precision']}. Supported values are: {SUPPORTED_PRECISION_LIST}."
        )

    kwargs["use_pptrt"] = kwargs.pop("use_tensorrt")
    kwargs["pptrt_precision"] = kwargs.pop("precision")

    return kwargs


def prepare_common_init_args(model_name, common_args):
    device = common_args["device"]
    if device is None:
        device = get_default_device()
    device_type, _ = parse_device(device)

    init_kwargs = {}
    init_kwargs["device"] = device
    init_kwargs["use_hpip"] = common_args["enable_hpi"]

    pp_option = PaddlePredictorOption()
    if device_type == "gpu":
        if common_args["use_pptrt"]:
            if common_args["pptrt_precision"] == "fp32":
                pp_option.run_mode = "trt_fp32"
            else:
                assert common_args["pptrt_precision"] == "fp16", common_args[
                    "pptrt_precision"
                ]
                pp_option.run_mode = "trt_fp16"
        else:
            pp_option.run_mode = "paddle"
    elif device_type == "cpu":
        enable_mkldnn = common_args["enable_mkldnn"]
        if enable_mkldnn:
            pp_option.mkldnn_cache_capacity = common_args["mkldnn_cache_capacity"]
        else:
            pp_option.run_mode = "paddle"
        pp_option.cpu_threads = common_args["cpu_threads"]
    else:
        pp_option.run_mode = "paddle"
    pp_option.enable_cinn = common_args["enable_cinn"]
    init_kwargs["pp_option"] = pp_option

    return init_kwargs


def add_common_cli_opts(parser, *, default_enable_hpi, allow_multiple_devices):
    if allow_multiple_devices:
        help_ = "Device(s) to use for inference, e.g., `cpu`, `gpu`, `npu`, `gpu:0`, `gpu:0,1`. If multiple devices are specified, inference will be performed in parallel. Note that parallel inference is not always supported. By default, GPU 0 will be used if available; otherwise, the CPU will be used."
    else:
        help_ = "Device to use for inference, e.g., `cpu`, `gpu`, `npu`, `gpu:0`. By default, GPU 0 will be used if available; otherwise, the CPU will be used."
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=help_,
    )
    parser.add_argument(
        "--enable_hpi",
        type=str2bool,
        default=default_enable_hpi,
        help="Enable the high performance inference.",
    )
    parser.add_argument(
        "--use_tensorrt",
        type=str2bool,
        default=DEFAULT_USE_TENSORRT,
        help="Whether to use the Paddle Inference TensorRT subgraph engine. If the model does not support TensorRT acceleration, even if this flag is set, acceleration will not be used.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=DEFAULT_PRECISION,
        choices=SUPPORTED_PRECISION_LIST,
        help="Precision for TensorRT when using the Paddle Inference TensorRT subgraph engine.",
    )
    parser.add_argument(
        "--enable_mkldnn",
        type=str2bool,
        default=DEFAULT_ENABLE_MKLDNN,
        help="Enable MKL-DNN acceleration for inference. If MKL-DNN is unavailable or the model does not support it, acceleration will not be used even if this flag is set.",
    )
    parser.add_argument(
        "--mkldnn_cache_capacity",
        type=int,
        default=DEFAULT_MKLDNN_CACHE_CAPACITY,
        help="MKL-DNN cache capacity.",
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=DEFAULT_CPU_THREADS,
        help="Number of threads to use for inference on CPUs.",
    )
    parser.add_argument(
        "--enable_cinn",
        type=str2bool,
        default=DEFAULT_USE_CINN,
        help="Whether to use the CINN compiler.",
    )
