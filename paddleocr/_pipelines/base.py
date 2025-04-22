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

# TODO: Should we use a third-party CLI library to auto-generate command-line
# arguments from the pipeline class, to reduce boilerplate and improve
# maintainability?

import abc

import yaml
from paddlex import create_pipeline
from paddlex.inference import PaddlePredictorOption, load_pipeline_config

from .._abstract import CLISubcommandExecutor
from ..utils.cli import str2bool

_DEFAULT_ENABLE_HPI = False
_DEFAULT_AUTO_PADDLE2ONNX = True
_DEFAULT_USE_TENSORRT = False
_DEFAULT_MIN_SUBGRAPH_SIZE = 3
_DEFAULT_PRECISION = "fp32"
_DEFAULT_ENABLE_MKLDNN = False
_DEFAULT_CPU_THREADS = 8
_SUPPORTED_PRECISION_LIST = ["fp32", "fp16"]


def _merge_dicts(d1, d2):
    res = d1.copy()
    for k, v in d2.items():
        if k in res and isinstance(res[k], dict) and isinstance(v, dict):
            res[k] = _merge_dicts(res[k], v)
        else:
            res[k] = v
    return res


class PaddleXPipelineWrapper(metaclass=abc.ABCMeta):
    def __init__(
        self,
        *,
        paddlex_config=None,
        device=None,
        enable_hpi=_DEFAULT_ENABLE_HPI,
        auto_paddle2onnx=_DEFAULT_AUTO_PADDLE2ONNX,
        use_tensorrt=_DEFAULT_USE_TENSORRT,
        min_subgraph_size=_DEFAULT_MIN_SUBGRAPH_SIZE,
        precision=_DEFAULT_PRECISION,
        enable_mkldnn=_DEFAULT_ENABLE_MKLDNN,
        cpu_threads=_DEFAULT_CPU_THREADS,
    ):
        super().__init__()
        self._paddlex_config = paddlex_config
        self._device = device
        self._enable_hpi = enable_hpi
        self._auto_paddle2onnx = auto_paddle2onnx
        self._use_pptrt = use_tensorrt
        self._pptrt_min_subgraph_size = min_subgraph_size
        if precision not in _SUPPORTED_PRECISION_LIST:
            raise ValueError(
                f"Invalid precision: {precision}. Supported values are: {_SUPPORTED_PRECISION_LIST}."
            )
        self._pptrt_precision = precision
        if self._use_pptrt and enable_mkldnn:
            raise ValueError("oneDNN should not be enabled when TensorRT is used.")
        self._enable_mkldnn = enable_mkldnn
        self._cpu_threads = cpu_threads
        self._merged_paddlex_config = self._get_merged_paddlex_config()
        self._paddlex_pipeline = self._create_paddlex_pipeline()

    @property
    @abc.abstractmethod
    def _paddlex_pipeline_name(self):
        raise NotImplementedError

    def export_paddlex_config_to_yaml(self, yaml_path):
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._merged_paddlex_config, f)

    @classmethod
    @abc.abstractmethod
    def get_cli_subcommand_executor(cls):
        raise NotImplementedError

    def _get_basic_paddlex_config_overrides(self):
        overrides = {}
        overrides["device"] = self._device
        if self._enable_hpi:
            overrides["use_hpip"] = True
            if self._auto_paddle2onnx:
                overrides["hpi_config"] = {
                    "auto_paddle2onnx": True,
                }
        pp_option = PaddlePredictorOption(None)
        assert not (self._use_pptrt and self._enable_mkldnn)
        if self._use_pptrt:
            if self._pptrt_precision == "fp32":
                pp_option.run_mode = "trt_fp32"
            else:
                assert self._pptrt_precision == "fp16", self._pptrt_precision
                pp_option.run_mode = "trt_fp16"
        elif self._enable_mkldnn:
            pp_option.run_mode = "mkldnn"
        pp_option.cpu_threads = self._cpu_threads
        return overrides

    def _get_extended_paddlex_config_overrides(self):
        return {}

    def _get_merged_paddlex_config(self):
        if self._paddlex_config is None:
            config = load_pipeline_config(self._paddlex_pipeline_name)
        elif isinstance(self._config, str):
            config = load_pipeline_config(self._paddlex_config)
        else:
            config = self._paddlex_config

        overrides = self._get_basic_paddlex_config_overrides()
        extended_overrides = self._get_extended_paddlex_config_overrides()
        if extended_overrides:
            overrides = _merge_dicts(overrides, extended_overrides)

        return _merge_dicts(config, overrides)

    def _create_paddlex_pipeline(self):
        return create_pipeline(config=self._merged_paddlex_config)


class PipelineCLISubcommandExecutor(CLISubcommandExecutor):
    @property
    @abc.abstractmethod
    def subparser_name(self):
        raise NotImplementedError

    def add_subparser(self, subparsers):
        subparser = subparsers.add_parser(name=self.subparser_name)
        subparser.add_argument(
            "--paddlex_config",
            type=str,
            help="Path to PaddleX pipeline configuration file.",
        )
        subparser.add_argument(
            "--device", type=str, help="Device to use for inference."
        )
        subparser.add_argument(
            "--enable_hpi",
            type=str2bool,
            default=_DEFAULT_ENABLE_HPI,
            help="Enable the high performance inference.",
        )
        subparser.add_argument(
            "--auto_paddle2onnx",
            type=str2bool,
            default=_DEFAULT_AUTO_PADDLE2ONNX,
            help="Whether to allow automatic Paddle-to-ONNX model conversion before performing inference.",
        )
        subparser.add_argument(
            "--use_tensorrt",
            type=str2bool,
            default=_DEFAULT_AUTO_PADDLE2ONNX,
            help="Whether to use the Paddle Inference TensorRT subgraph engine.",
        )
        subparser.add_argument(
            "--min_subgraph_size",
            type=int,
            default=_DEFAULT_MIN_SUBGRAPH_SIZE,
            help="Minimum subgraph size for TensorRT when using the Paddle Inference TensorRT subgraph engine.",
        )
        subparser.add_argument(
            "--precision",
            type=str,
            default=_DEFAULT_PRECISION,
            choices=_SUPPORTED_PRECISION_LIST,
            help="Precision for TensorRT when using the Paddle Inference TensorRT subgraph engine.",
        )
        subparser.add_argument(
            "--enable_mkldnn",
            type=str2bool,
            default=_DEFAULT_ENABLE_MKLDNN,
            help="Enalbe oneDNN (formerly known as MKL-DNN) acceleration for inference.",
        )
        subparser.add_argument(
            "--cpu_threads",
            type=int,
            default=_DEFAULT_CPU_THREADS,
            help="Number of threads to use for inference on CPUs.",
        )
        self._update_subparser(subparser)
        return subparser

    @abc.abstractmethod
    def _update_subparser(self, subparser):
        raise NotImplementedError
