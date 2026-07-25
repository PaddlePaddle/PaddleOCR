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

import abc

import yaml
from paddlex import create_pipeline
from paddlex.inference import load_pipeline_config
from paddlex.utils.config import AttrDict
from paddlex.utils.deps import DependencyError

from .._abstract import CLISubcommandExecutor
from .._common_args import (
    add_common_cli_opts,
    parse_common_args,
    prepare_common_init_args,
)

_DEFAULT_ENABLE_HPI = None


def _merge_dicts(d1, d2):
    """Recursively merge d2 into d1, with d2 values taking precedence."""
    res = d1.copy()
    for k, v in d2.items():
        if k in res and isinstance(res[k], dict) and isinstance(v, dict):
            res[k] = _merge_dicts(res[k], v)
        else:
            res[k] = v
    return res


def _to_builtin(obj):
    """Recursively convert AttrDict and nested structures to plain Python dicts/lists."""
    if isinstance(obj, AttrDict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_builtin(item) for item in obj]
    else:
        return obj


class PaddleXPipelineWrapper(metaclass=abc.ABCMeta):
    """Base class for PaddleOCR pipeline wrappers.

    Subclasses wrap a PaddleX pipeline and expose a simplified Python API
    with optional CLI support. Each subclass must declare
    ``_paddlex_pipeline_name`` and ``get_cli_subcommand_executor``.

    Args:
        paddlex_config (str | dict | None): Path to a PaddleX pipeline YAML
            config file, a pre-loaded config dict, or ``None`` to use the
            default config for the pipeline.
        **common_args: Common inference arguments forwarded to PaddleX (e.g.
            ``device``, ``use_hpip``, ``use_tensorrt``).

    Example:
        >>> from paddleocr import PaddleOCR
        >>> ocr = PaddleOCR(lang="en")
        >>> results = ocr.predict("image.png")
    """

    def __init__(
        self,
        *,
        paddlex_config=None,
        **common_args,
    ):
        super().__init__()
        self._paddlex_config = paddlex_config
        self._common_args = parse_common_args(
            common_args, default_enable_hpi=_DEFAULT_ENABLE_HPI
        )
        self._merged_paddlex_config = self._get_merged_paddlex_config()
        self.paddlex_pipeline = self._create_paddlex_pipeline()

    @property
    @abc.abstractmethod
    def _paddlex_pipeline_name(self):
        raise NotImplementedError

    def export_paddlex_config_to_yaml(self, yaml_path):
        with open(yaml_path, "w", encoding="utf-8") as f:
            config = _to_builtin(self._merged_paddlex_config)
            yaml.safe_dump(config, f)

    def close(self):
        self.paddlex_pipeline.close()

    @classmethod
    @abc.abstractmethod
    def get_cli_subcommand_executor(cls):
        raise NotImplementedError

    def _get_paddlex_config_overrides(self):
        return {}

    def _get_merged_paddlex_config(self):
        if self._paddlex_config is None:
            config = load_pipeline_config(self._paddlex_pipeline_name)
        elif isinstance(self._paddlex_config, str):
            config = load_pipeline_config(self._paddlex_config)
        else:
            config = self._paddlex_config

        overrides = self._get_paddlex_config_overrides()

        return _merge_dicts(config, overrides)

    def _create_paddlex_pipeline(self):
        kwargs = prepare_common_init_args(None, self._common_args)
        try:
            return create_pipeline(config=self._merged_paddlex_config, **kwargs)
        except DependencyError as e:
            raise RuntimeError(
                "A dependency error occurred during pipeline creation. Please refer to the installation documentation to ensure all required dependencies are installed."
            ) from e


class PipelineCLISubcommandExecutor(CLISubcommandExecutor):
    """Base class for pipeline CLI subcommand executors.

    Registers a pipeline as a subcommand of the ``paddleocr`` CLI and handles
    argument parsing and execution.
    """

    @property
    @abc.abstractmethod
    def subparser_name(self):
        raise NotImplementedError

    def add_subparser(self, subparsers):
        subparser = subparsers.add_parser(name=self.subparser_name)
        self._update_subparser(subparser)
        add_common_cli_opts(
            subparser,
            default_enable_hpi=_DEFAULT_ENABLE_HPI,
            allow_multiple_devices=True,
        )
        subparser.add_argument(
            "--paddlex_config",
            type=str,
            help="Path to PaddleX pipeline configuration file.",
        )
        return subparser

    @abc.abstractmethod
    def _update_subparser(self, subparser):
        raise NotImplementedError
