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

from paddlex import create_predictor
from paddlex.utils.deps import DependencyError

from .._abstract import CLISubcommandExecutor
from .._common_args import (
    add_common_cli_opts,
    parse_common_args,
    prepare_common_init_args,
)

_DEFAULT_ENABLE_HPI = False


class PaddleXPredictorWrapper(metaclass=abc.ABCMeta):
    """Base class for single-model PaddleOCR wrappers.

    Subclasses wrap a PaddleX predictor for a specific model (e.g. text
    detection, text recognition) and expose ``predict`` / ``predict_iter``
    methods along with optional CLI support.  Each subclass must declare
    ``default_model_name`` and ``get_cli_subcommand_executor``.

    Args:
        model_name (str | None): Name of the model to load. Defaults to
            ``default_model_name`` when ``None``.
        model_dir (str | None): Local directory containing model files.
            Downloads from the model hub when ``None``.
        **common_args: Common inference arguments forwarded to PaddleX (e.g.
            ``device``, ``use_hpip``, ``use_tensorrt``).

    Example:
        >>> from paddleocr import TextDetection
        >>> detector = TextDetection()
        >>> results = detector.predict("image.png")
    """

    def __init__(
        self,
        *,
        model_name=None,
        model_dir=None,
        **common_args,
    ):
        super().__init__()
        self._model_name = (
            model_name if model_name is not None else self.default_model_name
        )
        self._model_dir = model_dir
        self._common_args = parse_common_args(
            common_args, default_enable_hpi=_DEFAULT_ENABLE_HPI
        )
        self.paddlex_predictor = self._create_paddlex_predictor()

    @property
    @abc.abstractmethod
    def default_model_name(self):
        raise NotImplementedError

    def predict_iter(self, *args, **kwargs):
        return self.paddlex_predictor.predict(*args, **kwargs)

    def predict(self, *args, **kwargs):
        result = list(self.predict_iter(*args, **kwargs))
        return result

    def close(self):
        self.paddlex_predictor.close()

    @classmethod
    @abc.abstractmethod
    def get_cli_subcommand_executor(cls):
        raise NotImplementedError

    def _get_extra_paddlex_predictor_init_args(self):
        return {}

    def _create_paddlex_predictor(self):
        kwargs = prepare_common_init_args(self._model_name, self._common_args)
        kwargs = {**self._get_extra_paddlex_predictor_init_args(), **kwargs}
        # Should we check model names?
        try:
            return create_predictor(
                model_name=self._model_name, model_dir=self._model_dir, **kwargs
            )
        except DependencyError as e:
            raise RuntimeError(
                "A dependency error occurred during predictor creation. Please refer to the installation documentation to ensure all required dependencies are installed."
            ) from e


class PredictorCLISubcommandExecutor(CLISubcommandExecutor):
    """Base class for single-model CLI subcommand executors.

    Registers a model predictor as a subcommand of the ``paddleocr`` CLI and
    handles argument parsing and execution.
    """

    @property
    @abc.abstractmethod
    def subparser_name(self):
        raise NotImplementedError

    def add_subparser(self, subparsers):
        subparser = subparsers.add_parser(name=self.subparser_name)
        self._update_subparser(subparser)
        subparser.add_argument("--model_name", type=str, help="Name of the model.")
        subparser.add_argument(
            "--model_dir", type=str, help="Directory where the model is stored."
        )
        add_common_cli_opts(
            subparser,
            default_enable_hpi=_DEFAULT_ENABLE_HPI,
            allow_multiple_devices=False,
        )
        return subparser

    @abc.abstractmethod
    def _update_subparser(self, subparser):
        raise NotImplementedError
