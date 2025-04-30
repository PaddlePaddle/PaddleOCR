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

import argparse
import warnings

from ._models import (
    DocImgOrientationClassification,
    DocVLM,
    FormulaRecognition,
    LayoutDetection,
    SealTextDetection,
    TableCellsDetection,
    TableClassification,
    TableStructureRecognition,
    TextDetection,
    TextImageUnwarping,
    TextLineOrientationClassification,
    TextRecognition,
)
from ._pipelines import (
    DocPreprocessor,
    DocUnderstanding,
    FormulaRecognitionPipeline,
    PaddleOCR,
    PPChatOCRv4Doc,
    PPStructureV3,
    SealRecognition,
    TableRecognitionPipelineV2,
)
from ._version import version
from .utils.deprecation import CLIDeprecationWarning


def _register_pipelines(subparsers):
    for cls in [
        DocPreprocessor,
        DocUnderstanding,
        FormulaRecognitionPipeline,
        PaddleOCR,
        PPChatOCRv4Doc,
        PPStructureV3,
        SealRecognition,
        TableRecognitionPipelineV2,
    ]:
        subcommand_executor = cls.get_cli_subcommand_executor()
        subparser = subcommand_executor.add_subparser(subparsers)
        subparser.set_defaults(executor=subcommand_executor.execute_with_args)


def _register_models(subparsers):
    for cls in [
        DocImgOrientationClassification,
        DocVLM,
        FormulaRecognition,
        LayoutDetection,
        SealTextDetection,
        TableCellsDetection,
        TableClassification,
        TableStructureRecognition,
        TextDetection,
        TextImageUnwarping,
        TextLineOrientationClassification,
        TextRecognition,
    ]:
        subcommand_executor = cls.get_cli_subcommand_executor()
        subparser = subcommand_executor.add_subparser(subparsers)
        subparser.set_defaults(executor=subcommand_executor.execute_with_args)


def _parse_args():
    parser = argparse.ArgumentParser(prog="paddleocr")
    parser.add_argument("--version", action="version", version=f"%(prog)s {version}")
    subparsers = parser.add_subparsers(dest="subcommand")
    _register_pipelines(subparsers)
    _register_models(subparsers)
    return parser.parse_args()


def _execute(args):
    args.executor(args)


def main():
    warnings.filterwarnings("default", category=CLIDeprecationWarning)
    args = _parse_args()
    _execute(args)
