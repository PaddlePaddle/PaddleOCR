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

from .text_detection import (
    TextDetection,
    TextDetectionSubcommandExecutor,
)


class SealTextDetection(TextDetection):
    @property
    def default_model_name(self):
        return "PP-OCRv4_mobile_seal_det"

    @classmethod
    def get_cli_subcommand_executor(cls):
        return SealTextDetectionSubcommandExecutor()


class SealTextDetectionSubcommandExecutor(TextDetectionSubcommandExecutor):
    @property
    def subparser_name(self):
        return "seal_text_detection"

    @property
    def wrapper_cls(self):
        return SealTextDetection
