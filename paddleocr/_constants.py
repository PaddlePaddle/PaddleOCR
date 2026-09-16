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

import os
import sys

DEFAULT_DEVICE = None
DEFAULT_USE_TENSORRT = False
DEFAULT_PRECISION = "fp32"
# oneDNN (mkldnn) is broken in the PaddlePaddle 3.3+ PIR executor on Windows
# CPU: inference crashes with `ConvertPirAttribute2RuntimeAttribute` /
# `[pir::ArrayAttribute<pir::DoubleAttribute>]` (see
# https://github.com/PaddlePaddle/Paddle/issues/77340 and
# https://github.com/PaddlePaddle/PaddleOCR/issues/17869). Disable it by
# default on win32 so out-of-the-box CPU inference works; users on other
# platforms (or with a fixed Paddle) can still opt in with
# `PaddleOCR(enable_mkldnn=True)`.
DEFAULT_ENABLE_MKLDNN = False if sys.platform == "win32" else True
DEFAULT_MKLDNN_CACHE_CAPACITY = 10
DEFAULT_CPU_THREADS = 10
SUPPORTED_PRECISION_LIST = ["fp32", "fp16"]
DEFAULT_USE_CINN = False
