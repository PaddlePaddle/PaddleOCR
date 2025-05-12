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

import time

from .logging import logger


def result_gen_to_list(gen, *, log_item=False):
    result = []
    t1 = time.time()
    for i, res in enumerate(gen):
        logger.info(f"Processed item {i} in {(time.time()-t1) * 1000} ms")
        t1 = time.time()
        if log_item:
            logger.info("%s", res)
        result.append(res)
    return result
