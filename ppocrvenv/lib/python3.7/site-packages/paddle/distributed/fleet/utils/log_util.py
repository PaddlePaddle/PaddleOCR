# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import sys

__all__ = []


class LoggerFactory:
    @staticmethod
    def build_logger(name=None, level=logging.INFO):
        assert name is not None, "name for logger should not be None"

        formatter = logging.Formatter(
            "%(asctime)s-%(levelname)s: "
            "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s")

        _logger = logging.getLogger(name)
        _logger.setLevel(level)
        _logger.propagate = False
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(formatter)
        handler.setLevel(level)
        _logger.addHandler(handler)
        return _logger


logger = LoggerFactory.build_logger(name="HybridParallel", level=logging.INFO)


def layer_to_str(base, *args, **kwargs):
    name = base + "("
    if args:
        name += ", ".join(str(arg) for arg in args)
        if kwargs:
            name += ", "
    if kwargs:
        name += ", ".join("{}={}".format(key, str(value))
                          for key, value in kwargs.items())
    name += ")"
    return name
