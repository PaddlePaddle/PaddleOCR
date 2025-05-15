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
import sys
import warnings

from typing_extensions import deprecated as deprecated


class CLIDeprecationWarning(DeprecationWarning):
    pass


class DeprecatedOptionAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        assert option_string
        warnings.warn(
            f"The option `{option_string}` has been deprecated and will be removed in the future. Please refer to the documentation for more details.",
            CLIDeprecationWarning,
        )
        setattr(namespace, self.dest, values)


def warn_deprecated_param(name, new_name=None):
    msg = (
        f"The parameter `{name}` has been deprecated and will be removed in the future."
    )
    if new_name is not None:
        msg += f" Please use `{new_name}` instead."
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
