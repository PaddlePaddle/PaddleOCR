#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""
All util layers.
"""

from __future__ import print_function

from .layer_function_generator import autodoc
from ..framework import unique_name
from ..layer_helper import LayerHelper
from paddle.utils import deprecated

__all__ = []


@deprecated(since='0.15.0', update_to="paddle.fluid.ParallelExecutor")
@autodoc()
def get_places(device_count=None, device_type=None):
    helper = LayerHelper('get_places', **locals())
    out_places = helper.create_variable(
        name=unique_name.generate_with_ignorable_key(helper.name + ".out"))
    attrs = dict()
    if device_count is not None:
        attrs['device_count'] = int(device_count)
    if device_type is not None:
        attrs['device_type'] = str(device_type)

    helper.append_op(
        type='get_places', outputs={"Out": [out_places]}, attrs=attrs)

    return out_places
