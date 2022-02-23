#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""This is definition of generator class, which is for managing the state of the algorithm that produces pseudo random numbers."""

from . import core
from .framework import _get_paddle_place

__all__ = ['Generator']


class Generator(core.Generator):
    """Generator class"""

    def __init__(self, place=None):
        """
        Create a generator object which manages the random number generation. ( Experimental Feature )

        Parameters:
            place(CPUPlace|CUDAPinnedPlace|CUDAPlace|str,optional): The place to allocate Tensor. Can be  
                CPUPlace, CUDAPinnedPlace, CUDAPlace. Default: None, means global place. If ``place`` is
                string, it can be ``cpu`` and ``gpu:x``, where ``x`` is the index of the GPUs.

        Returns:
            Generator: A generator object.

        """
        self.place = _get_paddle_place(place)
        if not place:
            place = core.CPUPlace()
        if isinstance(place, core.CPUPlace):
            super(Generator, self).__init__()
        else:
            raise ValueError(
                "Generator class with %s does is not supported yet, currently only support generator with CPUPlace "
                % place)
