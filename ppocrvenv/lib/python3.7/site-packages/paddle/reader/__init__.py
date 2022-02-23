# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
r"""
At training and testing time, PaddlePaddle programs need to read data. To ease
the users' work to write data reading code, we define that

- A *reader* is a function that reads data (from file, network, random number
  generator, etc) and yields data items.
- A *reader creator* is a function that returns a reader function.
- A *reader decorator* is a function, which accepts one or more readers, and
  returns a reader.
- A *batch reader* is a function that reads data (from *reader*, file, network,
  random number generator, etc) and yields a batch of data items.

#####################
Data Reader Interface
#####################

Indeed, *data reader* doesn't have to be a function that reads and yields data
items. It can be any function with no parameter that creates a iterable
(anything can be used in :code:`for x in iterable`)\:

..  code-block:: python

    iterable = data_reader()

Element produced from the iterable should be a **single** entry of data,
**not** a mini batch. That entry of data could be a single item, or a tuple of
items.
Item should be of supported type (e.g., numpy array or list/tuple of float 
or int).

An example implementation for single item data reader creator:

..  code-block:: python

    def reader_creator_random_image(width, height):
        def reader():
            while True:
                yield numpy.random.uniform(-1, 1, size=width*height)
    return reader

An example implementation for multiple item data reader creator:

..  code-block:: python

    def reader_creator_random_image_and_label(width, height, label):
        def reader():
            while True:
                yield numpy.random.uniform(-1, 1, size=width*height), label
    return reader

"""

from paddle.reader.decorator import map_readers  # noqa: F401
from paddle.reader.decorator import shuffle  # noqa: F401
from paddle.reader.decorator import xmap_readers  # noqa: F401
from paddle.reader.decorator import firstn  # noqa: F401
from paddle.reader.decorator import buffered  # noqa: F401
from paddle.reader.decorator import compose  # noqa: F401
from paddle.reader.decorator import cache  # noqa: F401
from paddle.reader.decorator import ComposeNotAligned  # noqa: F401
from paddle.reader.decorator import chain  # noqa: F401
from paddle.reader.decorator import multiprocess_reader  # noqa: F401

__all__ = []
