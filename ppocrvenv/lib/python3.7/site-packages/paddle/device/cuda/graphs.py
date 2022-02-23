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

from paddle.fluid.core import is_compiled_with_cuda, is_compiled_with_rocm, CUDAPlace

if is_compiled_with_cuda() and not is_compiled_with_rocm():
    from paddle.fluid.core import CUDAGraph as CoreCUDAGraph

    class CUDAGraph:
        def __init__(self, place=None, mode="thread_local"):
            ALL_MODES = ["global", "thread_local", "relaxed"]
            self._graph = None
            if place is None:
                place = CUDAPlace(0)
            self._place = place
            assert mode in ALL_MODES
            self._mode = ALL_MODES.index(mode)

        def capture_begin(self):
            CoreCUDAGraph.begin_capture(self._place, self._mode)

        def capture_end(self):
            self._graph = CoreCUDAGraph.end_capture()

        def replay(self):
            self._graph.replay()

        def reset(self):
            self._graph.reset()
else:

    class CUDAGraph:
        def __init__(self, place=None, mode="thread_local"):
            raise NotImplementedError()

        def capture_begin(self):
            raise NotImplementedError()

        def capture_end(self):
            raise NotImplementedError()

        def replay(self):
            raise NotImplementedError()

        def reset(self):
            raise NotImplementedError()
