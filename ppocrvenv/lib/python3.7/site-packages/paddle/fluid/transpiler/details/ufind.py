# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function


class UnionFind(object):
    """ Union-find data structure.

    Union-find is a data structure that keeps track of a set of elements partitioned
    into a number of disjoint (non-overlapping) subsets.

    Reference:
    https://en.wikipedia.org/wiki/Disjoint-set_data_structure

    Args:
      elements(list): The initialize element list.
    """

    def __init__(self, elementes=None):
        self._parents = []  # index -> parent index
        self._index = {}  # element -> index
        self._curr_idx = 0
        if not elementes:
            elementes = []
        for ele in elementes:
            self._parents.append(self._curr_idx)
            self._index.update({ele: self._curr_idx})
            self._curr_idx += 1

    def find(self, x):
        # Find the root index of given element x,
        # execute the path compress while findind the root index
        if not x in self._index:
            return -1
        idx = self._index[x]
        while idx != self._parents[idx]:
            t = self._parents[idx]
            self._parents[idx] = self._parents[t]
            idx = t
        return idx

    def union(self, x, y):
        # Union two given element
        x_root = self.find(x)
        y_root = self.find(y)

        if x_root == y_root:
            return
        self._parents[x_root] = y_root

    def is_connected(self, x, y):
        # If two given elements have the same root index,
        # then they are connected.
        return self.find(x) == self.find(y)
