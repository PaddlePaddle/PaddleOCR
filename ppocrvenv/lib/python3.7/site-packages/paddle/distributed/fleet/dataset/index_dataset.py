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
from paddle.fluid import core

__all__ = []


class Index(object):
    def __init__(self, name):
        self._name = name


class TreeIndex(Index):
    def __init__(self, name, path):
        super(TreeIndex, self).__init__(name)
        self._wrapper = core.IndexWrapper()
        self._wrapper.insert_tree_index(name, path)
        self._tree = self._wrapper.get_tree_index(name)
        self._height = self._tree.height()
        self._branch = self._tree.branch()
        self._total_node_nums = self._tree.total_node_nums()
        self._emb_size = self._tree.emb_size()
        self._layerwise_sampler = None

    def height(self):
        return self._height

    def branch(self):
        return self._branch

    def total_node_nums(self):
        return self._total_node_nums

    def emb_size(self):
        return self._emb_size

    def get_all_leafs(self):
        return self._tree.get_all_leafs()

    def get_nodes(self, codes):
        return self._tree.get_nodes(codes)

    def get_layer_codes(self, level):
        return self._tree.get_layer_codes(level)

    def get_travel_codes(self, id, start_level=0):
        return self._tree.get_travel_codes(id, start_level)

    def get_ancestor_codes(self, ids, level):
        return self._tree.get_ancestor_codes(ids, level)

    def get_children_codes(self, ancestor, level):
        return self._tree.get_children_codes(ancestor, level)

    def get_travel_path(self, child, ancestor):
        res = []
        while (child > ancestor):
            res.append(child)
            child = int((child - 1) / self._branch)
        return res

    def get_pi_relation(self, ids, level):
        codes = self.get_ancestor_codes(ids, level)
        return dict(zip(ids, codes))

    def init_layerwise_sampler(self,
                               layer_sample_counts,
                               start_sample_layer=1,
                               seed=0):
        assert self._layerwise_sampler is None
        self._layerwise_sampler = core.IndexSampler("by_layerwise", self._name)
        self._layerwise_sampler.init_layerwise_conf(layer_sample_counts,
                                                    start_sample_layer, seed)

    def layerwise_sample(self, user_input, index_input, with_hierarchy=False):
        if self._layerwise_sampler is None:
            raise ValueError("please init layerwise_sampler first.")
        return self._layerwise_sampler.sample(user_input, index_input,
                                              with_hierarchy)
