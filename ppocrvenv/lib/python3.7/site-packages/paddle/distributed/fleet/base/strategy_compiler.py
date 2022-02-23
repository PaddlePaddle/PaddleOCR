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

__all__ = []


def create_graph(optimizer_list):
    nsize = len(optimizer_list)

    edge = [[0] * nsize for _ in range(nsize)]  # adjacency matrix
    indegree = [0] * nsize
    for i, opt in enumerate(optimizer_list):
        for j, opt_inner in enumerate(optimizer_list):
            if opt._can_update(opt_inner):
                edge[i][j] = 1  # weight
                indegree[j] += 1

    return edge, indegree


def topo_sort(edge, indegree):
    nsize = len(indegree)

    topo = [-1] * nsize
    for i in range(nsize):
        j = 0
        while j < nsize and indegree[j] != 0:
            j += 1
        assert j < nsize, 'The combination of meta optimizers contains ring'

        topo[i] = j
        indegree[j] = -1
        for k in range(nsize):
            if edge[j][k] != 0:
                indegree[k] -= 1

    return topo


def floyd(edge):
    nsize = len(edge)
    max_len = -1
    max_edge = [-1, -1]

    max_path = [[[] for _ in range(nsize)] for _ in range(nsize)]
    for i in range(nsize):
        for j in range(nsize):
            if edge[i][j] > 0:
                max_path[i][j] = [j]

                if edge[i][j] > max_len:
                    max_len = edge[i][j]
                    max_edge = [i, j]

    # use floyd algorithm to find max_path
    for k in range(nsize):
        for i in range(nsize):
            for j in range(nsize):
                # if a-->b-->c, but a-/->c, can only apply a-->b or b-->c,
                # however if a-->b-->c, and a-->c, can apply a->b->c
                if edge[i][j] == 0:
                    continue

                if edge[i][k] == 0 or edge[k][j] == 0:
                    continue

                if edge[i][j] < edge[i][k] + edge[k][j]:
                    edge[i][j] = edge[i][k] + edge[k][j]
                    max_path[i][j] = max_path[i][k] + max_path[k][j]

                    max_len = edge[i][j]
                    max_edge = [i, j]

    if max_len == -1:
        return [0]

    return [max_edge[0]] + max_path[max_edge[0]][max_edge[1]]


def maximum_path_len_algo(optimizer_list):
    if len(optimizer_list) == 0:
        return None

    edge, indegree = create_graph(optimizer_list)
    topo_sort(edge, indegree)
    max_path = floyd(edge)

    candidate = []
    for idx in max_path:
        candidate.append(optimizer_list[idx])

    for idx, opt in enumerate(candidate[:-1]):
        opt._update_inner_optimizer(candidate[idx + 1])

    return candidate


class StrategyCompilerBase(object):
    def __init__(self):
        pass


class StrategyCompiler(StrategyCompilerBase):
    """
    StrategyCompiler is responsible for meta optimizers combination
    Generally, a user can define serveral distributed strategies that
    can generate serveral meta optimizer. The combination of these 
    meta optimizers should have the right order to apply the optimizers'
    minimize function.
    This class is responsible for the executable distributed optimizer
    generation.
    """

    def __init__(self):
        super(StrategyCompiler, self).__init__()
        self._meta_optimizers = []
        self._graph_optimizers = []
        self._valid_optimizer_list = None
        self._user_defined_strategy = None
        self._meta_optimizer_candidates = []
        self._graph_optimizer_candidates = []

    def _get_applied_meta_optimizer(self):
        return self._meta_optimizers

    def _get_applied_meta_list(self):
        return [type(opt).__name__ for opt in self._meta_optimizers]

    def _get_applied_graph_list(self):
        return [type(opt).__name__ for opt in self._graph_optimizers]

    def _get_valid_strategy(self, dist_strategy, can_not_apply_optimizer_list):
        import copy
        valid_strategy = copy.deepcopy(dist_strategy)
        invalid_optimizers = []
        for candidate in self._meta_optimizer_candidates:
            is_valid = False
            for valid in self._meta_optimizers:
                if candidate.__class__.__name__ == valid.__class__.__name__:
                    is_valid = True
                    break
            if not is_valid:
                invalid_optimizers.append(candidate)
        for opt in invalid_optimizers:
            opt._disable_strategy(valid_strategy)
        for opt in can_not_apply_optimizer_list:
            opt._disable_strategy(valid_strategy)
        return valid_strategy

    """
    Meta Optimizer Type A: rewrite forward, backward. e.g. recompute, async, sync, pipeline.
                           results will be splitted in async, sync, pipeline
    Meta Optimizer Type B: rewrite forward, 
                           e.g. AMP and the corresponding backward is generated by rewritten forward
    Meta Opitmizer Type B: rewrite backward. e.g. gradient fusion
    Meta Optimizer Type D: rewrite optimize. e.g. lars, lamb, localsgd, gradient merge, dgc
    Meta Optimizer Type E: only transpile to Graph structure for runtime,
                           currently, grad fusion and kernel fusion, sync batch-norm included.
                           we will remove grad fusion and sync batch-norm
    """

    def generate_optimizer(self, loss, role_maker, optimizer,
                           user_defined_strategy, meta_optimizer_list,
                           graph_optimizer_list):
        self._user_defined_strategy = user_defined_strategy
        self._meta_optimizer_candidates = meta_optimizer_list
        self._graph_optimizer_candidates = graph_optimizer_list

        if len(meta_optimizer_list) == 0 and len(graph_optimizer_list) == 0:
            return optimizer, None
        else:
            # currently, we use heuristic algorithm to select
            # meta optimizers combinations
            meta_optimizers = maximum_path_len_algo(meta_optimizer_list)
            graph_optimizers = maximum_path_len_algo(graph_optimizer_list)
            # should design a distributed strategy update interface
            # when we have finally decided the combination of meta_optimizer
            # and graph_optimizer, the corresponding distributed strategy
            # should be updated.

            self._meta_optimizers = [] if meta_optimizers is None else meta_optimizers
            self._graph_optimizers = [] if graph_optimizers is None else graph_optimizers

            return_meta = None if meta_optimizers == None else meta_optimizers[
                0]
            return_graph = None if graph_optimizers == None else graph_optimizers[
                0]

            if meta_optimizers == None or graph_optimizers == None:
                return return_meta, return_graph

            # do heuristic filter here, if any meta optimizer in graph optimizers is in 
            # any meta optimizers' black list, set return_graph to None
            need_graph_opt = True
            for graph_opt in graph_optimizers:
                for program_opt in meta_optimizers:
                    if graph_opt.__class__.__name__ in program_opt.meta_optimizers_black_list:
                        need_graph_opt = False
            if not need_graph_opt:
                return_graph = None

            return return_meta, return_graph
