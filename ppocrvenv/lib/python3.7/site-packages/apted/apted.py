#
# The MIT License
#
# Copyright 2017 Joao Felipe Pimentel

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""APTED implementation"""
from __future__ import (absolute_import, division)

import math
from .node_indexer import NodeIndexer
from .config import Config
from .single_path_functions import spf1, SinglePathFunction, LEFT, RIGHT, INNER
# pylint: disable=invalid-name
# pylint: disable=fixme

class Cost(object):
    """Represents a Cost for opt strategy calculation"""
    # pylint: disable=too-few-public-methods
    def __init__(self, vcls):
        self.l = 0.0
        self.r = 0.0
        self.i = 0.0
        self.path = vcls()

    def set_lri(self, value):
        """Sets l, r, and i values"""
        self.l = self.r = self.i = value


class APTED(object):
    """Implements APTED algorithm [1,2].

    - Optimal strategy with all paths.
    - Single-node single path function supports currently only unit cost.
    - Two-node single path function not included.
    - \\Delta^L and \\Delta^R based on Zhang and Shasha's algorithm for executing
      left and right paths (as in [3]). If only left and right paths are used
      in the strategy, the memory usage is reduced by one quadratic array.
    - For any other path \\Delta^A from [1] is used.

    References:
    - [1] M. Pawlik and N. Augsten. Efficient Computation of the Tree Edit
      Distance. ACM Transactions on Database Systems (TODS) 40(1). 2015.
    - [2] M. Pawlik and N. Augsten. Tree edit distance: Robust and memory-
      efficient. Information Systems 56. 2016.
    - [3] M. Pawlik and N. Augsten. RTED: A Robust Algorithm for the Tree Edit
      Distance. PVLDB 5(4). 2011.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, tree1, tree2, config=None, spf=SinglePathFunction):
        # Config object that specifies how to calculate the edit distance
        self.config = config or Config()

        # Single path function class
        self.spf = spf

        # The distance matrix [1, Sections 3.4,8.2,8.3]
        # Used to store intermediate distances between pairs of subtrees
        self.delta = []

        # Stores the number of subproblems encountered while computing the
        # distance. See [1, Section 10].
        self.counter = 0

        # Stores the indexes of the first input tree
        self.it1 = NodeIndexer(tree1, 0, self.config)

        # Stores the indexes of the second input tree
        self.it2 = NodeIndexer(tree2, 1, self.config)

        # Stores the result
        self.result = None
        self.mapping = None


    def compute_edit_distance(self):
        """Compute tree edit distance between source and destination trees
        using APTED algorithm [1,2]."""
        # Initialize delta array
        if self.result is None:
            if self.it1.lchl < self.it1.rchl:
                self.delta = self.compute_opt_strategy_post_l()
            else:
                self.delta = self.compute_opt_strategy_post_r()

            self.ted_init()
            self.result = self.gted()
        return self.result

    def compute_edit_distance_spf_test(self, spf_type):
        """This method is only for testing purspose. It computes TED with a
        fixed path type in the strategy to trigger execution of a specific
        single-path function"""
        # Initialise delta array.
        if self.result is None:
            index_1, vcls = self.it1.pre_ltr_info, self.config.valuecls
            size1, size2 = self.it1.tree_size, self.it2.tree_size
            self.delta = [
                [vcls() for _ in range(size2)]
                for _ in range(size1)
            ]
            # Fix a path type to trigger specific spf.
            for i in range(size1):
                for j in range(size2):
                    # Fix path type
                    if spf_type == LEFT:
                        self.delta[i][j] = vcls(index_1[i].lld.pre_ltr + 1)
                    elif spf_type == RIGHT:
                        self.delta[i][j] = vcls(index_1[i].rld.pre_ltr + 1)
            self.ted_init()
            self.result = self.gted()
        return self.result

    def ted_init(self):
        """After the optimal strategy is computed, initializes distances of
        deleting and inserting subtrees without their root nodes."""
        it1, it2, vcls = self.it1, self.it2, self.config.valuecls

        delta = self.delta
        # Reset the subproblems counter.
        self.counter = 0

        # Computer subtree distances without the root nodes when one of
        # the subtrees is a single node

        insert = self.config.insert
        delete = self.config.delete

        # Loop over the nodes in reversed(?) left-to-right preorder.
        for x, node1 in enumerate(it1.pre_ltr_info):
            size1 = node1.size
            for y, node2 in enumerate(it2.pre_ltr_info):
                # Set values in delta based on the sums of deletion and
                # insertion costs. Substract the costs for root nodes.
                # In this method we don't have to verify the order of the
                # input trees because it is equal to the original.
                size2 = node2.size
                if size1 == 1 and size2 == 1:
                    delta[x][y] = vcls(0.0)
                elif size1 == 1:
                    delta[x][y] = node2.sum_cost - insert(node2.node)
                elif size2 == 1:
                    delta[x][y] = node1.sum_cost - delete(node1.node)

    def compute_opt_strategy_post_l(self):
        """Compute the optimal strategy using left-to-right postorder traversal
        of the nodes. See [2, Algorithm 1].
        """
        def update_parent(min_cost, node, cost, parent_cost):
            """Update parent cost according to node cost and min_cost"""
            update_path = None
            parent_cost.r += min_cost
            tmp_cost = -min_cost + cost.i
            if tmp_cost < parent_cost.i:
                parent_cost.i = tmp_cost
                update_path = parent_cost.path = cost.path
            if node.type_r:
                parent_cost.i += parent_cost.r
                parent_cost.r += cost.r - min_cost
            if node.type_l:
                parent_cost.l += cost.l
            else:
                parent_cost.l += min_cost

            return update_path
        order1 = self.it1.post_ltr_info
        order2 = self.it2.post_ltr_info
        cost_index = lambda node: node.post_ltr
        return self.compute_opt_strategy_post(
            order1, order2, cost_index, update_parent
        )

    def compute_opt_strategy_post_r(self):
        """Compute the optimal strategy using right-to-left postorder traversal
        of the nodes. See [2, Algorithm 1].
        """
        def update_parent(min_cost, node, cost, parent_cost):
            """Update parent cost according to node cost and min_cost"""
            update_path = None
            parent_cost.l += min_cost
            tmp_cost = -min_cost + cost.i
            if tmp_cost < parent_cost.i:
                parent_cost.i = tmp_cost
                update_path = parent_cost.path = cost.path
            if node.type_l:
                parent_cost.i += parent_cost.l
                parent_cost.l += cost.l - min_cost
            if node.type_r:
                parent_cost.r += cost.r
            else:
                parent_cost.r += min_cost

            return update_path
        order1 = self.it1.post_rtl_info
        order2 = self.it2.post_rtl_info
        cost_index = lambda node: node.post_rtl
        return self.compute_opt_strategy_post(
            order1, order2, cost_index, update_parent
        )

    def compute_opt_strategy_post(self, order1, order2, costi, update_parent):
        """Compute the optimal strategy using generic post-ortder traversal
        See [2, Algorithm 1].
        """
        # pylint: disable=too-many-statements
        # pylint: disable=too-many-branches
        # pylint: disable=no-self-use
        # pylint: disable=too-many-locals
        # pylint: disable=cell-var-from-loop
        it1, it2, vcls = self.it1, self.it2, self.config.valuecls
        size1, size2 = it1.tree_size, it2.tree_size
        strategy = [
            [vcls() for _ in range(size2)]
            for _ in range(size1)
        ]
        cost1 = [None] * size1

        leaf_row = [Cost(vcls) for _ in range(size2)]

        path_id_offset = size1
        min_cost = float('inf')
        strategy_path = -1

        rows_to_reuse = []
        pre_rtl_1 = it1.pre_rtl_info
        pre_rtl_2 = it2.pre_rtl_info

        for node1 in order1:
            v_cost = costi(node1)
            v_in_pre_ltr = node1.pre_ltr

            strategy_v = strategy[v_in_pre_ltr]

            parent1 = node1.parent
            size_v = node1.size
            kr_sum_v = node1.kr_sum
            rev_kr_sum_v = node1.rev_kr_sum
            desc_sum_v = node1.desc_sum

            # this is the left path's ID which is the leftmost leaf node:
            # l-r_preorder(r-l_preorder(v) + |Fv| - 1)
            left_path_v = vcls(-(pre_rtl_1[node1.pre_rtl + size_v - 1].pre_ltr + 1))
            # this is the right path's ID which is the rightmost leaf node:
            # l-r_preorder(v) + |Fv| - 1
            right_path_v = vcls(v_in_pre_ltr + size_v)


            if not node1.children:
                cost_pointer_v = cost1[v_cost] = leaf_row
                for node2 in order2:
                    w_cost, w_pre = costi(node2), node2.pre_ltr
                    strategy_v[w_pre] = cost_pointer_v[w_cost].path = vcls(v_in_pre_ltr)
            else:
                cost_pointer_v = cost1[v_cost]

            if parent1:
                parent_v = costi(parent1)
                if cost1[parent_v] is None:
                    if rows_to_reuse:
                        cost1[parent_v] = rows_to_reuse.pop()
                    else:
                        cost1[parent_v] = [Cost(vcls) for _ in range(size2)]

                cost_pointer_parent_v = cost1[parent_v]

            cost2 = [Cost(vcls) for _ in range(size2)]

            for node2 in order2:
                w_cost = costi(node2)
                w_in_pre_ltr = node2.pre_ltr

                cost_pointer_w = cost2[w_cost]
                cost_pointer_vw = cost_pointer_v[w_cost]

                parent2 = node2.parent

                size_w = node2.size
                if not node2.children:
                    cost_pointer_w.set_lri(0)
                    cost_pointer_w.path = vcls(w_in_pre_ltr)

                if size_v <= 1 or size_w <= 1:
                    min_cost = max(size_v, size_w)
                    strategy_path = -1
                else:
                    min_cost, _, strategy_fn = min(
                        (
                            size_v * node2.kr_sum + cost_pointer_vw.l, 1,
                            lambda: left_path_v
                        ),
                        (
                            size_v * node2.rev_kr_sum + cost_pointer_vw.r, 2,
                            lambda: right_path_v
                        ),
                        (
                            size_v * node2.desc_sum + cost_pointer_vw.i, 3,
                            lambda: cost_pointer_vw.path + vcls(1)
                        ),
                        (
                            size_w * kr_sum_v + cost_pointer_w.l, 4,
                            lambda: vcls(
                                - pre_rtl_2[node2.pre_rtl + size_w - 1].pre_ltr
                                - path_id_offset - 1
                            )
                        ),
                        (
                            size_w * rev_kr_sum_v + cost_pointer_w.r, 5,
                            lambda: vcls(w_in_pre_ltr + size_w + path_id_offset)
                        ),
                        (
                            size_w * desc_sum_v + cost_pointer_w.i, 6,
                            lambda: cost_pointer_w.path + vcls(path_id_offset + 1)
                        )
                    )
                    strategy_path = strategy_fn()

                if parent1:
                    new_path = update_parent(
                        min_cost, node1, cost_pointer_vw,
                        cost_pointer_parent_v[w_cost]
                    )
                    if new_path is not None:
                        strategy_v[w_in_pre_ltr] = new_path

                if parent2:
                    update_parent(
                        min_cost, node2, cost_pointer_w, cost2[costi(parent2)]
                    )

                cost_pointer_vw.path = strategy_path
                strategy_v[w_in_pre_ltr] = cost_pointer_vw.path

            if node1.children:
                for cost in cost_pointer_v:
                    cost.set_lri(0)
                rows_to_reuse.append(cost_pointer_v)

        return strategy

    def gted(self, data=None):
        """Implements GTED algorithm [1, Section 3.4].

        Return the tree edit distance between the source and destination trees.
        """
        it1, it2 = self.it1, self.it2
        data = data or [it1.pre_ltr_info[0], it2.pre_ltr_info[0]]
        tree1, tree2 = data

        if tree1.size == 1 or tree2.size == 1:  # Use spf1
            result = spf1(it1, it2, self.config, tree1, tree2)
            self.delta[tree1.pre_ltr][tree2.pre_ltr] = result
            return result

        path_id = int(self.delta[tree1.pre_ltr][tree2.pre_ltr])
        node_id = abs(path_id) - 1

        if node_id < it1.tree_size:  # Apply on subtree 1
            return self.sub_gted(data, it1, it2, tree1, path_id, node_id, False)

        # Apply on subtree 2
        node_id -= it1.tree_size
        return self.sub_gted(data, it2, it1, tree2, path_id, node_id, True)

    def sub_gted(self, data, it_f, it_s, tree_f, path_id, node_id, reverse):
        """Apply gted to subtree"""
        # pylint: disable=too-many-arguments
        size = self.it1.tree_size

        strategy = self.get_strategy_path_type(path_id, size, tree_f)
        current = it_f.pre_ltr_info[node_id]

        for parent, last in it_f.traverse_up(current, tree_f):
            for child in parent.children:
                if child is not last:
                    data[it_f.num] = child
                    self.gted(data)


        data[it_f.num] = tree_f
        # Pass to spfs a boolean that says says if the order of input
        # subtrees has been swapped compared to the order of the initial
        # input trees. Used for accessing delta array and deciding on the
        # edit operation. See [1, Section 3.4].

        return self.spf(
            it_f, it_s,
            data[it_f.num], data[it_s.num],
            self, node_id, strategy, reverse
        )()

    def get_strategy_path_type(self, strategy_path_id, size1, current):
        """Decodes the path from the optimal strategy to its type.

        Params:
          strategy_path_id: raw path id from strategy array.
          size1: offset used to distinguish between paths in the source and
            destination trees.
          it: node indexer
          current: current subtree processed in tree decomposition phase

        Return type of the strategy path: LEFT, RIGHT, INNER"""
        # pylint: disable=no-self-use
        if math.copysign(1, strategy_path_id) == -1:
            return LEFT
        path_id = abs(strategy_path_id) - 1
        if path_id >= size1:
            path_id = path_id - size1
        if path_id == (current.pre_ltr + current.size - 1):
            return RIGHT
        return INNER

    def compute_edit_mapping(self):
        """Compute the edit mapping between two trees.

        Returns list of pairs of nodes that are mapped as pairs
        Nodes that are delete or inserted are mapped to None
        """
        return self.config.compute_edit_mapping(self)

    def mapping_cost(self, mapping):
        """Calculates the cost of an edit mapping"""
        return self.config.mapping_cost(mapping)
