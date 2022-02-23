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
"""APTED configuration classes"""

from .helpers import ChainedValue


class Config(object):
    """Algorithm configuration"""

    valuecls = int

    def delete(self, node):
        """Calculates the cost of deleting a node"""
        return 1

    def insert(self, node):
        """Calculates the cost of inserting a node"""
        return 1

    def rename(self, node1, node2):
        """Calculates the cost of renaming the label of the source node
        to the label of the destination node"""
        return int(node1.name != node2.name)

    def children(self, node):
        """Returns children of node"""
        return getattr(node, 'children', [])

    def compute_edit_mapping(self, apted):
        """Compute the edit mapping between two trees. The trees are input trees
        to the distance computation and the distance must be computed before
        computing the edit mapping (distances of subtree pairs are required)

        Returns list of pairs of nodes that are mapped as pairs
        Nodes that are delete or inserted are mapped to 0
        """
        # pylint: disable=too-many-locals
        apted.compute_edit_distance()
        it1, it2 = apted.it1, apted.it2
        post_ltr_1, post_ltr_2 = it1.post_ltr_info, it2.post_ltr_info
        size1, size2 = it1.tree_size, it2.tree_size
        delete, insert = self.delete, self.insert

        forestdist = [
            [0 for _ in range(size2 + 1)]
            for _ in range(size1 + 1)
        ]
        # Empty edit mapping
        edit_mapping = []
        # Stack of tree pairs starting with the pair (ted1, ted2)
        tree_pairs = [(size1, size2)]

        while tree_pairs:
            # Get next pair to be processed
            id1, id2 = tree_pairs.pop()
            info1, info2 = post_ltr_1[id1 - 1], post_ltr_2[id2 - 1]

            # compute forest distance matrix
            self.forest_dist(apted, id1, id2, forestdist)

            # compute mapping for current forest distance matrix
            first1, first2 = info1.lld.post_ltr, info2.lld.post_ltr
            while id1 > first1 or id2 > first2:
                fdrc = forestdist[id1][id2]
                is_delete = (
                    id1 > first1 and
                    forestdist[id1 - 1][id2] + delete(info1.node) == fdrc
                )
                is_insert = (
                    id2 > first2 and
                    forestdist[id1][id2 - 1] + insert(info2.node) == fdrc
                )
                if is_delete:  # Node with post ltr id1 is deleted from ted1
                    edit_mapping.append((info1.node, None))
                    id1 -= 1
                elif is_insert:  # Node with post ltr id2 is inserted intro ted2
                    edit_mapping.append((None, info2.node))
                    id2 -= 1
                else:
                    # Node with post ltr id1 in ted1 is renamed to node id2 in
                    # ted 2
                    lld1, lld2 = info1.lld.post_ltr, info2.lld.post_ltr
                    if lld1 == first1 and lld2 == first2:
                        edit_mapping.append((info1.node, info2.node))
                        id1, id2 = id1 - 1, id2 - 1
                    else:
                        # append subtree pair
                        tree_pairs.append((id1, id2))
                        id1, id2 = lld1, lld2
                info1, info2 = post_ltr_1[id1 - 1], post_ltr_2[id2 - 1]

        return edit_mapping

    def forest_dist(self, apted, i, j, forestdist):
        """Recalculates distances between subforests of two subtrees.
        These values are used in mapping computation to track back the origin of
        minimum values. It is basen on Zhang and Shasha algorithm.

        The rename cost must be added in the last line. Otherwise the formula is
        incorrect. This is due to delta storing distances between subtrees
        without the root nodes.

        i and j are postorder ids of the nodes - starting with 1.
        Nodes that are delete or inserted are mapped to None
        """
        # pylint: disable=invalid-name, too-many-locals
        delete, insert = self.delete, self.insert
        rename, delta = self.rename, apted.delta

        it1, it2 = apted.it1, apted.it2
        post_ltr_1, post_ltr_2 = it1.post_ltr_info, it2.post_ltr_info

        lld1 = post_ltr_1[i - 1].lld.post_ltr
        lld2 = post_ltr_2[j - 1].lld.post_ltr

        forestdist[lld1][lld2] = 0

        for di in range(lld1 + 1, i + 1):
            info1 = post_ltr_1[di - 1]
            forestdist[di][lld2] = (
                forestdist[di - 1][lld2] + delete(info1.node)
            )
            for dj in range(lld2 + 1, j + 1):
                info2 = post_ltr_2[dj - 1]
                forestdist[lld1][dj] = (
                    forestdist[lld1][dj - 1] + insert(info2.node)
                )
                cost_ren = rename(info1.node, info2.node)
                # todo: the first two elements of the minimum can be computed
                # here, similarly to spfl and spfr
                if info1.lld.post_ltr == lld1 and info2.lld.post_ltr == lld2:
                    forestdist[di][dj] = min(
                        forestdist[di - 1][dj] + delete(info1.node),
                        forestdist[di][dj - 1] + insert(info2.node),
                        forestdist[di - 1][dj - 1] + cost_ren
                    )
                    # If substituted with delta, this will overwrite the value
                    # in delta.
                    # It looks that we don't have to write this value.
                    # Conceptually it is correct because we already have all
                    # the values in delta for subtrees without the root nodes,
                    # and we need these.
                    # treedist[di][dj] = forestdist[di][dj];

                else:
                    # di and dj are postorder ids of the nodes - starting with 1
                    # Substituted 'treedist[di][dj]' with
                    # 'delta[it1.postL_to_preL[di-1]][it2.postL_to_preL[dj-1]]'
                    forestdist[di][dj] = min(
                        forestdist[di - 1][dj] + delete(info1.node),
                        forestdist[di][dj - 1] + insert(info2.node),
                        forestdist[info1.lld.post_ltr][info2.lld.post_ltr] +
                        delta[info1.pre_ltr][info2.pre_ltr] + cost_ren
                    )

    def mapping_cost(self, mapping):
        """Calculates the cost of an edit mapping. It traverses the mapping and
        sums up the cost of each operation. The costs are taken from the cost
        model."""
        delete, insert = self.delete, self.insert
        rename = self.rename
        cost = 0
        for node1, node2 in mapping:
            if node1 is None: # insertion
                cost += insert(node2)
            elif node2 is None: # deletion
                cost += delete(node1)
            else:
                cost += rename(node1, node2)
        return cost


class PerEditOperationConfig(Config):
    """Algorithm configuration"""

    valuecls = int

    def __init__(self, del_cost, ins_cost, ren_cost):
        self.del_cost = del_cost
        self.ins_cost = ins_cost
        self.ren_cost = ren_cost

    def delete(self, node):
        """Calculates the cost of deleting a node"""
        return self.del_cost

    def insert(self, node):
        """Calculates the cost of inserting a node"""
        return self.ins_cost

    def rename(self, node1, node2):
        """Calculates the cost of renaming the label of the source node
        to the label of the destination node"""
        return (
            super(PerEditOperationConfig, self).rename(node1, node2) *
            self.ren_cost
        )


def meta_chained_config(config_cls):
    """Creates a config class that keeps track of the chain"""
    class ChainedConfig(config_cls):
        """Chained config class"""

        valuecls = ChainedValue

        def delete(self, node):
            """Calculates the cost of deleting a node"""
            return ChainedValue(
                super(ChainedConfig, self).delete(node),
                [(id(node), None)]
            )

        def insert(self, node):
            """Calculates the cost of inserting a node"""
            return ChainedValue(
                super(ChainedConfig, self).insert(node),
                [(None, id(node))]
            )

        def rename(self, node1, node2):
            """Calculates the cost of renaming the label of the source node
            to the label of the destination node"""
            return ChainedValue(
                super(ChainedConfig, self).rename(node1, node2),
                [(id(node1), id(node2))]
            )

        def compute_edit_mapping(self, apted):
            """Compute the edit mapping between two trees.

            Returns list of pairs of nodes that are mapped as pairs
            Nodes that are delete or inserted are mapped to None
            """
            # pylint: disable=no-self-use
            value = apted.compute_edit_distance()
            node_1, node_2 = apted.it1.node_info, apted.it2.node_info
            if apted.mapping is None:
                result = set()
                rem_list = set()
                for pair in value.chain:
                    if pair[0] == "R":
                        for rem_pair in pair[1]:
                            try:
                                result.remove(rem_pair)
                            except KeyError:
                                rem_list.add(rem_pair)
                    elif pair not in rem_list:
                        result.add(pair)
                    else:
                        rem_list.remove(pair)

                mapping = []
                for id1, id2 in result:
                    mapping.append((
                        getattr(node_1.get(id1), 'node', None),
                        getattr(node_2.get(id2), 'node', None)
                    ))

                apted.mapping = mapping

            return apted.mapping

        def mapping_cost(self, mapping):
            """Calculates the cost of an edit mapping. It traverses the mapping and
            sums up the cost of each operation. The costs are taken from the cost
            model."""
            cost = 0
            for row, col in mapping:
                if row is None: # insertion
                    cost += config_cls.insert(self, col)
                elif col is None: # deletion
                    cost += config_cls.delete(self, row)
                else:
                    cost += config_cls.rename(self, row, col)
            return cost

    return ChainedConfig
