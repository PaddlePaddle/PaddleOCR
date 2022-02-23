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
"""Implements an exponential algorithm for the tree edit distance. It
computes all possible TED mappings between two trees and calculated their
minimal cost."""
from __future__ import (absolute_import, division)

from copy import copy
from .config import Config
from .node_indexer import NodeIndexer


class AllPossibleMappingsTED(object):
    """Implements an exponential algorithm for the tree edit distance. It
    computes all possible TED mappings between two trees and calculated their
    minimal cost."""

    def __init__(self, tree1, tree2, config=None):
        self.config = config or Config()
        """Config object that specifies how to calculate the edit distance"""

        self.it1 = NodeIndexer(tree1, 0, self.config)
        """Stores the indexes of the first input tree"""

        self.it2 = NodeIndexer(tree2, 1, self.config)
        """Stores the indexes of the second input tree"""


    def compute_edit_distance(self):
        """Computes the tree edit distance between two trees by trying all
        possible TED mappings. It uses the specified cost model."""
        mappings = [
            mapping for mapping in self.generate_all_one_to_one_mappins()
            if self.is_ted_mapping(mapping)
        ]
        return self.get_min_cost(mappings)


    def generate_all_one_to_one_mappins(self):
        """Generate all possible 1-1 mappings.

        These mappings do not conform to TED conditions (sibling-order and
        ancestor-descendant).

        A mapping is a list of pairs (arrays) of preorder IDs (identifying
        nodes).

        return set of all 1-1 mappings
        """
        mappings = [
            [(node1, None) for node1 in self.it1.pre_ltr_info] +
            [(None, node2) for node2 in self.it2.pre_ltr_info]
        ]
        # For each node in the source tree
        for node1 in self.it1.pre_ltr_info:
            # Duplicate all mappings and store in mappings_copy
            mappings_copy = [
                copy(x) for x in mappings
            ]
            # For each node in the destination tree
            for node2 in self.it2.pre_ltr_info:
                # For each mapping (produced for all n1 values smaller than
                # current n1)
                for mapping in mappings_copy:
                    # Produce new mappings with the pair (n1, n2) by adding this
                    # pair to all mappings where it is valid to add
                    element_add = True
                    # Verify if (n1, n2) can be added to mapping m.
                    # All elements in m are checked with (n1, n2) for possible
                    # violation
                    # One-to-one condition
                    for ele1, ele2 in mapping:
                        # n1 is not in any of previous mappings
                        if ele1 and ele2 and ele2 is node2:
                            element_add = False
                            break
                    # New mappings must be produces by duplicating a previous
                    # mapping and extending it by (n1, n2)
                    if element_add:
                        m_copy = copy(mapping)
                        m_copy.append((node1, node2))
                        m_copy.remove((node1, None))
                        m_copy.remove((None, node2))
                        mappings.append(m_copy)
        return mappings

    def is_ted_mapping(self, mapping):
        """Test if a 1-1 mapping is a TED mapping"""
        # pylint: disable=no-self-use, invalid-name
        # Validade each pait of pairs of mapped nodes in the mapping
        for node_a1, node_a2 in mapping:
            # Use only pairs of mapped nodes for validation.
            if node_a1 is None or node_a2 is None:
                continue
            for node_b1, node_b2 in mapping:
                # Use only pairs of mapped nodes for validation.
                if node_b1 is None or node_b2 is None:
                    continue
                # If any of the conditions below doesn't hold, discard m.
                # Validate ancestor-descendant condition.
                n1 = (
                    node_a1.pre_ltr < node_b1.pre_ltr and
                    node_a1.pre_rtl < node_b1.pre_rtl
                )
                n2 = (
                    node_a2.pre_ltr < node_b2.pre_ltr and
                    node_a2.pre_rtl < node_b2.pre_rtl
                )
                if (n1 and not n2) or (not n1 and n2):
                    # Discard the mapping.
                    # If this condition doesn't hold, the next condition
                    # doesn't have to be verified any more and any other
                    # pair doesn't have to be verified any more.
                    return False
                # Validade sibling-order condition
                n1 = (
                    node_a1.pre_ltr < node_b1.pre_ltr and
                    node_a1.pre_rtl > node_b1.pre_rtl
                )
                n2 = (
                    node_a2.pre_ltr < node_b2.pre_ltr and
                    node_a2.pre_rtl > node_b2.pre_rtl
                )
                if (n1 and not n2) or (not n1 and n2):
                    # Discard the mapping.
                    return False
        return True

    def get_min_cost(self, mappings):
        """Given list of all TED mappings, calculate the cost of the
        minimal-cost mapping."""
        insert, delete = self.config.insert, self.config.delete
        rename = self.config.rename

        # Initialize min_cost to the upper bound
        min_cost = float('inf')
        # verify cost of each mapping
        for mapping in mappings:
            m_cost = 0
            # Sum up edit costs for all elements in the mapping m.
            for node1, node2 in mapping:
                if node1 and node2:
                    m_cost += rename(node1.node, node2.node)
                elif node1:
                    m_cost += delete(node1.node)
                else:
                    m_cost += insert(node2.node)
                # Break as soon as the current min_cost is exceeded.
                # Only for early loop break.
                if m_cost > min_cost:
                    break
            # Store the minimal cost - compare m_cost and min_cost
            min_cost = min(min_cost, m_cost)
        return min_cost
