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
"""NodeIndexer and NodeInfo implementations"""
from __future__ import (absolute_import, division)
from .config import Config


class NodeIndexer(object):
    """Indexes nodes of the input tree to the algorithm that is already
    parsed to tree structure. Stores various indices on nodes required for
    efficient computation of APTED [1,2]. Additionally, it stores single-value
    properties of the tree.

    For indexing we use four tree traversals that assign ids to the nodes:

    - left-to-right preorder [1],
    - right-to-left preorder [1],
    - left-to-right postorder [2],
    - right-to-left postorder [2].

    See the source code for more algorithm-related comments.

    References:
    [1] M. Pawlik and N. Augsten. Efficient Computation of the Tree Edit
        Distance. ACM Transactions on Database Systems (TODS) 40(1). 2015.
    [2] M. Pawlik and N. Augsten. Tree edit distance: Robust and memory-
        efficient. Information Systems 56. 2016.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, tree, num, config=None):
        # pylint: disable=too-many-statements

        self.num = num

        # Config object that specifies how to calculate the edit distance
        self.config = config or Config()

        # Temporary variable that stores preorder index
        self.preorder_tmp = 0

        # Map left-to-right preorder index to NodeInfo
        self.pre_ltr_info = []

        # Map left-to-right postorder index to NodeInfo
        self.post_ltr_info = []

        # Map right-to-left postorder index to NodeInfo
        self.post_rtl_info = []

        # Map right-to-left preorder index to NodeInfo
        self.pre_rtl_info = []

        # Map node object id to Info
        self.node_info = {}

        # Size of input tree
        self.tree_size = 0

        # Number of leftmost-child leaf nodes in the input tree
        # See [2, Section 5.3].
        self.lchl = 0

        # Number of rightmost-child leaf nodes in the input tree
        # See [2, Section 5.3].
        self.rchl = 0

        root, _ = self.index_nodes(tree, -1)
        root.parent = NodeInfo(None, -1, self.config)
        root.parent.num = num
        root.parent.fake_child = root


        self.tree_size = self.pre_ltr_info[0].size
        self.pre_rtl_info = [None] * self.tree_size
        self.post_rtl_info = [None] * self.tree_size

        self.post_traversal_indexing()

    def index_nodes(self, node, postorder):
        """Preprocesses each node of the tree and creates associated NodeInfo
        Computes the following attributes: node, pre_ltr, post_ltr, parent,
          children, size, desc_sum, kr_sum, rev_kr_sum
        Also computes the following indices: pre_ltr_info, post_ltr_info

        It is a recursive method that traverses the tree once

        node is the current node while traversing the input tree
        postorder is the postorder id of the current node

        return NodeInfo for node and accumulated sum of subtree sizes
        rooted at descendant nodes
        """
        desc_sizes = current_size = kr_sizes_sum = revkr_sizes_sum = 0

        node_info = NodeInfo(node, self.preorder_tmp, self.config)
        node_info.num = self.num
        self.node_info[id(node)] = node_info
        self.preorder_tmp += 1
        self.pre_ltr_info.append(node_info)

        children = self.config.children(node)
        last_index = len(children) - 1

        for child_index, child in enumerate(children):
            child_info, desc_sizes_tmp = self.index_nodes(child, postorder)
            child_info.parent = node_info
            node_info.children.append(child_info)

            postorder = child_info.post_ltr

            current_size += child_info.size
            desc_sizes += desc_sizes_tmp

            kr_sizes_sum += child_info.kr_sum
            revkr_sizes_sum += child_info.rev_kr_sum
            if child_index == 0:
                kr_sizes_sum -= child_info.size
                child_info.type_l = True
            if child_index == last_index:
                revkr_sizes_sum -= child_info.size
                child_info.type_r = True

        postorder += 1
        self.post_ltr_info.append(node_info)
        node_info.post_ltr = postorder

        node_info.size = nsize = current_size + 1
        desc_sizes_tmp = desc_sizes + nsize

        node_info.desc_sum = (nsize * (nsize + 3)) / 2 - desc_sizes_tmp
        node_info.kr_sum = kr_sizes_sum + nsize
        node_info.rev_kr_sum = revkr_sizes_sum + nsize

        return node_info, desc_sizes_tmp

    def post_traversal_indexing(self):
        """Indexes the nodes of the input tree.
        It computes the following indices, which could not be computed
        immediately while traversing the tree in index_nodes: pre_ltr_to_ln,
          post_ltr_to_lld, post_rtl_to_rlf, pre_rtl_to_ln

        Runs in linear time in the input tree size.
        Currently requires two loops over input tree nodes.
        Can be reduced to one loop (See the code)
        """

        current_leaf = NodeInfo.EMPTY
        delete = self.config.insert if self.num else self.config.delete

        for i, node in enumerate(self.pre_ltr_info):
            node.pre_rtl = self.tree_size - 1 - node.post_ltr
            node.post_rtl = self.tree_size - 1 - node.pre_ltr
            self.pre_rtl_info[node.pre_rtl] = node
            self.post_rtl_info[node.post_rtl] = node

            node.lnl = current_leaf
            if not node.children:
                current_leaf = node

                # Count lchl and rchl if node is leaf
                # Note that it must visit parent before child in order
                # to have pre_rtl computed
                parent = node.parent
                if parent:
                    if parent.pre_ltr + 1 == node.pre_ltr:
                        self.lchl += 1
                    elif parent.pre_rtl + 1 == node.pre_rtl:
                        self.rchl += 1

            # Sum up costs of deleting and inserting entire subtrees.
            # Reverse the node index.
            # Traverses nodes bottom-up
            sum_node = self.pre_ltr_info[self.tree_size - i - 1]
            sum_parent = sum_node.parent
            # Update myself
            sum_node.sum_cost += delete(sum_node.node)
            if sum_parent:
                sum_parent.sum_cost += sum_node.sum_cost


        current_leaf = NodeInfo.EMPTY
        for i in range(self.tree_size):
            # right-to-left preorder traversal
            node = self.pre_rtl_info[i]
            node.lnr = current_leaf
            if not node.children:
                current_leaf = node

            # left-to-right postorder traversal
            # Stores leftmost leaf descendants for each node.
            # Used for mapping computation.
            node = self.post_ltr_info[i]
            node.lld = node if node.size == 1 else node.children[0].lld

            # right-to-left postorder traversal
            # Stores rightmost leaf descendants for each node.
            node = self.post_rtl_info[i]
            node.rld = node if node.size == 1 else node.children[-1].rld

    def preorder_ltr(self, tree, target=None):
        """Generator that traverses tree in left-to-right preorder"""
        if target is None:
            target = tree.pre_ltr + tree.size
        for pre_ltr in range(tree.pre_ltr, target):
            yield pre_ltr, self.pre_ltr_info[pre_ltr]

    def traverse_up(self, current, target=None):
        """Traverse up to parent until it reaches the target
        If target is not specified, it is the root of the tree
        Generates:
           parent Info
           last visited node
        """
        target = target or self.pre_ltr_info[0]
        target_pre_ltr = target.pre_ltr
        parent = current.parent
        while parent and parent.pre_ltr >= target_pre_ltr:
            yield parent, current
            current, parent = parent, parent.parent

    def walk_up(self, current, target=None):
        """Same thing as traverse_up, but has a initial iteration
        that generates current, None"""
        child = NodeInfo(None, -1)
        child.parent = current
        current.fake_child = child
        yield current, child
        current.fake_child = None
        for parent, curr in self.traverse_up(current, target):
            yield parent, curr


class NodeInfo(object):
    """Represents a Tree Node with extra information"""
    # pylint: disable=too-many-instance-attributes

    EMPTY = None

    def __init__(self, node, preorder, config=Config):
        # Config obj/cls
        self.config = config

        # Node referred by this info
        self.node = node

        # Left-to-right preorder traversal index
        self.pre_ltr = preorder

        # Right-to-left preorder traversal index
        self.pre_rtl = -1

        # Left-to-right postorder traversal index
        self.post_ltr = -1

        # Rigth-to-left postorder traversal index
        self.post_rtl = -1

        # Parent node_info
        self.parent = None

        # Node children in left-to-right preorder
        self._children = []

        # Node lies on the leftmost path starting at its parent
        # See [2, Section 5.3, Algorithm 1, Lines 26,36]
        self.type_l = False

        # Node lies on the rightmost path starting at its parent
        # See [2, Section 5.3, Algorithm 1, Lines 26,36]
        self.type_r = False

        # Cost of spf_A (single path function using an inner path)
        # for the subtree rooted at this node. See [1, Section 5.2]
        self.desc_sum = 0

        # Cost of spf_L (single path function using the leftmost path)
        # for the subtree rooted at this node. See [1, Section 5.2]
        self.kr_sum = 0

        # Cost of spf_R (single path function using the rightmost path)
        # for the subtree rooted at this node. See [1, Section 5.2]
        self.rev_kr_sum = 0

        # Size of node subtree (including itself and all its descendants)
        self.size = 1

        # First leaf node to the left of this node. See [1, Section 8.4].
        self.lnl = self.EMPTY

        # First leaf node to the right of this node. See [1, Section 8.4].
        self.lnr = self.EMPTY

        # Leftmost leaf descendant of this node
        self.lld = self.EMPTY

        # Rightmost leaf descendant of this node
        self.rld = self.EMPTY

        # Cost of deleting/inserting all nodes in the subtree rooted at this node
        self.sum_cost = config.valuecls()

        # Overrides leftmost and rightmost child
        self.fake_child = None

        self.num = -1


    def __bool__(self):
        return bool(self.node)

    def __nonzero__(self):
        return bool(self.node)

    @property
    def children(self):
        """Returns node children"""
        if self.fake_child is not None:
            return [self.fake_child]
        return self._children

    @property
    def rightmost(self):
        """Returns rightmost child"""
        children = self.children
        if not children:
            return NodeInfo(None, -1, self.config)
        return children[-1]

    @property
    def leftmost(self):
        """Returns leftmost child"""
        children = self.children
        if not children:
            return NodeInfo(None, -1, self.config)
        return children[0]

    def __repr__(self):
        return str(self.post_ltr + 1)

EMPTY = NodeInfo.EMPTY = NodeInfo(None, -1)
EMPTY.lnl = EMPTY.lnr = EMPTY.lld = EMPTY.rld = EMPTY
