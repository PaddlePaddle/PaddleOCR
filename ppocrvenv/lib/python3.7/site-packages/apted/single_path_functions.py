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
"""Single path functions implementations"""
from __future__ import (absolute_import, division)

from itertools import chain

LEFT, RIGHT, INNER = 0, 1, 2


def get(matrix, a, b, swap, da=0, db=0):
    """Get value from matrix[a][b] or matrix[b][a], according to swap"""
    # pylint: disable=invalid-name
    # pylint: disable=too-many-arguments
    a, b = swap(a, b)
    return matrix[a + da][b + db]


def spf1(it1, it2, config, subtree1, subtree2):
    """Implements spf1 single path function for the case when one of the
    subtrees  is a single node [2, Section 6.1, Algorithm 2].

    We allow an arbitrary cost model which in principle may allow renames
    to have a lower cost than the respective deletion plus insertion.
    Thus, Formula 4 in [2] has to be modified to account for that case.

    In this method we don't have to verify if input subtrees have been
    swapped because they're always passed in the original input order.

    Params:
      ni1 -- node indexer for the source input subtree
      subtree1 -- root node of a subtree in the source input tree
      ni2 -- node indexer for destination input subtree
      subtree2 -- root node of a subtree in the destination input tree

    Returns the tree edit distance between two subtrees of the source and
    destination input subtrees.


    """
    #todo: Merge the initialisation loop in tedInit with this method.
    #   Currently, spf1 doesn't have to store distances in delta, because
    #   all of them have been stored in tedInit.
    # pylint: disable=invalid-name
    delete, insert, rename = config.delete, config.insert, config.rename

    if subtree1.size == 1 and subtree2.size == 1:
        n1, n2 = subtree1.node, subtree2.node
        return min(rename(n1, n2), delete(n1) + insert(n2))
    if subtree1.size == 1:
        n1 = subtree1.node
        return sub_spf1(
            it2, subtree1, subtree2, delete(n1),
            lambda n1, n2: rename(n1.node, n2.node) - insert(n2.node)
        )

    # subtree2.size == 1:
    n2 = subtree2.node
    return sub_spf1(
        it1, subtree2, subtree1, insert(n2),
        lambda n2, n1: rename(n1.node, n2.node) - delete(n1.node)
    )


def sub_spf1(ni, subtree1, subtree2, op, calculate):
    """Implements spf1 single path function for the case when the
    other subtree is a single node

    Params:
      ni -- node indexer for the subtree that has more than one element
      subtree1 -- subtree that has a single element
      subtree2 -- subtree that has more than one element
      op -- cost of deleting/inserting node
      calculate -- function(node, other) that returns the cost of
        renaming nodes
    """
    # pylint: disable=invalid-name
    # pylint: disable=too-many-arguments
    cost = subtree2.sum_cost
    max_cost = cost + op
    min_ren_minus_op = min(chain([cost], [
        calculate(subtree1, info)
        for _, info in ni.preorder_ltr(subtree2)
    ]))
    return min(min_ren_minus_op + cost, max_cost)


class SinglePathFunction(object):
    """Implements the single-path functions"""

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    # pylint: disable=invalid-name
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-instance-attributes


    def __init__(self, it1, it2, c1, c2, apted, path_id, path_type, swapped):

        self.path_id, self.path_type = path_id, path_type
        self.swapped = swapped
        self.apted = apted
        self.it1, self.it2 = it1, it2
        self.current1, self.current2 = c1, c2
        vcls = self.vcls = apted.config.valuecls

        # Swapped
        if swapped:
            self.swap = lambda a, b: (b, a)
            self.delete = apted.config.insert
            self.insert = apted.config.delete
            self.rename = lambda x, y: apted.config.rename(y, x)
        else:
            self.swap = lambda a, b: (a, b)
            self.delete = apted.config.delete
            self.insert = apted.config.insert
            self.rename = apted.config.rename

        self.it2_loff = self.current2.pre_ltr
        self.it2_roff = self.current2.pre_rtl

        self.max_size = max(it1.tree_size, it2.tree_size) + 1

        # todo: Do not use fn and ft arrays [1, Section 8.4]
        # Array used in the algorithm before [1]. Using it does not change
        # the complexity. Do not use it [1, Section 8.4].
        self.fn = [0] * (self.max_size + 1)
        # Array used in the algorithm before [1]. Using it does not change
        # the complexity. TODO: Do not use it [1, Section 8.4].
        self.ft = [0] * (self.max_size + 1)
        # One of distance arrays to store intermediate distances in spfA.
        self.q = [vcls() for _ in range(self.max_size)]

        size1, size2 = self.current1.size + 1, self.current2.size + 1
        self.delta = apted.delta
        self.s = [
            [vcls() for _ in range(size2)]
            for _ in range(size1)
        ]
        self.t = [
            [vcls() for _ in range(size2)]
            for _ in range(size2)
        ]

        # SPF-side
        self.ald = None

        # Loop A iterators
        self.v, self.vparent = None, None
        self.it1_off = -1

        # Symmetry variables
        # Map id (pre_ltr/pre_rtl) to Node Info
        self.atb_1, self.atb_2 = None, None
        self.bta_1, self.bta_2 = None, None

        # Get index from NodeInfo
        self.atb = None # Returns pre_ltr/pre_rtl
        self.bta = None # Returns pre_rtl/pre_ltr

        # Loop B
        self.lna = None
        self.amost_child = None
        self.bmost_child = None
        self.left_right = None
        self.it2_doff = -1
        self.a_part = False

        # Loop C
        self.cost1 = vcls()  # Current forest 1 cost
        self.size1 = 0  # Current forest 1 size
        self.cost2 = vcls()  # Current forest 2 cost
        self.size2 = 0  # Current forest 2 size
        self.counter = 0

        # Loop D
        self.sp1_func = None
        self.sp2_func = None
        self.sp2_func_after = None
        self.sp3_pre = None
        self.sp3_func = None
        self.sp3_func_after = None

    def spf_l(self):
        """Implements single-path function for left paths
        [1, Sections 3.3,3.4,3.5].
        We use this single-path function due to better performance compared
        to spfA.
        """
        self.ald = lambda x: x.lld
        self.atb = lambda x: x.post_ltr
        self.atb_1 = self.it1.post_ltr_info
        self.atb_2 = self.it2.post_ltr_info
        self.left_right = self.swap
        return self.spf_generic_side()

    def spf_r(self):
        """Implements single-path function for right paths
        [1, Sections 3.3,3.4,3.5].
        We use this single-path function due to better performance compared
        to spfA.
        """
        self.ald = lambda x: x.rld
        self.atb = lambda x: x.post_rtl
        self.atb_1 = self.it1.post_rtl_info
        self.atb_2 = self.it2.post_rtl_info
        self.left_right = self.swap
        return self.spf_generic_side()

    def spf_generic_side(self):
        """Implements single-path function for side(left/right) paths
        [1, Sections 3.3,3.4,3.5].
        We use this single-path function due to better performance compared
        to spfA.
        """
        current1, current2 = self.current1, self.current2
        size1, size2 = current1.size, current2.size
        tree_edit_dist = self.tree_edit_dist
        vcls = self.vcls
        # Initialise the array to store the keyroot nodes in the right-hand
        # input subtree
        keyroots = [None] * size2
        # Get the leftmost/rightmost leaf node of the right-hand input subtree
        leaf = self.ald(current2)

        # Calculate the keyroot nodes in the right-hand input subtree.
        # first_keyroot is the index in keyroots of the first keyroot node that
        # we have to process. We need this index because keyRoots array is
        # larger than the number of keyroot nodes.
        first_keyroot = self.compute_keyroots(current2, leaf, keyroots, 0)
        # Initialise an array to store intermediate distances for subforest pairs.
        forestdist = [
            [vcls() for _ in range(size2 + 1)]
            for _ in range(size1 + 1)
        ]
        # Compute the distances between pairs of keyroot nodes. In the left-hand
        # input subtree only the root is the keyroot. Thus, we compute the distance
        # between the left-hand input subtree and all keyroot nodes in the
        # right-hand input subtree.
        for i in range(first_keyroot - 1, -1, -1):
            tree_edit_dist(current1, keyroots[i], forestdist)

        return forestdist[size1][size2]

    def compute_keyroots(self, root, leaf, keyroots, index):
        """Calculates and stores keyroot nodes for left paths of the given
        subtree recursively."""
        ald = self.ald
        # The root is a keyroot node. Add it to keyroots.
        keyroots[index] = root
        # Increment the index to know where to store the next keyroot node.
        index += 1
        # Walk up the left/right path starting with the leftmost/rightmost
        # leaf of root until the child of root.
        path = leaf
        root_id = root.pre_ltr
        while path.pre_ltr > root_id:
            parent = path.parent
            # For each sibling to the right/left of pathNode,
            # execute this method recursively.
            # Each right/left sibling of pathNode is a keyroot node.
            for child in parent.children:
                if child is not path:
                    index = self.compute_keyroots(child, ald(child), keyroots, index)
            path = parent
        return index

    def tree_edit_dist(self, subtree1, subtree2, forestdist):
        """Implements the core of spfL/spfR. Fills in forestdist array with
        intermediate distances of subforest pairs in dynamic-programming fashion
        """
        delete, insert, rename = self.delete, self.insert, self.rename
        atb_1, atb_2, atb = self.atb_1, self.atb_2, self.atb
        delta, ald, vcls = self.delta, self.ald, self.vcls
        lr = self.left_right

        i, j = atb(subtree1), atb(subtree2)
        # We need to offset the node ids for accessing forestdist array which
        # has indices from 0 to subtree size. However, the subtree node indices
        # do not necessarily start with 0.
        # Whenever the original LTR/RTL postorder id has to be accessed,
        # use i+ioff and j+joff.
        ioff = atb(ald(subtree1)) - 1
        joff = atb(ald(subtree2)) - 1
        # Variables holding costs of each minimum element.
        da = db = dc = 0
        # Initialize forestdist array with deletion and insertion costs of each
        # relevant subforest.
        forestdist[0][0] = vcls(0)
        for i1 in range(1, i - ioff + 1):
            info = atb_1[i1 + ioff]
            forestdist[i1][0] = forestdist[i1 - 1][0] + delete(info.node)
        for j1 in range(1, j - joff + 1):
            info = atb_2[j1 + joff]
            forestdist[0][j1] = forestdist[0][j1 - 1] + insert(info.node)

        # Fill in the remaining costs.
        for i1 in range(1, i - ioff + 1):
            info1 = atb_1[i1 + ioff]
            info1_node = info1.node
            info1_ald = ald(info1)

            for j1 in range(1, j - joff + 1):
                info2 = atb_2[j1 + joff]
                info2_node = info2.node
                info2_ald = ald(info2)
                # Increment the number of subproblems.
                self.counter += 1
                # Calculate partial distance values for this subproblem.
                u = rename(info1_node, info2_node)
                da = forestdist[i1 - 1][j1] + delete(info1_node)
                db = forestdist[i1][j1 - 1] + insert(info2_node)
                # If current subforests are subtrees.
                a, b = self.swap(info1.pre_ltr, info2.pre_ltr)
                if info1_ald is ald(subtree1) and info2_ald is ald(subtree2):
                    dc = forestdist[i1 - 1][j1 - 1] + u
                    # Store the relevant distance value in delta array.
                    delta[a][b] = forestdist[i1 - 1][j1 - 1]
                else:
                    id1 = atb(info1_ald) - 1 - ioff
                    id2 = atb(info2_ald) - 1 - joff
                    dc = forestdist[id1][id2] + delta[a][b] + u
                # Calculate final minimum.
                forestdist[i1][j1] = min(da, db, dc)

    def spf_a(self):
        """Implements the single-path function spfA. Here, we use it strictly
        for inner paths (spfL and spfR have better performance for leaft and
        right  paths, respectively) [1, Sections 7 and 8]. However, in this
        stage it also executes correctly for left and right paths."""
        return self.loop_a()

    def loop_a(self):
        """Loop A [1, Algorithm 3] - walk up the path"""
        pre_ltr_1, pre_ltr_2 = self.it1.pre_ltr_info, self.it2.pre_ltr_info
        pre_rtl_1, pre_rtl_2 = self.it1.pre_rtl_info, self.it2.pre_rtl_info
        path, path_leaf = self.path_type, self.it1.pre_ltr_info[self.path_id]

        result = 0, -1, []

        # for v in F on the path from dummy E to root
        for vparent, v in self.it1.walk_up(path_leaf, self.current1):
            self.vparent, self.v = vparent, v
            # start is v; end is p(v)
            vparent_pre_ltr, vparent_pre_rtl = vparent.pre_ltr, vparent.pre_rtl
            v_pre_ltr, v_pre_rtl = v.pre_ltr, v.pre_rtl

            left_part = bool(v) and v_pre_ltr - vparent_pre_ltr > 1
            right_part = bool(v) and v_pre_rtl - vparent_pre_rtl > 1
            neither = not left_part and not right_part

            # Deal with nodes to the left of the path
            if path == RIGHT or path == INNER and left_part:
                if v:
                    rf_first = v_pre_rtl
                    lf_first = v_pre_ltr - 1
                else:
                    rf_first = vparent_pre_rtl
                    lf_first = vparent_pre_ltr

                lf_last = vparent_pre_ltr + int(right_part)
                rf_last = -1 if right_part else vparent_pre_rtl

                # B: left(F_p(v), v) \\cup lv_last
                lf_range = range(lf_first, lf_last - 1, -1)

                # Adjust symmetry variables
                self.it1_off = vparent_pre_ltr
                self.it2_doff = self.it2_loff
                self.atb_1, self.atb_2 = pre_ltr_1, pre_ltr_2
                self.bta_1, self.bta_2 = pre_rtl_1, pre_rtl_2
                self.atb = lambda x: x.pre_ltr
                self.bta = lambda x: x.pre_rtl
                self.amost_child = lambda x: x.leftmost
                self.bmost_child = lambda x: x.rightmost
                self.lna = lambda x: x.lnl
                self.left_right = lambda ag, bg: (ag, bg)
                self.a_part = left_part

                #l = a. r = b
                result = self.loop_b(lf_range, rf_first, rf_last)

            # Deal with nodes to the right of the path
            if path == LEFT or path == INNER and (right_part or neither):
                if v:
                    lf_first = vparent_pre_ltr + 1
                    rf_first = v_pre_rtl - 1
                else:
                    lf_first = vparent_pre_ltr
                    rf_first = vparent_pre_rtl

                rf_last = vparent_pre_rtl
                lf_last = vparent_pre_ltr

                # B': right(F_p(v), v) \\cup {p(v)}
                rf_range = range(rf_first, rf_last - 1, -1)

                # Adjust symmetry variables
                self.it1_off = vparent_pre_rtl
                self.it2_doff = self.it2_roff
                self.atb_1, self.atb_2 = pre_rtl_1, pre_rtl_2
                self.bta_1, self.bta_2 = pre_ltr_1, pre_ltr_2
                self.atb = lambda x: x.pre_rtl
                self.bta = lambda x: x.pre_ltr
                self.amost_child = lambda x: x.rightmost
                self.bmost_child = lambda x: x.leftmost
                self.lna = lambda x: x.lnr
                self.left_right = lambda ag, bg: (bg, ag)
                self.a_part = right_part

                #r = a. l = b
                result = self.loop_b(rf_range, lf_first, lf_last)
        return result

    def loop_b(self, af_range, bf_first, bf_last):
        """Loop B [1, Algoritm 3] - for all nodes in G (right-hand input tree)."""
        current2, bta_2 = self.current2, self.bta_2
        atb, bta, lna = self.atb, self.bta, self.lna
        amost_child, bmost_child = self.amost_child, self.bmost_child

        result = 0, -1, []

        self.fn = [-1] * self.max_size
        self.ft = [-1] * self.max_size

        # Store the current size and cost of forest in F.
        tmpsize1, tmpcost1 = self.size1, self.cost1
        current2_atb = atb(current2)

        bg_last = bta(current2)
        bg_first = bg_last + current2.size - 1
        for bg in range(bg_first, bg_last - 1, -1):  # Reversed preorder bta
            bg_info = bta_2[bg]
            bg_atb = atb(bg_info)
            bg_ln = atb(lna(bg_info))
            self.update_fn_array(bg_ln, bg_atb, current2_atb)
            self.update_ft_array(bg_ln, bg_atb)

            # B: {rw} \\cup left(G',rw) in reverse LTR preorder
            # B': {lw} \\cup right(G',lw) in reverse RTL preorder
            ag_first = bg_info
            ag_last = current2 if bg_info is current2 else amost_child(current2)
            ag_range = list(self.each_ft(ag_first, ag_last))
            self.size1, self.cost1 = tmpsize1, tmpcost1

            result = self.loop_c(
                bg, af_range, ag_range, bf_first, bf_last,
                bg_info.size
            )
            bg_parent = bg_info.parent
            self.update_delta(
                af_range, ag_range, bg, bg_parent,
                bg > bg_last and bg_info is bmost_child(bg_parent)
            )

        return result

    def update_delta(self, af_range, ag_range, bg, bg_parent, is_bmost_child):
        """Update Delta and q. See [1, Algorithm 4]"""
        loff2, roff2, doff2 = self.it2_loff, self.it2_roff, self.it2_doff
        delta, q, s, t = self.delta, self.q, self.s, self.t
        off1, left_right = self.it1_off, self.left_right

        vp = self.vparent
        vp_parent = vp.parent
        af_last = af_range[-1]

        # if bg is rightmost/leftmost child of p(bg)
        if is_bmost_child:
            amost = self.atb(bg_parent) + 1
            if self.a_part:
                a, b = self.swap(vp.pre_ltr, bg_parent.pre_ltr)
                delta[a][b] = s[af_last + 1 - off1][amost - doff2]

            if vp_parent and vp is vp_parent.rightmost is vp_parent.leftmost:
                a, b = self.swap(vp_parent.pre_ltr, bg_parent.pre_ltr)
                delta[a][b] = s[af_last - off1][amost - doff2]
            # B: for node lv in left(F_p(v), v) \\cup lv_last in reverse LTR
            # B': for node rv in right(F_p(v), v) \\cup {p(v)}
            for af in af_range:
                q[af] = s[af - off1][amost - doff2]

        # B: foreach lw in {rw} \\cup left(G', rw) in reverse LTR preorder
        # B': foreach rw in {lw} \\cup right(G', lw) in reverse RTL preorder
        # : first pointers can be precomputed
        for ag, _ in ag_range:
            lg, rg = left_right(ag, bg)
            t[lg - loff2][rg - roff2] = s[af_last - off1][ag - doff2]

    def loop_c(self, bg, af_range, ag_range, bf, bf_last, g_size):
        """Loop C of [1, Algorithm 3]
        C: foreach node lv in left(F_p(v), v) \\cup lv_last in reverse LTR pre
        C': foreach node rv in right(F_p(v), v) \\cup {p(v)} in reverse RTL pre
        """
        # pylint: disable=cell-var-from-loop
        atb, bta, atb_1 = self.atb, self.bta, self.atb_1
        loff2, roff2, doff2 = self.it2_loff, self.it2_roff, self.it2_doff
        off1, fn, lr = self.it1_off, self.fn, self.left_right
        swap, delta, vcls = self.swap, self.delta, self.vcls
        q, t, s = self.q, self.t, self.s
        v_atb = atb(self.v)
        result = 0, -1, []
        af_last = af_range[-1]

        for af in af_range:
            af_info = atb_1[af]

            if af == af_last:  # if lv == p(v) then rv <- p(v)
                bf = bf_last

            af_node = af_info.node

            # Increment size and cost of F forest by node lf
            self.size1 += 1
            self.cost1 += self.delete(af_node)
            # Reset size and cost of forest in G to subtree G_lg_first
            self.size2, self.cost2 = g_size, vcls(0)

            af_subtree_size = af_info.size
            af_is_consecutive = af_is_side_sibling = False
            distance = v_atb - af
            af_is_consecutive = distance == 1
            af_is_side_sibling = distance == af_subtree_size

            sp1source = 1  # Search sp1 value in s array by default
            sp3source = 1  # Search second part of sp3 value in s array by default
            if af_is_consecutive:  # F_{lF,rF}-af is the path node subtree.
                sp1source = 2
            if bta(af_info) == bf: # F_{lF,rF} is a tree.
                sp3_0 = vcls(0)
                if af_subtree_size == 1: # F_{lF,rF} is a single node.
                    sp1source = 3
                sp3source = 2
            else:
                sp3_0 = self.cost1 - af_info.sum_cost
                if af_is_side_sibling:
                    sp3source = 3

            # Set SP1
            sp1_s = s[af + 1 - off1]
            self.sp1_func = {
                1: lambda ag: sp1_s[ag - doff2],
                2: lambda ag: get(t, ag, bg, lr, -loff2, -roff2),
                3: lambda ag: self.cost2,
            }[sp1source]

            # Set SP2
            sp2_s = s[af - off1]
            self.sp2_func = lambda ag: self.cost1 if g_size == 1 else q[af]
            self.sp2_func_after = lambda ag: sp2_s[fn[ag] - doff2]

            # Set SP3
            sp3_s = s[0]
            if sp3source == 1:
                sp3_s = s[af + af_subtree_size - off1]
            self.sp3_func = lambda lg, info: sp3_0
            self.sp3_func_after = {
                1: lambda ag, info: sp3_s[fn[ag + info.size - 1] - doff2],
                2: lambda ag, info: self.cost2 - info.sum_cost,
                3: lambda ag, info: get(t, fn[ag + info.size - 1], bg, lr, -loff2, -roff2)
            }[sp3source]
            af_pre = af_info.pre_ltr
            self.sp3_pre = lambda bg_pre: get(delta, af_pre, bg_pre, swap)

            result = self.loop_d(s[af - off1], ag_range, af_node)

        return result

    def loop_d(self, swrite, ag_range, f_node):
        """Loop D of [1, Algorithm 3]
        D: foreach node lw in {rw} \\cup left(G', rw) in reverse LTR preorder
        D': foreach node rw in {lw} \\cup rigth(G', lw) in reverse RTL preorder
        """
        doff2 = self.it2_doff
        result = 0, -1, []
        iterator = iter(ag_range)
        # First iteration
        ag, ag_info = next(iterator)
        ag_node = ag_info.node
        self.cost2 += ag_info.sum_cost
        result = swrite[ag - doff2] = self.calculate_min_loop_d(
            ag, ag_info, ag_node, f_node
        )

        self.counter += 1
        self.sp2_func = self.sp2_func_after
        self.sp3_func = self.sp3_func_after

        # Other iterations
        for ag, ag_info in iterator:
            ag_node = ag_info.node
            self.cost2 += self.insert(ag_node)
            result = swrite[ag - doff2] = self.calculate_min_loop_d(
                ag, ag_info, ag_node, f_node
            )

            self.counter += 1

        return result

    def calculate_min_loop_d(self, index, g_info, g_node, f_node):
        """Compute delta(F_(l_v,r_v), G_(l_w,r_w))
        Return min between sp1, sp2, and sp3
        Index is g_info.pre_ltr in Loop D
              or g_info.pre_rtl in Loop D'
        """
        return min(
            self.sp1_func(index) + self.delete(f_node),
            self.sp2_func(index) + self.insert(g_node),
            self.sp3_pre(g_info.pre_ltr) + self.sp3_func(index, g_info)
            + self.rename(f_node, g_node)
        )

    def each_ft(self, start, finish):
        """Iterate on ft array"""
        ft, atb, atb_2 = self.ft, self.atb, self.atb_2
        finish_atb = atb(finish)
        current, current_atb = start, atb(start)
        yield current_atb, current
        current_atb = ft[current_atb]
        while current_atb >= finish_atb:
            current = atb_2[current_atb]
            yield current_atb, current
            current_atb = ft[current_atb]

    def update_fn_array(self, ln_index, node_index, current_index):
        """fn array used in the algorithm before [1]. Using it does not change
        the complexity

        TODO: Do not use it [1, Section 8.4].
        """
        fn = self.fn
        ln = ln_index if ln_index >= current_index else - 1
        fn[node_index], fn[ln] = fn[ln], node_index

    def update_ft_array(self, ln_index, node_index):
        """ft array used in the algorithm before [1]. Using it does not change
        the complexity

        TODO: Do not use it [1, Section 8.4].
        """
        ft, fn = self.ft, self.fn
        ft[node_index] = ln_index
        if fn[node_index] > -1:
            ft[fn[node_index]] = node_index

    def __call__(self):
        result = {
            LEFT: self.spf_l,
            RIGHT: self.spf_r,
            INNER: self.spf_a,
        }[self.path_type]()
        self.apted.counter = self.counter
        return result

class UseOnlySPFA(SinglePathFunction):
    """Single Path Function that uses only SPFA"""

    def __call__(self):
        result = self.spf_a()
        self.apted.counter = self.counter
        return result
