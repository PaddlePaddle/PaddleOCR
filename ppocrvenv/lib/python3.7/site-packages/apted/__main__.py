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
"""This is the command line interface for executing APTED algorithm."""
from __future__ import (absolute_import, division, print_function)

import argparse
import time
from textwrap import dedent
from .helpers import Tree
from .apted import APTED

EPILOG = r"""
DESCRIPTION

    Compute the edit distance between two trees with APTED algorithm [1,2].
    APTED supersedes the RTED algorithm [3].
    By default unit cost model is supported where each edit operation
    has cost 1 (in case of equal labels the cost is 0).

    For implementing other cost models see the details on github website
    (https://github.com/JoaoFelipe/apted).

    This implementation is a port from the original Java implementation
    also available on github (https://github.com/DatabaseGroup/apted)

LICENSE

    The source code of this program is published under the MIT licence and
    can be found on github (https://github.com/JoaoFelipe/apted).

EXAMPLES

    python -m apted -t {a{b}{c}} {a{b{d}}}
    python -m apted -f 1.tree 2.tree
    python -m apted -t {a{b}{c}} {a{b{d}}} -m -v

REFERENCES

    [1] M. Pawlik and N. Augsten. Efficient Computation of the Tree Edit
        Distance. ACM Transactions on Database Systems (TODS) 40(1). 2015.
    [2] M. Pawlik and N. Augsten. Tree edit distance: Robust and memory-
        efficient. Information Systems 56. 2016.
    [3] M. Pawlik and N. Augsten. RTED: A Robust Algorithm for the Tree Edit
        Distance. PVLDB 5(4). 2011.

ORIGINAL AUTHORS

    Mateusz Pawlik, Nikolaus Augsten

IMPLEMENTATION AUTHOR

    Joao Felipe Pimentel
"""


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Compute the edit distance between two trees.",
        epilog=EPILOG
    )
    parser.add_argument("-m", "--mapping", action="store_true", help=dedent("""\
        compute the minimal edit mapping between two trees. There might
        be multiple minimal edit mappings. This option computes only one
        of them. The first line of the output is the cost of the mapping.
        The following lines represent the edit operations. n and m are
        postorder IDs (beginning with 1) of nodes in the left-hand and
        the right-hand trees respectively.
            n->m - rename node n to m
            n->0 - delete node n
            0->m - insert node m
        """))
    parser.add_argument("-v", "--verbose", action="store_true", help=dedent("""\
        print verbose output, including tree edit distance, runtime,
        number of relevant subproblems and strategy statistics.
        """))

    group = parser.add_argument_group('Input')
    group_ex = group.add_mutually_exclusive_group(required=True)
    group_ex.add_argument(
        '-t', '--trees', nargs=2, metavar=("TREE1", "TREE2"), type=str,
        help=dedent("""\
        compute the tree edit distance between TREE1 and TREE2. The
        trees are encoded in the bracket notation, for example, in tree
        {A{B{X}{Y}{F}}{C}} the root node has label A and two children
        with labels B and C. B has three children with labels X, Y, F.
        """)
    )

    group_ex.add_argument(
        '-f', '--files', nargs=2, metavar=("FILE1", "FILE2"),
        type=argparse.FileType('r'), help=dedent("""\
        compute the tree edit distance between the two trees stored in
        the files FILE1 and FILE2. The trees are encoded in bracket
        notation
        """)
    )


    namespace, _ = parser.parse_known_args()
    if namespace.trees:
        tree1, tree2 = map(Tree.from_text, namespace.trees)
    elif namespace.files:
        tree1, tree2 = map(lambda x: Tree.from_text(x.read()), namespace.files)

    apted = APTED(tree1, tree2)
    time1 = time.time()
    ted = apted.compute_edit_distance()
    time2 = time.time()
    if namespace.verbose:
        print("distance:            ", ted)
        print("runtime:             ", (time2 - time1))
    else:
        print(ted)

    if namespace.mapping:
        node_1, node_2 = apted.it1.node_info, apted.it2.node_info
        mapping = apted.compute_edit_mapping()
        for node1, node2 in mapping:
            if not namespace.verbose:
                value1 = node_1[id(node1)].post_ltr + 1 if node1 else 0
                value2 = node_2[id(node2)].post_ltr + 1 if node2 else 0
                print(value1, "->", value2)
            else:
                print(node1, "->", node2)

if __name__ == '__main__':
    main()
