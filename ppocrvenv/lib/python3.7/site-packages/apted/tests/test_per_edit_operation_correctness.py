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
"""Correctness unit tests of distance computation for node labels with
a single string value and per-edit-operation cost model."""
# pylint: disable=missing-docstring
import unittest
import json
import os

from ..helpers import Tree
from ..config import PerEditOperationConfig
from ..apted import APTED
from ..all_possible_mappings_ted import AllPossibleMappingsTED


def test_factory(test):
    """Creates testcase for test dict"""
    tree1 = Tree.from_text(test["t1"])
    tree2 = Tree.from_text(test["t2"])
    config = PerEditOperationConfig(.4, .4, .6)

    class TestPerEditOperationCorrectness(unittest.TestCase):
        """Correctness unit tests of distance computation for node labels with
        a single string value and per-edit-operation cost model."""

        def test_distance_unit_cost(self):
            apted = APTED(tree1, tree2, config)
            apmted = AllPossibleMappingsTED(tree1, tree2, config)
            self.assertAlmostEqual(
                apmted.compute_edit_distance(),  # correct result
                apted.compute_edit_distance()    # result
            )

    return type(
        "TestPerEditOperationCorrectness{}".format(test["testID"]),
        (TestPerEditOperationCorrectness,), {}
    )


def create_testcases():
    """Creates correctness testcases"""
    name = os.path.abspath(os.path.join(
        __file__, "..", "..", "resources", "mini.json"
    ))
    with open(name) as data_file:
        return [test_factory(test) for test in json.load(data_file)]

for testcase in create_testcases():
    locals()[testcase.__name__] = testcase
