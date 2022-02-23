"""Correctness unit tests of distance and mapping computation."""
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
# pylint: disable=missing-docstring
import unittest
import json
import os

from ..helpers import Tree
from ..config import Config, meta_chained_config
from ..apted import APTED
from ..single_path_functions import UseOnlySPFA

def test_factory(test):
    """Creates testcase for test dict"""
    tree1 = Tree.from_text(test["t1"])
    tree2 = Tree.from_text(test["t2"])
    config = meta_chained_config(Config)()

    class TestChained(unittest.TestCase):
        """Correctness unit tests of distance and mapping computation."""
        def test_distance_unit_cost(self):
            apted = APTED(tree1, tree2, config)
            self.assertEqual(test["d"], apted.compute_edit_distance().value)

            apted = APTED(tree2, tree1, config)
            self.assertEqual(test["d"], apted.compute_edit_distance().value)

        def test_distance_unit_cost_spf_l(self):
            apted = APTED(tree1, tree2, config)
            self.assertEqual(
                test["d"], apted.compute_edit_distance_spf_test(0).value)

        def test_distance_unit_cost_spf_r(self):
            apted = APTED(tree1, tree2, config)
            self.assertEqual(
                test["d"], apted.compute_edit_distance_spf_test(1).value)

        def test_distance_unit_cost_spf_a(self):
            apted = APTED(tree1, tree2, config, spf=UseOnlySPFA)
            self.assertEqual(test["d"], apted.compute_edit_distance().value)

            apted = APTED(tree2, tree1, config, spf=UseOnlySPFA)
            self.assertEqual(test["d"], apted.compute_edit_distance().value)

        def test_mapping_cost_unit(self):
            apted = APTED(tree1, tree2, config)
            mapping = apted.compute_edit_mapping()
            self.assertEqual(test["d"], apted.mapping_cost(mapping))

    return type(
        "TestChained{}".format(test["testID"]), (TestChained,), {}
    )


def create_testcases():
    """Creates correctness testcases"""
    name = os.path.abspath(os.path.join(
        __file__, "..", "..", "resources",
        "correctness_test_cases.json"
    ))
    with open(name) as data_file:
        return [test_factory(test) for test in json.load(data_file)]


for testcase in create_testcases():
    locals()[testcase.__name__] = testcase
