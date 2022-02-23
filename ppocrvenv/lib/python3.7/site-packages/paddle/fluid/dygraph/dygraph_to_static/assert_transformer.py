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

from __future__ import print_function

from paddle.utils import gast

from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code


class AssertTransformer(gast.NodeTransformer):
    """
    A class transforms python assert to convert_assert.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of AssertTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node

    def transform(self):
        self.visit(self.root)

    def visit_Assert(self, node):
        convert_assert_node = gast.parse(
            'paddle.jit.dy2static.convert_assert({test}, {msg})'.format(
                test=ast_to_source_code(node.test),
                msg=ast_to_source_code(node.msg)
                if node.msg else "")).body[0].value

        return gast.Expr(value=convert_assert_node)
