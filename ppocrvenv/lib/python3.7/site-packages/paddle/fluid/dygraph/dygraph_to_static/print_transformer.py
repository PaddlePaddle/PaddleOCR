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

from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper, StaticAnalysisVisitor


class PrintTransformer(gast.NodeTransformer):
    """
    This class transforms python print function to fluid.layers.Print.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of PrintTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node

        self.static_analysis_visitor = StaticAnalysisVisitor(self.root)
        self.node_to_wrapper_map = self.static_analysis_visitor.get_node_to_wrapper_map(
        )

    def transform(self):
        self.visit(self.root)

    # NOTE: deal with print in PY3
    def visit_Call(self, node):
        if isinstance(node.func, gast.Name) and node.func.id == 'print':
            node = self._create_print_node(node.args)
        return node

    # NOTE: deal with print in PY2
    def visit_Print(self, node):
        convert_print_node = self._create_print_node(node.values)
        return gast.Expr(value=convert_print_node)

    def _create_print_node(self, print_args):
        convert_print_func = gast.parse(
            'paddle.jit.dy2static.convert_print').body[0].value
        return gast.Call(func=convert_print_func, args=print_args, keywords=[])
