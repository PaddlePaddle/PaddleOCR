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


class CastTransformer(gast.NodeTransformer):
    """
    This class transforms type casting into Static Graph Ast.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of CastTransformer."
        self._root = wrapper_root.node
        self._castable_type = {'bool', 'int', 'float'}

    def transform(self):
        self.visit(self._root)

    def visit_Call(self, node):
        self.generic_visit(node)
        func_str = ast_to_source_code(node.func).strip()
        if func_str in self._castable_type and len(node.args) > 0:
            args_str = ast_to_source_code(node.args[0]).strip()
            new_func_str = "paddle.jit.dy2static.convert_var_dtype({}, '{}')".format(
                args_str, func_str)
            new_node = gast.parse(new_func_str).body[0].value
            return new_node

        return node
