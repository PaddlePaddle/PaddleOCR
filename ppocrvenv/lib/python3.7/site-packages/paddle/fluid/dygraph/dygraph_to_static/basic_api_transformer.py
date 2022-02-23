# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import astor
from paddle.utils import gast

from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper
from paddle.fluid.dygraph.dygraph_to_static import utils


class BasicApiTransformer(gast.NodeTransformer):
    """
    Class to transform basic API from dygraph to static graph.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of BasicApiTransformer."

        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node
        self.class_node_dict = {}

    def transform(self):
        to_tensor_transformer = ToTensorTransformer(self.root)
        to_tensor_transformer.transform()
        self.visit(self.root)

        return self.wrapper_root

    def visit_Assign(self, node):
        if self._update_class_node_dict(node):
            return None

        for child_node in gast.walk(node.value):
            if isinstance(child_node, gast.Call):
                self._visit_Call(child_node)
        return node

    def visit_Expr(self, node):
        value_node = node.value
        for child_node in gast.walk(value_node):
            if isinstance(child_node, gast.Call):
                # TODO(liym27):
                #  Considers that a dygraph api which modifies the input or has a output.
                if utils.is_dygraph_api(child_node):
                    return
                else:
                    self._visit_Call(child_node)
        return node

    def _visit_Call(self, node):
        assert isinstance(node, gast.Call)
        func_name = astor.to_source(gast.gast_to_ast(node.func))

        if self._is_dygraph_forward(func_name):
            class_node = self._get_class_node(func_name)
            static_node = utils.to_static_ast(node, class_node)
            return static_node
        else:
            return node

    def _is_dygraph_forward(self, func_id):
        return func_id in self.class_node_dict

    def _get_class_node(self, func_id):
        return self.class_node_dict[func_id]

    def _update_class_node_dict(self, node):
        assert isinstance(node, gast.Assign)
        node_value = node.value
        if isinstance(node_value, gast.Call):
            if is_to_variable(node_value):
                return False

            if utils.is_dygraph_api(node_value):
                dygraph_api = node_value.func.attr
                if not utils.dygraph_class_to_static_api.get(dygraph_api):
                    return False

                utils.update_args_of_func(node_value, node_value, "__init__")
                target_str = astor.to_source(gast.gast_to_ast(node.targets[0]))
                self.class_node_dict[target_str] = node_value
                return True
            # TODO: node.value is not dygraph class
        return False


class ToTensorTransformer(gast.NodeTransformer):
    """
    Class to transform paddle.to_tensor and paddle.to_variable to paddle.assign
    """

    def __init__(self, node):
        assert isinstance(
            node, gast.AST
        ), "Input non-gast.AST node for the initialization of ToTensorTransformer."
        self.root = node

    def transform(self):
        self.visit(self.root)
        return self.root

    def visit_Call(self, node):
        assert isinstance(node, gast.Call)
        if is_to_variable(node):
            node = to_assign_node(node)
        self.generic_visit(node)
        return node


def is_to_variable(node):
    assert isinstance(node, gast.Call)
    api_name = utils.ast_to_source_code(node.func).strip()

    if utils.is_dygraph_api(node):
        return api_name.endswith("to_variable")

    if utils.is_paddle_api(node):
        return api_name.endswith("to_tensor")

    return False


def to_assign_node(node):
    # Transform dygraph api `fluid.dygraph.to_variable` alias `paddle.to_tensor` to static api `paddle.assign`.
    # NOTE:
    #   1. Api `to_variable` supports data type {float16, float32, float64, int16, int32, int64, uint8, uint16},
    #   but api `assign` only supports {float32, float64, int32, int64, bool};
    #   2. If the input of api `assign` is numpy.ndarray, its size cannot be greater than 1024 * 1024.

    assert isinstance(node, gast.Call)
    assign_api = gast.parse('paddle.assign').body[0].value
    node.func = assign_api

    if node.args:
        node.args = [node.args[0]]
        node.keywords = []
    else:
        for idx, kw in enumerate(node.keywords):
            if kw.arg == 'value' or kw.arg == 'data':
                node.keywords[idx].arg = 'x'
                node.keywords = [node.keywords[idx]]
                node.args = []
                break
    return node
