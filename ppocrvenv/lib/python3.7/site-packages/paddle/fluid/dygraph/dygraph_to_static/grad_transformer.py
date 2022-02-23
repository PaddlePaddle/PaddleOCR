#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import warnings

from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper
from paddle.fluid.dygraph.dygraph_to_static import utils


class GradTransformer(gast.NodeTransformer):
    """
    A class transforms dygraph paddle.grad to static graph paddle.gradients. The
    transformation is applied to support double grad mode.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of GradTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node

    def transform(self):
        self.visit(self.root)

    def visit_Call(self, node):
        self.generic_visit(node)
        if not is_grad_api_node(node):
            return node

        dygraph_grad_parameters = [
            "outputs", "inputs", "grad_outputs", "retain_graph", "create_graph",
            "only_inputs", "allow_unused", "no_grad_vars"
        ]
        to_static_grad_param = {
            "outputs": "targets",
            "inputs": "inputs",
            "grad_outputs": "target_gradients",
            "no_grad_vars": "no_grad_set"
        }
        static_keywords = []

        for kw in node.keywords:
            if kw.arg not in dygraph_grad_parameters or kw.arg not in to_static_grad_param:
                warnings.warn("paddle.grad has unsupported parameter in jit: " +
                              kw.arg + ", jit will discard it")
                continue
            dygraph_grad_parameters.remove(kw.arg)
            kw.arg = to_static_grad_param[kw.arg]
            static_keywords.append(kw)

        for i in range(len(node.args)):
            arg_name = dygraph_grad_parameters[i]
            if arg_name not in to_static_grad_param:
                warnings.warn("paddle.grad has unsupported parameter in jit: " +
                              kw.arg + ", jit will discard it")
                continue
            kw = gast.keyword(
                arg=to_static_grad_param[arg_name], value=node.args[i])
            static_keywords.append(kw)

        node.func = gast.parse('paddle.static.gradients').body[0].value
        node.keywords = static_keywords
        node.args = []
        return node


def is_grad_api_node(node):
    assert isinstance(node, gast.Call)
    api_name = utils.ast_to_source_code(node.func).strip()
    if utils.is_paddle_api(node):
        if 'no_grad' in api_name:
            warnings.warn(
                "paddle.no_grad is only supported for inference model, and not supported for training under @to_static."
            )
            return False
        return api_name.endswith("grad")
    return False
