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

import astor
from paddle.utils import gast

from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper, StaticAnalysisVisitor
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code
from paddle.fluid.dygraph.dygraph_to_static.utils import slice_is_num
from paddle.fluid.dygraph.dygraph_to_static.utils import is_control_flow_to_transform

from paddle.fluid.dygraph.dygraph_to_static.utils import SplitAssignTransformer


class ListTransformer(gast.NodeTransformer):
    """
    This class transforms python list used in control flow into Static Graph Ast.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of ListTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node
        self.list_name_to_updated = dict()
        self.list_nodes = set()

        self.static_analysis_visitor = StaticAnalysisVisitor(self.root)
        self.node_to_wrapper_map = self.static_analysis_visitor.get_node_to_wrapper_map(
        )
        var_env = self.static_analysis_visitor.get_var_env()
        var_env.cur_scope = var_env.cur_scope.sub_scopes[0]
        self.scope_var_type_dict = var_env.get_scope_var_type()

    def transform(self):
        SplitAssignTransformer(self.root).transform()
        self.visit(self.root)
        self.replace_list_with_tensor_array(self.root)

    def visit_Call(self, node):
        if isinstance(node.func, gast.Attribute):
            func_name = node.func.attr
            if func_name == "pop":
                node = self._replace_pop(node)
        return node

    def visit_Assign(self, node):
        if self._update_list_name_to_updated(node):
            return node

        if self._need_to_array_write_node(node):
            return self._transform_slice_to_tensor_write(node)

        self.generic_visit(node)
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if is_control_flow_to_transform(node, self.static_analysis_visitor,
                                        self.scope_var_type_dict):
            self._transform_list_append_in_control_flow(node)
        return node

    def visit_While(self, node):
        self.generic_visit(node)
        if is_control_flow_to_transform(node, self.static_analysis_visitor,
                                        self.scope_var_type_dict):
            self._transform_list_append_in_control_flow(node)
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        if is_control_flow_to_transform(node, self.static_analysis_visitor,
                                        self.scope_var_type_dict):
            self._transform_list_append_in_control_flow(node)
        return node

    def replace_list_with_tensor_array(self, node):
        for child_node in gast.walk(node):
            if isinstance(child_node, gast.Assign):
                if self._need_to_create_tensor_array(child_node):
                    child_node.value = self._create_tensor_array(
                        child_node.value)

    def _transform_list_append_in_control_flow(self, node):
        for child_node in gast.walk(node):
            if self._need_to_array_write_node(child_node):
                child_node.value = \
                    self._to_array_write_node(child_node.value)

    def _need_to_array_write_node(self, node):
        if isinstance(node, gast.Expr):
            if isinstance(node.value, gast.Call):
                if self._is_list_append_tensor(node.value):
                    return True

        if isinstance(node, gast.Assign):
            target_node = node.targets[0]
            if isinstance(target_node, gast.Subscript):
                list_name = ast_to_source_code(target_node.value).strip()
                if list_name in self.list_name_to_updated:
                    if self.list_name_to_updated[list_name] == True:
                        return True
        return False

    def _transform_slice_to_tensor_write(self, node):
        assert isinstance(node, gast.Assign)
        target_node = node.targets[0]

        target_name = target_node.value.id
        slice_node = target_node.slice

        if isinstance(slice_node, gast.Slice):
            pass
        elif slice_is_num(target_node):
            value_code = ast_to_source_code(node.value)
            i = "paddle.cast(" \
                "x=paddle.jit.dy2static.to_static_variable({})," \
                "dtype='int64')".format(ast_to_source_code(slice_node))
            assign_code = "{} = paddle.tensor.array_write(x={}, i={}, array={})" \
                .format(target_name, value_code, i, target_name)
            assign_node = gast.parse(assign_code).body[0]
        return assign_node

    def _is_list_append_tensor(self, node):
        """
        a.append(b): a is list, b is Tensor
        self.x.append(b): self.x is list, b is Tensor
        """
        assert isinstance(node, gast.Call)
        # 1. The func is `append`.
        if not isinstance(node.func, gast.Attribute):
            return False
        if node.func.attr != 'append':
            return False

        # 2. It's a `python list` to call append().
        value_name = astor.to_source(gast.gast_to_ast(node.func.value)).strip()
        if value_name not in self.list_name_to_updated:
            return False

        # 3. The number of arg of append() is one
        # Only one argument is supported in Python list.append()
        if len(node.args) != 1:
            return False

        # TODO(liym27): The arg of append() should be Tensor. But because the type of arg is often wrong with static analysis,
        #   the arg is not required to be Tensor here.
        # 4. The arg of append() is Tensor
        # arg = node.args[0]
        # if isinstance(arg, gast.Name):
        #     # TODO: `arg.id` may be not in scope_var_type_dict if `arg.id` is the arg of decorated function
        #     # Need a better way to confirm whether `arg.id` is a Tensor.
        #     try:
        #         var_type_set = self.scope_var_type_dict[arg.id]
        #     except KeyError:
        #         return False
        #     if NodeVarType.NUMPY_NDARRAY in var_type_set:
        #         return False
        #     if NodeVarType.TENSOR not in var_type_set and NodeVarType.PADDLE_RETURN_TYPES not in var_type_set:
        #         return False
        # # TODO: Consider that `arg` may be a gast.Call about Paddle Api. eg: list_a.append(paddle.reshape(x))
        # # else:
        # # return True
        self.list_name_to_updated[value_name.strip()] = True
        return True

    def _need_to_create_tensor_array(self, node):
        assert isinstance(node, gast.Assign)
        target_node = node.targets[0]
        try:
            target_id = target_node.id
        except AttributeError:
            return False
        if self.list_name_to_updated.get(target_id) and node in self.list_nodes:
            return True
        return False

    def _create_tensor_array(self, value_node):
        # Although `dtype='float32'`, other types such as `int32` can also be supported
        init_value = ast_to_source_code(value_node).strip()
        func_code = "paddle.tensor.create_array('float32', {})".format(
            init_value)
        func_node = gast.parse(func_code).body[0].value
        return func_node

    def _to_array_write_node(self, node):
        assert isinstance(node, gast.Call)
        array = astor.to_source(gast.gast_to_ast(node.func.value))
        x = astor.to_source(gast.gast_to_ast(node.args[0]))
        i = "paddle.tensor.array_length({})".format(array)
        func_code = "paddle.tensor.array_write(x={}, i={}, array={})".format(
            x, i, array)
        return gast.parse(func_code).body[0].value

    def _update_list_name_to_updated(self, node):
        assert isinstance(node, gast.Assign)
        target_node = node.targets[0]
        # NOTE: Code like `x, y = a, []` has been transformed to `x=a; y=[]`
        try:
            target_id = target_node.id
        except AttributeError:
            return False
        value_node = node.value
        if isinstance(value_node, gast.List):
            self.list_name_to_updated[target_id] = False
            self.list_nodes.add(node)
            return True
        elif target_id in self.list_name_to_updated and \
                self.list_name_to_updated[target_id] == False:
            del self.list_name_to_updated[target_id]
        return False

    def _replace_pop(self, node):
        """
        Replace a pop statement for a list or dict.
        For example:

            list_a = [0,1,2,3,4]
            x = list_a.pop()  # --> convert_pop(list_a)
            y = list_a.pop(1) # --> convert_pop(list_a, 1)

            dict_a = {"red":0, "blue":1, "yellow":2}
            m = dict_a.pop("red")           # --> convert_pop(dict_a, "red")
            n = dict_a.pop("black", 3)      # --> convert_pop(dict_a, "black", 3)

        """
        assert isinstance(node, gast.Call)
        assert isinstance(node.func, gast.Attribute)

        target_node = node.func.value
        target_str = ast_to_source_code(target_node).strip()

        args_str = [ast_to_source_code(arg).strip() for arg in node.args]

        # NOTE(liym27):
        # 1. pop stmt for a list if len(args_str) == 0
        # 2. pop stmt for a list or dict if len(args_str) == 1
        # 3. pop stmt for a dict if len(args_str) == 2
        if len(args_str) <= 2:
            new_pop_str = "paddle.jit.dy2static.convert_pop({}, {})"\
                .format(target_str, ",".join(args_str))
            new_pop_node = gast.parse(new_pop_str).body[0].value
            return new_pop_node
        else:
            return node
