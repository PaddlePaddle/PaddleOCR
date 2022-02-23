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

import copy
from paddle.utils import gast

from paddle.fluid import unique_name
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code
from paddle.fluid.dygraph.dygraph_to_static.utils import slice_is_num
from paddle.fluid.dygraph.dygraph_to_static.utils import is_paddle_api
from paddle.fluid.dygraph.dygraph_to_static.utils import SplitAssignTransformer
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import StaticAnalysisVisitor

STATIC_CONVERT_VAR_SHAPE_SUFFIX = '__static_convert_var_shape_suffix'


def create_convert_shape_node(var_shape_node,
                              slice_node=None,
                              in_control_flow=False):
    assert isinstance(var_shape_node, (gast.Attribute, gast.Subscript))

    if isinstance(var_shape_node, gast.Attribute):
        args = [ast_to_source_code(var_shape_node.value).strip()]
        # (1) A slice can be a simple number such as 1, -2, i.e. gast.Index or gast.Constant
        # (2) A slice can also be represented by bounds such as 2:-1, i.e. not gast.Index or gast.Constant
        # In (1) case, we pass the number as 'idx' argument in convert_var_shape
        # In (2) case, we have to make it like `convert_var_shape(x)[slice]`
        if slice_node is not None and slice_is_num(slice_node):
            args.append(ast_to_source_code(slice_node.slice).strip())

        convert_var_shape_func = "paddle.jit.dy2static.convert_var_shape({}, in_control_flow={})".format(
            ",".join(args), in_control_flow)
        api_shape_node = gast.parse(convert_var_shape_func).body[0].value

        if slice_node is not None and not slice_is_num(slice_node):
            return gast.Subscript(
                value=api_shape_node, slice=slice_node.slice, ctx=gast.Load())
        return api_shape_node

    if isinstance(var_shape_node, gast.Subscript):
        result_node = copy.deepcopy(var_shape_node)
        result_node = create_convert_shape_node(result_node.value, result_node,
                                                in_control_flow)
        return result_node


def create_choose_shape_node(attr_shape_name, api_shape_name, slice_node=None):
    eval_exist_func = "paddle.jit.dy2static.eval_if_exist_else_none('{}', globals())".format(
        api_shape_name)
    args = [attr_shape_name, eval_exist_func]

    if slice_node is not None and slice_is_num(slice_node):
        args.append(ast_to_source_code(slice_node.slice).strip())
    choose_shape_func = "paddle.jit.dy2static.choose_shape_attr_or_api({})".format(
        ",".join(args))
    choose_shape_node = gast.parse(choose_shape_func).body[0].value
    if slice_node is not None and not slice_is_num(slice_node):
        return gast.Subscript(
            value=choose_shape_node, slice=slice_node.slice, ctx=gast.Load())
    return choose_shape_node


class ShapeAttributeTransformer(gast.NodeTransformer):
    """
    Input a node like `x.shape` or `x[4].shape[0]` (self._is_var_shape(node) is True),
    return a new node changes input to static shape API like `convert_var_shape(x)`,
    `convert_var_shape(x[4])[0]`.
    """

    def visit_Attribute(self, node):
        if node.attr == 'shape':
            args = ast_to_source_code(node.value).strip()
            convert_var_shape_func = "paddle.jit.dy2static.convert_var_shape_simple({})".format(
                args)
            api_shape_node = gast.parse(convert_var_shape_func).body[0].value
            return api_shape_node
        return node


class TensorShapeTransformer(gast.NodeTransformer):
    """
    This class transforms variable.shape used in Paddle Apis or control flow conditions into Static Graph Ast.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of TensorShapeTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node
        # stores origin var string name (like "x" in `x = t.shape`) to
        # static shape var string name (like "x_SUFFIX" in `x_SUFFIX = shape(t)`)
        self.name_to_var_shape = {}

        self.static_analysis_visitor = StaticAnalysisVisitor(self.root)
        self.node_to_wrapper_map = self.static_analysis_visitor.get_node_to_wrapper_map(
        )
        var_env = self.static_analysis_visitor.get_var_env()
        var_env.cur_scope = var_env.cur_scope.sub_scopes[0]
        self.scope_var_type_dict = var_env.get_scope_var_type()

    def transform(self):
        SplitAssignTransformer(self.root).transform()
        self.visit(self.root)

    def visit_Assign(self, node):
        update_static_shape_var_node = self._update_name_to_var_shape(node)
        if update_static_shape_var_node is not None:
            ret = [node]
            ret.extend(update_static_shape_var_node)
            return ret
        self.generic_visit(node)
        return node

    def visit_Subscript(self, node):
        value_node = node.value
        slice_node = node.slice
        if isinstance(value_node, gast.Name):
            if value_node.id in self.name_to_var_shape and self._used_by_paddle_api(
                    value_node):
                return create_choose_shape_node(
                    value_node.id, self.name_to_var_shape[value_node.id], node)
        elif isinstance(value_node, gast.Attribute):
            if self._used_by_paddle_api(value_node):
                value_name = ast_to_source_code(value_node).strip()
                if value_name in self.name_to_var_shape:
                    return create_choose_shape_node(
                        value_name, self.name_to_var_shape[value_name], node)
                if self._is_var_shape(value_node):
                    return create_convert_shape_node(value_node, node)
        return node

    def visit_Attribute(self, node):
        if self._used_by_paddle_api(node):
            name = ast_to_source_code(node).strip()
            if name in self.name_to_var_shape:
                return create_choose_shape_node(name,
                                                self.name_to_var_shape[name])
            if self._is_var_shape(node):
                return create_convert_shape_node(node)
        return node

    def visit_Name(self, node):
        if node.id in self.name_to_var_shape:
            if self._used_by_paddle_api(node):
                return create_choose_shape_node(node.id,
                                                self.name_to_var_shape[node.id])
        return node

    def visit_Call(self, node):
        if is_paddle_api(node):
            # Visit gast.Attribute and gast.Name to replace var.shape if necessary.
            self.generic_visit(node)
        # Don't have to visit other APIs
        return node

    def visit_If(self, node):
        # Call generic_visit first to transform var.shape that is used in Paddle Api.
        self.generic_visit(node)
        cond = node.test
        self._transform_var_shape_if_necessary(cond)

        return node

    def visit_While(self, node):
        self.generic_visit(node)
        cond = node.test
        self._transform_var_shape_if_necessary(cond)
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        iter = node.iter
        self._transform_var_shape_if_necessary(iter)

        # If var.shape is a gast.Name and it is used in range function, transform it
        self._transform_var_shape_in_range(node)
        return node

    def _transform_var_shape_in_range(self, node):
        assert isinstance(node, gast.For)
        if not isinstance(node.iter, gast.Call):
            return False
        if not isinstance(node.iter.func, gast.Name):
            return False
        if node.iter.func.id != "range":
            return False
        args = node.iter.args
        for idx, arg in enumerate(args):
            if isinstance(arg, gast.Name) and arg.id in self.name_to_var_shape:
                args[idx] = create_choose_shape_node(
                    arg.id, self.name_to_var_shape[arg.id])
        return True

    def _transform_var_shape_if_necessary(self, cond):
        need_transformed = False
        for child_node in gast.walk(cond):
            var_shape_node = None
            if isinstance(child_node,
                          (gast.Name, gast.Attribute, gast.Subscript)):
                child_name = ast_to_source_code(child_node).strip()
                if child_name in self.name_to_var_shape:
                    var_shape_node = create_choose_shape_node(
                        child_name, self.name_to_var_shape[child_name])
                elif self._is_var_shape(child_node):
                    var_shape_node = child_node

            if var_shape_node:
                need_transformed = True
                wrapper_node = self.node_to_wrapper_map.get(child_node)
                parent_node = wrapper_node.parent.node
                for field, value in gast.iter_fields(parent_node):
                    if child_node is value:
                        if var_shape_node is child_node:
                            setattr(parent_node, field,
                                    create_convert_shape_node(var_shape_node,
                                                              None, True))
                        else:
                            setattr(parent_node, field, var_shape_node)
                        break
                    # Some child_node may be in a list such as gast.Compare
                    if isinstance(value, list):
                        has_converted_shape = False
                        for i, v in enumerate(value):
                            if child_node is v:
                                if var_shape_node is child_node:
                                    value[i] = create_convert_shape_node(
                                        var_shape_node, None, True)
                                else:
                                    value[i] = var_shape_node
                                has_converted_shape = True
                                break
                        if has_converted_shape:
                            break
        return need_transformed

    def _used_by_paddle_api(self, node):
        """
        Whether node is used in paddle api as arguments.
        For example:
            1) Return True in `paddle.relu(x)` where node is `x` (gast.Name)
            2) Return True in `paddle.add(self.x)` where node is `self.x` (gast.Attribute)
            3) Return False in `paddle.add(self.x)` where node is `paddle.add` (gast.Attribute),
               because the role of node is not arguments but `gast.Call.func`.
        """
        assert isinstance(node, (gast.Attribute, gast.Name))
        wrapper_node = self.node_to_wrapper_map.get(node)
        if not wrapper_node:
            # Transformed node is not in node_to_wrapper_map
            return False
        while wrapper_node.parent:
            parent_node = wrapper_node.parent.node
            if isinstance(parent_node, gast.Call):
                # Note(Aurelius84): Filter the case when the role of node is `gast.Call.func`.
                if is_paddle_api(parent_node) and parent_node.func != node:
                    return True
                else:
                    return False
            wrapper_node = wrapper_node.parent

        return False

    def _is_var_shape(self, node):
        """
        Return True if node is like `x.shape` or `x.shape[0]`, return False otherwise.
        """
        if not isinstance(node, (gast.Attribute, gast.Subscript)):
            return False

        if isinstance(node, gast.Attribute):
            # If node is `paddle.shape`, return False
            if (node.attr == 'shape' and isinstance(node.value, gast.Name) and
                    node.value.id == 'paddle'):
                return False
            if node.attr != 'shape':
                return False
            return True

        if isinstance(node, gast.Subscript):
            value_node = node.value
            return self._is_var_shape(value_node)

        return False

    def _update_name_to_var_shape(self, node):
        def replace_dot(name):
            # replace all '.' into '_'
            return name.replace('.', '_')

        assert isinstance(node, gast.Assign)
        target_node = node.targets[0]
        value_node = node.value

        update_static_shape_var_node = None
        if isinstance(target_node, gast.Tuple):
            update_static_shape_var_node = []
            for idx, element in enumerate(target_node.elts):
                target_id = ast_to_source_code(element).strip()

                if isinstance(value_node, gast.Name):
                    if value_node.id in self.name_to_var_shape:
                        # TODO(zhhsplendid): is context a problem for the result node of gast.parse?
                        static_shape_var_name = unique_name.generate(
                            replace_dot(target_id) +
                            STATIC_CONVERT_VAR_SHAPE_SUFFIX)
                        static_shape_var_node = gast.parse(
                            static_shape_var_name).body[0].value

                        static_shape_value_name = self.name_to_var_shape[
                            value_node.id]

                        sub_node_str = "{}[{}]".format(static_shape_value_name,
                                                       idx)
                        sub_node = gast.parse(sub_node_str).body[0].value

                        update_static_shape_var_node.append(
                            gast.Assign(
                                targets=[static_shape_var_node],
                                value=sub_node))

                        self.name_to_var_shape[
                            target_id] = static_shape_var_name
                if isinstance(value_node, gast.Attribute):
                    if self._is_var_shape(value_node):  # eg: x.shape
                        static_shape_var_name = unique_name.generate(
                            replace_dot(target_id) +
                            STATIC_CONVERT_VAR_SHAPE_SUFFIX)
                        static_shape_var_node = gast.parse(
                            static_shape_var_name).body[0].value

                        static_shape_value_node = copy.deepcopy(value_node)
                        # x.shape becomes convert_var_shape_simple(x)
                        static_shape_value_node = ShapeAttributeTransformer(
                        ).visit(static_shape_value_node)

                        sub_node_str = "{}[{}]".format(
                            ast_to_source_code(static_shape_value_node).strip(),
                            idx)
                        sub_node = gast.parse(sub_node_str).body[0].value
                        # Note(Aurelius84): Becuase static_shape_var_name is used in
                        # eval_if_exist_else_none() as plain string, so it will not 
                        # be pasred as argument in convert_loop/ifelse. We delcare it
                        # as global var because it has unique name.
                        update_static_shape_var_node.append(
                            gast.Global(names=[static_shape_var_name]))

                        update_static_shape_var_node.append(
                            gast.Assign(
                                targets=[static_shape_var_node],
                                value=sub_node))
                        self.name_to_var_shape[
                            target_id] = static_shape_var_name
            return update_static_shape_var_node
        else:
            target_id = ast_to_source_code(target_node).strip()
            if isinstance(value_node, gast.Name):
                if value_node.id in self.name_to_var_shape:
                    static_shape_var_name = unique_name.generate(
                        replace_dot(target_id) +
                        STATIC_CONVERT_VAR_SHAPE_SUFFIX)
                    static_shape_var_node = gast.parse(
                        static_shape_var_name).body[0].value
                    static_shape_value_name = self.name_to_var_shape[
                        value_node.id]
                    static_shape_value_node = gast.parse(
                        static_shape_value_name).body[0].value

                    update_static_shape_var_node = [
                        gast.Assign(
                            targets=[static_shape_var_node],
                            value=static_shape_value_node)
                    ]
                    self.name_to_var_shape[target_id] = static_shape_var_name
            elif self._is_var_shape(value_node):  # eg: x.shape or x.shape[0]
                static_shape_var_name = unique_name.generate(
                    replace_dot(target_id) + STATIC_CONVERT_VAR_SHAPE_SUFFIX)
                static_shape_var_node = gast.parse(static_shape_var_name).body[
                    0].value
                static_shape_value_node = copy.deepcopy(value_node)
                # x.shape becomes convert_var_shape_simple(x)
                static_shape_value_node = ShapeAttributeTransformer().visit(
                    static_shape_value_node)
                # Declare static_shape_var_name as global var
                update_static_shape_var_node = [
                    gast.Global(names=[static_shape_var_name])
                ]
                update_static_shape_var_node.append(
                    gast.Assign(
                        targets=[static_shape_var_node],
                        value=static_shape_value_node))
                self.name_to_var_shape[target_id] = static_shape_var_name
        return update_static_shape_var_node
