#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from .utils import is_paddle_api, is_dygraph_api, is_numpy_api, index_in_list

__all__ = ['AstNodeWrapper', 'NodeVarType', 'StaticAnalysisVisitor']


class NodeVarType(object):
    """
    Enum class of python variable types. We have to know some variable types
    during compile time to transfer AST. For example, a string variable and a
    tensor variable in if clause may lead to different conversion from dygraph
    to static graph.
    """
    ERROR = -1  # Returns when static analysis gets error
    UNKNOWN = 0  # Reserve for AST nodes have not known the type
    STATEMENT = 1  # For nodes representing statement (non-variable type)
    CALLABLE = 2

    # python data types
    NONE = 100
    BOOLEAN = 101
    INT = 102
    FLOAT = 103
    STRING = 104
    TENSOR = 105
    NUMPY_NDARRAY = 106

    # python collections
    LIST = 200
    SET = 201
    DICT = 202

    PADDLE_DYGRAPH_API = 300
    PADDLE_CONTROL_IF = 301
    PADDLE_CONTROL_WHILE = 302
    PADDLE_CONTROL_FOR = 303
    # Paddle API may not be visible to get source code.
    # We use this enum value to denote the type return by a Paddle API
    PADDLE_RETURN_TYPES = 304

    # If node.node_var_type in TENSOR_TYPES, it can be considered as tensor-dependent.
    TENSOR_TYPES = {TENSOR, PADDLE_RETURN_TYPES}

    @staticmethod
    def binary_op_output_type(in_type1, in_type2):
        if in_type1 == in_type2:
            return in_type1

        if in_type1 == NodeVarType.UNKNOWN:
            return in_type2
        if in_type2 == NodeVarType.UNKNOWN:
            return in_type1

        supported_types = [
            NodeVarType.BOOLEAN, NodeVarType.INT, NodeVarType.FLOAT,
            NodeVarType.NUMPY_NDARRAY, NodeVarType.TENSOR,
            NodeVarType.PADDLE_RETURN_TYPES
        ]

        if in_type1 not in supported_types:
            return NodeVarType.UNKNOWN
        if in_type2 not in supported_types:
            return NodeVarType.UNKNOWN

        forbidden_types = [NodeVarType.NUMPY_NDARRAY, NodeVarType.TENSOR]
        if in_type1 in forbidden_types and in_type2 in forbidden_types:
            return NodeVarType.UNKNOWN
        return max(in_type1, in_type2)


class AstNodeWrapper(object):
    """
    Wrapper for python gast.node. We need a node wrapper because gast.node
    doesn't store all required information when we are transforming AST.
    We should collect additional information which the actual transformation
    needs.
    """

    def __init__(self, node):
        self.node = node
        self.parent = None
        self.children = []
        self.node_var_type = {NodeVarType.UNKNOWN}


class AstVarScope(object):
    """
    AstVarScope is a class holding the map from current scope variable to its
    type.
    """
    SCOPE_TYPE_SCRIPT = 0
    SCOPE_TYPE_FUNCTION = 1
    SCOPE_TYPE_CLASS = 2

    def __init__(self,
                 scope_name='',
                 scope_type=SCOPE_TYPE_SCRIPT,
                 parent_scope=None):
        self.sub_scopes = []
        self.name_to_id = {}
        self.id_to_type = {}
        self.cur_id = 0

        self.scope_name = scope_name
        self.scope_type = scope_type
        self.parent_scope = parent_scope
        if parent_scope is not None:
            parent_scope.sub_scopes.append(self)

    def add_var_type(self, var_name, node_var_type):
        var_type = self.get_var_type(var_name)
        if var_type == {NodeVarType.UNKNOWN}:
            self.set_var_type(var_name, node_var_type)
        else:
            if isinstance(node_var_type, set):
                var_type.update(node_var_type)
            else:
                var_type.add(node_var_type)

    def set_var_type(self, var_name, node_var_type):
        if var_name in self.name_to_id:
            num_id = self.name_to_id[var_name]
        else:
            num_id = self.cur_id
            self.cur_id += 1
            self.name_to_id[var_name] = num_id
        self.id_to_type[num_id] = node_var_type if isinstance(
            node_var_type, set) else {node_var_type}

    def get_var_type(self, var_name):
        if var_name in self.name_to_id:
            num_id = self.name_to_id[var_name]
            return self.id_to_type[num_id]
        if self.parent_scope is None:
            return {NodeVarType.UNKNOWN}
        return self.parent_scope.get_var_type(var_name)


class AstVarEnv(object):
    """
    A class maintains scopes and mapping from name strings to type.
    """

    def __init__(self):
        self.cur_scope = AstVarScope()

    def enter_scope(self, scope_name, scope_type):
        self.cur_scope = AstVarScope(
            scope_name, scope_type, parent_scope=self.cur_scope)
        return self.cur_scope

    def exit_scope(self):
        assert self.cur_scope.parent_scope is not None, "Call exit_scope in "\
            "AstVarEnv when current scope doesn't have parent scope."
        self.cur_scope = self.cur_scope.parent_scope
        return self.cur_scope

    def get_parent_scope(self):
        assert self.cur_scope.parent_scope is not None, "Call parent_scope in "\
            "AstVarEnv when current scope doesn't have parent scope."
        return self.cur_scope.parent_scope

    def add_var_type(self, var_name, node_var_type):
        self.cur_scope.add_var_type(var_name, node_var_type)

    def set_var_type(self, var_name, node_var_type):
        self.cur_scope.set_var_type(var_name, node_var_type)

    def get_var_type(self, var_name):
        return self.cur_scope.get_var_type(var_name)

    def get_scope_var_type(self):
        '''
        Returns a dict mapping from variable name to type. Used for debug and
        test.
        '''
        cur_scope_dict = {}
        for name in self.cur_scope.name_to_id:
            node_var_type = self.cur_scope.get_var_type(name)
            cur_scope_dict[name] = node_var_type
        return cur_scope_dict


class StaticAnalysisVisitor(object):
    """
    A class that does static analysis
    """

    def __init__(self, ast_root=None):
        if ast_root is not None:
            self.run(ast_root)

    def run(self, ast_root):
        self.node_wrapper_root = None
        self.ancestor_wrappers = []
        self.node_to_wrapper_map = {}
        self.var_env = AstVarEnv()

        self.dfs_visit(ast_root)

    def dfs_visit(self, node):
        # AST reuses some gast.nodes, such as Param node of expr_context
        if node not in self.node_to_wrapper_map:
            cur_wrapper = AstNodeWrapper(node)
            self.node_to_wrapper_map[node] = cur_wrapper
        else:
            cur_wrapper = self.node_to_wrapper_map[node]

        if self.node_wrapper_root is None:
            self.node_wrapper_root = cur_wrapper

        if len(self.ancestor_wrappers) != 0:
            last_wrapper = self.ancestor_wrappers[-1]
            last_wrapper.children.append(cur_wrapper)
            cur_wrapper.parent = last_wrapper

        self.ancestor_wrappers.append(cur_wrapper)
        for child in gast.iter_child_nodes(node):
            if isinstance(child, gast.FunctionDef) or isinstance(
                    child, gast.AsyncFunctionDef):
                # TODO: current version is function name mapping to its type
                # consider complex case involving parameters
                self.var_env.enter_scope(child.name,
                                         AstVarScope.SCOPE_TYPE_FUNCTION)
                func_type = self.dfs_visit(child)
                self.var_env.exit_scope()
            else:
                self.dfs_visit(child)
        self.ancestor_wrappers.pop()

        cur_wrapper.node_var_type = self._get_node_var_type(cur_wrapper)
        return cur_wrapper.node_var_type

    def get_node_wrapper_root(self):
        return self.node_wrapper_root

    def get_node_to_wrapper_map(self):
        return self.node_to_wrapper_map

    def get_var_env(self):
        return self.var_env

    def is_tensor_node(self, node):
        tensor_types = {NodeVarType.TENSOR, NodeVarType.PADDLE_RETURN_TYPES}
        node_wrapper = self.node_to_wrapper_map.get(node, None)
        if node_wrapper is None:
            return False
        if node_wrapper.node_var_type & tensor_types:
            return True

    def _get_constant_node_type(self, node):
        assert isinstance(node, gast.Constant), \
            "Type of input node should be gast.Constant, but received %s" % type(node)
        # singleton: None, True or False
        if node.value is None:
            return {NodeVarType.NONE}
        if isinstance(node.value, bool):
            return {NodeVarType.BOOLEAN}
        if isinstance(node.value, int):
            return {NodeVarType.INT}
        if isinstance(node.value, float):
            return {NodeVarType.FLOAT}
        if isinstance(node.value, str):
            return {NodeVarType.STRING}

        return {NodeVarType.UNKNOWN}

    def _get_node_var_type(self, cur_wrapper):
        node = cur_wrapper.node
        if isinstance(node, gast.Constant):
            return self._get_constant_node_type(node)

        if isinstance(node, gast.BoolOp):
            return {NodeVarType.BOOLEAN}
        if isinstance(node, gast.Compare):
            return {NodeVarType.BOOLEAN}

        if isinstance(node, gast.Dict):
            return {NodeVarType.DICT}
        if isinstance(node, gast.Set):
            return {NodeVarType.SET}

        if isinstance(node, gast.UnaryOp):
            return self.node_to_wrapper_map[node.operand].node_var_type

        if isinstance(node, gast.BinOp):
            left_type = self.node_to_wrapper_map[node.left].node_var_type
            right_type = self.node_to_wrapper_map[node.right].node_var_type
            result_type = set()
            for l in left_type:
                for r in right_type:
                    result_type.add(NodeVarType.binary_op_output_type(l, r))
            return result_type

        if isinstance(node, gast.Assign):
            ret_type = self.node_to_wrapper_map[node.value].node_var_type
            for target in node.targets:
                if isinstance(target, gast.Name):
                    self.node_to_wrapper_map[target].node_var_type = ret_type
                    self.var_env.set_var_type(target.id, ret_type)
            return ret_type

        if isinstance(node, gast.Name):
            if node.id == "None":
                return {NodeVarType.NONE}
            if node.id in {"True", "False"}:
                return {NodeVarType.BOOLEAN}
            # If node is child of functionDef.arguments
            parent_node_wrapper = cur_wrapper.parent
            if parent_node_wrapper and isinstance(parent_node_wrapper.node,
                                                  gast.arguments):
                parent_node = parent_node_wrapper.node
                var_type = {NodeVarType.UNKNOWN}
                if parent_node.defaults:
                    index = index_in_list(parent_node.args, node)
                    args_len = len(parent_node.args)
                    if index != -1 and args_len - index <= len(
                            parent_node.defaults):
                        defaults_node = parent_node.defaults[index - args_len]
                        if isinstance(defaults_node, gast.Constant):
                            var_type = self._get_constant_node_type(
                                defaults_node)

                            # Add node with identified type into cur_env.
                            self.var_env.set_var_type(node.id, var_type)
                return var_type

            return self.var_env.get_var_type(node.id)

        if isinstance(node, gast.Return):
            # If return nothing:
            if node.value is None:
                return {NodeVarType.NONE}

            return_type = self.node_to_wrapper_map[node.value].node_var_type
            assert self.var_env.cur_scope.scope_type == AstVarScope.SCOPE_TYPE_FUNCTION, "Return at non-function scope"
            func_name = self.var_env.cur_scope.scope_name
            parent_scope = self.var_env.get_parent_scope()
            parent_scope.add_var_type(func_name, return_type)
            return return_type

        if isinstance(node, gast.Call):
            if is_dygraph_api(node):
                if isinstance(node.func, gast.Attribute):
                    if node.func.attr == "to_variable":
                        return {NodeVarType.TENSOR}
            if is_paddle_api(node):
                return {NodeVarType.PADDLE_RETURN_TYPES}
            if is_numpy_api(node):
                # In this simple version we assume numpy api returns nd-array
                return {NodeVarType.NUMPY_NDARRAY}

            if isinstance(node.func, gast.Name):
                return self.var_env.get_var_type(node.func.id)
        if isinstance(node, gast.Subscript):
            if self.is_tensor_node(node.value):
                return {NodeVarType.TENSOR}

        return {NodeVarType.STATEMENT}
