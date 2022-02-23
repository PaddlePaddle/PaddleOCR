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

import six
import copy
from collections import defaultdict

# gast is a generic AST to represent Python2 and Python3's Abstract Syntax Tree(AST).
# It provides a compatibility layer between the AST of various Python versions,
# as produced by ast.parse from the standard ast module.
# See details in https://github.com/serge-sans-paille/gast/
from paddle.utils import gast
from paddle.fluid import unique_name

from paddle.fluid.dygraph.dygraph_to_static.utils import create_funcDef_node, ast_to_source_code
from paddle.fluid.dygraph.dygraph_to_static.utils import create_assign_node
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import StaticAnalysisVisitor
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper
from paddle.fluid.dygraph.dygraph_to_static.variable_trans_func import create_static_variable_gast_node

TRUE_FUNC_PREFIX = 'true_fn'
FALSE_FUNC_PREFIX = 'false_fn'


class IfElseTransformer(gast.NodeTransformer):
    """
    Transform if/else statement of Dygraph into Static Graph.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Type of input node should be AstNodeWrapper, but received %s ." % type(
            wrapper_root)
        self.root = wrapper_root.node
        self.static_analysis_visitor = StaticAnalysisVisitor(self.root)

    def transform(self):
        """
        Main function to transform AST.
        """
        self.visit(self.root)

    def visit_If(self, node):
        self.generic_visit(node)
        new_vars_stmts, true_func_node, false_func_node, return_name_ids = transform_if_else(
            node, self.root)

        new_node = create_convert_ifelse_node(return_name_ids, node.test,
                                              true_func_node, false_func_node)

        return new_vars_stmts + [true_func_node, false_func_node] + [new_node]

    def visit_Call(self, node):
        # Remove `numpy()` statement, like `Tensor.numpy()[i]` -> `Tensor[i]`
        if isinstance(node.func, gast.Attribute):
            attribute = node.func
            if attribute.attr == 'numpy':
                node = attribute.value
        self.generic_visit(node)
        return node

    def visit_IfExp(self, node):
        """
        Transformation with `true_fn(x) if Tensor > 0 else false_fn(x)`
        """
        self.generic_visit(node)

        new_node = create_convert_ifelse_node(None, node.test, node.body,
                                              node.orelse, True)
        # Note: A blank line will be added separately if transform gast.Expr
        # into source code. Using gast.Expr.value instead to avoid syntax error
        # in python.
        if isinstance(new_node, gast.Expr):
            new_node = new_node.value

        return new_node


class NameVisitor(gast.NodeVisitor):
    def __init__(self, after_node=None, end_node=None):
        # The start node (exclusive) of the visitor
        self.after_node = after_node
        # The terminate node of the visitor.
        self.end_node = end_node
        # Dict to store the names and ctxs of vars.
        self.name_ids = defaultdict(list)
        # List of current visited nodes
        self.ancestor_nodes = []
        # True when in range (after_node, end_node).
        self._in_range = after_node is None
        self._candidate_ctxs = (gast.Store, gast.Load, gast.Param)
        self._def_func_names = set()

    def visit(self, node):
        """Visit a node."""
        if self.after_node is not None and node == self.after_node:
            self._in_range = True
            return
        if node == self.end_node:
            self._in_range = False
            return

        self.ancestor_nodes.append(node)
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        ret = visitor(node)
        self.ancestor_nodes.pop()

        return ret

    def visit_If(self, node):
        """
        For nested `if/else`, the created vars are not always visible for parent node.
        In addition, the vars created in `if.body` are not visible for `if.orelse`.

        Case 1:
            x = 1
            if m > 1:
                res = new_tensor
            res = res + 1   # Error, `res` is not visible here.

        Case 2:
            if x_tensor > 0:
                res = new_tensor
            else:
                res = res + 1   # Error, `res` is not visible here.

        In above two cases, we should consider to manage the scope of vars to parsing
        the arguments and returned vars correctly.
        """
        if not self._in_range or not self.end_node:
            self.generic_visit(node)
            return
        else:
            before_if_name_ids = copy.deepcopy(self.name_ids)
            body_name_ids = self._visit_child(node.body)
            # If traversal process stops early in `if.body`, return the currently seen name_ids.
            if not self._in_range:
                self._update_name_ids(before_if_name_ids)
            else:
                else_name_ids = self._visit_child(node.orelse)
                # If traversal process stops early in `if.orelse`, return the currently seen name_ids.
                if not self._in_range:
                    self._update_name_ids(before_if_name_ids)
                else:
                    # Blocks the vars in `if.body` and only inserts the vars both created in 'if/else' branch
                    # into name_ids.
                    new_name_ids = self._find_new_name_ids(body_name_ids,
                                                           else_name_ids)
                    for new_name_id in new_name_ids:
                        before_if_name_ids[new_name_id].append(gast.Store())

                    self.name_ids = before_if_name_ids

    def visit_Attribute(self, node):
        if not self._in_range or not self._is_call_func_name_node(node):
            self.generic_visit(node)

    def visit_Name(self, node):
        if not self._in_range:
            self.generic_visit(node)
            return
        blacklist = {'True', 'False', 'None'}
        if node.id in blacklist: return
        if node.id in self._def_func_names:
            return
        if not self._is_call_func_name_node(node):
            if isinstance(node.ctx, self._candidate_ctxs):
                self.name_ids[node.id].append(node.ctx)

    def visit_Assign(self, node):
        if not self._in_range:
            self.generic_visit(node)
            return
        # Visit `value` firstly.
        node._fields = ('value', 'targets')
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not self._in_range:
            self.generic_visit(node)
            return
        self._def_func_names.add(node.name)
        if not self.end_node:
            self.generic_visit(node)
        else:
            before_name_ids = copy.deepcopy(self.name_ids)
            self.name_ids = defaultdict(list)
            self.generic_visit(node)

            if not self._in_range:
                self._update_name_ids(before_name_ids)
            else:
                self.name_ids = before_name_ids

    def _visit_child(self, node):
        self.name_ids = defaultdict(list)
        if isinstance(node, list):
            for item in node:
                if isinstance(item, gast.AST):
                    self.visit(item)
        elif isinstance(node, gast.AST):
            self.visit(node)

        return copy.deepcopy(self.name_ids)

    def _find_new_name_ids(self, body_name_ids, else_name_ids):
        def is_required_ctx(ctxs, required_ctx):
            for ctx in ctxs:
                if isinstance(ctx, required_ctx):
                    return True
            return False

        candidate_name_ids = set(body_name_ids.keys()) & set(else_name_ids.keys(
        ))
        store_ctx = gast.Store
        new_name_ids = set()
        for name_id in candidate_name_ids:
            if is_required_ctx(body_name_ids[name_id],
                               store_ctx) and is_required_ctx(
                                   else_name_ids[name_id], store_ctx):
                new_name_ids.add(name_id)

        return new_name_ids

    def _is_call_func_name_node(self, node):
        white_func_names = set(['append', 'extend'])
        if len(self.ancestor_nodes) > 1:
            assert self.ancestor_nodes[-1] == node
            parent_node = self.ancestor_nodes[-2]
            if isinstance(parent_node, gast.Call) and parent_node.func == node:
                # e.g: var_list.append(elem), var_list is also a name_id.
                should_skip = isinstance(
                    node, gast.Attribute) and node.attr in white_func_names
                if not should_skip:
                    return True
        return False

    def _update_name_ids(self, new_name_ids):
        for name_id, ctxs in six.iteritems(new_name_ids):
            self.name_ids[name_id] = ctxs + self.name_ids[name_id]


def get_name_ids(nodes, after_node=None, end_node=None):
    """
    Return all ast.Name.id of python variable in nodes range from
    (after_node, end_node) exclusively. If after_node or end_node is None, the
    range is unlimited.
    """
    name_visitor = NameVisitor(after_node, end_node)
    for node in nodes:
        name_visitor.visit(node)
    return name_visitor.name_ids


def parse_cond_args(parent_ids_dict,
                    var_ids_dict,
                    modified_ids_dict=None,
                    ctx=gast.Load):
    """
    Find out the ast.Name.id list of input by analyzing node's AST information.
    """

    # 1. filter the var fit the ctx
    arg_name_ids = [
        var_id for var_id, var_ctx in six.iteritems(var_ids_dict)
        if isinstance(var_ctx[0], ctx)
    ]

    # 2. args should contain modified var ids in if-body or else-body
    #  case:
    #
    #   ```
    #   if b < 1:
    #     z = y
    #   else:
    #     z = x
    #   ```
    #
    #   In the above case, `z` should be in the args of cond()
    if modified_ids_dict:
        arg_name_ids = set(arg_name_ids) | set(modified_ids_dict)

    # 3. args should not contain the vars not in parent ids
    #  case :
    #
    #   ```
    #   x = 1
    #   if x > y:
    #     z = [v for v in range(i)]
    #   ```
    #
    #   In the above case, `v` should not be in the args of cond()
    arg_name_ids = list(set(arg_name_ids) & set(parent_ids_dict))

    arg_name_ids.sort()
    args = [
        gast.Name(
            id=name_id, ctx=gast.Load(), annotation=None, type_comment=None)
        for name_id in arg_name_ids
    ]
    arguments = gast.arguments(
        args=args,
        posonlyargs=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=None,
        kwarg=None,
        defaults=[])

    return arguments


def parse_cond_return(parent_vars_dict, if_vars_dict, else_vars_dict,
                      after_ifelse_vars_dict):
    """
    Find out the ast.Name list of output by analyzing node's AST information.
    One of the following conditions should be satisfied while determining whether a variable is a return value:
    1. the var in parent scope is modified in If.body or If.orelse node.
    2. new var is both created in If.body and If.orelse node.
    3. new var is created only in one of If.body or If.orelse node, and it used as gast.Load firstly after gast.If node.

    For example:
        x, y = 5, 10
        if x > 4:
            x = x+1
            z = x*x
            q = 10
        else:
            y = y - 1
            z = y*y
            m = 20
            n = 20

        print(q)
        n = 30
        print(n)


    The return_ids are (x, y, z, q) for `If.body` and `If.orelse`node, because
    1. x is modified in If.body node,
    2. y is modified in If.body node,
    3. z is both created in If.body and If.orelse node,
    4. q is created only in If.body, and it is used by `print(q)` as gast.Load.
    Note:
        After transformed, q and z are created in parent scope. For example,

        x, y = 5, 10
        q = paddle.jit.dy2static.data_layer_not_check(name='q', shape=[-1], dtype='float32')
        z = paddle.jit.dy2static.data_layer_not_check(name='z', shape=[-1], dtype='float32')

        def true_func(x, y, q):
            x = x+1
            z = x*x
            q = 10
            return x,y,z,q

        def false_func(x, y, q):
            y = y - 1
            z = y*y
            m = 20
            n = 20
            return x,y,z,q

        x,y,z,q = fluid.layers.cond(x>4, lambda: true_func(x, y), lambda: false_func(x, y, q))

    m and n are not in return_ids, because
    5. m is created only in If.orelse, but it is not used after gast.If node.
    6. n is created only in If.orelse, and it is used by `n = 30` and `print(n)`, but it is not used as gast.Load firstly but gast.Store .

    """

    def _is_return_var(ctxs):
        for ctx in ctxs:
            if isinstance(ctx, (gast.Store, gast.Param)):
                return True
        return False

    def _vars_with_store(ids_dict):
        vars = []
        for k, ctxs in six.iteritems(ids_dict):
            if _is_return_var(ctxs):
                vars.append(k)
        return vars

    def _modified_vars(child_dict, parent_dict):
        return set([
            var for var in _vars_with_store(child_dict) if var in parent_dict
        ])

    def _vars_loaded(ids_dict):
        """
        gast.Param is also a kind of `load` semantic.
        """
        new_dict = defaultdict(list)
        for k, ctxs in six.iteritems(ids_dict):
            for ctx in ctxs:
                if isinstance(ctx, (gast.Load, gast.Param)):
                    new_dict[k].append(ctx)
        return new_dict

    # modified vars
    body_modified_vars = _modified_vars(if_vars_dict, parent_vars_dict)
    orelse_modified_vars = _modified_vars(else_vars_dict, parent_vars_dict)
    modified_vars = body_modified_vars | orelse_modified_vars

    # new vars
    body_new_vars = set([
        var for var in _vars_with_store(if_vars_dict)
        if var not in parent_vars_dict
    ])
    orelse_new_vars = set([
        var for var in _vars_with_store(else_vars_dict)
        if var not in parent_vars_dict
    ])
    new_vars_in_body_or_orelse = body_new_vars | orelse_new_vars
    new_vars_in_one_of_body_or_orelse = body_new_vars ^ orelse_new_vars

    # 1. the var in parent scope is modified in If.body or If.orelse node.
    modified_vars_from_parent = modified_vars - new_vars_in_body_or_orelse

    # 2. new var is both created in If.body and If.orelse node.
    new_vars_in_body_and_orelse = body_new_vars & orelse_new_vars

    # 3. new var is created only in one of If.body or If.orelse node, and it used as gast.Load firstly after gast.If node.
    # TODO(zhhsplendid): the _vars_loaded can be optimized as _vars_loaded_before_store. Because if a variable is stored before load,
    # the value would change by the store statement, we don't have to return to change the value. However, analysis is
    # complex because if the IfElse is nested and outer IfElse store statement may not run at all. We will put this optimization
    # as the future TODO
    used_vars_after_ifelse = set(
        [var for var in _vars_loaded(after_ifelse_vars_dict)])
    new_vars_to_create = new_vars_in_one_of_body_or_orelse & used_vars_after_ifelse | new_vars_in_body_and_orelse

    # 4. generate return_ids of if/else node.
    return_ids = list(modified_vars_from_parent | new_vars_in_body_and_orelse |
                      new_vars_to_create)
    return_ids.sort()

    return return_ids, modified_vars_from_parent, new_vars_to_create


def transform_if_else(node, root):
    """
    Transform ast.If into control flow statement of Paddle static graph.
    """
    # TODO(liym27): Consider variable like `self.a` modified in if/else node.
    parent_name_ids = get_name_ids([root], end_node=node)
    body_name_ids = get_name_ids(node.body)
    orelse_name_ids = get_name_ids(node.orelse)
    # Get after_ifelse_name_ids, which means used var names after If.body and If.orelse node.
    after_ifelse_name_ids = get_name_ids([root], after_node=node)

    return_name_ids, modified_name_ids_from_parent, new_vars_to_create = parse_cond_return(
        parent_name_ids, body_name_ids, orelse_name_ids, after_ifelse_name_ids)

    # NOTE: Python can create variable only in if body or only in else body, and use it out of if/else.
    # E.g.
    #
    # if x > 5:
    #   a = 10
    # print(a)
    #
    # Create static variable for those variables
    create_new_vars_in_parent_stmts = []
    for name in new_vars_to_create:
        # NOTE: Consider variable like `self.a` modified in if/else node.
        if "." not in name:
            create_new_vars_in_parent_stmts.append(
                create_static_variable_gast_node(name))

    modified_name_ids = modified_name_ids_from_parent | new_vars_to_create

    true_func_node = create_funcDef_node(
        node.body,
        name=unique_name.generate(TRUE_FUNC_PREFIX),
        input_args=parse_cond_args(parent_name_ids, body_name_ids,
                                   modified_name_ids),
        return_name_ids=return_name_ids)
    false_func_node = create_funcDef_node(
        node.orelse,
        name=unique_name.generate(FALSE_FUNC_PREFIX),
        input_args=parse_cond_args(parent_name_ids, orelse_name_ids,
                                   modified_name_ids),
        return_name_ids=return_name_ids)
    return create_new_vars_in_parent_stmts, true_func_node, false_func_node, return_name_ids


def create_convert_ifelse_node(return_name_ids,
                               pred,
                               true_func,
                               false_func,
                               is_if_expr=False):
    """
    Create `paddle.jit.dy2static.convert_ifelse(
            pred, true_fn, false_fn, true_args, false_args, return_vars)`
    to replace original `python if/else` statement.
    """

    def create_name_nodes(name_ids):
        if not name_ids:
            return gast.Tuple(elts=[], ctx=gast.Load())

        gast_names = [
            gast.Name(
                id=name_id, ctx=gast.Load(), annotation=None, type_comment=None)
            for name_id in name_ids
        ]
        name_node = gast.Tuple(elts=gast_names, ctx=gast.Load())
        return name_node

    if is_if_expr:
        true_args = gast.Tuple(elts=[], ctx=gast.Load())
        false_args = gast.Tuple(elts=[], ctx=gast.Load())
        true_func_source = "lambda : {}".format(ast_to_source_code(true_func))
        false_func_source = "lambda : {}".format(ast_to_source_code(false_func))
    else:
        true_args = gast.Tuple(elts=true_func.args.args, ctx=gast.Load())
        false_args = gast.Tuple(elts=false_func.args.args, ctx=gast.Load())
        true_func_source = true_func.name
        false_func_source = false_func.name

    return_vars = create_name_nodes(return_name_ids)

    convert_ifelse_layer = gast.parse(
        'paddle.jit.dy2static.convert_ifelse('
        '{pred}, {true_fn}, {false_fn}, {true_args}, {false_args}, {return_vars})'.
        format(
            pred=ast_to_source_code(pred),
            true_fn=true_func_source,
            false_fn=false_func_source,
            true_args=ast_to_source_code(true_args),
            false_args=ast_to_source_code(false_args),
            return_vars=ast_to_source_code(return_vars))).body[0].value

    if return_name_ids:
        _, cond_node = create_assign_node(return_name_ids, convert_ifelse_layer)
    else:  # No variables can be returned if no assign statement in if.body.
        cond_node = gast.Expr(value=convert_ifelse_layer)

    return cond_node
