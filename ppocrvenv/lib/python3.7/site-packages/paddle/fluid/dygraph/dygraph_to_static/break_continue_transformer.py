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

from paddle.fluid import unique_name
from paddle.fluid.dygraph.dygraph_to_static.utils import index_in_list
from paddle.fluid.dygraph.dygraph_to_static.utils import ForNodeVisitor
from paddle.fluid.dygraph.dygraph_to_static.utils import BaseNodeVisitor
from paddle.fluid.dygraph.dygraph_to_static.variable_trans_func import create_fill_constant_node

__all__ = ['BreakContinueTransformer']

BREAK_NAME_PREFIX = '__break'
CONTINUE_NAME_PREFIX = '__continue'


class ForToWhileTransformer(gast.NodeTransformer):
    """
    Transform python for loop into while loop and add condition node in the
    loop test
    """

    def __init__(self, parent_node, loop_node, condition_node):
        assert isinstance(
            loop_node,
            gast.For), "loop_node is not gast.For in ForToWhileTransformer"
        self.parent_node = parent_node
        self.loop_node = loop_node
        self.condition_node = condition_node

    def transform(self):
        if hasattr(self.parent_node, 'body'):
            body_list = self.parent_node.body
            i = index_in_list(body_list, self.loop_node)
            if i != -1:
                new_stmts = self.get_for_stmt_nodes(body_list[i])
                body_list[i:i + 1] = new_stmts
                i += len(new_stmts)
                return new_stmts
        if hasattr(self.parent_node, 'orelse'):
            body_list = self.parent_node.orelse
            i = index_in_list(body_list, self.loop_node)
            if i != -1:
                new_stmts = self.get_for_stmt_nodes(body_list[i])
                body_list[i:i + 1] = new_stmts
                i += len(new_stmts)
                return new_stmts
        raise ValueError(
            "parent_node doesn't contain the loop_node in ForToWhileTransformer")

    def get_for_stmt_nodes(self, node):
        assert isinstance(
            node, gast.For), "Input node is NOT gast.For in get_for_stmt_nodes"

        # 1. parse current gast.For node
        current_for_node_parser = ForNodeVisitor(node)
        stmts_tuple = current_for_node_parser.parse()
        if stmts_tuple is None:
            return [node]
        init_stmts, cond_stmt, body_stmts = stmts_tuple

        # 2. append break statement
        new_cond_stmt = gast.BoolOp(
            op=gast.And(), values=[cond_stmt, self.condition_node])

        # 3. construct gast.While node
        while_node = gast.While(
            test=new_cond_stmt, body=body_stmts, orelse=node.orelse)
        init_stmts.append(while_node)
        return init_stmts


class BreakContinueTransformer(BaseNodeVisitor):
    """
    Rewrite 'break' and 'continue' key words in a if-else python way to make
    it equivalent to original control flow
    
    The main idea of this class is:

        1. Map the 'break/continue' stmt with an unique boolean variable V.

        2. Find the first ancestor block containing this 'break/continue', a
        block can be a node containing stmt list. We should remove all stmts
        after the 'break/continue' and set the V to True here.

        3. Add 'if V' for stmts in ancestor blocks between the first one
        (exclusive) and the ancestor loop (inclusive)

        4. For 'break' add break into condition of the loop. For 'continue',
        set continue to False at the beginning of each loop

        TODO: more details should be summarized as design document

    Note: The class is inherited from BaseNodeVisitor instead of NodeTransformer,
          because ancestor nodes will be modified inplace for `Break/Continue` here.
          In general, we recommend to inheriting NodeTransformer to modify node!
    """

    def __init__(self, wrapper_root):
        super(BreakContinueTransformer, self).__init__()

        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node

    def transform(self):
        self.visit(self.root)

    def visit_Break(self, node):
        loop_node_index = _find_ancestor_loop_index(node, self.ancestor_nodes)
        assert loop_node_index != -1, "SyntaxError: 'break' outside loop"
        loop_node = self.ancestor_nodes[loop_node_index]

        # 1. Map the 'break/continue' stmt with an unique boolean variable V.
        variable_name = unique_name.generate(BREAK_NAME_PREFIX)

        # 2. Find the first ancestor block containing this 'break/continue', a
        # block can be a node containing stmt list. We should remove all stmts
        # after the 'break/continue' and set the V to True here.
        first_block_index = self._remove_stmts_after_break_continue(
            node, variable_name, loop_node_index)

        # 3. Add 'if not V' for stmts in ancestor blocks between the first one
        # (exclusive) and the ancestor loop (inclusive)
        self._replace_if_stmt(loop_node_index, first_block_index, variable_name)

        # 4. For 'break' add break into condition of the loop.
        assign_false_node = create_fill_constant_node(variable_name, False)
        self._add_stmt_before_cur_node(loop_node_index, assign_false_node)

        cond_var_node = gast.UnaryOp(
            op=gast.Not(),
            operand=gast.Name(
                id=variable_name,
                ctx=gast.Load(),
                annotation=None,
                type_comment=None))

        if isinstance(loop_node, gast.While):
            loop_node.test = gast.BoolOp(
                op=gast.And(), values=[loop_node.test, cond_var_node])
        elif isinstance(loop_node, gast.For):
            parent_node = self.ancestor_nodes[loop_node_index - 1]
            for_to_while = ForToWhileTransformer(parent_node, loop_node,
                                                 cond_var_node)
            for_to_while.transform()

    def visit_Continue(self, node):
        loop_node_index = _find_ancestor_loop_index(node, self.ancestor_nodes)
        assert loop_node_index != -1, "SyntaxError: 'continue' outside loop"
        loop_node = self.ancestor_nodes[loop_node_index]

        # 1. Map the 'break/continue' stmt with an unique boolean variable V.
        variable_name = unique_name.generate(CONTINUE_NAME_PREFIX)

        # 2. Find the first ancestor block containing this 'break/continue', a
        # block can be a node containing stmt list. We should remove all stmts
        # after the 'break/continue' and set the V to True here.
        first_block_index = self._remove_stmts_after_break_continue(
            node, variable_name, loop_node_index)

        # 3. Add 'if not V' for stmts in ancestor blocks between the first one
        # (exclusive) and the ancestor loop (inclusive)
        self._replace_if_stmt(loop_node_index, first_block_index, variable_name)

        # 4. For 'continue', set continue to False at the beginning of each loop
        assign_false_node = create_fill_constant_node(variable_name, False)
        loop_node.body.insert(0, assign_false_node)

    def _remove_stmts_after_break_continue(
            self, break_continue_node, break_continue_name, loop_node_index):
        for first_block_index in range(
                len(self.ancestor_nodes) - 1, loop_node_index - 1, -1):
            first_block = self.ancestor_nodes[first_block_index]
            if hasattr(first_block,
                       "body") and self._replace_break_continue_in_stmt_list(
                           first_block.body, break_continue_node,
                           break_continue_name):
                return first_block_index

            if hasattr(first_block,
                       "orelse") and self._replace_break_continue_in_stmt_list(
                           first_block.orelse, break_continue_node,
                           break_continue_name):
                return first_block_index

        return first_block_index

    def _replace_if_stmt(self, loop_node_index, first_block_index,
                         break_continue_name):
        for i in range(first_block_index - 1, loop_node_index - 1, -1):
            cur_node = self.ancestor_nodes[i]
            son_node = self.ancestor_nodes[i + 1]
            if hasattr(cur_node,
                       'body') and self._replace_after_node_to_if_in_stmt_list(
                           cur_node.body, son_node, break_continue_name):
                continue
            if hasattr(
                    cur_node,
                    'orelse') and self._replace_after_node_to_if_in_stmt_list(
                        cur_node.orelse, son_node, break_continue_name):
                continue

    def _replace_break_continue_in_stmt_list(
            self, stmt_list, break_continue_node, break_continue_name):
        i = index_in_list(stmt_list, break_continue_node)
        if i == -1:
            return False
        assign_true_node = create_fill_constant_node(break_continue_name, True)
        stmt_list[i:] = [assign_true_node]
        return True

    def _replace_after_node_to_if_in_stmt_list(self, stmt_list, node,
                                               break_continue_name):
        i = index_in_list(stmt_list, node)
        if i == -1:
            return False

        if i == len(stmt_list) - 1:
            # No need to add, we consider this as added successfully
            return True

        if_stmt = gast.If(test=gast.UnaryOp(
            op=gast.Not(),
            operand=gast.Name(
                id=break_continue_name,
                ctx=gast.Store(),
                annotation=None,
                type_comment=None)),
                          body=stmt_list[i + 1:],
                          orelse=[])
        stmt_list[i + 1:] = []
        stmt_list.append(if_stmt)
        return True

    def _add_stmt_before_cur_node(self, cur_node_index, stmt_node):
        cur_node = self.ancestor_nodes[cur_node_index]
        parent_node = self.ancestor_nodes[cur_node_index - 1]
        if hasattr(parent_node,
                   "body") and self._add_stmt_into_list_before_node(
                       parent_node.body, cur_node, stmt_node):
            return True
        if hasattr(parent_node,
                   "orelse") and self._add_stmt_into_list_before_node(
                       parent_node.orelse, cur_node, stmt_node):
            return True
        return False

    def _add_stmt_into_list_before_node(self, stmt_list, node, stmt_node):
        i = index_in_list(stmt_list, node)
        if i == -1:
            return False
        stmt_list.insert(i, stmt_node)
        return True


def _find_ancestor_loop_index(node, ancestor_nodes):
    for i in range(len(ancestor_nodes) - 1, -1, -1):
        if isinstance(ancestor_nodes[i], (gast.For, gast.While)):
            return i
    return -1


class BreakTransformOptimizer(BaseNodeVisitor):
    """
    In specific pattern, the transformed code could be optimized by joining the 
    If.test with while.test. 
    
    Currently supported pattern is:
    ```
        while cond1:            while cond1 and not cond2:
            if cond2:    --->       do_something()
                break
            do_something()
    ```
    
    See following example:

    >>> def foo(x):
    ...     i = paddle.to_tensor(1, dtype='int32')
    ...     while i < 10:
    ...         if x.mean() > 5:
    ...             break
    ...         x += i
    ...         i += 1
    ...     return x

    The generated code after applying optimization will be:
    ```
        def foo(x):
            i = paddle.to_tensor(1, dtype='int32')
            while i < 10 and not x.mean() > 5:
                x += i
                i += 1
            return x
    ```
    It can avoid wrapping all ops after `break` statement into `cond_op` that 
    usually brings very heavy overhead.
    """

    def __init__(self, wrapper_root):
        super(BreakTransformOptimizer, self).__init__()

        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node

    def transform(self):
        self.visit(self.root)

    def visit_Break(self, node):
        loop_node_index = _find_ancestor_loop_index(node, self.ancestor_nodes)
        assert loop_node_index != -1, "SyntaxError: 'break' outside loop"
        loop_node = self.ancestor_nodes[loop_node_index]

        if self._is_break_cond_pattern(node, loop_node):
            cond_var_node = self._join_with_while_cond(node, loop_node)

            if isinstance(loop_node, gast.While):
                loop_node.test = gast.BoolOp(
                    op=gast.And(), values=[loop_node.test, cond_var_node])
            elif isinstance(loop_node, gast.For):
                parent_node = self.ancestor_nodes[loop_node_index - 1]
                for_to_while = ForToWhileTransformer(parent_node, loop_node,
                                                     cond_var_node)
                for_to_while.transform()

    def _is_break_cond_pattern(self, break_node, loop_node):
        """
        Judge whether if match the pattern to join `If.test` with `while.test`
        """
        # while/for -> if -> break
        if len(self.ancestor_nodes) < 3 or self.ancestor_nodes[-3] != loop_node:
            return False

        assert self.ancestor_nodes[-1] == break_node
        parent_if_node = self.ancestor_nodes[-2]

        is_matched = False
        if isinstance(parent_if_node, gast.If):
            # gast.If only contains `break`
            break_first_in_if = parent_if_node.body[0] == break_node and len(
                parent_if_node.orelse) == 0
            # gast.If is first node of loop_node
            if_first_in_loop = loop_node.body[0] == parent_if_node

            is_matched = if_first_in_loop and break_first_in_if

        return is_matched

    def _join_with_while_cond(self, break_node, loop_node):
        """
        Join the `If.test` with `While.test` together.
        """
        parent_if_node = self.ancestor_nodes[-2]

        cond_var_node = gast.UnaryOp(op=gast.Not(), operand=parent_if_node.test)

        # remove the gast.If node that contains the gast.Break.
        assert loop_node.body[0] == parent_if_node
        loop_node.body.pop(0)

        return cond_var_node
