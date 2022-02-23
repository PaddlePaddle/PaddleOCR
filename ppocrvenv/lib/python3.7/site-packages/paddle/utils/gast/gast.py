# Copyright (c) 2016, Serge Guelton
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 	Redistributions of source code must retain the above copyright notice, this
# 	list of conditions and the following disclaimer.

# 	Redistributions in binary form must reproduce the above copyright notice,
# 	this list of conditions and the following disclaimer in the documentation
# 	and/or other materials provided with the distribution.

# 	Neither the name of HPCProject, Serge Guelton nor the names of its
# 	contributors may be used to endorse or promote products derived from this
# 	software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# NOTE(paddle-dev): We introduce third-party library Gast as unified AST
# representation. See https://github.com/serge-sans-paille/gast for details.

import sys as _sys
import ast as _ast
from ast import boolop, cmpop, excepthandler, expr, expr_context, operator
from ast import slice, stmt, unaryop, mod, AST
from ast import iter_child_nodes, walk

try:
    from ast import TypeIgnore
except ImportError:

    class TypeIgnore(AST):
        pass


def _make_node(Name, Fields, Attributes, Bases):
    def create_node(self, *args, **kwargs):
        nbparam = len(args) + len(kwargs)
        assert nbparam in (0, len(Fields)), \
            "Bad argument number for {}: {}, expecting {}".\
            format(Name, nbparam, len(Fields))
        self._fields = Fields
        self._attributes = Attributes
        for argname, argval in zip(self._fields, args):
            setattr(self, argname, argval)
        for argname, argval in kwargs.items():
            assert argname in Fields, \
                    "Invalid Keyword argument for {}: {}".format(Name, argname)
            setattr(self, argname, argval)

    setattr(_sys.modules[__name__], Name,
            type(Name, Bases, {'__init__': create_node}))


_nodes = (
    # mod
    ('Module', (('body', 'type_ignores'), (), (mod, ))),
    ('Interactive', (('body', ), (), (mod, ))),
    ('Expression', (('body', ), (), (mod, ))),
    ('FunctionType', (('argtypes', 'returns'), (), (mod, ))),
    ('Suite', (('body', ), (), (mod, ))),

    # stmt
    ('FunctionDef', (('name', 'args', 'body', 'decorator_list', 'returns',
                      'type_comment'), (
                          'lineno',
                          'col_offset',
                          'end_lineno',
                          'end_col_offset', ), (stmt, ))),
    ('AsyncFunctionDef', (('name', 'args', 'body', 'decorator_list', 'returns',
                           'type_comment'), (
                               'lineno',
                               'col_offset',
                               'end_lineno',
                               'end_col_offset', ), (stmt, ))),
    ('ClassDef', ((
        'name',
        'bases',
        'keywords',
        'body',
        'decorator_list', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (stmt, ))),
    ('Return', (('value', ), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (stmt, ))),
    ('Delete', (('targets', ), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (stmt, ))),
    ('Assign', ((
        'targets',
        'value', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (stmt, ))),
    ('AugAssign', ((
        'target',
        'op',
        'value', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (stmt, ))),
    ('AnnAssign', ((
        'target',
        'annotation',
        'value',
        'simple', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (stmt, ))),
    ('Print', ((
        'dest',
        'values',
        'nl', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (stmt, ))),
    ('For', (('target', 'iter', 'body', 'orelse', 'type_comment'), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (stmt, ))),
    ('AsyncFor', (('target', 'iter', 'body', 'orelse', 'type_comment'), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (stmt, ))),
    ('While', ((
        'test',
        'body',
        'orelse', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (stmt, ))),
    ('If', ((
        'test',
        'body',
        'orelse', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (stmt, ))),
    ('With', (('items', 'body', 'type_comment'), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (stmt, ))),
    ('AsyncWith', (('items', 'body', 'type_comment'), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (stmt, ))),
    ('Raise', ((
        'exc',
        'cause', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (stmt, ))),
    ('Try', ((
        'body',
        'handlers',
        'orelse',
        'finalbody', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (stmt, ))),
    ('Assert', ((
        'test',
        'msg', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (stmt, ))),
    ('Import', (('names', ), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (stmt, ))),
    ('ImportFrom', ((
        'module',
        'names',
        'level', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (stmt, ))),
    ('Exec', ((
        'body',
        'globals',
        'locals', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (stmt, ))),
    ('Global', (('names', ), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (stmt, ))),
    ('Nonlocal', (('names', ), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (stmt, ))),
    ('Expr', (('value', ), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (stmt, ))),
    ('Pass', ((), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (stmt, ))),
    ('Break', ((), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (stmt, ))),
    ('Continue', ((), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (stmt, ))),

    # expr
    ('BoolOp', ((
        'op',
        'values', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (expr, ))),
    ('BinOp', ((
        'left',
        'op',
        'right', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (expr, ))),
    ('UnaryOp', ((
        'op',
        'operand', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (expr, ))),
    ('Lambda', ((
        'args',
        'body', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (expr, ))),
    ('IfExp', ((
        'test',
        'body',
        'orelse', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (expr, ))),
    ('Dict', ((
        'keys',
        'values', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (expr, ))),
    ('Set', (('elts', ), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (expr, ))),
    ('ListComp', ((
        'elt',
        'generators', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (expr, ))),
    ('SetComp', ((
        'elt',
        'generators', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (expr, ))),
    ('DictComp', ((
        'key',
        'value',
        'generators', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (expr, ))),
    ('GeneratorExp', ((
        'elt',
        'generators', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (expr, ))),
    ('Await', (('value', ), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (expr, ))),
    ('Yield', (('value', ), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (expr, ))),
    ('YieldFrom', (('value', ), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (expr, ))),
    ('Compare', ((
        'left',
        'ops',
        'comparators', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (expr, ))),
    ('Call', ((
        'func',
        'args',
        'keywords', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (expr, ))),
    ('Repr', (('value', ), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (expr, ))),
    ('FormattedValue', ((
        'value',
        'conversion',
        'format_spec', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (expr, ))),
    ('JoinedStr', (('values', ), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (expr, ))),
    ('Constant', (('value', 'kind'), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (expr, ))),
    ('Attribute', ((
        'value',
        'attr',
        'ctx', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (expr, ))),
    ('Subscript', ((
        'value',
        'slice',
        'ctx', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (expr, ))),
    ('Starred', ((
        'value',
        'ctx', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (expr, ))),
    ('Name', (('id', 'ctx', 'annotation', 'type_comment'), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (expr, ))),
    ('List', ((
        'elts',
        'ctx', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (expr, ))),
    ('Tuple', ((
        'elts',
        'ctx', ), (
            'lineno',
            'col_offset',
            'end_lineno',
            'end_col_offset', ), (expr, ))),

    # expr_context
    ('Load', ((), (), (expr_context, ))),
    ('Store', ((), (), (expr_context, ))),
    ('Del', ((), (), (expr_context, ))),
    ('AugLoad', ((), (), (expr_context, ))),
    ('AugStore', ((), (), (expr_context, ))),
    ('Param', ((), (), (expr_context, ))),

    # slice
    ('Slice', (('lower', 'upper', 'step'), (
        'lineno',
        'col_offset',
        'end_lineno',
        'end_col_offset', ), (slice, ))),

    # boolop
    ('And', ((), (), (boolop, ))),
    ('Or', ((), (), (boolop, ))),

    # operator
    ('Add', ((), (), (operator, ))),
    ('Sub', ((), (), (operator, ))),
    ('Mult', ((), (), (operator, ))),
    ('MatMult', ((), (), (operator, ))),
    ('Div', ((), (), (operator, ))),
    ('Mod', ((), (), (operator, ))),
    ('Pow', ((), (), (operator, ))),
    ('LShift', ((), (), (operator, ))),
    ('RShift', ((), (), (operator, ))),
    ('BitOr', ((), (), (operator, ))),
    ('BitXor', ((), (), (operator, ))),
    ('BitAnd', ((), (), (operator, ))),
    ('FloorDiv', ((), (), (operator, ))),

    # unaryop
    ('Invert', ((), (), (
        unaryop,
        AST, ))),
    ('Not', ((), (), (
        unaryop,
        AST, ))),
    ('UAdd', ((), (), (
        unaryop,
        AST, ))),
    ('USub', ((), (), (
        unaryop,
        AST, ))),

    # cmpop
    ('Eq', ((), (), (cmpop, ))),
    ('NotEq', ((), (), (cmpop, ))),
    ('Lt', ((), (), (cmpop, ))),
    ('LtE', ((), (), (cmpop, ))),
    ('Gt', ((), (), (cmpop, ))),
    ('GtE', ((), (), (cmpop, ))),
    ('Is', ((), (), (cmpop, ))),
    ('IsNot', ((), (), (cmpop, ))),
    ('In', ((), (), (cmpop, ))),
    ('NotIn', ((), (), (cmpop, ))),

    # comprehension
    ('comprehension', (('target', 'iter', 'ifs', 'is_async'), (), (AST, ))),

    # excepthandler
    ('ExceptHandler', (('type', 'name', 'body'),
                       ('lineno', 'col_offset', 'end_lineno',
                        'end_col_offset'), (excepthandler, ))),

    # arguments
    ('arguments', (('args', 'posonlyargs', 'vararg', 'kwonlyargs',
                    'kw_defaults', 'kwarg', 'defaults'), (), (AST, ))),

    # keyword
    ('keyword',
     (('arg', 'value'),
      ('lineno', 'col_offset', 'end_lineno', 'end_col_offset'), (AST, ))),

    # alias
    ('alias', (('name', 'asname'), (), (AST, ))),

    # withitem
    ('withitem', (('context_expr', 'optional_vars'), (), (AST, ))),

    # type_ignore
    ('type_ignore', ((), ('lineno', 'tag'), (TypeIgnore, ))), )

for name, descr in _nodes:
    _make_node(name, *descr)

py_version = _sys.version_info.major
if py_version != 3:
    raise RuntimeError(
        'Required Python version >= 3, but received Python version == {}'.
        format(py_version))

from .ast3 import ast_to_gast, gast_to_ast


def parse(*args, **kwargs):
    return ast_to_gast(_ast.parse(*args, **kwargs))


def literal_eval(node_or_string):
    if isinstance(node_or_string, AST):
        node_or_string = gast_to_ast(node_or_string)
    return _ast.literal_eval(node_or_string)


def get_docstring(node, clean=True):
    if not isinstance(node, (FunctionDef, ClassDef, Module)):
        raise TypeError("%r can't have docstrings" % node.__class__.__name__)
    if node.body and isinstance(node.body[0], Expr) and \
       isinstance(node.body[0].value, Constant):
        if clean:
            import inspect
            holder = node.body[0].value
            return inspect.cleandoc(getattr(holder, holder._fields[0]))
        return node.body[0].value.s


# the following are directly imported from python3.8's Lib/ast.py  #


def copy_location(new_node, old_node):
    """
    Copy source location (`lineno`, `col_offset`, `end_lineno`, and
    `end_col_offset` attributes) from *old_node* to *new_node* if possible,
    and return *new_node*.
    """
    for attr in 'lineno', 'col_offset', 'end_lineno', 'end_col_offset':
        if attr in old_node._attributes and attr in new_node._attributes \
           and hasattr(old_node, attr):
            setattr(new_node, attr, getattr(old_node, attr))
    return new_node


def fix_missing_locations(node):
    """
    When you compile a node tree with compile(), the compiler expects lineno
    and col_offset attributes for every node that supports them.  This is
    rather tedious to fill in for generated nodes, so this helper adds these
    attributes recursively where not already set, by setting them to the values
    of the parent node.  It works recursively starting at *node*.
    """

    def _fix(node, lineno, col_offset, end_lineno, end_col_offset):
        if 'lineno' in node._attributes:
            if not hasattr(node, 'lineno'):
                node.lineno = lineno
            else:
                lineno = node.lineno
        if 'end_lineno' in node._attributes:
            if not hasattr(node, 'end_lineno'):
                node.end_lineno = end_lineno
            else:
                end_lineno = node.end_lineno
        if 'col_offset' in node._attributes:
            if not hasattr(node, 'col_offset'):
                node.col_offset = col_offset
            else:
                col_offset = node.col_offset
        if 'end_col_offset' in node._attributes:
            if not hasattr(node, 'end_col_offset'):
                node.end_col_offset = end_col_offset
            else:
                end_col_offset = node.end_col_offset
        for child in iter_child_nodes(node):
            _fix(child, lineno, col_offset, end_lineno, end_col_offset)

    _fix(node, 1, 0, 1, 0)
    return node


def increment_lineno(node, n=1):
    """
    Increment the line number and end line number of each node in the tree
    starting at *node* by *n*. This is useful to "move code" to a different
    location in a file.
    """
    for child in walk(node):
        if 'lineno' in child._attributes:
            child.lineno = (getattr(child, 'lineno', 0) or 0) + n
        if 'end_lineno' in child._attributes:
            child.end_lineno = (getattr(child, 'end_lineno', 0) or 0) + n
    return node
