# -*- coding: utf-8 -*-
"""
Part of the astor library for Python AST manipulation.

License: 3-clause BSD

Copyright (c) 2015 Patrick Maupin

This module provides data and functions for mapping
AST nodes to symbols and precedences.

"""

import ast

op_data = """
    GeneratorExp                1

          Assign                1
       AnnAssign                1
       AugAssign                0
            Expr                0
           Yield                1
       YieldFrom                0
              If                1
             For                0
        AsyncFor                0
           While                0
          Return                1

           Slice                1
       Subscript                0
           Index                1
        ExtSlice                1
    comprehension_target        1
           Tuple                0
  FormattedValue                0

           Comma                1
       NamedExpr                1
          Assert                0
           Raise                0
    call_one_arg                1

          Lambda                1
           IfExp                0

   comprehension                1
              Or   or           1
             And   and          1
             Not   not          1

              Eq   ==           1
              Gt   >            0
             GtE   >=           0
              In   in           0
              Is   is           0
           NotEq   !=           0
              Lt   <            0
             LtE   <=           0
           NotIn   not in       0
           IsNot   is not       0

           BitOr   |            1
          BitXor   ^            1
          BitAnd   &            1
          LShift   <<           1
          RShift   >>           0
             Add   +            1
             Sub   -            0
            Mult   *            1
             Div   /            0
             Mod   %            0
        FloorDiv   //           0
         MatMult   @            0
          PowRHS                1
          Invert   ~            1
            UAdd   +            0
            USub   -            0
             Pow   **           1
           Await                1
             Num                1
        Constant                1
"""

op_data = [x.split() for x in op_data.splitlines()]
op_data = [[x[0], ' '.join(x[1:-1]), int(x[-1])] for x in op_data if x]
for index in range(1, len(op_data)):
    op_data[index][2] *= 2
    op_data[index][2] += op_data[index - 1][2]

precedence_data = dict((getattr(ast, x, None), z) for x, y, z in op_data)
symbol_data = dict((getattr(ast, x, None), y) for x, y, z in op_data)


def get_op_symbol(obj, fmt='%s', symbol_data=symbol_data, type=type):
    """Given an AST node object, returns a string containing the symbol.
    """
    return fmt % symbol_data[type(obj)]


def get_op_precedence(obj, precedence_data=precedence_data, type=type):
    """Given an AST node object, returns the precedence.
    """
    return precedence_data[type(obj)]


class Precedence(object):
    vars().update((x, z) for x, y, z in op_data)
    highest = max(z for x, y, z in op_data) + 2
