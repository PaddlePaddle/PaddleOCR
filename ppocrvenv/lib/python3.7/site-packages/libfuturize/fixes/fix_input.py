"""
Fixer for input.

Does a check for `from builtins import input` before running the lib2to3 fixer.
The fixer will not run when the input is already present.


this:
    a = input()
becomes:
    from builtins import input
    a = eval(input())

and this:
    from builtins import input
    a = input()
becomes (no change):
    from builtins import input
    a = input()
"""

import lib2to3.fixes.fix_input
from lib2to3.fixer_util import does_tree_import


class FixInput(lib2to3.fixes.fix_input.FixInput):
    def transform(self, node, results):

        if does_tree_import('builtins', 'input', node):
            return

        return super(FixInput, self).transform(node, results)
