"""
For the ``future`` package.

Turns any print statements into functions and adds this import line:

    from __future__ import print_function

at the top to retain compatibility with Python 2.6+.
"""

from libfuturize.fixes.fix_print import FixPrint
from libfuturize.fixer_util import future_import

class FixPrintWithImport(FixPrint):
    run_order = 7
    def transform(self, node, results):
        # Add the __future__ import first. (Otherwise any shebang or encoding
        # comment line attached as a prefix to the print statement will be
        # copied twice and appear twice.)
        future_import(u'print_function', node)
        n_stmt = super(FixPrintWithImport, self).transform(node, results)
        return n_stmt
