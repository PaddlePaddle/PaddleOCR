"""
For the ``future`` package.

Adds this import line::

    from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                          int, list, map, next, object, oct, open, pow,
                          range, round, str, super, zip)

to a module, irrespective of whether each definition is used.

Adds these imports after any other imports (in an initial block of them).
"""

from __future__ import unicode_literals

from lib2to3 import fixer_base

from libfuturize.fixer_util import touch_import_top


class FixAddAllFutureBuiltins(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = "file_input"
    run_order = 1

    def transform(self, node, results):
        # import_str = """(ascii, bytes, chr, dict, filter, hex, input,
        #                      int, list, map, next, object, oct, open, pow,
        #                      range, round, str, super, zip)"""
        touch_import_top(u'builtins', '*', node)

        # builtins = """ascii bytes chr dict filter hex input
        #                      int list map next object oct open pow
        #                      range round str super zip"""
        # for builtin in sorted(builtins.split(), reverse=True):
        #     touch_import_top(u'builtins', builtin, node)
