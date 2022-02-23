"""
For the ``future`` package.

Adds this import line::

    from builtins import XYZ

for each of the functions XYZ that is used in the module.

Adds these imports after any other imports (in an initial block of them).
"""

from __future__ import unicode_literals

from lib2to3 import fixer_base
from lib2to3.pygram import python_symbols as syms
from lib2to3.fixer_util import Name, Call, in_special_context

from libfuturize.fixer_util import touch_import_top

# All builtins are:
#     from future.builtins.iterators import (filter, map, zip)
#     from future.builtins.misc import (ascii, chr, hex, input, isinstance, oct, open, round, super)
#     from future.types import (bytes, dict, int, range, str)
# We don't need isinstance any more.

replaced_builtin_fns = '''filter map zip
                       ascii chr hex input next oct
                       bytes range str raw_input'''.split()
                       # This includes raw_input as a workaround for the
                       # lib2to3 fixer for raw_input on Py3 (only), allowing
                       # the correct import to be included. (Py3 seems to run
                       # the fixers the wrong way around, perhaps ignoring the
                       # run_order class attribute below ...)

expression = '|'.join(["name='{0}'".format(name) for name in replaced_builtin_fns])


class FixFutureBuiltins(fixer_base.BaseFix):
    BM_compatible = True
    run_order = 7

    # Currently we only match uses as a function. This doesn't match e.g.:
    #     if isinstance(s, str):
    #         ...
    PATTERN = """
              power<
                 ({0}) trailer< '(' [arglist=any] ')' >
              rest=any* >
              |
              power<
                  'map' trailer< '(' [arglist=any] ')' >
              >
              """.format(expression)

    def transform(self, node, results):
        name = results["name"]
        touch_import_top(u'builtins', name.value, node)
        # name.replace(Name(u"input", prefix=name.prefix))
