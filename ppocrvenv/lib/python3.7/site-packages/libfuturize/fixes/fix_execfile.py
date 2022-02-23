# coding: utf-8
"""
Fixer for the execfile() function on Py2, which was removed in Py3.

The Lib/lib2to3/fixes/fix_execfile.py module has some problems: see
python-future issue #37. This fixer merely imports execfile() from
past.builtins and leaves the code alone.

Adds this import line::

    from past.builtins import execfile

for the function execfile() that was removed from Py3.
"""

from __future__ import unicode_literals
from lib2to3 import fixer_base

from libfuturize.fixer_util import touch_import_top


expression = "name='execfile'"


class FixExecfile(fixer_base.BaseFix):
    BM_compatible = True
    run_order = 9

    PATTERN = """
              power<
                 ({0}) trailer< '(' args=[any] ')' >
              rest=any* >
              """.format(expression)

    def transform(self, node, results):
        name = results["name"]
        touch_import_top(u'past.builtins', name.value, node)
