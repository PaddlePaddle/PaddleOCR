"""
Fixer that adds ``from past.builtins import basestring`` if there is a
reference to ``basestring``
"""

from lib2to3 import fixer_base

from libfuturize.fixer_util import touch_import_top


class FixBasestring(fixer_base.BaseFix):
    BM_compatible = True

    PATTERN = "'basestring'"

    def transform(self, node, results):
        touch_import_top(u'past.builtins', 'basestring', node)
