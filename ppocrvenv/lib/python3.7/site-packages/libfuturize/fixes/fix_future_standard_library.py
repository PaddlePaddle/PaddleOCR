"""
For the ``future`` package.

Changes any imports needed to reflect the standard library reorganization. Also
Also adds these import lines:

    from future import standard_library
    standard_library.install_aliases()

after any __future__ imports but before any other imports.
"""

from lib2to3.fixes.fix_imports import FixImports
from libfuturize.fixer_util import touch_import_top


class FixFutureStandardLibrary(FixImports):
    run_order = 8

    def transform(self, node, results):
        result = super(FixFutureStandardLibrary, self).transform(node, results)
        # TODO: add a blank line between any __future__ imports and this?
        touch_import_top(u'future', u'standard_library', node)
        return result
