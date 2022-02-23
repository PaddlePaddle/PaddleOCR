"""
For the ``future`` package.

Adds this import line:

    from future import standard_library

after any __future__ imports but before any other imports. Doesn't actually
change the imports to Py3 style.
"""

from lib2to3 import fixer_base
from libfuturize.fixer_util import touch_import_top

class FixAddFutureStandardLibraryImport(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = "file_input"
    run_order = 8

    def transform(self, node, results):
        # TODO: add a blank line between any __future__ imports and this?
        touch_import_top(u'future', u'standard_library', node)
        # TODO: also add standard_library.install_hooks()
