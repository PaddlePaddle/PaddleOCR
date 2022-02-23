"""
Fixer for removing any of these lines:

    from __future__ import with_statement
    from __future__ import nested_scopes
    from __future__ import generators

The reason is that __future__ imports like these are required to be the first
line of code (after docstrings) on Python 2.6+, which can get in the way.

These imports are always enabled in Python 2.6+, which is the minimum sane
version to target for Py2/3 compatibility.
"""

from lib2to3 import fixer_base
from libfuturize.fixer_util import remove_future_import

class FixRemoveOldFutureImports(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = "file_input"
    run_order = 1

    def transform(self, node, results):
        remove_future_import(u"with_statement", node)
        remove_future_import(u"nested_scopes", node)
        remove_future_import(u"generators", node)
