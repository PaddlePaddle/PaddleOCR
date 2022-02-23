"""
Fixer for adding:

    from __future__ import absolute_import
    from __future__ import division
    from __future__ import print_function

This is "stage 1": hopefully uncontroversial changes.

Stage 2 adds ``unicode_literals``.
"""

from lib2to3 import fixer_base
from libfuturize.fixer_util import future_import

class FixAddFutureImportsExceptUnicodeLiterals(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = "file_input"

    run_order = 9

    def transform(self, node, results):
        # Reverse order:
        future_import(u"absolute_import", node)
        future_import(u"division", node)
        future_import(u"print_function", node)
