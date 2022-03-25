"""LAP
``lap`` is a linear assignment problem solver using Jonker-Volgenant
algorithm for dense (LAPJV) or sparse (LAPMOD) matrices.

Functions
---------

lapjv
    Find optimal (minimum-cost) assignment for a dense cost matrix.
lapmod
    Find optimal (minimum-cost) assignment for a sparse cost matrix.
"""

import sys

__version__ = '0.4.0'

try:
    __LAP_SETUP__
except NameError:
    __LAP_SETUP__ = False
if __LAP_SETUP__:
    sys.stderr.write('Partial import of lap during the build process.\n')
else:
    from ._lapjv import (
            lapjv,
            LARGE_ as LARGE,
            FP_1_ as FP_1, FP_2_ as FP_2, FP_DYNAMIC_ as FP_DYNAMIC)
    from .lapmod import lapmod
    __all__ = ['lapjv', 'lapmod', 'FP_1', 'FP_2', 'FP_DYNAMIC', 'LARGE']
