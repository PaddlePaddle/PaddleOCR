"""
sparsetools is not a public module in scipy.sparse, but this file is
for backward compatibility if someone happens to use it.
"""
from numpy import deprecate

# This file shouldn't be imported by scipy --- SciPy code should use
# internally scipy.sparse._sparsetools


@deprecate(old_name="scipy.sparse.sparsetools",
           message=("scipy.sparse.sparsetools is a private module for scipy.sparse, "
                    "and should not be used."))
def _deprecated():
    pass


del deprecate

try:
    _deprecated()
except DeprecationWarning:
    # don't fail import if DeprecationWarnings raise error -- works around
    # the situation with NumPy's test framework
    pass

from ._sparsetools import *
