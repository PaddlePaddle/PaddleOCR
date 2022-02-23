"""
==========================================
Miscellaneous routines (:mod:`scipy.misc`)
==========================================

.. currentmodule:: scipy.misc

Various utilities that don't have another home.

.. autosummary::
   :toctree: generated/

   ascent - Get example image for processing
   central_diff_weights - Weights for an n-point central mth derivative
   derivative - Find the nth derivative of a function at a point
   face - Get example image for processing
   electrocardiogram - Load an example of a 1-D signal.

"""

from . import doccer
from .common import *

__all__ = ['doccer']

from . import common
__all__ += common.__all__
del common

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
