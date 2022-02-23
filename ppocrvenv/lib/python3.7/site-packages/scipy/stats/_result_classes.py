# This module exists only to allow Sphinx to generate docs
# for the result objects returned by some functions in stats.

"""
Result classes
--------------

.. currentmodule:: scipy.stats._result_classes

.. autosummary::
   :toctree: generated/

   RelativeRiskResult
   BinomTestResult

"""

__all__ = ['BinomTestResult', 'RelativeRiskResult']


from ._binomtest import BinomTestResult
from ._relative_risk import RelativeRiskResult
