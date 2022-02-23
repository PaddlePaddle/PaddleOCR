"""
Spatial Transformations (:mod:`scipy.spatial.transform`)
========================================================

.. currentmodule:: scipy.spatial.transform

This package implements various spatial transformations. For now,
only rotations are supported.

Rotations in 3 dimensions
-------------------------
.. autosummary::
   :toctree: generated/

   Rotation
   Slerp
   RotationSpline
"""
from .rotation import Rotation, Slerp
from ._rotation_spline import RotationSpline

__all__ = ['Rotation', 'Slerp', 'RotationSpline']

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
