"""
========================================
Interpolation (:mod:`scipy.interpolate`)
========================================

.. currentmodule:: scipy.interpolate

Sub-package for objects used in interpolation.

As listed below, this sub-package contains spline functions and classes,
1-D and multidimensional (univariate and multivariate)
interpolation classes, Lagrange and Taylor polynomial interpolators, and
wrappers for `FITPACK <http://www.netlib.org/dierckx/>`__
and DFITPACK functions.

Univariate interpolation
========================

.. autosummary::
   :toctree: generated/

   interp1d
   BarycentricInterpolator
   KroghInterpolator
   barycentric_interpolate
   krogh_interpolate
   pchip_interpolate
   CubicHermiteSpline
   PchipInterpolator
   Akima1DInterpolator
   CubicSpline
   PPoly
   BPoly


Multivariate interpolation
==========================

Unstructured data:

.. autosummary::
   :toctree: generated/

   griddata
   LinearNDInterpolator
   NearestNDInterpolator
   CloughTocher2DInterpolator
   RBFInterpolator
   Rbf
   interp2d

For data on a grid:

.. autosummary::
   :toctree: generated/

   interpn
   RegularGridInterpolator
   RectBivariateSpline

.. seealso::

    `scipy.ndimage.map_coordinates`

Tensor product polynomials:

.. autosummary::
   :toctree: generated/

   NdPPoly


1-D Splines
===========

.. autosummary::
   :toctree: generated/

   BSpline
   make_interp_spline
   make_lsq_spline

Functional interface to FITPACK routines:

.. autosummary::
   :toctree: generated/

   splrep
   splprep
   splev
   splint
   sproot
   spalde
   splder
   splantider
   insert

Object-oriented FITPACK interface:

.. autosummary::
   :toctree: generated/

   UnivariateSpline
   InterpolatedUnivariateSpline
   LSQUnivariateSpline



2-D Splines
===========

For data on a grid:

.. autosummary::
   :toctree: generated/

   RectBivariateSpline
   RectSphereBivariateSpline

For unstructured data:

.. autosummary::
   :toctree: generated/

   BivariateSpline
   SmoothBivariateSpline
   SmoothSphereBivariateSpline
   LSQBivariateSpline
   LSQSphereBivariateSpline

Low-level interface to FITPACK functions:

.. autosummary::
   :toctree: generated/

   bisplrep
   bisplev

Additional tools
================

.. autosummary::
   :toctree: generated/

   lagrange
   approximate_taylor_polynomial
   pade

.. seealso::

   `scipy.ndimage.map_coordinates`,
   `scipy.ndimage.spline_filter`,
   `scipy.signal.resample`,
   `scipy.signal.bspline`,
   `scipy.signal.gauss_spline`,
   `scipy.signal.qspline1d`,
   `scipy.signal.cspline1d`,
   `scipy.signal.qspline1d_eval`,
   `scipy.signal.cspline1d_eval`,
   `scipy.signal.qspline2d`,
   `scipy.signal.cspline2d`.

``pchip`` is an alias of `PchipInterpolator` for backward compatibility
(should not be used in new code).
"""
from .interpolate import *
from .fitpack import *

# New interface to fitpack library:
from .fitpack2 import *

from .rbf import Rbf

from ._rbfinterp import *

from .polyint import *

from ._cubic import *

from .ndgriddata import *

from ._bsplines import *

from ._pade import *

__all__ = [s for s in dir() if not s.startswith('_')]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester

# Backward compatibility
pchip = PchipInterpolator
