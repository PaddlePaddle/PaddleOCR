"""
=========================================================
Multidimensional image processing (:mod:`scipy.ndimage`)
=========================================================

.. currentmodule:: scipy.ndimage

This package contains various functions for multidimensional image
processing.


Filters
=======

.. autosummary::
   :toctree: generated/

   convolve - Multidimensional convolution
   convolve1d - 1-D convolution along the given axis
   correlate - Multidimensional correlation
   correlate1d - 1-D correlation along the given axis
   gaussian_filter
   gaussian_filter1d
   gaussian_gradient_magnitude
   gaussian_laplace
   generic_filter - Multidimensional filter using a given function
   generic_filter1d - 1-D generic filter along the given axis
   generic_gradient_magnitude
   generic_laplace
   laplace - N-D Laplace filter based on approximate second derivatives
   maximum_filter
   maximum_filter1d
   median_filter - Calculates a multidimensional median filter
   minimum_filter
   minimum_filter1d
   percentile_filter - Calculates a multidimensional percentile filter
   prewitt
   rank_filter - Calculates a multidimensional rank filter
   sobel
   uniform_filter - Multidimensional uniform filter
   uniform_filter1d - 1-D uniform filter along the given axis

Fourier filters
===============

.. autosummary::
   :toctree: generated/

   fourier_ellipsoid
   fourier_gaussian
   fourier_shift
   fourier_uniform

Interpolation
=============

.. autosummary::
   :toctree: generated/

   affine_transform - Apply an affine transformation
   geometric_transform - Apply an arbritrary geometric transform
   map_coordinates - Map input array to new coordinates by interpolation
   rotate - Rotate an array
   shift - Shift an array
   spline_filter
   spline_filter1d
   zoom - Zoom an array

Measurements
============

.. autosummary::
   :toctree: generated/

   center_of_mass - The center of mass of the values of an array at labels
   extrema - Min's and max's of an array at labels, with their positions
   find_objects - Find objects in a labeled array
   histogram - Histogram of the values of an array, optionally at labels
   label - Label features in an array
   labeled_comprehension
   maximum
   maximum_position
   mean - Mean of the values of an array at labels
   median
   minimum
   minimum_position
   standard_deviation - Standard deviation of an N-D image array
   sum_labels - Sum of the values of the array
   variance - Variance of the values of an N-D image array
   watershed_ift

Morphology
==========

.. autosummary::
   :toctree: generated/

   binary_closing
   binary_dilation
   binary_erosion
   binary_fill_holes
   binary_hit_or_miss
   binary_opening
   binary_propagation
   black_tophat
   distance_transform_bf
   distance_transform_cdt
   distance_transform_edt
   generate_binary_structure
   grey_closing
   grey_dilation
   grey_erosion
   grey_opening
   iterate_structure
   morphological_gradient
   morphological_laplace
   white_tophat

"""

# Copyright (C) 2003-2005 Peter J. Verveer
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from .filters import *
from .fourier import *
from .interpolation import *
from .measurements import *
from .morphology import *

__version__ = '2.0'

__all__ = [s for s in dir() if not s.startswith('_')]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
