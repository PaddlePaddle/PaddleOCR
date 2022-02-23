__all__ = ['interp1d', 'interp2d', 'lagrange', 'PPoly', 'BPoly', 'NdPPoly',
           'RegularGridInterpolator', 'interpn']

import itertools
import warnings

import numpy as np
from numpy import (array, transpose, searchsorted, atleast_1d, atleast_2d,
                   ravel, poly1d, asarray, intp)

import scipy.special as spec
from scipy.special import comb
from scipy._lib._util import prod

from . import fitpack
from . import dfitpack
from . import _fitpack
from .polyint import _Interpolator1D
from . import _ppoly
from .fitpack2 import RectBivariateSpline
from .interpnd import _ndim_coords_from_arrays
from ._bsplines import make_interp_spline, BSpline


def lagrange(x, w):
    r"""
    Return a Lagrange interpolating polynomial.

    Given two 1-D arrays `x` and `w,` returns the Lagrange interpolating
    polynomial through the points ``(x, w)``.

    Warning: This implementation is numerically unstable. Do not expect to
    be able to use more than about 20 points even if they are chosen optimally.

    Parameters
    ----------
    x : array_like
        `x` represents the x-coordinates of a set of datapoints.
    w : array_like
        `w` represents the y-coordinates of a set of datapoints, i.e., f(`x`).

    Returns
    -------
    lagrange : `numpy.poly1d` instance
        The Lagrange interpolating polynomial.

    Examples
    --------
    Interpolate :math:`f(x) = x^3` by 3 points.

    >>> from scipy.interpolate import lagrange
    >>> x = np.array([0, 1, 2])
    >>> y = x**3
    >>> poly = lagrange(x, y)

    Since there are only 3 points, Lagrange polynomial has degree 2. Explicitly,
    it is given by

    .. math::

        \begin{aligned}
            L(x) &= 1\times \frac{x (x - 2)}{-1} + 8\times \frac{x (x-1)}{2} \\
                 &= x (-2 + 3x)
        \end{aligned}

    >>> from numpy.polynomial.polynomial import Polynomial
    >>> Polynomial(poly).coef
    array([ 3., -2.,  0.])

    """

    M = len(x)
    p = poly1d(0.0)
    for j in range(M):
        pt = poly1d(w[j])
        for k in range(M):
            if k == j:
                continue
            fac = x[j]-x[k]
            pt *= poly1d([1.0, -x[k]])/fac
        p += pt
    return p


# !! Need to find argument for keeping initialize. If it isn't
# !! found, get rid of it!


class interp2d:
    """
    interp2d(x, y, z, kind='linear', copy=True, bounds_error=False,
             fill_value=None)

    Interpolate over a 2-D grid.

    `x`, `y` and `z` are arrays of values used to approximate some function
    f: ``z = f(x, y)`` which returns a scalar value `z`. This class returns a
    function whose call method uses spline interpolation to find the value
    of new points.

    If `x` and `y` represent a regular grid, consider using
    `RectBivariateSpline`.

    If `z` is a vector value, consider using `interpn`.

    Note that calling `interp2d` with NaNs present in input values results in
    undefined behaviour.

    Methods
    -------
    __call__

    Parameters
    ----------
    x, y : array_like
        Arrays defining the data point coordinates.

        If the points lie on a regular grid, `x` can specify the column
        coordinates and `y` the row coordinates, for example::

          >>> x = [0,1,2];  y = [0,3]; z = [[1,2,3], [4,5,6]]

        Otherwise, `x` and `y` must specify the full coordinates for each
        point, for example::

          >>> x = [0,1,2,0,1,2];  y = [0,0,0,3,3,3]; z = [1,2,3,4,5,6]

        If `x` and `y` are multidimensional, they are flattened before use.
    z : array_like
        The values of the function to interpolate at the data points. If
        `z` is a multidimensional array, it is flattened before use.  The
        length of a flattened `z` array is either
        len(`x`)*len(`y`) if `x` and `y` specify the column and row coordinates
        or ``len(z) == len(x) == len(y)`` if `x` and `y` specify coordinates
        for each point.
    kind : {'linear', 'cubic', 'quintic'}, optional
        The kind of spline interpolation to use. Default is 'linear'.
    copy : bool, optional
        If True, the class makes internal copies of x, y and z.
        If False, references may be used. The default is to copy.
    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data (x,y), a ValueError is raised.
        If False, then `fill_value` is used.
    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If omitted (None), values outside
        the domain are extrapolated via nearest-neighbor extrapolation.

    See Also
    --------
    RectBivariateSpline :
        Much faster 2-D interpolation if your input data is on a grid
    bisplrep, bisplev :
        Spline interpolation based on FITPACK
    BivariateSpline : a more recent wrapper of the FITPACK routines
    interp1d : 1-D version of this function

    Notes
    -----
    The minimum number of data points required along the interpolation
    axis is ``(k+1)**2``, with k=1 for linear, k=3 for cubic and k=5 for
    quintic interpolation.

    The interpolator is constructed by `bisplrep`, with a smoothing factor
    of 0. If more control over smoothing is needed, `bisplrep` should be
    used directly.

    Examples
    --------
    Construct a 2-D grid and interpolate on it:

    >>> from scipy import interpolate
    >>> x = np.arange(-5.01, 5.01, 0.25)
    >>> y = np.arange(-5.01, 5.01, 0.25)
    >>> xx, yy = np.meshgrid(x, y)
    >>> z = np.sin(xx**2+yy**2)
    >>> f = interpolate.interp2d(x, y, z, kind='cubic')

    Now use the obtained interpolation function and plot the result:

    >>> import matplotlib.pyplot as plt
    >>> xnew = np.arange(-5.01, 5.01, 1e-2)
    >>> ynew = np.arange(-5.01, 5.01, 1e-2)
    >>> znew = f(xnew, ynew)
    >>> plt.plot(x, z[0, :], 'ro-', xnew, znew[0, :], 'b-')
    >>> plt.show()
    """

    def __init__(self, x, y, z, kind='linear', copy=True, bounds_error=False,
                 fill_value=None):
        x = ravel(x)
        y = ravel(y)
        z = asarray(z)

        rectangular_grid = (z.size == len(x) * len(y))
        if rectangular_grid:
            if z.ndim == 2:
                if z.shape != (len(y), len(x)):
                    raise ValueError("When on a regular grid with x.size = m "
                                     "and y.size = n, if z.ndim == 2, then z "
                                     "must have shape (n, m)")
            if not np.all(x[1:] >= x[:-1]):
                j = np.argsort(x)
                x = x[j]
                z = z[:, j]
            if not np.all(y[1:] >= y[:-1]):
                j = np.argsort(y)
                y = y[j]
                z = z[j, :]
            z = ravel(z.T)
        else:
            z = ravel(z)
            if len(x) != len(y):
                raise ValueError(
                    "x and y must have equal lengths for non rectangular grid")
            if len(z) != len(x):
                raise ValueError(
                    "Invalid length for input z for non rectangular grid")

        interpolation_types = {'linear': 1, 'cubic': 3, 'quintic': 5}
        try:
            kx = ky = interpolation_types[kind]
        except KeyError as e:
            raise ValueError(
                f"Unsupported interpolation type {repr(kind)}, must be "
                f"either of {', '.join(map(repr, interpolation_types))}."
            ) from e

        if not rectangular_grid:
            # TODO: surfit is really not meant for interpolation!
            self.tck = fitpack.bisplrep(x, y, z, kx=kx, ky=ky, s=0.0)
        else:
            nx, tx, ny, ty, c, fp, ier = dfitpack.regrid_smth(
                x, y, z, None, None, None, None,
                kx=kx, ky=ky, s=0.0)
            self.tck = (tx[:nx], ty[:ny], c[:(nx - kx - 1) * (ny - ky - 1)],
                        kx, ky)

        self.bounds_error = bounds_error
        self.fill_value = fill_value
        self.x, self.y, self.z = [array(a, copy=copy) for a in (x, y, z)]

        self.x_min, self.x_max = np.amin(x), np.amax(x)
        self.y_min, self.y_max = np.amin(y), np.amax(y)

    def __call__(self, x, y, dx=0, dy=0, assume_sorted=False):
        """Interpolate the function.

        Parameters
        ----------
        x : 1-D array
            x-coordinates of the mesh on which to interpolate.
        y : 1-D array
            y-coordinates of the mesh on which to interpolate.
        dx : int >= 0, < kx
            Order of partial derivatives in x.
        dy : int >= 0, < ky
            Order of partial derivatives in y.
        assume_sorted : bool, optional
            If False, values of `x` and `y` can be in any order and they are
            sorted first.
            If True, `x` and `y` have to be arrays of monotonically
            increasing values.

        Returns
        -------
        z : 2-D array with shape (len(y), len(x))
            The interpolated values.
        """

        x = atleast_1d(x)
        y = atleast_1d(y)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y should both be 1-D arrays")

        if not assume_sorted:
            x = np.sort(x, kind="mergesort")
            y = np.sort(y, kind="mergesort")

        if self.bounds_error or self.fill_value is not None:
            out_of_bounds_x = (x < self.x_min) | (x > self.x_max)
            out_of_bounds_y = (y < self.y_min) | (y > self.y_max)

            any_out_of_bounds_x = np.any(out_of_bounds_x)
            any_out_of_bounds_y = np.any(out_of_bounds_y)

        if self.bounds_error and (any_out_of_bounds_x or any_out_of_bounds_y):
            raise ValueError("Values out of range; x must be in %r, y in %r"
                             % ((self.x_min, self.x_max),
                                (self.y_min, self.y_max)))

        z = fitpack.bisplev(x, y, self.tck, dx, dy)
        z = atleast_2d(z)
        z = transpose(z)

        if self.fill_value is not None:
            if any_out_of_bounds_x:
                z[:, out_of_bounds_x] = self.fill_value
            if any_out_of_bounds_y:
                z[out_of_bounds_y, :] = self.fill_value

        if len(z) == 1:
            z = z[0]
        return array(z)


def _check_broadcast_up_to(arr_from, shape_to, name):
    """Helper to check that arr_from broadcasts up to shape_to"""
    shape_from = arr_from.shape
    if len(shape_to) >= len(shape_from):
        for t, f in zip(shape_to[::-1], shape_from[::-1]):
            if f != 1 and f != t:
                break
        else:  # all checks pass, do the upcasting that we need later
            if arr_from.size != 1 and arr_from.shape != shape_to:
                arr_from = np.ones(shape_to, arr_from.dtype) * arr_from
            return arr_from.ravel()
    # at least one check failed
    raise ValueError('%s argument must be able to broadcast up '
                     'to shape %s but had shape %s'
                     % (name, shape_to, shape_from))


def _do_extrapolate(fill_value):
    """Helper to check if fill_value == "extrapolate" without warnings"""
    return (isinstance(fill_value, str) and
            fill_value == 'extrapolate')


class interp1d(_Interpolator1D):
    """
    Interpolate a 1-D function.

    `x` and `y` are arrays of values used to approximate some function f:
    ``y = f(x)``. This class returns a function whose call method uses
    interpolation to find the value of new points.

    Parameters
    ----------
    x : (N,) array_like
        A 1-D array of real values.
    y : (...,N,...) array_like
        A N-D array of real values. The length of `y` along the interpolation
        axis must be equal to the length of `x`.
    kind : str or int, optional
        Specifies the kind of interpolation as a string or as an integer
        specifying the order of the spline interpolator to use.
        The string has to be one of 'linear', 'nearest', 'nearest-up', 'zero',
        'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 'zero',
        'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of
        zeroth, first, second or third order; 'previous' and 'next' simply
        return the previous or next value of the point; 'nearest-up' and
        'nearest' differ when interpolating half-integers (e.g. 0.5, 1.5)
        in that 'nearest-up' rounds up and 'nearest' rounds down. Default
        is 'linear'.
    axis : int, optional
        Specifies the axis of `y` along which to interpolate.
        Interpolation defaults to the last axis of `y`.
    copy : bool, optional
        If True, the class makes internal copies of x and y.
        If False, references to `x` and `y` are used. The default is to copy.
    bounds_error : bool, optional
        If True, a ValueError is raised any time interpolation is attempted on
        a value outside of the range of x (where extrapolation is
        necessary). If False, out of bounds values are assigned `fill_value`.
        By default, an error is raised unless ``fill_value="extrapolate"``.
    fill_value : array-like or (array-like, array_like) or "extrapolate", optional
        - if a ndarray (or float), this value will be used to fill in for
          requested points outside of the data range. If not provided, then
          the default is NaN. The array-like must broadcast properly to the
          dimensions of the non-interpolation axes.
        - If a two-element tuple, then the first element is used as a
          fill value for ``x_new < x[0]`` and the second element is used for
          ``x_new > x[-1]``. Anything that is not a 2-element tuple (e.g.,
          list or ndarray, regardless of shape) is taken to be a single
          array-like argument meant to be used for both bounds as
          ``below, above = fill_value, fill_value``.

          .. versionadded:: 0.17.0
        - If "extrapolate", then points outside the data range will be
          extrapolated.

          .. versionadded:: 0.17.0
    assume_sorted : bool, optional
        If False, values of `x` can be in any order and they are sorted first.
        If True, `x` has to be an array of monotonically increasing values.

    Attributes
    ----------
    fill_value

    Methods
    -------
    __call__

    See Also
    --------
    splrep, splev
        Spline interpolation/smoothing based on FITPACK.
    UnivariateSpline : An object-oriented wrapper of the FITPACK routines.
    interp2d : 2-D interpolation

    Notes
    -----
    Calling `interp1d` with NaNs present in input values results in
    undefined behaviour.

    Input values `x` and `y` must be convertible to `float` values like
    `int` or `float`.
    
    If the values in `x` are not unique, the resulting behavior is
    undefined and specific to the choice of `kind`, i.e., changing
    `kind` will change the behavior for duplicates.


    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy import interpolate
    >>> x = np.arange(0, 10)
    >>> y = np.exp(-x/3.0)
    >>> f = interpolate.interp1d(x, y)

    >>> xnew = np.arange(0, 9, 0.1)
    >>> ynew = f(xnew)   # use interpolation function returned by `interp1d`
    >>> plt.plot(x, y, 'o', xnew, ynew, '-')
    >>> plt.show()
    """

    def __init__(self, x, y, kind='linear', axis=-1,
                 copy=True, bounds_error=None, fill_value=np.nan,
                 assume_sorted=False):
        """ Initialize a 1-D linear interpolation class."""
        _Interpolator1D.__init__(self, x, y, axis=axis)

        self.bounds_error = bounds_error  # used by fill_value setter
        self.copy = copy

        if kind in ['zero', 'slinear', 'quadratic', 'cubic']:
            order = {'zero': 0, 'slinear': 1,
                     'quadratic': 2, 'cubic': 3}[kind]
            kind = 'spline'
        elif isinstance(kind, int):
            order = kind
            kind = 'spline'
        elif kind not in ('linear', 'nearest', 'nearest-up', 'previous',
                          'next'):
            raise NotImplementedError("%s is unsupported: Use fitpack "
                                      "routines for other types." % kind)
        x = array(x, copy=self.copy)
        y = array(y, copy=self.copy)

        if not assume_sorted:
            ind = np.argsort(x, kind="mergesort")
            x = x[ind]
            y = np.take(y, ind, axis=axis)

        if x.ndim != 1:
            raise ValueError("the x array must have exactly one dimension.")
        if y.ndim == 0:
            raise ValueError("the y array must have at least one dimension.")

        # Force-cast y to a floating-point type, if it's not yet one
        if not issubclass(y.dtype.type, np.inexact):
            y = y.astype(np.float_)

        # Backward compatibility
        self.axis = axis % y.ndim

        # Interpolation goes internally along the first axis
        self.y = y
        self._y = self._reshape_yi(self.y)
        self.x = x
        del y, x  # clean up namespace to prevent misuse; use attributes
        self._kind = kind
        self.fill_value = fill_value  # calls the setter, can modify bounds_err

        # Adjust to interpolation kind; store reference to *unbound*
        # interpolation methods, in order to avoid circular references to self
        # stored in the bound instance methods, and therefore delayed garbage
        # collection.  See: https://docs.python.org/reference/datamodel.html
        if kind in ('linear', 'nearest', 'nearest-up', 'previous', 'next'):
            # Make a "view" of the y array that is rotated to the interpolation
            # axis.
            minval = 2
            if kind == 'nearest':
                # Do division before addition to prevent possible integer
                # overflow
                self._side = 'left'
                self.x_bds = self.x / 2.0
                self.x_bds = self.x_bds[1:] + self.x_bds[:-1]

                self._call = self.__class__._call_nearest
            elif kind == 'nearest-up':
                # Do division before addition to prevent possible integer
                # overflow
                self._side = 'right'
                self.x_bds = self.x / 2.0
                self.x_bds = self.x_bds[1:] + self.x_bds[:-1]

                self._call = self.__class__._call_nearest
            elif kind == 'previous':
                # Side for np.searchsorted and index for clipping
                self._side = 'left'
                self._ind = 0
                # Move x by one floating point value to the left
                self._x_shift = np.nextafter(self.x, -np.inf)
                self._call = self.__class__._call_previousnext
            elif kind == 'next':
                self._side = 'right'
                self._ind = 1
                # Move x by one floating point value to the right
                self._x_shift = np.nextafter(self.x, np.inf)
                self._call = self.__class__._call_previousnext
            else:
                # Check if we can delegate to numpy.interp (2x-10x faster).
                cond = self.x.dtype == np.float_ and self.y.dtype == np.float_
                cond = cond and self.y.ndim == 1
                cond = cond and not _do_extrapolate(fill_value)

                if cond:
                    self._call = self.__class__._call_linear_np
                else:
                    self._call = self.__class__._call_linear
        else:
            minval = order + 1

            rewrite_nan = False
            xx, yy = self.x, self._y
            if order > 1:
                # Quadratic or cubic spline. If input contains even a single
                # nan, then the output is all nans. We cannot just feed data
                # with nans to make_interp_spline because it calls LAPACK.
                # So, we make up a bogus x and y with no nans and use it
                # to get the correct shape of the output, which we then fill
                # with nans.
                # For slinear or zero order spline, we just pass nans through.
                mask = np.isnan(self.x)
                if mask.any():
                    sx = self.x[~mask]
                    if sx.size == 0:
                        raise ValueError("`x` array is all-nan")
                    xx = np.linspace(np.nanmin(self.x),
                                     np.nanmax(self.x),
                                     len(self.x))
                    rewrite_nan = True
                if np.isnan(self._y).any():
                    yy = np.ones_like(self._y)
                    rewrite_nan = True

            self._spline = make_interp_spline(xx, yy, k=order,
                                              check_finite=False)
            if rewrite_nan:
                self._call = self.__class__._call_nan_spline
            else:
                self._call = self.__class__._call_spline

        if len(self.x) < minval:
            raise ValueError("x and y arrays must have at "
                             "least %d entries" % minval)

    @property
    def fill_value(self):
        """The fill value."""
        # backwards compat: mimic a public attribute
        return self._fill_value_orig

    @fill_value.setter
    def fill_value(self, fill_value):
        # extrapolation only works for nearest neighbor and linear methods
        if _do_extrapolate(fill_value):
            if self.bounds_error:
                raise ValueError("Cannot extrapolate and raise "
                                 "at the same time.")
            self.bounds_error = False
            self._extrapolate = True
        else:
            broadcast_shape = (self.y.shape[:self.axis] +
                               self.y.shape[self.axis + 1:])
            if len(broadcast_shape) == 0:
                broadcast_shape = (1,)
            # it's either a pair (_below_range, _above_range) or a single value
            # for both above and below range
            if isinstance(fill_value, tuple) and len(fill_value) == 2:
                below_above = [np.asarray(fill_value[0]),
                               np.asarray(fill_value[1])]
                names = ('fill_value (below)', 'fill_value (above)')
                for ii in range(2):
                    below_above[ii] = _check_broadcast_up_to(
                        below_above[ii], broadcast_shape, names[ii])
            else:
                fill_value = np.asarray(fill_value)
                below_above = [_check_broadcast_up_to(
                    fill_value, broadcast_shape, 'fill_value')] * 2
            self._fill_value_below, self._fill_value_above = below_above
            self._extrapolate = False
            if self.bounds_error is None:
                self.bounds_error = True
        # backwards compat: fill_value was a public attr; make it writeable
        self._fill_value_orig = fill_value

    def _call_linear_np(self, x_new):
        # Note that out-of-bounds values are taken care of in self._evaluate
        return np.interp(x_new, self.x, self.y)

    def _call_linear(self, x_new):
        # 2. Find where in the original data, the values to interpolate
        #    would be inserted.
        #    Note: If x_new[n] == x[m], then m is returned by searchsorted.
        x_new_indices = searchsorted(self.x, x_new)

        # 3. Clip x_new_indices so that they are within the range of
        #    self.x indices and at least 1. Removes mis-interpolation
        #    of x_new[n] = x[0]
        x_new_indices = x_new_indices.clip(1, len(self.x)-1).astype(int)

        # 4. Calculate the slope of regions that each x_new value falls in.
        lo = x_new_indices - 1
        hi = x_new_indices

        x_lo = self.x[lo]
        x_hi = self.x[hi]
        y_lo = self._y[lo]
        y_hi = self._y[hi]

        # Note that the following two expressions rely on the specifics of the
        # broadcasting semantics.
        slope = (y_hi - y_lo) / (x_hi - x_lo)[:, None]

        # 5. Calculate the actual value for each entry in x_new.
        y_new = slope*(x_new - x_lo)[:, None] + y_lo

        return y_new

    def _call_nearest(self, x_new):
        """ Find nearest neighbor interpolated y_new = f(x_new)."""

        # 2. Find where in the averaged data the values to interpolate
        #    would be inserted.
        #    Note: use side='left' (right) to searchsorted() to define the
        #    halfway point to be nearest to the left (right) neighbor
        x_new_indices = searchsorted(self.x_bds, x_new, side=self._side)

        # 3. Clip x_new_indices so that they are within the range of x indices.
        x_new_indices = x_new_indices.clip(0, len(self.x)-1).astype(intp)

        # 4. Calculate the actual value for each entry in x_new.
        y_new = self._y[x_new_indices]

        return y_new

    def _call_previousnext(self, x_new):
        """Use previous/next neighbor of x_new, y_new = f(x_new)."""

        # 1. Get index of left/right value
        x_new_indices = searchsorted(self._x_shift, x_new, side=self._side)

        # 2. Clip x_new_indices so that they are within the range of x indices.
        x_new_indices = x_new_indices.clip(1-self._ind,
                                           len(self.x)-self._ind).astype(intp)

        # 3. Calculate the actual value for each entry in x_new.
        y_new = self._y[x_new_indices+self._ind-1]

        return y_new

    def _call_spline(self, x_new):
        return self._spline(x_new)

    def _call_nan_spline(self, x_new):
        out = self._spline(x_new)
        out[...] = np.nan
        return out

    def _evaluate(self, x_new):
        # 1. Handle values in x_new that are outside of x. Throw error,
        #    or return a list of mask array indicating the outofbounds values.
        #    The behavior is set by the bounds_error variable.
        x_new = asarray(x_new)
        y_new = self._call(self, x_new)
        if not self._extrapolate:
            below_bounds, above_bounds = self._check_bounds(x_new)
            if len(y_new) > 0:
                # Note fill_value must be broadcast up to the proper size
                # and flattened to work here
                y_new[below_bounds] = self._fill_value_below
                y_new[above_bounds] = self._fill_value_above
        return y_new

    def _check_bounds(self, x_new):
        """Check the inputs for being in the bounds of the interpolated data.

        Parameters
        ----------
        x_new : array

        Returns
        -------
        out_of_bounds : bool array
            The mask on x_new of values that are out of the bounds.
        """

        # If self.bounds_error is True, we raise an error if any x_new values
        # fall outside the range of x. Otherwise, we return an array indicating
        # which values are outside the boundary region.
        below_bounds = x_new < self.x[0]
        above_bounds = x_new > self.x[-1]

        # !! Could provide more information about which values are out of bounds
        if self.bounds_error and below_bounds.any():
            raise ValueError("A value in x_new is below the interpolation "
                             "range.")
        if self.bounds_error and above_bounds.any():
            raise ValueError("A value in x_new is above the interpolation "
                             "range.")

        # !! Should we emit a warning if some values are out of bounds?
        # !! matlab does not.
        return below_bounds, above_bounds


class _PPolyBase:
    """Base class for piecewise polynomials."""
    __slots__ = ('c', 'x', 'extrapolate', 'axis')

    def __init__(self, c, x, extrapolate=None, axis=0):
        self.c = np.asarray(c)
        self.x = np.ascontiguousarray(x, dtype=np.float64)

        if extrapolate is None:
            extrapolate = True
        elif extrapolate != 'periodic':
            extrapolate = bool(extrapolate)
        self.extrapolate = extrapolate

        if self.c.ndim < 2:
            raise ValueError("Coefficients array must be at least "
                             "2-dimensional.")

        if not (0 <= axis < self.c.ndim - 1):
            raise ValueError("axis=%s must be between 0 and %s" %
                             (axis, self.c.ndim-1))

        self.axis = axis
        if axis != 0:
            # roll the interpolation axis to be the first one in self.c
            # More specifically, the target shape for self.c is (k, m, ...),
            # and axis !=0 means that we have c.shape (..., k, m, ...)
            #                                               ^
            #                                              axis
            # So we roll two of them.
            self.c = np.rollaxis(self.c, axis+1)
            self.c = np.rollaxis(self.c, axis+1)

        if self.x.ndim != 1:
            raise ValueError("x must be 1-dimensional")
        if self.x.size < 2:
            raise ValueError("at least 2 breakpoints are needed")
        if self.c.ndim < 2:
            raise ValueError("c must have at least 2 dimensions")
        if self.c.shape[0] == 0:
            raise ValueError("polynomial must be at least of order 0")
        if self.c.shape[1] != self.x.size-1:
            raise ValueError("number of coefficients != len(x)-1")
        dx = np.diff(self.x)
        if not (np.all(dx >= 0) or np.all(dx <= 0)):
            raise ValueError("`x` must be strictly increasing or decreasing.")

        dtype = self._get_dtype(self.c.dtype)
        self.c = np.ascontiguousarray(self.c, dtype=dtype)

    def _get_dtype(self, dtype):
        if np.issubdtype(dtype, np.complexfloating) \
               or np.issubdtype(self.c.dtype, np.complexfloating):
            return np.complex_
        else:
            return np.float_

    @classmethod
    def construct_fast(cls, c, x, extrapolate=None, axis=0):
        """
        Construct the piecewise polynomial without making checks.

        Takes the same parameters as the constructor. Input arguments
        ``c`` and ``x`` must be arrays of the correct shape and type. The
        ``c`` array can only be of dtypes float and complex, and ``x``
        array must have dtype float.
        """
        self = object.__new__(cls)
        self.c = c
        self.x = x
        self.axis = axis
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = extrapolate
        return self

    def _ensure_c_contiguous(self):
        """
        c and x may be modified by the user. The Cython code expects
        that they are C contiguous.
        """
        if not self.x.flags.c_contiguous:
            self.x = self.x.copy()
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()

    def extend(self, c, x, right=None):
        """
        Add additional breakpoints and coefficients to the polynomial.

        Parameters
        ----------
        c : ndarray, size (k, m, ...)
            Additional coefficients for polynomials in intervals. Note that
            the first additional interval will be formed using one of the
            ``self.x`` end points.
        x : ndarray, size (m,)
            Additional breakpoints. Must be sorted in the same order as
            ``self.x`` and either to the right or to the left of the current
            breakpoints.
        right
            Deprecated argument. Has no effect.

            .. deprecated:: 0.19
        """
        if right is not None:
            warnings.warn("`right` is deprecated and will be removed.")

        c = np.asarray(c)
        x = np.asarray(x)

        if c.ndim < 2:
            raise ValueError("invalid dimensions for c")
        if x.ndim != 1:
            raise ValueError("invalid dimensions for x")
        if x.shape[0] != c.shape[1]:
            raise ValueError("Shapes of x {} and c {} are incompatible"
                             .format(x.shape, c.shape))
        if c.shape[2:] != self.c.shape[2:] or c.ndim != self.c.ndim:
            raise ValueError("Shapes of c {} and self.c {} are incompatible"
                             .format(c.shape, self.c.shape))

        if c.size == 0:
            return

        dx = np.diff(x)
        if not (np.all(dx >= 0) or np.all(dx <= 0)):
            raise ValueError("`x` is not sorted.")

        if self.x[-1] >= self.x[0]:
            if not x[-1] >= x[0]:
                raise ValueError("`x` is in the different order "
                                 "than `self.x`.")

            if x[0] >= self.x[-1]:
                action = 'append'
            elif x[-1] <= self.x[0]:
                action = 'prepend'
            else:
                raise ValueError("`x` is neither on the left or on the right "
                                 "from `self.x`.")
        else:
            if not x[-1] <= x[0]:
                raise ValueError("`x` is in the different order "
                                 "than `self.x`.")

            if x[0] <= self.x[-1]:
                action = 'append'
            elif x[-1] >= self.x[0]:
                action = 'prepend'
            else:
                raise ValueError("`x` is neither on the left or on the right "
                                 "from `self.x`.")

        dtype = self._get_dtype(c.dtype)

        k2 = max(c.shape[0], self.c.shape[0])
        c2 = np.zeros((k2, self.c.shape[1] + c.shape[1]) + self.c.shape[2:],
                      dtype=dtype)

        if action == 'append':
            c2[k2-self.c.shape[0]:, :self.c.shape[1]] = self.c
            c2[k2-c.shape[0]:, self.c.shape[1]:] = c
            self.x = np.r_[self.x, x]
        elif action == 'prepend':
            c2[k2-self.c.shape[0]:, :c.shape[1]] = c
            c2[k2-c.shape[0]:, c.shape[1]:] = self.c
            self.x = np.r_[x, self.x]

        self.c = c2

    def __call__(self, x, nu=0, extrapolate=None):
        """
        Evaluate the piecewise polynomial or its derivative.

        Parameters
        ----------
        x : array_like
            Points to evaluate the interpolant at.
        nu : int, optional
            Order of derivative to evaluate. Must be non-negative.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used.
            If None (default), use `self.extrapolate`.

        Returns
        -------
        y : array_like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.
        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        x = np.asarray(x)
        x_shape, x_ndim = x.shape, x.ndim
        x = np.ascontiguousarray(x.ravel(), dtype=np.float_)

        # With periodic extrapolation we map x to the segment
        # [self.x[0], self.x[-1]].
        if extrapolate == 'periodic':
            x = self.x[0] + (x - self.x[0]) % (self.x[-1] - self.x[0])
            extrapolate = False

        out = np.empty((len(x), prod(self.c.shape[2:])), dtype=self.c.dtype)
        self._ensure_c_contiguous()
        self._evaluate(x, nu, extrapolate, out)
        out = out.reshape(x_shape + self.c.shape[2:])
        if self.axis != 0:
            # transpose to move the calculated values to the interpolation axis
            l = list(range(out.ndim))
            l = l[x_ndim:x_ndim+self.axis] + l[:x_ndim] + l[x_ndim+self.axis:]
            out = out.transpose(l)
        return out


class PPoly(_PPolyBase):
    """
    Piecewise polynomial in terms of coefficients and breakpoints

    The polynomial between ``x[i]`` and ``x[i + 1]`` is written in the
    local power basis::

        S = sum(c[m, i] * (xp - x[i])**(k-m) for m in range(k+1))

    where ``k`` is the degree of the polynomial.

    Parameters
    ----------
    c : ndarray, shape (k, m, ...)
        Polynomial coefficients, order `k` and `m` intervals.
    x : ndarray, shape (m+1,)
        Polynomial breakpoints. Must be sorted in either increasing or
        decreasing order.
    extrapolate : bool or 'periodic', optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. Default is True.
    axis : int, optional
        Interpolation axis. Default is zero.

    Attributes
    ----------
    x : ndarray
        Breakpoints.
    c : ndarray
        Coefficients of the polynomials. They are reshaped
        to a 3-D array with the last dimension representing
        the trailing dimensions of the original coefficient array.
    axis : int
        Interpolation axis.

    Methods
    -------
    __call__
    derivative
    antiderivative
    integrate
    solve
    roots
    extend
    from_spline
    from_bernstein_basis
    construct_fast

    See also
    --------
    BPoly : piecewise polynomials in the Bernstein basis

    Notes
    -----
    High-order polynomials in the power basis can be numerically
    unstable. Precision problems can start to appear for orders
    larger than 20-30.
    """
    def _evaluate(self, x, nu, extrapolate, out):
        _ppoly.evaluate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                        self.x, x, nu, bool(extrapolate), out)

    def derivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the derivative.

        Parameters
        ----------
        nu : int, optional
            Order of derivative to evaluate. Default is 1, i.e., compute the
            first derivative. If negative, the antiderivative is returned.

        Returns
        -------
        pp : PPoly
            Piecewise polynomial of order k2 = k - n representing the derivative
            of this polynomial.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.
        """
        if nu < 0:
            return self.antiderivative(-nu)

        # reduce order
        if nu == 0:
            c2 = self.c.copy()
        else:
            c2 = self.c[:-nu, :].copy()

        if c2.shape[0] == 0:
            # derivative of order 0 is zero
            c2 = np.zeros((1,) + c2.shape[1:], dtype=c2.dtype)

        # multiply by the correct rising factorials
        factor = spec.poch(np.arange(c2.shape[0], 0, -1), nu)
        c2 *= factor[(slice(None),) + (None,)*(c2.ndim-1)]

        # construct a compatible polynomial
        return self.construct_fast(c2, self.x, self.extrapolate, self.axis)

    def antiderivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the antiderivative.

        Antiderivative is also the indefinite integral of the function,
        and derivative is its inverse operation.

        Parameters
        ----------
        nu : int, optional
            Order of antiderivative to evaluate. Default is 1, i.e., compute
            the first integral. If negative, the derivative is returned.

        Returns
        -------
        pp : PPoly
            Piecewise polynomial of order k2 = k + n representing
            the antiderivative of this polynomial.

        Notes
        -----
        The antiderivative returned by this function is continuous and
        continuously differentiable to order n-1, up to floating point
        rounding error.

        If antiderivative is computed and ``self.extrapolate='periodic'``,
        it will be set to False for the returned instance. This is done because
        the antiderivative is no longer periodic and its correct evaluation
        outside of the initially given x interval is difficult.
        """
        if nu <= 0:
            return self.derivative(-nu)

        c = np.zeros((self.c.shape[0] + nu, self.c.shape[1]) + self.c.shape[2:],
                     dtype=self.c.dtype)
        c[:-nu] = self.c

        # divide by the correct rising factorials
        factor = spec.poch(np.arange(self.c.shape[0], 0, -1), nu)
        c[:-nu] /= factor[(slice(None),) + (None,)*(c.ndim-1)]

        # fix continuity of added degrees of freedom
        self._ensure_c_contiguous()
        _ppoly.fix_continuity(c.reshape(c.shape[0], c.shape[1], -1),
                              self.x, nu - 1)

        if self.extrapolate == 'periodic':
            extrapolate = False
        else:
            extrapolate = self.extrapolate

        # construct a compatible polynomial
        return self.construct_fast(c, self.x, extrapolate, self.axis)

    def integrate(self, a, b, extrapolate=None):
        """
        Compute a definite integral over a piecewise polynomial.

        Parameters
        ----------
        a : float
            Lower integration bound
        b : float
            Upper integration bound
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used.
            If None (default), use `self.extrapolate`.

        Returns
        -------
        ig : array_like
            Definite integral of the piecewise polynomial over [a, b]
        """
        if extrapolate is None:
            extrapolate = self.extrapolate

        # Swap integration bounds if needed
        sign = 1
        if b < a:
            a, b = b, a
            sign = -1

        range_int = np.empty((prod(self.c.shape[2:]),), dtype=self.c.dtype)
        self._ensure_c_contiguous()

        # Compute the integral.
        if extrapolate == 'periodic':
            # Split the integral into the part over period (can be several
            # of them) and the remaining part.

            xs, xe = self.x[0], self.x[-1]
            period = xe - xs
            interval = b - a
            n_periods, left = divmod(interval, period)

            if n_periods > 0:
                _ppoly.integrate(
                    self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                    self.x, xs, xe, False, out=range_int)
                range_int *= n_periods
            else:
                range_int.fill(0)

            # Map a to [xs, xe], b is always a + left.
            a = xs + (a - xs) % period
            b = a + left

            # If b <= xe then we need to integrate over [a, b], otherwise
            # over [a, xe] and from xs to what is remained.
            remainder_int = np.empty_like(range_int)
            if b <= xe:
                _ppoly.integrate(
                    self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                    self.x, a, b, False, out=remainder_int)
                range_int += remainder_int
            else:
                _ppoly.integrate(
                    self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                    self.x, a, xe, False, out=remainder_int)
                range_int += remainder_int

                _ppoly.integrate(
                    self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                    self.x, xs, xs + left + a - xe, False, out=remainder_int)
                range_int += remainder_int
        else:
            _ppoly.integrate(
                self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                self.x, a, b, bool(extrapolate), out=range_int)

        # Return
        range_int *= sign
        return range_int.reshape(self.c.shape[2:])

    def solve(self, y=0., discontinuity=True, extrapolate=None):
        """
        Find real solutions of the the equation ``pp(x) == y``.

        Parameters
        ----------
        y : float, optional
            Right-hand side. Default is zero.
        discontinuity : bool, optional
            Whether to report sign changes across discontinuities at
            breakpoints as roots.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to return roots from the polynomial
            extrapolated based on first and last intervals, 'periodic' works
            the same as False. If None (default), use `self.extrapolate`.

        Returns
        -------
        roots : ndarray
            Roots of the polynomial(s).

            If the PPoly object describes multiple polynomials, the
            return value is an object array whose each element is an
            ndarray containing the roots.

        Notes
        -----
        This routine works only on real-valued polynomials.

        If the piecewise polynomial contains sections that are
        identically zero, the root list will contain the start point
        of the corresponding interval, followed by a ``nan`` value.

        If the polynomial is discontinuous across a breakpoint, and
        there is a sign change across the breakpoint, this is reported
        if the `discont` parameter is True.

        Examples
        --------

        Finding roots of ``[x**2 - 1, (x - 1)**2]`` defined on intervals
        ``[-2, 1], [1, 2]``:

        >>> from scipy.interpolate import PPoly
        >>> pp = PPoly(np.array([[1, -4, 3], [1, 0, 0]]).T, [-2, 1, 2])
        >>> pp.solve()
        array([-1.,  1.])
        """
        if extrapolate is None:
            extrapolate = self.extrapolate

        self._ensure_c_contiguous()

        if np.issubdtype(self.c.dtype, np.complexfloating):
            raise ValueError("Root finding is only for "
                             "real-valued polynomials")

        y = float(y)
        r = _ppoly.real_roots(self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                              self.x, y, bool(discontinuity),
                              bool(extrapolate))
        if self.c.ndim == 2:
            return r[0]
        else:
            r2 = np.empty(prod(self.c.shape[2:]), dtype=object)
            # this for-loop is equivalent to ``r2[...] = r``, but that's broken
            # in NumPy 1.6.0
            for ii, root in enumerate(r):
                r2[ii] = root

            return r2.reshape(self.c.shape[2:])

    def roots(self, discontinuity=True, extrapolate=None):
        """
        Find real roots of the the piecewise polynomial.

        Parameters
        ----------
        discontinuity : bool, optional
            Whether to report sign changes across discontinuities at
            breakpoints as roots.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to return roots from the polynomial
            extrapolated based on first and last intervals, 'periodic' works
            the same as False. If None (default), use `self.extrapolate`.

        Returns
        -------
        roots : ndarray
            Roots of the polynomial(s).

            If the PPoly object describes multiple polynomials, the
            return value is an object array whose each element is an
            ndarray containing the roots.

        See Also
        --------
        PPoly.solve
        """
        return self.solve(0, discontinuity, extrapolate)

    @classmethod
    def from_spline(cls, tck, extrapolate=None):
        """
        Construct a piecewise polynomial from a spline

        Parameters
        ----------
        tck
            A spline, as returned by `splrep` or a BSpline object.
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.
        """
        if isinstance(tck, BSpline):
            t, c, k = tck.tck
            if extrapolate is None:
                extrapolate = tck.extrapolate
        else:
            t, c, k = tck

        cvals = np.empty((k + 1, len(t)-1), dtype=c.dtype)
        for m in range(k, -1, -1):
            y = fitpack.splev(t[:-1], tck, der=m)
            cvals[k - m, :] = y/spec.gamma(m+1)

        return cls.construct_fast(cvals, t, extrapolate)

    @classmethod
    def from_bernstein_basis(cls, bp, extrapolate=None):
        """
        Construct a piecewise polynomial in the power basis
        from a polynomial in Bernstein basis.

        Parameters
        ----------
        bp : BPoly
            A Bernstein basis polynomial, as created by BPoly
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.
        """
        if not isinstance(bp, BPoly):
            raise TypeError(".from_bernstein_basis only accepts BPoly instances. "
                            "Got %s instead." % type(bp))

        dx = np.diff(bp.x)
        k = bp.c.shape[0] - 1  # polynomial order

        rest = (None,)*(bp.c.ndim-2)

        c = np.zeros_like(bp.c)
        for a in range(k+1):
            factor = (-1)**a * comb(k, a) * bp.c[a]
            for s in range(a, k+1):
                val = comb(k-a, s-a) * (-1)**s
                c[k-s] += factor * val / dx[(slice(None),)+rest]**s

        if extrapolate is None:
            extrapolate = bp.extrapolate

        return cls.construct_fast(c, bp.x, extrapolate, bp.axis)


class BPoly(_PPolyBase):
    """Piecewise polynomial in terms of coefficients and breakpoints.

    The polynomial between ``x[i]`` and ``x[i + 1]`` is written in the
    Bernstein polynomial basis::

        S = sum(c[a, i] * b(a, k; x) for a in range(k+1)),

    where ``k`` is the degree of the polynomial, and::

        b(a, k; x) = binom(k, a) * t**a * (1 - t)**(k - a),

    with ``t = (x - x[i]) / (x[i+1] - x[i])`` and ``binom`` is the binomial
    coefficient.

    Parameters
    ----------
    c : ndarray, shape (k, m, ...)
        Polynomial coefficients, order `k` and `m` intervals
    x : ndarray, shape (m+1,)
        Polynomial breakpoints. Must be sorted in either increasing or
        decreasing order.
    extrapolate : bool, optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. Default is True.
    axis : int, optional
        Interpolation axis. Default is zero.

    Attributes
    ----------
    x : ndarray
        Breakpoints.
    c : ndarray
        Coefficients of the polynomials. They are reshaped
        to a 3-D array with the last dimension representing
        the trailing dimensions of the original coefficient array.
    axis : int
        Interpolation axis.

    Methods
    -------
    __call__
    extend
    derivative
    antiderivative
    integrate
    construct_fast
    from_power_basis
    from_derivatives

    See also
    --------
    PPoly : piecewise polynomials in the power basis

    Notes
    -----
    Properties of Bernstein polynomials are well documented in the literature,
    see for example [1]_ [2]_ [3]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Bernstein_polynomial

    .. [2] Kenneth I. Joy, Bernstein polynomials,
       http://www.idav.ucdavis.edu/education/CAGDNotes/Bernstein-Polynomials.pdf

    .. [3] E. H. Doha, A. H. Bhrawy, and M. A. Saker, Boundary Value Problems,
           vol 2011, article ID 829546, :doi:`10.1155/2011/829543`.

    Examples
    --------
    >>> from scipy.interpolate import BPoly
    >>> x = [0, 1]
    >>> c = [[1], [2], [3]]
    >>> bp = BPoly(c, x)

    This creates a 2nd order polynomial

    .. math::

        B(x) = 1 \\times b_{0, 2}(x) + 2 \\times b_{1, 2}(x) + 3 \\times b_{2, 2}(x) \\\\
             = 1 \\times (1-x)^2 + 2 \\times 2 x (1 - x) + 3 \\times x^2

    """

    def _evaluate(self, x, nu, extrapolate, out):
        _ppoly.evaluate_bernstein(
            self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
            self.x, x, nu, bool(extrapolate), out)

    def derivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the derivative.

        Parameters
        ----------
        nu : int, optional
            Order of derivative to evaluate. Default is 1, i.e., compute the
            first derivative. If negative, the antiderivative is returned.

        Returns
        -------
        bp : BPoly
            Piecewise polynomial of order k - nu representing the derivative of
            this polynomial.

        """
        if nu < 0:
            return self.antiderivative(-nu)

        if nu > 1:
            bp = self
            for k in range(nu):
                bp = bp.derivative()
            return bp

        # reduce order
        if nu == 0:
            c2 = self.c.copy()
        else:
            # For a polynomial
            #    B(x) = \sum_{a=0}^{k} c_a b_{a, k}(x),
            # we use the fact that
            #   b'_{a, k} = k ( b_{a-1, k-1} - b_{a, k-1} ),
            # which leads to
            #   B'(x) = \sum_{a=0}^{k-1} (c_{a+1} - c_a) b_{a, k-1}
            #
            # finally, for an interval [y, y + dy] with dy != 1,
            # we need to correct for an extra power of dy

            rest = (None,)*(self.c.ndim-2)

            k = self.c.shape[0] - 1
            dx = np.diff(self.x)[(None, slice(None))+rest]
            c2 = k * np.diff(self.c, axis=0) / dx

        if c2.shape[0] == 0:
            # derivative of order 0 is zero
            c2 = np.zeros((1,) + c2.shape[1:], dtype=c2.dtype)

        # construct a compatible polynomial
        return self.construct_fast(c2, self.x, self.extrapolate, self.axis)

    def antiderivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the antiderivative.

        Parameters
        ----------
        nu : int, optional
            Order of antiderivative to evaluate. Default is 1, i.e., compute
            the first integral. If negative, the derivative is returned.

        Returns
        -------
        bp : BPoly
            Piecewise polynomial of order k + nu representing the
            antiderivative of this polynomial.

        Notes
        -----
        If antiderivative is computed and ``self.extrapolate='periodic'``,
        it will be set to False for the returned instance. This is done because
        the antiderivative is no longer periodic and its correct evaluation
        outside of the initially given x interval is difficult.
        """
        if nu <= 0:
            return self.derivative(-nu)

        if nu > 1:
            bp = self
            for k in range(nu):
                bp = bp.antiderivative()
            return bp

        # Construct the indefinite integrals on individual intervals
        c, x = self.c, self.x
        k = c.shape[0]
        c2 = np.zeros((k+1,) + c.shape[1:], dtype=c.dtype)

        c2[1:, ...] = np.cumsum(c, axis=0) / k
        delta = x[1:] - x[:-1]
        c2 *= delta[(None, slice(None)) + (None,)*(c.ndim-2)]

        # Now fix continuity: on the very first interval, take the integration
        # constant to be zero; on an interval [x_j, x_{j+1}) with j>0,
        # the integration constant is then equal to the jump of the `bp` at x_j.
        # The latter is given by the coefficient of B_{n+1, n+1}
        # *on the previous interval* (other B. polynomials are zero at the
        # breakpoint). Finally, use the fact that BPs form a partition of unity.
        c2[:,1:] += np.cumsum(c2[k, :], axis=0)[:-1]

        if self.extrapolate == 'periodic':
            extrapolate = False
        else:
            extrapolate = self.extrapolate

        return self.construct_fast(c2, x, extrapolate, axis=self.axis)

    def integrate(self, a, b, extrapolate=None):
        """
        Compute a definite integral over a piecewise polynomial.

        Parameters
        ----------
        a : float
            Lower integration bound
        b : float
            Upper integration bound
        extrapolate : {bool, 'periodic', None}, optional
            Whether to extrapolate to out-of-bounds points based on first
            and last intervals, or to return NaNs. If 'periodic', periodic
            extrapolation is used. If None (default), use `self.extrapolate`.

        Returns
        -------
        array_like
            Definite integral of the piecewise polynomial over [a, b]

        """
        # XXX: can probably use instead the fact that
        # \int_0^{1} B_{j, n}(x) \dx = 1/(n+1)
        ib = self.antiderivative()
        if extrapolate is None:
            extrapolate = self.extrapolate

        # ib.extrapolate shouldn't be 'periodic', it is converted to
        # False for 'periodic. in antiderivative() call.
        if extrapolate != 'periodic':
            ib.extrapolate = extrapolate

        if extrapolate == 'periodic':
            # Split the integral into the part over period (can be several
            # of them) and the remaining part.

            # For simplicity and clarity convert to a <= b case.
            if a <= b:
                sign = 1
            else:
                a, b = b, a
                sign = -1

            xs, xe = self.x[0], self.x[-1]
            period = xe - xs
            interval = b - a
            n_periods, left = divmod(interval, period)
            res = n_periods * (ib(xe) - ib(xs))

            # Map a and b to [xs, xe].
            a = xs + (a - xs) % period
            b = a + left

            # If b <= xe then we need to integrate over [a, b], otherwise
            # over [a, xe] and from xs to what is remained.
            if b <= xe:
                res += ib(b) - ib(a)
            else:
                res += ib(xe) - ib(a) + ib(xs + left + a - xe) - ib(xs)

            return sign * res
        else:
            return ib(b) - ib(a)

    def extend(self, c, x, right=None):
        k = max(self.c.shape[0], c.shape[0])
        self.c = self._raise_degree(self.c, k - self.c.shape[0])
        c = self._raise_degree(c, k - c.shape[0])
        return _PPolyBase.extend(self, c, x, right)
    extend.__doc__ = _PPolyBase.extend.__doc__

    @classmethod
    def from_power_basis(cls, pp, extrapolate=None):
        """
        Construct a piecewise polynomial in Bernstein basis
        from a power basis polynomial.

        Parameters
        ----------
        pp : PPoly
            A piecewise polynomial in the power basis
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.
        """
        if not isinstance(pp, PPoly):
            raise TypeError(".from_power_basis only accepts PPoly instances. "
                            "Got %s instead." % type(pp))

        dx = np.diff(pp.x)
        k = pp.c.shape[0] - 1   # polynomial order

        rest = (None,)*(pp.c.ndim-2)

        c = np.zeros_like(pp.c)
        for a in range(k+1):
            factor = pp.c[a] / comb(k, k-a) * dx[(slice(None),)+rest]**(k-a)
            for j in range(k-a, k+1):
                c[j] += factor * comb(j, k-a)

        if extrapolate is None:
            extrapolate = pp.extrapolate

        return cls.construct_fast(c, pp.x, extrapolate, pp.axis)

    @classmethod
    def from_derivatives(cls, xi, yi, orders=None, extrapolate=None):
        """Construct a piecewise polynomial in the Bernstein basis,
        compatible with the specified values and derivatives at breakpoints.

        Parameters
        ----------
        xi : array_like
            sorted 1-D array of x-coordinates
        yi : array_like or list of array_likes
            ``yi[i][j]`` is the ``j``th derivative known at ``xi[i]``
        orders : None or int or array_like of ints. Default: None.
            Specifies the degree of local polynomials. If not None, some
            derivatives are ignored.
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.

        Notes
        -----
        If ``k`` derivatives are specified at a breakpoint ``x``, the
        constructed polynomial is exactly ``k`` times continuously
        differentiable at ``x``, unless the ``order`` is provided explicitly.
        In the latter case, the smoothness of the polynomial at
        the breakpoint is controlled by the ``order``.

        Deduces the number of derivatives to match at each end
        from ``order`` and the number of derivatives available. If
        possible it uses the same number of derivatives from
        each end; if the number is odd it tries to take the
        extra one from y2. In any case if not enough derivatives
        are available at one end or another it draws enough to
        make up the total from the other end.

        If the order is too high and not enough derivatives are available,
        an exception is raised.

        Examples
        --------

        >>> from scipy.interpolate import BPoly
        >>> BPoly.from_derivatives([0, 1], [[1, 2], [3, 4]])

        Creates a polynomial `f(x)` of degree 3, defined on `[0, 1]`
        such that `f(0) = 1, df/dx(0) = 2, f(1) = 3, df/dx(1) = 4`

        >>> BPoly.from_derivatives([0, 1, 2], [[0, 1], [0], [2]])

        Creates a piecewise polynomial `f(x)`, such that
        `f(0) = f(1) = 0`, `f(2) = 2`, and `df/dx(0) = 1`.
        Based on the number of derivatives provided, the order of the
        local polynomials is 2 on `[0, 1]` and 1 on `[1, 2]`.
        Notice that no restriction is imposed on the derivatives at
        ``x = 1`` and ``x = 2``.

        Indeed, the explicit form of the polynomial is::

            f(x) = | x * (1 - x),  0 <= x < 1
                   | 2 * (x - 1),  1 <= x <= 2

        So that f'(1-0) = -1 and f'(1+0) = 2

        """
        xi = np.asarray(xi)
        if len(xi) != len(yi):
            raise ValueError("xi and yi need to have the same length")
        if np.any(xi[1:] - xi[:1] <= 0):
            raise ValueError("x coordinates are not in increasing order")

        # number of intervals
        m = len(xi) - 1

        # global poly order is k-1, local orders are <=k and can vary
        try:
            k = max(len(yi[i]) + len(yi[i+1]) for i in range(m))
        except TypeError as e:
            raise ValueError(
                "Using a 1-D array for y? Please .reshape(-1, 1)."
            ) from e

        if orders is None:
            orders = [None] * m
        else:
            if isinstance(orders, (int, np.integer)):
                orders = [orders] * m
            k = max(k, max(orders))

            if any(o <= 0 for o in orders):
                raise ValueError("Orders must be positive.")

        c = []
        for i in range(m):
            y1, y2 = yi[i], yi[i+1]
            if orders[i] is None:
                n1, n2 = len(y1), len(y2)
            else:
                n = orders[i]+1
                n1 = min(n//2, len(y1))
                n2 = min(n - n1, len(y2))
                n1 = min(n - n2, len(y2))
                if n1+n2 != n:
                    mesg = ("Point %g has %d derivatives, point %g"
                            " has %d derivatives, but order %d requested" % (
                               xi[i], len(y1), xi[i+1], len(y2), orders[i]))
                    raise ValueError(mesg)

                if not (n1 <= len(y1) and n2 <= len(y2)):
                    raise ValueError("`order` input incompatible with"
                                     " length y1 or y2.")

            b = BPoly._construct_from_derivatives(xi[i], xi[i+1],
                                                  y1[:n1], y2[:n2])
            if len(b) < k:
                b = BPoly._raise_degree(b, k - len(b))
            c.append(b)

        c = np.asarray(c)
        return cls(c.swapaxes(0, 1), xi, extrapolate)

    @staticmethod
    def _construct_from_derivatives(xa, xb, ya, yb):
        r"""Compute the coefficients of a polynomial in the Bernstein basis
        given the values and derivatives at the edges.

        Return the coefficients of a polynomial in the Bernstein basis
        defined on ``[xa, xb]`` and having the values and derivatives at the
        endpoints `xa` and `xb` as specified by `ya`` and `yb`.
        The polynomial constructed is of the minimal possible degree, i.e.,
        if the lengths of `ya` and `yb` are `na` and `nb`, the degree
        of the polynomial is ``na + nb - 1``.

        Parameters
        ----------
        xa : float
            Left-hand end point of the interval
        xb : float
            Right-hand end point of the interval
        ya : array_like
            Derivatives at `xa`. `ya[0]` is the value of the function, and
            `ya[i]` for ``i > 0`` is the value of the ``i``th derivative.
        yb : array_like
            Derivatives at `xb`.

        Returns
        -------
        array
            coefficient array of a polynomial having specified derivatives

        Notes
        -----
        This uses several facts from life of Bernstein basis functions.
        First of all,

            .. math:: b'_{a, n} = n (b_{a-1, n-1} - b_{a, n-1})

        If B(x) is a linear combination of the form

            .. math:: B(x) = \sum_{a=0}^{n} c_a b_{a, n},

        then :math: B'(x) = n \sum_{a=0}^{n-1} (c_{a+1} - c_{a}) b_{a, n-1}.
        Iterating the latter one, one finds for the q-th derivative

            .. math:: B^{q}(x) = n!/(n-q)! \sum_{a=0}^{n-q} Q_a b_{a, n-q},

        with

          .. math:: Q_a = \sum_{j=0}^{q} (-)^{j+q} comb(q, j) c_{j+a}

        This way, only `a=0` contributes to :math: `B^{q}(x = xa)`, and
        `c_q` are found one by one by iterating `q = 0, ..., na`.

        At ``x = xb`` it's the same with ``a = n - q``.

        """
        ya, yb = np.asarray(ya), np.asarray(yb)
        if ya.shape[1:] != yb.shape[1:]:
            raise ValueError('Shapes of ya {} and yb {} are incompatible'
                             .format(ya.shape, yb.shape))

        dta, dtb = ya.dtype, yb.dtype
        if (np.issubdtype(dta, np.complexfloating) or
               np.issubdtype(dtb, np.complexfloating)):
            dt = np.complex_
        else:
            dt = np.float_

        na, nb = len(ya), len(yb)
        n = na + nb

        c = np.empty((na+nb,) + ya.shape[1:], dtype=dt)

        # compute coefficients of a polynomial degree na+nb-1
        # walk left-to-right
        for q in range(0, na):
            c[q] = ya[q] / spec.poch(n - q, q) * (xb - xa)**q
            for j in range(0, q):
                c[q] -= (-1)**(j+q) * comb(q, j) * c[j]

        # now walk right-to-left
        for q in range(0, nb):
            c[-q-1] = yb[q] / spec.poch(n - q, q) * (-1)**q * (xb - xa)**q
            for j in range(0, q):
                c[-q-1] -= (-1)**(j+1) * comb(q, j+1) * c[-q+j]

        return c

    @staticmethod
    def _raise_degree(c, d):
        r"""Raise a degree of a polynomial in the Bernstein basis.

        Given the coefficients of a polynomial degree `k`, return (the
        coefficients of) the equivalent polynomial of degree `k+d`.

        Parameters
        ----------
        c : array_like
            coefficient array, 1-D
        d : integer

        Returns
        -------
        array
            coefficient array, 1-D array of length `c.shape[0] + d`

        Notes
        -----
        This uses the fact that a Bernstein polynomial `b_{a, k}` can be
        identically represented as a linear combination of polynomials of
        a higher degree `k+d`:

            .. math:: b_{a, k} = comb(k, a) \sum_{j=0}^{d} b_{a+j, k+d} \
                                 comb(d, j) / comb(k+d, a+j)

        """
        if d == 0:
            return c

        k = c.shape[0] - 1
        out = np.zeros((c.shape[0] + d,) + c.shape[1:], dtype=c.dtype)

        for a in range(c.shape[0]):
            f = c[a] * comb(k, a)
            for j in range(d+1):
                out[a+j] += f * comb(d, j) / comb(k+d, a+j)
        return out


class NdPPoly:
    """
    Piecewise tensor product polynomial

    The value at point ``xp = (x', y', z', ...)`` is evaluated by first
    computing the interval indices `i` such that::

        x[0][i[0]] <= x' < x[0][i[0]+1]
        x[1][i[1]] <= y' < x[1][i[1]+1]
        ...

    and then computing::

        S = sum(c[k0-m0-1,...,kn-mn-1,i[0],...,i[n]]
                * (xp[0] - x[0][i[0]])**m0
                * ...
                * (xp[n] - x[n][i[n]])**mn
                for m0 in range(k[0]+1)
                ...
                for mn in range(k[n]+1))

    where ``k[j]`` is the degree of the polynomial in dimension j. This
    representation is the piecewise multivariate power basis.

    Parameters
    ----------
    c : ndarray, shape (k0, ..., kn, m0, ..., mn, ...)
        Polynomial coefficients, with polynomial order `kj` and
        `mj+1` intervals for each dimension `j`.
    x : ndim-tuple of ndarrays, shapes (mj+1,)
        Polynomial breakpoints for each dimension. These must be
        sorted in increasing order.
    extrapolate : bool, optional
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs. Default: True.

    Attributes
    ----------
    x : tuple of ndarrays
        Breakpoints.
    c : ndarray
        Coefficients of the polynomials.

    Methods
    -------
    __call__
    derivative
    antiderivative
    integrate
    integrate_1d
    construct_fast

    See also
    --------
    PPoly : piecewise polynomials in 1D

    Notes
    -----
    High-order polynomials in the power basis can be numerically
    unstable.

    """

    def __init__(self, c, x, extrapolate=None):
        self.x = tuple(np.ascontiguousarray(v, dtype=np.float64) for v in x)
        self.c = np.asarray(c)
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = bool(extrapolate)

        ndim = len(self.x)
        if any(v.ndim != 1 for v in self.x):
            raise ValueError("x arrays must all be 1-dimensional")
        if any(v.size < 2 for v in self.x):
            raise ValueError("x arrays must all contain at least 2 points")
        if c.ndim < 2*ndim:
            raise ValueError("c must have at least 2*len(x) dimensions")
        if any(np.any(v[1:] - v[:-1] < 0) for v in self.x):
            raise ValueError("x-coordinates are not in increasing order")
        if any(a != b.size - 1 for a, b in zip(c.shape[ndim:2*ndim], self.x)):
            raise ValueError("x and c do not agree on the number of intervals")

        dtype = self._get_dtype(self.c.dtype)
        self.c = np.ascontiguousarray(self.c, dtype=dtype)

    @classmethod
    def construct_fast(cls, c, x, extrapolate=None):
        """
        Construct the piecewise polynomial without making checks.

        Takes the same parameters as the constructor. Input arguments
        ``c`` and ``x`` must be arrays of the correct shape and type.  The
        ``c`` array can only be of dtypes float and complex, and ``x``
        array must have dtype float.

        """
        self = object.__new__(cls)
        self.c = c
        self.x = x
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = extrapolate
        return self

    def _get_dtype(self, dtype):
        if np.issubdtype(dtype, np.complexfloating) \
               or np.issubdtype(self.c.dtype, np.complexfloating):
            return np.complex_
        else:
            return np.float_

    def _ensure_c_contiguous(self):
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()
        if not isinstance(self.x, tuple):
            self.x = tuple(self.x)

    def __call__(self, x, nu=None, extrapolate=None):
        """
        Evaluate the piecewise polynomial or its derivative

        Parameters
        ----------
        x : array-like
            Points to evaluate the interpolant at.
        nu : tuple, optional
            Orders of derivatives to evaluate. Each must be non-negative.
        extrapolate : bool, optional
            Whether to extrapolate to out-of-bounds points based on first
            and last intervals, or to return NaNs.

        Returns
        -------
        y : array-like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.

        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        else:
            extrapolate = bool(extrapolate)

        ndim = len(self.x)

        x = _ndim_coords_from_arrays(x)
        x_shape = x.shape
        x = np.ascontiguousarray(x.reshape(-1, x.shape[-1]), dtype=np.float_)

        if nu is None:
            nu = np.zeros((ndim,), dtype=np.intc)
        else:
            nu = np.asarray(nu, dtype=np.intc)
            if nu.ndim != 1 or nu.shape[0] != ndim:
                raise ValueError("invalid number of derivative orders nu")

        dim1 = prod(self.c.shape[:ndim])
        dim2 = prod(self.c.shape[ndim:2*ndim])
        dim3 = prod(self.c.shape[2*ndim:])
        ks = np.array(self.c.shape[:ndim], dtype=np.intc)

        out = np.empty((x.shape[0], dim3), dtype=self.c.dtype)
        self._ensure_c_contiguous()

        _ppoly.evaluate_nd(self.c.reshape(dim1, dim2, dim3),
                           self.x,
                           ks,
                           x,
                           nu,
                           bool(extrapolate),
                           out)

        return out.reshape(x_shape[:-1] + self.c.shape[2*ndim:])

    def _derivative_inplace(self, nu, axis):
        """
        Compute 1-D derivative along a selected dimension in-place
        May result to non-contiguous c array.
        """
        if nu < 0:
            return self._antiderivative_inplace(-nu, axis)

        ndim = len(self.x)
        axis = axis % ndim

        # reduce order
        if nu == 0:
            # noop
            return
        else:
            sl = [slice(None)]*ndim
            sl[axis] = slice(None, -nu, None)
            c2 = self.c[tuple(sl)]

        if c2.shape[axis] == 0:
            # derivative of order 0 is zero
            shp = list(c2.shape)
            shp[axis] = 1
            c2 = np.zeros(shp, dtype=c2.dtype)

        # multiply by the correct rising factorials
        factor = spec.poch(np.arange(c2.shape[axis], 0, -1), nu)
        sl = [None]*c2.ndim
        sl[axis] = slice(None)
        c2 *= factor[tuple(sl)]

        self.c = c2

    def _antiderivative_inplace(self, nu, axis):
        """
        Compute 1-D antiderivative along a selected dimension
        May result to non-contiguous c array.
        """
        if nu <= 0:
            return self._derivative_inplace(-nu, axis)

        ndim = len(self.x)
        axis = axis % ndim

        perm = list(range(ndim))
        perm[0], perm[axis] = perm[axis], perm[0]
        perm = perm + list(range(ndim, self.c.ndim))

        c = self.c.transpose(perm)

        c2 = np.zeros((c.shape[0] + nu,) + c.shape[1:],
                     dtype=c.dtype)
        c2[:-nu] = c

        # divide by the correct rising factorials
        factor = spec.poch(np.arange(c.shape[0], 0, -1), nu)
        c2[:-nu] /= factor[(slice(None),) + (None,)*(c.ndim-1)]

        # fix continuity of added degrees of freedom
        perm2 = list(range(c2.ndim))
        perm2[1], perm2[ndim+axis] = perm2[ndim+axis], perm2[1]

        c2 = c2.transpose(perm2)
        c2 = c2.copy()
        _ppoly.fix_continuity(c2.reshape(c2.shape[0], c2.shape[1], -1),
                              self.x[axis], nu-1)

        c2 = c2.transpose(perm2)
        c2 = c2.transpose(perm)

        # Done
        self.c = c2

    def derivative(self, nu):
        """
        Construct a new piecewise polynomial representing the derivative.

        Parameters
        ----------
        nu : ndim-tuple of int
            Order of derivatives to evaluate for each dimension.
            If negative, the antiderivative is returned.

        Returns
        -------
        pp : NdPPoly
            Piecewise polynomial of orders (k[0] - nu[0], ..., k[n] - nu[n])
            representing the derivative of this polynomial.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals in each dimension are
        considered half-open, ``[a, b)``, except for the last interval
        which is closed ``[a, b]``.

        """
        p = self.construct_fast(self.c.copy(), self.x, self.extrapolate)

        for axis, n in enumerate(nu):
            p._derivative_inplace(n, axis)

        p._ensure_c_contiguous()
        return p

    def antiderivative(self, nu):
        """
        Construct a new piecewise polynomial representing the antiderivative.

        Antiderivative is also the indefinite integral of the function,
        and derivative is its inverse operation.

        Parameters
        ----------
        nu : ndim-tuple of int
            Order of derivatives to evaluate for each dimension.
            If negative, the derivative is returned.

        Returns
        -------
        pp : PPoly
            Piecewise polynomial of order k2 = k + n representing
            the antiderivative of this polynomial.

        Notes
        -----
        The antiderivative returned by this function is continuous and
        continuously differentiable to order n-1, up to floating point
        rounding error.

        """
        p = self.construct_fast(self.c.copy(), self.x, self.extrapolate)

        for axis, n in enumerate(nu):
            p._antiderivative_inplace(n, axis)

        p._ensure_c_contiguous()
        return p

    def integrate_1d(self, a, b, axis, extrapolate=None):
        r"""
        Compute NdPPoly representation for one dimensional definite integral

        The result is a piecewise polynomial representing the integral:

        .. math::

           p(y, z, ...) = \int_a^b dx\, p(x, y, z, ...)

        where the dimension integrated over is specified with the
        `axis` parameter.

        Parameters
        ----------
        a, b : float
            Lower and upper bound for integration.
        axis : int
            Dimension over which to compute the 1-D integrals
        extrapolate : bool, optional
            Whether to extrapolate to out-of-bounds points based on first
            and last intervals, or to return NaNs.

        Returns
        -------
        ig : NdPPoly or array-like
            Definite integral of the piecewise polynomial over [a, b].
            If the polynomial was 1D, an array is returned,
            otherwise, an NdPPoly object.

        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        else:
            extrapolate = bool(extrapolate)

        ndim = len(self.x)
        axis = int(axis) % ndim

        # reuse 1-D integration routines
        c = self.c
        swap = list(range(c.ndim))
        swap.insert(0, swap[axis])
        del swap[axis + 1]
        swap.insert(1, swap[ndim + axis])
        del swap[ndim + axis + 1]

        c = c.transpose(swap)
        p = PPoly.construct_fast(c.reshape(c.shape[0], c.shape[1], -1),
                                 self.x[axis],
                                 extrapolate=extrapolate)
        out = p.integrate(a, b, extrapolate=extrapolate)

        # Construct result
        if ndim == 1:
            return out.reshape(c.shape[2:])
        else:
            c = out.reshape(c.shape[2:])
            x = self.x[:axis] + self.x[axis+1:]
            return self.construct_fast(c, x, extrapolate=extrapolate)

    def integrate(self, ranges, extrapolate=None):
        """
        Compute a definite integral over a piecewise polynomial.

        Parameters
        ----------
        ranges : ndim-tuple of 2-tuples float
            Sequence of lower and upper bounds for each dimension,
            ``[(a[0], b[0]), ..., (a[ndim-1], b[ndim-1])]``
        extrapolate : bool, optional
            Whether to extrapolate to out-of-bounds points based on first
            and last intervals, or to return NaNs.

        Returns
        -------
        ig : array_like
            Definite integral of the piecewise polynomial over
            [a[0], b[0]] x ... x [a[ndim-1], b[ndim-1]]

        """

        ndim = len(self.x)

        if extrapolate is None:
            extrapolate = self.extrapolate
        else:
            extrapolate = bool(extrapolate)

        if not hasattr(ranges, '__len__') or len(ranges) != ndim:
            raise ValueError("Range not a sequence of correct length")

        self._ensure_c_contiguous()

        # Reuse 1D integration routine
        c = self.c
        for n, (a, b) in enumerate(ranges):
            swap = list(range(c.ndim))
            swap.insert(1, swap[ndim - n])
            del swap[ndim - n + 1]

            c = c.transpose(swap)

            p = PPoly.construct_fast(c, self.x[n], extrapolate=extrapolate)
            out = p.integrate(a, b, extrapolate=extrapolate)
            c = out.reshape(c.shape[2:])

        return c


class RegularGridInterpolator:
    """
    Interpolation on a regular grid in arbitrary dimensions

    The data must be defined on a regular grid; the grid spacing however may be
    uneven. Linear and nearest-neighbor interpolation are supported. After
    setting up the interpolator object, the interpolation method (*linear* or
    *nearest*) may be chosen at each evaluation.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.

    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.

    method : str, optional
        The method of interpolation to perform. Supported are "linear" and
        "nearest". This parameter will become the default for the object's
        ``__call__`` method. Default is "linear".

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.

    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.

    Methods
    -------
    __call__

    Notes
    -----
    Contrary to LinearNDInterpolator and NearestNDInterpolator, this class
    avoids expensive triangulation of the input data by taking advantage of the
    regular grid structure.

    If any of `points` have a dimension of size 1, linear interpolation will
    return an array of `nan` values. Nearest-neighbor interpolation will work
    as usual in this case.

    .. versionadded:: 0.14

    Examples
    --------
    Evaluate a simple example function on the points of a 3-D grid:

    >>> from scipy.interpolate import RegularGridInterpolator
    >>> def f(x, y, z):
    ...     return 2 * x**3 + 3 * y**2 - z
    >>> x = np.linspace(1, 4, 11)
    >>> y = np.linspace(4, 7, 22)
    >>> z = np.linspace(7, 9, 33)
    >>> xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
    >>> data = f(xg, yg, zg)

    ``data`` is now a 3-D array with ``data[i,j,k] = f(x[i], y[j], z[k])``.
    Next, define an interpolating function from this data:

    >>> my_interpolating_function = RegularGridInterpolator((x, y, z), data)

    Evaluate the interpolating function at the two points
    ``(x,y,z) = (2.1, 6.2, 8.3)`` and ``(3.3, 5.2, 7.1)``:

    >>> pts = np.array([[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]])
    >>> my_interpolating_function(pts)
    array([ 125.80469388,  146.30069388])

    which is indeed a close approximation to
    ``[f(2.1, 6.2, 8.3), f(3.3, 5.2, 7.1)]``.

    See also
    --------
    NearestNDInterpolator : Nearest neighbor interpolation on unstructured
                            data in N dimensions

    LinearNDInterpolator : Piecewise linear interpolant on unstructured data
                           in N dimensions

    References
    ----------
    .. [1] Python package *regulargrid* by Johannes Buchner, see
           https://pypi.python.org/pypi/regulargrid/
    .. [2] Wikipedia, "Trilinear interpolation",
           https://en.wikipedia.org/wiki/Trilinear_interpolation
    .. [3] Weiser, Alan, and Sergio E. Zarantonello. "A note on piecewise linear
           and multilinear table interpolation in many dimensions." MATH.
           COMPUT. 50.181 (1988): 189-196.
           https://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917826-0/S0025-5718-1988-0917826-0.pdf

    """
    # this class is based on code originally programmed by Johannes Buchner,
    # see https://github.com/JohannesBuchner/regulargrid

    def __init__(self, points, values, method="linear", bounds_error=True,
                 fill_value=np.nan):
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)
        self.method = method
        self.bounds_error = bounds_error

        if not hasattr(values, 'ndim'):
            # allow reasonable duck-typed values
            values = np.asarray(values)

        if len(points) > values.ndim:
            raise ValueError("There are %d point arrays, but values has %d "
                             "dimensions" % (len(points), values.ndim))

        if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            if not np.issubdtype(values.dtype, np.inexact):
                values = values.astype(float)

        self.fill_value = fill_value
        if fill_value is not None:
            fill_value_dtype = np.asarray(fill_value).dtype
            if (hasattr(values, 'dtype') and not
                    np.can_cast(fill_value_dtype, values.dtype,
                                casting='same_kind')):
                raise ValueError("fill_value must be either 'None' or "
                                 "of a type compatible with values")

        for i, p in enumerate(points):
            if not np.all(np.diff(p) > 0.):
                raise ValueError("The points in dimension %d must be strictly "
                                 "ascending" % i)
            if not np.asarray(p).ndim == 1:
                raise ValueError("The points in dimension %d must be "
                                 "1-dimensional" % i)
            if not values.shape[i] == len(p):
                raise ValueError("There are %d points and %d values in "
                                 "dimension %d" % (len(p), values.shape[i], i))
        self.grid = tuple([np.asarray(p) for p in points])
        self.values = values

    def __call__(self, xi, method=None):
        """
        Interpolation at coordinates

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at

        method : str
            The method of interpolation to perform. Supported are "linear" and
            "nearest".

        """
        method = self.method if method is None else method
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)

        ndim = len(self.grid)
        xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             "%d, but this RegularGridInterpolator has "
                             "dimension %d" % (xi.shape[1], ndim))

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not np.logical_and(np.all(self.grid[i][0] <= p),
                                      np.all(p <= self.grid[i][-1])):
                    raise ValueError("One of the requested xi is out of bounds "
                                     "in dimension %d" % i)

        indices, norm_distances, out_of_bounds = self._find_indices(xi.T)
        if method == "linear":
            result = self._evaluate_linear(indices,
                                           norm_distances,
                                           out_of_bounds)
        elif method == "nearest":
            result = self._evaluate_nearest(indices,
                                            norm_distances,
                                            out_of_bounds)
        if not self.bounds_error and self.fill_value is not None:
            result[out_of_bounds] = self.fill_value

        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    def _evaluate_linear(self, indices, norm_distances, out_of_bounds):
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents a edge
        edges = itertools.product(*[[i, i + 1] for i in indices])
        values = 0.
        for edge_indices in edges:
            weight = 1.
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight *= np.where(ei == i, 1 - yi, yi)
            values += np.asarray(self.values[edge_indices]) * weight[vslice]
        return values

    def _evaluate_nearest(self, indices, norm_distances, out_of_bounds):
        idx_res = [np.where(yi <= .5, i, i + 1)
                   for i, yi in zip(indices, norm_distances)]
        return self.values[tuple(idx_res)]

    def _find_indices(self, xi):
        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # check for out of bounds xi
        out_of_bounds = np.zeros((xi.shape[1]), dtype=bool)
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            i = np.searchsorted(grid, x) - 1
            i[i < 0] = 0
            i[i > grid.size - 2] = grid.size - 2
            indices.append(i)
            norm_distances.append((x - grid[i]) /
                                  (grid[i + 1] - grid[i]))
            if not self.bounds_error:
                out_of_bounds += x < grid[0]
                out_of_bounds += x > grid[-1]
        return indices, norm_distances, out_of_bounds


def interpn(points, values, xi, method="linear", bounds_error=True,
            fill_value=np.nan):
    """
    Multidimensional interpolation on regular grids.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.

    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.

    xi : ndarray of shape (..., ndim)
        The coordinates to sample the gridded data at

    method : str, optional
        The method of interpolation to perform. Supported are "linear" and
        "nearest", and "splinef2d". "splinef2d" is only supported for
        2-dimensional data.

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.

    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.  Extrapolation is not supported by method
        "splinef2d".

    Returns
    -------
    values_x : ndarray, shape xi.shape[:-1] + values.shape[ndim:]
        Interpolated values at input coordinates.

    Notes
    -----

    .. versionadded:: 0.14

    Examples
    --------
    Evaluate a simple example function on the points of a regular 3-D grid:

    >>> from scipy.interpolate import interpn
    >>> def value_func_3d(x, y, z):
    ...     return 2 * x + 3 * y - z
    >>> x = np.linspace(0, 4, 5)
    >>> y = np.linspace(0, 5, 6)
    >>> z = np.linspace(0, 6, 7)
    >>> points = (x, y, z)
    >>> values = value_func_3d(*np.meshgrid(*points, indexing='ij'))

    Evaluate the interpolating function at a point

    >>> point = np.array([2.21, 3.12, 1.15])
    >>> print(interpn(points, values, point))
    [12.63]

    See also
    --------
    NearestNDInterpolator : Nearest neighbor interpolation on unstructured
                            data in N dimensions

    LinearNDInterpolator : Piecewise linear interpolant on unstructured data
                           in N dimensions

    RegularGridInterpolator : Linear and nearest-neighbor Interpolation on a
                              regular grid in arbitrary dimensions

    RectBivariateSpline : Bivariate spline approximation over a rectangular mesh

    """
    # sanity check 'method' kwarg
    if method not in ["linear", "nearest", "splinef2d"]:
        raise ValueError("interpn only understands the methods 'linear', "
                         "'nearest', and 'splinef2d'. You provided %s." %
                         method)

    if not hasattr(values, 'ndim'):
        values = np.asarray(values)

    ndim = values.ndim
    if ndim > 2 and method == "splinef2d":
        raise ValueError("The method splinef2d can only be used for "
                         "2-dimensional input data")
    if not bounds_error and fill_value is None and method == "splinef2d":
        raise ValueError("The method splinef2d does not support extrapolation.")

    # sanity check consistency of input dimensions
    if len(points) > ndim:
        raise ValueError("There are %d point arrays, but values has %d "
                         "dimensions" % (len(points), ndim))
    if len(points) != ndim and method == 'splinef2d':
        raise ValueError("The method splinef2d can only be used for "
                         "scalar data with one point per coordinate")

    # sanity check input grid
    for i, p in enumerate(points):
        if not np.all(np.diff(p) > 0.):
            raise ValueError("The points in dimension %d must be strictly "
                             "ascending" % i)
        if not np.asarray(p).ndim == 1:
            raise ValueError("The points in dimension %d must be "
                             "1-dimensional" % i)
        if not values.shape[i] == len(p):
            raise ValueError("There are %d points and %d values in "
                             "dimension %d" % (len(p), values.shape[i], i))
    grid = tuple([np.asarray(p) for p in points])

    # sanity check requested xi
    xi = _ndim_coords_from_arrays(xi, ndim=len(grid))
    if xi.shape[-1] != len(grid):
        raise ValueError("The requested sample points xi have dimension "
                         "%d, but this RegularGridInterpolator has "
                         "dimension %d" % (xi.shape[1], len(grid)))

    if bounds_error:
        for i, p in enumerate(xi.T):
            if not np.logical_and(np.all(grid[i][0] <= p),
                                                np.all(p <= grid[i][-1])):
                raise ValueError("One of the requested xi is out of bounds "
                                "in dimension %d" % i)

    # perform interpolation
    if method == "linear":
        interp = RegularGridInterpolator(points, values, method="linear",
                                         bounds_error=bounds_error,
                                         fill_value=fill_value)
        return interp(xi)
    elif method == "nearest":
        interp = RegularGridInterpolator(points, values, method="nearest",
                                         bounds_error=bounds_error,
                                         fill_value=fill_value)
        return interp(xi)
    elif method == "splinef2d":
        xi_shape = xi.shape
        xi = xi.reshape(-1, xi.shape[-1])

        # RectBivariateSpline doesn't support fill_value; we need to wrap here
        idx_valid = np.all((grid[0][0] <= xi[:, 0], xi[:, 0] <= grid[0][-1],
                            grid[1][0] <= xi[:, 1], xi[:, 1] <= grid[1][-1]),
                           axis=0)
        result = np.empty_like(xi[:, 0])

        # make a copy of values for RectBivariateSpline
        interp = RectBivariateSpline(points[0], points[1], values[:])
        result[idx_valid] = interp.ev(xi[idx_valid, 0], xi[idx_valid, 1])
        result[np.logical_not(idx_valid)] = fill_value

        return result.reshape(xi_shape[:-1])


# backward compatibility wrapper
class _ppform(PPoly):
    """
    Deprecated piecewise polynomial class.

    New code should use the `PPoly` class instead.

    """

    def __init__(self, coeffs, breaks, fill=0.0, sort=False):
        warnings.warn("_ppform is deprecated -- use PPoly instead",
                      category=DeprecationWarning)

        if sort:
            breaks = np.sort(breaks)
        else:
            breaks = np.asarray(breaks)

        PPoly.__init__(self, coeffs, breaks)

        self.coeffs = self.c
        self.breaks = self.x
        self.K = self.coeffs.shape[0]
        self.fill = fill
        self.a = self.breaks[0]
        self.b = self.breaks[-1]

    def __call__(self, x):
        return PPoly.__call__(self, x, 0, False)

    def _evaluate(self, x, nu, extrapolate, out):
        PPoly._evaluate(self, x, nu, extrapolate, out)
        out[~((x >= self.a) & (x <= self.b))] = self.fill
        return out

    @classmethod
    def fromspline(cls, xk, cvals, order, fill=0.0):
        # Note: this spline representation is incompatible with FITPACK
        N = len(xk)-1
        sivals = np.empty((order+1, N), dtype=float)
        for m in range(order, -1, -1):
            fact = spec.gamma(m+1)
            res = _fitpack._bspleval(xk[:-1], xk, cvals, order, m)
            res /= fact
            sivals[order-m, :] = res
        return cls(sivals, xk, fill=fill)
