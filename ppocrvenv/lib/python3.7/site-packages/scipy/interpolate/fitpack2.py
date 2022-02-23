"""
fitpack --- curve and surface fitting with splines

fitpack is based on a collection of Fortran routines DIERCKX
by P. Dierckx (see http://www.netlib.org/dierckx/) transformed
to double routines by Pearu Peterson.
"""
# Created by Pearu Peterson, June,August 2003
__all__ = [
    'UnivariateSpline',
    'InterpolatedUnivariateSpline',
    'LSQUnivariateSpline',
    'BivariateSpline',
    'LSQBivariateSpline',
    'SmoothBivariateSpline',
    'LSQSphereBivariateSpline',
    'SmoothSphereBivariateSpline',
    'RectBivariateSpline',
    'RectSphereBivariateSpline']


import warnings

from numpy import zeros, concatenate, ravel, diff, array, ones
import numpy as np

from . import fitpack
from . import dfitpack


dfitpack_int = dfitpack.types.intvar.dtype


# ############### Univariate spline ####################

_curfit_messages = {1: """
The required storage space exceeds the available storage space, as
specified by the parameter nest: nest too small. If nest is already
large (say nest > m/2), it may also indicate that s is too small.
The approximation returned is the weighted least-squares spline
according to the knots t[0],t[1],...,t[n-1]. (n=nest) the parameter fp
gives the corresponding weighted sum of squared residuals (fp>s).
""",
                    2: """
A theoretically impossible result was found during the iteration
process for finding a smoothing spline with fp = s: s too small.
There is an approximation returned but the corresponding weighted sum
of squared residuals does not satisfy the condition abs(fp-s)/s < tol.""",
                    3: """
The maximal number of iterations maxit (set to 20 by the program)
allowed for finding a smoothing spline with fp=s has been reached: s
too small.
There is an approximation returned but the corresponding weighted sum
of squared residuals does not satisfy the condition abs(fp-s)/s < tol.""",
                    10: """
Error on entry, no approximation returned. The following conditions
must hold:
xb<=x[0]<x[1]<...<x[m-1]<=xe, w[i]>0, i=0..m-1
if iopt=-1:
  xb<t[k+1]<t[k+2]<...<t[n-k-2]<xe"""
                    }


# UnivariateSpline, ext parameter can be an int or a string
_extrap_modes = {0: 0, 'extrapolate': 0,
                 1: 1, 'zeros': 1,
                 2: 2, 'raise': 2,
                 3: 3, 'const': 3}


class UnivariateSpline:
    """
    1-D smoothing spline fit to a given set of data points.

    Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.  `s`
    specifies the number of knots by specifying a smoothing condition.

    Parameters
    ----------
    x : (N,) array_like
        1-D array of independent input data. Must be increasing;
        must be strictly increasing if `s` is 0.
    y : (N,) array_like
        1-D array of dependent input data, of the same length as `x`.
    w : (N,) array_like, optional
        Weights for spline fitting.  Must be positive.  If `w` is None,
        weights are all equal. Default is None.
    bbox : (2,) array_like, optional
        2-sequence specifying the boundary of the approximation interval. If
        `bbox` is None, ``bbox=[x[0], x[-1]]``. Default is None.
    k : int, optional
        Degree of the smoothing spline.  Must be 1 <= `k` <= 5.
        ``k = 3`` is a cubic spline. Default is 3.
    s : float or None, optional
        Positive smoothing factor used to choose the number of knots.  Number
        of knots will be increased until the smoothing condition is satisfied::

            sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s

        If `s` is None, ``s = len(w)`` which should be a good value if
        ``1/w[i]`` is an estimate of the standard deviation of ``y[i]``.
        If 0, spline will interpolate through all data points. Default is None.
    ext : int or str, optional
        Controls the extrapolation mode for elements
        not in the interval defined by the knot sequence.

        * if ext=0 or 'extrapolate', return the extrapolated value.
        * if ext=1 or 'zeros', return 0
        * if ext=2 or 'raise', raise a ValueError
        * if ext=3 of 'const', return the boundary value.

        Default is 0.

    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination or non-sensical results) if the inputs
        do contain infinities or NaNs.
        Default is False.

    See Also
    --------
    BivariateSpline :
        a base class for bivariate splines.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh
    InterpolatedUnivariateSpline :
        a interpolating univariate spline for a given set of data points.
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives
    splrep :
        a function to find the B-spline representation of a 1-D curve
    splev :
        a function to evaluate a B-spline or its derivatives
    sproot :
        a function to find the roots of a cubic B-spline
    splint :
        a function to evaluate the definite integral of a B-spline between two
        given points
    spalde :
        a function to evaluate all derivatives of a B-spline

    Notes
    -----
    The number of data points must be larger than the spline degree `k`.

    **NaN handling**: If the input arrays contain ``nan`` values, the result
    is not useful, since the underlying spline fitting routines cannot deal
    with ``nan``. A workaround is to use zero weights for not-a-number
    data points:

    >>> from scipy.interpolate import UnivariateSpline
    >>> x, y = np.array([1, 2, 3, 4]), np.array([1, np.nan, 3, 4])
    >>> w = np.isnan(y)
    >>> y[w] = 0.
    >>> spl = UnivariateSpline(x, y, w=~w)

    Notice the need to replace a ``nan`` by a numerical value (precise value
    does not matter as long as the corresponding weight is zero.)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import UnivariateSpline
    >>> rng = np.random.default_rng()
    >>> x = np.linspace(-3, 3, 50)
    >>> y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)
    >>> plt.plot(x, y, 'ro', ms=5)

    Use the default value for the smoothing parameter:

    >>> spl = UnivariateSpline(x, y)
    >>> xs = np.linspace(-3, 3, 1000)
    >>> plt.plot(xs, spl(xs), 'g', lw=3)

    Manually change the amount of smoothing:

    >>> spl.set_smoothing_factor(0.5)
    >>> plt.plot(xs, spl(xs), 'b', lw=3)
    >>> plt.show()

    """
    def __init__(self, x, y, w=None, bbox=[None]*2, k=3, s=None,
                 ext=0, check_finite=False):

        x, y, w, bbox, self.ext = self.validate_input(x, y, w, bbox, k, s, ext,
                                                      check_finite)

        # _data == x,y,w,xb,xe,k,s,n,t,c,fp,fpint,nrdata,ier
        data = dfitpack.fpcurf0(x, y, k, w=w, xb=bbox[0],
                                xe=bbox[1], s=s)
        if data[-1] == 1:
            # nest too small, setting to maximum bound
            data = self._reset_nest(data)
        self._data = data
        self._reset_class()

    @staticmethod
    def validate_input(x, y, w, bbox, k, s, ext, check_finite):
        x, y, bbox = np.asarray(x), np.asarray(y), np.asarray(bbox)
        if w is not None:
            w = np.asarray(w)
        if check_finite:
            w_finite = np.isfinite(w).all() if w is not None else True
            if (not np.isfinite(x).all() or not np.isfinite(y).all() or
                    not w_finite):
                raise ValueError("x and y array must not contain "
                                 "NaNs or infs.")
        if s is None or s > 0:
            if not np.all(diff(x) >= 0.0):
                raise ValueError("x must be increasing if s > 0")
        else:
            if not np.all(diff(x) > 0.0):
                raise ValueError("x must be strictly increasing if s = 0")
        if x.size != y.size:
            raise ValueError("x and y should have a same length")
        elif w is not None and not x.size == y.size == w.size:
            raise ValueError("x, y, and w should have a same length")
        elif bbox.shape != (2,):
            raise ValueError("bbox shape should be (2,)")
        elif not (1 <= k <= 5):
            raise ValueError("k should be 1 <= k <= 5")
        elif s is not None and not s >= 0.0:
            raise ValueError("s should be s >= 0.0")

        try:
            ext = _extrap_modes[ext]
        except KeyError as e:
            raise ValueError("Unknown extrapolation mode %s." % ext) from e

        return x, y, w, bbox, ext

    @classmethod
    def _from_tck(cls, tck, ext=0):
        """Construct a spline object from given tck"""
        self = cls.__new__(cls)
        t, c, k = tck
        self._eval_args = tck
        # _data == x,y,w,xb,xe,k,s,n,t,c,fp,fpint,nrdata,ier
        self._data = (None, None, None, None, None, k, None, len(t), t,
                      c, None, None, None, None)
        self.ext = ext
        return self

    def _reset_class(self):
        data = self._data
        n, t, c, k, ier = data[7], data[8], data[9], data[5], data[-1]
        self._eval_args = t[:n], c[:n], k
        if ier == 0:
            # the spline returned has a residual sum of squares fp
            # such that abs(fp-s)/s <= tol with tol a relative
            # tolerance set to 0.001 by the program
            pass
        elif ier == -1:
            # the spline returned is an interpolating spline
            self._set_class(InterpolatedUnivariateSpline)
        elif ier == -2:
            # the spline returned is the weighted least-squares
            # polynomial of degree k. In this extreme case fp gives
            # the upper bound fp0 for the smoothing factor s.
            self._set_class(LSQUnivariateSpline)
        else:
            # error
            if ier == 1:
                self._set_class(LSQUnivariateSpline)
            message = _curfit_messages.get(ier, 'ier=%s' % (ier))
            warnings.warn(message)

    def _set_class(self, cls):
        self._spline_class = cls
        if self.__class__ in (UnivariateSpline, InterpolatedUnivariateSpline,
                              LSQUnivariateSpline):
            self.__class__ = cls
        else:
            # It's an unknown subclass -- don't change class. cf. #731
            pass

    def _reset_nest(self, data, nest=None):
        n = data[10]
        if nest is None:
            k, m = data[5], len(data[0])
            nest = m+k+1  # this is the maximum bound for nest
        else:
            if not n <= nest:
                raise ValueError("`nest` can only be increased")
        t, c, fpint, nrdata = [np.resize(data[j], nest) for j in
                               [8, 9, 11, 12]]

        args = data[:8] + (t, c, n, fpint, nrdata, data[13])
        data = dfitpack.fpcurf1(*args)
        return data

    def set_smoothing_factor(self, s):
        """ Continue spline computation with the given smoothing
        factor s and with the knots found at the last call.

        This routine modifies the spline in place.

        """
        data = self._data
        if data[6] == -1:
            warnings.warn('smoothing factor unchanged for'
                          'LSQ spline with fixed knots')
            return
        args = data[:6] + (s,) + data[7:]
        data = dfitpack.fpcurf1(*args)
        if data[-1] == 1:
            # nest too small, setting to maximum bound
            data = self._reset_nest(data)
        self._data = data
        self._reset_class()

    def __call__(self, x, nu=0, ext=None):
        """
        Evaluate spline (or its nu-th derivative) at positions x.

        Parameters
        ----------
        x : array_like
            A 1-D array of points at which to return the value of the smoothed
            spline or its derivatives. Note: `x` can be unordered but the
            evaluation is more efficient if `x` is (partially) ordered.
        nu  : int
            The order of derivative of the spline to compute.
        ext : int
            Controls the value returned for elements of `x` not in the
            interval defined by the knot sequence.

            * if ext=0 or 'extrapolate', return the extrapolated value.
            * if ext=1 or 'zeros', return 0
            * if ext=2 or 'raise', raise a ValueError
            * if ext=3 or 'const', return the boundary value.

            The default value is 0, passed from the initialization of
            UnivariateSpline.

        """
        x = np.asarray(x)
        # empty input yields empty output
        if x.size == 0:
            return array([])
        if ext is None:
            ext = self.ext
        else:
            try:
                ext = _extrap_modes[ext]
            except KeyError as e:
                raise ValueError("Unknown extrapolation mode %s." % ext) from e
        return fitpack.splev(x, self._eval_args, der=nu, ext=ext)

    def get_knots(self):
        """ Return positions of interior knots of the spline.

        Internally, the knot vector contains ``2*k`` additional boundary knots.
        """
        data = self._data
        k, n = data[5], data[7]
        return data[8][k:n-k]

    def get_coeffs(self):
        """Return spline coefficients."""
        data = self._data
        k, n = data[5], data[7]
        return data[9][:n-k-1]

    def get_residual(self):
        """Return weighted sum of squared residuals of the spline approximation.

           This is equivalent to::

                sum((w[i] * (y[i]-spl(x[i])))**2, axis=0)

        """
        return self._data[10]

    def integral(self, a, b):
        """ Return definite integral of the spline between two given points.

        Parameters
        ----------
        a : float
            Lower limit of integration.
        b : float
            Upper limit of integration.

        Returns
        -------
        integral : float
            The value of the definite integral of the spline between limits.

        Examples
        --------
        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, 3, 11)
        >>> y = x**2
        >>> spl = UnivariateSpline(x, y)
        >>> spl.integral(0, 3)
        9.0

        which agrees with :math:`\\int x^2 dx = x^3 / 3` between the limits
        of 0 and 3.

        A caveat is that this routine assumes the spline to be zero outside of
        the data limits:

        >>> spl.integral(-1, 4)
        9.0
        >>> spl.integral(-1, 0)
        0.0

        """
        return dfitpack.splint(*(self._eval_args+(a, b)))

    def derivatives(self, x):
        """ Return all derivatives of the spline at the point x.

        Parameters
        ----------
        x : float
            The point to evaluate the derivatives at.

        Returns
        -------
        der : ndarray, shape(k+1,)
            Derivatives of the orders 0 to k.

        Examples
        --------
        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, 3, 11)
        >>> y = x**2
        >>> spl = UnivariateSpline(x, y)
        >>> spl.derivatives(1.5)
        array([2.25, 3.0, 2.0, 0])

        """
        d, ier = dfitpack.spalde(*(self._eval_args+(x,)))
        if not ier == 0:
            raise ValueError("Error code returned by spalde: %s" % ier)
        return d

    def roots(self):
        """ Return the zeros of the spline.

        Restriction: only cubic splines are supported by fitpack.
        """
        k = self._data[5]
        if k == 3:
            z, m, ier = dfitpack.sproot(*self._eval_args[:2])
            if not ier == 0:
                raise ValueError("Error code returned by spalde: %s" % ier)
            return z[:m]
        raise NotImplementedError('finding roots unsupported for '
                                  'non-cubic splines')

    def derivative(self, n=1):
        """
        Construct a new spline representing the derivative of this spline.

        Parameters
        ----------
        n : int, optional
            Order of derivative to evaluate. Default: 1

        Returns
        -------
        spline : UnivariateSpline
            Spline of order k2=k-n representing the derivative of this
            spline.

        See Also
        --------
        splder, antiderivative

        Notes
        -----

        .. versionadded:: 0.13.0

        Examples
        --------
        This can be used for finding maxima of a curve:

        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, 10, 70)
        >>> y = np.sin(x)
        >>> spl = UnivariateSpline(x, y, k=4, s=0)

        Now, differentiate the spline and find the zeros of the
        derivative. (NB: `sproot` only works for order 3 splines, so we
        fit an order 4 spline):

        >>> spl.derivative().roots() / np.pi
        array([ 0.50000001,  1.5       ,  2.49999998])

        This agrees well with roots :math:`\\pi/2 + n\\pi` of
        :math:`\\cos(x) = \\sin'(x)`.

        """
        tck = fitpack.splder(self._eval_args, n)
        # if self.ext is 'const', derivative.ext will be 'zeros'
        ext = 1 if self.ext == 3 else self.ext
        return UnivariateSpline._from_tck(tck, ext=ext)

    def antiderivative(self, n=1):
        """
        Construct a new spline representing the antiderivative of this spline.

        Parameters
        ----------
        n : int, optional
            Order of antiderivative to evaluate. Default: 1

        Returns
        -------
        spline : UnivariateSpline
            Spline of order k2=k+n representing the antiderivative of this
            spline.

        Notes
        -----

        .. versionadded:: 0.13.0

        See Also
        --------
        splantider, derivative

        Examples
        --------
        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, np.pi/2, 70)
        >>> y = 1 / np.sqrt(1 - 0.8*np.sin(x)**2)
        >>> spl = UnivariateSpline(x, y, s=0)

        The derivative is the inverse operation of the antiderivative,
        although some floating point error accumulates:

        >>> spl(1.7), spl.antiderivative().derivative()(1.7)
        (array(2.1565429877197317), array(2.1565429877201865))

        Antiderivative can be used to evaluate definite integrals:

        >>> ispl = spl.antiderivative()
        >>> ispl(np.pi/2) - ispl(0)
        2.2572053588768486

        This is indeed an approximation to the complete elliptic integral
        :math:`K(m) = \\int_0^{\\pi/2} [1 - m\\sin^2 x]^{-1/2} dx`:

        >>> from scipy.special import ellipk
        >>> ellipk(0.8)
        2.2572053268208538

        """
        tck = fitpack.splantider(self._eval_args, n)
        return UnivariateSpline._from_tck(tck, self.ext)


class InterpolatedUnivariateSpline(UnivariateSpline):
    """
    1-D interpolating spline for a given set of data points.

    Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.
    Spline function passes through all provided points. Equivalent to
    `UnivariateSpline` with  s=0.

    Parameters
    ----------
    x : (N,) array_like
        Input dimension of data points -- must be strictly increasing
    y : (N,) array_like
        input dimension of data points
    w : (N,) array_like, optional
        Weights for spline fitting.  Must be positive.  If None (default),
        weights are all equal.
    bbox : (2,) array_like, optional
        2-sequence specifying the boundary of the approximation interval. If
        None (default), ``bbox=[x[0], x[-1]]``.
    k : int, optional
        Degree of the smoothing spline.  Must be 1 <= `k` <= 5.
    ext : int or str, optional
        Controls the extrapolation mode for elements
        not in the interval defined by the knot sequence.

        * if ext=0 or 'extrapolate', return the extrapolated value.
        * if ext=1 or 'zeros', return 0
        * if ext=2 or 'raise', raise a ValueError
        * if ext=3 of 'const', return the boundary value.

        The default value is 0.

    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination or non-sensical results) if the inputs
        do contain infinities or NaNs.
        Default is False.

    See Also
    --------
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    LSQUnivariateSpline :
        a spline for which knots are user-selected
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    splrep :
        a function to find the B-spline representation of a 1-D curve
    splev :
        a function to evaluate a B-spline or its derivatives
    sproot :
        a function to find the roots of a cubic B-spline
    splint :
        a function to evaluate the definite integral of a B-spline between two
        given points
    spalde :
        a function to evaluate all derivatives of a B-spline

    Notes
    -----
    The number of data points must be larger than the spline degree `k`.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import InterpolatedUnivariateSpline
    >>> rng = np.random.default_rng()
    >>> x = np.linspace(-3, 3, 50)
    >>> y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)
    >>> spl = InterpolatedUnivariateSpline(x, y)
    >>> plt.plot(x, y, 'ro', ms=5)
    >>> xs = np.linspace(-3, 3, 1000)
    >>> plt.plot(xs, spl(xs), 'g', lw=3, alpha=0.7)
    >>> plt.show()

    Notice that the ``spl(x)`` interpolates `y`:

    >>> spl.get_residual()
    0.0

    """
    def __init__(self, x, y, w=None, bbox=[None]*2, k=3,
                 ext=0, check_finite=False):

        x, y, w, bbox, self.ext = self.validate_input(x, y, w, bbox, k, None,
                                            ext, check_finite)
        if not np.all(diff(x) > 0.0):
            raise ValueError('x must be strictly increasing')

        # _data == x,y,w,xb,xe,k,s,n,t,c,fp,fpint,nrdata,ier
        self._data = dfitpack.fpcurf0(x, y, k, w=w, xb=bbox[0],
                                      xe=bbox[1], s=0)
        self._reset_class()


_fpchec_error_string = """The input parameters have been rejected by fpchec. \
This means that at least one of the following conditions is violated:

1) k+1 <= n-k-1 <= m
2) t(1) <= t(2) <= ... <= t(k+1)
   t(n-k) <= t(n-k+1) <= ... <= t(n)
3) t(k+1) < t(k+2) < ... < t(n-k)
4) t(k+1) <= x(i) <= t(n-k)
5) The conditions specified by Schoenberg and Whitney must hold
   for at least one subset of data points, i.e., there must be a
   subset of data points y(j) such that
       t(j) < y(j) < t(j+k+1), j=1,2,...,n-k-1
"""


class LSQUnivariateSpline(UnivariateSpline):
    """
    1-D spline with explicit internal knots.

    Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.  `t`
    specifies the internal knots of the spline

    Parameters
    ----------
    x : (N,) array_like
        Input dimension of data points -- must be increasing
    y : (N,) array_like
        Input dimension of data points
    t : (M,) array_like
        interior knots of the spline.  Must be in ascending order and::

            bbox[0] < t[0] < ... < t[-1] < bbox[-1]

    w : (N,) array_like, optional
        weights for spline fitting. Must be positive. If None (default),
        weights are all equal.
    bbox : (2,) array_like, optional
        2-sequence specifying the boundary of the approximation interval. If
        None (default), ``bbox = [x[0], x[-1]]``.
    k : int, optional
        Degree of the smoothing spline.  Must be 1 <= `k` <= 5.
        Default is `k` = 3, a cubic spline.
    ext : int or str, optional
        Controls the extrapolation mode for elements
        not in the interval defined by the knot sequence.

        * if ext=0 or 'extrapolate', return the extrapolated value.
        * if ext=1 or 'zeros', return 0
        * if ext=2 or 'raise', raise a ValueError
        * if ext=3 of 'const', return the boundary value.

        The default value is 0.

    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination or non-sensical results) if the inputs
        do contain infinities or NaNs.
        Default is False.

    Raises
    ------
    ValueError
        If the interior knots do not satisfy the Schoenberg-Whitney conditions

    See Also
    --------
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    InterpolatedUnivariateSpline :
        a interpolating univariate spline for a given set of data points.
    splrep :
        a function to find the B-spline representation of a 1-D curve
    splev :
        a function to evaluate a B-spline or its derivatives
    sproot :
        a function to find the roots of a cubic B-spline
    splint :
        a function to evaluate the definite integral of a B-spline between two
        given points
    spalde :
        a function to evaluate all derivatives of a B-spline

    Notes
    -----
    The number of data points must be larger than the spline degree `k`.

    Knots `t` must satisfy the Schoenberg-Whitney conditions,
    i.e., there must be a subset of data points ``x[j]`` such that
    ``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.

    Examples
    --------
    >>> from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = np.linspace(-3, 3, 50)
    >>> y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)

    Fit a smoothing spline with a pre-defined internal knots:

    >>> t = [-1, 0, 1]
    >>> spl = LSQUnivariateSpline(x, y, t)

    >>> xs = np.linspace(-3, 3, 1000)
    >>> plt.plot(x, y, 'ro', ms=5)
    >>> plt.plot(xs, spl(xs), 'g-', lw=3)
    >>> plt.show()

    Check the knot vector:

    >>> spl.get_knots()
    array([-3., -1., 0., 1., 3.])

    Constructing lsq spline using the knots from another spline:

    >>> x = np.arange(10)
    >>> s = UnivariateSpline(x, x, s=0)
    >>> s.get_knots()
    array([ 0.,  2.,  3.,  4.,  5.,  6.,  7.,  9.])
    >>> knt = s.get_knots()
    >>> s1 = LSQUnivariateSpline(x, x, knt[1:-1])    # Chop 1st and last knot
    >>> s1.get_knots()
    array([ 0.,  2.,  3.,  4.,  5.,  6.,  7.,  9.])

    """

    def __init__(self, x, y, t, w=None, bbox=[None]*2, k=3,
                 ext=0, check_finite=False):

        x, y, w, bbox, self.ext = self.validate_input(x, y, w, bbox, k, None,
                                                      ext, check_finite)
        if not np.all(diff(x) >= 0.0):
            raise ValueError('x must be increasing')

        # _data == x,y,w,xb,xe,k,s,n,t,c,fp,fpint,nrdata,ier
        xb = bbox[0]
        xe = bbox[1]
        if xb is None:
            xb = x[0]
        if xe is None:
            xe = x[-1]
        t = concatenate(([xb]*(k+1), t, [xe]*(k+1)))
        n = len(t)
        if not np.all(t[k+1:n-k]-t[k:n-k-1] > 0, axis=0):
            raise ValueError('Interior knots t must satisfy '
                             'Schoenberg-Whitney conditions')
        if not dfitpack.fpchec(x, t, k) == 0:
            raise ValueError(_fpchec_error_string)
        data = dfitpack.fpcurfm1(x, y, k, t, w=w, xb=xb, xe=xe)
        self._data = data[:-3] + (None, None, data[-1])
        self._reset_class()


# ############### Bivariate spline ####################

class _BivariateSplineBase:
    """ Base class for Bivariate spline s(x,y) interpolation on the rectangle
    [xb,xe] x [yb, ye] calculated from a given set of data points
    (x,y,z).

    See Also
    --------
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives
    BivariateSpline :
        a base class for bivariate splines.
    SphereBivariateSpline :
        a bivariate spline on a spherical grid
    """

    def get_residual(self):
        """ Return weighted sum of squared residuals of the spline
        approximation: sum ((w[i]*(z[i]-s(x[i],y[i])))**2,axis=0)
        """
        return self.fp

    def get_knots(self):
        """ Return a tuple (tx,ty) where tx,ty contain knots positions
        of the spline with respect to x-, y-variable, respectively.
        The position of interior and additional knots are given as
        t[k+1:-k-1] and t[:k+1]=b, t[-k-1:]=e, respectively.
        """
        return self.tck[:2]

    def get_coeffs(self):
        """ Return spline coefficients."""
        return self.tck[2]

    def __call__(self, x, y, dx=0, dy=0, grid=True):
        """
        Evaluate the spline or its derivatives at given positions.

        Parameters
        ----------
        x, y : array_like
            Input coordinates.

            If `grid` is False, evaluate the spline at points ``(x[i],
            y[i]), i=0, ..., len(x)-1``.  Standard Numpy broadcasting
            is obeyed.

            If `grid` is True: evaluate spline at the grid points
            defined by the coordinate arrays x, y. The arrays must be
            sorted to increasing order.

            Note that the axis ordering is inverted relative to
            the output of meshgrid.
        dx : int
            Order of x-derivative

            .. versionadded:: 0.14.0
        dy : int
            Order of y-derivative

            .. versionadded:: 0.14.0
        grid : bool
            Whether to evaluate the results on a grid spanned by the
            input arrays, or at points specified by the input arrays.

            .. versionadded:: 0.14.0

        """
        x = np.asarray(x)
        y = np.asarray(y)

        tx, ty, c = self.tck[:3]
        kx, ky = self.degrees
        if grid:
            if x.size == 0 or y.size == 0:
                return np.zeros((x.size, y.size), dtype=self.tck[2].dtype)

            if (x.size >= 2) and (not np.all(np.diff(x) >= 0.0)):
                raise ValueError("x must be strictly increasing when `grid` is True")
            if (y.size >= 2) and (not np.all(np.diff(y) >= 0.0)):
                raise ValueError("y must be strictly increasing when `grid` is True")

            if dx or dy:
                z, ier = dfitpack.parder(tx, ty, c, kx, ky, dx, dy, x, y)
                if not ier == 0:
                    raise ValueError("Error code returned by parder: %s" % ier)
            else:
                z, ier = dfitpack.bispev(tx, ty, c, kx, ky, x, y)
                if not ier == 0:
                    raise ValueError("Error code returned by bispev: %s" % ier)
        else:
            # standard Numpy broadcasting
            if x.shape != y.shape:
                x, y = np.broadcast_arrays(x, y)

            shape = x.shape
            x = x.ravel()
            y = y.ravel()

            if x.size == 0 or y.size == 0:
                return np.zeros(shape, dtype=self.tck[2].dtype)

            if dx or dy:
                z, ier = dfitpack.pardeu(tx, ty, c, kx, ky, dx, dy, x, y)
                if not ier == 0:
                    raise ValueError("Error code returned by pardeu: %s" % ier)
            else:
                z, ier = dfitpack.bispeu(tx, ty, c, kx, ky, x, y)
                if not ier == 0:
                    raise ValueError("Error code returned by bispeu: %s" % ier)

            z = z.reshape(shape)
        return z


_surfit_messages = {1: """
The required storage space exceeds the available storage space: nxest
or nyest too small, or s too small.
The weighted least-squares spline corresponds to the current set of
knots.""",
                    2: """
A theoretically impossible result was found during the iteration
process for finding a smoothing spline with fp = s: s too small or
badly chosen eps.
Weighted sum of squared residuals does not satisfy abs(fp-s)/s < tol.""",
                    3: """
the maximal number of iterations maxit (set to 20 by the program)
allowed for finding a smoothing spline with fp=s has been reached:
s too small.
Weighted sum of squared residuals does not satisfy abs(fp-s)/s < tol.""",
                    4: """
No more knots can be added because the number of b-spline coefficients
(nx-kx-1)*(ny-ky-1) already exceeds the number of data points m:
either s or m too small.
The weighted least-squares spline corresponds to the current set of
knots.""",
                    5: """
No more knots can be added because the additional knot would (quasi)
coincide with an old one: s too small or too large a weight to an
inaccurate data point.
The weighted least-squares spline corresponds to the current set of
knots.""",
                    10: """
Error on entry, no approximation returned. The following conditions
must hold:
xb<=x[i]<=xe, yb<=y[i]<=ye, w[i]>0, i=0..m-1
If iopt==-1, then
  xb<tx[kx+1]<tx[kx+2]<...<tx[nx-kx-2]<xe
  yb<ty[ky+1]<ty[ky+2]<...<ty[ny-ky-2]<ye""",
                    -3: """
The coefficients of the spline returned have been computed as the
minimal norm least-squares solution of a (numerically) rank deficient
system (deficiency=%i). If deficiency is large, the results may be
inaccurate. Deficiency may strongly depend on the value of eps."""
                    }


class BivariateSpline(_BivariateSplineBase):
    """
    Base class for bivariate splines.

    This describes a spline ``s(x, y)`` of degrees ``kx`` and ``ky`` on
    the rectangle ``[xb, xe] * [yb, ye]`` calculated from a given set
    of data points ``(x, y, z)``.

    This class is meant to be subclassed, not instantiated directly.
    To construct these splines, call either `SmoothBivariateSpline` or
    `LSQBivariateSpline` or `RectBivariateSpline`.

    See Also
    --------
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh.
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives
    """

    @classmethod
    def _from_tck(cls, tck):
        """Construct a spline object from given tck and degree"""
        self = cls.__new__(cls)
        if len(tck) != 5:
            raise ValueError("tck should be a 5 element tuple of tx,"
                             " ty, c, kx, ky")
        self.tck = tck[:3]
        self.degrees = tck[3:]
        return self

    def ev(self, xi, yi, dx=0, dy=0):
        """
        Evaluate the spline at points

        Returns the interpolated value at ``(xi[i], yi[i]),
        i=0,...,len(xi)-1``.

        Parameters
        ----------
        xi, yi : array_like
            Input coordinates. Standard Numpy broadcasting is obeyed.
        dx : int, optional
            Order of x-derivative

            .. versionadded:: 0.14.0
        dy : int, optional
            Order of y-derivative

            .. versionadded:: 0.14.0
        """
        return self.__call__(xi, yi, dx=dx, dy=dy, grid=False)

    def integral(self, xa, xb, ya, yb):
        """
        Evaluate the integral of the spline over area [xa,xb] x [ya,yb].

        Parameters
        ----------
        xa, xb : float
            The end-points of the x integration interval.
        ya, yb : float
            The end-points of the y integration interval.

        Returns
        -------
        integ : float
            The value of the resulting integral.

        """
        tx, ty, c = self.tck[:3]
        kx, ky = self.degrees
        return dfitpack.dblint(tx, ty, c, kx, ky, xa, xb, ya, yb)

    @staticmethod
    def _validate_input(x, y, z, w, kx, ky, eps):
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        if not x.size == y.size == z.size:
            raise ValueError('x, y, and z should have a same length')

        if w is not None:
            w = np.asarray(w)
            if x.size != w.size:
                raise ValueError('x, y, z, and w should have a same length')
            elif not np.all(w >= 0.0):
                raise ValueError('w should be positive')
        if (eps is not None) and (not 0.0 < eps < 1.0):
            raise ValueError('eps should be between (0, 1)')
        if not x.size >= (kx + 1) * (ky + 1):
            raise ValueError('The length of x, y and z should be at least'
                             ' (kx+1) * (ky+1)')
        return x, y, z, w


class SmoothBivariateSpline(BivariateSpline):
    """
    Smooth bivariate spline approximation.

    Parameters
    ----------
    x, y, z : array_like
        1-D sequences of data points (order is not important).
    w : array_like, optional
        Positive 1-D sequence of weights, of same length as `x`, `y` and `z`.
    bbox : array_like, optional
        Sequence of length 4 specifying the boundary of the rectangular
        approximation domain.  By default,
        ``bbox=[min(x), max(x), min(y), max(y)]``.
    kx, ky : ints, optional
        Degrees of the bivariate spline. Default is 3.
    s : float, optional
        Positive smoothing factor defined for estimation condition:
        ``sum((w[i]*(z[i]-s(x[i], y[i])))**2, axis=0) <= s``
        Default ``s=len(w)`` which should be a good value if ``1/w[i]`` is an
        estimate of the standard deviation of ``z[i]``.
    eps : float, optional
        A threshold for determining the effective rank of an over-determined
        linear system of equations. `eps` should have a value within the open
        interval ``(0, 1)``, the default is 1e-16.

    See Also
    --------
    BivariateSpline :
        a base class for bivariate splines.
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives

    Notes
    -----
    The length of `x`, `y` and `z` should be at least ``(kx+1) * (ky+1)``.

    """

    def __init__(self, x, y, z, w=None, bbox=[None] * 4, kx=3, ky=3, s=None,
                 eps=1e-16):

        x, y, z, w = self._validate_input(x, y, z, w, kx, ky, eps)
        bbox = ravel(bbox)
        if not bbox.shape == (4,):
            raise ValueError('bbox shape should be (4,)')
        if s is not None and not s >= 0.0:
            raise ValueError("s should be s >= 0.0")

        xb, xe, yb, ye = bbox
        nx, tx, ny, ty, c, fp, wrk1, ier = dfitpack.surfit_smth(x, y, z, w,
                                                                xb, xe, yb,
                                                                ye, kx, ky,
                                                                s=s, eps=eps,
                                                                lwrk2=1)
        if ier > 10:          # lwrk2 was to small, re-run
            nx, tx, ny, ty, c, fp, wrk1, ier = dfitpack.surfit_smth(x, y, z, w,
                                                                    xb, xe, yb,
                                                                    ye, kx, ky,
                                                                    s=s,
                                                                    eps=eps,
                                                                    lwrk2=ier)
        if ier in [0, -1, -2]:  # normal return
            pass
        else:
            message = _surfit_messages.get(ier, 'ier=%s' % (ier))
            warnings.warn(message)

        self.fp = fp
        self.tck = tx[:nx], ty[:ny], c[:(nx-kx-1)*(ny-ky-1)]
        self.degrees = kx, ky


class LSQBivariateSpline(BivariateSpline):
    """
    Weighted least-squares bivariate spline approximation.

    Parameters
    ----------
    x, y, z : array_like
        1-D sequences of data points (order is not important).
    tx, ty : array_like
        Strictly ordered 1-D sequences of knots coordinates.
    w : array_like, optional
        Positive 1-D array of weights, of the same length as `x`, `y` and `z`.
    bbox : (4,) array_like, optional
        Sequence of length 4 specifying the boundary of the rectangular
        approximation domain.  By default,
        ``bbox=[min(x,tx),max(x,tx), min(y,ty),max(y,ty)]``.
    kx, ky : ints, optional
        Degrees of the bivariate spline. Default is 3.
    eps : float, optional
        A threshold for determining the effective rank of an over-determined
        linear system of equations. `eps` should have a value within the open
        interval ``(0, 1)``, the default is 1e-16.

    See Also
    --------
    BivariateSpline :
        a base class for bivariate splines.
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh.
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives

    Notes
    -----
    The length of `x`, `y` and `z` should be at least ``(kx+1) * (ky+1)``.

    """

    def __init__(self, x, y, z, tx, ty, w=None, bbox=[None]*4, kx=3, ky=3,
                 eps=None):

        x, y, z, w = self._validate_input(x, y, z, w, kx, ky, eps)
        bbox = ravel(bbox)
        if not bbox.shape == (4,):
            raise ValueError('bbox shape should be (4,)')

        nx = 2*kx+2+len(tx)
        ny = 2*ky+2+len(ty)
        # The Fortran subroutine "surfit" (called as dfitpack.surfit_lsq)
        # requires that the knot arrays passed as input should be "real
        # array(s) of dimension nmax" where "nmax" refers to the greater of nx
        # and ny. We pad the tx1/ty1 arrays here so that this is satisfied, and
        # slice them to the desired sizes upon return.
        nmax = max(nx, ny)
        tx1 = zeros((nmax,), float)
        ty1 = zeros((nmax,), float)
        tx1[kx+1:nx-kx-1] = tx
        ty1[ky+1:ny-ky-1] = ty

        xb, xe, yb, ye = bbox
        tx1, ty1, c, fp, ier = dfitpack.surfit_lsq(x, y, z, nx, tx1, ny, ty1,
                                                   w, xb, xe, yb, ye,
                                                   kx, ky, eps, lwrk2=1)
        if ier > 10:
            tx1, ty1, c, fp, ier = dfitpack.surfit_lsq(x, y, z,
                                                       nx, tx1, ny, ty1, w,
                                                       xb, xe, yb, ye,
                                                       kx, ky, eps, lwrk2=ier)
        if ier in [0, -1, -2]:  # normal return
            pass
        else:
            if ier < -2:
                deficiency = (nx-kx-1)*(ny-ky-1)+ier
                message = _surfit_messages.get(-3) % (deficiency)
            else:
                message = _surfit_messages.get(ier, 'ier=%s' % (ier))
            warnings.warn(message)
        self.fp = fp
        self.tck = tx1[:nx], ty1[:ny], c
        self.degrees = kx, ky


class RectBivariateSpline(BivariateSpline):
    """
    Bivariate spline approximation over a rectangular mesh.

    Can be used for both smoothing and interpolating data.

    Parameters
    ----------
    x,y : array_like
        1-D arrays of coordinates in strictly ascending order.
    z : array_like
        2-D array of data with shape (x.size,y.size).
    bbox : array_like, optional
        Sequence of length 4 specifying the boundary of the rectangular
        approximation domain.  By default,
        ``bbox=[min(x), max(x), min(y), max(y)]``.
    kx, ky : ints, optional
        Degrees of the bivariate spline. Default is 3.
    s : float, optional
        Positive smoothing factor defined for estimation condition:
        ``sum((z[i]-f(x[i], y[i]))**2, axis=0) <= s`` where f is a spline
        function. Default is ``s=0``, which is for interpolation.

    See Also
    --------
    BivariateSpline :
        a base class for bivariate splines.
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives

    """

    def __init__(self, x, y, z, bbox=[None] * 4, kx=3, ky=3, s=0):
        x, y, bbox = ravel(x), ravel(y), ravel(bbox)
        z = np.asarray(z)
        if not np.all(diff(x) > 0.0):
            raise ValueError('x must be strictly increasing')
        if not np.all(diff(y) > 0.0):
            raise ValueError('y must be strictly increasing')
        if not x.size == z.shape[0]:
            raise ValueError('x dimension of z must have same number of '
                             'elements as x')
        if not y.size == z.shape[1]:
            raise ValueError('y dimension of z must have same number of '
                             'elements as y')
        if not bbox.shape == (4,):
            raise ValueError('bbox shape should be (4,)')
        if s is not None and not s >= 0.0:
            raise ValueError("s should be s >= 0.0")

        z = ravel(z)
        xb, xe, yb, ye = bbox
        nx, tx, ny, ty, c, fp, ier = dfitpack.regrid_smth(x, y, z, xb, xe, yb,
                                                          ye, kx, ky, s)

        if ier not in [0, -1, -2]:
            msg = _surfit_messages.get(ier, 'ier=%s' % (ier))
            raise ValueError(msg)

        self.fp = fp
        self.tck = tx[:nx], ty[:ny], c[:(nx - kx - 1) * (ny - ky - 1)]
        self.degrees = kx, ky


_spherefit_messages = _surfit_messages.copy()
_spherefit_messages[10] = """
ERROR. On entry, the input data are controlled on validity. The following
       restrictions must be satisfied:
            -1<=iopt<=1,  m>=2, ntest>=8 ,npest >=8, 0<eps<1,
            0<=teta(i)<=pi, 0<=phi(i)<=2*pi, w(i)>0, i=1,...,m
            lwrk1 >= 185+52*v+10*u+14*u*v+8*(u-1)*v**2+8*m
            kwrk >= m+(ntest-7)*(npest-7)
            if iopt=-1: 8<=nt<=ntest , 9<=np<=npest
                        0<tt(5)<tt(6)<...<tt(nt-4)<pi
                        0<tp(5)<tp(6)<...<tp(np-4)<2*pi
            if iopt>=0: s>=0
            if one of these conditions is found to be violated,control
            is immediately repassed to the calling program. in that
            case there is no approximation returned."""
_spherefit_messages[-3] = """
WARNING. The coefficients of the spline returned have been computed as the
         minimal norm least-squares solution of a (numerically) rank
         deficient system (deficiency=%i, rank=%i). Especially if the rank
         deficiency, which is computed by 6+(nt-8)*(np-7)+ier, is large,
         the results may be inaccurate. They could also seriously depend on
         the value of eps."""


class SphereBivariateSpline(_BivariateSplineBase):
    """
    Bivariate spline s(x,y) of degrees 3 on a sphere, calculated from a
    given set of data points (theta,phi,r).

    .. versionadded:: 0.11.0

    See Also
    --------
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQUnivariateSpline :
        a univariate spline using weighted least-squares fitting
    """

    def __call__(self, theta, phi, dtheta=0, dphi=0, grid=True):
        """
        Evaluate the spline or its derivatives at given positions.

        Parameters
        ----------
        theta, phi : array_like
            Input coordinates.

            If `grid` is False, evaluate the spline at points
            ``(theta[i], phi[i]), i=0, ..., len(x)-1``.  Standard
            Numpy broadcasting is obeyed.

            If `grid` is True: evaluate spline at the grid points
            defined by the coordinate arrays theta, phi. The arrays
            must be sorted to increasing order.
        dtheta : int, optional
            Order of theta-derivative

            .. versionadded:: 0.14.0
        dphi : int
            Order of phi-derivative

            .. versionadded:: 0.14.0
        grid : bool
            Whether to evaluate the results on a grid spanned by the
            input arrays, or at points specified by the input arrays.

            .. versionadded:: 0.14.0

        """
        theta = np.asarray(theta)
        phi = np.asarray(phi)

        if theta.size > 0 and (theta.min() < 0. or theta.max() > np.pi):
            raise ValueError("requested theta out of bounds.")

        return _BivariateSplineBase.__call__(self, theta, phi,
                                             dx=dtheta, dy=dphi, grid=grid)

    def ev(self, theta, phi, dtheta=0, dphi=0):
        """
        Evaluate the spline at points

        Returns the interpolated value at ``(theta[i], phi[i]),
        i=0,...,len(theta)-1``.

        Parameters
        ----------
        theta, phi : array_like
            Input coordinates. Standard Numpy broadcasting is obeyed.
        dtheta : int, optional
            Order of theta-derivative

            .. versionadded:: 0.14.0
        dphi : int, optional
            Order of phi-derivative

            .. versionadded:: 0.14.0
        """
        return self.__call__(theta, phi, dtheta=dtheta, dphi=dphi, grid=False)


class SmoothSphereBivariateSpline(SphereBivariateSpline):
    """
    Smooth bivariate spline approximation in spherical coordinates.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    theta, phi, r : array_like
        1-D sequences of data points (order is not important). Coordinates
        must be given in radians. Theta must lie within the interval
        ``[0, pi]``, and phi must lie within the interval ``[0, 2pi]``.
    w : array_like, optional
        Positive 1-D sequence of weights.
    s : float, optional
        Positive smoothing factor defined for estimation condition:
        ``sum((w(i)*(r(i) - s(theta(i), phi(i))))**2, axis=0) <= s``
        Default ``s=len(w)`` which should be a good value if ``1/w[i]`` is an
        estimate of the standard deviation of ``r[i]``.
    eps : float, optional
        A threshold for determining the effective rank of an over-determined
        linear system of equations. `eps` should have a value within the open
        interval ``(0, 1)``, the default is 1e-16.

    See Also
    --------
    BivariateSpline :
        a base class for bivariate splines.
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh.
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives

    Notes
    -----
    For more information, see the FITPACK_ site about this function.

    .. _FITPACK: http://www.netlib.org/dierckx/sphere.f

    Examples
    --------
    Suppose we have global data on a coarse grid (the input data does not
    have to be on a grid):

    >>> theta = np.linspace(0., np.pi, 7)
    >>> phi = np.linspace(0., 2*np.pi, 9)
    >>> data = np.empty((theta.shape[0], phi.shape[0]))
    >>> data[:,0], data[0,:], data[-1,:] = 0., 0., 0.
    >>> data[1:-1,1], data[1:-1,-1] = 1., 1.
    >>> data[1,1:-1], data[-2,1:-1] = 1., 1.
    >>> data[2:-2,2], data[2:-2,-2] = 2., 2.
    >>> data[2,2:-2], data[-3,2:-2] = 2., 2.
    >>> data[3,3:-2] = 3.
    >>> data = np.roll(data, 4, 1)

    We need to set up the interpolator object

    >>> lats, lons = np.meshgrid(theta, phi)
    >>> from scipy.interpolate import SmoothSphereBivariateSpline
    >>> lut = SmoothSphereBivariateSpline(lats.ravel(), lons.ravel(),
    ...                                   data.T.ravel(), s=3.5)

    As a first test, we'll see what the algorithm returns when run on the
    input coordinates

    >>> data_orig = lut(theta, phi)

    Finally we interpolate the data to a finer grid

    >>> fine_lats = np.linspace(0., np.pi, 70)
    >>> fine_lons = np.linspace(0., 2 * np.pi, 90)

    >>> data_smth = lut(fine_lats, fine_lons)

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(131)
    >>> ax1.imshow(data, interpolation='nearest')
    >>> ax2 = fig.add_subplot(132)
    >>> ax2.imshow(data_orig, interpolation='nearest')
    >>> ax3 = fig.add_subplot(133)
    >>> ax3.imshow(data_smth, interpolation='nearest')
    >>> plt.show()

    """

    def __init__(self, theta, phi, r, w=None, s=0., eps=1E-16):

        theta, phi, r = np.asarray(theta), np.asarray(phi), np.asarray(r)

        # input validation
        if not ((0.0 <= theta).all() and (theta <= np.pi).all()):
            raise ValueError('theta should be between [0, pi]')
        if not ((0.0 <= phi).all() and (phi <= 2.0 * np.pi).all()):
            raise ValueError('phi should be between [0, 2pi]')
        if w is not None:
            w = np.asarray(w)
            if not (w >= 0.0).all():
                raise ValueError('w should be positive')
        if not s >= 0.0:
            raise ValueError('s should be positive')
        if not 0.0 < eps < 1.0:
            raise ValueError('eps should be between (0, 1)')

        if np.issubclass_(w, float):
            w = ones(len(theta)) * w
        nt_, tt_, np_, tp_, c, fp, ier = dfitpack.spherfit_smth(theta, phi,
                                                                r, w=w, s=s,
                                                                eps=eps)
        if ier not in [0, -1, -2]:
            message = _spherefit_messages.get(ier, 'ier=%s' % (ier))
            raise ValueError(message)

        self.fp = fp
        self.tck = tt_[:nt_], tp_[:np_], c[:(nt_ - 4) * (np_ - 4)]
        self.degrees = (3, 3)

    def __call__(self, theta, phi, dtheta=0, dphi=0, grid=True):

        theta = np.asarray(theta)
        phi = np.asarray(phi)

        if phi.size > 0 and (phi.min() < 0. or phi.max() > 2. * np.pi):
            raise ValueError("requested phi out of bounds.")

        return SphereBivariateSpline.__call__(self, theta, phi, dtheta=dtheta,
                                              dphi=dphi, grid=grid)


class LSQSphereBivariateSpline(SphereBivariateSpline):
    """
    Weighted least-squares bivariate spline approximation in spherical
    coordinates.

    Determines a smoothing bicubic spline according to a given
    set of knots in the `theta` and `phi` directions.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    theta, phi, r : array_like
        1-D sequences of data points (order is not important). Coordinates
        must be given in radians. Theta must lie within the interval
        ``[0, pi]``, and phi must lie within the interval ``[0, 2pi]``.
    tt, tp : array_like
        Strictly ordered 1-D sequences of knots coordinates.
        Coordinates must satisfy ``0 < tt[i] < pi``, ``0 < tp[i] < 2*pi``.
    w : array_like, optional
        Positive 1-D sequence of weights, of the same length as `theta`, `phi`
        and `r`.
    eps : float, optional
        A threshold for determining the effective rank of an over-determined
        linear system of equations. `eps` should have a value within the
        open interval ``(0, 1)``, the default is 1e-16.

    See Also
    --------
    BivariateSpline :
        a base class for bivariate splines.
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh.
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives

    Notes
    -----
    For more information, see the FITPACK_ site about this function.

    .. _FITPACK: http://www.netlib.org/dierckx/sphere.f

    Examples
    --------
    Suppose we have global data on a coarse grid (the input data does not
    have to be on a grid):

    >>> from scipy.interpolate import LSQSphereBivariateSpline
    >>> import matplotlib.pyplot as plt

    >>> theta = np.linspace(0, np.pi, num=7)
    >>> phi = np.linspace(0, 2*np.pi, num=9)
    >>> data = np.empty((theta.shape[0], phi.shape[0]))
    >>> data[:,0], data[0,:], data[-1,:] = 0., 0., 0.
    >>> data[1:-1,1], data[1:-1,-1] = 1., 1.
    >>> data[1,1:-1], data[-2,1:-1] = 1., 1.
    >>> data[2:-2,2], data[2:-2,-2] = 2., 2.
    >>> data[2,2:-2], data[-3,2:-2] = 2., 2.
    >>> data[3,3:-2] = 3.
    >>> data = np.roll(data, 4, 1)

    We need to set up the interpolator object. Here, we must also specify the
    coordinates of the knots to use.

    >>> lats, lons = np.meshgrid(theta, phi)
    >>> knotst, knotsp = theta.copy(), phi.copy()
    >>> knotst[0] += .0001
    >>> knotst[-1] -= .0001
    >>> knotsp[0] += .0001
    >>> knotsp[-1] -= .0001
    >>> lut = LSQSphereBivariateSpline(lats.ravel(), lons.ravel(),
    ...                                data.T.ravel(), knotst, knotsp)

    As a first test, we'll see what the algorithm returns when run on the
    input coordinates

    >>> data_orig = lut(theta, phi)

    Finally we interpolate the data to a finer grid

    >>> fine_lats = np.linspace(0., np.pi, 70)
    >>> fine_lons = np.linspace(0., 2*np.pi, 90)
    >>> data_lsq = lut(fine_lats, fine_lons)

    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(131)
    >>> ax1.imshow(data, interpolation='nearest')
    >>> ax2 = fig.add_subplot(132)
    >>> ax2.imshow(data_orig, interpolation='nearest')
    >>> ax3 = fig.add_subplot(133)
    >>> ax3.imshow(data_lsq, interpolation='nearest')
    >>> plt.show()

    """

    def __init__(self, theta, phi, r, tt, tp, w=None, eps=1E-16):

        theta, phi, r = np.asarray(theta), np.asarray(phi), np.asarray(r)
        tt, tp = np.asarray(tt), np.asarray(tp)

        if not ((0.0 <= theta).all() and (theta <= np.pi).all()):
            raise ValueError('theta should be between [0, pi]')
        if not ((0.0 <= phi).all() and (phi <= 2*np.pi).all()):
            raise ValueError('phi should be between [0, 2pi]')
        if not ((0.0 < tt).all() and (tt < np.pi).all()):
            raise ValueError('tt should be between (0, pi)')
        if not ((0.0 < tp).all() and (tp < 2*np.pi).all()):
            raise ValueError('tp should be between (0, 2pi)')
        if w is not None:
            w = np.asarray(w)
            if not (w >= 0.0).all():
                raise ValueError('w should be positive')
        if not 0.0 < eps < 1.0:
            raise ValueError('eps should be between (0, 1)')

        if np.issubclass_(w, float):
            w = ones(len(theta)) * w
        nt_, np_ = 8 + len(tt), 8 + len(tp)
        tt_, tp_ = zeros((nt_,), float), zeros((np_,), float)
        tt_[4:-4], tp_[4:-4] = tt, tp
        tt_[-4:], tp_[-4:] = np.pi, 2. * np.pi
        tt_, tp_, c, fp, ier = dfitpack.spherfit_lsq(theta, phi, r, tt_, tp_,
                                                     w=w, eps=eps)
        if ier > 0:
            message = _spherefit_messages.get(ier, 'ier=%s' % (ier))
            raise ValueError(message)

        self.fp = fp
        self.tck = tt_, tp_, c
        self.degrees = (3, 3)

    def __call__(self, theta, phi, dtheta=0, dphi=0, grid=True):

        theta = np.asarray(theta)
        phi = np.asarray(phi)

        if phi.size > 0 and (phi.min() < 0. or phi.max() > 2. * np.pi):
            raise ValueError("requested phi out of bounds.")

        return SphereBivariateSpline.__call__(self, theta, phi, dtheta=dtheta,
                                              dphi=dphi, grid=grid)


_spfit_messages = _surfit_messages.copy()
_spfit_messages[10] = """
ERROR: on entry, the input data are controlled on validity
       the following restrictions must be satisfied.
          -1<=iopt(1)<=1, 0<=iopt(2)<=1, 0<=iopt(3)<=1,
          -1<=ider(1)<=1, 0<=ider(2)<=1, ider(2)=0 if iopt(2)=0.
          -1<=ider(3)<=1, 0<=ider(4)<=1, ider(4)=0 if iopt(3)=0.
          mu >= mumin (see above), mv >= 4, nuest >=8, nvest >= 8,
          kwrk>=5+mu+mv+nuest+nvest,
          lwrk >= 12+nuest*(mv+nvest+3)+nvest*24+4*mu+8*mv+max(nuest,mv+nvest)
          0< u(i-1)<u(i)< pi,i=2,..,mu,
          -pi<=v(1)< pi, v(1)<v(i-1)<v(i)<v(1)+2*pi, i=3,...,mv
          if iopt(1)=-1: 8<=nu<=min(nuest,mu+6+iopt(2)+iopt(3))
                         0<tu(5)<tu(6)<...<tu(nu-4)< pi
                         8<=nv<=min(nvest,mv+7)
                         v(1)<tv(5)<tv(6)<...<tv(nv-4)<v(1)+2*pi
                         the schoenberg-whitney conditions, i.e. there must be
                         subset of grid co-ordinates uu(p) and vv(q) such that
                            tu(p) < uu(p) < tu(p+4) ,p=1,...,nu-4
                            (iopt(2)=1 and iopt(3)=1 also count for a uu-value
                            tv(q) < vv(q) < tv(q+4) ,q=1,...,nv-4
                            (vv(q) is either a value v(j) or v(j)+2*pi)
          if iopt(1)>=0: s>=0
          if s=0: nuest>=mu+6+iopt(2)+iopt(3), nvest>=mv+7
       if one of these conditions is found to be violated,control is
       immediately repassed to the calling program. in that case there is no
       approximation returned."""


class RectSphereBivariateSpline(SphereBivariateSpline):
    """
    Bivariate spline approximation over a rectangular mesh on a sphere.

    Can be used for smoothing data.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    u : array_like
        1-D array of colatitude coordinates in strictly ascending order.
        Coordinates must be given in radians and lie within the open interval
        ``(0, pi)``.
    v : array_like
        1-D array of longitude coordinates in strictly ascending order.
        Coordinates must be given in radians. First element (``v[0]``) must lie
        within the interval ``[-pi, pi)``. Last element (``v[-1]``) must satisfy
        ``v[-1] <= v[0] + 2*pi``.
    r : array_like
        2-D array of data with shape ``(u.size, v.size)``.
    s : float, optional
        Positive smoothing factor defined for estimation condition
        (``s=0`` is for interpolation).
    pole_continuity : bool or (bool, bool), optional
        Order of continuity at the poles ``u=0`` (``pole_continuity[0]``) and
        ``u=pi`` (``pole_continuity[1]``).  The order of continuity at the pole
        will be 1 or 0 when this is True or False, respectively.
        Defaults to False.
    pole_values : float or (float, float), optional
        Data values at the poles ``u=0`` and ``u=pi``.  Either the whole
        parameter or each individual element can be None.  Defaults to None.
    pole_exact : bool or (bool, bool), optional
        Data value exactness at the poles ``u=0`` and ``u=pi``.  If True, the
        value is considered to be the right function value, and it will be
        fitted exactly. If False, the value will be considered to be a data
        value just like the other data values.  Defaults to False.
    pole_flat : bool or (bool, bool), optional
        For the poles at ``u=0`` and ``u=pi``, specify whether or not the
        approximation has vanishing derivatives.  Defaults to False.

    See Also
    --------
    BivariateSpline :
        a base class for bivariate splines.
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh.
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives

    Notes
    -----
    Currently, only the smoothing spline approximation (``iopt[0] = 0`` and
    ``iopt[0] = 1`` in the FITPACK routine) is supported.  The exact
    least-squares spline approximation is not implemented yet.

    When actually performing the interpolation, the requested `v` values must
    lie within the same length 2pi interval that the original `v` values were
    chosen from.

    For more information, see the FITPACK_ site about this function.

    .. _FITPACK: http://www.netlib.org/dierckx/spgrid.f

    Examples
    --------
    Suppose we have global data on a coarse grid

    >>> lats = np.linspace(10, 170, 9) * np.pi / 180.
    >>> lons = np.linspace(0, 350, 18) * np.pi / 180.
    >>> data = np.dot(np.atleast_2d(90. - np.linspace(-80., 80., 18)).T,
    ...               np.atleast_2d(180. - np.abs(np.linspace(0., 350., 9)))).T

    We want to interpolate it to a global one-degree grid

    >>> new_lats = np.linspace(1, 180, 180) * np.pi / 180
    >>> new_lons = np.linspace(1, 360, 360) * np.pi / 180
    >>> new_lats, new_lons = np.meshgrid(new_lats, new_lons)

    We need to set up the interpolator object

    >>> from scipy.interpolate import RectSphereBivariateSpline
    >>> lut = RectSphereBivariateSpline(lats, lons, data)

    Finally we interpolate the data.  The `RectSphereBivariateSpline` object
    only takes 1-D arrays as input, therefore we need to do some reshaping.

    >>> data_interp = lut.ev(new_lats.ravel(),
    ...                      new_lons.ravel()).reshape((360, 180)).T

    Looking at the original and the interpolated data, one can see that the
    interpolant reproduces the original data very well:

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(211)
    >>> ax1.imshow(data, interpolation='nearest')
    >>> ax2 = fig.add_subplot(212)
    >>> ax2.imshow(data_interp, interpolation='nearest')
    >>> plt.show()

    Choosing the optimal value of ``s`` can be a delicate task. Recommended
    values for ``s`` depend on the accuracy of the data values.  If the user
    has an idea of the statistical errors on the data, she can also find a
    proper estimate for ``s``. By assuming that, if she specifies the
    right ``s``, the interpolator will use a spline ``f(u,v)`` which exactly
    reproduces the function underlying the data, she can evaluate
    ``sum((r(i,j)-s(u(i),v(j)))**2)`` to find a good estimate for this ``s``.
    For example, if she knows that the statistical errors on her
    ``r(i,j)``-values are not greater than 0.1, she may expect that a good
    ``s`` should have a value not larger than ``u.size * v.size * (0.1)**2``.

    If nothing is known about the statistical error in ``r(i,j)``, ``s`` must
    be determined by trial and error.  The best is then to start with a very
    large value of ``s`` (to determine the least-squares polynomial and the
    corresponding upper bound ``fp0`` for ``s``) and then to progressively
    decrease the value of ``s`` (say by a factor 10 in the beginning, i.e.
    ``s = fp0 / 10, fp0 / 100, ...``  and more carefully as the approximation
    shows more detail) to obtain closer fits.

    The interpolation results for different values of ``s`` give some insight
    into this process:

    >>> fig2 = plt.figure()
    >>> s = [3e9, 2e9, 1e9, 1e8]
    >>> for ii in range(len(s)):
    ...     lut = RectSphereBivariateSpline(lats, lons, data, s=s[ii])
    ...     data_interp = lut.ev(new_lats.ravel(),
    ...                          new_lons.ravel()).reshape((360, 180)).T
    ...     ax = fig2.add_subplot(2, 2, ii+1)
    ...     ax.imshow(data_interp, interpolation='nearest')
    ...     ax.set_title("s = %g" % s[ii])
    >>> plt.show()

    """

    def __init__(self, u, v, r, s=0., pole_continuity=False, pole_values=None,
                 pole_exact=False, pole_flat=False):
        iopt = np.array([0, 0, 0], dtype=dfitpack_int)
        ider = np.array([-1, 0, -1, 0], dtype=dfitpack_int)
        if pole_values is None:
            pole_values = (None, None)
        elif isinstance(pole_values, (float, np.float32, np.float64)):
            pole_values = (pole_values, pole_values)
        if isinstance(pole_continuity, bool):
            pole_continuity = (pole_continuity, pole_continuity)
        if isinstance(pole_exact, bool):
            pole_exact = (pole_exact, pole_exact)
        if isinstance(pole_flat, bool):
            pole_flat = (pole_flat, pole_flat)

        r0, r1 = pole_values
        iopt[1:] = pole_continuity
        if r0 is None:
            ider[0] = -1
        else:
            ider[0] = pole_exact[0]

        if r1 is None:
            ider[2] = -1
        else:
            ider[2] = pole_exact[1]

        ider[1], ider[3] = pole_flat

        u, v = np.ravel(u), np.ravel(v)
        r = np.asarray(r)

        if not (0.0 < u[0] and u[-1] < np.pi):
            raise ValueError('u should be between (0, pi)')
        if not -np.pi <= v[0] < np.pi:
            raise ValueError('v[0] should be between [-pi, pi)')
        if not v[-1] <= v[0] + 2*np.pi:
            raise ValueError('v[-1] should be v[0] + 2pi or less ')

        if not np.all(np.diff(u) > 0.0):
            raise ValueError('u must be strictly increasing')
        if not np.all(np.diff(v) > 0.0):
            raise ValueError('v must be strictly increasing')

        if not u.size == r.shape[0]:
            raise ValueError('u dimension of r must have same number of '
                             'elements as u')
        if not v.size == r.shape[1]:
            raise ValueError('v dimension of r must have same number of '
                             'elements as v')

        if pole_continuity[1] is False and pole_flat[1] is True:
            raise ValueError('if pole_continuity is False, so must be '
                             'pole_flat')
        if pole_continuity[0] is False and pole_flat[0] is True:
            raise ValueError('if pole_continuity is False, so must be '
                             'pole_flat')

        if not s >= 0.0:
            raise ValueError('s should be positive')

        r = np.ravel(r)
        nu, tu, nv, tv, c, fp, ier = dfitpack.regrid_smth_spher(iopt, ider,
                                                                u.copy(),
                                                                v.copy(),
                                                                r.copy(),
                                                                r0, r1, s)

        if ier not in [0, -1, -2]:
            msg = _spfit_messages.get(ier, 'ier=%s' % (ier))
            raise ValueError(msg)

        self.fp = fp
        self.tck = tu[:nu], tv[:nv], c[:(nu - 4) * (nv-4)]
        self.degrees = (3, 3)
        self.v0 = v[0]

    def __call__(self, theta, phi, dtheta=0, dphi=0, grid=True):

        theta = np.asarray(theta)
        phi = np.asarray(phi)

        return SphereBivariateSpline.__call__(self, theta, phi, dtheta=dtheta,
                                              dphi=dphi, grid=grid)
