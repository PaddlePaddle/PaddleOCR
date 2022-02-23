"""
fitpack (dierckx in netlib) --- A Python-C wrapper to FITPACK (by P. Dierckx).
        FITPACK is a collection of FORTRAN programs for curve and surface
        fitting with splines and tensor product splines.

See
 https://web.archive.org/web/20010524124604/http://www.cs.kuleuven.ac.be:80/cwis/research/nalag/research/topics/fitpack.html
or
 http://www.netlib.org/dierckx/

Copyright 2002 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the SciPy (BSD style) license. See LICENSE.txt that came with
this distribution for specifics.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.

TODO: Make interfaces to the following fitpack functions:
    For univariate splines: cocosp, concon, fourco, insert
    For bivariate splines: profil, regrid, parsur, surev
"""

__all__ = ['splrep', 'splprep', 'splev', 'splint', 'sproot', 'spalde',
           'bisplrep', 'bisplev', 'insert', 'splder', 'splantider']

import warnings
import numpy as np
from . import _fitpack
from numpy import (atleast_1d, array, ones, zeros, sqrt, ravel, transpose,
                   empty, iinfo, asarray)

# Try to replace _fitpack interface with
#  f2py-generated version
from . import dfitpack


dfitpack_int = dfitpack.types.intvar.dtype


def _int_overflow(x, msg=None):
    """Cast the value to an dfitpack_int and raise an OverflowError if the value
    cannot fit.
    """
    if x > iinfo(dfitpack_int).max:
        if msg is None:
            msg = '%r cannot fit into an %r' % (x, dfitpack_int)
        raise OverflowError(msg)
    return dfitpack_int.type(x)


_iermess = {
    0: ["The spline has a residual sum of squares fp such that "
        "abs(fp-s)/s<=0.001", None],
    -1: ["The spline is an interpolating spline (fp=0)", None],
    -2: ["The spline is weighted least-squares polynomial of degree k.\n"
         "fp gives the upper bound fp0 for the smoothing factor s", None],
    1: ["The required storage space exceeds the available storage space.\n"
        "Probable causes: data (x,y) size is too small or smoothing parameter"
        "\ns is too small (fp>s).", ValueError],
    2: ["A theoretically impossible result when finding a smoothing spline\n"
        "with fp = s. Probable cause: s too small. (abs(fp-s)/s>0.001)",
        ValueError],
    3: ["The maximal number of iterations (20) allowed for finding smoothing\n"
        "spline with fp=s has been reached. Probable cause: s too small.\n"
        "(abs(fp-s)/s>0.001)", ValueError],
    10: ["Error on input data", ValueError],
    'unknown': ["An error occurred", TypeError]
}

_iermess2 = {
    0: ["The spline has a residual sum of squares fp such that "
        "abs(fp-s)/s<=0.001", None],
    -1: ["The spline is an interpolating spline (fp=0)", None],
    -2: ["The spline is weighted least-squares polynomial of degree kx and ky."
         "\nfp gives the upper bound fp0 for the smoothing factor s", None],
    -3: ["Warning. The coefficients of the spline have been computed as the\n"
         "minimal norm least-squares solution of a rank deficient system.",
         None],
    1: ["The required storage space exceeds the available storage space.\n"
        "Probable causes: nxest or nyest too small or s is too small. (fp>s)",
        ValueError],
    2: ["A theoretically impossible result when finding a smoothing spline\n"
        "with fp = s. Probable causes: s too small or badly chosen eps.\n"
        "(abs(fp-s)/s>0.001)", ValueError],
    3: ["The maximal number of iterations (20) allowed for finding smoothing\n"
        "spline with fp=s has been reached. Probable cause: s too small.\n"
        "(abs(fp-s)/s>0.001)", ValueError],
    4: ["No more knots can be added because the number of B-spline\n"
        "coefficients already exceeds the number of data points m.\n"
        "Probable causes: either s or m too small. (fp>s)", ValueError],
    5: ["No more knots can be added because the additional knot would\n"
        "coincide with an old one. Probable cause: s too small or too large\n"
        "a weight to an inaccurate data point. (fp>s)", ValueError],
    10: ["Error on input data", ValueError],
    11: ["rwrk2 too small, i.e., there is not enough workspace for computing\n"
         "the minimal least-squares solution of a rank deficient system of\n"
         "linear equations.", ValueError],
    'unknown': ["An error occurred", TypeError]
}

_parcur_cache = {'t': array([], float), 'wrk': array([], float),
                 'iwrk': array([], dfitpack_int), 'u': array([], float),
                 'ub': 0, 'ue': 1}


def splprep(x, w=None, u=None, ub=None, ue=None, k=3, task=0, s=None, t=None,
            full_output=0, nest=None, per=0, quiet=1):
    """
    Find the B-spline representation of an N-D curve.

    Given a list of N rank-1 arrays, `x`, which represent a curve in
    N-dimensional space parametrized by `u`, find a smooth approximating
    spline curve g(`u`). Uses the FORTRAN routine parcur from FITPACK.

    Parameters
    ----------
    x : array_like
        A list of sample vector arrays representing the curve.
    w : array_like, optional
        Strictly positive rank-1 array of weights the same length as `x[0]`.
        The weights are used in computing the weighted least-squares spline
        fit. If the errors in the `x` values have standard-deviation given by
        the vector d, then `w` should be 1/d. Default is ``ones(len(x[0]))``.
    u : array_like, optional
        An array of parameter values. If not given, these values are
        calculated automatically as ``M = len(x[0])``, where

            v[0] = 0

            v[i] = v[i-1] + distance(`x[i]`, `x[i-1]`)

            u[i] = v[i] / v[M-1]

    ub, ue : int, optional
        The end-points of the parameters interval. Defaults to
        u[0] and u[-1].
    k : int, optional
        Degree of the spline. Cubic splines are recommended.
        Even values of `k` should be avoided especially with a small s-value.
        ``1 <= k <= 5``, default is 3.
    task : int, optional
        If task==0 (default), find t and c for a given smoothing factor, s.
        If task==1, find t and c for another value of the smoothing factor, s.
        There must have been a previous call with task=0 or task=1
        for the same set of data.
        If task=-1 find the weighted least square spline for a given set of
        knots, t.
    s : float, optional
        A smoothing condition. The amount of smoothness is determined by
        satisfying the conditions: ``sum((w * (y - g))**2,axis=0) <= s``,
        where g(x) is the smoothed interpolation of (x,y).  The user can
        use `s` to control the trade-off between closeness and smoothness
        of fit. Larger `s` means more smoothing while smaller values of `s`
        indicate less smoothing. Recommended values of `s` depend on the
        weights, w.  If the weights represent the inverse of the
        standard-deviation of y, then a good `s` value should be found in
        the range ``(m-sqrt(2*m),m+sqrt(2*m))``, where m is the number of
        data points in x, y, and w.
    t : int, optional
        The knots needed for task=-1.
    full_output : int, optional
        If non-zero, then return optional outputs.
    nest : int, optional
        An over-estimate of the total number of knots of the spline to
        help in determining the storage space. By default nest=m/2.
        Always large enough is nest=m+k+1.
    per : int, optional
       If non-zero, data points are considered periodic with period
       ``x[m-1] - x[0]`` and a smooth periodic spline approximation is
       returned.  Values of ``y[m-1]`` and ``w[m-1]`` are not used.
    quiet : int, optional
         Non-zero to suppress messages.
         This parameter is deprecated; use standard Python warning filters
         instead.

    Returns
    -------
    tck : tuple
        A tuple (t,c,k) containing the vector of knots, the B-spline
        coefficients, and the degree of the spline.
    u : array
        An array of the values of the parameter.
    fp : float
        The weighted sum of squared residuals of the spline approximation.
    ier : int
        An integer flag about splrep success.  Success is indicated
        if ier<=0. If ier in [1,2,3] an error occurred but was not raised.
        Otherwise an error is raised.
    msg : str
        A message corresponding to the integer flag, ier.

    See Also
    --------
    splrep, splev, sproot, spalde, splint,
    bisplrep, bisplev
    UnivariateSpline, BivariateSpline

    Notes
    -----
    See `splev` for evaluation of the spline and its derivatives.
    The number of dimensions N must be smaller than 11.

    References
    ----------
    .. [1] P. Dierckx, "Algorithms for smoothing data with periodic and
        parametric splines, Computer Graphics and Image Processing",
        20 (1982) 171-184.
    .. [2] P. Dierckx, "Algorithms for smoothing data with periodic and
        parametric splines", report tw55, Dept. Computer Science,
        K.U.Leuven, 1981.
    .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs on
        Numerical Analysis, Oxford University Press, 1993.

    """
    if task <= 0:
        _parcur_cache = {'t': array([], float), 'wrk': array([], float),
                         'iwrk': array([], dfitpack_int), 'u': array([], float),
                         'ub': 0, 'ue': 1}
    x = atleast_1d(x)
    idim, m = x.shape
    if per:
        for i in range(idim):
            if x[i][0] != x[i][-1]:
                if quiet < 2:
                    warnings.warn(RuntimeWarning('Setting x[%d][%d]=x[%d][0]' %
                                                 (i, m, i)))
                x[i][-1] = x[i][0]
    if not 0 < idim < 11:
        raise TypeError('0 < idim < 11 must hold')
    if w is None:
        w = ones(m, float)
    else:
        w = atleast_1d(w)
    ipar = (u is not None)
    if ipar:
        _parcur_cache['u'] = u
        if ub is None:
            _parcur_cache['ub'] = u[0]
        else:
            _parcur_cache['ub'] = ub
        if ue is None:
            _parcur_cache['ue'] = u[-1]
        else:
            _parcur_cache['ue'] = ue
    else:
        _parcur_cache['u'] = zeros(m, float)
    if not (1 <= k <= 5):
        raise TypeError('1 <= k= %d <=5 must hold' % k)
    if not (-1 <= task <= 1):
        raise TypeError('task must be -1, 0 or 1')
    if (not len(w) == m) or (ipar == 1 and (not len(u) == m)):
        raise TypeError('Mismatch of input dimensions')
    if s is None:
        s = m - sqrt(2*m)
    if t is None and task == -1:
        raise TypeError('Knots must be given for task=-1')
    if t is not None:
        _parcur_cache['t'] = atleast_1d(t)
    n = len(_parcur_cache['t'])
    if task == -1 and n < 2*k + 2:
        raise TypeError('There must be at least 2*k+2 knots for task=-1')
    if m <= k:
        raise TypeError('m > k must hold')
    if nest is None:
        nest = m + 2*k

    if (task >= 0 and s == 0) or (nest < 0):
        if per:
            nest = m + 2*k
        else:
            nest = m + k + 1
    nest = max(nest, 2*k + 3)
    u = _parcur_cache['u']
    ub = _parcur_cache['ub']
    ue = _parcur_cache['ue']
    t = _parcur_cache['t']
    wrk = _parcur_cache['wrk']
    iwrk = _parcur_cache['iwrk']
    t, c, o = _fitpack._parcur(ravel(transpose(x)), w, u, ub, ue, k,
                               task, ipar, s, t, nest, wrk, iwrk, per)
    _parcur_cache['u'] = o['u']
    _parcur_cache['ub'] = o['ub']
    _parcur_cache['ue'] = o['ue']
    _parcur_cache['t'] = t
    _parcur_cache['wrk'] = o['wrk']
    _parcur_cache['iwrk'] = o['iwrk']
    ier = o['ier']
    fp = o['fp']
    n = len(t)
    u = o['u']
    c.shape = idim, n - k - 1
    tcku = [t, list(c), k], u
    if ier <= 0 and not quiet:
        warnings.warn(RuntimeWarning(_iermess[ier][0] +
                                     "\tk=%d n=%d m=%d fp=%f s=%f" %
                                     (k, len(t), m, fp, s)))
    if ier > 0 and not full_output:
        if ier in [1, 2, 3]:
            warnings.warn(RuntimeWarning(_iermess[ier][0]))
        else:
            try:
                raise _iermess[ier][1](_iermess[ier][0])
            except KeyError as e:
                raise _iermess['unknown'][1](_iermess['unknown'][0]) from e
    if full_output:
        try:
            return tcku, fp, ier, _iermess[ier][0]
        except KeyError:
            return tcku, fp, ier, _iermess['unknown'][0]
    else:
        return tcku


_curfit_cache = {'t': array([], float), 'wrk': array([], float),
                 'iwrk': array([], dfitpack_int)}


def splrep(x, y, w=None, xb=None, xe=None, k=3, task=0, s=None, t=None,
           full_output=0, per=0, quiet=1):
    """
    Find the B-spline representation of 1-D curve.

    Given the set of data points ``(x[i], y[i])`` determine a smooth spline
    approximation of degree k on the interval ``xb <= x <= xe``.

    Parameters
    ----------
    x, y : array_like
        The data points defining a curve y = f(x).
    w : array_like, optional
        Strictly positive rank-1 array of weights the same length as x and y.
        The weights are used in computing the weighted least-squares spline
        fit. If the errors in the y values have standard-deviation given by the
        vector d, then w should be 1/d. Default is ones(len(x)).
    xb, xe : float, optional
        The interval to fit.  If None, these default to x[0] and x[-1]
        respectively.
    k : int, optional
        The order of the spline fit. It is recommended to use cubic splines.
        Even order splines should be avoided especially with small s values.
        1 <= k <= 5
    task : {1, 0, -1}, optional
        If task==0 find t and c for a given smoothing factor, s.

        If task==1 find t and c for another value of the smoothing factor, s.
        There must have been a previous call with task=0 or task=1 for the same
        set of data (t will be stored an used internally)

        If task=-1 find the weighted least square spline for a given set of
        knots, t. These should be interior knots as knots on the ends will be
        added automatically.
    s : float, optional
        A smoothing condition. The amount of smoothness is determined by
        satisfying the conditions: sum((w * (y - g))**2,axis=0) <= s, where g(x)
        is the smoothed interpolation of (x,y). The user can use s to control
        the tradeoff between closeness and smoothness of fit. Larger s means
        more smoothing while smaller values of s indicate less smoothing.
        Recommended values of s depend on the weights, w. If the weights
        represent the inverse of the standard-deviation of y, then a good s
        value should be found in the range (m-sqrt(2*m),m+sqrt(2*m)) where m is
        the number of datapoints in x, y, and w. default : s=m-sqrt(2*m) if
        weights are supplied. s = 0.0 (interpolating) if no weights are
        supplied.
    t : array_like, optional
        The knots needed for task=-1. If given then task is automatically set
        to -1.
    full_output : bool, optional
        If non-zero, then return optional outputs.
    per : bool, optional
        If non-zero, data points are considered periodic with period x[m-1] -
        x[0] and a smooth periodic spline approximation is returned. Values of
        y[m-1] and w[m-1] are not used.
    quiet : bool, optional
        Non-zero to suppress messages.
        This parameter is deprecated; use standard Python warning filters
        instead.

    Returns
    -------
    tck : tuple
        (t,c,k) a tuple containing the vector of knots, the B-spline
        coefficients, and the degree of the spline.
    fp : array, optional
        The weighted sum of squared residuals of the spline approximation.
    ier : int, optional
        An integer flag about splrep success. Success is indicated if ier<=0.
        If ier in [1,2,3] an error occurred but was not raised. Otherwise an
        error is raised.
    msg : str, optional
        A message corresponding to the integer flag, ier.

    See Also
    --------
    UnivariateSpline, BivariateSpline
    splprep, splev, sproot, spalde, splint
    bisplrep, bisplev

    Notes
    -----
    See splev for evaluation of the spline and its derivatives. Uses the
    FORTRAN routine curfit from FITPACK.

    The user is responsible for assuring that the values of *x* are unique.
    Otherwise, *splrep* will not return sensible results.

    If provided, knots `t` must satisfy the Schoenberg-Whitney conditions,
    i.e., there must be a subset of data points ``x[j]`` such that
    ``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.

    References
    ----------
    Based on algorithms described in [1]_, [2]_, [3]_, and [4]_:

    .. [1] P. Dierckx, "An algorithm for smoothing, differentiation and
       integration of experimental data using spline functions",
       J.Comp.Appl.Maths 1 (1975) 165-184.
    .. [2] P. Dierckx, "A fast algorithm for smoothing data on a rectangular
       grid while using spline functions", SIAM J.Numer.Anal. 19 (1982)
       1286-1304.
    .. [3] P. Dierckx, "An improved algorithm for curve fitting with spline
       functions", report tw54, Dept. Computer Science,K.U. Leuven, 1981.
    .. [4] P. Dierckx, "Curve and surface fitting with splines", Monographs on
       Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import splev, splrep
    >>> x = np.linspace(0, 10, 10)
    >>> y = np.sin(x)
    >>> tck = splrep(x, y)
    >>> x2 = np.linspace(0, 10, 200)
    >>> y2 = splev(x2, tck)
    >>> plt.plot(x, y, 'o', x2, y2)
    >>> plt.show()

    """
    if task <= 0:
        _curfit_cache = {}
    x, y = map(atleast_1d, [x, y])
    m = len(x)
    if w is None:
        w = ones(m, float)
        if s is None:
            s = 0.0
    else:
        w = atleast_1d(w)
        if s is None:
            s = m - sqrt(2*m)
    if not len(w) == m:
        raise TypeError('len(w)=%d is not equal to m=%d' % (len(w), m))
    if (m != len(y)) or (m != len(w)):
        raise TypeError('Lengths of the first three arguments (x,y,w) must '
                        'be equal')
    if not (1 <= k <= 5):
        raise TypeError('Given degree of the spline (k=%d) is not supported. '
                        '(1<=k<=5)' % k)
    if m <= k:
        raise TypeError('m > k must hold')
    if xb is None:
        xb = x[0]
    if xe is None:
        xe = x[-1]
    if not (-1 <= task <= 1):
        raise TypeError('task must be -1, 0 or 1')
    if t is not None:
        task = -1
    if task == -1:
        if t is None:
            raise TypeError('Knots must be given for task=-1')
        numknots = len(t)
        _curfit_cache['t'] = empty((numknots + 2*k + 2,), float)
        _curfit_cache['t'][k+1:-k-1] = t
        nest = len(_curfit_cache['t'])
    elif task == 0:
        if per:
            nest = max(m + 2*k, 2*k + 3)
        else:
            nest = max(m + k + 1, 2*k + 3)
        t = empty((nest,), float)
        _curfit_cache['t'] = t
    if task <= 0:
        if per:
            _curfit_cache['wrk'] = empty((m*(k + 1) + nest*(8 + 5*k),), float)
        else:
            _curfit_cache['wrk'] = empty((m*(k + 1) + nest*(7 + 3*k),), float)
        _curfit_cache['iwrk'] = empty((nest,), dfitpack_int)
    try:
        t = _curfit_cache['t']
        wrk = _curfit_cache['wrk']
        iwrk = _curfit_cache['iwrk']
    except KeyError as e:
        raise TypeError("must call with task=1 only after"
                        " call with task=0,-1") from e
    if not per:
        n, c, fp, ier = dfitpack.curfit(task, x, y, w, t, wrk, iwrk,
                                        xb, xe, k, s)
    else:
        n, c, fp, ier = dfitpack.percur(task, x, y, w, t, wrk, iwrk, k, s)
    tck = (t[:n], c[:n], k)
    if ier <= 0 and not quiet:
        _mess = (_iermess[ier][0] + "\tk=%d n=%d m=%d fp=%f s=%f" %
                 (k, len(t), m, fp, s))
        warnings.warn(RuntimeWarning(_mess))
    if ier > 0 and not full_output:
        if ier in [1, 2, 3]:
            warnings.warn(RuntimeWarning(_iermess[ier][0]))
        else:
            try:
                raise _iermess[ier][1](_iermess[ier][0])
            except KeyError as e:
                raise _iermess['unknown'][1](_iermess['unknown'][0]) from e
    if full_output:
        try:
            return tck, fp, ier, _iermess[ier][0]
        except KeyError:
            return tck, fp, ier, _iermess['unknown'][0]
    else:
        return tck


def splev(x, tck, der=0, ext=0):
    """
    Evaluate a B-spline or its derivatives.

    Given the knots and coefficients of a B-spline representation, evaluate
    the value of the smoothing polynomial and its derivatives. This is a
    wrapper around the FORTRAN routines splev and splder of FITPACK.

    Parameters
    ----------
    x : array_like
        An array of points at which to return the value of the smoothed
        spline or its derivatives. If `tck` was returned from `splprep`,
        then the parameter values, u should be given.
    tck : tuple
        A sequence of length 3 returned by `splrep` or `splprep` containing
        the knots, coefficients, and degree of the spline.
    der : int, optional
        The order of derivative of the spline to compute (must be less than
        or equal to k).
    ext : int, optional
        Controls the value returned for elements of ``x`` not in the
        interval defined by the knot sequence.

        * if ext=0, return the extrapolated value.
        * if ext=1, return 0
        * if ext=2, raise a ValueError
        * if ext=3, return the boundary value.

        The default value is 0.

    Returns
    -------
    y : ndarray or list of ndarrays
        An array of values representing the spline function evaluated at
        the points in ``x``.  If `tck` was returned from `splprep`, then this
        is a list of arrays representing the curve in N-D space.

    See Also
    --------
    splprep, splrep, sproot, spalde, splint
    bisplrep, bisplev

    References
    ----------
    .. [1] C. de Boor, "On calculating with b-splines", J. Approximation
        Theory, 6, p.50-62, 1972.
    .. [2] M.G. Cox, "The numerical evaluation of b-splines", J. Inst. Maths
        Applics, 10, p.134-149, 1972.
    .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs
        on Numerical Analysis, Oxford University Press, 1993.

    """
    t, c, k = tck
    try:
        c[0][0]
        parametric = True
    except Exception:
        parametric = False
    if parametric:
        return list(map(lambda c, x=x, t=t, k=k, der=der:
                        splev(x, [t, c, k], der, ext), c))
    else:
        if not (0 <= der <= k):
            raise ValueError("0<=der=%d<=k=%d must hold" % (der, k))
        if ext not in (0, 1, 2, 3):
            raise ValueError("ext = %s not in (0, 1, 2, 3) " % ext)

        x = asarray(x)
        shape = x.shape
        x = atleast_1d(x).ravel()
        y, ier = _fitpack._spl_(x, der, t, c, k, ext)

        if ier == 10:
            raise ValueError("Invalid input data")
        if ier == 1:
            raise ValueError("Found x value not in the domain")
        if ier:
            raise TypeError("An error occurred")

        return y.reshape(shape)


def splint(a, b, tck, full_output=0):
    """
    Evaluate the definite integral of a B-spline.

    Given the knots and coefficients of a B-spline, evaluate the definite
    integral of the smoothing polynomial between two given points.

    Parameters
    ----------
    a, b : float
        The end-points of the integration interval.
    tck : tuple
        A tuple (t,c,k) containing the vector of knots, the B-spline
        coefficients, and the degree of the spline (see `splev`).
    full_output : int, optional
        Non-zero to return optional output.

    Returns
    -------
    integral : float
        The resulting integral.
    wrk : ndarray
        An array containing the integrals of the normalized B-splines
        defined on the set of knots.

    Notes
    -----
    splint silently assumes that the spline function is zero outside the data
    interval (a, b).

    See Also
    --------
    splprep, splrep, sproot, spalde, splev
    bisplrep, bisplev
    UnivariateSpline, BivariateSpline

    References
    ----------
    .. [1] P.W. Gaffney, The calculation of indefinite integrals of b-splines",
        J. Inst. Maths Applics, 17, p.37-41, 1976.
    .. [2] P. Dierckx, "Curve and surface fitting with splines", Monographs
        on Numerical Analysis, Oxford University Press, 1993.

    """
    t, c, k = tck
    try:
        c[0][0]
        parametric = True
    except Exception:
        parametric = False
    if parametric:
        return list(map(lambda c, a=a, b=b, t=t, k=k:
                        splint(a, b, [t, c, k]), c))
    else:
        aint, wrk = _fitpack._splint(t, c, k, a, b)
        if full_output:
            return aint, wrk
        else:
            return aint


def sproot(tck, mest=10):
    """
    Find the roots of a cubic B-spline.

    Given the knots (>=8) and coefficients of a cubic B-spline return the
    roots of the spline.

    Parameters
    ----------
    tck : tuple
        A tuple (t,c,k) containing the vector of knots,
        the B-spline coefficients, and the degree of the spline.
        The number of knots must be >= 8, and the degree must be 3.
        The knots must be a montonically increasing sequence.
    mest : int, optional
        An estimate of the number of zeros (Default is 10).

    Returns
    -------
    zeros : ndarray
        An array giving the roots of the spline.

    See also
    --------
    splprep, splrep, splint, spalde, splev
    bisplrep, bisplev
    UnivariateSpline, BivariateSpline


    References
    ----------
    .. [1] C. de Boor, "On calculating with b-splines", J. Approximation
        Theory, 6, p.50-62, 1972.
    .. [2] M.G. Cox, "The numerical evaluation of b-splines", J. Inst. Maths
        Applics, 10, p.134-149, 1972.
    .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs
        on Numerical Analysis, Oxford University Press, 1993.

    """
    t, c, k = tck
    if k != 3:
        raise ValueError("sproot works only for cubic (k=3) splines")
    try:
        c[0][0]
        parametric = True
    except Exception:
        parametric = False
    if parametric:
        return list(map(lambda c, t=t, k=k, mest=mest:
                        sproot([t, c, k], mest), c))
    else:
        if len(t) < 8:
            raise TypeError("The number of knots %d>=8" % len(t))
        z, ier = _fitpack._sproot(t, c, k, mest)
        if ier == 10:
            raise TypeError("Invalid input data. "
                            "t1<=..<=t4<t5<..<tn-3<=..<=tn must hold.")
        if ier == 0:
            return z
        if ier == 1:
            warnings.warn(RuntimeWarning("The number of zeros exceeds mest"))
            return z
        raise TypeError("Unknown error")


def spalde(x, tck):
    """
    Evaluate all derivatives of a B-spline.

    Given the knots and coefficients of a cubic B-spline compute all
    derivatives up to order k at a point (or set of points).

    Parameters
    ----------
    x : array_like
        A point or a set of points at which to evaluate the derivatives.
        Note that ``t(k) <= x <= t(n-k+1)`` must hold for each `x`.
    tck : tuple
        A tuple (t,c,k) containing the vector of knots,
        the B-spline coefficients, and the degree of the spline.

    Returns
    -------
    results : {ndarray, list of ndarrays}
        An array (or a list of arrays) containing all derivatives
        up to order k inclusive for each point `x`.

    See Also
    --------
    splprep, splrep, splint, sproot, splev, bisplrep, bisplev,
    UnivariateSpline, BivariateSpline

    References
    ----------
    .. [1] de Boor C : On calculating with b-splines, J. Approximation Theory
       6 (1972) 50-62.
    .. [2] Cox M.G. : The numerical evaluation of b-splines, J. Inst. Maths
       applics 10 (1972) 134-149.
    .. [3] Dierckx P. : Curve and surface fitting with splines, Monographs on
       Numerical Analysis, Oxford University Press, 1993.

    """
    t, c, k = tck
    try:
        c[0][0]
        parametric = True
    except Exception:
        parametric = False
    if parametric:
        return list(map(lambda c, x=x, t=t, k=k:
                        spalde(x, [t, c, k]), c))
    else:
        x = atleast_1d(x)
        if len(x) > 1:
            return list(map(lambda x, tck=tck: spalde(x, tck), x))
        d, ier = _fitpack._spalde(t, c, k, x[0])
        if ier == 0:
            return d
        if ier == 10:
            raise TypeError("Invalid input data. t(k)<=x<=t(n-k+1) must hold.")
        raise TypeError("Unknown error")

# def _curfit(x,y,w=None,xb=None,xe=None,k=3,task=0,s=None,t=None,
#           full_output=0,nest=None,per=0,quiet=1):


_surfit_cache = {'tx': array([], float), 'ty': array([], float),
                 'wrk': array([], float), 'iwrk': array([], dfitpack_int)}


def bisplrep(x, y, z, w=None, xb=None, xe=None, yb=None, ye=None,
             kx=3, ky=3, task=0, s=None, eps=1e-16, tx=None, ty=None,
             full_output=0, nxest=None, nyest=None, quiet=1):
    """
    Find a bivariate B-spline representation of a surface.

    Given a set of data points (x[i], y[i], z[i]) representing a surface
    z=f(x,y), compute a B-spline representation of the surface. Based on
    the routine SURFIT from FITPACK.

    Parameters
    ----------
    x, y, z : ndarray
        Rank-1 arrays of data points.
    w : ndarray, optional
        Rank-1 array of weights. By default ``w=np.ones(len(x))``.
    xb, xe : float, optional
        End points of approximation interval in `x`.
        By default ``xb = x.min(), xe=x.max()``.
    yb, ye : float, optional
        End points of approximation interval in `y`.
        By default ``yb=y.min(), ye = y.max()``.
    kx, ky : int, optional
        The degrees of the spline (1 <= kx, ky <= 5).
        Third order (kx=ky=3) is recommended.
    task : int, optional
        If task=0, find knots in x and y and coefficients for a given
        smoothing factor, s.
        If task=1, find knots and coefficients for another value of the
        smoothing factor, s.  bisplrep must have been previously called
        with task=0 or task=1.
        If task=-1, find coefficients for a given set of knots tx, ty.
    s : float, optional
        A non-negative smoothing factor. If weights correspond
        to the inverse of the standard-deviation of the errors in z,
        then a good s-value should be found in the range
        ``(m-sqrt(2*m),m+sqrt(2*m))`` where m=len(x).
    eps : float, optional
        A threshold for determining the effective rank of an
        over-determined linear system of equations (0 < eps < 1).
        `eps` is not likely to need changing.
    tx, ty : ndarray, optional
        Rank-1 arrays of the knots of the spline for task=-1
    full_output : int, optional
        Non-zero to return optional outputs.
    nxest, nyest : int, optional
        Over-estimates of the total number of knots. If None then
        ``nxest = max(kx+sqrt(m/2),2*kx+3)``,
        ``nyest = max(ky+sqrt(m/2),2*ky+3)``.
    quiet : int, optional
        Non-zero to suppress printing of messages.
        This parameter is deprecated; use standard Python warning filters
        instead.

    Returns
    -------
    tck : array_like
        A list [tx, ty, c, kx, ky] containing the knots (tx, ty) and
        coefficients (c) of the bivariate B-spline representation of the
        surface along with the degree of the spline.
    fp : ndarray
        The weighted sum of squared residuals of the spline approximation.
    ier : int
        An integer flag about splrep success. Success is indicated if
        ier<=0. If ier in [1,2,3] an error occurred but was not raised.
        Otherwise an error is raised.
    msg : str
        A message corresponding to the integer flag, ier.

    See Also
    --------
    splprep, splrep, splint, sproot, splev
    UnivariateSpline, BivariateSpline

    Notes
    -----
    See `bisplev` to evaluate the value of the B-spline given its tck
    representation.

    References
    ----------
    .. [1] Dierckx P.:An algorithm for surface fitting with spline functions
       Ima J. Numer. Anal. 1 (1981) 267-283.
    .. [2] Dierckx P.:An algorithm for surface fitting with spline functions
       report tw50, Dept. Computer Science,K.U.Leuven, 1980.
    .. [3] Dierckx P.:Curve and surface fitting with splines, Monographs on
       Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------
    Examples are given :ref:`in the tutorial <tutorial-interpolate_2d_spline>`.

    """
    x, y, z = map(ravel, [x, y, z])  # ensure 1-d arrays.
    m = len(x)
    if not (m == len(y) == len(z)):
        raise TypeError('len(x)==len(y)==len(z) must hold.')
    if w is None:
        w = ones(m, float)
    else:
        w = atleast_1d(w)
    if not len(w) == m:
        raise TypeError('len(w)=%d is not equal to m=%d' % (len(w), m))
    if xb is None:
        xb = x.min()
    if xe is None:
        xe = x.max()
    if yb is None:
        yb = y.min()
    if ye is None:
        ye = y.max()
    if not (-1 <= task <= 1):
        raise TypeError('task must be -1, 0 or 1')
    if s is None:
        s = m - sqrt(2*m)
    if tx is None and task == -1:
        raise TypeError('Knots_x must be given for task=-1')
    if tx is not None:
        _surfit_cache['tx'] = atleast_1d(tx)
    nx = len(_surfit_cache['tx'])
    if ty is None and task == -1:
        raise TypeError('Knots_y must be given for task=-1')
    if ty is not None:
        _surfit_cache['ty'] = atleast_1d(ty)
    ny = len(_surfit_cache['ty'])
    if task == -1 and nx < 2*kx+2:
        raise TypeError('There must be at least 2*kx+2 knots_x for task=-1')
    if task == -1 and ny < 2*ky+2:
        raise TypeError('There must be at least 2*ky+2 knots_x for task=-1')
    if not ((1 <= kx <= 5) and (1 <= ky <= 5)):
        raise TypeError('Given degree of the spline (kx,ky=%d,%d) is not '
                        'supported. (1<=k<=5)' % (kx, ky))
    if m < (kx + 1)*(ky + 1):
        raise TypeError('m >= (kx+1)(ky+1) must hold')
    if nxest is None:
        nxest = int(kx + sqrt(m/2))
    if nyest is None:
        nyest = int(ky + sqrt(m/2))
    nxest, nyest = max(nxest, 2*kx + 3), max(nyest, 2*ky + 3)
    if task >= 0 and s == 0:
        nxest = int(kx + sqrt(3*m))
        nyest = int(ky + sqrt(3*m))
    if task == -1:
        _surfit_cache['tx'] = atleast_1d(tx)
        _surfit_cache['ty'] = atleast_1d(ty)
    tx, ty = _surfit_cache['tx'], _surfit_cache['ty']
    wrk = _surfit_cache['wrk']
    u = nxest - kx - 1
    v = nyest - ky - 1
    km = max(kx, ky) + 1
    ne = max(nxest, nyest)
    bx, by = kx*v + ky + 1, ky*u + kx + 1
    b1, b2 = bx, bx + v - ky
    if bx > by:
        b1, b2 = by, by + u - kx
    msg = "Too many data points to interpolate"
    lwrk1 = _int_overflow(u*v*(2 + b1 + b2) +
                          2*(u + v + km*(m + ne) + ne - kx - ky) + b2 + 1,
                          msg=msg)
    lwrk2 = _int_overflow(u*v*(b2 + 1) + b2, msg=msg)
    tx, ty, c, o = _fitpack._surfit(x, y, z, w, xb, xe, yb, ye, kx, ky,
                                    task, s, eps, tx, ty, nxest, nyest,
                                    wrk, lwrk1, lwrk2)
    _curfit_cache['tx'] = tx
    _curfit_cache['ty'] = ty
    _curfit_cache['wrk'] = o['wrk']
    ier, fp = o['ier'], o['fp']
    tck = [tx, ty, c, kx, ky]

    ierm = min(11, max(-3, ier))
    if ierm <= 0 and not quiet:
        _mess = (_iermess2[ierm][0] +
                 "\tkx,ky=%d,%d nx,ny=%d,%d m=%d fp=%f s=%f" %
                 (kx, ky, len(tx), len(ty), m, fp, s))
        warnings.warn(RuntimeWarning(_mess))
    if ierm > 0 and not full_output:
        if ier in [1, 2, 3, 4, 5]:
            _mess = ("\n\tkx,ky=%d,%d nx,ny=%d,%d m=%d fp=%f s=%f" %
                     (kx, ky, len(tx), len(ty), m, fp, s))
            warnings.warn(RuntimeWarning(_iermess2[ierm][0] + _mess))
        else:
            try:
                raise _iermess2[ierm][1](_iermess2[ierm][0])
            except KeyError as e:
                raise _iermess2['unknown'][1](_iermess2['unknown'][0]) from e
    if full_output:
        try:
            return tck, fp, ier, _iermess2[ierm][0]
        except KeyError:
            return tck, fp, ier, _iermess2['unknown'][0]
    else:
        return tck


def bisplev(x, y, tck, dx=0, dy=0):
    """
    Evaluate a bivariate B-spline and its derivatives.

    Return a rank-2 array of spline function values (or spline derivative
    values) at points given by the cross-product of the rank-1 arrays `x` and
    `y`.  In special cases, return an array or just a float if either `x` or
    `y` or both are floats.  Based on BISPEV from FITPACK.

    Parameters
    ----------
    x, y : ndarray
        Rank-1 arrays specifying the domain over which to evaluate the
        spline or its derivative.
    tck : tuple
        A sequence of length 5 returned by `bisplrep` containing the knot
        locations, the coefficients, and the degree of the spline:
        [tx, ty, c, kx, ky].
    dx, dy : int, optional
        The orders of the partial derivatives in `x` and `y` respectively.

    Returns
    -------
    vals : ndarray
        The B-spline or its derivative evaluated over the set formed by
        the cross-product of `x` and `y`.

    See Also
    --------
    splprep, splrep, splint, sproot, splev
    UnivariateSpline, BivariateSpline

    Notes
    -----
        See `bisplrep` to generate the `tck` representation.

    References
    ----------
    .. [1] Dierckx P. : An algorithm for surface fitting
       with spline functions
       Ima J. Numer. Anal. 1 (1981) 267-283.
    .. [2] Dierckx P. : An algorithm for surface fitting
       with spline functions
       report tw50, Dept. Computer Science,K.U.Leuven, 1980.
    .. [3] Dierckx P. : Curve and surface fitting with splines,
       Monographs on Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------
    Examples are given :ref:`in the tutorial <tutorial-interpolate_2d_spline>`.

    """
    tx, ty, c, kx, ky = tck
    if not (0 <= dx < kx):
        raise ValueError("0 <= dx = %d < kx = %d must hold" % (dx, kx))
    if not (0 <= dy < ky):
        raise ValueError("0 <= dy = %d < ky = %d must hold" % (dy, ky))
    x, y = map(atleast_1d, [x, y])
    if (len(x.shape) != 1) or (len(y.shape) != 1):
        raise ValueError("First two entries should be rank-1 arrays.")
    z, ier = _fitpack._bispev(tx, ty, c, kx, ky, x, y, dx, dy)
    if ier == 10:
        raise ValueError("Invalid input data")
    if ier:
        raise TypeError("An error occurred")
    z.shape = len(x), len(y)
    if len(z) > 1:
        return z
    if len(z[0]) > 1:
        return z[0]
    return z[0][0]


def dblint(xa, xb, ya, yb, tck):
    """Evaluate the integral of a spline over area [xa,xb] x [ya,yb].

    Parameters
    ----------
    xa, xb : float
        The end-points of the x integration interval.
    ya, yb : float
        The end-points of the y integration interval.
    tck : list [tx, ty, c, kx, ky]
        A sequence of length 5 returned by bisplrep containing the knot
        locations tx, ty, the coefficients c, and the degrees kx, ky
        of the spline.

    Returns
    -------
    integ : float
        The value of the resulting integral.
    """
    tx, ty, c, kx, ky = tck
    return dfitpack.dblint(tx, ty, c, kx, ky, xa, xb, ya, yb)


def insert(x, tck, m=1, per=0):
    """
    Insert knots into a B-spline.

    Given the knots and coefficients of a B-spline representation, create a
    new B-spline with a knot inserted `m` times at point `x`.
    This is a wrapper around the FORTRAN routine insert of FITPACK.

    Parameters
    ----------
    x (u) : array_like
        A 1-D point at which to insert a new knot(s).  If `tck` was returned
        from ``splprep``, then the parameter values, u should be given.
    tck : tuple
        A tuple (t,c,k) returned by ``splrep`` or ``splprep`` containing
        the vector of knots, the B-spline coefficients,
        and the degree of the spline.
    m : int, optional
        The number of times to insert the given knot (its multiplicity).
        Default is 1.
    per : int, optional
        If non-zero, the input spline is considered periodic.

    Returns
    -------
    tck : tuple
        A tuple (t,c,k) containing the vector of knots, the B-spline
        coefficients, and the degree of the new spline.
        ``t(k+1) <= x <= t(n-k)``, where k is the degree of the spline.
        In case of a periodic spline (``per != 0``) there must be
        either at least k interior knots t(j) satisfying ``t(k+1)<t(j)<=x``
        or at least k interior knots t(j) satisfying ``x<=t(j)<t(n-k)``.

    Notes
    -----
    Based on algorithms from [1]_ and [2]_.

    References
    ----------
    .. [1] W. Boehm, "Inserting new knots into b-spline curves.",
        Computer Aided Design, 12, p.199-201, 1980.
    .. [2] P. Dierckx, "Curve and surface fitting with splines, Monographs on
        Numerical Analysis", Oxford University Press, 1993.

    """
    t, c, k = tck
    try:
        c[0][0]
        parametric = True
    except Exception:
        parametric = False
    if parametric:
        cc = []
        for c_vals in c:
            tt, cc_val, kk = insert(x, [t, c_vals, k], m)
            cc.append(cc_val)
        return (tt, cc, kk)
    else:
        tt, cc, ier = _fitpack._insert(per, t, c, k, x, m)
        if ier == 10:
            raise ValueError("Invalid input data")
        if ier:
            raise TypeError("An error occurred")
        return (tt, cc, k)


def splder(tck, n=1):
    """
    Compute the spline representation of the derivative of a given spline

    Parameters
    ----------
    tck : tuple of (t, c, k)
        Spline whose derivative to compute
    n : int, optional
        Order of derivative to evaluate. Default: 1

    Returns
    -------
    tck_der : tuple of (t2, c2, k2)
        Spline of order k2=k-n representing the derivative
        of the input spline.

    Notes
    -----

    .. versionadded:: 0.13.0

    See Also
    --------
    splantider, splev, spalde

    Examples
    --------
    This can be used for finding maxima of a curve:

    >>> from scipy.interpolate import splrep, splder, sproot
    >>> x = np.linspace(0, 10, 70)
    >>> y = np.sin(x)
    >>> spl = splrep(x, y, k=4)

    Now, differentiate the spline and find the zeros of the
    derivative. (NB: `sproot` only works for order 3 splines, so we
    fit an order 4 spline):

    >>> dspl = splder(spl)
    >>> sproot(dspl) / np.pi
    array([ 0.50000001,  1.5       ,  2.49999998])

    This agrees well with roots :math:`\\pi/2 + n\\pi` of
    :math:`\\cos(x) = \\sin'(x)`.

    """
    if n < 0:
        return splantider(tck, -n)

    t, c, k = tck

    if n > k:
        raise ValueError(("Order of derivative (n = %r) must be <= "
                          "order of spline (k = %r)") % (n, tck[2]))

    # Extra axes for the trailing dims of the `c` array:
    sh = (slice(None),) + ((None,)*len(c.shape[1:]))

    with np.errstate(invalid='raise', divide='raise'):
        try:
            for j in range(n):
                # See e.g. Schumaker, Spline Functions: Basic Theory, Chapter 5

                # Compute the denominator in the differentiation formula.
                # (and append traling dims, if necessary)
                dt = t[k+1:-1] - t[1:-k-1]
                dt = dt[sh]
                # Compute the new coefficients
                c = (c[1:-1-k] - c[:-2-k]) * k / dt
                # Pad coefficient array to same size as knots (FITPACK
                # convention)
                c = np.r_[c, np.zeros((k,) + c.shape[1:])]
                # Adjust knots
                t = t[1:-1]
                k -= 1
        except FloatingPointError as e:
            raise ValueError(("The spline has internal repeated knots "
                              "and is not differentiable %d times") % n) from e

    return t, c, k


def splantider(tck, n=1):
    """
    Compute the spline for the antiderivative (integral) of a given spline.

    Parameters
    ----------
    tck : tuple of (t, c, k)
        Spline whose antiderivative to compute
    n : int, optional
        Order of antiderivative to evaluate. Default: 1

    Returns
    -------
    tck_ader : tuple of (t2, c2, k2)
        Spline of order k2=k+n representing the antiderivative of the input
        spline.

    See Also
    --------
    splder, splev, spalde

    Notes
    -----
    The `splder` function is the inverse operation of this function.
    Namely, ``splder(splantider(tck))`` is identical to `tck`, modulo
    rounding error.

    .. versionadded:: 0.13.0

    Examples
    --------
    >>> from scipy.interpolate import splrep, splder, splantider, splev
    >>> x = np.linspace(0, np.pi/2, 70)
    >>> y = 1 / np.sqrt(1 - 0.8*np.sin(x)**2)
    >>> spl = splrep(x, y)

    The derivative is the inverse operation of the antiderivative,
    although some floating point error accumulates:

    >>> splev(1.7, spl), splev(1.7, splder(splantider(spl)))
    (array(2.1565429877197317), array(2.1565429877201865))

    Antiderivative can be used to evaluate definite integrals:

    >>> ispl = splantider(spl)
    >>> splev(np.pi/2, ispl) - splev(0, ispl)
    2.2572053588768486

    This is indeed an approximation to the complete elliptic integral
    :math:`K(m) = \\int_0^{\\pi/2} [1 - m\\sin^2 x]^{-1/2} dx`:

    >>> from scipy.special import ellipk
    >>> ellipk(0.8)
    2.2572053268208538

    """
    if n < 0:
        return splder(tck, -n)

    t, c, k = tck

    # Extra axes for the trailing dims of the `c` array:
    sh = (slice(None),) + (None,)*len(c.shape[1:])

    for j in range(n):
        # This is the inverse set of operations to splder.

        # Compute the multiplier in the antiderivative formula.
        dt = t[k+1:] - t[:-k-1]
        dt = dt[sh]
        # Compute the new coefficients
        c = np.cumsum(c[:-k-1] * dt, axis=0) / (k + 1)
        c = np.r_[np.zeros((1,) + c.shape[1:]),
                  c,
                  [c[-1]] * (k+2)]
        # New knots
        t = np.r_[t[0], t, t[-1]]
        k += 1

    return t, c, k
