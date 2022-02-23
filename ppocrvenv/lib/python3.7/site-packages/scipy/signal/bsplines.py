from numpy import (logical_and, asarray, pi, zeros_like,
                   piecewise, array, arctan2, tan, zeros, arange, floor)
from numpy.core.umath import (sqrt, exp, greater, less, cos, add, sin,
                              less_equal, greater_equal)

# From splinemodule.c
from .spline import cspline2d, sepfir2d

from scipy.special import comb
from scipy._lib._util import float_factorial

__all__ = ['spline_filter', 'bspline', 'gauss_spline', 'cubic', 'quadratic',
           'cspline1d', 'qspline1d', 'cspline1d_eval', 'qspline1d_eval']


def spline_filter(Iin, lmbda=5.0):
    """Smoothing spline (cubic) filtering of a rank-2 array.

    Filter an input data set, `Iin`, using a (cubic) smoothing spline of
    fall-off `lmbda`.

    Parameters
    ----------
    Iin : array_like
        input data set
    lmbda : float, optional
        spline smooghing fall-off value, default is `5.0`.

    Returns
    -------
    res : ndarray
        filterd input data

    Examples
    --------
    We can filter an multi dimentional signal (ex: 2D image) using cubic
    B-spline filter:

    >>> from scipy.signal import spline_filter
    >>> import matplotlib.pyplot as plt
    >>> orig_img = np.eye(20)  # create an image
    >>> orig_img[10, :] = 1.0
    >>> sp_filter = spline_filter(orig_img, lmbda=0.1)
    >>> f, ax = plt.subplots(1, 2, sharex=True)
    >>> for ind, data in enumerate([[orig_img, "original image"],
    ...                             [sp_filter, "spline filter"]]):
    ...     ax[ind].imshow(data[0], cmap='gray_r')
    ...     ax[ind].set_title(data[1])
    >>> plt.tight_layout()
    >>> plt.show()

    """
    intype = Iin.dtype.char
    hcol = array([1.0, 4.0, 1.0], 'f') / 6.0
    if intype in ['F', 'D']:
        Iin = Iin.astype('F')
        ckr = cspline2d(Iin.real, lmbda)
        cki = cspline2d(Iin.imag, lmbda)
        outr = sepfir2d(ckr, hcol, hcol)
        outi = sepfir2d(cki, hcol, hcol)
        out = (outr + 1j * outi).astype(intype)
    elif intype in ['f', 'd']:
        ckr = cspline2d(Iin, lmbda)
        out = sepfir2d(ckr, hcol, hcol)
        out = out.astype(intype)
    else:
        raise TypeError("Invalid data type for Iin")
    return out


_splinefunc_cache = {}


def _bspline_piecefunctions(order):
    """Returns the function defined over the left-side pieces for a bspline of
    a given order.

    The 0th piece is the first one less than 0. The last piece is a function
    identical to 0 (returned as the constant 0). (There are order//2 + 2 total
    pieces).

    Also returns the condition functions that when evaluated return boolean
    arrays for use with `numpy.piecewise`.
    """
    try:
        return _splinefunc_cache[order]
    except KeyError:
        pass

    def condfuncgen(num, val1, val2):
        if num == 0:
            return lambda x: logical_and(less_equal(x, val1),
                                         greater_equal(x, val2))
        elif num == 2:
            return lambda x: less_equal(x, val2)
        else:
            return lambda x: logical_and(less(x, val1),
                                         greater_equal(x, val2))

    last = order // 2 + 2
    if order % 2:
        startbound = -1.0
    else:
        startbound = -0.5
    condfuncs = [condfuncgen(0, 0, startbound)]
    bound = startbound
    for num in range(1, last - 1):
        condfuncs.append(condfuncgen(1, bound, bound - 1))
        bound = bound - 1
    condfuncs.append(condfuncgen(2, 0, -(order + 1) / 2.0))

    # final value of bound is used in piecefuncgen below

    # the functions to evaluate are taken from the left-hand side
    #  in the general expression derived from the central difference
    #  operator (because they involve fewer terms).

    fval = float_factorial(order)

    def piecefuncgen(num):
        Mk = order // 2 - num
        if (Mk < 0):
            return 0  # final function is 0
        coeffs = [(1 - 2 * (k % 2)) * float(comb(order + 1, k, exact=1)) / fval
                  for k in range(Mk + 1)]
        shifts = [-bound - k for k in range(Mk + 1)]

        def thefunc(x):
            res = 0.0
            for k in range(Mk + 1):
                res += coeffs[k] * (x + shifts[k]) ** order
            return res
        return thefunc

    funclist = [piecefuncgen(k) for k in range(last)]

    _splinefunc_cache[order] = (funclist, condfuncs)

    return funclist, condfuncs


def bspline(x, n):
    """B-spline basis function of order n.

    Parameters
    ----------
    x : array_like
        a knot vector
    n : int
        The order of the spline. Must be non-negative, i.e., n >= 0

    Returns
    -------
    res : ndarray
        B-spline basis function values

    See Also
    --------
    cubic : A cubic B-spline.
    quadratic : A quadratic B-spline.

    Notes
    -----
    Uses numpy.piecewise and automatic function-generator.

    Examples
    --------
    We can calculate B-Spline basis function of several orders:

    >>> from scipy.signal import bspline, cubic, quadratic
    >>> bspline(0.0, 1)
    1

    >>> knots = [-1.0, 0.0, -1.0]
    >>> bspline(knots, 2)
    array([0.125, 0.75, 0.125])

    >>> np.array_equal(bspline(knots, 2), quadratic(knots))
    True

    >>> np.array_equal(bspline(knots, 3), cubic(knots))
    True

    """
    ax = -abs(asarray(x))
    # number of pieces on the left-side is (n+1)/2
    funclist, condfuncs = _bspline_piecefunctions(n)
    condlist = [func(ax) for func in condfuncs]
    return piecewise(ax, condlist, funclist)


def gauss_spline(x, n):
    r"""Gaussian approximation to B-spline basis function of order n.

    Parameters
    ----------
    x : array_like
        a knot vector
    n : int
        The order of the spline. Must be non-negative, i.e., n >= 0

    Returns
    -------
    res : ndarray
        B-spline basis function values approximated by a zero-mean Gaussian
        function.

    Notes
    -----
    The B-spline basis function can be approximated well by a zero-mean
    Gaussian function with standard-deviation equal to :math:`\sigma=(n+1)/12`
    for large `n` :

    .. math::  \frac{1}{\sqrt {2\pi\sigma^2}}exp(-\frac{x^2}{2\sigma})

    References
    ----------
    .. [1] Bouma H., Vilanova A., Bescos J.O., ter Haar Romeny B.M., Gerritsen
       F.A. (2007) Fast and Accurate Gaussian Derivatives Based on B-Splines. In:
       Sgallari F., Murli A., Paragios N. (eds) Scale Space and Variational
       Methods in Computer Vision. SSVM 2007. Lecture Notes in Computer
       Science, vol 4485. Springer, Berlin, Heidelberg
    .. [2] http://folk.uio.no/inf3330/scripting/doc/python/SciPy/tutorial/old/node24.html

    Examples
    --------
    We can calculate B-Spline basis functions approximated by a gaussian
    distribution:

    >>> from scipy.signal import gauss_spline, bspline
    >>> knots = np.array([-1.0, 0.0, -1.0])
    >>> gauss_spline(knots, 3)
    array([0.15418033, 0.6909883, 0.15418033])  # may vary

    >>> bspline(knots, 3)
    array([0.16666667, 0.66666667, 0.16666667])  # may vary

    """
    x = asarray(x)
    signsq = (n + 1) / 12.0
    return 1 / sqrt(2 * pi * signsq) * exp(-x ** 2 / 2 / signsq)


def cubic(x):
    """A cubic B-spline.

    This is a special case of `bspline`, and equivalent to ``bspline(x, 3)``.

    Parameters
    ----------
    x : array_like
        a knot vector

    Returns
    -------
    res : ndarray
        Cubic B-spline basis function values

    See Also
    --------
    bspline : B-spline basis function of order n
    quadratic : A quadratic B-spline.

    Examples
    --------
    We can calculate B-Spline basis function of several orders:

    >>> from scipy.signal import bspline, cubic, quadratic
    >>> bspline(0.0, 1)
    1

    >>> knots = [-1.0, 0.0, -1.0]
    >>> bspline(knots, 2)
    array([0.125, 0.75, 0.125])

    >>> np.array_equal(bspline(knots, 2), quadratic(knots))
    True

    >>> np.array_equal(bspline(knots, 3), cubic(knots))
    True

    """
    ax = abs(asarray(x))
    res = zeros_like(ax)
    cond1 = less(ax, 1)
    if cond1.any():
        ax1 = ax[cond1]
        res[cond1] = 2.0 / 3 - 1.0 / 2 * ax1 ** 2 * (2 - ax1)
    cond2 = ~cond1 & less(ax, 2)
    if cond2.any():
        ax2 = ax[cond2]
        res[cond2] = 1.0 / 6 * (2 - ax2) ** 3
    return res


def quadratic(x):
    """A quadratic B-spline.

    This is a special case of `bspline`, and equivalent to ``bspline(x, 2)``.

    Parameters
    ----------
    x : array_like
        a knot vector

    Returns
    -------
    res : ndarray
        Quadratic B-spline basis function values

    See Also
    --------
    bspline : B-spline basis function of order n
    cubic : A cubic B-spline.

    Examples
    --------
    We can calculate B-Spline basis function of several orders:

    >>> from scipy.signal import bspline, cubic, quadratic
    >>> bspline(0.0, 1)
    1

    >>> knots = [-1.0, 0.0, -1.0]
    >>> bspline(knots, 2)
    array([0.125, 0.75, 0.125])

    >>> np.array_equal(bspline(knots, 2), quadratic(knots))
    True

    >>> np.array_equal(bspline(knots, 3), cubic(knots))
    True

    """
    ax = abs(asarray(x))
    res = zeros_like(ax)
    cond1 = less(ax, 0.5)
    if cond1.any():
        ax1 = ax[cond1]
        res[cond1] = 0.75 - ax1 ** 2
    cond2 = ~cond1 & less(ax, 1.5)
    if cond2.any():
        ax2 = ax[cond2]
        res[cond2] = (ax2 - 1.5) ** 2 / 2.0
    return res


def _coeff_smooth(lam):
    xi = 1 - 96 * lam + 24 * lam * sqrt(3 + 144 * lam)
    omeg = arctan2(sqrt(144 * lam - 1), sqrt(xi))
    rho = (24 * lam - 1 - sqrt(xi)) / (24 * lam)
    rho = rho * sqrt((48 * lam + 24 * lam * sqrt(3 + 144 * lam)) / xi)
    return rho, omeg


def _hc(k, cs, rho, omega):
    return (cs / sin(omega) * (rho ** k) * sin(omega * (k + 1)) *
            greater(k, -1))


def _hs(k, cs, rho, omega):
    c0 = (cs * cs * (1 + rho * rho) / (1 - rho * rho) /
          (1 - 2 * rho * rho * cos(2 * omega) + rho ** 4))
    gamma = (1 - rho * rho) / (1 + rho * rho) / tan(omega)
    ak = abs(k)
    return c0 * rho ** ak * (cos(omega * ak) + gamma * sin(omega * ak))


def _cubic_smooth_coeff(signal, lamb):
    rho, omega = _coeff_smooth(lamb)
    cs = 1 - 2 * rho * cos(omega) + rho * rho
    K = len(signal)
    yp = zeros((K,), signal.dtype.char)
    k = arange(K)
    yp[0] = (_hc(0, cs, rho, omega) * signal[0] +
             add.reduce(_hc(k + 1, cs, rho, omega) * signal))

    yp[1] = (_hc(0, cs, rho, omega) * signal[0] +
             _hc(1, cs, rho, omega) * signal[1] +
             add.reduce(_hc(k + 2, cs, rho, omega) * signal))

    for n in range(2, K):
        yp[n] = (cs * signal[n] + 2 * rho * cos(omega) * yp[n - 1] -
                 rho * rho * yp[n - 2])

    y = zeros((K,), signal.dtype.char)

    y[K - 1] = add.reduce((_hs(k, cs, rho, omega) +
                           _hs(k + 1, cs, rho, omega)) * signal[::-1])
    y[K - 2] = add.reduce((_hs(k - 1, cs, rho, omega) +
                           _hs(k + 2, cs, rho, omega)) * signal[::-1])

    for n in range(K - 3, -1, -1):
        y[n] = (cs * yp[n] + 2 * rho * cos(omega) * y[n + 1] -
                rho * rho * y[n + 2])

    return y


def _cubic_coeff(signal):
    zi = -2 + sqrt(3)
    K = len(signal)
    yplus = zeros((K,), signal.dtype.char)
    powers = zi ** arange(K)
    yplus[0] = signal[0] + zi * add.reduce(powers * signal)
    for k in range(1, K):
        yplus[k] = signal[k] + zi * yplus[k - 1]
    output = zeros((K,), signal.dtype)
    output[K - 1] = zi / (zi - 1) * yplus[K - 1]
    for k in range(K - 2, -1, -1):
        output[k] = zi * (output[k + 1] - yplus[k])
    return output * 6.0


def _quadratic_coeff(signal):
    zi = -3 + 2 * sqrt(2.0)
    K = len(signal)
    yplus = zeros((K,), signal.dtype.char)
    powers = zi ** arange(K)
    yplus[0] = signal[0] + zi * add.reduce(powers * signal)
    for k in range(1, K):
        yplus[k] = signal[k] + zi * yplus[k - 1]
    output = zeros((K,), signal.dtype.char)
    output[K - 1] = zi / (zi - 1) * yplus[K - 1]
    for k in range(K - 2, -1, -1):
        output[k] = zi * (output[k + 1] - yplus[k])
    return output * 8.0


def cspline1d(signal, lamb=0.0):
    """
    Compute cubic spline coefficients for rank-1 array.

    Find the cubic spline coefficients for a 1-D signal assuming
    mirror-symmetric boundary conditions. To obtain the signal back from the
    spline representation mirror-symmetric-convolve these coefficients with a
    length 3 FIR window [1.0, 4.0, 1.0]/ 6.0 .

    Parameters
    ----------
    signal : ndarray
        A rank-1 array representing samples of a signal.
    lamb : float, optional
        Smoothing coefficient, default is 0.0.

    Returns
    -------
    c : ndarray
        Cubic spline coefficients.

    See Also
    --------
    cspline1d_eval : Evaluate a cubic spline at the new set of points.

    Examples
    --------
    We can filter a signal to reduce and smooth out high-frequency noise with
    a cubic spline:

    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import cspline1d, cspline1d_eval
    >>> rng = np.random.default_rng()
    >>> sig = np.repeat([0., 1., 0.], 100)
    >>> sig += rng.standard_normal(len(sig))*0.05  # add noise
    >>> time = np.linspace(0, len(sig))
    >>> filtered = cspline1d_eval(cspline1d(sig), time)
    >>> plt.plot(sig, label="signal")
    >>> plt.plot(time, filtered, label="filtered")
    >>> plt.legend()
    >>> plt.show()

    """
    if lamb != 0.0:
        return _cubic_smooth_coeff(signal, lamb)
    else:
        return _cubic_coeff(signal)


def qspline1d(signal, lamb=0.0):
    """Compute quadratic spline coefficients for rank-1 array.

    Parameters
    ----------
    signal : ndarray
        A rank-1 array representing samples of a signal.
    lamb : float, optional
        Smoothing coefficient (must be zero for now).

    Returns
    -------
    c : ndarray
        Quadratic spline coefficients.

    See Also
    --------
    qspline1d_eval : Evaluate a quadratic spline at the new set of points.

    Notes
    -----
    Find the quadratic spline coefficients for a 1-D signal assuming
    mirror-symmetric boundary conditions. To obtain the signal back from the
    spline representation mirror-symmetric-convolve these coefficients with a
    length 3 FIR window [1.0, 6.0, 1.0]/ 8.0 .

    Examples
    --------
    We can filter a signal to reduce and smooth out high-frequency noise with
    a quadratic spline:

    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import qspline1d, qspline1d_eval
    >>> rng = np.random.default_rng()
    >>> sig = np.repeat([0., 1., 0.], 100)
    >>> sig += rng.standard_normal(len(sig))*0.05  # add noise
    >>> time = np.linspace(0, len(sig))
    >>> filtered = qspline1d_eval(qspline1d(sig), time)
    >>> plt.plot(sig, label="signal")
    >>> plt.plot(time, filtered, label="filtered")
    >>> plt.legend()
    >>> plt.show()

    """
    if lamb != 0.0:
        raise ValueError("Smoothing quadratic splines not supported yet.")
    else:
        return _quadratic_coeff(signal)


def cspline1d_eval(cj, newx, dx=1.0, x0=0):
    """Evaluate a cubic spline at the new set of points.

    `dx` is the old sample-spacing while `x0` was the old origin. In
    other-words the old-sample points (knot-points) for which the `cj`
    represent spline coefficients were at equally-spaced points of:

      oldx = x0 + j*dx  j=0...N-1, with N=len(cj)

    Edges are handled using mirror-symmetric boundary conditions.

    Parameters
    ----------
    cj : ndarray
        cublic spline coefficients
    newx : ndarray
        New set of points.
    dx : float, optional
        Old sample-spacing, the default value is 1.0.
    x0 : int, optional
        Old origin, the default value is 0.

    Returns
    -------
    res : ndarray
        Evaluated a cubic spline points.

    See Also
    --------
    cspline1d : Compute cubic spline coefficients for rank-1 array.

    Examples
    --------
    We can filter a signal to reduce and smooth out high-frequency noise with
    a cubic spline:

    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import cspline1d, cspline1d_eval
    >>> rng = np.random.default_rng()
    >>> sig = np.repeat([0., 1., 0.], 100)
    >>> sig += rng.standard_normal(len(sig))*0.05  # add noise
    >>> time = np.linspace(0, len(sig))
    >>> filtered = cspline1d_eval(cspline1d(sig), time)
    >>> plt.plot(sig, label="signal")
    >>> plt.plot(time, filtered, label="filtered")
    >>> plt.legend()
    >>> plt.show()

    """
    newx = (asarray(newx) - x0) / float(dx)
    res = zeros_like(newx, dtype=cj.dtype)
    if res.size == 0:
        return res
    N = len(cj)
    cond1 = newx < 0
    cond2 = newx > (N - 1)
    cond3 = ~(cond1 | cond2)
    # handle general mirror-symmetry
    res[cond1] = cspline1d_eval(cj, -newx[cond1])
    res[cond2] = cspline1d_eval(cj, 2 * (N - 1) - newx[cond2])
    newx = newx[cond3]
    if newx.size == 0:
        return res
    result = zeros_like(newx, dtype=cj.dtype)
    jlower = floor(newx - 2).astype(int) + 1
    for i in range(4):
        thisj = jlower + i
        indj = thisj.clip(0, N - 1)  # handle edge cases
        result += cj[indj] * cubic(newx - thisj)
    res[cond3] = result
    return res


def qspline1d_eval(cj, newx, dx=1.0, x0=0):
    """Evaluate a quadratic spline at the new set of points.

    Parameters
    ----------
    cj : ndarray
        Quadratic spline coefficients
    newx : ndarray
        New set of points.
    dx : float, optional
        Old sample-spacing, the default value is 1.0.
    x0 : int, optional
        Old origin, the default value is 0.

    Returns
    -------
    res : ndarray
        Evaluated a quadratic spline points.

    See Also
    --------
    qspline1d : Compute quadratic spline coefficients for rank-1 array.

    Notes
    -----
    `dx` is the old sample-spacing while `x0` was the old origin. In
    other-words the old-sample points (knot-points) for which the `cj`
    represent spline coefficients were at equally-spaced points of::

      oldx = x0 + j*dx  j=0...N-1, with N=len(cj)

    Edges are handled using mirror-symmetric boundary conditions.

    Examples
    --------
    We can filter a signal to reduce and smooth out high-frequency noise with
    a quadratic spline:

    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import qspline1d, qspline1d_eval
    >>> rng = np.random.default_rng()
    >>> sig = np.repeat([0., 1., 0.], 100)
    >>> sig += rng.standard_normal(len(sig))*0.05  # add noise
    >>> time = np.linspace(0, len(sig))
    >>> filtered = qspline1d_eval(qspline1d(sig), time)
    >>> plt.plot(sig, label="signal")
    >>> plt.plot(time, filtered, label="filtered")
    >>> plt.legend()
    >>> plt.show()

    """
    newx = (asarray(newx) - x0) / dx
    res = zeros_like(newx)
    if res.size == 0:
        return res
    N = len(cj)
    cond1 = newx < 0
    cond2 = newx > (N - 1)
    cond3 = ~(cond1 | cond2)
    # handle general mirror-symmetry
    res[cond1] = qspline1d_eval(cj, -newx[cond1])
    res[cond2] = qspline1d_eval(cj, 2 * (N - 1) - newx[cond2])
    newx = newx[cond3]
    if newx.size == 0:
        return res
    result = zeros_like(newx)
    jlower = floor(newx - 1.5).astype(int) + 1
    for i in range(3):
        thisj = jlower + i
        indj = thisj.clip(0, N - 1)  # handle edge cases
        result += cj[indj] * quadratic(newx - thisj)
    res[cond3] = result
    return res
