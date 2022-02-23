#
# Author:  Travis Oliphant, 2002
#

import operator
import numpy as np
import math
from numpy import (pi, asarray, floor, isscalar, iscomplex, real,
                   imag, sqrt, where, mgrid, sin, place, issubdtype,
                   extract, inexact, nan, zeros, sinc)
from . import _ufuncs as ufuncs
from ._ufuncs import (mathieu_a, mathieu_b, iv, jv, gamma,
                      psi, hankel1, hankel2, yv, kv, ndtri,
                      poch, binom, hyp0f1)
from . import specfun
from . import orthogonal
from ._comb import _comb_int


__all__ = [
    'ai_zeros',
    'assoc_laguerre',
    'bei_zeros',
    'beip_zeros',
    'ber_zeros',
    'bernoulli',
    'berp_zeros',
    'bi_zeros',
    'clpmn',
    'comb',
    'digamma',
    'diric',
    'erf_zeros',
    'euler',
    'factorial',
    'factorial2',
    'factorialk',
    'fresnel_zeros',
    'fresnelc_zeros',
    'fresnels_zeros',
    'gamma',
    'h1vp',
    'h2vp',
    'hankel1',
    'hankel2',
    'hyp0f1',
    'iv',
    'ivp',
    'jn_zeros',
    'jnjnp_zeros',
    'jnp_zeros',
    'jnyn_zeros',
    'jv',
    'jvp',
    'kei_zeros',
    'keip_zeros',
    'kelvin_zeros',
    'ker_zeros',
    'kerp_zeros',
    'kv',
    'kvp',
    'lmbda',
    'lpmn',
    'lpn',
    'lqmn',
    'lqn',
    'mathieu_a',
    'mathieu_b',
    'mathieu_even_coef',
    'mathieu_odd_coef',
    'ndtri',
    'obl_cv_seq',
    'pbdn_seq',
    'pbdv_seq',
    'pbvv_seq',
    'perm',
    'polygamma',
    'pro_cv_seq',
    'psi',
    'riccati_jn',
    'riccati_yn',
    'sinc',
    'y0_zeros',
    'y1_zeros',
    'y1p_zeros',
    'yn_zeros',
    'ynp_zeros',
    'yv',
    'yvp',
    'zeta'
]


def _nonneg_int_or_fail(n, var_name, strict=True):
    try:
        if strict:
            # Raises an exception if float
            n = operator.index(n)
        elif n == floor(n):
            n = int(n)
        else:
            raise ValueError()
        if n < 0:
            raise ValueError()
    except (ValueError, TypeError) as err:
        raise err.__class__("{} must be a non-negative integer".format(var_name)) from err
    return n


def diric(x, n):
    """Periodic sinc function, also called the Dirichlet function.

    The Dirichlet function is defined as::

        diric(x, n) = sin(x * n/2) / (n * sin(x / 2)),

    where `n` is a positive integer.

    Parameters
    ----------
    x : array_like
        Input data
    n : int
        Integer defining the periodicity.

    Returns
    -------
    diric : ndarray

    Examples
    --------
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(-8*np.pi, 8*np.pi, num=201)
    >>> plt.figure(figsize=(8, 8));
    >>> for idx, n in enumerate([2, 3, 4, 9]):
    ...     plt.subplot(2, 2, idx+1)
    ...     plt.plot(x, special.diric(x, n))
    ...     plt.title('diric, n={}'.format(n))
    >>> plt.show()

    The following example demonstrates that `diric` gives the magnitudes
    (modulo the sign and scaling) of the Fourier coefficients of a
    rectangular pulse.

    Suppress output of values that are effectively 0:

    >>> np.set_printoptions(suppress=True)

    Create a signal `x` of length `m` with `k` ones:

    >>> m = 8
    >>> k = 3
    >>> x = np.zeros(m)
    >>> x[:k] = 1

    Use the FFT to compute the Fourier transform of `x`, and
    inspect the magnitudes of the coefficients:

    >>> np.abs(np.fft.fft(x))
    array([ 3.        ,  2.41421356,  1.        ,  0.41421356,  1.        ,
            0.41421356,  1.        ,  2.41421356])

    Now find the same values (up to sign) using `diric`. We multiply
    by `k` to account for the different scaling conventions of
    `numpy.fft.fft` and `diric`:

    >>> theta = np.linspace(0, 2*np.pi, m, endpoint=False)
    >>> k * special.diric(theta, k)
    array([ 3.        ,  2.41421356,  1.        , -0.41421356, -1.        ,
           -0.41421356,  1.        ,  2.41421356])
    """
    x, n = asarray(x), asarray(n)
    n = asarray(n + (x-x))
    x = asarray(x + (n-n))
    if issubdtype(x.dtype, inexact):
        ytype = x.dtype
    else:
        ytype = float
    y = zeros(x.shape, ytype)

    # empirical minval for 32, 64 or 128 bit float computations
    # where sin(x/2) < minval, result is fixed at +1 or -1
    if np.finfo(ytype).eps < 1e-18:
        minval = 1e-11
    elif np.finfo(ytype).eps < 1e-15:
        minval = 1e-7
    else:
        minval = 1e-3

    mask1 = (n <= 0) | (n != floor(n))
    place(y, mask1, nan)

    x = x / 2
    denom = sin(x)
    mask2 = (1-mask1) & (abs(denom) < minval)
    xsub = extract(mask2, x)
    nsub = extract(mask2, n)
    zsub = xsub / pi
    place(y, mask2, pow(-1, np.round(zsub)*(nsub-1)))

    mask = (1-mask1) & (1-mask2)
    xsub = extract(mask, x)
    nsub = extract(mask, n)
    dsub = extract(mask, denom)
    place(y, mask, sin(nsub*xsub)/(nsub*dsub))
    return y


def jnjnp_zeros(nt):
    """Compute zeros of integer-order Bessel functions Jn and Jn'.

    Results are arranged in order of the magnitudes of the zeros.

    Parameters
    ----------
    nt : int
        Number (<=1200) of zeros to compute

    Returns
    -------
    zo[l-1] : ndarray
        Value of the lth zero of Jn(x) and Jn'(x). Of length `nt`.
    n[l-1] : ndarray
        Order of the Jn(x) or Jn'(x) associated with lth zero. Of length `nt`.
    m[l-1] : ndarray
        Serial number of the zeros of Jn(x) or Jn'(x) associated
        with lth zero. Of length `nt`.
    t[l-1] : ndarray
        0 if lth zero in zo is zero of Jn(x), 1 if it is a zero of Jn'(x). Of
        length `nt`.

    See Also
    --------
    jn_zeros, jnp_zeros : to get separated arrays of zeros.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt > 1200):
        raise ValueError("Number must be integer <= 1200.")
    nt = int(nt)
    n, m, t, zo = specfun.jdzo(nt)
    return zo[1:nt+1], n[:nt], m[:nt], t[:nt]


def jnyn_zeros(n, nt):
    """Compute nt zeros of Bessel functions Jn(x), Jn'(x), Yn(x), and Yn'(x).

    Returns 4 arrays of length `nt`, corresponding to the first `nt`
    zeros of Jn(x), Jn'(x), Yn(x), and Yn'(x), respectively. The zeros
    are returned in ascending order.

    Parameters
    ----------
    n : int
        Order of the Bessel functions
    nt : int
        Number (<=1200) of zeros to compute

    Returns
    -------
    Jn : ndarray
        First `nt` zeros of Jn
    Jnp : ndarray
        First `nt` zeros of Jn'
    Yn : ndarray
        First `nt` zeros of Yn
    Ynp : ndarray
        First `nt` zeros of Yn'

    See Also
    --------
    jn_zeros, jnp_zeros, yn_zeros, ynp_zeros

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not (isscalar(nt) and isscalar(n)):
        raise ValueError("Arguments must be scalars.")
    if (floor(n) != n) or (floor(nt) != nt):
        raise ValueError("Arguments must be integers.")
    if (nt <= 0):
        raise ValueError("nt > 0")
    return specfun.jyzo(abs(n), nt)


def jn_zeros(n, nt):
    r"""Compute zeros of integer-order Bessel functions Jn.

    Compute `nt` zeros of the Bessel functions :math:`J_n(x)` on the
    interval :math:`(0, \infty)`. The zeros are returned in ascending
    order. Note that this interval excludes the zero at :math:`x = 0`
    that exists for :math:`n > 0`.

    Parameters
    ----------
    n : int
        Order of Bessel function
    nt : int
        Number of zeros to return

    Returns
    -------
    ndarray
        First `n` zeros of the Bessel function.

    See Also
    --------
    jv

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    Examples
    --------
    >>> import scipy.special as sc

    We can check that we are getting approximations of the zeros by
    evaluating them with `jv`.

    >>> n = 1
    >>> x = sc.jn_zeros(n, 3)
    >>> x
    array([ 3.83170597,  7.01558667, 10.17346814])
    >>> sc.jv(n, x)
    array([-0.00000000e+00,  1.72975330e-16,  2.89157291e-16])

    Note that the zero at ``x = 0`` for ``n > 0`` is not included.

    >>> sc.jv(1, 0)
    0.0

    """
    return jnyn_zeros(n, nt)[0]


def jnp_zeros(n, nt):
    r"""Compute zeros of integer-order Bessel function derivatives Jn'.

    Compute `nt` zeros of the functions :math:`J_n'(x)` on the
    interval :math:`(0, \infty)`. The zeros are returned in ascending
    order. Note that this interval excludes the zero at :math:`x = 0`
    that exists for :math:`n > 1`.

    Parameters
    ----------
    n : int
        Order of Bessel function
    nt : int
        Number of zeros to return

    Returns
    -------
    ndarray
        First `n` zeros of the Bessel function.

    See Also
    --------
    jvp, jv

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    Examples
    --------
    >>> import scipy.special as sc

    We can check that we are getting approximations of the zeros by
    evaluating them with `jvp`.

    >>> n = 2
    >>> x = sc.jnp_zeros(n, 3)
    >>> x
    array([3.05423693, 6.70613319, 9.96946782])
    >>> sc.jvp(n, x)
    array([ 2.77555756e-17,  2.08166817e-16, -3.01841885e-16])

    Note that the zero at ``x = 0`` for ``n > 1`` is not included.

    >>> sc.jvp(n, 0)
    0.0

    """
    return jnyn_zeros(n, nt)[1]


def yn_zeros(n, nt):
    r"""Compute zeros of integer-order Bessel function Yn(x).

    Compute `nt` zeros of the functions :math:`Y_n(x)` on the interval
    :math:`(0, \infty)`. The zeros are returned in ascending order.

    Parameters
    ----------
    n : int
        Order of Bessel function
    nt : int
        Number of zeros to return

    Returns
    -------
    ndarray
        First `n` zeros of the Bessel function.

    See Also
    --------
    yn, yv

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    Examples
    --------
    >>> import scipy.special as sc

    We can check that we are getting approximations of the zeros by
    evaluating them with `yn`.

    >>> n = 2
    >>> x = sc.yn_zeros(n, 3)
    >>> x
    array([ 3.38424177,  6.79380751, 10.02347798])
    >>> sc.yn(n, x)
    array([-1.94289029e-16,  8.32667268e-17, -1.52655666e-16])

    """
    return jnyn_zeros(n, nt)[2]


def ynp_zeros(n, nt):
    r"""Compute zeros of integer-order Bessel function derivatives Yn'(x).

    Compute `nt` zeros of the functions :math:`Y_n'(x)` on the
    interval :math:`(0, \infty)`. The zeros are returned in ascending
    order.

    Parameters
    ----------
    n : int
        Order of Bessel function
    nt : int
        Number of zeros to return

    See Also
    --------
    yvp

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    Examples
    --------
    >>> import scipy.special as sc

    We can check that we are getting approximations of the zeros by
    evaluating them with `yvp`.

    >>> n = 2
    >>> x = sc.ynp_zeros(n, 3)
    >>> x
    array([ 5.00258293,  8.3507247 , 11.57419547])
    >>> sc.yvp(n, x)
    array([ 2.22044605e-16, -3.33066907e-16,  2.94902991e-16])

    """
    return jnyn_zeros(n, nt)[3]


def y0_zeros(nt, complex=False):
    """Compute nt zeros of Bessel function Y0(z), and derivative at each zero.

    The derivatives are given by Y0'(z0) = -Y1(z0) at each zero z0.

    Parameters
    ----------
    nt : int
        Number of zeros to return
    complex : bool, default False
        Set to False to return only the real zeros; set to True to return only
        the complex zeros with negative real part and positive imaginary part.
        Note that the complex conjugates of the latter are also zeros of the
        function, but are not returned by this routine.

    Returns
    -------
    z0n : ndarray
        Location of nth zero of Y0(z)
    y0pz0n : ndarray
        Value of derivative Y0'(z0) for nth zero

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("Arguments must be scalar positive integer.")
    kf = 0
    kc = not complex
    return specfun.cyzo(nt, kf, kc)


def y1_zeros(nt, complex=False):
    """Compute nt zeros of Bessel function Y1(z), and derivative at each zero.

    The derivatives are given by Y1'(z1) = Y0(z1) at each zero z1.

    Parameters
    ----------
    nt : int
        Number of zeros to return
    complex : bool, default False
        Set to False to return only the real zeros; set to True to return only
        the complex zeros with negative real part and positive imaginary part.
        Note that the complex conjugates of the latter are also zeros of the
        function, but are not returned by this routine.

    Returns
    -------
    z1n : ndarray
        Location of nth zero of Y1(z)
    y1pz1n : ndarray
        Value of derivative Y1'(z1) for nth zero

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("Arguments must be scalar positive integer.")
    kf = 1
    kc = not complex
    return specfun.cyzo(nt, kf, kc)


def y1p_zeros(nt, complex=False):
    """Compute nt zeros of Bessel derivative Y1'(z), and value at each zero.

    The values are given by Y1(z1) at each z1 where Y1'(z1)=0.

    Parameters
    ----------
    nt : int
        Number of zeros to return
    complex : bool, default False
        Set to False to return only the real zeros; set to True to return only
        the complex zeros with negative real part and positive imaginary part.
        Note that the complex conjugates of the latter are also zeros of the
        function, but are not returned by this routine.

    Returns
    -------
    z1pn : ndarray
        Location of nth zero of Y1'(z)
    y1z1pn : ndarray
        Value of derivative Y1(z1) for nth zero

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("Arguments must be scalar positive integer.")
    kf = 2
    kc = not complex
    return specfun.cyzo(nt, kf, kc)


def _bessel_diff_formula(v, z, n, L, phase):
    # from AMS55.
    # L(v, z) = J(v, z), Y(v, z), H1(v, z), H2(v, z), phase = -1
    # L(v, z) = I(v, z) or exp(v*pi*i)K(v, z), phase = 1
    # For K, you can pull out the exp((v-k)*pi*i) into the caller
    v = asarray(v)
    p = 1.0
    s = L(v-n, z)
    for i in range(1, n+1):
        p = phase * (p * (n-i+1)) / i   # = choose(k, i)
        s += p*L(v-n + i*2, z)
    return s / (2.**n)


def jvp(v, z, n=1):
    """Compute derivatives of Bessel functions of the first kind.

    Compute the nth derivative of the Bessel function `Jv` with
    respect to `z`.

    Parameters
    ----------
    v : float
        Order of Bessel function
    z : complex
        Argument at which to evaluate the derivative; can be real or
        complex.
    n : int, default 1
        Order of derivative

    Returns
    -------
    scalar or ndarray
        Values of the derivative of the Bessel function.

    Notes
    -----
    The derivative is computed using the relation DLFM 10.6.7 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.6.E7

    """
    n = _nonneg_int_or_fail(n, 'n')
    if n == 0:
        return jv(v, z)
    else:
        return _bessel_diff_formula(v, z, n, jv, -1)


def yvp(v, z, n=1):
    """Compute derivatives of Bessel functions of the second kind.

    Compute the nth derivative of the Bessel function `Yv` with
    respect to `z`.

    Parameters
    ----------
    v : float
        Order of Bessel function
    z : complex
        Argument at which to evaluate the derivative
    n : int, default 1
        Order of derivative

    Returns
    -------
    scalar or ndarray
        nth derivative of the Bessel function.

    Notes
    -----
    The derivative is computed using the relation DLFM 10.6.7 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.6.E7

    """
    n = _nonneg_int_or_fail(n, 'n')
    if n == 0:
        return yv(v, z)
    else:
        return _bessel_diff_formula(v, z, n, yv, -1)


def kvp(v, z, n=1):
    """Compute nth derivative of real-order modified Bessel function Kv(z)

    Kv(z) is the modified Bessel function of the second kind.
    Derivative is calculated with respect to `z`.

    Parameters
    ----------
    v : array_like of float
        Order of Bessel function
    z : array_like of complex
        Argument at which to evaluate the derivative
    n : int
        Order of derivative.  Default is first derivative.

    Returns
    -------
    out : ndarray
        The results

    Examples
    --------
    Calculate multiple values at order 5:

    >>> from scipy.special import kvp
    >>> kvp(5, (1, 2, 3+5j))
    array([-1.84903536e+03+0.j        , -2.57735387e+01+0.j        ,
           -3.06627741e-02+0.08750845j])


    Calculate for a single value at multiple orders:

    >>> kvp((4, 4.5, 5), 1)
    array([ -184.0309,  -568.9585, -1849.0354])

    Notes
    -----
    The derivative is computed using the relation DLFM 10.29.5 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 6.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.29.E5

    """
    n = _nonneg_int_or_fail(n, 'n')
    if n == 0:
        return kv(v, z)
    else:
        return (-1)**n * _bessel_diff_formula(v, z, n, kv, 1)


def ivp(v, z, n=1):
    """Compute derivatives of modified Bessel functions of the first kind.

    Compute the nth derivative of the modified Bessel function `Iv`
    with respect to `z`.

    Parameters
    ----------
    v : array_like
        Order of Bessel function
    z : array_like
        Argument at which to evaluate the derivative; can be real or
        complex.
    n : int, default 1
        Order of derivative

    Returns
    -------
    scalar or ndarray
        nth derivative of the modified Bessel function.

    See Also
    --------
    iv

    Notes
    -----
    The derivative is computed using the relation DLFM 10.29.5 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 6.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.29.E5

    """
    n = _nonneg_int_or_fail(n, 'n')
    if n == 0:
        return iv(v, z)
    else:
        return _bessel_diff_formula(v, z, n, iv, 1)


def h1vp(v, z, n=1):
    """Compute nth derivative of Hankel function H1v(z) with respect to `z`.

    Parameters
    ----------
    v : array_like
        Order of Hankel function
    z : array_like
        Argument at which to evaluate the derivative. Can be real or
        complex.
    n : int, default 1
        Order of derivative

    Returns
    -------
    scalar or ndarray
        Values of the derivative of the Hankel function.

    Notes
    -----
    The derivative is computed using the relation DLFM 10.6.7 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.6.E7

    """
    n = _nonneg_int_or_fail(n, 'n')
    if n == 0:
        return hankel1(v, z)
    else:
        return _bessel_diff_formula(v, z, n, hankel1, -1)


def h2vp(v, z, n=1):
    """Compute nth derivative of Hankel function H2v(z) with respect to `z`.

    Parameters
    ----------
    v : array_like
        Order of Hankel function
    z : array_like
        Argument at which to evaluate the derivative. Can be real or
        complex.
    n : int, default 1
        Order of derivative

    Returns
    -------
    scalar or ndarray
        Values of the derivative of the Hankel function.

    Notes
    -----
    The derivative is computed using the relation DLFM 10.6.7 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.6.E7

    """
    n = _nonneg_int_or_fail(n, 'n')
    if n == 0:
        return hankel2(v, z)
    else:
        return _bessel_diff_formula(v, z, n, hankel2, -1)


def riccati_jn(n, x):
    r"""Compute Ricatti-Bessel function of the first kind and its derivative.

    The Ricatti-Bessel function of the first kind is defined as :math:`x
    j_n(x)`, where :math:`j_n` is the spherical Bessel function of the first
    kind of order :math:`n`.

    This function computes the value and first derivative of the
    Ricatti-Bessel function for all orders up to and including `n`.

    Parameters
    ----------
    n : int
        Maximum order of function to compute
    x : float
        Argument at which to evaluate

    Returns
    -------
    jn : ndarray
        Value of j0(x), ..., jn(x)
    jnp : ndarray
        First derivative j0'(x), ..., jn'(x)

    Notes
    -----
    The computation is carried out via backward recurrence, using the
    relation DLMF 10.51.1 [2]_.

    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
    Jin [1]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.51.E1

    """
    if not (isscalar(n) and isscalar(x)):
        raise ValueError("arguments must be scalars.")
    n = _nonneg_int_or_fail(n, 'n', strict=False)
    if (n == 0):
        n1 = 1
    else:
        n1 = n
    nm, jn, jnp = specfun.rctj(n1, x)
    return jn[:(n+1)], jnp[:(n+1)]


def riccati_yn(n, x):
    """Compute Ricatti-Bessel function of the second kind and its derivative.

    The Ricatti-Bessel function of the second kind is defined as :math:`x
    y_n(x)`, where :math:`y_n` is the spherical Bessel function of the second
    kind of order :math:`n`.

    This function computes the value and first derivative of the function for
    all orders up to and including `n`.

    Parameters
    ----------
    n : int
        Maximum order of function to compute
    x : float
        Argument at which to evaluate

    Returns
    -------
    yn : ndarray
        Value of y0(x), ..., yn(x)
    ynp : ndarray
        First derivative y0'(x), ..., yn'(x)

    Notes
    -----
    The computation is carried out via ascending recurrence, using the
    relation DLMF 10.51.1 [2]_.

    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
    Jin [1]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.51.E1

    """
    if not (isscalar(n) and isscalar(x)):
        raise ValueError("arguments must be scalars.")
    n = _nonneg_int_or_fail(n, 'n', strict=False)
    if (n == 0):
        n1 = 1
    else:
        n1 = n
    nm, jn, jnp = specfun.rcty(n1, x)
    return jn[:(n+1)], jnp[:(n+1)]


def erf_zeros(nt):
    """Compute the first nt zero in the first quadrant, ordered by absolute value.

    Zeros in the other quadrants can be obtained by using the symmetries erf(-z) = erf(z) and
    erf(conj(z)) = conj(erf(z)).


    Parameters
    ----------
    nt : int
        The number of zeros to compute

    Returns
    -------
    The locations of the zeros of erf : ndarray (complex)
        Complex values at which zeros of erf(z)

    Examples
    --------
    >>> from scipy import special
    >>> special.erf_zeros(1)
    array([1.45061616+1.880943j])

    Check that erf is (close to) zero for the value returned by erf_zeros

    >>> special.erf(special.erf_zeros(1))
    array([4.95159469e-14-1.16407394e-16j])

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if (floor(nt) != nt) or (nt <= 0) or not isscalar(nt):
        raise ValueError("Argument must be positive scalar integer.")
    return specfun.cerzo(nt)


def fresnelc_zeros(nt):
    """Compute nt complex zeros of cosine Fresnel integral C(z).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if (floor(nt) != nt) or (nt <= 0) or not isscalar(nt):
        raise ValueError("Argument must be positive scalar integer.")
    return specfun.fcszo(1, nt)


def fresnels_zeros(nt):
    """Compute nt complex zeros of sine Fresnel integral S(z).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if (floor(nt) != nt) or (nt <= 0) or not isscalar(nt):
        raise ValueError("Argument must be positive scalar integer.")
    return specfun.fcszo(2, nt)


def fresnel_zeros(nt):
    """Compute nt complex zeros of sine and cosine Fresnel integrals S(z) and C(z).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if (floor(nt) != nt) or (nt <= 0) or not isscalar(nt):
        raise ValueError("Argument must be positive scalar integer.")
    return specfun.fcszo(2, nt), specfun.fcszo(1, nt)


def assoc_laguerre(x, n, k=0.0):
    """Compute the generalized (associated) Laguerre polynomial of degree n and order k.

    The polynomial :math:`L^{(k)}_n(x)` is orthogonal over ``[0, inf)``,
    with weighting function ``exp(-x) * x**k`` with ``k > -1``.

    Notes
    -----
    `assoc_laguerre` is a simple wrapper around `eval_genlaguerre`, with
    reversed argument order ``(x, n, k=0.0) --> (n, k, x)``.

    """
    return orthogonal.eval_genlaguerre(n, k, x)


digamma = psi


def polygamma(n, x):
    r"""Polygamma functions.

    Defined as :math:`\psi^{(n)}(x)` where :math:`\psi` is the
    `digamma` function. See [dlmf]_ for details.

    Parameters
    ----------
    n : array_like
        The order of the derivative of the digamma function; must be
        integral
    x : array_like
        Real valued input

    Returns
    -------
    ndarray
        Function results

    See Also
    --------
    digamma

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/5.15

    Examples
    --------
    >>> from scipy import special
    >>> x = [2, 3, 25.5]
    >>> special.polygamma(1, x)
    array([ 0.64493407,  0.39493407,  0.03999467])
    >>> special.polygamma(0, x) == special.psi(x)
    array([ True,  True,  True], dtype=bool)

    """
    n, x = asarray(n), asarray(x)
    fac2 = (-1.0)**(n+1) * gamma(n+1.0) * zeta(n+1, x)
    return where(n == 0, psi(x), fac2)


def mathieu_even_coef(m, q):
    r"""Fourier coefficients for even Mathieu and modified Mathieu functions.

    The Fourier series of the even solutions of the Mathieu differential
    equation are of the form

    .. math:: \mathrm{ce}_{2n}(z, q) = \sum_{k=0}^{\infty} A_{(2n)}^{(2k)} \cos 2kz

    .. math:: \mathrm{ce}_{2n+1}(z, q) = \sum_{k=0}^{\infty} A_{(2n+1)}^{(2k+1)} \cos (2k+1)z

    This function returns the coefficients :math:`A_{(2n)}^{(2k)}` for even
    input m=2n, and the coefficients :math:`A_{(2n+1)}^{(2k+1)}` for odd input
    m=2n+1.

    Parameters
    ----------
    m : int
        Order of Mathieu functions.  Must be non-negative.
    q : float (>=0)
        Parameter of Mathieu functions.  Must be non-negative.

    Returns
    -------
    Ak : ndarray
        Even or odd Fourier coefficients, corresponding to even or odd m.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/28.4#i

    """
    if not (isscalar(m) and isscalar(q)):
        raise ValueError("m and q must be scalars.")
    if (q < 0):
        raise ValueError("q >=0")
    if (m != floor(m)) or (m < 0):
        raise ValueError("m must be an integer >=0.")

    if (q <= 1):
        qm = 7.5 + 56.1*sqrt(q) - 134.7*q + 90.7*sqrt(q)*q
    else:
        qm = 17.0 + 3.1*sqrt(q) - .126*q + .0037*sqrt(q)*q
    km = int(qm + 0.5*m)
    if km > 251:
        print("Warning, too many predicted coefficients.")
    kd = 1
    m = int(floor(m))
    if m % 2:
        kd = 2

    a = mathieu_a(m, q)
    fc = specfun.fcoef(kd, m, q, a)
    return fc[:km]


def mathieu_odd_coef(m, q):
    r"""Fourier coefficients for even Mathieu and modified Mathieu functions.

    The Fourier series of the odd solutions of the Mathieu differential
    equation are of the form

    .. math:: \mathrm{se}_{2n+1}(z, q) = \sum_{k=0}^{\infty} B_{(2n+1)}^{(2k+1)} \sin (2k+1)z

    .. math:: \mathrm{se}_{2n+2}(z, q) = \sum_{k=0}^{\infty} B_{(2n+2)}^{(2k+2)} \sin (2k+2)z

    This function returns the coefficients :math:`B_{(2n+2)}^{(2k+2)}` for even
    input m=2n+2, and the coefficients :math:`B_{(2n+1)}^{(2k+1)}` for odd
    input m=2n+1.

    Parameters
    ----------
    m : int
        Order of Mathieu functions.  Must be non-negative.
    q : float (>=0)
        Parameter of Mathieu functions.  Must be non-negative.

    Returns
    -------
    Bk : ndarray
        Even or odd Fourier coefficients, corresponding to even or odd m.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not (isscalar(m) and isscalar(q)):
        raise ValueError("m and q must be scalars.")
    if (q < 0):
        raise ValueError("q >=0")
    if (m != floor(m)) or (m <= 0):
        raise ValueError("m must be an integer > 0")

    if (q <= 1):
        qm = 7.5 + 56.1*sqrt(q) - 134.7*q + 90.7*sqrt(q)*q
    else:
        qm = 17.0 + 3.1*sqrt(q) - .126*q + .0037*sqrt(q)*q
    km = int(qm + 0.5*m)
    if km > 251:
        print("Warning, too many predicted coefficients.")
    kd = 4
    m = int(floor(m))
    if m % 2:
        kd = 3

    b = mathieu_b(m, q)
    fc = specfun.fcoef(kd, m, q, b)
    return fc[:km]


def lpmn(m, n, z):
    """Sequence of associated Legendre functions of the first kind.

    Computes the associated Legendre function of the first kind of order m and
    degree n, ``Pmn(z)`` = :math:`P_n^m(z)`, and its derivative, ``Pmn'(z)``.
    Returns two arrays of size ``(m+1, n+1)`` containing ``Pmn(z)`` and
    ``Pmn'(z)`` for all orders from ``0..m`` and degrees from ``0..n``.

    This function takes a real argument ``z``. For complex arguments ``z``
    use clpmn instead.

    Parameters
    ----------
    m : int
       ``|m| <= n``; the order of the Legendre function.
    n : int
       where ``n >= 0``; the degree of the Legendre function.  Often
       called ``l`` (lower case L) in descriptions of the associated
       Legendre function
    z : float
        Input value.

    Returns
    -------
    Pmn_z : (m+1, n+1) array
       Values for all orders 0..m and degrees 0..n
    Pmn_d_z : (m+1, n+1) array
       Derivatives for all orders 0..m and degrees 0..n

    See Also
    --------
    clpmn: associated Legendre functions of the first kind for complex z

    Notes
    -----
    In the interval (-1, 1), Ferrer's function of the first kind is
    returned. The phase convention used for the intervals (1, inf)
    and (-inf, -1) is such that the result is always real.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/14.3

    """
    if not isscalar(m) or (abs(m) > n):
        raise ValueError("m must be <= n.")
    if not isscalar(n) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if not isscalar(z):
        raise ValueError("z must be scalar.")
    if iscomplex(z):
        raise ValueError("Argument must be real. Use clpmn instead.")
    if (m < 0):
        mp = -m
        mf, nf = mgrid[0:mp+1, 0:n+1]
        with ufuncs.errstate(all='ignore'):
            if abs(z) < 1:
                # Ferrer function; DLMF 14.9.3
                fixarr = where(mf > nf, 0.0,
                               (-1)**mf * gamma(nf-mf+1) / gamma(nf+mf+1))
            else:
                # Match to clpmn; DLMF 14.9.13
                fixarr = where(mf > nf, 0.0, gamma(nf-mf+1) / gamma(nf+mf+1))
    else:
        mp = m
    p, pd = specfun.lpmn(mp, n, z)
    if (m < 0):
        p = p * fixarr
        pd = pd * fixarr
    return p, pd


def clpmn(m, n, z, type=3):
    """Associated Legendre function of the first kind for complex arguments.

    Computes the associated Legendre function of the first kind of order m and
    degree n, ``Pmn(z)`` = :math:`P_n^m(z)`, and its derivative, ``Pmn'(z)``.
    Returns two arrays of size ``(m+1, n+1)`` containing ``Pmn(z)`` and
    ``Pmn'(z)`` for all orders from ``0..m`` and degrees from ``0..n``.

    Parameters
    ----------
    m : int
       ``|m| <= n``; the order of the Legendre function.
    n : int
       where ``n >= 0``; the degree of the Legendre function.  Often
       called ``l`` (lower case L) in descriptions of the associated
       Legendre function
    z : float or complex
        Input value.
    type : int, optional
       takes values 2 or 3
       2: cut on the real axis ``|x| > 1``
       3: cut on the real axis ``-1 < x < 1`` (default)

    Returns
    -------
    Pmn_z : (m+1, n+1) array
       Values for all orders ``0..m`` and degrees ``0..n``
    Pmn_d_z : (m+1, n+1) array
       Derivatives for all orders ``0..m`` and degrees ``0..n``

    See Also
    --------
    lpmn: associated Legendre functions of the first kind for real z

    Notes
    -----
    By default, i.e. for ``type=3``, phase conventions are chosen according
    to [1]_ such that the function is analytic. The cut lies on the interval
    (-1, 1). Approaching the cut from above or below in general yields a phase
    factor with respect to Ferrer's function of the first kind
    (cf. `lpmn`).

    For ``type=2`` a cut at ``|x| > 1`` is chosen. Approaching the real values
    on the interval (-1, 1) in the complex plane yields Ferrer's function
    of the first kind.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/14.21

    """
    if not isscalar(m) or (abs(m) > n):
        raise ValueError("m must be <= n.")
    if not isscalar(n) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if not isscalar(z):
        raise ValueError("z must be scalar.")
    if not(type == 2 or type == 3):
        raise ValueError("type must be either 2 or 3.")
    if (m < 0):
        mp = -m
        mf, nf = mgrid[0:mp+1, 0:n+1]
        with ufuncs.errstate(all='ignore'):
            if type == 2:
                fixarr = where(mf > nf, 0.0,
                               (-1)**mf * gamma(nf-mf+1) / gamma(nf+mf+1))
            else:
                fixarr = where(mf > nf, 0.0, gamma(nf-mf+1) / gamma(nf+mf+1))
    else:
        mp = m
    p, pd = specfun.clpmn(mp, n, real(z), imag(z), type)
    if (m < 0):
        p = p * fixarr
        pd = pd * fixarr
    return p, pd


def lqmn(m, n, z):
    """Sequence of associated Legendre functions of the second kind.

    Computes the associated Legendre function of the second kind of order m and
    degree n, ``Qmn(z)`` = :math:`Q_n^m(z)`, and its derivative, ``Qmn'(z)``.
    Returns two arrays of size ``(m+1, n+1)`` containing ``Qmn(z)`` and
    ``Qmn'(z)`` for all orders from ``0..m`` and degrees from ``0..n``.

    Parameters
    ----------
    m : int
       ``|m| <= n``; the order of the Legendre function.
    n : int
       where ``n >= 0``; the degree of the Legendre function.  Often
       called ``l`` (lower case L) in descriptions of the associated
       Legendre function
    z : complex
        Input value.

    Returns
    -------
    Qmn_z : (m+1, n+1) array
       Values for all orders 0..m and degrees 0..n
    Qmn_d_z : (m+1, n+1) array
       Derivatives for all orders 0..m and degrees 0..n

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not isscalar(m) or (m < 0):
        raise ValueError("m must be a non-negative integer.")
    if not isscalar(n) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if not isscalar(z):
        raise ValueError("z must be scalar.")
    m = int(m)
    n = int(n)

    # Ensure neither m nor n == 0
    mm = max(1, m)
    nn = max(1, n)

    if iscomplex(z):
        q, qd = specfun.clqmn(mm, nn, z)
    else:
        q, qd = specfun.lqmn(mm, nn, z)
    return q[:(m+1), :(n+1)], qd[:(m+1), :(n+1)]


def bernoulli(n):
    """Bernoulli numbers B0..Bn (inclusive).

    Parameters
    ----------
    n : int
        Indicated the number of terms in the Bernoulli series to generate.

    Returns
    -------
    ndarray
        The Bernoulli numbers ``[B(0), B(1), ..., B(n)]``.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
    .. [2] "Bernoulli number", Wikipedia, https://en.wikipedia.org/wiki/Bernoulli_number

    Examples
    --------
    >>> from scipy.special import bernoulli, zeta
    >>> bernoulli(4)
    array([ 1.        , -0.5       ,  0.16666667,  0.        , -0.03333333])

    The Wikipedia article ([2]_) points out the relationship between the
    Bernoulli numbers and the zeta function, ``B_n^+ = -n * zeta(1 - n)``
    for ``n > 0``:

    >>> n = np.arange(1, 5)
    >>> -n * zeta(1 - n)
    array([ 0.5       ,  0.16666667, -0.        , -0.03333333])

    Note that, in the notation used in the wikipedia article,
    `bernoulli` computes ``B_n^-`` (i.e. it used the convention that
    ``B_1`` is -1/2).  The relation given above is for ``B_n^+``, so the
    sign of 0.5 does not match the output of ``bernoulli(4)``.

    """
    if not isscalar(n) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    n = int(n)
    if (n < 2):
        n1 = 2
    else:
        n1 = n
    return specfun.bernob(int(n1))[:(n+1)]


def euler(n):
    """Euler numbers E(0), E(1), ..., E(n).

    The Euler numbers [1]_ are also known as the secant numbers.

    Because ``euler(n)`` returns floating point values, it does not give
    exact values for large `n`.  The first inexact value is E(22).

    Parameters
    ----------
    n : int
        The highest index of the Euler number to be returned.

    Returns
    -------
    ndarray
        The Euler numbers [E(0), E(1), ..., E(n)].
        The odd Euler numbers, which are all zero, are included.

    References
    ----------
    .. [1] Sequence A122045, The On-Line Encyclopedia of Integer Sequences,
           https://oeis.org/A122045
    .. [2] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    Examples
    --------
    >>> from scipy.special import euler
    >>> euler(6)
    array([  1.,   0.,  -1.,   0.,   5.,   0., -61.])

    >>> euler(13).astype(np.int64)
    array([      1,       0,      -1,       0,       5,       0,     -61,
                 0,    1385,       0,  -50521,       0, 2702765,       0])

    >>> euler(22)[-1]  # Exact value of E(22) is -69348874393137901.
    -69348874393137976.0

    """
    if not isscalar(n) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    n = int(n)
    if (n < 2):
        n1 = 2
    else:
        n1 = n
    return specfun.eulerb(n1)[:(n+1)]


def lpn(n, z):
    """Legendre function of the first kind.

    Compute sequence of Legendre functions of the first kind (polynomials),
    Pn(z) and derivatives for all degrees from 0 to n (inclusive).

    See also special.legendre for polynomial class.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not (isscalar(n) and isscalar(z)):
        raise ValueError("arguments must be scalars.")
    n = _nonneg_int_or_fail(n, 'n', strict=False)
    if (n < 1):
        n1 = 1
    else:
        n1 = n
    if iscomplex(z):
        pn, pd = specfun.clpn(n1, z)
    else:
        pn, pd = specfun.lpn(n1, z)
    return pn[:(n+1)], pd[:(n+1)]


def lqn(n, z):
    """Legendre function of the second kind.

    Compute sequence of Legendre functions of the second kind, Qn(z) and
    derivatives for all degrees from 0 to n (inclusive).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not (isscalar(n) and isscalar(z)):
        raise ValueError("arguments must be scalars.")
    n = _nonneg_int_or_fail(n, 'n', strict=False)
    if (n < 1):
        n1 = 1
    else:
        n1 = n
    if iscomplex(z):
        qn, qd = specfun.clqn(n1, z)
    else:
        qn, qd = specfun.lqnb(n1, z)
    return qn[:(n+1)], qd[:(n+1)]


def ai_zeros(nt):
    """
    Compute `nt` zeros and values of the Airy function Ai and its derivative.

    Computes the first `nt` zeros, `a`, of the Airy function Ai(x);
    first `nt` zeros, `ap`, of the derivative of the Airy function Ai'(x);
    the corresponding values Ai(a');
    and the corresponding values Ai'(a).

    Parameters
    ----------
    nt : int
        Number of zeros to compute

    Returns
    -------
    a : ndarray
        First `nt` zeros of Ai(x)
    ap : ndarray
        First `nt` zeros of Ai'(x)
    ai : ndarray
        Values of Ai(x) evaluated at first `nt` zeros of Ai'(x)
    aip : ndarray
        Values of Ai'(x) evaluated at first `nt` zeros of Ai(x)

    Examples
    --------
    >>> from scipy import special
    >>> a, ap, ai, aip = special.ai_zeros(3)
    >>> a
    array([-2.33810741, -4.08794944, -5.52055983])
    >>> ap
    array([-1.01879297, -3.24819758, -4.82009921])
    >>> ai
    array([ 0.53565666, -0.41901548,  0.38040647])
    >>> aip
    array([ 0.70121082, -0.80311137,  0.86520403])

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    kf = 1
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be a positive integer scalar.")
    return specfun.airyzo(nt, kf)


def bi_zeros(nt):
    """
    Compute `nt` zeros and values of the Airy function Bi and its derivative.

    Computes the first `nt` zeros, b, of the Airy function Bi(x);
    first `nt` zeros, b', of the derivative of the Airy function Bi'(x);
    the corresponding values Bi(b');
    and the corresponding values Bi'(b).

    Parameters
    ----------
    nt : int
        Number of zeros to compute

    Returns
    -------
    b : ndarray
        First `nt` zeros of Bi(x)
    bp : ndarray
        First `nt` zeros of Bi'(x)
    bi : ndarray
        Values of Bi(x) evaluated at first `nt` zeros of Bi'(x)
    bip : ndarray
        Values of Bi'(x) evaluated at first `nt` zeros of Bi(x)

    Examples
    --------
    >>> from scipy import special
    >>> b, bp, bi, bip = special.bi_zeros(3)
    >>> b
    array([-1.17371322, -3.2710933 , -4.83073784])
    >>> bp
    array([-2.29443968, -4.07315509, -5.51239573])
    >>> bi
    array([-0.45494438,  0.39652284, -0.36796916])
    >>> bip
    array([ 0.60195789, -0.76031014,  0.83699101])

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    kf = 2
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be a positive integer scalar.")
    return specfun.airyzo(nt, kf)


def lmbda(v, x):
    r"""Jahnke-Emden Lambda function, Lambdav(x).

    This function is defined as [2]_,

    .. math:: \Lambda_v(x) = \Gamma(v+1) \frac{J_v(x)}{(x/2)^v},

    where :math:`\Gamma` is the gamma function and :math:`J_v` is the
    Bessel function of the first kind.

    Parameters
    ----------
    v : float
        Order of the Lambda function
    x : float
        Value at which to evaluate the function and derivatives

    Returns
    -------
    vl : ndarray
        Values of Lambda_vi(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.
    dl : ndarray
        Derivatives Lambda_vi'(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
    .. [2] Jahnke, E. and Emde, F. "Tables of Functions with Formulae and
           Curves" (4th ed.), Dover, 1945
    """
    if not (isscalar(v) and isscalar(x)):
        raise ValueError("arguments must be scalars.")
    if (v < 0):
        raise ValueError("argument must be > 0.")
    n = int(v)
    v0 = v - n
    if (n < 1):
        n1 = 1
    else:
        n1 = n
    v1 = n1 + v0
    if (v != floor(v)):
        vm, vl, dl = specfun.lamv(v1, x)
    else:
        vm, vl, dl = specfun.lamn(v1, x)
    return vl[:(n+1)], dl[:(n+1)]


def pbdv_seq(v, x):
    """Parabolic cylinder functions Dv(x) and derivatives.

    Parameters
    ----------
    v : float
        Order of the parabolic cylinder function
    x : float
        Value at which to evaluate the function and derivatives

    Returns
    -------
    dv : ndarray
        Values of D_vi(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.
    dp : ndarray
        Derivatives D_vi'(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 13.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not (isscalar(v) and isscalar(x)):
        raise ValueError("arguments must be scalars.")
    n = int(v)
    v0 = v-n
    if (n < 1):
        n1 = 1
    else:
        n1 = n
    v1 = n1 + v0
    dv, dp, pdf, pdd = specfun.pbdv(v1, x)
    return dv[:n1+1], dp[:n1+1]


def pbvv_seq(v, x):
    """Parabolic cylinder functions Vv(x) and derivatives.

    Parameters
    ----------
    v : float
        Order of the parabolic cylinder function
    x : float
        Value at which to evaluate the function and derivatives

    Returns
    -------
    dv : ndarray
        Values of V_vi(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.
    dp : ndarray
        Derivatives V_vi'(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 13.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not (isscalar(v) and isscalar(x)):
        raise ValueError("arguments must be scalars.")
    n = int(v)
    v0 = v-n
    if (n <= 1):
        n1 = 1
    else:
        n1 = n
    v1 = n1 + v0
    dv, dp, pdf, pdd = specfun.pbvv(v1, x)
    return dv[:n1+1], dp[:n1+1]


def pbdn_seq(n, z):
    """Parabolic cylinder functions Dn(z) and derivatives.

    Parameters
    ----------
    n : int
        Order of the parabolic cylinder function
    z : complex
        Value at which to evaluate the function and derivatives

    Returns
    -------
    dv : ndarray
        Values of D_i(z), for i=0, ..., i=n.
    dp : ndarray
        Derivatives D_i'(z), for i=0, ..., i=n.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 13.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not (isscalar(n) and isscalar(z)):
        raise ValueError("arguments must be scalars.")
    if (floor(n) != n):
        raise ValueError("n must be an integer.")
    if (abs(n) <= 1):
        n1 = 1
    else:
        n1 = n
    cpb, cpd = specfun.cpbdn(n1, z)
    return cpb[:n1+1], cpd[:n1+1]


def ber_zeros(nt):
    """Compute nt zeros of the Kelvin function ber.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the Kelvin function.

    See Also
    --------
    ber

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return specfun.klvnzo(nt, 1)


def bei_zeros(nt):
    """Compute nt zeros of the Kelvin function bei.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the Kelvin function.

    See Also
    --------
    bei

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return specfun.klvnzo(nt, 2)


def ker_zeros(nt):
    """Compute nt zeros of the Kelvin function ker.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the Kelvin function.

    See Also
    --------
    ker

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return specfun.klvnzo(nt, 3)


def kei_zeros(nt):
    """Compute nt zeros of the Kelvin function kei.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the Kelvin function.

    See Also
    --------
    kei

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return specfun.klvnzo(nt, 4)


def berp_zeros(nt):
    """Compute nt zeros of the derivative of the Kelvin function ber.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the derivative of the Kelvin function.

    See Also
    --------
    ber, berp

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return specfun.klvnzo(nt, 5)


def beip_zeros(nt):
    """Compute nt zeros of the derivative of the Kelvin function bei.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the derivative of the Kelvin function.

    See Also
    --------
    bei, beip

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return specfun.klvnzo(nt, 6)


def kerp_zeros(nt):
    """Compute nt zeros of the derivative of the Kelvin function ker.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the derivative of the Kelvin function.

    See Also
    --------
    ker, kerp

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return specfun.klvnzo(nt, 7)


def keip_zeros(nt):
    """Compute nt zeros of the derivative of the Kelvin function kei.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the derivative of the Kelvin function.

    See Also
    --------
    kei, keip

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return specfun.klvnzo(nt, 8)


def kelvin_zeros(nt):
    """Compute nt zeros of all Kelvin functions.

    Returned in a length-8 tuple of arrays of length nt.  The tuple contains
    the arrays of zeros of (ber, bei, ker, kei, ber', bei', ker', kei').

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return (specfun.klvnzo(nt, 1),
            specfun.klvnzo(nt, 2),
            specfun.klvnzo(nt, 3),
            specfun.klvnzo(nt, 4),
            specfun.klvnzo(nt, 5),
            specfun.klvnzo(nt, 6),
            specfun.klvnzo(nt, 7),
            specfun.klvnzo(nt, 8))


def pro_cv_seq(m, n, c):
    """Characteristic values for prolate spheroidal wave functions.

    Compute a sequence of characteristic values for the prolate
    spheroidal wave functions for mode m and n'=m..n and spheroidal
    parameter c.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not (isscalar(m) and isscalar(n) and isscalar(c)):
        raise ValueError("Arguments must be scalars.")
    if (n != floor(n)) or (m != floor(m)):
        raise ValueError("Modes must be integers.")
    if (n-m > 199):
        raise ValueError("Difference between n and m is too large.")
    maxL = n-m+1
    return specfun.segv(m, n, c, 1)[1][:maxL]


def obl_cv_seq(m, n, c):
    """Characteristic values for oblate spheroidal wave functions.

    Compute a sequence of characteristic values for the oblate
    spheroidal wave functions for mode m and n'=m..n and spheroidal
    parameter c.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """
    if not (isscalar(m) and isscalar(n) and isscalar(c)):
        raise ValueError("Arguments must be scalars.")
    if (n != floor(n)) or (m != floor(m)):
        raise ValueError("Modes must be integers.")
    if (n-m > 199):
        raise ValueError("Difference between n and m is too large.")
    maxL = n-m+1
    return specfun.segv(m, n, c, -1)[1][:maxL]


def comb(N, k, exact=False, repetition=False):
    """The number of combinations of N things taken k at a time.

    This is often expressed as "N choose k".

    Parameters
    ----------
    N : int, ndarray
        Number of things.
    k : int, ndarray
        Number of elements taken.
    exact : bool, optional
        If `exact` is False, then floating point precision is used, otherwise
        exact long integer is computed.
    repetition : bool, optional
        If `repetition` is True, then the number of combinations with
        repetition is computed.

    Returns
    -------
    val : int, float, ndarray
        The total number of combinations.

    See Also
    --------
    binom : Binomial coefficient ufunc

    Notes
    -----
    - Array arguments accepted only for exact=False case.
    - If N < 0, or k < 0, then 0 is returned.
    - If k > N and repetition=False, then 0 is returned.

    Examples
    --------
    >>> from scipy.special import comb
    >>> k = np.array([3, 4])
    >>> n = np.array([10, 10])
    >>> comb(n, k, exact=False)
    array([ 120.,  210.])
    >>> comb(10, 3, exact=True)
    120
    >>> comb(10, 3, exact=True, repetition=True)
    220

    """
    if repetition:
        return comb(N + k - 1, k, exact)
    if exact:
        return _comb_int(N, k)
    else:
        k, N = asarray(k), asarray(N)
        cond = (k <= N) & (N >= 0) & (k >= 0)
        vals = binom(N, k)
        if isinstance(vals, np.ndarray):
            vals[~cond] = 0
        elif not cond:
            vals = np.float64(0)
        return vals


def perm(N, k, exact=False):
    """Permutations of N things taken k at a time, i.e., k-permutations of N.

    It's also known as "partial permutations".

    Parameters
    ----------
    N : int, ndarray
        Number of things.
    k : int, ndarray
        Number of elements taken.
    exact : bool, optional
        If `exact` is False, then floating point precision is used, otherwise
        exact long integer is computed.

    Returns
    -------
    val : int, ndarray
        The number of k-permutations of N.

    Notes
    -----
    - Array arguments accepted only for exact=False case.
    - If k > N, N < 0, or k < 0, then a 0 is returned.

    Examples
    --------
    >>> from scipy.special import perm
    >>> k = np.array([3, 4])
    >>> n = np.array([10, 10])
    >>> perm(n, k)
    array([  720.,  5040.])
    >>> perm(10, 3, exact=True)
    720

    """
    if exact:
        if (k > N) or (N < 0) or (k < 0):
            return 0
        val = 1
        for i in range(N - k + 1, N + 1):
            val *= i
        return val
    else:
        k, N = asarray(k), asarray(N)
        cond = (k <= N) & (N >= 0) & (k >= 0)
        vals = poch(N - k + 1, k)
        if isinstance(vals, np.ndarray):
            vals[~cond] = 0
        elif not cond:
            vals = np.float64(0)
        return vals


# https://stackoverflow.com/a/16327037
def _range_prod(lo, hi):
    """
    Product of a range of numbers.

    Returns the product of
    lo * (lo+1) * (lo+2) * ... * (hi-2) * (hi-1) * hi
    = hi! / (lo-1)!

    Breaks into smaller products first for speed:
    _range_prod(2, 9) = ((2*3)*(4*5))*((6*7)*(8*9))
    """
    if lo + 1 < hi:
        mid = (hi + lo) // 2
        return _range_prod(lo, mid) * _range_prod(mid + 1, hi)
    if lo == hi:
        return lo
    return lo * hi


def factorial(n, exact=False):
    """
    The factorial of a number or array of numbers.

    The factorial of non-negative integer `n` is the product of all
    positive integers less than or equal to `n`::

        n! = n * (n - 1) * (n - 2) * ... * 1

    Parameters
    ----------
    n : int or array_like of ints
        Input values.  If ``n < 0``, the return value is 0.
    exact : bool, optional
        If True, calculate the answer exactly using long integer arithmetic.
        If False, result is approximated in floating point rapidly using the
        `gamma` function.
        Default is False.

    Returns
    -------
    nf : float or int or ndarray
        Factorial of `n`, as integer or float depending on `exact`.

    Notes
    -----
    For arrays with ``exact=True``, the factorial is computed only once, for
    the largest input, with each other result computed in the process.
    The output dtype is increased to ``int64`` or ``object`` if necessary.

    With ``exact=False`` the factorial is approximated using the gamma
    function:

    .. math:: n! = \\Gamma(n+1)

    Examples
    --------
    >>> from scipy.special import factorial
    >>> arr = np.array([3, 4, 5])
    >>> factorial(arr, exact=False)
    array([   6.,   24.,  120.])
    >>> factorial(arr, exact=True)
    array([  6,  24, 120])
    >>> factorial(5, exact=True)
    120

    """
    if exact:
        if np.ndim(n) == 0:
            if np.isnan(n):
                return n
            return 0 if n < 0 else math.factorial(n)
        else:
            n = asarray(n)
            un = np.unique(n).astype(object)

            # Convert to object array of long ints if np.int_ can't handle size
            if np.isnan(n).any():
                dt = float
            elif un[-1] > 20:
                dt = object
            elif un[-1] > 12:
                dt = np.int64
            else:
                dt = np.int_

            out = np.empty_like(n, dtype=dt)

            # Handle invalid/trivial values
            # Ignore runtime warning when less operator used w/np.nan
            with np.errstate(all='ignore'):
                un = un[un > 1]
                out[n < 2] = 1
                out[n < 0] = 0

            # Calculate products of each range of numbers
            if un.size:
                val = math.factorial(un[0])
                out[n == un[0]] = val
                for i in range(len(un) - 1):
                    prev = un[i] + 1
                    current = un[i + 1]
                    val *= _range_prod(prev, current)
                    out[n == current] = val

            if np.isnan(n).any():
                out = out.astype(np.float64)
                out[np.isnan(n)] = n[np.isnan(n)]
            return out
    else:
        out = ufuncs._factorial(n)
        return out


def factorial2(n, exact=False):
    """Double factorial.

    This is the factorial with every second value skipped.  E.g., ``7!! = 7 * 5
    * 3 * 1``.  It can be approximated numerically as::

      n!! = special.gamma(n/2+1)*2**((m+1)/2)/sqrt(pi)  n odd
          = 2**(n/2) * (n/2)!                           n even

    Parameters
    ----------
    n : int or array_like
        Calculate ``n!!``.  Arrays are only supported with `exact` set
        to False.  If ``n < 0``, the return value is 0.
    exact : bool, optional
        The result can be approximated rapidly using the gamma-formula
        above (default).  If `exact` is set to True, calculate the
        answer exactly using integer arithmetic.

    Returns
    -------
    nff : float or int
        Double factorial of `n`, as an int or a float depending on
        `exact`.

    Examples
    --------
    >>> from scipy.special import factorial2
    >>> factorial2(7, exact=False)
    array(105.00000000000001)
    >>> factorial2(7, exact=True)
    105

    """
    if exact:
        if n < -1:
            return 0
        if n <= 0:
            return 1
        val = 1
        for k in range(n, 0, -2):
            val *= k
        return val
    else:
        n = asarray(n)
        vals = zeros(n.shape, 'd')
        cond1 = (n % 2) & (n >= -1)
        cond2 = (1-(n % 2)) & (n >= -1)
        oddn = extract(cond1, n)
        evenn = extract(cond2, n)
        nd2o = oddn / 2.0
        nd2e = evenn / 2.0
        place(vals, cond1, gamma(nd2o + 1) / sqrt(pi) * pow(2.0, nd2o + 0.5))
        place(vals, cond2, gamma(nd2e + 1) * pow(2.0, nd2e))
        return vals


def factorialk(n, k, exact=True):
    """Multifactorial of n of order k, n(!!...!).

    This is the multifactorial of n skipping k values.  For example,

      factorialk(17, 4) = 17!!!! = 17 * 13 * 9 * 5 * 1

    In particular, for any integer ``n``, we have

      factorialk(n, 1) = factorial(n)

      factorialk(n, 2) = factorial2(n)

    Parameters
    ----------
    n : int
        Calculate multifactorial. If `n` < 0, the return value is 0.
    k : int
        Order of multifactorial.
    exact : bool, optional
        If exact is set to True, calculate the answer exactly using
        integer arithmetic.

    Returns
    -------
    val : int
        Multifactorial of `n`.

    Raises
    ------
    NotImplementedError
        Raises when exact is False

    Examples
    --------
    >>> from scipy.special import factorialk
    >>> factorialk(5, 1, exact=True)
    120
    >>> factorialk(5, 3, exact=True)
    10

    """
    if exact:
        if n < 1-k:
            return 0
        if n <= 0:
            return 1
        val = 1
        for j in range(n, 0, -k):
            val = val*j
        return val
    else:
        raise NotImplementedError


def zeta(x, q=None, out=None):
    r"""
    Riemann or Hurwitz zeta function.

    Parameters
    ----------
    x : array_like of float
        Input data, must be real
    q : array_like of float, optional
        Input data, must be real.  Defaults to Riemann zeta.
    out : ndarray, optional
        Output array for the computed values.

    Returns
    -------
    out : array_like
        Values of zeta(x).

    Notes
    -----
    The two-argument version is the Hurwitz zeta function

    .. math::

        \zeta(x, q) = \sum_{k=0}^{\infty} \frac{1}{(k + q)^x};

    see [dlmf]_ for details. The Riemann zeta function corresponds to
    the case when ``q = 1``.

    See Also
    --------
    zetac

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/25.11#i

    Examples
    --------
    >>> from scipy.special import zeta, polygamma, factorial

    Some specific values:

    >>> zeta(2), np.pi**2/6
    (1.6449340668482266, 1.6449340668482264)

    >>> zeta(4), np.pi**4/90
    (1.0823232337111381, 1.082323233711138)

    Relation to the `polygamma` function:

    >>> m = 3
    >>> x = 1.25
    >>> polygamma(m, x)
    array(2.782144009188397)
    >>> (-1)**(m+1) * factorial(m) * zeta(m+1, x)
    2.7821440091883969

    """
    if q is None:
        return ufuncs._riemann_zeta(x, out)
    else:
        return ufuncs._zeta(x, q, out)
