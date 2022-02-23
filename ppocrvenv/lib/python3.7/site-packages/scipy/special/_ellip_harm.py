import numpy as np

from ._ufuncs import _ellip_harm
from ._ellip_harm_2 import _ellipsoid, _ellipsoid_norm


def ellip_harm(h2, k2, n, p, s, signm=1, signn=1):
    r"""
    Ellipsoidal harmonic functions E^p_n(l)

    These are also known as Lame functions of the first kind, and are
    solutions to the Lame equation:

    .. math:: (s^2 - h^2)(s^2 - k^2)E''(s) + s(2s^2 - h^2 - k^2)E'(s) + (a - q s^2)E(s) = 0

    where :math:`q = (n+1)n` and :math:`a` is the eigenvalue (not
    returned) corresponding to the solutions.

    Parameters
    ----------
    h2 : float
        ``h**2``
    k2 : float
        ``k**2``; should be larger than ``h**2``
    n : int
        Degree
    s : float
        Coordinate
    p : int
        Order, can range between [1,2n+1]
    signm : {1, -1}, optional
        Sign of prefactor of functions. Can be +/-1. See Notes.
    signn : {1, -1}, optional
        Sign of prefactor of functions. Can be +/-1. See Notes.

    Returns
    -------
    E : float
        the harmonic :math:`E^p_n(s)`

    See Also
    --------
    ellip_harm_2, ellip_normal

    Notes
    -----
    The geometric interpretation of the ellipsoidal functions is
    explained in [2]_, [3]_, [4]_. The `signm` and `signn` arguments control the
    sign of prefactors for functions according to their type::

        K : +1
        L : signm
        M : signn
        N : signm*signn

    .. versionadded:: 0.15.0

    References
    ----------
    .. [1] Digital Library of Mathematical Functions 29.12
       https://dlmf.nist.gov/29.12
    .. [2] Bardhan and Knepley, "Computational science and
       re-discovery: open-source implementations of
       ellipsoidal harmonics for problems in potential theory",
       Comput. Sci. Disc. 5, 014006 (2012)
       :doi:`10.1088/1749-4699/5/1/014006`.
    .. [3] David J.and Dechambre P, "Computation of Ellipsoidal
       Gravity Field Harmonics for small solar system bodies"
       pp. 30-36, 2000
    .. [4] George Dassios, "Ellipsoidal Harmonics: Theory and Applications"
       pp. 418, 2012

    Examples
    --------
    >>> from scipy.special import ellip_harm
    >>> w = ellip_harm(5,8,1,1,2.5)
    >>> w
    2.5

    Check that the functions indeed are solutions to the Lame equation:

    >>> from scipy.interpolate import UnivariateSpline
    >>> def eigenvalue(f, df, ddf):
    ...     r = ((s**2 - h**2)*(s**2 - k**2)*ddf + s*(2*s**2 - h**2 - k**2)*df - n*(n+1)*s**2*f)/f
    ...     return -r.mean(), r.std()
    >>> s = np.linspace(0.1, 10, 200)
    >>> k, h, n, p = 8.0, 2.2, 3, 2
    >>> E = ellip_harm(h**2, k**2, n, p, s)
    >>> E_spl = UnivariateSpline(s, E)
    >>> a, a_err = eigenvalue(E_spl(s), E_spl(s,1), E_spl(s,2))
    >>> a, a_err
    (583.44366156701483, 6.4580890640310646e-11)

    """
    return _ellip_harm(h2, k2, n, p, s, signm, signn)


_ellip_harm_2_vec = np.vectorize(_ellipsoid, otypes='d')


def ellip_harm_2(h2, k2, n, p, s):
    r"""
    Ellipsoidal harmonic functions F^p_n(l)

    These are also known as Lame functions of the second kind, and are
    solutions to the Lame equation:

    .. math:: (s^2 - h^2)(s^2 - k^2)F''(s) + s(2s^2 - h^2 - k^2)F'(s) + (a - q s^2)F(s) = 0

    where :math:`q = (n+1)n` and :math:`a` is the eigenvalue (not
    returned) corresponding to the solutions.

    Parameters
    ----------
    h2 : float
        ``h**2``
    k2 : float
        ``k**2``; should be larger than ``h**2``
    n : int
        Degree.
    p : int
        Order, can range between [1,2n+1].
    s : float
        Coordinate

    Returns
    -------
    F : float
        The harmonic :math:`F^p_n(s)`

    Notes
    -----
    Lame functions of the second kind are related to the functions of the first kind:

    .. math::

       F^p_n(s)=(2n + 1)E^p_n(s)\int_{0}^{1/s}\frac{du}{(E^p_n(1/u))^2\sqrt{(1-u^2k^2)(1-u^2h^2)}}

    .. versionadded:: 0.15.0

    See Also
    --------
    ellip_harm, ellip_normal

    Examples
    --------
    >>> from scipy.special import ellip_harm_2
    >>> w = ellip_harm_2(5,8,2,1,10)
    >>> w
    0.00108056853382

    """
    with np.errstate(all='ignore'):
        return _ellip_harm_2_vec(h2, k2, n, p, s)


def _ellip_normal_vec(h2, k2, n, p):
    return _ellipsoid_norm(h2, k2, n, p)


_ellip_normal_vec = np.vectorize(_ellip_normal_vec, otypes='d')


def ellip_normal(h2, k2, n, p):
    r"""
    Ellipsoidal harmonic normalization constants gamma^p_n

    The normalization constant is defined as

    .. math::

       \gamma^p_n=8\int_{0}^{h}dx\int_{h}^{k}dy\frac{(y^2-x^2)(E^p_n(y)E^p_n(x))^2}{\sqrt((k^2-y^2)(y^2-h^2)(h^2-x^2)(k^2-x^2)}

    Parameters
    ----------
    h2 : float
        ``h**2``
    k2 : float
        ``k**2``; should be larger than ``h**2``
    n : int
        Degree.
    p : int
        Order, can range between [1,2n+1].

    Returns
    -------
    gamma : float
        The normalization constant :math:`\gamma^p_n`

    See Also
    --------
    ellip_harm, ellip_harm_2

    Notes
    -----
    .. versionadded:: 0.15.0

    Examples
    --------
    >>> from scipy.special import ellip_normal
    >>> w = ellip_normal(5,8,3,7)
    >>> w
    1723.38796997

    """
    with np.errstate(all='ignore'):
        return _ellip_normal_vec(h2, k2, n, p)
