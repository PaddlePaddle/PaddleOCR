"""
A collection of functions to find the weights and abscissas for
Gaussian Quadrature.

These calculations are done by finding the eigenvalues of a
tridiagonal matrix whose entries are dependent on the coefficients
in the recursion formula for the orthogonal polynomials with the
corresponding weighting function over the interval.

Many recursion relations for orthogonal polynomials are given:

.. math::

    a1n f_{n+1} (x) = (a2n + a3n x ) f_n (x) - a4n f_{n-1} (x)

The recursion relation of interest is

.. math::

    P_{n+1} (x) = (x - A_n) P_n (x) - B_n P_{n-1} (x)

where :math:`P` has a different normalization than :math:`f`.

The coefficients can be found as:

.. math::

    A_n = -a2n / a3n
    \\qquad
    B_n = ( a4n / a3n \\sqrt{h_n-1 / h_n})^2

where

.. math::

    h_n = \\int_a^b w(x) f_n(x)^2

assume:

.. math::

    P_0 (x) = 1
    \\qquad
    P_{-1} (x) == 0

For the mathematical background, see [golub.welsch-1969-mathcomp]_ and
[abramowitz.stegun-1965]_.

References
----------
.. [golub.welsch-1969-mathcomp]
   Golub, Gene H, and John H Welsch. 1969. Calculation of Gauss
   Quadrature Rules. *Mathematics of Computation* 23, 221-230+s1--s10.

.. [abramowitz.stegun-1965]
   Abramowitz, Milton, and Irene A Stegun. (1965) *Handbook of
   Mathematical Functions: with Formulas, Graphs, and Mathematical
   Tables*. Gaithersburg, MD: National Bureau of Standards.
   http://www.math.sfu.ca/~cbm/aands/

.. [townsend.trogdon.olver-2014]
   Townsend, A. and Trogdon, T. and Olver, S. (2014)
   *Fast computation of Gauss quadrature nodes and
   weights on the whole real line*. :arXiv:`1410.5286`.

.. [townsend.trogdon.olver-2015]
   Townsend, A. and Trogdon, T. and Olver, S. (2015)
   *Fast computation of Gauss quadrature nodes and
   weights on the whole real line*.
   IMA Journal of Numerical Analysis
   :doi:`10.1093/imanum/drv002`.
"""
#
# Author:  Travis Oliphant 2000
# Updated Sep. 2003 (fixed bugs --- tested to be accurate)

# SciPy imports.
import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
                   hstack, arccos, arange)
from scipy import linalg
from scipy.special import airy

# Local imports.
from . import _ufuncs
from . import _ufuncs as cephes
_gam = cephes.gamma
from . import specfun

_polyfuns = ['legendre', 'chebyt', 'chebyu', 'chebyc', 'chebys',
             'jacobi', 'laguerre', 'genlaguerre', 'hermite',
             'hermitenorm', 'gegenbauer', 'sh_legendre', 'sh_chebyt',
             'sh_chebyu', 'sh_jacobi']

# Correspondence between new and old names of root functions
_rootfuns_map = {'roots_legendre': 'p_roots',
               'roots_chebyt': 't_roots',
               'roots_chebyu': 'u_roots',
               'roots_chebyc': 'c_roots',
               'roots_chebys': 's_roots',
               'roots_jacobi': 'j_roots',
               'roots_laguerre': 'l_roots',
               'roots_genlaguerre': 'la_roots',
               'roots_hermite': 'h_roots',
               'roots_hermitenorm': 'he_roots',
               'roots_gegenbauer': 'cg_roots',
               'roots_sh_legendre': 'ps_roots',
               'roots_sh_chebyt': 'ts_roots',
               'roots_sh_chebyu': 'us_roots',
               'roots_sh_jacobi': 'js_roots'}

_evalfuns = ['eval_legendre', 'eval_chebyt', 'eval_chebyu',
             'eval_chebyc', 'eval_chebys', 'eval_jacobi',
             'eval_laguerre', 'eval_genlaguerre', 'eval_hermite',
             'eval_hermitenorm', 'eval_gegenbauer',
             'eval_sh_legendre', 'eval_sh_chebyt', 'eval_sh_chebyu',
             'eval_sh_jacobi']

__all__ = _polyfuns + list(_rootfuns_map.keys()) + _evalfuns + ['poch', 'binom']


class orthopoly1d(np.poly1d):

    def __init__(self, roots, weights=None, hn=1.0, kn=1.0, wfunc=None,
                 limits=None, monic=False, eval_func=None):
        equiv_weights = [weights[k] / wfunc(roots[k]) for
                         k in range(len(roots))]
        mu = sqrt(hn)
        if monic:
            evf = eval_func
            if evf:
                knn = kn
                eval_func = lambda x: evf(x) / knn
            mu = mu / abs(kn)
            kn = 1.0

        # compute coefficients from roots, then scale
        poly = np.poly1d(roots, r=True)
        np.poly1d.__init__(self, poly.coeffs * float(kn))

        self.weights = np.array(list(zip(roots, weights, equiv_weights)))
        self.weight_func = wfunc
        self.limits = limits
        self.normcoef = mu

        # Note: eval_func will be discarded on arithmetic
        self._eval_func = eval_func

    def __call__(self, v):
        if self._eval_func and not isinstance(v, np.poly1d):
            return self._eval_func(v)
        else:
            return np.poly1d.__call__(self, v)

    def _scale(self, p):
        if p == 1.0:
            return
        self._coeffs *= p

        evf = self._eval_func
        if evf:
            self._eval_func = lambda x: evf(x) * p
        self.normcoef *= p


def _gen_roots_and_weights(n, mu0, an_func, bn_func, f, df, symmetrize, mu):
    """[x,w] = gen_roots_and_weights(n,an_func,sqrt_bn_func,mu)

    Returns the roots (x) of an nth order orthogonal polynomial,
    and weights (w) to use in appropriate Gaussian quadrature with that
    orthogonal polynomial.

    The polynomials have the recurrence relation
          P_n+1(x) = (x - A_n) P_n(x) - B_n P_n-1(x)

    an_func(n)          should return A_n
    sqrt_bn_func(n)     should return sqrt(B_n)
    mu ( = h_0 )        is the integral of the weight over the orthogonal
                        interval
    """
    k = np.arange(n, dtype='d')
    c = np.zeros((2, n))
    c[0,1:] = bn_func(k[1:])
    c[1,:] = an_func(k)
    x = linalg.eigvals_banded(c, overwrite_a_band=True)

    # improve roots by one application of Newton's method
    y = f(n, x)
    dy = df(n, x)
    x -= y/dy

    fm = f(n-1, x)
    fm /= np.abs(fm).max()
    dy /= np.abs(dy).max()
    w = 1.0 / (fm * dy)

    if symmetrize:
        w = (w + w[::-1]) / 2
        x = (x - x[::-1]) / 2

    w *= mu0 / w.sum()

    if mu:
        return x, w, mu0
    else:
        return x, w

# Jacobi Polynomials 1               P^(alpha,beta)_n(x)


def roots_jacobi(n, alpha, beta, mu=False):
    r"""Gauss-Jacobi quadrature.

    Compute the sample points and weights for Gauss-Jacobi
    quadrature. The sample points are the roots of the nth degree
    Jacobi polynomial, :math:`P^{\alpha, \beta}_n(x)`. These sample
    points and weights correctly integrate polynomials of degree
    :math:`2n - 1` or less over the interval :math:`[-1, 1]` with
    weight function :math:`w(x) = (1 - x)^{\alpha} (1 +
    x)^{\beta}`. See 22.2.1 in [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    alpha : float
        alpha must be > -1
    beta : float
        beta must be > -1
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.quadrature
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError("n must be a positive integer.")
    if alpha <= -1 or beta <= -1:
        raise ValueError("alpha and beta must be greater than -1.")

    if alpha == 0.0 and beta == 0.0:
        return roots_legendre(m, mu)
    if alpha == beta:
        return roots_gegenbauer(m, alpha+0.5, mu)

    mu0 = 2.0**(alpha+beta+1)*cephes.beta(alpha+1, beta+1)
    a = alpha
    b = beta
    if a + b == 0.0:
        an_func = lambda k: np.where(k == 0, (b-a)/(2+a+b), 0.0)
    else:
        an_func = lambda k: np.where(k == 0, (b-a)/(2+a+b),
                  (b*b - a*a) / ((2.0*k+a+b)*(2.0*k+a+b+2)))

    bn_func = lambda k: 2.0 / (2.0*k+a+b)*np.sqrt((k+a)*(k+b) / (2*k+a+b+1)) \
              * np.where(k == 1, 1.0, np.sqrt(k*(k+a+b) / (2.0*k+a+b-1)))

    f = lambda n, x: cephes.eval_jacobi(n, a, b, x)
    df = lambda n, x: 0.5 * (n + a + b + 1) \
                      * cephes.eval_jacobi(n-1, a+1, b+1, x)
    return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, False, mu)


def jacobi(n, alpha, beta, monic=False):
    r"""Jacobi polynomial.

    Defined to be the solution of

    .. math::
        (1 - x^2)\frac{d^2}{dx^2}P_n^{(\alpha, \beta)}
          + (\beta - \alpha - (\alpha + \beta + 2)x)
            \frac{d}{dx}P_n^{(\alpha, \beta)}
          + n(n + \alpha + \beta + 1)P_n^{(\alpha, \beta)} = 0

    for :math:`\alpha, \beta > -1`; :math:`P_n^{(\alpha, \beta)}` is a
    polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    alpha : float
        Parameter, must be greater than -1.
    beta : float
        Parameter, must be greater than -1.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    P : orthopoly1d
        Jacobi polynomial.

    Notes
    -----
    For fixed :math:`\alpha, \beta`, the polynomials
    :math:`P_n^{(\alpha, \beta)}` are orthogonal over :math:`[-1, 1]`
    with weight function :math:`(1 - x)^\alpha(1 + x)^\beta`.

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    wfunc = lambda x: (1 - x)**alpha * (1 + x)**beta
    if n == 0:
        return orthopoly1d([], [], 1.0, 1.0, wfunc, (-1, 1), monic,
                           eval_func=np.ones_like)
    x, w, mu = roots_jacobi(n, alpha, beta, mu=True)
    ab1 = alpha + beta + 1.0
    hn = 2**ab1 / (2 * n + ab1) * _gam(n + alpha + 1)
    hn *= _gam(n + beta + 1.0) / _gam(n + 1) / _gam(n + ab1)
    kn = _gam(2 * n + ab1) / 2.0**n / _gam(n + 1) / _gam(n + ab1)
    # here kn = coefficient on x^n term
    p = orthopoly1d(x, w, hn, kn, wfunc, (-1, 1), monic,
                    lambda x: eval_jacobi(n, alpha, beta, x))
    return p

# Jacobi Polynomials shifted         G_n(p,q,x)


def roots_sh_jacobi(n, p1, q1, mu=False):
    """Gauss-Jacobi (shifted) quadrature.

    Compute the sample points and weights for Gauss-Jacobi (shifted)
    quadrature. The sample points are the roots of the nth degree
    shifted Jacobi polynomial, :math:`G^{p,q}_n(x)`. These sample
    points and weights correctly integrate polynomials of degree
    :math:`2n - 1` or less over the interval :math:`[0, 1]` with
    weight function :math:`w(x) = (1 - x)^{p-q} x^{q-1}`. See 22.2.2
    in [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    p1 : float
        (p1 - q1) must be > -1
    q1 : float
        q1 must be > 0
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.quadrature
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    if (p1-q1) <= -1 or q1 <= 0:
        raise ValueError("(p - q) must be greater than -1, and q must be greater than 0.")
    x, w, m = roots_jacobi(n, p1-q1, q1-1, True)
    x = (x + 1) / 2
    scale = 2.0**p1
    w /= scale
    m /= scale
    if mu:
        return x, w, m
    else:
        return x, w

def sh_jacobi(n, p, q, monic=False):
    r"""Shifted Jacobi polynomial.

    Defined by

    .. math::

        G_n^{(p, q)}(x)
          = \binom{2n + p - 1}{n}^{-1}P_n^{(p - q, q - 1)}(2x - 1),

    where :math:`P_n^{(\cdot, \cdot)}` is the nth Jacobi polynomial.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    p : float
        Parameter, must have :math:`p > q - 1`.
    q : float
        Parameter, must be greater than 0.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    G : orthopoly1d
        Shifted Jacobi polynomial.

    Notes
    -----
    For fixed :math:`p, q`, the polynomials :math:`G_n^{(p, q)}` are
    orthogonal over :math:`[0, 1]` with weight function :math:`(1 -
    x)^{p - q}x^{q - 1}`.

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    wfunc = lambda x: (1.0 - x)**(p - q) * (x)**(q - 1.)
    if n == 0:
        return orthopoly1d([], [], 1.0, 1.0, wfunc, (-1, 1), monic,
                           eval_func=np.ones_like)
    n1 = n
    x, w = roots_sh_jacobi(n1, p, q)
    hn = _gam(n + 1) * _gam(n + q) * _gam(n + p) * _gam(n + p - q + 1)
    hn /= (2 * n + p) * (_gam(2 * n + p)**2)
    # kn = 1.0 in standard form so monic is redundant. Kept for compatibility.
    kn = 1.0
    pp = orthopoly1d(x, w, hn, kn, wfunc=wfunc, limits=(0, 1), monic=monic,
                     eval_func=lambda x: eval_sh_jacobi(n, p, q, x))
    return pp

# Generalized Laguerre               L^(alpha)_n(x)


def roots_genlaguerre(n, alpha, mu=False):
    r"""Gauss-generalized Laguerre quadrature.

    Compute the sample points and weights for Gauss-generalized
    Laguerre quadrature. The sample points are the roots of the nth
    degree generalized Laguerre polynomial, :math:`L^{\alpha}_n(x)`.
    These sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[0,
    \infty]` with weight function :math:`w(x) = x^{\alpha}
    e^{-x}`. See 22.3.9 in [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    alpha : float
        alpha must be > -1
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.quadrature
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError("n must be a positive integer.")
    if alpha < -1:
        raise ValueError("alpha must be greater than -1.")

    mu0 = cephes.gamma(alpha + 1)

    if m == 1:
        x = np.array([alpha+1.0], 'd')
        w = np.array([mu0], 'd')
        if mu:
            return x, w, mu0
        else:
            return x, w

    an_func = lambda k: 2 * k + alpha + 1
    bn_func = lambda k: -np.sqrt(k * (k + alpha))
    f = lambda n, x: cephes.eval_genlaguerre(n, alpha, x)
    df = lambda n, x: (n*cephes.eval_genlaguerre(n, alpha, x)
                     - (n + alpha)*cephes.eval_genlaguerre(n-1, alpha, x))/x
    return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, False, mu)


def genlaguerre(n, alpha, monic=False):
    r"""Generalized (associated) Laguerre polynomial.

    Defined to be the solution of

    .. math::
        x\frac{d^2}{dx^2}L_n^{(\alpha)}
          + (\alpha + 1 - x)\frac{d}{dx}L_n^{(\alpha)}
          + nL_n^{(\alpha)} = 0,

    where :math:`\alpha > -1`; :math:`L_n^{(\alpha)}` is a polynomial
    of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    alpha : float
        Parameter, must be greater than -1.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    L : orthopoly1d
        Generalized Laguerre polynomial.

    Notes
    -----
    For fixed :math:`\alpha`, the polynomials :math:`L_n^{(\alpha)}`
    are orthogonal over :math:`[0, \infty)` with weight function
    :math:`e^{-x}x^\alpha`.

    The Laguerre polynomials are the special case where :math:`\alpha
    = 0`.

    See Also
    --------
    laguerre : Laguerre polynomial.

    """
    if alpha <= -1:
        raise ValueError("alpha must be > -1")
    if n < 0:
        raise ValueError("n must be nonnegative.")

    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_genlaguerre(n1, alpha)
    wfunc = lambda x: exp(-x) * x**alpha
    if n == 0:
        x, w = [], []
    hn = _gam(n + alpha + 1) / _gam(n + 1)
    kn = (-1)**n / _gam(n + 1)
    p = orthopoly1d(x, w, hn, kn, wfunc, (0, inf), monic,
                    lambda x: eval_genlaguerre(n, alpha, x))
    return p

# Laguerre                      L_n(x)


def roots_laguerre(n, mu=False):
    r"""Gauss-Laguerre quadrature.

    Compute the sample points and weights for Gauss-Laguerre
    quadrature. The sample points are the roots of the nth degree
    Laguerre polynomial, :math:`L_n(x)`. These sample points and
    weights correctly integrate polynomials of degree :math:`2n - 1`
    or less over the interval :math:`[0, \infty]` with weight function
    :math:`w(x) = e^{-x}`. See 22.2.13 in [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.quadrature
    scipy.integrate.fixed_quad
    numpy.polynomial.laguerre.laggauss

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    return roots_genlaguerre(n, 0.0, mu=mu)


def laguerre(n, monic=False):
    r"""Laguerre polynomial.

    Defined to be the solution of

    .. math::
        x\frac{d^2}{dx^2}L_n + (1 - x)\frac{d}{dx}L_n + nL_n = 0;

    :math:`L_n` is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    L : orthopoly1d
        Laguerre Polynomial.

    Notes
    -----
    The polynomials :math:`L_n` are orthogonal over :math:`[0,
    \infty)` with weight function :math:`e^{-x}`.

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_laguerre(n1)
    if n == 0:
        x, w = [], []
    hn = 1.0
    kn = (-1)**n / _gam(n + 1)
    p = orthopoly1d(x, w, hn, kn, lambda x: exp(-x), (0, inf), monic,
                    lambda x: eval_laguerre(n, x))
    return p

# Hermite  1                         H_n(x)


def roots_hermite(n, mu=False):
    r"""Gauss-Hermite (physicist's) quadrature.

    Compute the sample points and weights for Gauss-Hermite
    quadrature. The sample points are the roots of the nth degree
    Hermite polynomial, :math:`H_n(x)`. These sample points and
    weights correctly integrate polynomials of degree :math:`2n - 1`
    or less over the interval :math:`[-\infty, \infty]` with weight
    function :math:`w(x) = e^{-x^2}`. See 22.2.14 in [AS]_ for
    details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    Notes
    -----
    For small n up to 150 a modified version of the Golub-Welsch
    algorithm is used. Nodes are computed from the eigenvalue
    problem and improved by one step of a Newton iteration.
    The weights are computed from the well-known analytical formula.

    For n larger than 150 an optimal asymptotic algorithm is applied
    which computes nodes and weights in a numerically stable manner.
    The algorithm has linear runtime making computation for very
    large n (several thousand or more) feasible.

    See Also
    --------
    scipy.integrate.quadrature
    scipy.integrate.fixed_quad
    numpy.polynomial.hermite.hermgauss
    roots_hermitenorm

    References
    ----------
    .. [townsend.trogdon.olver-2014]
        Townsend, A. and Trogdon, T. and Olver, S. (2014)
        *Fast computation of Gauss quadrature nodes and
        weights on the whole real line*. :arXiv:`1410.5286`.
    .. [townsend.trogdon.olver-2015]
        Townsend, A. and Trogdon, T. and Olver, S. (2015)
        *Fast computation of Gauss quadrature nodes and
        weights on the whole real line*.
        IMA Journal of Numerical Analysis
        :doi:`10.1093/imanum/drv002`.
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError("n must be a positive integer.")

    mu0 = np.sqrt(np.pi)
    if n <= 150:
        an_func = lambda k: 0.0*k
        bn_func = lambda k: np.sqrt(k/2.0)
        f = cephes.eval_hermite
        df = lambda n, x: 2.0 * n * cephes.eval_hermite(n-1, x)
        return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, True, mu)
    else:
        nodes, weights = _roots_hermite_asy(m)
        if mu:
            return nodes, weights, mu0
        else:
            return nodes, weights


def _compute_tauk(n, k, maxit=5):
    """Helper function for Tricomi initial guesses

    For details, see formula 3.1 in lemma 3.1 in the
    original paper.

    Parameters
    ----------
    n : int
        Quadrature order
    k : ndarray of type int
        Index of roots :math:`\tau_k` to compute
    maxit : int
        Number of Newton maxit performed, the default
        value of 5 is sufficient.

    Returns
    -------
    tauk : ndarray
        Roots of equation 3.1

    See Also
    --------
    initial_nodes_a
    roots_hermite_asy
    """
    a = n % 2 - 0.5
    c = (4.0*floor(n/2.0) - 4.0*k + 3.0)*pi / (4.0*floor(n/2.0) + 2.0*a + 2.0)
    f = lambda x: x - sin(x) - c
    df = lambda x: 1.0 - cos(x)
    xi = 0.5*pi
    for i in range(maxit):
        xi = xi - f(xi)/df(xi)
    return xi


def _initial_nodes_a(n, k):
    r"""Tricomi initial guesses

    Computes an initial approximation to the square of the `k`-th
    (positive) root :math:`x_k` of the Hermite polynomial :math:`H_n`
    of order :math:`n`. The formula is the one from lemma 3.1 in the
    original paper. The guesses are accurate except in the region
    near :math:`\sqrt{2n + 1}`.

    Parameters
    ----------
    n : int
        Quadrature order
    k : ndarray of type int
        Index of roots to compute

    Returns
    -------
    xksq : ndarray
        Square of the approximate roots

    See Also
    --------
    initial_nodes
    roots_hermite_asy
    """
    tauk = _compute_tauk(n, k)
    sigk = cos(0.5*tauk)**2
    a = n % 2 - 0.5
    nu = 4.0*floor(n/2.0) + 2.0*a + 2.0
    # Initial approximation of Hermite roots (square)
    xksq = nu*sigk - 1.0/(3.0*nu) * (5.0/(4.0*(1.0-sigk)**2) - 1.0/(1.0-sigk) - 0.25)
    return xksq


def _initial_nodes_b(n, k):
    r"""Gatteschi initial guesses

    Computes an initial approximation to the square of the kth
    (positive) root :math:`x_k` of the Hermite polynomial :math:`H_n`
    of order :math:`n`. The formula is the one from lemma 3.2 in the
    original paper. The guesses are accurate in the region just
    below :math:`\sqrt{2n + 1}`.

    Parameters
    ----------
    n : int
        Quadrature order
    k : ndarray of type int
        Index of roots to compute

    Returns
    -------
    xksq : ndarray
        Square of the approximate root

    See Also
    --------
    initial_nodes
    roots_hermite_asy
    """
    a = n % 2 - 0.5
    nu = 4.0*floor(n/2.0) + 2.0*a + 2.0
    # Airy roots by approximation
    ak = specfun.airyzo(k.max(), 1)[0][::-1]
    # Initial approximation of Hermite roots (square)
    xksq = (nu +
            2.0**(2.0/3.0) * ak * nu**(1.0/3.0) +
            1.0/5.0 * 2.0**(4.0/3.0) * ak**2 * nu**(-1.0/3.0) +
            (9.0/140.0 - 12.0/175.0 * ak**3) * nu**(-1.0) +
            (16.0/1575.0 * ak + 92.0/7875.0 * ak**4) * 2.0**(2.0/3.0) * nu**(-5.0/3.0) -
            (15152.0/3031875.0 * ak**5 + 1088.0/121275.0 * ak**2) * 2.0**(1.0/3.0) * nu**(-7.0/3.0))
    return xksq


def _initial_nodes(n):
    """Initial guesses for the Hermite roots

    Computes an initial approximation to the non-negative
    roots :math:`x_k` of the Hermite polynomial :math:`H_n`
    of order :math:`n`. The Tricomi and Gatteschi initial
    guesses are used in the region where they are accurate.

    Parameters
    ----------
    n : int
        Quadrature order

    Returns
    -------
    xk : ndarray
        Approximate roots

    See Also
    --------
    roots_hermite_asy
    """
    # Turnover point
    # linear polynomial fit to error of 10, 25, 40, ..., 1000 point rules
    fit = 0.49082003*n - 4.37859653
    turnover = around(fit).astype(int)
    # Compute all approximations
    ia = arange(1, int(floor(n*0.5)+1))
    ib = ia[::-1]
    xasq = _initial_nodes_a(n, ia[:turnover+1])
    xbsq = _initial_nodes_b(n, ib[turnover+1:])
    # Combine
    iv = sqrt(hstack([xasq, xbsq]))
    # Central node is always zero
    if n % 2 == 1:
        iv = hstack([0.0, iv])
    return iv


def _pbcf(n, theta):
    r"""Asymptotic series expansion of parabolic cylinder function

    The implementation is based on sections 3.2 and 3.3 from the
    original paper. Compared to the published version this code
    adds one more term to the asymptotic series. The detailed
    formulas can be found at [parabolic-asymptotics]_. The evaluation
    is done in a transformed variable :math:`\theta := \arccos(t)`
    where :math:`t := x / \mu` and :math:`\mu := \sqrt{2n + 1}`.

    Parameters
    ----------
    n : int
        Quadrature order
    theta : ndarray
        Transformed position variable

    Returns
    -------
    U : ndarray
        Value of the parabolic cylinder function :math:`U(a, \theta)`.
    Ud : ndarray
        Value of the derivative :math:`U^{\prime}(a, \theta)` of
        the parabolic cylinder function.

    See Also
    --------
    roots_hermite_asy

    References
    ----------
    .. [parabolic-asymptotics]
       https://dlmf.nist.gov/12.10#vii
    """
    st = sin(theta)
    ct = cos(theta)
    # https://dlmf.nist.gov/12.10#vii
    mu = 2.0*n + 1.0
    # https://dlmf.nist.gov/12.10#E23
    eta = 0.5*theta - 0.5*st*ct
    # https://dlmf.nist.gov/12.10#E39
    zeta = -(3.0*eta/2.0) ** (2.0/3.0)
    # https://dlmf.nist.gov/12.10#E40
    phi = (-zeta / st**2) ** (0.25)
    # Coefficients
    # https://dlmf.nist.gov/12.10#E43
    a0 = 1.0
    a1 = 0.10416666666666666667
    a2 = 0.08355034722222222222
    a3 = 0.12822657455632716049
    a4 = 0.29184902646414046425
    a5 = 0.88162726744375765242
    b0 = 1.0
    b1 = -0.14583333333333333333
    b2 = -0.09874131944444444444
    b3 = -0.14331205391589506173
    b4 = -0.31722720267841354810
    b5 = -0.94242914795712024914
    # Polynomials
    # https://dlmf.nist.gov/12.10#E9
    # https://dlmf.nist.gov/12.10#E10
    ctp = ct ** arange(16).reshape((-1,1))
    u0 = 1.0
    u1 = (1.0*ctp[3,:] - 6.0*ct) / 24.0
    u2 = (-9.0*ctp[4,:] + 249.0*ctp[2,:] + 145.0) / 1152.0
    u3 = (-4042.0*ctp[9,:] + 18189.0*ctp[7,:] - 28287.0*ctp[5,:] - 151995.0*ctp[3,:] - 259290.0*ct) / 414720.0
    u4 = (72756.0*ctp[10,:] - 321339.0*ctp[8,:] - 154982.0*ctp[6,:] + 50938215.0*ctp[4,:] + 122602962.0*ctp[2,:] + 12773113.0) / 39813120.0
    u5 = (82393456.0*ctp[15,:] - 617950920.0*ctp[13,:] + 1994971575.0*ctp[11,:] - 3630137104.0*ctp[9,:] + 4433574213.0*ctp[7,:]
          - 37370295816.0*ctp[5,:] - 119582875013.0*ctp[3,:] - 34009066266.0*ct) / 6688604160.0
    v0 = 1.0
    v1 = (1.0*ctp[3,:] + 6.0*ct) / 24.0
    v2 = (15.0*ctp[4,:] - 327.0*ctp[2,:] - 143.0) / 1152.0
    v3 = (-4042.0*ctp[9,:] + 18189.0*ctp[7,:] - 36387.0*ctp[5,:] + 238425.0*ctp[3,:] + 259290.0*ct) / 414720.0
    v4 = (-121260.0*ctp[10,:] + 551733.0*ctp[8,:] - 151958.0*ctp[6,:] - 57484425.0*ctp[4,:] - 132752238.0*ctp[2,:] - 12118727) / 39813120.0
    v5 = (82393456.0*ctp[15,:] - 617950920.0*ctp[13,:] + 2025529095.0*ctp[11,:] - 3750839308.0*ctp[9,:] + 3832454253.0*ctp[7,:]
          + 35213253348.0*ctp[5,:] + 130919230435.0*ctp[3,:] + 34009066266*ct) / 6688604160.0
    # Airy Evaluation (Bi and Bip unused)
    Ai, Aip, Bi, Bip = airy(mu**(4.0/6.0) * zeta)
    # Prefactor for U
    P = 2.0*sqrt(pi) * mu**(1.0/6.0) * phi
    # Terms for U
    # https://dlmf.nist.gov/12.10#E42
    phip = phi ** arange(6, 31, 6).reshape((-1,1))
    A0 = b0*u0
    A1 = (b2*u0 + phip[0,:]*b1*u1 + phip[1,:]*b0*u2) / zeta**3
    A2 = (b4*u0 + phip[0,:]*b3*u1 + phip[1,:]*b2*u2 + phip[2,:]*b1*u3 + phip[3,:]*b0*u4) / zeta**6
    B0 = -(a1*u0 + phip[0,:]*a0*u1) / zeta**2
    B1 = -(a3*u0 + phip[0,:]*a2*u1 + phip[1,:]*a1*u2 + phip[2,:]*a0*u3) / zeta**5
    B2 = -(a5*u0 + phip[0,:]*a4*u1 + phip[1,:]*a3*u2 + phip[2,:]*a2*u3 + phip[3,:]*a1*u4 + phip[4,:]*a0*u5) / zeta**8
    # U
    # https://dlmf.nist.gov/12.10#E35
    U = P * (Ai * (A0 + A1/mu**2.0 + A2/mu**4.0) +
             Aip * (B0 + B1/mu**2.0 + B2/mu**4.0) / mu**(8.0/6.0))
    # Prefactor for derivative of U
    Pd = sqrt(2.0*pi) * mu**(2.0/6.0) / phi
    # Terms for derivative of U
    # https://dlmf.nist.gov/12.10#E46
    C0 = -(b1*v0 + phip[0,:]*b0*v1) / zeta
    C1 = -(b3*v0 + phip[0,:]*b2*v1 + phip[1,:]*b1*v2 + phip[2,:]*b0*v3) / zeta**4
    C2 = -(b5*v0 + phip[0,:]*b4*v1 + phip[1,:]*b3*v2 + phip[2,:]*b2*v3 + phip[3,:]*b1*v4 + phip[4,:]*b0*v5) / zeta**7
    D0 = a0*v0
    D1 = (a2*v0 + phip[0,:]*a1*v1 + phip[1,:]*a0*v2) / zeta**3
    D2 = (a4*v0 + phip[0,:]*a3*v1 + phip[1,:]*a2*v2 + phip[2,:]*a1*v3 + phip[3,:]*a0*v4) / zeta**6
    # Derivative of U
    # https://dlmf.nist.gov/12.10#E36
    Ud = Pd * (Ai * (C0 + C1/mu**2.0 + C2/mu**4.0) / mu**(4.0/6.0) +
               Aip * (D0 + D1/mu**2.0 + D2/mu**4.0))
    return U, Ud


def _newton(n, x_initial, maxit=5):
    """Newton iteration for polishing the asymptotic approximation
    to the zeros of the Hermite polynomials.

    Parameters
    ----------
    n : int
        Quadrature order
    x_initial : ndarray
        Initial guesses for the roots
    maxit : int
        Maximal number of Newton iterations.
        The default 5 is sufficient, usually
        only one or two steps are needed.

    Returns
    -------
    nodes : ndarray
        Quadrature nodes
    weights : ndarray
        Quadrature weights

    See Also
    --------
    roots_hermite_asy
    """
    # Variable transformation
    mu = sqrt(2.0*n + 1.0)
    t = x_initial / mu
    theta = arccos(t)
    # Newton iteration
    for i in range(maxit):
        u, ud = _pbcf(n, theta)
        dtheta = u / (sqrt(2.0) * mu * sin(theta) * ud)
        theta = theta + dtheta
        if max(abs(dtheta)) < 1e-14:
            break
    # Undo variable transformation
    x = mu * cos(theta)
    # Central node is always zero
    if n % 2 == 1:
        x[0] = 0.0
    # Compute weights
    w = exp(-x**2) / (2.0*ud**2)
    return x, w


def _roots_hermite_asy(n):
    r"""Gauss-Hermite (physicist's) quadrature for large n.

    Computes the sample points and weights for Gauss-Hermite quadrature.
    The sample points are the roots of the nth degree Hermite polynomial,
    :math:`H_n(x)`. These sample points and weights correctly integrate
    polynomials of degree :math:`2n - 1` or less over the interval
    :math:`[-\infty, \infty]` with weight function :math:`f(x) = e^{-x^2}`.

    This method relies on asymptotic expansions which work best for n > 150.
    The algorithm has linear runtime making computation for very large n
    feasible.

    Parameters
    ----------
    n : int
        quadrature order

    Returns
    -------
    nodes : ndarray
        Quadrature nodes
    weights : ndarray
        Quadrature weights

    See Also
    --------
    roots_hermite

    References
    ----------
    .. [townsend.trogdon.olver-2014]
       Townsend, A. and Trogdon, T. and Olver, S. (2014)
       *Fast computation of Gauss quadrature nodes and
       weights on the whole real line*. :arXiv:`1410.5286`.

    .. [townsend.trogdon.olver-2015]
       Townsend, A. and Trogdon, T. and Olver, S. (2015)
       *Fast computation of Gauss quadrature nodes and
       weights on the whole real line*.
       IMA Journal of Numerical Analysis
       :doi:`10.1093/imanum/drv002`.
    """
    iv = _initial_nodes(n)
    nodes, weights = _newton(n, iv)
    # Combine with negative parts
    if n % 2 == 0:
        nodes = hstack([-nodes[::-1], nodes])
        weights = hstack([weights[::-1], weights])
    else:
        nodes = hstack([-nodes[-1:0:-1], nodes])
        weights = hstack([weights[-1:0:-1], weights])
    # Scale weights
    weights *= sqrt(pi) / sum(weights)
    return nodes, weights


def hermite(n, monic=False):
    r"""Physicist's Hermite polynomial.

    Defined by

    .. math::

        H_n(x) = (-1)^ne^{x^2}\frac{d^n}{dx^n}e^{-x^2};

    :math:`H_n` is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    H : orthopoly1d
        Hermite polynomial.

    Notes
    -----
    The polynomials :math:`H_n` are orthogonal over :math:`(-\infty,
    \infty)` with weight function :math:`e^{-x^2}`.

    Examples
    --------
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> p_monic = special.hermite(3, monic=True)
    >>> p_monic
    poly1d([ 1. ,  0. , -1.5,  0. ])
    >>> p_monic(1)
    -0.49999999999999983
    >>> x = np.linspace(-3, 3, 400)
    >>> y = p_monic(x)
    >>> plt.plot(x, y)
    >>> plt.title("Monic Hermite polynomial of degree 3")
    >>> plt.xlabel("x")
    >>> plt.ylabel("H_3(x)")
    >>> plt.show()

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_hermite(n1)
    wfunc = lambda x: exp(-x * x)
    if n == 0:
        x, w = [], []
    hn = 2**n * _gam(n + 1) * sqrt(pi)
    kn = 2**n
    p = orthopoly1d(x, w, hn, kn, wfunc, (-inf, inf), monic,
                    lambda x: eval_hermite(n, x))
    return p

# Hermite  2                         He_n(x)


def roots_hermitenorm(n, mu=False):
    r"""Gauss-Hermite (statistician's) quadrature.

    Compute the sample points and weights for Gauss-Hermite
    quadrature. The sample points are the roots of the nth degree
    Hermite polynomial, :math:`He_n(x)`. These sample points and
    weights correctly integrate polynomials of degree :math:`2n - 1`
    or less over the interval :math:`[-\infty, \infty]` with weight
    function :math:`w(x) = e^{-x^2/2}`. See 22.2.15 in [AS]_ for more
    details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    Notes
    -----
    For small n up to 150 a modified version of the Golub-Welsch
    algorithm is used. Nodes are computed from the eigenvalue
    problem and improved by one step of a Newton iteration.
    The weights are computed from the well-known analytical formula.

    For n larger than 150 an optimal asymptotic algorithm is used
    which computes nodes and weights in a numerical stable manner.
    The algorithm has linear runtime making computation for very
    large n (several thousand or more) feasible.

    See Also
    --------
    scipy.integrate.quadrature
    scipy.integrate.fixed_quad
    numpy.polynomial.hermite_e.hermegauss

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError("n must be a positive integer.")

    mu0 = np.sqrt(2.0*np.pi)
    if n <= 150:
        an_func = lambda k: 0.0*k
        bn_func = lambda k: np.sqrt(k)
        f = cephes.eval_hermitenorm
        df = lambda n, x: n * cephes.eval_hermitenorm(n-1, x)
        return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, True, mu)
    else:
        nodes, weights = _roots_hermite_asy(m)
        # Transform
        nodes *= sqrt(2)
        weights *= sqrt(2)
        if mu:
            return nodes, weights, mu0
        else:
            return nodes, weights


def hermitenorm(n, monic=False):
    r"""Normalized (probabilist's) Hermite polynomial.

    Defined by

    .. math::

        He_n(x) = (-1)^ne^{x^2/2}\frac{d^n}{dx^n}e^{-x^2/2};

    :math:`He_n` is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    He : orthopoly1d
        Hermite polynomial.

    Notes
    -----

    The polynomials :math:`He_n` are orthogonal over :math:`(-\infty,
    \infty)` with weight function :math:`e^{-x^2/2}`.

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_hermitenorm(n1)
    wfunc = lambda x: exp(-x * x / 2.0)
    if n == 0:
        x, w = [], []
    hn = sqrt(2 * pi) * _gam(n + 1)
    kn = 1.0
    p = orthopoly1d(x, w, hn, kn, wfunc=wfunc, limits=(-inf, inf), monic=monic,
                    eval_func=lambda x: eval_hermitenorm(n, x))
    return p

# The remainder of the polynomials can be derived from the ones above.

# Ultraspherical (Gegenbauer)        C^(alpha)_n(x)


def roots_gegenbauer(n, alpha, mu=False):
    r"""Gauss-Gegenbauer quadrature.

    Compute the sample points and weights for Gauss-Gegenbauer
    quadrature. The sample points are the roots of the nth degree
    Gegenbauer polynomial, :math:`C^{\alpha}_n(x)`. These sample
    points and weights correctly integrate polynomials of degree
    :math:`2n - 1` or less over the interval :math:`[-1, 1]` with
    weight function :math:`w(x) = (1 - x^2)^{\alpha - 1/2}`. See
    22.2.3 in [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    alpha : float
        alpha must be > -0.5
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.quadrature
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError("n must be a positive integer.")
    if alpha < -0.5:
        raise ValueError("alpha must be greater than -0.5.")
    elif alpha == 0.0:
        # C(n,0,x) == 0 uniformly, however, as alpha->0, C(n,alpha,x)->T(n,x)
        # strictly, we should just error out here, since the roots are not
        # really defined, but we used to return something useful, so let's
        # keep doing so.
        return roots_chebyt(n, mu)

    mu0 = np.sqrt(np.pi) * cephes.gamma(alpha + 0.5) / cephes.gamma(alpha + 1)
    an_func = lambda k: 0.0 * k
    bn_func = lambda k: np.sqrt(k * (k + 2 * alpha - 1)
                        / (4 * (k + alpha) * (k + alpha - 1)))
    f = lambda n, x: cephes.eval_gegenbauer(n, alpha, x)
    df = lambda n, x: (-n*x*cephes.eval_gegenbauer(n, alpha, x)
         + (n + 2*alpha - 1)*cephes.eval_gegenbauer(n-1, alpha, x))/(1-x**2)
    return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, True, mu)


def gegenbauer(n, alpha, monic=False):
    r"""Gegenbauer (ultraspherical) polynomial.

    Defined to be the solution of

    .. math::
        (1 - x^2)\frac{d^2}{dx^2}C_n^{(\alpha)}
          - (2\alpha + 1)x\frac{d}{dx}C_n^{(\alpha)}
          + n(n + 2\alpha)C_n^{(\alpha)} = 0

    for :math:`\alpha > -1/2`; :math:`C_n^{(\alpha)}` is a polynomial
    of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    alpha : float
        Parameter, must be greater than -0.5.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    C : orthopoly1d
        Gegenbauer polynomial.

    Notes
    -----
    The polynomials :math:`C_n^{(\alpha)}` are orthogonal over
    :math:`[-1,1]` with weight function :math:`(1 - x^2)^{(\alpha -
    1/2)}`.

    Examples
    --------
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt

    We can initialize a variable ``p`` as a Gegenbauer polynomial using the
    `gegenbauer` function and evaluate at a point ``x = 1``.

    >>> p = special.gegenbauer(3, 0.5, monic=False)
    >>> p
    poly1d([ 2.5,  0. , -1.5,  0. ])
    >>> p(1)
    1.0

    To evaluate ``p`` at various points ``x`` in the interval ``(-3, 3)``,
    simply pass an array ``x`` to ``p`` as follows:

    >>> x = np.linspace(-3, 3, 400)
    >>> y = p(x)

    We can then visualize ``x, y`` using `matplotlib.pyplot`.

    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, y)
    >>> ax.set_title("Gegenbauer (ultraspherical) polynomial of degree 3")
    >>> ax.set_xlabel("x")
    >>> ax.set_ylabel("G_3(x)")
    >>> plt.show()

    """
    base = jacobi(n, alpha - 0.5, alpha - 0.5, monic=monic)
    if monic:
        return base
    #  Abrahmowitz and Stegan 22.5.20
    factor = (_gam(2*alpha + n) * _gam(alpha + 0.5) /
              _gam(2*alpha) / _gam(alpha + 0.5 + n))
    base._scale(factor)
    base.__dict__['_eval_func'] = lambda x: eval_gegenbauer(float(n), alpha, x)
    return base

# Chebyshev of the first kind: T_n(x) =
#     n! sqrt(pi) / _gam(n+1./2)* P^(-1/2,-1/2)_n(x)
# Computed anew.


def roots_chebyt(n, mu=False):
    r"""Gauss-Chebyshev (first kind) quadrature.

    Computes the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    Chebyshev polynomial of the first kind, :math:`T_n(x)`. These
    sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[-1, 1]`
    with weight function :math:`w(x) = 1/\sqrt{1 - x^2}`. See 22.2.4
    in [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.quadrature
    scipy.integrate.fixed_quad
    numpy.polynomial.chebyshev.chebgauss

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError('n must be a positive integer.')
    x = _ufuncs._sinpi(np.arange(-m + 1, m, 2) / (2*m))
    w = np.full_like(x, pi/m)
    if mu:
        return x, w, pi
    else:
        return x, w


def chebyt(n, monic=False):
    r"""Chebyshev polynomial of the first kind.

    Defined to be the solution of

    .. math::
        (1 - x^2)\frac{d^2}{dx^2}T_n - x\frac{d}{dx}T_n + n^2T_n = 0;

    :math:`T_n` is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    T : orthopoly1d
        Chebyshev polynomial of the first kind.

    Notes
    -----
    The polynomials :math:`T_n` are orthogonal over :math:`[-1, 1]`
    with weight function :math:`(1 - x^2)^{-1/2}`.

    See Also
    --------
    chebyu : Chebyshev polynomial of the second kind.

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    wfunc = lambda x: 1.0 / sqrt(1 - x * x)
    if n == 0:
        return orthopoly1d([], [], pi, 1.0, wfunc, (-1, 1), monic,
                           lambda x: eval_chebyt(n, x))
    n1 = n
    x, w, mu = roots_chebyt(n1, mu=True)
    hn = pi / 2
    kn = 2**(n - 1)
    p = orthopoly1d(x, w, hn, kn, wfunc, (-1, 1), monic,
                    lambda x: eval_chebyt(n, x))
    return p

# Chebyshev of the second kind
#    U_n(x) = (n+1)! sqrt(pi) / (2*_gam(n+3./2)) * P^(1/2,1/2)_n(x)


def roots_chebyu(n, mu=False):
    r"""Gauss-Chebyshev (second kind) quadrature.

    Computes the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    Chebyshev polynomial of the second kind, :math:`U_n(x)`. These
    sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[-1, 1]`
    with weight function :math:`w(x) = \sqrt{1 - x^2}`. See 22.2.5 in
    [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.quadrature
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError('n must be a positive integer.')
    t = np.arange(m, 0, -1) * pi / (m + 1)
    x = np.cos(t)
    w = pi * np.sin(t)**2 / (m + 1)
    if mu:
        return x, w, pi / 2
    else:
        return x, w


def chebyu(n, monic=False):
    r"""Chebyshev polynomial of the second kind.

    Defined to be the solution of

    .. math::
        (1 - x^2)\frac{d^2}{dx^2}U_n - 3x\frac{d}{dx}U_n
          + n(n + 2)U_n = 0;

    :math:`U_n` is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    U : orthopoly1d
        Chebyshev polynomial of the second kind.

    Notes
    -----
    The polynomials :math:`U_n` are orthogonal over :math:`[-1, 1]`
    with weight function :math:`(1 - x^2)^{1/2}`.

    See Also
    --------
    chebyt : Chebyshev polynomial of the first kind.

    """
    base = jacobi(n, 0.5, 0.5, monic=monic)
    if monic:
        return base
    factor = sqrt(pi) / 2.0 * _gam(n + 2) / _gam(n + 1.5)
    base._scale(factor)
    return base

# Chebyshev of the first kind        C_n(x)


def roots_chebyc(n, mu=False):
    r"""Gauss-Chebyshev (first kind) quadrature.

    Compute the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    Chebyshev polynomial of the first kind, :math:`C_n(x)`. These
    sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[-2, 2]`
    with weight function :math:`w(x) = 1 / \sqrt{1 - (x/2)^2}`. See
    22.2.6 in [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.quadrature
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    x, w, m = roots_chebyt(n, True)
    x *= 2
    w *= 2
    m *= 2
    if mu:
        return x, w, m
    else:
        return x, w


def chebyc(n, monic=False):
    r"""Chebyshev polynomial of the first kind on :math:`[-2, 2]`.

    Defined as :math:`C_n(x) = 2T_n(x/2)`, where :math:`T_n` is the
    nth Chebychev polynomial of the first kind.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    C : orthopoly1d
        Chebyshev polynomial of the first kind on :math:`[-2, 2]`.

    Notes
    -----
    The polynomials :math:`C_n(x)` are orthogonal over :math:`[-2, 2]`
    with weight function :math:`1/\sqrt{1 - (x/2)^2}`.

    See Also
    --------
    chebyt : Chebyshev polynomial of the first kind.

    References
    ----------
    .. [1] Abramowitz and Stegun, "Handbook of Mathematical Functions"
           Section 22. National Bureau of Standards, 1972.

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_chebyc(n1)
    if n == 0:
        x, w = [], []
    hn = 4 * pi * ((n == 0) + 1)
    kn = 1.0
    p = orthopoly1d(x, w, hn, kn,
                    wfunc=lambda x: 1.0 / sqrt(1 - x * x / 4.0),
                    limits=(-2, 2), monic=monic)
    if not monic:
        p._scale(2.0 / p(2))
        p.__dict__['_eval_func'] = lambda x: eval_chebyc(n, x)
    return p

# Chebyshev of the second kind       S_n(x)


def roots_chebys(n, mu=False):
    r"""Gauss-Chebyshev (second kind) quadrature.

    Compute the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    Chebyshev polynomial of the second kind, :math:`S_n(x)`. These
    sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[-2, 2]`
    with weight function :math:`w(x) = \sqrt{1 - (x/2)^2}`. See 22.2.7
    in [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.quadrature
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    x, w, m = roots_chebyu(n, True)
    x *= 2
    w *= 2
    m *= 2
    if mu:
        return x, w, m
    else:
        return x, w


def chebys(n, monic=False):
    r"""Chebyshev polynomial of the second kind on :math:`[-2, 2]`.

    Defined as :math:`S_n(x) = U_n(x/2)` where :math:`U_n` is the
    nth Chebychev polynomial of the second kind.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    S : orthopoly1d
        Chebyshev polynomial of the second kind on :math:`[-2, 2]`.

    Notes
    -----
    The polynomials :math:`S_n(x)` are orthogonal over :math:`[-2, 2]`
    with weight function :math:`\sqrt{1 - (x/2)}^2`.

    See Also
    --------
    chebyu : Chebyshev polynomial of the second kind

    References
    ----------
    .. [1] Abramowitz and Stegun, "Handbook of Mathematical Functions"
           Section 22. National Bureau of Standards, 1972.

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_chebys(n1)
    if n == 0:
        x, w = [], []
    hn = pi
    kn = 1.0
    p = orthopoly1d(x, w, hn, kn,
                    wfunc=lambda x: sqrt(1 - x * x / 4.0),
                    limits=(-2, 2), monic=monic)
    if not monic:
        factor = (n + 1.0) / p(2)
        p._scale(factor)
        p.__dict__['_eval_func'] = lambda x: eval_chebys(n, x)
    return p

# Shifted Chebyshev of the first kind     T^*_n(x)


def roots_sh_chebyt(n, mu=False):
    r"""Gauss-Chebyshev (first kind, shifted) quadrature.

    Compute the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    shifted Chebyshev polynomial of the first kind, :math:`T_n(x)`.
    These sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[0, 1]`
    with weight function :math:`w(x) = 1/\sqrt{x - x^2}`. See 22.2.8
    in [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.quadrature
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    xw = roots_chebyt(n, mu)
    return ((xw[0] + 1) / 2,) + xw[1:]


def sh_chebyt(n, monic=False):
    r"""Shifted Chebyshev polynomial of the first kind.

    Defined as :math:`T^*_n(x) = T_n(2x - 1)` for :math:`T_n` the nth
    Chebyshev polynomial of the first kind.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    T : orthopoly1d
        Shifted Chebyshev polynomial of the first kind.

    Notes
    -----
    The polynomials :math:`T^*_n` are orthogonal over :math:`[0, 1]`
    with weight function :math:`(x - x^2)^{-1/2}`.

    """
    base = sh_jacobi(n, 0.0, 0.5, monic=monic)
    if monic:
        return base
    if n > 0:
        factor = 4**n / 2.0
    else:
        factor = 1.0
    base._scale(factor)
    return base


# Shifted Chebyshev of the second kind    U^*_n(x)
def roots_sh_chebyu(n, mu=False):
    r"""Gauss-Chebyshev (second kind, shifted) quadrature.

    Computes the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    shifted Chebyshev polynomial of the second kind, :math:`U_n(x)`.
    These sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[0, 1]`
    with weight function :math:`w(x) = \sqrt{x - x^2}`. See 22.2.9 in
    [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.quadrature
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    x, w, m = roots_chebyu(n, True)
    x = (x + 1) / 2
    m_us = cephes.beta(1.5, 1.5)
    w *= m_us / m
    if mu:
        return x, w, m_us
    else:
        return x, w


def sh_chebyu(n, monic=False):
    r"""Shifted Chebyshev polynomial of the second kind.

    Defined as :math:`U^*_n(x) = U_n(2x - 1)` for :math:`U_n` the nth
    Chebyshev polynomial of the second kind.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    U : orthopoly1d
        Shifted Chebyshev polynomial of the second kind.

    Notes
    -----
    The polynomials :math:`U^*_n` are orthogonal over :math:`[0, 1]`
    with weight function :math:`(x - x^2)^{1/2}`.

    """
    base = sh_jacobi(n, 2.0, 1.5, monic=monic)
    if monic:
        return base
    factor = 4**n
    base._scale(factor)
    return base

# Legendre


def roots_legendre(n, mu=False):
    r"""Gauss-Legendre quadrature.

    Compute the sample points and weights for Gauss-Legendre
    quadrature. The sample points are the roots of the nth degree
    Legendre polynomial :math:`P_n(x)`. These sample points and
    weights correctly integrate polynomials of degree :math:`2n - 1`
    or less over the interval :math:`[-1, 1]` with weight function
    :math:`w(x) = 1.0`. See 2.2.10 in [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.quadrature
    scipy.integrate.fixed_quad
    numpy.polynomial.legendre.leggauss

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError("n must be a positive integer.")

    mu0 = 2.0
    an_func = lambda k: 0.0 * k
    bn_func = lambda k: k * np.sqrt(1.0 / (4 * k * k - 1))
    f = cephes.eval_legendre
    df = lambda n, x: (-n*x*cephes.eval_legendre(n, x)
                     + n*cephes.eval_legendre(n-1, x))/(1-x**2)
    return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, True, mu)


def legendre(n, monic=False):
    r"""Legendre polynomial.

    Defined to be the solution of

    .. math::
        \frac{d}{dx}\left[(1 - x^2)\frac{d}{dx}P_n(x)\right]
          + n(n + 1)P_n(x) = 0;

    :math:`P_n(x)` is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    P : orthopoly1d
        Legendre polynomial.

    Notes
    -----
    The polynomials :math:`P_n` are orthogonal over :math:`[-1, 1]`
    with weight function 1.

    Examples
    --------
    Generate the 3rd-order Legendre polynomial 1/2*(5x^3 + 0x^2 - 3x + 0):

    >>> from scipy.special import legendre
    >>> legendre(3)
    poly1d([ 2.5,  0. , -1.5,  0. ])

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_legendre(n1)
    if n == 0:
        x, w = [], []
    hn = 2.0 / (2 * n + 1)
    kn = _gam(2 * n + 1) / _gam(n + 1)**2 / 2.0**n
    p = orthopoly1d(x, w, hn, kn, wfunc=lambda x: 1.0, limits=(-1, 1),
                    monic=monic, eval_func=lambda x: eval_legendre(n, x))
    return p

# Shifted Legendre              P^*_n(x)


def roots_sh_legendre(n, mu=False):
    r"""Gauss-Legendre (shifted) quadrature.

    Compute the sample points and weights for Gauss-Legendre
    quadrature. The sample points are the roots of the nth degree
    shifted Legendre polynomial :math:`P^*_n(x)`. These sample points
    and weights correctly integrate polynomials of degree :math:`2n -
    1` or less over the interval :math:`[0, 1]` with weight function
    :math:`w(x) = 1.0`. See 2.2.11 in [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.quadrature
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    x, w = roots_legendre(n)
    x = (x + 1) / 2
    w /= 2
    if mu:
        return x, w, 1.0
    else:
        return x, w

def sh_legendre(n, monic=False):
    r"""Shifted Legendre polynomial.

    Defined as :math:`P^*_n(x) = P_n(2x - 1)` for :math:`P_n` the nth
    Legendre polynomial.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    P : orthopoly1d
        Shifted Legendre polynomial.

    Notes
    -----
    The polynomials :math:`P^*_n` are orthogonal over :math:`[0, 1]`
    with weight function 1.

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    wfunc = lambda x: 0.0 * x + 1.0
    if n == 0:
        return orthopoly1d([], [], 1.0, 1.0, wfunc, (0, 1), monic,
                           lambda x: eval_sh_legendre(n, x))
    x, w = roots_sh_legendre(n)
    hn = 1.0 / (2 * n + 1.0)
    kn = _gam(2 * n + 1) / _gam(n + 1)**2
    p = orthopoly1d(x, w, hn, kn, wfunc, limits=(0, 1), monic=monic,
                    eval_func=lambda x: eval_sh_legendre(n, x))
    return p


# -----------------------------------------------------------------------------
# Code for backwards compatibility
# -----------------------------------------------------------------------------

# Import functions in case someone is still calling the orthogonal
# module directly. (They shouldn't be; it's not in the public API).
poch = cephes.poch

# eval_chebyu, eval_sh_chebyt and eval_sh_chebyu: These functions are not
# used in orthogonal.py, they are not in _rootfuns_map, but their names
# do appear in _evalfuns, so they must be kept.
from ._ufuncs import (binom, eval_jacobi, eval_sh_jacobi, eval_gegenbauer,
                      eval_chebyt, eval_chebyu, eval_chebys, eval_chebyc,
                      eval_sh_chebyt, eval_sh_chebyu, eval_legendre,
                      eval_sh_legendre, eval_genlaguerre, eval_laguerre,
                      eval_hermite, eval_hermitenorm)

# Make the old root function names an alias for the new ones
_modattrs = globals()
for newfun, oldfun in _rootfuns_map.items():
    _modattrs[oldfun] = _modattrs[newfun]
    __all__.append(oldfun)
