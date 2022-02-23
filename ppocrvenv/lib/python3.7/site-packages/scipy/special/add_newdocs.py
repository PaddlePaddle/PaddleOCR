# Docstrings for generated ufuncs
#
# The syntax is designed to look like the function add_newdoc is being
# called from numpy.lib, but in this file add_newdoc puts the
# docstrings in a dictionary. This dictionary is used in
# _generate_pyx.py to generate the docstrings for the ufuncs in
# scipy.special at the C level when the ufuncs are created at compile
# time.
from typing import Dict

docdict: Dict[str, str] = {}


def get(name):
    return docdict.get(name)


def add_newdoc(name, doc):
    docdict[name] = doc


add_newdoc("_sf_error_test_function",
    """
    Private function; do not use.
    """)


add_newdoc("_cosine_cdf",
    """
    _cosine_cdf(x)

    Cumulative distribution function (CDF) of the cosine distribution::

                 {             0,              x < -pi
        cdf(x) = { (pi + x + sin(x))/(2*pi),   -pi <= x <= pi
                 {             1,              x > pi

    Parameters
    ----------
    x : array_like
        `x` must contain real numbers.

    Returns
    -------
    float
        The cosine distribution CDF evaluated at `x`.

    """)

add_newdoc("_cosine_invcdf",
    """
    _cosine_invcdf(p)

    Inverse of the cumulative distribution function (CDF) of the cosine
    distribution.

    The CDF of the cosine distribution is::

        cdf(x) = (pi + x + sin(x))/(2*pi)

    This function computes the inverse of cdf(x).

    Parameters
    ----------
    p : array_like
        `p` must contain real numbers in the interval ``0 <= p <= 1``.
        `nan` is returned for values of `p` outside the interval [0, 1].

    Returns
    -------
    float
        The inverse of the cosine distribution CDF evaluated at `p`.

    """)

add_newdoc("sph_harm",
    r"""
    sph_harm(m, n, theta, phi)

    Compute spherical harmonics.

    The spherical harmonics are defined as

    .. math::

        Y^m_n(\theta,\phi) = \sqrt{\frac{2n+1}{4\pi} \frac{(n-m)!}{(n+m)!}}
          e^{i m \theta} P^m_n(\cos(\phi))

    where :math:`P_n^m` are the associated Legendre functions; see `lpmv`.

    Parameters
    ----------
    m : array_like
        Order of the harmonic (int); must have ``|m| <= n``.
    n : array_like
       Degree of the harmonic (int); must have ``n >= 0``. This is
       often denoted by ``l`` (lower case L) in descriptions of
       spherical harmonics.
    theta : array_like
       Azimuthal (longitudinal) coordinate; must be in ``[0, 2*pi]``.
    phi : array_like
       Polar (colatitudinal) coordinate; must be in ``[0, pi]``.

    Returns
    -------
    y_mn : complex float
       The harmonic :math:`Y^m_n` sampled at ``theta`` and ``phi``.

    Notes
    -----
    There are different conventions for the meanings of the input
    arguments ``theta`` and ``phi``. In SciPy ``theta`` is the
    azimuthal angle and ``phi`` is the polar angle. It is common to
    see the opposite convention, that is, ``theta`` as the polar angle
    and ``phi`` as the azimuthal angle.

    Note that SciPy's spherical harmonics include the Condon-Shortley
    phase [2]_ because it is part of `lpmv`.

    With SciPy's conventions, the first several spherical harmonics
    are

    .. math::

        Y_0^0(\theta, \phi) &= \frac{1}{2} \sqrt{\frac{1}{\pi}} \\
        Y_1^{-1}(\theta, \phi) &= \frac{1}{2} \sqrt{\frac{3}{2\pi}}
                                    e^{-i\theta} \sin(\phi) \\
        Y_1^0(\theta, \phi) &= \frac{1}{2} \sqrt{\frac{3}{\pi}}
                                 \cos(\phi) \\
        Y_1^1(\theta, \phi) &= -\frac{1}{2} \sqrt{\frac{3}{2\pi}}
                                 e^{i\theta} \sin(\phi).

    References
    ----------
    .. [1] Digital Library of Mathematical Functions, 14.30.
           https://dlmf.nist.gov/14.30
    .. [2] https://en.wikipedia.org/wiki/Spherical_harmonics#Condon.E2.80.93Shortley_phase
    """)

add_newdoc("_ellip_harm",
    """
    Internal function, use `ellip_harm` instead.
    """)

add_newdoc("_ellip_norm",
    """
    Internal function, use `ellip_norm` instead.
    """)

add_newdoc("_lambertw",
    """
    Internal function, use `lambertw` instead.
    """)

add_newdoc("voigt_profile",
    r"""
    voigt_profile(x, sigma, gamma, out=None)

    Voigt profile.

    The Voigt profile is a convolution of a 1-D Normal distribution with
    standard deviation ``sigma`` and a 1-D Cauchy distribution with half-width at
    half-maximum ``gamma``.

    If ``sigma = 0``, PDF of Cauchy distribution is returned.
    Conversely, if ``gamma = 0``, PDF of Normal distribution is returned.
    If ``sigma = gamma = 0``, the return value is ``Inf`` for ``x = 0``, and ``0`` for all other ``x``.

    Parameters
    ----------
    x : array_like
        Real argument
    sigma : array_like
        The standard deviation of the Normal distribution part
    gamma : array_like
        The half-width at half-maximum of the Cauchy distribution part
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        The Voigt profile at the given arguments

    Notes
    -----
    It can be expressed in terms of Faddeeva function

    .. math:: V(x; \sigma, \gamma) = \frac{Re[w(z)]}{\sigma\sqrt{2\pi}},
    .. math:: z = \frac{x + i\gamma}{\sqrt{2}\sigma}

    where :math:`w(z)` is the Faddeeva function.

    See Also
    --------
    wofz : Faddeeva function

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Voigt_profile

    """)

add_newdoc("wrightomega",
    r"""
    wrightomega(z, out=None)

    Wright Omega function.

    Defined as the solution to

    .. math::

        \omega + \log(\omega) = z

    where :math:`\log` is the principal branch of the complex logarithm.

    Parameters
    ----------
    z : array_like
        Points at which to evaluate the Wright Omega function

    Returns
    -------
    omega : ndarray
        Values of the Wright Omega function

    Notes
    -----
    .. versionadded:: 0.19.0

    The function can also be defined as

    .. math::

        \omega(z) = W_{K(z)}(e^z)

    where :math:`K(z) = \lceil (\Im(z) - \pi)/(2\pi) \rceil` is the
    unwinding number and :math:`W` is the Lambert W function.

    The implementation here is taken from [1]_.

    See Also
    --------
    lambertw : The Lambert W function

    References
    ----------
    .. [1] Lawrence, Corless, and Jeffrey, "Algorithm 917: Complex
           Double-Precision Evaluation of the Wright :math:`\omega`
           Function." ACM Transactions on Mathematical Software,
           2012. :doi:`10.1145/2168773.2168779`.

    """)


add_newdoc("agm",
    """
    agm(a, b)

    Compute the arithmetic-geometric mean of `a` and `b`.

    Start with a_0 = a and b_0 = b and iteratively compute::

        a_{n+1} = (a_n + b_n)/2
        b_{n+1} = sqrt(a_n*b_n)

    a_n and b_n converge to the same limit as n increases; their common
    limit is agm(a, b).

    Parameters
    ----------
    a, b : array_like
        Real values only. If the values are both negative, the result
        is negative. If one value is negative and the other is positive,
        `nan` is returned.

    Returns
    -------
    float
        The arithmetic-geometric mean of `a` and `b`.

    Examples
    --------
    >>> from scipy.special import agm
    >>> a, b = 24.0, 6.0
    >>> agm(a, b)
    13.458171481725614

    Compare that result to the iteration:

    >>> while a != b:
    ...     a, b = (a + b)/2, np.sqrt(a*b)
    ...     print("a = %19.16f  b=%19.16f" % (a, b))
    ...
    a = 15.0000000000000000  b=12.0000000000000000
    a = 13.5000000000000000  b=13.4164078649987388
    a = 13.4582039324993694  b=13.4581390309909850
    a = 13.4581714817451772  b=13.4581714817060547
    a = 13.4581714817256159  b=13.4581714817256159

    When array-like arguments are given, broadcasting applies:

    >>> a = np.array([[1.5], [3], [6]])  # a has shape (3, 1).
    >>> b = np.array([6, 12, 24, 48])    # b has shape (4,).
    >>> agm(a, b)
    array([[  3.36454287,   5.42363427,   9.05798751,  15.53650756],
           [  4.37037309,   6.72908574,  10.84726853,  18.11597502],
           [  6.        ,   8.74074619,  13.45817148,  21.69453707]])
    """)

add_newdoc("airy",
    r"""
    airy(z)

    Airy functions and their derivatives.

    Parameters
    ----------
    z : array_like
        Real or complex argument.

    Returns
    -------
    Ai, Aip, Bi, Bip : ndarrays
        Airy functions Ai and Bi, and their derivatives Aip and Bip.

    Notes
    -----
    The Airy functions Ai and Bi are two independent solutions of

    .. math:: y''(x) = x y(x).

    For real `z` in [-10, 10], the computation is carried out by calling
    the Cephes [1]_ `airy` routine, which uses power series summation
    for small `z` and rational minimax approximations for large `z`.

    Outside this range, the AMOS [2]_ `zairy` and `zbiry` routines are
    employed.  They are computed using power series for :math:`|z| < 1` and
    the following relations to modified Bessel functions for larger `z`
    (where :math:`t \equiv 2 z^{3/2}/3`):

    .. math::

        Ai(z) = \frac{1}{\pi \sqrt{3}} K_{1/3}(t)

        Ai'(z) = -\frac{z}{\pi \sqrt{3}} K_{2/3}(t)

        Bi(z) = \sqrt{\frac{z}{3}} \left(I_{-1/3}(t) + I_{1/3}(t) \right)

        Bi'(z) = \frac{z}{\sqrt{3}} \left(I_{-2/3}(t) + I_{2/3}(t)\right)

    See also
    --------
    airye : exponentially scaled Airy functions.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    .. [2] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/

    Examples
    --------
    Compute the Airy functions on the interval [-15, 5].

    >>> from scipy import special
    >>> x = np.linspace(-15, 5, 201)
    >>> ai, aip, bi, bip = special.airy(x)

    Plot Ai(x) and Bi(x).

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, ai, 'r', label='Ai(x)')
    >>> plt.plot(x, bi, 'b--', label='Bi(x)')
    >>> plt.ylim(-0.5, 1.0)
    >>> plt.grid()
    >>> plt.legend(loc='upper left')
    >>> plt.show()

    """)

add_newdoc("airye",
    """
    airye(z)

    Exponentially scaled Airy functions and their derivatives.

    Scaling::

        eAi  = Ai  * exp(2.0/3.0*z*sqrt(z))
        eAip = Aip * exp(2.0/3.0*z*sqrt(z))
        eBi  = Bi  * exp(-abs(2.0/3.0*(z*sqrt(z)).real))
        eBip = Bip * exp(-abs(2.0/3.0*(z*sqrt(z)).real))

    Parameters
    ----------
    z : array_like
        Real or complex argument.

    Returns
    -------
    eAi, eAip, eBi, eBip : array_like
        Exponentially scaled Airy functions eAi and eBi, and their derivatives
        eAip and eBip

    Notes
    -----
    Wrapper for the AMOS [1]_ routines `zairy` and `zbiry`.

    See also
    --------
    airy

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/

    Examples
    --------
    We can compute exponentially scaled Airy functions and their derivatives:

    >>> from scipy.special import airye
    >>> import matplotlib.pyplot as plt
    >>> z = np.linspace(0, 50, 500)
    >>> eAi, eAip, eBi, eBip = airye(z)
    >>> f, ax = plt.subplots(2, 1, sharex=True)
    >>> for ind, data in enumerate([[eAi, eAip, ["eAi", "eAip"]],
    ...                             [eBi, eBip, ["eBi", "eBip"]]]):
    ...     ax[ind].plot(z, data[0], "-r", z, data[1], "-b")
    ...     ax[ind].legend(data[2])
    ...     ax[ind].grid(True)
    >>> plt.show()

    We can compute these using usual non-scaled Airy functions by:

    >>> from scipy.special import airy
    >>> Ai, Aip, Bi, Bip = airy(z)
    >>> np.allclose(eAi, Ai * np.exp(2.0 / 3.0 * z * np.sqrt(z)))
    True
    >>> np.allclose(eAip, Aip * np.exp(2.0 / 3.0 * z * np.sqrt(z)))
    True
    >>> np.allclose(eBi, Bi * np.exp(-abs(np.real(2.0 / 3.0 * z * np.sqrt(z)))))
    True
    >>> np.allclose(eBip, Bip * np.exp(-abs(np.real(2.0 / 3.0 * z * np.sqrt(z)))))
    True

    Comparing non-scaled and exponentially scaled ones, the usual non-scaled
    function quickly underflows for large values, whereas the exponentially
    scaled function does not.

    >>> airy(200)
    (0.0, 0.0, nan, nan)
    >>> airye(200)
    (0.07501041684381093, -1.0609012305109042, 0.15003188417418148, 2.1215836725571093)

    """)

add_newdoc("bdtr",
    r"""
    bdtr(k, n, p)

    Binomial distribution cumulative distribution function.

    Sum of the terms 0 through `floor(k)` of the Binomial probability density.

    .. math::
        \mathrm{bdtr}(k, n, p) = \sum_{j=0}^{\lfloor k \rfloor} {{n}\choose{j}} p^j (1-p)^{n-j}

    Parameters
    ----------
    k : array_like
        Number of successes (double), rounded down to the nearest integer.
    n : array_like
        Number of events (int).
    p : array_like
        Probability of success in a single event (float).

    Returns
    -------
    y : ndarray
        Probability of `floor(k)` or fewer successes in `n` independent events with
        success probabilities of `p`.

    Notes
    -----
    The terms are not summed directly; instead the regularized incomplete beta
    function is employed, according to the formula,

    .. math::
        \mathrm{bdtr}(k, n, p) = I_{1 - p}(n - \lfloor k \rfloor, \lfloor k \rfloor + 1).

    Wrapper for the Cephes [1]_ routine `bdtr`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    """)

add_newdoc("bdtrc",
    r"""
    bdtrc(k, n, p)

    Binomial distribution survival function.

    Sum of the terms `floor(k) + 1` through `n` of the binomial probability
    density,

    .. math::
        \mathrm{bdtrc}(k, n, p) = \sum_{j=\lfloor k \rfloor +1}^n {{n}\choose{j}} p^j (1-p)^{n-j}

    Parameters
    ----------
    k : array_like
        Number of successes (double), rounded down to nearest integer.
    n : array_like
        Number of events (int)
    p : array_like
        Probability of success in a single event.

    Returns
    -------
    y : ndarray
        Probability of `floor(k) + 1` or more successes in `n` independent
        events with success probabilities of `p`.

    See also
    --------
    bdtr
    betainc

    Notes
    -----
    The terms are not summed directly; instead the regularized incomplete beta
    function is employed, according to the formula,

    .. math::
        \mathrm{bdtrc}(k, n, p) = I_{p}(\lfloor k \rfloor + 1, n - \lfloor k \rfloor).

    Wrapper for the Cephes [1]_ routine `bdtrc`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    """)

add_newdoc("bdtri",
    r"""
    bdtri(k, n, y)

    Inverse function to `bdtr` with respect to `p`.

    Finds the event probability `p` such that the sum of the terms 0 through
    `k` of the binomial probability density is equal to the given cumulative
    probability `y`.

    Parameters
    ----------
    k : array_like
        Number of successes (float), rounded down to the nearest integer.
    n : array_like
        Number of events (float)
    y : array_like
        Cumulative probability (probability of `k` or fewer successes in `n`
        events).

    Returns
    -------
    p : ndarray
        The event probability such that `bdtr(\lfloor k \rfloor, n, p) = y`.

    See also
    --------
    bdtr
    betaincinv

    Notes
    -----
    The computation is carried out using the inverse beta integral function
    and the relation,::

        1 - p = betaincinv(n - k, k + 1, y).

    Wrapper for the Cephes [1]_ routine `bdtri`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

add_newdoc("bdtrik",
    """
    bdtrik(y, n, p)

    Inverse function to `bdtr` with respect to `k`.

    Finds the number of successes `k` such that the sum of the terms 0 through
    `k` of the Binomial probability density for `n` events with probability
    `p` is equal to the given cumulative probability `y`.

    Parameters
    ----------
    y : array_like
        Cumulative probability (probability of `k` or fewer successes in `n`
        events).
    n : array_like
        Number of events (float).
    p : array_like
        Success probability (float).

    Returns
    -------
    k : ndarray
        The number of successes `k` such that `bdtr(k, n, p) = y`.

    See also
    --------
    bdtr

    Notes
    -----
    Formula 26.5.24 of [1]_ is used to reduce the binomial distribution to the
    cumulative incomplete beta distribution.

    Computation of `k` involves a search for a value that produces the desired
    value of `y`. The search relies on the monotonicity of `y` with `k`.

    Wrapper for the CDFLIB [2]_ Fortran routine `cdfbin`.

    References
    ----------
    .. [1] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
    .. [2] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.

    """)

add_newdoc("bdtrin",
    """
    bdtrin(k, y, p)

    Inverse function to `bdtr` with respect to `n`.

    Finds the number of events `n` such that the sum of the terms 0 through
    `k` of the Binomial probability density for events with probability `p` is
    equal to the given cumulative probability `y`.

    Parameters
    ----------
    k : array_like
        Number of successes (float).
    y : array_like
        Cumulative probability (probability of `k` or fewer successes in `n`
        events).
    p : array_like
        Success probability (float).

    Returns
    -------
    n : ndarray
        The number of events `n` such that `bdtr(k, n, p) = y`.

    See also
    --------
    bdtr

    Notes
    -----
    Formula 26.5.24 of [1]_ is used to reduce the binomial distribution to the
    cumulative incomplete beta distribution.

    Computation of `n` involves a search for a value that produces the desired
    value of `y`. The search relies on the monotonicity of `y` with `n`.

    Wrapper for the CDFLIB [2]_ Fortran routine `cdfbin`.

    References
    ----------
    .. [1] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
    .. [2] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    """)

add_newdoc("binom",
    """
    binom(n, k)

    Binomial coefficient

    See Also
    --------
    comb : The number of combinations of N things taken k at a time.

    """)

add_newdoc("btdtria",
    r"""
    btdtria(p, b, x)

    Inverse of `btdtr` with respect to `a`.

    This is the inverse of the beta cumulative distribution function, `btdtr`,
    considered as a function of `a`, returning the value of `a` for which
    `btdtr(a, b, x) = p`, or

    .. math::
        p = \int_0^x \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} t^{a-1} (1-t)^{b-1}\,dt

    Parameters
    ----------
    p : array_like
        Cumulative probability, in [0, 1].
    b : array_like
        Shape parameter (`b` > 0).
    x : array_like
        The quantile, in [0, 1].

    Returns
    -------
    a : ndarray
        The value of the shape parameter `a` such that `btdtr(a, b, x) = p`.

    See Also
    --------
    btdtr : Cumulative distribution function of the beta distribution.
    btdtri : Inverse with respect to `x`.
    btdtrib : Inverse with respect to `b`.

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdfbet`.

    The cumulative distribution function `p` is computed using a routine by
    DiDinato and Morris [2]_. Computation of `a` involves a search for a value
    that produces the desired value of `p`. The search relies on the
    monotonicity of `p` with `a`.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] DiDinato, A. R. and Morris, A. H.,
           Algorithm 708: Significant Digit Computation of the Incomplete Beta
           Function Ratios. ACM Trans. Math. Softw. 18 (1993), 360-373.

    """)

add_newdoc("btdtrib",
    r"""
    btdtria(a, p, x)

    Inverse of `btdtr` with respect to `b`.

    This is the inverse of the beta cumulative distribution function, `btdtr`,
    considered as a function of `b`, returning the value of `b` for which
    `btdtr(a, b, x) = p`, or

    .. math::
        p = \int_0^x \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} t^{a-1} (1-t)^{b-1}\,dt

    Parameters
    ----------
    a : array_like
        Shape parameter (`a` > 0).
    p : array_like
        Cumulative probability, in [0, 1].
    x : array_like
        The quantile, in [0, 1].

    Returns
    -------
    b : ndarray
        The value of the shape parameter `b` such that `btdtr(a, b, x) = p`.

    See Also
    --------
    btdtr : Cumulative distribution function of the beta distribution.
    btdtri : Inverse with respect to `x`.
    btdtria : Inverse with respect to `a`.

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdfbet`.

    The cumulative distribution function `p` is computed using a routine by
    DiDinato and Morris [2]_. Computation of `b` involves a search for a value
    that produces the desired value of `p`. The search relies on the
    monotonicity of `p` with `b`.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] DiDinato, A. R. and Morris, A. H.,
           Algorithm 708: Significant Digit Computation of the Incomplete Beta
           Function Ratios. ACM Trans. Math. Softw. 18 (1993), 360-373.


    """)

add_newdoc("bei",
    r"""
    bei(x, out=None)

    Kelvin function bei.

    Defined as

    .. math::

        \mathrm{bei}(x) = \Im[J_0(x e^{3 \pi i / 4})]

    where :math:`J_0` is the Bessel function of the first kind of
    order zero (see `jv`). See [dlmf]_ for more details.

    Parameters
    ----------
    x : array_like
        Real argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of the Kelvin function.

    See Also
    --------
    ber : the corresponding real part
    beip : the derivative of bei
    jv : Bessel function of the first kind

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/10.61

    Examples
    --------
    It can be expressed using Bessel functions.

    >>> import scipy.special as sc
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> sc.jv(0, x * np.exp(3 * np.pi * 1j / 4)).imag
    array([0.24956604, 0.97229163, 1.93758679, 2.29269032])
    >>> sc.bei(x)
    array([0.24956604, 0.97229163, 1.93758679, 2.29269032])

    """)

add_newdoc("beip",
    r"""
    beip(x, out=None)

    Derivative of the Kelvin function bei.

    Parameters
    ----------
    x : array_like
        Real argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        The values of the derivative of bei.

    See Also
    --------
    bei

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/10#PT5

    """)

add_newdoc("ber",
    r"""
    ber(x, out=None)

    Kelvin function ber.

    Defined as

    .. math::

        \mathrm{ber}(x) = \Re[J_0(x e^{3 \pi i / 4})]

    where :math:`J_0` is the Bessel function of the first kind of
    order zero (see `jv`). See [dlmf]_ for more details.

    Parameters
    ----------
    x : array_like
        Real argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of the Kelvin function.

    See Also
    --------
    bei : the corresponding real part
    berp : the derivative of bei
    jv : Bessel function of the first kind

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/10.61

    Examples
    --------
    It can be expressed using Bessel functions.

    >>> import scipy.special as sc
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> sc.jv(0, x * np.exp(3 * np.pi * 1j / 4)).real
    array([ 0.98438178,  0.75173418, -0.22138025, -2.56341656])
    >>> sc.ber(x)
    array([ 0.98438178,  0.75173418, -0.22138025, -2.56341656])

    """)

add_newdoc("berp",
    r"""
    berp(x, out=None)

    Derivative of the Kelvin function ber.

    Parameters
    ----------
    x : array_like
        Real argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        The values of the derivative of ber.

    See Also
    --------
    ber

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/10#PT5

    """)

add_newdoc("besselpoly",
    r"""
    besselpoly(a, lmb, nu, out=None)

    Weighted integral of the Bessel function of the first kind.

    Computes

    .. math::

       \int_0^1 x^\lambda J_\nu(2 a x) \, dx

    where :math:`J_\nu` is a Bessel function and :math:`\lambda=lmb`,
    :math:`\nu=nu`.

    Parameters
    ----------
    a : array_like
        Scale factor inside the Bessel function.
    lmb : array_like
        Power of `x`
    nu : array_like
        Order of the Bessel function.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Value of the integral.

    """)

add_newdoc("beta",
    r"""
    beta(a, b, out=None)

    Beta function.

    This function is defined in [1]_ as

    .. math::

        B(a, b) = \int_0^1 t^{a-1}(1-t)^{b-1}dt
                = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)},

    where :math:`\Gamma` is the gamma function.

    Parameters
    ----------
    a, b : array-like
        Real-valued arguments
    out : ndarray, optional
        Optional output array for the function result

    Returns
    -------
    scalar or ndarray
        Value of the beta function

    See Also
    --------
    gamma : the gamma function
    betainc :  the incomplete beta function
    betaln : the natural logarithm of the absolute
             value of the beta function

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions,
           Eq. 5.12.1. https://dlmf.nist.gov/5.12

    Examples
    --------
    >>> import scipy.special as sc

    The beta function relates to the gamma function by the
    definition given above:

    >>> sc.beta(2, 3)
    0.08333333333333333
    >>> sc.gamma(2)*sc.gamma(3)/sc.gamma(2 + 3)
    0.08333333333333333

    As this relationship demonstrates, the beta function
    is symmetric:

    >>> sc.beta(1.7, 2.4)
    0.16567527689031739
    >>> sc.beta(2.4, 1.7)
    0.16567527689031739

    This function satisfies :math:`B(1, b) = 1/b`:

    >>> sc.beta(1, 4)
    0.25

    """)

add_newdoc("betainc",
    r"""
    betainc(a, b, x, out=None)

    Incomplete beta function.

    Computes the incomplete beta function, defined as [1]_:

    .. math::

        I_x(a, b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \int_0^x
        t^{a-1}(1-t)^{b-1}dt,

    for :math:`0 \leq x \leq 1`.

    Parameters
    ----------
    a, b : array-like
           Positive, real-valued parameters
    x : array-like
        Real-valued such that :math:`0 \leq x \leq 1`,
        the upper limit of integration
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    array-like
        Value of the incomplete beta function

    See Also
    --------
    beta : beta function
    betaincinv : inverse of the incomplete beta function

    Notes
    -----
    The incomplete beta function is also sometimes defined
    without the `gamma` terms, in which case the above
    definition is the so-called regularized incomplete beta
    function. Under this definition, you can get the incomplete
    beta function by multiplying the result of the SciPy
    function by `beta`.

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/8.17

    Examples
    --------

    Let :math:`B(a, b)` be the `beta` function.

    >>> import scipy.special as sc

    The coefficient in terms of `gamma` is equal to
    :math:`1/B(a, b)`. Also, when :math:`x=1`
    the integral is equal to :math:`B(a, b)`.
    Therefore, :math:`I_{x=1}(a, b) = 1` for any :math:`a, b`.

    >>> sc.betainc(0.2, 3.5, 1.0)
    1.0

    It satisfies
    :math:`I_x(a, b) = x^a F(a, 1-b, a+1, x)/ (aB(a, b))`,
    where :math:`F` is the hypergeometric function `hyp2f1`:

    >>> a, b, x = 1.4, 3.1, 0.5
    >>> x**a * sc.hyp2f1(a, 1 - b, a + 1, x)/(a * sc.beta(a, b))
    0.8148904036225295
    >>> sc.betainc(a, b, x)
    0.8148904036225296

    This functions satisfies the relationship
    :math:`I_x(a, b) = 1 - I_{1-x}(b, a)`:

    >>> sc.betainc(2.2, 3.1, 0.4)
    0.49339638807619446
    >>> 1 - sc.betainc(3.1, 2.2, 1 - 0.4)
    0.49339638807619446

    """)

add_newdoc("betaincinv",
    r"""
    betaincinv(a, b, y, out=None)

    Inverse of the incomplete beta function.

    Computes :math:`x` such that:

    .. math::

        y = I_x(a, b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}
        \int_0^x t^{a-1}(1-t)^{b-1}dt,

    where :math:`I_x` is the normalized incomplete beta
    function `betainc` and
    :math:`\Gamma` is the `gamma` function [1]_.

    Parameters
    ----------
    a, b : array-like
        Positive, real-valued parameters
    y : array-like
        Real-valued input
    out : ndarray, optional
        Optional output array for function values

    Returns
    -------
    array-like
        Value of the inverse of the incomplete beta function

    See Also
    --------
    betainc : incomplete beta function
    gamma : gamma function

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/8.17

    Examples
    --------
    >>> import scipy.special as sc

    This function is the inverse of `betainc` for fixed
    values of :math:`a` and :math:`b`.

    >>> a, b = 1.2, 3.1
    >>> y = sc.betainc(a, b, 0.2)
    >>> sc.betaincinv(a, b, y)
    0.2
    >>>
    >>> a, b = 7.5, 0.4
    >>> x = sc.betaincinv(a, b, 0.5)
    >>> sc.betainc(a, b, x)
    0.5

    """)

add_newdoc("betaln",
    """
    betaln(a, b)

    Natural logarithm of absolute value of beta function.

    Computes ``ln(abs(beta(a, b)))``.
    """)

add_newdoc("boxcox",
    """
    boxcox(x, lmbda)

    Compute the Box-Cox transformation.

    The Box-Cox transformation is::

        y = (x**lmbda - 1) / lmbda  if lmbda != 0
            log(x)                  if lmbda == 0

    Returns `nan` if ``x < 0``.
    Returns `-inf` if ``x == 0`` and ``lmbda < 0``.

    Parameters
    ----------
    x : array_like
        Data to be transformed.
    lmbda : array_like
        Power parameter of the Box-Cox transform.

    Returns
    -------
    y : array
        Transformed data.

    Notes
    -----

    .. versionadded:: 0.14.0

    Examples
    --------
    >>> from scipy.special import boxcox
    >>> boxcox([1, 4, 10], 2.5)
    array([   0.        ,   12.4       ,  126.09110641])
    >>> boxcox(2, [0, 1, 2])
    array([ 0.69314718,  1.        ,  1.5       ])
    """)

add_newdoc("boxcox1p",
    """
    boxcox1p(x, lmbda)

    Compute the Box-Cox transformation of 1 + `x`.

    The Box-Cox transformation computed by `boxcox1p` is::

        y = ((1+x)**lmbda - 1) / lmbda  if lmbda != 0
            log(1+x)                    if lmbda == 0

    Returns `nan` if ``x < -1``.
    Returns `-inf` if ``x == -1`` and ``lmbda < 0``.

    Parameters
    ----------
    x : array_like
        Data to be transformed.
    lmbda : array_like
        Power parameter of the Box-Cox transform.

    Returns
    -------
    y : array
        Transformed data.

    Notes
    -----

    .. versionadded:: 0.14.0

    Examples
    --------
    >>> from scipy.special import boxcox1p
    >>> boxcox1p(1e-4, [0, 0.5, 1])
    array([  9.99950003e-05,   9.99975001e-05,   1.00000000e-04])
    >>> boxcox1p([0.01, 0.1], 0.25)
    array([ 0.00996272,  0.09645476])
    """)

add_newdoc("inv_boxcox",
    """
    inv_boxcox(y, lmbda)

    Compute the inverse of the Box-Cox transformation.

    Find ``x`` such that::

        y = (x**lmbda - 1) / lmbda  if lmbda != 0
            log(x)                  if lmbda == 0

    Parameters
    ----------
    y : array_like
        Data to be transformed.
    lmbda : array_like
        Power parameter of the Box-Cox transform.

    Returns
    -------
    x : array
        Transformed data.

    Notes
    -----

    .. versionadded:: 0.16.0

    Examples
    --------
    >>> from scipy.special import boxcox, inv_boxcox
    >>> y = boxcox([1, 4, 10], 2.5)
    >>> inv_boxcox(y, 2.5)
    array([1., 4., 10.])
    """)

add_newdoc("inv_boxcox1p",
    """
    inv_boxcox1p(y, lmbda)

    Compute the inverse of the Box-Cox transformation.

    Find ``x`` such that::

        y = ((1+x)**lmbda - 1) / lmbda  if lmbda != 0
            log(1+x)                    if lmbda == 0

    Parameters
    ----------
    y : array_like
        Data to be transformed.
    lmbda : array_like
        Power parameter of the Box-Cox transform.

    Returns
    -------
    x : array
        Transformed data.

    Notes
    -----

    .. versionadded:: 0.16.0

    Examples
    --------
    >>> from scipy.special import boxcox1p, inv_boxcox1p
    >>> y = boxcox1p([1, 4, 10], 2.5)
    >>> inv_boxcox1p(y, 2.5)
    array([1., 4., 10.])
    """)

add_newdoc("btdtr",
    r"""
    btdtr(a, b, x)

    Cumulative distribution function of the beta distribution.

    Returns the integral from zero to `x` of the beta probability density
    function,

    .. math::
        I = \int_0^x \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} t^{a-1} (1-t)^{b-1}\,dt

    where :math:`\Gamma` is the gamma function.

    Parameters
    ----------
    a : array_like
        Shape parameter (a > 0).
    b : array_like
        Shape parameter (b > 0).
    x : array_like
        Upper limit of integration, in [0, 1].

    Returns
    -------
    I : ndarray
        Cumulative distribution function of the beta distribution with
        parameters `a` and `b` at `x`.

    See Also
    --------
    betainc

    Notes
    -----
    This function is identical to the incomplete beta integral function
    `betainc`.

    Wrapper for the Cephes [1]_ routine `btdtr`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    """)

add_newdoc("btdtri",
    r"""
    btdtri(a, b, p)

    The `p`-th quantile of the beta distribution.

    This function is the inverse of the beta cumulative distribution function,
    `btdtr`, returning the value of `x` for which `btdtr(a, b, x) = p`, or

    .. math::
        p = \int_0^x \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} t^{a-1} (1-t)^{b-1}\,dt

    Parameters
    ----------
    a : array_like
        Shape parameter (`a` > 0).
    b : array_like
        Shape parameter (`b` > 0).
    p : array_like
        Cumulative probability, in [0, 1].

    Returns
    -------
    x : ndarray
        The quantile corresponding to `p`.

    See Also
    --------
    betaincinv
    btdtr

    Notes
    -----
    The value of `x` is found by interval halving or Newton iterations.

    Wrapper for the Cephes [1]_ routine `incbi`, which solves the equivalent
    problem of finding the inverse of the incomplete beta integral.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    """)

add_newdoc("cbrt",
    """
    cbrt(x)

    Element-wise cube root of `x`.

    Parameters
    ----------
    x : array_like
        `x` must contain real numbers.

    Returns
    -------
    float
        The cube root of each value in `x`.

    Examples
    --------
    >>> from scipy.special import cbrt

    >>> cbrt(8)
    2.0
    >>> cbrt([-8, -3, 0.125, 1.331])
    array([-2.        , -1.44224957,  0.5       ,  1.1       ])

    """)

add_newdoc("chdtr",
    r"""
    chdtr(v, x, out=None)

    Chi square cumulative distribution function.

    Returns the area under the left tail (from 0 to `x`) of the Chi
    square probability density function with `v` degrees of freedom:

    .. math::

        \frac{1}{2^{v/2} \Gamma(v/2)} \int_0^x t^{v/2 - 1} e^{-t/2} dt

    Here :math:`\Gamma` is the Gamma function; see `gamma`. This
    integral can be expressed in terms of the regularized lower
    incomplete gamma function `gammainc` as
    ``gammainc(v / 2, x / 2)``. [1]_

    Parameters
    ----------
    v : array_like
        Degrees of freedom.
    x : array_like
        Upper bound of the integral.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of the cumulative distribution function.

    See Also
    --------
    chdtrc, chdtri, chdtriv, gammainc

    References
    ----------
    .. [1] Chi-Square distribution,
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm

    Examples
    --------
    >>> import scipy.special as sc

    It can be expressed in terms of the regularized lower incomplete
    gamma function.

    >>> v = 1
    >>> x = np.arange(4)
    >>> sc.chdtr(v, x)
    array([0.        , 0.68268949, 0.84270079, 0.91673548])
    >>> sc.gammainc(v / 2, x / 2)
    array([0.        , 0.68268949, 0.84270079, 0.91673548])

    """)

add_newdoc("chdtrc",
    r"""
    chdtrc(v, x, out=None)

    Chi square survival function.

    Returns the area under the right hand tail (from `x` to infinity)
    of the Chi square probability density function with `v` degrees of
    freedom:

    .. math::

        \frac{1}{2^{v/2} \Gamma(v/2)} \int_x^\infty t^{v/2 - 1} e^{-t/2} dt

    Here :math:`\Gamma` is the Gamma function; see `gamma`. This
    integral can be expressed in terms of the regularized upper
    incomplete gamma function `gammaincc` as
    ``gammaincc(v / 2, x / 2)``. [1]_

    Parameters
    ----------
    v : array_like
        Degrees of freedom.
    x : array_like
        Lower bound of the integral.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of the survival function.

    See Also
    --------
    chdtr, chdtri, chdtriv, gammaincc

    References
    ----------
    .. [1] Chi-Square distribution,
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm

    Examples
    --------
    >>> import scipy.special as sc

    It can be expressed in terms of the regularized upper incomplete
    gamma function.

    >>> v = 1
    >>> x = np.arange(4)
    >>> sc.chdtrc(v, x)
    array([1.        , 0.31731051, 0.15729921, 0.08326452])
    >>> sc.gammaincc(v / 2, x / 2)
    array([1.        , 0.31731051, 0.15729921, 0.08326452])

    """)

add_newdoc("chdtri",
    """
    chdtri(v, p, out=None)

    Inverse to `chdtrc` with respect to `x`.

    Returns `x` such that ``chdtrc(v, x) == p``.

    Parameters
    ----------
    v : array_like
        Degrees of freedom.
    p : array_like
        Probability.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    x : scalar or ndarray
        Value so that the probability a Chi square random variable
        with `v` degrees of freedom is greater than `x` equals `p`.

    See Also
    --------
    chdtrc, chdtr, chdtriv

    References
    ----------
    .. [1] Chi-Square distribution,
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm

    Examples
    --------
    >>> import scipy.special as sc

    It inverts `chdtrc`.

    >>> v, p = 1, 0.3
    >>> sc.chdtrc(v, sc.chdtri(v, p))
    0.3
    >>> x = 1
    >>> sc.chdtri(v, sc.chdtrc(v, x))
    1.0

    """)

add_newdoc("chdtriv",
    """
    chdtriv(p, x, out=None)

    Inverse to `chdtr` with respect to `v`.

    Returns `v` such that ``chdtr(v, x) == p``.

    Parameters
    ----------
    p : array_like
        Probability that the Chi square random variable is less than
        or equal to `x`.
    x : array_like
        Nonnegative input.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Degrees of freedom.

    See Also
    --------
    chdtr, chdtrc, chdtri

    References
    ----------
    .. [1] Chi-Square distribution,
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm

    Examples
    --------
    >>> import scipy.special as sc

    It inverts `chdtr`.

    >>> p, x = 0.5, 1
    >>> sc.chdtr(sc.chdtriv(p, x), x)
    0.5000000000202172
    >>> v = 1
    >>> sc.chdtriv(sc.chdtr(v, x), v)
    1.0000000000000013

    """)

add_newdoc("chndtr",
    """
    chndtr(x, df, nc)

    Non-central chi square cumulative distribution function

    """)

add_newdoc("chndtrix",
    """
    chndtrix(p, df, nc)

    Inverse to `chndtr` vs `x`
    """)

add_newdoc("chndtridf",
    """
    chndtridf(x, p, nc)

    Inverse to `chndtr` vs `df`
    """)

add_newdoc("chndtrinc",
    """
    chndtrinc(x, df, p)

    Inverse to `chndtr` vs `nc`
    """)

add_newdoc("cosdg",
    """
    cosdg(x, out=None)

    Cosine of the angle `x` given in degrees.

    Parameters
    ----------
    x : array_like
        Angle, given in degrees.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Cosine of the input.

    See Also
    --------
    sindg, tandg, cotdg

    Examples
    --------
    >>> import scipy.special as sc

    It is more accurate than using cosine directly.

    >>> x = 90 + 180 * np.arange(3)
    >>> sc.cosdg(x)
    array([-0.,  0., -0.])
    >>> np.cos(x * np.pi / 180)
    array([ 6.1232340e-17, -1.8369702e-16,  3.0616170e-16])

    """)

add_newdoc("cosm1",
    """
    cosm1(x, out=None)

    cos(x) - 1 for use when `x` is near zero.

    Parameters
    ----------
    x : array_like
        Real valued argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of ``cos(x) - 1``.

    See Also
    --------
    expm1, log1p

    Examples
    --------
    >>> import scipy.special as sc

    It is more accurate than computing ``cos(x) - 1`` directly for
    ``x`` around 0.

    >>> x = 1e-30
    >>> np.cos(x) - 1
    0.0
    >>> sc.cosm1(x)
    -5.0000000000000005e-61

    """)

add_newdoc("cotdg",
    """
    cotdg(x, out=None)

    Cotangent of the angle `x` given in degrees.

    Parameters
    ----------
    x : array_like
        Angle, given in degrees.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Cotangent at the input.

    See Also
    --------
    sindg, cosdg, tandg

    Examples
    --------
    >>> import scipy.special as sc

    It is more accurate than using cotangent directly.

    >>> x = 90 + 180 * np.arange(3)
    >>> sc.cotdg(x)
    array([0., 0., 0.])
    >>> 1 / np.tan(x * np.pi / 180)
    array([6.1232340e-17, 1.8369702e-16, 3.0616170e-16])

    """)

add_newdoc("dawsn",
    """
    dawsn(x)

    Dawson's integral.

    Computes::

        exp(-x**2) * integral(exp(t**2), t=0..x).

    See Also
    --------
    wofz, erf, erfc, erfcx, erfi

    References
    ----------
    .. [1] Steven G. Johnson, Faddeeva W function implementation.
       http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-15, 15, num=1000)
    >>> plt.plot(x, special.dawsn(x))
    >>> plt.xlabel('$x$')
    >>> plt.ylabel('$dawsn(x)$')
    >>> plt.show()

    """)

add_newdoc("ellipe",
    r"""
    ellipe(m)

    Complete elliptic integral of the second kind

    This function is defined as

    .. math:: E(m) = \int_0^{\pi/2} [1 - m \sin(t)^2]^{1/2} dt

    Parameters
    ----------
    m : array_like
        Defines the parameter of the elliptic integral.

    Returns
    -------
    E : ndarray
        Value of the elliptic integral.

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `ellpe`.

    For `m > 0` the computation uses the approximation,

    .. math:: E(m) \approx P(1-m) - (1-m) \log(1-m) Q(1-m),

    where :math:`P` and :math:`Q` are tenth-order polynomials.  For
    `m < 0`, the relation

    .. math:: E(m) = E(m/(m - 1)) \sqrt(1-m)

    is used.

    The parameterization in terms of :math:`m` follows that of section
    17.2 in [2]_. Other parameterizations in terms of the
    complementary parameter :math:`1 - m`, modular angle
    :math:`\sin^2(\alpha) = m`, or modulus :math:`k^2 = m` are also
    used, so be careful that you choose the correct parameter.

    See Also
    --------
    ellipkm1 : Complete elliptic integral of the first kind, near `m` = 1
    ellipk : Complete elliptic integral of the first kind
    ellipkinc : Incomplete elliptic integral of the first kind
    ellipeinc : Incomplete elliptic integral of the second kind

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    This function is used in finding the circumference of an
    ellipse with semi-major axis `a` and semi-minor axis `b`.

    >>> from scipy import special

    >>> a = 3.5
    >>> b = 2.1
    >>> e_sq = 1.0 - b**2/a**2  # eccentricity squared

    Then the circumference is found using the following:

    >>> C = 4*a*special.ellipe(e_sq)  # circumference formula
    >>> C
    17.868899204378693

    When `a` and `b` are the same (meaning eccentricity is 0),
    this reduces to the circumference of a circle.

    >>> 4*a*special.ellipe(0.0)  # formula for ellipse with a = b
    21.991148575128552
    >>> 2*np.pi*a  # formula for circle of radius a
    21.991148575128552

    """)

add_newdoc("ellipeinc",
    r"""
    ellipeinc(phi, m)

    Incomplete elliptic integral of the second kind

    This function is defined as

    .. math:: E(\phi, m) = \int_0^{\phi} [1 - m \sin(t)^2]^{1/2} dt

    Parameters
    ----------
    phi : array_like
        amplitude of the elliptic integral.

    m : array_like
        parameter of the elliptic integral.

    Returns
    -------
    E : ndarray
        Value of the elliptic integral.

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `ellie`.

    Computation uses arithmetic-geometric means algorithm.

    The parameterization in terms of :math:`m` follows that of section
    17.2 in [2]_. Other parameterizations in terms of the
    complementary parameter :math:`1 - m`, modular angle
    :math:`\sin^2(\alpha) = m`, or modulus :math:`k^2 = m` are also
    used, so be careful that you choose the correct parameter.

    See Also
    --------
    ellipkm1 : Complete elliptic integral of the first kind, near `m` = 1
    ellipk : Complete elliptic integral of the first kind
    ellipkinc : Incomplete elliptic integral of the first kind
    ellipe : Complete elliptic integral of the second kind

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
    """)

add_newdoc("ellipj",
    """
    ellipj(u, m)

    Jacobian elliptic functions

    Calculates the Jacobian elliptic functions of parameter `m` between
    0 and 1, and real argument `u`.

    Parameters
    ----------
    m : array_like
        Parameter.
    u : array_like
        Argument.

    Returns
    -------
    sn, cn, dn, ph : ndarrays
        The returned functions::

            sn(u|m), cn(u|m), dn(u|m)

        The value `ph` is such that if `u = ellipkinc(ph, m)`,
        then `sn(u|m) = sin(ph)` and `cn(u|m) = cos(ph)`.

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `ellpj`.

    These functions are periodic, with quarter-period on the real axis
    equal to the complete elliptic integral `ellipk(m)`.

    Relation to incomplete elliptic integral: If `u = ellipkinc(phi,m)`, then
    `sn(u|m) = sin(phi)`, and `cn(u|m) = cos(phi)`. The `phi` is called
    the amplitude of `u`.

    Computation is by means of the arithmetic-geometric mean algorithm,
    except when `m` is within 1e-9 of 0 or 1. In the latter case with `m`
    close to 1, the approximation applies only for `phi < pi/2`.

    See also
    --------
    ellipk : Complete elliptic integral of the first kind
    ellipkinc : Incomplete elliptic integral of the first kind

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

add_newdoc("ellipkm1",
    """
    ellipkm1(p)

    Complete elliptic integral of the first kind around `m` = 1

    This function is defined as

    .. math:: K(p) = \\int_0^{\\pi/2} [1 - m \\sin(t)^2]^{-1/2} dt

    where `m = 1 - p`.

    Parameters
    ----------
    p : array_like
        Defines the parameter of the elliptic integral as `m = 1 - p`.

    Returns
    -------
    K : ndarray
        Value of the elliptic integral.

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `ellpk`.

    For `p <= 1`, computation uses the approximation,

    .. math:: K(p) \\approx P(p) - \\log(p) Q(p),

    where :math:`P` and :math:`Q` are tenth-order polynomials.  The
    argument `p` is used internally rather than `m` so that the logarithmic
    singularity at `m = 1` will be shifted to the origin; this preserves
    maximum accuracy.  For `p > 1`, the identity

    .. math:: K(p) = K(1/p)/\\sqrt(p)

    is used.

    See Also
    --------
    ellipk : Complete elliptic integral of the first kind
    ellipkinc : Incomplete elliptic integral of the first kind
    ellipe : Complete elliptic integral of the second kind
    ellipeinc : Incomplete elliptic integral of the second kind

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

add_newdoc("ellipk",
    r"""
    ellipk(m)

    Complete elliptic integral of the first kind.

    This function is defined as

    .. math:: K(m) = \int_0^{\pi/2} [1 - m \sin(t)^2]^{-1/2} dt

    Parameters
    ----------
    m : array_like
        The parameter of the elliptic integral.

    Returns
    -------
    K : array_like
        Value of the elliptic integral.

    Notes
    -----
    For more precision around point m = 1, use `ellipkm1`, which this
    function calls.

    The parameterization in terms of :math:`m` follows that of section
    17.2 in [1]_. Other parameterizations in terms of the
    complementary parameter :math:`1 - m`, modular angle
    :math:`\sin^2(\alpha) = m`, or modulus :math:`k^2 = m` are also
    used, so be careful that you choose the correct parameter.

    See Also
    --------
    ellipkm1 : Complete elliptic integral of the first kind around m = 1
    ellipkinc : Incomplete elliptic integral of the first kind
    ellipe : Complete elliptic integral of the second kind
    ellipeinc : Incomplete elliptic integral of the second kind

    References
    ----------
    .. [1] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("ellipkinc",
    r"""
    ellipkinc(phi, m)

    Incomplete elliptic integral of the first kind

    This function is defined as

    .. math:: K(\phi, m) = \int_0^{\phi} [1 - m \sin(t)^2]^{-1/2} dt

    This function is also called `F(phi, m)`.

    Parameters
    ----------
    phi : array_like
        amplitude of the elliptic integral

    m : array_like
        parameter of the elliptic integral

    Returns
    -------
    K : ndarray
        Value of the elliptic integral

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `ellik`.  The computation is
    carried out using the arithmetic-geometric mean algorithm.

    The parameterization in terms of :math:`m` follows that of section
    17.2 in [2]_. Other parameterizations in terms of the
    complementary parameter :math:`1 - m`, modular angle
    :math:`\sin^2(\alpha) = m`, or modulus :math:`k^2 = m` are also
    used, so be careful that you choose the correct parameter.

    See Also
    --------
    ellipkm1 : Complete elliptic integral of the first kind, near `m` = 1
    ellipk : Complete elliptic integral of the first kind
    ellipe : Complete elliptic integral of the second kind
    ellipeinc : Incomplete elliptic integral of the second kind

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.
    """)

add_newdoc("entr",
    r"""
    entr(x)

    Elementwise function for computing entropy.

    .. math:: \text{entr}(x) = \begin{cases} - x \log(x) & x > 0  \\ 0 & x = 0 \\ -\infty & \text{otherwise} \end{cases}

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    res : ndarray
        The value of the elementwise entropy function at the given points `x`.

    See Also
    --------
    kl_div, rel_entr

    Notes
    -----
    This function is concave.

    .. versionadded:: 0.15.0

    """)

add_newdoc("erf",
    """
    erf(z)

    Returns the error function of complex argument.

    It is defined as ``2/sqrt(pi)*integral(exp(-t**2), t=0..z)``.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    res : ndarray
        The values of the error function at the given points `x`.

    See Also
    --------
    erfc, erfinv, erfcinv, wofz, erfcx, erfi

    Notes
    -----
    The cumulative of the unit normal distribution is given by
    ``Phi(z) = 1/2[1 + erf(z/sqrt(2))]``.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Error_function
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover,
        1972. http://www.math.sfu.ca/~cbm/aands/page_297.htm
    .. [3] Steven G. Johnson, Faddeeva W function implementation.
       http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-3, 3)
    >>> plt.plot(x, special.erf(x))
    >>> plt.xlabel('$x$')
    >>> plt.ylabel('$erf(x)$')
    >>> plt.show()

    """)

add_newdoc("erfc",
    """
    erfc(x, out=None)

    Complementary error function, ``1 - erf(x)``.

    Parameters
    ----------
    x : array_like
        Real or complex valued argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the complementary error function

    See Also
    --------
    erf, erfi, erfcx, dawsn, wofz

    References
    ----------
    .. [1] Steven G. Johnson, Faddeeva W function implementation.
       http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-3, 3)
    >>> plt.plot(x, special.erfc(x))
    >>> plt.xlabel('$x$')
    >>> plt.ylabel('$erfc(x)$')
    >>> plt.show()

    """)

add_newdoc("erfi",
    """
    erfi(z, out=None)

    Imaginary error function, ``-i erf(i z)``.

    Parameters
    ----------
    z : array_like
        Real or complex valued argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the imaginary error function

    See Also
    --------
    erf, erfc, erfcx, dawsn, wofz

    Notes
    -----

    .. versionadded:: 0.12.0

    References
    ----------
    .. [1] Steven G. Johnson, Faddeeva W function implementation.
       http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-3, 3)
    >>> plt.plot(x, special.erfi(x))
    >>> plt.xlabel('$x$')
    >>> plt.ylabel('$erfi(x)$')
    >>> plt.show()

    """)

add_newdoc("erfcx",
    """
    erfcx(x, out=None)

    Scaled complementary error function, ``exp(x**2) * erfc(x)``.

    Parameters
    ----------
    x : array_like
        Real or complex valued argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the scaled complementary error function


    See Also
    --------
    erf, erfc, erfi, dawsn, wofz

    Notes
    -----

    .. versionadded:: 0.12.0

    References
    ----------
    .. [1] Steven G. Johnson, Faddeeva W function implementation.
       http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-3, 3)
    >>> plt.plot(x, special.erfcx(x))
    >>> plt.xlabel('$x$')
    >>> plt.ylabel('$erfcx(x)$')
    >>> plt.show()

    """)

add_newdoc("erfinv",
    """Inverse of the error function.

    Computes the inverse of the error function.

    In the complex domain, there is no unique complex number w satisfying
    erf(w)=z. This indicates a true inverse function would have multi-value.
    When the domain restricts to the real, -1 < x < 1, there is a unique real
    number satisfying erf(erfinv(x)) = x.

    Parameters
    ----------
    y : ndarray
        Argument at which to evaluate. Domain: [-1, 1]

    Returns
    -------
    erfinv : ndarray
        The inverse of erf of y, element-wise)

    See Also
    --------
    erf : Error function of a complex argument
    erfc : Complementary error function, ``1 - erf(x)``
    erfcinv : Inverse of the complementary error function

    Examples
    --------
    1) evaluating a float number

    >>> from scipy import special
    >>> special.erfinv(0.5)
    0.4769362762044698

    2) evaluating an ndarray

    >>> from scipy import special
    >>> y = np.linspace(-1.0, 1.0, num=10)
    >>> special.erfinv(y)
    array([       -inf, -0.86312307, -0.5407314 , -0.30457019, -0.0987901 ,
            0.0987901 ,  0.30457019,  0.5407314 ,  0.86312307,         inf])

    """)

add_newdoc("erfcinv",
    """Inverse of the complementary error function.

    Computes the inverse of the complementary error function.

    In the complex domain, there is no unique complex number w satisfying
    erfc(w)=z. This indicates a true inverse function would have multi-value.
    When the domain restricts to the real, 0 < x < 2, there is a unique real
    number satisfying erfc(erfcinv(x)) = erfcinv(erfc(x)).

    It is related to inverse of the error function by erfcinv(1-x) = erfinv(x)

    Parameters
    ----------
    y : ndarray
        Argument at which to evaluate. Domain: [0, 2]

    Returns
    -------
    erfcinv : ndarray
        The inverse of erfc of y, element-wise

    See Also
    --------
    erf : Error function of a complex argument
    erfc : Complementary error function, ``1 - erf(x)``
    erfinv : Inverse of the error function

    Examples
    --------
    1) evaluating a float number

    >>> from scipy import special
    >>> special.erfcinv(0.5)
    0.4769362762044698

    2) evaluating an ndarray

    >>> from scipy import special
    >>> y = np.linspace(0.0, 2.0, num=11)
    >>> special.erfcinv(y)
    array([        inf,  0.9061938 ,  0.59511608,  0.37080716,  0.17914345,
           -0.        , -0.17914345, -0.37080716, -0.59511608, -0.9061938 ,
                  -inf])

    """)

add_newdoc("eval_jacobi",
    r"""
    eval_jacobi(n, alpha, beta, x, out=None)

    Evaluate Jacobi polynomial at a point.

    The Jacobi polynomials can be defined via the Gauss hypergeometric
    function :math:`{}_2F_1` as

    .. math::

        P_n^{(\alpha, \beta)}(x) = \frac{(\alpha + 1)_n}{\Gamma(n + 1)}
          {}_2F_1(-n, 1 + \alpha + \beta + n; \alpha + 1; (1 - z)/2)

    where :math:`(\cdot)_n` is the Pochhammer symbol; see `poch`. When
    :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.42 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer the result is
        determined via the relation to the Gauss hypergeometric
        function.
    alpha : array_like
        Parameter
    beta : array_like
        Parameter
    x : array_like
        Points at which to evaluate the polynomial

    Returns
    -------
    P : ndarray
        Values of the Jacobi polynomial

    See Also
    --------
    roots_jacobi : roots and quadrature weights of Jacobi polynomials
    jacobi : Jacobi polynomial object
    hyp2f1 : Gauss hypergeometric function

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_sh_jacobi",
    r"""
    eval_sh_jacobi(n, p, q, x, out=None)

    Evaluate shifted Jacobi polynomial at a point.

    Defined by

    .. math::

        G_n^{(p, q)}(x)
          = \binom{2n + p - 1}{n}^{-1} P_n^{(p - q, q - 1)}(2x - 1),

    where :math:`P_n^{(\cdot, \cdot)}` is the n-th Jacobi
    polynomial. See 22.5.2 in [AS]_ for details.

    Parameters
    ----------
    n : int
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to `binom` and `eval_jacobi`.
    p : float
        Parameter
    q : float
        Parameter

    Returns
    -------
    G : ndarray
        Values of the shifted Jacobi polynomial.

    See Also
    --------
    roots_sh_jacobi : roots and quadrature weights of shifted Jacobi
                      polynomials
    sh_jacobi : shifted Jacobi polynomial object
    eval_jacobi : evaluate Jacobi polynomials

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_gegenbauer",
    r"""
    eval_gegenbauer(n, alpha, x, out=None)

    Evaluate Gegenbauer polynomial at a point.

    The Gegenbauer polynomials can be defined via the Gauss
    hypergeometric function :math:`{}_2F_1` as

    .. math::

        C_n^{(\alpha)} = \frac{(2\alpha)_n}{\Gamma(n + 1)}
          {}_2F_1(-n, 2\alpha + n; \alpha + 1/2; (1 - z)/2).

    When :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.46 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to the Gauss hypergeometric
        function.
    alpha : array_like
        Parameter
    x : array_like
        Points at which to evaluate the Gegenbauer polynomial

    Returns
    -------
    C : ndarray
        Values of the Gegenbauer polynomial

    See Also
    --------
    roots_gegenbauer : roots and quadrature weights of Gegenbauer
                       polynomials
    gegenbauer : Gegenbauer polynomial object
    hyp2f1 : Gauss hypergeometric function

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_chebyt",
    r"""
    eval_chebyt(n, x, out=None)

    Evaluate Chebyshev polynomial of the first kind at a point.

    The Chebyshev polynomials of the first kind can be defined via the
    Gauss hypergeometric function :math:`{}_2F_1` as

    .. math::

        T_n(x) = {}_2F_1(n, -n; 1/2; (1 - x)/2).

    When :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.47 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to the Gauss hypergeometric
        function.
    x : array_like
        Points at which to evaluate the Chebyshev polynomial

    Returns
    -------
    T : ndarray
        Values of the Chebyshev polynomial

    See Also
    --------
    roots_chebyt : roots and quadrature weights of Chebyshev
                   polynomials of the first kind
    chebyu : Chebychev polynomial object
    eval_chebyu : evaluate Chebyshev polynomials of the second kind
    hyp2f1 : Gauss hypergeometric function
    numpy.polynomial.chebyshev.Chebyshev : Chebyshev series

    Notes
    -----
    This routine is numerically stable for `x` in ``[-1, 1]`` at least
    up to order ``10000``.

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_chebyu",
    r"""
    eval_chebyu(n, x, out=None)

    Evaluate Chebyshev polynomial of the second kind at a point.

    The Chebyshev polynomials of the second kind can be defined via
    the Gauss hypergeometric function :math:`{}_2F_1` as

    .. math::

        U_n(x) = (n + 1) {}_2F_1(-n, n + 2; 3/2; (1 - x)/2).

    When :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.48 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to the Gauss hypergeometric
        function.
    x : array_like
        Points at which to evaluate the Chebyshev polynomial

    Returns
    -------
    U : ndarray
        Values of the Chebyshev polynomial

    See Also
    --------
    roots_chebyu : roots and quadrature weights of Chebyshev
                   polynomials of the second kind
    chebyu : Chebyshev polynomial object
    eval_chebyt : evaluate Chebyshev polynomials of the first kind
    hyp2f1 : Gauss hypergeometric function

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_chebys",
    r"""
    eval_chebys(n, x, out=None)

    Evaluate Chebyshev polynomial of the second kind on [-2, 2] at a
    point.

    These polynomials are defined as

    .. math::

        S_n(x) = U_n(x/2)

    where :math:`U_n` is a Chebyshev polynomial of the second
    kind. See 22.5.13 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to `eval_chebyu`.
    x : array_like
        Points at which to evaluate the Chebyshev polynomial

    Returns
    -------
    S : ndarray
        Values of the Chebyshev polynomial

    See Also
    --------
    roots_chebys : roots and quadrature weights of Chebyshev
                   polynomials of the second kind on [-2, 2]
    chebys : Chebyshev polynomial object
    eval_chebyu : evaluate Chebyshev polynomials of the second kind

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    >>> import scipy.special as sc

    They are a scaled version of the Chebyshev polynomials of the
    second kind.

    >>> x = np.linspace(-2, 2, 6)
    >>> sc.eval_chebys(3, x)
    array([-4.   ,  0.672,  0.736, -0.736, -0.672,  4.   ])
    >>> sc.eval_chebyu(3, x / 2)
    array([-4.   ,  0.672,  0.736, -0.736, -0.672,  4.   ])

    """)

add_newdoc("eval_chebyc",
    r"""
    eval_chebyc(n, x, out=None)

    Evaluate Chebyshev polynomial of the first kind on [-2, 2] at a
    point.

    These polynomials are defined as

    .. math::

        C_n(x) = 2 T_n(x/2)

    where :math:`T_n` is a Chebyshev polynomial of the first kind. See
    22.5.11 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to `eval_chebyt`.
    x : array_like
        Points at which to evaluate the Chebyshev polynomial

    Returns
    -------
    C : ndarray
        Values of the Chebyshev polynomial

    See Also
    --------
    roots_chebyc : roots and quadrature weights of Chebyshev
                   polynomials of the first kind on [-2, 2]
    chebyc : Chebyshev polynomial object
    numpy.polynomial.chebyshev.Chebyshev : Chebyshev series
    eval_chebyt : evaluate Chebycshev polynomials of the first kind

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    >>> import scipy.special as sc

    They are a scaled version of the Chebyshev polynomials of the
    first kind.

    >>> x = np.linspace(-2, 2, 6)
    >>> sc.eval_chebyc(3, x)
    array([-2.   ,  1.872,  1.136, -1.136, -1.872,  2.   ])
    >>> 2 * sc.eval_chebyt(3, x / 2)
    array([-2.   ,  1.872,  1.136, -1.136, -1.872,  2.   ])

    """)

add_newdoc("eval_sh_chebyt",
    r"""
    eval_sh_chebyt(n, x, out=None)

    Evaluate shifted Chebyshev polynomial of the first kind at a
    point.

    These polynomials are defined as

    .. math::

        T_n^*(x) = T_n(2x - 1)

    where :math:`T_n` is a Chebyshev polynomial of the first kind. See
    22.5.14 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to `eval_chebyt`.
    x : array_like
        Points at which to evaluate the shifted Chebyshev polynomial

    Returns
    -------
    T : ndarray
        Values of the shifted Chebyshev polynomial

    See Also
    --------
    roots_sh_chebyt : roots and quadrature weights of shifted
                      Chebyshev polynomials of the first kind
    sh_chebyt : shifted Chebyshev polynomial object
    eval_chebyt : evaluate Chebyshev polynomials of the first kind
    numpy.polynomial.chebyshev.Chebyshev : Chebyshev series

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_sh_chebyu",
    r"""
    eval_sh_chebyu(n, x, out=None)

    Evaluate shifted Chebyshev polynomial of the second kind at a
    point.

    These polynomials are defined as

    .. math::

        U_n^*(x) = U_n(2x - 1)

    where :math:`U_n` is a Chebyshev polynomial of the first kind. See
    22.5.15 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to `eval_chebyu`.
    x : array_like
        Points at which to evaluate the shifted Chebyshev polynomial

    Returns
    -------
    U : ndarray
        Values of the shifted Chebyshev polynomial

    See Also
    --------
    roots_sh_chebyu : roots and quadrature weights of shifted
                      Chebychev polynomials of the second kind
    sh_chebyu : shifted Chebyshev polynomial object
    eval_chebyu : evaluate Chebyshev polynomials of the second kind

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_legendre",
    r"""
    eval_legendre(n, x, out=None)

    Evaluate Legendre polynomial at a point.

    The Legendre polynomials can be defined via the Gauss
    hypergeometric function :math:`{}_2F_1` as

    .. math::

        P_n(x) = {}_2F_1(-n, n + 1; 1; (1 - x)/2).

    When :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.49 in [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to the Gauss hypergeometric
        function.
    x : array_like
        Points at which to evaluate the Legendre polynomial

    Returns
    -------
    P : ndarray
        Values of the Legendre polynomial

    See Also
    --------
    roots_legendre : roots and quadrature weights of Legendre
                     polynomials
    legendre : Legendre polynomial object
    hyp2f1 : Gauss hypergeometric function
    numpy.polynomial.legendre.Legendre : Legendre series

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    >>> from scipy.special import eval_legendre

    Evaluate the zero-order Legendre polynomial at x = 0

    >>> eval_legendre(0, 0)
    1.0

    Evaluate the first-order Legendre polynomial between -1 and 1

    >>> X = np.linspace(-1, 1, 5)  # Domain of Legendre polynomials
    >>> eval_legendre(1, X)
    array([-1. , -0.5,  0. ,  0.5,  1. ])

    Evaluate Legendre polynomials of order 0 through 4 at x = 0

    >>> N = range(0, 5)
    >>> eval_legendre(N, 0)
    array([ 1.   ,  0.   , -0.5  ,  0.   ,  0.375])

    Plot Legendre polynomials of order 0 through 4

    >>> X = np.linspace(-1, 1)

    >>> import matplotlib.pyplot as plt
    >>> for n in range(0, 5):
    ...     y = eval_legendre(n, X)
    ...     plt.plot(X, y, label=r'$P_{}(x)$'.format(n))

    >>> plt.title("Legendre Polynomials")
    >>> plt.xlabel("x")
    >>> plt.ylabel(r'$P_n(x)$')
    >>> plt.legend(loc='lower right')
    >>> plt.show()

    """)

add_newdoc("eval_sh_legendre",
    r"""
    eval_sh_legendre(n, x, out=None)

    Evaluate shifted Legendre polynomial at a point.

    These polynomials are defined as

    .. math::

        P_n^*(x) = P_n(2x - 1)

    where :math:`P_n` is a Legendre polynomial. See 2.2.11 in [AS]_
    for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the value is
        determined via the relation to `eval_legendre`.
    x : array_like
        Points at which to evaluate the shifted Legendre polynomial

    Returns
    -------
    P : ndarray
        Values of the shifted Legendre polynomial

    See Also
    --------
    roots_sh_legendre : roots and quadrature weights of shifted
                        Legendre polynomials
    sh_legendre : shifted Legendre polynomial object
    eval_legendre : evaluate Legendre polynomials
    numpy.polynomial.legendre.Legendre : Legendre series

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_genlaguerre",
    r"""
    eval_genlaguerre(n, alpha, x, out=None)

    Evaluate generalized Laguerre polynomial at a point.

    The generalized Laguerre polynomials can be defined via the
    confluent hypergeometric function :math:`{}_1F_1` as

    .. math::

        L_n^{(\alpha)}(x) = \binom{n + \alpha}{n}
          {}_1F_1(-n, \alpha + 1, x).

    When :math:`n` is an integer the result is a polynomial of degree
    :math:`n`. See 22.5.54 in [AS]_ for details. The Laguerre
    polynomials are the special case where :math:`\alpha = 0`.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer, the result is
        determined via the relation to the confluent hypergeometric
        function.
    alpha : array_like
        Parameter; must have ``alpha > -1``
    x : array_like
        Points at which to evaluate the generalized Laguerre
        polynomial

    Returns
    -------
    L : ndarray
        Values of the generalized Laguerre polynomial

    See Also
    --------
    roots_genlaguerre : roots and quadrature weights of generalized
                        Laguerre polynomials
    genlaguerre : generalized Laguerre polynomial object
    hyp1f1 : confluent hypergeometric function
    eval_laguerre : evaluate Laguerre polynomials

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_laguerre",
    r"""
    eval_laguerre(n, x, out=None)

    Evaluate Laguerre polynomial at a point.

    The Laguerre polynomials can be defined via the confluent
    hypergeometric function :math:`{}_1F_1` as

    .. math::

        L_n(x) = {}_1F_1(-n, 1, x).

    See 22.5.16 and 22.5.54 in [AS]_ for details. When :math:`n` is an
    integer the result is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial. If not an integer the result is
        determined via the relation to the confluent hypergeometric
        function.
    x : array_like
        Points at which to evaluate the Laguerre polynomial

    Returns
    -------
    L : ndarray
        Values of the Laguerre polynomial

    See Also
    --------
    roots_laguerre : roots and quadrature weights of Laguerre
                     polynomials
    laguerre : Laguerre polynomial object
    numpy.polynomial.laguerre.Laguerre : Laguerre series
    eval_genlaguerre : evaluate generalized Laguerre polynomials

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

     """)

add_newdoc("eval_hermite",
    r"""
    eval_hermite(n, x, out=None)

    Evaluate physicist's Hermite polynomial at a point.

    Defined by

    .. math::

        H_n(x) = (-1)^n e^{x^2} \frac{d^n}{dx^n} e^{-x^2};

    :math:`H_n` is a polynomial of degree :math:`n`. See 22.11.7 in
    [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial
    x : array_like
        Points at which to evaluate the Hermite polynomial

    Returns
    -------
    H : ndarray
        Values of the Hermite polynomial

    See Also
    --------
    roots_hermite : roots and quadrature weights of physicist's
                    Hermite polynomials
    hermite : physicist's Hermite polynomial object
    numpy.polynomial.hermite.Hermite : Physicist's Hermite series
    eval_hermitenorm : evaluate Probabilist's Hermite polynomials

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("eval_hermitenorm",
    r"""
    eval_hermitenorm(n, x, out=None)

    Evaluate probabilist's (normalized) Hermite polynomial at a
    point.

    Defined by

    .. math::

        He_n(x) = (-1)^n e^{x^2/2} \frac{d^n}{dx^n} e^{-x^2/2};

    :math:`He_n` is a polynomial of degree :math:`n`. See 22.11.8 in
    [AS]_ for details.

    Parameters
    ----------
    n : array_like
        Degree of the polynomial
    x : array_like
        Points at which to evaluate the Hermite polynomial

    Returns
    -------
    He : ndarray
        Values of the Hermite polynomial

    See Also
    --------
    roots_hermitenorm : roots and quadrature weights of probabilist's
                        Hermite polynomials
    hermitenorm : probabilist's Hermite polynomial object
    numpy.polynomial.hermite_e.HermiteE : Probabilist's Hermite series
    eval_hermite : evaluate physicist's Hermite polynomials

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("exp1",
    r"""
    exp1(z, out=None)

    Exponential integral E1.

    For complex :math:`z \ne 0` the exponential integral can be defined as
    [1]_

    .. math::

       E_1(z) = \int_z^\infty \frac{e^{-t}}{t} dt,

    where the path of the integral does not cross the negative real
    axis or pass through the origin.

    Parameters
    ----------
    z: array_like
        Real or complex argument.
    out: ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the exponential integral E1

    See Also
    --------
    expi : exponential integral :math:`Ei`
    expn : generalization of :math:`E_1`

    Notes
    -----
    For :math:`x > 0` it is related to the exponential integral
    :math:`Ei` (see `expi`) via the relation

    .. math::

       E_1(x) = -Ei(-x).

    References
    ----------
    .. [1] Digital Library of Mathematical Functions, 6.2.1
           https://dlmf.nist.gov/6.2#E1

    Examples
    --------
    >>> import scipy.special as sc

    It has a pole at 0.

    >>> sc.exp1(0)
    inf

    It has a branch cut on the negative real axis.

    >>> sc.exp1(-1)
    nan
    >>> sc.exp1(complex(-1, 0))
    (-1.8951178163559368-3.141592653589793j)
    >>> sc.exp1(complex(-1, -0.0))
    (-1.8951178163559368+3.141592653589793j)

    It approaches 0 along the positive real axis.

    >>> sc.exp1([1, 10, 100, 1000])
    array([2.19383934e-01, 4.15696893e-06, 3.68359776e-46, 0.00000000e+00])

    It is related to `expi`.

    >>> x = np.array([1, 2, 3, 4])
    >>> sc.exp1(x)
    array([0.21938393, 0.04890051, 0.01304838, 0.00377935])
    >>> -sc.expi(-x)
    array([0.21938393, 0.04890051, 0.01304838, 0.00377935])

    """)

add_newdoc("exp10",
    """
    exp10(x)

    Compute ``10**x`` element-wise.

    Parameters
    ----------
    x : array_like
        `x` must contain real numbers.

    Returns
    -------
    float
        ``10**x``, computed element-wise.

    Examples
    --------
    >>> from scipy.special import exp10

    >>> exp10(3)
    1000.0
    >>> x = np.array([[-1, -0.5, 0], [0.5, 1, 1.5]])
    >>> exp10(x)
    array([[  0.1       ,   0.31622777,   1.        ],
           [  3.16227766,  10.        ,  31.6227766 ]])

    """)

add_newdoc("exp2",
    """
    exp2(x)

    Compute ``2**x`` element-wise.

    Parameters
    ----------
    x : array_like
        `x` must contain real numbers.

    Returns
    -------
    float
        ``2**x``, computed element-wise.

    Examples
    --------
    >>> from scipy.special import exp2

    >>> exp2(3)
    8.0
    >>> x = np.array([[-1, -0.5, 0], [0.5, 1, 1.5]])
    >>> exp2(x)
    array([[ 0.5       ,  0.70710678,  1.        ],
           [ 1.41421356,  2.        ,  2.82842712]])
    """)

add_newdoc("expi",
    r"""
    expi(x, out=None)

    Exponential integral Ei.

    For real :math:`x`, the exponential integral is defined as [1]_

    .. math::

        Ei(x) = \int_{-\infty}^x \frac{e^t}{t} dt.

    For :math:`x > 0` the integral is understood as a Cauchy principle
    value.

    It is extended to the complex plane by analytic continuation of
    the function on the interval :math:`(0, \infty)`. The complex
    variant has a branch cut on the negative real axis.

    Parameters
    ----------
    x: array_like
        Real or complex valued argument
    out: ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the exponential integral

    Notes
    -----
    The exponential integrals :math:`E_1` and :math:`Ei` satisfy the
    relation

    .. math::

        E_1(x) = -Ei(-x)

    for :math:`x > 0`.

    See Also
    --------
    exp1 : Exponential integral :math:`E_1`
    expn : Generalized exponential integral :math:`E_n`

    References
    ----------
    .. [1] Digital Library of Mathematical Functions, 6.2.5
           https://dlmf.nist.gov/6.2#E5

    Examples
    --------
    >>> import scipy.special as sc

    It is related to `exp1`.

    >>> x = np.array([1, 2, 3, 4])
    >>> -sc.expi(-x)
    array([0.21938393, 0.04890051, 0.01304838, 0.00377935])
    >>> sc.exp1(x)
    array([0.21938393, 0.04890051, 0.01304838, 0.00377935])

    The complex variant has a branch cut on the negative real axis.

    >>> import scipy.special as sc
    >>> sc.expi(-1 + 1e-12j)
    (-0.21938393439552062+3.1415926535894254j)
    >>> sc.expi(-1 - 1e-12j)
    (-0.21938393439552062-3.1415926535894254j)

    As the complex variant approaches the branch cut, the real parts
    approach the value of the real variant.

    >>> sc.expi(-1)
    -0.21938393439552062

    The SciPy implementation returns the real variant for complex
    values on the branch cut.

    >>> sc.expi(complex(-1, 0.0))
    (-0.21938393439552062-0j)
    >>> sc.expi(complex(-1, -0.0))
    (-0.21938393439552062-0j)

    """)

add_newdoc('expit',
    """
    expit(x)

    Expit (a.k.a. logistic sigmoid) ufunc for ndarrays.

    The expit function, also known as the logistic sigmoid function, is
    defined as ``expit(x) = 1/(1+exp(-x))``.  It is the inverse of the
    logit function.

    Parameters
    ----------
    x : ndarray
        The ndarray to apply expit to element-wise.

    Returns
    -------
    out : ndarray
        An ndarray of the same shape as x. Its entries
        are `expit` of the corresponding entry of x.

    See Also
    --------
    logit

    Notes
    -----
    As a ufunc expit takes a number of optional
    keyword arguments. For more information
    see `ufuncs <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_

    .. versionadded:: 0.10.0

    Examples
    --------
    >>> from scipy.special import expit, logit

    >>> expit([-np.inf, -1.5, 0, 1.5, np.inf])
    array([ 0.        ,  0.18242552,  0.5       ,  0.81757448,  1.        ])

    `logit` is the inverse of `expit`:

    >>> logit(expit([-2.5, 0, 3.1, 5.0]))
    array([-2.5,  0. ,  3.1,  5. ])

    Plot expit(x) for x in [-6, 6]:

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-6, 6, 121)
    >>> y = expit(x)
    >>> plt.plot(x, y)
    >>> plt.grid()
    >>> plt.xlim(-6, 6)
    >>> plt.xlabel('x')
    >>> plt.title('expit(x)')
    >>> plt.show()

    """)

add_newdoc("expm1",
    """
    expm1(x)

    Compute ``exp(x) - 1``.

    When `x` is near zero, ``exp(x)`` is near 1, so the numerical calculation
    of ``exp(x) - 1`` can suffer from catastrophic loss of precision.
    ``expm1(x)`` is implemented to avoid the loss of precision that occurs when
    `x` is near zero.

    Parameters
    ----------
    x : array_like
        `x` must contain real numbers.

    Returns
    -------
    float
        ``exp(x) - 1`` computed element-wise.

    Examples
    --------
    >>> from scipy.special import expm1

    >>> expm1(1.0)
    1.7182818284590451
    >>> expm1([-0.2, -0.1, 0, 0.1, 0.2])
    array([-0.18126925, -0.09516258,  0.        ,  0.10517092,  0.22140276])

    The exact value of ``exp(7.5e-13) - 1`` is::

        7.5000000000028125000000007031250000001318...*10**-13.

    Here is what ``expm1(7.5e-13)`` gives:

    >>> expm1(7.5e-13)
    7.5000000000028135e-13

    Compare that to ``exp(7.5e-13) - 1``, where the subtraction results in
    a "catastrophic" loss of precision:

    >>> np.exp(7.5e-13) - 1
    7.5006667543675576e-13

    """)

add_newdoc("expn",
    r"""
    expn(n, x, out=None)

    Generalized exponential integral En.

    For integer :math:`n \geq 0` and real :math:`x \geq 0` the
    generalized exponential integral is defined as [dlmf]_

    .. math::

        E_n(x) = x^{n - 1} \int_x^\infty \frac{e^{-t}}{t^n} dt.

    Parameters
    ----------
    n: array_like
        Non-negative integers
    x: array_like
        Real argument
    out: ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the generalized exponential integral

    See Also
    --------
    exp1 : special case of :math:`E_n` for :math:`n = 1`
    expi : related to :math:`E_n` when :math:`n = 1`

    References
    ----------
    .. [dlmf] Digital Library of Mathematical Functions, 8.19.2
              https://dlmf.nist.gov/8.19#E2

    Examples
    --------
    >>> import scipy.special as sc

    Its domain is nonnegative n and x.

    >>> sc.expn(-1, 1.0), sc.expn(1, -1.0)
    (nan, nan)

    It has a pole at ``x = 0`` for ``n = 1, 2``; for larger ``n`` it
    is equal to ``1 / (n - 1)``.

    >>> sc.expn([0, 1, 2, 3, 4], 0)
    array([       inf,        inf, 1.        , 0.5       , 0.33333333])

    For n equal to 0 it reduces to ``exp(-x) / x``.

    >>> x = np.array([1, 2, 3, 4])
    >>> sc.expn(0, x)
    array([0.36787944, 0.06766764, 0.01659569, 0.00457891])
    >>> np.exp(-x) / x
    array([0.36787944, 0.06766764, 0.01659569, 0.00457891])

    For n equal to 1 it reduces to `exp1`.

    >>> sc.expn(1, x)
    array([0.21938393, 0.04890051, 0.01304838, 0.00377935])
    >>> sc.exp1(x)
    array([0.21938393, 0.04890051, 0.01304838, 0.00377935])

    """)

add_newdoc("exprel",
    r"""
    exprel(x)

    Relative error exponential, ``(exp(x) - 1)/x``.

    When `x` is near zero, ``exp(x)`` is near 1, so the numerical calculation
    of ``exp(x) - 1`` can suffer from catastrophic loss of precision.
    ``exprel(x)`` is implemented to avoid the loss of precision that occurs when
    `x` is near zero.

    Parameters
    ----------
    x : ndarray
        Input array.  `x` must contain real numbers.

    Returns
    -------
    float
        ``(exp(x) - 1)/x``, computed element-wise.

    See Also
    --------
    expm1

    Notes
    -----
    .. versionadded:: 0.17.0

    Examples
    --------
    >>> from scipy.special import exprel

    >>> exprel(0.01)
    1.0050167084168056
    >>> exprel([-0.25, -0.1, 0, 0.1, 0.25])
    array([ 0.88479687,  0.95162582,  1.        ,  1.05170918,  1.13610167])

    Compare ``exprel(5e-9)`` to the naive calculation.  The exact value
    is ``1.00000000250000000416...``.

    >>> exprel(5e-9)
    1.0000000025

    >>> (np.exp(5e-9) - 1)/5e-9
    0.99999999392252903
    """)

add_newdoc("fdtr",
    r"""
    fdtr(dfn, dfd, x)

    F cumulative distribution function.

    Returns the value of the cumulative distribution function of the
    F-distribution, also known as Snedecor's F-distribution or the
    Fisher-Snedecor distribution.

    The F-distribution with parameters :math:`d_n` and :math:`d_d` is the
    distribution of the random variable,

    .. math::
        X = \frac{U_n/d_n}{U_d/d_d},

    where :math:`U_n` and :math:`U_d` are random variables distributed
    :math:`\chi^2`, with :math:`d_n` and :math:`d_d` degrees of freedom,
    respectively.

    Parameters
    ----------
    dfn : array_like
        First parameter (positive float).
    dfd : array_like
        Second parameter (positive float).
    x : array_like
        Argument (nonnegative float).

    Returns
    -------
    y : ndarray
        The CDF of the F-distribution with parameters `dfn` and `dfd` at `x`.

    Notes
    -----
    The regularized incomplete beta function is used, according to the
    formula,

    .. math::
        F(d_n, d_d; x) = I_{xd_n/(d_d + xd_n)}(d_n/2, d_d/2).

    Wrapper for the Cephes [1]_ routine `fdtr`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    """)

add_newdoc("fdtrc",
    r"""
    fdtrc(dfn, dfd, x)

    F survival function.

    Returns the complemented F-distribution function (the integral of the
    density from `x` to infinity).

    Parameters
    ----------
    dfn : array_like
        First parameter (positive float).
    dfd : array_like
        Second parameter (positive float).
    x : array_like
        Argument (nonnegative float).

    Returns
    -------
    y : ndarray
        The complemented F-distribution function with parameters `dfn` and
        `dfd` at `x`.

    See also
    --------
    fdtr

    Notes
    -----
    The regularized incomplete beta function is used, according to the
    formula,

    .. math::
        F(d_n, d_d; x) = I_{d_d/(d_d + xd_n)}(d_d/2, d_n/2).

    Wrapper for the Cephes [1]_ routine `fdtrc`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

add_newdoc("fdtri",
    r"""
    fdtri(dfn, dfd, p)

    The `p`-th quantile of the F-distribution.

    This function is the inverse of the F-distribution CDF, `fdtr`, returning
    the `x` such that `fdtr(dfn, dfd, x) = p`.

    Parameters
    ----------
    dfn : array_like
        First parameter (positive float).
    dfd : array_like
        Second parameter (positive float).
    p : array_like
        Cumulative probability, in [0, 1].

    Returns
    -------
    x : ndarray
        The quantile corresponding to `p`.

    Notes
    -----
    The computation is carried out using the relation to the inverse
    regularized beta function, :math:`I^{-1}_x(a, b)`.  Let
    :math:`z = I^{-1}_p(d_d/2, d_n/2).`  Then,

    .. math::
        x = \frac{d_d (1 - z)}{d_n z}.

    If `p` is such that :math:`x < 0.5`, the following relation is used
    instead for improved stability: let
    :math:`z' = I^{-1}_{1 - p}(d_n/2, d_d/2).` Then,

    .. math::
        x = \frac{d_d z'}{d_n (1 - z')}.

    Wrapper for the Cephes [1]_ routine `fdtri`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    """)

add_newdoc("fdtridfd",
    """
    fdtridfd(dfn, p, x)

    Inverse to `fdtr` vs dfd

    Finds the F density argument dfd such that ``fdtr(dfn, dfd, x) == p``.
    """)

add_newdoc("fdtridfn",
    """
    fdtridfn(p, dfd, x)

    Inverse to `fdtr` vs dfn

    finds the F density argument dfn such that ``fdtr(dfn, dfd, x) == p``.
    """)

add_newdoc("fresnel",
    r"""
    fresnel(z, out=None)

    Fresnel integrals.

    The Fresnel integrals are defined as

    .. math::

       S(z) &= \int_0^z \sin(\pi t^2 /2) dt \\
       C(z) &= \int_0^z \cos(\pi t^2 /2) dt.

    See [dlmf]_ for details.

    Parameters
    ----------
    z : array_like
        Real or complex valued argument
    out : 2-tuple of ndarrays, optional
        Optional output arrays for the function results

    Returns
    -------
    S, C : 2-tuple of scalar or ndarray
        Values of the Fresnel integrals

    See Also
    --------
    fresnel_zeros : zeros of the Fresnel integrals

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/7.2#iii

    Examples
    --------
    >>> import scipy.special as sc

    As z goes to infinity along the real axis, S and C converge to 0.5.

    >>> S, C = sc.fresnel([0.1, 1, 10, 100, np.inf])
    >>> S
    array([0.00052359, 0.43825915, 0.46816998, 0.4968169 , 0.5       ])
    >>> C
    array([0.09999753, 0.7798934 , 0.49989869, 0.4999999 , 0.5       ])

    They are related to the error function `erf`.

    >>> z = np.array([1, 2, 3, 4])
    >>> zeta = 0.5 * np.sqrt(np.pi) * (1 - 1j) * z
    >>> S, C = sc.fresnel(z)
    >>> C + 1j*S
    array([0.7798934 +0.43825915j, 0.48825341+0.34341568j,
           0.60572079+0.496313j  , 0.49842603+0.42051575j])
    >>> 0.5 * (1 + 1j) * sc.erf(zeta)
    array([0.7798934 +0.43825915j, 0.48825341+0.34341568j,
           0.60572079+0.496313j  , 0.49842603+0.42051575j])

    """)

add_newdoc("gamma",
    r"""
    gamma(z)

    gamma function.

    The gamma function is defined as

    .. math::

       \Gamma(z) = \int_0^\infty t^{z-1} e^{-t} dt

    for :math:`\Re(z) > 0` and is extended to the rest of the complex
    plane by analytic continuation. See [dlmf]_ for more details.

    Parameters
    ----------
    z : array_like
        Real or complex valued argument

    Returns
    -------
    scalar or ndarray
        Values of the gamma function

    Notes
    -----
    The gamma function is often referred to as the generalized
    factorial since :math:`\Gamma(n + 1) = n!` for natural numbers
    :math:`n`. More generally it satisfies the recurrence relation
    :math:`\Gamma(z + 1) = z \cdot \Gamma(z)` for complex :math:`z`,
    which, combined with the fact that :math:`\Gamma(1) = 1`, implies
    the above identity for :math:`z = n`.

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/5.2#E1

    Examples
    --------
    >>> from scipy.special import gamma, factorial

    >>> gamma([0, 0.5, 1, 5])
    array([         inf,   1.77245385,   1.        ,  24.        ])

    >>> z = 2.5 + 1j
    >>> gamma(z)
    (0.77476210455108352+0.70763120437959293j)
    >>> gamma(z+1), z*gamma(z)  # Recurrence property
    ((1.2292740569981171+2.5438401155000685j),
     (1.2292740569981158+2.5438401155000658j))

    >>> gamma(0.5)**2  # gamma(0.5) = sqrt(pi)
    3.1415926535897927

    Plot gamma(x) for real x

    >>> x = np.linspace(-3.5, 5.5, 2251)
    >>> y = gamma(x)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y, 'b', alpha=0.6, label='gamma(x)')
    >>> k = np.arange(1, 7)
    >>> plt.plot(k, factorial(k-1), 'k*', alpha=0.6,
    ...          label='(x-1)!, x = 1, 2, ...')
    >>> plt.xlim(-3.5, 5.5)
    >>> plt.ylim(-10, 25)
    >>> plt.grid()
    >>> plt.xlabel('x')
    >>> plt.legend(loc='lower right')
    >>> plt.show()

    """)

add_newdoc("gammainc",
    r"""
    gammainc(a, x)

    Regularized lower incomplete gamma function.

    It is defined as

    .. math::

        P(a, x) = \frac{1}{\Gamma(a)} \int_0^x t^{a - 1}e^{-t} dt

    for :math:`a > 0` and :math:`x \geq 0`. See [dlmf]_ for details.

    Parameters
    ----------
    a : array_like
        Positive parameter
    x : array_like
        Nonnegative argument

    Returns
    -------
    scalar or ndarray
        Values of the lower incomplete gamma function

    Notes
    -----
    The function satisfies the relation ``gammainc(a, x) +
    gammaincc(a, x) = 1`` where `gammaincc` is the regularized upper
    incomplete gamma function.

    The implementation largely follows that of [boost]_.

    See also
    --------
    gammaincc : regularized upper incomplete gamma function
    gammaincinv : inverse of the regularized lower incomplete gamma
        function with respect to `x`
    gammainccinv : inverse of the regularized upper incomplete gamma
        function with respect to `x`

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical functions
              https://dlmf.nist.gov/8.2#E4
    .. [boost] Maddock et. al., "Incomplete Gamma Functions",
       https://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html

    Examples
    --------
    >>> import scipy.special as sc

    It is the CDF of the gamma distribution, so it starts at 0 and
    monotonically increases to 1.

    >>> sc.gammainc(0.5, [0, 1, 10, 100])
    array([0.        , 0.84270079, 0.99999226, 1.        ])

    It is equal to one minus the upper incomplete gamma function.

    >>> a, x = 0.5, 0.4
    >>> sc.gammainc(a, x)
    0.6289066304773024
    >>> 1 - sc.gammaincc(a, x)
    0.6289066304773024

    """)

add_newdoc("gammaincc",
    r"""
    gammaincc(a, x)

    Regularized upper incomplete gamma function.

    It is defined as

    .. math::

        Q(a, x) = \frac{1}{\Gamma(a)} \int_x^\infty t^{a - 1}e^{-t} dt

    for :math:`a > 0` and :math:`x \geq 0`. See [dlmf]_ for details.

    Parameters
    ----------
    a : array_like
        Positive parameter
    x : array_like
        Nonnegative argument

    Returns
    -------
    scalar or ndarray
        Values of the upper incomplete gamma function

    Notes
    -----
    The function satisfies the relation ``gammainc(a, x) +
    gammaincc(a, x) = 1`` where `gammainc` is the regularized lower
    incomplete gamma function.

    The implementation largely follows that of [boost]_.

    See also
    --------
    gammainc : regularized lower incomplete gamma function
    gammaincinv : inverse of the regularized lower incomplete gamma
        function with respect to `x`
    gammainccinv : inverse to of the regularized upper incomplete
        gamma function with respect to `x`

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical functions
              https://dlmf.nist.gov/8.2#E4
    .. [boost] Maddock et. al., "Incomplete Gamma Functions",
       https://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html

    Examples
    --------
    >>> import scipy.special as sc

    It is the survival function of the gamma distribution, so it
    starts at 1 and monotonically decreases to 0.

    >>> sc.gammaincc(0.5, [0, 1, 10, 100, 1000])
    array([1.00000000e+00, 1.57299207e-01, 7.74421643e-06, 2.08848758e-45,
           0.00000000e+00])

    It is equal to one minus the lower incomplete gamma function.

    >>> a, x = 0.5, 0.4
    >>> sc.gammaincc(a, x)
    0.37109336952269756
    >>> 1 - sc.gammainc(a, x)
    0.37109336952269756

    """)

add_newdoc("gammainccinv",
    """
    gammainccinv(a, y)

    Inverse of the upper incomplete gamma function with respect to `x`

    Given an input :math:`y` between 0 and 1, returns :math:`x` such
    that :math:`y = Q(a, x)`. Here :math:`Q` is the upper incomplete
    gamma function; see `gammaincc`. This is well-defined because the
    upper incomplete gamma function is monotonic as can be seen from
    its definition in [dlmf]_.

    Parameters
    ----------
    a : array_like
        Positive parameter
    y : array_like
        Argument between 0 and 1, inclusive

    Returns
    -------
    scalar or ndarray
        Values of the inverse of the upper incomplete gamma function

    See Also
    --------
    gammaincc : regularized upper incomplete gamma function
    gammainc : regularized lower incomplete gamma function
    gammaincinv : inverse of the regularized lower incomplete gamma
        function with respect to `x`

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/8.2#E4

    Examples
    --------
    >>> import scipy.special as sc

    It starts at infinity and monotonically decreases to 0.

    >>> sc.gammainccinv(0.5, [0, 0.1, 0.5, 1])
    array([       inf, 1.35277173, 0.22746821, 0.        ])

    It inverts the upper incomplete gamma function.

    >>> a, x = 0.5, [0, 0.1, 0.5, 1]
    >>> sc.gammaincc(a, sc.gammainccinv(a, x))
    array([0. , 0.1, 0.5, 1. ])

    >>> a, x = 0.5, [0, 10, 50]
    >>> sc.gammainccinv(a, sc.gammaincc(a, x))
    array([ 0., 10., 50.])

    """)

add_newdoc("gammaincinv",
    """
    gammaincinv(a, y)

    Inverse to the lower incomplete gamma function with respect to `x`.

    Given an input :math:`y` between 0 and 1, returns :math:`x` such
    that :math:`y = P(a, x)`. Here :math:`P` is the regularized lower
    incomplete gamma function; see `gammainc`. This is well-defined
    because the lower incomplete gamma function is monotonic as can be
    seen from its definition in [dlmf]_.

    Parameters
    ----------
    a : array_like
        Positive parameter
    y : array_like
        Parameter between 0 and 1, inclusive

    Returns
    -------
    scalar or ndarray
        Values of the inverse of the lower incomplete gamma function

    See Also
    --------
    gammainc : regularized lower incomplete gamma function
    gammaincc : regularized upper incomplete gamma function
    gammainccinv : inverse of the regualizred upper incomplete gamma
        function with respect to `x`

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/8.2#E4

    Examples
    --------
    >>> import scipy.special as sc

    It starts at 0 and monotonically increases to infinity.

    >>> sc.gammaincinv(0.5, [0, 0.1 ,0.5, 1])
    array([0.        , 0.00789539, 0.22746821,        inf])

    It inverts the lower incomplete gamma function.

    >>> a, x = 0.5, [0, 0.1, 0.5, 1]
    >>> sc.gammainc(a, sc.gammaincinv(a, x))
    array([0. , 0.1, 0.5, 1. ])

    >>> a, x = 0.5, [0, 10, 25]
    >>> sc.gammaincinv(a, sc.gammainc(a, x))
    array([ 0.        , 10.        , 25.00001465])

    """)

add_newdoc("gammaln",
    r"""
    gammaln(x, out=None)

    Logarithm of the absolute value of the gamma function.

    Defined as

    .. math::

       \ln(\lvert\Gamma(x)\rvert)

    where :math:`\Gamma` is the gamma function. For more details on
    the gamma function, see [dlmf]_.

    Parameters
    ----------
    x : array_like
        Real argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the log of the absolute value of gamma

    See Also
    --------
    gammasgn : sign of the gamma function
    loggamma : principal branch of the logarithm of the gamma function

    Notes
    -----
    It is the same function as the Python standard library function
    :func:`math.lgamma`.

    When used in conjunction with `gammasgn`, this function is useful
    for working in logspace on the real axis without having to deal
    with complex numbers via the relation ``exp(gammaln(x)) =
    gammasgn(x) * gamma(x)``.

    For complex-valued log-gamma, use `loggamma` instead of `gammaln`.

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/5

    Examples
    --------
    >>> import scipy.special as sc

    It has two positive zeros.

    >>> sc.gammaln([1, 2])
    array([0., 0.])

    It has poles at nonpositive integers.

    >>> sc.gammaln([0, -1, -2, -3, -4])
    array([inf, inf, inf, inf, inf])

    It asymptotically approaches ``x * log(x)`` (Stirling's formula).

    >>> x = np.array([1e10, 1e20, 1e40, 1e80])
    >>> sc.gammaln(x)
    array([2.20258509e+11, 4.50517019e+21, 9.11034037e+41, 1.83206807e+82])
    >>> x * np.log(x)
    array([2.30258509e+11, 4.60517019e+21, 9.21034037e+41, 1.84206807e+82])

    """)

add_newdoc("gammasgn",
    r"""
    gammasgn(x)

    Sign of the gamma function.

    It is defined as

    .. math::

       \text{gammasgn}(x) =
       \begin{cases}
         +1 & \Gamma(x) > 0 \\
         -1 & \Gamma(x) < 0
       \end{cases}

    where :math:`\Gamma` is the gamma function; see `gamma`. This
    definition is complete since the gamma function is never zero;
    see the discussion after [dlmf]_.

    Parameters
    ----------
    x : array_like
        Real argument

    Returns
    -------
    scalar or ndarray
        Sign of the gamma function

    Notes
    -----
    The gamma function can be computed as ``gammasgn(x) *
    np.exp(gammaln(x))``.

    See Also
    --------
    gamma : the gamma function
    gammaln : log of the absolute value of the gamma function
    loggamma : analytic continuation of the log of the gamma function

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/5.2#E1

    Examples
    --------
    >>> import scipy.special as sc

    It is 1 for `x > 0`.

    >>> sc.gammasgn([1, 2, 3, 4])
    array([1., 1., 1., 1.])

    It alternates between -1 and 1 for negative integers.

    >>> sc.gammasgn([-0.5, -1.5, -2.5, -3.5])
    array([-1.,  1., -1.,  1.])

    It can be used to compute the gamma function.

    >>> x = [1.5, 0.5, -0.5, -1.5]
    >>> sc.gammasgn(x) * np.exp(sc.gammaln(x))
    array([ 0.88622693,  1.77245385, -3.5449077 ,  2.3632718 ])
    >>> sc.gamma(x)
    array([ 0.88622693,  1.77245385, -3.5449077 ,  2.3632718 ])

    """)

add_newdoc("gdtr",
    r"""
    gdtr(a, b, x)

    Gamma distribution cumulative distribution function.

    Returns the integral from zero to `x` of the gamma probability density
    function,

    .. math::

        F = \int_0^x \frac{a^b}{\Gamma(b)} t^{b-1} e^{-at}\,dt,

    where :math:`\Gamma` is the gamma function.

    Parameters
    ----------
    a : array_like
        The rate parameter of the gamma distribution, sometimes denoted
        :math:`\beta` (float).  It is also the reciprocal of the scale
        parameter :math:`\theta`.
    b : array_like
        The shape parameter of the gamma distribution, sometimes denoted
        :math:`\alpha` (float).
    x : array_like
        The quantile (upper limit of integration; float).

    See also
    --------
    gdtrc : 1 - CDF of the gamma distribution.

    Returns
    -------
    F : ndarray
        The CDF of the gamma distribution with parameters `a` and `b`
        evaluated at `x`.

    Notes
    -----
    The evaluation is carried out using the relation to the incomplete gamma
    integral (regularized gamma function).

    Wrapper for the Cephes [1]_ routine `gdtr`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    """)

add_newdoc("gdtrc",
    r"""
    gdtrc(a, b, x)

    Gamma distribution survival function.

    Integral from `x` to infinity of the gamma probability density function,

    .. math::

        F = \int_x^\infty \frac{a^b}{\Gamma(b)} t^{b-1} e^{-at}\,dt,

    where :math:`\Gamma` is the gamma function.

    Parameters
    ----------
    a : array_like
        The rate parameter of the gamma distribution, sometimes denoted
        :math:`\beta` (float). It is also the reciprocal of the scale
        parameter :math:`\theta`.
    b : array_like
        The shape parameter of the gamma distribution, sometimes denoted
        :math:`\alpha` (float).
    x : array_like
        The quantile (lower limit of integration; float).

    Returns
    -------
    F : ndarray
        The survival function of the gamma distribution with parameters `a`
        and `b` evaluated at `x`.

    See Also
    --------
    gdtr, gdtrix

    Notes
    -----
    The evaluation is carried out using the relation to the incomplete gamma
    integral (regularized gamma function).

    Wrapper for the Cephes [1]_ routine `gdtrc`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    """)

add_newdoc("gdtria",
    """
    gdtria(p, b, x, out=None)

    Inverse of `gdtr` vs a.

    Returns the inverse with respect to the parameter `a` of ``p =
    gdtr(a, b, x)``, the cumulative distribution function of the gamma
    distribution.

    Parameters
    ----------
    p : array_like
        Probability values.
    b : array_like
        `b` parameter values of `gdtr(a, b, x)`. `b` is the "shape" parameter
        of the gamma distribution.
    x : array_like
        Nonnegative real values, from the domain of the gamma distribution.
    out : ndarray, optional
        If a fourth argument is given, it must be a numpy.ndarray whose size
        matches the broadcast result of `a`, `b` and `x`.  `out` is then the
        array returned by the function.

    Returns
    -------
    a : ndarray
        Values of the `a` parameter such that `p = gdtr(a, b, x)`.  `1/a`
        is the "scale" parameter of the gamma distribution.

    See Also
    --------
    gdtr : CDF of the gamma distribution.
    gdtrib : Inverse with respect to `b` of `gdtr(a, b, x)`.
    gdtrix : Inverse with respect to `x` of `gdtr(a, b, x)`.

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdfgam`.

    The cumulative distribution function `p` is computed using a routine by
    DiDinato and Morris [2]_. Computation of `a` involves a search for a value
    that produces the desired value of `p`. The search relies on the
    monotonicity of `p` with `a`.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] DiDinato, A. R. and Morris, A. H.,
           Computation of the incomplete gamma function ratios and their
           inverse.  ACM Trans. Math. Softw. 12 (1986), 377-393.

    Examples
    --------
    First evaluate `gdtr`.

    >>> from scipy.special import gdtr, gdtria
    >>> p = gdtr(1.2, 3.4, 5.6)
    >>> print(p)
    0.94378087442

    Verify the inverse.

    >>> gdtria(p, 3.4, 5.6)
    1.2
    """)

add_newdoc("gdtrib",
    """
    gdtrib(a, p, x, out=None)

    Inverse of `gdtr` vs b.

    Returns the inverse with respect to the parameter `b` of ``p =
    gdtr(a, b, x)``, the cumulative distribution function of the gamma
    distribution.

    Parameters
    ----------
    a : array_like
        `a` parameter values of `gdtr(a, b, x)`. `1/a` is the "scale"
        parameter of the gamma distribution.
    p : array_like
        Probability values.
    x : array_like
        Nonnegative real values, from the domain of the gamma distribution.
    out : ndarray, optional
        If a fourth argument is given, it must be a numpy.ndarray whose size
        matches the broadcast result of `a`, `b` and `x`.  `out` is then the
        array returned by the function.

    Returns
    -------
    b : ndarray
        Values of the `b` parameter such that `p = gdtr(a, b, x)`.  `b` is
        the "shape" parameter of the gamma distribution.

    See Also
    --------
    gdtr : CDF of the gamma distribution.
    gdtria : Inverse with respect to `a` of `gdtr(a, b, x)`.
    gdtrix : Inverse with respect to `x` of `gdtr(a, b, x)`.

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdfgam`.

    The cumulative distribution function `p` is computed using a routine by
    DiDinato and Morris [2]_. Computation of `b` involves a search for a value
    that produces the desired value of `p`. The search relies on the
    monotonicity of `p` with `b`.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] DiDinato, A. R. and Morris, A. H.,
           Computation of the incomplete gamma function ratios and their
           inverse.  ACM Trans. Math. Softw. 12 (1986), 377-393.

    Examples
    --------
    First evaluate `gdtr`.

    >>> from scipy.special import gdtr, gdtrib
    >>> p = gdtr(1.2, 3.4, 5.6)
    >>> print(p)
    0.94378087442

    Verify the inverse.

    >>> gdtrib(1.2, p, 5.6)
    3.3999999999723882
    """)

add_newdoc("gdtrix",
    """
    gdtrix(a, b, p, out=None)

    Inverse of `gdtr` vs x.

    Returns the inverse with respect to the parameter `x` of ``p =
    gdtr(a, b, x)``, the cumulative distribution function of the gamma
    distribution. This is also known as the pth quantile of the
    distribution.

    Parameters
    ----------
    a : array_like
        `a` parameter values of `gdtr(a, b, x)`. `1/a` is the "scale"
        parameter of the gamma distribution.
    b : array_like
        `b` parameter values of `gdtr(a, b, x)`. `b` is the "shape" parameter
        of the gamma distribution.
    p : array_like
        Probability values.
    out : ndarray, optional
        If a fourth argument is given, it must be a numpy.ndarray whose size
        matches the broadcast result of `a`, `b` and `x`. `out` is then the
        array returned by the function.

    Returns
    -------
    x : ndarray
        Values of the `x` parameter such that `p = gdtr(a, b, x)`.

    See Also
    --------
    gdtr : CDF of the gamma distribution.
    gdtria : Inverse with respect to `a` of `gdtr(a, b, x)`.
    gdtrib : Inverse with respect to `b` of `gdtr(a, b, x)`.

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdfgam`.

    The cumulative distribution function `p` is computed using a routine by
    DiDinato and Morris [2]_. Computation of `x` involves a search for a value
    that produces the desired value of `p`. The search relies on the
    monotonicity of `p` with `x`.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] DiDinato, A. R. and Morris, A. H.,
           Computation of the incomplete gamma function ratios and their
           inverse.  ACM Trans. Math. Softw. 12 (1986), 377-393.

    Examples
    --------
    First evaluate `gdtr`.

    >>> from scipy.special import gdtr, gdtrix
    >>> p = gdtr(1.2, 3.4, 5.6)
    >>> print(p)
    0.94378087442

    Verify the inverse.

    >>> gdtrix(1.2, 3.4, p)
    5.5999999999999996
    """)

add_newdoc("hankel1",
    r"""
    hankel1(v, z)

    Hankel function of the first kind

    Parameters
    ----------
    v : array_like
        Order (float).
    z : array_like
        Argument (float or complex).

    Returns
    -------
    out : Values of the Hankel function of the first kind.

    Notes
    -----
    A wrapper for the AMOS [1]_ routine `zbesh`, which carries out the
    computation using the relation,

    .. math:: H^{(1)}_v(z) = \frac{2}{\imath\pi} \exp(-\imath \pi v/2) K_v(z \exp(-\imath\pi/2))

    where :math:`K_v` is the modified Bessel function of the second kind.
    For negative orders, the relation

    .. math:: H^{(1)}_{-v}(z) = H^{(1)}_v(z) \exp(\imath\pi v)

    is used.

    See also
    --------
    hankel1e : this function with leading exponential behavior stripped off.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    """)

add_newdoc("hankel1e",
    r"""
    hankel1e(v, z)

    Exponentially scaled Hankel function of the first kind

    Defined as::

        hankel1e(v, z) = hankel1(v, z) * exp(-1j * z)

    Parameters
    ----------
    v : array_like
        Order (float).
    z : array_like
        Argument (float or complex).

    Returns
    -------
    out : Values of the exponentially scaled Hankel function.

    Notes
    -----
    A wrapper for the AMOS [1]_ routine `zbesh`, which carries out the
    computation using the relation,

    .. math:: H^{(1)}_v(z) = \frac{2}{\imath\pi} \exp(-\imath \pi v/2) K_v(z \exp(-\imath\pi/2))

    where :math:`K_v` is the modified Bessel function of the second kind.
    For negative orders, the relation

    .. math:: H^{(1)}_{-v}(z) = H^{(1)}_v(z) \exp(\imath\pi v)

    is used.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    """)

add_newdoc("hankel2",
    r"""
    hankel2(v, z)

    Hankel function of the second kind

    Parameters
    ----------
    v : array_like
        Order (float).
    z : array_like
        Argument (float or complex).

    Returns
    -------
    out : Values of the Hankel function of the second kind.

    Notes
    -----
    A wrapper for the AMOS [1]_ routine `zbesh`, which carries out the
    computation using the relation,

    .. math:: H^{(2)}_v(z) = -\frac{2}{\imath\pi} \exp(\imath \pi v/2) K_v(z \exp(\imath\pi/2))

    where :math:`K_v` is the modified Bessel function of the second kind.
    For negative orders, the relation

    .. math:: H^{(2)}_{-v}(z) = H^{(2)}_v(z) \exp(-\imath\pi v)

    is used.

    See also
    --------
    hankel2e : this function with leading exponential behavior stripped off.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    """)

add_newdoc("hankel2e",
    r"""
    hankel2e(v, z)

    Exponentially scaled Hankel function of the second kind

    Defined as::

        hankel2e(v, z) = hankel2(v, z) * exp(1j * z)

    Parameters
    ----------
    v : array_like
        Order (float).
    z : array_like
        Argument (float or complex).

    Returns
    -------
    out : Values of the exponentially scaled Hankel function of the second kind.

    Notes
    -----
    A wrapper for the AMOS [1]_ routine `zbesh`, which carries out the
    computation using the relation,

    .. math:: H^{(2)}_v(z) = -\frac{2}{\imath\pi} \exp(\frac{\imath \pi v}{2}) K_v(z exp(\frac{\imath\pi}{2}))

    where :math:`K_v` is the modified Bessel function of the second kind.
    For negative orders, the relation

    .. math:: H^{(2)}_{-v}(z) = H^{(2)}_v(z) \exp(-\imath\pi v)

    is used.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/

    """)

add_newdoc("huber",
    r"""
    huber(delta, r)

    Huber loss function.

    .. math:: \text{huber}(\delta, r) = \begin{cases} \infty & \delta < 0  \\ \frac{1}{2}r^2 & 0 \le \delta, | r | \le \delta \\ \delta ( |r| - \frac{1}{2}\delta ) & \text{otherwise} \end{cases}

    Parameters
    ----------
    delta : ndarray
        Input array, indicating the quadratic vs. linear loss changepoint.
    r : ndarray
        Input array, possibly representing residuals.

    Returns
    -------
    res : ndarray
        The computed Huber loss function values.

    Notes
    -----
    This function is convex in r.

    .. versionadded:: 0.15.0

    """)

add_newdoc("hyp0f1",
    r"""
    hyp0f1(v, z, out=None)

    Confluent hypergeometric limit function 0F1.

    Parameters
    ----------
    v : array_like
        Real-valued parameter
    z : array_like
        Real- or complex-valued argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        The confluent hypergeometric limit function

    Notes
    -----
    This function is defined as:

    .. math:: _0F_1(v, z) = \sum_{k=0}^{\infty}\frac{z^k}{(v)_k k!}.

    It's also the limit as :math:`q \to \infty` of :math:`_1F_1(q; v; z/q)`,
    and satisfies the differential equation :math:`f''(z) + vf'(z) =
    f(z)`. See [1]_ for more information.

    References
    ----------
    .. [1] Wolfram MathWorld, "Confluent Hypergeometric Limit Function",
           http://mathworld.wolfram.com/ConfluentHypergeometricLimitFunction.html

    Examples
    --------
    >>> import scipy.special as sc

    It is one when `z` is zero.

    >>> sc.hyp0f1(1, 0)
    1.0

    It is the limit of the confluent hypergeometric function as `q`
    goes to infinity.

    >>> q = np.array([1, 10, 100, 1000])
    >>> v = 1
    >>> z = 1
    >>> sc.hyp1f1(q, v, z / q)
    array([2.71828183, 2.31481985, 2.28303778, 2.27992985])
    >>> sc.hyp0f1(v, z)
    2.2795853023360673

    It is related to Bessel functions.

    >>> n = 1
    >>> x = np.linspace(0, 1, 5)
    >>> sc.jv(n, x)
    array([0.        , 0.12402598, 0.24226846, 0.3492436 , 0.44005059])
    >>> (0.5 * x)**n / sc.factorial(n) * sc.hyp0f1(n + 1, -0.25 * x**2)
    array([0.        , 0.12402598, 0.24226846, 0.3492436 , 0.44005059])

    """)

add_newdoc("hyp1f1",
    r"""
    hyp1f1(a, b, x, out=None)

    Confluent hypergeometric function 1F1.

    The confluent hypergeometric function is defined by the series

    .. math::

       {}_1F_1(a; b; x) = \sum_{k = 0}^\infty \frac{(a)_k}{(b)_k k!} x^k.

    See [dlmf]_ for more details. Here :math:`(\cdot)_k` is the
    Pochhammer symbol; see `poch`.

    Parameters
    ----------
    a, b : array_like
        Real parameters
    x : array_like
        Real or complex argument
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the confluent hypergeometric function

    See also
    --------
    hyperu : another confluent hypergeometric function
    hyp0f1 : confluent hypergeometric limit function
    hyp2f1 : Gaussian hypergeometric function

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/13.2#E2

    Examples
    --------
    >>> import scipy.special as sc

    It is one when `x` is zero:

    >>> sc.hyp1f1(0.5, 0.5, 0)
    1.0

    It is singular when `b` is a nonpositive integer.

    >>> sc.hyp1f1(0.5, -1, 0)
    inf

    It is a polynomial when `a` is a nonpositive integer.

    >>> a, b, x = -1, 0.5, np.array([1.0, 2.0, 3.0, 4.0])
    >>> sc.hyp1f1(a, b, x)
    array([-1., -3., -5., -7.])
    >>> 1 + (a / b) * x
    array([-1., -3., -5., -7.])

    It reduces to the exponential function when `a = b`.

    >>> sc.hyp1f1(2, 2, [1, 2, 3, 4])
    array([ 2.71828183,  7.3890561 , 20.08553692, 54.59815003])
    >>> np.exp([1, 2, 3, 4])
    array([ 2.71828183,  7.3890561 , 20.08553692, 54.59815003])

    """)

add_newdoc("hyp2f1",
    r"""
    hyp2f1(a, b, c, z)

    Gauss hypergeometric function 2F1(a, b; c; z)

    Parameters
    ----------
    a, b, c : array_like
        Arguments, should be real-valued.
    z : array_like
        Argument, real or complex.

    Returns
    -------
    hyp2f1 : scalar or ndarray
        The values of the gaussian hypergeometric function.

    See also
    --------
    hyp0f1 : confluent hypergeometric limit function.
    hyp1f1 : Kummer's (confluent hypergeometric) function.

    Notes
    -----
    This function is defined for :math:`|z| < 1` as

    .. math::

       \mathrm{hyp2f1}(a, b, c, z) = \sum_{n=0}^\infty
       \frac{(a)_n (b)_n}{(c)_n}\frac{z^n}{n!},

    and defined on the rest of the complex z-plane by analytic
    continuation [1]_.
    Here :math:`(\cdot)_n` is the Pochhammer symbol; see `poch`. When
    :math:`n` is an integer the result is a polynomial of degree :math:`n`.

    The implementation for complex values of ``z`` is described in [2]_.

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/15.2
    .. [2] S. Zhang and J.M. Jin, "Computation of Special Functions", Wiley 1996
    .. [3] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    Examples
    --------
    >>> import scipy.special as sc

    It has poles when `c` is a negative integer.

    >>> sc.hyp2f1(1, 1, -2, 1)
    inf

    It is a polynomial when `a` or `b` is a negative integer.

    >>> a, b, c = -1, 1, 1.5
    >>> z = np.linspace(0, 1, 5)
    >>> sc.hyp2f1(a, b, c, z)
    array([1.        , 0.83333333, 0.66666667, 0.5       , 0.33333333])
    >>> 1 + a * b * z / c
    array([1.        , 0.83333333, 0.66666667, 0.5       , 0.33333333])

    It is symmetric in `a` and `b`.

    >>> a = np.linspace(0, 1, 5)
    >>> b = np.linspace(0, 1, 5)
    >>> sc.hyp2f1(a, b, 1, 0.5)
    array([1.        , 1.03997334, 1.1803406 , 1.47074441, 2.        ])
    >>> sc.hyp2f1(b, a, 1, 0.5)
    array([1.        , 1.03997334, 1.1803406 , 1.47074441, 2.        ])

    It contains many other functions as special cases.

    >>> z = 0.5
    >>> sc.hyp2f1(1, 1, 2, z)
    1.3862943611198901
    >>> -np.log(1 - z) / z
    1.3862943611198906

    >>> sc.hyp2f1(0.5, 1, 1.5, z**2)
    1.098612288668109
    >>> np.log((1 + z) / (1 - z)) / (2 * z)
    1.0986122886681098

    >>> sc.hyp2f1(0.5, 1, 1.5, -z**2)
    0.9272952180016117
    >>> np.arctan(z) / z
    0.9272952180016123

    """)

add_newdoc("hyperu",
    r"""
    hyperu(a, b, x, out=None)

    Confluent hypergeometric function U

    It is defined as the solution to the equation

    .. math::

       x \frac{d^2w}{dx^2} + (b - x) \frac{dw}{dx} - aw = 0

    which satisfies the property

    .. math::

       U(a, b, x) \sim x^{-a}

    as :math:`x \to \infty`. See [dlmf]_ for more details.

    Parameters
    ----------
    a, b : array_like
        Real-valued parameters
    x : array_like
        Real-valued argument
    out : ndarray
        Optional output array for the function values

    Returns
    -------
    scalar or ndarray
        Values of `U`

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematics Functions
              https://dlmf.nist.gov/13.2#E6

    Examples
    --------
    >>> import scipy.special as sc

    It has a branch cut along the negative `x` axis.

    >>> x = np.linspace(-0.1, -10, 5)
    >>> sc.hyperu(1, 1, x)
    array([nan, nan, nan, nan, nan])

    It approaches zero as `x` goes to infinity.

    >>> x = np.array([1, 10, 100])
    >>> sc.hyperu(1, 1, x)
    array([0.59634736, 0.09156333, 0.00990194])

    It satisfies Kummer's transformation.

    >>> a, b, x = 2, 1, 1
    >>> sc.hyperu(a, b, x)
    0.1926947246463881
    >>> x**(1 - b) * sc.hyperu(a - b + 1, 2 - b, x)
    0.1926947246463881

    """)

add_newdoc("i0",
    r"""
    i0(x)

    Modified Bessel function of order 0.

    Defined as,

    .. math::
        I_0(x) = \sum_{k=0}^\infty \frac{(x^2/4)^k}{(k!)^2} = J_0(\imath x),

    where :math:`J_0` is the Bessel function of the first kind of order 0.

    Parameters
    ----------
    x : array_like
        Argument (float)

    Returns
    -------
    I : ndarray
        Value of the modified Bessel function of order 0 at `x`.

    Notes
    -----
    The range is partitioned into the two intervals [0, 8] and (8, infinity).
    Chebyshev polynomial expansions are employed in each interval.

    This function is a wrapper for the Cephes [1]_ routine `i0`.

    See also
    --------
    iv
    i0e

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

add_newdoc("i0e",
    """
    i0e(x)

    Exponentially scaled modified Bessel function of order 0.

    Defined as::

        i0e(x) = exp(-abs(x)) * i0(x).

    Parameters
    ----------
    x : array_like
        Argument (float)

    Returns
    -------
    I : ndarray
        Value of the exponentially scaled modified Bessel function of order 0
        at `x`.

    Notes
    -----
    The range is partitioned into the two intervals [0, 8] and (8, infinity).
    Chebyshev polynomial expansions are employed in each interval. The
    polynomial expansions used are the same as those in `i0`, but
    they are not multiplied by the dominant exponential factor.

    This function is a wrapper for the Cephes [1]_ routine `i0e`.

    See also
    --------
    iv
    i0

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

add_newdoc("i1",
    r"""
    i1(x)

    Modified Bessel function of order 1.

    Defined as,

    .. math::
        I_1(x) = \frac{1}{2}x \sum_{k=0}^\infty \frac{(x^2/4)^k}{k! (k + 1)!}
               = -\imath J_1(\imath x),

    where :math:`J_1` is the Bessel function of the first kind of order 1.

    Parameters
    ----------
    x : array_like
        Argument (float)

    Returns
    -------
    I : ndarray
        Value of the modified Bessel function of order 1 at `x`.

    Notes
    -----
    The range is partitioned into the two intervals [0, 8] and (8, infinity).
    Chebyshev polynomial expansions are employed in each interval.

    This function is a wrapper for the Cephes [1]_ routine `i1`.

    See also
    --------
    iv
    i1e

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

add_newdoc("i1e",
    """
    i1e(x)

    Exponentially scaled modified Bessel function of order 1.

    Defined as::

        i1e(x) = exp(-abs(x)) * i1(x)

    Parameters
    ----------
    x : array_like
        Argument (float)

    Returns
    -------
    I : ndarray
        Value of the exponentially scaled modified Bessel function of order 1
        at `x`.

    Notes
    -----
    The range is partitioned into the two intervals [0, 8] and (8, infinity).
    Chebyshev polynomial expansions are employed in each interval. The
    polynomial expansions used are the same as those in `i1`, but
    they are not multiplied by the dominant exponential factor.

    This function is a wrapper for the Cephes [1]_ routine `i1e`.

    See also
    --------
    iv
    i1

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

add_newdoc("_igam_fac",
    """
    Internal function, do not use.
    """)

add_newdoc("it2i0k0",
    r"""
    it2i0k0(x, out=None)

    Integrals related to modified Bessel functions of order 0.

    Computes the integrals

    .. math::

        \int_0^x \frac{I_0(t) - 1}{t} dt \\
        \int_x^\infty \frac{K_0(t)}{t} dt.

    Parameters
    ----------
    x : array_like
        Values at which to evaluate the integrals.
    out : tuple of ndarrays, optional
        Optional output arrays for the function results.

    Returns
    -------
    ii0 : scalar or ndarray
        The integral for `i0`
    ik0 : scalar or ndarray
        The integral for `k0`

    """)

add_newdoc("it2j0y0",
    r"""
    it2j0y0(x, out=None)

    Integrals related to Bessel functions of the first kind of order 0.

    Computes the integrals

    .. math::

        \int_0^x \frac{1 - J_0(t)}{t} dt \\
        \int_x^\infty \frac{Y_0(t)}{t} dt.

    For more on :math:`J_0` and :math:`Y_0` see `j0` and `y0`.

    Parameters
    ----------
    x : array_like
        Values at which to evaluate the integrals.
    out : tuple of ndarrays, optional
        Optional output arrays for the function results.

    Returns
    -------
    ij0 : scalar or ndarray
        The integral for `j0`
    iy0 : scalar or ndarray
        The integral for `y0`

    """)

add_newdoc("it2struve0",
    r"""
    it2struve0(x)

    Integral related to the Struve function of order 0.

    Returns the integral,

    .. math::
        \int_x^\infty \frac{H_0(t)}{t}\,dt

    where :math:`H_0` is the Struve function of order 0.

    Parameters
    ----------
    x : array_like
        Lower limit of integration.

    Returns
    -------
    I : ndarray
        The value of the integral.

    See also
    --------
    struve

    Notes
    -----
    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
    Jin [1]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
    """)

add_newdoc("itairy",
    """
    itairy(x)

    Integrals of Airy functions

    Calculates the integrals of Airy functions from 0 to `x`.

    Parameters
    ----------

    x: array_like
        Upper limit of integration (float).

    Returns
    -------
    Apt
        Integral of Ai(t) from 0 to x.
    Bpt
        Integral of Bi(t) from 0 to x.
    Ant
        Integral of Ai(-t) from 0 to x.
    Bnt
        Integral of Bi(-t) from 0 to x.

    Notes
    -----

    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
    Jin [1]_.

    References
    ----------

    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
    """)

add_newdoc("iti0k0",
    r"""
    iti0k0(x, out=None)

    Integrals of modified Bessel functions of order 0.

    Computes the integrals

    .. math::

        \int_0^x I_0(t) dt \\
        \int_0^x K_0(t) dt.

    For more on :math:`I_0` and :math:`K_0` see `i0` and `k0`.

    Parameters
    ----------
    x : array_like
        Values at which to evaluate the integrals.
    out : tuple of ndarrays, optional
        Optional output arrays for the function results.

    Returns
    -------
    ii0 : scalar or ndarray
        The integral for `i0`
    ik0 : scalar or ndarray
        The integral for `k0`
    """)

add_newdoc("itj0y0",
    r"""
    itj0y0(x, out=None)

    Integrals of Bessel functions of the first kind of order 0.

    Computes the integrals

    .. math::

        \int_0^x J_0(t) dt \\
        \int_0^x Y_0(t) dt.

    For more on :math:`J_0` and :math:`Y_0` see `j0` and `y0`.

    Parameters
    ----------
    x : array_like
        Values at which to evaluate the integrals.
    out : tuple of ndarrays, optional
        Optional output arrays for the function results.

    Returns
    -------
    ij0 : scalar or ndarray
        The integral of `j0`
    iy0 : scalar or ndarray
        The integral of `y0`

    """)

add_newdoc("itmodstruve0",
    r"""
    itmodstruve0(x)

    Integral of the modified Struve function of order 0.

    .. math::
        I = \int_0^x L_0(t)\,dt

    Parameters
    ----------
    x : array_like
        Upper limit of integration (float).

    Returns
    -------
    I : ndarray
        The integral of :math:`L_0` from 0 to `x`.

    Notes
    -----
    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
    Jin [1]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """)

add_newdoc("itstruve0",
    r"""
    itstruve0(x)

    Integral of the Struve function of order 0.

    .. math::
        I = \int_0^x H_0(t)\,dt

    Parameters
    ----------
    x : array_like
        Upper limit of integration (float).

    Returns
    -------
    I : ndarray
        The integral of :math:`H_0` from 0 to `x`.

    See also
    --------
    struve

    Notes
    -----
    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
    Jin [1]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html

    """)

add_newdoc("iv",
    r"""
    iv(v, z)

    Modified Bessel function of the first kind of real order.

    Parameters
    ----------
    v : array_like
        Order. If `z` is of real type and negative, `v` must be integer
        valued.
    z : array_like of float or complex
        Argument.

    Returns
    -------
    out : ndarray
        Values of the modified Bessel function.

    Notes
    -----
    For real `z` and :math:`v \in [-50, 50]`, the evaluation is carried out
    using Temme's method [1]_.  For larger orders, uniform asymptotic
    expansions are applied.

    For complex `z` and positive `v`, the AMOS [2]_ `zbesi` routine is
    called. It uses a power series for small `z`, the asymptotic expansion
    for large `abs(z)`, the Miller algorithm normalized by the Wronskian
    and a Neumann series for intermediate magnitudes, and the uniform
    asymptotic expansions for :math:`I_v(z)` and :math:`J_v(z)` for large
    orders. Backward recurrence is used to generate sequences or reduce
    orders when necessary.

    The calculations above are done in the right half plane and continued
    into the left half plane by the formula,

    .. math:: I_v(z \exp(\pm\imath\pi)) = \exp(\pm\pi v) I_v(z)

    (valid when the real part of `z` is positive).  For negative `v`, the
    formula

    .. math:: I_{-v}(z) = I_v(z) + \frac{2}{\pi} \sin(\pi v) K_v(z)

    is used, where :math:`K_v(z)` is the modified Bessel function of the
    second kind, evaluated using the AMOS routine `zbesk`.

    See also
    --------
    kve : This function with leading exponential behavior stripped off.

    References
    ----------
    .. [1] Temme, Journal of Computational Physics, vol 21, 343 (1976)
    .. [2] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    """)

add_newdoc("ive",
    r"""
    ive(v, z)

    Exponentially scaled modified Bessel function of the first kind

    Defined as::

        ive(v, z) = iv(v, z) * exp(-abs(z.real))

    Parameters
    ----------
    v : array_like of float
        Order.
    z : array_like of float or complex
        Argument.

    Returns
    -------
    out : ndarray
        Values of the exponentially scaled modified Bessel function.

    Notes
    -----
    For positive `v`, the AMOS [1]_ `zbesi` routine is called. It uses a
    power series for small `z`, the asymptotic expansion for large
    `abs(z)`, the Miller algorithm normalized by the Wronskian and a
    Neumann series for intermediate magnitudes, and the uniform asymptotic
    expansions for :math:`I_v(z)` and :math:`J_v(z)` for large orders.
    Backward recurrence is used to generate sequences or reduce orders when
    necessary.

    The calculations above are done in the right half plane and continued
    into the left half plane by the formula,

    .. math:: I_v(z \exp(\pm\imath\pi)) = \exp(\pm\pi v) I_v(z)

    (valid when the real part of `z` is positive).  For negative `v`, the
    formula

    .. math:: I_{-v}(z) = I_v(z) + \frac{2}{\pi} \sin(\pi v) K_v(z)

    is used, where :math:`K_v(z)` is the modified Bessel function of the
    second kind, evaluated using the AMOS routine `zbesk`.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    """)

add_newdoc("j0",
    r"""
    j0(x)

    Bessel function of the first kind of order 0.

    Parameters
    ----------
    x : array_like
        Argument (float).

    Returns
    -------
    J : ndarray
        Value of the Bessel function of the first kind of order 0 at `x`.

    Notes
    -----
    The domain is divided into the intervals [0, 5] and (5, infinity). In the
    first interval the following rational approximation is used:

    .. math::

        J_0(x) \approx (w - r_1^2)(w - r_2^2) \frac{P_3(w)}{Q_8(w)},

    where :math:`w = x^2` and :math:`r_1`, :math:`r_2` are the zeros of
    :math:`J_0`, and :math:`P_3` and :math:`Q_8` are polynomials of degrees 3
    and 8, respectively.

    In the second interval, the Hankel asymptotic expansion is employed with
    two rational functions of degree 6/6 and 7/7.

    This function is a wrapper for the Cephes [1]_ routine `j0`.
    It should not be confused with the spherical Bessel functions (see
    `spherical_jn`).

    See also
    --------
    jv : Bessel function of real order and complex argument.
    spherical_jn : spherical Bessel functions.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

add_newdoc("j1",
    """
    j1(x)

    Bessel function of the first kind of order 1.

    Parameters
    ----------
    x : array_like
        Argument (float).

    Returns
    -------
    J : ndarray
        Value of the Bessel function of the first kind of order 1 at `x`.

    Notes
    -----
    The domain is divided into the intervals [0, 8] and (8, infinity). In the
    first interval a 24 term Chebyshev expansion is used. In the second, the
    asymptotic trigonometric representation is employed using two rational
    functions of degree 5/5.

    This function is a wrapper for the Cephes [1]_ routine `j1`.
    It should not be confused with the spherical Bessel functions (see
    `spherical_jn`).

    See also
    --------
    jv
    spherical_jn : spherical Bessel functions.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    """)

add_newdoc("jn",
    """
    jn(n, x)

    Bessel function of the first kind of integer order and real argument.

    Notes
    -----
    `jn` is an alias of `jv`.
    Not to be confused with the spherical Bessel functions (see `spherical_jn`).

    See also
    --------
    jv
    spherical_jn : spherical Bessel functions.

    """)

add_newdoc("jv",
    r"""
    jv(v, z)

    Bessel function of the first kind of real order and complex argument.

    Parameters
    ----------
    v : array_like
        Order (float).
    z : array_like
        Argument (float or complex).

    Returns
    -------
    J : ndarray
        Value of the Bessel function, :math:`J_v(z)`.

    Notes
    -----
    For positive `v` values, the computation is carried out using the AMOS
    [1]_ `zbesj` routine, which exploits the connection to the modified
    Bessel function :math:`I_v`,

    .. math::
        J_v(z) = \exp(v\pi\imath/2) I_v(-\imath z)\qquad (\Im z > 0)

        J_v(z) = \exp(-v\pi\imath/2) I_v(\imath z)\qquad (\Im z < 0)

    For negative `v` values the formula,

    .. math:: J_{-v}(z) = J_v(z) \cos(\pi v) - Y_v(z) \sin(\pi v)

    is used, where :math:`Y_v(z)` is the Bessel function of the second
    kind, computed using the AMOS routine `zbesy`.  Note that the second
    term is exactly zero for integer `v`; to improve accuracy the second
    term is explicitly omitted for `v` values such that `v = floor(v)`.

    Not to be confused with the spherical Bessel functions (see `spherical_jn`).

    See also
    --------
    jve : :math:`J_v` with leading exponential behavior stripped off.
    spherical_jn : spherical Bessel functions.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    """)

add_newdoc("jve",
    r"""
    jve(v, z)

    Exponentially scaled Bessel function of order `v`.

    Defined as::

        jve(v, z) = jv(v, z) * exp(-abs(z.imag))

    Parameters
    ----------
    v : array_like
        Order (float).
    z : array_like
        Argument (float or complex).

    Returns
    -------
    J : ndarray
        Value of the exponentially scaled Bessel function.

    Notes
    -----
    For positive `v` values, the computation is carried out using the AMOS
    [1]_ `zbesj` routine, which exploits the connection to the modified
    Bessel function :math:`I_v`,

    .. math::
        J_v(z) = \exp(v\pi\imath/2) I_v(-\imath z)\qquad (\Im z > 0)

        J_v(z) = \exp(-v\pi\imath/2) I_v(\imath z)\qquad (\Im z < 0)

    For negative `v` values the formula,

    .. math:: J_{-v}(z) = J_v(z) \cos(\pi v) - Y_v(z) \sin(\pi v)

    is used, where :math:`Y_v(z)` is the Bessel function of the second
    kind, computed using the AMOS routine `zbesy`.  Note that the second
    term is exactly zero for integer `v`; to improve accuracy the second
    term is explicitly omitted for `v` values such that `v = floor(v)`.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    """)

add_newdoc("k0",
    r"""
    k0(x)

    Modified Bessel function of the second kind of order 0, :math:`K_0`.

    This function is also sometimes referred to as the modified Bessel
    function of the third kind of order 0.

    Parameters
    ----------
    x : array_like
        Argument (float).

    Returns
    -------
    K : ndarray
        Value of the modified Bessel function :math:`K_0` at `x`.

    Notes
    -----
    The range is partitioned into the two intervals [0, 2] and (2, infinity).
    Chebyshev polynomial expansions are employed in each interval.

    This function is a wrapper for the Cephes [1]_ routine `k0`.

    See also
    --------
    kv
    k0e

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

add_newdoc("k0e",
    """
    k0e(x)

    Exponentially scaled modified Bessel function K of order 0

    Defined as::

        k0e(x) = exp(x) * k0(x).

    Parameters
    ----------
    x : array_like
        Argument (float)

    Returns
    -------
    K : ndarray
        Value of the exponentially scaled modified Bessel function K of order
        0 at `x`.

    Notes
    -----
    The range is partitioned into the two intervals [0, 2] and (2, infinity).
    Chebyshev polynomial expansions are employed in each interval.

    This function is a wrapper for the Cephes [1]_ routine `k0e`.

    See also
    --------
    kv
    k0

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

add_newdoc("k1",
    """
    k1(x)

    Modified Bessel function of the second kind of order 1, :math:`K_1(x)`.

    Parameters
    ----------
    x : array_like
        Argument (float)

    Returns
    -------
    K : ndarray
        Value of the modified Bessel function K of order 1 at `x`.

    Notes
    -----
    The range is partitioned into the two intervals [0, 2] and (2, infinity).
    Chebyshev polynomial expansions are employed in each interval.

    This function is a wrapper for the Cephes [1]_ routine `k1`.

    See also
    --------
    kv
    k1e

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

add_newdoc("k1e",
    """
    k1e(x)

    Exponentially scaled modified Bessel function K of order 1

    Defined as::

        k1e(x) = exp(x) * k1(x)

    Parameters
    ----------
    x : array_like
        Argument (float)

    Returns
    -------
    K : ndarray
        Value of the exponentially scaled modified Bessel function K of order
        1 at `x`.

    Notes
    -----
    The range is partitioned into the two intervals [0, 2] and (2, infinity).
    Chebyshev polynomial expansions are employed in each interval.

    This function is a wrapper for the Cephes [1]_ routine `k1e`.

    See also
    --------
    kv
    k1

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

add_newdoc("kei",
    r"""
    kei(x, out=None)

    Kelvin function kei.

    Defined as

    .. math::

        \mathrm{kei}(x) = \Im[K_0(x e^{\pi i / 4})]

    where :math:`K_0` is the modified Bessel function of the second
    kind (see `kv`). See [dlmf]_ for more details.

    Parameters
    ----------
    x : array_like
        Real argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of the Kelvin function.

    See Also
    --------
    ker : the corresponding real part
    keip : the derivative of kei
    kv : modified Bessel function of the second kind

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/10.61

    Examples
    --------
    It can be expressed using the modified Bessel function of the
    second kind.

    >>> import scipy.special as sc
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> sc.kv(0, x * np.exp(np.pi * 1j / 4)).imag
    array([-0.49499464, -0.20240007, -0.05112188,  0.0021984 ])
    >>> sc.kei(x)
    array([-0.49499464, -0.20240007, -0.05112188,  0.0021984 ])

    """)

add_newdoc("keip",
    r"""
    keip(x, out=None)

    Derivative of the Kelvin function kei.

    Parameters
    ----------
    x : array_like
        Real argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        The values of the derivative of kei.

    See Also
    --------
    kei

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/10#PT5

    """)

add_newdoc("kelvin",
    """
    kelvin(x)

    Kelvin functions as complex numbers

    Returns
    -------
    Be, Ke, Bep, Kep
        The tuple (Be, Ke, Bep, Kep) contains complex numbers
        representing the real and imaginary Kelvin functions and their
        derivatives evaluated at `x`.  For example, kelvin(x)[0].real =
        ber x and kelvin(x)[0].imag = bei x with similar relationships
        for ker and kei.
    """)

add_newdoc("ker",
    r"""
    ker(x, out=None)

    Kelvin function ker.

    Defined as

    .. math::

        \mathrm{ker}(x) = \Re[K_0(x e^{\pi i / 4})]

    Where :math:`K_0` is the modified Bessel function of the second
    kind (see `kv`). See [dlmf]_ for more details.

    Parameters
    ----------
    x : array_like
        Real argument.
    out : ndarray, optional
        Optional output array for the function results.

    See Also
    --------
    kei : the corresponding imaginary part
    kerp : the derivative of ker
    kv : modified Bessel function of the second kind

    Returns
    -------
    scalar or ndarray
        Values of the Kelvin function.

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/10.61

    Examples
    --------
    It can be expressed using the modified Bessel function of the
    second kind.

    >>> import scipy.special as sc
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> sc.kv(0, x * np.exp(np.pi * 1j / 4)).real
    array([ 0.28670621, -0.04166451, -0.06702923, -0.03617885])
    >>> sc.ker(x)
    array([ 0.28670621, -0.04166451, -0.06702923, -0.03617885])

    """)

add_newdoc("kerp",
    r"""
    kerp(x, out=None)

    Derivative of the Kelvin function ker.

    Parameters
    ----------
    x : array_like
        Real argument.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of the derivative of ker.

    See Also
    --------
    ker

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/10#PT5

    """)

add_newdoc("kl_div",
    r"""
    kl_div(x, y, out=None)

    Elementwise function for computing Kullback-Leibler divergence.

    .. math::

        \mathrm{kl\_div}(x, y) =
          \begin{cases}
            x \log(x / y) - x + y & x > 0, y > 0 \\
            y & x = 0, y \ge 0 \\
            \infty & \text{otherwise}
          \end{cases}

    Parameters
    ----------
    x, y : array_like
        Real arguments
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Values of the Kullback-Liebler divergence.

    See Also
    --------
    entr, rel_entr

    Notes
    -----
    .. versionadded:: 0.15.0

    This function is non-negative and is jointly convex in `x` and `y`.

    The origin of this function is in convex programming; see [1]_ for
    details. This is why the the function contains the extra :math:`-x
    + y` terms over what might be expected from the Kullback-Leibler
    divergence. For a version of the function without the extra terms,
    see `rel_entr`.

    References
    ----------
    .. [1] Grant, Boyd, and Ye, "CVX: Matlab Software for Disciplined Convex
        Programming", http://cvxr.com/cvx/


    """)

add_newdoc("kn",
    r"""
    kn(n, x)

    Modified Bessel function of the second kind of integer order `n`

    Returns the modified Bessel function of the second kind for integer order
    `n` at real `z`.

    These are also sometimes called functions of the third kind, Basset
    functions, or Macdonald functions.

    Parameters
    ----------
    n : array_like of int
        Order of Bessel functions (floats will truncate with a warning)
    z : array_like of float
        Argument at which to evaluate the Bessel functions

    Returns
    -------
    out : ndarray
        The results

    Notes
    -----
    Wrapper for AMOS [1]_ routine `zbesk`.  For a discussion of the
    algorithm used, see [2]_ and the references therein.

    See Also
    --------
    kv : Same function, but accepts real order and complex argument
    kvp : Derivative of this function

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    .. [2] Donald E. Amos, "Algorithm 644: A portable package for Bessel
           functions of a complex argument and nonnegative order", ACM
           TOMS Vol. 12 Issue 3, Sept. 1986, p. 265

    Examples
    --------
    Plot the function of several orders for real input:

    >>> from scipy.special import kn
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 5, 1000)
    >>> for N in range(6):
    ...     plt.plot(x, kn(N, x), label='$K_{}(x)$'.format(N))
    >>> plt.ylim(0, 10)
    >>> plt.legend()
    >>> plt.title(r'Modified Bessel function of the second kind $K_n(x)$')
    >>> plt.show()

    Calculate for a single value at multiple orders:

    >>> kn([4, 5, 6], 1)
    array([   44.23241585,   360.9605896 ,  3653.83831186])
    """)

add_newdoc("kolmogi",
    """
    kolmogi(p)

    Inverse Survival Function of Kolmogorov distribution

    It is the inverse function to `kolmogorov`.
    Returns y such that ``kolmogorov(y) == p``.

    Parameters
    ----------
    p : float array_like
        Probability

    Returns
    -------
    float
        The value(s) of kolmogi(p)

    Notes
    -----
    `kolmogorov` is used by `stats.kstest` in the application of the
    Kolmogorov-Smirnov Goodness of Fit test. For historial reasons this
    function is exposed in `scpy.special`, but the recommended way to achieve
    the most accurate CDF/SF/PDF/PPF/ISF computations is to use the
    `stats.kstwobign` distribution.

    See Also
    --------
    kolmogorov : The Survival Function for the distribution
    scipy.stats.kstwobign : Provides the functionality as a continuous distribution
    smirnov, smirnovi : Functions for the one-sided distribution

    Examples
    --------
    >>> from scipy.special import kolmogi
    >>> kolmogi([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    array([        inf,  1.22384787,  1.01918472,  0.82757356,  0.67644769,
            0.57117327,  0.        ])

    """)

add_newdoc("kolmogorov",
    r"""
    kolmogorov(y)

    Complementary cumulative distribution (Survival Function) function of
    Kolmogorov distribution.

    Returns the complementary cumulative distribution function of
    Kolmogorov's limiting distribution (``D_n*\sqrt(n)`` as n goes to infinity)
    of a two-sided test for equality between an empirical and a theoretical
    distribution. It is equal to the (limit as n->infinity of the)
    probability that ``sqrt(n) * max absolute deviation > y``.

    Parameters
    ----------
    y : float array_like
      Absolute deviation between the Empirical CDF (ECDF) and the target CDF,
      multiplied by sqrt(n).

    Returns
    -------
    float
        The value(s) of kolmogorov(y)

    Notes
    -----
    `kolmogorov` is used by `stats.kstest` in the application of the
    Kolmogorov-Smirnov Goodness of Fit test. For historial reasons this
    function is exposed in `scpy.special`, but the recommended way to achieve
    the most accurate CDF/SF/PDF/PPF/ISF computations is to use the
    `stats.kstwobign` distribution.

    See Also
    --------
    kolmogi : The Inverse Survival Function for the distribution
    scipy.stats.kstwobign : Provides the functionality as a continuous distribution
    smirnov, smirnovi : Functions for the one-sided distribution

    Examples
    --------
    Show the probability of a gap at least as big as 0, 0.5 and 1.0.

    >>> from scipy.special import kolmogorov
    >>> from scipy.stats import kstwobign
    >>> kolmogorov([0, 0.5, 1.0])
    array([ 1.        ,  0.96394524,  0.26999967])

    Compare a sample of size 1000 drawn from a Laplace(0, 1) distribution against
    the target distribution, a Normal(0, 1) distribution.

    >>> from scipy.stats import norm, laplace
    >>> rng = np.random.default_rng()
    >>> n = 1000
    >>> lap01 = laplace(0, 1)
    >>> x = np.sort(lap01.rvs(n, random_state=rng))
    >>> np.mean(x), np.std(x)
    (-0.05841730131499543, 1.3968109101997568)

    Construct the Empirical CDF and the K-S statistic Dn.

    >>> target = norm(0,1)  # Normal mean 0, stddev 1
    >>> cdfs = target.cdf(x)
    >>> ecdfs = np.arange(n+1, dtype=float)/n
    >>> gaps = np.column_stack([cdfs - ecdfs[:n], ecdfs[1:] - cdfs])
    >>> Dn = np.max(gaps)
    >>> Kn = np.sqrt(n) * Dn
    >>> print('Dn=%f, sqrt(n)*Dn=%f' % (Dn, Kn))
    Dn=0.043363, sqrt(n)*Dn=1.371265
    >>> print(chr(10).join(['For a sample of size n drawn from a N(0, 1) distribution:',
    ...   ' the approximate Kolmogorov probability that sqrt(n)*Dn>=%f is %f' %  (Kn, kolmogorov(Kn)),
    ...   ' the approximate Kolmogorov probability that sqrt(n)*Dn<=%f is %f' %  (Kn, kstwobign.cdf(Kn))]))
    For a sample of size n drawn from a N(0, 1) distribution:
     the approximate Kolmogorov probability that sqrt(n)*Dn>=1.371265 is 0.046533
     the approximate Kolmogorov probability that sqrt(n)*Dn<=1.371265 is 0.953467

    Plot the Empirical CDF against the target N(0, 1) CDF.

    >>> import matplotlib.pyplot as plt
    >>> plt.step(np.concatenate([[-3], x]), ecdfs, where='post', label='Empirical CDF')
    >>> x3 = np.linspace(-3, 3, 100)
    >>> plt.plot(x3, target.cdf(x3), label='CDF for N(0, 1)')
    >>> plt.ylim([0, 1]); plt.grid(True); plt.legend();
    >>> # Add vertical lines marking Dn+ and Dn-
    >>> iminus, iplus = np.argmax(gaps, axis=0)
    >>> plt.vlines([x[iminus]], ecdfs[iminus], cdfs[iminus], color='r', linestyle='dashed', lw=4)
    >>> plt.vlines([x[iplus]], cdfs[iplus], ecdfs[iplus+1], color='r', linestyle='dashed', lw=4)
    >>> plt.show()
    """)

add_newdoc("_kolmogc",
    r"""
    Internal function, do not use.
    """)

add_newdoc("_kolmogci",
    r"""
    Internal function, do not use.
    """)

add_newdoc("_kolmogp",
    r"""
    Internal function, do not use.
    """)

add_newdoc("kv",
    r"""
    kv(v, z)

    Modified Bessel function of the second kind of real order `v`

    Returns the modified Bessel function of the second kind for real order
    `v` at complex `z`.

    These are also sometimes called functions of the third kind, Basset
    functions, or Macdonald functions.  They are defined as those solutions
    of the modified Bessel equation for which,

    .. math::
        K_v(x) \sim \sqrt{\pi/(2x)} \exp(-x)

    as :math:`x \to \infty` [3]_.

    Parameters
    ----------
    v : array_like of float
        Order of Bessel functions
    z : array_like of complex
        Argument at which to evaluate the Bessel functions

    Returns
    -------
    out : ndarray
        The results. Note that input must be of complex type to get complex
        output, e.g. ``kv(3, -2+0j)`` instead of ``kv(3, -2)``.

    Notes
    -----
    Wrapper for AMOS [1]_ routine `zbesk`.  For a discussion of the
    algorithm used, see [2]_ and the references therein.

    See Also
    --------
    kve : This function with leading exponential behavior stripped off.
    kvp : Derivative of this function

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    .. [2] Donald E. Amos, "Algorithm 644: A portable package for Bessel
           functions of a complex argument and nonnegative order", ACM
           TOMS Vol. 12 Issue 3, Sept. 1986, p. 265
    .. [3] NIST Digital Library of Mathematical Functions,
           Eq. 10.25.E3. https://dlmf.nist.gov/10.25.E3

    Examples
    --------
    Plot the function of several orders for real input:

    >>> from scipy.special import kv
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 5, 1000)
    >>> for N in np.linspace(0, 6, 5):
    ...     plt.plot(x, kv(N, x), label='$K_{{{}}}(x)$'.format(N))
    >>> plt.ylim(0, 10)
    >>> plt.legend()
    >>> plt.title(r'Modified Bessel function of the second kind $K_\nu(x)$')
    >>> plt.show()

    Calculate for a single value at multiple orders:

    >>> kv([4, 4.5, 5], 1+2j)
    array([ 0.1992+2.3892j,  2.3493+3.6j   ,  7.2827+3.8104j])

    """)

add_newdoc("kve",
    r"""
    kve(v, z)

    Exponentially scaled modified Bessel function of the second kind.

    Returns the exponentially scaled, modified Bessel function of the
    second kind (sometimes called the third kind) for real order `v` at
    complex `z`::

        kve(v, z) = kv(v, z) * exp(z)

    Parameters
    ----------
    v : array_like of float
        Order of Bessel functions
    z : array_like of complex
        Argument at which to evaluate the Bessel functions

    Returns
    -------
    out : ndarray
        The exponentially scaled modified Bessel function of the second kind.

    Notes
    -----
    Wrapper for AMOS [1]_ routine `zbesk`.  For a discussion of the
    algorithm used, see [2]_ and the references therein.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    .. [2] Donald E. Amos, "Algorithm 644: A portable package for Bessel
           functions of a complex argument and nonnegative order", ACM
           TOMS Vol. 12 Issue 3, Sept. 1986, p. 265
    """)

add_newdoc("_lanczos_sum_expg_scaled",
    """
    Internal function, do not use.
    """)

add_newdoc("_lgam1p",
    """
    Internal function, do not use.
    """)

add_newdoc("log1p",
    """
    log1p(x, out=None)

    Calculates log(1 + x) for use when `x` is near zero.

    Parameters
    ----------
    x : array_like
        Real or complex valued input.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of ``log(1 + x)``.

    See Also
    --------
    expm1, cosm1

    Examples
    --------
    >>> import scipy.special as sc

    It is more accurate than using ``log(1 + x)`` directly for ``x``
    near 0. Note that in the below example ``1 + 1e-17 == 1`` to
    double precision.

    >>> sc.log1p(1e-17)
    1e-17
    >>> np.log(1 + 1e-17)
    0.0

    """)

add_newdoc("_log1pmx",
    """
    Internal function, do not use.
    """)

add_newdoc('logit',
    """
    logit(x)

    Logit ufunc for ndarrays.

    The logit function is defined as logit(p) = log(p/(1-p)).
    Note that logit(0) = -inf, logit(1) = inf, and logit(p)
    for p<0 or p>1 yields nan.

    Parameters
    ----------
    x : ndarray
        The ndarray to apply logit to element-wise.

    Returns
    -------
    out : ndarray
        An ndarray of the same shape as x. Its entries
        are logit of the corresponding entry of x.

    See Also
    --------
    expit

    Notes
    -----
    As a ufunc logit takes a number of optional
    keyword arguments. For more information
    see `ufuncs <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_

    .. versionadded:: 0.10.0

    Examples
    --------
    >>> from scipy.special import logit, expit

    >>> logit([0, 0.25, 0.5, 0.75, 1])
    array([       -inf, -1.09861229,  0.        ,  1.09861229,         inf])

    `expit` is the inverse of `logit`:

    >>> expit(logit([0.1, 0.75, 0.999]))
    array([ 0.1  ,  0.75 ,  0.999])

    Plot logit(x) for x in [0, 1]:

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 1, 501)
    >>> y = logit(x)
    >>> plt.plot(x, y)
    >>> plt.grid()
    >>> plt.ylim(-6, 6)
    >>> plt.xlabel('x')
    >>> plt.title('logit(x)')
    >>> plt.show()

    """)

add_newdoc("lpmv",
    r"""
    lpmv(m, v, x)

    Associated Legendre function of integer order and real degree.

    Defined as

    .. math::

        P_v^m = (-1)^m (1 - x^2)^{m/2} \frac{d^m}{dx^m} P_v(x)

    where

    .. math::

        P_v = \sum_{k = 0}^\infty \frac{(-v)_k (v + 1)_k}{(k!)^2}
                \left(\frac{1 - x}{2}\right)^k

    is the Legendre function of the first kind. Here :math:`(\cdot)_k`
    is the Pochhammer symbol; see `poch`.

    Parameters
    ----------
    m : array_like
        Order (int or float). If passed a float not equal to an
        integer the function returns NaN.
    v : array_like
        Degree (float).
    x : array_like
        Argument (float). Must have ``|x| <= 1``.

    Returns
    -------
    pmv : ndarray
        Value of the associated Legendre function.

    See Also
    --------
    lpmn : Compute the associated Legendre function for all orders
           ``0, ..., m`` and degrees ``0, ..., n``.
    clpmn : Compute the associated Legendre function at complex
            arguments.

    Notes
    -----
    Note that this implementation includes the Condon-Shortley phase.

    References
    ----------
    .. [1] Zhang, Jin, "Computation of Special Functions", John Wiley
           and Sons, Inc, 1996.

    """)

add_newdoc("mathieu_a",
    """
    mathieu_a(m, q)

    Characteristic value of even Mathieu functions

    Returns the characteristic value for the even solution,
    ``ce_m(z, q)``, of Mathieu's equation.
    """)

add_newdoc("mathieu_b",
    """
    mathieu_b(m, q)

    Characteristic value of odd Mathieu functions

    Returns the characteristic value for the odd solution,
    ``se_m(z, q)``, of Mathieu's equation.
    """)

add_newdoc("mathieu_cem",
    """
    mathieu_cem(m, q, x)

    Even Mathieu function and its derivative

    Returns the even Mathieu function, ``ce_m(x, q)``, of order `m` and
    parameter `q` evaluated at `x` (given in degrees).  Also returns the
    derivative with respect to `x` of ce_m(x, q)

    Parameters
    ----------
    m
        Order of the function
    q
        Parameter of the function
    x
        Argument of the function, *given in degrees, not radians*

    Returns
    -------
    y
        Value of the function
    yp
        Value of the derivative vs x
    """)

add_newdoc("mathieu_modcem1",
    """
    mathieu_modcem1(m, q, x)

    Even modified Mathieu function of the first kind and its derivative

    Evaluates the even modified Mathieu function of the first kind,
    ``Mc1m(x, q)``, and its derivative at `x` for order `m` and parameter
    `q`.

    Returns
    -------
    y
        Value of the function
    yp
        Value of the derivative vs x
    """)

add_newdoc("mathieu_modcem2",
    """
    mathieu_modcem2(m, q, x)

    Even modified Mathieu function of the second kind and its derivative

    Evaluates the even modified Mathieu function of the second kind,
    Mc2m(x, q), and its derivative at `x` (given in degrees) for order `m`
    and parameter `q`.

    Returns
    -------
    y
        Value of the function
    yp
        Value of the derivative vs x
    """)

add_newdoc("mathieu_modsem1",
    """
    mathieu_modsem1(m, q, x)

    Odd modified Mathieu function of the first kind and its derivative

    Evaluates the odd modified Mathieu function of the first kind,
    Ms1m(x, q), and its derivative at `x` (given in degrees) for order `m`
    and parameter `q`.

    Returns
    -------
    y
        Value of the function
    yp
        Value of the derivative vs x
    """)

add_newdoc("mathieu_modsem2",
    """
    mathieu_modsem2(m, q, x)

    Odd modified Mathieu function of the second kind and its derivative

    Evaluates the odd modified Mathieu function of the second kind,
    Ms2m(x, q), and its derivative at `x` (given in degrees) for order `m`
    and parameter q.

    Returns
    -------
    y
        Value of the function
    yp
        Value of the derivative vs x
    """)

add_newdoc("mathieu_sem",
    """
    mathieu_sem(m, q, x)

    Odd Mathieu function and its derivative

    Returns the odd Mathieu function, se_m(x, q), of order `m` and
    parameter `q` evaluated at `x` (given in degrees).  Also returns the
    derivative with respect to `x` of se_m(x, q).

    Parameters
    ----------
    m
        Order of the function
    q
        Parameter of the function
    x
        Argument of the function, *given in degrees, not radians*.

    Returns
    -------
    y
        Value of the function
    yp
        Value of the derivative vs x
    """)

add_newdoc("modfresnelm",
    """
    modfresnelm(x)

    Modified Fresnel negative integrals

    Returns
    -------
    fm
        Integral ``F_-(x)``: ``integral(exp(-1j*t*t), t=x..inf)``
    km
        Integral ``K_-(x)``: ``1/sqrt(pi)*exp(1j*(x*x+pi/4))*fp``
    """)

add_newdoc("modfresnelp",
    """
    modfresnelp(x)

    Modified Fresnel positive integrals

    Returns
    -------
    fp
        Integral ``F_+(x)``: ``integral(exp(1j*t*t), t=x..inf)``
    kp
        Integral ``K_+(x)``: ``1/sqrt(pi)*exp(-1j*(x*x+pi/4))*fp``
    """)

add_newdoc("modstruve",
    r"""
    modstruve(v, x)

    Modified Struve function.

    Return the value of the modified Struve function of order `v` at `x`.  The
    modified Struve function is defined as,

    .. math::
        L_v(x) = -\imath \exp(-\pi\imath v/2) H_v(\imath x),

    where :math:`H_v` is the Struve function.

    Parameters
    ----------
    v : array_like
        Order of the modified Struve function (float).
    x : array_like
        Argument of the Struve function (float; must be positive unless `v` is
        an integer).

    Returns
    -------
    L : ndarray
        Value of the modified Struve function of order `v` at `x`.

    Notes
    -----
    Three methods discussed in [1]_ are used to evaluate the function:

    - power series
    - expansion in Bessel functions (if :math:`|x| < |v| + 20`)
    - asymptotic large-x expansion (if :math:`x \geq 0.7v + 12`)

    Rounding errors are estimated based on the largest terms in the sums, and
    the result associated with the smallest error is returned.

    See also
    --------
    struve

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/11
    """)

add_newdoc("nbdtr",
    r"""
    nbdtr(k, n, p)

    Negative binomial cumulative distribution function.

    Returns the sum of the terms 0 through `k` of the negative binomial
    distribution probability mass function,

    .. math::

        F = \sum_{j=0}^k {{n + j - 1}\choose{j}} p^n (1 - p)^j.

    In a sequence of Bernoulli trials with individual success probabilities
    `p`, this is the probability that `k` or fewer failures precede the nth
    success.

    Parameters
    ----------
    k : array_like
        The maximum number of allowed failures (nonnegative int).
    n : array_like
        The target number of successes (positive int).
    p : array_like
        Probability of success in a single event (float).

    Returns
    -------
    F : ndarray
        The probability of `k` or fewer failures before `n` successes in a
        sequence of events with individual success probability `p`.

    See also
    --------
    nbdtrc

    Notes
    -----
    If floating point values are passed for `k` or `n`, they will be truncated
    to integers.

    The terms are not summed directly; instead the regularized incomplete beta
    function is employed, according to the formula,

    .. math::
        \mathrm{nbdtr}(k, n, p) = I_{p}(n, k + 1).

    Wrapper for the Cephes [1]_ routine `nbdtr`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    """)

add_newdoc("nbdtrc",
    r"""
    nbdtrc(k, n, p)

    Negative binomial survival function.

    Returns the sum of the terms `k + 1` to infinity of the negative binomial
    distribution probability mass function,

    .. math::

        F = \sum_{j=k + 1}^\infty {{n + j - 1}\choose{j}} p^n (1 - p)^j.

    In a sequence of Bernoulli trials with individual success probabilities
    `p`, this is the probability that more than `k` failures precede the nth
    success.

    Parameters
    ----------
    k : array_like
        The maximum number of allowed failures (nonnegative int).
    n : array_like
        The target number of successes (positive int).
    p : array_like
        Probability of success in a single event (float).

    Returns
    -------
    F : ndarray
        The probability of `k + 1` or more failures before `n` successes in a
        sequence of events with individual success probability `p`.

    Notes
    -----
    If floating point values are passed for `k` or `n`, they will be truncated
    to integers.

    The terms are not summed directly; instead the regularized incomplete beta
    function is employed, according to the formula,

    .. math::
        \mathrm{nbdtrc}(k, n, p) = I_{1 - p}(k + 1, n).

    Wrapper for the Cephes [1]_ routine `nbdtrc`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

add_newdoc("nbdtri",
    """
    nbdtri(k, n, y)

    Inverse of `nbdtr` vs `p`.

    Returns the inverse with respect to the parameter `p` of
    `y = nbdtr(k, n, p)`, the negative binomial cumulative distribution
    function.

    Parameters
    ----------
    k : array_like
        The maximum number of allowed failures (nonnegative int).
    n : array_like
        The target number of successes (positive int).
    y : array_like
        The probability of `k` or fewer failures before `n` successes (float).

    Returns
    -------
    p : ndarray
        Probability of success in a single event (float) such that
        `nbdtr(k, n, p) = y`.

    See also
    --------
    nbdtr : Cumulative distribution function of the negative binomial.
    nbdtrik : Inverse with respect to `k` of `nbdtr(k, n, p)`.
    nbdtrin : Inverse with respect to `n` of `nbdtr(k, n, p)`.

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `nbdtri`.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/

    """)

add_newdoc("nbdtrik",
    r"""
    nbdtrik(y, n, p)

    Inverse of `nbdtr` vs `k`.

    Returns the inverse with respect to the parameter `k` of
    `y = nbdtr(k, n, p)`, the negative binomial cumulative distribution
    function.

    Parameters
    ----------
    y : array_like
        The probability of `k` or fewer failures before `n` successes (float).
    n : array_like
        The target number of successes (positive int).
    p : array_like
        Probability of success in a single event (float).

    Returns
    -------
    k : ndarray
        The maximum number of allowed failures such that `nbdtr(k, n, p) = y`.

    See also
    --------
    nbdtr : Cumulative distribution function of the negative binomial.
    nbdtri : Inverse with respect to `p` of `nbdtr(k, n, p)`.
    nbdtrin : Inverse with respect to `n` of `nbdtr(k, n, p)`.

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdfnbn`.

    Formula 26.5.26 of [2]_,

    .. math::
        \sum_{j=k + 1}^\infty {{n + j - 1}\choose{j}} p^n (1 - p)^j = I_{1 - p}(k + 1, n),

    is used to reduce calculation of the cumulative distribution function to
    that of a regularized incomplete beta :math:`I`.

    Computation of `k` involves a search for a value that produces the desired
    value of `y`.  The search relies on the monotonicity of `y` with `k`.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("nbdtrin",
    r"""
    nbdtrin(k, y, p)

    Inverse of `nbdtr` vs `n`.

    Returns the inverse with respect to the parameter `n` of
    `y = nbdtr(k, n, p)`, the negative binomial cumulative distribution
    function.

    Parameters
    ----------
    k : array_like
        The maximum number of allowed failures (nonnegative int).
    y : array_like
        The probability of `k` or fewer failures before `n` successes (float).
    p : array_like
        Probability of success in a single event (float).

    Returns
    -------
    n : ndarray
        The number of successes `n` such that `nbdtr(k, n, p) = y`.

    See also
    --------
    nbdtr : Cumulative distribution function of the negative binomial.
    nbdtri : Inverse with respect to `p` of `nbdtr(k, n, p)`.
    nbdtrik : Inverse with respect to `k` of `nbdtr(k, n, p)`.

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdfnbn`.

    Formula 26.5.26 of [2]_,

    .. math::
        \sum_{j=k + 1}^\infty {{n + j - 1}\choose{j}} p^n (1 - p)^j = I_{1 - p}(k + 1, n),

    is used to reduce calculation of the cumulative distribution function to
    that of a regularized incomplete beta :math:`I`.

    Computation of `n` involves a search for a value that produces the desired
    value of `y`.  The search relies on the monotonicity of `y` with `n`.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.

    """)

add_newdoc("ncfdtr",
    r"""
    ncfdtr(dfn, dfd, nc, f)

    Cumulative distribution function of the non-central F distribution.

    The non-central F describes the distribution of,

    .. math::
        Z = \frac{X/d_n}{Y/d_d}

    where :math:`X` and :math:`Y` are independently distributed, with
    :math:`X` distributed non-central :math:`\chi^2` with noncentrality
    parameter `nc` and :math:`d_n` degrees of freedom, and :math:`Y`
    distributed :math:`\chi^2` with :math:`d_d` degrees of freedom.

    Parameters
    ----------
    dfn : array_like
        Degrees of freedom of the numerator sum of squares.  Range (0, inf).
    dfd : array_like
        Degrees of freedom of the denominator sum of squares.  Range (0, inf).
    nc : array_like
        Noncentrality parameter.  Should be in range (0, 1e4).
    f : array_like
        Quantiles, i.e. the upper limit of integration.

    Returns
    -------
    cdf : float or ndarray
        The calculated CDF.  If all inputs are scalar, the return will be a
        float.  Otherwise it will be an array.

    See Also
    --------
    ncfdtri : Quantile function; inverse of `ncfdtr` with respect to `f`.
    ncfdtridfd : Inverse of `ncfdtr` with respect to `dfd`.
    ncfdtridfn : Inverse of `ncfdtr` with respect to `dfn`.
    ncfdtrinc : Inverse of `ncfdtr` with respect to `nc`.

    Notes
    -----
    Wrapper for the CDFLIB [1]_ Fortran routine `cdffnc`.

    The cumulative distribution function is computed using Formula 26.6.20 of
    [2]_:

    .. math::
        F(d_n, d_d, n_c, f) = \sum_{j=0}^\infty e^{-n_c/2} \frac{(n_c/2)^j}{j!} I_{x}(\frac{d_n}{2} + j, \frac{d_d}{2}),

    where :math:`I` is the regularized incomplete beta function, and
    :math:`x = f d_n/(f d_n + d_d)`.

    The computation time required for this routine is proportional to the
    noncentrality parameter `nc`.  Very large values of this parameter can
    consume immense computer resources.  This is why the search range is
    bounded by 10,000.

    References
    ----------
    .. [1] Barry Brown, James Lovato, and Kathy Russell,
           CDFLIB: Library of Fortran Routines for Cumulative Distribution
           Functions, Inverses, and Other Parameters.
    .. [2] Milton Abramowitz and Irene A. Stegun, eds.
           Handbook of Mathematical Functions with Formulas,
           Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    >>> from scipy import special
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    Plot the CDF of the non-central F distribution, for nc=0.  Compare with the
    F-distribution from scipy.stats:

    >>> x = np.linspace(-1, 8, num=500)
    >>> dfn = 3
    >>> dfd = 2
    >>> ncf_stats = stats.f.cdf(x, dfn, dfd)
    >>> ncf_special = special.ncfdtr(dfn, dfd, 0, x)

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(x, ncf_stats, 'b-', lw=3)
    >>> ax.plot(x, ncf_special, 'r-')
    >>> plt.show()

    """)

add_newdoc("ncfdtri",
    """
    ncfdtri(dfn, dfd, nc, p)

    Inverse with respect to `f` of the CDF of the non-central F distribution.

    See `ncfdtr` for more details.

    Parameters
    ----------
    dfn : array_like
        Degrees of freedom of the numerator sum of squares.  Range (0, inf).
    dfd : array_like
        Degrees of freedom of the denominator sum of squares.  Range (0, inf).
    nc : array_like
        Noncentrality parameter.  Should be in range (0, 1e4).
    p : array_like
        Value of the cumulative distribution function.  Must be in the
        range [0, 1].

    Returns
    -------
    f : float
        Quantiles, i.e., the upper limit of integration.

    See Also
    --------
    ncfdtr : CDF of the non-central F distribution.
    ncfdtridfd : Inverse of `ncfdtr` with respect to `dfd`.
    ncfdtridfn : Inverse of `ncfdtr` with respect to `dfn`.
    ncfdtrinc : Inverse of `ncfdtr` with respect to `nc`.

    Examples
    --------
    >>> from scipy.special import ncfdtr, ncfdtri

    Compute the CDF for several values of `f`:

    >>> f = [0.5, 1, 1.5]
    >>> p = ncfdtr(2, 3, 1.5, f)
    >>> p
    array([ 0.20782291,  0.36107392,  0.47345752])

    Compute the inverse.  We recover the values of `f`, as expected:

    >>> ncfdtri(2, 3, 1.5, p)
    array([ 0.5,  1. ,  1.5])

    """)

add_newdoc("ncfdtridfd",
    """
    ncfdtridfd(dfn, p, nc, f)

    Calculate degrees of freedom (denominator) for the noncentral F-distribution.

    This is the inverse with respect to `dfd` of `ncfdtr`.
    See `ncfdtr` for more details.

    Parameters
    ----------
    dfn : array_like
        Degrees of freedom of the numerator sum of squares.  Range (0, inf).
    p : array_like
        Value of the cumulative distribution function.  Must be in the
        range [0, 1].
    nc : array_like
        Noncentrality parameter.  Should be in range (0, 1e4).
    f : array_like
        Quantiles, i.e., the upper limit of integration.

    Returns
    -------
    dfd : float
        Degrees of freedom of the denominator sum of squares.

    See Also
    --------
    ncfdtr : CDF of the non-central F distribution.
    ncfdtri : Quantile function; inverse of `ncfdtr` with respect to `f`.
    ncfdtridfn : Inverse of `ncfdtr` with respect to `dfn`.
    ncfdtrinc : Inverse of `ncfdtr` with respect to `nc`.

    Notes
    -----
    The value of the cumulative noncentral F distribution is not necessarily
    monotone in either degrees of freedom. There thus may be two values that
    provide a given CDF value. This routine assumes monotonicity and will
    find an arbitrary one of the two values.

    Examples
    --------
    >>> from scipy.special import ncfdtr, ncfdtridfd

    Compute the CDF for several values of `dfd`:

    >>> dfd = [1, 2, 3]
    >>> p = ncfdtr(2, dfd, 0.25, 15)
    >>> p
    array([ 0.8097138 ,  0.93020416,  0.96787852])

    Compute the inverse.  We recover the values of `dfd`, as expected:

    >>> ncfdtridfd(2, p, 0.25, 15)
    array([ 1.,  2.,  3.])

    """)

add_newdoc("ncfdtridfn",
    """
    ncfdtridfn(p, dfd, nc, f)

    Calculate degrees of freedom (numerator) for the noncentral F-distribution.

    This is the inverse with respect to `dfn` of `ncfdtr`.
    See `ncfdtr` for more details.

    Parameters
    ----------
    p : array_like
        Value of the cumulative distribution function. Must be in the
        range [0, 1].
    dfd : array_like
        Degrees of freedom of the denominator sum of squares. Range (0, inf).
    nc : array_like
        Noncentrality parameter.  Should be in range (0, 1e4).
    f : float
        Quantiles, i.e., the upper limit of integration.

    Returns
    -------
    dfn : float
        Degrees of freedom of the numerator sum of squares.

    See Also
    --------
    ncfdtr : CDF of the non-central F distribution.
    ncfdtri : Quantile function; inverse of `ncfdtr` with respect to `f`.
    ncfdtridfd : Inverse of `ncfdtr` with respect to `dfd`.
    ncfdtrinc : Inverse of `ncfdtr` with respect to `nc`.

    Notes
    -----
    The value of the cumulative noncentral F distribution is not necessarily
    monotone in either degrees of freedom. There thus may be two values that
    provide a given CDF value. This routine assumes monotonicity and will
    find an arbitrary one of the two values.

    Examples
    --------
    >>> from scipy.special import ncfdtr, ncfdtridfn

    Compute the CDF for several values of `dfn`:

    >>> dfn = [1, 2, 3]
    >>> p = ncfdtr(dfn, 2, 0.25, 15)
    >>> p
    array([ 0.92562363,  0.93020416,  0.93188394])

    Compute the inverse. We recover the values of `dfn`, as expected:

    >>> ncfdtridfn(p, 2, 0.25, 15)
    array([ 1.,  2.,  3.])

    """)

add_newdoc("ncfdtrinc",
    """
    ncfdtrinc(dfn, dfd, p, f)

    Calculate non-centrality parameter for non-central F distribution.

    This is the inverse with respect to `nc` of `ncfdtr`.
    See `ncfdtr` for more details.

    Parameters
    ----------
    dfn : array_like
        Degrees of freedom of the numerator sum of squares. Range (0, inf).
    dfd : array_like
        Degrees of freedom of the denominator sum of squares. Range (0, inf).
    p : array_like
        Value of the cumulative distribution function. Must be in the
        range [0, 1].
    f : array_like
        Quantiles, i.e., the upper limit of integration.

    Returns
    -------
    nc : float
        Noncentrality parameter.

    See Also
    --------
    ncfdtr : CDF of the non-central F distribution.
    ncfdtri : Quantile function; inverse of `ncfdtr` with respect to `f`.
    ncfdtridfd : Inverse of `ncfdtr` with respect to `dfd`.
    ncfdtridfn : Inverse of `ncfdtr` with respect to `dfn`.

    Examples
    --------
    >>> from scipy.special import ncfdtr, ncfdtrinc

    Compute the CDF for several values of `nc`:

    >>> nc = [0.5, 1.5, 2.0]
    >>> p = ncfdtr(2, 3, nc, 15)
    >>> p
    array([ 0.96309246,  0.94327955,  0.93304098])

    Compute the inverse. We recover the values of `nc`, as expected:

    >>> ncfdtrinc(2, 3, p, 15)
    array([ 0.5,  1.5,  2. ])

    """)

add_newdoc("nctdtr",
    """
    nctdtr(df, nc, t)

    Cumulative distribution function of the non-central `t` distribution.

    Parameters
    ----------
    df : array_like
        Degrees of freedom of the distribution. Should be in range (0, inf).
    nc : array_like
        Noncentrality parameter. Should be in range (-1e6, 1e6).
    t : array_like
        Quantiles, i.e., the upper limit of integration.

    Returns
    -------
    cdf : float or ndarray
        The calculated CDF. If all inputs are scalar, the return will be a
        float. Otherwise, it will be an array.

    See Also
    --------
    nctdtrit : Inverse CDF (iCDF) of the non-central t distribution.
    nctdtridf : Calculate degrees of freedom, given CDF and iCDF values.
    nctdtrinc : Calculate non-centrality parameter, given CDF iCDF values.

    Examples
    --------
    >>> from scipy import special
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    Plot the CDF of the non-central t distribution, for nc=0. Compare with the
    t-distribution from scipy.stats:

    >>> x = np.linspace(-5, 5, num=500)
    >>> df = 3
    >>> nct_stats = stats.t.cdf(x, df)
    >>> nct_special = special.nctdtr(df, 0, x)

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(x, nct_stats, 'b-', lw=3)
    >>> ax.plot(x, nct_special, 'r-')
    >>> plt.show()

    """)

add_newdoc("nctdtridf",
    """
    nctdtridf(p, nc, t)

    Calculate degrees of freedom for non-central t distribution.

    See `nctdtr` for more details.

    Parameters
    ----------
    p : array_like
        CDF values, in range (0, 1].
    nc : array_like
        Noncentrality parameter. Should be in range (-1e6, 1e6).
    t : array_like
        Quantiles, i.e., the upper limit of integration.

    """)

add_newdoc("nctdtrinc",
    """
    nctdtrinc(df, p, t)

    Calculate non-centrality parameter for non-central t distribution.

    See `nctdtr` for more details.

    Parameters
    ----------
    df : array_like
        Degrees of freedom of the distribution. Should be in range (0, inf).
    p : array_like
        CDF values, in range (0, 1].
    t : array_like
        Quantiles, i.e., the upper limit of integration.

    """)

add_newdoc("nctdtrit",
    """
    nctdtrit(df, nc, p)

    Inverse cumulative distribution function of the non-central t distribution.

    See `nctdtr` for more details.

    Parameters
    ----------
    df : array_like
        Degrees of freedom of the distribution. Should be in range (0, inf).
    nc : array_like
        Noncentrality parameter. Should be in range (-1e6, 1e6).
    p : array_like
        CDF values, in range (0, 1].

    """)

add_newdoc("ndtr",
    r"""
    ndtr(x)

    Gaussian cumulative distribution function.

    Returns the area under the standard Gaussian probability
    density function, integrated from minus infinity to `x`

    .. math::

       \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x \exp(-t^2/2) dt

    Parameters
    ----------
    x : array_like, real or complex
        Argument

    Returns
    -------
    ndarray
        The value of the normal CDF evaluated at `x`

    See Also
    --------
    erf
    erfc
    scipy.stats.norm
    log_ndtr

    """)


add_newdoc("nrdtrimn",
    """
    nrdtrimn(p, x, std)

    Calculate mean of normal distribution given other params.

    Parameters
    ----------
    p : array_like
        CDF values, in range (0, 1].
    x : array_like
        Quantiles, i.e. the upper limit of integration.
    std : array_like
        Standard deviation.

    Returns
    -------
    mn : float or ndarray
        The mean of the normal distribution.

    See Also
    --------
    nrdtrimn, ndtr

    """)

add_newdoc("nrdtrisd",
    """
    nrdtrisd(p, x, mn)

    Calculate standard deviation of normal distribution given other params.

    Parameters
    ----------
    p : array_like
        CDF values, in range (0, 1].
    x : array_like
        Quantiles, i.e. the upper limit of integration.
    mn : float or ndarray
        The mean of the normal distribution.

    Returns
    -------
    std : array_like
        Standard deviation.

    See Also
    --------
    ndtr

    """)

add_newdoc("log_ndtr",
    """
    log_ndtr(x)

    Logarithm of Gaussian cumulative distribution function.

    Returns the log of the area under the standard Gaussian probability
    density function, integrated from minus infinity to `x`::

        log(1/sqrt(2*pi) * integral(exp(-t**2 / 2), t=-inf..x))

    Parameters
    ----------
    x : array_like, real or complex
        Argument

    Returns
    -------
    ndarray
        The value of the log of the normal CDF evaluated at `x`

    See Also
    --------
    erf
    erfc
    scipy.stats.norm
    ndtr

    """)

add_newdoc("ndtri",
    """
    ndtri(y)

    Inverse of `ndtr` vs x

    Returns the argument x for which the area under the Gaussian
    probability density function (integrated from minus infinity to `x`)
    is equal to y.
    """)

add_newdoc("obl_ang1",
    """
    obl_ang1(m, n, c, x)

    Oblate spheroidal angular function of the first kind and its derivative

    Computes the oblate spheroidal angular function of the first kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.

    Returns
    -------
    s
        Value of the function
    sp
        Value of the derivative vs x
    """)

add_newdoc("obl_ang1_cv",
    """
    obl_ang1_cv(m, n, c, cv, x)

    Oblate spheroidal angular function obl_ang1 for precomputed characteristic value

    Computes the oblate spheroidal angular function of the first kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires
    pre-computed characteristic value.

    Returns
    -------
    s
        Value of the function
    sp
        Value of the derivative vs x
    """)

add_newdoc("obl_cv",
    """
    obl_cv(m, n, c)

    Characteristic value of oblate spheroidal function

    Computes the characteristic value of oblate spheroidal wave
    functions of order `m`, `n` (n>=m) and spheroidal parameter `c`.
    """)

add_newdoc("obl_rad1",
    """
    obl_rad1(m, n, c, x)

    Oblate spheroidal radial function of the first kind and its derivative

    Computes the oblate spheroidal radial function of the first kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.

    Returns
    -------
    s
        Value of the function
    sp
        Value of the derivative vs x
    """)

add_newdoc("obl_rad1_cv",
    """
    obl_rad1_cv(m, n, c, cv, x)

    Oblate spheroidal radial function obl_rad1 for precomputed characteristic value

    Computes the oblate spheroidal radial function of the first kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires
    pre-computed characteristic value.

    Returns
    -------
    s
        Value of the function
    sp
        Value of the derivative vs x
    """)

add_newdoc("obl_rad2",
    """
    obl_rad2(m, n, c, x)

    Oblate spheroidal radial function of the second kind and its derivative.

    Computes the oblate spheroidal radial function of the second kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.

    Returns
    -------
    s
        Value of the function
    sp
        Value of the derivative vs x
    """)

add_newdoc("obl_rad2_cv",
    """
    obl_rad2_cv(m, n, c, cv, x)

    Oblate spheroidal radial function obl_rad2 for precomputed characteristic value

    Computes the oblate spheroidal radial function of the second kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires
    pre-computed characteristic value.

    Returns
    -------
    s
        Value of the function
    sp
        Value of the derivative vs x
    """)

add_newdoc("pbdv",
    """
    pbdv(v, x)

    Parabolic cylinder function D

    Returns (d, dp) the parabolic cylinder function Dv(x) in d and the
    derivative, Dv'(x) in dp.

    Returns
    -------
    d
        Value of the function
    dp
        Value of the derivative vs x
    """)

add_newdoc("pbvv",
    """
    pbvv(v, x)

    Parabolic cylinder function V

    Returns the parabolic cylinder function Vv(x) in v and the
    derivative, Vv'(x) in vp.

    Returns
    -------
    v
        Value of the function
    vp
        Value of the derivative vs x
    """)

add_newdoc("pbwa",
    r"""
    pbwa(a, x)

    Parabolic cylinder function W.

    The function is a particular solution to the differential equation

    .. math::

        y'' + \left(\frac{1}{4}x^2 - a\right)y = 0,

    for a full definition see section 12.14 in [1]_.

    Parameters
    ----------
    a : array_like
        Real parameter
    x : array_like
        Real argument

    Returns
    -------
    w : scalar or ndarray
        Value of the function
    wp : scalar or ndarray
        Value of the derivative in x

    Notes
    -----
    The function is a wrapper for a Fortran routine by Zhang and Jin
    [2]_. The implementation is accurate only for ``|a|, |x| < 5`` and
    returns NaN outside that range.

    References
    ----------
    .. [1] Digital Library of Mathematical Functions, 14.30.
           https://dlmf.nist.gov/14.30
    .. [2] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
    """)

add_newdoc("pdtr",
    r"""
    pdtr(k, m, out=None)

    Poisson cumulative distribution function.

    Defined as the probability that a Poisson-distributed random
    variable with event rate :math:`m` is less than or equal to
    :math:`k`. More concretely, this works out to be [1]_

    .. math::

       \exp(-m) \sum_{j = 0}^{\lfloor{k}\rfloor} \frac{m^j}{m!}.

    Parameters
    ----------
    k : array_like
        Nonnegative real argument
    m : array_like
        Nonnegative real shape parameter
    out : ndarray
        Optional output array for the function results

    See Also
    --------
    pdtrc : Poisson survival function
    pdtrik : inverse of `pdtr` with respect to `k`
    pdtri : inverse of `pdtr` with respect to `m`

    Returns
    -------
    scalar or ndarray
        Values of the Poisson cumulative distribution function

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Poisson_distribution

    Examples
    --------
    >>> import scipy.special as sc

    It is a cumulative distribution function, so it converges to 1
    monotonically as `k` goes to infinity.

    >>> sc.pdtr([1, 10, 100, np.inf], 1)
    array([0.73575888, 0.99999999, 1.        , 1.        ])

    It is discontinuous at integers and constant between integers.

    >>> sc.pdtr([1, 1.5, 1.9, 2], 1)
    array([0.73575888, 0.73575888, 0.73575888, 0.9196986 ])

    """)

add_newdoc("pdtrc",
    """
    pdtrc(k, m)

    Poisson survival function

    Returns the sum of the terms from k+1 to infinity of the Poisson
    distribution: sum(exp(-m) * m**j / j!, j=k+1..inf) = gammainc(
    k+1, m). Arguments must both be non-negative doubles.
    """)

add_newdoc("pdtri",
    """
    pdtri(k, y)

    Inverse to `pdtr` vs m

    Returns the Poisson variable `m` such that the sum from 0 to `k` of
    the Poisson density is equal to the given probability `y`:
    calculated by gammaincinv(k+1, y). `k` must be a nonnegative
    integer and `y` between 0 and 1.
    """)

add_newdoc("pdtrik",
    """
    pdtrik(p, m)

    Inverse to `pdtr` vs k

    Returns the quantile k such that ``pdtr(k, m) = p``
    """)

add_newdoc("poch",
    r"""
    poch(z, m)

    Pochhammer symbol.

    The Pochhammer symbol (rising factorial) is defined as

    .. math::

        (z)_m = \frac{\Gamma(z + m)}{\Gamma(z)}

    For positive integer `m` it reads

    .. math::

        (z)_m = z (z + 1) ... (z + m - 1)

    See [dlmf]_ for more details.

    Parameters
    ----------
    z, m : array_like
        Real-valued arguments.

    Returns
    -------
    scalar or ndarray
        The value of the function.

    References
    ----------
    .. [dlmf] Nist, Digital Library of Mathematical Functions
        https://dlmf.nist.gov/5.2#iii

    Examples
    --------
    >>> import scipy.special as sc

    It is 1 when m is 0.

    >>> sc.poch([1, 2, 3, 4], 0)
    array([1., 1., 1., 1.])

    For z equal to 1 it reduces to the factorial function.

    >>> sc.poch(1, 5)
    120.0
    >>> 1 * 2 * 3 * 4 * 5
    120

    It can be expressed in terms of the gamma function.

    >>> z, m = 3.7, 2.1
    >>> sc.poch(z, m)
    20.529581933776953
    >>> sc.gamma(z + m) / sc.gamma(z)
    20.52958193377696

    """)

add_newdoc("pro_ang1",
    """
    pro_ang1(m, n, c, x)

    Prolate spheroidal angular function of the first kind and its derivative

    Computes the prolate spheroidal angular function of the first kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.

    Returns
    -------
    s
        Value of the function
    sp
        Value of the derivative vs x
    """)

add_newdoc("pro_ang1_cv",
    """
    pro_ang1_cv(m, n, c, cv, x)

    Prolate spheroidal angular function pro_ang1 for precomputed characteristic value

    Computes the prolate spheroidal angular function of the first kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires
    pre-computed characteristic value.

    Returns
    -------
    s
        Value of the function
    sp
        Value of the derivative vs x
    """)

add_newdoc("pro_cv",
    """
    pro_cv(m, n, c)

    Characteristic value of prolate spheroidal function

    Computes the characteristic value of prolate spheroidal wave
    functions of order `m`, `n` (n>=m) and spheroidal parameter `c`.
    """)

add_newdoc("pro_rad1",
    """
    pro_rad1(m, n, c, x)

    Prolate spheroidal radial function of the first kind and its derivative

    Computes the prolate spheroidal radial function of the first kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.

    Returns
    -------
    s
        Value of the function
    sp
        Value of the derivative vs x
    """)

add_newdoc("pro_rad1_cv",
    """
    pro_rad1_cv(m, n, c, cv, x)

    Prolate spheroidal radial function pro_rad1 for precomputed characteristic value

    Computes the prolate spheroidal radial function of the first kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires
    pre-computed characteristic value.

    Returns
    -------
    s
        Value of the function
    sp
        Value of the derivative vs x
    """)

add_newdoc("pro_rad2",
    """
    pro_rad2(m, n, c, x)

    Prolate spheroidal radial function of the second kind and its derivative

    Computes the prolate spheroidal radial function of the second kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.

    Returns
    -------
    s
        Value of the function
    sp
        Value of the derivative vs x
    """)

add_newdoc("pro_rad2_cv",
    """
    pro_rad2_cv(m, n, c, cv, x)

    Prolate spheroidal radial function pro_rad2 for precomputed characteristic value

    Computes the prolate spheroidal radial function of the second kind
    and its derivative (with respect to `x`) for mode parameters m>=0
    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires
    pre-computed characteristic value.

    Returns
    -------
    s
        Value of the function
    sp
        Value of the derivative vs x
    """)

add_newdoc("pseudo_huber",
    r"""
    pseudo_huber(delta, r)

    Pseudo-Huber loss function.

    .. math:: \mathrm{pseudo\_huber}(\delta, r) = \delta^2 \left( \sqrt{ 1 + \left( \frac{r}{\delta} \right)^2 } - 1 \right)

    Parameters
    ----------
    delta : ndarray
        Input array, indicating the soft quadratic vs. linear loss changepoint.
    r : ndarray
        Input array, possibly representing residuals.

    Returns
    -------
    res : ndarray
        The computed Pseudo-Huber loss function values.

    Notes
    -----
    This function is convex in :math:`r`.

    .. versionadded:: 0.15.0

    """)

add_newdoc("psi",
    """
    psi(z, out=None)

    The digamma function.

    The logarithmic derivative of the gamma function evaluated at ``z``.

    Parameters
    ----------
    z : array_like
        Real or complex argument.
    out : ndarray, optional
        Array for the computed values of ``psi``.

    Returns
    -------
    digamma : ndarray
        Computed values of ``psi``.

    Notes
    -----
    For large values not close to the negative real axis, ``psi`` is
    computed using the asymptotic series (5.11.2) from [1]_. For small
    arguments not close to the negative real axis, the recurrence
    relation (5.5.2) from [1]_ is used until the argument is large
    enough to use the asymptotic series. For values close to the
    negative real axis, the reflection formula (5.5.4) from [1]_ is
    used first. Note that ``psi`` has a family of zeros on the
    negative real axis which occur between the poles at nonpositive
    integers. Around the zeros the reflection formula suffers from
    cancellation and the implementation loses precision. The sole
    positive zero and the first negative zero, however, are handled
    separately by precomputing series expansions using [2]_, so the
    function should maintain full accuracy around the origin.

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/5
    .. [2] Fredrik Johansson and others.
           "mpmath: a Python library for arbitrary-precision floating-point arithmetic"
           (Version 0.19) http://mpmath.org/

    Examples
    --------
    >>> from scipy.special import psi
    >>> z = 3 + 4j
    >>> psi(z)
    (1.55035981733341+1.0105022091860445j)

    Verify psi(z) = psi(z + 1) - 1/z:

    >>> psi(z + 1) - 1/z
    (1.55035981733341+1.0105022091860445j)
    """)

add_newdoc("radian",
    """
    radian(d, m, s, out=None)

    Convert from degrees to radians.

    Returns the angle given in (d)egrees, (m)inutes, and (s)econds in
    radians.

    Parameters
    ----------
    d : array_like
        Degrees, can be real-valued.
    m : array_like
        Minutes, can be real-valued.
    s : array_like
        Seconds, can be real-valued.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Values of the inputs in radians.

    Examples
    --------
    >>> import scipy.special as sc

    There are many ways to specify an angle.

    >>> sc.radian(90, 0, 0)
    1.5707963267948966
    >>> sc.radian(0, 60 * 90, 0)
    1.5707963267948966
    >>> sc.radian(0, 0, 60**2 * 90)
    1.5707963267948966

    The inputs can be real-valued.

    >>> sc.radian(1.5, 0, 0)
    0.02617993877991494
    >>> sc.radian(1, 30, 0)
    0.02617993877991494

    """)

add_newdoc("rel_entr",
    r"""
    rel_entr(x, y, out=None)

    Elementwise function for computing relative entropy.

    .. math::

        \mathrm{rel\_entr}(x, y) =
            \begin{cases}
                x \log(x / y) & x > 0, y > 0 \\
                0 & x = 0, y \ge 0 \\
                \infty & \text{otherwise}
            \end{cases}

    Parameters
    ----------
    x, y : array_like
        Input arrays
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Relative entropy of the inputs

    See Also
    --------
    entr, kl_div

    Notes
    -----
    .. versionadded:: 0.15.0

    This function is jointly convex in x and y.

    The origin of this function is in convex programming; see
    [1]_. Given two discrete probability distributions :math:`p_1,
    \ldots, p_n` and :math:`q_1, \ldots, q_n`, to get the relative
    entropy of statistics compute the sum

    .. math::

        \sum_{i = 1}^n \mathrm{rel\_entr}(p_i, q_i).

    See [2]_ for details.

    References
    ----------
    .. [1] Grant, Boyd, and Ye, "CVX: Matlab Software for Disciplined Convex
        Programming", http://cvxr.com/cvx/
    .. [2] Kullback-Leibler divergence,
        https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    """)

add_newdoc("rgamma",
    r"""
    rgamma(z, out=None)

    Reciprocal of the gamma function.

    Defined as :math:`1 / \Gamma(z)`, where :math:`\Gamma` is the
    gamma function. For more on the gamma function see `gamma`.

    Parameters
    ----------
    z : array_like
        Real or complex valued input
    out : ndarray, optional
        Optional output array for the function results

    Returns
    -------
    scalar or ndarray
        Function results

    Notes
    -----
    The gamma function has no zeros and has simple poles at
    nonpositive integers, so `rgamma` is an entire function with zeros
    at the nonpositive integers. See the discussion in [dlmf]_ for
    more details.

    See Also
    --------
    gamma, gammaln, loggamma

    References
    ----------
    .. [dlmf] Nist, Digital Library of Mathematical functions,
        https://dlmf.nist.gov/5.2#i

    Examples
    --------
    >>> import scipy.special as sc

    It is the reciprocal of the gamma function.

    >>> sc.rgamma([1, 2, 3, 4])
    array([1.        , 1.        , 0.5       , 0.16666667])
    >>> 1 / sc.gamma([1, 2, 3, 4])
    array([1.        , 1.        , 0.5       , 0.16666667])

    It is zero at nonpositive integers.

    >>> sc.rgamma([0, -1, -2, -3])
    array([0., 0., 0., 0.])

    It rapidly underflows to zero along the positive real axis.

    >>> sc.rgamma([10, 100, 179])
    array([2.75573192e-006, 1.07151029e-156, 0.00000000e+000])

    """)

add_newdoc("round",
    """
    round(x, out=None)

    Round to the nearest integer.

    Returns the nearest integer to `x`.  If `x` ends in 0.5 exactly,
    the nearest even integer is chosen.

    Parameters
    ----------
    x : array_like
        Real valued input.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        The nearest integers to the elements of `x`. The result is of
        floating type, not integer type.

    Examples
    --------
    >>> import scipy.special as sc

    It rounds to even.

    >>> sc.round([0.5, 1.5])
    array([0., 2.])

    """)

add_newdoc("shichi",
    r"""
    shichi(x, out=None)

    Hyperbolic sine and cosine integrals.

    The hyperbolic sine integral is

    .. math::

      \int_0^x \frac{\sinh{t}}{t}dt

    and the hyperbolic cosine integral is

    .. math::

      \gamma + \log(x) + \int_0^x \frac{\cosh{t} - 1}{t} dt

    where :math:`\gamma` is Euler's constant and :math:`\log` is the
    principle branch of the logarithm.

    Parameters
    ----------
    x : array_like
        Real or complex points at which to compute the hyperbolic sine
        and cosine integrals.

    Returns
    -------
    si : ndarray
        Hyperbolic sine integral at ``x``
    ci : ndarray
        Hyperbolic cosine integral at ``x``

    Notes
    -----
    For real arguments with ``x < 0``, ``chi`` is the real part of the
    hyperbolic cosine integral. For such points ``chi(x)`` and ``chi(x
    + 0j)`` differ by a factor of ``1j*pi``.

    For real arguments the function is computed by calling Cephes'
    [1]_ *shichi* routine. For complex arguments the algorithm is based
    on Mpmath's [2]_ *shi* and *chi* routines.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    .. [2] Fredrik Johansson and others.
           "mpmath: a Python library for arbitrary-precision floating-point arithmetic"
           (Version 0.19) http://mpmath.org/
    """)

add_newdoc("sici",
    r"""
    sici(x, out=None)

    Sine and cosine integrals.

    The sine integral is

    .. math::

      \int_0^x \frac{\sin{t}}{t}dt

    and the cosine integral is

    .. math::

      \gamma + \log(x) + \int_0^x \frac{\cos{t} - 1}{t}dt

    where :math:`\gamma` is Euler's constant and :math:`\log` is the
    principle branch of the logarithm.

    Parameters
    ----------
    x : array_like
        Real or complex points at which to compute the sine and cosine
        integrals.

    Returns
    -------
    si : ndarray
        Sine integral at ``x``
    ci : ndarray
        Cosine integral at ``x``

    Notes
    -----
    For real arguments with ``x < 0``, ``ci`` is the real part of the
    cosine integral. For such points ``ci(x)`` and ``ci(x + 0j)``
    differ by a factor of ``1j*pi``.

    For real arguments the function is computed by calling Cephes'
    [1]_ *sici* routine. For complex arguments the algorithm is based
    on Mpmath's [2]_ *si* and *ci* routines.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    .. [2] Fredrik Johansson and others.
           "mpmath: a Python library for arbitrary-precision floating-point arithmetic"
           (Version 0.19) http://mpmath.org/
    """)

add_newdoc("sindg",
    """
    sindg(x, out=None)

    Sine of the angle `x` given in degrees.

    Parameters
    ----------
    x : array_like
        Angle, given in degrees.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Sine at the input.

    See Also
    --------
    cosdg, tandg, cotdg

    Examples
    --------
    >>> import scipy.special as sc

    It is more accurate than using sine directly.

    >>> x = 180 * np.arange(3)
    >>> sc.sindg(x)
    array([ 0., -0.,  0.])
    >>> np.sin(x * np.pi / 180)
    array([ 0.0000000e+00,  1.2246468e-16, -2.4492936e-16])

    """)

add_newdoc("smirnov",
    r"""
    smirnov(n, d)

    Kolmogorov-Smirnov complementary cumulative distribution function

    Returns the exact Kolmogorov-Smirnov complementary cumulative
    distribution function,(aka the Survival Function) of Dn+ (or Dn-)
    for a one-sided test of equality between an empirical and a
    theoretical distribution. It is equal to the probability that the
    maximum difference between a theoretical distribution and an empirical
    one based on `n` samples is greater than d.

    Parameters
    ----------
    n : int
      Number of samples
    d : float array_like
      Deviation between the Empirical CDF (ECDF) and the target CDF.

    Returns
    -------
    float
        The value(s) of smirnov(n, d), Prob(Dn+ >= d) (Also Prob(Dn- >= d))

    Notes
    -----
    `smirnov` is used by `stats.kstest` in the application of the
    Kolmogorov-Smirnov Goodness of Fit test. For historial reasons this
    function is exposed in `scpy.special`, but the recommended way to achieve
    the most accurate CDF/SF/PDF/PPF/ISF computations is to use the
    `stats.ksone` distribution.

    See Also
    --------
    smirnovi : The Inverse Survival Function for the distribution
    scipy.stats.ksone : Provides the functionality as a continuous distribution
    kolmogorov, kolmogi : Functions for the two-sided distribution

    Examples
    --------
    >>> from scipy.special import smirnov

    Show the probability of a gap at least as big as 0, 0.5 and 1.0 for a sample of size 5

    >>> smirnov(5, [0, 0.5, 1.0])
    array([ 1.   ,  0.056,  0.   ])

    Compare a sample of size 5 drawn from a source N(0.5, 1) distribution against
    a target N(0, 1) CDF.

    >>> from scipy.stats import norm
    >>> rng = np.random.default_rng()
    >>> n = 5
    >>> gendist = norm(0.5, 1)       # Normal distribution, mean 0.5, stddev 1
    >>> x = np.sort(gendist.rvs(size=n, random_state=rng))
    >>> x
    array([-1.3922078 , -0.13526532,  0.1371477 ,  0.18981686,  1.81948167])
    >>> target = norm(0, 1)
    >>> cdfs = target.cdf(x)
    >>> cdfs
    array([0.08192974, 0.44620105, 0.55454297, 0.57527368, 0.96558101])
    # Construct the Empirical CDF and the K-S statistics (Dn+, Dn-, Dn)
    >>> ecdfs = np.arange(n+1, dtype=float)/n
    >>> cols = np.column_stack([x, ecdfs[1:], cdfs, cdfs - ecdfs[:n], ecdfs[1:] - cdfs])
    >>> np.set_printoptions(precision=3)
    >>> cols
    array([[-1.392,  0.2  ,  0.082,  0.082,  0.118],
           [-0.135,  0.4  ,  0.446,  0.246, -0.046],
           [ 0.137,  0.6  ,  0.555,  0.155,  0.045],
           [ 0.19 ,  0.8  ,  0.575, -0.025,  0.225],
           [ 1.819,  1.   ,  0.966,  0.166,  0.034]])
    >>> gaps = cols[:, -2:]
    >>> Dnpm = np.max(gaps, axis=0)
    >>> print('Dn-=%f, Dn+=%f' % (Dnpm[0], Dnpm[1]))
    Dn-=0.246201, Dn+=0.224726
    >>> probs = smirnov(n, Dnpm)
    >>> print(chr(10).join(['For a sample of size %d drawn from a N(0, 1) distribution:' % n,
    ...      ' Smirnov n=%d: Prob(Dn- >= %f) = %.4f' % (n, Dnpm[0], probs[0]),
    ...      ' Smirnov n=%d: Prob(Dn+ >= %f) = %.4f' % (n, Dnpm[1], probs[1])]))
    For a sample of size 5 drawn from a N(0, 1) distribution:
     Smirnov n=5: Prob(Dn- >= 0.246201) = 0.4713
     Smirnov n=5: Prob(Dn+ >= 0.224726) = 0.5243

    Plot the Empirical CDF against the target N(0, 1) CDF

    >>> import matplotlib.pyplot as plt
    >>> plt.step(np.concatenate([[-3], x]), ecdfs, where='post', label='Empirical CDF')
    >>> x3 = np.linspace(-3, 3, 100)
    >>> plt.plot(x3, target.cdf(x3), label='CDF for N(0, 1)')
    >>> plt.ylim([0, 1]); plt.grid(True); plt.legend();
    # Add vertical lines marking Dn+ and Dn-
    >>> iminus, iplus = np.argmax(gaps, axis=0)
    >>> plt.vlines([x[iminus]], ecdfs[iminus], cdfs[iminus], color='r', linestyle='dashed', lw=4)
    >>> plt.vlines([x[iplus]], cdfs[iplus], ecdfs[iplus+1], color='m', linestyle='dashed', lw=4)
    >>> plt.show()
    """)

add_newdoc("smirnovi",
    """
    smirnovi(n, p)

    Inverse to `smirnov`

    Returns `d` such that ``smirnov(n, d) == p``, the critical value
    corresponding to `p`.

    Parameters
    ----------
    n : int
      Number of samples
    p : float array_like
        Probability

    Returns
    -------
    float
        The value(s) of smirnovi(n, p), the critical values.

    Notes
    -----
    `smirnov` is used by `stats.kstest` in the application of the
    Kolmogorov-Smirnov Goodness of Fit test. For historial reasons this
    function is exposed in `scpy.special`, but the recommended way to achieve
    the most accurate CDF/SF/PDF/PPF/ISF computations is to use the
    `stats.ksone` distribution.

    See Also
    --------
    smirnov  : The Survival Function (SF) for the distribution
    scipy.stats.ksone : Provides the functionality as a continuous distribution
    kolmogorov, kolmogi, scipy.stats.kstwobign : Functions for the two-sided distribution
    """)

add_newdoc("_smirnovc",
    """
    _smirnovc(n, d)
     Internal function, do not use.
    """)

add_newdoc("_smirnovci",
    """
     Internal function, do not use.
    """)

add_newdoc("_smirnovp",
    """
    _smirnovp(n, p)
     Internal function, do not use.
    """)

add_newdoc("spence",
    r"""
    spence(z, out=None)

    Spence's function, also known as the dilogarithm.

    It is defined to be

    .. math::
      \int_1^z \frac{\log(t)}{1 - t}dt

    for complex :math:`z`, where the contour of integration is taken
    to avoid the branch cut of the logarithm. Spence's function is
    analytic everywhere except the negative real axis where it has a
    branch cut.

    Parameters
    ----------
    z : array_like
        Points at which to evaluate Spence's function

    Returns
    -------
    s : ndarray
        Computed values of Spence's function

    Notes
    -----
    There is a different convention which defines Spence's function by
    the integral

    .. math::
      -\int_0^z \frac{\log(1 - t)}{t}dt;

    this is our ``spence(1 - z)``.
    """)

add_newdoc("stdtr",
    """
    stdtr(df, t)

    Student t distribution cumulative distribution function

    Returns the integral from minus infinity to t of the Student t
    distribution with df > 0 degrees of freedom::

       gamma((df+1)/2)/(sqrt(df*pi)*gamma(df/2)) *
       integral((1+x**2/df)**(-df/2-1/2), x=-inf..t)

    """)

add_newdoc("stdtridf",
    """
    stdtridf(p, t)

    Inverse of `stdtr` vs df

    Returns the argument df such that stdtr(df, t) is equal to `p`.
    """)

add_newdoc("stdtrit",
    """
    stdtrit(df, p)

    Inverse of `stdtr` vs `t`

    Returns the argument `t` such that stdtr(df, t) is equal to `p`.
    """)

add_newdoc("struve",
    r"""
    struve(v, x)

    Struve function.

    Return the value of the Struve function of order `v` at `x`.  The Struve
    function is defined as,

    .. math::
        H_v(x) = (z/2)^{v + 1} \sum_{n=0}^\infty \frac{(-1)^n (z/2)^{2n}}{\Gamma(n + \frac{3}{2}) \Gamma(n + v + \frac{3}{2})},

    where :math:`\Gamma` is the gamma function.

    Parameters
    ----------
    v : array_like
        Order of the Struve function (float).
    x : array_like
        Argument of the Struve function (float; must be positive unless `v` is
        an integer).

    Returns
    -------
    H : ndarray
        Value of the Struve function of order `v` at `x`.

    Notes
    -----
    Three methods discussed in [1]_ are used to evaluate the Struve function:

    - power series
    - expansion in Bessel functions (if :math:`|z| < |v| + 20`)
    - asymptotic large-z expansion (if :math:`z \geq 0.7v + 12`)

    Rounding errors are estimated based on the largest terms in the sums, and
    the result associated with the smallest error is returned.

    See also
    --------
    modstruve

    References
    ----------
    .. [1] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/11

    """)

add_newdoc("tandg",
    """
    tandg(x, out=None)

    Tangent of angle `x` given in degrees.

    Parameters
    ----------
    x : array_like
        Angle, given in degrees.
    out : ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    scalar or ndarray
        Tangent at the input.

    See Also
    --------
    sindg, cosdg, cotdg

    Examples
    --------
    >>> import scipy.special as sc

    It is more accurate than using tangent directly.

    >>> x = 180 * np.arange(3)
    >>> sc.tandg(x)
    array([0., 0., 0.])
    >>> np.tan(x * np.pi / 180)
    array([ 0.0000000e+00, -1.2246468e-16, -2.4492936e-16])

    """)

add_newdoc("tklmbda",
    """
    tklmbda(x, lmbda)

    Tukey-Lambda cumulative distribution function

    """)

add_newdoc("wofz",
    """
    wofz(z)

    Faddeeva function

    Returns the value of the Faddeeva function for complex argument::

        exp(-z**2) * erfc(-i*z)

    See Also
    --------
    dawsn, erf, erfc, erfcx, erfi

    References
    ----------
    .. [1] Steven G. Johnson, Faddeeva W function implementation.
       http://ab-initio.mit.edu/Faddeeva

    Examples
    --------
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(-3, 3)
    >>> z = special.wofz(x)

    >>> plt.plot(x, z.real, label='wofz(x).real')
    >>> plt.plot(x, z.imag, label='wofz(x).imag')
    >>> plt.xlabel('$x$')
    >>> plt.legend(framealpha=1, shadow=True)
    >>> plt.grid(alpha=0.25)
    >>> plt.show()

    """)

add_newdoc("xlogy",
    """
    xlogy(x, y)

    Compute ``x*log(y)`` so that the result is 0 if ``x = 0``.

    Parameters
    ----------
    x : array_like
        Multiplier
    y : array_like
        Argument

    Returns
    -------
    z : array_like
        Computed x*log(y)

    Notes
    -----

    .. versionadded:: 0.13.0

    """)

add_newdoc("xlog1py",
    """
    xlog1py(x, y)

    Compute ``x*log1p(y)`` so that the result is 0 if ``x = 0``.

    Parameters
    ----------
    x : array_like
        Multiplier
    y : array_like
        Argument

    Returns
    -------
    z : array_like
        Computed x*log1p(y)

    Notes
    -----

    .. versionadded:: 0.13.0

    """)

add_newdoc("y0",
    r"""
    y0(x)

    Bessel function of the second kind of order 0.

    Parameters
    ----------
    x : array_like
        Argument (float).

    Returns
    -------
    Y : ndarray
        Value of the Bessel function of the second kind of order 0 at `x`.

    Notes
    -----

    The domain is divided into the intervals [0, 5] and (5, infinity). In the
    first interval a rational approximation :math:`R(x)` is employed to
    compute,

    .. math::

        Y_0(x) = R(x) + \frac{2 \log(x) J_0(x)}{\pi},

    where :math:`J_0` is the Bessel function of the first kind of order 0.

    In the second interval, the Hankel asymptotic expansion is employed with
    two rational functions of degree 6/6 and 7/7.

    This function is a wrapper for the Cephes [1]_ routine `y0`.

    See also
    --------
    j0
    yv

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

add_newdoc("y1",
    """
    y1(x)

    Bessel function of the second kind of order 1.

    Parameters
    ----------
    x : array_like
        Argument (float).

    Returns
    -------
    Y : ndarray
        Value of the Bessel function of the second kind of order 1 at `x`.

    Notes
    -----

    The domain is divided into the intervals [0, 8] and (8, infinity). In the
    first interval a 25 term Chebyshev expansion is used, and computing
    :math:`J_1` (the Bessel function of the first kind) is required. In the
    second, the asymptotic trigonometric representation is employed using two
    rational functions of degree 5/5.

    This function is a wrapper for the Cephes [1]_ routine `y1`.

    See also
    --------
    j1
    yn
    yv

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

add_newdoc("yn",
    r"""
    yn(n, x)

    Bessel function of the second kind of integer order and real argument.

    Parameters
    ----------
    n : array_like
        Order (integer).
    z : array_like
        Argument (float).

    Returns
    -------
    Y : ndarray
        Value of the Bessel function, :math:`Y_n(x)`.

    Notes
    -----
    Wrapper for the Cephes [1]_ routine `yn`.

    The function is evaluated by forward recurrence on `n`, starting with
    values computed by the Cephes routines `y0` and `y1`. If `n = 0` or 1,
    the routine for `y0` or `y1` is called directly.

    See also
    --------
    yv : For real order and real or complex argument.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library,
           http://www.netlib.org/cephes/
    """)

add_newdoc("yv",
    r"""
    yv(v, z)

    Bessel function of the second kind of real order and complex argument.

    Parameters
    ----------
    v : array_like
        Order (float).
    z : array_like
        Argument (float or complex).

    Returns
    -------
    Y : ndarray
        Value of the Bessel function of the second kind, :math:`Y_v(x)`.

    Notes
    -----
    For positive `v` values, the computation is carried out using the
    AMOS [1]_ `zbesy` routine, which exploits the connection to the Hankel
    Bessel functions :math:`H_v^{(1)}` and :math:`H_v^{(2)}`,

    .. math:: Y_v(z) = \frac{1}{2\imath} (H_v^{(1)} - H_v^{(2)}).

    For negative `v` values the formula,

    .. math:: Y_{-v}(z) = Y_v(z) \cos(\pi v) + J_v(z) \sin(\pi v)

    is used, where :math:`J_v(z)` is the Bessel function of the first kind,
    computed using the AMOS routine `zbesj`.  Note that the second term is
    exactly zero for integer `v`; to improve accuracy the second term is
    explicitly omitted for `v` values such that `v = floor(v)`.

    See also
    --------
    yve : :math:`Y_v` with leading exponential behavior stripped off.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/

    """)

add_newdoc("yve",
    r"""
    yve(v, z)

    Exponentially scaled Bessel function of the second kind of real order.

    Returns the exponentially scaled Bessel function of the second
    kind of real order `v` at complex `z`::

        yve(v, z) = yv(v, z) * exp(-abs(z.imag))

    Parameters
    ----------
    v : array_like
        Order (float).
    z : array_like
        Argument (float or complex).

    Returns
    -------
    Y : ndarray
        Value of the exponentially scaled Bessel function.

    Notes
    -----
    For positive `v` values, the computation is carried out using the
    AMOS [1]_ `zbesy` routine, which exploits the connection to the Hankel
    Bessel functions :math:`H_v^{(1)}` and :math:`H_v^{(2)}`,

    .. math:: Y_v(z) = \frac{1}{2\imath} (H_v^{(1)} - H_v^{(2)}).

    For negative `v` values the formula,

    .. math:: Y_{-v}(z) = Y_v(z) \cos(\pi v) + J_v(z) \sin(\pi v)

    is used, where :math:`J_v(z)` is the Bessel function of the first kind,
    computed using the AMOS routine `zbesj`.  Note that the second term is
    exactly zero for integer `v`; to improve accuracy the second term is
    explicitly omitted for `v` values such that `v = floor(v)`.

    References
    ----------
    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
           of a Complex Argument and Nonnegative Order",
           http://netlib.org/amos/
    """)

add_newdoc("_zeta",
    """
    _zeta(x, q)

    Internal function, Hurwitz zeta.

    """)

add_newdoc("zetac",
    """
    zetac(x)

    Riemann zeta function minus 1.

    This function is defined as

    .. math:: \\zeta(x) = \\sum_{k=2}^{\\infty} 1 / k^x,

    where ``x > 1``.  For ``x < 1`` the analytic continuation is
    computed. For more information on the Riemann zeta function, see
    [dlmf]_.

    Parameters
    ----------
    x : array_like of float
        Values at which to compute zeta(x) - 1 (must be real).

    Returns
    -------
    out : array_like
        Values of zeta(x) - 1.

    See Also
    --------
    zeta

    Examples
    --------
    >>> from scipy.special import zetac, zeta

    Some special values:

    >>> zetac(2), np.pi**2/6 - 1
    (0.64493406684822641, 0.6449340668482264)

    >>> zetac(-1), -1.0/12 - 1
    (-1.0833333333333333, -1.0833333333333333)

    Compare ``zetac(x)`` to ``zeta(x) - 1`` for large `x`:

    >>> zetac(60), zeta(60) - 1
    (8.673617380119933e-19, 0.0)

    References
    ----------
    .. [dlmf] NIST Digital Library of Mathematical Functions
              https://dlmf.nist.gov/25

    """)

add_newdoc("_riemann_zeta",
    """
    Internal function, use `zeta` instead.
    """)

add_newdoc("_struve_asymp_large_z",
    """
    _struve_asymp_large_z(v, z, is_h)

    Internal function for testing `struve` & `modstruve`

    Evaluates using asymptotic expansion

    Returns
    -------
    v, err
    """)

add_newdoc("_struve_power_series",
    """
    _struve_power_series(v, z, is_h)

    Internal function for testing `struve` & `modstruve`

    Evaluates using power series

    Returns
    -------
    v, err
    """)

add_newdoc("_struve_bessel_series",
    """
    _struve_bessel_series(v, z, is_h)

    Internal function for testing `struve` & `modstruve`

    Evaluates using Bessel function series

    Returns
    -------
    v, err
    """)

add_newdoc("_spherical_jn",
    """
    Internal function, use `spherical_jn` instead.
    """)

add_newdoc("_spherical_jn_d",
    """
    Internal function, use `spherical_jn` instead.
    """)

add_newdoc("_spherical_yn",
    """
    Internal function, use `spherical_yn` instead.
    """)

add_newdoc("_spherical_yn_d",
    """
    Internal function, use `spherical_yn` instead.
    """)

add_newdoc("_spherical_in",
    """
    Internal function, use `spherical_in` instead.
    """)

add_newdoc("_spherical_in_d",
    """
    Internal function, use `spherical_in` instead.
    """)

add_newdoc("_spherical_kn",
    """
    Internal function, use `spherical_kn` instead.
    """)

add_newdoc("_spherical_kn_d",
    """
    Internal function, use `spherical_kn` instead.
    """)

add_newdoc("loggamma",
    r"""
    loggamma(z, out=None)

    Principal branch of the logarithm of the gamma function.

    Defined to be :math:`\log(\Gamma(x))` for :math:`x > 0` and
    extended to the complex plane by analytic continuation. The
    function has a single branch cut on the negative real axis.

    .. versionadded:: 0.18.0

    Parameters
    ----------
    z : array-like
        Values in the complex plain at which to compute ``loggamma``
    out : ndarray, optional
        Output array for computed values of ``loggamma``

    Returns
    -------
    loggamma : ndarray
        Values of ``loggamma`` at z.

    Notes
    -----
    It is not generally true that :math:`\log\Gamma(z) =
    \log(\Gamma(z))`, though the real parts of the functions do
    agree. The benefit of not defining `loggamma` as
    :math:`\log(\Gamma(z))` is that the latter function has a
    complicated branch cut structure whereas `loggamma` is analytic
    except for on the negative real axis.

    The identities

    .. math::
      \exp(\log\Gamma(z)) &= \Gamma(z) \\
      \log\Gamma(z + 1) &= \log(z) + \log\Gamma(z)

    make `loggamma` useful for working in complex logspace.

    On the real line `loggamma` is related to `gammaln` via
    ``exp(loggamma(x + 0j)) = gammasgn(x)*exp(gammaln(x))``, up to
    rounding error.

    The implementation here is based on [hare1997]_.

    See also
    --------
    gammaln : logarithm of the absolute value of the gamma function
    gammasgn : sign of the gamma function

    References
    ----------
    .. [hare1997] D.E.G. Hare,
      *Computing the Principal Branch of log-Gamma*,
      Journal of Algorithms, Volume 25, Issue 2, November 1997, pages 221-236.
    """)

add_newdoc("_sinpi",
    """
    Internal function, do not use.
    """)

add_newdoc("_cospi",
    """
    Internal function, do not use.
    """)

add_newdoc("owens_t",
    """
    owens_t(h, a)

    Owen's T Function.

    The function T(h, a) gives the probability of the event
    (X > h and 0 < Y < a * X) where X and Y are independent
    standard normal random variables.

    Parameters
    ----------
    h: array_like
        Input value.
    a: array_like
        Input value.

    Returns
    -------
    t: scalar or ndarray
        Probability of the event (X > h and 0 < Y < a * X),
        where X and Y are independent standard normal random variables.

    Examples
    --------
    >>> from scipy import special
    >>> a = 3.5
    >>> h = 0.78
    >>> special.owens_t(h, a)
    0.10877216734852274

    References
    ----------
    .. [1] M. Patefield and D. Tandy, "Fast and accurate calculation of
           Owen's T Function", Statistical Software vol. 5, pp. 1-25, 2000.
    """)

add_newdoc("_factorial",
    """
    Internal function, do not use.
    """)

add_newdoc("wright_bessel",
    r"""
    wright_bessel(a, b, x)

    Wright's generalized Bessel function.

    Wright's generalized Bessel function is an entire function and defined as

    .. math:: \Phi(a, b; x) = \sum_{k=0}^\infty \frac{x^k}{k! \Gamma(a k + b)}

    See also [1].

    Parameters
    ----------
    a : array_like of float
        a >= 0
    b : array_like of float
        b >= 0
    x : array_like of float
        x >= 0

    Notes
    -----
    Due to the compexity of the function with its three parameters, only
    non-negative arguments are implemented.

    Examples
    --------
    >>> from scipy.special import wright_bessel
    >>> a, b, x = 1.5, 1.1, 2.5
    >>> wright_bessel(a, b-1, x)
    4.5314465939443025

    Now, let us verify the relation

    .. math:: \Phi(a, b-1; x) = a x \Phi(a, b+a; x) + (b-1) \Phi(a, b; x)

    >>> a * x * wright_bessel(a, b+a, x) + (b-1) * wright_bessel(a, b, x)
    4.5314465939443025

    References
    ----------
    .. [1] Digital Library of Mathematical Functions, 10.46.
           https://dlmf.nist.gov/10.46.E1
    """)


add_newdoc("ndtri_exp",
    r"""
    ndtri_exp(y)

    Inverse of `log_ndtr` vs x. Allows for greater precision than
    `ndtri` composed with `numpy.exp` for very small values of y and for
    y close to 0.

    Parameters
    ----------
    y : array_like of float

    Returns
    -------
    scalar or ndarray
        Inverse of the log CDF of the standard normal distribution, evaluated
        at y.

    Examples
    --------
    >>> import scipy.special as sc

    `ndtri_exp` agrees with the naive implementation when the latter does
    not suffer from underflow.

    >>> sc.ndtri_exp(-1)
    -0.33747496376420244
    >>> sc.ndtri(np.exp(-1))
    -0.33747496376420244

    For extreme values of y, the naive approach fails

    >>> sc.ndtri(np.exp(-800))
    -inf
    >>> sc.ndtri(np.exp(-1e-20))
    inf

    whereas `ndtri_exp` is still able to compute the result to high precision.

    >>> sc.ndtri_exp(-800)
    -39.88469483825668
    >>> sc.ndtri_exp(-1e-20)
    9.262340089798409

    See Also
    --------
    log_ndtr, ndtri, ndtr
    """)
