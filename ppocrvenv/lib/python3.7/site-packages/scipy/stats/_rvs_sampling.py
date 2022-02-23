# -*- coding: utf-8 -*-
import numpy as np
from scipy._lib._util import check_random_state
from scipy.interpolate import CubicHermiteSpline
from scipy import stats


def rvs_ratio_uniforms(pdf, umax, vmin, vmax, size=1, c=0, random_state=None):
    """
    Generate random samples from a probability density function using the
    ratio-of-uniforms method.

    Parameters
    ----------
    pdf : callable
        A function with signature `pdf(x)` that is proportional to the
        probability density function of the distribution.
    umax : float
        The upper bound of the bounding rectangle in the u-direction.
    vmin : float
        The lower bound of the bounding rectangle in the v-direction.
    vmax : float
        The upper bound of the bounding rectangle in the v-direction.
    size : int or tuple of ints, optional
        Defining number of random variates (default is 1).
    c : float, optional.
        Shift parameter of ratio-of-uniforms method, see Notes. Default is 0.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    rvs : ndarray
        The random variates distributed according to the probability
        distribution defined by the pdf.

    Notes
    -----
    Given a univariate probability density function `pdf` and a constant `c`,
    define the set ``A = {(u, v) : 0 < u <= sqrt(pdf(v/u + c))}``.
    If `(U, V)` is a random vector uniformly distributed over `A`,
    then `V/U + c` follows a distribution according to `pdf`.

    The above result (see [1]_, [2]_) can be used to sample random variables
    using only the pdf, i.e. no inversion of the cdf is required. Typical
    choices of `c` are zero or the mode of `pdf`. The set `A` is a subset of
    the rectangle ``R = [0, umax] x [vmin, vmax]`` where

    - ``umax = sup sqrt(pdf(x))``
    - ``vmin = inf (x - c) sqrt(pdf(x))``
    - ``vmax = sup (x - c) sqrt(pdf(x))``

    In particular, these values are finite if `pdf` is bounded and
    ``x**2 * pdf(x)`` is bounded (i.e. subquadratic tails).
    One can generate `(U, V)` uniformly on `R` and return
    `V/U + c` if `(U, V)` are also in `A` which can be directly
    verified.

    The algorithm is not changed if one replaces `pdf` by k * `pdf` for any
    constant k > 0. Thus, it is often convenient to work with a function
    that is proportional to the probability density function by dropping
    unneccessary normalization factors.

    Intuitively, the method works well if `A` fills up most of the
    enclosing rectangle such that the probability is high that `(U, V)`
    lies in `A` whenever it lies in `R` as the number of required
    iterations becomes too large otherwise. To be more precise, note that
    the expected number of iterations to draw `(U, V)` uniformly
    distributed on `R` such that `(U, V)` is also in `A` is given by
    the ratio ``area(R) / area(A) = 2 * umax * (vmax - vmin) / area(pdf)``,
    where `area(pdf)` is the integral of `pdf` (which is equal to one if the
    probability density function is used but can take on other values if a
    function proportional to the density is used). The equality holds since
    the area of `A` is equal to 0.5 * area(pdf) (Theorem 7.1 in [1]_).
    If the sampling fails to generate a single random variate after 50000
    iterations (i.e. not a single draw is in `A`), an exception is raised.

    If the bounding rectangle is not correctly specified (i.e. if it does not
    contain `A`), the algorithm samples from a distribution different from
    the one given by `pdf`. It is therefore recommended to perform a
    test such as `~scipy.stats.kstest` as a check.

    References
    ----------
    .. [1] L. Devroye, "Non-Uniform Random Variate Generation",
       Springer-Verlag, 1986.

    .. [2] W. Hoermann and J. Leydold, "Generating generalized inverse Gaussian
       random variates", Statistics and Computing, 24(4), p. 547--557, 2014.

    .. [3] A.J. Kinderman and J.F. Monahan, "Computer Generation of Random
       Variables Using the Ratio of Uniform Deviates",
       ACM Transactions on Mathematical Software, 3(3), p. 257--260, 1977.

    Examples
    --------
    >>> from scipy import stats
    >>> rng = np.random.default_rng()

    Simulate normally distributed random variables. It is easy to compute the
    bounding rectangle explicitly in that case. For simplicity, we drop the
    normalization factor of the density.

    >>> f = lambda x: np.exp(-x**2 / 2)
    >>> v_bound = np.sqrt(f(np.sqrt(2))) * np.sqrt(2)
    >>> umax, vmin, vmax = np.sqrt(f(0)), -v_bound, v_bound
    >>> rvs = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=2500,
    ...                                random_state=rng)

    The K-S test confirms that the random variates are indeed normally
    distributed (normality is not rejected at 5% significance level):

    >>> stats.kstest(rvs, 'norm')[1]
    0.250634764150542

    The exponential distribution provides another example where the bounding
    rectangle can be determined explicitly.

    >>> rvs = stats.rvs_ratio_uniforms(lambda x: np.exp(-x), umax=1,
    ...                                vmin=0, vmax=2*np.exp(-1), size=1000,
    ...                                random_state=rng)
    >>> stats.kstest(rvs, 'expon')[1]
    0.21121052054580314

    """
    if vmin >= vmax:
        raise ValueError("vmin must be smaller than vmax.")

    if umax <= 0:
        raise ValueError("umax must be positive.")

    size1d = tuple(np.atleast_1d(size))
    N = np.prod(size1d)  # number of rvs needed, reshape upon return

    # start sampling using ratio of uniforms method
    rng = check_random_state(random_state)
    x = np.zeros(N)
    simulated, i = 0, 1

    # loop until N rvs have been generated: expected runtime is finite.
    # to avoid infinite loop, raise exception if not a single rv has been
    # generated after 50000 tries. even if the expected numer of iterations
    # is 1000, the probability of this event is (1-1/1000)**50000
    # which is of order 10e-22
    while simulated < N:
        k = N - simulated
        # simulate uniform rvs on [0, umax] and [vmin, vmax]
        u1 = umax * rng.uniform(size=k)
        v1 = rng.uniform(vmin, vmax, size=k)
        # apply rejection method
        rvs = v1 / u1 + c
        accept = (u1**2 <= pdf(rvs))
        num_accept = np.sum(accept)
        if num_accept > 0:
            x[simulated:(simulated + num_accept)] = rvs[accept]
            simulated += num_accept

        if (simulated == 0) and (i*N >= 50000):
            msg = ("Not a single random variate could be generated in {} "
                   "attempts. The ratio of uniforms method does not appear "
                   "to work for the provided parameters. Please check the "
                   "pdf and the bounds.".format(i*N))
            raise RuntimeError(msg)
        i += 1

    return np.reshape(x, size1d)


class NumericalInverseHermite:
    r"""
    A Hermite spline fast numerical inverse of a probability distribution.

    The initializer of `NumericalInverseHermite` accepts `dist`, an object
    representing a continuous distribution, and provides an object with methods
    that approximate `dist.ppf` and `dist.rvs`. For most distributions,
    these methods are faster than those of `dist` itself.

    Parameters
    ----------
    dist : object
        Object representing the distribution for which a fast numerical inverse
        is desired; for instance, a frozen instance of a `scipy.stats`
        continuous distribution. See Notes and Examples for details.
    tol : float, optional
        u-error tolerance (see Notes). The default is 1e-12.
    max_intervals : int, optional
        Maximum number of intervals in the cubic Hermite spline used to
        approximate the percent point function. The default is 100000.

    Attributes
    ----------
    intervals : int
        The number of intervals of the interpolant.
    midpoint_error : float
        The maximum u-error at an interpolant interval midpoint.

    Notes
    -----
    `NumericalInverseHermite` approximates the inverse of a continuous
    statistical distribution's CDF with a cubic Hermite spline.

    As described in [1]_, it begins by evaluating the distribution's PDF and
    CDF at a mesh of quantiles ``x`` within the distribution's support.
    It uses the results to fit a cubic Hermite spline ``H`` such that
    ``H(p) == x``, where ``p`` is the array of percentiles corresponding
    with the quantiles ``x``. Therefore, the spline approximates the inverse
    of the distribution's CDF to machine precision at the percentiles ``p``,
    but typically, the spline will not be as accurate at the midpoints between
    the percentile points::

        p_mid = (p[:-1] + p[1:])/2

    so the mesh of quantiles is refined as needed to reduce the maximum
    "u-error"::

        u_error = np.max(np.abs(dist.cdf(H(p_mid)) - p_mid))

    below the specified tolerance `tol`. Refinement stops when the required
    tolerance is achieved or when the number of mesh intervals after the next
    refinement could exceed the maximum allowed number `max_intervals`.

    The object `dist` must have methods ``pdf``, ``cdf``, and ``ppf`` that
    behave like those of a *frozen* instance of `scipy.stats.rv_continuous`.
    Specifically, it must have methods ``pdf`` and ``cdf`` that accept exactly
    one ndarray argument ``x`` and return the probability density function and
    cumulative density function (respectively) at ``x``. The object must also
    have a method ``ppf`` that accepts a float ``p`` and returns the percentile
    point function at ``p``. The object may also have a method ``isf`` that
    accepts a float ``p`` and returns the inverse survival function at ``p``;
    if it does not, it will be assigned an attribute ``isf`` that calculates
    the inverse survival function using ``ppf``. The ``ppf`` and
    ``isf` methods will each be evaluated at a small positive float ``p``
    (e.g. ``p = utol/10``), and the domain over which the approximate numerical
    inverse is defined will be ``ppf(p)`` to ``isf(p)``. The approximation will
    not be accurate in the extreme tails beyond this domain.

    References
    ----------
    .. [1] Hörmann, Wolfgang, and Josef Leydold. "Continuous random variate
           generation by fast numerical inversion." ACM Transactions on
           Modeling and Computer Simulation (TOMACS) 13.4 (2003): 347-362.

    Examples
    --------
    For some distributions, ``dist.ppf`` and ``dist.rvs`` are quite slow.
    For instance, consider `scipy.stats.genexpon`. We freeze the distribution
    by passing all shape parameters into its initializer and time the resulting
    object's ``ppf`` and ``rvs`` functions.

    >>> import numpy as np
    >>> from scipy import stats
    >>> from timeit import timeit
    >>> time_once = lambda f: f"{timeit(f, number=1)*1000:.6} ms"
    >>> dist = stats.genexpon(9, 16, 3)  # freeze the distribution
    >>> p = np.linspace(0.01, 0.99, 99)  # percentiles from 1% to 99%
    >>> time_once(lambda: dist.ppf(p))
    '154.565 ms'  # may vary

    >>> time_once(lambda: dist.rvs(size=100))
    '148.979 ms'  # may vary

    The `NumericalInverseHermite` has a method that approximates ``dist.ppf``.

    >>> from scipy.stats import NumericalInverseHermite
    >>> fni = NumericalInverseHermite(dist)
    >>> np.allclose(fni.ppf(p), dist.ppf(p))
    True

    In some cases, it is faster to both generate the fast numerical inverse
    and use it than to call ``dist.ppf``.

    >>> def time_me():
    ...     fni = NumericalInverseHermite(dist)
    ...     fni.ppf(p)
    >>> time_once(time_me)
    '11.9222 ms'  # may vary

    After generating the fast numerical inverse, subsequent calls to its
    methods are much faster.
    >>> time_once(lambda: fni.ppf(p))
    '0.0819 ms'  # may vary

    The fast numerical inverse can also be used to generate random variates
    using inverse transform sampling.

    >>> time_once(lambda: fni.rvs(size=100))
    '0.0911 ms'  # may vary

    Depending on the implementation of the distribution's random sampling
    method, the random variates generated may be nearly identical, given
    the same random state.

    >>> # `seed` ensures identical random streams are used by each `rvs` method
    >>> seed = 500072020
    >>> rvs1 = dist.rvs(size=100, random_state=np.random.default_rng(seed))
    >>> rvs2 = fni.rvs(size=100, random_state=np.random.default_rng(seed))
    >>> np.allclose(rvs1, rvs2)
    True

    To use `NumericalInverseHermite` with a custom distribution, users may
    subclass  `scipy.stats.rv_continuous` and initialize a frozen instance or
    create an object with equivalent ``pdf``, ``cdf``, and ``ppf`` methods.
    For instance, the following object represents the standard normal
    distribution. For simplicity, we use `scipy.special.ndtr` and
    `scipy.special.ndtri` to compute the ``cdf`` and ``ppf``, respectively.

    >>> from scipy.special import ndtr, ndtri
    >>>
    >>> class MyNormal:
    ...
    ...     def pdf(self, x):
    ...        return 1/np.sqrt(2*np.pi) * np.exp(-x**2 / 2)
    ...
    ...     def cdf(self, x):
    ...        return ndtr(x)
    ...
    ...     def ppf(self, x):
    ...        return ndtri(x)
    ...
    >>> dist1 = MyNormal()
    >>> fni1 = NumericalInverseHermite(dist1)
    >>>
    >>> dist2 = stats.norm()
    >>> fni2 = NumericalInverseHermite(dist2)
    >>>
    >>> print(fni1.rvs(random_state=seed), fni2.rvs(random_state=seed))
    -1.9603810921759424 -1.9603810921747074

    """

    def __init__(self, dist, *, tol=1e-12, max_intervals=100000):
        res = _fast_numerical_inverse(dist, tol, max_intervals)
        H, eu, intervals, a, b = res
        self.H = H
        self.midpoint_error = eu
        self.intervals = intervals
        self._a, self._b = a, b

    def ppf(self, q):
        r"""
        Approximate percent point function (inverse `cdf`) of the given RV.

        Parameters
        ----------
        q : array_like
            lower tail probability.

        Returns
        -------
        x : array_like
            quantile corresponding to the lower tail probability `q`.

        """
        q = np.asarray(q)  # no harm; self.H always returns an array
        result = np.zeros_like(q, dtype=np.float64)
        i = (q >= 0) & (q <= 1)
        result[i] = self.H(q[i])
        result[~i] = np.nan
        return result

    def rvs(self, size=None, random_state=None):
        """
        Random variates of the given RV.

        The `random_state` is used to draw uniform pseudo-random variates, and
        these are converted to pseudo-random variates of the given RV using
        inverse transform sampling.

        Parameters
        ----------
        size : int, tuple of ints, or None; optional
            Defines shape of array of random variates. Default is ``None``.
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            Defines the object to use for drawing pseudorandom variates.
            If `random_state` is ``None`` the `np.random.RandomState`
            singleton is used.
            If `random_state` is an ``int``, a new ``RandomState`` instance is
            used, seeded with `random_state`.
            If `random_state` is already a ``RandomState`` or ``Generator``
            instance, then that object is used.
            Default is None.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`. If `size` is ``None``, a scalar
            is returned.
        """
        random_state = check_random_state(random_state)
        uniform = random_state.uniform(size=size)
        # scale to valid domain of interpolant
        uniform = self._a + uniform * (self._b - self._a)
        return self.ppf(uniform)

    def qrvs(self, size=None, d=None, qmc_engine=None):
        """
        Quasi-random variates of the given RV.

        The `qmc_engine` is used to draw uniform quasi-random variates, and
        these are converted to quasi-random variates of the given RV using
        inverse transform sampling.

        Parameters
        ----------
        size : int, tuple of ints, or None; optional
            Defines shape of random variates array. Default is ``None``.
        d : int or None, optional
            Defines dimension of uniform quasi-random variates to be
            transformed. Default is ``None``.
        qmc_engine : scipy.stats.qmc.QMCEngine(d=1), optional
            Defines the object to use for drawing
            quasi-random variates. Default is ``None``, which uses
            `scipy.stats.qmc.Halton(1)`.

        Returns
        -------
        rvs : ndarray or scalar
            Quasi-random variates. See Notes for shape information.

        Notes
        -----
        The shape of the output array depends on `size`, `d`, and `qmc_engine`.
        The intent is for the interface to be natural, but the detailed rules
        to achieve this are complicated.

        - If `qmc_engine` is ``None``, a `scipy.stats.qmc.Halton` instance is
          created with dimension `d`. If `d` is not provided, ``d=1``.
        - If `qmc_engine` is not ``None`` and `d` is ``None``, `d` is
          determined from the dimension of the `qmc_engine`.
        - If `qmc_engine` is not ``None`` and `d` is not ``None`` but the
          dimensions are inconsistent, a ``ValueError`` is raised.
        - After `d` is determined according to the rules above, the output
          shape is ``tuple_shape + d_shape``, where:

              - ``tuple_shape = tuple()`` if `size` is ``None``,
              - ``tuple_shape = (size,)`` if `size` is an ``int``,
              - ``tuple_shape = size`` if `size` is a sequence,
              - ``d_shape = tuple()`` if `d` is ``None`` or `d` is 1, and
              - ``d_shape = (d,)`` if `d` is greater than 1.

        The elements of the returned array are part of a low-discrepancy
        sequence. If `d` is 1, this means that none of the samples are truly
        independent. If `d` > 1, each slice ``rvs[..., i]`` will be of a
        quasi-independent sequence; see `scipy.stats.qmc.QMCEngine` for
        details. Note that when `d` > 1, the samples returned are still those
        of the provided univariate distribution, not a multivariate
        generalization of that distribution.

        """
        # Input validation for `qmc_engine` and `d`
        # Error messages for invalid `d` are raised by QMCEngine
        # we could probably use a stats.qmc.check_qrandom_state
        if isinstance(qmc_engine, stats.qmc.QMCEngine):
            message = "`d` must be consistent with dimension of `qmc_engine`."
            if d is not None and qmc_engine.d != d:
                raise ValueError(message)
            d = qmc_engine.d if d is None else d
        elif qmc_engine is None:
            d = 1 if d is None else d
            qmc_engine = stats.qmc.Halton(d)
        else:
            message = ("`qmc_engine` must be an instance of "
                       "`scipy.stats.qmc.QMCEngine` or `None`.")
            raise ValueError(message)

        # `rvs` is flexible about whether `size` is an int or tuple, so this
        # should be, too.
        try:
            tuple_size = tuple(size)
        except TypeError:
            tuple_size = (size,)

        # Get uniform QRVS from qmc_random and transform it
        uniform = qmc_engine.random(np.prod(tuple_size) or 1)
        # scale to valid domain of interpolant
        uniform = self._a + uniform * (self._b - self._a)
        qrvs = self.ppf(uniform)

        # Output reshaping for user convenience
        if size is None:
            return qrvs.squeeze()[()]
        else:
            if d == 1:
                return qrvs.reshape(tuple_size)
            else:
                return qrvs.reshape(tuple_size + (d,))


def _fni_input_validation(dist, tol, max_intervals):
    """
    Input validation and standardization for _fast_numerical_inverse.

    """

    has_pdf = hasattr(dist, 'pdf') and callable(dist.pdf)
    has_cdf = hasattr(dist, 'cdf') and callable(dist.cdf)
    has_ppf = hasattr(dist, 'ppf') and callable(dist.ppf)
    has_isf = hasattr(dist, 'isf') and callable(dist.isf)

    if not (has_pdf and has_cdf and has_ppf):
        raise ValueError("`dist` must have methods `pdf`, `cdf`, and `ppf`.")

    if not has_isf:
        def isf(x):
            return 1 - dist.ppf(x)
        dist.isf = isf

    tol = float(tol)  # if there's an exception, raise it now

    if int(max_intervals) != max_intervals or max_intervals <= 1:
        raise ValueError("`max_intervals' must be an integer greater than 1.")

    return dist, tol, max_intervals


def _fast_numerical_inverse(dist, tol=1e-12, max_intervals=100000):
    """
    Generate fast, approximate PPF (inverse CDF) of probability distribution.

    `_fast_numerical_inverse` accepts `dist`, an object representing the
    distribution for which a fast approximate PPF is desired, and returns an
    object `fni` with methods that approximate `dist.ppf` and `dist.rvs`.
    For some distributions, these methods may be faster than those of `dist`
    itself.

    Parameters
    ----------
    dist : object
        Object representing distribution for which fast approximate PPF is
        desired; e.g., a frozen instance of `scipy.stats.rv_continuous`.
    tol : float, optional
        u-error tolerance. The default is 1e-12.
    max_intervals : int, optional
        Maximum number of intervals in the cubic Hermite Spline used to
        approximate the percent point function. The default is 100000.

    Returns
    -------
    H : scipy.interpolate.CubicHermiteSpline
        Interpolant of the distributions's PPF.
    intervals : int
        The number of intervals of the interpolant.
    midpoint_error : float
        The maximum u-error at an interpolant interval midpoint.
    a, b : float
        The left and right endpoints of the valid domain of the interpolant.

    """
    dist, tol, max_intervals = _fni_input_validation(dist, tol, max_intervals)

    # [1] Section 2.1: "For distributions with unbounded domain, we have to
    # chop off its tails at [a] and [b] such that F(a) and 1-F(b) are small
    # compared to the maximal tolerated approximation error."
    p = np.array([dist.ppf(tol/10), dist.isf(tol/10)])  # initial interval

    # [1] Section 2.3: "We then halve this interval recursively until
    # |u[i+1]-u[i]| is smaller than some threshold value, for example, 0.05."
    u = dist.cdf(p)
    while p.size-1 <= np.ceil(max_intervals/2):
        i = np.nonzero(np.diff(u) > 0.05)[0]
        if not i.size:
            break

        p_mid = (p[i] + p[i+1])/2
        # Compute only the new values and insert them in the right places
        # [1] uses a linked list; we can't do that efficiently
        u_mid = dist.cdf(p_mid)
        p = np.concatenate((p, p_mid))
        u = np.concatenate((u, u_mid))
        i_sort = np.argsort(p)
        p = p[i_sort]
        u = u[i_sort]

    # [1] Section 2.3: "Now we continue with checking the error estimate in
    # each of the intervals and continue with splitting them until [it] is
    # smaller than a given error bound."
    u = dist.cdf(p)
    f = dist.pdf(p)
    while p.size-1 <= max_intervals:
        # [1] Equation 4-8
        try:
            H = CubicHermiteSpline(u, p, 1/f)
        except ValueError:
            message = ("The interpolating spline could not be created. This "
                       "is often caused by inaccurate CDF evaluation in a "
                       "tail of the distribution. Increasing `tol` can "
                       "resolve this error at the expense of lower accuracy.")
            raise ValueError(message)
        # To improve performance, add update feature to CubicHermiteSpline

        # [1] Equation 12
        u_mid = (u[:-1] + u[1:])/2
        eu = np.abs(dist.cdf(H(u_mid)) - u_mid)

        i = np.nonzero(eu > tol)[0]
        if not i.size:
            break

        p_mid = (p[i] + p[i+1])/2
        u_mid = dist.cdf(p_mid)
        f_mid = dist.pdf(p_mid)
        p = np.concatenate((p, p_mid))
        u = np.concatenate((u, u_mid))
        f = np.concatenate((f, f_mid))
        i_sort = np.argsort(p)
        p = p[i_sort]
        u = u[i_sort]
        f = f[i_sort]

    # todo: add test for monotonicity [1] Section 2.4
    # todo: deal with vanishing density [1] Section 2.5
    return H, eu, p.size-1, u[0], u[-1]
