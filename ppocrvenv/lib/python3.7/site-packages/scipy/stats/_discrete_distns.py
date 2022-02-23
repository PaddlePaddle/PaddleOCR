#
# Author:  Travis Oliphant  2002-2011 with contributions from
#          SciPy Developers 2004-2011
#
from functools import partial
from scipy import special
from scipy.special import entr, logsumexp, betaln, gammaln as gamln, zeta
from scipy._lib._util import _lazywhere, rng_integers
from scipy.interpolate import interp1d

from numpy import floor, ceil, log, exp, sqrt, log1p, expm1, tanh, cosh, sinh

import numpy as np

from ._distn_infrastructure import (
    rv_discrete, _ncx2_pdf, _ncx2_cdf, get_distribution_names,
    _check_shape)
import scipy.stats._boost as _boost
from .biasedurn import (_PyFishersNCHypergeometric,
                        _PyWalleniusNCHypergeometric,
                        _PyStochasticLib3)

class binom_gen(rv_discrete):
    r"""A binomial discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `binom` is:

    .. math::

       f(k) = \binom{n}{k} p^k (1-p)^{n-k}

    for :math:`k \in \{0, 1, \dots, n\}`, :math:`0 \leq p \leq 1`

    `binom` takes :math:`n` and :math:`p` as shape parameters,
    where :math:`p` is the probability of a single success
    and :math:`1-p` is the probability of a single failure.

    %(after_notes)s

    %(example)s

    See Also
    --------
    hypergeom, nbinom, nhypergeom

    """
    def _rvs(self, n, p, size=None, random_state=None):
        return random_state.binomial(n, p, size)

    def _argcheck(self, n, p):
        return (n >= 0) & (p >= 0) & (p <= 1)

    def _get_support(self, n, p):
        return self.a, n

    def _logpmf(self, x, n, p):
        k = floor(x)
        combiln = (gamln(n+1) - (gamln(k+1) + gamln(n-k+1)))
        return combiln + special.xlogy(k, p) + special.xlog1py(n-k, -p)

    def _pmf(self, x, n, p):
        # binom.pmf(k) = choose(n, k) * p**k * (1-p)**(n-k)
        return _boost._binom_pdf(x, n, p)

    def _cdf(self, x, n, p):
        k = floor(x)
        return _boost._binom_cdf(k, n, p)

    def _sf(self, x, n, p):
        k = floor(x)
        return _boost._binom_sf(k, n, p)

    def _isf(self, x, n, p):
        return _boost._binom_isf(x, n, p)
    
    def _ppf(self, q, n, p):
        return _boost._binom_ppf(q, n, p)

    def _stats(self, n, p, moments='mv'):
        mu = _boost._binom_mean(n, p)
        var = _boost._binom_variance(n, p)
        g1, g2 = None, None
        if 's' in moments:
            g1 = _boost._binom_skewness(n, p)
        if 'k' in moments:
            g2 = _boost._binom_kurtosis_excess(n, p)
        return mu, var, g1, g2

    def _entropy(self, n, p):
        k = np.r_[0:n + 1]
        vals = self._pmf(k, n, p)
        return np.sum(entr(vals), axis=0)


binom = binom_gen(name='binom')


class bernoulli_gen(binom_gen):
    r"""A Bernoulli discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `bernoulli` is:

    .. math::

       f(k) = \begin{cases}1-p  &\text{if } k = 0\\
                           p    &\text{if } k = 1\end{cases}

    for :math:`k` in :math:`\{0, 1\}`, :math:`0 \leq p \leq 1`

    `bernoulli` takes :math:`p` as shape parameter,
    where :math:`p` is the probability of a single success
    and :math:`1-p` is the probability of a single failure.

    %(after_notes)s

    %(example)s

    """
    def _rvs(self, p, size=None, random_state=None):
        return binom_gen._rvs(self, 1, p, size=size, random_state=random_state)

    def _argcheck(self, p):
        return (p >= 0) & (p <= 1)

    def _get_support(self, p):
        # Overrides binom_gen._get_support!x
        return self.a, self.b

    def _logpmf(self, x, p):
        return binom._logpmf(x, 1, p)

    def _pmf(self, x, p):
        # bernoulli.pmf(k) = 1-p  if k = 0
        #                  = p    if k = 1
        return binom._pmf(x, 1, p)

    def _cdf(self, x, p):
        return binom._cdf(x, 1, p)

    def _sf(self, x, p):
        return binom._sf(x, 1, p)

    def _isf(self, x, p):
        return binom._isf(x, 1, p)        
    
    def _ppf(self, q, p):
        return binom._ppf(q, 1, p)

    def _stats(self, p):
        return binom._stats(1, p)

    def _entropy(self, p):
        return entr(p) + entr(1-p)


bernoulli = bernoulli_gen(b=1, name='bernoulli')


class betabinom_gen(rv_discrete):
    r"""A beta-binomial discrete random variable.

    %(before_notes)s

    Notes
    -----
    The beta-binomial distribution is a binomial distribution with a
    probability of success `p` that follows a beta distribution.

    The probability mass function for `betabinom` is:

    .. math::

       f(k) = \binom{n}{k} \frac{B(k + a, n - k + b)}{B(a, b)}

    for :math:`k \in \{0, 1, \dots, n\}`, :math:`n \geq 0`, :math:`a > 0`,
    :math:`b > 0`, where :math:`B(a, b)` is the beta function.

    `betabinom` takes :math:`n`, :math:`a`, and :math:`b` as shape parameters.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Beta-binomial_distribution

    %(after_notes)s

    .. versionadded:: 1.4.0

    See Also
    --------
    beta, binom

    %(example)s

    """

    def _rvs(self, n, a, b, size=None, random_state=None):
        p = random_state.beta(a, b, size)
        return random_state.binomial(n, p, size)

    def _get_support(self, n, a, b):
        return 0, n

    def _argcheck(self, n, a, b):
        return (n >= 0) & (a > 0) & (b > 0)

    def _logpmf(self, x, n, a, b):
        k = floor(x)
        combiln = -log(n + 1) - betaln(n - k + 1, k + 1)
        return combiln + betaln(k + a, n - k + b) - betaln(a, b)

    def _pmf(self, x, n, a, b):
        return exp(self._logpmf(x, n, a, b))

    def _stats(self, n, a, b, moments='mv'):
        e_p = a / (a + b)
        e_q = 1 - e_p
        mu = n * e_p
        var = n * (a + b + n) * e_p * e_q / (a + b + 1)
        g1, g2 = None, None
        if 's' in moments:
            g1 = 1.0 / sqrt(var)
            g1 *= (a + b + 2 * n) * (b - a)
            g1 /= (a + b + 2) * (a + b)
        if 'k' in moments:
            g2 = a + b
            g2 *= (a + b - 1 + 6 * n)
            g2 += 3 * a * b * (n - 2)
            g2 += 6 * n ** 2
            g2 -= 3 * e_p * b * n * (6 - n)
            g2 -= 18 * e_p * e_q * n ** 2
            g2 *= (a + b) ** 2 * (1 + a + b)
            g2 /= (n * a * b * (a + b + 2) * (a + b + 3) * (a + b + n))
            g2 -= 3
        return mu, var, g1, g2


betabinom = betabinom_gen(name='betabinom')


class nbinom_gen(rv_discrete):
    r"""A negative binomial discrete random variable.

    %(before_notes)s

    Notes
    -----
    Negative binomial distribution describes a sequence of i.i.d. Bernoulli
    trials, repeated until a predefined, non-random number of successes occurs.

    The probability mass function of the number of failures for `nbinom` is:

    .. math::

       f(k) = \binom{k+n-1}{n-1} p^n (1-p)^k

    for :math:`k \ge 0`, :math:`0 < p \leq 1`

    `nbinom` takes :math:`n` and :math:`p` as shape parameters where n is the
    number of successes, :math:`p` is the probability of a single success,
    and :math:`1-p` is the probability of a single failure.

    Another common parameterization of the negative binomial distribution is
    in terms of the mean number of failures :math:`\mu` to achieve :math:`n`
    successes. The mean :math:`\mu` is related to the probability of success
    as

    .. math::

       p = \frac{n}{n + \mu}

    The number of successes :math:`n` may also be specified in terms of a
    "dispersion", "heterogeneity", or "aggregation" parameter :math:`\alpha`,
    which relates the mean :math:`\mu` to the variance :math:`\sigma^2`,
    e.g. :math:`\sigma^2 = \mu + \alpha \mu^2`. Regardless of the convention
    used for :math:`\alpha`,

    .. math::

       p &= \frac{\mu}{\sigma^2} \\
       n &= \frac{\mu^2}{\sigma^2 - \mu}

    %(after_notes)s

    %(example)s

    See Also
    --------
    hypergeom, binom, nhypergeom

    """
    def _rvs(self, n, p, size=None, random_state=None):
        return random_state.negative_binomial(n, p, size)

    def _argcheck(self, n, p):
        return (n > 0) & (p > 0) & (p <= 1)

    def _pmf(self, x, n, p):
        # nbinom.pmf(k) = choose(k+n-1, n-1) * p**n * (1-p)**k
        return _boost._nbinom_pdf(x, n, p)

    def _logpmf(self, x, n, p):
        coeff = gamln(n+x) - gamln(x+1) - gamln(n)
        return coeff + n*log(p) + special.xlog1py(x, -p)

    def _cdf(self, x, n, p):
        k = floor(x)
        return _boost._nbinom_cdf(k, n, p)

    def _logcdf(self, x, n, p):
        k = floor(x)
        cdf = self._cdf(k, n, p)
        cond = cdf > 0.5

        def f1(k, n, p):
            return np.log1p(-special.betainc(k + 1, n, 1 - p))

        def f2(k, n, p):
            return np.log(cdf)

        with np.errstate(divide='ignore'):
            return _lazywhere(cond, (x, n, p), f=f1, f2=f2)

    def _sf(self, x, n, p):
        k = floor(x)
        return _boost._nbinom_sf(k, n, p)

    def _isf(self, x, n, p):
        return _boost._nbinom_isf(x, n, p)
    
    def _ppf(self, q, n, p):
        return _boost._nbinom_ppf(q, n, p)

    def _stats(self, n, p):
        return(
            _boost._nbinom_mean(n, p),
            _boost._nbinom_variance(n, p),
            _boost._nbinom_skewness(n, p),
            _boost._nbinom_kurtosis_excess(n, p),
        )


nbinom = nbinom_gen(name='nbinom')


class geom_gen(rv_discrete):
    r"""A geometric discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `geom` is:

    .. math::

        f(k) = (1-p)^{k-1} p

    for :math:`k \ge 1`, :math:`0 < p \leq 1`

    `geom` takes :math:`p` as shape parameter,
    where :math:`p` is the probability of a single success
    and :math:`1-p` is the probability of a single failure.

    %(after_notes)s

    See Also
    --------
    planck

    %(example)s

    """
    def _rvs(self, p, size=None, random_state=None):
        return random_state.geometric(p, size=size)

    def _argcheck(self, p):
        return (p <= 1) & (p > 0)

    def _pmf(self, k, p):
        return np.power(1-p, k-1) * p

    def _logpmf(self, k, p):
        return special.xlog1py(k - 1, -p) + log(p)

    def _cdf(self, x, p):
        k = floor(x)
        return -expm1(log1p(-p)*k)

    def _sf(self, x, p):
        return np.exp(self._logsf(x, p))

    def _logsf(self, x, p):
        k = floor(x)
        return k*log1p(-p)

    def _ppf(self, q, p):
        vals = ceil(log1p(-q) / log1p(-p))
        temp = self._cdf(vals-1, p)
        return np.where((temp >= q) & (vals > 0), vals-1, vals)

    def _stats(self, p):
        mu = 1.0/p
        qr = 1.0-p
        var = qr / p / p
        g1 = (2.0-p) / sqrt(qr)
        g2 = np.polyval([1, -6, 6], p)/(1.0-p)
        return mu, var, g1, g2


geom = geom_gen(a=1, name='geom', longname="A geometric")


class hypergeom_gen(rv_discrete):
    r"""A hypergeometric discrete random variable.

    The hypergeometric distribution models drawing objects from a bin.
    `M` is the total number of objects, `n` is total number of Type I objects.
    The random variate represents the number of Type I objects in `N` drawn
    without replacement from the total population.

    %(before_notes)s

    Notes
    -----
    The symbols used to denote the shape parameters (`M`, `n`, and `N`) are not
    universally accepted.  See the Examples for a clarification of the
    definitions used here.

    The probability mass function is defined as,

    .. math:: p(k, M, n, N) = \frac{\binom{n}{k} \binom{M - n}{N - k}}
                                   {\binom{M}{N}}

    for :math:`k \in [\max(0, N - M + n), \min(n, N)]`, where the binomial
    coefficients are defined as,

    .. math:: \binom{n}{k} \equiv \frac{n!}{k! (n - k)!}.

    %(after_notes)s

    Examples
    --------
    >>> from scipy.stats import hypergeom
    >>> import matplotlib.pyplot as plt

    Suppose we have a collection of 20 animals, of which 7 are dogs.  Then if
    we want to know the probability of finding a given number of dogs if we
    choose at random 12 of the 20 animals, we can initialize a frozen
    distribution and plot the probability mass function:

    >>> [M, n, N] = [20, 7, 12]
    >>> rv = hypergeom(M, n, N)
    >>> x = np.arange(0, n+1)
    >>> pmf_dogs = rv.pmf(x)

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(x, pmf_dogs, 'bo')
    >>> ax.vlines(x, 0, pmf_dogs, lw=2)
    >>> ax.set_xlabel('# of dogs in our group of chosen animals')
    >>> ax.set_ylabel('hypergeom PMF')
    >>> plt.show()

    Instead of using a frozen distribution we can also use `hypergeom`
    methods directly.  To for example obtain the cumulative distribution
    function, use:

    >>> prb = hypergeom.cdf(x, M, n, N)

    And to generate random numbers:

    >>> R = hypergeom.rvs(M, n, N, size=10)

    See Also
    --------
    nhypergeom, binom, nbinom

    """
    def _rvs(self, M, n, N, size=None, random_state=None):
        return random_state.hypergeometric(n, M-n, N, size=size)

    def _get_support(self, M, n, N):
        return np.maximum(N-(M-n), 0), np.minimum(n, N)

    def _argcheck(self, M, n, N):
        cond = (M > 0) & (n >= 0) & (N >= 0)
        cond &= (n <= M) & (N <= M)
        return cond

    def _logpmf(self, k, M, n, N):
        tot, good = M, n
        bad = tot - good
        result = (betaln(good+1, 1) + betaln(bad+1, 1) + betaln(tot-N+1, N+1) -
                  betaln(k+1, good-k+1) - betaln(N-k+1, bad-N+k+1) -
                  betaln(tot+1, 1))
        return result

    def _pmf(self, k, M, n, N):
        # same as the following but numerically more precise
        # return comb(good, k) * comb(bad, N-k) / comb(tot, N)
        return exp(self._logpmf(k, M, n, N))

    def _stats(self, M, n, N):
        # tot, good, sample_size = M, n, N
        # "wikipedia".replace('N', 'M').replace('n', 'N').replace('K', 'n')
        M, n, N = 1.*M, 1.*n, 1.*N
        m = M - n
        p = n/M
        mu = N*p

        var = m*n*N*(M - N)*1.0/(M*M*(M-1))
        g1 = (m - n)*(M-2*N) / (M-2.0) * sqrt((M-1.0) / (m*n*N*(M-N)))

        g2 = M*(M+1) - 6.*N*(M-N) - 6.*n*m
        g2 *= (M-1)*M*M
        g2 += 6.*n*N*(M-N)*m*(5.*M-6)
        g2 /= n * N * (M-N) * m * (M-2.) * (M-3.)
        return mu, var, g1, g2

    def _entropy(self, M, n, N):
        k = np.r_[N - (M - n):min(n, N) + 1]
        vals = self.pmf(k, M, n, N)
        return np.sum(entr(vals), axis=0)

    def _sf(self, k, M, n, N):
        # This for loop is needed because `k` can be an array. If that's the
        # case, the sf() method makes M, n and N arrays of the same shape. We
        # therefore unpack all inputs args, so we can do the manual
        # integration.
        res = []
        for quant, tot, good, draw in zip(*np.broadcast_arrays(k, M, n, N)):
            # Manual integration over probability mass function. More accurate
            # than integrate.quad.
            k2 = np.arange(quant + 1, draw + 1)
            res.append(np.sum(self._pmf(k2, tot, good, draw)))
        return np.asarray(res)

    def _logsf(self, k, M, n, N):
        res = []
        for quant, tot, good, draw in zip(*np.broadcast_arrays(k, M, n, N)):
            if (quant + 0.5) * (tot + 0.5) < (good - 0.5) * (draw - 0.5):
                # Less terms to sum if we calculate log(1-cdf)
                res.append(log1p(-exp(self.logcdf(quant, tot, good, draw))))
            else:
                # Integration over probability mass function using logsumexp
                k2 = np.arange(quant + 1, draw + 1)
                res.append(logsumexp(self._logpmf(k2, tot, good, draw)))
        return np.asarray(res)

    def _logcdf(self, k, M, n, N):
        res = []
        for quant, tot, good, draw in zip(*np.broadcast_arrays(k, M, n, N)):
            if (quant + 0.5) * (tot + 0.5) > (good - 0.5) * (draw - 0.5):
                # Less terms to sum if we calculate log(1-sf)
                res.append(log1p(-exp(self.logsf(quant, tot, good, draw))))
            else:
                # Integration over probability mass function using logsumexp
                k2 = np.arange(0, quant + 1)
                res.append(logsumexp(self._logpmf(k2, tot, good, draw)))
        return np.asarray(res)


hypergeom = hypergeom_gen(name='hypergeom')


class nhypergeom_gen(rv_discrete):
    r"""A negative hypergeometric discrete random variable.

    Consider a box containing :math:`M` balls:, :math:`n` red and
    :math:`M-n` blue. We randomly sample balls from the box, one
    at a time and *without* replacement, until we have picked :math:`r`
    blue balls. `nhypergeom` is the distribution of the number of
    red balls :math:`k` we have picked.

    %(before_notes)s

    Notes
    -----
    The symbols used to denote the shape parameters (`M`, `n`, and `r`) are not
    universally accepted. See the Examples for a clarification of the
    definitions used here.

    The probability mass function is defined as,

    .. math:: f(k; M, n, r) = \frac{{{k+r-1}\choose{k}}{{M-r-k}\choose{n-k}}}
                                   {{M \choose n}}

    for :math:`k \in [0, n]`, :math:`n \in [0, M]`, :math:`r \in [0, M-n]`,
    and the binomial coefficient is:

    .. math:: \binom{n}{k} \equiv \frac{n!}{k! (n - k)!}.

    It is equivalent to observing :math:`k` successes in :math:`k+r-1`
    samples with :math:`k+r`'th sample being a failure. The former
    can be modelled as a hypergeometric distribution. The probability
    of the latter is simply the number of failures remaining
    :math:`M-n-(r-1)` divided by the size of the remaining population
    :math:`M-(k+r-1)`. This relationship can be shown as:

    .. math:: NHG(k;M,n,r) = HG(k;M,n,k+r-1)\frac{(M-n-(r-1))}{(M-(k+r-1))}

    where :math:`NHG` is probability mass function (PMF) of the
    negative hypergeometric distribution and :math:`HG` is the
    PMF of the hypergeometric distribution.

    %(after_notes)s

    Examples
    --------
    >>> from scipy.stats import nhypergeom
    >>> import matplotlib.pyplot as plt

    Suppose we have a collection of 20 animals, of which 7 are dogs.
    Then if we want to know the probability of finding a given number
    of dogs (successes) in a sample with exactly 12 animals that
    aren't dogs (failures), we can initialize a frozen distribution
    and plot the probability mass function:

    >>> M, n, r = [20, 7, 12]
    >>> rv = nhypergeom(M, n, r)
    >>> x = np.arange(0, n+2)
    >>> pmf_dogs = rv.pmf(x)

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(x, pmf_dogs, 'bo')
    >>> ax.vlines(x, 0, pmf_dogs, lw=2)
    >>> ax.set_xlabel('# of dogs in our group with given 12 failures')
    >>> ax.set_ylabel('nhypergeom PMF')
    >>> plt.show()

    Instead of using a frozen distribution we can also use `nhypergeom`
    methods directly.  To for example obtain the probability mass
    function, use:

    >>> prb = nhypergeom.pmf(x, M, n, r)

    And to generate random numbers:

    >>> R = nhypergeom.rvs(M, n, r, size=10)

    To verify the relationship between `hypergeom` and `nhypergeom`, use:

    >>> from scipy.stats import hypergeom, nhypergeom
    >>> M, n, r = 45, 13, 8
    >>> k = 6
    >>> nhypergeom.pmf(k, M, n, r)
    0.06180776620271643
    >>> hypergeom.pmf(k, M, n, k+r-1) * (M - n - (r-1)) / (M - (k+r-1))
    0.06180776620271644

    See Also
    --------
    hypergeom, binom, nbinom

    References
    ----------
    .. [1] Negative Hypergeometric Distribution on Wikipedia
           https://en.wikipedia.org/wiki/Negative_hypergeometric_distribution

    .. [2] Negative Hypergeometric Distribution from
           http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Negativehypergeometric.pdf

    """

    def _get_support(self, M, n, r):
        return 0, n

    def _argcheck(self, M, n, r):
        cond = (n >= 0) & (n <= M) & (r >= 0) & (r <= M-n)
        return cond

    def _rvs(self, M, n, r, size=None, random_state=None):

        @_vectorize_rvs_over_shapes
        def _rvs1(M, n, r, size, random_state):
            # invert cdf by calculating all values in support, scalar M, n, r
            a, b = self.support(M, n, r)
            ks = np.arange(a, b+1)
            cdf = self.cdf(ks, M, n, r)
            ppf = interp1d(cdf, ks, kind='next', fill_value='extrapolate')
            rvs = ppf(random_state.uniform(size=size)).astype(int)
            if size is None:
                return rvs.item()
            return rvs

        return _rvs1(M, n, r, size=size, random_state=random_state)

    def _logpmf(self, k, M, n, r):
        cond = ((r == 0) & (k == 0))
        result = _lazywhere(~cond, (k, M, n, r),
                            lambda k, M, n, r:
                                (-betaln(k+1, r) + betaln(k+r, 1) -
                                 betaln(n-k+1, M-r-n+1) + betaln(M-r-k+1, 1) +
                                 betaln(n+1, M-n+1) - betaln(M+1, 1)),
                            fillvalue=0.0)
        return result

    def _pmf(self, k, M, n, r):
        # same as the following but numerically more precise
        # return comb(k+r-1, k) * comb(M-r-k, n-k) / comb(M, n)
        return exp(self._logpmf(k, M, n, r))

    def _stats(self, M, n, r):
        # Promote the datatype to at least float
        # mu = rn / (M-n+1)
        M, n, r = 1.*M, 1.*n, 1.*r
        mu = r*n / (M-n+1)

        var = r*(M+1)*n / ((M-n+1)*(M-n+2)) * (1 - r / (M-n+1))

        # The skew and kurtosis are mathematically
        # intractable so return `None`. See [2]_.
        g1, g2 = None, None
        return mu, var, g1, g2


nhypergeom = nhypergeom_gen(name='nhypergeom')


# FIXME: Fails _cdfvec
class logser_gen(rv_discrete):
    r"""A Logarithmic (Log-Series, Series) discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `logser` is:

    .. math::

        f(k) = - \frac{p^k}{k \log(1-p)}

    for :math:`k \ge 1`, :math:`0 < p < 1`

    `logser` takes :math:`p` as shape parameter,
    where :math:`p` is the probability of a single success
    and :math:`1-p` is the probability of a single failure.

    %(after_notes)s

    %(example)s

    """
    def _rvs(self, p, size=None, random_state=None):
        # looks wrong for p>0.5, too few k=1
        # trying to use generic is worse, no k=1 at all
        return random_state.logseries(p, size=size)

    def _argcheck(self, p):
        return (p > 0) & (p < 1)

    def _pmf(self, k, p):
        # logser.pmf(k) = - p**k / (k*log(1-p))
        return -np.power(p, k) * 1.0 / k / special.log1p(-p)

    def _stats(self, p):
        r = special.log1p(-p)
        mu = p / (p - 1.0) / r
        mu2p = -p / r / (p - 1.0)**2
        var = mu2p - mu*mu
        mu3p = -p / r * (1.0+p) / (1.0 - p)**3
        mu3 = mu3p - 3*mu*mu2p + 2*mu**3
        g1 = mu3 / np.power(var, 1.5)

        mu4p = -p / r * (
            1.0 / (p-1)**2 - 6*p / (p - 1)**3 + 6*p*p / (p-1)**4)
        mu4 = mu4p - 4*mu3p*mu + 6*mu2p*mu*mu - 3*mu**4
        g2 = mu4 / var**2 - 3.0
        return mu, var, g1, g2


logser = logser_gen(a=1, name='logser', longname='A logarithmic')


class poisson_gen(rv_discrete):
    r"""A Poisson discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `poisson` is:

    .. math::

        f(k) = \exp(-\mu) \frac{\mu^k}{k!}

    for :math:`k \ge 0`.

    `poisson` takes :math:`\mu \geq 0` as shape parameter.
    When :math:`\mu = 0`, the ``pmf`` method
    returns ``1.0`` at quantile :math:`k = 0`.

    %(after_notes)s

    %(example)s

    """

    # Override rv_discrete._argcheck to allow mu=0.
    def _argcheck(self, mu):
        return mu >= 0

    def _rvs(self, mu, size=None, random_state=None):
        return random_state.poisson(mu, size)

    def _logpmf(self, k, mu):
        Pk = special.xlogy(k, mu) - gamln(k + 1) - mu
        return Pk

    def _pmf(self, k, mu):
        # poisson.pmf(k) = exp(-mu) * mu**k / k!
        return exp(self._logpmf(k, mu))

    def _cdf(self, x, mu):
        k = floor(x)
        return special.pdtr(k, mu)

    def _sf(self, x, mu):
        k = floor(x)
        return special.pdtrc(k, mu)

    def _ppf(self, q, mu):
        vals = ceil(special.pdtrik(q, mu))
        vals1 = np.maximum(vals - 1, 0)
        temp = special.pdtr(vals1, mu)
        return np.where(temp >= q, vals1, vals)

    def _stats(self, mu):
        var = mu
        tmp = np.asarray(mu)
        mu_nonzero = tmp > 0
        g1 = _lazywhere(mu_nonzero, (tmp,), lambda x: sqrt(1.0/x), np.inf)
        g2 = _lazywhere(mu_nonzero, (tmp,), lambda x: 1.0/x, np.inf)
        return mu, var, g1, g2


poisson = poisson_gen(name="poisson", longname='A Poisson')


class planck_gen(rv_discrete):
    r"""A Planck discrete exponential random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `planck` is:

    .. math::

        f(k) = (1-\exp(-\lambda)) \exp(-\lambda k)

    for :math:`k \ge 0` and :math:`\lambda > 0`.

    `planck` takes :math:`\lambda` as shape parameter. The Planck distribution
    can be written as a geometric distribution (`geom`) with
    :math:`p = 1 - \exp(-\lambda)` shifted by ``loc = -1``.

    %(after_notes)s

    See Also
    --------
    geom

    %(example)s

    """
    def _argcheck(self, lambda_):
        return lambda_ > 0

    def _pmf(self, k, lambda_):
        return -expm1(-lambda_)*exp(-lambda_*k)

    def _cdf(self, x, lambda_):
        k = floor(x)
        return -expm1(-lambda_*(k+1))

    def _sf(self, x, lambda_):
        return exp(self._logsf(x, lambda_))

    def _logsf(self, x, lambda_):
        k = floor(x)
        return -lambda_*(k+1)

    def _ppf(self, q, lambda_):
        vals = ceil(-1.0/lambda_ * log1p(-q)-1)
        vals1 = (vals-1).clip(*(self._get_support(lambda_)))
        temp = self._cdf(vals1, lambda_)
        return np.where(temp >= q, vals1, vals)

    def _rvs(self, lambda_, size=None, random_state=None):
        # use relation to geometric distribution for sampling
        p = -expm1(-lambda_)
        return random_state.geometric(p, size=size) - 1.0

    def _stats(self, lambda_):
        mu = 1/expm1(lambda_)
        var = exp(-lambda_)/(expm1(-lambda_))**2
        g1 = 2*cosh(lambda_/2.0)
        g2 = 4+2*cosh(lambda_)
        return mu, var, g1, g2

    def _entropy(self, lambda_):
        C = -expm1(-lambda_)
        return lambda_*exp(-lambda_)/C - log(C)


planck = planck_gen(a=0, name='planck', longname='A discrete exponential ')


class boltzmann_gen(rv_discrete):
    r"""A Boltzmann (Truncated Discrete Exponential) random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `boltzmann` is:

    .. math::

        f(k) = (1-\exp(-\lambda)) \exp(-\lambda k) / (1-\exp(-\lambda N))

    for :math:`k = 0,..., N-1`.

    `boltzmann` takes :math:`\lambda > 0` and :math:`N > 0` as shape parameters.

    %(after_notes)s

    %(example)s

    """
    def _argcheck(self, lambda_, N):
        return (lambda_ > 0) & (N > 0)

    def _get_support(self, lambda_, N):
        return self.a, N - 1

    def _pmf(self, k, lambda_, N):
        # boltzmann.pmf(k) =
        #               (1-exp(-lambda_)*exp(-lambda_*k)/(1-exp(-lambda_*N))
        fact = (1-exp(-lambda_))/(1-exp(-lambda_*N))
        return fact*exp(-lambda_*k)

    def _cdf(self, x, lambda_, N):
        k = floor(x)
        return (1-exp(-lambda_*(k+1)))/(1-exp(-lambda_*N))

    def _ppf(self, q, lambda_, N):
        qnew = q*(1-exp(-lambda_*N))
        vals = ceil(-1.0/lambda_ * log(1-qnew)-1)
        vals1 = (vals-1).clip(0.0, np.inf)
        temp = self._cdf(vals1, lambda_, N)
        return np.where(temp >= q, vals1, vals)

    def _stats(self, lambda_, N):
        z = exp(-lambda_)
        zN = exp(-lambda_*N)
        mu = z/(1.0-z)-N*zN/(1-zN)
        var = z/(1.0-z)**2 - N*N*zN/(1-zN)**2
        trm = (1-zN)/(1-z)
        trm2 = (z*trm**2 - N*N*zN)
        g1 = z*(1+z)*trm**3 - N**3*zN*(1+zN)
        g1 = g1 / trm2**(1.5)
        g2 = z*(1+4*z+z*z)*trm**4 - N**4 * zN*(1+4*zN+zN*zN)
        g2 = g2 / trm2 / trm2
        return mu, var, g1, g2


boltzmann = boltzmann_gen(name='boltzmann', a=0,
                          longname='A truncated discrete exponential ')


class randint_gen(rv_discrete):
    r"""A uniform discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `randint` is:

    .. math::

        f(k) = \frac{1}{\texttt{high} - \texttt{low}}

    for :math:`k \in \{\texttt{low}, \dots, \texttt{high} - 1\}`.

    `randint` takes :math:`\texttt{low}` and :math:`\texttt{high}` as shape
    parameters.

    %(after_notes)s

    %(example)s

    """
    def _argcheck(self, low, high):
        return (high > low)

    def _get_support(self, low, high):
        return low, high-1

    def _pmf(self, k, low, high):
        # randint.pmf(k) = 1./(high - low)
        p = np.ones_like(k) / (high - low)
        return np.where((k >= low) & (k < high), p, 0.)

    def _cdf(self, x, low, high):
        k = floor(x)
        return (k - low + 1.) / (high - low)

    def _ppf(self, q, low, high):
        vals = ceil(q * (high - low) + low) - 1
        vals1 = (vals - 1).clip(low, high)
        temp = self._cdf(vals1, low, high)
        return np.where(temp >= q, vals1, vals)

    def _stats(self, low, high):
        m2, m1 = np.asarray(high), np.asarray(low)
        mu = (m2 + m1 - 1.0) / 2
        d = m2 - m1
        var = (d*d - 1) / 12.0
        g1 = 0.0
        g2 = -6.0/5.0 * (d*d + 1.0) / (d*d - 1.0)
        return mu, var, g1, g2

    def _rvs(self, low, high, size=None, random_state=None):
        """An array of *size* random integers >= ``low`` and < ``high``."""
        if np.asarray(low).size == 1 and np.asarray(high).size == 1:
            # no need to vectorize in that case
            return rng_integers(random_state, low, high, size=size)

        if size is not None:
            # NumPy's RandomState.randint() doesn't broadcast its arguments.
            # Use `broadcast_to()` to extend the shapes of low and high
            # up to size.  Then we can use the numpy.vectorize'd
            # randint without needing to pass it a `size` argument.
            low = np.broadcast_to(low, size)
            high = np.broadcast_to(high, size)
        randint = np.vectorize(partial(rng_integers, random_state),
                               otypes=[np.int_])
        return randint(low, high)

    def _entropy(self, low, high):
        return log(high - low)


randint = randint_gen(name='randint', longname='A discrete uniform '
                      '(random integer)')


# FIXME: problems sampling.
class zipf_gen(rv_discrete):
    r"""A Zipf (Zeta) discrete random variable.

    %(before_notes)s

    See Also
    --------
    zipfian

    Notes
    -----
    The probability mass function for `zipf` is:

    .. math::

        f(k, a) = \frac{1}{\zeta(a) k^a}

    for :math:`k \ge 1`, :math:`a > 1`.

    `zipf` takes :math:`a > 1` as shape parameter. :math:`\zeta` is the
    Riemann zeta function (`scipy.special.zeta`)

    The Zipf distribution is also known as the zeta distribution, which is
    a special case of the Zipfian distribution (`zipfian`).

    %(after_notes)s

    References
    ----------
    .. [1] "Zeta Distribution", Wikipedia,
           https://en.wikipedia.org/wiki/Zeta_distribution

    %(example)s

    Confirm that `zipf` is the large `n` limit of `zipfian`.

    >>> from scipy.stats import zipfian
    >>> k = np.arange(11)
    >>> np.allclose(zipf.pmf(k, a), zipfian.pmf(k, a, n=10000000))
    True

    """
    def _rvs(self, a, size=None, random_state=None):
        return random_state.zipf(a, size=size)

    def _argcheck(self, a):
        return a > 1

    def _pmf(self, k, a):
        # zipf.pmf(k, a) = 1/(zeta(a) * k**a)
        Pk = 1.0 / special.zeta(a, 1) / k**a
        return Pk

    def _munp(self, n, a):
        return _lazywhere(
            a > n + 1, (a, n),
            lambda a, n: special.zeta(a - n, 1) / special.zeta(a, 1),
            np.inf)


zipf = zipf_gen(a=1, name='zipf', longname='A Zipf')


def _gen_harmonic_gt1(n, a):
    """Generalized harmonic number, a > 1"""
    # See https://en.wikipedia.org/wiki/Harmonic_number; search for "hurwitz"
    return zeta(a, 1) - zeta(a, n+1)


def _gen_harmonic_leq1(n, a):
    """Generalized harmonic number, a <= 1"""
    if not np.size(n):
        return n
    n_max = np.max(n)  # loop starts at maximum of all n
    out = np.zeros_like(a, dtype=float)
    # add terms of harmonic series; starting from smallest to avoid roundoff
    for i in np.arange(n_max, 0, -1, dtype=float):
        mask = i <= n  # don't add terms after nth
        out[mask] += 1/i**a[mask]
    return out


def _gen_harmonic(n, a):
    """Generalized harmonic number"""
    n, a = np.broadcast_arrays(n, a)
    return _lazywhere(a > 1, (n, a),
                      f=_gen_harmonic_gt1, f2=_gen_harmonic_leq1)


class zipfian_gen(rv_discrete):
    r"""A Zipfian discrete random variable.

    %(before_notes)s

    See Also
    --------
    zipf

    Notes
    -----
    The probability mass function for `zipfian` is:

    .. math::

        f(k, a, n) = \frac{1}{H_{n,a} k^a}

    for :math:`k \in \{1, 2, \dots, n-1, n\}`, :math:`a \ge 0`,
    :math:`n \in \{1, 2, 3, \dots\}`.

    `zipfian` takes :math:`a` and :math:`n` as shape parameters.
    :math:`H_{n,a}` is the :math:`n`:sup:`th` generalized harmonic
    number of order :math:`a`.

    The Zipfian distribution reduces to the Zipf (zeta) distribution as
    :math:`n \rightarrow \infty`.

    %(after_notes)s

    References
    ----------
    .. [1] "Zipf's Law", Wikipedia, https://en.wikipedia.org/wiki/Zipf's_law
    .. [2] Larry Leemis, "Zipf Distribution", Univariate Distribution
           Relationships. http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Zipf.pdf

    %(example)s

    Confirm that `zipfian` reduces to `zipf` for large `n`, `a > 1`.

    >>> from scipy.stats import zipf
    >>> k = np.arange(11)
    >>> np.allclose(zipfian.pmf(k, a=3.5, n=10000000), zipf.pmf(k, a=3.5))
    True

    """
    def _argcheck(self, a, n):
        # we need np.asarray here because moment (maybe others) don't convert
        return (a >= 0) & (n > 0) & (n == np.asarray(n, dtype=int))

    def _get_support(self, a, n):
        return 1, n

    def _pmf(self, k, a, n):
        return 1.0 / _gen_harmonic(n, a) / k**a

    def _cdf(self, k, a, n):
        return  _gen_harmonic(k, a) / _gen_harmonic(n, a)

    def _sf(self, k, a, n):
        k = k + 1  # # to match SciPy convention
        # see http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Zipf.pdf
        return ((k**a*(_gen_harmonic(n, a) - _gen_harmonic(k, a)) + 1)
                / (k**a*_gen_harmonic(n, a)))

    def _stats(self, a, n):
        # see # see http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Zipf.pdf
        Hna = _gen_harmonic(n, a)
        Hna1 = _gen_harmonic(n, a-1)
        Hna2 = _gen_harmonic(n, a-2)
        Hna3 = _gen_harmonic(n, a-3)
        Hna4 = _gen_harmonic(n, a-4)
        mu1 = Hna1/Hna
        mu2n = (Hna2*Hna - Hna1**2)
        mu2d = Hna**2
        mu2 = mu2n / mu2d
        g1 = (Hna3/Hna - 3*Hna1*Hna2/Hna**2 + 2*Hna1**3/Hna**3)/mu2**(3/2)
        g2 = (Hna**3*Hna4 - 4*Hna**2*Hna1*Hna3 + 6*Hna*Hna1**2*Hna2
              - 3*Hna1**4) / mu2n**2
        g2 -= 3
        return mu1, mu2, g1, g2

zipfian = zipfian_gen(a=1, name='zipfian', longname='A Zipfian')


class dlaplace_gen(rv_discrete):
    r"""A  Laplacian discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `dlaplace` is:

    .. math::

        f(k) = \tanh(a/2) \exp(-a |k|)

    for integers :math:`k` and :math:`a > 0`.

    `dlaplace` takes :math:`a` as shape parameter.

    %(after_notes)s

    %(example)s

    """
    def _pmf(self, k, a):
        # dlaplace.pmf(k) = tanh(a/2) * exp(-a*abs(k))
        return tanh(a/2.0) * exp(-a * abs(k))

    def _cdf(self, x, a):
        k = floor(x)
        f = lambda k, a: 1.0 - exp(-a * k) / (exp(a) + 1)
        f2 = lambda k, a: exp(a * (k+1)) / (exp(a) + 1)
        return _lazywhere(k >= 0, (k, a), f=f, f2=f2)

    def _ppf(self, q, a):
        const = 1 + exp(a)
        vals = ceil(np.where(q < 1.0 / (1 + exp(-a)),
                             log(q*const) / a - 1,
                             -log((1-q) * const) / a))
        vals1 = vals - 1
        return np.where(self._cdf(vals1, a) >= q, vals1, vals)

    def _stats(self, a):
        ea = exp(a)
        mu2 = 2.*ea/(ea-1.)**2
        mu4 = 2.*ea*(ea**2+10.*ea+1.) / (ea-1.)**4
        return 0., mu2, 0., mu4/mu2**2 - 3.

    def _entropy(self, a):
        return a / sinh(a) - log(tanh(a/2.0))

    def _rvs(self, a, size=None, random_state=None):
        # The discrete Laplace is equivalent to the two-sided geometric
        # distribution with PMF:
        #   f(k) = (1 - alpha)/(1 + alpha) * alpha^abs(k)
        #   Reference:
        #     https://www.sciencedirect.com/science/
        #     article/abs/pii/S0378375804003519
        # Furthermore, the two-sided geometric distribution is
        # equivalent to the difference between two iid geometric
        # distributions.
        #   Reference (page 179):
        #     https://pdfs.semanticscholar.org/61b3/
        #     b99f466815808fd0d03f5d2791eea8b541a1.pdf
        # Thus, we can leverage the following:
        #   1) alpha = e^-a
        #   2) probability_of_success = 1 - alpha (Bernoulli trial)
        probOfSuccess = -np.expm1(-np.asarray(a))
        x = random_state.geometric(probOfSuccess, size=size)
        y = random_state.geometric(probOfSuccess, size=size)
        return x - y


dlaplace = dlaplace_gen(a=-np.inf,
                        name='dlaplace', longname='A discrete Laplacian')


class skellam_gen(rv_discrete):
    r"""A  Skellam discrete random variable.

    %(before_notes)s

    Notes
    -----
    Probability distribution of the difference of two correlated or
    uncorrelated Poisson random variables.

    Let :math:`k_1` and :math:`k_2` be two Poisson-distributed r.v. with
    expected values :math:`\lambda_1` and :math:`\lambda_2`. Then,
    :math:`k_1 - k_2` follows a Skellam distribution with parameters
    :math:`\mu_1 = \lambda_1 - \rho \sqrt{\lambda_1 \lambda_2}` and
    :math:`\mu_2 = \lambda_2 - \rho \sqrt{\lambda_1 \lambda_2}`, where
    :math:`\rho` is the correlation coefficient between :math:`k_1` and
    :math:`k_2`. If the two Poisson-distributed r.v. are independent then
    :math:`\rho = 0`.

    Parameters :math:`\mu_1` and :math:`\mu_2` must be strictly positive.

    For details see: https://en.wikipedia.org/wiki/Skellam_distribution

    `skellam` takes :math:`\mu_1` and :math:`\mu_2` as shape parameters.

    %(after_notes)s

    %(example)s

    """
    def _rvs(self, mu1, mu2, size=None, random_state=None):
        n = size
        return (random_state.poisson(mu1, n) -
                random_state.poisson(mu2, n))

    def _pmf(self, x, mu1, mu2):
        px = np.where(x < 0,
                      _ncx2_pdf(2*mu2, 2*(1-x), 2*mu1)*2,
                      _ncx2_pdf(2*mu1, 2*(1+x), 2*mu2)*2)
        # ncx2.pdf() returns nan's for extremely low probabilities
        return px

    def _cdf(self, x, mu1, mu2):
        x = floor(x)
        px = np.where(x < 0,
                      _ncx2_cdf(2*mu2, -2*x, 2*mu1),
                      1 - _ncx2_cdf(2*mu1, 2*(x+1), 2*mu2))
        return px

    def _stats(self, mu1, mu2):
        mean = mu1 - mu2
        var = mu1 + mu2
        g1 = mean / sqrt((var)**3)
        g2 = 1 / var
        return mean, var, g1, g2


skellam = skellam_gen(a=-np.inf, name="skellam", longname='A Skellam')


class yulesimon_gen(rv_discrete):
    r"""A Yule-Simon discrete random variable.

    %(before_notes)s

    Notes
    -----

    The probability mass function for the `yulesimon` is:

    .. math::

        f(k) =  \alpha B(k, \alpha+1)

    for :math:`k=1,2,3,...`, where :math:`\alpha>0`.
    Here :math:`B` refers to the `scipy.special.beta` function.

    The sampling of random variates is based on pg 553, Section 6.3 of [1]_.
    Our notation maps to the referenced logic via :math:`\alpha=a-1`.

    For details see the wikipedia entry [2]_.

    References
    ----------
    .. [1] Devroye, Luc. "Non-uniform Random Variate Generation",
         (1986) Springer, New York.

    .. [2] https://en.wikipedia.org/wiki/Yule-Simon_distribution

    %(after_notes)s

    %(example)s

    """
    def _rvs(self, alpha, size=None, random_state=None):
        E1 = random_state.standard_exponential(size)
        E2 = random_state.standard_exponential(size)
        ans = ceil(-E1 / log1p(-exp(-E2 / alpha)))
        return ans

    def _pmf(self, x, alpha):
        return alpha * special.beta(x, alpha + 1)

    def _argcheck(self, alpha):
        return (alpha > 0)

    def _logpmf(self, x, alpha):
        return log(alpha) + special.betaln(x, alpha + 1)

    def _cdf(self, x, alpha):
        return 1 - x * special.beta(x, alpha + 1)

    def _sf(self, x, alpha):
        return x * special.beta(x, alpha + 1)

    def _logsf(self, x, alpha):
        return log(x) + special.betaln(x, alpha + 1)

    def _stats(self, alpha):
        mu = np.where(alpha <= 1, np.inf, alpha / (alpha - 1))
        mu2 = np.where(alpha > 2,
                alpha**2 / ((alpha - 2.0) * (alpha - 1)**2),
                np.inf)
        mu2 = np.where(alpha <= 1, np.nan, mu2)
        g1 = np.where(alpha > 3,
                sqrt(alpha - 2) * (alpha + 1)**2 / (alpha * (alpha - 3)),
                np.inf)
        g1 = np.where(alpha <= 2, np.nan, g1)
        g2 = np.where(alpha > 4,
                (alpha + 3) + (alpha**3 - 49 * alpha - 22) / (alpha *
                        (alpha - 4) * (alpha - 3)), np.inf)
        g2 = np.where(alpha <= 2, np.nan, g2)
        return mu, mu2, g1, g2


yulesimon = yulesimon_gen(name='yulesimon', a=1)


def _vectorize_rvs_over_shapes(_rvs1):
    """Decorator that vectorizes _rvs method to work on ndarray shapes"""
    # _rvs1 must be a _function_ that accepts _scalar_ args as positional
    # arguments, `size` and `random_state` as keyword arguments.
    # _rvs1 must return a random variate array with shape `size`. If `size` is
    # None, _rvs1 must return a scalar.
    # When applied to _rvs1, this decorator broadcasts ndarray args
    # and loops over them, calling _rvs1 for each set of scalar args.
    # For usage example, see _nchypergeom_gen
    def _rvs(*args, size, random_state):
        _rvs1_size, _rvs1_indices = _check_shape(args[0].shape, size)

        size = np.array(size)
        _rvs1_size = np.array(_rvs1_size)
        _rvs1_indices = np.array(_rvs1_indices)

        if np.all(_rvs1_indices):  # all args are scalars
            return _rvs1(*args, size, random_state)

        out = np.empty(size)

        # out.shape can mix dimensions associated with arg_shape and _rvs1_size
        # Sort them to arg_shape + _rvs1_size for easy indexing of dimensions
        # corresponding with the different sets of scalar args
        j0 = np.arange(out.ndim)
        j1 = np.hstack((j0[~_rvs1_indices], j0[_rvs1_indices]))
        out = np.moveaxis(out, j1, j0)

        for i in np.ndindex(*size[~_rvs1_indices]):
            # arg can be squeezed because singleton dimensions will be
            # associated with _rvs1_size, not arg_shape per _check_shape
            out[i] = _rvs1(*[np.squeeze(arg)[i] for arg in args],
                           _rvs1_size, random_state)

        return np.moveaxis(out, j0, j1)  # move axes back before returning
    return _rvs


class _nchypergeom_gen(rv_discrete):
    r"""A noncentral hypergeometric discrete random variable.

    For subclassing by nchypergeom_fisher_gen and nchypergeom_wallenius_gen.

    """

    rvs_name = None
    dist = None

    def _get_support(self, M, n, N, odds):
        N, m1, n = M, n, N  # follow Wikipedia notation
        m2 = N - m1
        x_min = np.maximum(0, n - m2)
        x_max = np.minimum(n, m1)
        return x_min, x_max

    def _argcheck(self, M, n, N, odds):
        M, n = np.asarray(M), np.asarray(n),
        N, odds = np.asarray(N), np.asarray(odds)
        cond1 = (M.astype(int) == M) & (M >= 0)
        cond2 = (n.astype(int) == n) & (n >= 0)
        cond3 = (N.astype(int) == N) & (N >= 0)
        cond4 = odds > 0
        cond5 = N <= M
        cond6 = n <= M
        return cond1 & cond2 & cond3 & cond4 & cond5 & cond6

    def _rvs(self, M, n, N, odds, size=None, random_state=None):

        @_vectorize_rvs_over_shapes
        def _rvs1(M, n, N, odds, size, random_state):
            length = np.prod(size)
            urn = _PyStochasticLib3()
            rv_gen = getattr(urn, self.rvs_name)
            rvs = rv_gen(N, n, M, odds, length, random_state)
            rvs = rvs.reshape(size)
            return rvs

        return _rvs1(M, n, N, odds, size=size, random_state=random_state)

    def _pmf(self, x, M, n, N, odds):

        @np.vectorize
        def _pmf1(x, M, n, N, odds):
            urn = self.dist(N, n, M, odds, 1e-12)
            return urn.probability(x)

        return _pmf1(x, M, n, N, odds)

    def _stats(self, M, n, N, odds, moments):

        @np.vectorize
        def _moments1(M, n, N, odds):
            urn = self.dist(N, n, M, odds, 1e-12)
            return urn.moments()

        m, v = _moments1(M, n, N, odds) if ("m" in moments
                                            or "v" in moments) else None
        s, k = None, None
        return m, v, s, k


class nchypergeom_fisher_gen(_nchypergeom_gen):
    r"""A Fisher's noncentral hypergeometric discrete random variable.

    Fisher's noncentral hypergeometric distribution models drawing objects of
    two types from a bin. `M` is the total number of objects, `n` is the
    number of Type I objects, and `odds` is the odds ratio: the odds of
    selecting a Type I object rather than a Type II object when there is only
    one object of each type.
    The random variate represents the number of Type I objects drawn if we
    take a handful of objects from the bin at once and find out afterwards
    that we took `N` objects.

    %(before_notes)s

    See Also
    --------
    nchypergeom_wallenius, hypergeom, nhypergeom

    Notes
    -----
    Let mathematical symbols :math:`N`, :math:`n`, and :math:`M` correspond
    with parameters `N`, `n`, and `M` (respectively) as defined above.

    The probability mass function is defined as

    .. math::

        p(x; M, n, N, \omega) =
        \frac{\binom{n}{x}\binom{M - n}{N-x}\omega^x}{P_0},

    for
    :math:`x \in [x_l, x_u]`,
    :math:`M \in {\mathbb N}`,
    :math:`n \in [0, M]`,
    :math:`N \in [0, M]`,
    :math:`\omega > 0`,
    where
    :math:`x_l = \max(0, N - (M - n))`,
    :math:`x_u = \min(N, n)`,

    .. math::

        P_0 = \sum_{y=x_l}^{x_u} \binom{n}{y}\binom{M - n}{N-y}\omega^y,

    and the binomial coefficients are defined as

    .. math:: \binom{n}{k} \equiv \frac{n!}{k! (n - k)!}.

    `nchypergeom_fisher` uses the BiasedUrn package by Agner Fog with
    permission for it to be distributed under SciPy's license.

    The symbols used to denote the shape parameters (`N`, `n`, and `M`) are not
    universally accepted; they are chosen for consistency with `hypergeom`.

    Note that Fisher's noncentral hypergeometric distribution is distinct
    from Wallenius' noncentral hypergeometric distribution, which models
    drawing a pre-determined `N` objects from a bin one by one.
    When the odds ratio is unity, however, both distributions reduce to the
    ordinary hypergeometric distribution.

    %(after_notes)s

    References
    ----------
    .. [1] Agner Fog, "Biased Urn Theory".
           https://cran.r-project.org/web/packages/BiasedUrn/vignettes/UrnTheory.pdf

    .. [2] "Fisher's noncentral hypergeometric distribution", Wikipedia,
           https://en.wikipedia.org/wiki/Fisher's_noncentral_hypergeometric_distribution

    %(example)s

    """

    rvs_name = "rvs_fisher"
    dist = _PyFishersNCHypergeometric


nchypergeom_fisher = nchypergeom_fisher_gen(
    name='nchypergeom_fisher',
    longname="A Fisher's noncentral hypergeometric")


class nchypergeom_wallenius_gen(_nchypergeom_gen):
    r"""A Wallenius' noncentral hypergeometric discrete random variable.

    Wallenius' noncentral hypergeometric distribution models drawing objects of
    two types from a bin. `M` is the total number of objects, `n` is the
    number of Type I objects, and `odds` is the odds ratio: the odds of
    selecting a Type I object rather than a Type II object when there is only
    one object of each type.
    The random variate represents the number of Type I objects drawn if we
    draw a pre-determined `N` objects from a bin one by one.

    %(before_notes)s

    See Also
    --------
    nchypergeom_fisher, hypergeom, nhypergeom

    Notes
    -----
    Let mathematical symbols :math:`N`, :math:`n`, and :math:`M` correspond
    with parameters `N`, `n`, and `M` (respectively) as defined above.

    The probability mass function is defined as

    .. math::

        p(x; N, n, M) = \binom{n}{x} \binom{M - n}{N-x}
        \int_0^1 \left(1-t^{\omega/D}\right)^x\left(1-t^{1/D}\right)^{N-x} dt

    for
    :math:`x \in [x_l, x_u]`,
    :math:`M \in {\mathbb N}`,
    :math:`n \in [0, M]`,
    :math:`N \in [0, M]`,
    :math:`\omega > 0`,
    where
    :math:`x_l = \max(0, N - (M - n))`,
    :math:`x_u = \min(N, n)`,

    .. math::

        D = \omega(n - x) + ((M - n)-(N-x)),

    and the binomial coefficients are defined as

    .. math:: \binom{n}{k} \equiv \frac{n!}{k! (n - k)!}.

    `nchypergeom_wallenius` uses the BiasedUrn package by Agner Fog with
    permission for it to be distributed under SciPy's license.

    The symbols used to denote the shape parameters (`N`, `n`, and `M`) are not
    universally accepted; they are chosen for consistency with `hypergeom`.

    Note that Wallenius' noncentral hypergeometric distribution is distinct
    from Fisher's noncentral hypergeometric distribution, which models
    take a handful of objects from the bin at once, finding out afterwards
    that `N` objects were taken.
    When the odds ratio is unity, however, both distributions reduce to the
    ordinary hypergeometric distribution.

    %(after_notes)s

    References
    ----------
    .. [1] Agner Fog, "Biased Urn Theory".
           https://cran.r-project.org/web/packages/BiasedUrn/vignettes/UrnTheory.pdf

    .. [2] "Wallenius' noncentral hypergeometric distribution", Wikipedia,
           https://en.wikipedia.org/wiki/Wallenius'_noncentral_hypergeometric_distribution

    %(example)s

    """

    rvs_name = "rvs_wallenius"
    dist = _PyWalleniusNCHypergeometric


nchypergeom_wallenius = nchypergeom_wallenius_gen(
    name='nchypergeom_wallenius',
    longname="A Wallenius' noncentral hypergeometric")


# Collect names of classes and objects in this module.
pairs = list(globals().items())
_distn_names, _distn_gen_names = get_distribution_names(pairs, rv_discrete)

__all__ = _distn_names + _distn_gen_names
