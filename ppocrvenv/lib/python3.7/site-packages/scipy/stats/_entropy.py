# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:06:05 2021

@author: matth
"""

from __future__ import annotations
import math
import numpy as np
from scipy import special
from typing import Optional, Union

__all__ = ['entropy', 'differential_entropy']


def entropy(pk, qk=None, base=None, axis=0):
    """Calculate the entropy of a distribution for given probability values.

    If only probabilities `pk` are given, the entropy is calculated as
    ``S = -sum(pk * log(pk), axis=axis)``.

    If `qk` is not None, then compute the Kullback-Leibler divergence
    ``S = sum(pk * log(pk / qk), axis=axis)``.

    This routine will normalize `pk` and `qk` if they don't sum to 1.

    Parameters
    ----------
    pk : sequence
        Defines the (discrete) distribution. ``pk[i]`` is the (possibly
        unnormalized) probability of event ``i``.
    qk : sequence, optional
        Sequence against which the relative entropy is computed. Should be in
        the same format as `pk`.
    base : float, optional
        The logarithmic base to use, defaults to ``e`` (natural logarithm).
    axis: int, optional
        The axis along which the entropy is calculated. Default is 0.

    Returns
    -------
    S : float
        The calculated entropy.

    Examples
    --------

    >>> from scipy.stats import entropy

    Bernoulli trial with different p.
    The outcome of a fair coin is the most uncertain:

    >>> entropy([1/2, 1/2], base=2)
    1.0

    The outcome of a biased coin is less uncertain:

    >>> entropy([9/10, 1/10], base=2)
    0.46899559358928117

    Relative entropy:

    >>> entropy([1/2, 1/2], qk=[9/10, 1/10])
    0.5108256237659907

    """
    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")

    pk = np.asarray(pk)
    pk = 1.0*pk / np.sum(pk, axis=axis, keepdims=True)
    if qk is None:
        vec = special.entr(pk)
    else:
        qk = np.asarray(qk)
        pk, qk = np.broadcast_arrays(pk, qk)
        qk = 1.0*qk / np.sum(qk, axis=axis, keepdims=True)
        vec = special.rel_entr(pk, qk)
    S = np.sum(vec, axis=axis)
    if base is not None:
        S /= np.log(base)
    return S


def differential_entropy(
    values: np.typing.ArrayLike,
    *,
    window_length: Optional[int] = None,
    base: Optional[float] = None,
    axis: int = 0,
    method: str = "auto",
) -> Union[np.number, np.ndarray]:
    r"""Given a sample of a distribution, estimate the differential entropy.

    Several estimation methods are available using the `method` parameter. By
    default, a method is selected based the size of the sample.

    Parameters
    ----------
    values : sequence
        Sample from a continuous distribution.
    window_length : int, optional
        Window length for computing Vasicek estimate. Must be an integer
        between 1 and half of the sample size. If ``None`` (the default), it
        uses the heuristic value

        .. math::
            \left \lfloor \sqrt{n} + 0.5 \right \rfloor

        where :math:`n` is the sample size. This heuristic was originally
        proposed in [2]_ and has become common in the literature.
    base : float, optional
        The logarithmic base to use, defaults to ``e`` (natural logarithm).
    axis : int, optional
        The axis along which the differential entropy is calculated.
        Default is 0.
    method : {'vasicek', 'van es', 'ebrahimi', 'correa', 'auto'}, optional
        The method used to estimate the differential entropy from the sample.
        Default is ``'auto'``.  See Notes for more information.

    Returns
    -------
    entropy : float
        The calculated differential entropy.

    Notes
    -----
    This function will converge to the true differential entropy in the limit

    .. math::
        n \to \infty, \quad m \to \infty, \quad \frac{m}{n} \to 0

    The optimal choice of ``window_length`` for a given sample size depends on
    the (unknown) distribution. Typically, the smoother the density of the
    distribution, the larger the optimal value of ``window_length`` [1]_.

    The following options are available for the `method` parameter.

    * ``'vasicek'`` uses the estimator presented in [1]_. This is
      one of the first and most influential estimators of differential entropy.
    * ``'van es'`` uses the bias-corrected estimator presented in [3]_, which
      is not only consistent but, under some conditions, asymptotically normal.
    * ``'ebrahimi'`` uses an estimator presented in [4]_, which was shown
      in simulation to have smaller bias and mean squared error than
      the Vasicek estimator.
    * ``'correa'`` uses the estimator presented in [5]_ based on local linear
      regression. In a simulation study, it had consistently smaller mean
      square error than the Vasiceck estimator, but it is more expensive to
      compute.
    * ``'auto'`` selects the method automatically (default). Currently,
      this selects ``'van es'`` for very small samples (<10), ``'ebrahimi'``
      for moderate sample sizes (11-1000), and ``'vasicek'`` for larger
      samples, but this behavior is subject to change in future versions.

    All estimators are implemented as described in [6]_.

    References
    ----------
    .. [1] Vasicek, O. (1976). A test for normality based on sample entropy.
           Journal of the Royal Statistical Society:
           Series B (Methodological), 38(1), 54-59.
    .. [2] Crzcgorzewski, P., & Wirczorkowski, R. (1999). Entropy-based
           goodness-of-fit test for exponentiality. Communications in
           Statistics-Theory and Methods, 28(5), 1183-1202.
    .. [3] Van Es, B. (1992). Estimating functionals related to a density by a
           class of statistics based on spacings. Scandinavian Journal of
           Statistics, 61-72.
    .. [4] Ebrahimi, N., Pflughoeft, K., & Soofi, E. S. (1994). Two measures
           of sample entropy. Statistics & Probability Letters, 20(3), 225-234.
    .. [5] Correa, J. C. (1995). A new estimator of entropy. Communications
           in Statistics-Theory and Methods, 24(10), 2439-2449.
    .. [6] Noughabi, H. A. (2015). Entropy Estimation Using Numerical Methods.
           Annals of Data Science, 2(2), 231-241.
           https://link.springer.com/article/10.1007/s40745-015-0045-9

    Examples
    --------
    >>> from scipy.stats import differential_entropy, norm

    Entropy of a standard normal distribution:

    >>> rng = np.random.default_rng()
    >>> values = rng.standard_normal(100)
    >>> differential_entropy(values)
    1.3407817436640392

    Compare with the true entropy:

    >>> float(norm.entropy())
    1.4189385332046727

    For several sample sizes between 5 and 1000, compare the accuracy of
    the ``'vasicek'``, ``'van es'``, and ``'ebrahimi'`` methods. Specifically,
    compare the root mean squared error (over 1000 trials) between the estimate
    and the true differential entropy of the distribution.

    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>>
    >>>
    >>> def rmse(res, expected):
    ...     '''Root mean squared error'''
    ...     return np.sqrt(np.mean((res - expected)**2))
    >>>
    >>>
    >>> a, b = np.log10(5), np.log10(1000)
    >>> ns = np.round(np.logspace(a, b, 10)).astype(int)
    >>> reps = 1000  # number of repetitions for each sample size
    >>> expected = stats.expon.entropy()
    >>>
    >>> method_errors = {'vasicek': [], 'van es': [], 'ebrahimi': []}
    >>> for method in method_errors:
    ...     for n in ns:
    ...        rvs = stats.expon.rvs(size=(reps, n), random_state=rng)
    ...        res = stats.differential_entropy(rvs, method=method, axis=-1)
    ...        error = rmse(res, expected)
    ...        method_errors[method].append(error)
    >>>
    >>> for method, errors in method_errors.items():
    ...     plt.loglog(ns, errors, label=method)
    >>>
    >>> plt.legend()
    >>> plt.xlabel('sample size')
    >>> plt.ylabel('RMSE (1000 trials)')
    >>> plt.title('Entropy Estimator Error (Exponential Distribution)')

    """
    values = np.asarray(values)
    values = np.moveaxis(values, axis, -1)
    n = values.shape[-1]  # number of observations

    if window_length is None:
        window_length = math.floor(math.sqrt(n) + 0.5)

    if not 2 <= 2 * window_length < n:
        raise ValueError(
            f"Window length ({window_length}) must be positive and less "
            f"than half the sample size ({n}).",
        )

    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")

    sorted_data = np.sort(values, axis=-1)

    methods = {"vasicek": _vasicek_entropy,
               "van es": _van_es_entropy,
               "correa": _correa_entropy,
               "ebrahimi": _ebrahimi_entropy,
               "auto": _vasicek_entropy}
    method = method.lower()
    if method not in methods:
        message = f"`method` must be one of {set(methods)}"
        raise ValueError(message)

    if method == "auto":
        if n <= 10:
            method = 'van es'
        elif n <= 1000:
            method = 'ebrahimi'
        else:
            method = 'vasicek'

    res = methods[method](sorted_data, window_length)

    if base is not None:
        res /= np.log(base)

    return res


def _pad_along_last_axis(X, m):
    """Pad the data for computing the rolling window difference."""
    # scales a  bit better than method in _vasicek_like_entropy
    shape = np.array(X.shape)
    shape[-1] = m
    Xl = np.broadcast_to(X[..., [0]], shape)  # [0] vs 0 to maintain shape
    Xr = np.broadcast_to(X[..., [-1]], shape)
    return np.concatenate((Xl, X, Xr), axis=-1)


def _vasicek_entropy(X, m):
    """Compute the Vasicek estimator as described in [6] Eq. 1.3."""
    n = X.shape[-1]
    X = _pad_along_last_axis(X, m)
    differences = X[..., 2 * m:] - X[..., : -2 * m:]
    logs = np.log(n/(2*m) * differences)
    return np.mean(logs, axis=-1)


def _van_es_entropy(X, m):
    """Compute the van Es estimator as described in [6]."""
    # No equation number, but referred to as HVE_mn.
    # Typo: there should be a log within the summation.
    n = X.shape[-1]
    difference = X[..., m:] - X[..., :-m]
    term1 = 1/(n-m) * np.sum(np.log((n+1)/m * difference), axis=-1)
    k = np.arange(m, n+1)
    return term1 + np.sum(1/k) + np.log(m) - np.log(n+1)


def _ebrahimi_entropy(X, m):
    """Compute the Ebrahimi estimator as described in [6]."""
    # No equation number, but referred to as HE_mn
    n = X.shape[-1]
    X = _pad_along_last_axis(X, m)

    differences = X[..., 2 * m:] - X[..., : -2 * m:]

    i = np.arange(1, n+1).astype(float)
    ci = np.ones_like(i)*2
    ci[i <= m] = 1 + (i[i <= m] - 1)/m
    ci[i >= n - m + 1] = 1 + (n - i[i >= n-m+1])/m

    logs = np.log(n * differences / (ci * m))
    return np.mean(logs, axis=-1)


def _correa_entropy(X, m):
    """Compute the Correa estimator as described in [6]."""
    # No equation number, but referred to as HC_mn
    n = X.shape[-1]
    X = _pad_along_last_axis(X, m)

    i = np.arange(1, n+1)
    dj = np.arange(-m, m+1)[:, None]
    j = i + dj
    j0 = j + m - 1  # 0-indexed version of j

    Xibar = np.mean(X[..., j0], axis=-2, keepdims=True)
    difference = X[..., j0] - Xibar
    num = np.sum(difference*dj, axis=-2)  # dj is d-i
    den = n*np.sum(difference**2, axis=-2)
    return -np.mean(np.log(num/den), axis=-1)
