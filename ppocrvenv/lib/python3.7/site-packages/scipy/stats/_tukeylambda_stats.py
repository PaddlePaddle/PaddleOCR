import numpy as np
from numpy import poly1d
from scipy.special import beta


# The following code was used to generate the Pade coefficients for the
# Tukey Lambda variance function.  Version 0.17 of mpmath was used.
#---------------------------------------------------------------------------
# import mpmath as mp
#
# mp.mp.dps = 60
#
# one   = mp.mpf(1)
# two   = mp.mpf(2)
#
# def mpvar(lam):
#     if lam == 0:
#         v = mp.pi**2 / three
#     else:
#         v = (two / lam**2) * (one / (one + two*lam) -
#                               mp.beta(lam + one, lam + one))
#     return v
#
# t = mp.taylor(mpvar, 0, 8)
# p, q = mp.pade(t, 4, 4)
# print("p =", [mp.fp.mpf(c) for c in p])
# print("q =", [mp.fp.mpf(c) for c in q])
#---------------------------------------------------------------------------

# Pade coefficients for the Tukey Lambda variance function.
_tukeylambda_var_pc = [3.289868133696453, 0.7306125098871127,
                       -0.5370742306855439, 0.17292046290190008,
                       -0.02371146284628187]
_tukeylambda_var_qc = [1.0, 3.683605511659861, 4.184152498888124,
                       1.7660926747377275, 0.2643989311168465]

# numpy.poly1d instances for the numerator and denominator of the
# Pade approximation to the Tukey Lambda variance.
_tukeylambda_var_p = poly1d(_tukeylambda_var_pc[::-1])
_tukeylambda_var_q = poly1d(_tukeylambda_var_qc[::-1])


def tukeylambda_variance(lam):
    """Variance of the Tukey Lambda distribution.

    Parameters
    ----------
    lam : array_like
        The lambda values at which to compute the variance.

    Returns
    -------
    v : ndarray
        The variance.  For lam < -0.5, the variance is not defined, so
        np.nan is returned.  For lam = 0.5, np.inf is returned.

    Notes
    -----
    In an interval around lambda=0, this function uses the [4,4] Pade
    approximation to compute the variance.  Otherwise it uses the standard
    formula (https://en.wikipedia.org/wiki/Tukey_lambda_distribution).  The
    Pade approximation is used because the standard formula has a removable
    discontinuity at lambda = 0, and does not produce accurate numerical
    results near lambda = 0.
    """
    lam = np.asarray(lam)
    shp = lam.shape
    lam = np.atleast_1d(lam).astype(np.float64)

    # For absolute values of lam less than threshold, use the Pade
    # approximation.
    threshold = 0.075

    # Play games with masks to implement the conditional evaluation of
    # the distribution.
    # lambda < -0.5:  var = nan
    low_mask = lam < -0.5
    # lambda == -0.5: var = inf
    neghalf_mask = lam == -0.5
    # abs(lambda) < threshold:  use Pade approximation
    small_mask = np.abs(lam) < threshold
    # else the "regular" case:  use the explicit formula.
    reg_mask = ~(low_mask | neghalf_mask | small_mask)

    # Get the 'lam' values for the cases where they are needed.
    small = lam[small_mask]
    reg = lam[reg_mask]

    # Compute the function for each case.
    v = np.empty_like(lam)
    v[low_mask] = np.nan
    v[neghalf_mask] = np.inf
    if small.size > 0:
        # Use the Pade approximation near lambda = 0.
        v[small_mask] = _tukeylambda_var_p(small) / _tukeylambda_var_q(small)
    if reg.size > 0:
        v[reg_mask] = (2.0 / reg**2) * (1.0 / (1.0 + 2 * reg) -
                                      beta(reg + 1, reg + 1))
    v.shape = shp
    return v


# The following code was used to generate the Pade coefficients for the
# Tukey Lambda kurtosis function.  Version 0.17 of mpmath was used.
#---------------------------------------------------------------------------
# import mpmath as mp
#
# mp.mp.dps = 60
#
# one   = mp.mpf(1)
# two   = mp.mpf(2)
# three = mp.mpf(3)
# four  = mp.mpf(4)
#
# def mpkurt(lam):
#     if lam == 0:
#         k = mp.mpf(6)/5
#     else:
#         numer = (one/(four*lam+one) - four*mp.beta(three*lam+one, lam+one) +
#                  three*mp.beta(two*lam+one, two*lam+one))
#         denom = two*(one/(two*lam+one) - mp.beta(lam+one,lam+one))**2
#         k = numer / denom - three
#     return k
#
# # There is a bug in mpmath 0.17: when we use the 'method' keyword of the
# # taylor function and we request a degree 9 Taylor polynomial, we actually
# # get degree 8.
# t = mp.taylor(mpkurt, 0, 9, method='quad', radius=0.01)
# t = [mp.chop(c, tol=1e-15) for c in t]
# p, q = mp.pade(t, 4, 4)
# print("p =", [mp.fp.mpf(c) for c in p])
# print("q =", [mp.fp.mpf(c) for c in q])
#---------------------------------------------------------------------------

# Pade coefficients for the Tukey Lambda kurtosis function.
_tukeylambda_kurt_pc = [1.2, -5.853465139719495, -22.653447381131077,
                        0.20601184383406815, 4.59796302262789]
_tukeylambda_kurt_qc = [1.0, 7.171149192233599, 12.96663094361842,
                        0.43075235247853005, -2.789746758009912]

# numpy.poly1d instances for the numerator and denominator of the
# Pade approximation to the Tukey Lambda kurtosis.
_tukeylambda_kurt_p = poly1d(_tukeylambda_kurt_pc[::-1])
_tukeylambda_kurt_q = poly1d(_tukeylambda_kurt_qc[::-1])


def tukeylambda_kurtosis(lam):
    """Kurtosis of the Tukey Lambda distribution.

    Parameters
    ----------
    lam : array_like
        The lambda values at which to compute the variance.

    Returns
    -------
    v : ndarray
        The variance.  For lam < -0.25, the variance is not defined, so
        np.nan is returned.  For lam = 0.25, np.inf is returned.

    """
    lam = np.asarray(lam)
    shp = lam.shape
    lam = np.atleast_1d(lam).astype(np.float64)

    # For absolute values of lam less than threshold, use the Pade
    # approximation.
    threshold = 0.055

    # Use masks to implement the conditional evaluation of the kurtosis.
    # lambda < -0.25:  kurtosis = nan
    low_mask = lam < -0.25
    # lambda == -0.25: kurtosis = inf
    negqrtr_mask = lam == -0.25
    # lambda near 0:  use Pade approximation
    small_mask = np.abs(lam) < threshold
    # else the "regular" case:  use the explicit formula.
    reg_mask = ~(low_mask | negqrtr_mask | small_mask)

    # Get the 'lam' values for the cases where they are needed.
    small = lam[small_mask]
    reg = lam[reg_mask]

    # Compute the function for each case.
    k = np.empty_like(lam)
    k[low_mask] = np.nan
    k[negqrtr_mask] = np.inf
    if small.size > 0:
        k[small_mask] = _tukeylambda_kurt_p(small) / _tukeylambda_kurt_q(small)
    if reg.size > 0:
        numer = (1.0 / (4 * reg + 1) - 4 * beta(3 * reg + 1, reg + 1) +
                 3 * beta(2 * reg + 1, 2 * reg + 1))
        denom = 2 * (1.0/(2 * reg + 1) - beta(reg + 1, reg + 1))**2
        k[reg_mask] = numer / denom - 3

    # The return value will be a numpy array; resetting the shape ensures that
    # if `lam` was a scalar, the return value is a 0-d array.
    k.shape = shp
    return k
