
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special._ufuncs import _cosine_cdf, _cosine_invcdf


# These values are (x, p) where p is the expected exact value of
# _cosine_cdf(x).  These values will be tested for exact agreement.
_coscdf_exact = [
    (-4.0, 0.0),
    (0, 0.5),
    (np.pi, 1.0),
    (4.0, 1.0),
]

@pytest.mark.parametrize("x, expected", _coscdf_exact)
def test_cosine_cdf_exact(x, expected):
    assert _cosine_cdf(x) == expected


# These values are (x, p), where p is the expected value of
# _cosine_cdf(x). The expected values were computed with mpmath using
# 50 digits of precision.  These values will be tested for agreement
# with the computed values using a very small relative tolerance.
# The value at -np.pi is not 0, because -np.pi does not equal -π.
_coscdf_close = [
    (3.1409, 0.999999999991185),
    (2.25, 0.9819328173287907),
    # -1.6 is the threshold below which the Pade approximant is used.
    (-1.599, 0.08641959838382553),
    (-1.601, 0.086110582992713),
    (-2.0, 0.0369709335961611),
    (-3.0, 7.522387241801384e-05),
    (-3.1415, 2.109869685443648e-14),
    (-3.14159, 4.956444476505336e-19),
    (-np.pi, 4.871934450264861e-50),
]

@pytest.mark.parametrize("x, expected", _coscdf_close)
def test_cosine_cdf(x, expected):
    assert_allclose(_cosine_cdf(x), expected, rtol=5e-15)


# These values are (p, x) where x is the expected exact value of
# _cosine_invcdf(p).  These values will be tested for exact agreement.
_cosinvcdf_exact = [
    (0.0, -np.pi),
    (0.5, 0.0),
    (1.0, np.pi),
]

@pytest.mark.parametrize("p, expected", _cosinvcdf_exact)
def test_cosine_invcdf_exact(p, expected):
    assert _cosine_invcdf(p) == expected


def test_cosine_invcdf_invalid_p():
    # Check that p values outside of [0, 1] return nan.
    assert np.isnan(_cosine_invcdf([-0.1, 1.1])).all()


# These values are (p, x), where x is the expected value of _cosine_invcdf(p).
# The expected values were computed with mpmath using 50 digits of precision.
_cosinvcdf_close = [
    (1e-50, -np.pi),
    (1e-14, -3.1415204137058454),
    (1e-08, -3.1343686589124524),
    (0.0018001, -2.732563923138336),
    (0.010, -2.41276589008678),
    (0.060, -1.7881244975330157),
    (0.125, -1.3752523669869274),
    (0.250, -0.831711193579736),
    (0.400, -0.3167954512395289),
    (0.419, -0.25586025626919906),
    (0.421, -0.24947570750445663),
    (0.750, 0.831711193579736),
    (0.940, 1.7881244975330153),
    (0.9999999996, 3.1391220839917167),
]

@pytest.mark.parametrize("p, expected", _cosinvcdf_close)
def test_cosine_invcdf(p, expected):
    assert_allclose(_cosine_invcdf(p), expected, rtol=1e-14)
