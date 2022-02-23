import pytest
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_equal

import scipy.special as sc


class TestHyperu:

    def test_negative_x(self):
        a, b, x = np.meshgrid(
            [-1, -0.5, 0, 0.5, 1],
            [-1, -0.5, 0, 0.5, 1],
            np.linspace(-100, -1, 10),
        )
        assert np.all(np.isnan(sc.hyperu(a, b, x)))

    def test_special_cases(self):
        assert sc.hyperu(0, 1, 1) == 1.0

    @pytest.mark.parametrize('a', [0.5, 1, np.nan])
    @pytest.mark.parametrize('b', [1, 2, np.nan])
    @pytest.mark.parametrize('x', [0.25, 3, np.nan])
    def test_nan_inputs(self, a, b, x):
        assert np.isnan(sc.hyperu(a, b, x)) == np.any(np.isnan([a, b, x]))

class TestHyp1f1:

    @pytest.mark.parametrize('a, b, x', [
        (np.nan, 1, 1),
        (1, np.nan, 1),
        (1, 1, np.nan)
    ])
    def test_nan_inputs(self, a, b, x):
        assert np.isnan(sc.hyp1f1(a, b, x))

    def test_poles(self):
        assert_equal(sc.hyp1f1(1, [0, -1, -2, -3, -4], 0.5), np.infty)

    @pytest.mark.parametrize('a, b, x, result', [
        (-1, 1, 0.5, 0.5),
        (1, 1, 0.5, 1.6487212707001281468),
        (2, 1, 0.5, 2.4730819060501922203),
        (1, 2, 0.5, 1.2974425414002562937),
        (-10, 1, 0.5, -0.38937441413785204475)
    ])
    def test_special_cases(self, a, b, x, result):
        # Hit all the special case branches at the beginning of the
        # function. Desired answers computed using Mpmath.
        assert_allclose(sc.hyp1f1(a, b, x), result, atol=0, rtol=1e-15)

    @pytest.mark.parametrize('a, b, x, result', [
        (1, 1, 0.44, 1.5527072185113360455),
        (-1, 1, 0.44, 0.55999999999999999778),
        (100, 100, 0.89, 2.4351296512898745592),
        (-100, 100, 0.89, 0.40739062490768104667),
        (1.5, 100, 59.99, 3.8073513625965598107),
        (-1.5, 100, 59.99, 0.25099240047125826943)
    ])
    def test_geometric_convergence(self, a, b, x, result):
        # Test the region where we are relying on the ratio of
        #
        # (|a| + 1) * |x| / |b|
        #
        # being small. Desired answers computed using Mpmath
        assert_allclose(sc.hyp1f1(a, b, x), result, atol=0, rtol=1e-15)

    @pytest.mark.parametrize('a, b, x, result', [
        (-1, 1, 1.5, -0.5),
        (-10, 1, 1.5, 0.41801777430943080357),
        (-25, 1, 1.5, 0.25114491646037839809),
        (-50, 1, 1.5, -0.25683643975194756115),
        (-51, 1, 1.5, -0.19843162753845452972)
    ])
    def test_a_negative_integer(self, a, b, x, result):
        # Desired answers computed using Mpmath. After -51 the
        # relative error becomes unsatisfactory and we start returning
        # NaN.
        assert_allclose(sc.hyp1f1(a, b, x), result, atol=0, rtol=1e-9)

    def test_gh_3492(self):
        desired = 0.99973683897677527773  # Computed using Mpmath
        assert_allclose(
            sc.hyp1f1(0.01, 150, -4),
            desired,
            atol=0,
            rtol=1e-15
        )

    def test_gh_3593(self):
        desired = 1.0020033381011970966  # Computed using Mpmath
        assert_allclose(
            sc.hyp1f1(1, 5, 0.01),
            desired,
            atol=0,
            rtol=1e-15
        )

    @pytest.mark.parametrize('a, b, x, desired', [
        (-1, -2, 2, 2),
        (-1, -4, 10, 3.5),
        (-2, -2, 1, 2.5)
    ])
    def test_gh_11099(self, a, b, x, desired):
        # All desired results computed using Mpmath
        assert sc.hyp1f1(a, b, x) == desired
