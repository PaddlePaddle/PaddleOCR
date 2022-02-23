import pytest
import numpy as np
from numpy.testing import assert_equal
from scipy.special import log_ndtr, ndtri_exp
from scipy.special._testutils import assert_func_equal


def log_ndtr_ndtri_exp(y):
    return log_ndtr(ndtri_exp(y))


@pytest.fixture(scope="class")
def uniform_random_points():
    random_state = np.random.RandomState(1234)
    points = random_state.random_sample(1000)
    return points


class TestNdtriExp:
    """Tests that ndtri_exp is sufficiently close to an inverse of log_ndtr.

    We have separate tests for the five intervals (-inf, -10),
    [-10, -2), [-2, -0.14542), [-0.14542, -1e-6), and [-1e-6, 0).
    ndtri_exp(y) is computed in three different ways depending on if y
    is in (-inf, -2), [-2, log(1 - exp(-2))], or [log(1 - exp(-2), 0).
    Each of these intervals is given its own test with two additional tests
    for handling very small values and values very close to zero.
    """

    @pytest.mark.parametrize(
        "test_input", [-1e1, -1e2, -1e10, -1e20, -np.finfo(float).max]
    )
    def test_very_small_arg(self, test_input, uniform_random_points):
        scale = test_input
        points = scale * (0.5 * uniform_random_points + 0.5)
        assert_func_equal(
            log_ndtr_ndtri_exp,
            lambda y: y, points,
            rtol=1e-14,
            nan_ok=True
        )

    @pytest.mark.parametrize(
        "interval,expected_rtol",
        [
            ((-10, -2), 1e-14),
            ((-2, -0.14542), 1e-12),
            ((-0.14542, -1e-6), 1e-10),
            ((-1e-6, 0), 1e-6),
        ],
    )
    def test_in_interval(self, interval, expected_rtol, uniform_random_points):
        left, right = interval
        points = (right - left) * uniform_random_points + left
        assert_func_equal(
            log_ndtr_ndtri_exp,
            lambda y: y, points,
            rtol=expected_rtol,
            nan_ok=True
        )

    def test_extreme(self):
        assert_func_equal(
            log_ndtr_ndtri_exp,
            lambda y: y,
            [-np.finfo(float).max, -np.finfo(float).min],
            rtol=1e-12,
            nan_ok=True
        )

    def test_asymptotes(self):
        assert_equal(ndtri_exp([-np.inf, 0.0]), [-np.inf, np.inf])

    def test_outside_domain(self):
        assert np.isnan(ndtri_exp(1.0))
