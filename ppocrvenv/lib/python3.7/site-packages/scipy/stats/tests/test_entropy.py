
import numpy as np
from numpy.testing import assert_equal, assert_allclose
# avoid new uses of the following; prefer assert/np.testing.assert_allclose
from numpy.testing import (assert_, assert_almost_equal,
                           assert_array_almost_equal)

import pytest
from pytest import raises as assert_raises
import scipy.stats as stats


class TestEntropy:
    def test_entropy_positive(self):
        # See ticket #497
        pk = [0.5, 0.2, 0.3]
        qk = [0.1, 0.25, 0.65]
        eself = stats.entropy(pk, pk)
        edouble = stats.entropy(pk, qk)
        assert_(0.0 == eself)
        assert_(edouble >= 0.0)

    def test_entropy_base(self):
        pk = np.ones(16, float)
        S = stats.entropy(pk, base=2.)
        assert_(abs(S - 4.) < 1.e-5)

        qk = np.ones(16, float)
        qk[:8] = 2.
        S = stats.entropy(pk, qk)
        S2 = stats.entropy(pk, qk, base=2.)
        assert_(abs(S/S2 - np.log(2.)) < 1.e-5)

    def test_entropy_zero(self):
        # Test for PR-479
        assert_almost_equal(stats.entropy([0, 1, 2]), 0.63651416829481278,
                            decimal=12)

    def test_entropy_2d(self):
        pk = [[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]]
        qk = [[0.2, 0.1], [0.3, 0.6], [0.5, 0.3]]
        assert_array_almost_equal(stats.entropy(pk, qk),
                                  [0.1933259, 0.18609809])

    def test_entropy_2d_zero(self):
        pk = [[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]]
        qk = [[0.0, 0.1], [0.3, 0.6], [0.5, 0.3]]
        assert_array_almost_equal(stats.entropy(pk, qk),
                                  [np.inf, 0.18609809])

        pk[0][0] = 0.0
        assert_array_almost_equal(stats.entropy(pk, qk),
                                  [0.17403988, 0.18609809])

    def test_entropy_base_2d_nondefault_axis(self):
        pk = [[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]]
        assert_array_almost_equal(stats.entropy(pk, axis=1),
                                  [0.63651417, 0.63651417, 0.66156324])

    def test_entropy_2d_nondefault_axis(self):
        pk = [[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]]
        qk = [[0.2, 0.1], [0.3, 0.6], [0.5, 0.3]]
        assert_array_almost_equal(stats.entropy(pk, qk, axis=1),
                                  [0.231049, 0.231049, 0.127706])

    def test_entropy_raises_value_error(self):
        pk = [[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]]
        qk = [[0.1, 0.2], [0.6, 0.3]]
        assert_raises(ValueError, stats.entropy, pk, qk)

    def test_base_entropy_with_axis_0_is_equal_to_default(self):
        pk = [[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]]
        assert_array_almost_equal(stats.entropy(pk, axis=0),
                                  stats.entropy(pk))

    def test_entropy_with_axis_0_is_equal_to_default(self):
        pk = [[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]]
        qk = [[0.2, 0.1], [0.3, 0.6], [0.5, 0.3]]
        assert_array_almost_equal(stats.entropy(pk, qk, axis=0),
                                  stats.entropy(pk, qk))

    def test_base_entropy_transposed(self):
        pk = np.array([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        assert_array_almost_equal(stats.entropy(pk.T).T,
                                  stats.entropy(pk, axis=1))

    def test_entropy_transposed(self):
        pk = np.array([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        qk = np.array([[0.2, 0.1], [0.3, 0.6], [0.5, 0.3]])
        assert_array_almost_equal(stats.entropy(pk.T, qk.T).T,
                                  stats.entropy(pk, qk, axis=1))

    def test_entropy_broadcasting(self):
        np.random.rand(0)
        x = np.random.rand(3)
        y = np.random.rand(2, 1)
        res = stats.entropy(x, y, axis=-1)
        assert_equal(res[0], stats.entropy(x, y[0]))
        assert_equal(res[1], stats.entropy(x, y[1]))

    def test_entropy_shape_mismatch(self):
        x = np.random.rand(10, 1, 12)
        y = np.random.rand(11, 2)
        message = "shape mismatch: objects cannot be broadcast"
        with pytest.raises(ValueError, match=message):
            stats.entropy(x, y)

    def test_input_validation(self):
        x = np.random.rand(10)
        message = "`base` must be a positive number."
        with pytest.raises(ValueError, match=message):
            stats.entropy(x, base=-2)


class TestDifferentialEntropy:
    """
    Vasicek results are compared with the R package vsgoftest.

    # library(vsgoftest)
    #
    # samp <- c(<values>)
    # entropy.estimate(x = samp, window = <window_length>)

    """

    def test_differential_entropy_vasicek(self):

        random_state = np.random.RandomState(0)
        values = random_state.standard_normal(100)

        entropy = stats.differential_entropy(values, method='vasicek')
        assert_allclose(entropy, 1.342551, rtol=1e-6)

        entropy = stats.differential_entropy(values, window_length=1,
                                             method='vasicek')
        assert_allclose(entropy, 1.122044, rtol=1e-6)

        entropy = stats.differential_entropy(values, window_length=8,
                                             method='vasicek')
        assert_allclose(entropy, 1.349401, rtol=1e-6)

    def test_differential_entropy_vasicek_2d_nondefault_axis(self):
        random_state = np.random.RandomState(0)
        values = random_state.standard_normal((3, 100))

        entropy = stats.differential_entropy(values, axis=1, method='vasicek')
        assert_allclose(
            entropy,
            [1.342551, 1.341826, 1.293775],
            rtol=1e-6,
        )

        entropy = stats.differential_entropy(values, axis=1, window_length=1,
                                             method='vasicek')
        assert_allclose(
            entropy,
            [1.122044, 1.102944, 1.129616],
            rtol=1e-6,
        )

        entropy = stats.differential_entropy(values, axis=1, window_length=8,
                                             method='vasicek')
        assert_allclose(
            entropy,
            [1.349401, 1.338514, 1.292332],
            rtol=1e-6,
        )

    def test_differential_entropy_raises_value_error(self):
        random_state = np.random.RandomState(0)
        values = random_state.standard_normal((3, 100))

        error_str = (
            r"Window length \({window_length}\) must be positive and less "
            r"than half the sample size \({sample_size}\)."
        )

        sample_size = values.shape[1]

        for window_length in {-1, 0, sample_size//2, sample_size}:

            formatted_error_str = error_str.format(
                window_length=window_length,
                sample_size=sample_size,
            )

            with assert_raises(ValueError, match=formatted_error_str):
                stats.differential_entropy(
                    values,
                    window_length=window_length,
                    axis=1,
                )

    def test_base_differential_entropy_with_axis_0_is_equal_to_default(self):
        random_state = np.random.RandomState(0)
        values = random_state.standard_normal((100, 3))

        entropy = stats.differential_entropy(values, axis=0)
        default_entropy = stats.differential_entropy(values)
        assert_allclose(entropy, default_entropy)

    def test_base_differential_entropy_transposed(self):
        random_state = np.random.RandomState(0)
        values = random_state.standard_normal((3, 100))

        assert_allclose(
            stats.differential_entropy(values.T).T,
            stats.differential_entropy(values, axis=1),
        )

    def test_input_validation(self):
        x = np.random.rand(10)

        message = "`base` must be a positive number or `None`."
        with pytest.raises(ValueError, match=message):
            stats.differential_entropy(x, base=-2)

        message = "`method` must be one of..."
        with pytest.raises(ValueError, match=message):
            stats.differential_entropy(x, method='ekki-ekki')

    @pytest.mark.parametrize('method', ['vasicek', 'van es',
                                        'ebrahimi', 'correa'])
    def test_consistency(self, method):
        # test that method is a consistent estimator
        n = 10000 if method == 'correa' else 1000000
        rvs = stats.norm.rvs(size=n, random_state=0)
        expected = stats.norm.entropy()
        res = stats.differential_entropy(rvs, method=method)
        assert_allclose(res, expected, rtol=0.005)

    # values from differential_entropy reference [6], table 1, n=50, m=7
    norm_rmse_std_cases = {  # method: (RMSE, STD)
                           'vasicek': (0.198, 0.109),
                           'van es': (0.212, 0.110),
                           'correa': (0.135, 0.112),
                           'ebrahimi': (0.128, 0.109)
                           }

    @pytest.mark.parametrize('method, expected',
                             list(norm_rmse_std_cases.items()))
    def test_norm_rmse_std(self, method, expected):
        # test that RMSE and standard deviation of estimators matches values
        # given in differential_entropy reference [6]. Incidentally, also
        # tests vectorization.
        reps, n, m = 10000, 50, 7
        rmse_expected, std_expected = expected
        rvs = stats.norm.rvs(size=(reps, n), random_state=0)
        true_entropy = stats.norm.entropy()
        res = stats.differential_entropy(rvs, window_length=m,
                                         method=method, axis=-1)
        assert_allclose(np.sqrt(np.mean((res - true_entropy)**2)),
                        rmse_expected, atol=0.005)
        assert_allclose(np.std(res), std_expected, atol=0.002)

    # values from differential_entropy reference [6], table 2, n=50, m=7
    expon_rmse_std_cases = {  # method: (RMSE, STD)
                            'vasicek': (0.194, 0.148),
                            'van es': (0.179, 0.149),
                            'correa': (0.155, 0.152),
                            'ebrahimi': (0.151, 0.148)
                            }

    @pytest.mark.parametrize('method, expected',
                             list(expon_rmse_std_cases.items()))
    def test_expon_rmse_std(self, method, expected):
        # test that RMSE and standard deviation of estimators matches values
        # given in differential_entropy reference [6]. Incidentally, also
        # tests vectorization.
        reps, n, m = 10000, 50, 7
        rmse_expected, std_expected = expected
        rvs = stats.expon.rvs(size=(reps, n), random_state=0)
        true_entropy = stats.expon.entropy()
        res = stats.differential_entropy(rvs, window_length=m,
                                         method=method, axis=-1)
        assert_allclose(np.sqrt(np.mean((res - true_entropy)**2)),
                        rmse_expected, atol=0.005)
        assert_allclose(np.std(res), std_expected, atol=0.002)

    @pytest.mark.parametrize('n, method', [(8, 'van es'),
                                           (12, 'ebrahimi'),
                                           (1001, 'vasicek')])
    def test_method_auto(self, n, method):
        rvs = stats.norm.rvs(size=(n,), random_state=0)
        res1 = stats.differential_entropy(rvs)
        res2 = stats.differential_entropy(rvs, method=method)
        assert res1 == res2
