import numpy as np
from numpy.testing import (assert_array_equal,
        assert_array_almost_equal_nulp, assert_almost_equal)
from pytest import raises as assert_raises

from scipy.special import gammaln, multigammaln


class TestMultiGammaLn:

    def test1(self):
        # A test of the identity
        #     Gamma_1(a) = Gamma(a)
        np.random.seed(1234)
        a = np.abs(np.random.randn())
        assert_array_equal(multigammaln(a, 1), gammaln(a))

    def test2(self):
        # A test of the identity
        #     Gamma_2(a) = sqrt(pi) * Gamma(a) * Gamma(a - 0.5)
        a = np.array([2.5, 10.0])
        result = multigammaln(a, 2)
        expected = np.log(np.sqrt(np.pi)) + gammaln(a) + gammaln(a - 0.5)
        assert_almost_equal(result, expected)

    def test_bararg(self):
        assert_raises(ValueError, multigammaln, 0.5, 1.2)


def _check_multigammaln_array_result(a, d):
    # Test that the shape of the array returned by multigammaln
    # matches the input shape, and that all the values match
    # the value computed when multigammaln is called with a scalar.
    result = multigammaln(a, d)
    assert_array_equal(a.shape, result.shape)
    a1 = a.ravel()
    result1 = result.ravel()
    for i in range(a.size):
        assert_array_almost_equal_nulp(result1[i], multigammaln(a1[i], d))


def test_multigammaln_array_arg():
    # Check that the array returned by multigammaln has the correct
    # shape and contains the correct values.  The cases have arrays
    # with several differnent shapes.
    # The cases include a regression test for ticket #1849
    # (a = np.array([2.0]), an array with a single element).
    np.random.seed(1234)

    cases = [
        # a, d
        (np.abs(np.random.randn(3, 2)) + 5, 5),
        (np.abs(np.random.randn(1, 2)) + 5, 5),
        (np.arange(10.0, 18.0).reshape(2, 2, 2), 3),
        (np.array([2.0]), 3),
        (np.float64(2.0), 3),
    ]

    for a, d in cases:
        _check_multigammaln_array_result(a, d)

