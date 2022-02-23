from __future__ import division, print_function, absolute_import

from itertools import product

import numpy as np
import pytest
from numpy.testing import (assert_, assert_equal, assert_allclose,
                           assert_almost_equal)  # avoid new uses
from pytest import raises as assert_raises

import scipy.stats as stats
from scipy.stats import distributions
from scipy.stats._hypotests import (epps_singleton_2samp, cramervonmises,
                                    _cdf_cvm, cramervonmises_2samp,
                                    _pval_cvm_2samp_exact, barnard_exact,
                                    boschloo_exact)
from scipy.stats._mannwhitneyu import mannwhitneyu, _mwu_state
from .common_tests import check_named_results


class TestEppsSingleton:
    def test_statistic_1(self):
        # first example in Goerg & Kaiser, also in original paper of
        # Epps & Singleton. Note: values do not match exactly, the
        # value of the interquartile range varies depending on how
        # quantiles are computed
        x = np.array([-0.35, 2.55, 1.73, 0.73, 0.35,
                      2.69, 0.46, -0.94, -0.37, 12.07])
        y = np.array([-1.15, -0.15, 2.48, 3.25, 3.71,
                      4.29, 5.00, 7.74, 8.38, 8.60])
        w, p = epps_singleton_2samp(x, y)
        assert_almost_equal(w, 15.14, decimal=1)
        assert_almost_equal(p, 0.00442, decimal=3)

    def test_statistic_2(self):
        # second example in Goerg & Kaiser, again not a perfect match
        x = np.array((0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5, 5, 5, 6, 10,
                      10, 10, 10))
        y = np.array((10, 4, 0, 5, 10, 10, 0, 5, 6, 7, 10, 3, 1, 7, 0, 8, 1,
                      5, 8, 10))
        w, p = epps_singleton_2samp(x, y)
        assert_allclose(w, 8.900, atol=0.001)
        assert_almost_equal(p, 0.06364, decimal=3)

    def test_epps_singleton_array_like(self):
        np.random.seed(1234)
        x, y = np.arange(30), np.arange(28)

        w1, p1 = epps_singleton_2samp(list(x), list(y))
        w2, p2 = epps_singleton_2samp(tuple(x), tuple(y))
        w3, p3 = epps_singleton_2samp(x, y)

        assert_(w1 == w2 == w3)
        assert_(p1 == p2 == p3)

    def test_epps_singleton_size(self):
        # raise error if less than 5 elements
        x, y = (1, 2, 3, 4), np.arange(10)
        assert_raises(ValueError, epps_singleton_2samp, x, y)

    def test_epps_singleton_nonfinite(self):
        # raise error if there are non-finite values
        x, y = (1, 2, 3, 4, 5, np.inf), np.arange(10)
        assert_raises(ValueError, epps_singleton_2samp, x, y)
        x, y = np.arange(10), (1, 2, 3, 4, 5, np.nan)
        assert_raises(ValueError, epps_singleton_2samp, x, y)

    def test_epps_singleton_1d_input(self):
        x = np.arange(100).reshape(-1, 1)
        assert_raises(ValueError, epps_singleton_2samp, x, x)

    def test_names(self):
        x, y = np.arange(20), np.arange(30)
        res = epps_singleton_2samp(x, y)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes)


class TestCvm:
    # the expected values of the cdfs are taken from Table 1 in
    # Csorgo / Faraway: The Exact and Asymptotic Distribution of
    # Cramér-von Mises Statistics, 1996.
    def test_cdf_4(self):
        assert_allclose(
                _cdf_cvm([0.02983, 0.04111, 0.12331, 0.94251], 4),
                [0.01, 0.05, 0.5, 0.999],
                atol=1e-4)

    def test_cdf_10(self):
        assert_allclose(
                _cdf_cvm([0.02657, 0.03830, 0.12068, 0.56643], 10),
                [0.01, 0.05, 0.5, 0.975],
                atol=1e-4)

    def test_cdf_1000(self):
        assert_allclose(
                _cdf_cvm([0.02481, 0.03658, 0.11889, 1.16120], 1000),
                [0.01, 0.05, 0.5, 0.999],
                atol=1e-4)

    def test_cdf_inf(self):
        assert_allclose(
                _cdf_cvm([0.02480, 0.03656, 0.11888, 1.16204]),
                [0.01, 0.05, 0.5, 0.999],
                atol=1e-4)

    def test_cdf_support(self):
        # cdf has support on [1/(12*n), n/3]
        assert_equal(_cdf_cvm([1/(12*533), 533/3], 533), [0, 1])
        assert_equal(_cdf_cvm([1/(12*(27 + 1)), (27 + 1)/3], 27), [0, 1])

    def test_cdf_large_n(self):
        # test that asymptotic cdf and cdf for large samples are close
        assert_allclose(
                _cdf_cvm([0.02480, 0.03656, 0.11888, 1.16204, 100], 10000),
                _cdf_cvm([0.02480, 0.03656, 0.11888, 1.16204, 100]),
                atol=1e-4)

    def test_large_x(self):
        # for large values of x and n, the series used to compute the cdf
        # converges slowly.
        # this leads to bug in R package goftest and MAPLE code that is
        # the basis of the implemenation in scipy
        # note: cdf = 1 for x >= 1000/3 and n = 1000
        assert_(0.99999 < _cdf_cvm(333.3, 1000) < 1.0)
        assert_(0.99999 < _cdf_cvm(333.3) < 1.0)

    def test_low_p(self):
        # _cdf_cvm can return values larger than 1. In that case, we just
        # return a p-value of zero.
        n = 12
        res = cramervonmises(np.ones(n)*0.8, 'norm')
        assert_(_cdf_cvm(res.statistic, n) > 1.0)
        assert_equal(res.pvalue, 0)

    def test_invalid_input(self):
        x = np.arange(10).reshape((2, 5))
        assert_raises(ValueError, cramervonmises, x, "norm")
        assert_raises(ValueError, cramervonmises, [1.5], "norm")
        assert_raises(ValueError, cramervonmises, (), "norm")

    def test_values_R(self):
        # compared against R package goftest, version 1.1.1
        # goftest::cvm.test(c(-1.7, 2, 0, 1.3, 4, 0.1, 0.6), "pnorm")
        res = cramervonmises([-1.7, 2, 0, 1.3, 4, 0.1, 0.6], "norm")
        assert_allclose(res.statistic, 0.288156, atol=1e-6)
        assert_allclose(res.pvalue, 0.1453465, atol=1e-6)

        # goftest::cvm.test(c(-1.7, 2, 0, 1.3, 4, 0.1, 0.6),
        #                   "pnorm", mean = 3, sd = 1.5)
        res = cramervonmises([-1.7, 2, 0, 1.3, 4, 0.1, 0.6], "norm", (3, 1.5))
        assert_allclose(res.statistic, 0.9426685, atol=1e-6)
        assert_allclose(res.pvalue, 0.002026417, atol=1e-6)

        # goftest::cvm.test(c(1, 2, 5, 1.4, 0.14, 11, 13, 0.9, 7.5), "pexp")
        res = cramervonmises([1, 2, 5, 1.4, 0.14, 11, 13, 0.9, 7.5], "expon")
        assert_allclose(res.statistic, 0.8421854, atol=1e-6)
        assert_allclose(res.pvalue, 0.004433406, atol=1e-6)

    def test_callable_cdf(self):
        x, args = np.arange(5), (1.4, 0.7)
        r1 = cramervonmises(x, distributions.expon.cdf)
        r2 = cramervonmises(x, "expon")
        assert_equal((r1.statistic, r1.pvalue), (r2.statistic, r2.pvalue))

        r1 = cramervonmises(x, distributions.beta.cdf, args)
        r2 = cramervonmises(x, "beta", args)
        assert_equal((r1.statistic, r1.pvalue), (r2.statistic, r2.pvalue))


class TestMannWhitneyU:
    # All magic numbers are from R wilcox.test unless otherwise specied
    # https://rdrr.io/r/stats/wilcox.test.html

    # --- Test Input Validation ---

    def test_input_validation(self):
        x = np.array([1, 2])  # generic, valid inputs
        y = np.array([3, 4])
        with assert_raises(ValueError, match="`x` and `y` must be of nonzero"):
            mannwhitneyu([], y)
        with assert_raises(ValueError, match="`x` and `y` must be of nonzero"):
            mannwhitneyu(x, [])
        with assert_raises(ValueError, match="`x` and `y` must not contain"):
            mannwhitneyu([np.nan, 2], y)
        with assert_raises(ValueError, match="`use_continuity` must be one"):
            mannwhitneyu(x, y, use_continuity='ekki')
        with assert_raises(ValueError, match="`alternative` must be one of"):
            mannwhitneyu(x, y, alternative='ekki')
        with assert_raises(ValueError, match="`axis` must be an integer"):
            mannwhitneyu(x, y, axis=1.5)
        with assert_raises(ValueError, match="`method` must be one of"):
            mannwhitneyu(x, y, method='ekki')

    def test_auto(self):
        # Test that default method ('auto') chooses intended method

        np.random.seed(1)
        n = 8  # threshold to switch from exact to asymptotic

        # both inputs are smaller than threshold; should use exact
        x = np.random.rand(n-1)
        y = np.random.rand(n-1)
        auto = mannwhitneyu(x, y)
        asymptotic = mannwhitneyu(x, y, method='asymptotic')
        exact = mannwhitneyu(x, y, method='exact')
        assert auto.pvalue == exact.pvalue
        assert auto.pvalue != asymptotic.pvalue

        # one input is smaller than threshold; should use exact
        x = np.random.rand(n-1)
        y = np.random.rand(n+1)
        auto = mannwhitneyu(x, y)
        asymptotic = mannwhitneyu(x, y, method='asymptotic')
        exact = mannwhitneyu(x, y, method='exact')
        assert auto.pvalue == exact.pvalue
        assert auto.pvalue != asymptotic.pvalue

        # other input is smaller than threshold; should use exact
        auto = mannwhitneyu(y, x)
        asymptotic = mannwhitneyu(x, y, method='asymptotic')
        exact = mannwhitneyu(x, y, method='exact')
        assert auto.pvalue == exact.pvalue
        assert auto.pvalue != asymptotic.pvalue

        # both inputs are larger than threshold; should use asymptotic
        x = np.random.rand(n+1)
        y = np.random.rand(n+1)
        auto = mannwhitneyu(x, y)
        asymptotic = mannwhitneyu(x, y, method='asymptotic')
        exact = mannwhitneyu(x, y, method='exact')
        assert auto.pvalue != exact.pvalue
        assert auto.pvalue == asymptotic.pvalue

        # both inputs are smaller than threshold, but there is a tie
        # should use asymptotic
        x = np.random.rand(n-1)
        y = np.random.rand(n-1)
        y[3] = x[3]
        auto = mannwhitneyu(x, y)
        asymptotic = mannwhitneyu(x, y, method='asymptotic')
        exact = mannwhitneyu(x, y, method='exact')
        assert auto.pvalue != exact.pvalue
        assert auto.pvalue == asymptotic.pvalue

    # --- Test Basic Functionality ---

    x = [210.052110, 110.190630, 307.918612]
    y = [436.08811482466416, 416.37397329768191, 179.96975939463582,
         197.8118754228619, 34.038757281225756, 138.54220550921517,
         128.7769351470246, 265.92721427951852, 275.6617533155341,
         592.34083395416258, 448.73177590617018, 300.61495185038905,
         187.97508449019588]

    # This test was written for mann_whitney_u in gh-4933.
    # Originally, the p-values for alternatives were swapped;
    # this has been corrected and the tests have been refactored for
    # compactness, but otherwise the tests are unchanged.
    # R code for comparison, e.g.:
    # options(digits = 16)
    # x = c(210.052110, 110.190630, 307.918612)
    # y = c(436.08811482466416, 416.37397329768191, 179.96975939463582,
    #       197.8118754228619, 34.038757281225756, 138.54220550921517,
    #       128.7769351470246, 265.92721427951852, 275.6617533155341,
    #       592.34083395416258, 448.73177590617018, 300.61495185038905,
    #       187.97508449019588)
    # wilcox.test(x, y, alternative="g", exact=TRUE)
    cases_basic = [[{"alternative": 'two-sided', "method": "asymptotic"},
                    (16, 0.6865041817876)],
                   [{"alternative": 'less', "method": "asymptotic"},
                    (16, 0.3432520908938)],
                   [{"alternative": 'greater', "method": "asymptotic"},
                    (16, 0.7047591913255)],
                   [{"alternative": 'two-sided', "method": "exact"},
                    (16, 0.7035714285714)],
                   [{"alternative": 'less', "method": "exact"},
                    (16, 0.3517857142857)],
                   [{"alternative": 'greater', "method": "exact"},
                    (16, 0.6946428571429)]]

    @pytest.mark.parametrize(("kwds", "expected"), cases_basic)
    def test_basic(self, kwds, expected):
        res = mannwhitneyu(self.x, self.y, **kwds)
        assert_allclose(res, expected)

    cases_continuity = [[{"alternative": 'two-sided', "use_continuity": True},
                         (23, 0.6865041817876)],
                        [{"alternative": 'less', "use_continuity": True},
                         (23, 0.7047591913255)],
                        [{"alternative": 'greater', "use_continuity": True},
                         (23, 0.3432520908938)],
                        [{"alternative": 'two-sided', "use_continuity": False},
                         (23, 0.6377328900502)],
                        [{"alternative": 'less', "use_continuity": False},
                         (23, 0.6811335549749)],
                        [{"alternative": 'greater', "use_continuity": False},
                         (23, 0.3188664450251)]]

    @pytest.mark.parametrize(("kwds", "expected"), cases_continuity)
    def test_continuity(self, kwds, expected):
        # When x and y are interchanged, less and greater p-values should
        # swap (compare to above). This wouldn't happen if the continuity
        # correction were applied in the wrong direction. Note that less and
        # greater p-values do not sum to 1 when continuity correction is on,
        # which is what we'd expect. Also check that results match R when
        # continuity correction is turned off.
        # Note that method='asymptotic' -> exact=FALSE
        # and use_continuity=False -> correct=FALSE, e.g.:
        # wilcox.test(x, y, alternative="t", exact=FALSE, correct=FALSE)
        res = mannwhitneyu(self.y, self.x, method='asymptotic', **kwds)
        assert_allclose(res, expected)

    def test_tie_correct(self):
        # Test tie correction against R's wilcox.test
        # options(digits = 16)
        # x = c(1, 2, 3, 4)
        # y = c(1, 2, 3, 4, 5)
        # wilcox.test(x, y, exact=FALSE)
        x = [1, 2, 3, 4]
        y0 = np.array([1, 2, 3, 4, 5])
        dy = np.array([0, 1, 0, 1, 0])*0.01
        dy2 = np.array([0, 0, 1, 0, 0])*0.01
        y = [y0-0.01, y0-dy, y0-dy2, y0, y0+dy2, y0+dy, y0+0.01]
        res = mannwhitneyu(x, y, axis=-1, method="asymptotic")
        U_expected = [10, 9, 8.5, 8, 7.5, 7, 6]
        p_expected = [1, 0.9017048037317, 0.804080657472, 0.7086240584439,
                      0.6197963884941, 0.5368784563079, 0.3912672792826]
        assert_equal(res.statistic, U_expected)
        assert_allclose(res.pvalue, p_expected)

    # --- Test Exact Distribution of U ---

    # These are tabulated values of the CDF of the exact distribution of
    # the test statistic from pg 52 of reference [1] (Mann-Whitney Original)
    pn3 = {1: [0.25, 0.5, 0.75], 2: [0.1, 0.2, 0.4, 0.6],
           3: [0.05, .1, 0.2, 0.35, 0.5, 0.65]}
    pn4 = {1: [0.2, 0.4, 0.6], 2: [0.067, 0.133, 0.267, 0.4, 0.6],
           3: [0.028, 0.057, 0.114, 0.2, .314, 0.429, 0.571],
           4: [0.014, 0.029, 0.057, 0.1, 0.171, 0.243, 0.343, 0.443, 0.557]}
    pm5 = {1: [0.167, 0.333, 0.5, 0.667],
           2: [0.047, 0.095, 0.19, 0.286, 0.429, 0.571],
           3: [0.018, 0.036, 0.071, 0.125, 0.196, 0.286, 0.393, 0.5, 0.607],
           4: [0.008, 0.016, 0.032, 0.056, 0.095, 0.143,
               0.206, 0.278, 0.365, 0.452, 0.548],
           5: [0.004, 0.008, 0.016, 0.028, 0.048, 0.075, 0.111,
               0.155, 0.21, 0.274, 0.345, .421, 0.5, 0.579]}
    pm6 = {1: [0.143, 0.286, 0.428, 0.571],
           2: [0.036, 0.071, 0.143, 0.214, 0.321, 0.429, 0.571],
           3: [0.012, 0.024, 0.048, 0.083, 0.131,
               0.19, 0.274, 0.357, 0.452, 0.548],
           4: [0.005, 0.01, 0.019, 0.033, 0.057, 0.086, 0.129,
               0.176, 0.238, 0.305, 0.381, 0.457, 0.543],  # the last element
           # of the previous list, 0.543, has been modified from 0.545;
           # I assume it was a typo
           5: [0.002, 0.004, 0.009, 0.015, 0.026, 0.041, 0.063, 0.089,
               0.123, 0.165, 0.214, 0.268, 0.331, 0.396, 0.465, 0.535],
           6: [0.001, 0.002, 0.004, 0.008, 0.013, 0.021, 0.032, 0.047,
               0.066, 0.09, 0.12, 0.155, 0.197, 0.242, 0.294, 0.350,
               0.409, 0.469, 0.531]}

    def test_exact_distribution(self):
        # I considered parametrize. I decided against it.
        p_tables = {3: self.pn3, 4: self.pn4, 5: self.pm5, 6: self.pm6}
        for n, table in p_tables.items():
            for m, p in table.items():
                # check p-value against table
                u = np.arange(0, len(p))
                assert_allclose(_mwu_state.cdf(k=u, m=m, n=n), p, atol=1e-3)

                # check identity CDF + SF - PMF = 1
                # ( In this implementation, SF(U) includes PMF(U) )
                u2 = np.arange(0, m*n+1)
                assert_allclose(_mwu_state.cdf(k=u2, m=m, n=n)
                                + _mwu_state.sf(k=u2, m=m, n=n)
                                - _mwu_state.pmf(k=u2, m=m, n=n), 1)

                # check symmetry about mean of U, i.e. pmf(U) = pmf(m*n-U)
                pmf = _mwu_state.pmf(k=u2, m=m, n=n)
                assert_allclose(pmf, pmf[::-1])

                # check symmetry w.r.t. interchange of m, n
                pmf2 = _mwu_state.pmf(k=u2, m=n, n=m)
                assert_allclose(pmf, pmf2)

    def test_asymptotic_behavior(self):
        np.random.seed(0)

        # for small samples, the asymptotic test is not very accurate
        x = np.random.rand(5)
        y = np.random.rand(5)
        res1 = mannwhitneyu(x, y, method="exact")
        res2 = mannwhitneyu(x, y, method="asymptotic")
        assert res1.statistic == res2.statistic
        assert np.abs(res1.pvalue - res2.pvalue) > 1e-2

        # for large samples, they agree reasonably well
        x = np.random.rand(40)
        y = np.random.rand(40)
        res1 = mannwhitneyu(x, y, method="exact")
        res2 = mannwhitneyu(x, y, method="asymptotic")
        assert res1.statistic == res2.statistic
        assert np.abs(res1.pvalue - res2.pvalue) < 1e-3

    # --- Test Corner Cases ---

    def test_exact_U_equals_mean(self):
        # Test U == m*n/2 with exact method
        # Without special treatment, two-sided p-value > 1 because both
        # one-sided p-values are > 0.5
        res_l = mannwhitneyu([1, 2, 3], [1.5, 2.5], alternative="less",
                             method="exact")
        res_g = mannwhitneyu([1, 2, 3], [1.5, 2.5], alternative="greater",
                             method="exact")
        assert_equal(res_l.pvalue, res_g.pvalue)
        assert res_l.pvalue > 0.5

        res = mannwhitneyu([1, 2, 3], [1.5, 2.5], alternative="two-sided",
                           method="exact")
        assert_equal(res, (3, 1))
        # U == m*n/2 for asymptotic case tested in test_gh_2118
        # The reason it's tricky for the asymptotic test has to do with
        # continuity correction.

    cases_scalar = [[{"alternative": 'two-sided', "method": "asymptotic"},
                     (0, 1)],
                    [{"alternative": 'less', "method": "asymptotic"},
                     (0, 0.5)],
                    [{"alternative": 'greater', "method": "asymptotic"},
                     (0, 0.977249868052)],
                    [{"alternative": 'two-sided', "method": "exact"}, (0, 1)],
                    [{"alternative": 'less', "method": "exact"}, (0, 0.5)],
                    [{"alternative": 'greater', "method": "exact"}, (0, 1)]]

    @pytest.mark.parametrize(("kwds", "result"), cases_scalar)
    def test_scalar_data(self, kwds, result):
        # just making sure scalars work
        assert_allclose(mannwhitneyu(1, 2, **kwds), result)

    def test_equal_scalar_data(self):
        # when two scalars are equal, there is an -0.5/0 in the asymptotic
        # approximation. R gives pvalue=1.0 for alternatives 'less' and
        # 'greater' but NA for 'two-sided'. I don't see why, so I don't
        # see a need for a special case to match that behavior.
        assert_equal(mannwhitneyu(1, 1, method="exact"), (0.5, 1))
        assert_equal(mannwhitneyu(1, 1, method="asymptotic"), (0.5, 1))

        # without continuity correction, this becomes 0/0, which really
        # is undefined
        assert_equal(mannwhitneyu(1, 1, method="asymptotic",
                                  use_continuity=False), (0.5, np.nan))

    # --- Test Enhancements / Bug Reports ---

    @pytest.mark.parametrize("method", ["asymptotic", "exact"])
    def test_gh_12837_11113(self, method):
        # Test that behavior for broadcastable nd arrays is appropriate:
        # output shape is correct and all values are equal to when the test
        # is performed on one pair of samples at a time.
        # Tests that gh-12837 and gh-11113 (requests for n-d input)
        # are resolved
        np.random.seed(0)

        # arrays are broadcastable except for axis = -3
        axis = -3
        m, n = 7, 10  # sample sizes
        x = np.random.rand(m, 3, 8)
        y = np.random.rand(6, n, 1, 8) + 0.1
        res = mannwhitneyu(x, y, method=method, axis=axis)

        shape = (6, 3, 8)  # appropriate shape of outputs, given inputs
        assert(res.pvalue.shape == shape)
        assert(res.statistic.shape == shape)

        # move axis of test to end for simplicity
        x, y = np.moveaxis(x, axis, -1), np.moveaxis(y, axis, -1)

        x = x[None, ...]  # give x a zeroth dimension
        assert(x.ndim == y.ndim)

        x = np.broadcast_to(x, shape + (m,))
        y = np.broadcast_to(y, shape + (n,))
        assert(x.shape[:-1] == shape)
        assert(y.shape[:-1] == shape)

        # loop over pairs of samples
        statistics = np.zeros(shape)
        pvalues = np.zeros(shape)
        for indices in product(*[range(i) for i in shape]):
            xi = x[indices]
            yi = y[indices]
            temp = mannwhitneyu(xi, yi, method=method)
            statistics[indices] = temp.statistic
            pvalues[indices] = temp.pvalue

        np.testing.assert_equal(res.pvalue, pvalues)
        np.testing.assert_equal(res.statistic, statistics)

    def test_gh_11355(self):
        # Test for correct behavior with NaN/Inf in input
        x = [1, 2, 3, 4]
        y = [3, 6, 7, 8, 9, 3, 2, 1, 4, 4, 5]
        res1 = mannwhitneyu(x, y)

        # Inf is not a problem. This is a rank test, and it's the largest value
        y[4] = np.inf
        res2 = mannwhitneyu(x, y)

        assert_equal(res1.statistic, res2.statistic)
        assert_equal(res1.pvalue, res2.pvalue)

        # NaNs should raise an error. No nan_policy for now.
        y[4] = np.nan
        with assert_raises(ValueError, match="`x` and `y` must not contain"):
            mannwhitneyu(x, y)

    cases_11355 = [([1, 2, 3, 4],
                    [3, 6, 7, 8, np.inf, 3, 2, 1, 4, 4, 5],
                    10, 0.1297704873477),
                   ([1, 2, 3, 4],
                    [3, 6, 7, 8, np.inf, np.inf, 2, 1, 4, 4, 5],
                    8.5, 0.08735617507695),
                   ([1, 2, np.inf, 4],
                    [3, 6, 7, 8, np.inf, 3, 2, 1, 4, 4, 5],
                    17.5, 0.5988856695752),
                   ([1, 2, np.inf, 4],
                    [3, 6, 7, 8, np.inf, np.inf, 2, 1, 4, 4, 5],
                    16, 0.4687165824462),
                   ([1, np.inf, np.inf, 4],
                    [3, 6, 7, 8, np.inf, np.inf, 2, 1, 4, 4, 5],
                    24.5, 0.7912517950119)]

    @pytest.mark.parametrize(("x", "y", "statistic", "pvalue"), cases_11355)
    def test_gh_11355b(self, x, y, statistic, pvalue):
        # Test for correct behavior with NaN/Inf in input
        res = mannwhitneyu(x, y, method='asymptotic')
        assert_allclose(res.statistic, statistic, atol=1e-12)
        assert_allclose(res.pvalue, pvalue, atol=1e-12)

    cases_9184 = [[True, "less", "asymptotic", 0.900775348204],
                  [True, "greater", "asymptotic", 0.1223118025635],
                  [True, "two-sided", "asymptotic", 0.244623605127],
                  [False, "less", "asymptotic", 0.8896643190401],
                  [False, "greater", "asymptotic", 0.1103356809599],
                  [False, "two-sided", "asymptotic", 0.2206713619198],
                  [True, "less", "exact", 0.8967698967699],
                  [True, "greater", "exact", 0.1272061272061],
                  [True, "two-sided", "exact", 0.2544122544123]]

    @pytest.mark.parametrize(("use_continuity", "alternative",
                              "method", "pvalue_exp"), cases_9184)
    def test_gh_9184(self, use_continuity, alternative, method, pvalue_exp):
        # gh-9184 might be considered a doc-only bug. Please see the
        # documentation to confirm that mannwhitneyu correctly notes
        # that the output statistic is that of the first sample (x). In any
        # case, check the case provided there against output from R.
        # R code:
        # options(digits=16)
        # x <- c(0.80, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46)
        # y <- c(1.15, 0.88, 0.90, 0.74, 1.21)
        # wilcox.test(x, y, alternative = "less", exact = FALSE)
        # wilcox.test(x, y, alternative = "greater", exact = FALSE)
        # wilcox.test(x, y, alternative = "two.sided", exact = FALSE)
        # wilcox.test(x, y, alternative = "less", exact = FALSE,
        #             correct=FALSE)
        # wilcox.test(x, y, alternative = "greater", exact = FALSE,
        #             correct=FALSE)
        # wilcox.test(x, y, alternative = "two.sided", exact = FALSE,
        #             correct=FALSE)
        # wilcox.test(x, y, alternative = "less", exact = TRUE)
        # wilcox.test(x, y, alternative = "greater", exact = TRUE)
        # wilcox.test(x, y, alternative = "two.sided", exact = TRUE)
        statistic_exp = 35
        x = (0.80, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46)
        y = (1.15, 0.88, 0.90, 0.74, 1.21)
        res = mannwhitneyu(x, y, use_continuity=use_continuity,
                           alternative=alternative, method=method)
        assert_equal(res.statistic, statistic_exp)
        assert_allclose(res.pvalue, pvalue_exp)

    def test_gh_6897(self):
        # Test for correct behavior with empty input
        with assert_raises(ValueError, match="`x` and `y` must be of nonzero"):
            mannwhitneyu([], [])

    def test_gh_4067(self):
        # Test for correct behavior with all NaN input
        a = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        b = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        with assert_raises(ValueError, match="`x` and `y` must not contain"):
            mannwhitneyu(a, b)

    # All cases checked against R wilcox.test, e.g.
    # options(digits=16)
    # x = c(1, 2, 3)
    # y = c(1.5, 2.5)
    # wilcox.test(x, y, exact=FALSE, alternative='less')

    cases_2118 = [[[1, 2, 3], [1.5, 2.5], "greater", (3, 0.6135850036578)],
                  [[1, 2, 3], [1.5, 2.5], "less", (3, 0.6135850036578)],
                  [[1, 2, 3], [1.5, 2.5], "two-sided", (3, 1.0)],
                  [[1, 2, 3], [2], "greater", (1.5, 0.681324055883)],
                  [[1, 2, 3], [2], "less", (1.5, 0.681324055883)],
                  [[1, 2, 3], [2], "two-sided", (1.5, 1)],
                  [[1, 2], [1, 2], "greater", (2, 0.667497228949)],
                  [[1, 2], [1, 2], "less", (2, 0.667497228949)],
                  [[1, 2], [1, 2], "two-sided", (2, 1)]]

    @pytest.mark.parametrize(["x", "y", "alternative", "expected"], cases_2118)
    def test_gh_2118(self, x, y, alternative, expected):
        # test cases in which U == m*n/2 when method is asymptotic
        # applying continuity correction could result in p-value > 1
        res = mannwhitneyu(x, y, use_continuity=True, alternative=alternative,
                           method="asymptotic")
        assert_allclose(res, expected, rtol=1e-12)


class TestSomersD:

    def test_like_kendalltau(self):
        # All tests correspond with one in test_stats.py `test_kendalltau`

        # case without ties, con-dis equal zero
        x = [5, 2, 1, 3, 6, 4, 7, 8]
        y = [5, 2, 6, 3, 1, 8, 7, 4]
        # Cross-check with result from SAS FREQ:
        expected = (0.000000000000000, 1.000000000000000)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)

        # case without ties, con-dis equal zero
        x = [0, 5, 2, 1, 3, 6, 4, 7, 8]
        y = [5, 2, 0, 6, 3, 1, 8, 7, 4]
        # Cross-check with result from SAS FREQ:
        expected = (0.000000000000000, 1.000000000000000)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)

        # case without ties, con-dis close to zero
        x = [5, 2, 1, 3, 6, 4, 7]
        y = [5, 2, 6, 3, 1, 7, 4]
        # Cross-check with result from SAS FREQ:
        expected = (-0.142857142857140, 0.630326953157670)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)

        # simple case without ties
        x = np.arange(10)
        y = np.arange(10)
        # Cross-check with result from SAS FREQ:
        # SAS p value is not provided.
        expected = (1.000000000000000, 0)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)

        # swap a couple values and a couple more
        x = np.arange(10)
        y = np.array([0, 2, 1, 3, 4, 6, 5, 7, 8, 9])
        # Cross-check with result from SAS FREQ:
        expected = (0.911111111111110, 0.000000000000000)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)

        # same in opposite direction
        x = np.arange(10)
        y = np.arange(10)[::-1]
        # Cross-check with result from SAS FREQ:
        # SAS p value is not provided.
        expected = (-1.000000000000000, 0)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)

        # swap a couple values and a couple more
        x = np.arange(10)
        y = np.array([9, 7, 8, 6, 5, 3, 4, 2, 1, 0])
        # Cross-check with result from SAS FREQ:
        expected = (-0.9111111111111111, 0.000000000000000)
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)

        # with some ties
        x1 = [12, 2, 1, 12, 2]
        x2 = [1, 4, 7, 1, 0]
        # Cross-check with result from SAS FREQ:
        expected = (-0.500000000000000, 0.304901788178780)
        res = stats.somersd(x1, x2)
        assert_allclose(res.statistic, expected[0], atol=1e-15)
        assert_allclose(res.pvalue, expected[1], atol=1e-15)

        # with only ties in one or both inputs
        # SAS will not produce an output for these:
        # NOTE: No statistics are computed for x * y because x has fewer
        # than 2 nonmissing levels.
        # WARNING: No OUTPUT data set is produced for this table because a
        # row or column variable has fewer than 2 nonmissing levels and no
        # statistics are computed.

        res = stats.somersd([2, 2, 2], [2, 2, 2])
        assert_allclose(res.statistic, np.nan)
        assert_allclose(res.pvalue, np.nan)

        res = stats.somersd([2, 0, 2], [2, 2, 2])
        assert_allclose(res.statistic, np.nan)
        assert_allclose(res.pvalue, np.nan)

        res = stats.somersd([2, 2, 2], [2, 0, 2])
        assert_allclose(res.statistic, np.nan)
        assert_allclose(res.pvalue, np.nan)

        res = stats.somersd([0], [0])
        assert_allclose(res.statistic, np.nan)
        assert_allclose(res.pvalue, np.nan)

        # empty arrays provided as input
        res = stats.somersd([], [])
        assert_allclose(res.statistic, np.nan)
        assert_allclose(res.pvalue, np.nan)

        # test unequal length inputs
        x = np.arange(10.)
        y = np.arange(20.)
        assert_raises(ValueError, stats.somersd, x, y)

    def test_asymmetry(self):
        # test that somersd is asymmetric w.r.t. input order and that
        # convention is as described: first input is row variable & independent
        # data is from Wikipedia:
        # https://en.wikipedia.org/wiki/Somers%27_D
        # but currently that example contradicts itself - it says X is
        # independent yet take D_XY

        x = [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 1, 2,
             2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
        y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        # Cross-check with result from SAS FREQ:
        d_cr = 0.272727272727270
        d_rc = 0.342857142857140
        p = 0.092891940883700  # same p-value for either direction
        res = stats.somersd(x, y)
        assert_allclose(res.statistic, d_cr, atol=1e-15)
        assert_allclose(res.pvalue, p, atol=1e-4)
        assert_equal(res.table.shape, (3, 2))
        res = stats.somersd(y, x)
        assert_allclose(res.statistic, d_rc, atol=1e-15)
        assert_allclose(res.pvalue, p, atol=1e-15)
        assert_equal(res.table.shape, (2, 3))

    def test_somers_original(self):
        # test against Somers' original paper [1]

        # Table 5A
        # Somers' convention was column IV
        table = np.array([[8, 2], [6, 5], [3, 4], [1, 3], [2, 3]])
        # Our convention (and that of SAS FREQ) is row IV
        table = table.T
        dyx = 129/340
        assert_allclose(stats.somersd(table).statistic, dyx)

        # table 7A - d_yx = 1
        table = np.array([[25, 0], [85, 0], [0, 30]])
        dxy, dyx = 3300/5425, 3300/3300
        assert_allclose(stats.somersd(table).statistic, dxy)
        assert_allclose(stats.somersd(table.T).statistic, dyx)

        # table 7B - d_yx < 0
        table = np.array([[25, 0], [0, 30], [85, 0]])
        dyx = -1800/3300
        assert_allclose(stats.somersd(table.T).statistic, dyx)

    def test_contingency_table_with_zero_rows_cols(self):
        # test that zero rows/cols in contingency table don't affect result

        N = 100
        shape = 4, 6
        size = np.prod(shape)

        np.random.seed(0)
        s = stats.multinomial.rvs(N, p=np.ones(size)/size).reshape(shape)
        res = stats.somersd(s)

        s2 = np.insert(s, 2, np.zeros(shape[1]), axis=0)
        res2 = stats.somersd(s2)

        s3 = np.insert(s, 2, np.zeros(shape[0]), axis=1)
        res3 = stats.somersd(s3)

        s4 = np.insert(s2, 2, np.zeros(shape[0]+1), axis=1)
        res4 = stats.somersd(s4)

        # Cross-check with result from SAS FREQ:
        assert_allclose(res.statistic, -0.116981132075470, atol=1e-15)
        assert_allclose(res.statistic, res2.statistic)
        assert_allclose(res.statistic, res3.statistic)
        assert_allclose(res.statistic, res4.statistic)

        assert_allclose(res.pvalue, 0.156376448188150, atol=1e-15)
        assert_allclose(res.pvalue, res2.pvalue)
        assert_allclose(res.pvalue, res3.pvalue)
        assert_allclose(res.pvalue, res4.pvalue)

    def test_invalid_contingency_tables(self):
        N = 100
        shape = 4, 6
        size = np.prod(shape)

        np.random.seed(0)
        # start with a valid contingency table
        s = stats.multinomial.rvs(N, p=np.ones(size)/size).reshape(shape)

        s5 = s - 2
        message = "All elements of the contingency table must be non-negative"
        with assert_raises(ValueError, match=message):
            stats.somersd(s5)

        s6 = s + 0.01
        message = "All elements of the contingency table must be integer"
        with assert_raises(ValueError, match=message):
            stats.somersd(s6)

        message = ("At least two elements of the contingency "
                   "table must be nonzero.")
        with assert_raises(ValueError, match=message):
            stats.somersd([[]])

        with assert_raises(ValueError, match=message):
            stats.somersd([[1]])

        s7 = np.zeros((3, 3))
        with assert_raises(ValueError, match=message):
            stats.somersd(s7)

        s7[0, 1] = 1
        with assert_raises(ValueError, match=message):
            stats.somersd(s7)

    def test_only_ranks_matter(self):
        # only ranks of input data should matter
        x = [1, 2, 3]
        x2 = [-1, 2.1, np.inf]
        y = [3, 2, 1]
        y2 = [0, -0.5, -np.inf]
        res = stats.somersd(x, y)
        res2 = stats.somersd(x2, y2)
        assert_equal(res.statistic, res2.statistic)
        assert_equal(res.pvalue, res2.pvalue)

    def test_contingency_table_return(self):
        # check that contingency table is returned
        x = np.arange(10)
        y = np.arange(10)
        res = stats.somersd(x, y)
        assert_equal(res.table, np.eye(10))


class TestBarnardExact:
    """Some tests to show that barnard_exact() works correctly."""

    @pytest.mark.parametrize(
        "input_sample,expected",
        [
            ([[43, 40], [10, 39]], (3.555406779643, 0.000362832367)),
            ([[100, 2], [1000, 5]], (-1.776382925679, 0.135126970878)),
            ([[2, 7], [8, 2]], (-2.518474945157, 0.019210815430)),
            ([[5, 1], [10, 10]], (1.449486150679, 0.156277546306)),
            ([[5, 15], [20, 20]], (-1.851640199545, 0.066363501421)),
            ([[5, 16], [20, 25]], (-1.609639949352, 0.116984852192)),
            ([[10, 5], [10, 1]], (-1.449486150679, 0.177536588915)),
            ([[5, 0], [1, 4]], (2.581988897472, 0.013671875000)),
            ([[0, 1], [3, 2]], (-1.095445115010, 0.509667991877)),
            ([[0, 2], [6, 4]], (-1.549193338483, 0.197019618792)),
            ([[2, 7], [8, 2]], (-2.518474945157, 0.019210815430)),
        ],
    )
    def test_precise(self, input_sample, expected):
        """The expected values have been generated by R, using a resolution
        for the nuisance parameter of 1e-6 :
        ```R
        library(Barnard)
        options(digits=10)
        barnard.test(43, 40, 10, 39, dp=1e-6, pooled=TRUE)
        ```
        """
        res = barnard_exact(input_sample)
        statistic, pvalue = res.statistic, res.pvalue
        assert_allclose([statistic, pvalue], expected)

    @pytest.mark.parametrize(
        "input_sample,expected",
        [
            ([[43, 40], [10, 39]], (3.920362887717, 0.000289470662)),
            ([[100, 2], [1000, 5]], (-1.139432816087, 0.950272080594)),
            ([[2, 7], [8, 2]], (-3.079373904042, 0.020172119141)),
            ([[5, 1], [10, 10]], (1.622375939458, 0.150599922226)),
            ([[5, 15], [20, 20]], (-1.974771239528, 0.063038448651)),
            ([[5, 16], [20, 25]], (-1.722122973346, 0.133329494287)),
            ([[10, 5], [10, 1]], (-1.765469659009, 0.250566655215)),
            ([[5, 0], [1, 4]], (5.477225575052, 0.007812500000)),
            ([[0, 1], [3, 2]], (-1.224744871392, 0.509667991877)),
            ([[0, 2], [6, 4]], (-1.732050807569, 0.197019618792)),
            ([[2, 7], [8, 2]], (-3.079373904042, 0.020172119141)),
        ],
    )
    def test_pooled_param(self, input_sample, expected):
        """The expected values have been generated by R, using a resolution
        for the nuisance parameter of 1e-6 :
        ```R
        library(Barnard)
        options(digits=10)
        barnard.test(43, 40, 10, 39, dp=1e-6, pooled=FALSE)
        ```
        """
        res = barnard_exact(input_sample, pooled=False)
        statistic, pvalue = res.statistic, res.pvalue
        assert_allclose([statistic, pvalue], expected)

    def test_raises(self):
        # test we raise an error for wrong input number of nuisances.
        error_msg = (
            "Number of points `n` must be strictly positive, found 0"
        )
        with assert_raises(ValueError, match=error_msg):
            barnard_exact([[1, 2], [3, 4]], n=0)

        # test we raise an error for wrong shape of input.
        error_msg = "The input `table` must be of shape \\(2, 2\\)."
        with assert_raises(ValueError, match=error_msg):
            barnard_exact(np.arange(6).reshape(2, 3))

        # Test all values must be positives
        error_msg = "All values in `table` must be nonnegative."
        with assert_raises(ValueError, match=error_msg):
            barnard_exact([[-1, 2], [3, 4]])

        # Test value error on wrong alternative param
        error_msg = (
            "`alternative` should be one of {'two-sided', 'less', 'greater'},"
            " found .*"
        )
        with assert_raises(ValueError, match=error_msg):
            barnard_exact([[1, 2], [3, 4]], "not-correct")

    @pytest.mark.parametrize(
        "input_sample,expected",
        [
            ([[0, 0], [4, 3]], (1.0, 0)),
        ],
    )
    def test_edge_cases(self, input_sample, expected):
        res = barnard_exact(input_sample)
        statistic, pvalue = res.statistic, res.pvalue
        assert_equal(pvalue, expected[0])
        assert_equal(statistic, expected[1])

    @pytest.mark.parametrize(
        "input_sample,expected",
        [
            ([[0, 5], [0, 10]], (1.0, np.nan)),
            ([[5, 0], [10, 0]], (1.0, np.nan)),
        ],
    )
    def test_row_or_col_zero(self, input_sample, expected):
        res = barnard_exact(input_sample)
        statistic, pvalue = res.statistic, res.pvalue
        assert_equal(pvalue, expected[0])
        assert_equal(statistic, expected[1])

    @pytest.mark.parametrize(
        "input_sample,expected",
        [
            ([[2, 7], [8, 2]], (-2.518474945157, 0.009886140845)),
            ([[7, 200], [300, 8]], (-21.320036698460, 0.0)),
            ([[21, 28], [1957, 6]], (-30.489638143953, 0.0)),
        ],
    )
    @pytest.mark.parametrize("alternative", ["greater", "less"])
    def test_less_greater(self, input_sample, expected, alternative):
        """
        "The expected values have been generated by R, using a resolution
        for the nuisance parameter of 1e-6 :
        ```R
        library(Barnard)
        options(digits=10)
        a = barnard.test(2, 7, 8, 2, dp=1e-6, pooled=TRUE)
        a$p.value[1]
        ```
        In this test, we are using the "one-sided" return value `a$p.value[1]`
        to test our pvalue.
        """
        expected_stat, less_pvalue_expect = expected

        if alternative == "greater":
            input_sample = np.array(input_sample)[:, ::-1]
            expected_stat = -expected_stat

        res = barnard_exact(input_sample, alternative=alternative)
        statistic, pvalue = res.statistic, res.pvalue
        assert_allclose(
            [statistic, pvalue], [expected_stat, less_pvalue_expect], atol=1e-7
        )


class TestBoschlooExact:
    """Some tests to show that boschloo_exact() works correctly."""

    ATOL = 1e-7
    @pytest.mark.parametrize(
        "input_sample,expected",
        [
            ([[2, 7], [8, 2]], (0.01852173, 0.009886142)),
            ([[5, 1], [10, 10]], (0.9782609, 0.9450994)),
            ([[5, 16], [20, 25]], (0.08913823, 0.05827348)),
            ([[10, 5], [10, 1]], (0.1652174, 0.08565611)),
            ([[5, 0], [1, 4]], (1, 1)),
            ([[0, 1], [3, 2]], (0.5, 0.34375)),
            ([[2, 7], [8, 2]], (0.01852173, 0.009886142)),
            ([[7, 12], [8, 3]], (0.06406797, 0.03410916)),
            ([[10, 24], [25, 37]], (0.2009359, 0.1512882)),
        ],
    )
    def test_less(self, input_sample, expected):
        """The expected values have been generated by R, using a resolution
        for the nuisance parameter of 1e-8 :
        ```R
        library(Exact)
        options(digits=10)
        data <- matrix(c(43, 10, 40, 39), 2, 2, byrow=TRUE)
        a = exact.test(data, method="Boschloo", alternative="less",
                       tsmethod="central", np.interval=TRUE, beta=1e-8)
        ```
        """
        res = boschloo_exact(input_sample, alternative="less")
        statistic, pvalue = res.statistic, res.pvalue
        assert_allclose([statistic, pvalue], expected, atol=self.ATOL)

    @pytest.mark.parametrize(
        "input_sample,expected",
        [
            ([[43, 40], [10, 39]], (0.0002875544, 0.0001615562)),
            ([[2, 7], [8, 2]], (0.9990149, 0.9918327)),
            ([[5, 1], [10, 10]], (0.1652174, 0.09008534)),
            ([[5, 15], [20, 20]], (0.9849087, 0.9706997)),
            ([[5, 16], [20, 25]], (0.972349, 0.9524124)),
            ([[5, 0], [1, 4]], (0.02380952, 0.006865367)),
            ([[0, 1], [3, 2]], (1, 1)),
            ([[0, 2], [6, 4]], (1, 1)),
            ([[2, 7], [8, 2]], (0.9990149, 0.9918327)),
            ([[7, 12], [8, 3]], (0.9895302, 0.9771215)),
            ([[10, 24], [25, 37]], (0.9012936, 0.8633275)),
        ],
    )
    def test_greater(self, input_sample, expected):
        """The expected values have been generated by R, using a resolution
        for the nuisance parameter of 1e-8 :
        ```R
        library(Exact)
        options(digits=10)
        data <- matrix(c(43, 10, 40, 39), 2, 2, byrow=TRUE)
        a = exact.test(data, method="Boschloo", alternative="greater",
                       tsmethod="central", np.interval=TRUE, beta=1e-8)
        ```
        """
        res = boschloo_exact(input_sample, alternative="greater")
        statistic, pvalue = res.statistic, res.pvalue
        assert_allclose([statistic, pvalue], expected, atol=self.ATOL)

    @pytest.mark.parametrize(
        "input_sample,expected",
        [
            ([[43, 40], [10, 39]], (0.0002875544, 0.0003231115)),
            ([[2, 7], [8, 2]], (0.01852173, 0.01977228)),
            ([[5, 1], [10, 10]], (0.1652174, 0.1801707)),
            ([[5, 16], [20, 25]], (0.08913823, 0.116547)),
            ([[5, 0], [1, 4]], (0.02380952, 0.01373073)),
            ([[0, 1], [3, 2]], (0.5, 0.6875)),
            ([[2, 7], [8, 2]], (0.01852173, 0.01977228)),
            ([[7, 12], [8, 3]], (0.06406797, 0.06821831)),
        ],
    )
    def test_two_sided(self, input_sample, expected):
        """The expected values have been generated by R, using a resolution
        for the nuisance parameter of 1e-8 :
        ```R
        library(Exact)
        options(digits=10)
        data <- matrix(c(43, 10, 40, 39), 2, 2, byrow=TRUE)
        a = exact.test(data, method="Boschloo", alternative="two.sided",
                       tsmethod="central", np.interval=TRUE, beta=1e-8)
        ```
        """
        res = boschloo_exact(input_sample, alternative="two-sided", n=64)
        # Need n = 64 for python 32-bit
        statistic, pvalue = res.statistic, res.pvalue
        assert_allclose([statistic, pvalue], expected, atol=self.ATOL)

    def test_raises(self):
        # test we raise an error for wrong input number of nuisances.
        error_msg = (
            "Number of points `n` must be strictly positive, found 0"
        )
        with assert_raises(ValueError, match=error_msg):
            boschloo_exact([[1, 2], [3, 4]], n=0)

        # test we raise an error for wrong shape of input.
        error_msg = "The input `table` must be of shape \\(2, 2\\)."
        with assert_raises(ValueError, match=error_msg):
            boschloo_exact(np.arange(6).reshape(2, 3))

        # Test all values must be positives
        error_msg = "All values in `table` must be nonnegative."
        with assert_raises(ValueError, match=error_msg):
            boschloo_exact([[-1, 2], [3, 4]])

        # Test value error on wrong alternative param
        error_msg = (
            r"`alternative` should be one of \('two-sided', 'less', "
            r"'greater'\), found .*"
        )
        with assert_raises(ValueError, match=error_msg):
            boschloo_exact([[1, 2], [3, 4]], "not-correct")

    @pytest.mark.parametrize(
        "input_sample,expected",
        [
            ([[0, 5], [0, 10]], (np.nan, np.nan)),
            ([[5, 0], [10, 0]], (np.nan, np.nan)),
        ],
    )
    def test_row_or_col_zero(self, input_sample, expected):
        res = boschloo_exact(input_sample)
        statistic, pvalue = res.statistic, res.pvalue
        assert_equal(pvalue, expected[0])
        assert_equal(statistic, expected[1])

class TestCvm_2samp:
    def test_invalid_input(self):
        x = np.arange(10).reshape((2, 5))
        y = np.arange(5)
        msg = 'The samples must be one-dimensional'
        with pytest.raises(ValueError, match=msg):
            cramervonmises_2samp(x, y)
        with pytest.raises(ValueError, match=msg):
            cramervonmises_2samp(y, x)
        msg = 'x and y must contain at least two observations.'
        with pytest.raises(ValueError, match=msg):
            cramervonmises_2samp([], y)
        with pytest.raises(ValueError, match=msg):
            cramervonmises_2samp(y, [1])
        msg = 'method must be either auto, exact or asymptotic'
        with pytest.raises(ValueError, match=msg):
            cramervonmises_2samp(y, y, 'xyz')

    def test_list_input(self):
        x = [2, 3, 4, 7, 6]
        y = [0.2, 0.7, 12, 18]
        r1 = cramervonmises_2samp(x, y)
        r2 = cramervonmises_2samp(np.array(x), np.array(y))
        assert_equal((r1.statistic, r1.pvalue), (r2.statistic, r2.pvalue))

    def test_example_conover(self):
        # Example 2 in Section 6.2 of W.J. Conover: Practical Nonparametric
        # Statistics, 1971.
        x = [7.6, 8.4, 8.6, 8.7, 9.3, 9.9, 10.1, 10.6, 11.2]
        y = [5.2, 5.7, 5.9, 6.5, 6.8, 8.2, 9.1, 9.8, 10.8, 11.3, 11.5, 12.3,
             12.5, 13.4, 14.6]
        r = cramervonmises_2samp(x, y)
        assert_allclose(r.statistic, 0.262, atol=1e-3)
        assert_allclose(r.pvalue, 0.18, atol=1e-2)

    @pytest.mark.parametrize('statistic, m, n, pval',
                             [(710, 5, 6, 48./462),
                              (1897, 7, 7, 117./1716),
                              (576, 4, 6, 2./210),
                              (1764, 6, 7, 2./1716)])
    def test_exact_pvalue(self, statistic, m, n, pval):
        # the exact values are taken from Anderson: On the distribution of the
        # two-sample Cramer-von-Mises criterion, 1962.
        # The values are taken from Table 2, 3, 4 and 5
        assert_equal(_pval_cvm_2samp_exact(statistic, m, n), pval)

    def test_large_sample(self):
        # for large samples, the statistic U gets very large
        # do a sanity check that p-value is not 0, 1 or nan
        np.random.seed(4367)
        x = distributions.norm.rvs(size=1000000)
        y = distributions.norm.rvs(size=900000)
        r = cramervonmises_2samp(x, y)
        assert_(0 < r.pvalue < 1)
        r = cramervonmises_2samp(x, y+0.1)
        assert_(0 < r.pvalue < 1)

    def test_exact_vs_asymptotic(self):
        np.random.seed(0)
        x = np.random.rand(7)
        y = np.random.rand(8)
        r1 = cramervonmises_2samp(x, y, method='exact')
        r2 = cramervonmises_2samp(x, y, method='asymptotic')
        assert_equal(r1.statistic, r2.statistic)
        assert_allclose(r1.pvalue, r2.pvalue, atol=1e-2)

    def test_method_auto(self):
        x = np.arange(10)
        y = [0.5, 4.7, 13.1]
        r1 = cramervonmises_2samp(x, y, method='exact')
        r2 = cramervonmises_2samp(x, y, method='auto')
        assert_equal(r1.pvalue, r2.pvalue)
        # switch to asymptotic if one sample has more than 10 observations
        x = np.arange(11)
        r1 = cramervonmises_2samp(x, y, method='asymptotic')
        r2 = cramervonmises_2samp(x, y, method='auto')
        assert_equal(r1.pvalue, r2.pvalue)

    def test_same_input(self):
        # make sure trivial edge case can be handled
        # note that _cdf_cvm_inf(0) = nan. implementation avoids nan by
        # returning pvalue=1 for very small values of the statistic
        x = np.arange(15)
        res = cramervonmises_2samp(x, x)
        assert_equal((res.statistic, res.pvalue), (0.0, 1.0))
        # check exact p-value
        res = cramervonmises_2samp(x[:4], x[:4])
        assert_equal((res.statistic, res.pvalue), (0.0, 1.0))
