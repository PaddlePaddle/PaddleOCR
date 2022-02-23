""" Test functions for stats module

    WRITTEN BY LOUIS LUANGKESORN <lluang@yahoo.com> FOR THE STATS MODULE
    BASED ON WILKINSON'S STATISTICS QUIZ
    https://www.stanford.edu/~clint/bench/wilk.txt

    Additional tests by a host of SciPy developers.
"""
import os
import warnings
from collections import namedtuple
from itertools import product
from copy import deepcopy

from numpy.testing import (assert_, assert_equal,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal, assert_approx_equal,
                           assert_allclose, assert_warns, suppress_warnings,
                           assert_string_equal)
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np

from scipy._lib._util import check_random_state
from scipy import special
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats.mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from .common_tests import check_named_results
from scipy.sparse.sputils import matrix
from scipy.spatial.distance import cdist
from scipy.stats._distr_params import distcont
from numpy.lib import NumpyVersion
from scipy.stats.stats import (_broadcast_concatenate,
                               AlexanderGovernConstantInputWarning)
from scipy.stats.stats import _calc_t_stat, _data_partitions

""" Numbers in docstrings beginning with 'W' refer to the section numbers
    and headings found in the STATISTICS QUIZ of Leland Wilkinson.  These are
    considered to be essential functionality.  True testing and
    evaluation of a statistics package requires use of the
    NIST Statistical test data.  See McCoullough(1999) Assessing The Reliability
    of Statistical Software for a test methodology and its
    implementation in testing SAS, SPSS, and S-Plus
"""

#  Datasets
#  These data sets are from the nasty.dat sets used by Wilkinson
#  For completeness, I should write the relevant tests and count them as failures
#  Somewhat acceptable, since this is still beta software.  It would count as a
#  good target for 1.0 status
X = array([1,2,3,4,5,6,7,8,9], float)
ZERO = array([0,0,0,0,0,0,0,0,0], float)
BIG = array([99999991,99999992,99999993,99999994,99999995,99999996,99999997,
             99999998,99999999], float)
LITTLE = array([0.99999991,0.99999992,0.99999993,0.99999994,0.99999995,0.99999996,
                0.99999997,0.99999998,0.99999999], float)
HUGE = array([1e+12,2e+12,3e+12,4e+12,5e+12,6e+12,7e+12,8e+12,9e+12], float)
TINY = array([1e-12,2e-12,3e-12,4e-12,5e-12,6e-12,7e-12,8e-12,9e-12], float)
ROUND = array([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5], float)


class TestTrimmedStats:
    # TODO: write these tests to handle missing values properly
    dprec = np.finfo(np.float64).precision

    def test_tmean(self):
        y = stats.tmean(X, (2, 8), (True, True))
        assert_approx_equal(y, 5.0, significant=self.dprec)

        y1 = stats.tmean(X, limits=(2, 8), inclusive=(False, False))
        y2 = stats.tmean(X, limits=None)
        assert_approx_equal(y1, y2, significant=self.dprec)

    def test_tvar(self):
        y = stats.tvar(X, limits=(2, 8), inclusive=(True, True))
        assert_approx_equal(y, 4.6666666666666661, significant=self.dprec)

        y = stats.tvar(X, limits=None)
        assert_approx_equal(y, X.var(ddof=1), significant=self.dprec)

        x_2d = arange(63, dtype=float64).reshape((9, 7))
        y = stats.tvar(x_2d, axis=None)
        assert_approx_equal(y, x_2d.var(ddof=1), significant=self.dprec)

        y = stats.tvar(x_2d, axis=0)
        assert_array_almost_equal(y[0], np.full((1, 7), 367.50000000), decimal=8)

        y = stats.tvar(x_2d, axis=1)
        assert_array_almost_equal(y[0], np.full((1, 9), 4.66666667), decimal=8)

        y = stats.tvar(x_2d[3, :])
        assert_approx_equal(y, 4.666666666666667, significant=self.dprec)

        with suppress_warnings() as sup:
            sup.record(RuntimeWarning, "Degrees of freedom <= 0 for slice.")

            # Limiting some values along one axis
            y = stats.tvar(x_2d, limits=(1, 5), axis=1, inclusive=(True, True))
            assert_approx_equal(y[0], 2.5, significant=self.dprec)

            # Limiting all values along one axis
            y = stats.tvar(x_2d, limits=(0, 6), axis=1, inclusive=(True, True))
            assert_approx_equal(y[0], 4.666666666666667, significant=self.dprec)
            assert_equal(y[1], np.nan)

    def test_tstd(self):
        y = stats.tstd(X, (2, 8), (True, True))
        assert_approx_equal(y, 2.1602468994692865, significant=self.dprec)

        y = stats.tstd(X, limits=None)
        assert_approx_equal(y, X.std(ddof=1), significant=self.dprec)

    def test_tmin(self):
        assert_equal(stats.tmin(4), 4)

        x = np.arange(10)
        assert_equal(stats.tmin(x), 0)
        assert_equal(stats.tmin(x, lowerlimit=0), 0)
        assert_equal(stats.tmin(x, lowerlimit=0, inclusive=False), 1)

        x = x.reshape((5, 2))
        assert_equal(stats.tmin(x, lowerlimit=0, inclusive=False), [2, 1])
        assert_equal(stats.tmin(x, axis=1), [0, 2, 4, 6, 8])
        assert_equal(stats.tmin(x, axis=None), 0)

        x = np.arange(10.)
        x[9] = np.nan
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning, "invalid value*")
            assert_equal(stats.tmin(x), np.nan)
            assert_equal(stats.tmin(x, nan_policy='omit'), 0.)
            assert_raises(ValueError, stats.tmin, x, nan_policy='raise')
            assert_raises(ValueError, stats.tmin, x, nan_policy='foobar')
            msg = "'propagate', 'raise', 'omit'"
            with assert_raises(ValueError, match=msg):
                stats.tmin(x, nan_policy='foo')

    def test_tmax(self):
        assert_equal(stats.tmax(4), 4)

        x = np.arange(10)
        assert_equal(stats.tmax(x), 9)
        assert_equal(stats.tmax(x, upperlimit=9), 9)
        assert_equal(stats.tmax(x, upperlimit=9, inclusive=False), 8)

        x = x.reshape((5, 2))
        assert_equal(stats.tmax(x, upperlimit=9, inclusive=False), [8, 7])
        assert_equal(stats.tmax(x, axis=1), [1, 3, 5, 7, 9])
        assert_equal(stats.tmax(x, axis=None), 9)

        x = np.arange(10.)
        x[6] = np.nan
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning, "invalid value*")
            assert_equal(stats.tmax(x), np.nan)
            assert_equal(stats.tmax(x, nan_policy='omit'), 9.)
            assert_raises(ValueError, stats.tmax, x, nan_policy='raise')
            assert_raises(ValueError, stats.tmax, x, nan_policy='foobar')

    def test_tsem(self):
        y = stats.tsem(X, limits=(3, 8), inclusive=(False, True))
        y_ref = np.array([4, 5, 6, 7, 8])
        assert_approx_equal(y, y_ref.std(ddof=1) / np.sqrt(y_ref.size),
                            significant=self.dprec)

        assert_approx_equal(stats.tsem(X, limits=[-1, 10]),
                            stats.tsem(X, limits=None),
                            significant=self.dprec)


class TestCorrPearsonr:
    """ W.II.D. Compute a correlation matrix on all the variables.

        All the correlations, except for ZERO and MISS, should be exactly 1.
        ZERO and MISS should have undefined or missing correlations with the
        other variables.  The same should go for SPEARMAN correlations, if
        your program has them.
    """

    def test_pXX(self):
        y = stats.pearsonr(X,X)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pXBIG(self):
        y = stats.pearsonr(X,BIG)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pXLITTLE(self):
        y = stats.pearsonr(X,LITTLE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pXHUGE(self):
        y = stats.pearsonr(X,HUGE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pXTINY(self):
        y = stats.pearsonr(X,TINY)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pXROUND(self):
        y = stats.pearsonr(X,ROUND)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pBIGBIG(self):
        y = stats.pearsonr(BIG,BIG)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pBIGLITTLE(self):
        y = stats.pearsonr(BIG,LITTLE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pBIGHUGE(self):
        y = stats.pearsonr(BIG,HUGE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pBIGTINY(self):
        y = stats.pearsonr(BIG,TINY)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pBIGROUND(self):
        y = stats.pearsonr(BIG,ROUND)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pLITTLELITTLE(self):
        y = stats.pearsonr(LITTLE,LITTLE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pLITTLEHUGE(self):
        y = stats.pearsonr(LITTLE,HUGE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pLITTLETINY(self):
        y = stats.pearsonr(LITTLE,TINY)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pLITTLEROUND(self):
        y = stats.pearsonr(LITTLE,ROUND)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pHUGEHUGE(self):
        y = stats.pearsonr(HUGE,HUGE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pHUGETINY(self):
        y = stats.pearsonr(HUGE,TINY)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pHUGEROUND(self):
        y = stats.pearsonr(HUGE,ROUND)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pTINYTINY(self):
        y = stats.pearsonr(TINY,TINY)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pTINYROUND(self):
        y = stats.pearsonr(TINY,ROUND)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pROUNDROUND(self):
        y = stats.pearsonr(ROUND,ROUND)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_r_almost_exactly_pos1(self):
        a = arange(3.0)
        r, prob = stats.pearsonr(a, a)

        assert_allclose(r, 1.0, atol=1e-15)
        # With n = len(a) = 3, the error in prob grows like the
        # square root of the error in r.
        assert_allclose(prob, 0.0, atol=np.sqrt(2*np.spacing(1.0)))

    def test_r_almost_exactly_neg1(self):
        a = arange(3.0)
        r, prob = stats.pearsonr(a, -a)

        assert_allclose(r, -1.0, atol=1e-15)
        # With n = len(a) = 3, the error in prob grows like the
        # square root of the error in r.
        assert_allclose(prob, 0.0, atol=np.sqrt(2*np.spacing(1.0)))

    def test_basic(self):
        # A basic test, with a correlation coefficient
        # that is not 1 or -1.
        a = array([-1, 0, 1])
        b = array([0, 0, 3])
        r, prob = stats.pearsonr(a, b)
        assert_approx_equal(r, np.sqrt(3)/2)
        assert_approx_equal(prob, 1/3)

    def test_constant_input(self):
        # Zero variance input
        # See https://github.com/scipy/scipy/issues/3728
        with assert_warns(stats.PearsonRConstantInputWarning):
            r, p = stats.pearsonr([0.667, 0.667, 0.667], [0.123, 0.456, 0.789])
            assert_equal(r, np.nan)
            assert_equal(p, np.nan)

    def test_near_constant_input(self):
        # Near constant input (but not constant):
        x = [2, 2, 2 + np.spacing(2)]
        y = [3, 3, 3 + 6*np.spacing(3)]
        with assert_warns(stats.PearsonRNearConstantInputWarning):
            # r and p are garbage, so don't bother checking them in this case.
            # (The exact value of r would be 1.)
            r, p = stats.pearsonr(x, y)

    def test_very_small_input_values(self):
        # Very small values in an input.  A naive implementation will
        # suffer from underflow.
        # See https://github.com/scipy/scipy/issues/9353
        x = [0.004434375, 0.004756007, 0.003911996, 0.0038005, 0.003409971]
        y = [2.48e-188, 7.41e-181, 4.09e-208, 2.08e-223, 2.66e-245]
        r, p = stats.pearsonr(x,y)

        # The expected values were computed using mpmath with 80 digits
        # of precision.
        assert_allclose(r, 0.7272930540750450)
        assert_allclose(p, 0.1637805429533202)

    def test_very_large_input_values(self):
        # Very large values in an input.  A naive implementation will
        # suffer from overflow.
        # See https://github.com/scipy/scipy/issues/8980
        x = 1e90*np.array([0, 0, 0, 1, 1, 1, 1])
        y = 1e90*np.arange(7)

        r, p = stats.pearsonr(x, y)

        # The expected values were computed using mpmath with 80 digits
        # of precision.
        assert_allclose(r, 0.8660254037844386)
        assert_allclose(p, 0.011724811003954638)

    def test_extremely_large_input_values(self):
        # Extremely large values in x and y.  These values would cause the
        # product sigma_x * sigma_y to overflow if the two factors were
        # computed independently.
        x = np.array([2.3e200, 4.5e200, 6.7e200, 8e200])
        y = np.array([1.2e199, 5.5e200, 3.3e201, 1.0e200])
        r, p = stats.pearsonr(x, y)

        # The expected values were computed using mpmath with 80 digits
        # of precision.
        assert_allclose(r, 0.351312332103289)
        assert_allclose(p, 0.648687667896711)

    def test_length_two_pos1(self):
        # Inputs with length 2.
        # See https://github.com/scipy/scipy/issues/7730
        r, p = stats.pearsonr([1, 2], [3, 5])
        assert_equal(r, 1)
        assert_equal(p, 1)

    def test_length_two_neg2(self):
        # Inputs with length 2.
        # See https://github.com/scipy/scipy/issues/7730
        r, p = stats.pearsonr([2, 1], [3, 5])
        assert_equal(r, -1)
        assert_equal(p, 1)

    def test_more_basic_examples(self):
        x = [1, 2, 3, 4]
        y = [0, 1, 0.5, 1]
        r, p = stats.pearsonr(x, y)

        # The expected values were computed using mpmath with 80 digits
        # of precision.
        assert_allclose(r, 0.674199862463242)
        assert_allclose(p, 0.325800137536758)

        x = [1, 2, 3]
        y = [5, -4, -13]
        r, p = stats.pearsonr(x, y)

        # The expected r and p are exact.
        assert_allclose(r, -1.0)
        assert_allclose(p, 0.0, atol=1e-7)

    def test_unequal_lengths(self):
        x = [1, 2, 3]
        y = [4, 5]
        assert_raises(ValueError, stats.pearsonr, x, y)

    def test_len1(self):
        x = [1]
        y = [2]
        assert_raises(ValueError, stats.pearsonr, x, y)


class TestFisherExact:
    """Some tests to show that fisher_exact() works correctly.

    Note that in SciPy 0.9.0 this was not working well for large numbers due to
    inaccuracy of the hypergeom distribution (see #1218). Fixed now.

    Also note that R and SciPy have different argument formats for their
    hypergeometric distribution functions.

    R:
    > phyper(18999, 99000, 110000, 39000, lower.tail = FALSE)
    [1] 1.701815e-09
    """

    def test_basic(self):
        fisher_exact = stats.fisher_exact

        res = fisher_exact([[14500, 20000], [30000, 40000]])[1]
        assert_approx_equal(res, 0.01106, significant=4)
        res = fisher_exact([[100, 2], [1000, 5]])[1]
        assert_approx_equal(res, 0.1301, significant=4)
        res = fisher_exact([[2, 7], [8, 2]])[1]
        assert_approx_equal(res, 0.0230141, significant=6)
        res = fisher_exact([[5, 1], [10, 10]])[1]
        assert_approx_equal(res, 0.1973244, significant=6)
        res = fisher_exact([[5, 15], [20, 20]])[1]
        assert_approx_equal(res, 0.0958044, significant=6)
        res = fisher_exact([[5, 16], [20, 25]])[1]
        assert_approx_equal(res, 0.1725862, significant=6)
        res = fisher_exact([[10, 5], [10, 1]])[1]
        assert_approx_equal(res, 0.1973244, significant=6)
        res = fisher_exact([[5, 0], [1, 4]])[1]
        assert_approx_equal(res, 0.04761904, significant=6)
        res = fisher_exact([[0, 1], [3, 2]])[1]
        assert_approx_equal(res, 1.0)
        res = fisher_exact([[0, 2], [6, 4]])[1]
        assert_approx_equal(res, 0.4545454545)
        res = fisher_exact([[2, 7], [8, 2]])
        assert_approx_equal(res[1], 0.0230141, significant=6)
        assert_approx_equal(res[0], 4.0 / 56)

    def test_precise(self):
        # results from R
        #
        # R defines oddsratio differently (see Notes section of fisher_exact
        # docstring), so those will not match.  We leave them in anyway, in
        # case they will be useful later on. We test only the p-value.
        tablist = [
            ([[100, 2], [1000, 5]], (2.505583993422285e-001, 1.300759363430016e-001)),
            ([[2, 7], [8, 2]], (8.586235135736206e-002, 2.301413756522114e-002)),
            ([[5, 1], [10, 10]], (4.725646047336584e+000, 1.973244147157190e-001)),
            ([[5, 15], [20, 20]], (3.394396617440852e-001, 9.580440012477637e-002)),
            ([[5, 16], [20, 25]], (3.960558326183334e-001, 1.725864953812994e-001)),
            ([[10, 5], [10, 1]], (2.116112781158483e-001, 1.973244147157190e-001)),
            ([[10, 5], [10, 0]], (0.000000000000000e+000, 6.126482213438734e-002)),
            ([[5, 0], [1, 4]], (np.inf, 4.761904761904762e-002)),
            ([[0, 5], [1, 4]], (0.000000000000000e+000, 1.000000000000000e+000)),
            ([[5, 1], [0, 4]], (np.inf, 4.761904761904758e-002)),
            ([[0, 1], [3, 2]], (0.000000000000000e+000, 1.000000000000000e+000))
            ]
        for table, res_r in tablist:
            res = stats.fisher_exact(np.asarray(table))
            np.testing.assert_almost_equal(res[1], res_r[1], decimal=11,
                                           verbose=True)

    @pytest.mark.slow
    def test_large_numbers(self):
        # Test with some large numbers. Regression test for #1401
        pvals = [5.56e-11, 2.666e-11, 1.363e-11]  # from R
        for pval, num in zip(pvals, [75, 76, 77]):
            res = stats.fisher_exact([[17704, 496], [1065, num]])[1]
            assert_approx_equal(res, pval, significant=4)

        res = stats.fisher_exact([[18000, 80000], [20000, 90000]])[1]
        assert_approx_equal(res, 0.2751, significant=4)

    def test_raises(self):
        # test we raise an error for wrong shape of input.
        assert_raises(ValueError, stats.fisher_exact,
                      np.arange(6).reshape(2, 3))

    def test_row_or_col_zero(self):
        tables = ([[0, 0], [5, 10]],
                  [[5, 10], [0, 0]],
                  [[0, 5], [0, 10]],
                  [[5, 0], [10, 0]])
        for table in tables:
            oddsratio, pval = stats.fisher_exact(table)
            assert_equal(pval, 1.0)
            assert_equal(oddsratio, np.nan)

    def test_less_greater(self):
        tables = (
            # Some tables to compare with R:
            [[2, 7], [8, 2]],
            [[200, 7], [8, 300]],
            [[28, 21], [6, 1957]],
            [[190, 800], [200, 900]],
            # Some tables with simple exact values
            # (includes regression test for ticket #1568):
            [[0, 2], [3, 0]],
            [[1, 1], [2, 1]],
            [[2, 0], [1, 2]],
            [[0, 1], [2, 3]],
            [[1, 0], [1, 4]],
            )
        pvals = (
            # from R:
            [0.018521725952066501, 0.9990149169715733],
            [1.0, 2.0056578803889148e-122],
            [1.0, 5.7284374608319831e-44],
            [0.7416227, 0.2959826],
            # Exact:
            [0.1, 1.0],
            [0.7, 0.9],
            [1.0, 0.3],
            [2./3, 1.0],
            [1.0, 1./3],
            )
        for table, pval in zip(tables, pvals):
            res = []
            res.append(stats.fisher_exact(table, alternative="less")[1])
            res.append(stats.fisher_exact(table, alternative="greater")[1])
            assert_allclose(res, pval, atol=0, rtol=1e-7)

    def test_gh3014(self):
        # check if issue #3014 has been fixed.
        # before, this would have risen a ValueError
        odds, pvalue = stats.fisher_exact([[1, 2], [9, 84419233]])


class TestCorrSpearmanr:
    """ W.II.D. Compute a correlation matrix on all the variables.

        All the correlations, except for ZERO and MISS, should be exactly 1.
        ZERO and MISS should have undefined or missing correlations with the
        other variables.  The same should go for SPEARMAN corelations, if
        your program has them.
    """

    def test_scalar(self):
        y = stats.spearmanr(4., 2.)
        assert_(np.isnan(y).all())

    def test_uneven_lengths(self):
        assert_raises(ValueError, stats.spearmanr, [1, 2, 1], [8, 9])
        assert_raises(ValueError, stats.spearmanr, [1, 2, 1], 8)

    def test_uneven_2d_shapes(self):
        # Different number of columns should work - those just get concatenated.
        np.random.seed(232324)
        x = np.random.randn(4, 3)
        y = np.random.randn(4, 2)
        assert stats.spearmanr(x, y).correlation.shape == (5, 5)
        assert stats.spearmanr(x.T, y.T, axis=1).pvalue.shape == (5, 5)

        assert_raises(ValueError, stats.spearmanr, x, y, axis=1)
        assert_raises(ValueError, stats.spearmanr, x.T, y.T)

    def test_ndim_too_high(self):
        np.random.seed(232324)
        x = np.random.randn(4, 3, 2)
        assert_raises(ValueError, stats.spearmanr, x)
        assert_raises(ValueError, stats.spearmanr, x, x)
        assert_raises(ValueError, stats.spearmanr, x, None, None)
        # But should work with axis=None (raveling axes) for two input arrays
        assert_allclose(stats.spearmanr(x, x, axis=None),
                        stats.spearmanr(x.flatten(), x.flatten(), axis=0))

    def test_nan_policy(self):
        x = np.arange(10.)
        x[9] = np.nan
        assert_array_equal(stats.spearmanr(x, x), (np.nan, np.nan))
        assert_array_equal(stats.spearmanr(x, x, nan_policy='omit'),
                           (1.0, 0.0))
        assert_raises(ValueError, stats.spearmanr, x, x, nan_policy='raise')
        assert_raises(ValueError, stats.spearmanr, x, x, nan_policy='foobar')

    def test_nan_policy_bug_12458(self):
        np.random.seed(5)
        x = np.random.rand(5, 10)
        k = 6
        x[:, k] = np.nan
        y = np.delete(x, k, axis=1)
        corx, px = stats.spearmanr(x, nan_policy='omit')
        cory, py = stats.spearmanr(y)
        corx = np.delete(np.delete(corx, k, axis=1), k, axis=0)
        px = np.delete(np.delete(px, k, axis=1), k, axis=0)
        assert_allclose(corx, cory, atol=1e-14)
        assert_allclose(px, py, atol=1e-14)

    def test_nan_policy_bug_12411(self):
        np.random.seed(5)
        m = 5
        n = 10
        x = np.random.randn(m, n)
        x[1, 0] = np.nan
        x[3, -1] = np.nan
        corr, pvalue = stats.spearmanr(x, axis=1, nan_policy="propagate")
        res = [[stats.spearmanr(x[i, :], x[j, :]).correlation for i in range(m)]
               for j in range(m)]
        assert_allclose(corr, res)

    def test_sXX(self):
        y = stats.spearmanr(X,X)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_sXBIG(self):
        y = stats.spearmanr(X,BIG)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_sXLITTLE(self):
        y = stats.spearmanr(X,LITTLE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_sXHUGE(self):
        y = stats.spearmanr(X,HUGE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_sXTINY(self):
        y = stats.spearmanr(X,TINY)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_sXROUND(self):
        y = stats.spearmanr(X,ROUND)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_sBIGBIG(self):
        y = stats.spearmanr(BIG,BIG)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_sBIGLITTLE(self):
        y = stats.spearmanr(BIG,LITTLE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_sBIGHUGE(self):
        y = stats.spearmanr(BIG,HUGE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_sBIGTINY(self):
        y = stats.spearmanr(BIG,TINY)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_sBIGROUND(self):
        y = stats.spearmanr(BIG,ROUND)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_sLITTLELITTLE(self):
        y = stats.spearmanr(LITTLE,LITTLE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_sLITTLEHUGE(self):
        y = stats.spearmanr(LITTLE,HUGE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_sLITTLETINY(self):
        y = stats.spearmanr(LITTLE,TINY)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_sLITTLEROUND(self):
        y = stats.spearmanr(LITTLE,ROUND)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_sHUGEHUGE(self):
        y = stats.spearmanr(HUGE,HUGE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_sHUGETINY(self):
        y = stats.spearmanr(HUGE,TINY)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_sHUGEROUND(self):
        y = stats.spearmanr(HUGE,ROUND)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_sTINYTINY(self):
        y = stats.spearmanr(TINY,TINY)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_sTINYROUND(self):
        y = stats.spearmanr(TINY,ROUND)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_sROUNDROUND(self):
        y = stats.spearmanr(ROUND,ROUND)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_spearmanr_result_attributes(self):
        res = stats.spearmanr(X, X)
        attributes = ('correlation', 'pvalue')
        check_named_results(res, attributes)

    def test_1d_vs_2d(self):
        x1 = [1, 2, 3, 4, 5, 6]
        x2 = [1, 2, 3, 4, 6, 5]
        res1 = stats.spearmanr(x1, x2)
        res2 = stats.spearmanr(np.asarray([x1, x2]).T)
        assert_allclose(res1, res2)

    def test_1d_vs_2d_nans(self):
        # Now the same with NaNs present.  Regression test for gh-9103.
        for nan_policy in ['propagate', 'omit']:
            x1 = [1, np.nan, 3, 4, 5, 6]
            x2 = [1, 2, 3, 4, 6, np.nan]
            res1 = stats.spearmanr(x1, x2, nan_policy=nan_policy)
            res2 = stats.spearmanr(np.asarray([x1, x2]).T, nan_policy=nan_policy)
            assert_allclose(res1, res2)

    def test_3cols(self):
        x1 = np.arange(6)
        x2 = -x1
        x3 = np.array([0, 1, 2, 3, 5, 4])
        x = np.asarray([x1, x2, x3]).T
        actual = stats.spearmanr(x)
        expected_corr = np.array([[1, -1, 0.94285714],
                                  [-1, 1, -0.94285714],
                                  [0.94285714, -0.94285714, 1]])
        expected_pvalue = np.zeros((3, 3), dtype=float)
        expected_pvalue[2, 0:2] = 0.00480466472
        expected_pvalue[0:2, 2] = 0.00480466472

        assert_allclose(actual.correlation, expected_corr)
        assert_allclose(actual.pvalue, expected_pvalue)

    def test_gh_9103(self):
        # Regression test for gh-9103.
        x = np.array([[np.nan, 3.0, 4.0, 5.0, 5.1, 6.0, 9.2],
                      [5.0, np.nan, 4.1, 4.8, 4.9, 5.0, 4.1],
                      [0.5, 4.0, 7.1, 3.8, 8.0, 5.1, 7.6]]).T
        corr = np.array([[np.nan, np.nan, np.nan],
                         [np.nan, np.nan, np.nan],
                         [np.nan, np.nan, 1.]])
        assert_allclose(stats.spearmanr(x, nan_policy='propagate').correlation,
                        corr)

        res = stats.spearmanr(x, nan_policy='omit').correlation
        assert_allclose((res[0][1], res[0][2], res[1][2]),
                        (0.2051957, 0.4857143, -0.4707919), rtol=1e-6)

    def test_gh_8111(self):
        # Regression test for gh-8111 (different result for float/int/bool).
        n = 100
        np.random.seed(234568)
        x = np.random.rand(n)
        m = np.random.rand(n) > 0.7

        # bool against float, no nans
        a = (x > .5)
        b = np.array(x)
        res1 = stats.spearmanr(a, b, nan_policy='omit').correlation

        # bool against float with NaNs
        b[m] = np.nan
        res2 = stats.spearmanr(a, b, nan_policy='omit').correlation

        # int against float with NaNs
        a = a.astype(np.int32)
        res3 = stats.spearmanr(a, b, nan_policy='omit').correlation

        expected = [0.865895477, 0.866100381, 0.866100381]
        assert_allclose([res1, res2, res3], expected)


class TestCorrSpearmanr2:
    """Some further tests of the spearmanr function."""

    def test_spearmanr_vs_r(self):
        # Cross-check with R:
        # cor.test(c(1,2,3,4,5),c(5,6,7,8,7),method="spearmanr")
        x1 = [1, 2, 3, 4, 5]
        x2 = [5, 6, 7, 8, 7]
        expected = (0.82078268166812329, 0.088587005313543798)
        res = stats.spearmanr(x1, x2)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])

    def test_empty_arrays(self):
        assert_equal(stats.spearmanr([], []), (np.nan, np.nan))

    def test_normal_draws(self):
        np.random.seed(7546)
        x = np.array([np.random.normal(loc=1, scale=1, size=500),
                    np.random.normal(loc=1, scale=1, size=500)])
        corr = [[1.0, 0.3],
                [0.3, 1.0]]
        x = np.dot(np.linalg.cholesky(corr), x)
        expected = (0.28659685838743354, 6.579862219051161e-11)
        res = stats.spearmanr(x[0], x[1])
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])

    def test_corr_1(self):
        assert_approx_equal(stats.spearmanr([1, 1, 2], [1, 1, 2])[0], 1.0)

    def test_nan_policies(self):
        x = np.arange(10.)
        x[9] = np.nan
        assert_array_equal(stats.spearmanr(x, x), (np.nan, np.nan))
        assert_allclose(stats.spearmanr(x, x, nan_policy='omit'),
                        (1.0, 0))
        assert_raises(ValueError, stats.spearmanr, x, x, nan_policy='raise')
        assert_raises(ValueError, stats.spearmanr, x, x, nan_policy='foobar')

    def test_unequal_lengths(self):
        x = np.arange(10.)
        y = np.arange(20.)
        assert_raises(ValueError, stats.spearmanr, x, y)

    def test_omit_paired_value(self):
        x1 = [1, 2, 3, 4]
        x2 = [8, 7, 6, np.nan]
        res1 = stats.spearmanr(x1, x2, nan_policy='omit')
        res2 = stats.spearmanr(x1[:3], x2[:3], nan_policy='omit')
        assert_equal(res1, res2)

    def test_gh_issue_6061_windows_overflow(self):
        x = list(range(2000))
        y = list(range(2000))
        y[0], y[9] = y[9], y[0]
        y[10], y[434] = y[434], y[10]
        y[435], y[1509] = y[1509], y[435]
        # rho = 1 - 6 * (2 * (9^2 + 424^2 + 1074^2))/(2000 * (2000^2 - 1))
        #     = 1 - (1 / 500)
        #     = 0.998
        x.append(np.nan)
        y.append(3.0)
        assert_almost_equal(stats.spearmanr(x, y, nan_policy='omit')[0], 0.998)

    def test_tie0(self):
        # with only ties in one or both inputs
        with assert_warns(stats.SpearmanRConstantInputWarning):
            r, p = stats.spearmanr([2, 2, 2], [2, 2, 2])
            assert_equal(r, np.nan)
            assert_equal(p, np.nan)
            r, p = stats.spearmanr([2, 0, 2], [2, 2, 2])
            assert_equal(r, np.nan)
            assert_equal(p, np.nan)
            r, p = stats.spearmanr([2, 2, 2], [2, 0, 2])
            assert_equal(r, np.nan)
            assert_equal(p, np.nan)

    def test_tie1(self):
        # Data
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.0, 2.0, 2.0, 3.0]
        # Ranks of the data, with tie-handling.
        xr = [1.0, 2.0, 3.0, 4.0]
        yr = [1.0, 2.5, 2.5, 4.0]
        # Result of spearmanr should be the same as applying
        # pearsonr to the ranks.
        sr = stats.spearmanr(x, y)
        pr = stats.pearsonr(xr, yr)
        assert_almost_equal(sr, pr)

    def test_tie2(self):
        # Test tie-handling if inputs contain nan's
        # Data without nan's
        x1 = [1, 2, 2.5, 2]
        y1 = [1, 3, 2.5, 4]
        # Same data with nan's
        x2 = [1, 2, 2.5, 2, np.nan]
        y2 = [1, 3, 2.5, 4, np.nan]

        # Results for two data sets should be the same if nan's are ignored
        sr1 = stats.spearmanr(x1, y1)
        sr2 = stats.spearmanr(x2, y2, nan_policy='omit')
        assert_almost_equal(sr1, sr2)

    def test_ties_axis_1(self):
        z1 = np.array([[1, 1, 1, 1], [1, 2, 3, 4]])
        z2 = np.array([[1, 2, 3, 4], [1, 1, 1, 1]])
        z3 = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
        with assert_warns(stats.SpearmanRConstantInputWarning):
            r, p = stats.spearmanr(z1, axis=1)
            assert_equal(r, np.nan)
            assert_equal(p, np.nan)
            r, p = stats.spearmanr(z2, axis=1)
            assert_equal(r, np.nan)
            assert_equal(p, np.nan)
            r, p = stats.spearmanr(z3, axis=1)
            assert_equal(r, np.nan)
            assert_equal(p, np.nan)

    def test_gh_11111(self):
        x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        y = np.array([0, 0.009783728115345005, 0, 0, 0.0019759230121848587,
            0.0007535430349118562, 0.0002661781514710257, 0, 0,
            0.0007835762419683435])
        with assert_warns(stats.SpearmanRConstantInputWarning):
            r, p = stats.spearmanr(x, y)
            assert_equal(r, np.nan)
            assert_equal(p, np.nan)

    def test_index_error(self):
        x = np.array([1.0, 7.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        y = np.array([0, 0.009783728115345005, 0, 0, 0.0019759230121848587,
            0.0007535430349118562, 0.0002661781514710257, 0, 0,
            0.0007835762419683435])
        assert_raises(ValueError, stats.spearmanr, x, y, axis=2)

    def test_alternative(self):
        # Test alternative parameter

        # Simple test - Based on the above ``test_spearmanr_vs_r``
        x1 = [1, 2, 3, 4, 5]
        x2 = [5, 6, 7, 8, 7]

        # strong positive correlation
        expected = (0.82078268166812329, 0.088587005313543798)

        # correlation > 0 -> large "less" p-value
        res = stats.spearmanr(x1, x2, alternative="less")
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], 1 - (expected[1] / 2))

        # correlation > 0 -> small "less" p-value
        res = stats.spearmanr(x1, x2, alternative="greater")
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1] / 2)

        with pytest.raises(ValueError, match="alternative must be 'less'..."):
            stats.spearmanr(x1, x2, alternative="ekki-ekki")

    @pytest.mark.parametrize("alternative", ('two-sided', 'less', 'greater'))
    def test_alternative_nan_policy(self, alternative):
        # Test nan policies
        x1 = [1, 2, 3, 4, 5]
        x2 = [5, 6, 7, 8, 7]
        x1nan = x1 + [np.nan]
        x2nan = x2 + [np.nan]

        # test nan_policy="propagate"
        assert_array_equal(stats.spearmanr(x1nan, x2nan), (np.nan, np.nan))

        # test nan_policy="omit"
        res_actual = stats.spearmanr(x1nan, x2nan, nan_policy='omit',
                                     alternative=alternative)
        res_expected = stats.spearmanr(x1, x2, alternative=alternative)
        assert_allclose(res_actual, res_expected)

        # test nan_policy="raise"
        message = 'The input contains nan values'
        with pytest.raises(ValueError, match=message):
            stats.spearmanr(x1nan, x2nan, nan_policy='raise',
                            alternative=alternative)

        # test invalid nan_policy
        message = "nan_policy must be one of..."
        with pytest.raises(ValueError, match=message):
            stats.spearmanr(x1nan, x2nan, nan_policy='ekki-ekki',
                            alternative=alternative)


#    W.II.E.  Tabulate X against X, using BIG as a case weight.  The values
#    should appear on the diagonal and the total should be 899999955.
#    If the table cannot hold these values, forget about working with
#    census data.  You can also tabulate HUGE against TINY.  There is no
#    reason a tabulation program should not be able to distinguish
#    different values regardless of their magnitude.

# I need to figure out how to do this one.


def test_kendalltau():
    # For the cases without ties, both variants should give the same
    # result.
    variants = ('b', 'c')

    # case without ties, con-dis equal zero
    x = [5, 2, 1, 3, 6, 4, 7, 8]
    y = [5, 2, 6, 3, 1, 8, 7, 4]
    # Cross-check with exact result from R:
    # cor.test(x,y,method="kendall",exact=1)
    expected = (0.0, 1.0)
    for taux in variants:
        res = stats.kendalltau(x, y)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])

    # case without ties, con-dis equal zero
    x = [0, 5, 2, 1, 3, 6, 4, 7, 8]
    y = [5, 2, 0, 6, 3, 1, 8, 7, 4]
    # Cross-check with exact result from R:
    # cor.test(x,y,method="kendall",exact=1)
    expected = (0.0, 1.0)
    for taux in variants:
        res = stats.kendalltau(x, y)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])

    # case without ties, con-dis close to zero
    x = [5, 2, 1, 3, 6, 4, 7]
    y = [5, 2, 6, 3, 1, 7, 4]
    # Cross-check with exact result from R:
    # cor.test(x,y,method="kendall",exact=1)
    expected = (-0.14285714286, 0.77261904762)
    for taux in variants:
        res = stats.kendalltau(x, y)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])

    # case without ties, con-dis close to zero
    x = [2, 1, 3, 6, 4, 7, 8]
    y = [2, 6, 3, 1, 8, 7, 4]
    # Cross-check with exact result from R:
    # cor.test(x,y,method="kendall",exact=1)
    expected = (0.047619047619, 1.0)
    for taux in variants:
        res = stats.kendalltau(x, y)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])

    # simple case without ties
    x = np.arange(10)
    y = np.arange(10)
    # Cross-check with exact result from R:
    # cor.test(x,y,method="kendall",exact=1)
    expected = (1.0, 5.511463844797e-07)
    for taux in variants:
        res = stats.kendalltau(x, y, variant=taux)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])

    # swap a couple of values
    b = y[1]
    y[1] = y[2]
    y[2] = b
    # Cross-check with exact result from R:
    # cor.test(x,y,method="kendall",exact=1)
    expected = (0.9555555555555556, 5.511463844797e-06)
    for taux in variants:
        res = stats.kendalltau(x, y, variant=taux)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])

    # swap a couple more
    b = y[5]
    y[5] = y[6]
    y[6] = b
    # Cross-check with exact result from R:
    # cor.test(x,y,method="kendall",exact=1)
    expected = (0.9111111111111111, 2.976190476190e-05)
    for taux in variants:
        res = stats.kendalltau(x, y, variant=taux)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])

    # same in opposite direction
    x = np.arange(10)
    y = np.arange(10)[::-1]
    # Cross-check with exact result from R:
    # cor.test(x,y,method="kendall",exact=1)
    expected = (-1.0, 5.511463844797e-07)
    for taux in variants:
        res = stats.kendalltau(x, y, variant=taux)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])

    # swap a couple of values
    b = y[1]
    y[1] = y[2]
    y[2] = b
    # Cross-check with exact result from R:
    # cor.test(x,y,method="kendall",exact=1)
    expected = (-0.9555555555555556, 5.511463844797e-06)
    for taux in variants:
        res = stats.kendalltau(x, y, variant=taux)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])

    # swap a couple more
    b = y[5]
    y[5] = y[6]
    y[6] = b
    # Cross-check with exact result from R:
    # cor.test(x,y,method="kendall",exact=1)
    expected = (-0.9111111111111111, 2.976190476190e-05)
    for taux in variants:
        res = stats.kendalltau(x, y, variant=taux)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])

    # Check a case where variants are different
    # Example values found from Kendall (1970).
    # P-value is the same for the both variants
    x = array([1, 2, 2, 4, 4, 6, 6, 8, 9, 9])
    y = array([1, 2, 4, 4, 4, 4, 8, 8, 8, 10])
    expected = 0.85895569
    assert_approx_equal(stats.kendalltau(x, y, variant='b')[0], expected)
    expected = 0.825
    assert_approx_equal(stats.kendalltau(x, y, variant='c')[0], expected)

    # check exception in case of ties and method='exact' requested
    y[2] = y[1]
    assert_raises(ValueError, stats.kendalltau, x, y, method='exact')

    # check exception in case of invalid method keyword
    assert_raises(ValueError, stats.kendalltau, x, y, method='banana')

    # check exception in case of invalid variant keyword
    assert_raises(ValueError, stats.kendalltau, x, y, variant='rms')

    # tau-b with some ties
    # Cross-check with R:
    # cor.test(c(12,2,1,12,2),c(1,4,7,1,0),method="kendall",exact=FALSE)
    x1 = [12, 2, 1, 12, 2]
    x2 = [1, 4, 7, 1, 0]
    expected = (-0.47140452079103173, 0.28274545993277478)
    res = stats.kendalltau(x1, x2)
    assert_approx_equal(res[0], expected[0])
    assert_approx_equal(res[1], expected[1])

    # test for namedtuple attribute results
    attributes = ('correlation', 'pvalue')
    for taux in variants:
        res = stats.kendalltau(x1, x2, variant=taux)
        check_named_results(res, attributes)

    # with only ties in one or both inputs in tau-b or tau-c
    for taux in variants:
        assert_equal(stats.kendalltau([2, 2, 2], [2, 2, 2], variant=taux),
                     (np.nan, np.nan))
        assert_equal(stats.kendalltau([2, 0, 2], [2, 2, 2], variant=taux),
                     (np.nan, np.nan))
        assert_equal(stats.kendalltau([2, 2, 2], [2, 0, 2], variant=taux),
                     (np.nan, np.nan))

    # empty arrays provided as input
    assert_equal(stats.kendalltau([], []), (np.nan, np.nan))

    # check with larger arrays
    np.random.seed(7546)
    x = np.array([np.random.normal(loc=1, scale=1, size=500),
                np.random.normal(loc=1, scale=1, size=500)])
    corr = [[1.0, 0.3],
            [0.3, 1.0]]
    x = np.dot(np.linalg.cholesky(corr), x)
    expected = (0.19291382765531062, 1.1337095377742629e-10)
    res = stats.kendalltau(x[0], x[1])
    assert_approx_equal(res[0], expected[0])
    assert_approx_equal(res[1], expected[1])

    # this should result in 1 for taub but not tau-c
    assert_approx_equal(stats.kendalltau([1, 1, 2], [1, 1, 2], variant='b')[0],
                        1.0)
    assert_approx_equal(stats.kendalltau([1, 1, 2], [1, 1, 2], variant='c')[0],
                        0.88888888)

    # test nan_policy
    x = np.arange(10.)
    x[9] = np.nan
    assert_array_equal(stats.kendalltau(x, x), (np.nan, np.nan))
    assert_allclose(stats.kendalltau(x, x, nan_policy='omit'),
                    (1.0, 5.5114638e-6), rtol=1e-06)
    assert_allclose(stats.kendalltau(x, x, nan_policy='omit', method='asymptotic'),
                    (1.0, 0.00017455009626808976), rtol=1e-06)
    assert_raises(ValueError, stats.kendalltau, x, x, nan_policy='raise')
    assert_raises(ValueError, stats.kendalltau, x, x, nan_policy='foobar')

    # test unequal length inputs
    x = np.arange(10.)
    y = np.arange(20.)
    assert_raises(ValueError, stats.kendalltau, x, y)

    # test all ties
    tau, p_value = stats.kendalltau([], [])
    assert_equal(np.nan, tau)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.kendalltau([0], [0])
    assert_equal(np.nan, tau)
    assert_equal(np.nan, p_value)

    # Regression test for GitHub issue #6061 - Overflow on Windows
    x = np.arange(2000, dtype=float)
    x = np.ma.masked_greater(x, 1995)
    y = np.arange(2000, dtype=float)
    y = np.concatenate((y[1000:], y[:1000]))
    assert_(np.isfinite(stats.kendalltau(x,y)[1]))


def test_kendalltau_vs_mstats_basic():
    np.random.seed(42)
    for s in range(2,10):
        a = []
        # Generate rankings with ties
        for i in range(s):
            a += [i]*i
        b = list(a)
        np.random.shuffle(a)
        np.random.shuffle(b)
        expected = mstats_basic.kendalltau(a, b)
        actual = stats.kendalltau(a, b)
        assert_approx_equal(actual[0], expected[0])
        assert_approx_equal(actual[1], expected[1])


def test_kendalltau_nan_2nd_arg():
    # regression test for gh-6134: nans in the second arg were not handled
    x = [1., 2., 3., 4.]
    y = [np.nan, 2.4, 3.4, 3.4]

    r1 = stats.kendalltau(x, y, nan_policy='omit')
    r2 = stats.kendalltau(x[1:], y[1:])
    assert_allclose(r1.correlation, r2.correlation, atol=1e-15)


def test_weightedtau():
    x = [12, 2, 1, 12, 2]
    y = [1, 4, 7, 1, 0]
    tau, p_value = stats.weightedtau(x, y)
    assert_approx_equal(tau, -0.56694968153682723)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau(x, y, additive=False)
    assert_approx_equal(tau, -0.62205716951801038)
    assert_equal(np.nan, p_value)
    # This must be exactly Kendall's tau
    tau, p_value = stats.weightedtau(x, y, weigher=lambda x: 1)
    assert_approx_equal(tau, -0.47140452079103173)
    assert_equal(np.nan, p_value)

    # Asymmetric, ranked version
    tau, p_value = stats.weightedtau(x, y, rank=None)
    assert_approx_equal(tau, -0.4157652301037516)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau(y, x, rank=None)
    assert_approx_equal(tau, -0.7181341329699029)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau(x, y, rank=None, additive=False)
    assert_approx_equal(tau, -0.40644850966246893)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau(y, x, rank=None, additive=False)
    assert_approx_equal(tau, -0.83766582937355172)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau(x, y, rank=False)
    assert_approx_equal(tau, -0.51604397940261848)
    assert_equal(np.nan, p_value)
    # This must be exactly Kendall's tau
    tau, p_value = stats.weightedtau(x, y, rank=True, weigher=lambda x: 1)
    assert_approx_equal(tau, -0.47140452079103173)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau(y, x, rank=True, weigher=lambda x: 1)
    assert_approx_equal(tau, -0.47140452079103173)
    assert_equal(np.nan, p_value)
    # Test argument conversion
    tau, p_value = stats.weightedtau(np.asarray(x, dtype=np.float64), y)
    assert_approx_equal(tau, -0.56694968153682723)
    tau, p_value = stats.weightedtau(np.asarray(x, dtype=np.int16), y)
    assert_approx_equal(tau, -0.56694968153682723)
    tau, p_value = stats.weightedtau(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    assert_approx_equal(tau, -0.56694968153682723)
    # All ties
    tau, p_value = stats.weightedtau([], [])
    assert_equal(np.nan, tau)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau([0], [0])
    assert_equal(np.nan, tau)
    assert_equal(np.nan, p_value)
    # Size mismatches
    assert_raises(ValueError, stats.weightedtau, [0, 1], [0, 1, 2])
    assert_raises(ValueError, stats.weightedtau, [0, 1], [0, 1], [0])
    # NaNs
    x = [12, 2, 1, 12, 2]
    y = [1, 4, 7, 1, np.nan]
    tau, p_value = stats.weightedtau(x, y)
    assert_approx_equal(tau, -0.56694968153682723)
    x = [12, 2, np.nan, 12, 2]
    tau, p_value = stats.weightedtau(x, y)
    assert_approx_equal(tau, -0.56694968153682723)
    # NaNs when the dtype of x and y are all np.float64
    x = [12.0, 2.0, 1.0, 12.0, 2.0]
    y = [1.0, 4.0, 7.0, 1.0, np.nan]
    tau, p_value = stats.weightedtau(x, y)
    assert_approx_equal(tau, -0.56694968153682723)
    x = [12.0, 2.0, np.nan, 12.0, 2.0]
    tau, p_value = stats.weightedtau(x, y)
    assert_approx_equal(tau, -0.56694968153682723)
    # NaNs when there are more than one NaN in x or y
    x = [12.0, 2.0, 1.0, 12.0, 1.0]
    y = [1.0, 4.0, 7.0, 1.0, 1.0]
    tau, p_value = stats.weightedtau(x, y)
    assert_approx_equal(tau, -0.6615242347139803)
    x = [12.0, 2.0, np.nan, 12.0, np.nan]
    tau, p_value = stats.weightedtau(x, y)
    assert_approx_equal(tau, -0.6615242347139803)
    y = [np.nan, 4.0, 7.0, np.nan, np.nan]
    tau, p_value = stats.weightedtau(x, y)
    assert_approx_equal(tau, -0.6615242347139803)


def test_segfault_issue_9710():
    # https://github.com/scipy/scipy/issues/9710
    # This test was created to check segfault
    # In issue SEGFAULT only repros in optimized builds after calling the function twice
    stats.weightedtau([1], [1.0])
    stats.weightedtau([1], [1.0])
    # The code below also caused SEGFAULT
    stats.weightedtau([np.nan], [52])


def test_kendall_tau_large():
    n = 172
    # Test omit policy
    x = np.arange(n + 1).astype(float)
    y = np.arange(n + 1).astype(float)
    y[-1] = np.nan
    _, pval = stats.kendalltau(x, y, method='exact', nan_policy='omit')
    assert_equal(pval, 0.0)


def test_weightedtau_vs_quadratic():
    # Trivial quadratic implementation, all parameters mandatory
    def wkq(x, y, rank, weigher, add):
        tot = conc = disc = u = v = 0
        for i in range(len(x)):
            for j in range(len(x)):
                w = weigher(rank[i]) + weigher(rank[j]) if add else weigher(rank[i]) * weigher(rank[j])
                tot += w
                if x[i] == x[j]:
                    u += w
                if y[i] == y[j]:
                    v += w
                if x[i] < x[j] and y[i] < y[j] or x[i] > x[j] and y[i] > y[j]:
                    conc += w
                elif x[i] < x[j] and y[i] > y[j] or x[i] > x[j] and y[i] < y[j]:
                    disc += w
        return (conc - disc) / np.sqrt(tot - u) / np.sqrt(tot - v)

    np.random.seed(42)
    for s in range(3,10):
        a = []
        # Generate rankings with ties
        for i in range(s):
            a += [i]*i
        b = list(a)
        np.random.shuffle(a)
        np.random.shuffle(b)
        # First pass: use element indices as ranks
        rank = np.arange(len(a), dtype=np.intp)
        for _ in range(2):
            for add in [True, False]:
                expected = wkq(a, b, rank, lambda x: 1./(x+1), add)
                actual = stats.weightedtau(a, b, rank, lambda x: 1./(x+1), add).correlation
                assert_approx_equal(expected, actual)
            # Second pass: use a random rank
            np.random.shuffle(rank)


class TestFindRepeats:

    def test_basic(self):
        a = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 5]
        res, nums = stats.find_repeats(a)
        assert_array_equal(res, [1, 2, 3, 4])
        assert_array_equal(nums, [3, 3, 2, 2])

    def test_empty_result(self):
        # Check that empty arrays are returned when there are no repeats.
        for a in [[10, 20, 50, 30, 40], []]:
            repeated, counts = stats.find_repeats(a)
            assert_array_equal(repeated, [])
            assert_array_equal(counts, [])


class TestRegression:

    def test_linregressBIGX(self):
        # W.II.F.  Regress BIG on X.
        result = stats.linregress(X, BIG)
        assert_almost_equal(result.intercept, 99999990)
        assert_almost_equal(result.rvalue, 1.0)
        # The uncertainty ought to be almost zero
        # since all points lie on a line
        assert_almost_equal(result.stderr, 0.0)
        assert_almost_equal(result.intercept_stderr, 0.0)

    def test_regressXX(self):
        # W.IV.B.  Regress X on X.
        # The constant should be exactly 0 and the regression coefficient
        # should be 1.  This is a perfectly valid regression and the
        # program should not complain.
        result = stats.linregress(X, X)
        assert_almost_equal(result.intercept, 0.0)
        assert_almost_equal(result.rvalue, 1.0)
        # The uncertainly on regression through two points ought to be 0
        assert_almost_equal(result.stderr, 0.0)
        assert_almost_equal(result.intercept_stderr, 0.0)

        # W.IV.C. Regress X on BIG and LITTLE (two predictors).  The program
        # should tell you that this model is "singular" because BIG and
        # LITTLE are linear combinations of each other.  Cryptic error
        # messages are unacceptable here.  Singularity is the most
        # fundamental regression error.
        #
        # Need to figure out how to handle multiple linear regression.
        # This is not obvious

    def test_regressZEROX(self):
        # W.IV.D. Regress ZERO on X.
        # The program should inform you that ZERO has no variance or it should
        # go ahead and compute the regression and report a correlation and
        # total sum of squares of exactly 0.
        result = stats.linregress(X, ZERO)
        assert_almost_equal(result.intercept, 0.0)
        assert_almost_equal(result.rvalue, 0.0)

    def test_regress_simple(self):
        # Regress a line with sinusoidal noise.
        x = np.linspace(0, 100, 100)
        y = 0.2 * np.linspace(0, 100, 100) + 10
        y += np.sin(np.linspace(0, 20, 100))

        result = stats.linregress(x, y)
        lr = stats._stats_mstats_common.LinregressResult
        assert_(isinstance(result, lr))
        assert_almost_equal(result.stderr, 2.3957814497838803e-3)

    def test_regress_alternative(self):
        # test alternative parameter
        x = np.linspace(0, 100, 100)
        y = 0.2 * np.linspace(0, 100, 100) + 10  # slope is greater than zero
        y += np.sin(np.linspace(0, 20, 100))

        with pytest.raises(ValueError, match="alternative must be 'less'..."):
            stats.linregress(x, y, alternative="ekki-ekki")

        res1 = stats.linregress(x, y, alternative="two-sided")

        # slope is greater than zero, so "less" p-value should be large
        res2 = stats.linregress(x, y, alternative="less")
        assert_allclose(res2.pvalue, 1 - (res1.pvalue / 2))

        # slope is greater than zero, so "greater" p-value should be small
        res3 = stats.linregress(x, y, alternative="greater")
        assert_allclose(res3.pvalue, res1.pvalue / 2)

        assert res1.rvalue == res2.rvalue == res3.rvalue

    def test_regress_against_R(self):
        # test against R `lm`
        # options(digits=16)
        # x <- c(151, 174, 138, 186, 128, 136, 179, 163, 152, 131)
        # y <- c(63, 81, 56, 91, 47, 57, 76, 72, 62, 48)
        # relation <- lm(y~x)
        # print(summary(relation))

        x = [151, 174, 138, 186, 128, 136, 179, 163, 152, 131]
        y = [63, 81, 56, 91, 47, 57, 76, 72, 62, 48]
        res = stats.linregress(x, y, alternative="two-sided")
        # expected values from R's `lm` above
        assert_allclose(res.slope, 0.6746104491292)
        assert_allclose(res.intercept, -38.4550870760770)
        assert_allclose(res.rvalue, np.sqrt(0.95478224775))
        assert_allclose(res.pvalue, 1.16440531074e-06)
        assert_allclose(res.stderr, 0.0519051424731)
        assert_allclose(res.intercept_stderr, 8.0490133029927)

    def test_regress_simple_onearg_rows(self):
        # Regress a line w sinusoidal noise,
        # with a single input of shape (2, N)
        x = np.linspace(0, 100, 100)
        y = 0.2 * np.linspace(0, 100, 100) + 10
        y += np.sin(np.linspace(0, 20, 100))
        rows = np.vstack((x, y))

        result = stats.linregress(rows)
        assert_almost_equal(result.stderr, 2.3957814497838803e-3)
        assert_almost_equal(result.intercept_stderr, 1.3866936078570702e-1)

    def test_regress_simple_onearg_cols(self):
        x = np.linspace(0, 100, 100)
        y = 0.2 * np.linspace(0, 100, 100) + 10
        y += np.sin(np.linspace(0, 20, 100))
        columns = np.hstack((np.expand_dims(x, 1), np.expand_dims(y, 1)))

        result = stats.linregress(columns)
        assert_almost_equal(result.stderr, 2.3957814497838803e-3)
        assert_almost_equal(result.intercept_stderr, 1.3866936078570702e-1)

    def test_regress_shape_error(self):
        # Check that a single input argument to linregress with wrong shape
        # results in a ValueError.
        assert_raises(ValueError, stats.linregress, np.ones((3, 3)))

    def test_linregress(self):
        # compared with multivariate ols with pinv
        x = np.arange(11)
        y = np.arange(5, 16)
        y[[(1), (-2)]] -= 1
        y[[(0), (-1)]] += 1

        result = stats.linregress(x, y)

        # This test used to use 'assert_array_almost_equal' but its
        # formualtion got confusing since LinregressResult became
        # _lib._bunch._make_tuple_bunch instead of namedtuple
        # (for backwards compatibility, see PR #12983)
        assert_ae = lambda x, y: assert_almost_equal(x, y, decimal=14)
        assert_ae(result.slope, 1.0)
        assert_ae(result.intercept, 5.0)
        assert_ae(result.rvalue, 0.98229948625750)
        assert_ae(result.pvalue, 7.45259691e-008)
        assert_ae(result.stderr, 0.063564172616372733)
        assert_ae(result.intercept_stderr, 0.37605071654517686)

    def test_regress_simple_negative_cor(self):
        # If the slope of the regression is negative the factor R tend
        # to -1 not 1.  Sometimes rounding errors makes it < -1
        # leading to stderr being NaN.
        a, n = 1e-71, 100000
        x = np.linspace(a, 2 * a, n)
        y = np.linspace(2 * a, a, n)
        result = stats.linregress(x, y)

        # Make sure propagated numerical errors
        # did not bring rvalue below -1 (or were coersced)
        assert_(result.rvalue >= -1)
        assert_almost_equal(result.rvalue, -1)

        # slope and intercept stderror should stay numeric
        assert_(not np.isnan(result.stderr))
        assert_(not np.isnan(result.intercept_stderr))

    def test_linregress_result_attributes(self):
        x = np.linspace(0, 100, 100)
        y = 0.2 * np.linspace(0, 100, 100) + 10
        y += np.sin(np.linspace(0, 20, 100))
        result = stats.linregress(x, y)

        # Result is of a correct class
        lr = stats._stats_mstats_common.LinregressResult
        assert_(isinstance(result, lr))

        # LinregressResult elements have correct names
        attributes = ('slope', 'intercept', 'rvalue', 'pvalue', 'stderr')
        check_named_results(result, attributes)
        # Also check that the extra attribute (intercept_stderr) is present
        assert 'intercept_stderr' in dir(result)

    def test_regress_two_inputs(self):
        # Regress a simple line formed by two points.
        x = np.arange(2)
        y = np.arange(3, 5)
        result = stats.linregress(x, y)

        # Non-horizontal line
        assert_almost_equal(result.pvalue, 0.0)

        # Zero error through two points
        assert_almost_equal(result.stderr, 0.0)
        assert_almost_equal(result.intercept_stderr, 0.0)

    def test_regress_two_inputs_horizontal_line(self):
        # Regress a horizontal line formed by two points.
        x = np.arange(2)
        y = np.ones(2)
        result = stats.linregress(x, y)

        # Horizontal line
        assert_almost_equal(result.pvalue, 1.0)

        # Zero error through two points
        assert_almost_equal(result.stderr, 0.0)
        assert_almost_equal(result.intercept_stderr, 0.0)

    def test_nist_norris(self):
        x = [0.2, 337.4, 118.2, 884.6, 10.1, 226.5, 666.3, 996.3, 448.6, 777.0,
             558.2, 0.4, 0.6, 775.5, 666.9, 338.0, 447.5, 11.6, 556.0, 228.1,
             995.8, 887.6, 120.2, 0.3, 0.3, 556.8, 339.1, 887.2, 999.0, 779.0,
             11.1, 118.3, 229.2, 669.1, 448.9, 0.5]

        y = [0.1, 338.8, 118.1, 888.0, 9.2, 228.1, 668.5, 998.5, 449.1, 778.9,
             559.2, 0.3, 0.1, 778.1, 668.8, 339.3, 448.9, 10.8, 557.7, 228.3,
             998.0, 888.8, 119.6, 0.3, 0.6, 557.6, 339.3, 888.0, 998.5, 778.9,
             10.2, 117.6, 228.9, 668.4, 449.2, 0.2]

        result = stats.linregress(x, y)

        assert_almost_equal(result.slope, 1.00211681802045)
        assert_almost_equal(result.intercept, -0.262323073774029)
        assert_almost_equal(result.rvalue**2, 0.999993745883712)
        assert_almost_equal(result.pvalue, 0.0)
        assert_almost_equal(result.stderr, 0.00042979684820)
        assert_almost_equal(result.intercept_stderr, 0.23281823430153)

    def test_compare_to_polyfit(self):
        x = np.linspace(0, 100, 100)
        y = 0.2 * np.linspace(0, 100, 100) + 10
        y += np.sin(np.linspace(0, 20, 100))
        result = stats.linregress(x, y)
        poly = np.polyfit(x, y, 1)  # Fit 1st degree polynomial

        # Make sure linear regression slope and intercept
        # match with results from numpy polyfit
        assert_almost_equal(result.slope, poly[0])
        assert_almost_equal(result.intercept, poly[1])

    def test_empty_input(self):
        assert_raises(ValueError, stats.linregress, [], [])

    def test_nan_input(self):
        x = np.arange(10.)
        x[9] = np.nan

        with np.errstate(invalid="ignore"):
            result = stats.linregress(x, x)

        # Make sure the resut still comes back as `LinregressResult`
        lr = stats._stats_mstats_common.LinregressResult
        assert_(isinstance(result, lr))
        assert_array_equal(result, (np.nan,)*5)
        assert_equal(result.intercept_stderr, np.nan)


def test_theilslopes():
    # Basic slope test.
    slope, intercept, lower, upper = stats.theilslopes([0,1,1])
    assert_almost_equal(slope, 0.5)
    assert_almost_equal(intercept, 0.5)

    # Test of confidence intervals.
    x = [1, 2, 3, 4, 10, 12, 18]
    y = [9, 15, 19, 20, 45, 55, 78]
    slope, intercept, lower, upper = stats.theilslopes(y, x, 0.07)
    assert_almost_equal(slope, 4)
    assert_almost_equal(upper, 4.38, decimal=2)
    assert_almost_equal(lower, 3.71, decimal=2)


def test_cumfreq():
    x = [1, 4, 2, 1, 3, 1]
    cumfreqs, lowlim, binsize, extrapoints = stats.cumfreq(x, numbins=4)
    assert_array_almost_equal(cumfreqs, np.array([3., 4., 5., 6.]))
    cumfreqs, lowlim, binsize, extrapoints = stats.cumfreq(x, numbins=4,
                                                      defaultreallimits=(1.5, 5))
    assert_(extrapoints == 3)

    # test for namedtuple attribute results
    attributes = ('cumcount', 'lowerlimit', 'binsize', 'extrapoints')
    res = stats.cumfreq(x, numbins=4, defaultreallimits=(1.5, 5))
    check_named_results(res, attributes)


def test_relfreq():
    a = np.array([1, 4, 2, 1, 3, 1])
    relfreqs, lowlim, binsize, extrapoints = stats.relfreq(a, numbins=4)
    assert_array_almost_equal(relfreqs,
                              array([0.5, 0.16666667, 0.16666667, 0.16666667]))

    # test for namedtuple attribute results
    attributes = ('frequency', 'lowerlimit', 'binsize', 'extrapoints')
    res = stats.relfreq(a, numbins=4)
    check_named_results(res, attributes)

    # check array_like input is accepted
    relfreqs2, lowlim, binsize, extrapoints = stats.relfreq([1, 4, 2, 1, 3, 1],
                                                            numbins=4)
    assert_array_almost_equal(relfreqs, relfreqs2)


class TestScoreatpercentile:
    def setup_method(self):
        self.a1 = [3, 4, 5, 10, -3, -5, 6]
        self.a2 = [3, -6, -2, 8, 7, 4, 2, 1]
        self.a3 = [3., 4, 5, 10, -3, -5, -6, 7.0]

    def test_basic(self):
        x = arange(8) * 0.5
        assert_equal(stats.scoreatpercentile(x, 0), 0.)
        assert_equal(stats.scoreatpercentile(x, 100), 3.5)
        assert_equal(stats.scoreatpercentile(x, 50), 1.75)

    def test_fraction(self):
        scoreatperc = stats.scoreatpercentile

        # Test defaults
        assert_equal(scoreatperc(list(range(10)), 50), 4.5)
        assert_equal(scoreatperc(list(range(10)), 50, (2,7)), 4.5)
        assert_equal(scoreatperc(list(range(100)), 50, limit=(1, 8)), 4.5)
        assert_equal(scoreatperc(np.array([1, 10,100]), 50, (10,100)), 55)
        assert_equal(scoreatperc(np.array([1, 10,100]), 50, (1,10)), 5.5)

        # explicitly specify interpolation_method 'fraction' (the default)
        assert_equal(scoreatperc(list(range(10)), 50, interpolation_method='fraction'),
                     4.5)
        assert_equal(scoreatperc(list(range(10)), 50, limit=(2, 7),
                                 interpolation_method='fraction'),
                     4.5)
        assert_equal(scoreatperc(list(range(100)), 50, limit=(1, 8),
                                 interpolation_method='fraction'),
                     4.5)
        assert_equal(scoreatperc(np.array([1, 10,100]), 50, (10, 100),
                                 interpolation_method='fraction'),
                     55)
        assert_equal(scoreatperc(np.array([1, 10,100]), 50, (1,10),
                                 interpolation_method='fraction'),
                     5.5)

    def test_lower_higher(self):
        scoreatperc = stats.scoreatpercentile

        # interpolation_method 'lower'/'higher'
        assert_equal(scoreatperc(list(range(10)), 50,
                                 interpolation_method='lower'), 4)
        assert_equal(scoreatperc(list(range(10)), 50,
                                 interpolation_method='higher'), 5)
        assert_equal(scoreatperc(list(range(10)), 50, (2,7),
                                 interpolation_method='lower'), 4)
        assert_equal(scoreatperc(list(range(10)), 50, limit=(2,7),
                                 interpolation_method='higher'), 5)
        assert_equal(scoreatperc(list(range(100)), 50, (1,8),
                                 interpolation_method='lower'), 4)
        assert_equal(scoreatperc(list(range(100)), 50, (1,8),
                                 interpolation_method='higher'), 5)
        assert_equal(scoreatperc(np.array([1, 10, 100]), 50, (10, 100),
                                 interpolation_method='lower'), 10)
        assert_equal(scoreatperc(np.array([1, 10, 100]), 50, limit=(10, 100),
                                 interpolation_method='higher'), 100)
        assert_equal(scoreatperc(np.array([1, 10, 100]), 50, (1, 10),
                                 interpolation_method='lower'), 1)
        assert_equal(scoreatperc(np.array([1, 10, 100]), 50, limit=(1, 10),
                                 interpolation_method='higher'), 10)

    def test_sequence_per(self):
        x = arange(8) * 0.5
        expected = np.array([0, 3.5, 1.75])
        res = stats.scoreatpercentile(x, [0, 100, 50])
        assert_allclose(res, expected)
        assert_(isinstance(res, np.ndarray))
        # Test with ndarray.  Regression test for gh-2861
        assert_allclose(stats.scoreatpercentile(x, np.array([0, 100, 50])),
                        expected)
        # Also test combination of 2-D array, axis not None and array-like per
        res2 = stats.scoreatpercentile(np.arange(12).reshape((3,4)),
                                       np.array([0, 1, 100, 100]), axis=1)
        expected2 = array([[0, 4, 8],
                           [0.03, 4.03, 8.03],
                           [3, 7, 11],
                           [3, 7, 11]])
        assert_allclose(res2, expected2)

    def test_axis(self):
        scoreatperc = stats.scoreatpercentile
        x = arange(12).reshape(3, 4)

        assert_equal(scoreatperc(x, (25, 50, 100)), [2.75, 5.5, 11.0])

        r0 = [[2, 3, 4, 5], [4, 5, 6, 7], [8, 9, 10, 11]]
        assert_equal(scoreatperc(x, (25, 50, 100), axis=0), r0)

        r1 = [[0.75, 4.75, 8.75], [1.5, 5.5, 9.5], [3, 7, 11]]
        assert_equal(scoreatperc(x, (25, 50, 100), axis=1), r1)

        x = array([[1, 1, 1],
                   [1, 1, 1],
                   [4, 4, 3],
                   [1, 1, 1],
                   [1, 1, 1]])
        score = stats.scoreatpercentile(x, 50)
        assert_equal(score.shape, ())
        assert_equal(score, 1.0)
        score = stats.scoreatpercentile(x, 50, axis=0)
        assert_equal(score.shape, (3,))
        assert_equal(score, [1, 1, 1])

    def test_exception(self):
        assert_raises(ValueError, stats.scoreatpercentile, [1, 2], 56,
            interpolation_method='foobar')
        assert_raises(ValueError, stats.scoreatpercentile, [1], 101)
        assert_raises(ValueError, stats.scoreatpercentile, [1], -1)

    def test_empty(self):
        assert_equal(stats.scoreatpercentile([], 50), np.nan)
        assert_equal(stats.scoreatpercentile(np.array([[], []]), 50), np.nan)
        assert_equal(stats.scoreatpercentile([], [50, 99]), [np.nan, np.nan])


class TestItemfreq:
    a = [5, 7, 1, 2, 1, 5, 7] * 10
    b = [1, 2, 5, 7]

    def test_numeric_types(self):
        # Check itemfreq works for all dtypes (adapted from np.unique tests)
        def _check_itemfreq(dt):
            a = np.array(self.a, dt)
            with suppress_warnings() as sup:
                sup.filter(DeprecationWarning)
                v = stats.itemfreq(a)
            assert_array_equal(v[:, 0], [1, 2, 5, 7])
            assert_array_equal(v[:, 1], np.array([20, 10, 20, 20], dtype=dt))

        dtypes = [np.int32, np.int64, np.float32, np.float64,
                  np.complex64, np.complex128]
        for dt in dtypes:
            _check_itemfreq(dt)

    def test_object_arrays(self):
        a, b = self.a, self.b
        dt = 'O'
        aa = np.empty(len(a), dt)
        aa[:] = a
        bb = np.empty(len(b), dt)
        bb[:] = b
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            v = stats.itemfreq(aa)
        assert_array_equal(v[:, 0], bb)

    def test_structured_arrays(self):
        a, b = self.a, self.b
        dt = [('', 'i'), ('', 'i')]
        aa = np.array(list(zip(a, a)), dt)
        bb = np.array(list(zip(b, b)), dt)
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            v = stats.itemfreq(aa)
        # Arrays don't compare equal because v[:,0] is object array
        assert_equal(tuple(v[2, 0]), tuple(bb[2]))


class TestMode:
    def test_empty(self):
        vals, counts = stats.mode([])
        assert_equal(vals, np.array([]))
        assert_equal(counts, np.array([]))

    def test_scalar(self):
        vals, counts = stats.mode(4.)
        assert_equal(vals, np.array([4.]))
        assert_equal(counts, np.array([1]))

    def test_basic(self):
        data1 = [3, 5, 1, 10, 23, 3, 2, 6, 8, 6, 10, 6]
        vals = stats.mode(data1)
        assert_equal(vals[0][0], 6)
        assert_equal(vals[1][0], 3)

    def test_axes(self):
        data1 = [10, 10, 30, 40]
        data2 = [10, 10, 10, 10]
        data3 = [20, 10, 20, 20]
        data4 = [30, 30, 30, 30]
        data5 = [40, 30, 30, 30]
        arr = np.array([data1, data2, data3, data4, data5])

        vals = stats.mode(arr, axis=None)
        assert_equal(vals[0], np.array([30]))
        assert_equal(vals[1], np.array([8]))

        vals = stats.mode(arr, axis=0)
        assert_equal(vals[0], np.array([[10, 10, 30, 30]]))
        assert_equal(vals[1], np.array([[2, 3, 3, 2]]))

        vals = stats.mode(arr, axis=1)
        assert_equal(vals[0], np.array([[10], [10], [20], [30], [30]]))
        assert_equal(vals[1], np.array([[2], [4], [3], [4], [3]]))

    def test_strings(self):
        data1 = ['rain', 'showers', 'showers']
        vals = stats.mode(data1)
        assert_equal(vals[0][0], 'showers')
        assert_equal(vals[1][0], 2)

    def test_mixed_objects(self):
        objects = [10, True, np.nan, 'hello', 10]
        arr = np.empty((5,), dtype=object)
        arr[:] = objects
        vals = stats.mode(arr)
        assert_equal(vals[0][0], 10)
        assert_equal(vals[1][0], 2)

    def test_objects(self):
        # Python objects must be sortable (le + eq) and have ne defined
        # for np.unique to work. hash is for set.
        class Point:
            def __init__(self, x):
                self.x = x

            def __eq__(self, other):
                return self.x == other.x

            def __ne__(self, other):
                return self.x != other.x

            def __lt__(self, other):
                return self.x < other.x

            def __hash__(self):
                return hash(self.x)

        points = [Point(x) for x in [1, 2, 3, 4, 3, 2, 2, 2]]
        arr = np.empty((8,), dtype=object)
        arr[:] = points
        assert_(len(set(points)) == 4)
        assert_equal(np.unique(arr).shape, (4,))
        vals = stats.mode(arr)

        assert_equal(vals[0][0], Point(2))
        assert_equal(vals[1][0], 4)

    def test_mode_result_attributes(self):
        data1 = [3, 5, 1, 10, 23, 3, 2, 6, 8, 6, 10, 6]
        data2 = []
        actual = stats.mode(data1)
        attributes = ('mode', 'count')
        check_named_results(actual, attributes)
        actual2 = stats.mode(data2)
        check_named_results(actual2, attributes)

    def test_mode_nan(self):
        data1 = [3, np.nan, 5, 1, 10, 23, 3, 2, 6, 8, 6, 10, 6]
        actual = stats.mode(data1)
        assert_equal(actual, (6, 3))

        actual = stats.mode(data1, nan_policy='omit')
        assert_equal(actual, (6, 3))
        assert_raises(ValueError, stats.mode, data1, nan_policy='raise')
        assert_raises(ValueError, stats.mode, data1, nan_policy='foobar')

    @pytest.mark.parametrize("data", [
        [3, 5, 1, 1, 3],
        [3, np.nan, 5, 1, 1, 3],
        [3, 5, 1],
        [3, np.nan, 5, 1],
    ])
    def test_smallest_equal(self, data):
        result = stats.mode(data, nan_policy='omit')
        assert_equal(result[0][0], 1)

    def test_obj_arrays_ndim(self):
        # regression test for gh-9645: `mode` fails for object arrays w/ndim > 1
        data = [['Oxidation'], ['Oxidation'], ['Polymerization'], ['Reduction']]
        ar = np.array(data, dtype=object)
        m = stats.mode(ar, axis=0)
        assert np.all(m.mode == 'Oxidation') and m.mode.shape == (1, 1)
        assert np.all(m.count == 2) and m.count.shape == (1, 1)

        data1 = data + [[np.nan]]
        ar1 = np.array(data1, dtype=object)
        m = stats.mode(ar1, axis=0)
        assert np.all(m.mode == 'Oxidation') and m.mode.shape == (1, 1)
        assert np.all(m.count == 2) and m.count.shape == (1, 1)


class TestSEM:

    testcase = [1, 2, 3, 4]
    scalar_testcase = 4.

    def test_sem(self):
        # This is not in R, so used:
        #     sqrt(var(testcase)*3/4)/sqrt(3)

        # y = stats.sem(self.shoes[0])
        # assert_approx_equal(y,0.775177399)
        with suppress_warnings() as sup, np.errstate(invalid="ignore"):
            sup.filter(RuntimeWarning, "Degrees of freedom <= 0 for slice")
            y = stats.sem(self.scalar_testcase)
        assert_(np.isnan(y))

        y = stats.sem(self.testcase)
        assert_approx_equal(y, 0.6454972244)
        n = len(self.testcase)
        assert_allclose(stats.sem(self.testcase, ddof=0) * np.sqrt(n/(n-2)),
                        stats.sem(self.testcase, ddof=2))

        x = np.arange(10.)
        x[9] = np.nan
        assert_equal(stats.sem(x), np.nan)
        assert_equal(stats.sem(x, nan_policy='omit'), 0.9128709291752769)
        assert_raises(ValueError, stats.sem, x, nan_policy='raise')
        assert_raises(ValueError, stats.sem, x, nan_policy='foobar')


class TestZmapZscore:

    @pytest.mark.parametrize(
        'x, y',
        [([1, 2, 3, 4], [1, 2, 3, 4]),
         ([1, 2, 3], [0, 1, 2, 3, 4])]
    )
    def test_zmap(self, x, y):
        z = stats.zmap(x, y)
        # For these simple cases, calculate the expected result directly
        # by using the formula for the z-score.
        expected = (x - np.mean(y))/np.std(y)
        assert_allclose(z, expected, rtol=1e-12)

    def test_zmap_axis(self):
        # Test use of 'axis' keyword in zmap.
        x = np.array([[0.0, 0.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0, 2.0],
                      [2.0, 0.0, 2.0, 0.0]])

        t1 = 1.0/np.sqrt(2.0/3)
        t2 = np.sqrt(3.)/3
        t3 = np.sqrt(2.)

        z0 = stats.zmap(x, x, axis=0)
        z1 = stats.zmap(x, x, axis=1)

        z0_expected = [[-t1, -t3/2, -t3/2, 0.0],
                       [0.0, t3, -t3/2, t1],
                       [t1, -t3/2, t3, -t1]]
        z1_expected = [[-1.0, -1.0, 1.0, 1.0],
                       [-t2, -t2, -t2, np.sqrt(3.)],
                       [1.0, -1.0, 1.0, -1.0]]

        assert_array_almost_equal(z0, z0_expected)
        assert_array_almost_equal(z1, z1_expected)

    def test_zmap_ddof(self):
        # Test use of 'ddof' keyword in zmap.
        x = np.array([[0.0, 0.0, 1.0, 1.0],
                      [0.0, 1.0, 2.0, 3.0]])

        z = stats.zmap(x, x, axis=1, ddof=1)

        z0_expected = np.array([-0.5, -0.5, 0.5, 0.5])/(1.0/np.sqrt(3))
        z1_expected = np.array([-1.5, -0.5, 0.5, 1.5])/(np.sqrt(5./3))
        assert_array_almost_equal(z[0], z0_expected)
        assert_array_almost_equal(z[1], z1_expected)

    @pytest.mark.parametrize('ddof', [0, 2])
    def test_zmap_nan_policy_omit(self, ddof):
        # nans in `scores` are propagated, regardless of `nan_policy`.
        # `nan_policy` only affects how nans in `compare` are handled.
        scores = np.array([-3, -1, 2, np.nan])
        compare = np.array([-8, -3, 2, 7, 12, np.nan])
        z = stats.zmap(scores, compare, ddof=ddof, nan_policy='omit')
        assert_allclose(z, stats.zmap(scores, compare[~np.isnan(compare)],
                                      ddof=ddof))

    @pytest.mark.parametrize('ddof', [0, 2])
    def test_zmap_nan_policy_omit_with_axis(self, ddof):
        scores = np.arange(-5.0, 9.0).reshape(2, -1)
        compare = np.linspace(-8, 6, 24).reshape(2, -1)
        compare[0, 4] = np.nan
        compare[0, 6] = np.nan
        compare[1, 1] = np.nan
        z = stats.zmap(scores, compare, nan_policy='omit', axis=1, ddof=ddof)
        expected = np.array([stats.zmap(scores[0],
                                        compare[0][~np.isnan(compare[0])],
                                        ddof=ddof),
                             stats.zmap(scores[1],
                                        compare[1][~np.isnan(compare[1])],
                                        ddof=ddof)])
        assert_allclose(z, expected, rtol=1e-14)

    def test_zmap_nan_policy_raise(self):
        scores = np.array([1, 2, 3])
        compare = np.array([-8, -3, 2, 7, 12, np.nan])
        with pytest.raises(ValueError, match='input contains nan'):
            stats.zmap(scores, compare, nan_policy='raise')

    def test_zscore(self):
        # not in R, so tested by using:
        #    (testcase[i] - mean(testcase, axis=0)) / sqrt(var(testcase) * 3/4)
        y = stats.zscore([1, 2, 3, 4])
        desired = ([-1.3416407864999, -0.44721359549996, 0.44721359549996,
                    1.3416407864999])
        assert_array_almost_equal(desired, y, decimal=12)

    def test_zscore_axis(self):
        # Test use of 'axis' keyword in zscore.
        x = np.array([[0.0, 0.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0, 2.0],
                      [2.0, 0.0, 2.0, 0.0]])

        t1 = 1.0/np.sqrt(2.0/3)
        t2 = np.sqrt(3.)/3
        t3 = np.sqrt(2.)

        z0 = stats.zscore(x, axis=0)
        z1 = stats.zscore(x, axis=1)

        z0_expected = [[-t1, -t3/2, -t3/2, 0.0],
                       [0.0, t3, -t3/2, t1],
                       [t1, -t3/2, t3, -t1]]
        z1_expected = [[-1.0, -1.0, 1.0, 1.0],
                       [-t2, -t2, -t2, np.sqrt(3.)],
                       [1.0, -1.0, 1.0, -1.0]]

        assert_array_almost_equal(z0, z0_expected)
        assert_array_almost_equal(z1, z1_expected)

    def test_zscore_ddof(self):
        # Test use of 'ddof' keyword in zscore.
        x = np.array([[0.0, 0.0, 1.0, 1.0],
                      [0.0, 1.0, 2.0, 3.0]])

        z = stats.zscore(x, axis=1, ddof=1)

        z0_expected = np.array([-0.5, -0.5, 0.5, 0.5])/(1.0/np.sqrt(3))
        z1_expected = np.array([-1.5, -0.5, 0.5, 1.5])/(np.sqrt(5./3))
        assert_array_almost_equal(z[0], z0_expected)
        assert_array_almost_equal(z[1], z1_expected)

    def test_zscore_nan_propagate(self):
        x = np.array([1, 2, np.nan, 4, 5])
        z = stats.zscore(x, nan_policy='propagate')
        assert all(np.isnan(z))

    def test_zscore_nan_omit(self):
        x = np.array([1, 2, np.nan, 4, 5])

        z = stats.zscore(x, nan_policy='omit')

        expected = np.array([-1.2649110640673518,
                             -0.6324555320336759,
                             np.nan,
                             0.6324555320336759,
                             1.2649110640673518
                             ])
        assert_array_almost_equal(z, expected)

    def test_zscore_nan_omit_with_ddof(self):
        x = np.array([np.nan, 1.0, 3.0, 5.0, 7.0, 9.0])
        z = stats.zscore(x, ddof=1, nan_policy='omit')
        expected = np.r_[np.nan, stats.zscore(x[1:], ddof=1)]
        assert_allclose(z, expected, rtol=1e-13)

    def test_zscore_nan_raise(self):
        x = np.array([1, 2, np.nan, 4, 5])

        assert_raises(ValueError, stats.zscore, x, nan_policy='raise')

    def test_zscore_constant_input_1d(self):
        x = [-0.087] * 3
        z = stats.zscore(x)
        assert_equal(z, np.full(len(x), np.nan))

    def test_zscore_constant_input_2d(self):
        x = np.array([[10.0, 10.0, 10.0, 10.0],
                      [10.0, 11.0, 12.0, 13.0]])
        z0 = stats.zscore(x, axis=0)
        assert_equal(z0, np.array([[np.nan, -1.0, -1.0, -1.0],
                                   [np.nan, 1.0, 1.0, 1.0]]))
        z1 = stats.zscore(x, axis=1)
        assert_equal(z1, np.array([[np.nan, np.nan, np.nan, np.nan],
                                   stats.zscore(x[1])]))
        z = stats.zscore(x, axis=None)
        assert_equal(z, stats.zscore(x.ravel()).reshape(x.shape))

        y = np.ones((3, 6))
        z = stats.zscore(y, axis=None)
        assert_equal(z, np.full(y.shape, np.nan))

    def test_zscore_constant_input_2d_nan_policy_omit(self):
        x = np.array([[10.0, 10.0, 10.0, 10.0],
                      [10.0, 11.0, 12.0, np.nan],
                      [10.0, 12.0, np.nan, 10.0]])
        z0 = stats.zscore(x, nan_policy='omit', axis=0)
        s = np.sqrt(3/2)
        s2 = np.sqrt(2)
        assert_allclose(z0, np.array([[np.nan, -s, -1.0, np.nan],
                                      [np.nan, 0, 1.0, np.nan],
                                      [np.nan, s, np.nan, np.nan]]))
        z1 = stats.zscore(x, nan_policy='omit', axis=1)
        assert_allclose(z1, np.array([[np.nan, np.nan, np.nan, np.nan],
                                      [-s, 0, s, np.nan],
                                      [-s2/2, s2, np.nan, -s2/2]]))

    def test_zscore_2d_all_nan_row(self):
        # A row is all nan, and we use axis=1.
        x = np.array([[np.nan, np.nan, np.nan, np.nan],
                      [10.0, 10.0, 12.0, 12.0]])
        z = stats.zscore(x, nan_policy='omit', axis=1)
        assert_equal(z, np.array([[np.nan, np.nan, np.nan, np.nan],
                                  [-1.0, -1.0, 1.0, 1.0]]))

    def test_zscore_2d_all_nan(self):
        # The entire 2d array is nan, and we use axis=None.
        y = np.full((2, 3), np.nan)
        z = stats.zscore(y, nan_policy='omit', axis=None)
        assert_equal(z, y)

    @pytest.mark.parametrize('x', [np.array([]), np.zeros((3, 0, 5))])
    def test_zscore_empty_input(self, x):
        z = stats.zscore(x)
        assert_equal(z, x)


class TestMedianAbsDeviation:
    def setup_class(self):
        self.dat_nan = np.array([2.20, 2.20, 2.4, 2.4, 2.5, 2.7, 2.8, 2.9,
                                 3.03, 3.03, 3.10, 3.37, 3.4, 3.4, 3.4, 3.5,
                                 3.6, 3.7, 3.7, 3.7, 3.7, 3.77, 5.28, np.nan])
        self.dat = np.array([2.20, 2.20, 2.4, 2.4, 2.5, 2.7, 2.8, 2.9, 3.03,
                             3.03, 3.10, 3.37, 3.4, 3.4, 3.4, 3.5, 3.6, 3.7,
                             3.7, 3.7, 3.7, 3.77, 5.28, 28.95])

    def test_median_abs_deviation(self):
        assert_almost_equal(stats.median_abs_deviation(self.dat, axis=None),
                            0.355)
        dat = self.dat.reshape(6, 4)
        mad = stats.median_abs_deviation(dat, axis=0)
        mad_expected = np.asarray([0.435, 0.5, 0.45, 0.4])
        assert_array_almost_equal(mad, mad_expected)

    def test_mad_nan_omit(self):
        mad = stats.median_abs_deviation(self.dat_nan, nan_policy='omit')
        assert_almost_equal(mad, 0.34)

    def test_axis_and_nan(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0, np.nan],
                      [1.0, 4.0, 5.0, 8.0, 9.0]])
        mad = stats.median_abs_deviation(x, axis=1)
        assert_equal(mad, np.array([np.nan, 3.0]))

    def test_nan_policy_omit_with_inf(sef):
        z = np.array([1, 3, 4, 6, 99, np.nan, np.inf])
        mad = stats.median_abs_deviation(z, nan_policy='omit')
        assert_equal(mad, 3.0)

    @pytest.mark.parametrize('axis', [0, 1, 2, None])
    def test_size_zero_with_axis(self, axis):
        x = np.zeros((3, 0, 4))
        mad = stats.median_abs_deviation(x, axis=axis)
        assert_equal(mad, np.full_like(x.sum(axis=axis), fill_value=np.nan))

    @pytest.mark.parametrize('nan_policy, expected',
                             [('omit', np.array([np.nan, 1.5, 1.5])),
                              ('propagate', np.array([np.nan, np.nan, 1.5]))])
    def test_nan_policy_with_axis(self, nan_policy, expected):
        x = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                      [1, 5, 3, 6, np.nan, np.nan],
                      [5, 6, 7, 9, 9, 10]])
        mad = stats.median_abs_deviation(x, nan_policy=nan_policy, axis=1)
        assert_equal(mad, expected)

    @pytest.mark.parametrize('axis, expected',
                             [(1, [2.5, 2.0, 12.0]), (None, 4.5)])
    def test_center_mean_with_nan(self, axis, expected):
        x = np.array([[1, 2, 4, 9, np.nan],
                      [0, 1, 1, 1, 12],
                      [-10, -10, -10, 20, 20]])
        mad = stats.median_abs_deviation(x, center=np.mean, nan_policy='omit',
                                         axis=axis)
        assert_allclose(mad, expected, rtol=1e-15, atol=1e-15)

    def test_center_not_callable(self):
        with pytest.raises(TypeError, match='callable'):
            stats.median_abs_deviation([1, 2, 3, 5], center=99)


class TestMedianAbsoluteDeviation:
    def setup_class(self):
        self.dat_nan = np.array([2.20, 2.20, 2.4, 2.4, 2.5, 2.7, 2.8, 2.9, 3.03,
                3.03, 3.10, 3.37, 3.4, 3.4, 3.4, 3.5, 3.6, 3.7, 3.7,
                3.7, 3.7, 3.77, 5.28, np.nan])
        self.dat = np.array([2.20, 2.20, 2.4, 2.4, 2.5, 2.7, 2.8, 2.9, 3.03,
                3.03, 3.10, 3.37, 3.4, 3.4, 3.4, 3.5, 3.6, 3.7, 3.7,
                3.7, 3.7, 3.77, 5.28, 28.95])

    def test_mad_empty(self):
        dat = []
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            mad = stats.median_absolute_deviation(dat)
        assert_equal(mad, np.nan)

    def test_mad_nan_shape1(self):
        z = np.ones((3, 0))
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            mad_axis0 = stats.median_absolute_deviation(z, axis=0)
            mad_axis1 = stats.median_absolute_deviation(z, axis=1)
        assert_equal(mad_axis0, np.nan)
        assert_equal(mad_axis1, np.array([np.nan, np.nan, np.nan]))
        assert_equal(mad_axis1.shape, (3,))

    def test_mad_nan_shape2(self):
        z = np.ones((3, 0, 2))
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            mad_axis0 = stats.median_absolute_deviation(z, axis=0)
            mad_axis1 = stats.median_absolute_deviation(z, axis=1)
            mad_axis2 = stats.median_absolute_deviation(z, axis=2)
        assert_equal(mad_axis0, np.nan)
        assert_equal(mad_axis1, np.array([[np.nan, np.nan],
                                          [np.nan, np.nan],
                                          [np.nan, np.nan]]))
        assert_equal(mad_axis1.shape, (3, 2))
        assert_equal(mad_axis2, np.nan)

    def test_mad_nan_propagate(self):
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            mad = stats.median_absolute_deviation(self.dat_nan,
                                                  nan_policy='propagate')
        assert_equal(mad, np.nan)

    def test_mad_nan_raise(self):
        with assert_raises(ValueError):
            with suppress_warnings() as sup:
                sup.filter(DeprecationWarning)
                stats.median_absolute_deviation(self.dat_nan,
                                                nan_policy='raise')

    def test_mad_scale_default(self):
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            mad = stats.median_absolute_deviation(self.dat, scale=1.0)
            mad_float = stats.median_absolute_deviation(self.dat, scale=1.0)
        assert_almost_equal(mad, 0.355)
        assert_almost_equal(mad, mad_float)

    def test_mad_scale_normal(self):
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            mad = stats.median_absolute_deviation(self.dat, scale="normal")
            scale = 1.4826022185056018
            mad_float = stats.median_absolute_deviation(self.dat, scale=scale)
        assert_almost_equal(mad, 0.526323787)
        assert_almost_equal(mad, mad_float)


def _check_warnings(warn_list, expected_type, expected_len):
    """
    Checks that all of the warnings from a list returned by
    `warnings.catch_all(record=True)` are of the required type and that the list
    contains expected number of warnings.
    """
    assert_equal(len(warn_list), expected_len, "number of warnings")
    for warn_ in warn_list:
        assert_(warn_.category is expected_type)


class TestIQR:

    def test_basic(self):
        x = np.arange(8) * 0.5
        np.random.shuffle(x)
        assert_equal(stats.iqr(x), 1.75)

    def test_api(self):
        d = np.ones((5, 5))
        stats.iqr(d)
        stats.iqr(d, None)
        stats.iqr(d, 1)
        stats.iqr(d, (0, 1))
        stats.iqr(d, None, (10, 90))
        stats.iqr(d, None, (30, 20), 1.0)
        stats.iqr(d, None, (25, 75), 1.5, 'propagate')
        stats.iqr(d, None, (50, 50), 'normal', 'raise', 'linear')
        stats.iqr(d, None, (25, 75), -0.4, 'omit', 'lower', True)

    def test_empty(self):
        assert_equal(stats.iqr([]), np.nan)
        assert_equal(stats.iqr(np.arange(0)), np.nan)

    def test_constant(self):
        # Constant array always gives 0
        x = np.ones((7, 4))
        assert_equal(stats.iqr(x), 0.0)
        assert_array_equal(stats.iqr(x, axis=0), np.zeros(4))
        assert_array_equal(stats.iqr(x, axis=1), np.zeros(7))
        assert_equal(stats.iqr(x, interpolation='linear'), 0.0)
        assert_equal(stats.iqr(x, interpolation='midpoint'), 0.0)
        assert_equal(stats.iqr(x, interpolation='nearest'), 0.0)
        assert_equal(stats.iqr(x, interpolation='lower'), 0.0)
        assert_equal(stats.iqr(x, interpolation='higher'), 0.0)

        # 0 only along constant dimensions
        # This also tests much of `axis`
        y = np.ones((4, 5, 6)) * np.arange(6)
        assert_array_equal(stats.iqr(y, axis=0), np.zeros((5, 6)))
        assert_array_equal(stats.iqr(y, axis=1), np.zeros((4, 6)))
        assert_array_equal(stats.iqr(y, axis=2), np.full((4, 5), 2.5))
        assert_array_equal(stats.iqr(y, axis=(0, 1)), np.zeros(6))
        assert_array_equal(stats.iqr(y, axis=(0, 2)), np.full(5, 3.))
        assert_array_equal(stats.iqr(y, axis=(1, 2)), np.full(4, 3.))

    def test_scalarlike(self):
        x = np.arange(1) + 7.0
        assert_equal(stats.iqr(x[0]), 0.0)
        assert_equal(stats.iqr(x), 0.0)
        assert_array_equal(stats.iqr(x, keepdims=True), [0.0])

    def test_2D(self):
        x = np.arange(15).reshape((3, 5))
        assert_equal(stats.iqr(x), 7.0)
        assert_array_equal(stats.iqr(x, axis=0), np.full(5, 5.))
        assert_array_equal(stats.iqr(x, axis=1), np.full(3, 2.))
        assert_array_equal(stats.iqr(x, axis=(0, 1)), 7.0)
        assert_array_equal(stats.iqr(x, axis=(1, 0)), 7.0)

    def test_axis(self):
        # The `axis` keyword is also put through its paces in `test_keepdims`.
        o = np.random.normal(size=(71, 23))
        x = np.dstack([o] * 10)                 # x.shape = (71, 23, 10)
        q = stats.iqr(o)

        assert_equal(stats.iqr(x, axis=(0, 1)), q)
        x = np.rollaxis(x, -1, 0)               # x.shape = (10, 71, 23)
        assert_equal(stats.iqr(x, axis=(2, 1)), q)
        x = x.swapaxes(0, 1)                    # x.shape = (71, 10, 23)
        assert_equal(stats.iqr(x, axis=(0, 2)), q)
        x = x.swapaxes(0, 1)                    # x.shape = (10, 71, 23)

        assert_equal(stats.iqr(x, axis=(0, 1, 2)),
                     stats.iqr(x, axis=None))
        assert_equal(stats.iqr(x, axis=(0,)),
                     stats.iqr(x, axis=0))

        d = np.arange(3 * 5 * 7 * 11)
        # Older versions of numpy only shuffle along axis=0.
        # Not sure about newer, don't care.
        np.random.shuffle(d)
        d = d.reshape((3, 5, 7, 11))
        assert_equal(stats.iqr(d, axis=(0, 1, 2))[0],
                     stats.iqr(d[:,:,:, 0].ravel()))
        assert_equal(stats.iqr(d, axis=(0, 1, 3))[1],
                     stats.iqr(d[:,:, 1,:].ravel()))
        assert_equal(stats.iqr(d, axis=(3, 1, -4))[2],
                     stats.iqr(d[:,:, 2,:].ravel()))
        assert_equal(stats.iqr(d, axis=(3, 1, 2))[2],
                     stats.iqr(d[2,:,:,:].ravel()))
        assert_equal(stats.iqr(d, axis=(3, 2))[2, 1],
                     stats.iqr(d[2, 1,:,:].ravel()))
        assert_equal(stats.iqr(d, axis=(1, -2))[2, 1],
                     stats.iqr(d[2, :, :, 1].ravel()))
        assert_equal(stats.iqr(d, axis=(1, 3))[2, 2],
                     stats.iqr(d[2, :, 2,:].ravel()))

        assert_raises(np.AxisError, stats.iqr, d, axis=4)
        assert_raises(ValueError, stats.iqr, d, axis=(0, 0))

    def test_rng(self):
        x = np.arange(5)
        assert_equal(stats.iqr(x), 2)
        assert_equal(stats.iqr(x, rng=(25, 87.5)), 2.5)
        assert_equal(stats.iqr(x, rng=(12.5, 75)), 2.5)
        assert_almost_equal(stats.iqr(x, rng=(10, 50)), 1.6)  # 3-1.4

        assert_raises(ValueError, stats.iqr, x, rng=(0, 101))
        assert_raises(ValueError, stats.iqr, x, rng=(np.nan, 25))
        assert_raises(TypeError, stats.iqr, x, rng=(0, 50, 60))

    def test_interpolation(self):
        x = np.arange(5)
        y = np.arange(4)
        # Default
        assert_equal(stats.iqr(x), 2)
        assert_equal(stats.iqr(y), 1.5)
        # Linear
        assert_equal(stats.iqr(x, interpolation='linear'), 2)
        assert_equal(stats.iqr(y, interpolation='linear'), 1.5)
        # Higher
        assert_equal(stats.iqr(x, interpolation='higher'), 2)
        assert_equal(stats.iqr(x, rng=(25, 80), interpolation='higher'), 3)
        assert_equal(stats.iqr(y, interpolation='higher'), 2)
        # Lower (will generally, but not always be the same as higher)
        assert_equal(stats.iqr(x, interpolation='lower'), 2)
        assert_equal(stats.iqr(x, rng=(25, 80), interpolation='lower'), 2)
        assert_equal(stats.iqr(y, interpolation='lower'), 2)
        # Nearest
        assert_equal(stats.iqr(x, interpolation='nearest'), 2)
        assert_equal(stats.iqr(y, interpolation='nearest'), 1)
        # Midpoint
        assert_equal(stats.iqr(x, interpolation='midpoint'), 2)
        assert_equal(stats.iqr(x, rng=(25, 80), interpolation='midpoint'), 2.5)
        assert_equal(stats.iqr(y, interpolation='midpoint'), 2)

        assert_raises(ValueError, stats.iqr, x, interpolation='foobar')

    def test_keepdims(self):
        # Also tests most of `axis`
        x = np.ones((3, 5, 7, 11))
        assert_equal(stats.iqr(x, axis=None, keepdims=False).shape, ())
        assert_equal(stats.iqr(x, axis=2, keepdims=False).shape, (3, 5, 11))
        assert_equal(stats.iqr(x, axis=(0, 1), keepdims=False).shape, (7, 11))
        assert_equal(stats.iqr(x, axis=(0, 3), keepdims=False).shape, (5, 7))
        assert_equal(stats.iqr(x, axis=(1,), keepdims=False).shape, (3, 7, 11))
        assert_equal(stats.iqr(x, (0, 1, 2, 3), keepdims=False).shape, ())
        assert_equal(stats.iqr(x, axis=(0, 1, 3), keepdims=False).shape, (7,))

        assert_equal(stats.iqr(x, axis=None, keepdims=True).shape, (1, 1, 1, 1))
        assert_equal(stats.iqr(x, axis=2, keepdims=True).shape, (3, 5, 1, 11))
        assert_equal(stats.iqr(x, axis=(0, 1), keepdims=True).shape, (1, 1, 7, 11))
        assert_equal(stats.iqr(x, axis=(0, 3), keepdims=True).shape, (1, 5, 7, 1))
        assert_equal(stats.iqr(x, axis=(1,), keepdims=True).shape, (3, 1, 7, 11))
        assert_equal(stats.iqr(x, (0, 1, 2, 3), keepdims=True).shape, (1, 1, 1, 1))
        assert_equal(stats.iqr(x, axis=(0, 1, 3), keepdims=True).shape, (1, 1, 7, 1))

    def test_nanpolicy(self):
        x = np.arange(15.0).reshape((3, 5))

        # No NaNs
        assert_equal(stats.iqr(x, nan_policy='propagate'), 7)
        assert_equal(stats.iqr(x, nan_policy='omit'), 7)
        assert_equal(stats.iqr(x, nan_policy='raise'), 7)

        # Yes NaNs
        x[1, 2] = np.nan
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert_equal(stats.iqr(x, nan_policy='propagate'), np.nan)
            assert_equal(stats.iqr(x, axis=0, nan_policy='propagate'), [5, 5, np.nan, 5, 5])
            assert_equal(stats.iqr(x, axis=1, nan_policy='propagate'), [2, np.nan, 2])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert_equal(stats.iqr(x, nan_policy='omit'), 7.5)
            assert_equal(stats.iqr(x, axis=0, nan_policy='omit'), np.full(5, 5))
            assert_equal(stats.iqr(x, axis=1, nan_policy='omit'), [2, 2.5, 2])

        assert_raises(ValueError, stats.iqr, x, nan_policy='raise')
        assert_raises(ValueError, stats.iqr, x, axis=0, nan_policy='raise')
        assert_raises(ValueError, stats.iqr, x, axis=1, nan_policy='raise')

        # Bad policy
        assert_raises(ValueError, stats.iqr, x, nan_policy='barfood')

    def test_scale(self):
        x = np.arange(15.0).reshape((3, 5))

        # No NaNs
        assert_equal(stats.iqr(x, scale=1.0), 7)
        assert_almost_equal(stats.iqr(x, scale='normal'), 7 / 1.3489795)
        assert_equal(stats.iqr(x, scale=2.0), 3.5)

        # Yes NaNs
        x[1, 2] = np.nan
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert_equal(stats.iqr(x, scale=1.0, nan_policy='propagate'), np.nan)
            assert_equal(stats.iqr(x, scale='normal', nan_policy='propagate'), np.nan)
            assert_equal(stats.iqr(x, scale=2.0, nan_policy='propagate'), np.nan)
            # axis=1 chosen to show behavior with both nans and without
            assert_equal(stats.iqr(x, axis=1, scale=1.0,
                                   nan_policy='propagate'), [2, np.nan, 2])
            assert_almost_equal(stats.iqr(x, axis=1, scale='normal',
                                          nan_policy='propagate'),
                                np.array([2, np.nan, 2]) / 1.3489795)
            assert_equal(stats.iqr(x, axis=1, scale=2.0, nan_policy='propagate'),
                         [1, np.nan, 1])
            # Since NumPy 1.17.0.dev, warnings are no longer emitted by
            # np.percentile with nans, so we don't check the number of
            # warnings here. See https://github.com/numpy/numpy/pull/12679.

        assert_equal(stats.iqr(x, scale=1.0, nan_policy='omit'), 7.5)
        assert_almost_equal(stats.iqr(x, scale='normal', nan_policy='omit'),
                            7.5 / 1.3489795)
        assert_equal(stats.iqr(x, scale=2.0, nan_policy='omit'), 3.75)

        # Bad scale
        assert_raises(ValueError, stats.iqr, x, scale='foobar')


class TestMoments:
    """
        Comparison numbers are found using R v.1.5.1
        note that length(testcase) = 4
        testmathworks comes from documentation for the
        Statistics Toolbox for Matlab and can be found at both
        https://www.mathworks.com/help/stats/kurtosis.html
        https://www.mathworks.com/help/stats/skewness.html
        Note that both test cases came from here.
    """
    testcase = [1,2,3,4]
    scalar_testcase = 4.
    np.random.seed(1234)
    testcase_moment_accuracy = np.random.rand(42)
    testmathworks = [1.165, 0.6268, 0.0751, 0.3516, -0.6965]

    def _assert_equal(self, actual, expect, *, shape=None, dtype=None):
        expect = np.asarray(expect)
        if shape is not None:
            expect = np.broadcast_to(expect, shape)
        assert_array_equal(actual, expect)
        if dtype is None:
            dtype = expect.dtype
        assert actual.dtype == dtype

    def test_moment(self):
        # mean((testcase-mean(testcase))**power,axis=0),axis=0))**power))
        y = stats.moment(self.scalar_testcase)
        assert_approx_equal(y, 0.0)
        y = stats.moment(self.testcase, 0)
        assert_approx_equal(y, 1.0)
        y = stats.moment(self.testcase, 1)
        assert_approx_equal(y, 0.0, 10)
        y = stats.moment(self.testcase, 2)
        assert_approx_equal(y, 1.25)
        y = stats.moment(self.testcase, 3)
        assert_approx_equal(y, 0.0)
        y = stats.moment(self.testcase, 4)
        assert_approx_equal(y, 2.5625)

        # check array_like input for moment
        y = stats.moment(self.testcase, [1, 2, 3, 4])
        assert_allclose(y, [0, 1.25, 0, 2.5625])

        # check moment input consists only of integers
        y = stats.moment(self.testcase, 0.0)
        assert_approx_equal(y, 1.0)
        assert_raises(ValueError, stats.moment, self.testcase, 1.2)
        y = stats.moment(self.testcase, [1.0, 2, 3, 4.0])
        assert_allclose(y, [0, 1.25, 0, 2.5625])

        # test empty input
        y = stats.moment([])
        self._assert_equal(y, np.nan, dtype=np.float64)
        y = stats.moment(np.array([], dtype=np.float32))
        self._assert_equal(y, np.nan, dtype=np.float32)
        y = stats.moment(np.zeros((1, 0)), axis=0)
        self._assert_equal(y, [], shape=(0,), dtype=np.float64)
        y = stats.moment([[]], axis=1)
        self._assert_equal(y, np.nan, shape=(1,), dtype=np.float64)
        y = stats.moment([[]], moment=[0, 1], axis=0)
        self._assert_equal(y, [], shape=(2, 0))

        x = np.arange(10.)
        x[9] = np.nan
        assert_equal(stats.moment(x, 2), np.nan)
        assert_almost_equal(stats.moment(x, nan_policy='omit'), 0.0)
        assert_raises(ValueError, stats.moment, x, nan_policy='raise')
        assert_raises(ValueError, stats.moment, x, nan_policy='foobar')

    @pytest.mark.parametrize('dtype', [np.float32, np.float64, np.complex128])
    @pytest.mark.parametrize('expect, moment', [(0, 1), (1, 0)])
    def test_constant_moments(self, dtype, expect, moment):
        x = np.random.rand(5).astype(dtype)
        y = stats.moment(x, moment=moment)
        self._assert_equal(y, expect, dtype=dtype)

        y = stats.moment(np.broadcast_to(x, (6, 5)), axis=0, moment=moment)
        self._assert_equal(y, expect, shape=(5,), dtype=dtype)

        y = stats.moment(np.broadcast_to(x, (1, 2, 3, 4, 5)), axis=2,
                         moment=moment)
        self._assert_equal(y, expect, shape=(1, 2, 4, 5), dtype=dtype)

        y = stats.moment(np.broadcast_to(x, (1, 2, 3, 4, 5)), axis=None,
                         moment=moment)
        self._assert_equal(y, expect, shape=(), dtype=dtype)


    def test_moment_propagate_nan(self):
        # Check that the shape of the result is the same for inputs
        # with and without nans, cf gh-5817
        a = np.arange(8).reshape(2, -1).astype(float)
        a[1, 0] = np.nan
        mm = stats.moment(a, 2, axis=1, nan_policy="propagate")
        np.testing.assert_allclose(mm, [1.25, np.nan], atol=1e-15)

    def test_variation(self):
        # variation = samplestd / mean
        y = stats.variation(self.scalar_testcase)
        assert_approx_equal(y, 0.0)
        y = stats.variation(self.testcase)
        assert_approx_equal(y, 0.44721359549996, 10)

        x = np.arange(10.)
        x[9] = np.nan
        assert_equal(stats.variation(x), np.nan)
        assert_almost_equal(stats.variation(x, nan_policy='omit'),
                            0.6454972243679028)
        assert_raises(ValueError, stats.variation, x, nan_policy='raise')
        assert_raises(ValueError, stats.variation, x, nan_policy='foobar')

    def test_variation_propagate_nan(self):
        # Check that the shape of the result is the same for inputs
        # with and without nans, cf gh-5817
        a = np.arange(8).reshape(2, -1).astype(float)
        a[1, 0] = np.nan
        vv = stats.variation(a, axis=1, nan_policy="propagate")
        np.testing.assert_allclose(vv, [0.7453559924999299, np.nan], atol=1e-15)

    def test_variation_ddof(self):
        # test variation with delta degrees of freedom
        # regression test for gh-13341
        a = array([1, 2, 3, 4, 5])
        nan_a = array([1, 2, 3, np.nan, 4, 5, np.nan])
        y = stats.variation(a, ddof=1)
        nan_y = stats.variation(nan_a, nan_policy="omit", ddof=1)
        assert_approx_equal(y, 0.5270462766947299)
        np.testing.assert_equal(y, nan_y)

    def test_skewness(self):
        # Scalar test case
        y = stats.skew(self.scalar_testcase)
        assert_approx_equal(y, 0.0)
        # sum((testmathworks-mean(testmathworks,axis=0))**3,axis=0) /
        #     ((sqrt(var(testmathworks)*4/5))**3)/5
        y = stats.skew(self.testmathworks)
        assert_approx_equal(y, -0.29322304336607, 10)
        y = stats.skew(self.testmathworks, bias=0)
        assert_approx_equal(y, -0.437111105023940, 10)
        y = stats.skew(self.testcase)
        assert_approx_equal(y, 0.0, 10)

        x = np.arange(10.)
        x[9] = np.nan
        with np.errstate(invalid='ignore'):
            assert_equal(stats.skew(x), np.nan)
        assert_equal(stats.skew(x, nan_policy='omit'), 0.)
        assert_raises(ValueError, stats.skew, x, nan_policy='raise')
        assert_raises(ValueError, stats.skew, x, nan_policy='foobar')

    def test_skewness_scalar(self):
        # `skew` must return a scalar for 1-dim input
        assert_equal(stats.skew(arange(10)), 0.0)

    def test_skew_propagate_nan(self):
        # Check that the shape of the result is the same for inputs
        # with and without nans, cf gh-5817
        a = np.arange(8).reshape(2, -1).astype(float)
        a[1, 0] = np.nan
        with np.errstate(invalid='ignore'):
            s = stats.skew(a, axis=1, nan_policy="propagate")
        np.testing.assert_allclose(s, [0, np.nan], atol=1e-15)

    def test_skew_constant_value(self):
        # Skewness of a constant input should be zero even when the mean is not
        # exact (gh-13245)
        a = np.repeat(-0.27829495, 10)
        assert stats.skew(a) == 0.0
        assert stats.skew(a * float(2**50)) == 0.0
        assert stats.skew(a / float(2**50)) == 0.0
        assert stats.skew(a, bias=False) == 0.0

        # similarly, from gh-11086:
        assert stats.skew([14.3]*7) == 0.0
        assert stats.skew(1 + np.arange(-3, 4)*1e-16) == 0

    def test_kurtosis(self):
        # Scalar test case
        y = stats.kurtosis(self.scalar_testcase)
        assert_approx_equal(y, -3.0)
        #   sum((testcase-mean(testcase,axis=0))**4,axis=0)/((sqrt(var(testcase)*3/4))**4)/4
        #   sum((test2-mean(testmathworks,axis=0))**4,axis=0)/((sqrt(var(testmathworks)*4/5))**4)/5
        #   Set flags for axis = 0 and
        #   fisher=0 (Pearson's defn of kurtosis for compatibility with Matlab)
        y = stats.kurtosis(self.testmathworks, 0, fisher=0, bias=1)
        assert_approx_equal(y, 2.1658856802973, 10)

        # Note that MATLAB has confusing docs for the following case
        #  kurtosis(x,0) gives an unbiased estimate of Pearson's skewness
        #  kurtosis(x)  gives a biased estimate of Fisher's skewness (Pearson-3)
        #  The MATLAB docs imply that both should give Fisher's
        y = stats.kurtosis(self.testmathworks, fisher=0, bias=0)
        assert_approx_equal(y, 3.663542721189047, 10)
        y = stats.kurtosis(self.testcase, 0, 0)
        assert_approx_equal(y, 1.64)

        x = np.arange(10.)
        x[9] = np.nan
        assert_equal(stats.kurtosis(x), np.nan)
        assert_almost_equal(stats.kurtosis(x, nan_policy='omit'), -1.230000)
        assert_raises(ValueError, stats.kurtosis, x, nan_policy='raise')
        assert_raises(ValueError, stats.kurtosis, x, nan_policy='foobar')

    def test_kurtosis_array_scalar(self):
        assert_equal(type(stats.kurtosis([1,2,3])), float)

    def test_kurtosis_propagate_nan(self):
        # Check that the shape of the result is the same for inputs
        # with and without nans, cf gh-5817
        a = np.arange(8).reshape(2, -1).astype(float)
        a[1, 0] = np.nan
        k = stats.kurtosis(a, axis=1, nan_policy="propagate")
        np.testing.assert_allclose(k, [-1.36, np.nan], atol=1e-15)

    def test_kurtosis_constant_value(self):
        # Kurtosis of a constant input should be zero, even when the mean is not
        # exact (gh-13245)
        a = np.repeat(-0.27829495, 10)
        assert stats.kurtosis(a, fisher=False) == 0.0
        assert stats.kurtosis(a * float(2**50), fisher=False) == 0.0
        assert stats.kurtosis(a / float(2**50), fisher=False) == 0.0
        assert stats.kurtosis(a, fisher=False, bias=False) == 0.0

    def test_moment_accuracy(self):
        # 'moment' must have a small enough error compared to the slower
        #  but very accurate numpy.power() implementation.
        tc_no_mean = self.testcase_moment_accuracy - \
                     np.mean(self.testcase_moment_accuracy)
        assert_allclose(np.power(tc_no_mean, 42).mean(),
                            stats.moment(self.testcase_moment_accuracy, 42))


class TestStudentTest:
    X1 = np.array([-1, 0, 1])
    X2 = np.array([0, 1, 2])
    T1_0 = 0
    P1_0 = 1
    T1_1 = -1.7320508075
    P1_1 = 0.22540333075
    T1_2 = -3.464102
    P1_2 = 0.0741799
    T2_0 = 1.732051
    P2_0 = 0.2254033
    P1_1_l = P1_1 / 2
    P1_1_g = 1 - (P1_1 / 2)

    def test_onesample(self):
        with suppress_warnings() as sup, np.errstate(invalid="ignore"):
            sup.filter(RuntimeWarning, "Degrees of freedom <= 0 for slice")
            t, p = stats.ttest_1samp(4., 3.)
        assert_(np.isnan(t))
        assert_(np.isnan(p))

        t, p = stats.ttest_1samp(self.X1, 0)

        assert_array_almost_equal(t, self.T1_0)
        assert_array_almost_equal(p, self.P1_0)

        res = stats.ttest_1samp(self.X1, 0)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes)

        t, p = stats.ttest_1samp(self.X2, 0)

        assert_array_almost_equal(t, self.T2_0)
        assert_array_almost_equal(p, self.P2_0)

        t, p = stats.ttest_1samp(self.X1, 1)

        assert_array_almost_equal(t, self.T1_1)
        assert_array_almost_equal(p, self.P1_1)

        t, p = stats.ttest_1samp(self.X1, 2)

        assert_array_almost_equal(t, self.T1_2)
        assert_array_almost_equal(p, self.P1_2)

        # check nan policy
        x = stats.norm.rvs(loc=5, scale=10, size=51, random_state=7654567)
        x[50] = np.nan
        with np.errstate(invalid="ignore"):
            assert_array_equal(stats.ttest_1samp(x, 5.0), (np.nan, np.nan))

            assert_array_almost_equal(stats.ttest_1samp(x, 5.0, nan_policy='omit'),
                                      (-1.6412624074367159, 0.107147027334048005))
            assert_raises(ValueError, stats.ttest_1samp, x, 5.0, nan_policy='raise')
            assert_raises(ValueError, stats.ttest_1samp, x, 5.0,
                          nan_policy='foobar')

    def test_1samp_alternative(self):
        assert_raises(ValueError, stats.ttest_1samp, self.X1, 0,
                      alternative="error")

        t, p = stats.ttest_1samp(self.X1, 1, alternative="less")
        assert_allclose(p, self.P1_1_l)
        assert_allclose(t, self.T1_1)

        t, p = stats.ttest_1samp(self.X1, 1, alternative="greater")
        assert_allclose(p, self.P1_1_g)
        assert_allclose(t, self.T1_1)

def test_percentileofscore():
    pcos = stats.percentileofscore

    assert_equal(pcos([1,2,3,4,5,6,7,8,9,10],4), 40.0)

    for (kind, result) in [('mean', 35.0),
                           ('strict', 30.0),
                           ('weak', 40.0)]:
        assert_equal(pcos(np.arange(10) + 1, 4, kind=kind), result)

    # multiple - 2
    for (kind, result) in [('rank', 45.0),
                           ('strict', 30.0),
                           ('weak', 50.0),
                           ('mean', 40.0)]:
        assert_equal(pcos([1,2,3,4,4,5,6,7,8,9], 4, kind=kind), result)

    # multiple - 3
    assert_equal(pcos([1,2,3,4,4,4,5,6,7,8], 4), 50.0)
    for (kind, result) in [('rank', 50.0),
                           ('mean', 45.0),
                           ('strict', 30.0),
                           ('weak', 60.0)]:

        assert_equal(pcos([1,2,3,4,4,4,5,6,7,8], 4, kind=kind), result)

    # missing
    for kind in ('rank', 'mean', 'strict', 'weak'):
        assert_equal(pcos([1,2,3,5,6,7,8,9,10,11], 4, kind=kind), 30)

    # larger numbers
    for (kind, result) in [('mean', 35.0),
                           ('strict', 30.0),
                           ('weak', 40.0)]:
        assert_equal(
              pcos([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 40,
                   kind=kind), result)

    for (kind, result) in [('mean', 45.0),
                           ('strict', 30.0),
                           ('weak', 60.0)]:
        assert_equal(
              pcos([10, 20, 30, 40, 40, 40, 50, 60, 70, 80],
                   40, kind=kind), result)

    for kind in ('rank', 'mean', 'strict', 'weak'):
        assert_equal(
              pcos([10, 20, 30, 50, 60, 70, 80, 90, 100, 110],
                   40, kind=kind), 30.0)

    # boundaries
    for (kind, result) in [('rank', 10.0),
                           ('mean', 5.0),
                           ('strict', 0.0),
                           ('weak', 10.0)]:
        assert_equal(
              pcos([10, 20, 30, 50, 60, 70, 80, 90, 100, 110],
                   10, kind=kind), result)

    for (kind, result) in [('rank', 100.0),
                           ('mean', 95.0),
                           ('strict', 90.0),
                           ('weak', 100.0)]:
        assert_equal(
              pcos([10, 20, 30, 50, 60, 70, 80, 90, 100, 110],
                   110, kind=kind), result)

    # out of bounds
    for (kind, score, result) in [('rank', 200, 100.0),
                                  ('mean', 200, 100.0),
                                  ('mean', 0, 0.0)]:
        assert_equal(
              pcos([10, 20, 30, 50, 60, 70, 80, 90, 100, 110],
                   score, kind=kind), result)

    assert_raises(ValueError, pcos, [1, 2, 3, 3, 4], 3, kind='unrecognized')


PowerDivCase = namedtuple('Case',  # type: ignore[name-match]
                          ['f_obs', 'f_exp', 'ddof', 'axis',
                           'chi2',     # Pearson's
                           'log',      # G-test (log-likelihood)
                           'mod_log',  # Modified log-likelihood
                           'cr',       # Cressie-Read (lambda=2/3)
                          ])

# The details of the first two elements in power_div_1d_cases are used
# in a test in TestPowerDivergence.  Check that code before making
# any changes here.
power_div_1d_cases = [
    # Use the default f_exp.
    PowerDivCase(f_obs=[4, 8, 12, 8], f_exp=None, ddof=0, axis=None,
                 chi2=4,
                 log=2*(4*np.log(4/8) + 12*np.log(12/8)),
                 mod_log=2*(8*np.log(8/4) + 8*np.log(8/12)),
                 cr=(4*((4/8)**(2/3) - 1) + 12*((12/8)**(2/3) - 1))/(5/9)),
    # Give a non-uniform f_exp.
    PowerDivCase(f_obs=[4, 8, 12, 8], f_exp=[2, 16, 12, 2], ddof=0, axis=None,
                 chi2=24,
                 log=2*(4*np.log(4/2) + 8*np.log(8/16) + 8*np.log(8/2)),
                 mod_log=2*(2*np.log(2/4) + 16*np.log(16/8) + 2*np.log(2/8)),
                 cr=(4*((4/2)**(2/3) - 1) + 8*((8/16)**(2/3) - 1) +
                     8*((8/2)**(2/3) - 1))/(5/9)),
    # f_exp is a scalar.
    PowerDivCase(f_obs=[4, 8, 12, 8], f_exp=8, ddof=0, axis=None,
                 chi2=4,
                 log=2*(4*np.log(4/8) + 12*np.log(12/8)),
                 mod_log=2*(8*np.log(8/4) + 8*np.log(8/12)),
                 cr=(4*((4/8)**(2/3) - 1) + 12*((12/8)**(2/3) - 1))/(5/9)),
    # f_exp equal to f_obs.
    PowerDivCase(f_obs=[3, 5, 7, 9], f_exp=[3, 5, 7, 9], ddof=0, axis=0,
                 chi2=0, log=0, mod_log=0, cr=0),
]


power_div_empty_cases = [
    # Shape is (0,)--a data set with length 0.  The computed
    # test statistic should be 0.
    PowerDivCase(f_obs=[],
                 f_exp=None, ddof=0, axis=0,
                 chi2=0, log=0, mod_log=0, cr=0),
    # Shape is (0, 3).  This is 3 data sets, but each data set has
    # length 0, so the computed test statistic should be [0, 0, 0].
    PowerDivCase(f_obs=np.array([[],[],[]]).T,
                 f_exp=None, ddof=0, axis=0,
                 chi2=[0, 0, 0],
                 log=[0, 0, 0],
                 mod_log=[0, 0, 0],
                 cr=[0, 0, 0]),
    # Shape is (3, 0).  This represents an empty collection of
    # data sets in which each data set has length 3.  The test
    # statistic should be an empty array.
    PowerDivCase(f_obs=np.array([[],[],[]]),
                 f_exp=None, ddof=0, axis=0,
                 chi2=[],
                 log=[],
                 mod_log=[],
                 cr=[]),
]


class TestPowerDivergence:

    def check_power_divergence(self, f_obs, f_exp, ddof, axis, lambda_,
                               expected_stat):
        f_obs = np.asarray(f_obs)
        if axis is None:
            num_obs = f_obs.size
        else:
            b = np.broadcast(f_obs, f_exp)
            num_obs = b.shape[axis]

        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "Mean of empty slice")
            stat, p = stats.power_divergence(
                                f_obs=f_obs, f_exp=f_exp, ddof=ddof,
                                axis=axis, lambda_=lambda_)
            assert_allclose(stat, expected_stat)

            if lambda_ == 1 or lambda_ == "pearson":
                # Also test stats.chisquare.
                stat, p = stats.chisquare(f_obs=f_obs, f_exp=f_exp, ddof=ddof,
                                          axis=axis)
                assert_allclose(stat, expected_stat)

        ddof = np.asarray(ddof)
        expected_p = stats.distributions.chi2.sf(expected_stat,
                                                 num_obs - 1 - ddof)
        assert_allclose(p, expected_p)

    def test_basic(self):
        for case in power_div_1d_cases:
            self.check_power_divergence(
                   case.f_obs, case.f_exp, case.ddof, case.axis,
                   None, case.chi2)
            self.check_power_divergence(
                   case.f_obs, case.f_exp, case.ddof, case.axis,
                   "pearson", case.chi2)
            self.check_power_divergence(
                   case.f_obs, case.f_exp, case.ddof, case.axis,
                   1, case.chi2)
            self.check_power_divergence(
                   case.f_obs, case.f_exp, case.ddof, case.axis,
                   "log-likelihood", case.log)
            self.check_power_divergence(
                   case.f_obs, case.f_exp, case.ddof, case.axis,
                   "mod-log-likelihood", case.mod_log)
            self.check_power_divergence(
                   case.f_obs, case.f_exp, case.ddof, case.axis,
                   "cressie-read", case.cr)
            self.check_power_divergence(
                   case.f_obs, case.f_exp, case.ddof, case.axis,
                   2/3, case.cr)

    def test_basic_masked(self):
        for case in power_div_1d_cases:
            mobs = np.ma.array(case.f_obs)
            self.check_power_divergence(
                   mobs, case.f_exp, case.ddof, case.axis,
                   None, case.chi2)
            self.check_power_divergence(
                   mobs, case.f_exp, case.ddof, case.axis,
                   "pearson", case.chi2)
            self.check_power_divergence(
                   mobs, case.f_exp, case.ddof, case.axis,
                   1, case.chi2)
            self.check_power_divergence(
                   mobs, case.f_exp, case.ddof, case.axis,
                   "log-likelihood", case.log)
            self.check_power_divergence(
                   mobs, case.f_exp, case.ddof, case.axis,
                   "mod-log-likelihood", case.mod_log)
            self.check_power_divergence(
                   mobs, case.f_exp, case.ddof, case.axis,
                   "cressie-read", case.cr)
            self.check_power_divergence(
                   mobs, case.f_exp, case.ddof, case.axis,
                   2/3, case.cr)

    def test_axis(self):
        case0 = power_div_1d_cases[0]
        case1 = power_div_1d_cases[1]
        f_obs = np.vstack((case0.f_obs, case1.f_obs))
        f_exp = np.vstack((np.ones_like(case0.f_obs)*np.mean(case0.f_obs),
                           case1.f_exp))
        # Check the four computational code paths in power_divergence
        # using a 2D array with axis=1.
        self.check_power_divergence(
               f_obs, f_exp, 0, 1,
               "pearson", [case0.chi2, case1.chi2])
        self.check_power_divergence(
               f_obs, f_exp, 0, 1,
               "log-likelihood", [case0.log, case1.log])
        self.check_power_divergence(
               f_obs, f_exp, 0, 1,
               "mod-log-likelihood", [case0.mod_log, case1.mod_log])
        self.check_power_divergence(
               f_obs, f_exp, 0, 1,
               "cressie-read", [case0.cr, case1.cr])
        # Reshape case0.f_obs to shape (2,2), and use axis=None.
        # The result should be the same.
        self.check_power_divergence(
               np.array(case0.f_obs).reshape(2, 2), None, 0, None,
               "pearson", case0.chi2)

    def test_ddof_broadcasting(self):
        # Test that ddof broadcasts correctly.
        # ddof does not affect the test statistic.  It is broadcast
        # with the computed test statistic for the computation of
        # the p value.

        case0 = power_div_1d_cases[0]
        case1 = power_div_1d_cases[1]
        # Create 4x2 arrays of observed and expected frequencies.
        f_obs = np.vstack((case0.f_obs, case1.f_obs)).T
        f_exp = np.vstack((np.ones_like(case0.f_obs)*np.mean(case0.f_obs),
                           case1.f_exp)).T

        expected_chi2 = [case0.chi2, case1.chi2]

        # ddof has shape (2, 1).  This is broadcast with the computed
        # statistic, so p will have shape (2,2).
        ddof = np.array([[0], [1]])

        stat, p = stats.power_divergence(f_obs, f_exp, ddof=ddof)
        assert_allclose(stat, expected_chi2)

        # Compute the p values separately, passing in scalars for ddof.
        stat0, p0 = stats.power_divergence(f_obs, f_exp, ddof=ddof[0,0])
        stat1, p1 = stats.power_divergence(f_obs, f_exp, ddof=ddof[1,0])

        assert_array_equal(p, np.vstack((p0, p1)))

    def test_empty_cases(self):
        with warnings.catch_warnings():
            for case in power_div_empty_cases:
                self.check_power_divergence(
                       case.f_obs, case.f_exp, case.ddof, case.axis,
                       "pearson", case.chi2)
                self.check_power_divergence(
                       case.f_obs, case.f_exp, case.ddof, case.axis,
                       "log-likelihood", case.log)
                self.check_power_divergence(
                       case.f_obs, case.f_exp, case.ddof, case.axis,
                       "mod-log-likelihood", case.mod_log)
                self.check_power_divergence(
                       case.f_obs, case.f_exp, case.ddof, case.axis,
                       "cressie-read", case.cr)

    def test_power_divergence_result_attributes(self):
        f_obs = power_div_1d_cases[0].f_obs
        f_exp = power_div_1d_cases[0].f_exp
        ddof = power_div_1d_cases[0].ddof
        axis = power_div_1d_cases[0].axis

        res = stats.power_divergence(f_obs=f_obs, f_exp=f_exp, ddof=ddof,
                                     axis=axis, lambda_="pearson")
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes)

    def test_power_divergence_gh_12282(self):
        # The sums of observed and expected frequencies must match
        f_obs = np.array([[10, 20], [30, 20]])
        f_exp = np.array([[5, 15], [35, 25]])
        with assert_raises(ValueError, match='For each axis slice...'):
            stats.power_divergence(f_obs=[10, 20], f_exp=[30, 60])
        with assert_raises(ValueError, match='For each axis slice...'):
            stats.power_divergence(f_obs=f_obs, f_exp=f_exp, axis=1)
        stat, pval = stats.power_divergence(f_obs=f_obs, f_exp=f_exp)
        assert_allclose(stat, [5.71428571, 2.66666667])
        assert_allclose(pval, [0.01682741, 0.10247043])


def test_gh_chisquare_12282():
    # Currently `chisquare` is implemented via power_divergence
    # in case that ever changes, perform a basic test like
    # test_power_divergence_gh_12282
    with assert_raises(ValueError, match='For each axis slice...'):
        stats.chisquare(f_obs=[10, 20], f_exp=[30, 60])


@pytest.mark.parametrize("n, dtype", [(200, np.uint8), (1000000, np.int32)])
def test_chiquare_data_types(n, dtype):
    # Regression test for gh-10159.
    obs = np.array([n, 0], dtype=dtype)
    exp = np.array([n // 2, n // 2], dtype=dtype)
    stat, p = stats.chisquare(obs, exp)
    assert_allclose(stat, n, rtol=1e-13)


def test_chisquare_masked_arrays():
    # Test masked arrays.
    obs = np.array([[8, 8, 16, 32, -1], [-1, -1, 3, 4, 5]]).T
    mask = np.array([[0, 0, 0, 0, 1], [1, 1, 0, 0, 0]]).T
    mobs = np.ma.masked_array(obs, mask)
    expected_chisq = np.array([24.0, 0.5])
    expected_g = np.array([2*(2*8*np.log(0.5) + 32*np.log(2.0)),
                           2*(3*np.log(0.75) + 5*np.log(1.25))])

    chi2 = stats.distributions.chi2

    chisq, p = stats.chisquare(mobs)
    mat.assert_array_equal(chisq, expected_chisq)
    mat.assert_array_almost_equal(p, chi2.sf(expected_chisq,
                                             mobs.count(axis=0) - 1))

    g, p = stats.power_divergence(mobs, lambda_='log-likelihood')
    mat.assert_array_almost_equal(g, expected_g, decimal=15)
    mat.assert_array_almost_equal(p, chi2.sf(expected_g,
                                             mobs.count(axis=0) - 1))

    chisq, p = stats.chisquare(mobs.T, axis=1)
    mat.assert_array_equal(chisq, expected_chisq)
    mat.assert_array_almost_equal(p, chi2.sf(expected_chisq,
                                             mobs.T.count(axis=1) - 1))
    g, p = stats.power_divergence(mobs.T, axis=1, lambda_="log-likelihood")
    mat.assert_array_almost_equal(g, expected_g, decimal=15)
    mat.assert_array_almost_equal(p, chi2.sf(expected_g,
                                             mobs.count(axis=0) - 1))

    obs1 = np.ma.array([3, 5, 6, 99, 10], mask=[0, 0, 0, 1, 0])
    exp1 = np.ma.array([2, 4, 8, 10, 99], mask=[0, 0, 0, 0, 1])
    chi2, p = stats.chisquare(obs1, f_exp=exp1)
    # Because of the mask at index 3 of obs1 and at index 4 of exp1,
    # only the first three elements are included in the calculation
    # of the statistic.
    mat.assert_array_equal(chi2, 1/2 + 1/4 + 4/8)

    # When axis=None, the two values should have type np.float64.
    chisq, p = stats.chisquare(np.ma.array([1,2,3]), axis=None)
    assert_(isinstance(chisq, np.float64))
    assert_(isinstance(p, np.float64))
    assert_equal(chisq, 1.0)
    assert_almost_equal(p, stats.distributions.chi2.sf(1.0, 2))

    # Empty arrays:
    # A data set with length 0 returns a masked scalar.
    with np.errstate(invalid='ignore'):
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "Mean of empty slice")
            chisq, p = stats.chisquare(np.ma.array([]))
    assert_(isinstance(chisq, np.ma.MaskedArray))
    assert_equal(chisq.shape, ())
    assert_(chisq.mask)

    empty3 = np.ma.array([[],[],[]])

    # empty3 is a collection of 0 data sets (whose lengths would be 3, if
    # there were any), so the return value is an array with length 0.
    chisq, p = stats.chisquare(empty3)
    assert_(isinstance(chisq, np.ma.MaskedArray))
    mat.assert_array_equal(chisq, [])

    # empty3.T is an array containing 3 data sets, each with length 0,
    # so an array of size (3,) is returned, with all values masked.
    with np.errstate(invalid='ignore'):
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "Mean of empty slice")
            chisq, p = stats.chisquare(empty3.T)

    assert_(isinstance(chisq, np.ma.MaskedArray))
    assert_equal(chisq.shape, (3,))
    assert_(np.all(chisq.mask))


def test_power_divergence_against_cressie_read_data():
    # Test stats.power_divergence against tables 4 and 5 from
    # Cressie and Read, "Multimonial Goodness-of-Fit Tests",
    # J. R. Statist. Soc. B (1984), Vol 46, No. 3, pp. 440-464.
    # This tests the calculation for several values of lambda.

    # Table 4 data recalculated for greater precision according to:
    # Shelby J. Haberman, Analysis of Qualitative Data: Volume 1
    # Introductory Topics, Academic Press, New York, USA (1978).
    obs = np.array([15, 11, 14, 17, 5, 11, 10, 4, 8,
                    10, 7, 9, 11, 3, 6, 1, 1, 4])
    beta = -0.083769  # Haberman (1978), p. 15
    i = np.arange(1, len(obs) + 1)
    alpha = np.log(obs.sum() / np.exp(beta*i).sum())
    expected_counts = np.exp(alpha + beta*i)

    # `table4` holds just the second and third columns from Table 4.
    table4 = np.vstack((obs, expected_counts)).T

    table5 = np.array([
        # lambda, statistic
        -10.0, 72.2e3,
        -5.0, 28.9e1,
        -3.0, 65.6,
        -2.0, 40.6,
        -1.5, 34.0,
        -1.0, 29.5,
        -0.5, 26.5,
        0.0, 24.6,
        0.5, 23.4,
        0.67, 23.1,
        1.0, 22.7,
        1.5, 22.6,
        2.0, 22.9,
        3.0, 24.8,
        5.0, 35.5,
        10.0, 21.4e1,
        ]).reshape(-1, 2)

    for lambda_, expected_stat in table5:
        stat, p = stats.power_divergence(table4[:,0], table4[:,1],
                                         lambda_=lambda_)
        assert_allclose(stat, expected_stat, rtol=5e-3)


def test_friedmanchisquare():
    # see ticket:113
    # verified with matlab and R
    # From Demsar "Statistical Comparisons of Classifiers over Multiple Data Sets"
    # 2006, Xf=9.28 (no tie handling, tie corrected Xf >=9.28)
    x1 = [array([0.763, 0.599, 0.954, 0.628, 0.882, 0.936, 0.661, 0.583,
                 0.775, 1.0, 0.94, 0.619, 0.972, 0.957]),
          array([0.768, 0.591, 0.971, 0.661, 0.888, 0.931, 0.668, 0.583,
                 0.838, 1.0, 0.962, 0.666, 0.981, 0.978]),
          array([0.771, 0.590, 0.968, 0.654, 0.886, 0.916, 0.609, 0.563,
                 0.866, 1.0, 0.965, 0.614, 0.9751, 0.946]),
          array([0.798, 0.569, 0.967, 0.657, 0.898, 0.931, 0.685, 0.625,
                 0.875, 1.0, 0.962, 0.669, 0.975, 0.970])]

    # From "Bioestadistica para las ciencias de la salud" Xf=18.95 p<0.001:
    x2 = [array([4,3,5,3,5,3,2,5,4,4,4,3]),
          array([2,2,1,2,3,1,2,3,2,1,1,3]),
          array([2,4,3,3,4,3,3,4,4,1,2,1]),
          array([3,5,4,3,4,4,3,3,3,4,4,4])]

    # From Jerrorl H. Zar, "Biostatistical Analysis"(example 12.6), Xf=10.68, 0.005 < p < 0.01:
    # Probability from this example is inexact using Chisquare approximation of Friedman Chisquare.
    x3 = [array([7.0,9.9,8.5,5.1,10.3]),
          array([5.3,5.7,4.7,3.5,7.7]),
          array([4.9,7.6,5.5,2.8,8.4]),
          array([8.8,8.9,8.1,3.3,9.1])]

    assert_array_almost_equal(stats.friedmanchisquare(x1[0],x1[1],x1[2],x1[3]),
                              (10.2283464566929, 0.0167215803284414))
    assert_array_almost_equal(stats.friedmanchisquare(x2[0],x2[1],x2[2],x2[3]),
                              (18.9428571428571, 0.000280938375189499))
    assert_array_almost_equal(stats.friedmanchisquare(x3[0],x3[1],x3[2],x3[3]),
                              (10.68, 0.0135882729582176))
    assert_raises(ValueError, stats.friedmanchisquare,x3[0],x3[1])

    # test for namedtuple attribute results
    attributes = ('statistic', 'pvalue')
    res = stats.friedmanchisquare(*x1)
    check_named_results(res, attributes)

    # test using mstats
    assert_array_almost_equal(mstats.friedmanchisquare(x1[0], x1[1],
                                                       x1[2], x1[3]),
                              (10.2283464566929, 0.0167215803284414))
    # the following fails
    # assert_array_almost_equal(mstats.friedmanchisquare(x2[0],x2[1],x2[2],x2[3]),
    #                           (18.9428571428571, 0.000280938375189499))
    assert_array_almost_equal(mstats.friedmanchisquare(x3[0], x3[1],
                                                       x3[2], x3[3]),
                              (10.68, 0.0135882729582176))
    assert_raises(ValueError, mstats.friedmanchisquare,x3[0],x3[1])


class TestKSTest:
    """Tests kstest and ks_1samp agree with K-S various sizes, alternatives, modes."""

    def _testOne(self, x, alternative, expected_statistic, expected_prob, mode='auto', decimal=14):
        result = stats.kstest(x, 'norm', alternative=alternative, mode=mode)
        expected = np.array([expected_statistic, expected_prob])
        assert_array_almost_equal(np.array(result), expected, decimal=decimal)

    def _test_kstest_and_ks1samp(self, x, alternative, mode='auto', decimal=14):
        result = stats.kstest(x, 'norm', alternative=alternative, mode=mode)
        result_1samp = stats.ks_1samp(x, stats.norm.cdf, alternative=alternative, mode=mode)
        assert_array_almost_equal(np.array(result), result_1samp, decimal=decimal)

    def test_namedtuple_attributes(self):
        x = np.linspace(-1, 1, 9)
        # test for namedtuple attribute results
        attributes = ('statistic', 'pvalue')
        res = stats.kstest(x, 'norm')
        check_named_results(res, attributes)

    def test_agree_with_ks_1samp(self):
        x = np.linspace(-1, 1, 9)
        self._test_kstest_and_ks1samp(x, 'two-sided')

        x = np.linspace(-15, 15, 9)
        self._test_kstest_and_ks1samp(x, 'two-sided')

        x = [-1.23, 0.06, -0.60, 0.17, 0.66, -0.17, -0.08, 0.27, -0.98, -0.99]
        self._test_kstest_and_ks1samp(x, 'two-sided')
        self._test_kstest_and_ks1samp(x, 'greater', mode='exact')
        self._test_kstest_and_ks1samp(x, 'less', mode='exact')

    # missing: no test that uses *args

class TestKSOneSample:
    """Tests kstest and ks_samp 1-samples with K-S various sizes, alternatives, modes."""

    def _testOne(self, x, alternative, expected_statistic, expected_prob, mode='auto', decimal=14):
        result = stats.ks_1samp(x, stats.norm.cdf, alternative=alternative, mode=mode)
        expected = np.array([expected_statistic, expected_prob])
        assert_array_almost_equal(np.array(result), expected, decimal=decimal)

    def test_namedtuple_attributes(self):
        x = np.linspace(-1, 1, 9)
        # test for namedtuple attribute results
        attributes = ('statistic', 'pvalue')
        res = stats.ks_1samp(x, stats.norm.cdf)
        check_named_results(res, attributes)

    def test_agree_with_r(self):
        # comparing with some values from R
        x = np.linspace(-1, 1, 9)
        self._testOne(x, 'two-sided', 0.15865525393145705, 0.95164069201518386)

        x = np.linspace(-15, 15, 9)
        self._testOne(x, 'two-sided', 0.44435602715924361, 0.038850140086788665)

        x = [-1.23, 0.06, -0.60, 0.17, 0.66, -0.17, -0.08, 0.27, -0.98, -0.99]
        self._testOne(x, 'two-sided', 0.293580126801961, 0.293408463684361)
        self._testOne(x, 'greater', 0.293580126801961, 0.146988835042376, mode='exact')
        self._testOne(x, 'less', 0.109348552425692, 0.732768892470675, mode='exact')

    def test_known_examples(self):
        # the following tests rely on deterministically replicated rvs
        x = stats.norm.rvs(loc=0.2, size=100, random_state=987654321)
        self._testOne(x, 'two-sided', 0.12464329735846891, 0.089444888711820769, mode='asymp')
        self._testOne(x, 'less', 0.12464329735846891, 0.040989164077641749)
        self._testOne(x, 'greater', 0.0072115233216310994, 0.98531158590396228)

    def test_ks1samp_allpaths(self):
        # Check NaN input, output.
        assert_(np.isnan(kolmogn(np.nan, 1, True)))
        with assert_raises(ValueError, match='n is not integral: 1.5'):
            kolmogn(1.5, 1, True)
        assert_(np.isnan(kolmogn(-1, 1, True)))

        dataset = np.asarray([
            # Check x out of range
            (101, 1, True, 1.0),
            (101, 1.1, True, 1.0),
            (101, 0, True, 0.0),
            (101, -0.1, True, 0.0),

            (32, 1.0 / 64, True, 0.0),  # Ruben-Gambino
            (32, 1.0 / 64, False, 1.0),  # Ruben-Gambino

            (32, 0.5, True, 0.9999999363163307),  # Miller
            (32, 0.5, False, 6.368366937916623e-08),  # Miller 2 * special.smirnov(32, 0.5)

            # Check some other paths
            (32, 1.0 / 8, True, 0.34624229979775223),
            (32, 1.0 / 4, True, 0.9699508336558085),
            (1600, 0.49, False, 0.0),
            (1600, 1 / 16.0, False, 7.0837876229702195e-06),  # 2 * special.smirnov(1600, 1/16.0)
            (1600, 14 / 1600, False, 0.99962357317602),  # _kolmogn_DMTW
            (1600, 1 / 32, False, 0.08603386296651416),  # _kolmogn_PelzGood
        ])
        FuncData(kolmogn, dataset, (0, 1, 2), 3).check(dtypes=[int, float, bool])

    # missing: no test that uses *args


class TestKSTwoSamples:
    """Tests 2-samples with K-S various sizes, alternatives, modes."""

    def _testOne(self, x1, x2, alternative, expected_statistic, expected_prob, mode='auto'):
        result = stats.ks_2samp(x1, x2, alternative, mode=mode)
        expected = np.array([expected_statistic, expected_prob])
        assert_array_almost_equal(np.array(result), expected)

    def testSmall(self):
        self._testOne([0], [1], 'two-sided', 1.0/1, 1.0)
        self._testOne([0], [1], 'greater', 1.0/1, 0.5)
        self._testOne([0], [1], 'less', 0.0/1, 1.0)
        self._testOne([1], [0], 'two-sided', 1.0/1, 1.0)
        self._testOne([1], [0], 'greater', 0.0/1, 1.0)
        self._testOne([1], [0], 'less', 1.0/1, 0.5)

    def testTwoVsThree(self):
        data1 = np.array([1.0, 2.0])
        data1p = data1 + 0.01
        data1m = data1 - 0.01
        data2 = np.array([1.0, 2.0, 3.0])
        self._testOne(data1p, data2, 'two-sided', 1.0 / 3, 1.0)
        self._testOne(data1p, data2, 'greater', 1.0 / 3, 0.7)
        self._testOne(data1p, data2, 'less', 1.0 / 3, 0.7)
        self._testOne(data1m, data2, 'two-sided', 2.0 / 3, 0.6)
        self._testOne(data1m, data2, 'greater', 2.0 / 3, 0.3)
        self._testOne(data1m, data2, 'less', 0, 1.0)

    def testTwoVsFour(self):
        data1 = np.array([1.0, 2.0])
        data1p = data1 + 0.01
        data1m = data1 - 0.01
        data2 = np.array([1.0, 2.0, 3.0, 4.0])
        self._testOne(data1p, data2, 'two-sided', 2.0 / 4, 14.0/15)
        self._testOne(data1p, data2, 'greater', 2.0 / 4, 8.0/15)
        self._testOne(data1p, data2, 'less', 1.0 / 4, 12.0/15)

        self._testOne(data1m, data2, 'two-sided', 3.0 / 4, 6.0/15)
        self._testOne(data1m, data2, 'greater', 3.0 / 4, 3.0/15)
        self._testOne(data1m, data2, 'less', 0, 1.0)

    def test100_100(self):
        x100 = np.linspace(1, 100, 100)
        x100_2_p1 = x100 + 2 + 0.1
        x100_2_m1 = x100 + 2 - 0.1
        self._testOne(x100, x100_2_p1, 'two-sided', 3.0 / 100, 0.9999999999962055)
        self._testOne(x100, x100_2_p1, 'greater', 3.0 / 100, 0.9143290114276248)
        self._testOne(x100, x100_2_p1, 'less', 0, 1.0)
        self._testOne(x100, x100_2_m1, 'two-sided', 2.0 / 100, 1.0)
        self._testOne(x100, x100_2_m1, 'greater', 2.0 / 100, 0.960978450786184)
        self._testOne(x100, x100_2_m1, 'less', 0, 1.0)

    def test100_110(self):
        x100 = np.linspace(1, 100, 100)
        x110 = np.linspace(1, 100, 110)
        x110_20_p1 = x110 + 20 + 0.1
        x110_20_m1 = x110 + 20 - 0.1
        # 100, 110
        self._testOne(x100, x110_20_p1, 'two-sided', 232.0 / 1100, 0.015739183865607353)
        self._testOne(x100, x110_20_p1, 'greater', 232.0 / 1100, 0.007869594319053203)
        self._testOne(x100, x110_20_p1, 'less', 0, 1)
        self._testOne(x100, x110_20_m1, 'two-sided', 229.0 / 1100, 0.017803803861026313)
        self._testOne(x100, x110_20_m1, 'greater', 229.0 / 1100, 0.008901905958245056)
        self._testOne(x100, x110_20_m1, 'less', 0.0, 1.0)

    def testRepeatedValues(self):
        x2233 = np.array([2] * 3 + [3] * 4 + [5] * 5 + [6] * 4, dtype=int)
        x3344 = x2233 + 1
        x2356 = np.array([2] * 3 + [3] * 4 + [5] * 10 + [6] * 4, dtype=int)
        x3467 = np.array([3] * 10 + [4] * 2 + [6] * 10 + [7] * 4, dtype=int)
        self._testOne(x2233, x3344, 'two-sided', 5.0/16, 0.4262934613454952)
        self._testOne(x2233, x3344, 'greater', 5.0/16, 0.21465428276573786)
        self._testOne(x2233, x3344, 'less', 0.0/16, 1.0)
        self._testOne(x2356, x3467, 'two-sided', 190.0/21/26, 0.0919245790168125)
        self._testOne(x2356, x3467, 'greater', 190.0/21/26, 0.0459633806858544)
        self._testOne(x2356, x3467, 'less', 70.0/21/26, 0.6121593130022775)

    def testEqualSizes(self):
        data2 = np.array([1.0, 2.0, 3.0])
        self._testOne(data2, data2+1, 'two-sided', 1.0/3, 1.0)
        self._testOne(data2, data2+1, 'greater', 1.0/3, 0.75)
        self._testOne(data2, data2+1, 'less', 0.0/3, 1.)
        self._testOne(data2, data2+0.5, 'two-sided', 1.0/3, 1.0)
        self._testOne(data2, data2+0.5, 'greater', 1.0/3, 0.75)
        self._testOne(data2, data2+0.5, 'less', 0.0/3, 1.)
        self._testOne(data2, data2-0.5, 'two-sided', 1.0/3, 1.0)
        self._testOne(data2, data2-0.5, 'greater', 0.0/3, 1.0)
        self._testOne(data2, data2-0.5, 'less', 1.0/3, 0.75)

    @pytest.mark.slow
    def testMiddlingBoth(self):
        # 500, 600
        n1, n2 = 500, 600
        delta = 1.0/n1/n2/2/2
        x = np.linspace(1, 200, n1) - delta
        y = np.linspace(2, 200, n2)
        self._testOne(x, y, 'two-sided', 2000.0 / n1 / n2, 1.0, mode='auto')
        self._testOne(x, y, 'two-sided', 2000.0 / n1 / n2, 1.0, mode='asymp')
        self._testOne(x, y, 'greater', 2000.0 / n1 / n2, 0.9697596024683929, mode='asymp')
        self._testOne(x, y, 'less', 500.0 / n1 / n2, 0.9968735843165021, mode='asymp')
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "ks_2samp: Exact calculation unsuccessful. Switching to mode=asymp.")
            self._testOne(x, y, 'greater', 2000.0 / n1 / n2, 0.9697596024683929, mode='exact')
            self._testOne(x, y, 'less', 500.0 / n1 / n2, 0.9968735843165021, mode='exact')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._testOne(x, y, 'less', 500.0 / n1 / n2, 0.9968735843165021, mode='exact')
            _check_warnings(w, RuntimeWarning, 1)

    @pytest.mark.slow
    def testMediumBoth(self):
        # 1000, 1100
        n1, n2 = 1000, 1100
        delta = 1.0/n1/n2/2/2
        x = np.linspace(1, 200, n1) - delta
        y = np.linspace(2, 200, n2)
        self._testOne(x, y, 'two-sided', 6600.0 / n1 / n2, 1.0, mode='asymp')
        self._testOne(x, y, 'two-sided', 6600.0 / n1 / n2, 1.0, mode='auto')
        self._testOne(x, y, 'greater', 6600.0 / n1 / n2, 0.9573185808092622, mode='asymp')
        self._testOne(x, y, 'less', 1000.0 / n1 / n2, 0.9982410869433984, mode='asymp')

        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "ks_2samp: Exact calculation unsuccessful. Switching to mode=asymp.")
            self._testOne(x, y, 'greater', 6600.0 / n1 / n2, 0.9573185808092622, mode='exact')
            self._testOne(x, y, 'less', 1000.0 / n1 / n2, 0.9982410869433984, mode='exact')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._testOne(x, y, 'less', 1000.0 / n1 / n2, 0.9982410869433984, mode='exact')
            _check_warnings(w, RuntimeWarning, 1)

    def testLarge(self):
        # 10000, 110
        n1, n2 = 10000, 110
        lcm = n1*11.0
        delta = 1.0/n1/n2/2/2
        x = np.linspace(1, 200, n1) - delta
        y = np.linspace(2, 100, n2)
        self._testOne(x, y, 'two-sided', 55275.0 / lcm, 4.2188474935755949e-15)
        self._testOne(x, y, 'greater', 561.0 / lcm, 0.99115454582047591)
        self._testOne(x, y, 'less', 55275.0 / lcm, 3.1317328311518713e-26)

    def test_gh11184(self):
        # 3000, 3001, exact two-sided
        np.random.seed(123456)
        x = np.random.normal(size=3000)
        y = np.random.normal(size=3001) * 1.5
        self._testOne(x, y, 'two-sided', 0.11292880151060758, 2.7755575615628914e-15, mode='asymp')
        self._testOne(x, y, 'two-sided', 0.11292880151060758, 2.7755575615628914e-15, mode='exact')

    def test_gh11184_bigger(self):
        # 10000, 10001, exact two-sided
        np.random.seed(123456)
        x = np.random.normal(size=10000)
        y = np.random.normal(size=10001) * 1.5
        self._testOne(x, y, 'two-sided', 0.10597913208679133, 3.3149311398483503e-49, mode='asymp')
        self._testOne(x, y, 'two-sided', 0.10597913208679133, 2.7755575615628914e-15, mode='exact')
        self._testOne(x, y, 'greater', 0.10597913208679133, 2.7947433906389253e-41, mode='asymp')
        self._testOne(x, y, 'less', 0.09658002199780022, 2.7947433906389253e-41, mode='asymp')

    @pytest.mark.slow
    def testLargeBoth(self):
        # 10000, 11000
        n1, n2 = 10000, 11000
        lcm = n1*11.0
        delta = 1.0/n1/n2/2/2
        x = np.linspace(1, 200, n1) - delta
        y = np.linspace(2, 200, n2)
        self._testOne(x, y, 'two-sided', 563.0 / lcm, 0.9990660108966576, mode='asymp')
        self._testOne(x, y, 'two-sided', 563.0 / lcm, 0.9990456491488628, mode='exact')
        self._testOne(x, y, 'two-sided', 563.0 / lcm, 0.9990660108966576, mode='auto')
        self._testOne(x, y, 'greater', 563.0 / lcm, 0.7561851877420673)
        self._testOne(x, y, 'less', 10.0 / lcm, 0.9998239693191724)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "ks_2samp: Exact calculation unsuccessful. Switching to mode=asymp.")
            self._testOne(x, y, 'greater', 563.0 / lcm, 0.7561851877420673, mode='exact')
            self._testOne(x, y, 'less', 10.0 / lcm, 0.9998239693191724, mode='exact')

    def testNamedAttributes(self):
        # test for namedtuple attribute results
        attributes = ('statistic', 'pvalue')
        res = stats.ks_2samp([1, 2], [3])
        check_named_results(res, attributes)

    @pytest.mark.slow
    def test_some_code_paths(self):
        # Check that some code paths are executed
        from scipy.stats.stats import _count_paths_outside_method, _compute_prob_inside_method

        _compute_prob_inside_method(1, 1, 1, 1)
        _count_paths_outside_method(1000, 1, 1, 1001)

        assert_raises(FloatingPointError, _count_paths_outside_method, 1100, 1099, 1, 1)
        assert_raises(FloatingPointError, _count_paths_outside_method, 2000, 1000, 1, 1)

    def test_argument_checking(self):
        # Check that an empty array causes a ValueError
        assert_raises(ValueError, stats.ks_2samp, [], [1])
        assert_raises(ValueError, stats.ks_2samp, [1], [])
        assert_raises(ValueError, stats.ks_2samp, [], [])

    @pytest.mark.slow
    def test_gh12218(self):
        """Ensure gh-12218 is fixed."""
        # gh-1228 triggered a TypeError calculating sqrt(n1*n2*(n1+n2)).
        # n1, n2 both large integers, the product exceeded 2^64
        np.random.seed(12345678)
        n1 = 2097152  # 2*^21
        rvs1 = stats.uniform.rvs(size=n1, loc=0., scale=1)
        rvs2 = rvs1 + 1  # Exact value of rvs2 doesn't matter.
        stats.ks_2samp(rvs1, rvs2, alternative='greater', mode='asymp')
        stats.ks_2samp(rvs1, rvs2, alternative='less', mode='asymp')
        stats.ks_2samp(rvs1, rvs2, alternative='two-sided', mode='asymp')


def test_ttest_rel():
    # regression test
    tr,pr = 0.81248591389165692, 0.41846234511362157
    tpr = ([tr,-tr],[pr,pr])

    rvs1 = np.linspace(1,100,100)
    rvs2 = np.linspace(1.01,99.989,100)
    rvs1_2D = np.array([np.linspace(1,100,100), np.linspace(1.01,99.989,100)])
    rvs2_2D = np.array([np.linspace(1.01,99.989,100), np.linspace(1,100,100)])

    t,p = stats.ttest_rel(rvs1, rvs2, axis=0)
    assert_array_almost_equal([t,p],(tr,pr))
    t,p = stats.ttest_rel(rvs1_2D.T, rvs2_2D.T, axis=0)
    assert_array_almost_equal([t,p],tpr)
    t,p = stats.ttest_rel(rvs1_2D, rvs2_2D, axis=1)
    assert_array_almost_equal([t,p],tpr)

    # test scalars
    with suppress_warnings() as sup, np.errstate(invalid="ignore"):
        sup.filter(RuntimeWarning, "Degrees of freedom <= 0 for slice")
        t, p = stats.ttest_rel(4., 3.)
    assert_(np.isnan(t))
    assert_(np.isnan(p))

    # test for namedtuple attribute results
    attributes = ('statistic', 'pvalue')
    res = stats.ttest_rel(rvs1, rvs2, axis=0)
    check_named_results(res, attributes)

    # test on 3 dimensions
    rvs1_3D = np.dstack([rvs1_2D,rvs1_2D,rvs1_2D])
    rvs2_3D = np.dstack([rvs2_2D,rvs2_2D,rvs2_2D])
    t,p = stats.ttest_rel(rvs1_3D, rvs2_3D, axis=1)
    assert_array_almost_equal(np.abs(t), tr)
    assert_array_almost_equal(np.abs(p), pr)
    assert_equal(t.shape, (2, 3))

    t, p = stats.ttest_rel(np.rollaxis(rvs1_3D, 2), np.rollaxis(rvs2_3D, 2),
                           axis=2)
    assert_array_almost_equal(np.abs(t), tr)
    assert_array_almost_equal(np.abs(p), pr)
    assert_equal(t.shape, (3, 2))

    # test alternative parameter
    assert_raises(ValueError, stats.ttest_rel, rvs1, rvs2, alternative="error")

    t, p = stats.ttest_rel(rvs1, rvs2, axis=0, alternative="less")
    assert_allclose(p, 1 - pr/2)
    assert_allclose(t, tr)

    t, p = stats.ttest_rel(rvs1, rvs2, axis=0, alternative="greater")
    assert_allclose(p, pr/2)
    assert_allclose(t, tr)

    # check nan policy
    rng = np.random.RandomState(12345678)
    x = stats.norm.rvs(loc=5, scale=10, size=501, random_state=rng)
    x[500] = np.nan
    y = (stats.norm.rvs(loc=5, scale=10, size=501, random_state=rng) +
         stats.norm.rvs(scale=0.2, size=501, random_state=rng))
    y[500] = np.nan

    with np.errstate(invalid="ignore"):
        assert_array_equal(stats.ttest_rel(x, x), (np.nan, np.nan))

    assert_array_almost_equal(stats.ttest_rel(x, y, nan_policy='omit'),
                              (0.25299925303978066, 0.8003729814201519))
    assert_raises(ValueError, stats.ttest_rel, x, y, nan_policy='raise')
    assert_raises(ValueError, stats.ttest_rel, x, y, nan_policy='foobar')

    # test zero division problem
    t, p = stats.ttest_rel([0, 0, 0], [1, 1, 1])
    assert_equal((np.abs(t), p), (np.inf, 0))
    with np.errstate(invalid="ignore"):
        assert_equal(stats.ttest_rel([0, 0, 0], [0, 0, 0]), (np.nan, np.nan))

        # check that nan in input array result in nan output
        anan = np.array([[1, np.nan], [-1, 1]])
        assert_equal(stats.ttest_rel(anan, np.zeros((2, 2))),
                     ([0, np.nan], [1, np.nan]))

    # test incorrect input shape raise an error
    x = np.arange(24)
    assert_raises(ValueError, stats.ttest_rel, x.reshape((8, 3)),
                  x.reshape((2, 3, 4)))


def test_ttest_rel_nan_2nd_arg():
    # regression test for gh-6134: nans in the second arg were not handled
    x = [np.nan, 2.0, 3.0, 4.0]
    y = [1.0, 2.0, 1.0, 2.0]

    r1 = stats.ttest_rel(x, y, nan_policy='omit')
    r2 = stats.ttest_rel(y, x, nan_policy='omit')
    assert_allclose(r2.statistic, -r1.statistic, atol=1e-15)
    assert_allclose(r2.pvalue, r1.pvalue, atol=1e-15)

    # NB: arguments are paired when NaNs are dropped
    r3 = stats.ttest_rel(y[1:], x[1:])
    assert_allclose(r2, r3, atol=1e-15)

    # .. and this is consistent with R. R code:
    # x = c(NA, 2.0, 3.0, 4.0)
    # y = c(1.0, 2.0, 1.0, 2.0)
    # t.test(x, y, paired=TRUE)
    assert_allclose(r2, (-2, 0.1835), atol=1e-4)


def test_ttest_rel_empty_1d_returns_nan():
    # Two empty inputs should return a Ttest_relResult containing nan
    # for both values.
    result = stats.ttest_rel([], [])
    assert isinstance(result, stats.stats.Ttest_relResult)
    assert_equal(result, (np.nan, np.nan))


@pytest.mark.parametrize('b, expected_shape',
                         [(np.empty((1, 5, 0)), (3, 5)),
                          (np.empty((1, 0, 0)), (3, 0))])
def test_ttest_rel_axis_size_zero(b, expected_shape):
    # In this test, the length of the axis dimension is zero.
    # The results should be arrays containing nan with shape
    # given by the broadcast nonaxis dimensions.
    a = np.empty((3, 1, 0))
    result = stats.ttest_rel(a, b, axis=-1)
    assert isinstance(result, stats.stats.Ttest_relResult)
    expected_value = np.full(expected_shape, fill_value=np.nan)
    assert_equal(result.statistic, expected_value)
    assert_equal(result.pvalue, expected_value)


def test_ttest_rel_nonaxis_size_zero():
    # In this test, the length of the axis dimension is nonzero,
    # but one of the nonaxis dimensions has length 0.  Check that
    # we still get the correctly broadcast shape, which is (5, 0)
    # in this case.
    a = np.empty((1, 8, 0))
    b = np.empty((5, 8, 1))
    result = stats.ttest_rel(a, b, axis=1)
    assert isinstance(result, stats.stats.Ttest_relResult)
    assert_equal(result.statistic.shape, (5, 0))
    assert_equal(result.pvalue.shape, (5, 0))


def _desc_stats(x1, x2, axis=0):
    def _stats(x, axis=0):
        x = np.asarray(x)
        mu = np.mean(x, axis=axis)
        std = np.std(x, axis=axis, ddof=1)
        nobs = x.shape[axis]
        return mu, std, nobs
    return _stats(x1, axis) + _stats(x2, axis)


def test_ttest_ind():
    # regression test
    tr = 1.0912746897927283
    pr = 0.27647818616351882
    tpr = ([tr,-tr],[pr,pr])

    rvs2 = np.linspace(1,100,100)
    rvs1 = np.linspace(5,105,100)
    rvs1_2D = np.array([rvs1, rvs2])
    rvs2_2D = np.array([rvs2, rvs1])

    t,p = stats.ttest_ind(rvs1, rvs2, axis=0)
    assert_array_almost_equal([t,p],(tr,pr))
    # test from_stats API
    assert_array_almost_equal(stats.ttest_ind_from_stats(*_desc_stats(rvs1,
                                                                      rvs2)),
                              [t, p])
    t,p = stats.ttest_ind(rvs1_2D.T, rvs2_2D.T, axis=0)
    assert_array_almost_equal([t,p],tpr)
    args = _desc_stats(rvs1_2D.T, rvs2_2D.T)
    assert_array_almost_equal(stats.ttest_ind_from_stats(*args),
                              [t, p])
    t,p = stats.ttest_ind(rvs1_2D, rvs2_2D, axis=1)
    assert_array_almost_equal([t,p],tpr)
    args = _desc_stats(rvs1_2D, rvs2_2D, axis=1)
    assert_array_almost_equal(stats.ttest_ind_from_stats(*args),
                              [t, p])

    # test scalars
    with suppress_warnings() as sup, np.errstate(invalid="ignore"):
        sup.filter(RuntimeWarning, "Degrees of freedom <= 0 for slice")
        t, p = stats.ttest_ind(4., 3.)
    assert_(np.isnan(t))
    assert_(np.isnan(p))

    # test on 3 dimensions
    rvs1_3D = np.dstack([rvs1_2D,rvs1_2D,rvs1_2D])
    rvs2_3D = np.dstack([rvs2_2D,rvs2_2D,rvs2_2D])
    t,p = stats.ttest_ind(rvs1_3D, rvs2_3D, axis=1)
    assert_almost_equal(np.abs(t), np.abs(tr))
    assert_array_almost_equal(np.abs(p), pr)
    assert_equal(t.shape, (2, 3))

    t, p = stats.ttest_ind(np.rollaxis(rvs1_3D, 2), np.rollaxis(rvs2_3D, 2),
                           axis=2)
    assert_array_almost_equal(np.abs(t), np.abs(tr))
    assert_array_almost_equal(np.abs(p), pr)
    assert_equal(t.shape, (3, 2))

    # test alternative parameter
    assert_raises(ValueError, stats.ttest_ind, rvs1, rvs2, alternative="error")
    assert_raises(ValueError, stats.ttest_ind_from_stats,
                  *_desc_stats(rvs1_2D.T, rvs2_2D.T), alternative="error")

    t, p = stats.ttest_ind(rvs1, rvs2, alternative="less")
    assert_allclose(p, 1 - (pr/2))
    assert_allclose(t, tr)

    t, p = stats.ttest_ind(rvs1, rvs2, alternative="greater")
    assert_allclose(p, pr/2)
    assert_allclose(t, tr)

    # Below makes sure ttest_ind_from_stats p-val functions identically to
    # ttest_ind
    t, p = stats.ttest_ind(rvs1_2D.T, rvs2_2D.T, axis=0, alternative="less")
    args = _desc_stats(rvs1_2D.T, rvs2_2D.T)
    assert_allclose(
        stats.ttest_ind_from_stats(*args, alternative="less"), [t, p])

    t, p = stats.ttest_ind(rvs1_2D.T, rvs2_2D.T, axis=0, alternative="greater")
    args = _desc_stats(rvs1_2D.T, rvs2_2D.T)
    assert_allclose(
        stats.ttest_ind_from_stats(*args, alternative="greater"), [t, p])

    # check nan policy
    rng = np.random.RandomState(12345678)
    x = stats.norm.rvs(loc=5, scale=10, size=501, random_state=rng)
    x[500] = np.nan
    y = stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)

    with np.errstate(invalid="ignore"):
        assert_array_equal(stats.ttest_ind(x, y), (np.nan, np.nan))

    assert_array_almost_equal(stats.ttest_ind(x, y, nan_policy='omit'),
                              (0.24779670949091914, 0.80434267337517906))
    assert_raises(ValueError, stats.ttest_ind, x, y, nan_policy='raise')
    assert_raises(ValueError, stats.ttest_ind, x, y, nan_policy='foobar')

    # test zero division problem
    t, p = stats.ttest_ind([0, 0, 0], [1, 1, 1])
    assert_equal((np.abs(t), p), (np.inf, 0))

    with np.errstate(invalid="ignore"):
        assert_equal(stats.ttest_ind([0, 0, 0], [0, 0, 0]), (np.nan, np.nan))

        # check that nan in input array result in nan output
        anan = np.array([[1, np.nan], [-1, 1]])
        assert_equal(stats.ttest_ind(anan, np.zeros((2, 2))),
                     ([0, np.nan], [1, np.nan]))


class Test_ttest_ind_permutations():
    N = 20

    # data for most tests
    np.random.seed(0)
    a = np.vstack((np.arange(3*N//4), np.random.random(3*N//4)))
    b = np.vstack((np.arange(N//4) + 100, np.random.random(N//4)))

    # data for equal variance tests
    a2 = np.arange(10)
    b2 = np.arange(10) + 100

    # data for exact test
    a3 = [1, 2]
    b3 = [3, 4]

    # data for bigger test
    np.random.seed(0)
    rvs1 = stats.norm.rvs(loc=5, scale=10,  # type: ignore
                          size=500).reshape(100, 5)
    rvs2 = stats.norm.rvs(loc=8, scale=20, size=100)  # type: ignore

    p_d = [0, 0.676]  # desired pvalues
    p_d_gen = [0, 0.672]  # desired pvalues for Generator seed
    p_d_big = [0.993, 0.685, 0.84, 0.955, 0.255]

    params = [
        (a, b, {"axis": 1}, p_d),                     # basic test
        (a.T, b.T, {'axis': 0}, p_d),                 # along axis 0
        (a[0, :], b[0, :], {'axis': None}, p_d[0]),   # 1d data
        (a[0, :].tolist(), b[0, :].tolist(), {'axis': None}, p_d[0]),
        # different seeds
        (a, b, {'random_state': 0, "axis": 1}, p_d),
        (a, b, {'random_state': np.random.RandomState(0), "axis": 1}, p_d),
        (a2, b2, {'equal_var': True}, 0),  # equal variances
        (rvs1, rvs2, {'axis': 0, 'random_state': 0}, p_d_big),  # bigger test
        (a3, b3, {}, 1/3)  # exact test
        ]

    if NumpyVersion(np.__version__) >= '1.18.0':
        params.append(
            (a, b, {'random_state': np.random.default_rng(0), "axis": 1},
             p_d_gen),
            )

    @pytest.mark.parametrize("a,b,update,p_d", params)
    def test_ttest_ind_permutations(self, a, b, update, p_d):
        options_a = {'axis': None, 'equal_var': False}
        options_p = {'axis': None, 'equal_var': False,
                     'permutations': 1000, 'random_state': 0}
        options_a.update(update)
        options_p.update(update)

        stat_a, _ = stats.ttest_ind(a, b, **options_a)
        stat_p, pvalue = stats.ttest_ind(a, b, **options_p)
        assert_array_almost_equal(stat_a, stat_p, 5)
        assert_array_almost_equal(pvalue, p_d)

    def test_ttest_ind_exact_alternative(self):
        np.random.seed(0)
        N = 3
        a = np.random.rand(2, N, 2)
        b = np.random.rand(2, N, 2)

        options_p = {'axis': 1, 'permutations': 1000}

        options_p.update(alternative="greater")
        res_g_ab = stats.ttest_ind(a, b, **options_p)
        res_g_ba = stats.ttest_ind(b, a, **options_p)

        options_p.update(alternative="less")
        res_l_ab = stats.ttest_ind(a, b, **options_p)
        res_l_ba = stats.ttest_ind(b, a, **options_p)

        options_p.update(alternative="two-sided")
        res_2_ab = stats.ttest_ind(a, b, **options_p)
        res_2_ba = stats.ttest_ind(b, a, **options_p)

        # Alternative doesn't affect the statistic
        assert_equal(res_g_ab.statistic, res_l_ab.statistic)
        assert_equal(res_g_ab.statistic, res_2_ab.statistic)

        # Reversing order of inputs negates statistic
        assert_equal(res_g_ab.statistic, -res_g_ba.statistic)
        assert_equal(res_l_ab.statistic, -res_l_ba.statistic)
        assert_equal(res_2_ab.statistic, -res_2_ba.statistic)

        # Reversing order of inputs does not affect p-value of 2-sided test
        assert_equal(res_2_ab.pvalue, res_2_ba.pvalue)

        # In exact test, distribution is perfectly symmetric, so these
        # identities are exactly satisfied.
        assert_equal(res_g_ab.pvalue, res_l_ba.pvalue)
        assert_equal(res_l_ab.pvalue, res_g_ba.pvalue)
        mask = res_g_ab.pvalue <= 0.5
        assert_equal(res_g_ab.pvalue[mask] + res_l_ba.pvalue[mask],
                     res_2_ab.pvalue[mask])
        assert_equal(res_l_ab.pvalue[~mask] + res_g_ba.pvalue[~mask],
                     res_2_ab.pvalue[~mask])

    def test_ttest_ind_exact_selection(self):
        # test the various ways of activating the exact test
        np.random.seed(0)
        N = 3
        a = np.random.rand(N)
        b = np.random.rand(N)
        res0 = stats.ttest_ind(a, b)
        res1 = stats.ttest_ind(a, b, permutations=1000)
        res2 = stats.ttest_ind(a, b, permutations=0)
        res3 = stats.ttest_ind(a, b, permutations=np.inf)
        assert(res1.pvalue != res0.pvalue)
        assert(res2.pvalue == res0.pvalue)
        assert(res3.pvalue == res1.pvalue)

    def test_ttest_ind_exact_distribution(self):
        # the exact distribution of the test statistic should have
        # binom(na + nb, na) elements, all unique. This was not always true
        # in gh-4824; fixed by gh-13661.
        np.random.seed(0)
        a = np.random.rand(3)
        b = np.random.rand(4)

        data = np.concatenate((a, b))
        na, nb = len(a), len(b)

        permutations = 100000
        mat_perm, _ = _data_partitions(data, permutations, na)

        a = mat_perm[..., :na]
        b = mat_perm[..., nb:]
        t_stat = _calc_t_stat(a, b, True)
        n_unique = len(set(t_stat))
        assert n_unique == binom(na + nb, na)
        assert len(t_stat) == n_unique

    def test_ttest_ind_randperm_alternative(self):
        np.random.seed(0)
        N = 50
        a = np.random.rand(2, 3, N)
        b = np.random.rand(3, N)
        options_p = {'axis': -1, 'permutations': 1000, "random_state": 0}

        options_p.update(alternative="greater")
        res_g_ab = stats.ttest_ind(a, b, **options_p)
        res_g_ba = stats.ttest_ind(b, a, **options_p)

        options_p.update(alternative="less")
        res_l_ab = stats.ttest_ind(a, b, **options_p)
        res_l_ba = stats.ttest_ind(b, a, **options_p)

        # Alternative doesn't affect the statistic
        assert_equal(res_g_ab.statistic, res_l_ab.statistic)

        # Reversing order of inputs negates statistic
        assert_equal(res_g_ab.statistic, -res_g_ba.statistic)
        assert_equal(res_l_ab.statistic, -res_l_ba.statistic)

        # For random permutations, the chance of ties between the observed
        # test statistic and the population is small, so:
        assert_equal(res_g_ab.pvalue + res_l_ab.pvalue, 1)
        assert_equal(res_g_ba.pvalue + res_l_ba.pvalue, 1)

    @pytest.mark.slow()
    def test_ttest_ind_randperm_alternative2(self):
        np.random.seed(0)
        N = 50
        a = np.random.rand(N, 4)
        b = np.random.rand(N, 4)
        options_p = {'permutations': 20000, "random_state": 0}

        options_p.update(alternative="greater")
        res_g_ab = stats.ttest_ind(a, b, **options_p)

        options_p.update(alternative="less")
        res_l_ab = stats.ttest_ind(a, b, **options_p)

        options_p.update(alternative="two-sided")
        res_2_ab = stats.ttest_ind(a, b, **options_p)

        # For random permutations, the chance of ties between the observed
        # test statistic and the population is small, so:
        assert_equal(res_g_ab.pvalue + res_l_ab.pvalue, 1)

        # For for large sample sizes, the distribution should be approximately
        # symmetric, so these identities should be approximately satisfied
        mask = res_g_ab.pvalue <= 0.5
        assert_allclose(2 * res_g_ab.pvalue[mask],
                        res_2_ab.pvalue[mask], atol=2e-2)
        assert_allclose(2 * (1-res_g_ab.pvalue[~mask]),
                        res_2_ab.pvalue[~mask], atol=2e-2)
        assert_allclose(2 * res_l_ab.pvalue[~mask],
                        res_2_ab.pvalue[~mask], atol=2e-2)
        assert_allclose(2 * (1-res_l_ab.pvalue[mask]),
                        res_2_ab.pvalue[mask], atol=2e-2)

    def test_ttest_ind_permutation_nanpolicy(self):
        np.random.seed(0)
        N = 50
        a = np.random.rand(N, 5)
        b = np.random.rand(N, 5)
        a[5, 1] = np.nan
        b[8, 2] = np.nan
        a[9, 3] = np.nan
        b[9, 3] = np.nan
        options_p = {'permutations': 1000, "random_state": 0}

        # Raise
        options_p.update(nan_policy="raise")
        with assert_raises(ValueError, match="The input contains nan values"):
            res = stats.ttest_ind(a, b, **options_p)

        # Propagate
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning, "invalid value*")
            options_p.update(nan_policy="propagate")
            res = stats.ttest_ind(a, b, **options_p)

            mask = np.isnan(a).any(axis=0) | np.isnan(b).any(axis=0)
            res2 = stats.ttest_ind(a[:, ~mask], b[:, ~mask], **options_p)

            assert_equal(res.pvalue[mask], np.nan)
            assert_equal(res.statistic[mask], np.nan)

            assert_allclose(res.pvalue[~mask], res2.pvalue)
            assert_allclose(res.statistic[~mask], res2.statistic)

            # Propagate 1d
            res = stats.ttest_ind(a.ravel(), b.ravel(), **options_p)
            assert(np.isnan(res.pvalue))  # assert makes sure it's a scalar
            assert(np.isnan(res.statistic))

        # Omit
        options_p.update(nan_policy="omit")
        with assert_raises(ValueError,
                           match="nan-containing/masked inputs with"):
            res = stats.ttest_ind(a, b, **options_p)

    def test_ttest_ind_permutation_check_inputs(self):
        with assert_raises(ValueError, match="Permutations must be"):
            stats.ttest_ind(self.a2, self.b2, permutations=-3)
        with assert_raises(ValueError, match="Permutations must be"):
            stats.ttest_ind(self.a2, self.b2, permutations=1.5)
        with assert_raises(ValueError, match="'hello' cannot be used"):
            stats.ttest_ind(self.a, self.b, permutations=1,
                            random_state='hello')


class Test_ttest_ind_common:
    # for tests that are performed on variations of the t-test such as
    # permutations and trimming
    @pytest.mark.slow()
    @pytest.mark.parametrize("kwds", [{'permutations': 200, 'random_state': 0},
                                      {'trim': .2}, {}],
                             ids=["permutations", "trim", "basic"])
    @pytest.mark.parametrize('equal_var', [True, False],
                             ids=['equal_var', 'unequal_var'])
    def test_ttest_many_dims(self, kwds, equal_var):
        # Test that test works on many-dimensional arrays
        np.random.seed(0)
        a = np.random.rand(5, 4, 4, 7, 1, 6)
        b = np.random.rand(4, 1, 8, 2, 6)
        res = stats.ttest_ind(a, b, axis=-3, **kwds)

        # compare fully-vectorized t-test against t-test on smaller slice
        i, j, k = 2, 3, 1
        a2 = a[i, :, j, :, 0, :]
        b2 = b[:, 0, :, k, :]
        res2 = stats.ttest_ind(a2, b2, axis=-2, **kwds)
        assert_equal(res.statistic[i, :, j, k, :],
                     res2.statistic)
        assert_equal(res.pvalue[i, :, j, k, :],
                     res2.pvalue)

        # compare against t-test on one axis-slice at a time

        # manually broadcast with tile; move axis to end to simplify
        x = np.moveaxis(np.tile(a, (1, 1, 1, 1, 2, 1)), -3, -1)
        y = np.moveaxis(np.tile(b, (5, 1, 4, 1, 1, 1)), -3, -1)
        shape = x.shape[:-1]
        statistics = np.zeros(shape)
        pvalues = np.zeros(shape)
        for indices in product(*(range(i) for i in shape)):
            xi = x[indices]  # use tuple to index single axis slice
            yi = y[indices]
            res3 = stats.ttest_ind(xi, yi, axis=-1, **kwds)
            statistics[indices] = res3.statistic
            pvalues[indices] = res3.pvalue

        assert_allclose(statistics, res.statistic)
        assert_allclose(pvalues, res.pvalue)

    @pytest.mark.parametrize("kwds", [{'permutations': 200, 'random_state': 0},
                                      {'trim': .2}, {}],
                             ids=["trim", "permutations", "basic"])
    @pytest.mark.parametrize("axis", [-1, 0])
    def test_nans_on_axis(self, kwds, axis):
        # confirm that with `nan_policy='propagate'`, NaN results are returned
        # on the correct location
        a = np.random.randint(10, size=(5, 3, 10)).astype('float')
        b = np.random.randint(10, size=(5, 3, 10)).astype('float')
        # set some indices in `a` and `b` to be `np.nan`.
        a[0][2][3] = np.nan
        b[2][0][6] = np.nan

        # arbitrarily use `np.sum` as a baseline for which indices should be
        # NaNs
        expected = np.isnan(np.sum(a + b, axis=axis))
        # multidimensional inputs to `t.sf(np.abs(t), df)` with NaNs on some
        # indices throws an warning. See issue gh-13844
        with suppress_warnings() as sup, np.errstate(invalid="ignore"):
            sup.filter(RuntimeWarning,
                       "invalid value encountered in less_equal")
            res = stats.ttest_ind(a, b, axis=axis, **kwds)
        p_nans = np.isnan(res.pvalue)
        assert_array_equal(p_nans, expected)
        statistic_nans = np.isnan(res.statistic)
        assert_array_equal(statistic_nans, expected)


class Test_ttest_trim:
    params = [
        [[1, 2, 3], [1.1, 2.9, 4.2], 0.53619490753126731, -0.6864951273557258,
         .2],
        [[56, 128.6, 12, 123.8, 64.34, 78, 763.3], [1.1, 2.9, 4.2],
         0.00998909252078421, 4.591598691181999, .2],
        [[56, 128.6, 12, 123.8, 64.34, 78, 763.3], [1.1, 2.9, 4.2],
         0.10512380092302633, 2.832256715395378, .32],
        [[2.7, 2.7, 1.1, 3.0, 1.9, 3.0, 3.8, 3.8, 0.3, 1.9, 1.9],
         [6.5, 5.4, 8.1, 3.5, 0.5, 3.8, 6.8, 4.9, 9.5, 6.2, 4.1],
         0.002878909511344, -4.2461168970325, .2],
        [[-0.84504783, 0.13366078, 3.53601757, -0.62908581, 0.54119466,
          -1.16511574, -0.08836614, 1.18495416, 2.48028757, -1.58925028,
          -1.6706357, 0.3090472, -2.12258305, 0.3697304, -1.0415207,
          -0.57783497, -0.90997008, 1.09850192, 0.41270579, -1.4927376],
         [1.2725522, 1.1657899, 2.7509041, 1.2389013, -0.9490494, -1.0752459,
          1.1038576, 2.9912821, 3.5349111, 0.4171922, 1.0168959, -0.7625041,
          -0.4300008, 3.0431921, 1.6035947, 0.5285634, -0.7649405, 1.5575896,
          1.3670797, 1.1726023], 0.005293305834235, -3.0983317739483, .2]]

    @pytest.mark.parametrize("a,b,pr,tr,trim", params)
    def test_ttest_compare_r(self, a, b, pr, tr, trim):
        '''
        Using PairedData's yuen.t.test method. Something to note is that there
        are at least 3 R packages that come with a trimmed t-test method, and
        comparisons were made between them. It was found that PairedData's
        method's results match this method, SAS, and one of the other R
        methods. A notable discrepancy was the DescTools implementation of the
        function, which only sometimes agreed with SAS, WRS2, PairedData and
        this implementation. For this reason, most comparisons in R are made
        against PairedData's method.

        Rather than providing the input and output for all evaluations, here is
        a representative example:
        > library(PairedData)
        > a <- c(1, 2, 3)
        > b <- c(1.1, 2.9, 4.2)
        > options(digits=16)
        > yuen.t.test(a, b, tr=.2)

            Two-sample Yuen test, trim=0.2

        data:  x and y
        t = -0.68649512735573, df = 3.4104431643464, p-value = 0.5361949075313
        alternative hypothesis: true difference in trimmed means is not equal
        to 0
        95 percent confidence interval:
         -3.912777195645217  2.446110528978550
        sample estimates:
        trimmed mean of x trimmed mean of y
        2.000000000000000 2.73333333333333
        '''
        statistic, pvalue = stats.ttest_ind(a, b, trim=trim, equal_var=False)
        assert_allclose(statistic, tr, atol=1e-15)
        assert_allclose(pvalue, pr, atol=1e-15)

    def test_compare_SAS(self):
        # Source of the data used in this test:
        # https://support.sas.com/resources/papers/proceedings14/1660-2014.pdf
        a = [12, 14, 18, 25, 32, 44, 12, 14, 18, 25, 32, 44]
        b = [17, 22, 14, 12, 30, 29, 19, 17, 22, 14, 12, 30, 29, 19]
        # In this paper, a trimming percentage of 5% is used. However,
        # in their implementation, the number of values trimmed is rounded to
        # the nearest whole number. However, consistent with
        # `scipy.stats.trimmed_mean`, this test truncates to the lower
        # whole number. In this example, the paper notes that 1 value is
        # trimmed off of each side. 9% replicates this amount of trimming.
        statistic, pvalue = stats.ttest_ind(a, b, trim=.09, equal_var=False)
        assert_allclose(pvalue, 0.514522, atol=1e-6)
        assert_allclose(statistic, 0.669169, atol=1e-6)

    def test_equal_var(self):
        '''
        The PairedData library only supports unequal variances. To compare
        samples with equal variances, the multicon library is used.
        > library(multicon)
        > a <- c(2.7, 2.7, 1.1, 3.0, 1.9, 3.0, 3.8, 3.8, 0.3, 1.9, 1.9)
        > b <- c(6.5, 5.4, 8.1, 3.5, 0.5, 3.8, 6.8, 4.9, 9.5, 6.2, 4.1)
        > dv = c(a,b)
        > iv = c(rep('a', length(a)), rep('b', length(b)))
        > yuenContrast(dv~ iv, EQVAR = TRUE)
        $Ms
           N                 M wgt
        a 11 2.442857142857143   1
        b 11 5.385714285714286  -1

        $test
                              stat df              crit                   p
        results -4.246116897032513 12 2.178812829667228 0.00113508833897713
        '''
        a = [2.7, 2.7, 1.1, 3.0, 1.9, 3.0, 3.8, 3.8, 0.3, 1.9, 1.9]
        b = [6.5, 5.4, 8.1, 3.5, 0.5, 3.8, 6.8, 4.9, 9.5, 6.2, 4.1]
        # `equal_var=True` is default
        statistic, pvalue = stats.ttest_ind(a, b, trim=.2)
        assert_allclose(pvalue, 0.00113508833897713, atol=1e-10)
        assert_allclose(statistic, -4.246116897032513, atol=1e-10)

    @pytest.mark.parametrize('alt,pr,tr',
                             (('greater', 0.9985605452443, -4.2461168970325),
                              ('less', 0.001439454755672, -4.2461168970325),),
                             )
    def test_alternatives(self, alt, pr, tr):
        '''
        > library(PairedData)
        > a <- c(2.7,2.7,1.1,3.0,1.9,3.0,3.8,3.8,0.3,1.9,1.9)
        > b <- c(6.5,5.4,8.1,3.5,0.5,3.8,6.8,4.9,9.5,6.2,4.1)
        > options(digits=16)
        > yuen.t.test(a, b, alternative = 'greater')
        '''
        a = [2.7, 2.7, 1.1, 3.0, 1.9, 3.0, 3.8, 3.8, 0.3, 1.9, 1.9]
        b = [6.5, 5.4, 8.1, 3.5, 0.5, 3.8, 6.8, 4.9, 9.5, 6.2, 4.1]

        statistic, pvalue = stats.ttest_ind(a, b, trim=.2, equal_var=False,
                                            alternative=alt)
        assert_allclose(pvalue, pr, atol=1e-10)
        assert_allclose(statistic, tr, atol=1e-10)

    def test_errors_unsupported(self):
        # confirm that attempting to trim with NaNs or permutations raises an
        # error
        match = "Permutations are currently not supported with trimming."
        with assert_raises(ValueError, match=match):
            stats.ttest_ind([1, 2], [2, 3], trim=.2, permutations=2)
        match = ("not supported by permutation tests, one-sided asymptotic "
                 "tests, or trimmed tests.")
        with assert_raises(ValueError, match=match):
            stats.ttest_ind([1, 2], [2, np.nan, 3], trim=.2, nan_policy='omit')

    @pytest.mark.parametrize("trim", [-.2, .5, 1])
    def test_trim_bounds_error(self, trim):
        match = "Trimming percentage should be 0 <= `trim` < .5."
        with assert_raises(ValueError, match=match):
            stats.ttest_ind([1, 2], [2, 1], trim=trim)


def test__broadcast_concatenate():
    # test that _broadcast_concatenate properly broadcasts arrays along all
    # axes except `axis`, then concatenates along axis
    np.random.seed(0)
    a = np.random.rand(5, 4, 4, 3, 1, 6)
    b = np.random.rand(4, 1, 8, 2, 6)
    c = _broadcast_concatenate((a, b), axis=-3)
    # broadcast manually as an independent check
    a = np.tile(a, (1, 1, 1, 1, 2, 1))
    b = np.tile(b[None, ...], (5, 1, 4, 1, 1, 1))
    for index in product(*(range(i) for i in c.shape)):
        i, j, k, l, m, n = index
        if l < a.shape[-3]:
            assert a[i, j, k, l, m, n] == c[i, j, k, l, m, n]
        else:
            assert b[i, j, k, l - a.shape[-3], m, n] == c[i, j, k, l, m, n]


def test_ttest_ind_with_uneq_var():
    # check vs. R
    a = (1, 2, 3)
    b = (1.1, 2.9, 4.2)
    pr = 0.53619490753126731
    tr = -0.68649512735572582
    t, p = stats.ttest_ind(a, b, equal_var=False)
    assert_array_almost_equal([t,p], [tr, pr])
    # test from desc stats API
    assert_array_almost_equal(stats.ttest_ind_from_stats(*_desc_stats(a, b),
                                                         equal_var=False),
                              [t, p])

    a = (1, 2, 3, 4)
    pr = 0.84354139131608286
    tr = -0.2108663315950719
    t, p = stats.ttest_ind(a, b, equal_var=False)
    assert_array_almost_equal([t,p], [tr, pr])
    assert_array_almost_equal(stats.ttest_ind_from_stats(*_desc_stats(a, b),
                                                         equal_var=False),
                              [t, p])

    # regression test
    tr = 1.0912746897927283
    tr_uneq_n = 0.66745638708050492
    pr = 0.27647831993021388
    pr_uneq_n = 0.50873585065616544
    tpr = ([tr,-tr],[pr,pr])

    rvs3 = np.linspace(1,100, 25)
    rvs2 = np.linspace(1,100,100)
    rvs1 = np.linspace(5,105,100)
    rvs1_2D = np.array([rvs1, rvs2])

    rvs2_2D = np.array([rvs2, rvs1])

    t,p = stats.ttest_ind(rvs1, rvs2, axis=0, equal_var=False)
    assert_array_almost_equal([t,p],(tr,pr))
    assert_array_almost_equal(stats.ttest_ind_from_stats(*_desc_stats(rvs1,
                                                                      rvs2),
                                                         equal_var=False),
                              (t, p))

    t,p = stats.ttest_ind(rvs1, rvs3, axis=0, equal_var=False)
    assert_array_almost_equal([t,p], (tr_uneq_n, pr_uneq_n))
    assert_array_almost_equal(stats.ttest_ind_from_stats(*_desc_stats(rvs1,
                                                                      rvs3),
                                                         equal_var=False),
                              (t, p))

    t,p = stats.ttest_ind(rvs1_2D.T, rvs2_2D.T, axis=0, equal_var=False)
    assert_array_almost_equal([t,p],tpr)
    args = _desc_stats(rvs1_2D.T, rvs2_2D.T)
    assert_array_almost_equal(stats.ttest_ind_from_stats(*args,
                                                         equal_var=False),
                              (t, p))

    t,p = stats.ttest_ind(rvs1_2D, rvs2_2D, axis=1, equal_var=False)
    assert_array_almost_equal([t,p],tpr)
    args = _desc_stats(rvs1_2D, rvs2_2D, axis=1)
    assert_array_almost_equal(stats.ttest_ind_from_stats(*args,
                                                         equal_var=False),
                              (t, p))

    # test for namedtuple attribute results
    attributes = ('statistic', 'pvalue')
    res = stats.ttest_ind(rvs1, rvs2, axis=0, equal_var=False)
    check_named_results(res, attributes)

    # test on 3 dimensions
    rvs1_3D = np.dstack([rvs1_2D,rvs1_2D,rvs1_2D])
    rvs2_3D = np.dstack([rvs2_2D,rvs2_2D,rvs2_2D])
    t,p = stats.ttest_ind(rvs1_3D, rvs2_3D, axis=1, equal_var=False)
    assert_almost_equal(np.abs(t), np.abs(tr))
    assert_array_almost_equal(np.abs(p), pr)
    assert_equal(t.shape, (2, 3))
    args = _desc_stats(rvs1_3D, rvs2_3D, axis=1)
    t, p = stats.ttest_ind_from_stats(*args, equal_var=False)
    assert_almost_equal(np.abs(t), np.abs(tr))
    assert_array_almost_equal(np.abs(p), pr)
    assert_equal(t.shape, (2, 3))

    t,p = stats.ttest_ind(np.rollaxis(rvs1_3D,2), np.rollaxis(rvs2_3D,2),
                                   axis=2, equal_var=False)
    assert_array_almost_equal(np.abs(t), np.abs(tr))
    assert_array_almost_equal(np.abs(p), pr)
    assert_equal(t.shape, (3, 2))
    args = _desc_stats(np.rollaxis(rvs1_3D, 2),
                       np.rollaxis(rvs2_3D, 2), axis=2)
    t, p = stats.ttest_ind_from_stats(*args, equal_var=False)
    assert_array_almost_equal(np.abs(t), np.abs(tr))
    assert_array_almost_equal(np.abs(p), pr)
    assert_equal(t.shape, (3, 2))

    # test zero division problem
    t, p = stats.ttest_ind([0, 0, 0], [1, 1, 1], equal_var=False)
    assert_equal((np.abs(t), p), (np.inf, 0))
    with np.errstate(all='ignore'):
        assert_equal(stats.ttest_ind([0, 0, 0], [0, 0, 0], equal_var=False),
                     (np.nan, np.nan))

        # check that nan in input array result in nan output
        anan = np.array([[1, np.nan], [-1, 1]])
        assert_equal(stats.ttest_ind(anan, np.zeros((2, 2)), equal_var=False),
                     ([0, np.nan], [1, np.nan]))


def test_ttest_ind_nan_2nd_arg():
    # regression test for gh-6134: nans in the second arg were not handled
    x = [np.nan, 2.0, 3.0, 4.0]
    y = [1.0, 2.0, 1.0, 2.0]

    r1 = stats.ttest_ind(x, y, nan_policy='omit')
    r2 = stats.ttest_ind(y, x, nan_policy='omit')
    assert_allclose(r2.statistic, -r1.statistic, atol=1e-15)
    assert_allclose(r2.pvalue, r1.pvalue, atol=1e-15)

    # NB: arguments are not paired when NaNs are dropped
    r3 = stats.ttest_ind(y, x[1:])
    assert_allclose(r2, r3, atol=1e-15)

    # .. and this is consistent with R. R code:
    # x = c(NA, 2.0, 3.0, 4.0)
    # y = c(1.0, 2.0, 1.0, 2.0)
    # t.test(x, y, var.equal=TRUE)
    assert_allclose(r2, (-2.5354627641855498, 0.052181400457057901),
                    atol=1e-15)


def test_ttest_ind_empty_1d_returns_nan():
    # Two empty inputs should return a Ttest_indResult containing nan
    # for both values.
    result = stats.ttest_ind([], [])
    assert isinstance(result, stats.stats.Ttest_indResult)
    assert_equal(result, (np.nan, np.nan))


@pytest.mark.parametrize('b, expected_shape',
                         [(np.empty((1, 5, 0)), (3, 5)),
                          (np.empty((1, 0, 0)), (3, 0))])
def test_ttest_ind_axis_size_zero(b, expected_shape):
    # In this test, the length of the axis dimension is zero.
    # The results should be arrays containing nan with shape
    # given by the broadcast nonaxis dimensions.
    a = np.empty((3, 1, 0))
    result = stats.ttest_ind(a, b, axis=-1)
    assert isinstance(result, stats.stats.Ttest_indResult)
    expected_value = np.full(expected_shape, fill_value=np.nan)
    assert_equal(result.statistic, expected_value)
    assert_equal(result.pvalue, expected_value)


def test_ttest_ind_nonaxis_size_zero():
    # In this test, the length of the axis dimension is nonzero,
    # but one of the nonaxis dimensions has length 0.  Check that
    # we still get the correctly broadcast shape, which is (5, 0)
    # in this case.
    a = np.empty((1, 8, 0))
    b = np.empty((5, 8, 1))
    result = stats.ttest_ind(a, b, axis=1)
    assert isinstance(result, stats.stats.Ttest_indResult)
    assert_equal(result.statistic.shape, (5, 0))
    assert_equal(result.pvalue.shape, (5, 0))


def test_ttest_ind_nonaxis_size_zero_different_lengths():
    # In this test, the length of the axis dimension is nonzero,
    # and that size is different in the two inputs,
    # and one of the nonaxis dimensions has length 0.  Check that
    # we still get the correctly broadcast shape, which is (5, 0)
    # in this case.
    a = np.empty((1, 7, 0))
    b = np.empty((5, 8, 1))
    result = stats.ttest_ind(a, b, axis=1)
    assert isinstance(result, stats.stats.Ttest_indResult)
    assert_equal(result.statistic.shape, (5, 0))
    assert_equal(result.pvalue.shape, (5, 0))


def test_gh5686():
    mean1, mean2 = np.array([1, 2]), np.array([3, 4])
    std1, std2 = np.array([5, 3]), np.array([4, 5])
    nobs1, nobs2 = np.array([130, 140]), np.array([100, 150])
    # This will raise a TypeError unless gh-5686 is fixed.
    stats.ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2)


def test_ttest_ind_from_stats_inputs_zero():
    # Regression test for gh-6409.
    result = stats.ttest_ind_from_stats(0, 0, 6, 0, 0, 6, equal_var=False)
    assert_equal(result, [np.nan, np.nan])


def test_ttest_1samp_new():
    n1, n2, n3 = (10,15,20)
    rvn1 = stats.norm.rvs(loc=5,scale=10,size=(n1,n2,n3))

    # check multidimensional array and correct axis handling
    # deterministic rvn1 and rvn2 would be better as in test_ttest_rel
    t1,p1 = stats.ttest_1samp(rvn1[:,:,:], np.ones((n2,n3)),axis=0)
    t2,p2 = stats.ttest_1samp(rvn1[:,:,:], 1,axis=0)
    t3,p3 = stats.ttest_1samp(rvn1[:,0,0], 1)
    assert_array_almost_equal(t1,t2, decimal=14)
    assert_almost_equal(t1[0,0],t3, decimal=14)
    assert_equal(t1.shape, (n2,n3))

    t1,p1 = stats.ttest_1samp(rvn1[:,:,:], np.ones((n1,n3)),axis=1)
    t2,p2 = stats.ttest_1samp(rvn1[:,:,:], 1,axis=1)
    t3,p3 = stats.ttest_1samp(rvn1[0,:,0], 1)
    assert_array_almost_equal(t1,t2, decimal=14)
    assert_almost_equal(t1[0,0],t3, decimal=14)
    assert_equal(t1.shape, (n1,n3))

    t1,p1 = stats.ttest_1samp(rvn1[:,:,:], np.ones((n1,n2)),axis=2)
    t2,p2 = stats.ttest_1samp(rvn1[:,:,:], 1,axis=2)
    t3,p3 = stats.ttest_1samp(rvn1[0,0,:], 1)
    assert_array_almost_equal(t1,t2, decimal=14)
    assert_almost_equal(t1[0,0],t3, decimal=14)
    assert_equal(t1.shape, (n1,n2))

    # test zero division problem
    t, p = stats.ttest_1samp([0, 0, 0], 1)
    assert_equal((np.abs(t), p), (np.inf, 0))

    # test alternative parameter
    # Convert from two-sided p-values to one sided using T result data.
    def convert(t, p, alt):
        if (t < 0 and alt == "less") or (t > 0 and alt == "greater"):
            return p / 2
        return 1 - (p / 2)
    converter = np.vectorize(convert)
    tr, pr = stats.ttest_1samp(rvn1[:, :, :], 1)

    t, p = stats.ttest_1samp(rvn1[:, :, :], 1, alternative="greater")
    pc = converter(tr, pr, "greater")
    assert_allclose(p, pc)
    assert_allclose(t, tr)

    t, p = stats.ttest_1samp(rvn1[:, :, :], 1, alternative="less")
    pc = converter(tr, pr, "less")
    assert_allclose(p, pc)
    assert_allclose(t, tr)

    with np.errstate(all='ignore'):
        assert_equal(stats.ttest_1samp([0, 0, 0], 0), (np.nan, np.nan))

        # check that nan in input array result in nan output
        anan = np.array([[1, np.nan],[-1, 1]])
        assert_equal(stats.ttest_1samp(anan, 0), ([0, np.nan], [1, np.nan]))


class TestDescribe:
    def test_describe_scalar(self):
        with suppress_warnings() as sup, np.errstate(invalid="ignore"):
            sup.filter(RuntimeWarning, "Degrees of freedom <= 0 for slice")
            n, mm, m, v, sk, kurt = stats.describe(4.)
        assert_equal(n, 1)
        assert_equal(mm, (4.0, 4.0))
        assert_equal(m, 4.0)
        assert_(np.isnan(v))
        assert_array_almost_equal(sk, 0.0, decimal=13)
        assert_array_almost_equal(kurt, -3.0, decimal=13)

    def test_describe_numbers(self):
        x = np.vstack((np.ones((3,4)), np.full((2, 4), 2)))
        nc, mmc = (5, ([1., 1., 1., 1.], [2., 2., 2., 2.]))
        mc = np.array([1.4, 1.4, 1.4, 1.4])
        vc = np.array([0.3, 0.3, 0.3, 0.3])
        skc = [0.40824829046386357] * 4
        kurtc = [-1.833333333333333] * 4
        n, mm, m, v, sk, kurt = stats.describe(x)
        assert_equal(n, nc)
        assert_equal(mm, mmc)
        assert_equal(m, mc)
        assert_equal(v, vc)
        assert_array_almost_equal(sk, skc, decimal=13)
        assert_array_almost_equal(kurt, kurtc, decimal=13)
        n, mm, m, v, sk, kurt = stats.describe(x.T, axis=1)
        assert_equal(n, nc)
        assert_equal(mm, mmc)
        assert_equal(m, mc)
        assert_equal(v, vc)
        assert_array_almost_equal(sk, skc, decimal=13)
        assert_array_almost_equal(kurt, kurtc, decimal=13)

        x = np.arange(10.)
        x[9] = np.nan

        nc, mmc = (9, (0.0, 8.0))
        mc = 4.0
        vc = 7.5
        skc = 0.0
        kurtc = -1.2300000000000002
        n, mm, m, v, sk, kurt = stats.describe(x, nan_policy='omit')
        assert_equal(n, nc)
        assert_equal(mm, mmc)
        assert_equal(m, mc)
        assert_equal(v, vc)
        assert_array_almost_equal(sk, skc)
        assert_array_almost_equal(kurt, kurtc, decimal=13)

        assert_raises(ValueError, stats.describe, x, nan_policy='raise')
        assert_raises(ValueError, stats.describe, x, nan_policy='foobar')

    def test_describe_result_attributes(self):
        actual = stats.describe(np.arange(5))
        attributes = ('nobs', 'minmax', 'mean', 'variance', 'skewness',
                      'kurtosis')
        check_named_results(actual, attributes)

    def test_describe_ddof(self):
        x = np.vstack((np.ones((3, 4)), np.full((2, 4), 2)))
        nc, mmc = (5, ([1., 1., 1., 1.], [2., 2., 2., 2.]))
        mc = np.array([1.4, 1.4, 1.4, 1.4])
        vc = np.array([0.24, 0.24, 0.24, 0.24])
        skc = [0.40824829046386357] * 4
        kurtc = [-1.833333333333333] * 4
        n, mm, m, v, sk, kurt = stats.describe(x, ddof=0)
        assert_equal(n, nc)
        assert_allclose(mm, mmc, rtol=1e-15)
        assert_allclose(m, mc, rtol=1e-15)
        assert_allclose(v, vc, rtol=1e-15)
        assert_array_almost_equal(sk, skc, decimal=13)
        assert_array_almost_equal(kurt, kurtc, decimal=13)

    def test_describe_axis_none(self):
        x = np.vstack((np.ones((3, 4)), np.full((2, 4), 2)))

        # expected values
        e_nobs, e_minmax = (20, (1.0, 2.0))
        e_mean = 1.3999999999999999
        e_var = 0.25263157894736848
        e_skew = 0.4082482904638634
        e_kurt = -1.8333333333333333

        # actual values
        a = stats.describe(x, axis=None)

        assert_equal(a.nobs, e_nobs)
        assert_almost_equal(a.minmax, e_minmax)
        assert_almost_equal(a.mean, e_mean)
        assert_almost_equal(a.variance, e_var)
        assert_array_almost_equal(a.skewness, e_skew, decimal=13)
        assert_array_almost_equal(a.kurtosis, e_kurt, decimal=13)

    def test_describe_empty(self):
        assert_raises(ValueError, stats.describe, [])


def test_normalitytests():
    assert_raises(ValueError, stats.skewtest, 4.)
    assert_raises(ValueError, stats.kurtosistest, 4.)
    assert_raises(ValueError, stats.normaltest, 4.)

    # numbers verified with R: dagoTest in package fBasics
    st_normal, st_skew, st_kurt = (3.92371918, 1.98078826, -0.01403734)
    pv_normal, pv_skew, pv_kurt = (0.14059673, 0.04761502, 0.98880019)
    pv_skew_less, pv_kurt_less = 1 - pv_skew / 2, pv_kurt / 2
    pv_skew_greater, pv_kurt_greater = pv_skew / 2, 1 - pv_kurt / 2
    x = np.array((-2, -1, 0, 1, 2, 3)*4)**2
    attributes = ('statistic', 'pvalue')

    assert_array_almost_equal(stats.normaltest(x), (st_normal, pv_normal))
    check_named_results(stats.normaltest(x), attributes)
    assert_array_almost_equal(stats.skewtest(x), (st_skew, pv_skew))
    assert_array_almost_equal(stats.skewtest(x, alternative='less'),
                              (st_skew, pv_skew_less))
    assert_array_almost_equal(stats.skewtest(x, alternative='greater'),
                              (st_skew, pv_skew_greater))
    check_named_results(stats.skewtest(x), attributes)
    assert_array_almost_equal(stats.kurtosistest(x), (st_kurt, pv_kurt))
    assert_array_almost_equal(stats.kurtosistest(x, alternative='less'),
                              (st_kurt, pv_kurt_less))
    assert_array_almost_equal(stats.kurtosistest(x, alternative='greater'),
                              (st_kurt, pv_kurt_greater))
    check_named_results(stats.kurtosistest(x), attributes)

    # some more intuitive tests for kurtosistest and skewtest.
    # see gh-13549.
    # skew parameter is 1 > 0
    a1 = stats.skewnorm.rvs(a=1, size=10000, random_state=123)
    pval = stats.skewtest(a1, alternative='greater').pvalue
    assert_almost_equal(pval, 0.0, decimal=5)
    # excess kurtosis of laplace is 3 > 0
    a2 = stats.laplace.rvs(size=10000, random_state=123)
    pval = stats.kurtosistest(a2, alternative='greater').pvalue
    assert_almost_equal(pval, 0.0)

    # Test axis=None (equal to axis=0 for 1-D input)
    assert_array_almost_equal(stats.normaltest(x, axis=None),
           (st_normal, pv_normal))
    assert_array_almost_equal(stats.skewtest(x, axis=None),
           (st_skew, pv_skew))
    assert_array_almost_equal(stats.kurtosistest(x, axis=None),
           (st_kurt, pv_kurt))

    x = np.arange(10.)
    x[9] = np.nan
    with np.errstate(invalid="ignore"):
        assert_array_equal(stats.skewtest(x), (np.nan, np.nan))

    expected = (1.0184643553962129, 0.30845733195153502)
    assert_array_almost_equal(stats.skewtest(x, nan_policy='omit'), expected)

    with np.errstate(all='ignore'):
        assert_raises(ValueError, stats.skewtest, x, nan_policy='raise')
    assert_raises(ValueError, stats.skewtest, x, nan_policy='foobar')
    assert_raises(ValueError, stats.skewtest, x, nan_policy='omit',
                  alternative='less')
    assert_raises(ValueError, stats.skewtest, x, nan_policy='omit',
                  alternative='greater')
    assert_raises(ValueError, stats.skewtest, list(range(8)),
                  alternative='foobar')

    x = np.arange(30.)
    x[29] = np.nan
    with np.errstate(all='ignore'):
        assert_array_equal(stats.kurtosistest(x), (np.nan, np.nan))

    expected = (-2.2683547379505273, 0.023307594135872967)
    assert_array_almost_equal(stats.kurtosistest(x, nan_policy='omit'),
                              expected)

    assert_raises(ValueError, stats.kurtosistest, x, nan_policy='raise')
    assert_raises(ValueError, stats.kurtosistest, x, nan_policy='foobar')
    assert_raises(ValueError, stats.kurtosistest, x, nan_policy='omit',
                  alternative='less')
    assert_raises(ValueError, stats.kurtosistest, x, nan_policy='omit',
                  alternative='greater')
    assert_raises(ValueError, stats.kurtosistest, list(range(20)),
                  alternative='foobar')

    with np.errstate(all='ignore'):
        assert_array_equal(stats.normaltest(x), (np.nan, np.nan))

    expected = (6.2260409514287449, 0.04446644248650191)
    assert_array_almost_equal(stats.normaltest(x, nan_policy='omit'), expected)

    assert_raises(ValueError, stats.normaltest, x, nan_policy='raise')
    assert_raises(ValueError, stats.normaltest, x, nan_policy='foobar')

    # regression test for issue gh-9033: x cleary non-normal but power of
    # negtative denom needs to be handled correctly to reject normality
    counts = [128, 0, 58, 7, 0, 41, 16, 0, 0, 167]
    x = np.hstack([np.full(c, i) for i, c in enumerate(counts)])
    assert_equal(stats.kurtosistest(x)[1] < 0.01, True)


class TestRankSums:

    np.random.seed(0)
    x, y = np.random.rand(2, 10)

    @pytest.mark.parametrize('alternative', ['less', 'greater', 'two-sided'])
    def test_ranksums_result_attributes(self, alternative):
        # ranksums pval = mannwhitneyu pval w/out continuity or tie correction
        res1 = stats.ranksums(self.x, self.y,
                              alternative=alternative).pvalue
        res2 = stats.mannwhitneyu(self.x, self.y, use_continuity=False,
                                  alternative=alternative).pvalue
        assert_allclose(res1, res2)

    def test_ranksums_named_results(self):
        res = stats.ranksums(self.x, self.y)
        check_named_results(res, ('statistic', 'pvalue'))

    def test_input_validation(self):
        with assert_raises(ValueError, match="alternative must be 'less'"):
            stats.ranksums(self.x, self.y, alternative='foobar')


class TestJarqueBera:
    def test_jarque_bera_stats(self):
        np.random.seed(987654321)
        x = np.random.normal(0, 1, 100000)
        y = np.random.chisquare(10000, 100000)
        z = np.random.rayleigh(1, 100000)

        assert_equal(stats.jarque_bera(x)[0], stats.jarque_bera(x).statistic)
        assert_equal(stats.jarque_bera(x)[1], stats.jarque_bera(x).pvalue)

        assert_equal(stats.jarque_bera(y)[0], stats.jarque_bera(y).statistic)
        assert_equal(stats.jarque_bera(y)[1], stats.jarque_bera(y).pvalue)

        assert_equal(stats.jarque_bera(z)[0], stats.jarque_bera(z).statistic)
        assert_equal(stats.jarque_bera(z)[1], stats.jarque_bera(z).pvalue)

        assert_(stats.jarque_bera(x)[1] > stats.jarque_bera(y)[1])
        assert_(stats.jarque_bera(x).pvalue > stats.jarque_bera(y).pvalue)

        assert_(stats.jarque_bera(x)[1] > stats.jarque_bera(z)[1])
        assert_(stats.jarque_bera(x).pvalue > stats.jarque_bera(z).pvalue)

        assert_(stats.jarque_bera(y)[1] > stats.jarque_bera(z)[1])
        assert_(stats.jarque_bera(y).pvalue > stats.jarque_bera(z).pvalue)

    def test_jarque_bera_array_like(self):
        np.random.seed(987654321)
        x = np.random.normal(0, 1, 100000)

        jb_test1 = JB1, p1 = stats.jarque_bera(list(x))
        jb_test2 = JB2, p2 = stats.jarque_bera(tuple(x))
        jb_test3 = JB3, p3 = stats.jarque_bera(x.reshape(2, 50000))

        assert_(JB1 == JB2 == JB3 == jb_test1.statistic == jb_test2.statistic == jb_test3.statistic)
        assert_(p1 == p2 == p3 == jb_test1.pvalue == jb_test2.pvalue == jb_test3.pvalue)

    def test_jarque_bera_size(self):
        assert_raises(ValueError, stats.jarque_bera, [])


def test_skewtest_too_few_samples():
    # Regression test for ticket #1492.
    # skewtest requires at least 8 samples; 7 should raise a ValueError.
    x = np.arange(7.0)
    assert_raises(ValueError, stats.skewtest, x)


def test_kurtosistest_too_few_samples():
    # Regression test for ticket #1425.
    # kurtosistest requires at least 5 samples; 4 should raise a ValueError.
    x = np.arange(4.0)
    assert_raises(ValueError, stats.kurtosistest, x)


class TestMannWhitneyU:
    X = [19.8958398126694, 19.5452691647182, 19.0577309166425, 21.716543054589,
         20.3269502208702, 20.0009273294025, 19.3440043632957, 20.4216806548105,
         19.0649894736528, 18.7808043120398, 19.3680942943298, 19.4848044069953,
         20.7514611265663, 19.0894948874598, 19.4975522356628, 18.9971170734274,
         20.3239606288208, 20.6921298083835, 19.0724259532507, 18.9825187935021,
         19.5144462609601, 19.8256857844223, 20.5174677102032, 21.1122407995892,
         17.9490854922535, 18.2847521114727, 20.1072217648826, 18.6439891962179,
         20.4970638083542, 19.5567594734914]

    Y = [19.2790668029091, 16.993808441865, 18.5416338448258, 17.2634018833575,
         19.1577183624616, 18.5119655377495, 18.6068455037221, 18.8358343362655,
         19.0366413269742, 18.1135025515417, 19.2201873866958, 17.8344909022841,
         18.2894380745856, 18.6661374133922, 19.9688601693252, 16.0672254617636,
         19.00596360572, 19.201561539032, 19.0487501090183, 19.0847908674356]

    significant = 14

    def test_mannwhitneyu_one_sided(self):
        u1, p1 = stats.mannwhitneyu(self.X, self.Y, alternative='less')
        u2, p2 = stats.mannwhitneyu(self.Y, self.X, alternative='greater')
        u3, p3 = stats.mannwhitneyu(self.X, self.Y, alternative='greater')
        u4, p4 = stats.mannwhitneyu(self.Y, self.X, alternative='less')

        assert_equal(p1, p2)
        assert_equal(p3, p4)
        assert_(p1 != p3)
        assert_equal(u1, 498)
        assert_equal(u2, 102)
        assert_equal(u3, 498)
        assert_equal(u4, 102)
        assert_approx_equal(p1, 0.999957683256589, significant=self.significant)
        assert_approx_equal(p3, 4.5941632666275e-05, significant=self.significant)

    def test_mannwhitneyu_two_sided(self):
        u1, p1 = stats.mannwhitneyu(self.X, self.Y, alternative='two-sided')
        u2, p2 = stats.mannwhitneyu(self.Y, self.X, alternative='two-sided')

        assert_equal(p1, p2)
        assert_equal(u1, 498)
        assert_equal(u2, 102)
        assert_approx_equal(p1, 9.188326533255e-05,
                            significant=self.significant)

    def test_mannwhitneyu_no_correct_one_sided(self):
        u1, p1 = stats.mannwhitneyu(self.X, self.Y, False,
                                    alternative='less')
        u2, p2 = stats.mannwhitneyu(self.Y, self.X, False,
                                    alternative='greater')
        u3, p3 = stats.mannwhitneyu(self.X, self.Y, False,
                                    alternative='greater')
        u4, p4 = stats.mannwhitneyu(self.Y, self.X, False,
                                    alternative='less')

        assert_equal(p1, p2)
        assert_equal(p3, p4)
        assert_(p1 != p3)
        assert_equal(u1, 498)
        assert_equal(u2, 102)
        assert_equal(u3, 498)
        assert_equal(u4, 102)
        assert_approx_equal(p1, 0.999955905990004, significant=self.significant)
        assert_approx_equal(p3, 4.40940099958089e-05, significant=self.significant)

    def test_mannwhitneyu_no_correct_two_sided(self):
        u1, p1 = stats.mannwhitneyu(self.X, self.Y, False,
                                    alternative='two-sided')
        u2, p2 = stats.mannwhitneyu(self.Y, self.X, False,
                                    alternative='two-sided')

        assert_equal(p1, p2)
        assert_equal(u1, 498)
        assert_equal(u2, 102)
        assert_approx_equal(p1, 8.81880199916178e-05,
                            significant=self.significant)

    def test_mannwhitneyu_ones(self):
        # test for gh-1428
        x = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 2., 1., 1., 1., 1., 2., 1., 1., 2., 1., 1., 2.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 3., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1.])

        y = np.array([1., 1., 1., 1., 1., 1., 1., 2., 1., 2., 1., 1., 1., 1.,
                      2., 1., 1., 1., 2., 1., 1., 1., 1., 1., 2., 1., 1., 3.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 1., 2., 1.,
                      1., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 2.,
                      2., 1., 1., 2., 1., 1., 2., 1., 2., 1., 1., 1., 1., 2.,
                      2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 2., 1., 1., 1., 1., 1., 2., 2., 2., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      2., 1., 1., 2., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 2., 1., 1., 1., 2., 1., 1.,
                      1., 1., 1., 1.])

        # checked against R wilcox.test
        assert_allclose(stats.mannwhitneyu(x, y, alternative='less'),
                        (16980.5, 2.8214327656317373e-005))
        # p-value from R, e.g. wilcox.test(x, y, alternative="g")
        assert_allclose(stats.mannwhitneyu(x, y, alternative='greater'),
                        (16980.5, 0.9999719954296))
        assert_allclose(stats.mannwhitneyu(x, y, alternative='two-sided'),
                        (16980.5, 5.642865531266e-05))

    def test_mannwhitneyu_result_attributes(self):
        # test for namedtuple attribute results
        attributes = ('statistic', 'pvalue')
        res = stats.mannwhitneyu(self.X, self.Y, alternative="less")
        check_named_results(res, attributes)


def test_pointbiserial():
    # same as mstats test except for the nan
    # Test data: https://web.archive.org/web/20060504220742/https://support.sas.com/ctx/samples/index.jsp?sid=490&tab=output
    x = [1,0,1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,
         0,0,0,0,1]
    y = [14.8,13.8,12.4,10.1,7.1,6.1,5.8,4.6,4.3,3.5,3.3,3.2,3.0,
         2.8,2.8,2.5,2.4,2.3,2.1,1.7,1.7,1.5,1.3,1.3,1.2,1.2,1.1,
         0.8,0.7,0.6,0.5,0.2,0.2,0.1]
    assert_almost_equal(stats.pointbiserialr(x, y)[0], 0.36149, 5)

    # test for namedtuple attribute results
    attributes = ('correlation', 'pvalue')
    res = stats.pointbiserialr(x, y)
    check_named_results(res, attributes)


def test_obrientransform():
    # A couple tests calculated by hand.
    x1 = np.array([0, 2, 4])
    t1 = stats.obrientransform(x1)
    expected = [7, -2, 7]
    assert_allclose(t1[0], expected)

    x2 = np.array([0, 3, 6, 9])
    t2 = stats.obrientransform(x2)
    expected = np.array([30, 0, 0, 30])
    assert_allclose(t2[0], expected)

    # Test two arguments.
    a, b = stats.obrientransform(x1, x2)
    assert_equal(a, t1[0])
    assert_equal(b, t2[0])

    # Test three arguments.
    a, b, c = stats.obrientransform(x1, x2, x1)
    assert_equal(a, t1[0])
    assert_equal(b, t2[0])
    assert_equal(c, t1[0])

    # This is a regression test to check np.var replacement.
    # The author of this test didn't separately verify the numbers.
    x1 = np.arange(5)
    result = np.array(
      [[5.41666667, 1.04166667, -0.41666667, 1.04166667, 5.41666667],
       [21.66666667, 4.16666667, -1.66666667, 4.16666667, 21.66666667]])
    assert_array_almost_equal(stats.obrientransform(x1, 2*x1), result, decimal=8)

    # Example from "O'Brien Test for Homogeneity of Variance"
    # by Herve Abdi.
    values = range(5, 11)
    reps = np.array([5, 11, 9, 3, 2, 2])
    data = np.repeat(values, reps)
    transformed_values = np.array([3.1828, 0.5591, 0.0344,
                                   1.6086, 5.2817, 11.0538])
    expected = np.repeat(transformed_values, reps)
    result = stats.obrientransform(data)
    assert_array_almost_equal(result[0], expected, decimal=4)


def check_equal_gmean(array_like, desired, axis=None, dtype=None, rtol=1e-7, weights=None):
    # Note this doesn't test when axis is not specified
    x = stats.gmean(array_like, axis=axis, dtype=dtype, weights=weights)
    assert_allclose(x, desired, rtol=rtol)
    assert_equal(x.dtype, dtype)

def check_equal_hmean(array_like, desired, axis=None, dtype=None, rtol=1e-7):
    x = stats.hmean(array_like, axis=axis, dtype=dtype)
    assert_allclose(x, desired, rtol=rtol)
    assert_equal(x.dtype, dtype)


class TestHarMean:
    def test_1d_list(self):
        #  Test a 1d list
        a = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        desired = 34.1417152147
        check_equal_hmean(a, desired)

        a = [1, 2, 3, 4]
        desired = 4. / (1. / 1 + 1. / 2 + 1. / 3 + 1. / 4)
        check_equal_hmean(a, desired)

    def test_1d_array(self):
        #  Test a 1d array
        a = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        desired = 34.1417152147
        check_equal_hmean(a, desired)

    def test_1d_array_with_zero(self):
        a = np.array([1, 0])
        desired = 0.0
        assert_equal(stats.hmean(a), desired)

    def test_1d_array_with_negative_value(self):
        a = np.array([1, 0, -1])
        assert_raises(ValueError, stats.hmean, a)

    # Note the next tests use axis=None as default, not axis=0
    def test_2d_list(self):
        #  Test a 2d list
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = 38.6696271841
        check_equal_hmean(a, desired)

    def test_2d_array(self):
        #  Test a 2d array
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = 38.6696271841
        check_equal_hmean(np.array(a), desired)

    def test_2d_axis0(self):
        #  Test a 2d list with axis=0
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = np.array([22.88135593, 39.13043478, 52.90076336, 65.45454545])
        check_equal_hmean(a, desired, axis=0)

    def test_2d_axis0_with_zero(self):
        a = [[10, 0, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = np.array([22.88135593, 0.0, 52.90076336, 65.45454545])
        assert_allclose(stats.hmean(a, axis=0), desired)

    def test_2d_axis1(self):
        #  Test a 2d list with axis=1
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = np.array([19.2, 63.03939962, 103.80078637])
        check_equal_hmean(a, desired, axis=1)

    def test_2d_axis1_with_zero(self):
        a = [[10, 0, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = np.array([0.0, 63.03939962, 103.80078637])
        assert_allclose(stats.hmean(a, axis=1), desired)

    def test_2d_matrix_axis0(self):
        #  Test a 2d list with axis=0
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = matrix([[22.88135593, 39.13043478, 52.90076336, 65.45454545]])
        check_equal_hmean(matrix(a), desired, axis=0)

    def test_2d_matrix_axis1(self):
        #  Test a 2d list with axis=1
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = matrix([[19.2, 63.03939962, 103.80078637]]).T
        check_equal_hmean(matrix(a), desired, axis=1)


class TestGeoMean:
    def test_1d_list(self):
        #  Test a 1d list
        a = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        desired = 45.2872868812
        check_equal_gmean(a, desired)

        a = [1, 2, 3, 4]
        desired = power(1 * 2 * 3 * 4, 1. / 4.)
        check_equal_gmean(a, desired, rtol=1e-14)

    def test_1d_array(self):
        #  Test a 1d array
        a = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        desired = 45.2872868812
        check_equal_gmean(a, desired)

        a = array([1, 2, 3, 4], float32)
        desired = power(1 * 2 * 3 * 4, 1. / 4.)
        check_equal_gmean(a, desired, dtype=float32)

    # Note the next tests use axis=None as default, not axis=0
    def test_2d_list(self):
        #  Test a 2d list
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = 52.8885199
        check_equal_gmean(a, desired)

    def test_2d_array(self):
        #  Test a 2d array
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = 52.8885199
        check_equal_gmean(array(a), desired)

    def test_2d_axis0(self):
        #  Test a 2d list with axis=0
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = np.array([35.56893304, 49.32424149, 61.3579244, 72.68482371])
        check_equal_gmean(a, desired, axis=0)

        a = array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        desired = array([1, 2, 3, 4])
        check_equal_gmean(a, desired, axis=0, rtol=1e-14)

    def test_2d_axis1(self):
        #  Test a 2d list with axis=1
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = np.array([22.13363839, 64.02171746, 104.40086817])
        check_equal_gmean(a, desired, axis=1)

        a = array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        v = power(1 * 2 * 3 * 4, 1. / 4.)
        desired = array([v, v, v])
        check_equal_gmean(a, desired, axis=1, rtol=1e-14)

    def test_2d_matrix_axis0(self):
        #  Test a 2d list with axis=0
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = matrix([[35.56893304, 49.32424149, 61.3579244, 72.68482371]])
        check_equal_gmean(matrix(a), desired, axis=0)

        a = array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        desired = matrix([1, 2, 3, 4])
        check_equal_gmean(matrix(a), desired, axis=0, rtol=1e-14)

        a = array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        desired = matrix(stats.gmean(a, axis=0))
        check_equal_gmean(matrix(a), desired, axis=0, rtol=1e-14)

    def test_2d_matrix_axis1(self):
        #  Test a 2d list with axis=1
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = matrix([[22.13363839, 64.02171746, 104.40086817]]).T
        check_equal_gmean(matrix(a), desired, axis=1)

        a = array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        v = power(1 * 2 * 3 * 4, 1. / 4.)
        desired = matrix([[v], [v], [v]])
        check_equal_gmean(matrix(a), desired, axis=1, rtol=1e-14)

    def test_large_values(self):
        a = array([1e100, 1e200, 1e300])
        desired = 1e200
        check_equal_gmean(a, desired, rtol=1e-13)

    def test_1d_list0(self):
        #  Test a 1d list with zero element
        a = [10, 20, 30, 40, 50, 60, 70, 80, 90, 0]
        desired = 0.0  # due to exp(-inf)=0
        with np.errstate(all='ignore'):
            check_equal_gmean(a, desired)

    def test_1d_array0(self):
        #  Test a 1d array with zero element
        a = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 0])
        desired = 0.0  # due to exp(-inf)=0
        with np.errstate(divide='ignore'):
            check_equal_gmean(a, desired)

    def test_1d_list_neg(self):
        #  Test a 1d list with negative element
        a = [10, 20, 30, 40, 50, 60, 70, 80, 90, -1]
        desired = np.nan  # due to log(-1) = nan
        with np.errstate(invalid='ignore'):
            check_equal_gmean(a, desired)

    def test_weights_1d_list(self):
        # Desired result from:
        # https://www.dummies.com/education/math/business-statistics/how-to-find-the-weighted-geometric-mean-of-a-data-set/
        weights = [2, 5, 6, 4, 3]
        a = [1, 2, 3, 4, 5]
        desired = 2.77748
        check_equal_gmean(a, desired, weights=weights, rtol=1e-5)

    def test_weights_1d_array(self):
        # Desired result from:
        # https://www.dummies.com/education/math/business-statistics/how-to-find-the-weighted-geometric-mean-of-a-data-set/
        a = np.array([1, 2, 3, 4, 5])
        weights = np.array([2, 5, 6, 4, 3])
        desired = 2.77748
        check_equal_gmean(a, desired, weights=weights, rtol=1e-5)

    def test_weights_masked_1d_array(self):
        # Desired result from:
        # https://www.dummies.com/education/math/business-statistics/how-to-find-the-weighted-geometric-mean-of-a-data-set/
        a = np.array([1, 2, 3, 4, 5, 6])
        weights = np.ma.array([2, 5, 6, 4, 3, 5], mask=[0, 0, 0, 0, 0, 1])
        desired = 2.77748
        check_equal_gmean(a, desired, weights=weights, rtol=1e-5)


class TestGeometricStandardDeviation:
    # must add 1 as `gstd` is only defined for positive values
    array_1d = np.arange(2 * 3 * 4) + 1
    gstd_array_1d = 2.294407613602
    array_3d = array_1d.reshape(2, 3, 4)

    def test_1d_array(self):
        gstd_actual = stats.gstd(self.array_1d)
        assert_allclose(gstd_actual, self.gstd_array_1d)

    def test_1d_numeric_array_like_input(self):
        gstd_actual = stats.gstd(tuple(self.array_1d))
        assert_allclose(gstd_actual, self.gstd_array_1d)

    def test_raises_value_error_non_array_like_input(self):
        with pytest.raises(ValueError, match='Invalid array input'):
            stats.gstd('This should fail as it can not be cast to an array.')

    def test_raises_value_error_zero_entry(self):
        with pytest.raises(ValueError, match='Non positive value'):
            stats.gstd(np.append(self.array_1d, [0]))

    def test_raises_value_error_negative_entry(self):
        with pytest.raises(ValueError, match='Non positive value'):
            stats.gstd(np.append(self.array_1d, [-1]))

    def test_raises_value_error_inf_entry(self):
        with pytest.raises(ValueError, match='Infinite value'):
            stats.gstd(np.append(self.array_1d, [np.inf]))

    def test_propagates_nan_values(self):
        a = array([[1, 1, 1, 16], [np.nan, 1, 2, 3]])
        gstd_actual = stats.gstd(a, axis=1)
        assert_allclose(gstd_actual, np.array([4, np.nan]))

    def test_ddof_equal_to_number_of_observations(self):
        with pytest.raises(ValueError, match='Degrees of freedom <= 0'):
            stats.gstd(self.array_1d, ddof=self.array_1d.size)

    def test_3d_array(self):
        gstd_actual = stats.gstd(self.array_3d, axis=None)
        assert_allclose(gstd_actual, self.gstd_array_1d)

    def test_3d_array_axis_type_tuple(self):
        gstd_actual = stats.gstd(self.array_3d, axis=(1,2))
        assert_allclose(gstd_actual, [2.12939215, 1.22120169])

    def test_3d_array_axis_0(self):
        gstd_actual = stats.gstd(self.array_3d, axis=0)
        gstd_desired = np.array([
            [6.1330555493918, 3.958900210120, 3.1206598248344, 2.6651441426902],
            [2.3758135028411, 2.174581428192, 2.0260062829505, 1.9115518327308],
            [1.8205343606803, 1.746342404566, 1.6846557065742, 1.6325269194382]
        ])
        assert_allclose(gstd_actual, gstd_desired)

    def test_3d_array_axis_1(self):
        gstd_actual = stats.gstd(self.array_3d, axis=1)
        gstd_desired = np.array([
            [3.118993630946, 2.275985934063, 1.933995977619, 1.742896469724],
            [1.271693593916, 1.254158641801, 1.238774141609, 1.225164057869]
        ])
        assert_allclose(gstd_actual, gstd_desired)

    def test_3d_array_axis_2(self):
        gstd_actual = stats.gstd(self.array_3d, axis=2)
        gstd_desired = np.array([
            [1.8242475707664, 1.2243686572447, 1.1318311657788],
            [1.0934830582351, 1.0724479791887, 1.0591498540749]
        ])
        assert_allclose(gstd_actual, gstd_desired)

    def test_masked_3d_array(self):
        ma = np.ma.masked_where(self.array_3d > 16, self.array_3d)
        gstd_actual = stats.gstd(ma, axis=2)
        gstd_desired = stats.gstd(self.array_3d, axis=2)
        mask = [[0, 0, 0], [0, 1, 1]]
        assert_allclose(gstd_actual, gstd_desired)
        assert_equal(gstd_actual.mask, mask)


def test_binomtest():
    # precision tests compared to R for ticket:986
    pp = np.concatenate((np.linspace(0.1, 0.2, 5),
                         np.linspace(0.45, 0.65, 5),
                         np.linspace(0.85, 0.95, 5)))
    n = 501
    x = 450
    results = [0.0, 0.0, 1.0159969301994141e-304,
               2.9752418572150531e-275, 7.7668382922535275e-250,
               2.3381250925167094e-099, 7.8284591587323951e-081,
               9.9155947819961383e-065, 2.8729390725176308e-050,
               1.7175066298388421e-037, 0.0021070691951093692,
               0.12044570587262322, 0.88154763174802508, 0.027120993063129286,
               2.6102587134694721e-006]

    for p, res in zip(pp, results):
        assert_approx_equal(stats.binom_test(x, n, p), res,
                            significant=12, err_msg='fail forp=%f' % p)

    assert_approx_equal(stats.binom_test(50, 100, 0.1),
                        5.8320387857343647e-024,
                        significant=12)


def test_binomtest2():
    # test added for issue #2384
    res2 = [
        [1.0, 1.0],
        [0.5, 1.0, 0.5],
        [0.25, 1.00, 1.00, 0.25],
        [0.125, 0.625, 1.000, 0.625, 0.125],
        [0.0625, 0.3750, 1.0000, 1.0000, 0.3750, 0.0625],
        [0.03125, 0.21875, 0.68750, 1.00000, 0.68750, 0.21875, 0.03125],
        [0.015625, 0.125000, 0.453125, 1.000000, 1.000000, 0.453125, 0.125000,
         0.015625],
        [0.0078125, 0.0703125, 0.2890625, 0.7265625, 1.0000000, 0.7265625,
         0.2890625, 0.0703125, 0.0078125],
        [0.00390625, 0.03906250, 0.17968750, 0.50781250, 1.00000000,
         1.00000000, 0.50781250, 0.17968750, 0.03906250, 0.00390625],
        [0.001953125, 0.021484375, 0.109375000, 0.343750000, 0.753906250,
         1.000000000, 0.753906250, 0.343750000, 0.109375000, 0.021484375,
         0.001953125]
    ]

    for k in range(1, 11):
        res1 = [stats.binom_test(v, k, 0.5) for v in range(k + 1)]
        assert_almost_equal(res1, res2[k-1], decimal=10)


def test_binomtest3():
    # test added for issue #2384
    # test when x == n*p and neighbors
    res3 = [stats.binom_test(v, v*k, 1./k)
            for v in range(1, 11) for k in range(2, 11)]
    assert_equal(res3, np.ones(len(res3), int))

    # > bt=c()
    # > for(i in as.single(1:10)) {
    # +     for(k in as.single(2:10)) {
    # +         bt = c(bt, binom.test(i-1, k*i,(1/k))$p.value);
    # +         print(c(i+1, k*i,(1/k)))
    # +     }
    # + }
    binom_testm1 = np.array([
         0.5, 0.5555555555555556, 0.578125, 0.5904000000000003,
         0.5981224279835393, 0.603430543396034, 0.607304096221924,
         0.610255656871054, 0.612579511000001, 0.625, 0.670781893004115,
         0.68853759765625, 0.6980101120000006, 0.703906431368616,
         0.70793209416498, 0.7108561134173507, 0.713076544331419,
         0.714820192935702, 0.6875, 0.7268709038256367, 0.7418963909149174,
         0.74986110468096, 0.7548015520398076, 0.7581671424768577,
         0.760607984787832, 0.762459425024199, 0.7639120677676575, 0.7265625,
         0.761553963657302, 0.774800934828818, 0.7818005980538996,
         0.78613491480358, 0.789084353140195, 0.7912217659828884,
         0.79284214559524, 0.794112956558801, 0.75390625, 0.7856929451142176,
         0.7976688481430754, 0.8039848974727624, 0.807891868948366,
         0.8105487660137676, 0.812473307174702, 0.8139318233591120,
         0.815075399104785, 0.7744140625, 0.8037322594985427,
         0.814742863657656, 0.8205425178645808, 0.8241275984172285,
         0.8265645374416, 0.8283292196088257, 0.829666291102775,
         0.8307144686362666, 0.7905273437499996, 0.8178712053954738,
         0.828116983756619, 0.833508948940494, 0.8368403871552892,
         0.839104213210105, 0.840743186196171, 0.84198481438049,
         0.8429580531563676, 0.803619384765625, 0.829338573944648,
         0.8389591907548646, 0.84401876783902, 0.84714369697889,
         0.8492667010581667, 0.850803474598719, 0.851967542858308,
         0.8528799045949524, 0.8145294189453126, 0.838881732845347,
         0.847979024541911, 0.852760894015685, 0.8557134656773457,
         0.8577190131799202, 0.85917058278431, 0.860270010472127,
         0.861131648404582, 0.823802947998047, 0.846984756807511,
         0.855635653643743, 0.860180994825685, 0.86298688573253,
         0.864892525675245, 0.866271647085603, 0.867316125625004,
         0.8681346531755114
        ])

    # > bt=c()
    # > for(i in as.single(1:10)) {
    # +     for(k in as.single(2:10)) {
    # +         bt = c(bt, binom.test(i+1, k*i,(1/k))$p.value);
    # +         print(c(i+1, k*i,(1/k)))
    # +     }
    # + }

    binom_testp1 = np.array([
         0.5, 0.259259259259259, 0.26171875, 0.26272, 0.2632244513031551,
         0.2635138663069203, 0.2636951804161073, 0.2638162407564354,
         0.2639010709000002, 0.625, 0.4074074074074074, 0.42156982421875,
         0.4295746560000003, 0.43473045988554, 0.4383309503172684,
         0.4409884859402103, 0.4430309389962837, 0.444649849401104, 0.6875,
         0.4927602499618962, 0.5096031427383425, 0.5189636628480,
         0.5249280070771274, 0.5290623300865124, 0.5320974248125793,
         0.5344204730474308, 0.536255847400756, 0.7265625, 0.5496019313526808,
         0.5669248746708034, 0.576436455045805, 0.5824538812831795,
         0.5866053321547824, 0.589642781414643, 0.5919618019300193,
         0.593790427805202, 0.75390625, 0.590868349763505, 0.607983393277209,
         0.617303847446822, 0.623172512167948, 0.627208862156123,
         0.6301556891501057, 0.632401894928977, 0.6341708982290303,
         0.7744140625, 0.622562037497196, 0.639236102912278, 0.648263335014579,
         0.65392850011132, 0.657816519817211, 0.660650782947676,
         0.662808780346311, 0.6645068560246006, 0.7905273437499996,
         0.6478843304312477, 0.6640468318879372, 0.6727589686071775,
         0.6782129857784873, 0.681950188903695, 0.684671508668418,
         0.686741824999918, 0.688369886732168, 0.803619384765625,
         0.668716055304315, 0.684360013879534, 0.6927642396829181,
         0.6980155964704895, 0.701609591890657, 0.7042244320992127,
         0.7062125081341817, 0.707775152962577, 0.8145294189453126,
         0.686243374488305, 0.7013873696358975, 0.709501223328243,
         0.714563595144314, 0.718024953392931, 0.7205416252126137,
         0.722454130389843, 0.723956813292035, 0.823802947998047,
         0.701255953767043, 0.715928221686075, 0.723772209289768,
         0.7286603031173616, 0.7319999279787631, 0.7344267920995765,
         0.736270323773157, 0.737718376096348
        ])

    res4_p1 = [stats.binom_test(v+1, v*k, 1./k)
               for v in range(1, 11) for k in range(2, 11)]
    res4_m1 = [stats.binom_test(v-1, v*k, 1./k)
               for v in range(1, 11) for k in range(2, 11)]

    assert_almost_equal(res4_p1, binom_testp1, decimal=13)
    assert_almost_equal(res4_m1, binom_testm1, decimal=13)


class TestTrim:
    # test trim functions
    def test_trim1(self):
        a = np.arange(11)
        assert_equal(np.sort(stats.trim1(a, 0.1)), np.arange(10))
        assert_equal(np.sort(stats.trim1(a, 0.2)), np.arange(9))
        assert_equal(np.sort(stats.trim1(a, 0.2, tail='left')),
                     np.arange(2, 11))
        assert_equal(np.sort(stats.trim1(a, 3/11., tail='left')),
                     np.arange(3, 11))
        assert_equal(stats.trim1(a, 1.0), [])
        assert_equal(stats.trim1(a, 1.0, tail='left'), [])

        # empty input
        assert_equal(stats.trim1([], 0.1), [])
        assert_equal(stats.trim1([], 3/11., tail='left'), [])
        assert_equal(stats.trim1([], 4/6.), [])

    def test_trimboth(self):
        a = np.arange(11)
        assert_equal(np.sort(stats.trimboth(a, 3/11.)), np.arange(3, 8))
        assert_equal(np.sort(stats.trimboth(a, 0.2)),
                     np.array([2, 3, 4, 5, 6, 7, 8]))
        assert_equal(np.sort(stats.trimboth(np.arange(24).reshape(6, 4), 0.2)),
                     np.arange(4, 20).reshape(4, 4))
        assert_equal(np.sort(stats.trimboth(np.arange(24).reshape(4, 6).T,
                                            2/6.)),
                     np.array([[2, 8, 14, 20], [3, 9, 15, 21]]))
        assert_raises(ValueError, stats.trimboth,
                      np.arange(24).reshape(4, 6).T, 4/6.)

        # empty input
        assert_equal(stats.trimboth([], 0.1), [])
        assert_equal(stats.trimboth([], 3/11.), [])
        assert_equal(stats.trimboth([], 4/6.), [])

    def test_trim_mean(self):
        # don't use pre-sorted arrays
        a = np.array([4, 8, 2, 0, 9, 5, 10, 1, 7, 3, 6])
        idx = np.array([3, 5, 0, 1, 2, 4])
        a2 = np.arange(24).reshape(6, 4)[idx, :]
        a3 = np.arange(24).reshape(6, 4, order='F')[idx, :]
        assert_equal(stats.trim_mean(a3, 2/6.),
                     np.array([2.5, 8.5, 14.5, 20.5]))
        assert_equal(stats.trim_mean(a2, 2/6.),
                     np.array([10., 11., 12., 13.]))
        idx4 = np.array([1, 0, 3, 2])
        a4 = np.arange(24).reshape(4, 6)[idx4, :]
        assert_equal(stats.trim_mean(a4, 2/6.),
                     np.array([9., 10., 11., 12., 13., 14.]))
        # shuffled arange(24) as array_like
        a = [7, 11, 12, 21, 16, 6, 22, 1, 5, 0, 18, 10, 17, 9, 19, 15, 23,
             20, 2, 14, 4, 13, 8, 3]
        assert_equal(stats.trim_mean(a, 2/6.), 11.5)
        assert_equal(stats.trim_mean([5,4,3,1,2,0], 2/6.), 2.5)

        # check axis argument
        np.random.seed(1234)
        a = np.random.randint(20, size=(5, 6, 4, 7))
        for axis in [0, 1, 2, 3, -1]:
            res1 = stats.trim_mean(a, 2/6., axis=axis)
            res2 = stats.trim_mean(np.rollaxis(a, axis), 2/6.)
            assert_equal(res1, res2)

        res1 = stats.trim_mean(a, 2/6., axis=None)
        res2 = stats.trim_mean(a.ravel(), 2/6.)
        assert_equal(res1, res2)

        assert_raises(ValueError, stats.trim_mean, a, 0.6)

        # empty input
        assert_equal(stats.trim_mean([], 0.0), np.nan)
        assert_equal(stats.trim_mean([], 0.6), np.nan)


class TestSigmaClip:
    def test_sigmaclip1(self):
        a = np.concatenate((np.linspace(9.5, 10.5, 31), np.linspace(0, 20, 5)))
        fact = 4  # default
        c, low, upp = stats.sigmaclip(a)
        assert_(c.min() > low)
        assert_(c.max() < upp)
        assert_equal(low, c.mean() - fact*c.std())
        assert_equal(upp, c.mean() + fact*c.std())
        assert_equal(c.size, a.size)

    def test_sigmaclip2(self):
        a = np.concatenate((np.linspace(9.5, 10.5, 31), np.linspace(0, 20, 5)))
        fact = 1.5
        c, low, upp = stats.sigmaclip(a, fact, fact)
        assert_(c.min() > low)
        assert_(c.max() < upp)
        assert_equal(low, c.mean() - fact*c.std())
        assert_equal(upp, c.mean() + fact*c.std())
        assert_equal(c.size, 4)
        assert_equal(a.size, 36)  # check original array unchanged

    def test_sigmaclip3(self):
        a = np.concatenate((np.linspace(9.5, 10.5, 11),
                            np.linspace(-100, -50, 3)))
        fact = 1.8
        c, low, upp = stats.sigmaclip(a, fact, fact)
        assert_(c.min() > low)
        assert_(c.max() < upp)
        assert_equal(low, c.mean() - fact*c.std())
        assert_equal(upp, c.mean() + fact*c.std())
        assert_equal(c, np.linspace(9.5, 10.5, 11))

    def test_sigmaclip_result_attributes(self):
        a = np.concatenate((np.linspace(9.5, 10.5, 11),
                            np.linspace(-100, -50, 3)))
        fact = 1.8
        res = stats.sigmaclip(a, fact, fact)
        attributes = ('clipped', 'lower', 'upper')
        check_named_results(res, attributes)

    def test_std_zero(self):
        # regression test #8632
        x = np.ones(10)
        assert_equal(stats.sigmaclip(x)[0], x)


class TestAlexanderGovern:
    def test_compare_dtypes(self):
        args = [[13, 13, 13, 13, 13, 13, 13, 12, 12],
                [14, 13, 12, 12, 12, 12, 12, 11, 11],
                [14, 14, 13, 13, 13, 13, 13, 12, 12],
                [15, 14, 13, 13, 13, 12, 12, 12, 11]]
        args_int16 = np.array(args, dtype=np.int16)
        args_int32 = np.array(args, dtype=np.int32)
        args_uint8 = np.array(args, dtype=np.uint8)
        args_float64 = np.array(args, dtype=np.float64)

        res_int16 = stats.alexandergovern(*args_int16)
        res_int32 = stats.alexandergovern(*args_int32)
        res_unit8 = stats.alexandergovern(*args_uint8)
        res_float64 = stats.alexandergovern(*args_float64)

        assert (res_int16.pvalue == res_int32.pvalue ==
                res_unit8.pvalue == res_float64.pvalue)
        assert (res_int16.statistic == res_int32.statistic ==
                res_unit8.statistic == res_float64.statistic)

    def test_bad_inputs(self):
        # input array is of size zero
        with assert_raises(ValueError, match="Input sample size must be"
                                             " greater than one."):
            stats.alexandergovern([1, 2], [])
        # input is a singular non list element
        with assert_raises(ValueError, match="Input sample size must be"
                                             " greater than one."):
            stats.alexandergovern([1, 2], 2)
        # input list is of size 1
        with assert_raises(ValueError, match="Input sample size must be"
                                             " greater than one."):
            stats.alexandergovern([1, 2], [2])
        # inputs are not finite (infinity)
        with assert_raises(ValueError, match="Input samples must be finite."):
            stats.alexandergovern([1, 2], [np.inf, np.inf])
        # inputs are multidimensional
        with assert_raises(ValueError, match="Input samples must be one"
                                             "-dimensional"):
            stats.alexandergovern([1, 2], [[1, 2], [3, 4]])

    def test_compare_r(self):
        '''
        Data generated in R with
        > set.seed(1)
        > library("onewaytests")
        > library("tibble")
        > y <- c(rnorm(40, sd=10),
        +        rnorm(30, sd=15),
        +        rnorm(20, sd=20))
        > x <- c(rep("one", times=40),
        +        rep("two", times=30),
        +        rep("eight", times=20))
        > x <- factor(x)
        > ag.test(y ~ x, tibble(y,x))

        Alexander-Govern Test (alpha = 0.05)
        -------------------------------------------------------------
        data : y and x

        statistic  : 1.359941
        parameter  : 2
        p.value    : 0.5066321

        Result     : Difference is not statistically significant.
        -------------------------------------------------------------
        Example adapted from:
        https://eval-serv2.metpsy.uni-jena.de/wiki-metheval-hp/index.php/R_FUN_Alexander-Govern

        '''
        one = [-6.264538107423324, 1.8364332422208225, -8.356286124100471,
               15.952808021377916, 3.295077718153605, -8.204683841180152,
               4.874290524284853, 7.383247051292173, 5.757813516534923,
               -3.0538838715635603, 15.11781168450848, 3.898432364114311,
               -6.2124058054180376, -22.146998871774997, 11.249309181431082,
               -0.4493360901523085, -0.16190263098946087, 9.438362106852992,
               8.212211950980885, 5.939013212175088, 9.189773716082183,
               7.821363007310671, 0.745649833651906, -19.89351695863373,
               6.198257478947102, -0.5612873952900078, -1.557955067053293,
               -14.707523838992744, -4.781500551086204, 4.179415601997024,
               13.58679551529044, -1.0278772734299553, 3.876716115593691,
               -0.5380504058290512, -13.770595568286065, -4.149945632996798,
               -3.942899537103493, -0.5931339671118566, 11.000253719838831,
               7.631757484575442]

        two = [-2.4678539438038034, -3.8004252020476135, 10.454450631071062,
               8.34994798010486, -10.331335418242798, -10.612427354431794,
               5.468729432052455, 11.527993867731237, -1.6851931822534207,
               13.216615896813222, 5.971588205506021, -9.180395898761569,
               5.116795371366372, -16.94044644121189, 21.495355525515556,
               29.7059984775879, -5.508322146997636, -15.662019394747961,
               8.545794411636193, -2.0258190582123654, 36.024266407571645,
               -0.5886000409975387, 10.346090436761651, 0.4200323817099909,
               -11.14909813323608, 2.8318844927151434, -27.074379433365568,
               21.98332292344329, 2.2988000731784655, 32.58917505543229]

        eight = [9.510190577993251, -14.198928618436291, 12.214527069781099,
                 -18.68195263288503, -25.07266800478204, 5.828924710349257,
                 -8.86583746436866, 0.02210703263248262, 1.4868264830332811,
                 -11.79041892376144, -11.37337465637004, -2.7035723024766414,
                 23.56173993146409, -30.47133600859524, 11.878923752568431,
                 6.659007424270365, 21.261996745527256, -6.083678472686013,
                 7.400376198325763, 5.341975815444621]
        soln = stats.alexandergovern(one, two, eight)
        assert_allclose(soln.statistic, 1.3599405447999450836)
        assert_allclose(soln.pvalue, 0.50663205309676440091)

    def test_compare_scholar(self):
        '''
        Data taken from 'The Modification and Evaluation of the
        Alexander-Govern Test in Terms of Power' by Kingsley Ochuko, T.,
        Abdullah, S., Binti Zain, Z., & Soaad Syed Yahaya, S. (2015).
        '''
        young = [482.43, 484.36, 488.84, 495.15, 495.24, 502.69, 504.62,
                 518.29, 519.1, 524.1, 524.12, 531.18, 548.42, 572.1, 584.68,
                 609.09, 609.53, 666.63, 676.4]
        middle = [335.59, 338.43, 353.54, 404.27, 437.5, 469.01, 485.85,
                  487.3, 493.08, 494.31, 499.1, 886.41]
        old = [519.01, 528.5, 530.23, 536.03, 538.56, 538.83, 557.24, 558.61,
               558.95, 565.43, 586.39, 594.69, 629.22, 645.69, 691.84]
        soln = stats.alexandergovern(young, middle, old)
        assert_allclose(soln.statistic, 5.3237, atol=1e-3)
        assert_allclose(soln.pvalue, 0.06982, atol=1e-4)

        # verify with ag.test in r
        '''
        > library("onewaytests")
        > library("tibble")
        > young <- c(482.43, 484.36, 488.84, 495.15, 495.24, 502.69, 504.62,
        +                  518.29, 519.1, 524.1, 524.12, 531.18, 548.42, 572.1,
        +                  584.68, 609.09, 609.53, 666.63, 676.4)
        > middle <- c(335.59, 338.43, 353.54, 404.27, 437.5, 469.01, 485.85,
        +                   487.3, 493.08, 494.31, 499.1, 886.41)
        > old <- c(519.01, 528.5, 530.23, 536.03, 538.56, 538.83, 557.24,
        +                   558.61, 558.95, 565.43, 586.39, 594.69, 629.22,
        +                   645.69, 691.84)
        > young_fct <- c(rep("young", times=19))
        > middle_fct <-c(rep("middle", times=12))
        > old_fct <- c(rep("old", times=15))
        > ag.test(a ~ b, tibble(a=c(young, middle, old), b=factor(c(young_fct,
        +                                              middle_fct, old_fct))))

        Alexander-Govern Test (alpha = 0.05)
        -------------------------------------------------------------
        data : a and b

        statistic  : 5.324629
        parameter  : 2
        p.value    : 0.06978651

        Result     : Difference is not statistically significant.
        -------------------------------------------------------------

        '''
        assert_allclose(soln.statistic, 5.324629)
        assert_allclose(soln.pvalue, 0.06978651)

    def test_compare_scholar3(self):
        '''
        Data taken from 'Robustness And Comparative Power Of WelchAspin,
        Alexander-Govern And Yuen Tests Under Non-Normality And Variance
        Heteroscedasticity', by Ayed A. Almoied. 2017. Page 34-37.
        https://digitalcommons.wayne.edu/cgi/viewcontent.cgi?article=2775&context=oa_dissertations
        '''
        x1 = [-1.77559, -1.4113, -0.69457, -0.54148, -0.18808, -0.07152,
              0.04696, 0.051183, 0.148695, 0.168052, 0.422561, 0.458555,
              0.616123, 0.709968, 0.839956, 0.857226, 0.929159, 0.981442,
              0.999554, 1.642958]
        x2 = [-1.47973, -1.2722, -0.91914, -0.80916, -0.75977, -0.72253,
              -0.3601, -0.33273, -0.28859, -0.09637, -0.08969, -0.01824,
              0.260131, 0.289278, 0.518254, 0.683003, 0.877618, 1.172475,
              1.33964, 1.576766]
        soln = stats.alexandergovern(x1, x2)
        assert_allclose(soln.statistic, 0.713526, atol=1e-5)
        assert_allclose(soln.pvalue, 0.398276, atol=1e-5)

        '''
        tested in ag.test in R:
        > library("onewaytests")
        > library("tibble")
        > x1 <- c(-1.77559, -1.4113, -0.69457, -0.54148, -0.18808, -0.07152,
        +          0.04696, 0.051183, 0.148695, 0.168052, 0.422561, 0.458555,
        +          0.616123, 0.709968, 0.839956, 0.857226, 0.929159, 0.981442,
        +          0.999554, 1.642958)
        > x2 <- c(-1.47973, -1.2722, -0.91914, -0.80916, -0.75977, -0.72253,
        +         -0.3601, -0.33273, -0.28859, -0.09637, -0.08969, -0.01824,
        +         0.260131, 0.289278, 0.518254, 0.683003, 0.877618, 1.172475,
        +         1.33964, 1.576766)
        > x1_fact <- c(rep("x1", times=20))
        > x2_fact <- c(rep("x2", times=20))
        > a <- c(x1, x2)
        > b <- factor(c(x1_fact, x2_fact))
        > ag.test(a ~ b, tibble(a, b))
        Alexander-Govern Test (alpha = 0.05)
        -------------------------------------------------------------
        data : a and b

        statistic  : 0.7135182
        parameter  : 1
        p.value    : 0.3982783

        Result     : Difference is not statistically significant.
        -------------------------------------------------------------
        '''
        assert_allclose(soln.statistic, 0.7135182)
        assert_allclose(soln.pvalue, 0.3982783)

    def test_nan_policy_propogate(self):
        args = [[1, 2, 3, 4], [1, np.nan]]
        # default nan_policy is 'propagate'
        res = stats.alexandergovern(*args)
        assert_equal(res.pvalue, np.nan)
        assert_equal(res.statistic, np.nan)

    def test_nan_policy_raise(self):
        args = [[1, 2, 3, 4], [1, np.nan]]
        with assert_raises(ValueError, match="The input contains nan values"):
            stats.alexandergovern(*args, nan_policy='raise')

    def test_nan_policy_omit(self):
        args_nan = [[1, 2, 3, None, 4], [1, np.nan, 19, 25]]
        args_no_nan = [[1, 2, 3, 4], [1, 19, 25]]
        res_nan = stats.alexandergovern(*args_nan, nan_policy='omit')
        res_no_nan = stats.alexandergovern(*args_no_nan)
        assert_equal(res_nan.pvalue, res_no_nan.pvalue)
        assert_equal(res_nan.statistic, res_no_nan.statistic)

    def test_constant_input(self):
        # Zero variance input, consistent with `stats.pearsonr`
        with assert_warns(AlexanderGovernConstantInputWarning):
            res = stats.alexandergovern([0.667, 0.667, 0.667],
                                        [0.123, 0.456, 0.789])
            assert_equal(res.statistic, np.nan)
            assert_equal(res.pvalue, np.nan)


class TestFOneWay:

    def test_trivial(self):
        # A trivial test of stats.f_oneway, with F=0.
        F, p = stats.f_oneway([0, 2], [0, 2])
        assert_equal(F, 0.0)
        assert_equal(p, 1.0)

    def test_basic(self):
        # Despite being a floating point calculation, this data should
        # result in F being exactly 2.0.
        F, p = stats.f_oneway([0, 2], [2, 4])
        assert_equal(F, 2.0)
        assert_allclose(p, 1 - np.sqrt(0.5), rtol=1e-14)

    def test_known_exact(self):
        # Another trivial dataset for which the exact F and p can be
        # calculated.
        F, p = stats.f_oneway([2], [2], [2, 3, 4])
        # The use of assert_equal might be too optimistic, but the calculation
        # in this case is trivial enough that it is likely to go through with
        # no loss of precision.
        assert_equal(F, 3/5)
        assert_equal(p, 5/8)

    def test_large_integer_array(self):
        a = np.array([655, 788], dtype=np.uint16)
        b = np.array([789, 772], dtype=np.uint16)
        F, p = stats.f_oneway(a, b)
        # The expected value was verified by computing it with mpmath with
        # 40 digits of precision.
        assert_allclose(F, 0.77450216931805540, rtol=1e-14)

    def test_result_attributes(self):
        a = np.array([655, 788], dtype=np.uint16)
        b = np.array([789, 772], dtype=np.uint16)
        res = stats.f_oneway(a, b)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes)

    def test_nist(self):
        # These are the nist ANOVA files. They can be found at:
        # https://www.itl.nist.gov/div898/strd/anova/anova.html
        filenames = ['SiRstv.dat', 'SmLs01.dat', 'SmLs02.dat', 'SmLs03.dat',
                     'AtmWtAg.dat', 'SmLs04.dat', 'SmLs05.dat', 'SmLs06.dat',
                     'SmLs07.dat', 'SmLs08.dat', 'SmLs09.dat']

        for test_case in filenames:
            rtol = 1e-7
            fname = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 'data/nist_anova', test_case))
            with open(fname, 'r') as f:
                content = f.read().split('\n')
            certified = [line.split() for line in content[40:48]
                         if line.strip()]
            dataf = np.loadtxt(fname, skiprows=60)
            y, x = dataf.T
            y = y.astype(int)
            caty = np.unique(y)
            f = float(certified[0][-1])

            xlist = [x[y == i] for i in caty]
            res = stats.f_oneway(*xlist)

            # With the hard test cases we relax the tolerance a bit.
            hard_tc = ('SmLs07.dat', 'SmLs08.dat', 'SmLs09.dat')
            if test_case in hard_tc:
                rtol = 1e-4

            assert_allclose(res[0], f, rtol=rtol,
                            err_msg='Failing testcase: %s' % test_case)

    @pytest.mark.parametrize("a, b, expected", [
        (np.array([42, 42, 42]), np.array([7, 7, 7]), (np.inf, 0)),
        (np.array([42, 42, 42]), np.array([42, 42, 42]), (np.nan, np.nan))
        ])
    def test_constant_input(self, a, b, expected):
        # For more details, look on https://github.com/scipy/scipy/issues/11669
        with assert_warns(stats.F_onewayConstantInputWarning):
            f, p = stats.f_oneway(a, b)
            assert f, p == expected

    @pytest.mark.parametrize('axis', [-2, -1, 0, 1])
    def test_2d_inputs(self, axis):
        a = np.array([[1, 4, 3, 3],
                      [2, 5, 3, 3],
                      [3, 6, 3, 3],
                      [2, 3, 3, 3],
                      [1, 4, 3, 3]])
        b = np.array([[3, 1, 5, 3],
                      [4, 6, 5, 3],
                      [4, 3, 5, 3],
                      [1, 5, 5, 3],
                      [5, 5, 5, 3],
                      [2, 3, 5, 3],
                      [8, 2, 5, 3],
                      [2, 2, 5, 3]])
        c = np.array([[4, 3, 4, 3],
                      [4, 2, 4, 3],
                      [5, 4, 4, 3],
                      [5, 4, 4, 3]])

        if axis in [-1, 1]:
            a = a.T
            b = b.T
            c = c.T
            take_axis = 0
        else:
            take_axis = 1

        with assert_warns(stats.F_onewayConstantInputWarning):
            f, p = stats.f_oneway(a, b, c, axis=axis)

        # Verify that the result computed with the 2d arrays matches
        # the result of calling f_oneway individually on each slice.
        for j in [0, 1]:
            fj, pj = stats.f_oneway(np.take(a, j, take_axis),
                                    np.take(b, j, take_axis),
                                    np.take(c, j, take_axis))
            assert_allclose(f[j], fj, rtol=1e-14)
            assert_allclose(p[j], pj, rtol=1e-14)
        for j in [2, 3]:
            with assert_warns(stats.F_onewayConstantInputWarning):
                fj, pj = stats.f_oneway(np.take(a, j, take_axis),
                                        np.take(b, j, take_axis),
                                        np.take(c, j, take_axis))
                assert_equal(f[j], fj)
                assert_equal(p[j], pj)

    def test_3d_inputs(self):
        # Some 3-d arrays. (There is nothing special about the values.)
        a = 1/np.arange(1.0, 4*5*7 + 1).reshape(4, 5, 7)
        b = 2/np.arange(1.0, 4*8*7 + 1).reshape(4, 8, 7)
        c = np.cos(1/np.arange(1.0, 4*4*7 + 1).reshape(4, 4, 7))

        f, p = stats.f_oneway(a, b, c, axis=1)

        assert f.shape == (4, 7)
        assert p.shape == (4, 7)

        for i in range(a.shape[0]):
            for j in range(a.shape[2]):
                fij, pij = stats.f_oneway(a[i, :, j], b[i, :, j], c[i, :, j])
                assert_allclose(fij, f[i, j])
                assert_allclose(pij, p[i, j])

    def test_length0_1d_error(self):
        # Require at least one value in each group.
        with assert_warns(stats.F_onewayBadInputSizesWarning):
            result = stats.f_oneway([1, 2, 3], [], [4, 5, 6, 7])
            assert_equal(result, (np.nan, np.nan))

    def test_length0_2d_error(self):
        with assert_warns(stats.F_onewayBadInputSizesWarning):
            ncols = 3
            a = np.ones((4, ncols))
            b = np.ones((0, ncols))
            c = np.ones((5, ncols))
            f, p = stats.f_oneway(a, b, c)
            nans = np.full((ncols,), fill_value=np.nan)
            assert_equal(f, nans)
            assert_equal(p, nans)

    def test_all_length_one(self):
        with assert_warns(stats.F_onewayBadInputSizesWarning):
            result = stats.f_oneway([10], [11], [12], [13])
            assert_equal(result, (np.nan, np.nan))

    @pytest.mark.parametrize('args', [(), ([1, 2, 3],)])
    def test_too_few_inputs(self, args):
        with assert_raises(TypeError):
            stats.f_oneway(*args)

    def test_axis_error(self):
        a = np.ones((3, 4))
        b = np.ones((5, 4))
        with assert_raises(np.AxisError):
            stats.f_oneway(a, b, axis=2)

    def test_bad_shapes(self):
        a = np.ones((3, 4))
        b = np.ones((5, 4))
        with assert_raises(ValueError):
            stats.f_oneway(a, b, axis=1)


class TestKruskal:
    def test_simple(self):
        x = [1]
        y = [2]
        h, p = stats.kruskal(x, y)
        assert_equal(h, 1.0)
        assert_approx_equal(p, stats.distributions.chi2.sf(h, 1))
        h, p = stats.kruskal(np.array(x), np.array(y))
        assert_equal(h, 1.0)
        assert_approx_equal(p, stats.distributions.chi2.sf(h, 1))

    def test_basic(self):
        x = [1, 3, 5, 7, 9]
        y = [2, 4, 6, 8, 10]
        h, p = stats.kruskal(x, y)
        assert_approx_equal(h, 3./11, significant=10)
        assert_approx_equal(p, stats.distributions.chi2.sf(3./11, 1))
        h, p = stats.kruskal(np.array(x), np.array(y))
        assert_approx_equal(h, 3./11, significant=10)
        assert_approx_equal(p, stats.distributions.chi2.sf(3./11, 1))

    def test_simple_tie(self):
        x = [1]
        y = [1, 2]
        h_uncorr = 1.5**2 + 2*2.25**2 - 12
        corr = 0.75
        expected = h_uncorr / corr   # 0.5
        h, p = stats.kruskal(x, y)
        # Since the expression is simple and the exact answer is 0.5, it
        # should be safe to use assert_equal().
        assert_equal(h, expected)

    def test_another_tie(self):
        x = [1, 1, 1, 2]
        y = [2, 2, 2, 2]
        h_uncorr = (12. / 8. / 9.) * 4 * (3**2 + 6**2) - 3 * 9
        corr = 1 - float(3**3 - 3 + 5**3 - 5) / (8**3 - 8)
        expected = h_uncorr / corr
        h, p = stats.kruskal(x, y)
        assert_approx_equal(h, expected)

    def test_three_groups(self):
        # A test of stats.kruskal with three groups, with ties.
        x = [1, 1, 1]
        y = [2, 2, 2]
        z = [2, 2]
        h_uncorr = (12. / 8. / 9.) * (3*2**2 + 3*6**2 + 2*6**2) - 3 * 9  # 5.0
        corr = 1 - float(3**3 - 3 + 5**3 - 5) / (8**3 - 8)
        expected = h_uncorr / corr  # 7.0
        h, p = stats.kruskal(x, y, z)
        assert_approx_equal(h, expected)
        assert_approx_equal(p, stats.distributions.chi2.sf(h, 2))

    def test_empty(self):
        # A test of stats.kruskal with three groups, with ties.
        x = [1, 1, 1]
        y = [2, 2, 2]
        z = []
        assert_equal(stats.kruskal(x, y, z), (np.nan, np.nan))

    def test_nd_arrays(self):
        # Inputs must be exactly one-dimensional
        x = [1]
        y = [2]
        z = np.random.rand(2, 2)
        with assert_raises(ValueError,
                           match="Samples must be one-dimensional."):
            stats.kruskal(x, y, z)

    def test_kruskal_result_attributes(self):
        x = [1, 3, 5, 7, 9]
        y = [2, 4, 6, 8, 10]
        res = stats.kruskal(x, y)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes)

    def test_nan_policy(self):
        x = np.arange(10.)
        x[9] = np.nan
        assert_equal(stats.kruskal(x, x), (np.nan, np.nan))
        assert_almost_equal(stats.kruskal(x, x, nan_policy='omit'), (0.0, 1.0))
        assert_raises(ValueError, stats.kruskal, x, x, nan_policy='raise')
        assert_raises(ValueError, stats.kruskal, x, x, nan_policy='foobar')

    def test_large_no_samples(self):
        # Test to see if large samples are handled correctly.
        n = 50000
        x = np.random.randn(n)
        y = np.random.randn(n) + 50
        h, p = stats.kruskal(x, y)
        expected = 0
        assert_approx_equal(p, expected)


class TestCombinePvalues:

    def test_fisher(self):
        # Example taken from https://en.wikipedia.org/wiki/Fisher%27s_exact_test#Example
        xsq, p = stats.combine_pvalues([.01, .2, .3], method='fisher')
        assert_approx_equal(p, 0.02156, significant=4)

    def test_stouffer(self):
        Z, p = stats.combine_pvalues([.01, .2, .3], method='stouffer')
        assert_approx_equal(p, 0.01651, significant=4)

    def test_stouffer2(self):
        Z, p = stats.combine_pvalues([.5, .5, .5], method='stouffer')
        assert_approx_equal(p, 0.5, significant=4)

    def test_weighted_stouffer(self):
        Z, p = stats.combine_pvalues([.01, .2, .3], method='stouffer',
                                     weights=np.ones(3))
        assert_approx_equal(p, 0.01651, significant=4)

    def test_weighted_stouffer2(self):
        Z, p = stats.combine_pvalues([.01, .2, .3], method='stouffer',
                                     weights=np.array((1, 4, 9)))
        assert_approx_equal(p, 0.1464, significant=4)

    def test_pearson(self):
        Z, p = stats.combine_pvalues([.01, .2, .3], method='pearson')
        assert_approx_equal(p, 0.97787, significant=4)

    def test_tippett(self):
        Z, p = stats.combine_pvalues([.01, .2, .3], method='tippett')
        assert_approx_equal(p, 0.970299, significant=4)

    def test_mudholkar_george(self):
        Z, p = stats.combine_pvalues([.1, .1, .1], method='mudholkar_george')
        assert_approx_equal(p, 0.019462, significant=4)

    def test_mudholkar_george_equal_fisher_minus_pearson(self):
        Z, p = stats.combine_pvalues([.01, .2, .3], method='mudholkar_george')
        Z_f, p_f = stats.combine_pvalues([.01, .2, .3], method='fisher')
        Z_p, p_p = stats.combine_pvalues([.01, .2, .3], method='pearson')
        # 0.5 here is because logistic = log(u) - log(1-u), i.e. no 2 factors
        assert_approx_equal(0.5 * (Z_f-Z_p), Z, significant=4)


class TestCdfDistanceValidation:
    """
    Test that _cdf_distance() (via wasserstein_distance()) raises ValueErrors
    for bad inputs.
    """

    def test_distinct_value_and_weight_lengths(self):
        # When the number of weights does not match the number of values,
        # a ValueError should be raised.
        assert_raises(ValueError, stats.wasserstein_distance,
                      [1], [2], [4], [3, 1])
        assert_raises(ValueError, stats.wasserstein_distance, [1], [2], [1, 0])

    def test_zero_weight(self):
        # When a distribution is given zero weight, a ValueError should be
        # raised.
        assert_raises(ValueError, stats.wasserstein_distance,
                      [0, 1], [2], [0, 0])
        assert_raises(ValueError, stats.wasserstein_distance,
                      [0, 1], [2], [3, 1], [0])

    def test_negative_weights(self):
        # A ValueError should be raised if there are any negative weights.
        assert_raises(ValueError, stats.wasserstein_distance,
                      [0, 1], [2, 2], [1, 1], [3, -1])

    def test_empty_distribution(self):
        # A ValueError should be raised when trying to measure the distance
        # between something and nothing.
        assert_raises(ValueError, stats.wasserstein_distance, [], [2, 2])
        assert_raises(ValueError, stats.wasserstein_distance, [1], [])

    def test_inf_weight(self):
        # An inf weight is not valid.
        assert_raises(ValueError, stats.wasserstein_distance,
                      [1, 2, 1], [1, 1], [1, np.inf, 1], [1, 1])


class TestWassersteinDistance:
    """ Tests for wasserstein_distance() output values.
    """

    def test_simple(self):
        # For basic distributions, the value of the Wasserstein distance is
        # straightforward.
        assert_almost_equal(
            stats.wasserstein_distance([0, 1], [0], [1, 1], [1]),
            .5)
        assert_almost_equal(stats.wasserstein_distance(
            [0, 1], [0], [3, 1], [1]),
            .25)
        assert_almost_equal(stats.wasserstein_distance(
            [0, 2], [0], [1, 1], [1]),
            1)
        assert_almost_equal(stats.wasserstein_distance(
            [0, 1, 2], [1, 2, 3]),
            1)

    def test_same_distribution(self):
        # Any distribution moved to itself should have a Wasserstein distance of
        # zero.
        assert_equal(stats.wasserstein_distance([1, 2, 3], [2, 1, 3]), 0)
        assert_equal(
            stats.wasserstein_distance([1, 1, 1, 4], [4, 1],
                                       [1, 1, 1, 1], [1, 3]),
            0)

    def test_shift(self):
        # If the whole distribution is shifted by x, then the Wasserstein
        # distance should be x.
        assert_almost_equal(stats.wasserstein_distance([0], [1]), 1)
        assert_almost_equal(stats.wasserstein_distance([-5], [5]), 10)
        assert_almost_equal(
            stats.wasserstein_distance([1, 2, 3, 4, 5], [11, 12, 13, 14, 15]),
            10)
        assert_almost_equal(
            stats.wasserstein_distance([4.5, 6.7, 2.1], [4.6, 7, 9.2],
                                       [3, 1, 1], [1, 3, 1]),
            2.5)

    def test_combine_weights(self):
        # Assigning a weight w to a value is equivalent to including that value
        # w times in the value array with weight of 1.
        assert_almost_equal(
            stats.wasserstein_distance(
                [0, 0, 1, 1, 1, 1, 5], [0, 3, 3, 3, 3, 4, 4],
                [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]),
            stats.wasserstein_distance([5, 0, 1], [0, 4, 3],
                                       [1, 2, 4], [1, 2, 4]))

    def test_collapse(self):
        # Collapsing a distribution to a point distribution at zero is
        # equivalent to taking the average of the absolute values of the values.
        u = np.arange(-10, 30, 0.3)
        v = np.zeros_like(u)
        assert_almost_equal(
            stats.wasserstein_distance(u, v),
            np.mean(np.abs(u)))

        u_weights = np.arange(len(u))
        v_weights = u_weights[::-1]
        assert_almost_equal(
            stats.wasserstein_distance(u, v, u_weights, v_weights),
            np.average(np.abs(u), weights=u_weights))

    def test_zero_weight(self):
        # Values with zero weight have no impact on the Wasserstein distance.
        assert_almost_equal(
            stats.wasserstein_distance([1, 2, 100000], [1, 1],
                                       [1, 1, 0], [1, 1]),
            stats.wasserstein_distance([1, 2], [1, 1], [1, 1], [1, 1]))

    def test_inf_values(self):
        # Inf values can lead to an inf distance or trigger a RuntimeWarning
        # (and return NaN) if the distance is undefined.
        assert_equal(
            stats.wasserstein_distance([1, 2, np.inf], [1, 1]),
            np.inf)
        assert_equal(
            stats.wasserstein_distance([1, 2, np.inf], [-np.inf, 1]),
            np.inf)
        assert_equal(
            stats.wasserstein_distance([1, -np.inf, np.inf], [1, 1]),
            np.inf)
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning, "invalid value*")
            assert_equal(
                stats.wasserstein_distance([1, 2, np.inf], [np.inf, 1]),
                np.nan)


class TestEnergyDistance:
    """ Tests for energy_distance() output values.
    """

    def test_simple(self):
        # For basic distributions, the value of the energy distance is
        # straightforward.
        assert_almost_equal(
            stats.energy_distance([0, 1], [0], [1, 1], [1]),
            np.sqrt(2) * .5)
        assert_almost_equal(stats.energy_distance(
            [0, 1], [0], [3, 1], [1]),
            np.sqrt(2) * .25)
        assert_almost_equal(stats.energy_distance(
            [0, 2], [0], [1, 1], [1]),
            2 * .5)
        assert_almost_equal(
            stats.energy_distance([0, 1, 2], [1, 2, 3]),
            np.sqrt(2) * (3*(1./3**2))**.5)

    def test_same_distribution(self):
        # Any distribution moved to itself should have a energy distance of
        # zero.
        assert_equal(stats.energy_distance([1, 2, 3], [2, 1, 3]), 0)
        assert_equal(
            stats.energy_distance([1, 1, 1, 4], [4, 1], [1, 1, 1, 1], [1, 3]),
            0)

    def test_shift(self):
        # If a single-point distribution is shifted by x, then the energy
        # distance should be sqrt(2) * sqrt(x).
        assert_almost_equal(stats.energy_distance([0], [1]), np.sqrt(2))
        assert_almost_equal(
            stats.energy_distance([-5], [5]),
            np.sqrt(2) * 10**.5)

    def test_combine_weights(self):
        # Assigning a weight w to a value is equivalent to including that value
        # w times in the value array with weight of 1.
        assert_almost_equal(
            stats.energy_distance([0, 0, 1, 1, 1, 1, 5], [0, 3, 3, 3, 3, 4, 4],
                                  [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]),
            stats.energy_distance([5, 0, 1], [0, 4, 3], [1, 2, 4], [1, 2, 4]))

    def test_zero_weight(self):
        # Values with zero weight have no impact on the energy distance.
        assert_almost_equal(
            stats.energy_distance([1, 2, 100000], [1, 1], [1, 1, 0], [1, 1]),
            stats.energy_distance([1, 2], [1, 1], [1, 1], [1, 1]))

    def test_inf_values(self):
        # Inf values can lead to an inf distance or trigger a RuntimeWarning
        # (and return NaN) if the distance is undefined.
        assert_equal(stats.energy_distance([1, 2, np.inf], [1, 1]), np.inf)
        assert_equal(
            stats.energy_distance([1, 2, np.inf], [-np.inf, 1]),
            np.inf)
        assert_equal(
            stats.energy_distance([1, -np.inf, np.inf], [1, 1]),
            np.inf)
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning, "invalid value*")
            assert_equal(
                stats.energy_distance([1, 2, np.inf], [np.inf, 1]),
                np.nan)


class TestBrunnerMunzel:
    # Data from (Lumley, 1996)
    X = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1]
    Y = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]
    significant = 13

    def test_brunnermunzel_one_sided(self):
        # Results are compared with R's lawstat package.
        u1, p1 = stats.brunnermunzel(self.X, self.Y, alternative='less')
        u2, p2 = stats.brunnermunzel(self.Y, self.X, alternative='greater')
        u3, p3 = stats.brunnermunzel(self.X, self.Y, alternative='greater')
        u4, p4 = stats.brunnermunzel(self.Y, self.X, alternative='less')

        assert_approx_equal(p1, p2, significant=self.significant)
        assert_approx_equal(p3, p4, significant=self.significant)
        assert_(p1 != p3)
        assert_approx_equal(u1, 3.1374674823029505,
                            significant=self.significant)
        assert_approx_equal(u2, -3.1374674823029505,
                            significant=self.significant)
        assert_approx_equal(u3, 3.1374674823029505,
                            significant=self.significant)
        assert_approx_equal(u4, -3.1374674823029505,
                            significant=self.significant)
        assert_approx_equal(p1, 0.0028931043330757342,
                            significant=self.significant)
        assert_approx_equal(p3, 0.99710689566692423,
                            significant=self.significant)

    def test_brunnermunzel_two_sided(self):
        # Results are compared with R's lawstat package.
        u1, p1 = stats.brunnermunzel(self.X, self.Y, alternative='two-sided')
        u2, p2 = stats.brunnermunzel(self.Y, self.X, alternative='two-sided')

        assert_approx_equal(p1, p2, significant=self.significant)
        assert_approx_equal(u1, 3.1374674823029505,
                            significant=self.significant)
        assert_approx_equal(u2, -3.1374674823029505,
                            significant=self.significant)
        assert_approx_equal(p1, 0.0057862086661515377,
                            significant=self.significant)

    def test_brunnermunzel_default(self):
        # The default value for alternative is two-sided
        u1, p1 = stats.brunnermunzel(self.X, self.Y)
        u2, p2 = stats.brunnermunzel(self.Y, self.X)

        assert_approx_equal(p1, p2, significant=self.significant)
        assert_approx_equal(u1, 3.1374674823029505,
                            significant=self.significant)
        assert_approx_equal(u2, -3.1374674823029505,
                            significant=self.significant)
        assert_approx_equal(p1, 0.0057862086661515377,
                            significant=self.significant)

    def test_brunnermunzel_alternative_error(self):
        alternative = "error"
        distribution = "t"
        nan_policy = "propagate"
        assert_(alternative not in ["two-sided", "greater", "less"])
        assert_raises(ValueError,
                      stats.brunnermunzel,
                      self.X,
                      self.Y,
                      alternative,
                      distribution,
                      nan_policy)

    def test_brunnermunzel_distribution_norm(self):
        u1, p1 = stats.brunnermunzel(self.X, self.Y, distribution="normal")
        u2, p2 = stats.brunnermunzel(self.Y, self.X, distribution="normal")
        assert_approx_equal(p1, p2, significant=self.significant)
        assert_approx_equal(u1, 3.1374674823029505,
                            significant=self.significant)
        assert_approx_equal(u2, -3.1374674823029505,
                            significant=self.significant)
        assert_approx_equal(p1, 0.0017041417600383024,
                            significant=self.significant)

    def test_brunnermunzel_distribution_error(self):
        alternative = "two-sided"
        distribution = "error"
        nan_policy = "propagate"
        assert_(alternative not in ["t", "normal"])
        assert_raises(ValueError,
                      stats.brunnermunzel,
                      self.X,
                      self.Y,
                      alternative,
                      distribution,
                      nan_policy)

    def test_brunnermunzel_empty_imput(self):
        u1, p1 = stats.brunnermunzel(self.X, [])
        u2, p2 = stats.brunnermunzel([], self.Y)
        u3, p3 = stats.brunnermunzel([], [])

        assert_equal(u1, np.nan)
        assert_equal(p1, np.nan)
        assert_equal(u2, np.nan)
        assert_equal(p2, np.nan)
        assert_equal(u3, np.nan)
        assert_equal(p3, np.nan)

    def test_brunnermunzel_nan_input_propagate(self):
        X = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, np.nan]
        Y = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]
        u1, p1 = stats.brunnermunzel(X, Y, nan_policy="propagate")
        u2, p2 = stats.brunnermunzel(Y, X, nan_policy="propagate")

        assert_equal(u1, np.nan)
        assert_equal(p1, np.nan)
        assert_equal(u2, np.nan)
        assert_equal(p2, np.nan)

    def test_brunnermunzel_nan_input_raise(self):
        X = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, np.nan]
        Y = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]
        alternative = "two-sided"
        distribution = "t"
        nan_policy = "raise"

        assert_raises(ValueError,
                      stats.brunnermunzel,
                      X,
                      Y,
                      alternative,
                      distribution,
                      nan_policy)
        assert_raises(ValueError,
                      stats.brunnermunzel,
                      Y,
                      X,
                      alternative,
                      distribution,
                      nan_policy)

    def test_brunnermunzel_nan_input_omit(self):
        X = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, np.nan]
        Y = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]
        u1, p1 = stats.brunnermunzel(X, Y, nan_policy="omit")
        u2, p2 = stats.brunnermunzel(Y, X, nan_policy="omit")

        assert_approx_equal(p1, p2, significant=self.significant)
        assert_approx_equal(u1, 3.1374674823029505,
                            significant=self.significant)
        assert_approx_equal(u2, -3.1374674823029505,
                            significant=self.significant)
        assert_approx_equal(p1, 0.0057862086661515377,
                            significant=self.significant)


class TestRatioUniforms:
    """ Tests for rvs_ratio_uniforms.
    """

    def test_rv_generation(self):
        # use KS test to check distribution of rvs
        # normal distribution
        f = stats.norm.pdf
        v_bound = np.sqrt(f(np.sqrt(2))) * np.sqrt(2)
        umax, vmin, vmax = np.sqrt(f(0)), -v_bound, v_bound
        rvs = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=2500,
                                       random_state=12345)
        assert_equal(stats.kstest(rvs, 'norm')[1] > 0.25, True)

        # exponential distribution
        rvs = stats.rvs_ratio_uniforms(lambda x: np.exp(-x), umax=1,
                                       vmin=0, vmax=2*np.exp(-1),
                                       size=1000, random_state=12345)
        assert_equal(stats.kstest(rvs, 'expon')[1] > 0.25, True)

    def test_shape(self):
        # test shape of return value depending on size parameter
        f = stats.norm.pdf
        v_bound = np.sqrt(f(np.sqrt(2))) * np.sqrt(2)
        umax, vmin, vmax = np.sqrt(f(0)), -v_bound, v_bound

        r1 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=3,
                                      random_state=1234)
        r2 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=(3,),
                                      random_state=1234)
        r3 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=(3, 1),
                                      random_state=1234)
        assert_equal(r1, r2)
        assert_equal(r2, r3.flatten())
        assert_equal(r1.shape, (3,))
        assert_equal(r3.shape, (3, 1))

        r4 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=(3, 3, 3),
                                      random_state=12)
        r5 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=27,
                                      random_state=12)
        assert_equal(r4.flatten(), r5)
        assert_equal(r4.shape, (3, 3, 3))

        r6 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, random_state=1234)
        r7 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=1,
                                      random_state=1234)
        r8 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=(1, ),
                                      random_state=1234)
        assert_equal(r6, r7)
        assert_equal(r7, r8)

    def test_random_state(self):
        f = stats.norm.pdf
        v_bound = np.sqrt(f(np.sqrt(2))) * np.sqrt(2)
        umax, vmin, vmax = np.sqrt(f(0)), -v_bound, v_bound
        np.random.seed(1234)
        r1 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=(3, 4))
        r2 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=(3, 4),
                                      random_state=1234)
        assert_equal(r1, r2)

    def test_exceptions(self):
        f = stats.norm.pdf
        # need vmin < vmax
        assert_raises(ValueError,
                      stats.rvs_ratio_uniforms, pdf=f, umax=1, vmin=3, vmax=1)
        assert_raises(ValueError,
                      stats.rvs_ratio_uniforms, pdf=f, umax=1, vmin=1, vmax=1)
        # need umax > 0
        assert_raises(ValueError,
                      stats.rvs_ratio_uniforms, pdf=f, umax=-1, vmin=1, vmax=1)
        assert_raises(ValueError,
                      stats.rvs_ratio_uniforms, pdf=f, umax=0, vmin=1, vmax=1)


class TestMGCErrorWarnings:
    """ Tests errors and warnings derived from MGC.
    """
    def test_error_notndarray(self):
        # raises error if x or y is not a ndarray
        x = np.arange(20)
        y = [5] * 20
        assert_raises(ValueError, stats.multiscale_graphcorr, x, y)
        assert_raises(ValueError, stats.multiscale_graphcorr, y, x)

    def test_error_shape(self):
        # raises error if number of samples different (n)
        x = np.arange(100).reshape(25, 4)
        y = x.reshape(10, 10)
        assert_raises(ValueError, stats.multiscale_graphcorr, x, y)

    def test_error_lowsamples(self):
        # raises error if samples are low (< 3)
        x = np.arange(3)
        y = np.arange(3)
        assert_raises(ValueError, stats.multiscale_graphcorr, x, y)

    def test_error_nans(self):
        # raises error if inputs contain NaNs
        x = np.arange(20, dtype=float)
        x[0] = np.nan
        assert_raises(ValueError, stats.multiscale_graphcorr, x, x)

        y = np.arange(20)
        assert_raises(ValueError, stats.multiscale_graphcorr, x, y)

    def test_error_wrongdisttype(self):
        # raises error if metric is not a function
        x = np.arange(20)
        compute_distance = 0
        assert_raises(ValueError, stats.multiscale_graphcorr, x, x,
                      compute_distance=compute_distance)

    @pytest.mark.parametrize("reps", [
        -1,    # reps is negative
        '1',   # reps is not integer
    ])
    def test_error_reps(self, reps):
        # raises error if reps is negative
        x = np.arange(20)
        assert_raises(ValueError, stats.multiscale_graphcorr, x, x, reps=reps)

    def test_warns_reps(self):
        # raises warning when reps is less than 1000
        x = np.arange(20)
        reps = 100
        assert_warns(RuntimeWarning, stats.multiscale_graphcorr, x, x, reps=reps)

    def test_error_infty(self):
        # raises error if input contains infinities
        x = np.arange(20)
        y = np.ones(20) * np.inf
        assert_raises(ValueError, stats.multiscale_graphcorr, x, y)


class TestMGCStat:
    """ Test validity of MGC test statistic
    """
    def _simulations(self, samps=100, dims=1, sim_type=""):
        # linear simulation
        if sim_type == "linear":
            x = np.random.uniform(-1, 1, size=(samps, 1))
            y = x + 0.3 * np.random.random_sample(size=(x.size, 1))

        # spiral simulation
        elif sim_type == "nonlinear":
            unif = np.array(np.random.uniform(0, 5, size=(samps, 1)))
            x = unif * np.cos(np.pi * unif)
            y = unif * np.sin(np.pi * unif) + (0.4
                * np.random.random_sample(size=(x.size, 1)))

        # independence (tests type I simulation)
        elif sim_type == "independence":
            u = np.random.normal(0, 1, size=(samps, 1))
            v = np.random.normal(0, 1, size=(samps, 1))
            u_2 = np.random.binomial(1, p=0.5, size=(samps, 1))
            v_2 = np.random.binomial(1, p=0.5, size=(samps, 1))
            x = u/3 + 2*u_2 - 1
            y = v/3 + 2*v_2 - 1

        # raises error if not approved sim_type
        else:
            raise ValueError("sim_type must be linear, nonlinear, or "
                             "independence")

        # add dimensions of noise for higher dimensions
        if dims > 1:
            dims_noise = np.random.normal(0, 1, size=(samps, dims-1))
            x = np.concatenate((x, dims_noise), axis=1)

        return x, y

    @pytest.mark.slow
    @pytest.mark.parametrize("sim_type, obs_stat, obs_pvalue", [
        ("linear", 0.97, 1/1000),           # test linear simulation
        ("nonlinear", 0.163, 1/1000),       # test spiral simulation
        ("independence", -0.0094, 0.78)     # test independence simulation
    ])
    def test_oned(self, sim_type, obs_stat, obs_pvalue):
        np.random.seed(12345678)

        # generate x and y
        x, y = self._simulations(samps=100, dims=1, sim_type=sim_type)

        # test stat and pvalue
        stat, pvalue, _ = stats.multiscale_graphcorr(x, y)
        assert_approx_equal(stat, obs_stat, significant=1)
        assert_approx_equal(pvalue, obs_pvalue, significant=1)

    @pytest.mark.slow
    @pytest.mark.parametrize("sim_type, obs_stat, obs_pvalue", [
        ("linear", 0.184, 1/1000),           # test linear simulation
        ("nonlinear", 0.0190, 0.117),        # test spiral simulation
    ])
    def test_fived(self, sim_type, obs_stat, obs_pvalue):
        np.random.seed(12345678)

        # generate x and y
        x, y = self._simulations(samps=100, dims=5, sim_type=sim_type)

        # test stat and pvalue
        stat, pvalue, _ = stats.multiscale_graphcorr(x, y)
        assert_approx_equal(stat, obs_stat, significant=1)
        assert_approx_equal(pvalue, obs_pvalue, significant=1)

    @pytest.mark.slow
    def test_twosamp(self):
        np.random.seed(12345678)

        # generate x and y
        x = np.random.binomial(100, 0.5, size=(100, 5))
        y = np.random.normal(0, 1, size=(80, 5))

        # test stat and pvalue
        stat, pvalue, _ = stats.multiscale_graphcorr(x, y)
        assert_approx_equal(stat, 1.0, significant=1)
        assert_approx_equal(pvalue, 0.001, significant=1)

        # generate x and y
        y = np.random.normal(0, 1, size=(100, 5))

        # test stat and pvalue
        stat, pvalue, _ = stats.multiscale_graphcorr(x, y, is_twosamp=True)
        assert_approx_equal(stat, 1.0, significant=1)
        assert_approx_equal(pvalue, 0.001, significant=1)

    @pytest.mark.slow
    def test_workers(self):
        np.random.seed(12345678)

        # generate x and y
        x, y = self._simulations(samps=100, dims=1, sim_type="linear")

        # test stat and pvalue
        stat, pvalue, _ = stats.multiscale_graphcorr(x, y, workers=2)
        assert_approx_equal(stat, 0.97, significant=1)
        assert_approx_equal(pvalue, 0.001, significant=1)

    @pytest.mark.slow
    def test_random_state(self):
        # generate x and y
        x, y = self._simulations(samps=100, dims=1, sim_type="linear")

        # test stat and pvalue
        stat, pvalue, _ = stats.multiscale_graphcorr(x, y, random_state=1)
        assert_approx_equal(stat, 0.97, significant=1)
        assert_approx_equal(pvalue, 0.001, significant=1)

    @pytest.mark.slow
    def test_dist_perm(self):
        np.random.seed(12345678)
        # generate x and y
        x, y = self._simulations(samps=100, dims=1, sim_type="nonlinear")
        distx = cdist(x, x, metric="euclidean")
        disty = cdist(y, y, metric="euclidean")

        stat_dist, pvalue_dist, _ = stats.multiscale_graphcorr(distx, disty,
                                                               compute_distance=None,
                                                               random_state=1)
        assert_approx_equal(stat_dist, 0.163, significant=1)
        assert_approx_equal(pvalue_dist, 0.001, significant=1)


class TestNumericalInverseHermite:
    @pytest.mark.parametrize(("distname", "shapes"), distcont)
    def test_basic(self, distname, shapes):
        slow_dists = {'ksone', 'kstwo', 'levy_stable', 'skewnorm'}
        fail_dists = {'beta', 'gausshyper', 'geninvgauss', 'ncf', 'nct',
                      'norminvgauss', 'genhyperbolic', 'studentized_range'}

        if distname in slow_dists:
            pytest.skip("Distribution is too slow")
        if distname in fail_dists:
            # specific reasons documented in gh-13319
            # https://github.com/scipy/scipy/pull/13319#discussion_r626188955
            pytest.xfail("Fails - usually due to inaccurate CDF/PDF")

        np.random.seed(0)

        dist = getattr(stats, distname)(*shapes)

        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "overflow encountered")
            sup.filter(RuntimeWarning, "divide by zero")
            sup.filter(RuntimeWarning, "invalid value encountered")
            fni = stats.NumericalInverseHermite(dist)

        x = np.random.rand(10)
        p_tol = np.max(np.abs(dist.ppf(x)-fni.ppf(x))/np.abs(dist.ppf(x)))
        u_tol = np.max(np.abs(dist.cdf(fni.ppf(x)) - x))

        assert p_tol < 1e-8
        assert u_tol < 1e-12

    def test_input_validation(self):
        match = "`dist` must have methods `pdf`, `cdf`, and `ppf`"
        with pytest.raises(ValueError, match=match):
            stats.NumericalInverseHermite("norm")

        match = "could not convert string to float"
        with pytest.raises(ValueError, match=match):
            stats.NumericalInverseHermite(stats.norm(), tol='ekki')

        match = "`max_intervals' must be..."
        with pytest.raises(ValueError, match=match):
            stats.NumericalInverseHermite(stats.norm(), max_intervals=-1)

        match = "`qmc_engine` must be an instance of..."
        with pytest.raises(ValueError, match=match):
            fni = stats.NumericalInverseHermite(stats.norm())
            fni.qrvs(qmc_engine=0)

        if NumpyVersion(np.__version__) >= '1.18.0':
            # issues with QMCEngines and old NumPy
            fni = stats.NumericalInverseHermite(stats.norm())

            match = "`d` must be consistent with dimension of `qmc_engine`."
            with pytest.raises(ValueError, match=match):
                fni.qrvs(d=3, qmc_engine=stats.qmc.Halton(2))

    rngs = [None, 0, np.random.RandomState(0)]
    if NumpyVersion(np.__version__) >= '1.18.0':
        rngs.append(np.random.default_rng(0))  # type: ignore
    sizes = [(None, tuple()), (8, (8,)), ((4, 5, 6), (4, 5, 6))]

    @pytest.mark.parametrize('rng', rngs)
    @pytest.mark.parametrize('size_in, size_out', sizes)
    def test_RVS(self, rng, size_in, size_out):
        dist = stats.norm()
        fni = stats.NumericalInverseHermite(dist)

        rng2 = deepcopy(rng)
        rvs = fni.rvs(size=size_in, random_state=rng)
        assert(rvs.shape == size_out)

        if rng2 is not None:
            rng2 = check_random_state(rng2)
            uniform = rng2.uniform(size=size_in)
            rvs2 = stats.norm.ppf(uniform)
            assert_allclose(rvs, rvs2)

    if NumpyVersion(np.__version__) >= '1.18.0':
        qrngs = [None, stats.qmc.Sobol(1, seed=0), stats.qmc.Halton(3, seed=0)]
    else:
        qrngs = []
    # `size=None` should not add anything to the shape, `size=1` should
    sizes = [(None, tuple()), (1, (1,)), (4, (4,)),
             ((4,), (4,)), ((2, 4), (2, 4))]  # type: ignore
    # Neither `d=None` nor `d=1` should add anything to the shape
    ds = [(None,  tuple()), (1, tuple()), (3, (3,))]

    @pytest.mark.parametrize('qrng', qrngs)
    @pytest.mark.parametrize('size_in, size_out', sizes)
    @pytest.mark.parametrize('d_in, d_out', ds)
    def test_QRVS(self, qrng, size_in, size_out, d_in, d_out):
        dist = stats.norm()
        fni = stats.NumericalInverseHermite(dist)

        # If d and qrng.d are inconsistent, an error is raised
        if d_in is not None and qrng is not None and qrng.d != d_in:
            match = "`d` must be consistent with dimension of `qmc_engine`."
            with pytest.raises(ValueError, match=match):
                fni.qrvs(size_in, d=d_in, qmc_engine=qrng)
            return

        # Sometimes d is really determined by qrng
        if d_in is None and qrng is not None and qrng.d != 1:
            d_out = (qrng.d,)

        shape_expected = size_out + d_out

        qrng2 = deepcopy(qrng)
        qrvs = fni.qrvs(size=size_in, d=d_in, qmc_engine=qrng)
        assert(qrvs.shape == shape_expected)

        if qrng2 is not None:
            uniform = qrng2.random(np.prod(size_in) or 1)
            qrvs2 = stats.norm.ppf(uniform).reshape(shape_expected)
            assert_allclose(qrvs, qrvs2, atol=1e-12)

    def test_QRVS_size_tuple(self):
        # QMCEngine samples are always of shape (n, d). When `size` is a tuple,
        # we set `n = prod(size)` in the call to qmc_engine.random, transform
        # the sample, and reshape it to the final dimensions. When we reshape,
        # we need to be careful, because the _columns_ of the sample returned
        # by a QMCEngine are "independent"-ish, but the elements within the
        # columns are not. We need to make sure that this doesn't get mixed up
        # by reshaping: qrvs[..., i] should remain "independent"-ish of
        # qrvs[..., i+1], but the elements within qrvs[..., i] should be
        # transformed from the same low-discrepancy sequence.
        if NumpyVersion(np.__version__) <= '1.18.0':
            pytest.skip("QMC doesn't play well with old NumPy")

        dist = stats.norm()
        fni = stats.NumericalInverseHermite(dist)

        size = (3, 4)
        d = 5
        qrng = stats.qmc.Halton(d, seed=0)
        qrng2 = stats.qmc.Halton(d, seed=0)

        uniform = qrng2.random(np.prod(size))

        qrvs = fni.qrvs(size=size, d=d, qmc_engine=qrng)
        qrvs2 = stats.norm.ppf(uniform)

        for i in range(d):
            sample = qrvs[..., i]
            sample2 = qrvs2[:, i].reshape(size)
            assert_allclose(sample, sample2, atol=1e-12)

    def test_inaccurate_CDF(self):
        # CDF function with inaccurate tail cannot be inverted; see gh-13319
        # https://github.com/scipy/scipy/pull/13319#discussion_r626188955
        shapes = (2.3098496451481823, 0.6268795430096368)
        match = "The interpolating spline could not be created."

        # fails with default tol
        with pytest.raises(ValueError, match=match):
            stats.NumericalInverseHermite(stats.beta(*shapes))

        # no error with coarser tol
        stats.NumericalInverseHermite(stats.beta(*shapes), tol=1e-10)

    def test_custom_distribution(self):
        class MyNormal:

            def pdf(self, x):
                return 1/np.sqrt(2*np.pi) * np.exp(-x**2 / 2)

            def cdf(self, x):
                return special.ndtr(x)

            def ppf(self, x):
                return special.ndtri(x)

        dist1 = MyNormal()
        fni1 = stats.NumericalInverseHermite(dist1)

        dist2 = stats.norm()
        fni2 = stats.NumericalInverseHermite(dist2)

        assert_allclose(fni1.rvs(random_state=0), fni2.rvs(random_state=0))


class TestPageTrendTest:
    # expected statistic and p-values generated using R at
    # https://rdrr.io/cran/cultevo/, e.g.
    # library(cultevo)
    # data = rbind(c(72, 47, 73, 35, 47, 96, 30, 59, 41, 36, 56, 49, 81, 43,
    #                   70, 47, 28, 28, 62, 20, 61, 20, 80, 24, 50),
    #              c(68, 52, 60, 34, 44, 20, 65, 88, 21, 81, 48, 31, 31, 67,
    #                69, 94, 30, 24, 40, 87, 70, 43, 50, 96, 43),
    #              c(81, 13, 85, 35, 79, 12, 92, 86, 21, 64, 16, 64, 68, 17,
    #                16, 89, 71, 43, 43, 36, 54, 13, 66, 51, 55))
    # result = page.test(data, verbose=FALSE)
    # Most test cases generated to achieve common critical p-values so that
    # results could be checked (to limited precision) against tables in
    # scipy.stats.page_trend_test reference [1]

    np.random.seed(0)
    data_3_25 = np.random.rand(3, 25)
    data_10_26 = np.random.rand(10, 26)

    ts = [
          (12805, 0.3886487053947608, False, 'asymptotic', data_3_25),
          (49140, 0.02888978556179862, False, 'asymptotic', data_10_26),
          (12332, 0.7722477197436702, False, 'asymptotic',
           [[72, 47, 73, 35, 47, 96, 30, 59, 41, 36, 56, 49, 81,
             43, 70, 47, 28, 28, 62, 20, 61, 20, 80, 24, 50],
            [68, 52, 60, 34, 44, 20, 65, 88, 21, 81, 48, 31, 31,
             67, 69, 94, 30, 24, 40, 87, 70, 43, 50, 96, 43],
            [81, 13, 85, 35, 79, 12, 92, 86, 21, 64, 16, 64, 68,
             17, 16, 89, 71, 43, 43, 36, 54, 13, 66, 51, 55]]),
          (266, 4.121656378600823e-05, False, 'exact',
           [[1.5, 4., 8.3, 5, 19, 11],
            [5, 4, 3.5, 10, 20, 21],
            [8.4, 3.2, 10, 12, 14, 15]]),
          (332, 0.9566400920502488, True, 'exact',
           [[4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1],
            [4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1],
            [3, 4, 1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
            [1, 2, 3, 4], [1, 2, 3, 4]]),
          (241, 0.9622210164861476, True, 'exact',
           [[3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1],
            [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1],
            [3, 2, 1], [2, 1, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3],
            [1, 2, 3], [1, 2, 3], [1, 2, 3]]),
          (197, 0.9619432897162209, True, 'exact',
           [[6, 5, 4, 3, 2, 1], [6, 5, 4, 3, 2, 1], [1, 3, 4, 5, 2, 6]]),
          (423, 0.9590458306880073, True, 'exact',
           [[5, 4, 3, 2, 1], [5, 4, 3, 2, 1], [5, 4, 3, 2, 1],
            [5, 4, 3, 2, 1], [5, 4, 3, 2, 1], [5, 4, 3, 2, 1],
            [4, 1, 3, 2, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5]]),
          (217, 0.9693058575034678, True, 'exact',
           [[3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1],
            [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1],
            [2, 1, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3],
            [1, 2, 3]]),
          (395, 0.991530289351305, True, 'exact',
           [[7, 6, 5, 4, 3, 2, 1], [7, 6, 5, 4, 3, 2, 1],
            [6, 5, 7, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6, 7]]),
          (117, 0.9997817843373017, True, 'exact',
           [[3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1],
            [3, 2, 1], [3, 2, 1], [3, 2, 1], [2, 1, 3], [1, 2, 3]]),
         ]

    @pytest.mark.parametrize("L, p, ranked, method, data", ts)
    def test_accuracy(self, L, p, ranked, method, data):
        np.random.seed(42)
        res = stats.page_trend_test(data, ranked=ranked, method=method)
        assert_equal(L, res.statistic)
        assert_allclose(p, res.pvalue)
        assert_equal(method, res.method)

    ts2 = [
           (542, 0.9481266260876332, True, 'exact',
            [[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
             [1, 8, 4, 7, 6, 5, 9, 3, 2, 10]]),
           (1322, 0.9993113928199309, True, 'exact',
            [[10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
             [10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [9, 2, 8, 7, 6, 5, 4, 3, 10, 1],
             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
           (2286, 0.9908688345484833, True, 'exact',
            [[8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1],
             [8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1],
             [8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1],
             [8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1],
             [8, 7, 6, 5, 4, 3, 2, 1], [1, 3, 5, 6, 4, 7, 2, 8],
             [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8],
             [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8],
             [1, 2, 3, 4, 5, 6, 7, 8]]),
          ]

    # only the first of these appears slow because intermediate data are
    # cached and used on the rest
    @pytest.mark.parametrize("L, p, ranked, method, data", ts)
    @pytest.mark.slow()
    def test_accuracy2(self, L, p, ranked, method, data):
        np.random.seed(42)
        res = stats.page_trend_test(data, ranked=ranked, method=method)
        assert_equal(L, res.statistic)
        assert_allclose(p, res.pvalue)
        assert_equal(method, res.method)

    def test_options(self):
        np.random.seed(42)
        m, n = 10, 20
        predicted_ranks = np.arange(1, n+1)
        perm = np.random.permutation(np.arange(n))
        data = np.random.rand(m, n)
        ranks = stats.rankdata(data, axis=1)
        res1 = stats.page_trend_test(ranks)
        res2 = stats.page_trend_test(ranks, ranked=True)
        res3 = stats.page_trend_test(data, ranked=False)
        res4 = stats.page_trend_test(ranks, predicted_ranks=predicted_ranks)
        res5 = stats.page_trend_test(ranks[:, perm],
                                     predicted_ranks=predicted_ranks[perm])
        assert_equal(res1.statistic, res2.statistic)
        assert_equal(res1.statistic, res3.statistic)
        assert_equal(res1.statistic, res4.statistic)
        assert_equal(res1.statistic, res5.statistic)

    def test_Ames_assay(self):
        # test from _page_trend_test.py [2] page 151; data on page 144
        np.random.seed(42)

        data = [[101, 117, 111], [91, 90, 107], [103, 133, 121],
                [136, 140, 144], [190, 161, 201], [146, 120, 116]]
        data = np.array(data).T
        predicted_ranks = np.arange(1, 7)

        res = stats.page_trend_test(data, ranked=False,
                                    predicted_ranks=predicted_ranks,
                                    method="asymptotic")
        assert_equal(res.statistic, 257)
        assert_almost_equal(res.pvalue, 0.0035, decimal=4)

        res = stats.page_trend_test(data, ranked=False,
                                    predicted_ranks=predicted_ranks,
                                    method="exact")
        assert_equal(res.statistic, 257)
        assert_almost_equal(res.pvalue, 0.0023, decimal=4)

    def test_input_validation(self):
        # test data not a 2d array
        with assert_raises(ValueError, match="`data` must be a 2d array."):
            stats.page_trend_test(None)
        with assert_raises(ValueError, match="`data` must be a 2d array."):
            stats.page_trend_test([])
        with assert_raises(ValueError, match="`data` must be a 2d array."):
            stats.page_trend_test([1, 2])
        with assert_raises(ValueError, match="`data` must be a 2d array."):
            stats.page_trend_test([[[1]]])

        # test invalid dimensions
        with assert_raises(ValueError, match="Page's L is only appropriate"):
            stats.page_trend_test(np.random.rand(1, 3))
        with assert_raises(ValueError, match="Page's L is only appropriate"):
            stats.page_trend_test(np.random.rand(2, 2))

        # predicted ranks must include each integer [1, 2, 3] exactly once
        message = "`predicted_ranks` must include each integer"
        with assert_raises(ValueError, match=message):
            stats.page_trend_test(data=[[1, 2, 3], [1, 2, 3]],
                                  predicted_ranks=[0, 1, 2])
        with assert_raises(ValueError, match=message):
            stats.page_trend_test(data=[[1, 2, 3], [1, 2, 3]],
                                  predicted_ranks=[1.1, 2, 3])
        with assert_raises(ValueError, match=message):
            stats.page_trend_test(data=[[1, 2, 3], [1, 2, 3]],
                                  predicted_ranks=[1, 2, 3, 3])
        with assert_raises(ValueError, match=message):
            stats.page_trend_test(data=[[1, 2, 3], [1, 2, 3]],
                                  predicted_ranks="invalid")

        # test improperly ranked data
        with assert_raises(ValueError, match="`data` is not properly ranked"):
            stats.page_trend_test([[0, 2, 3], [1, 2, 3]], True)
        with assert_raises(ValueError, match="`data` is not properly ranked"):
            stats.page_trend_test([[1, 2, 3], [1, 2, 4]], True)

        # various
        with assert_raises(ValueError, match="`data` contains NaNs"):
            stats.page_trend_test([[1, 2, 3], [1, 2, np.nan]],
                                  ranked=False)
        with assert_raises(ValueError, match="`method` must be in"):
            stats.page_trend_test(data=[[1, 2, 3], [1, 2, 3]],
                                  method="ekki")
        with assert_raises(TypeError, match="`ranked` must be boolean."):
            stats.page_trend_test(data=[[1, 2, 3], [1, 2, 3]],
                                  ranked="ekki")
