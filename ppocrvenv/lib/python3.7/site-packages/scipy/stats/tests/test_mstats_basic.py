"""
Tests for the stats.mstats module (support for masked arrays)
"""
import warnings
import platform

import numpy as np
from numpy import nan
import numpy.ma as ma
from numpy.ma import masked, nomask

import scipy.stats.mstats as mstats
from scipy import stats
from .common_tests import check_named_results
import pytest
from pytest import raises as assert_raises
from numpy.ma.testutils import (assert_equal, assert_almost_equal,
    assert_array_almost_equal, assert_array_almost_equal_nulp, assert_,
    assert_allclose, assert_array_equal)
from numpy.testing import suppress_warnings
from scipy.stats import mstats_basic

class TestMquantiles:
    def test_mquantiles_limit_keyword(self):
        # Regression test for Trac ticket #867
        data = np.array([[6., 7., 1.],
                         [47., 15., 2.],
                         [49., 36., 3.],
                         [15., 39., 4.],
                         [42., 40., -999.],
                         [41., 41., -999.],
                         [7., -999., -999.],
                         [39., -999., -999.],
                         [43., -999., -999.],
                         [40., -999., -999.],
                         [36., -999., -999.]])
        desired = [[19.2, 14.6, 1.45],
                   [40.0, 37.5, 2.5],
                   [42.8, 40.05, 3.55]]
        quants = mstats.mquantiles(data, axis=0, limit=(0, 50))
        assert_almost_equal(quants, desired)


def check_equal_gmean(array_like, desired, axis=None, dtype=None, rtol=1e-7):
    # Note this doesn't test when axis is not specified
    x = mstats.gmean(array_like, axis=axis, dtype=dtype)
    assert_allclose(x, desired, rtol=rtol)
    assert_equal(x.dtype, dtype)

def check_equal_hmean(array_like, desired, axis=None, dtype=None, rtol=1e-7):
    x = stats.hmean(array_like, axis=axis, dtype=dtype)
    assert_allclose(x, desired, rtol=rtol)
    assert_equal(x.dtype, dtype)


class TestGeoMean:
    def test_1d(self):
        a = [1, 2, 3, 4]
        desired = np.power(1*2*3*4, 1./4.)
        check_equal_gmean(a, desired, rtol=1e-14)

    def test_1d_ma(self):
        #  Test a 1d masked array
        a = ma.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        desired = 45.2872868812
        check_equal_gmean(a, desired)

        a = ma.array([1, 2, 3, 4], mask=[0, 0, 0, 1])
        desired = np.power(1*2*3, 1./3.)
        check_equal_gmean(a, desired, rtol=1e-14)

    def test_1d_ma_value(self):
        #  Test a 1d masked array with a masked value
        a = np.ma.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], mask=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        desired = 41.4716627439
        check_equal_gmean(a, desired)

    def test_1d_ma0(self):
        #  Test a 1d masked array with zero element
        a = np.ma.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 0])
        desired = 41.4716627439
        with np.errstate(divide='ignore'):
            check_equal_gmean(a, desired)

    def test_1d_ma_inf(self):
        #  Test a 1d masked array with negative element
        a = np.ma.array([10, 20, 30, 40, 50, 60, 70, 80, 90, -1])
        desired = 41.4716627439
        with np.errstate(invalid='ignore'):
            check_equal_gmean(a, desired)

    @pytest.mark.skipif(not hasattr(np, 'float96'), reason='cannot find float96 so skipping')
    def test_1d_float96(self):
        a = ma.array([1, 2, 3, 4], mask=[0, 0, 0, 1])
        desired_dt = np.power(1*2*3, 1./3.).astype(np.float96)
        check_equal_gmean(a, desired_dt, dtype=np.float96, rtol=1e-14)

    def test_2d_ma(self):
        a = ma.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                     mask=[[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0]])
        desired = np.array([1, 2, 3, 4])
        check_equal_gmean(a, desired, axis=0, rtol=1e-14)

        desired = ma.array([np.power(1*2*3*4, 1./4.),
                            np.power(2*3, 1./2.),
                            np.power(1*4, 1./2.)])
        check_equal_gmean(a, desired, axis=-1, rtol=1e-14)

        #  Test a 2d masked array
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = 52.8885199
        check_equal_gmean(np.ma.array(a), desired)


class TestHarMean:
    def test_1d(self):
        a = ma.array([1, 2, 3, 4], mask=[0, 0, 0, 1])
        desired = 3. / (1./1 + 1./2 + 1./3)
        check_equal_hmean(a, desired, rtol=1e-14)

        a = np.ma.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        desired = 34.1417152147
        check_equal_hmean(a, desired)

        a = np.ma.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        mask=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        desired = 31.8137186141
        check_equal_hmean(a, desired)

    @pytest.mark.skipif(not hasattr(np, 'float96'), reason='cannot find float96 so skipping')
    def test_1d_float96(self):
        a = ma.array([1, 2, 3, 4], mask=[0, 0, 0, 1])
        desired_dt = np.asarray(3. / (1./1 + 1./2 + 1./3), dtype=np.float96)
        check_equal_hmean(a, desired_dt, dtype=np.float96)

    def test_2d(self):
        a = ma.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                     mask=[[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0]])
        desired = ma.array([1, 2, 3, 4])
        check_equal_hmean(a, desired, axis=0, rtol=1e-14)

        desired = [4./(1/1.+1/2.+1/3.+1/4.), 2./(1/2.+1/3.), 2./(1/1.+1/4.)]
        check_equal_hmean(a, desired, axis=-1, rtol=1e-14)

        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = 38.6696271841
        check_equal_hmean(np.ma.array(a), desired)


class TestRanking:
    def test_ranking(self):
        x = ma.array([0,1,1,1,2,3,4,5,5,6,])
        assert_almost_equal(mstats.rankdata(x),
                           [1,3,3,3,5,6,7,8.5,8.5,10])
        x[[3,4]] = masked
        assert_almost_equal(mstats.rankdata(x),
                           [1,2.5,2.5,0,0,4,5,6.5,6.5,8])
        assert_almost_equal(mstats.rankdata(x, use_missing=True),
                            [1,2.5,2.5,4.5,4.5,4,5,6.5,6.5,8])
        x = ma.array([0,1,5,1,2,4,3,5,1,6,])
        assert_almost_equal(mstats.rankdata(x),
                           [1,3,8.5,3,5,7,6,8.5,3,10])
        x = ma.array([[0,1,1,1,2], [3,4,5,5,6,]])
        assert_almost_equal(mstats.rankdata(x),
                            [[1,3,3,3,5], [6,7,8.5,8.5,10]])
        assert_almost_equal(mstats.rankdata(x, axis=1),
                           [[1,3,3,3,5], [1,2,3.5,3.5,5]])
        assert_almost_equal(mstats.rankdata(x,axis=0),
                           [[1,1,1,1,1], [2,2,2,2,2,]])


class TestCorr:
    def test_pearsonr(self):
        # Tests some computations of Pearson's r
        x = ma.arange(10)
        with warnings.catch_warnings():
            # The tests in this context are edge cases, with perfect
            # correlation or anticorrelation, or totally masked data.
            # None of these should trigger a RuntimeWarning.
            warnings.simplefilter("error", RuntimeWarning)

            assert_almost_equal(mstats.pearsonr(x, x)[0], 1.0)
            assert_almost_equal(mstats.pearsonr(x, x[::-1])[0], -1.0)

            x = ma.array(x, mask=True)
            pr = mstats.pearsonr(x, x)
            assert_(pr[0] is masked)
            assert_(pr[1] is masked)

        x1 = ma.array([-1.0, 0.0, 1.0])
        y1 = ma.array([0, 0, 3])
        r, p = mstats.pearsonr(x1, y1)
        assert_almost_equal(r, np.sqrt(3)/2)
        assert_almost_equal(p, 1.0/3)

        # (x2, y2) have the same unmasked data as (x1, y1).
        mask = [False, False, False, True]
        x2 = ma.array([-1.0, 0.0, 1.0, 99.0], mask=mask)
        y2 = ma.array([0, 0, 3, -1], mask=mask)
        r, p = mstats.pearsonr(x2, y2)
        assert_almost_equal(r, np.sqrt(3)/2)
        assert_almost_equal(p, 1.0/3)

    def test_pearsonr_misaligned_mask(self):
        mx = np.ma.masked_array([1, 2, 3, 4, 5, 6], mask=[0, 1, 0, 0, 0, 0])
        my = np.ma.masked_array([9, 8, 7, 6, 5, 9], mask=[0, 0, 1, 0, 0, 0])
        x = np.array([1, 4, 5, 6])
        y = np.array([9, 6, 5, 9])
        mr, mp = mstats.pearsonr(mx, my)
        r, p = stats.pearsonr(x, y)
        assert_equal(mr, r)
        assert_equal(mp, p)

    def test_spearmanr(self):
        # Tests some computations of Spearman's rho
        (x, y) = ([5.05,6.75,3.21,2.66], [1.65,2.64,2.64,6.95])
        assert_almost_equal(mstats.spearmanr(x,y)[0], -0.6324555)
        (x, y) = ([5.05,6.75,3.21,2.66,np.nan],[1.65,2.64,2.64,6.95,np.nan])
        (x, y) = (ma.fix_invalid(x), ma.fix_invalid(y))
        assert_almost_equal(mstats.spearmanr(x,y)[0], -0.6324555)

        x = [2.0, 47.4, 42.0, 10.8, 60.1, 1.7, 64.0, 63.1,
              1.0, 1.4, 7.9, 0.3, 3.9, 0.3, 6.7]
        y = [22.6, 8.3, 44.4, 11.9, 24.6, 0.6, 5.7, 41.6,
              0.0, 0.6, 6.7, 3.8, 1.0, 1.2, 1.4]
        assert_almost_equal(mstats.spearmanr(x,y)[0], 0.6887299)
        x = [2.0, 47.4, 42.0, 10.8, 60.1, 1.7, 64.0, 63.1,
              1.0, 1.4, 7.9, 0.3, 3.9, 0.3, 6.7, np.nan]
        y = [22.6, 8.3, 44.4, 11.9, 24.6, 0.6, 5.7, 41.6,
              0.0, 0.6, 6.7, 3.8, 1.0, 1.2, 1.4, np.nan]
        (x, y) = (ma.fix_invalid(x), ma.fix_invalid(y))
        assert_almost_equal(mstats.spearmanr(x,y)[0], 0.6887299)
        # Next test is to make sure calculation uses sufficient precision.
        # The denominator's value is ~n^3 and used to be represented as an
        # int. 2000**3 > 2**32 so these arrays would cause overflow on
        # some machines.
        x = list(range(2000))
        y = list(range(2000))
        y[0], y[9] = y[9], y[0]
        y[10], y[434] = y[434], y[10]
        y[435], y[1509] = y[1509], y[435]
        # rho = 1 - 6 * (2 * (9^2 + 424^2 + 1074^2))/(2000 * (2000^2 - 1))
        #     = 1 - (1 / 500)
        #     = 0.998
        assert_almost_equal(mstats.spearmanr(x,y)[0], 0.998)

        # test for namedtuple attributes
        res = mstats.spearmanr(x, y)
        attributes = ('correlation', 'pvalue')
        check_named_results(res, attributes, ma=True)

    def test_spearmanr_alternative(self):
        # check against R
        # options(digits=16)
        # cor.test(c(2.0, 47.4, 42.0, 10.8, 60.1, 1.7, 64.0, 63.1,
        #            1.0, 1.4, 7.9, 0.3, 3.9, 0.3, 6.7),
        #          c(22.6, 8.3, 44.4, 11.9, 24.6, 0.6, 5.7, 41.6,
        #            0.0, 0.6, 6.7, 3.8, 1.0, 1.2, 1.4),
        #          alternative='two.sided', method='spearman')
        x = [2.0, 47.4, 42.0, 10.8, 60.1, 1.7, 64.0, 63.1,
             1.0, 1.4, 7.9, 0.3, 3.9, 0.3, 6.7]
        y = [22.6, 8.3, 44.4, 11.9, 24.6, 0.6, 5.7, 41.6,
             0.0, 0.6, 6.7, 3.8, 1.0, 1.2, 1.4]

        r_exp = 0.6887298747763864  # from cor.test

        r, p = mstats.spearmanr(x, y)
        assert_allclose(r, r_exp)
        assert_allclose(p, 0.004519192910756)

        r, p = mstats.spearmanr(x, y, alternative='greater')
        assert_allclose(r, r_exp)
        assert_allclose(p, 0.002259596455378)

        r, p = mstats.spearmanr(x, y, alternative='less')
        assert_allclose(r, r_exp)
        assert_allclose(p, 0.9977404035446)

        # intuitive test (with obvious positive correlation)
        n = 100
        x = np.linspace(0, 5, n)
        y = 0.1*x + np.random.rand(n)  # y is positively correlated w/ x

        stat1, p1 = mstats.spearmanr(x, y)

        stat2, p2 = mstats.spearmanr(x, y, alternative="greater")
        assert_allclose(p2, p1 / 2)  # positive correlation -> small p

        stat3, p3 = mstats.spearmanr(x, y, alternative="less")
        assert_allclose(p3, 1 - p1 / 2)  # positive correlation -> large p

        assert stat1 == stat2 == stat3

        with pytest.raises(ValueError, match="alternative must be 'less'..."):
            mstats.spearmanr(x, y, alternative="ekki-ekki")

    @pytest.mark.skipif(platform.machine() == 'ppc64le',
                        reason="fails/crashes on ppc64le")
    def test_kendalltau(self):
        # check case with with maximum disorder and p=1
        x = ma.array(np.array([9, 2, 5, 6]))
        y = ma.array(np.array([4, 7, 9, 11]))
        # Cross-check with exact result from R:
        # cor.test(x,y,method="kendall",exact=1)
        expected = [0.0, 1.0]
        assert_almost_equal(np.asarray(mstats.kendalltau(x, y)), expected)

        # simple case without ties
        x = ma.array(np.arange(10))
        y = ma.array(np.arange(10))
        # Cross-check with exact result from R:
        # cor.test(x,y,method="kendall",exact=1)
        expected = [1.0, 5.511463844797e-07]
        assert_almost_equal(np.asarray(mstats.kendalltau(x, y)), expected)

        # check exception in case of invalid method keyword
        assert_raises(ValueError, mstats.kendalltau, x, y, method='banana')

        # swap a couple of values
        b = y[1]
        y[1] = y[2]
        y[2] = b
        # Cross-check with exact result from R:
        # cor.test(x,y,method="kendall",exact=1)
        expected = [0.9555555555555556, 5.511463844797e-06]
        assert_almost_equal(np.asarray(mstats.kendalltau(x, y)), expected)

        # swap a couple more
        b = y[5]
        y[5] = y[6]
        y[6] = b
        # Cross-check with exact result from R:
        # cor.test(x,y,method="kendall",exact=1)
        expected = [0.9111111111111111, 2.976190476190e-05]
        assert_almost_equal(np.asarray(mstats.kendalltau(x, y)), expected)

        # same in opposite direction
        x = ma.array(np.arange(10))
        y = ma.array(np.arange(10)[::-1])
        # Cross-check with exact result from R:
        # cor.test(x,y,method="kendall",exact=1)
        expected = [-1.0, 5.511463844797e-07]
        assert_almost_equal(np.asarray(mstats.kendalltau(x, y)), expected)

        # swap a couple of values
        b = y[1]
        y[1] = y[2]
        y[2] = b
        # Cross-check with exact result from R:
        # cor.test(x,y,method="kendall",exact=1)
        expected = [-0.9555555555555556, 5.511463844797e-06]
        assert_almost_equal(np.asarray(mstats.kendalltau(x, y)), expected)

        # swap a couple more
        b = y[5]
        y[5] = y[6]
        y[6] = b
        # Cross-check with exact result from R:
        # cor.test(x,y,method="kendall",exact=1)
        expected = [-0.9111111111111111, 2.976190476190e-05]
        assert_almost_equal(np.asarray(mstats.kendalltau(x, y)), expected)

        # Tests some computations of Kendall's tau
        x = ma.fix_invalid([5.05, 6.75, 3.21, 2.66, np.nan])
        y = ma.fix_invalid([1.65, 26.5, -5.93, 7.96, np.nan])
        z = ma.fix_invalid([1.65, 2.64, 2.64, 6.95, np.nan])
        assert_almost_equal(np.asarray(mstats.kendalltau(x, y)),
                            [+0.3333333, 0.75])
        assert_almost_equal(np.asarray(mstats.kendalltau(x, y, method='asymptotic')),
                            [+0.3333333, 0.4969059])
        assert_almost_equal(np.asarray(mstats.kendalltau(x, z)),
                            [-0.5477226, 0.2785987])
        #
        x = ma.fix_invalid([0, 0, 0, 0, 20, 20, 0, 60, 0, 20,
                            10, 10, 0, 40, 0, 20, 0, 0, 0, 0, 0, np.nan])
        y = ma.fix_invalid([0, 80, 80, 80, 10, 33, 60, 0, 67, 27,
                            25, 80, 80, 80, 80, 80, 80, 0, 10, 45, np.nan, 0])
        result = mstats.kendalltau(x, y)
        assert_almost_equal(np.asarray(result), [-0.1585188, 0.4128009])

        # test for namedtuple attributes
        attributes = ('correlation', 'pvalue')
        check_named_results(result, attributes, ma=True)

    @pytest.mark.skipif(platform.machine() == 'ppc64le',
                        reason="fails/crashes on ppc64le")
    @pytest.mark.slow
    def test_kendalltau_large(self):
        # make sure internal variable use correct precision with
        # larger arrays
        x = np.arange(2000, dtype=float)
        x = ma.masked_greater(x, 1995)
        y = np.arange(2000, dtype=float)
        y = np.concatenate((y[1000:], y[:1000]))
        assert_(np.isfinite(mstats.kendalltau(x, y)[1]))


    def test_kendalltau_seasonal(self):
        # Tests the seasonal Kendall tau.
        x = [[nan, nan, 4, 2, 16, 26, 5, 1, 5, 1, 2, 3, 1],
             [4, 3, 5, 3, 2, 7, 3, 1, 1, 2, 3, 5, 3],
             [3, 2, 5, 6, 18, 4, 9, 1, 1, nan, 1, 1, nan],
             [nan, 6, 11, 4, 17, nan, 6, 1, 1, 2, 5, 1, 1]]
        x = ma.fix_invalid(x).T
        output = mstats.kendalltau_seasonal(x)
        assert_almost_equal(output['global p-value (indep)'], 0.008, 3)
        assert_almost_equal(output['seasonal p-value'].round(2),
                            [0.18,0.53,0.20,0.04])

    def test_kendall_p_exact_medium(self):
        # Test for the exact method with medium samples (some n >= 171)
        # expected values generated using SymPy
        expectations = {(100, 2393): 0.62822615287956040664,
                        (101, 2436): 0.60439525773513602669,
                        (170, 0): 2.755801935583541e-307,
                        (171, 0): 0.0,
                        (171, 1): 2.755801935583541e-307,
                        (172, 1): 0.0,
                        (200, 9797): 0.74753983745929675209,
                        (201, 9656): 0.40959218958120363618}
        for nc, expected in expectations.items():
            res = mstats_basic._kendall_p_exact(nc[0], nc[1])
            assert_almost_equal(res, expected)

    @pytest.mark.slow
    def test_kendall_p_exact_large(self):
        # Test for the exact method with large samples (n >= 171)
        # expected values generated using SymPy
        expectations = {(400, 38965): 0.48444283672113314099,
                        (401, 39516): 0.66363159823474837662,
                        (800, 156772): 0.42265448483120932055,
                        (801, 157849): 0.53437553412194416236,
                        (1600, 637472): 0.84200727400323538419,
                        (1601, 630304): 0.34465255088058593946}

        for nc, expected in expectations.items():
            res = mstats_basic._kendall_p_exact(nc[0], nc[1])
            assert_almost_equal(res, expected)


    def test_pointbiserial(self):
        x = [1,0,1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,
             0,0,0,0,1,-1]
        y = [14.8,13.8,12.4,10.1,7.1,6.1,5.8,4.6,4.3,3.5,3.3,3.2,3.0,
             2.8,2.8,2.5,2.4,2.3,2.1,1.7,1.7,1.5,1.3,1.3,1.2,1.2,1.1,
             0.8,0.7,0.6,0.5,0.2,0.2,0.1,np.nan]
        assert_almost_equal(mstats.pointbiserialr(x, y)[0], 0.36149, 5)

        # test for namedtuple attributes
        res = mstats.pointbiserialr(x, y)
        attributes = ('correlation', 'pvalue')
        check_named_results(res, attributes, ma=True)


class TestTrimming:

    def test_trim(self):
        a = ma.arange(10)
        assert_equal(mstats.trim(a), [0,1,2,3,4,5,6,7,8,9])
        a = ma.arange(10)
        assert_equal(mstats.trim(a,(2,8)), [None,None,2,3,4,5,6,7,8,None])
        a = ma.arange(10)
        assert_equal(mstats.trim(a,limits=(2,8),inclusive=(False,False)),
                     [None,None,None,3,4,5,6,7,None,None])
        a = ma.arange(10)
        assert_equal(mstats.trim(a,limits=(0.1,0.2),relative=True),
                     [None,1,2,3,4,5,6,7,None,None])

        a = ma.arange(12)
        a[[0,-1]] = a[5] = masked
        assert_equal(mstats.trim(a, (2,8)),
                     [None, None, 2, 3, 4, None, 6, 7, 8, None, None, None])

        x = ma.arange(100).reshape(10, 10)
        expected = [1]*10 + [0]*70 + [1]*20
        trimx = mstats.trim(x, (0.1,0.2), relative=True, axis=None)
        assert_equal(trimx._mask.ravel(), expected)
        trimx = mstats.trim(x, (0.1,0.2), relative=True, axis=0)
        assert_equal(trimx._mask.ravel(), expected)
        trimx = mstats.trim(x, (0.1,0.2), relative=True, axis=-1)
        assert_equal(trimx._mask.T.ravel(), expected)

        # same as above, but with an extra masked row inserted
        x = ma.arange(110).reshape(11, 10)
        x[1] = masked
        expected = [1]*20 + [0]*70 + [1]*20
        trimx = mstats.trim(x, (0.1,0.2), relative=True, axis=None)
        assert_equal(trimx._mask.ravel(), expected)
        trimx = mstats.trim(x, (0.1,0.2), relative=True, axis=0)
        assert_equal(trimx._mask.ravel(), expected)
        trimx = mstats.trim(x.T, (0.1,0.2), relative=True, axis=-1)
        assert_equal(trimx.T._mask.ravel(), expected)

    def test_trim_old(self):
        x = ma.arange(100)
        assert_equal(mstats.trimboth(x).count(), 60)
        assert_equal(mstats.trimtail(x,tail='r').count(), 80)
        x[50:70] = masked
        trimx = mstats.trimboth(x)
        assert_equal(trimx.count(), 48)
        assert_equal(trimx._mask, [1]*16 + [0]*34 + [1]*20 + [0]*14 + [1]*16)
        x._mask = nomask
        x.shape = (10,10)
        assert_equal(mstats.trimboth(x).count(), 60)
        assert_equal(mstats.trimtail(x).count(), 80)

    def test_trimr(self):
        x = ma.arange(10)
        result = mstats.trimr(x, limits=(0.15, 0.14), inclusive=(False, False))
        expected = ma.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                            mask=[1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
        assert_equal(result, expected)
        assert_equal(result.mask, expected.mask)

    def test_trimmedmean(self):
        data = ma.array([77, 87, 88,114,151,210,219,246,253,262,
                         296,299,306,376,428,515,666,1310,2611])
        assert_almost_equal(mstats.trimmed_mean(data,0.1), 343, 0)
        assert_almost_equal(mstats.trimmed_mean(data,(0.1,0.1)), 343, 0)
        assert_almost_equal(mstats.trimmed_mean(data,(0.2,0.2)), 283, 0)

    def test_trimmed_stde(self):
        data = ma.array([77, 87, 88,114,151,210,219,246,253,262,
                         296,299,306,376,428,515,666,1310,2611])
        assert_almost_equal(mstats.trimmed_stde(data,(0.2,0.2)), 56.13193, 5)
        assert_almost_equal(mstats.trimmed_stde(data,0.2), 56.13193, 5)

    def test_winsorization(self):
        data = ma.array([77, 87, 88,114,151,210,219,246,253,262,
                         296,299,306,376,428,515,666,1310,2611])
        assert_almost_equal(mstats.winsorize(data,(0.2,0.2)).var(ddof=1),
                            21551.4, 1)
        assert_almost_equal(
            mstats.winsorize(data, (0.2,0.2),(False,False)).var(ddof=1),
            11887.3, 1)
        data[5] = masked
        winsorized = mstats.winsorize(data)
        assert_equal(winsorized.mask, data.mask)

    def test_winsorization_nan(self):
        data = ma.array([np.nan, np.nan, 0, 1, 2])
        assert_raises(ValueError, mstats.winsorize, data, (0.05, 0.05),
                      nan_policy='raise')
        # Testing propagate (default behavior)
        assert_equal(mstats.winsorize(data, (0.4, 0.4)),
                     ma.array([2, 2, 2, 2, 2]))
        assert_equal(mstats.winsorize(data, (0.8, 0.8)),
                     ma.array([np.nan, np.nan, np.nan, np.nan, np.nan]))
        assert_equal(mstats.winsorize(data, (0.4, 0.4), nan_policy='omit'),
                     ma.array([np.nan, np.nan, 2, 2, 2]))
        assert_equal(mstats.winsorize(data, (0.8, 0.8), nan_policy='omit'),
                     ma.array([np.nan, np.nan, 2, 2, 2]))

class TestMoments:
    # Comparison numbers are found using R v.1.5.1
    # note that length(testcase) = 4
    # testmathworks comes from documentation for the
    # Statistics Toolbox for Matlab and can be found at both
    # https://www.mathworks.com/help/stats/kurtosis.html
    # https://www.mathworks.com/help/stats/skewness.html
    # Note that both test cases came from here.
    testcase = [1,2,3,4]
    testmathworks = ma.fix_invalid([1.165, 0.6268, 0.0751, 0.3516, -0.6965,
                                    np.nan])
    testcase_2d = ma.array(
    np.array([[0.05245846, 0.50344235, 0.86589117, 0.36936353, 0.46961149],
           [0.11574073, 0.31299969, 0.45925772, 0.72618805, 0.75194407],
           [0.67696689, 0.91878127, 0.09769044, 0.04645137, 0.37615733],
           [0.05903624, 0.29908861, 0.34088298, 0.66216337, 0.83160998],
           [0.64619526, 0.94894632, 0.27855892, 0.0706151, 0.39962917]]),
    mask=np.array([[True, False, False, True, False],
           [True, True, True, False, True],
           [False, False, False, False, False],
           [True, True, True, True, True],
           [False, False, True, False, False]], dtype=bool))

    def _assert_equal(self, actual, expect, *, shape=None, dtype=None):
        expect = np.asarray(expect)
        if shape is not None:
            expect = np.broadcast_to(expect, shape)
        assert_array_equal(actual, expect)
        if dtype is None:
            dtype = expect.dtype
        assert actual.dtype == dtype

    def test_moment(self):
        y = mstats.moment(self.testcase,1)
        assert_almost_equal(y,0.0,10)
        y = mstats.moment(self.testcase,2)
        assert_almost_equal(y,1.25)
        y = mstats.moment(self.testcase,3)
        assert_almost_equal(y,0.0)
        y = mstats.moment(self.testcase,4)
        assert_almost_equal(y,2.5625)

        # check array_like input for moment
        y = mstats.moment(self.testcase, [1, 2, 3, 4])
        assert_allclose(y, [0, 1.25, 0, 2.5625])

        # check moment input consists only of integers
        y = mstats.moment(self.testcase, 0.0)
        assert_allclose(y, 1.0)
        assert_raises(ValueError, mstats.moment, self.testcase, 1.2)
        y = mstats.moment(self.testcase, [1.0, 2, 3, 4.0])
        assert_allclose(y, [0, 1.25, 0, 2.5625])

        # test empty input
        y = mstats.moment([])
        self._assert_equal(y, np.nan, dtype=np.float64)
        y = mstats.moment(np.array([], dtype=np.float32))
        self._assert_equal(y, np.nan, dtype=np.float32)
        y = mstats.moment(np.zeros((1, 0)), axis=0)
        self._assert_equal(y, [], shape=(0,), dtype=np.float64)
        y = mstats.moment([[]], axis=1)
        self._assert_equal(y, np.nan, shape=(1,), dtype=np.float64)
        y = mstats.moment([[]], moment=[0, 1], axis=0)
        self._assert_equal(y, [], shape=(2, 0))

        x = np.arange(10.)
        x[9] = np.nan
        assert_equal(mstats.moment(x, 2), ma.masked) # NaN value is ignored

    def test_variation(self):
        y = mstats.variation(self.testcase)
        assert_almost_equal(y,0.44721359549996, 10)

    def test_variation_ddof(self):
        # test variation with delta degrees of freedom
        # regression test for gh-13341
        a = np.array([1, 2, 3, 4, 5])
        y = mstats.variation(a, ddof=1)
        assert_almost_equal(y, 0.5270462766947299)

    def test_skewness(self):
        y = mstats.skew(self.testmathworks)
        assert_almost_equal(y,-0.29322304336607,10)
        y = mstats.skew(self.testmathworks,bias=0)
        assert_almost_equal(y,-0.437111105023940,10)
        y = mstats.skew(self.testcase)
        assert_almost_equal(y,0.0,10)

    def test_kurtosis(self):
        # Set flags for axis = 0 and fisher=0 (Pearson's definition of kurtosis
        # for compatibility with Matlab)
        y = mstats.kurtosis(self.testmathworks, 0, fisher=0, bias=1)
        assert_almost_equal(y, 2.1658856802973, 10)
        # Note that MATLAB has confusing docs for the following case
        #  kurtosis(x,0) gives an unbiased estimate of Pearson's skewness
        #  kurtosis(x) gives a biased estimate of Fisher's skewness (Pearson-3)
        #  The MATLAB docs imply that both should give Fisher's
        y = mstats.kurtosis(self.testmathworks, fisher=0, bias=0)
        assert_almost_equal(y, 3.663542721189047, 10)
        y = mstats.kurtosis(self.testcase, 0, 0)
        assert_almost_equal(y, 1.64)

        # test that kurtosis works on multidimensional masked arrays
        correct_2d = ma.array(np.array([-1.5, -3., -1.47247052385, 0.,
                                        -1.26979517952]),
                              mask=np.array([False, False, False, True,
                                             False], dtype=bool))
        assert_array_almost_equal(mstats.kurtosis(self.testcase_2d, 1),
                                  correct_2d)
        for i, row in enumerate(self.testcase_2d):
            assert_almost_equal(mstats.kurtosis(row), correct_2d[i])

        correct_2d_bias_corrected = ma.array(
            np.array([-1.5, -3., -1.88988209538, 0., -0.5234638463918877]),
            mask=np.array([False, False, False, True, False], dtype=bool))
        assert_array_almost_equal(mstats.kurtosis(self.testcase_2d, 1,
                                                  bias=False),
                                  correct_2d_bias_corrected)
        for i, row in enumerate(self.testcase_2d):
            assert_almost_equal(mstats.kurtosis(row, bias=False),
                                correct_2d_bias_corrected[i])

        # Check consistency between stats and mstats implementations
        assert_array_almost_equal_nulp(mstats.kurtosis(self.testcase_2d[2, :]),
                                       stats.kurtosis(self.testcase_2d[2, :]),
                                       nulp=4)

    def test_mode(self):
        a1 = [0,0,0,1,1,1,2,3,3,3,3,4,5,6,7]
        a2 = np.reshape(a1, (3,5))
        a3 = np.array([1,2,3,4,5,6])
        a4 = np.reshape(a3, (3,2))
        ma1 = ma.masked_where(ma.array(a1) > 2, a1)
        ma2 = ma.masked_where(a2 > 2, a2)
        ma3 = ma.masked_where(a3 < 2, a3)
        ma4 = ma.masked_where(ma.array(a4) < 2, a4)
        assert_equal(mstats.mode(a1, axis=None), (3,4))
        assert_equal(mstats.mode(a1, axis=0), (3,4))
        assert_equal(mstats.mode(ma1, axis=None), (0,3))
        assert_equal(mstats.mode(a2, axis=None), (3,4))
        assert_equal(mstats.mode(ma2, axis=None), (0,3))
        assert_equal(mstats.mode(a3, axis=None), (1,1))
        assert_equal(mstats.mode(ma3, axis=None), (2,1))
        assert_equal(mstats.mode(a2, axis=0), ([[0,0,0,1,1]], [[1,1,1,1,1]]))
        assert_equal(mstats.mode(ma2, axis=0), ([[0,0,0,1,1]], [[1,1,1,1,1]]))
        assert_equal(mstats.mode(a2, axis=-1), ([[0],[3],[3]], [[3],[3],[1]]))
        assert_equal(mstats.mode(ma2, axis=-1), ([[0],[1],[0]], [[3],[1],[0]]))
        assert_equal(mstats.mode(ma4, axis=0), ([[3,2]], [[1,1]]))
        assert_equal(mstats.mode(ma4, axis=-1), ([[2],[3],[5]], [[1],[1],[1]]))

        a1_res = mstats.mode(a1, axis=None)

        # test for namedtuple attributes
        attributes = ('mode', 'count')
        check_named_results(a1_res, attributes, ma=True)

    def test_mode_modifies_input(self):
        # regression test for gh-6428: mode(..., axis=None) may not modify
        # the input array
        im = np.zeros((100, 100))
        im[:50, :] += 1
        im[:, :50] += 1
        cp = im.copy()
        mstats.mode(im, None)
        assert_equal(im, cp)


class TestPercentile:
    def setup_method(self):
        self.a1 = [3, 4, 5, 10, -3, -5, 6]
        self.a2 = [3, -6, -2, 8, 7, 4, 2, 1]
        self.a3 = [3., 4, 5, 10, -3, -5, -6, 7.0]

    def test_percentile(self):
        x = np.arange(8) * 0.5
        assert_equal(mstats.scoreatpercentile(x, 0), 0.)
        assert_equal(mstats.scoreatpercentile(x, 100), 3.5)
        assert_equal(mstats.scoreatpercentile(x, 50), 1.75)

    def test_2D(self):
        x = ma.array([[1, 1, 1],
                      [1, 1, 1],
                      [4, 4, 3],
                      [1, 1, 1],
                      [1, 1, 1]])
        assert_equal(mstats.scoreatpercentile(x, 50), [1, 1, 1])


class TestVariability:
    """  Comparison numbers are found using R v.1.5.1
         note that length(testcase) = 4
    """
    testcase = ma.fix_invalid([1,2,3,4,np.nan])

    def test_sem(self):
        # This is not in R, so used: sqrt(var(testcase)*3/4) / sqrt(3)
        y = mstats.sem(self.testcase)
        assert_almost_equal(y, 0.6454972244)
        n = self.testcase.count()
        assert_allclose(mstats.sem(self.testcase, ddof=0) * np.sqrt(n/(n-2)),
                        mstats.sem(self.testcase, ddof=2))

    def test_zmap(self):
        # This is not in R, so tested by using:
        #    (testcase[i]-mean(testcase,axis=0)) / sqrt(var(testcase)*3/4)
        y = mstats.zmap(self.testcase, self.testcase)
        desired_unmaskedvals = ([-1.3416407864999, -0.44721359549996,
                                 0.44721359549996, 1.3416407864999])
        assert_array_almost_equal(desired_unmaskedvals,
                                  y.data[y.mask == False], decimal=12)

    def test_zscore(self):
        # This is not in R, so tested by using:
        #     (testcase[i]-mean(testcase,axis=0)) / sqrt(var(testcase)*3/4)
        y = mstats.zscore(self.testcase)
        desired = ma.fix_invalid([-1.3416407864999, -0.44721359549996,
                                  0.44721359549996, 1.3416407864999, np.nan])
        assert_almost_equal(desired, y, decimal=12)


class TestMisc:

    def test_obrientransform(self):
        args = [[5]*5+[6]*11+[7]*9+[8]*3+[9]*2+[10]*2,
                [6]+[7]*2+[8]*4+[9]*9+[10]*16]
        result = [5*[3.1828]+11*[0.5591]+9*[0.0344]+3*[1.6086]+2*[5.2817]+2*[11.0538],
                  [10.4352]+2*[4.8599]+4*[1.3836]+9*[0.0061]+16*[0.7277]]
        assert_almost_equal(np.round(mstats.obrientransform(*args).T, 4),
                            result, 4)

    def test_ks_2samp(self):
        x = [[nan,nan, 4, 2, 16, 26, 5, 1, 5, 1, 2, 3, 1],
             [4, 3, 5, 3, 2, 7, 3, 1, 1, 2, 3, 5, 3],
             [3, 2, 5, 6, 18, 4, 9, 1, 1, nan, 1, 1, nan],
             [nan, 6, 11, 4, 17, nan, 6, 1, 1, 2, 5, 1, 1]]
        x = ma.fix_invalid(x).T
        (winter, spring, summer, fall) = x.T

        assert_almost_equal(np.round(mstats.ks_2samp(winter, spring), 4),
                            (0.1818, 0.9628))
        assert_almost_equal(np.round(mstats.ks_2samp(winter, spring, 'g'), 4),
                            (0.1469, 0.6886))
        assert_almost_equal(np.round(mstats.ks_2samp(winter, spring, 'l'), 4),
                            (0.1818, 0.6011))

    def test_friedmanchisq(self):
        # No missing values
        args = ([9.0,9.5,5.0,7.5,9.5,7.5,8.0,7.0,8.5,6.0],
                [7.0,6.5,7.0,7.5,5.0,8.0,6.0,6.5,7.0,7.0],
                [6.0,8.0,4.0,6.0,7.0,6.5,6.0,4.0,6.5,3.0])
        result = mstats.friedmanchisquare(*args)
        assert_almost_equal(result[0], 10.4737, 4)
        assert_almost_equal(result[1], 0.005317, 6)
        # Missing values
        x = [[nan,nan, 4, 2, 16, 26, 5, 1, 5, 1, 2, 3, 1],
             [4, 3, 5, 3, 2, 7, 3, 1, 1, 2, 3, 5, 3],
             [3, 2, 5, 6, 18, 4, 9, 1, 1,nan, 1, 1,nan],
             [nan, 6, 11, 4, 17,nan, 6, 1, 1, 2, 5, 1, 1]]
        x = ma.fix_invalid(x)
        result = mstats.friedmanchisquare(*x)
        assert_almost_equal(result[0], 2.0156, 4)
        assert_almost_equal(result[1], 0.5692, 4)

        # test for namedtuple attributes
        attributes = ('statistic', 'pvalue')
        check_named_results(result, attributes, ma=True)


def test_regress_simple():
    # Regress a line with sinusoidal noise. Test for #1273.
    x = np.linspace(0, 100, 100)
    y = 0.2 * np.linspace(0, 100, 100) + 10
    y += np.sin(np.linspace(0, 20, 100))

    result = mstats.linregress(x, y)

    # Result is of a correct class and with correct fields
    lr = stats._stats_mstats_common.LinregressResult
    assert_(isinstance(result, lr))
    attributes = ('slope', 'intercept', 'rvalue', 'pvalue', 'stderr')
    check_named_results(result, attributes, ma=True)
    assert 'intercept_stderr' in dir(result)

    # Slope and intercept are estimated correctly
    assert_almost_equal(result.slope, 0.19644990055858422)
    assert_almost_equal(result.intercept, 10.211269918932341)
    assert_almost_equal(result.stderr, 0.002395781449783862)
    assert_almost_equal(result.intercept_stderr, 0.13866936078570702)

def test_theilslopes():
    # Test for basic slope and intercept.
    slope, intercept, lower, upper = mstats.theilslopes([0, 1, 1])
    assert_almost_equal(slope, 0.5)
    assert_almost_equal(intercept, 0.5)

    # Test for correct masking.
    y = np.ma.array([0, 1, 100, 1], mask=[False, False, True, False])
    slope, intercept, lower, upper = mstats.theilslopes(y)
    assert_almost_equal(slope, 1./3)
    assert_almost_equal(intercept, 2./3)

    # Test of confidence intervals from example in Sen (1968).
    x = [1, 2, 3, 4, 10, 12, 18]
    y = [9, 15, 19, 20, 45, 55, 78]
    slope, intercept, lower, upper = mstats.theilslopes(y, x, 0.07)
    assert_almost_equal(slope, 4)
    assert_almost_equal(upper, 4.38, decimal=2)
    assert_almost_equal(lower, 3.71, decimal=2)


def test_siegelslopes():
    # method should be exact for straight line
    y = 2 * np.arange(10) + 0.5
    assert_equal(mstats.siegelslopes(y), (2.0, 0.5))
    assert_equal(mstats.siegelslopes(y, method='separate'), (2.0, 0.5))

    x = 2 * np.arange(10)
    y = 5 * x - 3.0
    assert_equal(mstats.siegelslopes(y, x), (5.0, -3.0))
    assert_equal(mstats.siegelslopes(y, x, method='separate'), (5.0, -3.0))

    # method is robust to outliers: brekdown point of 50%
    y[:4] = 1000
    assert_equal(mstats.siegelslopes(y, x), (5.0, -3.0))

    # if there are no outliers, results should be comparble to linregress
    x = np.arange(10)
    y = -2.3 + 0.3*x + stats.norm.rvs(size=10, random_state=231)
    slope_ols, intercept_ols, _, _, _ = stats.linregress(x, y)

    slope, intercept = mstats.siegelslopes(y, x)
    assert_allclose(slope, slope_ols, rtol=0.1)
    assert_allclose(intercept, intercept_ols, rtol=0.1)

    slope, intercept = mstats.siegelslopes(y, x, method='separate')
    assert_allclose(slope, slope_ols, rtol=0.1)
    assert_allclose(intercept, intercept_ols, rtol=0.1)


def test_plotting_positions():
    # Regression test for #1256
    pos = mstats.plotting_positions(np.arange(3), 0, 0)
    assert_array_almost_equal(pos.data, np.array([0.25, 0.5, 0.75]))


class TestNormalitytests():

    def test_vs_nonmasked(self):
        x = np.array((-2, -1, 0, 1, 2, 3)*4)**2
        assert_array_almost_equal(mstats.normaltest(x),
                                  stats.normaltest(x))
        assert_array_almost_equal(mstats.skewtest(x),
                                  stats.skewtest(x))
        assert_array_almost_equal(mstats.kurtosistest(x),
                                  stats.kurtosistest(x))

        funcs = [stats.normaltest, stats.skewtest, stats.kurtosistest]
        mfuncs = [mstats.normaltest, mstats.skewtest, mstats.kurtosistest]
        x = [1, 2, 3, 4]
        for func, mfunc in zip(funcs, mfuncs):
            assert_raises(ValueError, func, x)
            assert_raises(ValueError, mfunc, x)

    def test_axis_None(self):
        # Test axis=None (equal to axis=0 for 1-D input)
        x = np.array((-2,-1,0,1,2,3)*4)**2
        assert_allclose(mstats.normaltest(x, axis=None), mstats.normaltest(x))
        assert_allclose(mstats.skewtest(x, axis=None), mstats.skewtest(x))
        assert_allclose(mstats.kurtosistest(x, axis=None),
                        mstats.kurtosistest(x))

    def test_maskedarray_input(self):
        # Add some masked values, test result doesn't change
        x = np.array((-2, -1, 0, 1, 2, 3)*4)**2
        xm = np.ma.array(np.r_[np.inf, x, 10],
                         mask=np.r_[True, [False] * x.size, True])
        assert_allclose(mstats.normaltest(xm), stats.normaltest(x))
        assert_allclose(mstats.skewtest(xm), stats.skewtest(x))
        assert_allclose(mstats.kurtosistest(xm), stats.kurtosistest(x))

    def test_nd_input(self):
        x = np.array((-2, -1, 0, 1, 2, 3)*4)**2
        x_2d = np.vstack([x] * 2).T
        for func in [mstats.normaltest, mstats.skewtest, mstats.kurtosistest]:
            res_1d = func(x)
            res_2d = func(x_2d)
            assert_allclose(res_2d[0], [res_1d[0]] * 2)
            assert_allclose(res_2d[1], [res_1d[1]] * 2)

    def test_normaltest_result_attributes(self):
        x = np.array((-2, -1, 0, 1, 2, 3)*4)**2
        res = mstats.normaltest(x)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes, ma=True)

    def test_kurtosistest_result_attributes(self):
        x = np.array((-2, -1, 0, 1, 2, 3)*4)**2
        res = mstats.kurtosistest(x)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes, ma=True)

    def regression_test_9033(self):
        # x cleary non-normal but power of negtative denom needs
        # to be handled correctly to reject normality
        counts = [128, 0, 58, 7, 0, 41, 16, 0, 0, 167]
        x = np.hstack([np.full(c, i) for i, c in enumerate(counts)])
        assert_equal(mstats.kurtosistest(x)[1] < 0.01, True)


class TestFOneway():
    def test_result_attributes(self):
        a = np.array([655, 788], dtype=np.uint16)
        b = np.array([789, 772], dtype=np.uint16)
        res = mstats.f_oneway(a, b)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes, ma=True)


class TestMannwhitneyu():
    # data from gh-1428
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

    def test_result_attributes(self):
        res = mstats.mannwhitneyu(self.x, self.y)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes, ma=True)

    def test_against_stats(self):
        # gh-4641 reported that stats.mannwhitneyu returned half the p-value
        # of mstats.mannwhitneyu. Default alternative of stats.mannwhitneyu
        # is now two-sided, so they match.
        res1 = mstats.mannwhitneyu(self.x, self.y)
        res2 = stats.mannwhitneyu(self.x, self.y)
        assert res1.statistic == res2.statistic
        assert_allclose(res1.pvalue, res2.pvalue)


class TestKruskal():
    def test_result_attributes(self):
        x = [1, 3, 5, 7, 9]
        y = [2, 4, 6, 8, 10]

        res = mstats.kruskal(x, y)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes, ma=True)


# TODO: for all ttest functions, add tests with masked array inputs
class TestTtest_rel():
    def test_vs_nonmasked(self):
        np.random.seed(1234567)
        outcome = np.random.randn(20, 4) + [0, 0, 1, 2]

        # 1-D inputs
        res1 = stats.ttest_rel(outcome[:, 0], outcome[:, 1])
        res2 = mstats.ttest_rel(outcome[:, 0], outcome[:, 1])
        assert_allclose(res1, res2)

        # 2-D inputs
        res1 = stats.ttest_rel(outcome[:, 0], outcome[:, 1], axis=None)
        res2 = mstats.ttest_rel(outcome[:, 0], outcome[:, 1], axis=None)
        assert_allclose(res1, res2)
        res1 = stats.ttest_rel(outcome[:, :2], outcome[:, 2:], axis=0)
        res2 = mstats.ttest_rel(outcome[:, :2], outcome[:, 2:], axis=0)
        assert_allclose(res1, res2)

        # Check default is axis=0
        res3 = mstats.ttest_rel(outcome[:, :2], outcome[:, 2:])
        assert_allclose(res2, res3)

    def test_fully_masked(self):
        np.random.seed(1234567)
        outcome = ma.masked_array(np.random.randn(3, 2),
                                  mask=[[1, 1, 1], [0, 0, 0]])
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in absolute")
            for pair in [(outcome[:, 0], outcome[:, 1]), ([np.nan, np.nan], [1.0, 2.0])]:
                t, p = mstats.ttest_rel(*pair)
                assert_array_equal(t, (np.nan, np.nan))
                assert_array_equal(p, (np.nan, np.nan))

    def test_result_attributes(self):
        np.random.seed(1234567)
        outcome = np.random.randn(20, 4) + [0, 0, 1, 2]

        res = mstats.ttest_rel(outcome[:, 0], outcome[:, 1])
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes, ma=True)

    def test_invalid_input_size(self):
        assert_raises(ValueError, mstats.ttest_rel,
                      np.arange(10), np.arange(11))
        x = np.arange(24)
        assert_raises(ValueError, mstats.ttest_rel,
                      x.reshape(2, 3, 4), x.reshape(2, 4, 3), axis=1)
        assert_raises(ValueError, mstats.ttest_rel,
                      x.reshape(2, 3, 4), x.reshape(2, 4, 3), axis=2)

    def test_empty(self):
        res1 = mstats.ttest_rel([], [])
        assert_(np.all(np.isnan(res1)))

    def test_zero_division(self):
        t, p = mstats.ttest_ind([0, 0, 0], [1, 1, 1])
        assert_equal((np.abs(t), p), (np.inf, 0))

        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in absolute")
            t, p = mstats.ttest_ind([0, 0, 0], [0, 0, 0])
            assert_array_equal(t, np.array([np.nan, np.nan]))
            assert_array_equal(p, np.array([np.nan, np.nan]))


class TestTtest_ind():
    def test_vs_nonmasked(self):
        np.random.seed(1234567)
        outcome = np.random.randn(20, 4) + [0, 0, 1, 2]

        # 1-D inputs
        res1 = stats.ttest_ind(outcome[:, 0], outcome[:, 1])
        res2 = mstats.ttest_ind(outcome[:, 0], outcome[:, 1])
        assert_allclose(res1, res2)

        # 2-D inputs
        res1 = stats.ttest_ind(outcome[:, 0], outcome[:, 1], axis=None)
        res2 = mstats.ttest_ind(outcome[:, 0], outcome[:, 1], axis=None)
        assert_allclose(res1, res2)
        res1 = stats.ttest_ind(outcome[:, :2], outcome[:, 2:], axis=0)
        res2 = mstats.ttest_ind(outcome[:, :2], outcome[:, 2:], axis=0)
        assert_allclose(res1, res2)

        # Check default is axis=0
        res3 = mstats.ttest_ind(outcome[:, :2], outcome[:, 2:])
        assert_allclose(res2, res3)

        # Check equal_var
        res4 = stats.ttest_ind(outcome[:, 0], outcome[:, 1], equal_var=True)
        res5 = mstats.ttest_ind(outcome[:, 0], outcome[:, 1], equal_var=True)
        assert_allclose(res4, res5)
        res4 = stats.ttest_ind(outcome[:, 0], outcome[:, 1], equal_var=False)
        res5 = mstats.ttest_ind(outcome[:, 0], outcome[:, 1], equal_var=False)
        assert_allclose(res4, res5)

    def test_fully_masked(self):
        np.random.seed(1234567)
        outcome = ma.masked_array(np.random.randn(3, 2), mask=[[1, 1, 1], [0, 0, 0]])
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in absolute")
            for pair in [(outcome[:, 0], outcome[:, 1]), ([np.nan, np.nan], [1.0, 2.0])]:
                t, p = mstats.ttest_ind(*pair)
                assert_array_equal(t, (np.nan, np.nan))
                assert_array_equal(p, (np.nan, np.nan))

    def test_result_attributes(self):
        np.random.seed(1234567)
        outcome = np.random.randn(20, 4) + [0, 0, 1, 2]

        res = mstats.ttest_ind(outcome[:, 0], outcome[:, 1])
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes, ma=True)

    def test_empty(self):
        res1 = mstats.ttest_ind([], [])
        assert_(np.all(np.isnan(res1)))

    def test_zero_division(self):
        t, p = mstats.ttest_ind([0, 0, 0], [1, 1, 1])
        assert_equal((np.abs(t), p), (np.inf, 0))

        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in absolute")
            t, p = mstats.ttest_ind([0, 0, 0], [0, 0, 0])
            assert_array_equal(t, (np.nan, np.nan))
            assert_array_equal(p, (np.nan, np.nan))

        t, p = mstats.ttest_ind([0, 0, 0], [1, 1, 1], equal_var=False)
        assert_equal((np.abs(t), p), (np.inf, 0))
        assert_array_equal(mstats.ttest_ind([0, 0, 0], [0, 0, 0],
                                            equal_var=False), (np.nan, np.nan))


class TestTtest_1samp():
    def test_vs_nonmasked(self):
        np.random.seed(1234567)
        outcome = np.random.randn(20, 4) + [0, 0, 1, 2]

        # 1-D inputs
        res1 = stats.ttest_1samp(outcome[:, 0], 1)
        res2 = mstats.ttest_1samp(outcome[:, 0], 1)
        assert_allclose(res1, res2)

        # 2-D inputs
        res1 = stats.ttest_1samp(outcome[:, 0], outcome[:, 1], axis=None)
        res2 = mstats.ttest_1samp(outcome[:, 0], outcome[:, 1], axis=None)
        assert_allclose(res1, res2)

        res1 = stats.ttest_1samp(outcome[:, :2], outcome[:, 2:], axis=0)
        res2 = mstats.ttest_1samp(outcome[:, :2], outcome[:, 2:], axis=0)
        assert_allclose(res1, res2, atol=1e-15)

        # Check default is axis=0
        res3 = mstats.ttest_1samp(outcome[:, :2], outcome[:, 2:])
        assert_allclose(res2, res3)

    def test_fully_masked(self):
        np.random.seed(1234567)
        outcome = ma.masked_array(np.random.randn(3), mask=[1, 1, 1])
        expected = (np.nan, np.nan)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in absolute")
            for pair in [((np.nan, np.nan), 0.0), (outcome, 0.0)]:
                t, p = mstats.ttest_1samp(*pair)
                assert_array_equal(p, expected)
                assert_array_equal(t, expected)

    def test_result_attributes(self):
        np.random.seed(1234567)
        outcome = np.random.randn(20, 4) + [0, 0, 1, 2]

        res = mstats.ttest_1samp(outcome[:, 0], 1)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes, ma=True)

    def test_empty(self):
        res1 = mstats.ttest_1samp([], 1)
        assert_(np.all(np.isnan(res1)))

    def test_zero_division(self):
        t, p = mstats.ttest_1samp([0, 0, 0], 1)
        assert_equal((np.abs(t), p), (np.inf, 0))

        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in absolute")
            t, p = mstats.ttest_1samp([0, 0, 0], 0)
            assert_(np.isnan(t))
            assert_array_equal(p, (np.nan, np.nan))


class TestDescribe:
    """
    Tests for mstats.describe.

    Note that there are also tests for `mstats.describe` in the
    class TestCompareWithStats.
    """
    def test_basic_with_axis(self):
        # This is a basic test that is also a regression test for gh-7303.
        a = np.ma.masked_array([[0, 1, 2, 3, 4, 9],
                                [5, 5, 0, 9, 3, 3]],
                               mask=[[0, 0, 0, 0, 0, 1],
                                     [0, 0, 1, 1, 0, 0]])
        result = mstats.describe(a, axis=1)
        assert_equal(result.nobs, [5, 4])
        amin, amax = result.minmax
        assert_equal(amin, [0, 3])
        assert_equal(amax, [4, 5])
        assert_equal(result.mean, [2.0, 4.0])
        assert_equal(result.variance, [2.0, 1.0])
        assert_equal(result.skewness, [0.0, 0.0])
        assert_allclose(result.kurtosis, [-1.3, -2.0])


class TestCompareWithStats:
    """
    Class to compare mstats results with stats results.

    It is in general assumed that scipy.stats is at a more mature stage than
    stats.mstats.  If a routine in mstats results in similar results like in
    scipy.stats, this is considered also as a proper validation of scipy.mstats
    routine.

    Different sample sizes are used for testing, as some problems between stats
    and mstats are dependent on sample size.

    Author: Alexander Loew

    NOTE that some tests fail. This might be caused by
    a) actual differences or bugs between stats and mstats
    b) numerical inaccuracies
    c) different definitions of routine interfaces

    These failures need to be checked. Current workaround is to have disabled these tests,
    but issuing reports on scipy-dev

    """
    def get_n(self):
        """ Returns list of sample sizes to be used for comparison. """
        return [1000, 100, 10, 5]

    def generate_xy_sample(self, n):
        # This routine generates numpy arrays and corresponding masked arrays
        # with the same data, but additional masked values
        np.random.seed(1234567)
        x = np.random.randn(n)
        y = x + np.random.randn(n)
        xm = np.full(len(x) + 5, 1e16)
        ym = np.full(len(y) + 5, 1e16)
        xm[0:len(x)] = x
        ym[0:len(y)] = y
        mask = xm > 9e15
        xm = np.ma.array(xm, mask=mask)
        ym = np.ma.array(ym, mask=mask)
        return x, y, xm, ym

    def generate_xy_sample2D(self, n, nx):
        x = np.full((n, nx), np.nan)
        y = np.full((n, nx), np.nan)
        xm = np.full((n+5, nx), np.nan)
        ym = np.full((n+5, nx), np.nan)

        for i in range(nx):
            x[:, i], y[:, i], dx, dy = self.generate_xy_sample(n)

        xm[0:n, :] = x[0:n]
        ym[0:n, :] = y[0:n]
        xm = np.ma.array(xm, mask=np.isnan(xm))
        ym = np.ma.array(ym, mask=np.isnan(ym))
        return x, y, xm, ym

    def test_linregress(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            result1 = stats.linregress(x, y)
            result2 = stats.mstats.linregress(xm, ym)
            assert_allclose(np.asarray(result1), np.asarray(result2))

    def test_pearsonr(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            r, p = stats.pearsonr(x, y)
            rm, pm = stats.mstats.pearsonr(xm, ym)

            assert_almost_equal(r, rm, decimal=14)
            assert_almost_equal(p, pm, decimal=14)

    def test_spearmanr(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            r, p = stats.spearmanr(x, y)
            rm, pm = stats.mstats.spearmanr(xm, ym)
            assert_almost_equal(r, rm, 14)
            assert_almost_equal(p, pm, 14)

    def test_spearmanr_backcompat_useties(self):
        # A regression test to ensure we don't break backwards compat
        # more than we have to (see gh-9204).
        x = np.arange(6)
        assert_raises(ValueError, mstats.spearmanr, x, x, False)

    def test_gmean(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            r = stats.gmean(abs(x))
            rm = stats.mstats.gmean(abs(xm))
            assert_allclose(r, rm, rtol=1e-13)

            r = stats.gmean(abs(y))
            rm = stats.mstats.gmean(abs(ym))
            assert_allclose(r, rm, rtol=1e-13)

    def test_hmean(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)

            r = stats.hmean(abs(x))
            rm = stats.mstats.hmean(abs(xm))
            assert_almost_equal(r, rm, 10)

            r = stats.hmean(abs(y))
            rm = stats.mstats.hmean(abs(ym))
            assert_almost_equal(r, rm, 10)

    def test_skew(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)

            r = stats.skew(x)
            rm = stats.mstats.skew(xm)
            assert_almost_equal(r, rm, 10)

            r = stats.skew(y)
            rm = stats.mstats.skew(ym)
            assert_almost_equal(r, rm, 10)

    def test_moment(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)

            r = stats.moment(x)
            rm = stats.mstats.moment(xm)
            assert_almost_equal(r, rm, 10)

            r = stats.moment(y)
            rm = stats.mstats.moment(ym)
            assert_almost_equal(r, rm, 10)

    def test_zscore(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)

            # reference solution
            zx = (x - x.mean()) / x.std()
            zy = (y - y.mean()) / y.std()

            # validate stats
            assert_allclose(stats.zscore(x), zx, rtol=1e-10)
            assert_allclose(stats.zscore(y), zy, rtol=1e-10)

            # compare stats and mstats
            assert_allclose(stats.zscore(x), stats.mstats.zscore(xm[0:len(x)]),
                            rtol=1e-10)
            assert_allclose(stats.zscore(y), stats.mstats.zscore(ym[0:len(y)]),
                            rtol=1e-10)

    def test_kurtosis(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            r = stats.kurtosis(x)
            rm = stats.mstats.kurtosis(xm)
            assert_almost_equal(r, rm, 10)

            r = stats.kurtosis(y)
            rm = stats.mstats.kurtosis(ym)
            assert_almost_equal(r, rm, 10)

    def test_sem(self):
        # example from stats.sem doc
        a = np.arange(20).reshape(5, 4)
        am = np.ma.array(a)
        r = stats.sem(a, ddof=1)
        rm = stats.mstats.sem(am, ddof=1)

        assert_allclose(r, 2.82842712, atol=1e-5)
        assert_allclose(rm, 2.82842712, atol=1e-5)

        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            assert_almost_equal(stats.mstats.sem(xm, axis=None, ddof=0),
                                stats.sem(x, axis=None, ddof=0), decimal=13)
            assert_almost_equal(stats.mstats.sem(ym, axis=None, ddof=0),
                                stats.sem(y, axis=None, ddof=0), decimal=13)
            assert_almost_equal(stats.mstats.sem(xm, axis=None, ddof=1),
                                stats.sem(x, axis=None, ddof=1), decimal=13)
            assert_almost_equal(stats.mstats.sem(ym, axis=None, ddof=1),
                                stats.sem(y, axis=None, ddof=1), decimal=13)

    def test_describe(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            r = stats.describe(x, ddof=1)
            rm = stats.mstats.describe(xm, ddof=1)
            for ii in range(6):
                assert_almost_equal(np.asarray(r[ii]),
                                    np.asarray(rm[ii]),
                                    decimal=12)

    def test_describe_result_attributes(self):
        actual = mstats.describe(np.arange(5))
        attributes = ('nobs', 'minmax', 'mean', 'variance', 'skewness',
                      'kurtosis')
        check_named_results(actual, attributes, ma=True)

    def test_rankdata(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            r = stats.rankdata(x)
            rm = stats.mstats.rankdata(x)
            assert_allclose(r, rm)

    def test_tmean(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            assert_almost_equal(stats.tmean(x),stats.mstats.tmean(xm), 14)
            assert_almost_equal(stats.tmean(y),stats.mstats.tmean(ym), 14)

    def test_tmax(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            assert_almost_equal(stats.tmax(x,2.),
                                stats.mstats.tmax(xm,2.), 10)
            assert_almost_equal(stats.tmax(y,2.),
                                stats.mstats.tmax(ym,2.), 10)

            assert_almost_equal(stats.tmax(x, upperlimit=3.),
                                stats.mstats.tmax(xm, upperlimit=3.), 10)
            assert_almost_equal(stats.tmax(y, upperlimit=3.),
                                stats.mstats.tmax(ym, upperlimit=3.), 10)

    def test_tmin(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            assert_equal(stats.tmin(x), stats.mstats.tmin(xm))
            assert_equal(stats.tmin(y), stats.mstats.tmin(ym))

            assert_almost_equal(stats.tmin(x, lowerlimit=-1.),
                                stats.mstats.tmin(xm, lowerlimit=-1.), 10)
            assert_almost_equal(stats.tmin(y, lowerlimit=-1.),
                                stats.mstats.tmin(ym, lowerlimit=-1.), 10)

    def test_zmap(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            z = stats.zmap(x, y)
            zm = stats.mstats.zmap(xm, ym)
            assert_allclose(z, zm[0:len(z)], atol=1e-10)

    def test_variation(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            assert_almost_equal(stats.variation(x), stats.mstats.variation(xm),
                                decimal=12)
            assert_almost_equal(stats.variation(y), stats.mstats.variation(ym),
                                decimal=12)

    def test_tvar(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            assert_almost_equal(stats.tvar(x), stats.mstats.tvar(xm),
                                decimal=12)
            assert_almost_equal(stats.tvar(y), stats.mstats.tvar(ym),
                                decimal=12)

    def test_trimboth(self):
        a = np.arange(20)
        b = stats.trimboth(a, 0.1)
        bm = stats.mstats.trimboth(a, 0.1)
        assert_allclose(np.sort(b), bm.data[~bm.mask])

    def test_tsem(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            assert_almost_equal(stats.tsem(x), stats.mstats.tsem(xm),
                                decimal=14)
            assert_almost_equal(stats.tsem(y), stats.mstats.tsem(ym),
                                decimal=14)
            assert_almost_equal(stats.tsem(x, limits=(-2., 2.)),
                                stats.mstats.tsem(xm, limits=(-2., 2.)),
                                decimal=14)

    def test_skewtest(self):
        # this test is for 1D data
        for n in self.get_n():
            if n > 8:
                x, y, xm, ym = self.generate_xy_sample(n)
                r = stats.skewtest(x)
                rm = stats.mstats.skewtest(xm)
                assert_allclose(r, rm)

    def test_skewtest_result_attributes(self):
        x = np.array((-2, -1, 0, 1, 2, 3)*4)**2
        res = mstats.skewtest(x)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes, ma=True)

    def test_skewtest_2D_notmasked(self):
        # a normal ndarray is passed to the masked function
        x = np.random.random((20, 2)) * 20.
        r = stats.skewtest(x)
        rm = stats.mstats.skewtest(x)
        assert_allclose(np.asarray(r), np.asarray(rm))

    def test_skewtest_2D_WithMask(self):
        nx = 2
        for n in self.get_n():
            if n > 8:
                x, y, xm, ym = self.generate_xy_sample2D(n, nx)
                r = stats.skewtest(x)
                rm = stats.mstats.skewtest(xm)

                assert_equal(r[0][0], rm[0][0])
                assert_equal(r[0][1], rm[0][1])

    def test_normaltest(self):
        with np.errstate(over='raise'), suppress_warnings() as sup:
            sup.filter(UserWarning, "kurtosistest only valid for n>=20")
            for n in self.get_n():
                if n > 8:
                    x, y, xm, ym = self.generate_xy_sample(n)
                    r = stats.normaltest(x)
                    rm = stats.mstats.normaltest(xm)
                    assert_allclose(np.asarray(r), np.asarray(rm))

    def test_find_repeats(self):
        x = np.asarray([1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4]).astype('float')
        tmp = np.asarray([1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]).astype('float')
        mask = (tmp == 5.)
        xm = np.ma.array(tmp, mask=mask)
        x_orig, xm_orig = x.copy(), xm.copy()

        r = stats.find_repeats(x)
        rm = stats.mstats.find_repeats(xm)

        assert_equal(r, rm)
        assert_equal(x, x_orig)
        assert_equal(xm, xm_orig)

        # This crazy behavior is expected by count_tied_groups, but is not
        # in the docstring...
        _, counts = stats.mstats.find_repeats([])
        assert_equal(counts, np.array(0, dtype=np.intp))

    def test_kendalltau(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            r = stats.kendalltau(x, y)
            rm = stats.mstats.kendalltau(xm, ym)
            assert_almost_equal(r[0], rm[0], decimal=10)
            assert_almost_equal(r[1], rm[1], decimal=7)

    def test_obrientransform(self):
        for n in self.get_n():
            x, y, xm, ym = self.generate_xy_sample(n)
            r = stats.obrientransform(x)
            rm = stats.mstats.obrientransform(xm)
            assert_almost_equal(r.T, rm[0:len(x)])

    def test_ks_1samp(self):
        """Checks that mstats.ks_1samp and stats.ks_1samp agree on masked arrays."""
        for mode in ['auto', 'exact', 'asymp']:
            with suppress_warnings() as sup:
                for alternative in ['less', 'greater', 'two-sided']:
                    for n in self.get_n():
                        x, y, xm, ym = self.generate_xy_sample(n)
                        res1 = stats.ks_1samp(x, stats.norm.cdf, alternative=alternative, mode=mode)
                        res2 = stats.mstats.ks_1samp(xm, stats.norm.cdf, alternative=alternative, mode=mode)
                        assert_equal(np.asarray(res1), np.asarray(res2))
                        res3 = stats.ks_1samp(xm, stats.norm.cdf, alternative=alternative, mode=mode)
                        assert_equal(np.asarray(res1), np.asarray(res3))

    def test_kstest_1samp(self):
        """Checks that 1-sample mstats.kstest and stats.kstest agree on masked arrays."""
        for mode in ['auto', 'exact', 'asymp']:
            with suppress_warnings() as sup:
                for alternative in ['less', 'greater', 'two-sided']:
                    for n in self.get_n():
                        x, y, xm, ym = self.generate_xy_sample(n)
                        res1 = stats.kstest(x, 'norm', alternative=alternative, mode=mode)
                        res2 = stats.mstats.kstest(xm, 'norm', alternative=alternative, mode=mode)
                        assert_equal(np.asarray(res1), np.asarray(res2))
                        res3 = stats.kstest(xm, 'norm', alternative=alternative, mode=mode)
                        assert_equal(np.asarray(res1), np.asarray(res3))

    def test_ks_2samp(self):
        """Checks that mstats.ks_2samp and stats.ks_2samp agree on masked arrays.
        gh-8431"""
        for mode in ['auto', 'exact', 'asymp']:
            with suppress_warnings() as sup:
                if mode in ['auto', 'exact']:
                    sup.filter(RuntimeWarning,
                               "ks_2samp: Exact calculation unsuccessful. Switching to mode=asymp.")
                for alternative in ['less', 'greater', 'two-sided']:
                    for n in self.get_n():
                        x, y, xm, ym = self.generate_xy_sample(n)
                        res1 = stats.ks_2samp(x, y, alternative=alternative, mode=mode)
                        res2 = stats.mstats.ks_2samp(xm, ym, alternative=alternative, mode=mode)
                        assert_equal(np.asarray(res1), np.asarray(res2))
                        res3 = stats.ks_2samp(xm, y, alternative=alternative, mode=mode)
                        assert_equal(np.asarray(res1), np.asarray(res3))

    def test_kstest_2samp(self):
        """Checks that 2-sample mstats.kstest and stats.kstest agree on masked arrays."""
        for mode in ['auto', 'exact', 'asymp']:
            with suppress_warnings() as sup:
                if mode in ['auto', 'exact']:
                    sup.filter(RuntimeWarning,
                               "ks_2samp: Exact calculation unsuccessful. Switching to mode=asymp.")
                for alternative in ['less', 'greater', 'two-sided']:
                    for n in self.get_n():
                        x, y, xm, ym = self.generate_xy_sample(n)
                        res1 = stats.kstest(x, y, alternative=alternative, mode=mode)
                        res2 = stats.mstats.kstest(xm, ym, alternative=alternative, mode=mode)
                        assert_equal(np.asarray(res1), np.asarray(res2))
                        res3 = stats.kstest(xm, y, alternative=alternative, mode=mode)
                        assert_equal(np.asarray(res1), np.asarray(res3))

    def test_nametuples_agree(self):
        result = stats.kstest([1, 2], [3, 4])
        assert_(isinstance(result, stats.stats.KstestResult))
        result2 = stats.stats.Ks_2sampResult(result.statistic, result.pvalue)
        assert_(isinstance(result2, stats.stats.Ks_2sampResult))
        assert_equal(result, result2)


class TestBrunnerMunzel:
    # Data from (Lumley, 1996)
    X = np.ma.masked_invalid([1, 2, 1, 1, 1, np.nan, 1, 1,
                              1, 1, 1, 2, 4, 1, 1, np.nan])
    Y = np.ma.masked_invalid([3, 3, 4, 3, np.nan, 1, 2, 3, 1, 1, 5, 4])
    significant = 14

    def test_brunnermunzel_one_sided(self):
        # Results are compared with R's lawstat package.
        u1, p1 = mstats.brunnermunzel(self.X, self.Y, alternative='less')
        u2, p2 = mstats.brunnermunzel(self.Y, self.X, alternative='greater')
        u3, p3 = mstats.brunnermunzel(self.X, self.Y, alternative='greater')
        u4, p4 = mstats.brunnermunzel(self.Y, self.X, alternative='less')

        assert_almost_equal(p1, p2, decimal=self.significant)
        assert_almost_equal(p3, p4, decimal=self.significant)
        assert_(p1 != p3)
        assert_almost_equal(u1, 3.1374674823029505,
                            decimal=self.significant)
        assert_almost_equal(u2, -3.1374674823029505,
                            decimal=self.significant)
        assert_almost_equal(u3, 3.1374674823029505,
                            decimal=self.significant)
        assert_almost_equal(u4, -3.1374674823029505,
                            decimal=self.significant)
        assert_almost_equal(p1, 0.0028931043330757342,
                            decimal=self.significant)
        assert_almost_equal(p3, 0.99710689566692423,
                            decimal=self.significant)

    def test_brunnermunzel_two_sided(self):
        # Results are compared with R's lawstat package.
        u1, p1 = mstats.brunnermunzel(self.X, self.Y, alternative='two-sided')
        u2, p2 = mstats.brunnermunzel(self.Y, self.X, alternative='two-sided')

        assert_almost_equal(p1, p2, decimal=self.significant)
        assert_almost_equal(u1, 3.1374674823029505,
                            decimal=self.significant)
        assert_almost_equal(u2, -3.1374674823029505,
                            decimal=self.significant)
        assert_almost_equal(p1, 0.0057862086661515377,
                            decimal=self.significant)

    def test_brunnermunzel_default(self):
        # The default value for alternative is two-sided
        u1, p1 = mstats.brunnermunzel(self.X, self.Y)
        u2, p2 = mstats.brunnermunzel(self.Y, self.X)

        assert_almost_equal(p1, p2, decimal=self.significant)
        assert_almost_equal(u1, 3.1374674823029505,
                            decimal=self.significant)
        assert_almost_equal(u2, -3.1374674823029505,
                            decimal=self.significant)
        assert_almost_equal(p1, 0.0057862086661515377,
                            decimal=self.significant)

    def test_brunnermunzel_alternative_error(self):
        alternative = "error"
        distribution = "t"
        assert_(alternative not in ["two-sided", "greater", "less"])
        assert_raises(ValueError,
                      mstats.brunnermunzel,
                      self.X,
                      self.Y,
                      alternative,
                      distribution)

    def test_brunnermunzel_distribution_norm(self):
        u1, p1 = mstats.brunnermunzel(self.X, self.Y, distribution="normal")
        u2, p2 = mstats.brunnermunzel(self.Y, self.X, distribution="normal")
        assert_almost_equal(p1, p2, decimal=self.significant)
        assert_almost_equal(u1, 3.1374674823029505,
                            decimal=self.significant)
        assert_almost_equal(u2, -3.1374674823029505,
                            decimal=self.significant)
        assert_almost_equal(p1, 0.0017041417600383024,
                            decimal=self.significant)

    def test_brunnermunzel_distribution_error(self):
        alternative = "two-sided"
        distribution = "error"
        assert_(alternative not in ["t", "normal"])
        assert_raises(ValueError,
                      mstats.brunnermunzel,
                      self.X,
                      self.Y,
                      alternative,
                      distribution)

    def test_brunnermunzel_empty_imput(self):
        u1, p1 = mstats.brunnermunzel(self.X, [])
        u2, p2 = mstats.brunnermunzel([], self.Y)
        u3, p3 = mstats.brunnermunzel([], [])

        assert_(np.isnan(u1))
        assert_(np.isnan(p1))
        assert_(np.isnan(u2))
        assert_(np.isnan(p2))
        assert_(np.isnan(u3))
        assert_(np.isnan(p3))
