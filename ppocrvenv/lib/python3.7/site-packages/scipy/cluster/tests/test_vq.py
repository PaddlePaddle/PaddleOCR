
import warnings
import sys

import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_allclose, assert_equal, assert_,
                           suppress_warnings)
import pytest
from pytest import raises as assert_raises

from scipy.cluster.vq import (kmeans, kmeans2, py_vq, vq, whiten,
                              ClusterError, _krandinit)
from scipy.cluster import _vq
from scipy.sparse.sputils import matrix


TESTDATA_2D = np.array([
    -2.2, 1.17, -1.63, 1.69, -2.04, 4.38, -3.09, 0.95, -1.7, 4.79, -1.68, 0.68,
    -2.26, 3.34, -2.29, 2.55, -1.72, -0.72, -1.99, 2.34, -2.75, 3.43, -2.45,
    2.41, -4.26, 3.65, -1.57, 1.87, -1.96, 4.03, -3.01, 3.86, -2.53, 1.28,
    -4.0, 3.95, -1.62, 1.25, -3.42, 3.17, -1.17, 0.12, -3.03, -0.27, -2.07,
    -0.55, -1.17, 1.34, -2.82, 3.08, -2.44, 0.24, -1.71, 2.48, -5.23, 4.29,
    -2.08, 3.69, -1.89, 3.62, -2.09, 0.26, -0.92, 1.07, -2.25, 0.88, -2.25,
    2.02, -4.31, 3.86, -2.03, 3.42, -2.76, 0.3, -2.48, -0.29, -3.42, 3.21,
    -2.3, 1.73, -2.84, 0.69, -1.81, 2.48, -5.24, 4.52, -2.8, 1.31, -1.67,
    -2.34, -1.18, 2.17, -2.17, 2.82, -1.85, 2.25, -2.45, 1.86, -6.79, 3.94,
    -2.33, 1.89, -1.55, 2.08, -1.36, 0.93, -2.51, 2.74, -2.39, 3.92, -3.33,
    2.99, -2.06, -0.9, -2.83, 3.35, -2.59, 3.05, -2.36, 1.85, -1.69, 1.8,
    -1.39, 0.66, -2.06, 0.38, -1.47, 0.44, -4.68, 3.77, -5.58, 3.44, -2.29,
    2.24, -1.04, -0.38, -1.85, 4.23, -2.88, 0.73, -2.59, 1.39, -1.34, 1.75,
    -1.95, 1.3, -2.45, 3.09, -1.99, 3.41, -5.55, 5.21, -1.73, 2.52, -2.17,
    0.85, -2.06, 0.49, -2.54, 2.07, -2.03, 1.3, -3.23, 3.09, -1.55, 1.44,
    -0.81, 1.1, -2.99, 2.92, -1.59, 2.18, -2.45, -0.73, -3.12, -1.3, -2.83,
    0.2, -2.77, 3.24, -1.98, 1.6, -4.59, 3.39, -4.85, 3.75, -2.25, 1.71, -3.28,
    3.38, -1.74, 0.88, -2.41, 1.92, -2.24, 1.19, -2.48, 1.06, -1.68, -0.62,
    -1.3, 0.39, -1.78, 2.35, -3.54, 2.44, -1.32, 0.66, -2.38, 2.76, -2.35,
    3.95, -1.86, 4.32, -2.01, -1.23, -1.79, 2.76, -2.13, -0.13, -5.25, 3.84,
    -2.24, 1.59, -4.85, 2.96, -2.41, 0.01, -0.43, 0.13, -3.92, 2.91, -1.75,
    -0.53, -1.69, 1.69, -1.09, 0.15, -2.11, 2.17, -1.53, 1.22, -2.1, -0.86,
    -2.56, 2.28, -3.02, 3.33, -1.12, 3.86, -2.18, -1.19, -3.03, 0.79, -0.83,
    0.97, -3.19, 1.45, -1.34, 1.28, -2.52, 4.22, -4.53, 3.22, -1.97, 1.75,
    -2.36, 3.19, -0.83, 1.53, -1.59, 1.86, -2.17, 2.3, -1.63, 2.71, -2.03,
    3.75, -2.57, -0.6, -1.47, 1.33, -1.95, 0.7, -1.65, 1.27, -1.42, 1.09, -3.0,
    3.87, -2.51, 3.06, -2.6, 0.74, -1.08, -0.03, -2.44, 1.31, -2.65, 2.99,
    -1.84, 1.65, -4.76, 3.75, -2.07, 3.98, -2.4, 2.67, -2.21, 1.49, -1.21,
    1.22, -5.29, 2.38, -2.85, 2.28, -5.6, 3.78, -2.7, 0.8, -1.81, 3.5, -3.75,
    4.17, -1.29, 2.99, -5.92, 3.43, -1.83, 1.23, -1.24, -1.04, -2.56, 2.37,
    -3.26, 0.39, -4.63, 2.51, -4.52, 3.04, -1.7, 0.36, -1.41, 0.04, -2.1, 1.0,
    -1.87, 3.78, -4.32, 3.59, -2.24, 1.38, -1.99, -0.22, -1.87, 1.95, -0.84,
    2.17, -5.38, 3.56, -1.27, 2.9, -1.79, 3.31, -5.47, 3.85, -1.44, 3.69,
    -2.02, 0.37, -1.29, 0.33, -2.34, 2.56, -1.74, -1.27, -1.97, 1.22, -2.51,
    -0.16, -1.64, -0.96, -2.99, 1.4, -1.53, 3.31, -2.24, 0.45, -2.46, 1.71,
    -2.88, 1.56, -1.63, 1.46, -1.41, 0.68, -1.96, 2.76, -1.61,
    2.11]).reshape((200, 2))


# Global data
X = np.array([[3.0, 3], [4, 3], [4, 2],
              [9, 2], [5, 1], [6, 2], [9, 4],
              [5, 2], [5, 4], [7, 4], [6, 5]])

CODET1 = np.array([[3.0000, 3.0000],
                   [6.2000, 4.0000],
                   [5.8000, 1.8000]])

CODET2 = np.array([[11.0/3, 8.0/3],
                   [6.7500, 4.2500],
                   [6.2500, 1.7500]])

LABEL1 = np.array([0, 1, 2, 2, 2, 2, 1, 2, 1, 1, 1])


class TestWhiten:
    def test_whiten(self):
        desired = np.array([[5.08738849, 2.97091878],
                            [3.19909255, 0.69660580],
                            [4.51041982, 0.02640918],
                            [4.38567074, 0.95120889],
                            [2.32191480, 1.63195503]])
        for tp in np.array, matrix:
            obs = tp([[0.98744510, 0.82766775],
                      [0.62093317, 0.19406729],
                      [0.87545741, 0.00735733],
                      [0.85124403, 0.26499712],
                      [0.45067590, 0.45464607]])
            assert_allclose(whiten(obs), desired, rtol=1e-5)

    def test_whiten_zero_std(self):
        desired = np.array([[0., 1.0, 2.86666544],
                            [0., 1.0, 1.32460034],
                            [0., 1.0, 3.74382172]])
        for tp in np.array, matrix:
            obs = tp([[0., 1., 0.74109533],
                      [0., 1., 0.34243798],
                      [0., 1., 0.96785929]])
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                assert_allclose(whiten(obs), desired, rtol=1e-5)
                assert_equal(len(w), 1)
                assert_(issubclass(w[-1].category, RuntimeWarning))

    def test_whiten_not_finite(self):
        for tp in np.array, matrix:
            for bad_value in np.nan, np.inf, -np.inf:
                obs = tp([[0.98744510, bad_value],
                          [0.62093317, 0.19406729],
                          [0.87545741, 0.00735733],
                          [0.85124403, 0.26499712],
                          [0.45067590, 0.45464607]])
                assert_raises(ValueError, whiten, obs)


class TestVq:
    def test_py_vq(self):
        initc = np.concatenate(([[X[0]], [X[1]], [X[2]]]))
        for tp in np.array, matrix:
            label1 = py_vq(tp(X), tp(initc))[0]
            assert_array_equal(label1, LABEL1)

    def test_vq(self):
        initc = np.concatenate(([[X[0]], [X[1]], [X[2]]]))
        for tp in np.array, matrix:
            label1, dist = _vq.vq(tp(X), tp(initc))
            assert_array_equal(label1, LABEL1)
            tlabel1, tdist = vq(tp(X), tp(initc))

    def test_vq_1d(self):
        # Test special rank 1 vq algo, python implementation.
        data = X[:, 0]
        initc = data[:3]
        a, b = _vq.vq(data, initc)
        ta, tb = py_vq(data[:, np.newaxis], initc[:, np.newaxis])
        assert_array_equal(a, ta)
        assert_array_equal(b, tb)

    def test__vq_sametype(self):
        a = np.array([1.0, 2.0], dtype=np.float64)
        b = a.astype(np.float32)
        assert_raises(TypeError, _vq.vq, a, b)

    def test__vq_invalid_type(self):
        a = np.array([1, 2], dtype=int)
        assert_raises(TypeError, _vq.vq, a, a)

    def test_vq_large_nfeat(self):
        X = np.random.rand(20, 20)
        code_book = np.random.rand(3, 20)

        codes0, dis0 = _vq.vq(X, code_book)
        codes1, dis1 = py_vq(X, code_book)
        assert_allclose(dis0, dis1, 1e-5)
        assert_array_equal(codes0, codes1)

        X = X.astype(np.float32)
        code_book = code_book.astype(np.float32)

        codes0, dis0 = _vq.vq(X, code_book)
        codes1, dis1 = py_vq(X, code_book)
        assert_allclose(dis0, dis1, 1e-5)
        assert_array_equal(codes0, codes1)

    def test_vq_large_features(self):
        X = np.random.rand(10, 5) * 1000000
        code_book = np.random.rand(2, 5) * 1000000

        codes0, dis0 = _vq.vq(X, code_book)
        codes1, dis1 = py_vq(X, code_book)
        assert_allclose(dis0, dis1, 1e-5)
        assert_array_equal(codes0, codes1)


class TestKMean:
    def test_large_features(self):
        # Generate a data set with large values, and run kmeans on it to
        # (regression for 1077).
        d = 300
        n = 100

        m1 = np.random.randn(d)
        m2 = np.random.randn(d)
        x = 10000 * np.random.randn(n, d) - 20000 * m1
        y = 10000 * np.random.randn(n, d) + 20000 * m2

        data = np.empty((x.shape[0] + y.shape[0], d), np.double)
        data[:x.shape[0]] = x
        data[x.shape[0]:] = y

        kmeans(data, 2)

    def test_kmeans_simple(self):
        np.random.seed(54321)
        initc = np.concatenate(([[X[0]], [X[1]], [X[2]]]))
        for tp in np.array, matrix:
            code1 = kmeans(tp(X), tp(initc), iter=1)[0]
            assert_array_almost_equal(code1, CODET2)

    def test_kmeans_lost_cluster(self):
        # This will cause kmeans to have a cluster with no points.
        data = TESTDATA_2D
        initk = np.array([[-1.8127404, -0.67128041],
                         [2.04621601, 0.07401111],
                         [-2.31149087, -0.05160469]])

        kmeans(data, initk)
        with suppress_warnings() as sup:
            sup.filter(UserWarning,
                       "One of the clusters is empty. Re-run kmeans with a "
                       "different initialization")
            kmeans2(data, initk, missing='warn')

        assert_raises(ClusterError, kmeans2, data, initk, missing='raise')

    def test_kmeans2_simple(self):
        np.random.seed(12345678)
        initc = np.concatenate(([[X[0]], [X[1]], [X[2]]]))
        for tp in np.array, matrix:
            code1 = kmeans2(tp(X), tp(initc), iter=1)[0]
            code2 = kmeans2(tp(X), tp(initc), iter=2)[0]

            assert_array_almost_equal(code1, CODET1)
            assert_array_almost_equal(code2, CODET2)

    def test_kmeans2_rank1(self):
        data = TESTDATA_2D
        data1 = data[:, 0]

        initc = data1[:3]
        code = initc.copy()
        kmeans2(data1, code, iter=1)[0]
        kmeans2(data1, code, iter=2)[0]

    def test_kmeans2_rank1_2(self):
        data = TESTDATA_2D
        data1 = data[:, 0]
        kmeans2(data1, 2, iter=1)

    def test_kmeans2_high_dim(self):
        # test kmeans2 when the number of dimensions exceeds the number
        # of input points
        data = TESTDATA_2D
        data = data.reshape((20, 20))[:10]
        kmeans2(data, 2)

    def test_kmeans2_init(self):
        np.random.seed(12345)
        data = TESTDATA_2D

        kmeans2(data, 3, minit='points')
        kmeans2(data[:, :1], 3, minit='points')  # special case (1-D)

        kmeans2(data, 3, minit='++')
        kmeans2(data[:, :1], 3, minit='++')  # special case (1-D)

        # minit='random' can give warnings, filter those
        with suppress_warnings() as sup:
            sup.filter(message="One of the clusters is empty. Re-run.")
            kmeans2(data, 3, minit='random')
            kmeans2(data[:, :1], 3, minit='random')  # special case (1-D)

    @pytest.mark.skipif(sys.platform == 'win32',
                        reason='Fails with MemoryError in Wine.')
    def test_krandinit(self):
        data = TESTDATA_2D
        datas = [data.reshape((200, 2)), data.reshape((20, 20))[:10]]
        k = int(1e6)
        for data in datas:
            # check that np.random.Generator can be used (numpy >= 1.17)
            if hasattr(np.random, 'default_rng'):
                rng = np.random.default_rng(1234)
            else:
                rng = np.random.RandomState(1234)

            init = _krandinit(data, k, rng)
            orig_cov = np.cov(data, rowvar=0)
            init_cov = np.cov(init, rowvar=0)
            assert_allclose(orig_cov, init_cov, atol=1e-2)

    def test_kmeans2_empty(self):
        # Regression test for gh-1032.
        assert_raises(ValueError, kmeans2, [], 2)

    def test_kmeans_0k(self):
        # Regression test for gh-1073: fail when k arg is 0.
        assert_raises(ValueError, kmeans, X, 0)
        assert_raises(ValueError, kmeans2, X, 0)
        assert_raises(ValueError, kmeans2, X, np.array([]))

    def test_kmeans_large_thres(self):
        # Regression test for gh-1774
        x = np.array([1, 2, 3, 4, 10], dtype=float)
        res = kmeans(x, 1, thresh=1e16)
        assert_allclose(res[0], np.array([4.]))
        assert_allclose(res[1], 2.3999999999999999)

    def test_kmeans2_kpp_low_dim(self):
        # Regression test for gh-11462
        prev_res = np.array([[-1.95266667, 0.898],
                             [-3.153375, 3.3945]])
        np.random.seed(42)
        res, _ = kmeans2(TESTDATA_2D, 2, minit='++')
        assert_allclose(res, prev_res)

    def test_kmeans2_kpp_high_dim(self):
        # Regression test for gh-11462
        n_dim = 100
        size = 10
        centers = np.vstack([5 * np.ones(n_dim),
                             -5 * np.ones(n_dim)])
        np.random.seed(42)
        data = np.vstack([
            np.random.multivariate_normal(centers[0], np.eye(n_dim), size=size),
            np.random.multivariate_normal(centers[1], np.eye(n_dim), size=size)
        ])
        res, _ = kmeans2(data, 2, minit='++')
        assert_array_almost_equal(res, centers, decimal=0)

    def test_kmeans_and_kmeans2_random_seed(self):

        seed_list = [1234, np.random.RandomState(1234)]

        # check that np.random.Generator can be used (numpy >= 1.17)
        if hasattr(np.random, 'default_rng'):
            seed_list.append(np.random.default_rng(1234))

        for seed in seed_list:
            # test for kmeans
            res1, _ = kmeans(TESTDATA_2D, 2, seed=seed)
            res2, _ = kmeans(TESTDATA_2D, 2, seed=seed)
            assert_allclose(res1, res1)  # should be same results

            # test for kmeans2
            for minit in ["random", "points", "++"]:
                res1, _ = kmeans2(TESTDATA_2D, 2, minit=minit, seed=seed)
                res2, _ = kmeans2(TESTDATA_2D, 2, minit=minit, seed=seed)
                assert_allclose(res1, res1)  # should be same results
