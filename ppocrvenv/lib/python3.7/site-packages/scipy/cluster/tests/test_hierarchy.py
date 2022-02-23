#
# Author: Damian Eads
# Date: April 17, 2008
#
# Copyright (C) 2008 Damian Eads
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_, assert_warns
import pytest
from pytest import raises as assert_raises

import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import (
    ClusterWarning, linkage, from_mlab_linkage, to_mlab_linkage,
    num_obs_linkage, inconsistent, cophenet, fclusterdata, fcluster,
    is_isomorphic, single, leaders,
    correspond, is_monotonic, maxdists, maxinconsts, maxRstat,
    is_valid_linkage, is_valid_im, to_tree, leaves_list, dendrogram,
    set_link_color_palette, cut_tree, optimal_leaf_ordering,
    _order_cluster_tree, _hierarchy, _LINKAGE_METHODS)
from scipy.spatial.distance import pdist
from scipy.cluster._hierarchy import Heap

from . import hierarchy_test_data


# Matplotlib is not a scipy dependency but is optionally used in dendrogram, so
# check if it's available
try:
    import matplotlib  # type: ignore[import]
    # and set the backend to be Agg (no gui)
    matplotlib.use('Agg')
    # before importing pyplot
    import matplotlib.pyplot as plt  # type: ignore[import]
    have_matplotlib = True
except Exception:
    have_matplotlib = False


class TestLinkage:
    def test_linkage_non_finite_elements_in_distance_matrix(self):
        # Tests linkage(Y) where Y contains a non-finite element (e.g. NaN or Inf).
        # Exception expected.
        y = np.zeros((6,))
        y[0] = np.nan
        assert_raises(ValueError, linkage, y)

    def test_linkage_empty_distance_matrix(self):
        # Tests linkage(Y) where Y is a 0x4 linkage matrix. Exception expected.
        y = np.zeros((0,))
        assert_raises(ValueError, linkage, y)

    def test_linkage_tdist(self):
        for method in ['single', 'complete', 'average', 'weighted']:
            self.check_linkage_tdist(method)

    def check_linkage_tdist(self, method):
        # Tests linkage(Y, method) on the tdist data set.
        Z = linkage(hierarchy_test_data.ytdist, method)
        expectedZ = getattr(hierarchy_test_data, 'linkage_ytdist_' + method)
        assert_allclose(Z, expectedZ, atol=1e-10)

    def test_linkage_X(self):
        for method in ['centroid', 'median', 'ward']:
            self.check_linkage_q(method)

    def check_linkage_q(self, method):
        # Tests linkage(Y, method) on the Q data set.
        Z = linkage(hierarchy_test_data.X, method)
        expectedZ = getattr(hierarchy_test_data, 'linkage_X_' + method)
        assert_allclose(Z, expectedZ, atol=1e-06)

        y = scipy.spatial.distance.pdist(hierarchy_test_data.X,
                                         metric="euclidean")
        Z = linkage(y, method)
        assert_allclose(Z, expectedZ, atol=1e-06)

    def test_compare_with_trivial(self):
        rng = np.random.RandomState(0)
        n = 20
        X = rng.rand(n, 2)
        d = pdist(X)

        for method, code in _LINKAGE_METHODS.items():
            Z_trivial = _hierarchy.linkage(d, n, code)
            Z = linkage(d, method)
            assert_allclose(Z_trivial, Z, rtol=1e-14, atol=1e-15)

    def test_optimal_leaf_ordering(self):
        Z = linkage(hierarchy_test_data.ytdist, optimal_ordering=True)
        expectedZ = getattr(hierarchy_test_data, 'linkage_ytdist_single_olo')
        assert_allclose(Z, expectedZ, atol=1e-10)


class TestLinkageTies:
    _expectations = {
        'single': np.array([[0, 1, 1.41421356, 2],
                            [2, 3, 1.41421356, 3]]),
        'complete': np.array([[0, 1, 1.41421356, 2],
                              [2, 3, 2.82842712, 3]]),
        'average': np.array([[0, 1, 1.41421356, 2],
                             [2, 3, 2.12132034, 3]]),
        'weighted': np.array([[0, 1, 1.41421356, 2],
                              [2, 3, 2.12132034, 3]]),
        'centroid': np.array([[0, 1, 1.41421356, 2],
                              [2, 3, 2.12132034, 3]]),
        'median': np.array([[0, 1, 1.41421356, 2],
                            [2, 3, 2.12132034, 3]]),
        'ward': np.array([[0, 1, 1.41421356, 2],
                          [2, 3, 2.44948974, 3]]),
    }

    def test_linkage_ties(self):
        for method in ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']:
            self.check_linkage_ties(method)

    def check_linkage_ties(self, method):
        X = np.array([[-1, -1], [0, 0], [1, 1]])
        Z = linkage(X, method=method)
        expectedZ = self._expectations[method]
        assert_allclose(Z, expectedZ, atol=1e-06)


class TestInconsistent:
    def test_inconsistent_tdist(self):
        for depth in hierarchy_test_data.inconsistent_ytdist:
            self.check_inconsistent_tdist(depth)

    def check_inconsistent_tdist(self, depth):
        Z = hierarchy_test_data.linkage_ytdist_single
        assert_allclose(inconsistent(Z, depth),
                        hierarchy_test_data.inconsistent_ytdist[depth])


class TestCopheneticDistance:
    def test_linkage_cophenet_tdist_Z(self):
        # Tests cophenet(Z) on tdist data set.
        expectedM = np.array([268, 295, 255, 255, 295, 295, 268, 268, 295, 295,
                              295, 138, 219, 295, 295])
        Z = hierarchy_test_data.linkage_ytdist_single
        M = cophenet(Z)
        assert_allclose(M, expectedM, atol=1e-10)

    def test_linkage_cophenet_tdist_Z_Y(self):
        # Tests cophenet(Z, Y) on tdist data set.
        Z = hierarchy_test_data.linkage_ytdist_single
        (c, M) = cophenet(Z, hierarchy_test_data.ytdist)
        expectedM = np.array([268, 295, 255, 255, 295, 295, 268, 268, 295, 295,
                              295, 138, 219, 295, 295])
        expectedc = 0.639931296433393415057366837573
        assert_allclose(c, expectedc, atol=1e-10)
        assert_allclose(M, expectedM, atol=1e-10)


class TestMLabLinkageConversion:
    def test_mlab_linkage_conversion_empty(self):
        # Tests from/to_mlab_linkage on empty linkage array.
        X = np.asarray([])
        assert_equal(from_mlab_linkage([]), X)
        assert_equal(to_mlab_linkage([]), X)

    def test_mlab_linkage_conversion_single_row(self):
        # Tests from/to_mlab_linkage on linkage array with single row.
        Z = np.asarray([[0., 1., 3., 2.]])
        Zm = [[1, 2, 3]]
        assert_equal(from_mlab_linkage(Zm), Z)
        assert_equal(to_mlab_linkage(Z), Zm)

    def test_mlab_linkage_conversion_multiple_rows(self):
        # Tests from/to_mlab_linkage on linkage array with multiple rows.
        Zm = np.asarray([[3, 6, 138], [4, 5, 219],
                         [1, 8, 255], [2, 9, 268], [7, 10, 295]])
        Z = np.array([[2., 5., 138., 2.],
                      [3., 4., 219., 2.],
                      [0., 7., 255., 3.],
                      [1., 8., 268., 4.],
                      [6., 9., 295., 6.]],
                      dtype=np.double)
        assert_equal(from_mlab_linkage(Zm), Z)
        assert_equal(to_mlab_linkage(Z), Zm)


class TestFcluster:
    def test_fclusterdata(self):
        for t in hierarchy_test_data.fcluster_inconsistent:
            self.check_fclusterdata(t, 'inconsistent')
        for t in hierarchy_test_data.fcluster_distance:
            self.check_fclusterdata(t, 'distance')
        for t in hierarchy_test_data.fcluster_maxclust:
            self.check_fclusterdata(t, 'maxclust')

    def check_fclusterdata(self, t, criterion):
        # Tests fclusterdata(X, criterion=criterion, t=t) on a random 3-cluster data set.
        expectedT = getattr(hierarchy_test_data, 'fcluster_' + criterion)[t]
        X = hierarchy_test_data.Q_X
        T = fclusterdata(X, criterion=criterion, t=t)
        assert_(is_isomorphic(T, expectedT))

    def test_fcluster(self):
        for t in hierarchy_test_data.fcluster_inconsistent:
            self.check_fcluster(t, 'inconsistent')
        for t in hierarchy_test_data.fcluster_distance:
            self.check_fcluster(t, 'distance')
        for t in hierarchy_test_data.fcluster_maxclust:
            self.check_fcluster(t, 'maxclust')

    def check_fcluster(self, t, criterion):
        # Tests fcluster(Z, criterion=criterion, t=t) on a random 3-cluster data set.
        expectedT = getattr(hierarchy_test_data, 'fcluster_' + criterion)[t]
        Z = single(hierarchy_test_data.Q_X)
        T = fcluster(Z, criterion=criterion, t=t)
        assert_(is_isomorphic(T, expectedT))

    def test_fcluster_monocrit(self):
        for t in hierarchy_test_data.fcluster_distance:
            self.check_fcluster_monocrit(t)
        for t in hierarchy_test_data.fcluster_maxclust:
            self.check_fcluster_maxclust_monocrit(t)

    def check_fcluster_monocrit(self, t):
        expectedT = hierarchy_test_data.fcluster_distance[t]
        Z = single(hierarchy_test_data.Q_X)
        T = fcluster(Z, t, criterion='monocrit', monocrit=maxdists(Z))
        assert_(is_isomorphic(T, expectedT))

    def check_fcluster_maxclust_monocrit(self, t):
        expectedT = hierarchy_test_data.fcluster_maxclust[t]
        Z = single(hierarchy_test_data.Q_X)
        T = fcluster(Z, t, criterion='maxclust_monocrit', monocrit=maxdists(Z))
        assert_(is_isomorphic(T, expectedT))


class TestLeaders:
    def test_leaders_single(self):
        # Tests leaders using a flat clustering generated by single linkage.
        X = hierarchy_test_data.Q_X
        Y = pdist(X)
        Z = linkage(Y)
        T = fcluster(Z, criterion='maxclust', t=3)
        Lright = (np.array([53, 55, 56]), np.array([2, 3, 1]))
        L = leaders(Z, T)
        assert_equal(L, Lright)


class TestIsIsomorphic:
    def test_is_isomorphic_1(self):
        # Tests is_isomorphic on test case #1 (one flat cluster, different labellings)
        a = [1, 1, 1]
        b = [2, 2, 2]
        assert_(is_isomorphic(a, b))
        assert_(is_isomorphic(b, a))

    def test_is_isomorphic_2(self):
        # Tests is_isomorphic on test case #2 (two flat clusters, different labelings)
        a = [1, 7, 1]
        b = [2, 3, 2]
        assert_(is_isomorphic(a, b))
        assert_(is_isomorphic(b, a))

    def test_is_isomorphic_3(self):
        # Tests is_isomorphic on test case #3 (no flat clusters)
        a = []
        b = []
        assert_(is_isomorphic(a, b))

    def test_is_isomorphic_4A(self):
        # Tests is_isomorphic on test case #4A (3 flat clusters, different labelings, isomorphic)
        a = [1, 2, 3]
        b = [1, 3, 2]
        assert_(is_isomorphic(a, b))
        assert_(is_isomorphic(b, a))

    def test_is_isomorphic_4B(self):
        # Tests is_isomorphic on test case #4B (3 flat clusters, different labelings, nonisomorphic)
        a = [1, 2, 3, 3]
        b = [1, 3, 2, 3]
        assert_(is_isomorphic(a, b) == False)
        assert_(is_isomorphic(b, a) == False)

    def test_is_isomorphic_4C(self):
        # Tests is_isomorphic on test case #4C (3 flat clusters, different labelings, isomorphic)
        a = [7, 2, 3]
        b = [6, 3, 2]
        assert_(is_isomorphic(a, b))
        assert_(is_isomorphic(b, a))

    def test_is_isomorphic_5(self):
        # Tests is_isomorphic on test case #5 (1000 observations, 2/3/5 random
        # clusters, random permutation of the labeling).
        for nc in [2, 3, 5]:
            self.help_is_isomorphic_randperm(1000, nc)

    def test_is_isomorphic_6(self):
        # Tests is_isomorphic on test case #5A (1000 observations, 2/3/5 random
        # clusters, random permutation of the labeling, slightly
        # nonisomorphic.)
        for nc in [2, 3, 5]:
            self.help_is_isomorphic_randperm(1000, nc, True, 5)

    def test_is_isomorphic_7(self):
        # Regression test for gh-6271
        assert_(not is_isomorphic([1, 2, 3], [1, 1, 1]))

    def help_is_isomorphic_randperm(self, nobs, nclusters, noniso=False, nerrors=0):
        for k in range(3):
            a = np.int_(np.random.rand(nobs) * nclusters)
            b = np.zeros(a.size, dtype=np.int_)
            P = np.random.permutation(nclusters)
            for i in range(0, a.shape[0]):
                b[i] = P[a[i]]
            if noniso:
                Q = np.random.permutation(nobs)
                b[Q[0:nerrors]] += 1
                b[Q[0:nerrors]] %= nclusters
            assert_(is_isomorphic(a, b) == (not noniso))
            assert_(is_isomorphic(b, a) == (not noniso))


class TestIsValidLinkage:
    def test_is_valid_linkage_various_size(self):
        for nrow, ncol, valid in [(2, 5, False), (2, 3, False),
                                  (1, 4, True), (2, 4, True)]:
            self.check_is_valid_linkage_various_size(nrow, ncol, valid)

    def check_is_valid_linkage_various_size(self, nrow, ncol, valid):
        # Tests is_valid_linkage(Z) with linkage matrics of various sizes
        Z = np.asarray([[0, 1, 3.0, 2, 5],
                        [3, 2, 4.0, 3, 3]], dtype=np.double)
        Z = Z[:nrow, :ncol]
        assert_(is_valid_linkage(Z) == valid)
        if not valid:
            assert_raises(ValueError, is_valid_linkage, Z, throw=True)

    def test_is_valid_linkage_int_type(self):
        # Tests is_valid_linkage(Z) with integer type.
        Z = np.asarray([[0, 1, 3.0, 2],
                        [3, 2, 4.0, 3]], dtype=int)
        assert_(is_valid_linkage(Z) == False)
        assert_raises(TypeError, is_valid_linkage, Z, throw=True)

    def test_is_valid_linkage_empty(self):
        # Tests is_valid_linkage(Z) with empty linkage.
        Z = np.zeros((0, 4), dtype=np.double)
        assert_(is_valid_linkage(Z) == False)
        assert_raises(ValueError, is_valid_linkage, Z, throw=True)

    def test_is_valid_linkage_4_and_up(self):
        # Tests is_valid_linkage(Z) on linkage on observation sets between
        # sizes 4 and 15 (step size 3).
        for i in range(4, 15, 3):
            y = np.random.rand(i*(i-1)//2)
            Z = linkage(y)
            assert_(is_valid_linkage(Z) == True)

    def test_is_valid_linkage_4_and_up_neg_index_left(self):
        # Tests is_valid_linkage(Z) on linkage on observation sets between
        # sizes 4 and 15 (step size 3) with negative indices (left).
        for i in range(4, 15, 3):
            y = np.random.rand(i*(i-1)//2)
            Z = linkage(y)
            Z[i//2,0] = -2
            assert_(is_valid_linkage(Z) == False)
            assert_raises(ValueError, is_valid_linkage, Z, throw=True)

    def test_is_valid_linkage_4_and_up_neg_index_right(self):
        # Tests is_valid_linkage(Z) on linkage on observation sets between
        # sizes 4 and 15 (step size 3) with negative indices (right).
        for i in range(4, 15, 3):
            y = np.random.rand(i*(i-1)//2)
            Z = linkage(y)
            Z[i//2,1] = -2
            assert_(is_valid_linkage(Z) == False)
            assert_raises(ValueError, is_valid_linkage, Z, throw=True)

    def test_is_valid_linkage_4_and_up_neg_dist(self):
        # Tests is_valid_linkage(Z) on linkage on observation sets between
        # sizes 4 and 15 (step size 3) with negative distances.
        for i in range(4, 15, 3):
            y = np.random.rand(i*(i-1)//2)
            Z = linkage(y)
            Z[i//2,2] = -0.5
            assert_(is_valid_linkage(Z) == False)
            assert_raises(ValueError, is_valid_linkage, Z, throw=True)

    def test_is_valid_linkage_4_and_up_neg_counts(self):
        # Tests is_valid_linkage(Z) on linkage on observation sets between
        # sizes 4 and 15 (step size 3) with negative counts.
        for i in range(4, 15, 3):
            y = np.random.rand(i*(i-1)//2)
            Z = linkage(y)
            Z[i//2,3] = -2
            assert_(is_valid_linkage(Z) == False)
            assert_raises(ValueError, is_valid_linkage, Z, throw=True)


class TestIsValidInconsistent:
    def test_is_valid_im_int_type(self):
        # Tests is_valid_im(R) with integer type.
        R = np.asarray([[0, 1, 3.0, 2],
                        [3, 2, 4.0, 3]], dtype=int)
        assert_(is_valid_im(R) == False)
        assert_raises(TypeError, is_valid_im, R, throw=True)

    def test_is_valid_im_various_size(self):
        for nrow, ncol, valid in [(2, 5, False), (2, 3, False),
                                  (1, 4, True), (2, 4, True)]:
            self.check_is_valid_im_various_size(nrow, ncol, valid)

    def check_is_valid_im_various_size(self, nrow, ncol, valid):
        # Tests is_valid_im(R) with linkage matrics of various sizes
        R = np.asarray([[0, 1, 3.0, 2, 5],
                        [3, 2, 4.0, 3, 3]], dtype=np.double)
        R = R[:nrow, :ncol]
        assert_(is_valid_im(R) == valid)
        if not valid:
            assert_raises(ValueError, is_valid_im, R, throw=True)

    def test_is_valid_im_empty(self):
        # Tests is_valid_im(R) with empty inconsistency matrix.
        R = np.zeros((0, 4), dtype=np.double)
        assert_(is_valid_im(R) == False)
        assert_raises(ValueError, is_valid_im, R, throw=True)

    def test_is_valid_im_4_and_up(self):
        # Tests is_valid_im(R) on im on observation sets between sizes 4 and 15
        # (step size 3).
        for i in range(4, 15, 3):
            y = np.random.rand(i*(i-1)//2)
            Z = linkage(y)
            R = inconsistent(Z)
            assert_(is_valid_im(R) == True)

    def test_is_valid_im_4_and_up_neg_index_left(self):
        # Tests is_valid_im(R) on im on observation sets between sizes 4 and 15
        # (step size 3) with negative link height means.
        for i in range(4, 15, 3):
            y = np.random.rand(i*(i-1)//2)
            Z = linkage(y)
            R = inconsistent(Z)
            R[i//2,0] = -2.0
            assert_(is_valid_im(R) == False)
            assert_raises(ValueError, is_valid_im, R, throw=True)

    def test_is_valid_im_4_and_up_neg_index_right(self):
        # Tests is_valid_im(R) on im on observation sets between sizes 4 and 15
        # (step size 3) with negative link height standard deviations.
        for i in range(4, 15, 3):
            y = np.random.rand(i*(i-1)//2)
            Z = linkage(y)
            R = inconsistent(Z)
            R[i//2,1] = -2.0
            assert_(is_valid_im(R) == False)
            assert_raises(ValueError, is_valid_im, R, throw=True)

    def test_is_valid_im_4_and_up_neg_dist(self):
        # Tests is_valid_im(R) on im on observation sets between sizes 4 and 15
        # (step size 3) with negative link counts.
        for i in range(4, 15, 3):
            y = np.random.rand(i*(i-1)//2)
            Z = linkage(y)
            R = inconsistent(Z)
            R[i//2,2] = -0.5
            assert_(is_valid_im(R) == False)
            assert_raises(ValueError, is_valid_im, R, throw=True)


class TestNumObsLinkage:
    def test_num_obs_linkage_empty(self):
        # Tests num_obs_linkage(Z) with empty linkage.
        Z = np.zeros((0, 4), dtype=np.double)
        assert_raises(ValueError, num_obs_linkage, Z)

    def test_num_obs_linkage_1x4(self):
        # Tests num_obs_linkage(Z) on linkage over 2 observations.
        Z = np.asarray([[0, 1, 3.0, 2]], dtype=np.double)
        assert_equal(num_obs_linkage(Z), 2)

    def test_num_obs_linkage_2x4(self):
        # Tests num_obs_linkage(Z) on linkage over 3 observations.
        Z = np.asarray([[0, 1, 3.0, 2],
                        [3, 2, 4.0, 3]], dtype=np.double)
        assert_equal(num_obs_linkage(Z), 3)

    def test_num_obs_linkage_4_and_up(self):
        # Tests num_obs_linkage(Z) on linkage on observation sets between sizes
        # 4 and 15 (step size 3).
        for i in range(4, 15, 3):
            y = np.random.rand(i*(i-1)//2)
            Z = linkage(y)
            assert_equal(num_obs_linkage(Z), i)


class TestLeavesList:
    def test_leaves_list_1x4(self):
        # Tests leaves_list(Z) on a 1x4 linkage.
        Z = np.asarray([[0, 1, 3.0, 2]], dtype=np.double)
        to_tree(Z)
        assert_equal(leaves_list(Z), [0, 1])

    def test_leaves_list_2x4(self):
        # Tests leaves_list(Z) on a 2x4 linkage.
        Z = np.asarray([[0, 1, 3.0, 2],
                        [3, 2, 4.0, 3]], dtype=np.double)
        to_tree(Z)
        assert_equal(leaves_list(Z), [0, 1, 2])

    def test_leaves_list_Q(self):
        for method in ['single', 'complete', 'average', 'weighted', 'centroid',
                       'median', 'ward']:
            self.check_leaves_list_Q(method)

    def check_leaves_list_Q(self, method):
        # Tests leaves_list(Z) on the Q data set
        X = hierarchy_test_data.Q_X
        Z = linkage(X, method)
        node = to_tree(Z)
        assert_equal(node.pre_order(), leaves_list(Z))

    def test_Q_subtree_pre_order(self):
        # Tests that pre_order() works when called on sub-trees.
        X = hierarchy_test_data.Q_X
        Z = linkage(X, 'single')
        node = to_tree(Z)
        assert_equal(node.pre_order(), (node.get_left().pre_order()
                                        + node.get_right().pre_order()))


class TestCorrespond:
    def test_correspond_empty(self):
        # Tests correspond(Z, y) with empty linkage and condensed distance matrix.
        y = np.zeros((0,))
        Z = np.zeros((0,4))
        assert_raises(ValueError, correspond, Z, y)

    def test_correspond_2_and_up(self):
        # Tests correspond(Z, y) on linkage and CDMs over observation sets of
        # different sizes.
        for i in range(2, 4):
            y = np.random.rand(i*(i-1)//2)
            Z = linkage(y)
            assert_(correspond(Z, y))
        for i in range(4, 15, 3):
            y = np.random.rand(i*(i-1)//2)
            Z = linkage(y)
            assert_(correspond(Z, y))

    def test_correspond_4_and_up(self):
        # Tests correspond(Z, y) on linkage and CDMs over observation sets of
        # different sizes. Correspondence should be false.
        for (i, j) in (list(zip(list(range(2, 4)), list(range(3, 5)))) +
                       list(zip(list(range(3, 5)), list(range(2, 4))))):
            y = np.random.rand(i*(i-1)//2)
            y2 = np.random.rand(j*(j-1)//2)
            Z = linkage(y)
            Z2 = linkage(y2)
            assert_equal(correspond(Z, y2), False)
            assert_equal(correspond(Z2, y), False)

    def test_correspond_4_and_up_2(self):
        # Tests correspond(Z, y) on linkage and CDMs over observation sets of
        # different sizes. Correspondence should be false.
        for (i, j) in (list(zip(list(range(2, 7)), list(range(16, 21)))) +
                       list(zip(list(range(2, 7)), list(range(16, 21))))):
            y = np.random.rand(i*(i-1)//2)
            y2 = np.random.rand(j*(j-1)//2)
            Z = linkage(y)
            Z2 = linkage(y2)
            assert_equal(correspond(Z, y2), False)
            assert_equal(correspond(Z2, y), False)

    def test_num_obs_linkage_multi_matrix(self):
        # Tests num_obs_linkage with observation matrices of multiple sizes.
        for n in range(2, 10):
            X = np.random.rand(n, 4)
            Y = pdist(X)
            Z = linkage(Y)
            assert_equal(num_obs_linkage(Z), n)


class TestIsMonotonic:
    def test_is_monotonic_empty(self):
        # Tests is_monotonic(Z) on an empty linkage.
        Z = np.zeros((0, 4))
        assert_raises(ValueError, is_monotonic, Z)

    def test_is_monotonic_1x4(self):
        # Tests is_monotonic(Z) on 1x4 linkage. Expecting True.
        Z = np.asarray([[0, 1, 0.3, 2]], dtype=np.double)
        assert_equal(is_monotonic(Z), True)

    def test_is_monotonic_2x4_T(self):
        # Tests is_monotonic(Z) on 2x4 linkage. Expecting True.
        Z = np.asarray([[0, 1, 0.3, 2],
                        [2, 3, 0.4, 3]], dtype=np.double)
        assert_equal(is_monotonic(Z), True)

    def test_is_monotonic_2x4_F(self):
        # Tests is_monotonic(Z) on 2x4 linkage. Expecting False.
        Z = np.asarray([[0, 1, 0.4, 2],
                        [2, 3, 0.3, 3]], dtype=np.double)
        assert_equal(is_monotonic(Z), False)

    def test_is_monotonic_3x4_T(self):
        # Tests is_monotonic(Z) on 3x4 linkage. Expecting True.
        Z = np.asarray([[0, 1, 0.3, 2],
                        [2, 3, 0.4, 2],
                        [4, 5, 0.6, 4]], dtype=np.double)
        assert_equal(is_monotonic(Z), True)

    def test_is_monotonic_3x4_F1(self):
        # Tests is_monotonic(Z) on 3x4 linkage (case 1). Expecting False.
        Z = np.asarray([[0, 1, 0.3, 2],
                        [2, 3, 0.2, 2],
                        [4, 5, 0.6, 4]], dtype=np.double)
        assert_equal(is_monotonic(Z), False)

    def test_is_monotonic_3x4_F2(self):
        # Tests is_monotonic(Z) on 3x4 linkage (case 2). Expecting False.
        Z = np.asarray([[0, 1, 0.8, 2],
                        [2, 3, 0.4, 2],
                        [4, 5, 0.6, 4]], dtype=np.double)
        assert_equal(is_monotonic(Z), False)

    def test_is_monotonic_3x4_F3(self):
        # Tests is_monotonic(Z) on 3x4 linkage (case 3). Expecting False
        Z = np.asarray([[0, 1, 0.3, 2],
                        [2, 3, 0.4, 2],
                        [4, 5, 0.2, 4]], dtype=np.double)
        assert_equal(is_monotonic(Z), False)

    def test_is_monotonic_tdist_linkage1(self):
        # Tests is_monotonic(Z) on clustering generated by single linkage on
        # tdist data set. Expecting True.
        Z = linkage(hierarchy_test_data.ytdist, 'single')
        assert_equal(is_monotonic(Z), True)

    def test_is_monotonic_tdist_linkage2(self):
        # Tests is_monotonic(Z) on clustering generated by single linkage on
        # tdist data set. Perturbing. Expecting False.
        Z = linkage(hierarchy_test_data.ytdist, 'single')
        Z[2,2] = 0.0
        assert_equal(is_monotonic(Z), False)

    def test_is_monotonic_Q_linkage(self):
        # Tests is_monotonic(Z) on clustering generated by single linkage on
        # Q data set. Expecting True.
        X = hierarchy_test_data.Q_X
        Z = linkage(X, 'single')
        assert_equal(is_monotonic(Z), True)


class TestMaxDists:
    def test_maxdists_empty_linkage(self):
        # Tests maxdists(Z) on empty linkage. Expecting exception.
        Z = np.zeros((0, 4), dtype=np.double)
        assert_raises(ValueError, maxdists, Z)

    def test_maxdists_one_cluster_linkage(self):
        # Tests maxdists(Z) on linkage with one cluster.
        Z = np.asarray([[0, 1, 0.3, 4]], dtype=np.double)
        MD = maxdists(Z)
        expectedMD = calculate_maximum_distances(Z)
        assert_allclose(MD, expectedMD, atol=1e-15)

    def test_maxdists_Q_linkage(self):
        for method in ['single', 'complete', 'ward', 'centroid', 'median']:
            self.check_maxdists_Q_linkage(method)

    def check_maxdists_Q_linkage(self, method):
        # Tests maxdists(Z) on the Q data set
        X = hierarchy_test_data.Q_X
        Z = linkage(X, method)
        MD = maxdists(Z)
        expectedMD = calculate_maximum_distances(Z)
        assert_allclose(MD, expectedMD, atol=1e-15)


class TestMaxInconsts:
    def test_maxinconsts_empty_linkage(self):
        # Tests maxinconsts(Z, R) on empty linkage. Expecting exception.
        Z = np.zeros((0, 4), dtype=np.double)
        R = np.zeros((0, 4), dtype=np.double)
        assert_raises(ValueError, maxinconsts, Z, R)

    def test_maxinconsts_difrow_linkage(self):
        # Tests maxinconsts(Z, R) on linkage and inconsistency matrices with
        # different numbers of clusters. Expecting exception.
        Z = np.asarray([[0, 1, 0.3, 4]], dtype=np.double)
        R = np.random.rand(2, 4)
        assert_raises(ValueError, maxinconsts, Z, R)

    def test_maxinconsts_one_cluster_linkage(self):
        # Tests maxinconsts(Z, R) on linkage with one cluster.
        Z = np.asarray([[0, 1, 0.3, 4]], dtype=np.double)
        R = np.asarray([[0, 0, 0, 0.3]], dtype=np.double)
        MD = maxinconsts(Z, R)
        expectedMD = calculate_maximum_inconsistencies(Z, R)
        assert_allclose(MD, expectedMD, atol=1e-15)

    def test_maxinconsts_Q_linkage(self):
        for method in ['single', 'complete', 'ward', 'centroid', 'median']:
            self.check_maxinconsts_Q_linkage(method)

    def check_maxinconsts_Q_linkage(self, method):
        # Tests maxinconsts(Z, R) on the Q data set
        X = hierarchy_test_data.Q_X
        Z = linkage(X, method)
        R = inconsistent(Z)
        MD = maxinconsts(Z, R)
        expectedMD = calculate_maximum_inconsistencies(Z, R)
        assert_allclose(MD, expectedMD, atol=1e-15)


class TestMaxRStat:
    def test_maxRstat_invalid_index(self):
        for i in [3.3, -1, 4]:
            self.check_maxRstat_invalid_index(i)

    def check_maxRstat_invalid_index(self, i):
        # Tests maxRstat(Z, R, i). Expecting exception.
        Z = np.asarray([[0, 1, 0.3, 4]], dtype=np.double)
        R = np.asarray([[0, 0, 0, 0.3]], dtype=np.double)
        if isinstance(i, int):
            assert_raises(ValueError, maxRstat, Z, R, i)
        else:
            assert_raises(TypeError, maxRstat, Z, R, i)

    def test_maxRstat_empty_linkage(self):
        for i in range(4):
            self.check_maxRstat_empty_linkage(i)

    def check_maxRstat_empty_linkage(self, i):
        # Tests maxRstat(Z, R, i) on empty linkage. Expecting exception.
        Z = np.zeros((0, 4), dtype=np.double)
        R = np.zeros((0, 4), dtype=np.double)
        assert_raises(ValueError, maxRstat, Z, R, i)

    def test_maxRstat_difrow_linkage(self):
        for i in range(4):
            self.check_maxRstat_difrow_linkage(i)

    def check_maxRstat_difrow_linkage(self, i):
        # Tests maxRstat(Z, R, i) on linkage and inconsistency matrices with
        # different numbers of clusters. Expecting exception.
        Z = np.asarray([[0, 1, 0.3, 4]], dtype=np.double)
        R = np.random.rand(2, 4)
        assert_raises(ValueError, maxRstat, Z, R, i)

    def test_maxRstat_one_cluster_linkage(self):
        for i in range(4):
            self.check_maxRstat_one_cluster_linkage(i)

    def check_maxRstat_one_cluster_linkage(self, i):
        # Tests maxRstat(Z, R, i) on linkage with one cluster.
        Z = np.asarray([[0, 1, 0.3, 4]], dtype=np.double)
        R = np.asarray([[0, 0, 0, 0.3]], dtype=np.double)
        MD = maxRstat(Z, R, 1)
        expectedMD = calculate_maximum_inconsistencies(Z, R, 1)
        assert_allclose(MD, expectedMD, atol=1e-15)

    def test_maxRstat_Q_linkage(self):
        for method in ['single', 'complete', 'ward', 'centroid', 'median']:
            for i in range(4):
                self.check_maxRstat_Q_linkage(method, i)

    def check_maxRstat_Q_linkage(self, method, i):
        # Tests maxRstat(Z, R, i) on the Q data set
        X = hierarchy_test_data.Q_X
        Z = linkage(X, method)
        R = inconsistent(Z)
        MD = maxRstat(Z, R, 1)
        expectedMD = calculate_maximum_inconsistencies(Z, R, 1)
        assert_allclose(MD, expectedMD, atol=1e-15)


class TestDendrogram:
    def test_dendrogram_single_linkage_tdist(self):
        # Tests dendrogram calculation on single linkage of the tdist data set.
        Z = linkage(hierarchy_test_data.ytdist, 'single')
        R = dendrogram(Z, no_plot=True)
        leaves = R["leaves"]
        assert_equal(leaves, [2, 5, 1, 0, 3, 4])

    def test_valid_orientation(self):
        Z = linkage(hierarchy_test_data.ytdist, 'single')
        assert_raises(ValueError, dendrogram, Z, orientation="foo")

    def test_labels_as_array_or_list(self):
        # test for gh-12418
        Z = linkage(hierarchy_test_data.ytdist, 'single')
        labels = np.array([1, 3, 2, 6, 4, 5])
        result1 = dendrogram(Z, labels=labels, no_plot=True)
        result2 = dendrogram(Z, labels=labels.tolist(), no_plot=True)
        assert result1 == result2

    @pytest.mark.skipif(not have_matplotlib, reason="no matplotlib")
    def test_valid_label_size(self):
        link = np.array([
            [0, 1, 1.0, 4],
            [2, 3, 1.0, 5],
            [4, 5, 2.0, 6],
        ])
        plt.figure()
        with pytest.raises(ValueError) as exc_info:
            dendrogram(link, labels=list(range(100)))
        assert "Dimensions of Z and labels must be consistent."\
               in str(exc_info.value)

        with pytest.raises(
                ValueError,
                match="Dimensions of Z and labels must be consistent."):
            dendrogram(link, labels=[])

        plt.close()

    @pytest.mark.skipif(not have_matplotlib, reason="no matplotlib")
    def test_dendrogram_plot(self):
        for orientation in ['top', 'bottom', 'left', 'right']:
            self.check_dendrogram_plot(orientation)

    def check_dendrogram_plot(self, orientation):
        # Tests dendrogram plotting.
        Z = linkage(hierarchy_test_data.ytdist, 'single')
        expected = {'color_list': ['C1', 'C0', 'C0', 'C0', 'C0'],
                    'dcoord': [[0.0, 138.0, 138.0, 0.0],
                               [0.0, 219.0, 219.0, 0.0],
                               [0.0, 255.0, 255.0, 219.0],
                               [0.0, 268.0, 268.0, 255.0],
                               [138.0, 295.0, 295.0, 268.0]],
                    'icoord': [[5.0, 5.0, 15.0, 15.0],
                               [45.0, 45.0, 55.0, 55.0],
                               [35.0, 35.0, 50.0, 50.0],
                               [25.0, 25.0, 42.5, 42.5],
                               [10.0, 10.0, 33.75, 33.75]],
                    'ivl': ['2', '5', '1', '0', '3', '4'],
                    'leaves': [2, 5, 1, 0, 3, 4],
                    'leaves_color_list': ['C1', 'C1', 'C0', 'C0', 'C0', 'C0'],
                    }

        fig = plt.figure()
        ax = fig.add_subplot(221)

        # test that dendrogram accepts ax keyword
        R1 = dendrogram(Z, ax=ax, orientation=orientation)
        assert_equal(R1, expected)

        # test that dendrogram accepts and handle the leaf_font_size and
        # leaf_rotation keywords
        dendrogram(Z, ax=ax, orientation=orientation,
                   leaf_font_size=20, leaf_rotation=90)
        testlabel = (
            ax.get_xticklabels()[0]
            if orientation in ['top', 'bottom']
            else ax.get_yticklabels()[0]
        )
        assert_equal(testlabel.get_rotation(), 90)
        assert_equal(testlabel.get_size(), 20)
        dendrogram(Z, ax=ax, orientation=orientation,
                   leaf_rotation=90)
        testlabel = (
            ax.get_xticklabels()[0]
            if orientation in ['top', 'bottom']
            else ax.get_yticklabels()[0]
        )
        assert_equal(testlabel.get_rotation(), 90)
        dendrogram(Z, ax=ax, orientation=orientation,
                   leaf_font_size=20)
        testlabel = (
            ax.get_xticklabels()[0]
            if orientation in ['top', 'bottom']
            else ax.get_yticklabels()[0]
        )
        assert_equal(testlabel.get_size(), 20)
        plt.close()

        # test plotting to gca (will import pylab)
        R2 = dendrogram(Z, orientation=orientation)
        plt.close()
        assert_equal(R2, expected)

    @pytest.mark.skipif(not have_matplotlib, reason="no matplotlib")
    def test_dendrogram_truncate_mode(self):
        Z = linkage(hierarchy_test_data.ytdist, 'single')

        R = dendrogram(Z, 2, 'lastp', show_contracted=True)
        plt.close()
        assert_equal(R, {'color_list': ['C0'],
                         'dcoord': [[0.0, 295.0, 295.0, 0.0]],
                         'icoord': [[5.0, 5.0, 15.0, 15.0]],
                         'ivl': ['(2)', '(4)'],
                         'leaves': [6, 9],
                         'leaves_color_list': ['C0', 'C0'],
                         })

        R = dendrogram(Z, 2, 'mtica', show_contracted=True)
        plt.close()
        assert_equal(R, {'color_list': ['C1', 'C0', 'C0', 'C0'],
                         'dcoord': [[0.0, 138.0, 138.0, 0.0],
                                    [0.0, 255.0, 255.0, 0.0],
                                    [0.0, 268.0, 268.0, 255.0],
                                    [138.0, 295.0, 295.0, 268.0]],
                         'icoord': [[5.0, 5.0, 15.0, 15.0],
                                    [35.0, 35.0, 45.0, 45.0],
                                    [25.0, 25.0, 40.0, 40.0],
                                    [10.0, 10.0, 32.5, 32.5]],
                         'ivl': ['2', '5', '1', '0', '(2)'],
                         'leaves': [2, 5, 1, 0, 7],
                         'leaves_color_list': ['C1', 'C1', 'C0', 'C0', 'C0'],
                         })

    def test_dendrogram_colors(self):
        # Tests dendrogram plots with alternate colors
        Z = linkage(hierarchy_test_data.ytdist, 'single')

        set_link_color_palette(['c', 'm', 'y', 'k'])
        R = dendrogram(Z, no_plot=True,
                       above_threshold_color='g', color_threshold=250)
        set_link_color_palette(['g', 'r', 'c', 'm', 'y', 'k'])

        color_list = R['color_list']
        assert_equal(color_list, ['c', 'm', 'g', 'g', 'g'])

        # reset color palette (global list)
        set_link_color_palette(None)


def calculate_maximum_distances(Z):
    # Used for testing correctness of maxdists.
    n = Z.shape[0] + 1
    B = np.zeros((n-1,))
    q = np.zeros((3,))
    for i in range(0, n - 1):
        q[:] = 0.0
        left = Z[i, 0]
        right = Z[i, 1]
        if left >= n:
            q[0] = B[int(left) - n]
        if right >= n:
            q[1] = B[int(right) - n]
        q[2] = Z[i, 2]
        B[i] = q.max()
    return B


def calculate_maximum_inconsistencies(Z, R, k=3):
    # Used for testing correctness of maxinconsts.
    n = Z.shape[0] + 1
    B = np.zeros((n-1,))
    q = np.zeros((3,))
    for i in range(0, n - 1):
        q[:] = 0.0
        left = Z[i, 0]
        right = Z[i, 1]
        if left >= n:
            q[0] = B[int(left) - n]
        if right >= n:
            q[1] = B[int(right) - n]
        q[2] = R[i, k]
        B[i] = q.max()
    return B


def within_tol(a, b, tol):
    return np.abs(a - b).max() < tol


def test_unsupported_uncondensed_distance_matrix_linkage_warning():
    assert_warns(ClusterWarning, linkage, [[0, 1], [1, 0]])


def test_euclidean_linkage_value_error():
    for method in scipy.cluster.hierarchy._EUCLIDEAN_METHODS:
        assert_raises(ValueError, linkage, [[1, 1], [1, 1]],
                      method=method, metric='cityblock')


def test_2x2_linkage():
    Z1 = linkage([1], method='single', metric='euclidean')
    Z2 = linkage([[0, 1], [0, 0]], method='single', metric='euclidean')
    assert_allclose(Z1, Z2)


def test_node_compare():
    np.random.seed(23)
    nobs = 50
    X = np.random.randn(nobs, 4)
    Z = scipy.cluster.hierarchy.ward(X)
    tree = to_tree(Z)
    assert_(tree > tree.get_left())
    assert_(tree.get_right() > tree.get_left())
    assert_(tree.get_right() == tree.get_right())
    assert_(tree.get_right() != tree.get_left())


def test_cut_tree():
    np.random.seed(23)
    nobs = 50
    X = np.random.randn(nobs, 4)
    Z = scipy.cluster.hierarchy.ward(X)
    cutree = cut_tree(Z)

    assert_equal(cutree[:, 0], np.arange(nobs))
    assert_equal(cutree[:, -1], np.zeros(nobs))
    assert_equal(cutree.max(0), np.arange(nobs - 1, -1, -1))

    assert_equal(cutree[:, [-5]], cut_tree(Z, n_clusters=5))
    assert_equal(cutree[:, [-5, -10]], cut_tree(Z, n_clusters=[5, 10]))
    assert_equal(cutree[:, [-10, -5]], cut_tree(Z, n_clusters=[10, 5]))

    nodes = _order_cluster_tree(Z)
    heights = np.array([node.dist for node in nodes])

    assert_equal(cutree[:, np.searchsorted(heights, [5])],
                 cut_tree(Z, height=5))
    assert_equal(cutree[:, np.searchsorted(heights, [5, 10])],
                 cut_tree(Z, height=[5, 10]))
    assert_equal(cutree[:, np.searchsorted(heights, [10, 5])],
                 cut_tree(Z, height=[10, 5]))


def test_optimal_leaf_ordering():
    # test with the distance vector y
    Z = optimal_leaf_ordering(linkage(hierarchy_test_data.ytdist),
                              hierarchy_test_data.ytdist)
    expectedZ = hierarchy_test_data.linkage_ytdist_single_olo
    assert_allclose(Z, expectedZ, atol=1e-10)

    # test with the observation matrix X
    Z = optimal_leaf_ordering(linkage(hierarchy_test_data.X, 'ward'),
                              hierarchy_test_data.X)
    expectedZ = hierarchy_test_data.linkage_X_ward_olo
    assert_allclose(Z, expectedZ, atol=1e-06)


def test_Heap():
    values = np.array([2, -1, 0, -1.5, 3])
    heap = Heap(values)

    pair = heap.get_min()
    assert_equal(pair['key'], 3)
    assert_equal(pair['value'], -1.5)

    heap.remove_min()
    pair = heap.get_min()
    assert_equal(pair['key'], 1)
    assert_equal(pair['value'], -1)

    heap.change_value(1, 2.5)
    pair = heap.get_min()
    assert_equal(pair['key'], 2)
    assert_equal(pair['value'], 0)

    heap.remove_min()
    heap.remove_min()

    heap.change_value(1, 10)
    pair = heap.get_min()
    assert_equal(pair['key'], 4)
    assert_equal(pair['value'], 3)

    heap.remove_min()
    pair = heap.get_min()
    assert_equal(pair['key'], 1)
    assert_equal(pair['value'], 10)
