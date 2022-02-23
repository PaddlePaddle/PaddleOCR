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

import os.path

from functools import wraps, partial
import weakref

import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
                           assert_array_equal, assert_equal,
                           assert_almost_equal, assert_allclose,
                           suppress_warnings)
import pytest
from pytest import raises as assert_raises

import scipy.spatial.distance
from scipy.spatial import _distance_pybind
from scipy.spatial.distance import (
    squareform, pdist, cdist, num_obs_y, num_obs_dm, is_valid_dm, is_valid_y,
    _validate_vector, _METRICS_NAMES, _METRICS)

# these were missing: chebyshev cityblock kulsinski
# jensenshannon, matching and seuclidean are referenced by string name.
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
                                    correlation, cosine, dice, euclidean,
                                    hamming, jaccard, jensenshannon,
                                    kulsinski, mahalanobis, matching,
                                    minkowski, rogerstanimoto, russellrao,
                                    seuclidean, sokalmichener, sokalsneath,
                                    sqeuclidean, yule)
from scipy.spatial.distance import wminkowski as old_wminkowski

_filenames = [
              "cdist-X1.txt",
              "cdist-X2.txt",
              "iris.txt",
              "pdist-boolean-inp.txt",
              "pdist-chebyshev-ml-iris.txt",
              "pdist-chebyshev-ml.txt",
              "pdist-cityblock-ml-iris.txt",
              "pdist-cityblock-ml.txt",
              "pdist-correlation-ml-iris.txt",
              "pdist-correlation-ml.txt",
              "pdist-cosine-ml-iris.txt",
              "pdist-cosine-ml.txt",
              "pdist-double-inp.txt",
              "pdist-euclidean-ml-iris.txt",
              "pdist-euclidean-ml.txt",
              "pdist-hamming-ml.txt",
              "pdist-jaccard-ml.txt",
              "pdist-jensenshannon-ml-iris.txt",
              "pdist-jensenshannon-ml.txt",
              "pdist-minkowski-3.2-ml-iris.txt",
              "pdist-minkowski-3.2-ml.txt",
              "pdist-minkowski-5.8-ml-iris.txt",
              "pdist-seuclidean-ml-iris.txt",
              "pdist-seuclidean-ml.txt",
              "pdist-spearman-ml.txt",
              "random-bool-data.txt",
              "random-double-data.txt",
              "random-int-data.txt",
              "random-uint-data.txt",
              ]

_tdist = np.array([[0, 662, 877, 255, 412, 996],
                      [662, 0, 295, 468, 268, 400],
                      [877, 295, 0, 754, 564, 138],
                      [255, 468, 754, 0, 219, 869],
                      [412, 268, 564, 219, 0, 669],
                      [996, 400, 138, 869, 669, 0]], dtype='double')

_ytdist = squareform(_tdist)

# A hashmap of expected output arrays for the tests. These arrays
# come from a list of text files, which are read prior to testing.
# Each test loads inputs and outputs from this dictionary.
eo = {}


def load_testing_files():
    for fn in _filenames:
        name = fn.replace(".txt", "").replace("-ml", "")
        fqfn = os.path.join(os.path.dirname(__file__), 'data', fn)
        fp = open(fqfn)
        eo[name] = np.loadtxt(fp)
        fp.close()
    eo['pdist-boolean-inp'] = np.bool_(eo['pdist-boolean-inp'])
    eo['random-bool-data'] = np.bool_(eo['random-bool-data'])
    eo['random-float32-data'] = np.float32(eo['random-double-data'])
    eo['random-int-data'] = np.int_(eo['random-int-data'])
    eo['random-uint-data'] = np.uint(eo['random-uint-data'])


load_testing_files()


def _is_32bit():
    return np.intp(0).itemsize < 8


def _chk_asarrays(arrays, axis=None):
    arrays = [np.asanyarray(a) for a in arrays]
    if axis is None:
        # np < 1.10 ravel removes subclass from arrays
        arrays = [np.ravel(a) if a.ndim != 1 else a
                  for a in arrays]
        axis = 0
    arrays = tuple(np.atleast_1d(a) for a in arrays)
    if axis < 0:
        if not all(a.ndim == arrays[0].ndim for a in arrays):
            raise ValueError("array ndim must be the same for neg axis")
        axis = range(arrays[0].ndim)[axis]
    return arrays + (axis,)


def _chk_weights(arrays, weights=None, axis=None,
                 force_weights=False, simplify_weights=True,
                 pos_only=False, neg_check=False,
                 nan_screen=False, mask_screen=False,
                 ddof=None):
    chked = _chk_asarrays(arrays, axis=axis)
    arrays, axis = chked[:-1], chked[-1]

    simplify_weights = simplify_weights and not force_weights
    if not force_weights and mask_screen:
        force_weights = any(np.ma.getmask(a) is not np.ma.nomask for a in arrays)

    if nan_screen:
        has_nans = [np.isnan(np.sum(a)) for a in arrays]
        if any(has_nans):
            mask_screen = True
            force_weights = True
            arrays = tuple(np.ma.masked_invalid(a) if has_nan else a
                           for a, has_nan in zip(arrays, has_nans))

    if weights is not None:
        weights = np.asanyarray(weights)
    elif force_weights:
        weights = np.ones(arrays[0].shape[axis])
    else:
        return arrays + (weights, axis)

    if ddof:
        weights = _freq_weights(weights)

    if mask_screen:
        weights = _weight_masked(arrays, weights, axis)

    if not all(weights.shape == (a.shape[axis],) for a in arrays):
        raise ValueError("weights shape must match arrays along axis")
    if neg_check and (weights < 0).any():
        raise ValueError("weights cannot be negative")

    if pos_only:
        pos_weights = np.nonzero(weights > 0)[0]
        if pos_weights.size < weights.size:
            arrays = tuple(np.take(a, pos_weights, axis=axis) for a in arrays)
            weights = weights[pos_weights]
    if simplify_weights and (weights == 1).all():
        weights = None
    return arrays + (weights, axis)


def _freq_weights(weights):
    if weights is None:
        return weights
    int_weights = weights.astype(int)
    if (weights != int_weights).any():
        raise ValueError("frequency (integer count-type) weights required %s" % weights)
    return int_weights


def _weight_masked(arrays, weights, axis):
    if axis is None:
        axis = 0
    weights = np.asanyarray(weights)
    for a in arrays:
        axis_mask = np.ma.getmask(a)
        if axis_mask is np.ma.nomask:
            continue
        if a.ndim > 1:
            not_axes = tuple(i for i in range(a.ndim) if i != axis)
            axis_mask = axis_mask.any(axis=not_axes)
        weights *= 1 - axis_mask.astype(int)
    return weights


def within_tol(a, b, tol):
    return np.abs(a - b).max() < tol


def _assert_within_tol(a, b, atol=0, rtol=0, verbose_=False):
    if verbose_:
        print(np.abs(a - b).max())
    assert_allclose(a, b, rtol=rtol, atol=atol)


def _rand_split(arrays, weights, axis, split_per, seed=None):
    # inverse operation for stats.collapse_weights
    weights = np.array(weights, dtype=np.float64)  # modified inplace; need a copy
    seeded_rand = np.random.RandomState(seed)

    def mytake(a, ix, axis):
        record = np.asanyarray(np.take(a, ix, axis=axis))
        return record.reshape([a.shape[i] if i != axis else 1
                               for i in range(a.ndim)])

    n_obs = arrays[0].shape[axis]
    assert all(a.shape[axis] == n_obs for a in arrays), "data must be aligned on sample axis"
    for i in range(int(split_per) * n_obs):
        split_ix = seeded_rand.randint(n_obs + i)
        prev_w = weights[split_ix]
        q = seeded_rand.rand()
        weights[split_ix] = q * prev_w
        weights = np.append(weights, (1. - q) * prev_w)
        arrays = [np.append(a, mytake(a, split_ix, axis=axis),
                            axis=axis) for a in arrays]
    return arrays, weights


def _rough_check(a, b, compare_assert=partial(assert_allclose, atol=1e-5),
                  key=lambda x: x, w=None):
    check_a = key(a)
    check_b = key(b)
    try:
        if np.array(check_a != check_b).any():  # try strict equality for string types
            compare_assert(check_a, check_b)
    except AttributeError:  # masked array
        compare_assert(check_a, check_b)
    except (TypeError, ValueError):  # nested data structure
        for a_i, b_i in zip(check_a, check_b):
            _rough_check(a_i, b_i, compare_assert=compare_assert)

# diff from test_stats:
#  n_args=2, weight_arg='w', default_axis=None
#  ma_safe = False, nan_safe = False
def _weight_checked(fn, n_args=2, default_axis=None, key=lambda x: x, weight_arg='w',
                    squeeze=True, silent=False,
                    ones_test=True, const_test=True, dup_test=True,
                    split_test=True, dud_test=True, ma_safe=False, ma_very_safe=False, nan_safe=False,
                    split_per=1.0, seed=0, compare_assert=partial(assert_allclose, atol=1e-5)):
    """runs fn on its arguments 2 or 3 ways, checks that the results are the same,
       then returns the same thing it would have returned before"""
    @wraps(fn)
    def wrapped(*args, **kwargs):
        result = fn(*args, **kwargs)

        arrays = args[:n_args]
        rest = args[n_args:]
        weights = kwargs.get(weight_arg, None)
        axis = kwargs.get('axis', default_axis)

        chked = _chk_weights(arrays, weights=weights, axis=axis, force_weights=True, mask_screen=True)
        arrays, weights, axis = chked[:-2], chked[-2], chked[-1]
        if squeeze:
            arrays = [np.atleast_1d(a.squeeze()) for a in arrays]

        try:
            # WEIGHTS CHECK 1: EQUAL WEIGHTED OBESERVATIONS
            args = tuple(arrays) + rest
            if ones_test:
                kwargs[weight_arg] = weights
                _rough_check(result, fn(*args, **kwargs), key=key)
            if const_test:
                kwargs[weight_arg] = weights * 101.0
                _rough_check(result, fn(*args, **kwargs), key=key)
                kwargs[weight_arg] = weights * 0.101
                try:
                    _rough_check(result, fn(*args, **kwargs), key=key)
                except Exception as e:
                    raise type(e)((e, arrays, weights)) from e

            # WEIGHTS CHECK 2: ADDL 0-WEIGHTED OBS
            if dud_test:
                # add randomly resampled rows, weighted at 0
                dud_arrays, dud_weights = _rand_split(arrays, weights, axis, split_per=split_per, seed=seed)
                dud_weights[:weights.size] = weights  # not exactly 1 because of masked arrays
                dud_weights[weights.size:] = 0
                dud_args = tuple(dud_arrays) + rest
                kwargs[weight_arg] = dud_weights
                _rough_check(result, fn(*dud_args, **kwargs), key=key)
                # increase the value of those 0-weighted rows
                for a in dud_arrays:
                    indexer = [slice(None)] * a.ndim
                    indexer[axis] = slice(weights.size, None)
                    indexer = tuple(indexer)
                    a[indexer] = a[indexer] * 101
                dud_args = tuple(dud_arrays) + rest
                _rough_check(result, fn(*dud_args, **kwargs), key=key)
                # set those 0-weighted rows to NaNs
                for a in dud_arrays:
                    indexer = [slice(None)] * a.ndim
                    indexer[axis] = slice(weights.size, None)
                    indexer = tuple(indexer)
                    a[indexer] = a[indexer] * np.nan
                if kwargs.get("nan_policy", None) == "omit" and nan_safe:
                    dud_args = tuple(dud_arrays) + rest
                    _rough_check(result, fn(*dud_args, **kwargs), key=key)
                # mask out those nan values
                if ma_safe:
                    dud_arrays = [np.ma.masked_invalid(a) for a in dud_arrays]
                    dud_args = tuple(dud_arrays) + rest
                    _rough_check(result, fn(*dud_args, **kwargs), key=key)
                    if ma_very_safe:
                        kwargs[weight_arg] = None
                        _rough_check(result, fn(*dud_args, **kwargs), key=key)
                del dud_arrays, dud_args, dud_weights

            # WEIGHTS CHECK 3: DUPLICATE DATA (DUMB SPLITTING)
            if dup_test:
                dup_arrays = [np.append(a, a, axis=axis) for a in arrays]
                dup_weights = np.append(weights, weights) / 2.0
                dup_args = tuple(dup_arrays) + rest
                kwargs[weight_arg] = dup_weights
                _rough_check(result, fn(*dup_args, **kwargs), key=key)
                del dup_args, dup_arrays, dup_weights

            # WEIGHT CHECK 3: RANDOM SPLITTING
            if split_test and split_per > 0:
                split_arrays, split_weights = _rand_split(arrays, weights, axis, split_per=split_per, seed=seed)
                split_args = tuple(split_arrays) + rest
                kwargs[weight_arg] = split_weights
                _rough_check(result, fn(*split_args, **kwargs), key=key)
        except NotImplementedError as e:
            # when some combination of arguments makes weighting impossible,
            #  this is the desired response
            if not silent:
                warnings.warn("%s NotImplemented weights: %s" % (fn.__name__, e))
        return result
    return wrapped


wcdist = _weight_checked(cdist, default_axis=1, squeeze=False)
wcdist_no_const = _weight_checked(cdist, default_axis=1, squeeze=False, const_test=False)
wpdist = _weight_checked(pdist, default_axis=1, squeeze=False, n_args=1)
wpdist_no_const = _weight_checked(pdist, default_axis=1, squeeze=False, const_test=False, n_args=1)
wrogerstanimoto = _weight_checked(rogerstanimoto)
wmatching = whamming = _weight_checked(hamming, dud_test=False)
wyule = _weight_checked(yule)
wdice = _weight_checked(dice)
wcityblock = _weight_checked(cityblock)
wchebyshev = _weight_checked(chebyshev)
wcosine = _weight_checked(cosine)
wcorrelation = _weight_checked(correlation)
wkulsinski = _weight_checked(kulsinski)
wminkowski = _weight_checked(minkowski, const_test=False)
wjaccard = _weight_checked(jaccard)
weuclidean = _weight_checked(euclidean, const_test=False)
wsqeuclidean = _weight_checked(sqeuclidean, const_test=False)
wbraycurtis = _weight_checked(braycurtis)
wcanberra = _weight_checked(canberra, const_test=False)
wsokalsneath = _weight_checked(sokalsneath)
wsokalmichener = _weight_checked(sokalmichener)
wrussellrao = _weight_checked(russellrao)


class TestCdist:

    def setup_method(self):
        self.rnd_eo_names = ['random-float32-data', 'random-int-data',
                             'random-uint-data', 'random-double-data',
                             'random-bool-data']
        self.valid_upcasts = {'bool': [np.uint, np.int_, np.float32, np.double],
                              'uint': [np.int_, np.float32, np.double],
                              'int': [np.float32, np.double],
                              'float32': [np.double]}

    def test_cdist_extra_args(self):
        # Tests that args and kwargs are correctly handled
        def _my_metric(x, y, arg, kwarg=1, kwarg2=2):
            return arg + kwarg + kwarg2

        X1 = [[1., 2., 3.], [1.2, 2.3, 3.4], [2.2, 2.3, 4.4]]
        X2 = [[7., 5., 8.], [7.5, 5.8, 8.4], [5.5, 5.8, 4.4]]
        kwargs = {'N0tV4l1D_p4raM': 3.14, "w":np.arange(3)}
        args = [3.14] * 200
        with suppress_warnings() as w:
            w.filter(DeprecationWarning)
            for metric in _METRICS_NAMES:
                assert_raises(TypeError, cdist, X1, X2,
                              metric=metric, **kwargs)
                assert_raises(TypeError, cdist, X1, X2,
                              metric=eval(metric), **kwargs)
                assert_raises(TypeError, cdist, X1, X2,
                              metric="test_" + metric, **kwargs)
                assert_raises(TypeError, cdist, X1, X2,
                              metric=metric, *args)
                assert_raises(TypeError, cdist, X1, X2,
                              metric=eval(metric), *args)
                assert_raises(TypeError, cdist, X1, X2,
                              metric="test_" + metric, *args)

            assert_raises(TypeError, cdist, X1, X2, _my_metric)
            assert_raises(TypeError, cdist, X1, X2, _my_metric, *args)
            assert_raises(TypeError, cdist, X1, X2, _my_metric, **kwargs)
            assert_raises(TypeError, cdist, X1, X2, _my_metric,
                          kwarg=2.2, kwarg2=3.3)
            assert_raises(TypeError, cdist, X1, X2, _my_metric, 1, 2, kwarg=2.2)

            assert_raises(TypeError, cdist, X1, X2, _my_metric, 1.1, 2.2, 3.3)
            assert_raises(TypeError, cdist, X1, X2, _my_metric, 1.1, 2.2)
            assert_raises(TypeError, cdist, X1, X2, _my_metric, 1.1)
            assert_raises(TypeError, cdist, X1, X2, _my_metric, 1.1,
                          kwarg=2.2, kwarg2=3.3)

            # this should work
            assert_allclose(cdist(X1, X2, metric=_my_metric,
                                  arg=1.1, kwarg2=3.3), 5.4)

    def test_cdist_euclidean_random_unicode(self):
        eps = 1e-07
        X1 = eo['cdist-X1']
        X2 = eo['cdist-X2']
        Y1 = wcdist_no_const(X1, X2, 'euclidean')
        Y2 = wcdist_no_const(X1, X2, 'test_euclidean')
        _assert_within_tol(Y1, Y2, eps, verbose > 2)

    @pytest.mark.parametrize("p", [1.0, 1.23, 2.0, 3.8, 4.6, np.inf])
    def test_cdist_minkowski_random(self, p):
        eps = 1e-07
        X1 = eo['cdist-X1']
        X2 = eo['cdist-X2']
        Y1 = wcdist_no_const(X1, X2, 'minkowski', p=p)
        Y2 = wcdist_no_const(X1, X2, 'test_minkowski', p=p)
        _assert_within_tol(Y1, Y2, eps, verbose > 2)

    def test_cdist_cosine_random(self):
        eps = 1e-07
        X1 = eo['cdist-X1']
        X2 = eo['cdist-X2']
        Y1 = wcdist(X1, X2, 'cosine')

        # Naive implementation
        def norms(X):
            return np.linalg.norm(X, axis=1).reshape(-1, 1)

        Y2 = 1 - np.dot((X1 / norms(X1)), (X2 / norms(X2)).T)

        _assert_within_tol(Y1, Y2, eps, verbose > 2)

    def test_cdist_mahalanobis(self):
        # 1-dimensional observations
        x1 = np.array([[2], [3]])
        x2 = np.array([[2], [5]])
        dist = cdist(x1, x2, metric='mahalanobis')
        assert_allclose(dist, [[0.0, np.sqrt(4.5)], [np.sqrt(0.5), np.sqrt(2)]])

        # 2-dimensional observations
        x1 = np.array([[0, 0], [-1, 0]])
        x2 = np.array([[0, 2], [1, 0], [0, -2]])
        dist = cdist(x1, x2, metric='mahalanobis')
        rt2 = np.sqrt(2)
        assert_allclose(dist, [[rt2, rt2, rt2], [2, 2 * rt2, 2]])

        # Too few observations
        assert_raises(ValueError,
                      cdist, [[0, 1]], [[2, 3]], metric='mahalanobis')

    def test_cdist_custom_notdouble(self):
        class myclass:
            pass

        def _my_metric(x, y):
            if not isinstance(x[0], myclass) or not isinstance(y[0], myclass):
                raise ValueError("Type has been changed")
            return 1.123
        data = np.array([[myclass()]], dtype=object)
        cdist_y = cdist(data, data, metric=_my_metric)
        right_y = 1.123
        assert_equal(cdist_y, right_y, verbose=verbose > 2)

    def _check_calling_conventions(self, X1, X2, metric, eps=1e-07, **kwargs):
        # helper function for test_cdist_calling_conventions
        try:
            y1 = cdist(X1, X2, metric=metric, **kwargs)
            y2 = cdist(X1, X2, metric=eval(metric), **kwargs)
            y3 = cdist(X1, X2, metric="test_" + metric, **kwargs)
        except Exception as e:
            e_cls = e.__class__
            if verbose > 2:
                print(e_cls.__name__)
                print(e)
            assert_raises(e_cls, cdist, X1, X2, metric=metric, **kwargs)
            assert_raises(e_cls, cdist, X1, X2, metric=eval(metric), **kwargs)
            assert_raises(e_cls, cdist, X1, X2, metric="test_" + metric, **kwargs)
        else:
            _assert_within_tol(y1, y2, rtol=eps, verbose_=verbose > 2)
            _assert_within_tol(y1, y3, rtol=eps, verbose_=verbose > 2)

    def test_cdist_calling_conventions(self):
        # Ensures that specifying the metric with a str or scipy function
        # gives the same behaviour (i.e. same result or same exception).
        # NOTE: The correctness should be checked within each metric tests.
        for eo_name in self.rnd_eo_names:
            # subsampling input data to speed-up tests
            # NOTE: num samples needs to be > than dimensions for mahalanobis
            X1 = eo[eo_name][::5, ::-2]
            X2 = eo[eo_name][1::5, ::2]
            for metric in _METRICS_NAMES:
                if verbose > 2:
                    print("testing: ", metric, " with: ", eo_name)
                if metric == 'wminkowski':
                    continue
                if metric in {'dice', 'yule', 'kulsinski', 'matching',
                              'rogerstanimoto', 'russellrao', 'sokalmichener',
                              'sokalsneath'} and 'bool' not in eo_name:
                    # python version permits non-bools e.g. for fuzzy logic
                    continue
                self._check_calling_conventions(X1, X2, metric)

                # Testing built-in metrics with extra args
                if metric == "seuclidean":
                    X12 = np.vstack([X1, X2]).astype(np.double)
                    V = np.var(X12, axis=0, ddof=1)
                    self._check_calling_conventions(X1, X2, metric, V=V)
                elif metric == "mahalanobis":
                    X12 = np.vstack([X1, X2]).astype(np.double)
                    V = np.atleast_2d(np.cov(X12.T))
                    VI = np.array(np.linalg.inv(V).T)
                    self._check_calling_conventions(X1, X2, metric, VI=VI)

    def test_cdist_dtype_equivalence(self):
        # Tests that the result is not affected by type up-casting
        eps = 1e-07
        tests = [(eo['random-bool-data'], self.valid_upcasts['bool']),
                 (eo['random-uint-data'], self.valid_upcasts['uint']),
                 (eo['random-int-data'], self.valid_upcasts['int']),
                 (eo['random-float32-data'], self.valid_upcasts['float32'])]
        for metric in _METRICS_NAMES:
            for test in tests:
                X1 = test[0][::5, ::-2]
                X2 = test[0][1::5, ::2]
                try:
                    y1 = cdist(X1, X2, metric=metric)
                except Exception as e:
                    e_cls = e.__class__
                    if verbose > 2:
                        print(e_cls.__name__)
                        print(e)
                    for new_type in test[1]:
                        X1new = new_type(X1)
                        X2new = new_type(X2)
                        assert_raises(e_cls, cdist, X1new, X2new, metric=metric)
                else:
                    for new_type in test[1]:
                        y2 = cdist(new_type(X1), new_type(X2), metric=metric)
                        _assert_within_tol(y1, y2, eps, verbose > 2)

    def test_cdist_out(self):
        # Test that out parameter works properly
        eps = 1e-07
        X1 = eo['cdist-X1']
        X2 = eo['cdist-X2']
        out_r, out_c = X1.shape[0], X2.shape[0]
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning,
                       message="'wminkowski' metric is deprecated")
            for metric in _METRICS_NAMES:
                kwargs = dict()
                if metric in ['minkowski', 'wminkowski']:
                    kwargs['p'] = 1.23
                    if metric == 'wminkowski':
                        kwargs['w'] = 1.0 / X1.std(axis=0)
                out1 = np.empty((out_r, out_c), dtype=np.double)
                Y1 = cdist(X1, X2, metric, **kwargs)
                Y2 = cdist(X1, X2, metric, out=out1, **kwargs)
                # test that output is numerically equivalent
                _assert_within_tol(Y1, Y2, eps, verbose > 2)
                # test that Y_test1 and out1 are the same object
                assert_(Y2 is out1)
                # test for incorrect shape
                out2 = np.empty((out_r-1, out_c+1), dtype=np.double)
                assert_raises(ValueError,
                              cdist, X1, X2, metric, out=out2, **kwargs)
                # test for C-contiguous order
                out3 = np.empty(
                    (2 * out_r, 2 * out_c), dtype=np.double)[::2, ::2]
                out4 = np.empty((out_r, out_c), dtype=np.double, order='F')
                assert_raises(ValueError,
                              cdist, X1, X2, metric, out=out3, **kwargs)
                assert_raises(ValueError,
                              cdist, X1, X2, metric, out=out4, **kwargs)

                # test for incorrect dtype
                out5 = np.empty((out_r, out_c), dtype=np.int64)
                assert_raises(ValueError,
                              cdist, X1, X2, metric, out=out5, **kwargs)

    def test_striding(self):
        # test that striding is handled correct with calls to
        # _copy_array_if_base_present
        eps = 1e-07
        X1 = eo['cdist-X1'][::2, ::2]
        X2 = eo['cdist-X2'][::2, ::2]
        X1_copy = X1.copy()
        X2_copy = X2.copy()

        # confirm equivalence
        assert_equal(X1, X1_copy)
        assert_equal(X2, X2_copy)
        # confirm contiguity
        assert_(not X1.flags.c_contiguous)
        assert_(not X2.flags.c_contiguous)
        assert_(X1_copy.flags.c_contiguous)
        assert_(X2_copy.flags.c_contiguous)

        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "'wminkowski' metric is deprecated")
            for metric in _METRICS_NAMES:
                kwargs = dict()
                if metric in ['minkowski', 'wminkowski']:
                    kwargs['p'] = 1.23
                    if metric == 'wminkowski':
                        kwargs['w'] = 1.0 / X1.std(axis=0)
                Y1 = cdist(X1, X2, metric, **kwargs)
                Y2 = cdist(X1_copy, X2_copy, metric, **kwargs)
                # test that output is numerically equivalent
                _assert_within_tol(Y1, Y2, eps, verbose > 2)

    def test_cdist_refcount(self):
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "'wminkowski' metric is deprecated")
            for metric in _METRICS_NAMES:
                x1 = np.random.rand(10, 10)
                x2 = np.random.rand(10, 10)

                kwargs = dict()
                if metric in ['minkowski', 'wminkowski']:
                    kwargs['p'] = 1.23
                    if metric == 'wminkowski':
                        kwargs['w'] = 1.0 / x1.std(axis=0)

                out = cdist(x1, x2, metric=metric, **kwargs)

                # Check reference counts aren't messed up. If we only hold weak
                # references, the arrays should be deallocated.
                weak_refs = [weakref.ref(v) for v in (x1, x2, out)]
                del x1, x2, out
                assert all(weak_ref() is None for weak_ref in weak_refs)


class TestPdist:

    def setup_method(self):
        self.rnd_eo_names = ['random-float32-data', 'random-int-data',
                             'random-uint-data', 'random-double-data',
                             'random-bool-data']
        self.valid_upcasts = {'bool': [np.uint, np.int_, np.float32, np.double],
                              'uint': [np.int_, np.float32, np.double],
                              'int': [np.float32, np.double],
                              'float32': [np.double]}

    def test_pdist_extra_args(self):
        # Tests that args and kwargs are correctly handled
        def _my_metric(x, y, arg, kwarg=1, kwarg2=2):
            return arg + kwarg + kwarg2

        X1 = [[1., 2.], [1.2, 2.3], [2.2, 2.3]]
        kwargs = {'N0tV4l1D_p4raM': 3.14, "w":np.arange(2)}
        args = [3.14] * 200
        with suppress_warnings() as w:
            w.filter(DeprecationWarning)
            for metric in _METRICS_NAMES:
                assert_raises(TypeError, pdist, X1, metric=metric, **kwargs)
                assert_raises(TypeError, pdist, X1,
                              metric=eval(metric), **kwargs)
                assert_raises(TypeError, pdist, X1,
                              metric="test_" + metric, **kwargs)
                assert_raises(TypeError, pdist, X1, metric=metric, *args)
                assert_raises(TypeError, pdist, X1, metric=eval(metric), *args)
                assert_raises(TypeError, pdist, X1,
                              metric="test_" + metric, *args)

            assert_raises(TypeError, pdist, X1, _my_metric)
            assert_raises(TypeError, pdist, X1, _my_metric, *args)
            assert_raises(TypeError, pdist, X1, _my_metric, **kwargs)
            assert_raises(TypeError, pdist, X1, _my_metric,
                          kwarg=2.2, kwarg2=3.3)
            assert_raises(TypeError, pdist, X1, _my_metric, 1, 2, kwarg=2.2)

            assert_raises(TypeError, pdist, X1, _my_metric, 1.1, 2.2, 3.3)
            assert_raises(TypeError, pdist, X1, _my_metric, 1.1, 2.2)
            assert_raises(TypeError, pdist, X1, _my_metric, 1.1)
            assert_raises(TypeError, pdist, X1, _my_metric, 1.1,
                          kwarg=2.2, kwarg2=3.3)

            # these should work
            assert_allclose(pdist(X1, metric=_my_metric,
                                  arg=1.1, kwarg2=3.3), 5.4)

    def test_pdist_euclidean_random(self):
        eps = 1e-07
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-euclidean']
        Y_test1 = wpdist_no_const(X, 'euclidean')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_euclidean_random_u(self):
        eps = 1e-07
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-euclidean']
        Y_test1 = wpdist_no_const(X, 'euclidean')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_euclidean_random_float32(self):
        eps = 1e-07
        X = np.float32(eo['pdist-double-inp'])
        Y_right = eo['pdist-euclidean']
        Y_test1 = wpdist_no_const(X, 'euclidean')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_euclidean_random_nonC(self):
        eps = 1e-07
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-euclidean']
        Y_test2 = wpdist_no_const(X, 'test_euclidean')
        _assert_within_tol(Y_test2, Y_right, eps)

    @pytest.mark.slow
    def test_pdist_euclidean_iris_double(self):
        eps = 1e-07
        X = eo['iris']
        Y_right = eo['pdist-euclidean-iris']
        Y_test1 = wpdist_no_const(X, 'euclidean')
        _assert_within_tol(Y_test1, Y_right, eps)

    @pytest.mark.slow
    def test_pdist_euclidean_iris_float32(self):
        eps = 1e-06
        X = np.float32(eo['iris'])
        Y_right = eo['pdist-euclidean-iris']
        Y_test1 = wpdist_no_const(X, 'euclidean')
        _assert_within_tol(Y_test1, Y_right, eps, verbose > 2)

    @pytest.mark.slow
    def test_pdist_euclidean_iris_nonC(self):
        # Test pdist(X, 'test_euclidean') [the non-C implementation] on the
        # Iris data set.
        eps = 1e-07
        X = eo['iris']
        Y_right = eo['pdist-euclidean-iris']
        Y_test2 = wpdist_no_const(X, 'test_euclidean')
        _assert_within_tol(Y_test2, Y_right, eps)

    def test_pdist_seuclidean_random(self):
        eps = 1e-05
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-seuclidean']
        Y_test1 = pdist(X, 'seuclidean')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_seuclidean_random_float32(self):
        eps = 1e-05
        X = np.float32(eo['pdist-double-inp'])
        Y_right = eo['pdist-seuclidean']
        Y_test1 = pdist(X, 'seuclidean')
        _assert_within_tol(Y_test1, Y_right, eps)

        # Check no error is raise when V has float32 dtype (#11171).
        V = np.var(X, axis=0, ddof=1)
        Y_test2 = pdist(X, 'seuclidean', V=V)
        _assert_within_tol(Y_test2, Y_right, eps)

    def test_pdist_seuclidean_random_nonC(self):
        # Test pdist(X, 'test_sqeuclidean') [the non-C implementation]
        eps = 1e-05
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-seuclidean']
        Y_test2 = pdist(X, 'test_seuclidean')
        _assert_within_tol(Y_test2, Y_right, eps)

    def test_pdist_seuclidean_iris(self):
        eps = 1e-05
        X = eo['iris']
        Y_right = eo['pdist-seuclidean-iris']
        Y_test1 = pdist(X, 'seuclidean')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_seuclidean_iris_float32(self):
        # Tests pdist(X, 'seuclidean') on the Iris data set (float32).
        eps = 1e-05
        X = np.float32(eo['iris'])
        Y_right = eo['pdist-seuclidean-iris']
        Y_test1 = pdist(X, 'seuclidean')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_seuclidean_iris_nonC(self):
        # Test pdist(X, 'test_seuclidean') [the non-C implementation] on the
        # Iris data set.
        eps = 1e-05
        X = eo['iris']
        Y_right = eo['pdist-seuclidean-iris']
        Y_test2 = pdist(X, 'test_seuclidean')
        _assert_within_tol(Y_test2, Y_right, eps)

    def test_pdist_cosine_random(self):
        eps = 1e-08
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-cosine']
        Y_test1 = wpdist(X, 'cosine')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_cosine_random_float32(self):
        eps = 1e-08
        X = np.float32(eo['pdist-double-inp'])
        Y_right = eo['pdist-cosine']
        Y_test1 = wpdist(X, 'cosine')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_cosine_random_nonC(self):
        # Test pdist(X, 'test_cosine') [the non-C implementation]
        eps = 1e-08
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-cosine']
        Y_test2 = wpdist(X, 'test_cosine')
        _assert_within_tol(Y_test2, Y_right, eps)

    @pytest.mark.slow
    def test_pdist_cosine_iris(self):
        eps = 1e-08
        X = eo['iris']
        Y_right = eo['pdist-cosine-iris']
        Y_test1 = wpdist(X, 'cosine')
        _assert_within_tol(Y_test1, Y_right, eps)

    @pytest.mark.slow
    def test_pdist_cosine_iris_float32(self):
        eps = 1e-07
        X = np.float32(eo['iris'])
        Y_right = eo['pdist-cosine-iris']
        Y_test1 = wpdist(X, 'cosine')
        _assert_within_tol(Y_test1, Y_right, eps, verbose > 2)

    @pytest.mark.slow
    def test_pdist_cosine_iris_nonC(self):
        eps = 1e-08
        X = eo['iris']
        Y_right = eo['pdist-cosine-iris']
        Y_test2 = wpdist(X, 'test_cosine')
        _assert_within_tol(Y_test2, Y_right, eps)

    def test_pdist_cosine_bounds(self):
        # Test adapted from @joernhees's example at gh-5208: case where
        # cosine distance used to be negative. XXX: very sensitive to the
        # specific norm computation.
        x = np.abs(np.random.RandomState(1337).rand(91))
        X = np.vstack([x, x])
        assert_(wpdist(X, 'cosine')[0] >= 0,
                msg='cosine distance should be non-negative')

    def test_pdist_cityblock_random(self):
        eps = 1e-06
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-cityblock']
        Y_test1 = wpdist_no_const(X, 'cityblock')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_cityblock_random_float32(self):
        eps = 1e-06
        X = np.float32(eo['pdist-double-inp'])
        Y_right = eo['pdist-cityblock']
        Y_test1 = wpdist_no_const(X, 'cityblock')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_cityblock_random_nonC(self):
        eps = 1e-06
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-cityblock']
        Y_test2 = wpdist_no_const(X, 'test_cityblock')
        _assert_within_tol(Y_test2, Y_right, eps)

    @pytest.mark.slow
    def test_pdist_cityblock_iris(self):
        eps = 1e-14
        X = eo['iris']
        Y_right = eo['pdist-cityblock-iris']
        Y_test1 = wpdist_no_const(X, 'cityblock')
        _assert_within_tol(Y_test1, Y_right, eps)

    @pytest.mark.slow
    def test_pdist_cityblock_iris_float32(self):
        eps = 1e-06
        X = np.float32(eo['iris'])
        Y_right = eo['pdist-cityblock-iris']
        Y_test1 = wpdist_no_const(X, 'cityblock')
        _assert_within_tol(Y_test1, Y_right, eps, verbose > 2)

    @pytest.mark.slow
    def test_pdist_cityblock_iris_nonC(self):
        # Test pdist(X, 'test_cityblock') [the non-C implementation] on the
        # Iris data set.
        eps = 1e-14
        X = eo['iris']
        Y_right = eo['pdist-cityblock-iris']
        Y_test2 = wpdist_no_const(X, 'test_cityblock')
        _assert_within_tol(Y_test2, Y_right, eps)

    def test_pdist_correlation_random(self):
        eps = 1e-07
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-correlation']
        Y_test1 = wpdist(X, 'correlation')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_correlation_random_float32(self):
        eps = 1e-07
        X = np.float32(eo['pdist-double-inp'])
        Y_right = eo['pdist-correlation']
        Y_test1 = wpdist(X, 'correlation')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_correlation_random_nonC(self):
        eps = 1e-07
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-correlation']
        Y_test2 = wpdist(X, 'test_correlation')
        _assert_within_tol(Y_test2, Y_right, eps)

    @pytest.mark.slow
    def test_pdist_correlation_iris(self):
        eps = 1e-08
        X = eo['iris']
        Y_right = eo['pdist-correlation-iris']
        Y_test1 = wpdist(X, 'correlation')
        _assert_within_tol(Y_test1, Y_right, eps)

    @pytest.mark.slow
    def test_pdist_correlation_iris_float32(self):
        eps = 1e-07
        X = eo['iris']
        Y_right = np.float32(eo['pdist-correlation-iris'])
        Y_test1 = wpdist(X, 'correlation')
        _assert_within_tol(Y_test1, Y_right, eps, verbose > 2)

    @pytest.mark.slow
    def test_pdist_correlation_iris_nonC(self):
        eps = 1e-08
        X = eo['iris']
        Y_right = eo['pdist-correlation-iris']
        Y_test2 = wpdist(X, 'test_correlation')
        _assert_within_tol(Y_test2, Y_right, eps)

    @pytest.mark.parametrize("p", [1.0, 2.0, 3.2, np.inf])
    def test_pdist_minkowski_random_p(self, p):
        eps = 1e-05
        X = eo['pdist-double-inp']
        Y1 = wpdist_no_const(X, 'minkowski', p=p)
        Y2 = wpdist_no_const(X, 'test_minkowski', p=p)
        _assert_within_tol(Y1, Y2, eps)

    def test_pdist_minkowski_random(self):
        eps = 1e-05
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-minkowski-3.2']
        Y_test1 = wpdist_no_const(X, 'minkowski', p=3.2)
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_minkowski_random_float32(self):
        eps = 1e-05
        X = np.float32(eo['pdist-double-inp'])
        Y_right = eo['pdist-minkowski-3.2']
        Y_test1 = wpdist_no_const(X, 'minkowski', p=3.2)
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_minkowski_random_nonC(self):
        eps = 1e-05
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-minkowski-3.2']
        Y_test2 = wpdist_no_const(X, 'test_minkowski', p=3.2)
        _assert_within_tol(Y_test2, Y_right, eps)

    @pytest.mark.slow
    def test_pdist_minkowski_3_2_iris(self):
        eps = 1e-07
        X = eo['iris']
        Y_right = eo['pdist-minkowski-3.2-iris']
        Y_test1 = wpdist_no_const(X, 'minkowski', p=3.2)
        _assert_within_tol(Y_test1, Y_right, eps)

    @pytest.mark.slow
    def test_pdist_minkowski_3_2_iris_float32(self):
        eps = 1e-06
        X = np.float32(eo['iris'])
        Y_right = eo['pdist-minkowski-3.2-iris']
        Y_test1 = wpdist_no_const(X, 'minkowski', p=3.2)
        _assert_within_tol(Y_test1, Y_right, eps)

    @pytest.mark.slow
    def test_pdist_minkowski_3_2_iris_nonC(self):
        eps = 1e-07
        X = eo['iris']
        Y_right = eo['pdist-minkowski-3.2-iris']
        Y_test2 = wpdist_no_const(X, 'test_minkowski', p=3.2)
        _assert_within_tol(Y_test2, Y_right, eps)

    @pytest.mark.slow
    def test_pdist_minkowski_5_8_iris(self):
        eps = 1e-07
        X = eo['iris']
        Y_right = eo['pdist-minkowski-5.8-iris']
        Y_test1 = wpdist_no_const(X, 'minkowski', p=5.8)
        _assert_within_tol(Y_test1, Y_right, eps)

    @pytest.mark.slow
    def test_pdist_minkowski_5_8_iris_float32(self):
        eps = 1e-06
        X = np.float32(eo['iris'])
        Y_right = eo['pdist-minkowski-5.8-iris']
        Y_test1 = wpdist_no_const(X, 'minkowski', p=5.8)
        _assert_within_tol(Y_test1, Y_right, eps, verbose > 2)

    @pytest.mark.slow
    def test_pdist_minkowski_5_8_iris_nonC(self):
        eps = 1e-07
        X = eo['iris']
        Y_right = eo['pdist-minkowski-5.8-iris']
        Y_test2 = wpdist_no_const(X, 'test_minkowski', p=5.8)
        _assert_within_tol(Y_test2, Y_right, eps)

    def test_pdist_mahalanobis(self):
        # 1-dimensional observations
        x = np.array([2.0, 2.0, 3.0, 5.0]).reshape(-1, 1)
        dist = pdist(x, metric='mahalanobis')
        assert_allclose(dist, [0.0, np.sqrt(0.5), np.sqrt(4.5),
                               np.sqrt(0.5), np.sqrt(4.5), np.sqrt(2.0)])

        # 2-dimensional observations
        x = np.array([[0, 0], [-1, 0], [0, 2], [1, 0], [0, -2]])
        dist = pdist(x, metric='mahalanobis')
        rt2 = np.sqrt(2)
        assert_allclose(dist, [rt2, rt2, rt2, rt2, 2, 2 * rt2, 2, 2, 2 * rt2, 2])

        # Too few observations
        assert_raises(ValueError,
                      wpdist, [[0, 1], [2, 3]], metric='mahalanobis')

    def test_pdist_hamming_random(self):
        eps = 1e-07
        X = eo['pdist-boolean-inp']
        Y_right = eo['pdist-hamming']
        Y_test1 = wpdist(X, 'hamming')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_hamming_random_float32(self):
        eps = 1e-07
        X = np.float32(eo['pdist-boolean-inp'])
        Y_right = eo['pdist-hamming']
        Y_test1 = wpdist(X, 'hamming')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_hamming_random_nonC(self):
        eps = 1e-07
        X = eo['pdist-boolean-inp']
        Y_right = eo['pdist-hamming']
        Y_test2 = wpdist(X, 'test_hamming')
        _assert_within_tol(Y_test2, Y_right, eps)

    def test_pdist_dhamming_random(self):
        eps = 1e-07
        X = np.float64(eo['pdist-boolean-inp'])
        Y_right = eo['pdist-hamming']
        Y_test1 = wpdist(X, 'hamming')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_dhamming_random_float32(self):
        eps = 1e-07
        X = np.float32(eo['pdist-boolean-inp'])
        Y_right = eo['pdist-hamming']
        Y_test1 = wpdist(X, 'hamming')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_dhamming_random_nonC(self):
        eps = 1e-07
        X = np.float64(eo['pdist-boolean-inp'])
        Y_right = eo['pdist-hamming']
        Y_test2 = wpdist(X, 'test_hamming')
        _assert_within_tol(Y_test2, Y_right, eps)

    def test_pdist_jaccard_random(self):
        eps = 1e-08
        X = eo['pdist-boolean-inp']
        Y_right = eo['pdist-jaccard']
        Y_test1 = wpdist(X, 'jaccard')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_jaccard_random_float32(self):
        eps = 1e-08
        X = np.float32(eo['pdist-boolean-inp'])
        Y_right = eo['pdist-jaccard']
        Y_test1 = wpdist(X, 'jaccard')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_jaccard_random_nonC(self):
        eps = 1e-08
        X = eo['pdist-boolean-inp']
        Y_right = eo['pdist-jaccard']
        Y_test2 = wpdist(X, 'test_jaccard')
        _assert_within_tol(Y_test2, Y_right, eps)

    def test_pdist_djaccard_random(self):
        eps = 1e-08
        X = np.float64(eo['pdist-boolean-inp'])
        Y_right = eo['pdist-jaccard']
        Y_test1 = wpdist(X, 'jaccard')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_djaccard_random_float32(self):
        eps = 1e-08
        X = np.float32(eo['pdist-boolean-inp'])
        Y_right = eo['pdist-jaccard']
        Y_test1 = wpdist(X, 'jaccard')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_djaccard_allzeros(self):
        eps = 1e-08
        Y = pdist(np.zeros((5, 3)), 'jaccard')
        _assert_within_tol(np.zeros(10), Y, eps)

    def test_pdist_djaccard_random_nonC(self):
        eps = 1e-08
        X = np.float64(eo['pdist-boolean-inp'])
        Y_right = eo['pdist-jaccard']
        Y_test2 = wpdist(X, 'test_jaccard')
        _assert_within_tol(Y_test2, Y_right, eps)

    def test_pdist_jensenshannon_random(self):
        eps = 1e-08
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-jensenshannon']
        Y_test1 = pdist(X, 'jensenshannon')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_jensenshannon_random_float32(self):
        eps = 1e-07
        X = np.float32(eo['pdist-double-inp'])
        Y_right = eo['pdist-jensenshannon']
        Y_test1 = pdist(X, 'jensenshannon')
        _assert_within_tol(Y_test1, Y_right, eps, verbose > 2)

    def test_pdist_jensenshannon_random_nonC(self):
        eps = 1e-08
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-jensenshannon']
        Y_test2 = pdist(X, 'test_jensenshannon')
        _assert_within_tol(Y_test2, Y_right, eps)

    def test_pdist_jensenshannon_iris(self):
        if _is_32bit():
            # Test failing on 32-bit Linux on Azure otherwise, see gh-12810
            eps = 1.5e-10
        else:
            eps = 1e-12

        X = eo['iris']
        Y_right = eo['pdist-jensenshannon-iris']
        Y_test1 = pdist(X, 'jensenshannon')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_jensenshannon_iris_float32(self):
        eps = 1e-06
        X = np.float32(eo['iris'])
        Y_right = eo['pdist-jensenshannon-iris']
        Y_test1 = pdist(X, 'jensenshannon')
        _assert_within_tol(Y_test1, Y_right, eps, verbose > 2)

    def test_pdist_jensenshannon_iris_nonC(self):
        eps = 5e-12
        X = eo['iris']
        Y_right = eo['pdist-jensenshannon-iris']
        Y_test2 = pdist(X, 'test_jensenshannon')
        _assert_within_tol(Y_test2, Y_right, eps)

    def test_pdist_djaccard_allzeros_nonC(self):
        eps = 1e-08
        Y = pdist(np.zeros((5, 3)), 'test_jaccard')
        _assert_within_tol(np.zeros(10), Y, eps)

    def test_pdist_chebyshev_random(self):
        eps = 1e-08
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-chebyshev']
        Y_test1 = pdist(X, 'chebyshev')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_chebyshev_random_float32(self):
        eps = 1e-07
        X = np.float32(eo['pdist-double-inp'])
        Y_right = eo['pdist-chebyshev']
        Y_test1 = pdist(X, 'chebyshev')
        _assert_within_tol(Y_test1, Y_right, eps, verbose > 2)

    def test_pdist_chebyshev_random_nonC(self):
        eps = 1e-08
        X = eo['pdist-double-inp']
        Y_right = eo['pdist-chebyshev']
        Y_test2 = pdist(X, 'test_chebyshev')
        _assert_within_tol(Y_test2, Y_right, eps)

    def test_pdist_chebyshev_iris(self):
        eps = 1e-15
        X = eo['iris']
        Y_right = eo['pdist-chebyshev-iris']
        Y_test1 = pdist(X, 'chebyshev')
        _assert_within_tol(Y_test1, Y_right, eps)

    def test_pdist_chebyshev_iris_float32(self):
        eps = 1e-06
        X = np.float32(eo['iris'])
        Y_right = eo['pdist-chebyshev-iris']
        Y_test1 = pdist(X, 'chebyshev')
        _assert_within_tol(Y_test1, Y_right, eps, verbose > 2)

    def test_pdist_chebyshev_iris_nonC(self):
        eps = 1e-15
        X = eo['iris']
        Y_right = eo['pdist-chebyshev-iris']
        Y_test2 = pdist(X, 'test_chebyshev')
        _assert_within_tol(Y_test2, Y_right, eps)

    def test_pdist_matching_mtica1(self):
        # Test matching(*,*) with mtica example #1 (nums).
        m = wmatching(np.array([1, 0, 1, 1, 0]),
                      np.array([1, 1, 0, 1, 1]))
        m2 = wmatching(np.array([1, 0, 1, 1, 0], dtype=bool),
                       np.array([1, 1, 0, 1, 1], dtype=bool))
        assert_allclose(m, 0.6, rtol=0, atol=1e-10)
        assert_allclose(m2, 0.6, rtol=0, atol=1e-10)

    def test_pdist_matching_mtica2(self):
        # Test matching(*,*) with mtica example #2.
        m = wmatching(np.array([1, 0, 1]),
                     np.array([1, 1, 0]))
        m2 = wmatching(np.array([1, 0, 1], dtype=bool),
                      np.array([1, 1, 0], dtype=bool))
        assert_allclose(m, 2 / 3, rtol=0, atol=1e-10)
        assert_allclose(m2, 2 / 3, rtol=0, atol=1e-10)

    def test_pdist_jaccard_mtica1(self):
        m = wjaccard(np.array([1, 0, 1, 1, 0]),
                     np.array([1, 1, 0, 1, 1]))
        m2 = wjaccard(np.array([1, 0, 1, 1, 0], dtype=bool),
                      np.array([1, 1, 0, 1, 1], dtype=bool))
        assert_allclose(m, 0.6, rtol=0, atol=1e-10)
        assert_allclose(m2, 0.6, rtol=0, atol=1e-10)

    def test_pdist_jaccard_mtica2(self):
        m = wjaccard(np.array([1, 0, 1]),
                     np.array([1, 1, 0]))
        m2 = wjaccard(np.array([1, 0, 1], dtype=bool),
                      np.array([1, 1, 0], dtype=bool))
        assert_allclose(m, 2 / 3, rtol=0, atol=1e-10)
        assert_allclose(m2, 2 / 3, rtol=0, atol=1e-10)

    def test_pdist_yule_mtica1(self):
        m = wyule(np.array([1, 0, 1, 1, 0]),
                  np.array([1, 1, 0, 1, 1]))
        m2 = wyule(np.array([1, 0, 1, 1, 0], dtype=bool),
                   np.array([1, 1, 0, 1, 1], dtype=bool))
        if verbose > 2:
            print(m)
        assert_allclose(m, 2, rtol=0, atol=1e-10)
        assert_allclose(m2, 2, rtol=0, atol=1e-10)

    def test_pdist_yule_mtica2(self):
        m = wyule(np.array([1, 0, 1]),
                  np.array([1, 1, 0]))
        m2 = wyule(np.array([1, 0, 1], dtype=bool),
                   np.array([1, 1, 0], dtype=bool))
        if verbose > 2:
            print(m)
        assert_allclose(m, 2, rtol=0, atol=1e-10)
        assert_allclose(m2, 2, rtol=0, atol=1e-10)

    def test_pdist_dice_mtica1(self):
        m = wdice(np.array([1, 0, 1, 1, 0]),
                  np.array([1, 1, 0, 1, 1]))
        m2 = wdice(np.array([1, 0, 1, 1, 0], dtype=bool),
                   np.array([1, 1, 0, 1, 1], dtype=bool))
        if verbose > 2:
            print(m)
        assert_allclose(m, 3 / 7, rtol=0, atol=1e-10)
        assert_allclose(m2, 3 / 7, rtol=0, atol=1e-10)

    def test_pdist_dice_mtica2(self):
        m = wdice(np.array([1, 0, 1]),
                  np.array([1, 1, 0]))
        m2 = wdice(np.array([1, 0, 1], dtype=bool),
                   np.array([1, 1, 0], dtype=bool))
        if verbose > 2:
            print(m)
        assert_allclose(m, 0.5, rtol=0, atol=1e-10)
        assert_allclose(m2, 0.5, rtol=0, atol=1e-10)

    def test_pdist_sokalsneath_mtica1(self):
        m = sokalsneath(np.array([1, 0, 1, 1, 0]),
                        np.array([1, 1, 0, 1, 1]))
        m2 = sokalsneath(np.array([1, 0, 1, 1, 0], dtype=bool),
                         np.array([1, 1, 0, 1, 1], dtype=bool))
        if verbose > 2:
            print(m)
        assert_allclose(m, 3 / 4, rtol=0, atol=1e-10)
        assert_allclose(m2, 3 / 4, rtol=0, atol=1e-10)

    def test_pdist_sokalsneath_mtica2(self):
        m = wsokalsneath(np.array([1, 0, 1]),
                         np.array([1, 1, 0]))
        m2 = wsokalsneath(np.array([1, 0, 1], dtype=bool),
                          np.array([1, 1, 0], dtype=bool))
        if verbose > 2:
            print(m)
        assert_allclose(m, 4 / 5, rtol=0, atol=1e-10)
        assert_allclose(m2, 4 / 5, rtol=0, atol=1e-10)

    def test_pdist_rogerstanimoto_mtica1(self):
        m = wrogerstanimoto(np.array([1, 0, 1, 1, 0]),
                            np.array([1, 1, 0, 1, 1]))
        m2 = wrogerstanimoto(np.array([1, 0, 1, 1, 0], dtype=bool),
                             np.array([1, 1, 0, 1, 1], dtype=bool))
        if verbose > 2:
            print(m)
        assert_allclose(m, 3 / 4, rtol=0, atol=1e-10)
        assert_allclose(m2, 3 / 4, rtol=0, atol=1e-10)

    def test_pdist_rogerstanimoto_mtica2(self):
        m = wrogerstanimoto(np.array([1, 0, 1]),
                            np.array([1, 1, 0]))
        m2 = wrogerstanimoto(np.array([1, 0, 1], dtype=bool),
                             np.array([1, 1, 0], dtype=bool))
        if verbose > 2:
            print(m)
        assert_allclose(m, 4 / 5, rtol=0, atol=1e-10)
        assert_allclose(m2, 4 / 5, rtol=0, atol=1e-10)

    def test_pdist_russellrao_mtica1(self):
        m = wrussellrao(np.array([1, 0, 1, 1, 0]),
                        np.array([1, 1, 0, 1, 1]))
        m2 = wrussellrao(np.array([1, 0, 1, 1, 0], dtype=bool),
                         np.array([1, 1, 0, 1, 1], dtype=bool))
        if verbose > 2:
            print(m)
        assert_allclose(m, 3 / 5, rtol=0, atol=1e-10)
        assert_allclose(m2, 3 / 5, rtol=0, atol=1e-10)

    def test_pdist_russellrao_mtica2(self):
        m = wrussellrao(np.array([1, 0, 1]),
                        np.array([1, 1, 0]))
        m2 = wrussellrao(np.array([1, 0, 1], dtype=bool),
                         np.array([1, 1, 0], dtype=bool))
        if verbose > 2:
            print(m)
        assert_allclose(m, 2 / 3, rtol=0, atol=1e-10)
        assert_allclose(m2, 2 / 3, rtol=0, atol=1e-10)

    @pytest.mark.slow
    def test_pdist_canberra_match(self):
        D = eo['iris']
        if verbose > 2:
            print(D.shape, D.dtype)
        eps = 1e-10
        y1 = wpdist_no_const(D, "canberra")
        y2 = wpdist_no_const(D, "test_canberra")
        _assert_within_tol(y1, y2, eps, verbose > 2)

    def test_pdist_canberra_ticket_711(self):
        # Test pdist(X, 'canberra') to see if Canberra gives the right result
        # as reported on gh-1238.
        eps = 1e-8
        pdist_y = wpdist_no_const(([3.3], [3.4]), "canberra")
        right_y = 0.01492537
        _assert_within_tol(pdist_y, right_y, eps, verbose > 2)

    def test_pdist_custom_notdouble(self):
        # tests that when using a custom metric the data type is not altered
        class myclass:
            pass

        def _my_metric(x, y):
            if not isinstance(x[0], myclass) or not isinstance(y[0], myclass):
                raise ValueError("Type has been changed")
            return 1.123
        data = np.array([[myclass()], [myclass()]], dtype=object)
        pdist_y = pdist(data, metric=_my_metric)
        right_y = 1.123
        assert_equal(pdist_y, right_y, verbose=verbose > 2)

    def _check_calling_conventions(self, X, metric, eps=1e-07, **kwargs):
        # helper function for test_pdist_calling_conventions
        try:
            y1 = pdist(X, metric=metric, **kwargs)
            y2 = pdist(X, metric=eval(metric), **kwargs)
            y3 = pdist(X, metric="test_" + metric, **kwargs)
        except Exception as e:
            e_cls = e.__class__
            if verbose > 2:
                print(e_cls.__name__)
                print(e)
            assert_raises(e_cls, pdist, X, metric=metric, **kwargs)
            assert_raises(e_cls, pdist, X, metric=eval(metric), **kwargs)
            assert_raises(e_cls, pdist, X, metric="test_" + metric, **kwargs)
        else:
            _assert_within_tol(y1, y2, rtol=eps, verbose_=verbose > 2)
            _assert_within_tol(y1, y3, rtol=eps, verbose_=verbose > 2)

    def test_pdist_calling_conventions(self):
        # Ensures that specifying the metric with a str or scipy function
        # gives the same behaviour (i.e. same result or same exception).
        # NOTE: The correctness should be checked within each metric tests.
        # NOTE: Extra args should be checked with a dedicated test
        for eo_name in self.rnd_eo_names:
            # subsampling input data to speed-up tests
            # NOTE: num samples needs to be > than dimensions for mahalanobis
            X = eo[eo_name][::5, ::2]
            for metric in _METRICS_NAMES:
                if metric == 'wminkowski':
                    continue
                if verbose > 2:
                    print("testing: ", metric, " with: ", eo_name)
                if metric in {'dice', 'yule', 'kulsinski', 'matching',
                              'rogerstanimoto', 'russellrao', 'sokalmichener',
                              'sokalsneath'} and 'bool' not in eo_name:
                    # python version permits non-bools e.g. for fuzzy logic
                    continue
                self._check_calling_conventions(X, metric)

                # Testing built-in metrics with extra args
                if metric == "seuclidean":
                    V = np.var(X.astype(np.double), axis=0, ddof=1)
                    self._check_calling_conventions(X, metric, V=V)
                elif metric == "mahalanobis":
                    V = np.atleast_2d(np.cov(X.astype(np.double).T))
                    VI = np.array(np.linalg.inv(V).T)
                    self._check_calling_conventions(X, metric, VI=VI)

    def test_pdist_dtype_equivalence(self):
        # Tests that the result is not affected by type up-casting
        eps = 1e-07
        tests = [(eo['random-bool-data'], self.valid_upcasts['bool']),
                 (eo['random-uint-data'], self.valid_upcasts['uint']),
                 (eo['random-int-data'], self.valid_upcasts['int']),
                 (eo['random-float32-data'], self.valid_upcasts['float32'])]
        for metric in _METRICS_NAMES:
            for test in tests:
                X1 = test[0][::5, ::2]
                try:
                    y1 = pdist(X1, metric=metric)
                except Exception as e:
                    e_cls = e.__class__
                    if verbose > 2:
                        print(e_cls.__name__)
                        print(e)
                    for new_type in test[1]:
                        X2 = new_type(X1)
                        assert_raises(e_cls, pdist, X2, metric=metric)
                else:
                    for new_type in test[1]:
                        y2 = pdist(new_type(X1), metric=metric)
                        _assert_within_tol(y1, y2, eps, verbose > 2)

    def test_pdist_out(self):
        # Test that out parameter works properly
        eps = 1e-07
        X = eo['random-float32-data'][::5, ::2]
        out_size = int((X.shape[0] * (X.shape[0] - 1)) / 2)
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "'wminkowski' metric is deprecated")
            for metric in _METRICS_NAMES:
                kwargs = dict()
                if metric in ['minkowski', 'wminkowski']:
                    kwargs['p'] = 1.23
                if metric == 'wminkowski':
                    kwargs['w'] = 1.0 / X.std(axis=0)
                out1 = np.empty(out_size, dtype=np.double)
                Y_right = pdist(X, metric, **kwargs)
                Y_test1 = pdist(X, metric, out=out1, **kwargs)
                # test that output is numerically equivalent
                _assert_within_tol(Y_test1, Y_right, eps)
                # test that Y_test1 and out1 are the same object
                assert_(Y_test1 is out1)
                # test for incorrect shape
                out2 = np.empty(out_size + 3, dtype=np.double)
                assert_raises(ValueError, pdist, X, metric, out=out2, **kwargs)
                # test for (C-)contiguous output
                out3 = np.empty(2 * out_size, dtype=np.double)[::2]
                assert_raises(ValueError, pdist, X, metric, out=out3, **kwargs)
                # test for incorrect dtype
                out5 = np.empty(out_size, dtype=np.int64)
                assert_raises(ValueError, pdist, X, metric, out=out5, **kwargs)

    def test_striding(self):
        # test that striding is handled correct with calls to
        # _copy_array_if_base_present
        eps = 1e-07
        X = eo['random-float32-data'][::5, ::2]
        X_copy = X.copy()

        # confirm contiguity
        assert_(not X.flags.c_contiguous)
        assert_(X_copy.flags.c_contiguous)

        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning,
                       message="'wminkowski' metric is deprecated")
            for metric in _METRICS_NAMES:
                kwargs = dict()
                if metric in ['minkowski', 'wminkowski']:
                    kwargs['p'] = 1.23
                if metric == 'wminkowski':
                    kwargs['w'] = 1.0 / X.std(axis=0)
                Y1 = pdist(X, metric, **kwargs)
                Y2 = pdist(X_copy, metric, **kwargs)
                # test that output is numerically equivalent
                _assert_within_tol(Y1, Y2, eps, verbose > 2)

class TestSomeDistanceFunctions:

    def setup_method(self):
        # 1D arrays
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 1.0, 5.0])

        self.cases = [(x, y)]

    def test_minkowski(self):
        for x, y in self.cases:
            dist1 = wminkowski(x, y, p=1)
            assert_almost_equal(dist1, 3.0)
            dist1p5 = wminkowski(x, y, p=1.5)
            assert_almost_equal(dist1p5, (1.0 + 2.0**1.5)**(2. / 3))
            wminkowski(x, y, p=2)

        # Check that casting input to minimum scalar type doesn't affect result
        # (issue #10262). This could be extended to more test inputs with
        # np.min_scalar_type(np.max(input_matrix)).
        a = np.array([352, 916])
        b = np.array([350, 660])
        assert_equal(minkowski(a, b),
                     minkowski(a.astype('uint16'), b.astype('uint16')))

    def test_old_wminkowski(self):
        with suppress_warnings() as wrn:
            wrn.filter(DeprecationWarning,
                       message=".*wminkowski is deprecated")
            w = np.array([1.0, 2.0, 0.5])
            for x, y in self.cases:
                dist1 = old_wminkowski(x, y, p=1, w=w)
                assert_almost_equal(dist1, 3.0)
                dist1p5 = old_wminkowski(x, y, p=1.5, w=w)
                assert_almost_equal(dist1p5, (2.0**1.5+1.0)**(2./3))
                dist2 = old_wminkowski(x, y, p=2, w=w)
                assert_almost_equal(dist2, np.sqrt(5))

            # test weights Issue #7893
            arr = np.arange(4)
            w = np.full_like(arr, 4)
            assert_almost_equal(old_wminkowski(arr, arr + 1, p=2, w=w), 8.0)
            assert_almost_equal(wminkowski(arr, arr + 1, p=2, w=w), 4.0)

    def test_euclidean(self):
        for x, y in self.cases:
            dist = weuclidean(x, y)
            assert_almost_equal(dist, np.sqrt(5))

    def test_sqeuclidean(self):
        for x, y in self.cases:
            dist = wsqeuclidean(x, y)
            assert_almost_equal(dist, 5.0)

    def test_cosine(self):
        for x, y in self.cases:
            dist = wcosine(x, y)
            assert_almost_equal(dist, 1.0 - 18.0 / (np.sqrt(14) * np.sqrt(27)))

    def test_correlation(self):
        xm = np.array([-1.0, 0, 1.0])
        ym = np.array([-4.0 / 3, -4.0 / 3, 5.0 - 7.0 / 3])
        for x, y in self.cases:
            dist = wcorrelation(x, y)
            assert_almost_equal(dist, 1.0 - np.dot(xm, ym) / (norm(xm) * norm(ym)))

    def test_correlation_positive(self):
        # Regression test for gh-12320 (negative return value due to rounding
        x = np.array([0., 0., 0., 0., 0., 0., -2., 0., 0., 0., -2., -2., -2.,
                      0., -2., 0., -2., 0., 0., -1., -2., 0., 1., 0., 0., -2.,
                      0., 0., -2., 0., -2., -2., -2., -2., -2., -2., 0.])
        y = np.array([1., 1., 1., 1., 1., 1., -1., 1., 1., 1., -1., -1., -1.,
                      1., -1., 1., -1., 1., 1., 0., -1., 1., 2., 1., 1., -1.,
                      1., 1., -1., 1., -1., -1., -1., -1., -1., -1., 1.])
        dist = correlation(x, y)
        assert 0 <= dist <= 10 * np.finfo(np.float64).eps

    def test_mahalanobis(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 1.0, 5.0])
        vi = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
        for x, y in self.cases:
            dist = mahalanobis(x, y, vi)
            assert_almost_equal(dist, np.sqrt(6.0))


def construct_squeeze_tests():
    # Construct a class like TestSomeDistanceFunctions but testing 2-d vectors
    # with a length-1 dimension which is deprecated
    def setup_method(self):
        # 1D arrays
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 1.0, 5.0])
        # 3x1 arrays
        x31 = x[:, np.newaxis]
        y31 = y[:, np.newaxis]
        # 1x3 arrays
        x13 = x31.T
        y13 = y31.T

        self.cases = [(x31, y31), (x13, y13), (x31, y13)]

    sup = suppress_warnings()
    sup.filter(DeprecationWarning,
            ".*distance metrics ignoring length-1 dimensions is deprecated.*")
    base = TestSomeDistanceFunctions
    attrs = {
        name: sup(getattr(base, name))
        for name in dir(base)
        if name.startswith('test_')
    }
    attrs['setup_method'] = setup_method
    name = 'TestDistanceFunctionsSqueeze'
    globals()[name] = type(name, (base,), attrs)


construct_squeeze_tests()


class TestSquareForm:
    checked_dtypes = [np.float64, np.float32, np.int32, np.int8, bool]

    def test_squareform_matrix(self):
        for dtype in self.checked_dtypes:
            self.check_squareform_matrix(dtype)

    def test_squareform_vector(self):
        for dtype in self.checked_dtypes:
            self.check_squareform_vector(dtype)

    def check_squareform_matrix(self, dtype):
        A = np.zeros((0, 0), dtype=dtype)
        rA = squareform(A)
        assert_equal(rA.shape, (0,))
        assert_equal(rA.dtype, dtype)

        A = np.zeros((1, 1), dtype=dtype)
        rA = squareform(A)
        assert_equal(rA.shape, (0,))
        assert_equal(rA.dtype, dtype)

        A = np.array([[0, 4.2], [4.2, 0]], dtype=dtype)
        rA = squareform(A)
        assert_equal(rA.shape, (1,))
        assert_equal(rA.dtype, dtype)
        assert_array_equal(rA, np.array([4.2], dtype=dtype))

    def check_squareform_vector(self, dtype):
        v = np.zeros((0,), dtype=dtype)
        rv = squareform(v)
        assert_equal(rv.shape, (1, 1))
        assert_equal(rv.dtype, dtype)
        assert_array_equal(rv, [[0]])

        v = np.array([8.3], dtype=dtype)
        rv = squareform(v)
        assert_equal(rv.shape, (2, 2))
        assert_equal(rv.dtype, dtype)
        assert_array_equal(rv, np.array([[0, 8.3], [8.3, 0]], dtype=dtype))

    def test_squareform_multi_matrix(self):
        for n in range(2, 5):
            self.check_squareform_multi_matrix(n)

    def check_squareform_multi_matrix(self, n):
        X = np.random.rand(n, 4)
        Y = wpdist_no_const(X)
        assert_equal(len(Y.shape), 1)
        A = squareform(Y)
        Yr = squareform(A)
        s = A.shape
        k = 0
        if verbose >= 3:
            print(A.shape, Y.shape, Yr.shape)
        assert_equal(len(s), 2)
        assert_equal(len(Yr.shape), 1)
        assert_equal(s[0], s[1])
        for i in range(0, s[0]):
            for j in range(i + 1, s[1]):
                if i != j:
                    assert_equal(A[i, j], Y[k])
                    k += 1
                else:
                    assert_equal(A[i, j], 0)


class TestNumObsY:

    def test_num_obs_y_multi_matrix(self):
        for n in range(2, 10):
            X = np.random.rand(n, 4)
            Y = wpdist_no_const(X)
            assert_equal(num_obs_y(Y), n)

    def test_num_obs_y_1(self):
        # Tests num_obs_y(y) on a condensed distance matrix over 1
        # observations. Expecting exception.
        assert_raises(ValueError, self.check_y, 1)

    def test_num_obs_y_2(self):
        # Tests num_obs_y(y) on a condensed distance matrix over 2
        # observations.
        assert_(self.check_y(2))

    def test_num_obs_y_3(self):
        assert_(self.check_y(3))

    def test_num_obs_y_4(self):
        assert_(self.check_y(4))

    def test_num_obs_y_5_10(self):
        for i in range(5, 16):
            self.minit(i)

    def test_num_obs_y_2_100(self):
        # Tests num_obs_y(y) on 100 improper condensed distance matrices.
        # Expecting exception.
        a = set([])
        for n in range(2, 16):
            a.add(n * (n - 1) / 2)
        for i in range(5, 105):
            if i not in a:
                assert_raises(ValueError, self.bad_y, i)

    def minit(self, n):
        assert_(self.check_y(n))

    def bad_y(self, n):
        y = np.random.rand(n)
        return num_obs_y(y)

    def check_y(self, n):
        return num_obs_y(self.make_y(n)) == n

    def make_y(self, n):
        return np.random.rand((n * (n - 1)) // 2)


class TestNumObsDM:

    def test_num_obs_dm_multi_matrix(self):
        for n in range(1, 10):
            X = np.random.rand(n, 4)
            Y = wpdist_no_const(X)
            A = squareform(Y)
            if verbose >= 3:
                print(A.shape, Y.shape)
            assert_equal(num_obs_dm(A), n)

    def test_num_obs_dm_0(self):
        # Tests num_obs_dm(D) on a 0x0 distance matrix. Expecting exception.
        assert_(self.check_D(0))

    def test_num_obs_dm_1(self):
        # Tests num_obs_dm(D) on a 1x1 distance matrix.
        assert_(self.check_D(1))

    def test_num_obs_dm_2(self):
        assert_(self.check_D(2))

    def test_num_obs_dm_3(self):
        assert_(self.check_D(2))

    def test_num_obs_dm_4(self):
        assert_(self.check_D(4))

    def check_D(self, n):
        return num_obs_dm(self.make_D(n)) == n

    def make_D(self, n):
        return np.random.rand(n, n)


def is_valid_dm_throw(D):
    return is_valid_dm(D, throw=True)


class TestIsValidDM:

    def test_is_valid_dm_improper_shape_1D_E(self):
        D = np.zeros((5,), dtype=np.double)
        assert_raises(ValueError, is_valid_dm_throw, (D))

    def test_is_valid_dm_improper_shape_1D_F(self):
        D = np.zeros((5,), dtype=np.double)
        assert_equal(is_valid_dm(D), False)

    def test_is_valid_dm_improper_shape_3D_E(self):
        D = np.zeros((3, 3, 3), dtype=np.double)
        assert_raises(ValueError, is_valid_dm_throw, (D))

    def test_is_valid_dm_improper_shape_3D_F(self):
        D = np.zeros((3, 3, 3), dtype=np.double)
        assert_equal(is_valid_dm(D), False)

    def test_is_valid_dm_nonzero_diagonal_E(self):
        y = np.random.rand(10)
        D = squareform(y)
        for i in range(0, 5):
            D[i, i] = 2.0
        assert_raises(ValueError, is_valid_dm_throw, (D))

    def test_is_valid_dm_nonzero_diagonal_F(self):
        y = np.random.rand(10)
        D = squareform(y)
        for i in range(0, 5):
            D[i, i] = 2.0
        assert_equal(is_valid_dm(D), False)

    def test_is_valid_dm_asymmetric_E(self):
        y = np.random.rand(10)
        D = squareform(y)
        D[1, 3] = D[3, 1] + 1
        assert_raises(ValueError, is_valid_dm_throw, (D))

    def test_is_valid_dm_asymmetric_F(self):
        y = np.random.rand(10)
        D = squareform(y)
        D[1, 3] = D[3, 1] + 1
        assert_equal(is_valid_dm(D), False)

    def test_is_valid_dm_correct_1_by_1(self):
        D = np.zeros((1, 1), dtype=np.double)
        assert_equal(is_valid_dm(D), True)

    def test_is_valid_dm_correct_2_by_2(self):
        y = np.random.rand(1)
        D = squareform(y)
        assert_equal(is_valid_dm(D), True)

    def test_is_valid_dm_correct_3_by_3(self):
        y = np.random.rand(3)
        D = squareform(y)
        assert_equal(is_valid_dm(D), True)

    def test_is_valid_dm_correct_4_by_4(self):
        y = np.random.rand(6)
        D = squareform(y)
        assert_equal(is_valid_dm(D), True)

    def test_is_valid_dm_correct_5_by_5(self):
        y = np.random.rand(10)
        D = squareform(y)
        assert_equal(is_valid_dm(D), True)


def is_valid_y_throw(y):
    return is_valid_y(y, throw=True)


class TestIsValidY:
    # If test case name ends on "_E" then an exception is expected for the
    # given input, if it ends in "_F" then False is expected for the is_valid_y
    # check.  Otherwise the input is expected to be valid.

    def test_is_valid_y_improper_shape_2D_E(self):
        y = np.zeros((3, 3,), dtype=np.double)
        assert_raises(ValueError, is_valid_y_throw, (y))

    def test_is_valid_y_improper_shape_2D_F(self):
        y = np.zeros((3, 3,), dtype=np.double)
        assert_equal(is_valid_y(y), False)

    def test_is_valid_y_improper_shape_3D_E(self):
        y = np.zeros((3, 3, 3), dtype=np.double)
        assert_raises(ValueError, is_valid_y_throw, (y))

    def test_is_valid_y_improper_shape_3D_F(self):
        y = np.zeros((3, 3, 3), dtype=np.double)
        assert_equal(is_valid_y(y), False)

    def test_is_valid_y_correct_2_by_2(self):
        y = self.correct_n_by_n(2)
        assert_equal(is_valid_y(y), True)

    def test_is_valid_y_correct_3_by_3(self):
        y = self.correct_n_by_n(3)
        assert_equal(is_valid_y(y), True)

    def test_is_valid_y_correct_4_by_4(self):
        y = self.correct_n_by_n(4)
        assert_equal(is_valid_y(y), True)

    def test_is_valid_y_correct_5_by_5(self):
        y = self.correct_n_by_n(5)
        assert_equal(is_valid_y(y), True)

    def test_is_valid_y_2_100(self):
        a = set([])
        for n in range(2, 16):
            a.add(n * (n - 1) / 2)
        for i in range(5, 105):
            if i not in a:
                assert_raises(ValueError, self.bad_y, i)

    def bad_y(self, n):
        y = np.random.rand(n)
        return is_valid_y(y, throw=True)

    def correct_n_by_n(self, n):
        y = np.random.rand((n * (n - 1)) // 2)
        return y


def test_bad_p():
    # Raise ValueError if p < 1.
    p = 0.5
    assert_raises(ValueError, wminkowski, [1, 2], [3, 4], p)
    assert_raises(ValueError, wminkowski, [1, 2], [3, 4], p, [1, 1])


def test_sokalsneath_all_false():
    # Regression test for ticket #876
    assert_raises(ValueError, sokalsneath, [False, False, False], [False, False, False])


def test_canberra():
    # Regression test for ticket #1430.
    assert_equal(wcanberra([1, 2, 3], [2, 4, 6]), 1)
    assert_equal(wcanberra([1, 1, 0, 0], [1, 0, 1, 0]), 2)


def test_braycurtis():
    # Regression test for ticket #1430.
    assert_almost_equal(wbraycurtis([1, 2, 3], [2, 4, 6]), 1. / 3, decimal=15)
    assert_almost_equal(wbraycurtis([1, 1, 0, 0], [1, 0, 1, 0]), 0.5, decimal=15)


def test_euclideans():
    # Regression test for ticket #1328.
    x1 = np.array([1, 1, 1])
    x2 = np.array([0, 0, 0])

    # Basic test of the calculation.
    assert_almost_equal(wsqeuclidean(x1, x2), 3.0, decimal=14)
    assert_almost_equal(weuclidean(x1, x2), np.sqrt(3), decimal=14)

    # Check flattening for (1, N) or (N, 1) inputs
    with pytest.warns(DeprecationWarning,
                      match="ignoring length-1 dimensions is deprecated"):
        assert_almost_equal(weuclidean(x1[np.newaxis, :], x2[np.newaxis, :]),
                            np.sqrt(3), decimal=14)
    with pytest.warns(DeprecationWarning,
                      match="ignoring length-1 dimensions is deprecated"):
        assert_almost_equal(wsqeuclidean(x1[np.newaxis, :], x2[np.newaxis, :]),
                            3.0, decimal=14)
    with pytest.warns(DeprecationWarning,
                      match="ignoring length-1 dimensions is deprecated"):
        assert_almost_equal(wsqeuclidean(x1[:, np.newaxis], x2[:, np.newaxis]),
                            3.0, decimal=14)

    # Distance metrics only defined for vectors (= 1-D)
    x = np.arange(4).reshape(2, 2)
    assert_raises(ValueError, weuclidean, x, x)
    assert_raises(ValueError, wsqeuclidean, x, x)

    # Another check, with random data.
    rs = np.random.RandomState(1234567890)
    x = rs.rand(10)
    y = rs.rand(10)
    d1 = weuclidean(x, y)
    d2 = wsqeuclidean(x, y)
    assert_almost_equal(d1**2, d2, decimal=14)


def test_hamming_unequal_length():
    # Regression test for gh-4290.
    x = [0, 0, 1]
    y = [1, 0, 1, 0]
    # Used to give an AttributeError from ndarray.mean called on bool
    assert_raises(ValueError, whamming, x, y)


def test_hamming_string_array():
    # https://github.com/scikit-learn/scikit-learn/issues/4014
    a = np.array(['eggs', 'spam', 'spam', 'eggs', 'spam', 'spam', 'spam',
                  'spam', 'spam', 'spam', 'spam', 'eggs', 'eggs', 'spam',
                  'eggs', 'eggs', 'eggs', 'eggs', 'eggs', 'spam'],
                  dtype='|S4')
    b = np.array(['eggs', 'spam', 'spam', 'eggs', 'eggs', 'spam', 'spam',
                  'spam', 'spam', 'eggs', 'spam', 'eggs', 'spam', 'eggs',
                  'spam', 'spam', 'eggs', 'spam', 'spam', 'eggs'],
                  dtype='|S4')
    desired = 0.45
    assert_allclose(whamming(a, b), desired)


def test_minkowski_w():
    # Regression test for gh-8142.
    arr_in = np.array([[83.33333333, 100., 83.33333333, 100., 36.,
                        60., 90., 150., 24., 48.],
                       [83.33333333, 100., 83.33333333, 100., 36.,
                        60., 90., 150., 24., 48.]])
    p0 = pdist(arr_in, metric='minkowski', p=1, w=None)
    c0 = cdist(arr_in, arr_in, metric='minkowski', p=1, w=None)
    p1 = pdist(arr_in, metric='minkowski', p=1)
    c1 = cdist(arr_in, arr_in, metric='minkowski', p=1)

    assert_allclose(p0, p1, rtol=1e-15)
    assert_allclose(c0, c1, rtol=1e-15)


def test_sqeuclidean_dtypes():
    # Assert that sqeuclidean returns the right types of values.
    # Integer types should be converted to floating for stability.
    # Floating point types should be the same as the input.
    x = [1, 2, 3]
    y = [4, 5, 6]

    for dtype in [np.int8, np.int16, np.int32, np.int64]:
        d = wsqeuclidean(np.asarray(x, dtype=dtype), np.asarray(y, dtype=dtype))
        assert_(np.issubdtype(d.dtype, np.floating))

    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        d1 = wsqeuclidean([0], np.asarray([-1], dtype=dtype))
        d2 = wsqeuclidean(np.asarray([-1], dtype=dtype), [0])

        assert_equal(d1, d2)
        assert_equal(d1, np.float64(np.iinfo(dtype).max)**2)

    dtypes = [np.float32, np.float64, np.complex64, np.complex128]
    for dtype in ['float16', 'float128']:
        # These aren't present in older numpy versions; float128 may also not
        # be present on all platforms.
        if hasattr(np, dtype):
            dtypes.append(getattr(np, dtype))

    for dtype in dtypes:
        d = wsqeuclidean(np.asarray(x, dtype=dtype), np.asarray(y, dtype=dtype))
        assert_equal(d.dtype, dtype)


def test_sokalmichener():
    # Test that sokalmichener has the same result for bool and int inputs.
    p = [True, True, False]
    q = [True, False, True]
    x = [int(b) for b in p]
    y = [int(b) for b in q]
    dist1 = sokalmichener(p, q)
    dist2 = sokalmichener(x, y)
    # These should be exactly the same.
    assert_equal(dist1, dist2)


def test_sokalmichener_with_weight():
    # from: | 1 |   | 0 |
    # to:   | 1 |   | 1 |
    # weight|   | 1 |   | 0.2
    ntf = 0 * 1 + 0 * 0.2
    nft = 0 * 1 + 1 * 0.2
    ntt = 1 * 1 + 0 * 0.2
    nff = 0 * 1 + 0 * 0.2
    expected = 2 * (nft + ntf) / (ntt + nff + 2 * (nft + ntf))
    assert_almost_equal(expected, 0.2857143)
    actual = sokalmichener([1, 0], [1, 1], w=[1, 0.2])
    assert_almost_equal(expected, actual)

    a1 = [False, False, True, True, True, False, False, True, True, True, True,
          True, True, False, True, False, False, False, True, True]
    a2 = [True, True, True, False, False, True, True, True, False, True,
          True, True, True, True, False, False, False, True, True, True]

    for w in [0.05, 0.1, 1.0, 20.0]:
        assert_almost_equal(sokalmichener(a2, a1, [w]), 0.6666666666666666)


def test_modifies_input():
    # test whether cdist or pdist modifies input arrays
    X1 = np.asarray([[1., 2., 3.],
                     [1.2, 2.3, 3.4],
                     [2.2, 2.3, 4.4],
                     [22.2, 23.3, 44.4]])
    X1_copy = X1.copy()
    with suppress_warnings() as w:
        w.filter(message="'wminkowski' metric is deprecated")
        for metric in _METRICS_NAMES:
            kwargs = {"w": 1.0 / X1.std(axis=0)} if metric == "wminkowski" else {}
            cdist(X1, X1, metric, **kwargs)
            pdist(X1, metric, **kwargs)
            assert_array_equal(X1, X1_copy)


def test_Xdist_deprecated_args():
    # testing both cdist and pdist deprecated warnings
    X1 = np.asarray([[1., 2., 3.],
                     [1.2, 2.3, 3.4],
                     [2.2, 2.3, 4.4],
                     [22.2, 23.3, 44.4]])
    weights = np.arange(3)
    for metric in _METRICS_NAMES:
        kwargs = {"w": weights} if metric == "wminkowski" else dict()
        with suppress_warnings() as w:
            w.filter(DeprecationWarning,
                    message="'wminkowski' metric is deprecated")
            with pytest.raises(TypeError):
                cdist(X1, X1, metric, 2., **kwargs)

            with pytest.raises(TypeError):
                pdist(X1, metric, 2., **kwargs)

        for arg in ["p", "V", "VI"]:
            kwargs = {arg:"foo"}

            if metric == "wminkowski":
                if "p" in kwargs or "w" in kwargs:
                    continue
                kwargs["w"] = weights

            if((arg == "V" and metric == "seuclidean") or
            (arg == "VI" and metric == "mahalanobis") or
            (arg == "p" and metric == "minkowski")):
                continue

            with suppress_warnings() as w:
                w.filter(DeprecationWarning,
                        message="'wminkowski' metric is deprecated")
                with pytest.raises(TypeError):
                    cdist(X1, X1, metric, **kwargs)

                with pytest.raises(TypeError):
                    pdist(X1, metric, **kwargs)


def test_Xdist_non_negative_weights():
    X = eo['random-float32-data'][::5, ::2]
    w = np.ones(X.shape[1])
    w[::5] = -w[::5]
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning,
                   message="'wminkowski' metric is deprecated")
        for metric in _METRICS_NAMES:
            if metric in ['seuclidean', 'mahalanobis', 'jensenshannon']:
                continue

            for m in [metric, eval(metric), "test_" + metric]:
                assert_raises(ValueError, pdist, X, m, w=w)
                assert_raises(ValueError, cdist, X, X, m, w=w)


def test__validate_vector():
    x = [1, 2, 3]
    y = _validate_vector(x)
    assert_array_equal(y, x)

    y = _validate_vector(x, dtype=np.float64)
    assert_array_equal(y, x)
    assert_equal(y.dtype, np.float64)

    x = [1]
    y = _validate_vector(x)
    assert_equal(y.ndim, 1)
    assert_equal(y, x)

    x = 1
    with pytest.warns(DeprecationWarning,
                      match="ignoring length-1 dimensions is deprecated"):
        y = _validate_vector(x)
    assert_equal(y.ndim, 1)
    assert_equal(y, [x])

    x = np.arange(5).reshape(1, -1, 1)
    with pytest.warns(DeprecationWarning,
                      match="ignoring length-1 dimensions is deprecated"):
        y = _validate_vector(x)
    assert_equal(y.ndim, 1)
    assert_array_equal(y, x[0, :, 0])

    x = [[1, 2], [3, 4]]
    assert_raises(ValueError, _validate_vector, x)

def test_yule_all_same():
    # Test yule avoids a divide by zero when exactly equal
    x = np.ones((2, 6), dtype=bool)
    d = wyule(x[0], x[0])
    assert d == 0.0

    d = pdist(x, 'yule')
    assert_equal(d, [0.0])

    d = cdist(x[:1], x[:1], 'yule')
    assert_equal(d, [[0.0]])


def test_jensenshannon():
    assert_almost_equal(jensenshannon([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0),
                        1.0)
    assert_almost_equal(jensenshannon([1.0, 0.0], [0.5, 0.5]),
                        0.46450140402245893)
    assert_almost_equal(jensenshannon([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]), 0.0)

    assert_almost_equal(jensenshannon([[1.0, 2.0]], [[0.5, 1.5]], axis=0),
                        [0.0, 0.0])
    assert_almost_equal(jensenshannon([[1.0, 2.0]], [[0.5, 1.5]], axis=1),
                        [0.0649045])
    assert_almost_equal(jensenshannon([[1.0, 2.0]], [[0.5, 1.5]], axis=0,
                                      keepdims=True), [[0.0, 0.0]])
    assert_almost_equal(jensenshannon([[1.0, 2.0]], [[0.5, 1.5]], axis=1,
                                      keepdims=True), [[0.0649045]])

    a = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
    b = np.array([[13, 14, 15, 16],
                  [17, 18, 19, 20],
                  [21, 22, 23, 24]])

    assert_almost_equal(jensenshannon(a, b, axis=0),
                        [0.1954288, 0.1447697, 0.1138377, 0.0927636])
    assert_almost_equal(jensenshannon(a, b, axis=1),
                        [0.1402339, 0.0399106, 0.0201815])
