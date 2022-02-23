from itertools import product

import pytest
import re
import numpy as np
from scipy.sparse import (
    bsr_matrix,
    coo_matrix,
    csc_matrix,
    csr_matrix,
    dok_matrix,
    lil_matrix,
    issparse,
)

from sklearn import metrics
from sklearn import neighbors, datasets
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import EfficiencyWarning
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import VALID_METRICS_SPARSE, VALID_METRICS
from sklearn.neighbors._base import _is_sorted_by_data, _check_precomputed
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.validation import check_random_state
from sklearn.utils.fixes import sp_version, parse_version

import joblib

rng = np.random.RandomState(0)
# load and shuffle iris dataset
iris = datasets.load_iris()
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# load and shuffle digits
digits = datasets.load_digits()
perm = rng.permutation(digits.target.size)
digits.data = digits.data[perm]
digits.target = digits.target[perm]

SPARSE_TYPES = (bsr_matrix, coo_matrix, csc_matrix, csr_matrix, dok_matrix, lil_matrix)
SPARSE_OR_DENSE = SPARSE_TYPES + (np.asarray,)

ALGORITHMS = ("ball_tree", "brute", "kd_tree", "auto")
P = (1, 2, 3, 4, np.inf)
JOBLIB_BACKENDS = list(joblib.parallel.BACKENDS.keys())

# Filter deprecation warnings.
neighbors.kneighbors_graph = ignore_warnings(neighbors.kneighbors_graph)
neighbors.radius_neighbors_graph = ignore_warnings(neighbors.radius_neighbors_graph)


def _weight_func(dist):
    """Weight function to replace lambda d: d ** -2.
    The lambda function is not valid because:
    if d==0 then 0^-2 is not valid."""

    # Dist could be multidimensional, flatten it so all values
    # can be looped
    with np.errstate(divide="ignore"):
        retval = 1.0 / dist
    return retval ** 2


def test_unsupervised_kneighbors(
    n_samples=20, n_features=5, n_query_pts=2, n_neighbors=5
):
    # Test unsupervised neighbors methods
    X = rng.rand(n_samples, n_features)

    test = rng.rand(n_query_pts, n_features)

    for p in P:
        results_nodist = []
        results = []

        for algorithm in ALGORITHMS:
            neigh = neighbors.NearestNeighbors(
                n_neighbors=n_neighbors, algorithm=algorithm, p=p
            )
            neigh.fit(X)

            results_nodist.append(neigh.kneighbors(test, return_distance=False))
            results.append(neigh.kneighbors(test, return_distance=True))

        for i in range(len(results) - 1):
            assert_array_almost_equal(results_nodist[i], results[i][1])
            assert_array_almost_equal(results[i][0], results[i + 1][0])
            assert_array_almost_equal(results[i][1], results[i + 1][1])


@pytest.mark.parametrize(
    "NearestNeighbors",
    [
        neighbors.KNeighborsClassifier,
        neighbors.KNeighborsRegressor,
        neighbors.NearestNeighbors,
    ],
)
def test_unsupervised_inputs(NearestNeighbors):
    # Test unsupervised inputs for neighbors estimators

    X = rng.random_sample((10, 3))
    y = rng.randint(3, size=10)
    nbrs_fid = neighbors.NearestNeighbors(n_neighbors=1)
    nbrs_fid.fit(X)

    dist1, ind1 = nbrs_fid.kneighbors(X)

    nbrs = NearestNeighbors(n_neighbors=1)

    for data in (nbrs_fid, neighbors.BallTree(X), neighbors.KDTree(X)):
        nbrs.fit(data, y)

        dist2, ind2 = nbrs.kneighbors(X)

        assert_array_almost_equal(dist1, dist2)
        assert_array_almost_equal(ind1, ind2)


def test_n_neighbors_datatype():
    # Test to check whether n_neighbors is integer
    X = [[1, 1], [1, 1], [1, 1]]
    expected_msg = "n_neighbors does not take .*float.* value, enter integer value"
    msg = "Expected n_neighbors > 0. Got -3"

    neighbors_ = neighbors.NearestNeighbors(n_neighbors=3.0)
    with pytest.raises(TypeError, match=expected_msg):
        neighbors_.fit(X)
    with pytest.raises(ValueError, match=msg):
        neighbors_.kneighbors(X=X, n_neighbors=-3)
    with pytest.raises(TypeError, match=expected_msg):
        neighbors_.kneighbors(X=X, n_neighbors=3.0)


def test_not_fitted_error_gets_raised():
    X = [[1]]
    neighbors_ = neighbors.NearestNeighbors()
    with pytest.raises(NotFittedError):
        neighbors_.kneighbors_graph(X)
    with pytest.raises(NotFittedError):
        neighbors_.radius_neighbors_graph(X)


@ignore_warnings(category=EfficiencyWarning)
def check_precomputed(make_train_test, estimators):
    """Tests unsupervised NearestNeighbors with a distance matrix."""
    # Note: smaller samples may result in spurious test success
    rng = np.random.RandomState(42)
    X = rng.random_sample((10, 4))
    Y = rng.random_sample((3, 4))
    DXX, DYX = make_train_test(X, Y)
    for method in [
        "kneighbors",
    ]:
        # TODO: also test radius_neighbors, but requires different assertion

        # As a feature matrix (n_samples by n_features)
        nbrs_X = neighbors.NearestNeighbors(n_neighbors=3)
        nbrs_X.fit(X)
        dist_X, ind_X = getattr(nbrs_X, method)(Y)

        # As a dense distance matrix (n_samples by n_samples)
        nbrs_D = neighbors.NearestNeighbors(
            n_neighbors=3, algorithm="brute", metric="precomputed"
        )
        nbrs_D.fit(DXX)
        dist_D, ind_D = getattr(nbrs_D, method)(DYX)
        assert_array_almost_equal(dist_X, dist_D)
        assert_array_almost_equal(ind_X, ind_D)

        # Check auto works too
        nbrs_D = neighbors.NearestNeighbors(
            n_neighbors=3, algorithm="auto", metric="precomputed"
        )
        nbrs_D.fit(DXX)
        dist_D, ind_D = getattr(nbrs_D, method)(DYX)
        assert_array_almost_equal(dist_X, dist_D)
        assert_array_almost_equal(ind_X, ind_D)

        # Check X=None in prediction
        dist_X, ind_X = getattr(nbrs_X, method)(None)
        dist_D, ind_D = getattr(nbrs_D, method)(None)
        assert_array_almost_equal(dist_X, dist_D)
        assert_array_almost_equal(ind_X, ind_D)

        # Must raise a ValueError if the matrix is not of correct shape
        with pytest.raises(ValueError):
            getattr(nbrs_D, method)(X)

    target = np.arange(X.shape[0])
    for Est in estimators:
        est = Est(metric="euclidean")
        est.radius = est.n_neighbors = 1
        pred_X = est.fit(X, target).predict(Y)
        est.metric = "precomputed"
        pred_D = est.fit(DXX, target).predict(DYX)
        assert_array_almost_equal(pred_X, pred_D)


def test_precomputed_dense():
    def make_train_test(X_train, X_test):
        return (
            metrics.pairwise_distances(X_train),
            metrics.pairwise_distances(X_test, X_train),
        )

    estimators = [
        neighbors.KNeighborsClassifier,
        neighbors.KNeighborsRegressor,
        neighbors.RadiusNeighborsClassifier,
        neighbors.RadiusNeighborsRegressor,
    ]
    check_precomputed(make_train_test, estimators)


@pytest.mark.parametrize("fmt", ["csr", "lil"])
def test_precomputed_sparse_knn(fmt):
    def make_train_test(X_train, X_test):
        nn = neighbors.NearestNeighbors(n_neighbors=3 + 1).fit(X_train)
        return (
            nn.kneighbors_graph(X_train, mode="distance").asformat(fmt),
            nn.kneighbors_graph(X_test, mode="distance").asformat(fmt),
        )

    # We do not test RadiusNeighborsClassifier and RadiusNeighborsRegressor
    # since the precomputed neighbors graph is built with k neighbors only.
    estimators = [
        neighbors.KNeighborsClassifier,
        neighbors.KNeighborsRegressor,
    ]
    check_precomputed(make_train_test, estimators)


@pytest.mark.parametrize("fmt", ["csr", "lil"])
def test_precomputed_sparse_radius(fmt):
    def make_train_test(X_train, X_test):
        nn = neighbors.NearestNeighbors(radius=1).fit(X_train)
        return (
            nn.radius_neighbors_graph(X_train, mode="distance").asformat(fmt),
            nn.radius_neighbors_graph(X_test, mode="distance").asformat(fmt),
        )

    # We do not test KNeighborsClassifier and KNeighborsRegressor
    # since the precomputed neighbors graph is built with a radius.
    estimators = [
        neighbors.RadiusNeighborsClassifier,
        neighbors.RadiusNeighborsRegressor,
    ]
    check_precomputed(make_train_test, estimators)


def test_is_sorted_by_data():
    # Test that _is_sorted_by_data works as expected. In CSR sparse matrix,
    # entries in each row can be sorted by indices, by data, or unsorted.
    # _is_sorted_by_data should return True when entries are sorted by data,
    # and False in all other cases.

    # Test with sorted 1D array
    X = csr_matrix(np.arange(10))
    assert _is_sorted_by_data(X)
    # Test with unsorted 1D array
    X[0, 2] = 5
    assert not _is_sorted_by_data(X)

    # Test when the data is sorted in each sample, but not necessarily
    # between samples
    X = csr_matrix([[0, 1, 2], [3, 0, 0], [3, 4, 0], [1, 0, 2]])
    assert _is_sorted_by_data(X)

    # Test with duplicates entries in X.indptr
    data, indices, indptr = [0, 4, 2, 2], [0, 1, 1, 1], [0, 2, 2, 4]
    X = csr_matrix((data, indices, indptr), shape=(3, 3))
    assert _is_sorted_by_data(X)


@ignore_warnings(category=EfficiencyWarning)
def test_check_precomputed():
    # Test that _check_precomputed returns a graph sorted by data
    X = csr_matrix(np.abs(np.random.RandomState(42).randn(10, 10)))
    assert not _is_sorted_by_data(X)
    Xt = _check_precomputed(X)
    assert _is_sorted_by_data(Xt)

    # est with a different number of nonzero entries for each sample
    mask = np.random.RandomState(42).randint(2, size=(10, 10))
    X = X.toarray()
    X[mask == 1] = 0
    X = csr_matrix(X)
    assert not _is_sorted_by_data(X)
    Xt = _check_precomputed(X)
    assert _is_sorted_by_data(Xt)


@ignore_warnings(category=EfficiencyWarning)
def test_precomputed_sparse_invalid():
    dist = np.array([[0.0, 2.0, 1.0], [2.0, 0.0, 3.0], [1.0, 3.0, 0.0]])
    dist_csr = csr_matrix(dist)
    neigh = neighbors.NearestNeighbors(n_neighbors=1, metric="precomputed")
    neigh.fit(dist_csr)
    neigh.kneighbors(None, n_neighbors=1)
    neigh.kneighbors(np.array([[0.0, 0.0, 0.0]]), n_neighbors=2)

    # Ensures enough number of nearest neighbors
    dist = np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 3.0], [0.0, 3.0, 0.0]])
    dist_csr = csr_matrix(dist)
    neigh.fit(dist_csr)
    msg = "2 neighbors per samples are required, but some samples have only 1"
    with pytest.raises(ValueError, match=msg):
        neigh.kneighbors(None, n_neighbors=1)

    # Checks error with inconsistent distance matrix
    dist = np.array([[5.0, 2.0, 1.0], [-2.0, 0.0, 3.0], [1.0, 3.0, 0.0]])
    dist_csr = csr_matrix(dist)
    msg = "Negative values in data passed to precomputed distance matrix."
    with pytest.raises(ValueError, match=msg):
        neigh.kneighbors(dist_csr, n_neighbors=1)


def test_precomputed_cross_validation():
    # Ensure array is split correctly
    rng = np.random.RandomState(0)
    X = rng.rand(20, 2)
    D = pairwise_distances(X, metric="euclidean")
    y = rng.randint(3, size=20)
    for Est in (
        neighbors.KNeighborsClassifier,
        neighbors.RadiusNeighborsClassifier,
        neighbors.KNeighborsRegressor,
        neighbors.RadiusNeighborsRegressor,
    ):
        metric_score = cross_val_score(Est(), X, y)
        precomp_score = cross_val_score(Est(metric="precomputed"), D, y)
        assert_array_equal(metric_score, precomp_score)


def test_unsupervised_radius_neighbors(
    n_samples=20, n_features=5, n_query_pts=2, radius=0.5, random_state=0
):
    # Test unsupervised radius-based query
    rng = np.random.RandomState(random_state)

    X = rng.rand(n_samples, n_features)

    test = rng.rand(n_query_pts, n_features)

    for p in P:
        results = []

        for algorithm in ALGORITHMS:
            neigh = neighbors.NearestNeighbors(radius=radius, algorithm=algorithm, p=p)
            neigh.fit(X)

            ind1 = neigh.radius_neighbors(test, return_distance=False)

            # sort the results: this is not done automatically for
            # radius searches
            dist, ind = neigh.radius_neighbors(test, return_distance=True)
            for (d, i, i1) in zip(dist, ind, ind1):
                j = d.argsort()
                d[:] = d[j]
                i[:] = i[j]
                i1[:] = i1[j]
            results.append((dist, ind))

            assert_array_almost_equal(
                np.concatenate(list(ind)), np.concatenate(list(ind1))
            )

        for i in range(len(results) - 1):
            assert_array_almost_equal(
                np.concatenate(list(results[i][0])),
                np.concatenate(list(results[i + 1][0])),
            ),
            assert_array_almost_equal(
                np.concatenate(list(results[i][1])),
                np.concatenate(list(results[i + 1][1])),
            )


def test_kneighbors_classifier(
    n_samples=40, n_features=5, n_test_pts=10, n_neighbors=5, random_state=0
):
    # Test k-neighbors classification
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = ((X ** 2).sum(axis=1) < 0.5).astype(int)
    y_str = y.astype(str)

    weight_func = _weight_func

    for algorithm in ALGORITHMS:
        for weights in ["uniform", "distance", weight_func]:
            knn = neighbors.KNeighborsClassifier(
                n_neighbors=n_neighbors, weights=weights, algorithm=algorithm
            )
            knn.fit(X, y)
            epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
            y_pred = knn.predict(X[:n_test_pts] + epsilon)
            assert_array_equal(y_pred, y[:n_test_pts])
            # Test prediction with y_str
            knn.fit(X, y_str)
            y_pred = knn.predict(X[:n_test_pts] + epsilon)
            assert_array_equal(y_pred, y_str[:n_test_pts])


def test_kneighbors_classifier_float_labels(
    n_samples=40, n_features=5, n_test_pts=10, n_neighbors=5, random_state=0
):
    # Test k-neighbors classification
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = ((X ** 2).sum(axis=1) < 0.5).astype(int)

    knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, y.astype(float))
    epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
    y_pred = knn.predict(X[:n_test_pts] + epsilon)
    assert_array_equal(y_pred, y[:n_test_pts])


def test_kneighbors_classifier_predict_proba():
    # Test KNeighborsClassifier.predict_proba() method
    X = np.array([[0, 2, 0], [0, 2, 1], [2, 0, 0], [2, 2, 0], [0, 0, 2], [0, 0, 1]])
    y = np.array([4, 4, 5, 5, 1, 1])
    cls = neighbors.KNeighborsClassifier(n_neighbors=3, p=1)  # cityblock dist
    cls.fit(X, y)
    y_prob = cls.predict_proba(X)
    real_prob = np.array(
        [
            [0, 2.0 / 3, 1.0 / 3],
            [1.0 / 3, 2.0 / 3, 0],
            [1.0 / 3, 0, 2.0 / 3],
            [0, 1.0 / 3, 2.0 / 3],
            [2.0 / 3, 1.0 / 3, 0],
            [2.0 / 3, 1.0 / 3, 0],
        ]
    )
    assert_array_equal(real_prob, y_prob)
    # Check that it also works with non integer labels
    cls.fit(X, y.astype(str))
    y_prob = cls.predict_proba(X)
    assert_array_equal(real_prob, y_prob)
    # Check that it works with weights='distance'
    cls = neighbors.KNeighborsClassifier(n_neighbors=2, p=1, weights="distance")
    cls.fit(X, y)
    y_prob = cls.predict_proba(np.array([[0, 2, 0], [2, 2, 2]]))
    real_prob = np.array([[0, 1, 0], [0, 0.4, 0.6]])
    assert_array_almost_equal(real_prob, y_prob)


def test_radius_neighbors_classifier(
    n_samples=40, n_features=5, n_test_pts=10, radius=0.5, random_state=0
):
    # Test radius-based classification
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = ((X ** 2).sum(axis=1) < 0.5).astype(int)
    y_str = y.astype(str)

    weight_func = _weight_func

    for algorithm in ALGORITHMS:
        for weights in ["uniform", "distance", weight_func]:
            neigh = neighbors.RadiusNeighborsClassifier(
                radius=radius, weights=weights, algorithm=algorithm
            )
            neigh.fit(X, y)
            epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
            y_pred = neigh.predict(X[:n_test_pts] + epsilon)
            assert_array_equal(y_pred, y[:n_test_pts])
            neigh.fit(X, y_str)
            y_pred = neigh.predict(X[:n_test_pts] + epsilon)
            assert_array_equal(y_pred, y_str[:n_test_pts])


# TODO: Remove in v1.2
def test_radius_neighbors_classifier_kwargs_is_deprecated():
    extra_kwargs = {
        "unused_param": "",
        "extra_param": None,
    }
    msg = (
        "Passing additional keyword parameters has no effect and is deprecated "
        "in 1.0. An error will be raised from 1.2 and beyond. The ignored "
        f"keyword parameter(s) are: {extra_kwargs.keys()}."
    )
    with pytest.warns(FutureWarning, match=re.escape(msg)):
        neighbors.RadiusNeighborsClassifier(**extra_kwargs)


def test_radius_neighbors_classifier_when_no_neighbors():
    # Test radius-based classifier when no neighbors found.
    # In this case it should rise an informative exception

    X = np.array([[1.0, 1.0], [2.0, 2.0]])
    y = np.array([1, 2])
    radius = 0.1

    z1 = np.array([[1.01, 1.01], [2.01, 2.01]])  # no outliers
    z2 = np.array([[1.01, 1.01], [1.4, 1.4]])  # one outlier

    weight_func = _weight_func

    for outlier_label in [0, -1, None]:
        for algorithm in ALGORITHMS:
            for weights in ["uniform", "distance", weight_func]:
                rnc = neighbors.RadiusNeighborsClassifier
                clf = rnc(
                    radius=radius,
                    weights=weights,
                    algorithm=algorithm,
                    outlier_label=outlier_label,
                )
                clf.fit(X, y)
                assert_array_equal(np.array([1, 2]), clf.predict(z1))
                if outlier_label is None:
                    with pytest.raises(ValueError):
                        clf.predict(z2)


def test_radius_neighbors_classifier_outlier_labeling():
    # Test radius-based classifier when no neighbors found and outliers
    # are labeled.

    X = np.array([[1.0, 1.0], [2.0, 2.0], [0.99, 0.99], [0.98, 0.98], [2.01, 2.01]])
    y = np.array([1, 2, 1, 1, 2])
    radius = 0.1

    z1 = np.array([[1.01, 1.01], [2.01, 2.01]])  # no outliers
    z2 = np.array([[1.4, 1.4], [1.01, 1.01], [2.01, 2.01]])  # one outlier
    correct_labels1 = np.array([1, 2])
    correct_labels2 = np.array([-1, 1, 2])
    outlier_proba = np.array([0, 0])

    weight_func = _weight_func

    for algorithm in ALGORITHMS:
        for weights in ["uniform", "distance", weight_func]:
            clf = neighbors.RadiusNeighborsClassifier(
                radius=radius, weights=weights, algorithm=algorithm, outlier_label=-1
            )
            clf.fit(X, y)
            assert_array_equal(correct_labels1, clf.predict(z1))
            assert_array_equal(correct_labels2, clf.predict(z2))
            assert_array_equal(outlier_proba, clf.predict_proba(z2)[0])

    # test outlier_labeling of using predict_proba()
    RNC = neighbors.RadiusNeighborsClassifier
    X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    y = np.array([0, 2, 2, 1, 1, 1, 3, 3, 3, 3])

    # test outlier_label scalar verification
    def check_array_exception():
        clf = RNC(radius=1, outlier_label=[[5]])
        clf.fit(X, y)

    with pytest.raises(TypeError):
        check_array_exception()

    # test invalid outlier_label dtype
    def check_dtype_exception():
        clf = RNC(radius=1, outlier_label="a")
        clf.fit(X, y)

    with pytest.raises(TypeError):
        check_dtype_exception()

    # test most frequent
    clf = RNC(radius=1, outlier_label="most_frequent")
    clf.fit(X, y)
    proba = clf.predict_proba([[1], [15]])
    assert_array_equal(proba[1, :], [0, 0, 0, 1])

    # test manual label in y
    clf = RNC(radius=1, outlier_label=1)
    clf.fit(X, y)
    proba = clf.predict_proba([[1], [15]])
    assert_array_equal(proba[1, :], [0, 1, 0, 0])
    pred = clf.predict([[1], [15]])
    assert_array_equal(pred, [2, 1])

    # test manual label out of y warning
    def check_warning():
        clf = RNC(radius=1, outlier_label=4)
        clf.fit(X, y)
        clf.predict_proba([[1], [15]])

    with pytest.warns(UserWarning):
        check_warning()

    # test multi output same outlier label
    y_multi = [
        [0, 1],
        [2, 1],
        [2, 2],
        [1, 2],
        [1, 2],
        [1, 3],
        [3, 3],
        [3, 3],
        [3, 0],
        [3, 0],
    ]
    clf = RNC(radius=1, outlier_label=1)
    clf.fit(X, y_multi)
    proba = clf.predict_proba([[7], [15]])
    assert_array_equal(proba[1][1, :], [0, 1, 0, 0])
    pred = clf.predict([[7], [15]])
    assert_array_equal(pred[1, :], [1, 1])

    # test multi output different outlier label
    y_multi = [
        [0, 0],
        [2, 2],
        [2, 2],
        [1, 1],
        [1, 1],
        [1, 1],
        [3, 3],
        [3, 3],
        [3, 3],
        [3, 3],
    ]
    clf = RNC(radius=1, outlier_label=[0, 1])
    clf.fit(X, y_multi)
    proba = clf.predict_proba([[7], [15]])
    assert_array_equal(proba[0][1, :], [1, 0, 0, 0])
    assert_array_equal(proba[1][1, :], [0, 1, 0, 0])
    pred = clf.predict([[7], [15]])
    assert_array_equal(pred[1, :], [0, 1])

    # test inconsistent outlier label list length
    def check_exception():
        clf = RNC(radius=1, outlier_label=[0, 1, 2])
        clf.fit(X, y_multi)

    with pytest.raises(ValueError):
        check_exception()


def test_radius_neighbors_classifier_zero_distance():
    # Test radius-based classifier, when distance to a sample is zero.

    X = np.array([[1.0, 1.0], [2.0, 2.0]])
    y = np.array([1, 2])
    radius = 0.1

    z1 = np.array([[1.01, 1.01], [2.0, 2.0]])
    correct_labels1 = np.array([1, 2])

    weight_func = _weight_func

    for algorithm in ALGORITHMS:
        for weights in ["uniform", "distance", weight_func]:
            clf = neighbors.RadiusNeighborsClassifier(
                radius=radius, weights=weights, algorithm=algorithm
            )
            clf.fit(X, y)
            with np.errstate(invalid="ignore"):
                # Ignore the warning raised in _weight_func when making
                # predictions with null distances resulting in np.inf values.
                assert_array_equal(correct_labels1, clf.predict(z1))


def test_neighbors_regressors_zero_distance():
    # Test radius-based regressor, when distance to a sample is zero.

    X = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.5, 2.5]])
    y = np.array([1.0, 1.5, 2.0, 0.0])
    radius = 0.2
    z = np.array([[1.1, 1.1], [2.0, 2.0]])

    rnn_correct_labels = np.array([1.25, 2.0])

    knn_correct_unif = np.array([1.25, 1.0])
    knn_correct_dist = np.array([1.25, 2.0])

    for algorithm in ALGORITHMS:
        # we don't test for weights=_weight_func since user will be expected
        # to handle zero distances themselves in the function.
        for weights in ["uniform", "distance"]:
            rnn = neighbors.RadiusNeighborsRegressor(
                radius=radius, weights=weights, algorithm=algorithm
            )
            rnn.fit(X, y)
            assert_array_almost_equal(rnn_correct_labels, rnn.predict(z))

        for weights, corr_labels in zip(
            ["uniform", "distance"], [knn_correct_unif, knn_correct_dist]
        ):
            knn = neighbors.KNeighborsRegressor(
                n_neighbors=2, weights=weights, algorithm=algorithm
            )
            knn.fit(X, y)
            assert_array_almost_equal(corr_labels, knn.predict(z))


def test_radius_neighbors_boundary_handling():
    """Test whether points lying on boundary are handled consistently

    Also ensures that even with only one query point, an object array
    is returned rather than a 2d array.
    """

    X = np.array([[1.5], [3.0], [3.01]])
    radius = 3.0

    for algorithm in ALGORITHMS:
        nbrs = neighbors.NearestNeighbors(radius=radius, algorithm=algorithm).fit(X)
        results = nbrs.radius_neighbors([[0.0]], return_distance=False)
        assert results.shape == (1,)
        assert results.dtype == object
        assert_array_equal(results[0], [0, 1])


def test_radius_neighbors_returns_array_of_objects():
    # check that we can pass precomputed distances to
    # NearestNeighbors.radius_neighbors()
    # non-regression test for
    # https://github.com/scikit-learn/scikit-learn/issues/16036
    X = csr_matrix(np.ones((4, 4)))
    X.setdiag([0, 0, 0, 0])

    nbrs = neighbors.NearestNeighbors(
        radius=0.5, algorithm="auto", leaf_size=30, metric="precomputed"
    ).fit(X)
    neigh_dist, neigh_ind = nbrs.radius_neighbors(X, return_distance=True)

    expected_dist = np.empty(X.shape[0], dtype=object)
    expected_dist[:] = [np.array([0]), np.array([0]), np.array([0]), np.array([0])]
    expected_ind = np.empty(X.shape[0], dtype=object)
    expected_ind[:] = [np.array([0]), np.array([1]), np.array([2]), np.array([3])]

    assert_array_equal(neigh_dist, expected_dist)
    assert_array_equal(neigh_ind, expected_ind)


@pytest.mark.parametrize("algorithm", ["ball_tree", "kd_tree", "brute"])
def test_query_equidistant_kth_nn(algorithm):
    # For several candidates for the k-th nearest neighbor position,
    # the first candidate should be chosen
    query_point = np.array([[0, 0]])
    equidistant_points = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    # The 3rd and 4th points should not replace the 2nd point
    # for the 2th nearest neighbor position
    k = 2
    knn_indices = np.array([[0, 1]])
    nn = neighbors.NearestNeighbors(algorithm=algorithm).fit(equidistant_points)
    indices = np.sort(nn.kneighbors(query_point, n_neighbors=k, return_distance=False))
    assert_array_equal(indices, knn_indices)


@pytest.mark.parametrize(
    ["algorithm", "metric"],
    [
        ("ball_tree", "euclidean"),
        ("kd_tree", "euclidean"),
        ("brute", "euclidean"),
        ("brute", "precomputed"),
    ],
)
def test_radius_neighbors_sort_results(algorithm, metric):
    # Test radius_neighbors[_graph] output when sort_result is True
    n_samples = 10
    rng = np.random.RandomState(42)
    X = rng.random_sample((n_samples, 4))

    if metric == "precomputed":
        X = neighbors.radius_neighbors_graph(X, radius=np.inf, mode="distance")
    model = neighbors.NearestNeighbors(algorithm=algorithm, metric=metric)
    model.fit(X)

    # self.radius_neighbors
    distances, indices = model.radius_neighbors(X=X, radius=np.inf, sort_results=True)
    for ii in range(n_samples):
        assert_array_equal(distances[ii], np.sort(distances[ii]))

    # sort_results=True and return_distance=False
    if metric != "precomputed":  # no need to raise with precomputed graph
        with pytest.raises(ValueError, match="return_distance must be True"):
            model.radius_neighbors(
                X=X, radius=np.inf, sort_results=True, return_distance=False
            )

    # self.radius_neighbors_graph
    graph = model.radius_neighbors_graph(
        X=X, radius=np.inf, mode="distance", sort_results=True
    )
    assert _is_sorted_by_data(graph)


def test_RadiusNeighborsClassifier_multioutput():
    # Test k-NN classifier on multioutput data
    rng = check_random_state(0)
    n_features = 2
    n_samples = 40
    n_output = 3

    X = rng.rand(n_samples, n_features)
    y = rng.randint(0, 3, (n_samples, n_output))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    weights = [None, "uniform", "distance", _weight_func]

    for algorithm, weights in product(ALGORITHMS, weights):
        # Stack single output prediction
        y_pred_so = []
        for o in range(n_output):
            rnn = neighbors.RadiusNeighborsClassifier(
                weights=weights, algorithm=algorithm
            )
            rnn.fit(X_train, y_train[:, o])
            y_pred_so.append(rnn.predict(X_test))

        y_pred_so = np.vstack(y_pred_so).T
        assert y_pred_so.shape == y_test.shape

        # Multioutput prediction
        rnn_mo = neighbors.RadiusNeighborsClassifier(
            weights=weights, algorithm=algorithm
        )
        rnn_mo.fit(X_train, y_train)
        y_pred_mo = rnn_mo.predict(X_test)

        assert y_pred_mo.shape == y_test.shape
        assert_array_almost_equal(y_pred_mo, y_pred_so)


def test_kneighbors_classifier_sparse(
    n_samples=40, n_features=5, n_test_pts=10, n_neighbors=5, random_state=0
):
    # Test k-NN classifier on sparse matrices
    # Like the above, but with various types of sparse matrices
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    X *= X > 0.2
    y = ((X ** 2).sum(axis=1) < 0.5).astype(int)

    for sparsemat in SPARSE_TYPES:
        knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm="auto")
        knn.fit(sparsemat(X), y)
        epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
        for sparsev in SPARSE_TYPES + (np.asarray,):
            X_eps = sparsev(X[:n_test_pts] + epsilon)
            y_pred = knn.predict(X_eps)
            assert_array_equal(y_pred, y[:n_test_pts])


def test_KNeighborsClassifier_multioutput():
    # Test k-NN classifier on multioutput data
    rng = check_random_state(0)
    n_features = 5
    n_samples = 50
    n_output = 3

    X = rng.rand(n_samples, n_features)
    y = rng.randint(0, 3, (n_samples, n_output))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    weights = [None, "uniform", "distance", _weight_func]

    for algorithm, weights in product(ALGORITHMS, weights):
        # Stack single output prediction
        y_pred_so = []
        y_pred_proba_so = []
        for o in range(n_output):
            knn = neighbors.KNeighborsClassifier(weights=weights, algorithm=algorithm)
            knn.fit(X_train, y_train[:, o])
            y_pred_so.append(knn.predict(X_test))
            y_pred_proba_so.append(knn.predict_proba(X_test))

        y_pred_so = np.vstack(y_pred_so).T
        assert y_pred_so.shape == y_test.shape
        assert len(y_pred_proba_so) == n_output

        # Multioutput prediction
        knn_mo = neighbors.KNeighborsClassifier(weights=weights, algorithm=algorithm)
        knn_mo.fit(X_train, y_train)
        y_pred_mo = knn_mo.predict(X_test)

        assert y_pred_mo.shape == y_test.shape
        assert_array_almost_equal(y_pred_mo, y_pred_so)

        # Check proba
        y_pred_proba_mo = knn_mo.predict_proba(X_test)
        assert len(y_pred_proba_mo) == n_output

        for proba_mo, proba_so in zip(y_pred_proba_mo, y_pred_proba_so):
            assert_array_almost_equal(proba_mo, proba_so)


def test_kneighbors_regressor(
    n_samples=40, n_features=5, n_test_pts=10, n_neighbors=3, random_state=0
):
    # Test k-neighbors regression
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = np.sqrt((X ** 2).sum(1))
    y /= y.max()

    y_target = y[:n_test_pts]

    weight_func = _weight_func

    for algorithm in ALGORITHMS:
        for weights in ["uniform", "distance", weight_func]:
            knn = neighbors.KNeighborsRegressor(
                n_neighbors=n_neighbors, weights=weights, algorithm=algorithm
            )
            knn.fit(X, y)
            epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
            y_pred = knn.predict(X[:n_test_pts] + epsilon)
            assert np.all(abs(y_pred - y_target) < 0.3)


def test_KNeighborsRegressor_multioutput_uniform_weight():
    # Test k-neighbors in multi-output regression with uniform weight
    rng = check_random_state(0)
    n_features = 5
    n_samples = 40
    n_output = 4

    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples, n_output)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    for algorithm, weights in product(ALGORITHMS, [None, "uniform"]):
        knn = neighbors.KNeighborsRegressor(weights=weights, algorithm=algorithm)
        knn.fit(X_train, y_train)

        neigh_idx = knn.kneighbors(X_test, return_distance=False)
        y_pred_idx = np.array([np.mean(y_train[idx], axis=0) for idx in neigh_idx])

        y_pred = knn.predict(X_test)

        assert y_pred.shape == y_test.shape
        assert y_pred_idx.shape == y_test.shape
        assert_array_almost_equal(y_pred, y_pred_idx)


def test_kneighbors_regressor_multioutput(
    n_samples=40, n_features=5, n_test_pts=10, n_neighbors=3, random_state=0
):
    # Test k-neighbors in multi-output regression
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = np.sqrt((X ** 2).sum(1))
    y /= y.max()
    y = np.vstack([y, y]).T

    y_target = y[:n_test_pts]

    weights = ["uniform", "distance", _weight_func]
    for algorithm, weights in product(ALGORITHMS, weights):
        knn = neighbors.KNeighborsRegressor(
            n_neighbors=n_neighbors, weights=weights, algorithm=algorithm
        )
        knn.fit(X, y)
        epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
        y_pred = knn.predict(X[:n_test_pts] + epsilon)
        assert y_pred.shape == y_target.shape

        assert np.all(np.abs(y_pred - y_target) < 0.3)


def test_radius_neighbors_regressor(
    n_samples=40, n_features=3, n_test_pts=10, radius=0.5, random_state=0
):
    # Test radius-based neighbors regression
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = np.sqrt((X ** 2).sum(1))
    y /= y.max()

    y_target = y[:n_test_pts]

    weight_func = _weight_func

    for algorithm in ALGORITHMS:
        for weights in ["uniform", "distance", weight_func]:
            neigh = neighbors.RadiusNeighborsRegressor(
                radius=radius, weights=weights, algorithm=algorithm
            )
            neigh.fit(X, y)
            epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
            y_pred = neigh.predict(X[:n_test_pts] + epsilon)
            assert np.all(abs(y_pred - y_target) < radius / 2)

    # test that nan is returned when no nearby observations
    for weights in ["uniform", "distance"]:
        neigh = neighbors.RadiusNeighborsRegressor(
            radius=radius, weights=weights, algorithm="auto"
        )
        neigh.fit(X, y)
        X_test_nan = np.full((1, n_features), -1.0)
        empty_warning_msg = (
            "One or more samples have no neighbors "
            "within specified radius; predicting NaN."
        )
        with pytest.warns(UserWarning, match=re.escape(empty_warning_msg)):
            pred = neigh.predict(X_test_nan)
        assert np.all(np.isnan(pred))


def test_RadiusNeighborsRegressor_multioutput_with_uniform_weight():
    # Test radius neighbors in multi-output regression (uniform weight)

    rng = check_random_state(0)
    n_features = 5
    n_samples = 40
    n_output = 4

    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples, n_output)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    for algorithm, weights in product(ALGORITHMS, [None, "uniform"]):

        rnn = neighbors.RadiusNeighborsRegressor(weights=weights, algorithm=algorithm)
        rnn.fit(X_train, y_train)

        neigh_idx = rnn.radius_neighbors(X_test, return_distance=False)
        y_pred_idx = np.array([np.mean(y_train[idx], axis=0) for idx in neigh_idx])

        y_pred_idx = np.array(y_pred_idx)
        y_pred = rnn.predict(X_test)

        assert y_pred_idx.shape == y_test.shape
        assert y_pred.shape == y_test.shape
        assert_array_almost_equal(y_pred, y_pred_idx)


def test_RadiusNeighborsRegressor_multioutput(
    n_samples=40, n_features=5, n_test_pts=10, random_state=0
):
    # Test k-neighbors in multi-output regression with various weight
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = np.sqrt((X ** 2).sum(1))
    y /= y.max()
    y = np.vstack([y, y]).T

    y_target = y[:n_test_pts]
    weights = ["uniform", "distance", _weight_func]

    for algorithm, weights in product(ALGORITHMS, weights):
        rnn = neighbors.RadiusNeighborsRegressor(weights=weights, algorithm=algorithm)
        rnn.fit(X, y)
        epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
        y_pred = rnn.predict(X[:n_test_pts] + epsilon)

        assert y_pred.shape == y_target.shape
        assert np.all(np.abs(y_pred - y_target) < 0.3)


@ignore_warnings(category=EfficiencyWarning)
def test_kneighbors_regressor_sparse(
    n_samples=40, n_features=5, n_test_pts=10, n_neighbors=5, random_state=0
):
    # Test radius-based regression on sparse matrices
    # Like the above, but with various types of sparse matrices
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = ((X ** 2).sum(axis=1) < 0.25).astype(int)

    for sparsemat in SPARSE_TYPES:
        knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, algorithm="auto")
        knn.fit(sparsemat(X), y)

        knn_pre = neighbors.KNeighborsRegressor(
            n_neighbors=n_neighbors, metric="precomputed"
        )
        knn_pre.fit(pairwise_distances(X, metric="euclidean"), y)

        for sparsev in SPARSE_OR_DENSE:
            X2 = sparsev(X)
            assert np.mean(knn.predict(X2).round() == y) > 0.95

            X2_pre = sparsev(pairwise_distances(X, metric="euclidean"))
            if sparsev in {dok_matrix, bsr_matrix}:
                msg = "not supported due to its handling of explicit zeros"
                with pytest.raises(TypeError, match=msg):
                    knn_pre.predict(X2_pre)
            else:
                assert np.mean(knn_pre.predict(X2_pre).round() == y) > 0.95


def test_neighbors_iris():
    # Sanity checks on the iris dataset
    # Puts three points of each label in the plane and performs a
    # nearest neighbor query on points near the decision boundary.

    for algorithm in ALGORITHMS:
        clf = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm=algorithm)
        clf.fit(iris.data, iris.target)
        assert_array_equal(clf.predict(iris.data), iris.target)

        clf.set_params(n_neighbors=9, algorithm=algorithm)
        clf.fit(iris.data, iris.target)
        assert np.mean(clf.predict(iris.data) == iris.target) > 0.95

        rgs = neighbors.KNeighborsRegressor(n_neighbors=5, algorithm=algorithm)
        rgs.fit(iris.data, iris.target)
        assert np.mean(rgs.predict(iris.data).round() == iris.target) > 0.95


def test_neighbors_digits():
    # Sanity check on the digits dataset
    # the 'brute' algorithm has been observed to fail if the input
    # dtype is uint8 due to overflow in distance calculations.

    X = digits.data.astype("uint8")
    Y = digits.target
    (n_samples, n_features) = X.shape
    train_test_boundary = int(n_samples * 0.8)
    train = np.arange(0, train_test_boundary)
    test = np.arange(train_test_boundary, n_samples)
    (X_train, Y_train, X_test, Y_test) = X[train], Y[train], X[test], Y[test]

    clf = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm="brute")
    score_uint8 = clf.fit(X_train, Y_train).score(X_test, Y_test)
    score_float = clf.fit(X_train.astype(float, copy=False), Y_train).score(
        X_test.astype(float, copy=False), Y_test
    )
    assert score_uint8 == score_float


def test_kneighbors_graph():
    # Test kneighbors_graph to build the k-Nearest Neighbor graph.
    X = np.array([[0, 1], [1.01, 1.0], [2, 0]])

    # n_neighbors = 1
    A = neighbors.kneighbors_graph(X, 1, mode="connectivity", include_self=True)
    assert_array_equal(A.toarray(), np.eye(A.shape[0]))

    A = neighbors.kneighbors_graph(X, 1, mode="distance")
    assert_array_almost_equal(
        A.toarray(), [[0.00, 1.01, 0.0], [1.01, 0.0, 0.0], [0.00, 1.40716026, 0.0]]
    )

    # n_neighbors = 2
    A = neighbors.kneighbors_graph(X, 2, mode="connectivity", include_self=True)
    assert_array_equal(A.toarray(), [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

    A = neighbors.kneighbors_graph(X, 2, mode="distance")
    assert_array_almost_equal(
        A.toarray(),
        [
            [0.0, 1.01, 2.23606798],
            [1.01, 0.0, 1.40716026],
            [2.23606798, 1.40716026, 0.0],
        ],
    )

    # n_neighbors = 3
    A = neighbors.kneighbors_graph(X, 3, mode="connectivity", include_self=True)
    assert_array_almost_equal(A.toarray(), [[1, 1, 1], [1, 1, 1], [1, 1, 1]])


def test_kneighbors_graph_sparse(seed=36):
    # Test kneighbors_graph to build the k-Nearest Neighbor graph
    # for sparse input.
    rng = np.random.RandomState(seed)
    X = rng.randn(10, 10)
    Xcsr = csr_matrix(X)

    for n_neighbors in [1, 2, 3]:
        for mode in ["connectivity", "distance"]:
            assert_array_almost_equal(
                neighbors.kneighbors_graph(X, n_neighbors, mode=mode).toarray(),
                neighbors.kneighbors_graph(Xcsr, n_neighbors, mode=mode).toarray(),
            )


def test_radius_neighbors_graph():
    # Test radius_neighbors_graph to build the Nearest Neighbor graph.
    X = np.array([[0, 1], [1.01, 1.0], [2, 0]])

    A = neighbors.radius_neighbors_graph(X, 1.5, mode="connectivity", include_self=True)
    assert_array_equal(A.toarray(), [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])

    A = neighbors.radius_neighbors_graph(X, 1.5, mode="distance")
    assert_array_almost_equal(
        A.toarray(), [[0.0, 1.01, 0.0], [1.01, 0.0, 1.40716026], [0.0, 1.40716026, 0.0]]
    )


def test_radius_neighbors_graph_sparse(seed=36):
    # Test radius_neighbors_graph to build the Nearest Neighbor graph
    # for sparse input.
    rng = np.random.RandomState(seed)
    X = rng.randn(10, 10)
    Xcsr = csr_matrix(X)

    for n_neighbors in [1, 2, 3]:
        for mode in ["connectivity", "distance"]:
            assert_array_almost_equal(
                neighbors.radius_neighbors_graph(X, n_neighbors, mode=mode).toarray(),
                neighbors.radius_neighbors_graph(
                    Xcsr, n_neighbors, mode=mode
                ).toarray(),
            )


def test_neighbors_badargs():
    # Test bad argument values: these should all raise ValueErrors
    X = rng.random_sample((10, 2))
    Xsparse = csr_matrix(X)
    X3 = rng.random_sample((10, 3))
    y = np.ones(10)

    est = neighbors.NearestNeighbors(algorithm="blah")
    with pytest.raises(ValueError):
        est.fit(X)

    for cls in (
        neighbors.KNeighborsClassifier,
        neighbors.RadiusNeighborsClassifier,
        neighbors.KNeighborsRegressor,
        neighbors.RadiusNeighborsRegressor,
    ):
        est = cls(weights="blah")
        with pytest.raises(ValueError):
            est.fit(X, y)
        est = cls(p=-1)
        with pytest.raises(ValueError):
            est.fit(X, y)
        est = cls(algorithm="blah")
        with pytest.raises(ValueError):
            est.fit(X, y)

        nbrs = cls(algorithm="ball_tree", metric="haversine")
        with pytest.raises(ValueError):
            nbrs.predict(X)
        with pytest.raises(ValueError):
            ignore_warnings(nbrs.fit(Xsparse, y))

        nbrs = cls(metric="haversine", algorithm="brute")
        nbrs.fit(X3, y)
        msg = "Haversine distance only valid in 2 dimensions"
        with pytest.raises(ValueError, match=msg):
            nbrs.predict(X3)

        nbrs = cls()
        with pytest.raises(ValueError):
            nbrs.fit(np.ones((0, 2)), np.ones(0))
        with pytest.raises(ValueError):
            nbrs.fit(X[:, :, None], y)
        nbrs.fit(X, y)
        with pytest.raises(ValueError):
            nbrs.predict([[]])
        if issubclass(cls, neighbors.KNeighborsClassifier) or issubclass(
            cls, neighbors.KNeighborsRegressor
        ):
            nbrs = cls(n_neighbors=-1)
            with pytest.raises(ValueError):
                nbrs.fit(X, y)

    nbrs = neighbors.NearestNeighbors().fit(X)

    with pytest.raises(ValueError):
        nbrs.kneighbors_graph(X, mode="blah")
    with pytest.raises(ValueError):
        nbrs.radius_neighbors_graph(X, mode="blah")


def test_neighbors_metrics(n_samples=20, n_features=3, n_query_pts=2, n_neighbors=5):
    # Test computing the neighbors for various metrics
    # create a symmetric matrix
    V = rng.rand(n_features, n_features)
    VI = np.dot(V, V.T)

    metrics = [
        ("euclidean", {}),
        ("manhattan", {}),
        ("minkowski", dict(p=1)),
        ("minkowski", dict(p=2)),
        ("minkowski", dict(p=3)),
        ("minkowski", dict(p=np.inf)),
        ("chebyshev", {}),
        ("seuclidean", dict(V=rng.rand(n_features))),
        ("mahalanobis", dict(VI=VI)),
        ("haversine", {}),
    ]
    if sp_version < parse_version("1.8.0.dev0"):
        # TODO: remove once we no longer support scipy < 1.8.0.
        # wminkowski was removed in scipy 1.8.0 but should work for previous
        # versions.
        metrics.append(
            ("wminkowski", dict(p=3, w=rng.rand(n_features))),
        )
    else:
        # Recent scipy versions accept weights in the Minkowski metric directly:
        metrics.append(
            ("minkowski", dict(p=3, w=rng.rand(n_features))),
        )

    algorithms = ["brute", "ball_tree", "kd_tree"]
    X = rng.rand(n_samples, n_features)

    test = rng.rand(n_query_pts, n_features)

    for metric, metric_params in metrics:
        results = {}
        p = metric_params.pop("p", 2)
        w = metric_params.get("w", None)
        for algorithm in algorithms:
            # KD tree doesn't support all metrics
            if algorithm == "kd_tree" and (
                metric not in neighbors.KDTree.valid_metrics or w is not None
            ):
                est = neighbors.NearestNeighbors(
                    algorithm=algorithm, metric=metric, metric_params=metric_params
                )
                with pytest.raises(ValueError):
                    est.fit(X)
                continue
            neigh = neighbors.NearestNeighbors(
                n_neighbors=n_neighbors,
                algorithm=algorithm,
                metric=metric,
                p=p,
                metric_params=metric_params,
            )

            # Haversine distance only accepts 2D data
            feature_sl = slice(None, 2) if metric == "haversine" else slice(None)

            neigh.fit(X[:, feature_sl])

            # wminkoski is deprecated in SciPy 1.6.0 and removed in 1.8.0
            ExceptionToAssert = None
            if (
                metric == "wminkowski"
                and algorithm == "brute"
                and sp_version >= parse_version("1.6.0")
            ):
                ExceptionToAssert = DeprecationWarning

            with pytest.warns(ExceptionToAssert):
                results[algorithm] = neigh.kneighbors(
                    test[:, feature_sl], return_distance=True
                )

        assert_array_almost_equal(results["brute"][0], results["ball_tree"][0])
        assert_array_almost_equal(results["brute"][1], results["ball_tree"][1])
        if "kd_tree" in results:
            assert_array_almost_equal(results["brute"][0], results["kd_tree"][0])
            assert_array_almost_equal(results["brute"][1], results["kd_tree"][1])


def test_callable_metric():
    def custom_metric(x1, x2):
        return np.sqrt(np.sum(x1 ** 2 + x2 ** 2))

    X = np.random.RandomState(42).rand(20, 2)
    nbrs1 = neighbors.NearestNeighbors(
        n_neighbors=3, algorithm="auto", metric=custom_metric
    )
    nbrs2 = neighbors.NearestNeighbors(
        n_neighbors=3, algorithm="brute", metric=custom_metric
    )

    nbrs1.fit(X)
    nbrs2.fit(X)

    dist1, ind1 = nbrs1.kneighbors(X)
    dist2, ind2 = nbrs2.kneighbors(X)

    assert_array_almost_equal(dist1, dist2)


def test_valid_brute_metric_for_auto_algorithm():
    X = rng.rand(12, 12)
    Xcsr = csr_matrix(X)

    # check that there is a metric that is valid for brute
    # but not ball_tree (so we actually test something)
    assert "cosine" in VALID_METRICS["brute"]
    assert "cosine" not in VALID_METRICS["ball_tree"]

    # Metric which don't required any additional parameter
    require_params = ["mahalanobis", "wminkowski", "seuclidean"]
    for metric in VALID_METRICS["brute"]:
        if metric != "precomputed" and metric not in require_params:
            nn = neighbors.NearestNeighbors(
                n_neighbors=3, algorithm="auto", metric=metric
            )
            if metric != "haversine":
                nn.fit(X)
                nn.kneighbors(X)
            else:
                nn.fit(X[:, :2])
                nn.kneighbors(X[:, :2])
        elif metric == "precomputed":
            X_precomputed = rng.random_sample((10, 4))
            Y_precomputed = rng.random_sample((3, 4))
            DXX = metrics.pairwise_distances(X_precomputed, metric="euclidean")
            DYX = metrics.pairwise_distances(
                Y_precomputed, X_precomputed, metric="euclidean"
            )
            nb_p = neighbors.NearestNeighbors(n_neighbors=3)
            nb_p.fit(DXX)
            nb_p.kneighbors(DYX)

    for metric in VALID_METRICS_SPARSE["brute"]:
        if metric != "precomputed" and metric not in require_params:
            nn = neighbors.NearestNeighbors(
                n_neighbors=3, algorithm="auto", metric=metric
            ).fit(Xcsr)
            nn.kneighbors(Xcsr)

    # Metric with parameter
    VI = np.dot(X, X.T)
    list_metrics = [
        ("seuclidean", dict(V=rng.rand(12))),
        ("wminkowski", dict(w=rng.rand(12))),
        ("mahalanobis", dict(VI=VI)),
    ]
    for metric, params in list_metrics:
        nn = neighbors.NearestNeighbors(
            n_neighbors=3, algorithm="auto", metric=metric, metric_params=params
        ).fit(X)
        nn.kneighbors(X)


def test_metric_params_interface():
    X = rng.rand(5, 5)
    y = rng.randint(0, 2, 5)
    est = neighbors.KNeighborsClassifier(metric_params={"p": 3})
    with pytest.warns(SyntaxWarning):
        est.fit(X, y)


def test_predict_sparse_ball_kd_tree():
    rng = np.random.RandomState(0)
    X = rng.rand(5, 5)
    y = rng.randint(0, 2, 5)
    nbrs1 = neighbors.KNeighborsClassifier(1, algorithm="kd_tree")
    nbrs2 = neighbors.KNeighborsRegressor(1, algorithm="ball_tree")
    for model in [nbrs1, nbrs2]:
        model.fit(X, y)
        with pytest.raises(ValueError):
            model.predict(csr_matrix(X))


def test_non_euclidean_kneighbors():
    rng = np.random.RandomState(0)
    X = rng.rand(5, 5)

    # Find a reasonable radius.
    dist_array = pairwise_distances(X).flatten()
    np.sort(dist_array)
    radius = dist_array[15]

    # Test kneighbors_graph
    for metric in ["manhattan", "chebyshev"]:
        nbrs_graph = neighbors.kneighbors_graph(
            X, 3, metric=metric, mode="connectivity", include_self=True
        ).toarray()
        nbrs1 = neighbors.NearestNeighbors(n_neighbors=3, metric=metric).fit(X)
        assert_array_equal(nbrs_graph, nbrs1.kneighbors_graph(X).toarray())

    # Test radiusneighbors_graph
    for metric in ["manhattan", "chebyshev"]:
        nbrs_graph = neighbors.radius_neighbors_graph(
            X, radius, metric=metric, mode="connectivity", include_self=True
        ).toarray()
        nbrs1 = neighbors.NearestNeighbors(metric=metric, radius=radius).fit(X)
        assert_array_equal(nbrs_graph, nbrs1.radius_neighbors_graph(X).A)

    # Raise error when wrong parameters are supplied,
    X_nbrs = neighbors.NearestNeighbors(n_neighbors=3, metric="manhattan")
    X_nbrs.fit(X)
    with pytest.raises(ValueError):
        neighbors.kneighbors_graph(X_nbrs, 3, metric="euclidean")
    X_nbrs = neighbors.NearestNeighbors(radius=radius, metric="manhattan")
    X_nbrs.fit(X)
    with pytest.raises(ValueError):
        neighbors.radius_neighbors_graph(X_nbrs, radius, metric="euclidean")


def check_object_arrays(nparray, list_check):
    for ind, ele in enumerate(nparray):
        assert_array_equal(ele, list_check[ind])


def test_k_and_radius_neighbors_train_is_not_query():
    # Test kneighbors et.al when query is not training data

    for algorithm in ALGORITHMS:

        nn = neighbors.NearestNeighbors(n_neighbors=1, algorithm=algorithm)

        X = [[0], [1]]
        nn.fit(X)
        test_data = [[2], [1]]

        # Test neighbors.
        dist, ind = nn.kneighbors(test_data)
        assert_array_equal(dist, [[1], [0]])
        assert_array_equal(ind, [[1], [1]])
        dist, ind = nn.radius_neighbors([[2], [1]], radius=1.5)
        check_object_arrays(dist, [[1], [1, 0]])
        check_object_arrays(ind, [[1], [0, 1]])

        # Test the graph variants.
        assert_array_equal(nn.kneighbors_graph(test_data).A, [[0.0, 1.0], [0.0, 1.0]])
        assert_array_equal(
            nn.kneighbors_graph([[2], [1]], mode="distance").A,
            np.array([[0.0, 1.0], [0.0, 0.0]]),
        )
        rng = nn.radius_neighbors_graph([[2], [1]], radius=1.5)
        assert_array_equal(rng.A, [[0, 1], [1, 1]])


def test_k_and_radius_neighbors_X_None():
    # Test kneighbors et.al when query is None
    for algorithm in ALGORITHMS:

        nn = neighbors.NearestNeighbors(n_neighbors=1, algorithm=algorithm)

        X = [[0], [1]]
        nn.fit(X)

        dist, ind = nn.kneighbors()
        assert_array_equal(dist, [[1], [1]])
        assert_array_equal(ind, [[1], [0]])
        dist, ind = nn.radius_neighbors(None, radius=1.5)
        check_object_arrays(dist, [[1], [1]])
        check_object_arrays(ind, [[1], [0]])

        # Test the graph variants.
        rng = nn.radius_neighbors_graph(None, radius=1.5)
        kng = nn.kneighbors_graph(None)
        for graph in [rng, kng]:
            assert_array_equal(graph.A, [[0, 1], [1, 0]])
            assert_array_equal(graph.data, [1, 1])
            assert_array_equal(graph.indices, [1, 0])

        X = [[0, 1], [0, 1], [1, 1]]
        nn = neighbors.NearestNeighbors(n_neighbors=2, algorithm=algorithm)
        nn.fit(X)
        assert_array_equal(
            nn.kneighbors_graph().A,
            np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0]]),
        )


def test_k_and_radius_neighbors_duplicates():
    # Test behavior of kneighbors when duplicates are present in query

    for algorithm in ALGORITHMS:
        nn = neighbors.NearestNeighbors(n_neighbors=1, algorithm=algorithm)
        nn.fit([[0], [1]])

        # Do not do anything special to duplicates.
        kng = nn.kneighbors_graph([[0], [1]], mode="distance")
        assert_array_equal(kng.A, np.array([[0.0, 0.0], [0.0, 0.0]]))
        assert_array_equal(kng.data, [0.0, 0.0])
        assert_array_equal(kng.indices, [0, 1])

        dist, ind = nn.radius_neighbors([[0], [1]], radius=1.5)
        check_object_arrays(dist, [[0, 1], [1, 0]])
        check_object_arrays(ind, [[0, 1], [0, 1]])

        rng = nn.radius_neighbors_graph([[0], [1]], radius=1.5)
        assert_array_equal(rng.A, np.ones((2, 2)))

        rng = nn.radius_neighbors_graph([[0], [1]], radius=1.5, mode="distance")
        rng.sort_indices()
        assert_array_equal(rng.A, [[0, 1], [1, 0]])
        assert_array_equal(rng.indices, [0, 1, 0, 1])
        assert_array_equal(rng.data, [0, 1, 1, 0])

        # Mask the first duplicates when n_duplicates > n_neighbors.
        X = np.ones((3, 1))
        nn = neighbors.NearestNeighbors(n_neighbors=1, algorithm="brute")
        nn.fit(X)
        dist, ind = nn.kneighbors()
        assert_array_equal(dist, np.zeros((3, 1)))
        assert_array_equal(ind, [[1], [0], [1]])

        # Test that zeros are explicitly marked in kneighbors_graph.
        kng = nn.kneighbors_graph(mode="distance")
        assert_array_equal(kng.A, np.zeros((3, 3)))
        assert_array_equal(kng.data, np.zeros(3))
        assert_array_equal(kng.indices, [1.0, 0.0, 1.0])
        assert_array_equal(
            nn.kneighbors_graph().A,
            np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        )


def test_include_self_neighbors_graph():
    # Test include_self parameter in neighbors_graph
    X = [[2, 3], [4, 5]]
    kng = neighbors.kneighbors_graph(X, 1, include_self=True).A
    kng_not_self = neighbors.kneighbors_graph(X, 1, include_self=False).A
    assert_array_equal(kng, [[1.0, 0.0], [0.0, 1.0]])
    assert_array_equal(kng_not_self, [[0.0, 1.0], [1.0, 0.0]])

    rng = neighbors.radius_neighbors_graph(X, 5.0, include_self=True).A
    rng_not_self = neighbors.radius_neighbors_graph(X, 5.0, include_self=False).A
    assert_array_equal(rng, [[1.0, 1.0], [1.0, 1.0]])
    assert_array_equal(rng_not_self, [[0.0, 1.0], [1.0, 0.0]])


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_same_knn_parallel(algorithm):
    X, y = datasets.make_classification(
        n_samples=30, n_features=5, n_redundant=0, random_state=0
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = neighbors.KNeighborsClassifier(n_neighbors=3, algorithm=algorithm)
    clf.fit(X_train, y_train)
    y = clf.predict(X_test)
    dist, ind = clf.kneighbors(X_test)
    graph = clf.kneighbors_graph(X_test, mode="distance").toarray()

    clf.set_params(n_jobs=3)
    clf.fit(X_train, y_train)
    y_parallel = clf.predict(X_test)
    dist_parallel, ind_parallel = clf.kneighbors(X_test)
    graph_parallel = clf.kneighbors_graph(X_test, mode="distance").toarray()

    assert_array_equal(y, y_parallel)
    assert_array_almost_equal(dist, dist_parallel)
    assert_array_equal(ind, ind_parallel)
    assert_array_almost_equal(graph, graph_parallel)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_same_radius_neighbors_parallel(algorithm):
    X, y = datasets.make_classification(
        n_samples=30, n_features=5, n_redundant=0, random_state=0
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = neighbors.RadiusNeighborsClassifier(radius=10, algorithm=algorithm)
    clf.fit(X_train, y_train)
    y = clf.predict(X_test)
    dist, ind = clf.radius_neighbors(X_test)
    graph = clf.radius_neighbors_graph(X_test, mode="distance").toarray()

    clf.set_params(n_jobs=3)
    clf.fit(X_train, y_train)
    y_parallel = clf.predict(X_test)
    dist_parallel, ind_parallel = clf.radius_neighbors(X_test)
    graph_parallel = clf.radius_neighbors_graph(X_test, mode="distance").toarray()

    assert_array_equal(y, y_parallel)
    for i in range(len(dist)):
        assert_array_almost_equal(dist[i], dist_parallel[i])
        assert_array_equal(ind[i], ind_parallel[i])
    assert_array_almost_equal(graph, graph_parallel)


@pytest.mark.parametrize("backend", JOBLIB_BACKENDS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_knn_forcing_backend(backend, algorithm):
    # Non-regression test which ensure the knn methods are properly working
    # even when forcing the global joblib backend.
    with joblib.parallel_backend(backend):
        X, y = datasets.make_classification(
            n_samples=30, n_features=5, n_redundant=0, random_state=0
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        clf = neighbors.KNeighborsClassifier(
            n_neighbors=3, algorithm=algorithm, n_jobs=3
        )
        clf.fit(X_train, y_train)
        clf.predict(X_test)
        clf.kneighbors(X_test)
        clf.kneighbors_graph(X_test, mode="distance").toarray()


def test_dtype_convert():
    classifier = neighbors.KNeighborsClassifier(n_neighbors=1)
    CLASSES = 15
    X = np.eye(CLASSES)
    y = [ch for ch in "ABCDEFGHIJKLMNOPQRSTU"[:CLASSES]]

    result = classifier.fit(X, y).predict(X)
    assert_array_equal(result, y)


def test_sparse_metric_callable():
    def sparse_metric(x, y):  # Metric accepting sparse matrix input (only)
        assert issparse(x) and issparse(y)
        return x.dot(y.T).A.item()

    X = csr_matrix(
        [[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 0, 1, 0, 0]]  # Population matrix
    )

    Y = csr_matrix([[1, 1, 0, 1, 1], [1, 0, 0, 0, 1]])  # Query matrix

    nn = neighbors.NearestNeighbors(
        algorithm="brute", n_neighbors=2, metric=sparse_metric
    ).fit(X)
    N = nn.kneighbors(Y, return_distance=False)

    # GS indices of nearest neighbours in `X` for `sparse_metric`
    gold_standard_nn = np.array([[2, 1], [2, 1]])

    assert_array_equal(N, gold_standard_nn)


# ignore conversion to boolean in pairwise_distances
@ignore_warnings(category=DataConversionWarning)
def test_pairwise_boolean_distance():
    # Non-regression test for #4523
    # 'brute': uses scipy.spatial.distance through pairwise_distances
    # 'ball_tree': uses sklearn.neighbors._dist_metrics
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(6, 5))
    NN = neighbors.NearestNeighbors

    nn1 = NN(metric="jaccard", algorithm="brute").fit(X)
    nn2 = NN(metric="jaccard", algorithm="ball_tree").fit(X)
    assert_array_equal(nn1.kneighbors(X)[0], nn2.kneighbors(X)[0])


def test_radius_neighbors_predict_proba():
    for seed in range(5):
        X, y = datasets.make_classification(
            n_samples=50,
            n_features=5,
            n_informative=3,
            n_redundant=0,
            n_classes=3,
            random_state=seed,
        )
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
        outlier_label = int(2 - seed)
        clf = neighbors.RadiusNeighborsClassifier(radius=2, outlier_label=outlier_label)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        proba = clf.predict_proba(X_te)
        proba_label = proba.argmax(axis=1)
        proba_label = np.where(proba.sum(axis=1) == 0, outlier_label, proba_label)
        assert_array_equal(pred, proba_label)


def test_pipeline_with_nearest_neighbors_transformer():
    # Test chaining KNeighborsTransformer and classifiers/regressors
    rng = np.random.RandomState(0)
    X = 2 * rng.rand(40, 5) - 1
    X2 = 2 * rng.rand(40, 5) - 1
    y = rng.rand(40, 1)

    n_neighbors = 12
    radius = 1.5
    # We precompute more neighbors than necessary, to have equivalence between
    # k-neighbors estimator after radius-neighbors transformer, and vice-versa.
    factor = 2

    k_trans = neighbors.KNeighborsTransformer(n_neighbors=n_neighbors, mode="distance")
    k_trans_factor = neighbors.KNeighborsTransformer(
        n_neighbors=int(n_neighbors * factor), mode="distance"
    )

    r_trans = neighbors.RadiusNeighborsTransformer(radius=radius, mode="distance")
    r_trans_factor = neighbors.RadiusNeighborsTransformer(
        radius=int(radius * factor), mode="distance"
    )

    k_reg = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors)
    r_reg = neighbors.RadiusNeighborsRegressor(radius=radius)

    test_list = [
        (k_trans, k_reg),
        (k_trans_factor, r_reg),
        (r_trans, r_reg),
        (r_trans_factor, k_reg),
    ]

    for trans, reg in test_list:
        # compare the chained version and the compact version
        reg_compact = clone(reg)
        reg_precomp = clone(reg)
        reg_precomp.set_params(metric="precomputed")

        reg_chain = make_pipeline(clone(trans), reg_precomp)

        y_pred_chain = reg_chain.fit(X, y).predict(X2)
        y_pred_compact = reg_compact.fit(X, y).predict(X2)
        assert_array_almost_equal(y_pred_chain, y_pred_compact)


@pytest.mark.parametrize(
    "X, metric, metric_params, expected_algo",
    [
        (np.random.randint(10, size=(10, 10)), "precomputed", None, "brute"),
        (np.random.randn(10, 20), "euclidean", None, "brute"),
        (np.random.randn(8, 5), "euclidean", None, "brute"),
        (np.random.randn(10, 5), "euclidean", None, "kd_tree"),
        (np.random.randn(10, 5), "seuclidean", {"V": [2] * 5}, "ball_tree"),
        (np.random.randn(10, 5), "correlation", None, "brute"),
    ],
)
def test_auto_algorithm(X, metric, metric_params, expected_algo):
    model = neighbors.NearestNeighbors(
        n_neighbors=4, algorithm="auto", metric=metric, metric_params=metric_params
    )
    model.fit(X)
    assert model._fit_method == expected_algo


# TODO: Remove in 1.1
@pytest.mark.parametrize(
    "NearestNeighbors",
    [
        neighbors.KNeighborsClassifier,
        neighbors.KNeighborsRegressor,
        neighbors.NearestNeighbors,
    ],  # type: ignore
)
def test_pairwise_deprecated(NearestNeighbors):
    nn = NearestNeighbors(metric="precomputed")
    msg = r"Attribute `_pairwise` was deprecated in version 0\.24"
    with pytest.warns(FutureWarning, match=msg):
        nn._pairwise


# TODO: Remove in 1.3
def test_neighbors_distance_metric_deprecation():
    from sklearn.neighbors import DistanceMetric
    from sklearn.metrics import DistanceMetric as ActualDistanceMetric

    msg = r"This import path will be removed in 1\.3"
    with pytest.warns(FutureWarning, match=msg):
        dist_metric = DistanceMetric.get_metric("euclidean")

    assert isinstance(dist_metric, ActualDistanceMetric)
