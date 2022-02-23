"""
Testing for Clustering methods

"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import assert_array_equal

from sklearn.cluster import AffinityPropagation
from sklearn.cluster._affinity_propagation import _equal_similarities_and_preferences
from sklearn.cluster import affinity_propagation
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances

n_clusters = 3
centers = np.array([[1, 1], [-1, -1], [1, -1]]) + 10
X, _ = make_blobs(
    n_samples=60,
    n_features=2,
    centers=centers,
    cluster_std=0.4,
    shuffle=True,
    random_state=0,
)


def test_affinity_propagation():
    # Affinity Propagation algorithm
    # Compute similarities
    S = -euclidean_distances(X, squared=True)
    preference = np.median(S) * 10
    # Compute Affinity Propagation
    cluster_centers_indices, labels = affinity_propagation(
        S, preference=preference, random_state=39
    )

    n_clusters_ = len(cluster_centers_indices)

    assert n_clusters == n_clusters_

    af = AffinityPropagation(
        preference=preference, affinity="precomputed", random_state=28
    )
    labels_precomputed = af.fit(S).labels_

    af = AffinityPropagation(preference=preference, verbose=True, random_state=37)
    labels = af.fit(X).labels_

    assert_array_equal(labels, labels_precomputed)

    cluster_centers_indices = af.cluster_centers_indices_

    n_clusters_ = len(cluster_centers_indices)
    assert np.unique(labels).size == n_clusters_
    assert n_clusters == n_clusters_

    # Test also with no copy
    _, labels_no_copy = affinity_propagation(
        S, preference=preference, copy=False, random_state=74
    )
    assert_array_equal(labels, labels_no_copy)


def test_affinity_propagation_affinity_shape():
    """Check the shape of the affinity matrix when using `affinity_propagation."""
    S = -euclidean_distances(X, squared=True)
    err_msg = "S must be a square array"
    with pytest.raises(ValueError, match=err_msg):
        affinity_propagation(S[:, :-1])


@pytest.mark.parametrize(
    "input, params, err_type, err_msg",
    [
        (X, {"damping": 0}, ValueError, "damping == 0, must be >= 0.5"),
        (X, {"damping": 2}, ValueError, "damping == 2, must be < 1"),
        (X, {"max_iter": 0}, ValueError, "max_iter == 0, must be >= 1."),
        (X, {"convergence_iter": 0}, ValueError, "convergence_iter == 0, must be >= 1"),
        (X, {"affinity": "unknown"}, ValueError, "Affinity must be"),
        (
            csr_matrix((3, 3)),
            {"affinity": "precomputed"},
            TypeError,
            "A sparse matrix was passed, but dense data is required",
        ),
    ],
)
def test_affinity_propagation_params_validation(input, params, err_type, err_msg):
    """Check the parameters validation in `AffinityPropagation`."""
    with pytest.raises(err_type, match=err_msg):
        AffinityPropagation(**params).fit(input)


def test_affinity_propagation_predict():
    # Test AffinityPropagation.predict
    af = AffinityPropagation(affinity="euclidean", random_state=63)
    labels = af.fit_predict(X)
    labels2 = af.predict(X)
    assert_array_equal(labels, labels2)


def test_affinity_propagation_predict_error():
    # Test exception in AffinityPropagation.predict
    # Not fitted.
    af = AffinityPropagation(affinity="euclidean")
    with pytest.raises(ValueError):
        af.predict(X)

    # Predict not supported when affinity="precomputed".
    S = np.dot(X, X.T)
    af = AffinityPropagation(affinity="precomputed", random_state=57)
    af.fit(S)
    with pytest.raises(ValueError):
        af.predict(X)


def test_affinity_propagation_fit_non_convergence():
    # In case of non-convergence of affinity_propagation(), the cluster
    # centers should be an empty array and training samples should be labelled
    # as noise (-1)
    X = np.array([[0, 0], [1, 1], [-2, -2]])

    # Force non-convergence by allowing only a single iteration
    af = AffinityPropagation(preference=-10, max_iter=1, random_state=82)

    with pytest.warns(ConvergenceWarning):
        af.fit(X)
    assert_array_equal(np.empty((0, 2)), af.cluster_centers_)
    assert_array_equal(np.array([-1, -1, -1]), af.labels_)


def test_affinity_propagation_equal_mutual_similarities():
    X = np.array([[-1, 1], [1, -1]])
    S = -euclidean_distances(X, squared=True)

    # setting preference > similarity
    with pytest.warns(UserWarning, match="mutually equal"):
        cluster_center_indices, labels = affinity_propagation(S, preference=0)

    # expect every sample to become an exemplar
    assert_array_equal([0, 1], cluster_center_indices)
    assert_array_equal([0, 1], labels)

    # setting preference < similarity
    with pytest.warns(UserWarning, match="mutually equal"):
        cluster_center_indices, labels = affinity_propagation(S, preference=-10)

    # expect one cluster, with arbitrary (first) sample as exemplar
    assert_array_equal([0], cluster_center_indices)
    assert_array_equal([0, 0], labels)

    # setting different preferences
    with pytest.warns(None) as record:
        cluster_center_indices, labels = affinity_propagation(
            S, preference=[-20, -10], random_state=37
        )
    assert not len(record)

    # expect one cluster, with highest-preference sample as exemplar
    assert_array_equal([1], cluster_center_indices)
    assert_array_equal([0, 0], labels)


def test_affinity_propagation_predict_non_convergence():
    # In case of non-convergence of affinity_propagation(), the cluster
    # centers should be an empty array
    X = np.array([[0, 0], [1, 1], [-2, -2]])

    # Force non-convergence by allowing only a single iteration
    with pytest.warns(ConvergenceWarning):
        af = AffinityPropagation(preference=-10, max_iter=1, random_state=75).fit(X)

    # At prediction time, consider new samples as noise since there are no
    # clusters
    to_predict = np.array([[2, 2], [3, 3], [4, 4]])
    with pytest.warns(ConvergenceWarning):
        y = af.predict(to_predict)
    assert_array_equal(np.array([-1, -1, -1]), y)


def test_affinity_propagation_non_convergence_regressiontest():
    X = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1]])
    af = AffinityPropagation(affinity="euclidean", max_iter=2, random_state=34).fit(X)
    assert_array_equal(np.array([-1, -1, -1]), af.labels_)


def test_equal_similarities_and_preferences():
    # Unequal distances
    X = np.array([[0, 0], [1, 1], [-2, -2]])
    S = -euclidean_distances(X, squared=True)

    assert not _equal_similarities_and_preferences(S, np.array(0))
    assert not _equal_similarities_and_preferences(S, np.array([0, 0]))
    assert not _equal_similarities_and_preferences(S, np.array([0, 1]))

    # Equal distances
    X = np.array([[0, 0], [1, 1]])
    S = -euclidean_distances(X, squared=True)

    # Different preferences
    assert not _equal_similarities_and_preferences(S, np.array([0, 1]))

    # Same preferences
    assert _equal_similarities_and_preferences(S, np.array([0, 0]))
    assert _equal_similarities_and_preferences(S, np.array(0))


def test_affinity_propagation_random_state():
    # Significance of random_state parameter
    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(
        n_samples=300, centers=centers, cluster_std=0.5, random_state=0
    )
    # random_state = 0
    ap = AffinityPropagation(convergence_iter=1, max_iter=2, random_state=0)
    ap.fit(X)
    centers0 = ap.cluster_centers_

    # random_state = 76
    ap = AffinityPropagation(convergence_iter=1, max_iter=2, random_state=76)
    ap.fit(X)
    centers76 = ap.cluster_centers_

    assert np.mean((centers0 - centers76) ** 2) > 1


@pytest.mark.parametrize("centers", [csr_matrix(np.zeros((1, 10))), np.zeros((1, 10))])
def test_affinity_propagation_convergence_warning_dense_sparse(centers):
    """Non-regression, see #13334"""
    rng = np.random.RandomState(42)
    X = rng.rand(40, 10)
    y = (4 * rng.rand(40)).astype(int)
    ap = AffinityPropagation(random_state=46)
    ap.fit(X, y)
    ap.cluster_centers_ = centers
    with pytest.warns(None) as record:
        assert_array_equal(ap.predict(X), np.zeros(X.shape[0], dtype=int))
    assert len(record) == 0


def test_affinity_propagation_float32():
    # Test to fix incorrect clusters due to dtype change
    # (non-regression test for issue #10832)
    X = np.array(
        [[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype="float32"
    )
    afp = AffinityPropagation(preference=1, affinity="precomputed", random_state=0).fit(
        X
    )
    expected = np.array([0, 1, 1, 2])
    assert_array_equal(afp.labels_, expected)


def test_sparse_input_for_predict():
    # Test to make sure sparse inputs are accepted for predict
    # (non-regression test for issue #20049)
    af = AffinityPropagation(affinity="euclidean", random_state=42)
    af.fit(X)
    labels = af.predict(csr_matrix((2, 2)))
    assert_array_equal(labels, (2, 2))


def test_sparse_input_for_fit_predict():
    # Test to make sure sparse inputs are accepted for fit_predict
    # (non-regression test for issue #20049)
    af = AffinityPropagation(affinity="euclidean", random_state=42)
    rng = np.random.RandomState(42)
    X = csr_matrix(rng.randint(0, 2, size=(5, 5)))
    labels = af.fit_predict(X)
    assert_array_equal(labels, (0, 1, 1, 2, 3))


# TODO: Remove in 1.1
def test_affinity_propagation_pairwise_is_deprecated():
    afp = AffinityPropagation(affinity="precomputed")
    msg = r"Attribute `_pairwise` was deprecated in version 0\.24"
    with pytest.warns(FutureWarning, match=msg):
        afp._pairwise
