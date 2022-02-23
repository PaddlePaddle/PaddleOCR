"""
This is testing the equivalence between some estimators with internal nearest
neighbors computations, and the corresponding pipeline versions with
KNeighborsTransformer or RadiusNeighborsTransformer to precompute the
neighbors.
"""

import numpy as np
import pytest

from sklearn.utils._testing import assert_array_almost_equal
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

from sklearn.neighbors import KNeighborsTransformer
from sklearn.neighbors import RadiusNeighborsTransformer

from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE


def test_spectral_clustering():
    # Test chaining KNeighborsTransformer and SpectralClustering
    n_neighbors = 5
    X, _ = make_blobs(random_state=0)

    # compare the chained version and the compact version
    est_chain = make_pipeline(
        KNeighborsTransformer(n_neighbors=n_neighbors, mode="connectivity"),
        SpectralClustering(
            n_neighbors=n_neighbors, affinity="precomputed", random_state=42
        ),
    )
    est_compact = SpectralClustering(
        n_neighbors=n_neighbors, affinity="nearest_neighbors", random_state=42
    )
    labels_compact = est_compact.fit_predict(X)
    labels_chain = est_chain.fit_predict(X)
    assert_array_almost_equal(labels_chain, labels_compact)


def test_spectral_embedding():
    # Test chaining KNeighborsTransformer and SpectralEmbedding
    n_neighbors = 5

    n_samples = 1000
    centers = np.array(
        [
            [0.0, 5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 5.0, 1.0],
        ]
    )
    S, true_labels = make_blobs(
        n_samples=n_samples, centers=centers, cluster_std=1.0, random_state=42
    )

    # compare the chained version and the compact version
    est_chain = make_pipeline(
        KNeighborsTransformer(n_neighbors=n_neighbors, mode="connectivity"),
        SpectralEmbedding(
            n_neighbors=n_neighbors, affinity="precomputed", random_state=42
        ),
    )
    est_compact = SpectralEmbedding(
        n_neighbors=n_neighbors, affinity="nearest_neighbors", random_state=42
    )
    St_compact = est_compact.fit_transform(S)
    St_chain = est_chain.fit_transform(S)
    assert_array_almost_equal(St_chain, St_compact)


def test_dbscan():
    # Test chaining RadiusNeighborsTransformer and DBSCAN
    radius = 0.3
    n_clusters = 3
    X = generate_clustered_data(n_clusters=n_clusters)

    # compare the chained version and the compact version
    est_chain = make_pipeline(
        RadiusNeighborsTransformer(radius=radius, mode="distance"),
        DBSCAN(metric="precomputed", eps=radius),
    )
    est_compact = DBSCAN(eps=radius)

    labels_chain = est_chain.fit_predict(X)
    labels_compact = est_compact.fit_predict(X)
    assert_array_almost_equal(labels_chain, labels_compact)


def test_isomap():
    # Test chaining KNeighborsTransformer and Isomap with
    # neighbors_algorithm='precomputed'
    algorithm = "auto"
    n_neighbors = 10

    X, _ = make_blobs(random_state=0)
    X2, _ = make_blobs(random_state=1)

    # compare the chained version and the compact version
    est_chain = make_pipeline(
        KNeighborsTransformer(
            n_neighbors=n_neighbors, algorithm=algorithm, mode="distance"
        ),
        Isomap(n_neighbors=n_neighbors, metric="precomputed"),
    )
    est_compact = Isomap(n_neighbors=n_neighbors, neighbors_algorithm=algorithm)

    Xt_chain = est_chain.fit_transform(X)
    Xt_compact = est_compact.fit_transform(X)
    assert_array_almost_equal(Xt_chain, Xt_compact)

    Xt_chain = est_chain.transform(X2)
    Xt_compact = est_compact.transform(X2)
    assert_array_almost_equal(Xt_chain, Xt_compact)


# TODO: Remove filterwarning in 1.2
@pytest.mark.filterwarnings("ignore:.*TSNE will change.*:FutureWarning")
def test_tsne():
    # Test chaining KNeighborsTransformer and TSNE
    n_iter = 250
    perplexity = 5
    n_neighbors = int(3.0 * perplexity + 1)

    rng = np.random.RandomState(0)
    X = rng.randn(20, 2)

    for metric in ["minkowski", "sqeuclidean"]:

        # compare the chained version and the compact version
        est_chain = make_pipeline(
            KNeighborsTransformer(
                n_neighbors=n_neighbors, mode="distance", metric=metric
            ),
            TSNE(
                metric="precomputed",
                perplexity=perplexity,
                method="barnes_hut",
                random_state=42,
                n_iter=n_iter,
                square_distances=True,
            ),
        )
        est_compact = TSNE(
            metric=metric,
            perplexity=perplexity,
            n_iter=n_iter,
            method="barnes_hut",
            random_state=42,
            square_distances=True,
        )

        Xt_chain = est_chain.fit_transform(X)
        Xt_compact = est_compact.fit_transform(X)
        assert_array_almost_equal(Xt_chain, Xt_compact)


def test_lof_novelty_false():
    # Test chaining KNeighborsTransformer and LocalOutlierFactor
    n_neighbors = 4

    rng = np.random.RandomState(0)
    X = rng.randn(40, 2)

    # compare the chained version and the compact version
    est_chain = make_pipeline(
        KNeighborsTransformer(n_neighbors=n_neighbors, mode="distance"),
        LocalOutlierFactor(
            metric="precomputed",
            n_neighbors=n_neighbors,
            novelty=False,
            contamination="auto",
        ),
    )
    est_compact = LocalOutlierFactor(
        n_neighbors=n_neighbors, novelty=False, contamination="auto"
    )

    pred_chain = est_chain.fit_predict(X)
    pred_compact = est_compact.fit_predict(X)
    assert_array_almost_equal(pred_chain, pred_compact)


def test_lof_novelty_true():
    # Test chaining KNeighborsTransformer and LocalOutlierFactor
    n_neighbors = 4

    rng = np.random.RandomState(0)
    X1 = rng.randn(40, 2)
    X2 = rng.randn(40, 2)

    # compare the chained version and the compact version
    est_chain = make_pipeline(
        KNeighborsTransformer(n_neighbors=n_neighbors, mode="distance"),
        LocalOutlierFactor(
            metric="precomputed",
            n_neighbors=n_neighbors,
            novelty=True,
            contamination="auto",
        ),
    )
    est_compact = LocalOutlierFactor(
        n_neighbors=n_neighbors, novelty=True, contamination="auto"
    )

    pred_chain = est_chain.fit(X1).predict(X2)
    pred_compact = est_compact.fit(X1).predict(X2)
    assert_array_almost_equal(pred_chain, pred_compact)


def test_kneighbors_regressor():
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

    k_trans = KNeighborsTransformer(n_neighbors=n_neighbors, mode="distance")
    k_trans_factor = KNeighborsTransformer(
        n_neighbors=int(n_neighbors * factor), mode="distance"
    )

    r_trans = RadiusNeighborsTransformer(radius=radius, mode="distance")
    r_trans_factor = RadiusNeighborsTransformer(
        radius=int(radius * factor), mode="distance"
    )

    k_reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    r_reg = RadiusNeighborsRegressor(radius=radius)

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
