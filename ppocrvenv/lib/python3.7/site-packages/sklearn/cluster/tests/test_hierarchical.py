"""
Several basic tests for hierarchical clustering procedures

"""
# Authors: Vincent Michel, 2010, Gael Varoquaux 2012,
#          Matteo Visconti di Oleggio Castello 2014
# License: BSD 3 clause
import itertools
from tempfile import mkdtemp
import shutil
import pytest
from functools import partial

import numpy as np
from scipy import sparse
from scipy.cluster import hierarchy
from scipy.sparse.csgraph import connected_components

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.tests.test_dist_metrics import METRICS_DEFAULT_PARAMS
from sklearn.utils._testing import assert_almost_equal, create_memmap_backed_data
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import ignore_warnings

from sklearn.cluster import ward_tree
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from sklearn.cluster._agglomerative import (
    _hc_cut,
    _TREE_BUILDERS,
    linkage_tree,
    _fix_connectivity,
)
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.metrics import DistanceMetric
from sklearn.metrics.pairwise import (
    PAIRED_DISTANCES,
    cosine_distances,
    manhattan_distances,
    pairwise_distances,
)
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster._hierarchical_fast import (
    average_merge,
    max_merge,
    mst_linkage_core,
)
from sklearn.utils._fast_dict import IntFloatDict
from sklearn.utils._testing import assert_array_equal
from sklearn.datasets import make_moons, make_circles


def test_linkage_misc():
    # Misc tests on linkage
    rng = np.random.RandomState(42)
    X = rng.normal(size=(5, 5))
    with pytest.raises(ValueError):
        AgglomerativeClustering(linkage="foo").fit(X)

    with pytest.raises(ValueError):
        linkage_tree(X, linkage="foo")

    with pytest.raises(ValueError):
        linkage_tree(X, connectivity=np.ones((4, 4)))

    # Smoke test FeatureAgglomeration
    FeatureAgglomeration().fit(X)

    # test hierarchical clustering on a precomputed distances matrix
    dis = cosine_distances(X)

    res = linkage_tree(dis, affinity="precomputed")
    assert_array_equal(res[0], linkage_tree(X, affinity="cosine")[0])

    # test hierarchical clustering on a precomputed distances matrix
    res = linkage_tree(X, affinity=manhattan_distances)
    assert_array_equal(res[0], linkage_tree(X, affinity="manhattan")[0])


def test_structured_linkage_tree():
    # Check that we obtain the correct solution for structured linkage trees.
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    # Avoiding a mask with only 'True' entries
    mask[4:7, 4:7] = 0
    X = rng.randn(50, 100)
    connectivity = grid_to_graph(*mask.shape)
    for tree_builder in _TREE_BUILDERS.values():
        children, n_components, n_leaves, parent = tree_builder(
            X.T, connectivity=connectivity
        )
        n_nodes = 2 * X.shape[1] - 1
        assert len(children) + n_leaves == n_nodes
        # Check that ward_tree raises a ValueError with a connectivity matrix
        # of the wrong shape
        with pytest.raises(ValueError):
            tree_builder(X.T, connectivity=np.ones((4, 4)))
        # Check that fitting with no samples raises an error
        with pytest.raises(ValueError):
            tree_builder(X.T[:0], connectivity=connectivity)


def test_unstructured_linkage_tree():
    # Check that we obtain the correct solution for unstructured linkage trees.
    rng = np.random.RandomState(0)
    X = rng.randn(50, 100)
    for this_X in (X, X[0]):
        # With specified a number of clusters just for the sake of
        # raising a warning and testing the warning code
        with ignore_warnings():
            with pytest.warns(UserWarning):
                children, n_nodes, n_leaves, parent = ward_tree(this_X.T, n_clusters=10)
        n_nodes = 2 * X.shape[1] - 1
        assert len(children) + n_leaves == n_nodes

    for tree_builder in _TREE_BUILDERS.values():
        for this_X in (X, X[0]):
            with ignore_warnings():
                with pytest.warns(UserWarning):
                    children, n_nodes, n_leaves, parent = tree_builder(
                        this_X.T, n_clusters=10
                    )
            n_nodes = 2 * X.shape[1] - 1
            assert len(children) + n_leaves == n_nodes


def test_height_linkage_tree():
    # Check that the height of the results of linkage tree is sorted.
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    X = rng.randn(50, 100)
    connectivity = grid_to_graph(*mask.shape)
    for linkage_func in _TREE_BUILDERS.values():
        children, n_nodes, n_leaves, parent = linkage_func(
            X.T, connectivity=connectivity
        )
        n_nodes = 2 * X.shape[1] - 1
        assert len(children) + n_leaves == n_nodes


def test_agglomerative_clustering_wrong_arg_memory():
    # Test either if an error is raised when memory is not
    # either a str or a joblib.Memory instance
    rng = np.random.RandomState(0)
    n_samples = 100
    X = rng.randn(n_samples, 50)
    memory = 5
    clustering = AgglomerativeClustering(memory=memory)
    with pytest.raises(ValueError):
        clustering.fit(X)


def test_zero_cosine_linkage_tree():
    # Check that zero vectors in X produce an error when
    # 'cosine' affinity is used
    X = np.array([[0, 1], [0, 0]])
    msg = "Cosine affinity cannot be used when X contains zero vectors"
    with pytest.raises(ValueError, match=msg):
        linkage_tree(X, affinity="cosine")


@pytest.mark.parametrize("n_clusters, distance_threshold", [(None, 0.5), (10, None)])
@pytest.mark.parametrize("compute_distances", [True, False])
@pytest.mark.parametrize("linkage", ["ward", "complete", "average", "single"])
def test_agglomerative_clustering_distances(
    n_clusters, compute_distances, distance_threshold, linkage
):
    # Check that when `compute_distances` is True or `distance_threshold` is
    # given, the fitted model has an attribute `distances_`.
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    n_samples = 100
    X = rng.randn(n_samples, 50)
    connectivity = grid_to_graph(*mask.shape)

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        connectivity=connectivity,
        linkage=linkage,
        distance_threshold=distance_threshold,
        compute_distances=compute_distances,
    )
    clustering.fit(X)
    if compute_distances or (distance_threshold is not None):
        assert hasattr(clustering, "distances_")
        n_children = clustering.children_.shape[0]
        n_nodes = n_children + 1
        assert clustering.distances_.shape == (n_nodes - 1,)
    else:
        assert not hasattr(clustering, "distances_")


def test_agglomerative_clustering():
    # Check that we obtain the correct number of clusters with
    # agglomerative clustering.
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    n_samples = 100
    X = rng.randn(n_samples, 50)
    connectivity = grid_to_graph(*mask.shape)
    for linkage in ("ward", "complete", "average", "single"):
        clustering = AgglomerativeClustering(
            n_clusters=10, connectivity=connectivity, linkage=linkage
        )
        clustering.fit(X)
        # test caching
        try:
            tempdir = mkdtemp()
            clustering = AgglomerativeClustering(
                n_clusters=10,
                connectivity=connectivity,
                memory=tempdir,
                linkage=linkage,
            )
            clustering.fit(X)
            labels = clustering.labels_
            assert np.size(np.unique(labels)) == 10
        finally:
            shutil.rmtree(tempdir)
        # Turn caching off now
        clustering = AgglomerativeClustering(
            n_clusters=10, connectivity=connectivity, linkage=linkage
        )
        # Check that we obtain the same solution with early-stopping of the
        # tree building
        clustering.compute_full_tree = False
        clustering.fit(X)
        assert_almost_equal(normalized_mutual_info_score(clustering.labels_, labels), 1)
        clustering.connectivity = None
        clustering.fit(X)
        assert np.size(np.unique(clustering.labels_)) == 10
        # Check that we raise a TypeError on dense matrices
        clustering = AgglomerativeClustering(
            n_clusters=10,
            connectivity=sparse.lil_matrix(connectivity.toarray()[:10, :10]),
            linkage=linkage,
        )
        with pytest.raises(ValueError):
            clustering.fit(X)

    # Test that using ward with another metric than euclidean raises an
    # exception
    clustering = AgglomerativeClustering(
        n_clusters=10,
        connectivity=connectivity.toarray(),
        affinity="manhattan",
        linkage="ward",
    )
    with pytest.raises(ValueError):
        clustering.fit(X)

    # Test using another metric than euclidean works with linkage complete
    for affinity in PAIRED_DISTANCES.keys():
        # Compare our (structured) implementation to scipy
        clustering = AgglomerativeClustering(
            n_clusters=10,
            connectivity=np.ones((n_samples, n_samples)),
            affinity=affinity,
            linkage="complete",
        )
        clustering.fit(X)
        clustering2 = AgglomerativeClustering(
            n_clusters=10, connectivity=None, affinity=affinity, linkage="complete"
        )
        clustering2.fit(X)
        assert_almost_equal(
            normalized_mutual_info_score(clustering2.labels_, clustering.labels_), 1
        )

    # Test that using a distance matrix (affinity = 'precomputed') has same
    # results (with connectivity constraints)
    clustering = AgglomerativeClustering(
        n_clusters=10, connectivity=connectivity, linkage="complete"
    )
    clustering.fit(X)
    X_dist = pairwise_distances(X)
    clustering2 = AgglomerativeClustering(
        n_clusters=10,
        connectivity=connectivity,
        affinity="precomputed",
        linkage="complete",
    )
    clustering2.fit(X_dist)
    assert_array_equal(clustering.labels_, clustering2.labels_)


def test_agglomerative_clustering_memory_mapped():
    """AgglomerativeClustering must work on mem-mapped dataset.

    Non-regression test for issue #19875.
    """
    rng = np.random.RandomState(0)
    Xmm = create_memmap_backed_data(rng.randn(50, 100))
    AgglomerativeClustering(affinity="euclidean", linkage="single").fit(Xmm)


def test_ward_agglomeration():
    # Check that we obtain the correct solution in a simplistic case
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    X = rng.randn(50, 100)
    connectivity = grid_to_graph(*mask.shape)
    agglo = FeatureAgglomeration(n_clusters=5, connectivity=connectivity)
    agglo.fit(X)
    assert np.size(np.unique(agglo.labels_)) == 5

    X_red = agglo.transform(X)
    assert X_red.shape[1] == 5
    X_full = agglo.inverse_transform(X_red)
    assert np.unique(X_full[0]).size == 5
    assert_array_almost_equal(agglo.transform(X_full), X_red)

    # Check that fitting with no samples raises a ValueError
    with pytest.raises(ValueError):
        agglo.fit(X[:0])


def test_single_linkage_clustering():
    # Check that we get the correct result in two emblematic cases
    moons, moon_labels = make_moons(noise=0.05, random_state=42)
    clustering = AgglomerativeClustering(n_clusters=2, linkage="single")
    clustering.fit(moons)
    assert_almost_equal(
        normalized_mutual_info_score(clustering.labels_, moon_labels), 1
    )

    circles, circle_labels = make_circles(factor=0.5, noise=0.025, random_state=42)
    clustering = AgglomerativeClustering(n_clusters=2, linkage="single")
    clustering.fit(circles)
    assert_almost_equal(
        normalized_mutual_info_score(clustering.labels_, circle_labels), 1
    )


def assess_same_labelling(cut1, cut2):
    """Util for comparison with scipy"""
    co_clust = []
    for cut in [cut1, cut2]:
        n = len(cut)
        k = cut.max() + 1
        ecut = np.zeros((n, k))
        ecut[np.arange(n), cut] = 1
        co_clust.append(np.dot(ecut, ecut.T))
    assert (co_clust[0] == co_clust[1]).all()


def test_sparse_scikit_vs_scipy():
    # Test scikit linkage with full connectivity (i.e. unstructured) vs scipy
    n, p, k = 10, 5, 3
    rng = np.random.RandomState(0)

    # Not using a lil_matrix here, just to check that non sparse
    # matrices are well handled
    connectivity = np.ones((n, n))
    for linkage in _TREE_BUILDERS.keys():
        for i in range(5):
            X = 0.1 * rng.normal(size=(n, p))
            X -= 4.0 * np.arange(n)[:, np.newaxis]
            X -= X.mean(axis=1)[:, np.newaxis]

            out = hierarchy.linkage(X, method=linkage)

            children_ = out[:, :2].astype(int, copy=False)
            children, _, n_leaves, _ = _TREE_BUILDERS[linkage](
                X, connectivity=connectivity
            )

            # Sort the order of child nodes per row for consistency
            children.sort(axis=1)
            assert_array_equal(
                children,
                children_,
                "linkage tree differs from scipy impl for linkage: " + linkage,
            )

            cut = _hc_cut(k, children, n_leaves)
            cut_ = _hc_cut(k, children_, n_leaves)
            assess_same_labelling(cut, cut_)

    # Test error management in _hc_cut
    with pytest.raises(ValueError):
        _hc_cut(n_leaves + 1, children, n_leaves)


# Make sure our custom mst_linkage_core gives
# the same results as scipy's builtin
@pytest.mark.parametrize("seed", range(5))
def test_vector_scikit_single_vs_scipy_single(seed):
    n_samples, n_features, n_clusters = 10, 5, 3
    rng = np.random.RandomState(seed)
    X = 0.1 * rng.normal(size=(n_samples, n_features))
    X -= 4.0 * np.arange(n_samples)[:, np.newaxis]
    X -= X.mean(axis=1)[:, np.newaxis]

    out = hierarchy.linkage(X, method="single")
    children_scipy = out[:, :2].astype(int)

    children, _, n_leaves, _ = _TREE_BUILDERS["single"](X)

    # Sort the order of child nodes per row for consistency
    children.sort(axis=1)
    assert_array_equal(
        children,
        children_scipy,
        "linkage tree differs from scipy impl for single linkage.",
    )

    cut = _hc_cut(n_clusters, children, n_leaves)
    cut_scipy = _hc_cut(n_clusters, children_scipy, n_leaves)
    assess_same_labelling(cut, cut_scipy)


@pytest.mark.parametrize("metric_param_grid", METRICS_DEFAULT_PARAMS)
def test_mst_linkage_core_memory_mapped(metric_param_grid):
    """The MST-LINKAGE-CORE algorithm must work on mem-mapped dataset.

    Non-regression test for issue #19875.
    """
    rng = np.random.RandomState(seed=1)
    X = rng.normal(size=(20, 4))
    Xmm = create_memmap_backed_data(X)
    metric, param_grid = metric_param_grid
    keys = param_grid.keys()
    for vals in itertools.product(*param_grid.values()):
        kwargs = dict(zip(keys, vals))
        distance_metric = DistanceMetric.get_metric(metric, **kwargs)
        mst = mst_linkage_core(X, distance_metric)
        mst_mm = mst_linkage_core(Xmm, distance_metric)
        np.testing.assert_equal(mst, mst_mm)


def test_identical_points():
    # Ensure identical points are handled correctly when using mst with
    # a sparse connectivity matrix
    X = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]])
    true_labels = np.array([0, 0, 1, 1, 2, 2])
    connectivity = kneighbors_graph(X, n_neighbors=3, include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)
    connectivity, n_components = _fix_connectivity(X, connectivity, "euclidean")

    for linkage in ("single", "average", "average", "ward"):
        clustering = AgglomerativeClustering(
            n_clusters=3, linkage=linkage, connectivity=connectivity
        )
        clustering.fit(X)

        assert_almost_equal(
            normalized_mutual_info_score(clustering.labels_, true_labels), 1
        )


def test_connectivity_propagation():
    # Check that connectivity in the ward tree is propagated correctly during
    # merging.
    X = np.array(
        [
            (0.014, 0.120),
            (0.014, 0.099),
            (0.014, 0.097),
            (0.017, 0.153),
            (0.017, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.152),
            (0.018, 0.149),
            (0.018, 0.144),
        ]
    )
    connectivity = kneighbors_graph(X, 10, include_self=False)
    ward = AgglomerativeClustering(
        n_clusters=4, connectivity=connectivity, linkage="ward"
    )
    # If changes are not propagated correctly, fit crashes with an
    # IndexError
    ward.fit(X)


def test_ward_tree_children_order():
    # Check that children are ordered in the same way for both structured and
    # unstructured versions of ward_tree.

    # test on five random datasets
    n, p = 10, 5
    rng = np.random.RandomState(0)

    connectivity = np.ones((n, n))
    for i in range(5):
        X = 0.1 * rng.normal(size=(n, p))
        X -= 4.0 * np.arange(n)[:, np.newaxis]
        X -= X.mean(axis=1)[:, np.newaxis]

        out_unstructured = ward_tree(X)
        out_structured = ward_tree(X, connectivity=connectivity)

        assert_array_equal(out_unstructured[0], out_structured[0])


def test_ward_linkage_tree_return_distance():
    # Test return_distance option on linkage and ward trees

    # test that return_distance when set true, gives same
    # output on both structured and unstructured clustering.
    n, p = 10, 5
    rng = np.random.RandomState(0)

    connectivity = np.ones((n, n))
    for i in range(5):
        X = 0.1 * rng.normal(size=(n, p))
        X -= 4.0 * np.arange(n)[:, np.newaxis]
        X -= X.mean(axis=1)[:, np.newaxis]

        out_unstructured = ward_tree(X, return_distance=True)
        out_structured = ward_tree(X, connectivity=connectivity, return_distance=True)

        # get children
        children_unstructured = out_unstructured[0]
        children_structured = out_structured[0]

        # check if we got the same clusters
        assert_array_equal(children_unstructured, children_structured)

        # check if the distances are the same
        dist_unstructured = out_unstructured[-1]
        dist_structured = out_structured[-1]

        assert_array_almost_equal(dist_unstructured, dist_structured)

        for linkage in ["average", "complete", "single"]:
            structured_items = linkage_tree(
                X, connectivity=connectivity, linkage=linkage, return_distance=True
            )[-1]
            unstructured_items = linkage_tree(X, linkage=linkage, return_distance=True)[
                -1
            ]
            structured_dist = structured_items[-1]
            unstructured_dist = unstructured_items[-1]
            structured_children = structured_items[0]
            unstructured_children = unstructured_items[0]
            assert_array_almost_equal(structured_dist, unstructured_dist)
            assert_array_almost_equal(structured_children, unstructured_children)

    # test on the following dataset where we know the truth
    # taken from scipy/cluster/tests/hierarchy_test_data.py
    X = np.array(
        [
            [1.43054825, -7.5693489],
            [6.95887839, 6.82293382],
            [2.87137846, -9.68248579],
            [7.87974764, -6.05485803],
            [8.24018364, -6.09495602],
            [7.39020262, 8.54004355],
        ]
    )
    # truth
    linkage_X_ward = np.array(
        [
            [3.0, 4.0, 0.36265956, 2.0],
            [1.0, 5.0, 1.77045373, 2.0],
            [0.0, 2.0, 2.55760419, 2.0],
            [6.0, 8.0, 9.10208346, 4.0],
            [7.0, 9.0, 24.7784379, 6.0],
        ]
    )

    linkage_X_complete = np.array(
        [
            [3.0, 4.0, 0.36265956, 2.0],
            [1.0, 5.0, 1.77045373, 2.0],
            [0.0, 2.0, 2.55760419, 2.0],
            [6.0, 8.0, 6.96742194, 4.0],
            [7.0, 9.0, 18.77445997, 6.0],
        ]
    )

    linkage_X_average = np.array(
        [
            [3.0, 4.0, 0.36265956, 2.0],
            [1.0, 5.0, 1.77045373, 2.0],
            [0.0, 2.0, 2.55760419, 2.0],
            [6.0, 8.0, 6.55832839, 4.0],
            [7.0, 9.0, 15.44089605, 6.0],
        ]
    )

    n_samples, n_features = np.shape(X)
    connectivity_X = np.ones((n_samples, n_samples))

    out_X_unstructured = ward_tree(X, return_distance=True)
    out_X_structured = ward_tree(X, connectivity=connectivity_X, return_distance=True)

    # check that the labels are the same
    assert_array_equal(linkage_X_ward[:, :2], out_X_unstructured[0])
    assert_array_equal(linkage_X_ward[:, :2], out_X_structured[0])

    # check that the distances are correct
    assert_array_almost_equal(linkage_X_ward[:, 2], out_X_unstructured[4])
    assert_array_almost_equal(linkage_X_ward[:, 2], out_X_structured[4])

    linkage_options = ["complete", "average", "single"]
    X_linkage_truth = [linkage_X_complete, linkage_X_average]
    for (linkage, X_truth) in zip(linkage_options, X_linkage_truth):
        out_X_unstructured = linkage_tree(X, return_distance=True, linkage=linkage)
        out_X_structured = linkage_tree(
            X, connectivity=connectivity_X, linkage=linkage, return_distance=True
        )

        # check that the labels are the same
        assert_array_equal(X_truth[:, :2], out_X_unstructured[0])
        assert_array_equal(X_truth[:, :2], out_X_structured[0])

        # check that the distances are correct
        assert_array_almost_equal(X_truth[:, 2], out_X_unstructured[4])
        assert_array_almost_equal(X_truth[:, 2], out_X_structured[4])


def test_connectivity_fixing_non_lil():
    # Check non regression of a bug if a non item assignable connectivity is
    # provided with more than one component.
    # create dummy data
    x = np.array([[0, 0], [1, 1]])
    # create a mask with several components to force connectivity fixing
    m = np.array([[True, False], [False, True]])
    c = grid_to_graph(n_x=2, n_y=2, mask=m)
    w = AgglomerativeClustering(connectivity=c, linkage="ward")
    with pytest.warns(UserWarning):
        w.fit(x)


def test_int_float_dict():
    rng = np.random.RandomState(0)
    keys = np.unique(rng.randint(100, size=10).astype(np.intp, copy=False))
    values = rng.rand(len(keys))

    d = IntFloatDict(keys, values)
    for key, value in zip(keys, values):
        assert d[key] == value

    other_keys = np.arange(50, dtype=np.intp)[::2]
    other_values = np.full(50, 0.5)[::2]
    other = IntFloatDict(other_keys, other_values)
    # Complete smoke test
    max_merge(d, other, mask=np.ones(100, dtype=np.intp), n_a=1, n_b=1)
    average_merge(d, other, mask=np.ones(100, dtype=np.intp), n_a=1, n_b=1)


def test_connectivity_callable():
    rng = np.random.RandomState(0)
    X = rng.rand(20, 5)
    connectivity = kneighbors_graph(X, 3, include_self=False)
    aglc1 = AgglomerativeClustering(connectivity=connectivity)
    aglc2 = AgglomerativeClustering(
        connectivity=partial(kneighbors_graph, n_neighbors=3, include_self=False)
    )
    aglc1.fit(X)
    aglc2.fit(X)
    assert_array_equal(aglc1.labels_, aglc2.labels_)


def test_connectivity_ignores_diagonal():
    rng = np.random.RandomState(0)
    X = rng.rand(20, 5)
    connectivity = kneighbors_graph(X, 3, include_self=False)
    connectivity_include_self = kneighbors_graph(X, 3, include_self=True)
    aglc1 = AgglomerativeClustering(connectivity=connectivity)
    aglc2 = AgglomerativeClustering(connectivity=connectivity_include_self)
    aglc1.fit(X)
    aglc2.fit(X)
    assert_array_equal(aglc1.labels_, aglc2.labels_)


def test_compute_full_tree():
    # Test that the full tree is computed if n_clusters is small
    rng = np.random.RandomState(0)
    X = rng.randn(10, 2)
    connectivity = kneighbors_graph(X, 5, include_self=False)

    # When n_clusters is less, the full tree should be built
    # that is the number of merges should be n_samples - 1
    agc = AgglomerativeClustering(n_clusters=2, connectivity=connectivity)
    agc.fit(X)
    n_samples = X.shape[0]
    n_nodes = agc.children_.shape[0]
    assert n_nodes == n_samples - 1

    # When n_clusters is large, greater than max of 100 and 0.02 * n_samples.
    # we should stop when there are n_clusters.
    n_clusters = 101
    X = rng.randn(200, 2)
    connectivity = kneighbors_graph(X, 10, include_self=False)
    agc = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity)
    agc.fit(X)
    n_samples = X.shape[0]
    n_nodes = agc.children_.shape[0]
    assert n_nodes == n_samples - n_clusters


def test_n_components():
    # Test n_components returned by linkage, average and ward tree
    rng = np.random.RandomState(0)
    X = rng.rand(5, 5)

    # Connectivity matrix having five components.
    connectivity = np.eye(5)

    for linkage_func in _TREE_BUILDERS.values():
        assert ignore_warnings(linkage_func)(X, connectivity=connectivity)[1] == 5


def test_agg_n_clusters():
    # Test that an error is raised when n_clusters <= 0

    rng = np.random.RandomState(0)
    X = rng.rand(20, 10)
    for n_clus in [-1, 0]:
        agc = AgglomerativeClustering(n_clusters=n_clus)
        msg = "n_clusters should be an integer greater than 0. %s was provided." % str(
            agc.n_clusters
        )
        with pytest.raises(ValueError, match=msg):
            agc.fit(X)


def test_affinity_passed_to_fix_connectivity():
    # Test that the affinity parameter is actually passed to the pairwise
    # function

    size = 2
    rng = np.random.RandomState(0)
    X = rng.randn(size, size)
    mask = np.array([True, False, False, True])

    connectivity = grid_to_graph(n_x=size, n_y=size, mask=mask, return_as=np.ndarray)

    class FakeAffinity:
        def __init__(self):
            self.counter = 0

        def increment(self, *args, **kwargs):
            self.counter += 1
            return self.counter

    fa = FakeAffinity()

    linkage_tree(X, connectivity=connectivity, affinity=fa.increment)

    assert fa.counter == 3


@pytest.mark.parametrize("linkage", ["ward", "complete", "average"])
def test_agglomerative_clustering_with_distance_threshold(linkage):
    # Check that we obtain the correct number of clusters with
    # agglomerative clustering with distance_threshold.
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    n_samples = 100
    X = rng.randn(n_samples, 50)
    connectivity = grid_to_graph(*mask.shape)
    # test when distance threshold is set to 10
    distance_threshold = 10
    for conn in [None, connectivity]:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            connectivity=conn,
            linkage=linkage,
        )
        clustering.fit(X)
        clusters_produced = clustering.labels_
        num_clusters_produced = len(np.unique(clustering.labels_))
        # test if the clusters produced match the point in the linkage tree
        # where the distance exceeds the threshold
        tree_builder = _TREE_BUILDERS[linkage]
        children, n_components, n_leaves, parent, distances = tree_builder(
            X, connectivity=conn, n_clusters=None, return_distance=True
        )
        num_clusters_at_threshold = (
            np.count_nonzero(distances >= distance_threshold) + 1
        )
        # test number of clusters produced
        assert num_clusters_at_threshold == num_clusters_produced
        # test clusters produced
        clusters_at_threshold = _hc_cut(
            n_clusters=num_clusters_produced, children=children, n_leaves=n_leaves
        )
        assert np.array_equiv(clusters_produced, clusters_at_threshold)


def test_small_distance_threshold():
    rng = np.random.RandomState(0)
    n_samples = 10
    X = rng.randint(-300, 300, size=(n_samples, 3))
    # this should result in all data in their own clusters, given that
    # their pairwise distances are bigger than .1 (which may not be the case
    # with a different random seed).
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1.0, linkage="single"
    ).fit(X)
    # check that the pairwise distances are indeed all larger than .1
    all_distances = pairwise_distances(X, metric="minkowski", p=2)
    np.fill_diagonal(all_distances, np.inf)
    assert np.all(all_distances > 0.1)
    assert clustering.n_clusters_ == n_samples


def test_cluster_distances_with_distance_threshold():
    rng = np.random.RandomState(0)
    n_samples = 100
    X = rng.randint(-10, 10, size=(n_samples, 3))
    # check the distances within the clusters and with other clusters
    distance_threshold = 4
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold, linkage="single"
    ).fit(X)
    labels = clustering.labels_
    D = pairwise_distances(X, metric="minkowski", p=2)
    # to avoid taking the 0 diagonal in min()
    np.fill_diagonal(D, np.inf)
    for label in np.unique(labels):
        in_cluster_mask = labels == label
        max_in_cluster_distance = (
            D[in_cluster_mask][:, in_cluster_mask].min(axis=0).max()
        )
        min_out_cluster_distance = (
            D[in_cluster_mask][:, ~in_cluster_mask].min(axis=0).min()
        )
        # single data point clusters only have that inf diagonal here
        if in_cluster_mask.sum() > 1:
            assert max_in_cluster_distance < distance_threshold
        assert min_out_cluster_distance >= distance_threshold


@pytest.mark.parametrize("linkage", ["ward", "complete", "average"])
@pytest.mark.parametrize(
    ("threshold", "y_true"), [(0.5, [1, 0]), (1.0, [1, 0]), (1.5, [0, 0])]
)
def test_agglomerative_clustering_with_distance_threshold_edge_case(
    linkage, threshold, y_true
):
    # test boundary case of distance_threshold matching the distance
    X = [[0], [1]]
    clusterer = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, linkage=linkage
    )
    y_pred = clusterer.fit_predict(X)
    assert adjusted_rand_score(y_true, y_pred) == 1


def test_dist_threshold_invalid_parameters():
    X = [[0], [1]]
    with pytest.raises(ValueError, match="Exactly one of "):
        AgglomerativeClustering(n_clusters=None, distance_threshold=None).fit(X)

    with pytest.raises(ValueError, match="Exactly one of "):
        AgglomerativeClustering(n_clusters=2, distance_threshold=1).fit(X)

    X = [[0], [1]]
    with pytest.raises(ValueError, match="compute_full_tree must be True if"):
        AgglomerativeClustering(
            n_clusters=None, distance_threshold=1, compute_full_tree=False
        ).fit(X)


def test_invalid_shape_precomputed_dist_matrix():
    # Check that an error is raised when affinity='precomputed'
    # and a non square matrix is passed (PR #16257).
    rng = np.random.RandomState(0)
    X = rng.rand(5, 3)
    with pytest.raises(
        ValueError,
        match=r"Distance matrix should be square, got matrix of shape \(5, 3\)",
    ):
        AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(X)


def test_precomputed_connectivity_affinity_with_2_connected_components():
    """Check that connecting components works when connectivity and
    affinity are both precomputed and the number of connected components is
    greater than 1. Non-regression test for #16151.
    """

    connectivity_matrix = np.array(
        [
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    # ensure that connectivity_matrix has two connected components
    assert connected_components(connectivity_matrix)[0] == 2

    rng = np.random.RandomState(0)
    X = rng.randn(5, 10)

    X_dist = pairwise_distances(X)
    clusterer_precomputed = AgglomerativeClustering(
        affinity="precomputed", connectivity=connectivity_matrix, linkage="complete"
    )
    msg = "Completing it to avoid stopping the tree early"
    with pytest.warns(UserWarning, match=msg):
        clusterer_precomputed.fit(X_dist)

    clusterer = AgglomerativeClustering(
        connectivity=connectivity_matrix, linkage="complete"
    )
    with pytest.warns(UserWarning, match=msg):
        clusterer.fit(X)

    assert_array_equal(clusterer.labels_, clusterer_precomputed.labels_)
    assert_array_equal(clusterer.children_, clusterer_precomputed.children_)
