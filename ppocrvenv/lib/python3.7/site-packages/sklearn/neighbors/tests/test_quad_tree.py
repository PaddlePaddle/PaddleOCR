import pickle
import numpy as np

import pytest

from sklearn.neighbors._quad_tree import _QuadTree
from sklearn.utils import check_random_state


def test_quadtree_boundary_computation():
    # Introduce a point into a quad tree with boundaries not easy to compute.
    Xs = []

    # check a random case
    Xs.append(np.array([[-1, 1], [-4, -1]], dtype=np.float32))
    # check the case where only 0 are inserted
    Xs.append(np.array([[0, 0], [0, 0]], dtype=np.float32))
    # check the case where only negative are inserted
    Xs.append(np.array([[-1, -2], [-4, 0]], dtype=np.float32))
    # check the case where only small numbers are inserted
    Xs.append(np.array([[-1e-6, 1e-6], [-4e-6, -1e-6]], dtype=np.float32))

    for X in Xs:
        tree = _QuadTree(n_dimensions=2, verbose=0)
        tree.build_tree(X)
        tree._check_coherence()


def test_quadtree_similar_point():
    # Introduce a point into a quad tree where a similar point already exists.
    # Test will hang if it doesn't complete.
    Xs = []

    # check the case where points are actually different
    Xs.append(np.array([[1, 2], [3, 4]], dtype=np.float32))
    # check the case where points are the same on X axis
    Xs.append(np.array([[1.0, 2.0], [1.0, 3.0]], dtype=np.float32))
    # check the case where points are arbitrarily close on X axis
    Xs.append(np.array([[1.00001, 2.0], [1.00002, 3.0]], dtype=np.float32))
    # check the case where points are the same on Y axis
    Xs.append(np.array([[1.0, 2.0], [3.0, 2.0]], dtype=np.float32))
    # check the case where points are arbitrarily close on Y axis
    Xs.append(np.array([[1.0, 2.00001], [3.0, 2.00002]], dtype=np.float32))
    # check the case where points are arbitrarily close on both axes
    Xs.append(np.array([[1.00001, 2.00001], [1.00002, 2.00002]], dtype=np.float32))

    # check the case where points are arbitrarily close on both axes
    # close to machine epsilon - x axis
    Xs.append(np.array([[1, 0.0003817754041], [2, 0.0003817753750]], dtype=np.float32))

    # check the case where points are arbitrarily close on both axes
    # close to machine epsilon - y axis
    Xs.append(
        np.array([[0.0003817754041, 1.0], [0.0003817753750, 2.0]], dtype=np.float32)
    )

    for X in Xs:
        tree = _QuadTree(n_dimensions=2, verbose=0)
        tree.build_tree(X)
        tree._check_coherence()


@pytest.mark.parametrize("n_dimensions", (2, 3))
@pytest.mark.parametrize("protocol", (0, 1, 2))
def test_quad_tree_pickle(n_dimensions, protocol):
    rng = check_random_state(0)

    X = rng.random_sample((10, n_dimensions))

    tree = _QuadTree(n_dimensions=n_dimensions, verbose=0)
    tree.build_tree(X)

    s = pickle.dumps(tree, protocol=protocol)
    bt2 = pickle.loads(s)

    for x in X:
        cell_x_tree = tree.get_cell(x)
        cell_x_bt2 = bt2.get_cell(x)
        assert cell_x_tree == cell_x_bt2


@pytest.mark.parametrize("n_dimensions", (2, 3))
def test_qt_insert_duplicate(n_dimensions):
    rng = check_random_state(0)

    X = rng.random_sample((10, n_dimensions))
    Xd = np.r_[X, X[:5]]
    tree = _QuadTree(n_dimensions=n_dimensions, verbose=0)
    tree.build_tree(Xd)

    cumulative_size = tree.cumulative_size
    leafs = tree.leafs

    # Assert that the first 5 are indeed duplicated and that the next
    # ones are single point leaf
    for i, x in enumerate(X):
        cell_id = tree.get_cell(x)
        assert leafs[cell_id]
        assert cumulative_size[cell_id] == 1 + (i < 5)


def test_summarize():
    # Simple check for quad tree's summarize

    angle = 0.9
    X = np.array(
        [[-10.0, -10.0], [9.0, 10.0], [10.0, 9.0], [10.0, 10.0]], dtype=np.float32
    )
    query_pt = X[0, :]
    n_dimensions = X.shape[1]
    offset = n_dimensions + 2

    qt = _QuadTree(n_dimensions, verbose=0)
    qt.build_tree(X)

    idx, summary = qt._py_summarize(query_pt, X, angle)

    node_dist = summary[n_dimensions]
    node_size = summary[n_dimensions + 1]

    # Summary should contain only 1 node with size 3 and distance to
    # X[1:] barycenter
    barycenter = X[1:].mean(axis=0)
    ds2c = ((X[0] - barycenter) ** 2).sum()

    assert idx == offset
    assert node_size == 3, "summary size = {}".format(node_size)
    assert np.isclose(node_dist, ds2c)

    # Summary should contain all 3 node with size 1 and distance to
    # each point in X[1:] for ``angle=0``
    idx, summary = qt._py_summarize(query_pt, X, 0.0)
    barycenter = X[1:].mean(axis=0)
    ds2c = ((X[0] - barycenter) ** 2).sum()

    assert idx == 3 * (offset)
    for i in range(3):
        node_dist = summary[i * offset + n_dimensions]
        node_size = summary[i * offset + n_dimensions + 1]

        ds2c = ((X[0] - X[i + 1]) ** 2).sum()

        assert node_size == 1, "summary size = {}".format(node_size)
        assert np.isclose(node_dist, ds2c)
