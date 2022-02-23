import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from scipy.sparse import csgraph


def test_weak_connections():
    Xde = np.array([[0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0]])

    Xsp = csgraph.csgraph_from_dense(Xde, null_value=0)

    for X in Xsp, Xde:
        n_components, labels =\
            csgraph.connected_components(X, directed=True,
                                         connection='weak')

        assert_equal(n_components, 2)
        assert_array_almost_equal(labels, [0, 0, 1])


def test_strong_connections():
    X1de = np.array([[0, 1, 0],
                     [0, 0, 0],
                     [0, 0, 0]])
    X2de = X1de + X1de.T

    X1sp = csgraph.csgraph_from_dense(X1de, null_value=0)
    X2sp = csgraph.csgraph_from_dense(X2de, null_value=0)

    for X in X1sp, X1de:
        n_components, labels =\
            csgraph.connected_components(X, directed=True,
                                         connection='strong')

        assert_equal(n_components, 3)
        labels.sort()
        assert_array_almost_equal(labels, [0, 1, 2])

    for X in X2sp, X2de:
        n_components, labels =\
            csgraph.connected_components(X, directed=True,
                                         connection='strong')

        assert_equal(n_components, 2)
        labels.sort()
        assert_array_almost_equal(labels, [0, 0, 1])


def test_strong_connections2():
    X = np.array([[0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0]])
    n_components, labels =\
        csgraph.connected_components(X, directed=True,
                                     connection='strong')
    assert_equal(n_components, 5)
    labels.sort()
    assert_array_almost_equal(labels, [0, 1, 2, 2, 3, 4])


def test_weak_connections2():
    X = np.array([[0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0]])
    n_components, labels =\
        csgraph.connected_components(X, directed=True,
                                     connection='weak')
    assert_equal(n_components, 2)
    labels.sort()
    assert_array_almost_equal(labels, [0, 0, 1, 1, 1, 1])


def test_ticket1876():
    # Regression test: this failed in the original implementation
    # There should be two strongly-connected components; previously gave one
    g = np.array([[0, 1, 1, 0],
                  [1, 0, 0, 1],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]])
    n_components, labels = csgraph.connected_components(g, connection='strong')

    assert_equal(n_components, 2)
    assert_equal(labels[0], labels[1])
    assert_equal(labels[2], labels[3])


def test_fully_connected_graph():
    # Fully connected dense matrices raised an exception.
    # https://github.com/scipy/scipy/issues/3818
    g = np.ones((4, 4))
    n_components, labels = csgraph.connected_components(g)
    assert_equal(n_components, 1)
