import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse.csgraph import (breadth_first_tree, depth_first_tree,
    csgraph_to_dense, csgraph_from_dense)


def test_graph_breadth_first():
    csgraph = np.array([[0, 1, 2, 0, 0],
                        [1, 0, 0, 0, 3],
                        [2, 0, 0, 7, 0],
                        [0, 0, 7, 0, 1],
                        [0, 3, 0, 1, 0]])
    csgraph = csgraph_from_dense(csgraph, null_value=0)

    bfirst = np.array([[0, 1, 2, 0, 0],
                       [0, 0, 0, 0, 3],
                       [0, 0, 0, 7, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]])

    for directed in [True, False]:
        bfirst_test = breadth_first_tree(csgraph, 0, directed)
        assert_array_almost_equal(csgraph_to_dense(bfirst_test),
                                  bfirst)


def test_graph_depth_first():
    csgraph = np.array([[0, 1, 2, 0, 0],
                        [1, 0, 0, 0, 3],
                        [2, 0, 0, 7, 0],
                        [0, 0, 7, 0, 1],
                        [0, 3, 0, 1, 0]])
    csgraph = csgraph_from_dense(csgraph, null_value=0)

    dfirst = np.array([[0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 3],
                       [0, 0, 0, 0, 0],
                       [0, 0, 7, 0, 0],
                       [0, 0, 0, 1, 0]])

    for directed in [True, False]:
        dfirst_test = depth_first_tree(csgraph, 0, directed)
        assert_array_almost_equal(csgraph_to_dense(dfirst_test),
                                  dfirst)


def test_graph_breadth_first_trivial_graph():
    csgraph = np.array([[0]])
    csgraph = csgraph_from_dense(csgraph, null_value=0)

    bfirst = np.array([[0]])

    for directed in [True, False]:
        bfirst_test = breadth_first_tree(csgraph, 0, directed)
        assert_array_almost_equal(csgraph_to_dense(bfirst_test),
                                  bfirst)


def test_graph_depth_first_trivial_graph():
    csgraph = np.array([[0]])
    csgraph = csgraph_from_dense(csgraph, null_value=0)

    bfirst = np.array([[0]])

    for directed in [True, False]:
        bfirst_test = depth_first_tree(csgraph, 0, directed)
        assert_array_almost_equal(csgraph_to_dense(bfirst_test),
                                  bfirst)
