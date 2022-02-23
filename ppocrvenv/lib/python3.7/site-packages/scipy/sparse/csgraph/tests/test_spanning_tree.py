"""Test the minimum spanning tree function"""
import numpy as np
from numpy.testing import assert_
import numpy.testing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


def test_minimum_spanning_tree():

    # Create a graph with two connected components.
    graph = [[0,1,0,0,0],
             [1,0,0,0,0],
             [0,0,0,8,5],
             [0,0,8,0,1],
             [0,0,5,1,0]]
    graph = np.asarray(graph)

    # Create the expected spanning tree.
    expected = [[0,1,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,5],
                [0,0,0,0,1],
                [0,0,0,0,0]]
    expected = np.asarray(expected)

    # Ensure minimum spanning tree code gives this expected output.
    csgraph = csr_matrix(graph)
    mintree = minimum_spanning_tree(csgraph)
    npt.assert_array_equal(mintree.todense(), expected,
        'Incorrect spanning tree found.')

    # Ensure that the original graph was not modified.
    npt.assert_array_equal(csgraph.todense(), graph,
        'Original graph was modified.')

    # Now let the algorithm modify the csgraph in place.
    mintree = minimum_spanning_tree(csgraph, overwrite=True)
    npt.assert_array_equal(mintree.todense(), expected,
        'Graph was not properly modified to contain MST.')

    np.random.seed(1234)
    for N in (5, 10, 15, 20):

        # Create a random graph.
        graph = 3 + np.random.random((N, N))
        csgraph = csr_matrix(graph)

        # The spanning tree has at most N - 1 edges.
        mintree = minimum_spanning_tree(csgraph)
        assert_(mintree.nnz < N)

        # Set the sub diagonal to 1 to create a known spanning tree.
        idx = np.arange(N-1)
        graph[idx,idx+1] = 1
        csgraph = csr_matrix(graph)
        mintree = minimum_spanning_tree(csgraph)

        # We expect to see this pattern in the spanning tree and otherwise
        # have this zero.
        expected = np.zeros((N, N))
        expected[idx, idx+1] = 1

        npt.assert_array_equal(mintree.todense(), expected,
            'Incorrect spanning tree found.')
