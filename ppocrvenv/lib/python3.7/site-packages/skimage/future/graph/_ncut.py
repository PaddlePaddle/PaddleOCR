import networkx as nx
import numpy as np
from scipy import sparse
from . import _ncut_cy


def DW_matrices(graph):
    """Returns the diagonal and weight matrices of a graph.

    Parameters
    ----------
    graph : RAG
        A Region Adjacency Graph.

    Returns
    -------
    D : csc_matrix
        The diagonal matrix of the graph. ``D[i, i]`` is the sum of weights of
        all edges incident on `i`. All other entries are `0`.
    W : csc_matrix
        The weight matrix of the graph. ``W[i, j]`` is the weight of the edge
        joining `i` to `j`.
    """
    # sparse.eighsh is most efficient with CSC-formatted input
    W = nx.to_scipy_sparse_matrix(graph, format='csc')
    entries = W.sum(axis=0)
    D = sparse.dia_matrix((entries, 0), shape=W.shape).tocsc()

    return D, W


def ncut_cost(cut, D, W):
    """Returns the N-cut cost of a bi-partition of a graph.

    Parameters
    ----------
    cut : ndarray
        The mask for the nodes in the graph. Nodes corresponding to a `True`
        value are in one set.
    D : csc_matrix
        The diagonal matrix of the graph.
    W : csc_matrix
        The weight matrix of the graph.

    Returns
    -------
    cost : float
        The cost of performing the N-cut.

    References
    ----------
    .. [1] Normalized Cuts and Image Segmentation, Jianbo Shi and
           Jitendra Malik, IEEE Transactions on Pattern Analysis and Machine
           Intelligence, Page 889, Equation 2.
    """
    cut = np.array(cut)
    cut_cost = _ncut_cy.cut_cost(cut, W)

    # D has elements only along the diagonal, one per node, so we can directly
    # index the data attribute with cut.
    assoc_a = D.data[cut].sum()
    assoc_b = D.data[~cut].sum()

    return (cut_cost / assoc_a) + (cut_cost / assoc_b)
