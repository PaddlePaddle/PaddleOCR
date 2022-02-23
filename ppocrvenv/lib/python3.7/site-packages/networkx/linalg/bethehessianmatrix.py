"""Bethe Hessian or deformed Laplacian matrix of graphs."""
import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["bethe_hessian_matrix"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def bethe_hessian_matrix(G, r=None, nodelist=None):
    r"""Returns the Bethe Hessian matrix of G.

    The Bethe Hessian is a family of matrices parametrized by r, defined as
    H(r) = (r^2 - 1) I - r A + D where A is the adjacency matrix, D is the
    diagonal matrix of node degrees, and I is the identify matrix. It is equal
    to the graph laplacian when the regularizer r = 1.

    The default choice of regularizer should be the ratio [2]

    .. math::
      r_m = \left(\sum k_i \right)^{-1}\left(\sum k_i^2 \right) - 1

    Parameters
    ----------
    G : Graph
       A NetworkX graph

    r : float
       Regularizer parameter

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().


    Returns
    -------
    H : Numpy matrix
      The Bethe Hessian matrix of G, with paramter r.

    Examples
    --------
    >>> k = [3, 2, 2, 1, 0]
    >>> G = nx.havel_hakimi_graph(k)
    >>> H = nx.modularity_matrix(G)


    See Also
    --------
    bethe_hessian_spectrum
    to_numpy_array
    adjacency_matrix
    laplacian_matrix

    References
    ----------
    .. [1] A. Saade, F. Krzakala and L. Zdeborová
       "Spectral clustering of graphs with the bethe hessian",
       Advances in Neural Information Processing Systems. 2014.
    .. [2] C. M. Lee, E. Levina
       "Estimating the number of communities in networks by spectral methods"
       arXiv:1507.00827, 2015.
    """
    import scipy as sp
    import scipy.sparse  # call as sp.sparse

    if nodelist is None:
        nodelist = list(G)
    if r is None:
        r = (
            sum([d ** 2 for v, d in nx.degree(G)]) / sum([d for v, d in nx.degree(G)])
            - 1
        )
    A = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, format="csr")
    n, m = A.shape
    diags = A.sum(axis=1)
    D = sp.sparse.spdiags(diags.flatten(), [0], m, n, format="csr")
    I = sp.sparse.eye(m, n, format="csr")
    return (r ** 2 - 1) * I - r * A + D
