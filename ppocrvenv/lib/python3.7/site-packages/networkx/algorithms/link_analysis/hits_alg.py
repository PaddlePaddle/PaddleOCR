"""Hubs and authorities analysis of graph structure.
"""
import networkx as nx

__all__ = ["hits", "hits_numpy", "hits_scipy", "authority_matrix", "hub_matrix"]


def hits(G, max_iter=100, tol=1.0e-8, nstart=None, normalized=True):
    """Returns HITS hubs and authorities values for nodes.

    The HITS algorithm computes two numbers for a node.
    Authorities estimates the node value based on the incoming links.
    Hubs estimates the node value based on outgoing links.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    max_iter : integer, optional
      Maximum number of iterations in power method.

    tol : float, optional
      Error tolerance used to check convergence in power method iteration.

    nstart : dictionary, optional
      Starting value of each node for power method iteration.

    normalized : bool (default=True)
       Normalize results by the sum of all of the values.

    Returns
    -------
    (hubs,authorities) : two-tuple of dictionaries
       Two dictionaries keyed by node containing the hub and authority
       values.

    Raises
    ------
    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> h, a = nx.hits(G)

    Notes
    -----
    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence.  The iteration will stop
    after max_iter iterations or an error tolerance of
    number_of_nodes(G)*tol has been reached.

    The HITS algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs.

    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Jon Kleinberg,
       Authoritative sources in a hyperlinked environment
       Journal of the ACM 46 (5): 604-32, 1999.
       doi:10.1145/324133.324140.
       http://www.cs.cornell.edu/home/kleinber/auth.pdf.
    """
    import numpy as np
    import scipy as sp
    import scipy.sparse.linalg  # call as sp.sparse.linalg

    if len(G) == 0:
        return {}, {}
    M = nx.adjacency_matrix(G, nodelist=list(G), dtype=float)

    if nstart is None:
        u, s, vt = sp.sparse.linalg.svds(M, k=1, maxiter=max_iter, tol=tol)
    else:
        nstart = np.array(list(nstart.values()))
        u, s, vt = sp.sparse.linalg.svds(M, k=1, v0=nstart, maxiter=max_iter, tol=tol)

    a = vt.flatten().real
    h = np.asarray(M * a).flatten()
    if normalized:
        h = h / h.sum()
        a = a / a.sum()
    hubs = dict(zip(G, map(float, h)))
    authorities = dict(zip(G, map(float, a)))
    return hubs, authorities


def _hits_python(G, max_iter=100, tol=1.0e-8, nstart=None, normalized=True):
    if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
        raise Exception("hits() not defined for graphs with multiedges.")
    if len(G) == 0:
        return {}, {}
    # choose fixed starting vector if not given
    if nstart is None:
        h = dict.fromkeys(G, 1.0 / G.number_of_nodes())
    else:
        h = nstart
        # normalize starting vector
        s = 1.0 / sum(h.values())
        for k in h:
            h[k] *= s
    for _ in range(max_iter):  # power iteration: make up to max_iter iterations
        hlast = h
        h = dict.fromkeys(hlast.keys(), 0)
        a = dict.fromkeys(hlast.keys(), 0)
        # this "matrix multiply" looks odd because it is
        # doing a left multiply a^T=hlast^T*G
        for n in h:
            for nbr in G[n]:
                a[nbr] += hlast[n] * G[n][nbr].get("weight", 1)
        # now multiply h=Ga
        for n in h:
            for nbr in G[n]:
                h[n] += a[nbr] * G[n][nbr].get("weight", 1)
        # normalize vector
        s = 1.0 / max(h.values())
        for n in h:
            h[n] *= s
        # normalize vector
        s = 1.0 / max(a.values())
        for n in a:
            a[n] *= s
        # check convergence, l1 norm
        err = sum([abs(h[n] - hlast[n]) for n in h])
        if err < tol:
            break
    else:
        raise nx.PowerIterationFailedConvergence(max_iter)
    if normalized:
        s = 1.0 / sum(a.values())
        for n in a:
            a[n] *= s
        s = 1.0 / sum(h.values())
        for n in h:
            h[n] *= s
    return h, a


def authority_matrix(G, nodelist=None):
    """Returns the HITS authority matrix.

    .. deprecated:: 2.6
    """
    import warnings

    msg = (
        "\nauthority_matrix is deprecated as of version 2.6 and will be removed "
        "in version 3.0.\n"
        "The authority matrix can be computed by::\n"
        "    >>> M = nx.to_numpy_array(G, nodelist=nodelist)\n"
        "    >>> M.T @ M"
    )
    warnings.warn(msg, DeprecationWarning)
    M = nx.to_numpy_array(G, nodelist=nodelist)
    return M.T @ M


def hub_matrix(G, nodelist=None):
    """Returns the HITS hub matrix.

    .. deprecated:: 2.6
    """
    import warnings

    msg = (
        "\nhub_matrix is deprecated as of version 2.6 and will be removed "
        "in version 3.0.\n"
        "The hub matrix can be computed by::\n"
        "    >>> M = nx.to_numpy_array(G, nodelist=nodelist)\n"
        "    >>> M @ M.T"
    )
    warnings.warn(msg, DeprecationWarning)
    M = nx.to_numpy_array(G, nodelist=nodelist)
    return M @ M.T


def hits_numpy(G, normalized=True):
    """Returns HITS hubs and authorities values for nodes.

    .. deprecated:: 2.6

       hits_numpy is deprecated and will be removed in networkx 3.0.

    The HITS algorithm computes two numbers for a node.
    Authorities estimates the node value based on the incoming links.
    Hubs estimates the node value based on outgoing links.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    normalized : bool (default=True)
       Normalize results by the sum of all of the values.

    Returns
    -------
    (hubs,authorities) : two-tuple of dictionaries
       Two dictionaries keyed by node containing the hub and authority
       values.

    Examples
    --------
    >>> G = nx.path_graph(4)

    The `hubs` and `authorities` are given by the eigenvectors corresponding to the
    maximum eigenvalues of the hubs_matrix and the authority_matrix, respectively.

    The ``hubs`` and ``authority`` matrices are computed from the adjancency
    matrix:

    >>> adj_ary = nx.to_numpy_array(G)
    >>> hubs_matrix = adj_ary @ adj_ary.T
    >>> authority_matrix = adj_ary.T @ adj_ary

    `hits_numpy` maps the eigenvector corresponding to the maximum eigenvalue
    of the respective matrices to the nodes in `G`:

    >>> hubs, authority = nx.hits_numpy(G)

    Notes
    -----
    The eigenvector calculation uses NumPy's interface to LAPACK.

    The HITS algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs.

    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Jon Kleinberg,
       Authoritative sources in a hyperlinked environment
       Journal of the ACM 46 (5): 604-32, 1999.
       doi:10.1145/324133.324140.
       http://www.cs.cornell.edu/home/kleinber/auth.pdf.
    """
    import numpy as np
    import warnings

    warnings.warn(
        (
            "networkx.hits_numpy is deprecated and will be removed"
            "in NetworkX 3.0, use networkx.hits instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )

    if len(G) == 0:
        return {}, {}
    adj_ary = nx.to_numpy_array(G)
    # Hub matrix
    H = adj_ary @ adj_ary.T
    e, ev = np.linalg.eig(H)
    h = ev[:, np.argmax(e)]  # eigenvector corresponding to the maximum eigenvalue
    # Authority matrix
    A = adj_ary.T @ adj_ary
    e, ev = np.linalg.eig(A)
    a = ev[:, np.argmax(e)]  # eigenvector corresponding to the maximum eigenvalue
    if normalized:
        h = h / h.sum()
        a = a / a.sum()
    else:
        h = h / h.max()
        a = a / a.max()
    hubs = dict(zip(G, map(float, h)))
    authorities = dict(zip(G, map(float, a)))
    return hubs, authorities


def hits_scipy(G, max_iter=100, tol=1.0e-6, nstart=None, normalized=True):
    """Returns HITS hubs and authorities values for nodes.

    .. deprecated:: 2.6

       hits_scipy is deprecated and will be removed in networkx 3.0

    The HITS algorithm computes two numbers for a node.
    Authorities estimates the node value based on the incoming links.
    Hubs estimates the node value based on outgoing links.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    max_iter : integer, optional
      Maximum number of iterations in power method.

    tol : float, optional
      Error tolerance used to check convergence in power method iteration.

    nstart : dictionary, optional
      Starting value of each node for power method iteration.

    normalized : bool (default=True)
       Normalize results by the sum of all of the values.

    Returns
    -------
    (hubs,authorities) : two-tuple of dictionaries
       Two dictionaries keyed by node containing the hub and authority
       values.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> h, a = nx.hits(G)

    Notes
    -----
    This implementation uses SciPy sparse matrices.

    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence.  The iteration will stop
    after max_iter iterations or an error tolerance of
    number_of_nodes(G)*tol has been reached.

    The HITS algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs.

    Raises
    ------
    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.

    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Jon Kleinberg,
       Authoritative sources in a hyperlinked environment
       Journal of the ACM 46 (5): 604-632, 1999.
       doi:10.1145/324133.324140.
       http://www.cs.cornell.edu/home/kleinber/auth.pdf.
    """
    import numpy as np
    import warnings

    warnings.warn(
        (
            "networkx.hits_scipy is deprecated and will be removed"
            "in NetworkX 3.0, use networkx.hits instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )

    if len(G) == 0:
        return {}, {}
    M = nx.to_scipy_sparse_matrix(G, nodelist=list(G))
    (n, m) = M.shape  # should be square
    A = M.T * M  # authority matrix
    x = np.ones((n, 1)) / n  # initial guess
    # choose fixed starting vector if not given
    if nstart is None:
        x = np.ones((n, 1)) / n  # initial guess
    else:
        x = np.array([nstart.get(n, 0) for n in list(G)], dtype=float)
        x = x / x.sum()

    # power iteration on authority matrix
    i = 0
    while True:
        xlast = x
        x = A * x
        x = x / x.max()
        # check convergence, l1 norm
        err = np.absolute(x - xlast).sum()
        if err < tol:
            break
        if i > max_iter:
            raise nx.PowerIterationFailedConvergence(max_iter)
        i += 1

    a = np.asarray(x).flatten()
    # h=M*a
    h = np.asarray(M * a).flatten()
    if normalized:
        h = h / h.sum()
        a = a / a.sum()
    hubs = dict(zip(G, map(float, h)))
    authorities = dict(zip(G, map(float, a)))
    return hubs, authorities
