"""Functions to convert NetworkX graphs to and from common data containers
like numpy arrays, scipy sparse matrices, and pandas DataFrames.

The preferred way of converting data to a NetworkX graph is through the
graph constructor.  The constructor calls the to_networkx_graph() function
which attempts to guess the input type and convert it automatically.

Examples
--------
Create a 10 node random graph from a numpy array

>>> import numpy as np
>>> a = np.random.randint(0, 2, size=(10, 10))
>>> D = nx.DiGraph(a)

or equivalently

>>> D = nx.to_networkx_graph(a, create_using=nx.DiGraph)

See Also
--------
nx_agraph, nx_pydot
"""

import itertools
import warnings
import networkx as nx
from networkx.utils import not_implemented_for

__all__ = [
    "from_numpy_matrix",
    "to_numpy_matrix",
    "from_pandas_adjacency",
    "to_pandas_adjacency",
    "from_pandas_edgelist",
    "to_pandas_edgelist",
    "to_numpy_recarray",
    "from_scipy_sparse_matrix",
    "to_scipy_sparse_matrix",
    "from_numpy_array",
    "to_numpy_array",
]


def to_pandas_adjacency(
    G,
    nodelist=None,
    dtype=None,
    order=None,
    multigraph_weight=sum,
    weight="weight",
    nonedge=0.0,
):
    """Returns the graph adjacency matrix as a Pandas DataFrame.

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the Pandas DataFrame.

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in `nodelist`.
       If `nodelist` is None, then the ordering is produced by G.nodes().

    multigraph_weight : {sum, min, max}, optional
        An operator that determines how weights in multigraphs are handled.
        The default is to sum the weights of the multiple edges.

    weight : string or None, optional
        The edge attribute that holds the numerical value used for
        the edge weight.  If an edge does not have that attribute, then the
        value 1 is used instead.

    nonedge : float, optional
        The matrix values corresponding to nonedges are typically set to zero.
        However, this could be undesirable if there are matrix values
        corresponding to actual edges that also have the value zero. If so,
        one might prefer nonedges to have some other value, such as nan.

    Returns
    -------
    df : Pandas DataFrame
       Graph adjacency matrix

    Notes
    -----
    For directed graphs, entry i,j corresponds to an edge from i to j.

    The DataFrame entries are assigned to the weight edge attribute. When
    an edge does not have a weight attribute, the value of the entry is set to
    the number 1.  For multiple (parallel) edges, the values of the entries
    are determined by the 'multigraph_weight' parameter.  The default is to
    sum the weight attributes for each of the parallel edges.

    When `nodelist` does not contain every node in `G`, the matrix is built
    from the subgraph of `G` that is induced by the nodes in `nodelist`.

    The convention used for self-loop edges in graphs is to assign the
    diagonal matrix entry value to the weight attribute of the edge
    (or the number 1 if the edge has no weight attribute).  If the
    alternate convention of doubling the edge weight is desired the
    resulting Pandas DataFrame can be modified as follows:

    >>> import pandas as pd
    >>> pd.options.display.max_columns = 20
    >>> import numpy as np
    >>> G = nx.Graph([(1, 1)])
    >>> df = nx.to_pandas_adjacency(G, dtype=int)
    >>> df
       1
    1  1
    >>> df.values[np.diag_indices_from(df)] *= 2
    >>> df
       1
    1  2

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(0, 1, weight=2)
    0
    >>> G.add_edge(1, 0)
    0
    >>> G.add_edge(2, 2, weight=3)
    0
    >>> G.add_edge(2, 2)
    1
    >>> nx.to_pandas_adjacency(G, nodelist=[0, 1, 2], dtype=int)
       0  1  2
    0  0  2  0
    1  1  0  0
    2  0  0  4

    """
    import pandas as pd

    M = to_numpy_array(
        G,
        nodelist=nodelist,
        dtype=dtype,
        order=order,
        multigraph_weight=multigraph_weight,
        weight=weight,
        nonedge=nonedge,
    )
    if nodelist is None:
        nodelist = list(G)
    return pd.DataFrame(data=M, index=nodelist, columns=nodelist)


def from_pandas_adjacency(df, create_using=None):
    r"""Returns a graph from Pandas DataFrame.

    The Pandas DataFrame is interpreted as an adjacency matrix for the graph.

    Parameters
    ----------
    df : Pandas DataFrame
      An adjacency matrix representation of a graph

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Notes
    -----
    For directed graphs, explicitly mention create_using=nx.DiGraph,
    and entry i,j of df corresponds to an edge from i to j.

    If `df` has a single data type for each entry it will be converted to an
    appropriate Python data type.

    If `df` has a user-specified compound data type the names
    of the data fields will be used as attribute keys in the resulting
    NetworkX graph.

    See Also
    --------
    to_pandas_adjacency

    Examples
    --------
    Simple integer weights on edges:

    >>> import pandas as pd
    >>> pd.options.display.max_columns = 20
    >>> df = pd.DataFrame([[1, 1], [2, 1]])
    >>> df
       0  1
    0  1  1
    1  2  1
    >>> G = nx.from_pandas_adjacency(df)
    >>> G.name = "Graph from pandas adjacency matrix"
    >>> print(nx.info(G))
    Graph named 'Graph from pandas adjacency matrix' with 2 nodes and 3 edges
    """

    try:
        df = df[df.index]
    except Exception as e:
        missing = list(set(df.index).difference(set(df.columns)))
        msg = f"{missing} not in columns"
        raise nx.NetworkXError("Columns must match Indices.", msg) from e

    A = df.values
    G = from_numpy_array(A, create_using=create_using)

    nx.relabel.relabel_nodes(G, dict(enumerate(df.columns)), copy=False)
    return G


def to_pandas_edgelist(
    G,
    source="source",
    target="target",
    nodelist=None,
    dtype=None,
    order=None,
    edge_key=None,
):
    """Returns the graph edge list as a Pandas DataFrame.

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the Pandas DataFrame.

    source : str or int, optional
        A valid column name (string or integer) for the source nodes (for the
        directed case).

    target : str or int, optional
        A valid column name (string or integer) for the target nodes (for the
        directed case).

    nodelist : list, optional
       Use only nodes specified in nodelist

    dtype : dtype, default None
        Use to create the DataFrame. Data type to force.
        Only a single dtype is allowed. If None, infer.

    order : None
        An unused parameter mistakenly included in the function.

        .. deprecated:: 2.6
            This is deprecated and will be removed in NetworkX v3.0.

    edge_key : str or int or None, optional (default=None)
        A valid column name (string or integer) for the edge keys (for the
        multigraph case). If None, edge keys are not stored in the DataFrame.

    Returns
    -------
    df : Pandas DataFrame
       Graph edge list

    Examples
    --------
    >>> G = nx.Graph(
    ...     [
    ...         ("A", "B", {"cost": 1, "weight": 7}),
    ...         ("C", "E", {"cost": 9, "weight": 10}),
    ...     ]
    ... )
    >>> df = nx.to_pandas_edgelist(G, nodelist=["A", "C"])
    >>> df[["source", "target", "cost", "weight"]]
      source target  cost  weight
    0      A      B     1       7
    1      C      E     9      10

    >>> G = nx.MultiGraph([('A', 'B', {'cost': 1}), ('A', 'B', {'cost': 9})])
    >>> df = nx.to_pandas_edgelist(G, nodelist=['A', 'C'], edge_key='ekey')
    >>> df[['source', 'target', 'cost', 'ekey']]
      source target  cost  ekey
    0      A      B     1     0
    1      A      B     9     1

    """
    import pandas as pd

    if nodelist is None:
        edgelist = G.edges(data=True)
    else:
        edgelist = G.edges(nodelist, data=True)
    source_nodes = [s for s, _, _ in edgelist]
    target_nodes = [t for _, t, _ in edgelist]

    all_attrs = set().union(*(d.keys() for _, _, d in edgelist))
    if source in all_attrs:
        raise nx.NetworkXError(f"Source name {source!r} is an edge attr name")
    if target in all_attrs:
        raise nx.NetworkXError(f"Target name {target!r} is an edge attr name")

    nan = float("nan")
    edge_attr = {k: [d.get(k, nan) for _, _, d in edgelist] for k in all_attrs}

    if G.is_multigraph() and edge_key is not None:
        if edge_key in all_attrs:
            raise nx.NetworkXError(f"Edge key name {edge_key!r} is an edge attr name")
        edge_keys = [k for _, _, k in G.edges(keys=True)]
        edgelistdict = {source: source_nodes, target: target_nodes, edge_key: edge_keys}
    else:
        edgelistdict = {source: source_nodes, target: target_nodes}

    edgelistdict.update(edge_attr)
    return pd.DataFrame(edgelistdict, dtype=dtype)


def from_pandas_edgelist(
    df,
    source="source",
    target="target",
    edge_attr=None,
    create_using=None,
    edge_key=None,
):
    """Returns a graph from Pandas DataFrame containing an edge list.

    The Pandas DataFrame should contain at least two columns of node names and
    zero or more columns of edge attributes. Each row will be processed as one
    edge instance.

    Note: This function iterates over DataFrame.values, which is not
    guaranteed to retain the data type across columns in the row. This is only
    a problem if your row is entirely numeric and a mix of ints and floats. In
    that case, all values will be returned as floats. See the
    DataFrame.iterrows documentation for an example.

    Parameters
    ----------
    df : Pandas DataFrame
        An edge list representation of a graph

    source : str or int
        A valid column name (string or integer) for the source nodes (for the
        directed case).

    target : str or int
        A valid column name (string or integer) for the target nodes (for the
        directed case).

    edge_attr : str or int, iterable, True, or None
        A valid column name (str or int) or iterable of column names that are
        used to retrieve items and add them to the graph as edge attributes.
        If `True`, all of the remaining columns will be added.
        If `None`, no edge attributes are added to the graph.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    edge_key : str or None, optional (default=None)
        A valid column name for the edge keys (for a MultiGraph). The values in
        this column are used for the edge keys when adding edges if create_using
        is a multigraph.

    See Also
    --------
    to_pandas_edgelist

    Examples
    --------
    Simple integer weights on edges:

    >>> import pandas as pd
    >>> pd.options.display.max_columns = 20
    >>> import numpy as np
    >>> rng = np.random.RandomState(seed=5)
    >>> ints = rng.randint(1, 11, size=(3, 2))
    >>> a = ["A", "B", "C"]
    >>> b = ["D", "A", "E"]
    >>> df = pd.DataFrame(ints, columns=["weight", "cost"])
    >>> df[0] = a
    >>> df["b"] = b
    >>> df[["weight", "cost", 0, "b"]]
       weight  cost  0  b
    0       4     7  A  D
    1       7     1  B  A
    2      10     9  C  E
    >>> G = nx.from_pandas_edgelist(df, 0, "b", ["weight", "cost"])
    >>> G["E"]["C"]["weight"]
    10
    >>> G["E"]["C"]["cost"]
    9
    >>> edges = pd.DataFrame(
    ...     {
    ...         "source": [0, 1, 2],
    ...         "target": [2, 2, 3],
    ...         "weight": [3, 4, 5],
    ...         "color": ["red", "blue", "blue"],
    ...     }
    ... )
    >>> G = nx.from_pandas_edgelist(edges, edge_attr=True)
    >>> G[0][2]["color"]
    'red'

    Build multigraph with custom keys:

    >>> edges = pd.DataFrame(
    ...     {
    ...         "source": [0, 1, 2, 0],
    ...         "target": [2, 2, 3, 2],
    ...         "my_edge_key": ["A", "B", "C", "D"],
    ...         "weight": [3, 4, 5, 6],
    ...         "color": ["red", "blue", "blue", "blue"],
    ...     }
    ... )
    >>> G = nx.from_pandas_edgelist(
    ...     edges,
    ...     edge_key="my_edge_key",
    ...     edge_attr=["weight", "color"],
    ...     create_using=nx.MultiGraph(),
    ... )
    >>> G[0][2]
    AtlasView({'A': {'weight': 3, 'color': 'red'}, 'D': {'weight': 6, 'color': 'blue'}})


    """
    g = nx.empty_graph(0, create_using)

    if edge_attr is None:
        g.add_edges_from(zip(df[source], df[target]))
        return g

    reserved_columns = [source, target]

    # Additional columns requested
    attr_col_headings = []
    attribute_data = []
    if edge_attr is True:
        attr_col_headings = [c for c in df.columns if c not in reserved_columns]
    elif isinstance(edge_attr, (list, tuple)):
        attr_col_headings = edge_attr
    else:
        attr_col_headings = [edge_attr]
    if len(attr_col_headings) == 0:
        raise nx.NetworkXError(
            f"Invalid edge_attr argument: No columns found with name: {attr_col_headings}"
        )

    try:
        attribute_data = zip(*[df[col] for col in attr_col_headings])
    except (KeyError, TypeError) as e:
        msg = f"Invalid edge_attr argument: {edge_attr}"
        raise nx.NetworkXError(msg) from e

    if g.is_multigraph():
        # => append the edge keys from the df to the bundled data
        if edge_key is not None:
            try:
                multigraph_edge_keys = df[edge_key]
                attribute_data = zip(attribute_data, multigraph_edge_keys)
            except (KeyError, TypeError) as e:
                msg = f"Invalid edge_key argument: {edge_key}"
                raise nx.NetworkXError(msg) from e

        for s, t, attrs in zip(df[source], df[target], attribute_data):
            if edge_key is not None:
                attrs, multigraph_edge_key = attrs
                key = g.add_edge(s, t, key=multigraph_edge_key)
            else:
                key = g.add_edge(s, t)

            g[s][t][key].update(zip(attr_col_headings, attrs))
    else:
        for s, t, attrs in zip(df[source], df[target], attribute_data):
            g.add_edge(s, t)
            g[s][t].update(zip(attr_col_headings, attrs))

    return g


def to_numpy_matrix(
    G,
    nodelist=None,
    dtype=None,
    order=None,
    multigraph_weight=sum,
    weight="weight",
    nonedge=0.0,
):
    """Returns the graph adjacency matrix as a NumPy matrix.

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the NumPy matrix.

    nodelist : list, optional
        The rows and columns are ordered according to the nodes in `nodelist`.
        If `nodelist` is None, then the ordering is produced by G.nodes().

    dtype : NumPy data type, optional
        A valid single NumPy data type used to initialize the array.
        This must be a simple type such as int or numpy.float64 and
        not a compound data type (see to_numpy_recarray)
        If None, then the NumPy default is used.

    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory. If None, then the NumPy default
        is used.

    multigraph_weight : {sum, min, max}, optional
        An operator that determines how weights in multigraphs are handled.
        The default is to sum the weights of the multiple edges.

    weight : string or None optional (default = 'weight')
        The edge attribute that holds the numerical value used for
        the edge weight. If an edge does not have that attribute, then the
        value 1 is used instead.

    nonedge : float (default = 0.0)
        The matrix values corresponding to nonedges are typically set to zero.
        However, this could be undesirable if there are matrix values
        corresponding to actual edges that also have the value zero. If so,
        one might prefer nonedges to have some other value, such as nan.

    Returns
    -------
    M : NumPy matrix
        Graph adjacency matrix

    See Also
    --------
    to_numpy_recarray

    Notes
    -----
    For directed graphs, entry i,j corresponds to an edge from i to j.

    The matrix entries are assigned to the weight edge attribute. When
    an edge does not have a weight attribute, the value of the entry is set to
    the number 1.  For multiple (parallel) edges, the values of the entries
    are determined by the `multigraph_weight` parameter.  The default is to
    sum the weight attributes for each of the parallel edges.

    When `nodelist` does not contain every node in `G`, the matrix is built
    from the subgraph of `G` that is induced by the nodes in `nodelist`.

    The convention used for self-loop edges in graphs is to assign the
    diagonal matrix entry value to the weight attribute of the edge
    (or the number 1 if the edge has no weight attribute).  If the
    alternate convention of doubling the edge weight is desired the
    resulting Numpy matrix can be modified as follows:

    >>> import numpy as np
    >>> G = nx.Graph([(1, 1)])
    >>> A = nx.to_numpy_matrix(G)
    >>> A
    matrix([[1.]])
    >>> A[np.diag_indices_from(A)] *= 2
    >>> A
    matrix([[2.]])

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(0, 1, weight=2)
    0
    >>> G.add_edge(1, 0)
    0
    >>> G.add_edge(2, 2, weight=3)
    0
    >>> G.add_edge(2, 2)
    1
    >>> nx.to_numpy_matrix(G, nodelist=[0, 1, 2])
    matrix([[0., 2., 0.],
            [1., 0., 0.],
            [0., 0., 4.]])

    """
    warnings.warn(
        (
            "to_numpy_matrix is deprecated and will be removed in NetworkX 3.0.\n"
            "Use to_numpy_array instead, e.g. np.asmatrix(to_numpy_array(G, **kwargs))"
        ),
        DeprecationWarning,
    )

    import numpy as np

    A = to_numpy_array(
        G,
        nodelist=nodelist,
        dtype=dtype,
        order=order,
        multigraph_weight=multigraph_weight,
        weight=weight,
        nonedge=nonedge,
    )
    M = np.asmatrix(A, dtype=dtype)
    return M


def from_numpy_matrix(A, parallel_edges=False, create_using=None):
    """Returns a graph from numpy matrix.

    The numpy matrix is interpreted as an adjacency matrix for the graph.

    Parameters
    ----------
    A : numpy matrix
        An adjacency matrix representation of a graph

    parallel_edges : Boolean
        If True, `create_using` is a multigraph, and `A` is an
        integer matrix, then entry *(i, j)* in the matrix is interpreted as the
        number of parallel edges joining vertices *i* and *j* in the graph.
        If False, then the entries in the adjacency matrix are interpreted as
        the weight of a single edge joining the vertices.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Notes
    -----
    For directed graphs, explicitly mention create_using=nx.DiGraph,
    and entry i,j of A corresponds to an edge from i to j.

    If `create_using` is :class:`networkx.MultiGraph` or
    :class:`networkx.MultiDiGraph`, `parallel_edges` is True, and the
    entries of `A` are of type :class:`int`, then this function returns a
    multigraph (constructed from `create_using`) with parallel edges.

    If `create_using` indicates an undirected multigraph, then only the edges
    indicated by the upper triangle of the matrix `A` will be added to the
    graph.

    If the numpy matrix has a single data type for each matrix entry it
    will be converted to an appropriate Python data type.

    If the numpy matrix has a user-specified compound data type the names
    of the data fields will be used as attribute keys in the resulting
    NetworkX graph.

    See Also
    --------
    to_numpy_recarray

    Examples
    --------
    Simple integer weights on edges:

    >>> import numpy as np
    >>> A = np.array([[1, 1], [2, 1]])
    >>> G = nx.from_numpy_matrix(A)

    If `create_using` indicates a multigraph and the matrix has only integer
    entries and `parallel_edges` is False, then the entries will be treated
    as weights for edges joining the nodes (without creating parallel edges):

    >>> A = np.array([[1, 1], [1, 2]])
    >>> G = nx.from_numpy_matrix(A, create_using=nx.MultiGraph)
    >>> G[1][1]
    AtlasView({0: {'weight': 2}})

    If `create_using` indicates a multigraph and the matrix has only integer
    entries and `parallel_edges` is True, then the entries will be treated
    as the number of parallel edges joining those two vertices:

    >>> A = np.array([[1, 1], [1, 2]])
    >>> temp = nx.MultiGraph()
    >>> G = nx.from_numpy_matrix(A, parallel_edges=True, create_using=temp)
    >>> G[1][1]
    AtlasView({0: {'weight': 1}, 1: {'weight': 1}})

    User defined compound data type on edges:

    >>> dt = [("weight", float), ("cost", int)]
    >>> A = np.array([[(1.0, 2)]], dtype=dt)
    >>> G = nx.from_numpy_matrix(A)
    >>> list(G.edges())
    [(0, 0)]
    >>> G[0][0]["cost"]
    2
    >>> G[0][0]["weight"]
    1.0

    """
    warnings.warn(
        (
            "from_numpy_matrix is deprecated and will be removed in NetworkX 3.0.\n"
            "Use from_numpy_array instead, e.g. from_numpy_array(A, **kwargs)"
        ),
        DeprecationWarning,
    )
    return from_numpy_array(A, parallel_edges=parallel_edges, create_using=create_using)


@not_implemented_for("multigraph")
def to_numpy_recarray(G, nodelist=None, dtype=None, order=None):
    """Returns the graph adjacency matrix as a NumPy recarray.

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the NumPy recarray.

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in `nodelist`.
       If `nodelist` is None, then the ordering is produced by G.nodes().

    dtype : NumPy data-type, optional
        A valid NumPy named dtype used to initialize the NumPy recarray.
        The data type names are assumed to be keys in the graph edge attribute
        dictionary.

    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory. If None, then the NumPy default
        is used.

    Returns
    -------
    M : NumPy recarray
       The graph with specified edge data as a Numpy recarray

    Notes
    -----
    When `nodelist` does not contain every node in `G`, the adjacency
    matrix is built from the subgraph of `G` that is induced by the nodes in
    `nodelist`.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edge(1, 2, weight=7.0, cost=5)
    >>> A = nx.to_numpy_recarray(G, dtype=[("weight", float), ("cost", int)])
    >>> print(A.weight)
    [[0. 7.]
     [7. 0.]]
    >>> print(A.cost)
    [[0 5]
     [5 0]]

    """
    import numpy as np

    if dtype is None:
        dtype = [("weight", float)]

    if nodelist is None:
        nodelist = list(G)
        nodeset = G
        nlen = len(G)
    else:
        nlen = len(nodelist)
        nodeset = set(G.nbunch_iter(nodelist))
        if nlen != len(nodeset):
            for n in nodelist:
                if n not in G:
                    raise nx.NetworkXError(f"Node {n} in nodelist is not in G")
            raise nx.NetworkXError("nodelist contains duplicates.")

    undirected = not G.is_directed()
    index = dict(zip(nodelist, range(nlen)))
    M = np.zeros((nlen, nlen), dtype=dtype, order=order)

    names = M.dtype.names
    for u, v, attrs in G.edges(data=True):
        if (u in nodeset) and (v in nodeset):
            i, j = index[u], index[v]
            values = tuple([attrs[n] for n in names])
            M[i, j] = values
            if undirected:
                M[j, i] = M[i, j]

    return M.view(np.recarray)


def to_scipy_sparse_matrix(G, nodelist=None, dtype=None, weight="weight", format="csr"):
    """Returns the graph adjacency matrix as a SciPy sparse matrix.

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the sparse matrix.

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in `nodelist`.
       If `nodelist` is None, then the ordering is produced by G.nodes().

    dtype : NumPy data-type, optional
        A valid NumPy dtype used to initialize the array. If None, then the
        NumPy default is used.

    weight : string or None   optional (default='weight')
        The edge attribute that holds the numerical value used for
        the edge weight.  If None then all edge weights are 1.

    format : str in {'bsr', 'csr', 'csc', 'coo', 'lil', 'dia', 'dok'}
        The type of the matrix to be returned (default 'csr').  For
        some algorithms different implementations of sparse matrices
        can perform better.  See [1]_ for details.

    Returns
    -------
    M : SciPy sparse matrix
       Graph adjacency matrix.

    Notes
    -----
    For directed graphs, matrix entry i,j corresponds to an edge from i to j.

    The matrix entries are populated using the edge attribute held in
    parameter weight. When an edge does not have that attribute, the
    value of the entry is 1.

    For multiple edges the matrix values are the sums of the edge weights.

    When `nodelist` does not contain every node in `G`, the adjacency matrix
    is built from the subgraph of `G` that is induced by the nodes in
    `nodelist`.

    The convention used for self-loop edges in graphs is to assign the
    diagonal matrix entry value to the weight attribute of the edge
    (or the number 1 if the edge has no weight attribute).  If the
    alternate convention of doubling the edge weight is desired the
    resulting Scipy sparse matrix can be modified as follows:

    >>> G = nx.Graph([(1, 1)])
    >>> A = nx.to_scipy_sparse_matrix(G)
    >>> print(A.todense())
    [[1]]
    >>> A.setdiag(A.diagonal() * 2)
    >>> print(A.todense())
    [[2]]

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(0, 1, weight=2)
    0
    >>> G.add_edge(1, 0)
    0
    >>> G.add_edge(2, 2, weight=3)
    0
    >>> G.add_edge(2, 2)
    1
    >>> S = nx.to_scipy_sparse_matrix(G, nodelist=[0, 1, 2])
    >>> print(S.todense())
    [[0 2 0]
     [1 0 0]
     [0 0 4]]

    References
    ----------
    .. [1] Scipy Dev. References, "Sparse Matrices",
       https://docs.scipy.org/doc/scipy/reference/sparse.html
    """
    import scipy as sp
    import scipy.sparse  # call as sp.sparse

    if len(G) == 0:
        raise nx.NetworkXError("Graph has no nodes or edges")

    if nodelist is None:
        nodelist = list(G)
        nlen = len(G)
    else:
        nlen = len(nodelist)
        if nlen == 0:
            raise nx.NetworkXError("nodelist has no nodes")
        nodeset = set(G.nbunch_iter(nodelist))
        if nlen != len(nodeset):
            for n in nodelist:
                if n not in G:
                    raise nx.NetworkXError(f"Node {n} in nodelist is not in G")
            raise nx.NetworkXError("nodelist contains duplicates.")
        if nlen < len(G):
            G = G.subgraph(nodelist)

    index = dict(zip(nodelist, range(nlen)))
    coefficients = zip(
        *((index[u], index[v], wt) for u, v, wt in G.edges(data=weight, default=1))
    )
    try:
        row, col, data = coefficients
    except ValueError:
        # there is no edge in the subgraph
        row, col, data = [], [], []

    if G.is_directed():
        M = sp.sparse.coo_matrix((data, (row, col)), shape=(nlen, nlen), dtype=dtype)
    else:
        # symmetrize matrix
        d = data + data
        r = row + col
        c = col + row
        # selfloop entries get double counted when symmetrizing
        # so we subtract the data on the diagonal
        selfloops = list(nx.selfloop_edges(G, data=weight, default=1))
        if selfloops:
            diag_index, diag_data = zip(*((index[u], -wt) for u, v, wt in selfloops))
            d += diag_data
            r += diag_index
            c += diag_index
        M = sp.sparse.coo_matrix((d, (r, c)), shape=(nlen, nlen), dtype=dtype)
    try:
        return M.asformat(format)
    # From Scipy 1.1.0, asformat will throw a ValueError instead of an
    # AttributeError if the format if not recognized.
    except (AttributeError, ValueError) as e:
        raise nx.NetworkXError(f"Unknown sparse matrix format: {format}") from e


def _csr_gen_triples(A):
    """Converts a SciPy sparse matrix in **Compressed Sparse Row** format to
    an iterable of weighted edge triples.

    """
    nrows = A.shape[0]
    data, indices, indptr = A.data, A.indices, A.indptr
    for i in range(nrows):
        for j in range(indptr[i], indptr[i + 1]):
            yield i, indices[j], data[j]


def _csc_gen_triples(A):
    """Converts a SciPy sparse matrix in **Compressed Sparse Column** format to
    an iterable of weighted edge triples.

    """
    ncols = A.shape[1]
    data, indices, indptr = A.data, A.indices, A.indptr
    for i in range(ncols):
        for j in range(indptr[i], indptr[i + 1]):
            yield indices[j], i, data[j]


def _coo_gen_triples(A):
    """Converts a SciPy sparse matrix in **Coordinate** format to an iterable
    of weighted edge triples.

    """
    row, col, data = A.row, A.col, A.data
    return zip(row, col, data)


def _dok_gen_triples(A):
    """Converts a SciPy sparse matrix in **Dictionary of Keys** format to an
    iterable of weighted edge triples.

    """
    for (r, c), v in A.items():
        yield r, c, v


def _generate_weighted_edges(A):
    """Returns an iterable over (u, v, w) triples, where u and v are adjacent
    vertices and w is the weight of the edge joining u and v.

    `A` is a SciPy sparse matrix (in any format).

    """
    if A.format == "csr":
        return _csr_gen_triples(A)
    if A.format == "csc":
        return _csc_gen_triples(A)
    if A.format == "dok":
        return _dok_gen_triples(A)
    # If A is in any other format (including COO), convert it to COO format.
    return _coo_gen_triples(A.tocoo())


def from_scipy_sparse_matrix(
    A, parallel_edges=False, create_using=None, edge_attribute="weight"
):
    """Creates a new graph from an adjacency matrix given as a SciPy sparse
    matrix.

    Parameters
    ----------
    A: scipy sparse matrix
      An adjacency matrix representation of a graph

    parallel_edges : Boolean
      If this is True, `create_using` is a multigraph, and `A` is an
      integer matrix, then entry *(i, j)* in the matrix is interpreted as the
      number of parallel edges joining vertices *i* and *j* in the graph.
      If it is False, then the entries in the matrix are interpreted as
      the weight of a single edge joining the vertices.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    edge_attribute: string
       Name of edge attribute to store matrix numeric value. The data will
       have the same type as the matrix entry (int, float, (real,imag)).

    Notes
    -----
    For directed graphs, explicitly mention create_using=nx.DiGraph,
    and entry i,j of A corresponds to an edge from i to j.

    If `create_using` is :class:`networkx.MultiGraph` or
    :class:`networkx.MultiDiGraph`, `parallel_edges` is True, and the
    entries of `A` are of type :class:`int`, then this function returns a
    multigraph (constructed from `create_using`) with parallel edges.
    In this case, `edge_attribute` will be ignored.

    If `create_using` indicates an undirected multigraph, then only the edges
    indicated by the upper triangle of the matrix `A` will be added to the
    graph.

    Examples
    --------
    >>> import scipy as sp
    >>> import scipy.sparse  # call as sp.sparse
    >>> A = sp.sparse.eye(2, 2, 1)
    >>> G = nx.from_scipy_sparse_matrix(A)

    If `create_using` indicates a multigraph and the matrix has only integer
    entries and `parallel_edges` is False, then the entries will be treated
    as weights for edges joining the nodes (without creating parallel edges):

    >>> A = sp.sparse.csr_matrix([[1, 1], [1, 2]])
    >>> G = nx.from_scipy_sparse_matrix(A, create_using=nx.MultiGraph)
    >>> G[1][1]
    AtlasView({0: {'weight': 2}})

    If `create_using` indicates a multigraph and the matrix has only integer
    entries and `parallel_edges` is True, then the entries will be treated
    as the number of parallel edges joining those two vertices:

    >>> A = sp.sparse.csr_matrix([[1, 1], [1, 2]])
    >>> G = nx.from_scipy_sparse_matrix(
    ...     A, parallel_edges=True, create_using=nx.MultiGraph
    ... )
    >>> G[1][1]
    AtlasView({0: {'weight': 1}, 1: {'weight': 1}})

    """
    G = nx.empty_graph(0, create_using)
    n, m = A.shape
    if n != m:
        raise nx.NetworkXError(f"Adjacency matrix not square: nx,ny={A.shape}")
    # Make sure we get even the isolated nodes of the graph.
    G.add_nodes_from(range(n))
    # Create an iterable over (u, v, w) triples and for each triple, add an
    # edge from u to v with weight w.
    triples = _generate_weighted_edges(A)
    # If the entries in the adjacency matrix are integers, the graph is a
    # multigraph, and parallel_edges is True, then create parallel edges, each
    # with weight 1, for each entry in the adjacency matrix. Otherwise, create
    # one edge for each positive entry in the adjacency matrix and set the
    # weight of that edge to be the entry in the matrix.
    if A.dtype.kind in ("i", "u") and G.is_multigraph() and parallel_edges:
        chain = itertools.chain.from_iterable
        # The following line is equivalent to:
        #
        #     for (u, v) in edges:
        #         for d in range(A[u, v]):
        #             G.add_edge(u, v, weight=1)
        #
        triples = chain(((u, v, 1) for d in range(w)) for (u, v, w) in triples)
    # If we are creating an undirected multigraph, only add the edges from the
    # upper triangle of the matrix. Otherwise, add all the edges. This relies
    # on the fact that the vertices created in the
    # `_generated_weighted_edges()` function are actually the row/column
    # indices for the matrix `A`.
    #
    # Without this check, we run into a problem where each edge is added twice
    # when `G.add_weighted_edges_from()` is invoked below.
    if G.is_multigraph() and not G.is_directed():
        triples = ((u, v, d) for u, v, d in triples if u <= v)
    G.add_weighted_edges_from(triples, weight=edge_attribute)
    return G


def to_numpy_array(
    G,
    nodelist=None,
    dtype=None,
    order=None,
    multigraph_weight=sum,
    weight="weight",
    nonedge=0.0,
):
    """Returns the graph adjacency matrix as a NumPy array.

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the NumPy array.

    nodelist : list, optional
        The rows and columns are ordered according to the nodes in `nodelist`.
        If `nodelist` is None, then the ordering is produced by G.nodes().

    dtype : NumPy data type, optional
        A valid single NumPy data type used to initialize the array.
        This must be a simple type such as int or numpy.float64 and
        not a compound data type (see to_numpy_recarray)
        If None, then the NumPy default is used.

    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory. If None, then the NumPy default
        is used.

    multigraph_weight : {sum, min, max}, optional
        An operator that determines how weights in multigraphs are handled.
        The default is to sum the weights of the multiple edges.

    weight : string or None optional (default = 'weight')
        The edge attribute that holds the numerical value used for
        the edge weight. If an edge does not have that attribute, then the
        value 1 is used instead.

    nonedge : float (default = 0.0)
        The array values corresponding to nonedges are typically set to zero.
        However, this could be undesirable if there are array values
        corresponding to actual edges that also have the value zero. If so,
        one might prefer nonedges to have some other value, such as nan.

    Returns
    -------
    A : NumPy ndarray
        Graph adjacency matrix

    See Also
    --------
    from_numpy_array

    Notes
    -----
    For directed graphs, entry i,j corresponds to an edge from i to j.

    Entries in the adjacency matrix are assigned to the weight edge attribute.
    When an edge does not have a weight attribute, the value of the entry is
    set to the number 1.  For multiple (parallel) edges, the values of the
    entries are determined by the `multigraph_weight` parameter. The default is
    to sum the weight attributes for each of the parallel edges.

    When `nodelist` does not contain every node in `G`, the adjacency matrix is
    built from the subgraph of `G` that is induced by the nodes in `nodelist`.

    The convention used for self-loop edges in graphs is to assign the
    diagonal array entry value to the weight attribute of the edge
    (or the number 1 if the edge has no weight attribute). If the
    alternate convention of doubling the edge weight is desired the
    resulting NumPy array can be modified as follows:

    >>> import numpy as np
    >>> G = nx.Graph([(1, 1)])
    >>> A = nx.to_numpy_array(G)
    >>> A
    array([[1.]])
    >>> A[np.diag_indices_from(A)] *= 2
    >>> A
    array([[2.]])

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(0, 1, weight=2)
    0
    >>> G.add_edge(1, 0)
    0
    >>> G.add_edge(2, 2, weight=3)
    0
    >>> G.add_edge(2, 2)
    1
    >>> nx.to_numpy_array(G, nodelist=[0, 1, 2])
    array([[0., 2., 0.],
           [1., 0., 0.],
           [0., 0., 4.]])

    """
    import numpy as np

    if nodelist is None:
        nodelist = list(G)
        nodeset = G
        nlen = len(G)
    else:
        nlen = len(nodelist)
        nodeset = set(G.nbunch_iter(nodelist))
        if nlen != len(nodeset):
            for n in nodelist:
                if n not in G:
                    raise nx.NetworkXError(f"Node {n} in nodelist is not in G")
            raise nx.NetworkXError("nodelist contains duplicates.")

    undirected = not G.is_directed()
    index = dict(zip(nodelist, range(nlen)))

    # Initially, we start with an array of nans.  Then we populate the array
    # using data from the graph.  Afterwards, any leftover nans will be
    # converted to the value of `nonedge`.  Note, we use nans initially,
    # instead of zero, for two reasons:
    #
    #   1) It can be important to distinguish a real edge with the value 0
    #      from a nonedge with the value 0.
    #
    #   2) When working with multi(di)graphs, we must combine the values of all
    #      edges between any two nodes in some manner.  This often takes the
    #      form of a sum, min, or max.  Using the value 0 for a nonedge would
    #      have undesirable effects with min and max, but using nanmin and
    #      nanmax with initially nan values is not problematic at all.
    #
    # That said, there are still some drawbacks to this approach. Namely, if
    # a real edge is nan, then that value is a) not distinguishable from
    # nonedges and b) is ignored by the default combinator (nansum, nanmin,
    # nanmax) functions used for multi(di)graphs. If this becomes an issue,
    # an alternative approach is to use masked arrays.  Initially, every
    # element is masked and set to some `initial` value. As we populate the
    # graph, elements are unmasked (automatically) when we combine the initial
    # value with the values given by real edges.  At the end, we convert all
    # masked values to `nonedge`. Using masked arrays fully addresses reason 1,
    # but for reason 2, we would still have the issue with min and max if the
    # initial values were 0.0.  Note: an initial value of +inf is appropriate
    # for min, while an initial value of -inf is appropriate for max. When
    # working with sum, an initial value of zero is appropriate. Ideally then,
    # we'd want to allow users to specify both a value for nonedges and also
    # an initial value.  For multi(di)graphs, the choice of the initial value
    # will, in general, depend on the combinator function---sensible defaults
    # can be provided.

    if G.is_multigraph():
        # Handle MultiGraphs and MultiDiGraphs
        A = np.full((nlen, nlen), np.nan, order=order)
        # use numpy nan-aware operations
        operator = {sum: np.nansum, min: np.nanmin, max: np.nanmax}
        try:
            op = operator[multigraph_weight]
        except Exception as e:
            raise ValueError("multigraph_weight must be sum, min, or max") from e

        for u, v, attrs in G.edges(data=True):
            if (u in nodeset) and (v in nodeset):
                i, j = index[u], index[v]
                e_weight = attrs.get(weight, 1)
                A[i, j] = op([e_weight, A[i, j]])
                if undirected:
                    A[j, i] = A[i, j]
    else:
        # Graph or DiGraph, this is much faster than above
        A = np.full((nlen, nlen), np.nan, order=order)
        for u, nbrdict in G.adjacency():
            for v, d in nbrdict.items():
                try:
                    A[index[u], index[v]] = d.get(weight, 1)
                except KeyError:
                    # This occurs when there are fewer desired nodes than
                    # there are nodes in the graph: len(nodelist) < len(G)
                    pass

    A[np.isnan(A)] = nonedge
    A = np.asarray(A, dtype=dtype)
    return A


def from_numpy_array(A, parallel_edges=False, create_using=None):
    """Returns a graph from a 2D NumPy array.

    The 2D NumPy array is interpreted as an adjacency matrix for the graph.

    Parameters
    ----------
    A : a 2D numpy.ndarray
        An adjacency matrix representation of a graph

    parallel_edges : Boolean
        If this is True, `create_using` is a multigraph, and `A` is an
        integer array, then entry *(i, j)* in the array is interpreted as the
        number of parallel edges joining vertices *i* and *j* in the graph.
        If it is False, then the entries in the array are interpreted as
        the weight of a single edge joining the vertices.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Notes
    -----
    For directed graphs, explicitly mention create_using=nx.DiGraph,
    and entry i,j of A corresponds to an edge from i to j.

    If `create_using` is :class:`networkx.MultiGraph` or
    :class:`networkx.MultiDiGraph`, `parallel_edges` is True, and the
    entries of `A` are of type :class:`int`, then this function returns a
    multigraph (of the same type as `create_using`) with parallel edges.

    If `create_using` indicates an undirected multigraph, then only the edges
    indicated by the upper triangle of the array `A` will be added to the
    graph.

    If the NumPy array has a single data type for each array entry it
    will be converted to an appropriate Python data type.

    If the NumPy array has a user-specified compound data type the names
    of the data fields will be used as attribute keys in the resulting
    NetworkX graph.

    See Also
    --------
    to_numpy_array

    Examples
    --------
    Simple integer weights on edges:

    >>> import numpy as np
    >>> A = np.array([[1, 1], [2, 1]])
    >>> G = nx.from_numpy_array(A)
    >>> G.edges(data=True)
    EdgeDataView([(0, 0, {'weight': 1}), (0, 1, {'weight': 2}), (1, 1, {'weight': 1})])

    If `create_using` indicates a multigraph and the array has only integer
    entries and `parallel_edges` is False, then the entries will be treated
    as weights for edges joining the nodes (without creating parallel edges):

    >>> A = np.array([[1, 1], [1, 2]])
    >>> G = nx.from_numpy_array(A, create_using=nx.MultiGraph)
    >>> G[1][1]
    AtlasView({0: {'weight': 2}})

    If `create_using` indicates a multigraph and the array has only integer
    entries and `parallel_edges` is True, then the entries will be treated
    as the number of parallel edges joining those two vertices:

    >>> A = np.array([[1, 1], [1, 2]])
    >>> temp = nx.MultiGraph()
    >>> G = nx.from_numpy_array(A, parallel_edges=True, create_using=temp)
    >>> G[1][1]
    AtlasView({0: {'weight': 1}, 1: {'weight': 1}})

    User defined compound data type on edges:

    >>> dt = [("weight", float), ("cost", int)]
    >>> A = np.array([[(1.0, 2)]], dtype=dt)
    >>> G = nx.from_numpy_array(A)
    >>> G.edges()
    EdgeView([(0, 0)])
    >>> G[0][0]["cost"]
    2
    >>> G[0][0]["weight"]
    1.0

    """
    kind_to_python_type = {
        "f": float,
        "i": int,
        "u": int,
        "b": bool,
        "c": complex,
        "S": str,
        "U": str,
        "V": "void",
    }
    G = nx.empty_graph(0, create_using)
    if A.ndim != 2:
        raise nx.NetworkXError(f"Input array must be 2D, not {A.ndim}")
    n, m = A.shape
    if n != m:
        raise nx.NetworkXError(f"Adjacency matrix not square: nx,ny={A.shape}")
    dt = A.dtype
    try:
        python_type = kind_to_python_type[dt.kind]
    except Exception as e:
        raise TypeError(f"Unknown numpy data type: {dt}") from e

    # Make sure we get even the isolated nodes of the graph.
    G.add_nodes_from(range(n))
    # Get a list of all the entries in the array with nonzero entries. These
    # coordinates become edges in the graph. (convert to int from np.int64)
    edges = ((int(e[0]), int(e[1])) for e in zip(*A.nonzero()))
    # handle numpy constructed data type
    if python_type == "void":
        # Sort the fields by their offset, then by dtype, then by name.
        fields = sorted(
            (offset, dtype, name) for name, (dtype, offset) in A.dtype.fields.items()
        )
        triples = (
            (
                u,
                v,
                {
                    name: kind_to_python_type[dtype.kind](val)
                    for (_, dtype, name), val in zip(fields, A[u, v])
                },
            )
            for u, v in edges
        )
    # If the entries in the adjacency matrix are integers, the graph is a
    # multigraph, and parallel_edges is True, then create parallel edges, each
    # with weight 1, for each entry in the adjacency matrix. Otherwise, create
    # one edge for each positive entry in the adjacency matrix and set the
    # weight of that edge to be the entry in the matrix.
    elif python_type is int and G.is_multigraph() and parallel_edges:
        chain = itertools.chain.from_iterable
        # The following line is equivalent to:
        #
        #     for (u, v) in edges:
        #         for d in range(A[u, v]):
        #             G.add_edge(u, v, weight=1)
        #
        triples = chain(
            ((u, v, {"weight": 1}) for d in range(A[u, v])) for (u, v) in edges
        )
    else:  # basic data type
        triples = ((u, v, dict(weight=python_type(A[u, v]))) for u, v in edges)
    # If we are creating an undirected multigraph, only add the edges from the
    # upper triangle of the matrix. Otherwise, add all the edges. This relies
    # on the fact that the vertices created in the
    # `_generated_weighted_edges()` function are actually the row/column
    # indices for the matrix `A`.
    #
    # Without this check, we run into a problem where each edge is added twice
    # when `G.add_edges_from()` is invoked below.
    if G.is_multigraph() and not G.is_directed():
        triples = ((u, v, d) for u, v, d in triples if u <= v)
    G.add_edges_from(triples)
    return G
