import networkx as nx
import numpy as np
from . import _ncut
from . import _ncut_cy
from scipy.sparse import linalg


def cut_threshold(labels, rag, thresh, in_place=True):
    """Combine regions separated by weight less than threshold.

    Given an image's labels and its RAG, output new labels by
    combining regions whose nodes are separated by a weight less
    than the given threshold.

    Parameters
    ----------
    labels : ndarray
        The array of labels.
    rag : RAG
        The region adjacency graph.
    thresh : float
        The threshold. Regions connected by edges with smaller weights are
        combined.
    in_place : bool
        If set, modifies `rag` in place. The function will remove the edges
        with weights less that `thresh`. If set to `False` the function
        makes a copy of `rag` before proceeding.

    Returns
    -------
    out : ndarray
        The new labelled array.

    Examples
    --------
    >>> from skimage import data, segmentation
    >>> from skimage.future import graph
    >>> img = data.astronaut()
    >>> labels = segmentation.slic(img)
    >>> rag = graph.rag_mean_color(img, labels)
    >>> new_labels = graph.cut_threshold(labels, rag, 10)

    References
    ----------
    .. [1] Alain Tremeau and Philippe Colantoni
           "Regions Adjacency Graph Applied To Color Image Segmentation"
           :DOI:`10.1109/83.841950`

    """
    if not in_place:
        rag = rag.copy()

    # Because deleting edges while iterating through them produces an error.
    to_remove = [(x, y) for x, y, d in rag.edges(data=True)
                 if d['weight'] >= thresh]
    rag.remove_edges_from(to_remove)

    comps = nx.connected_components(rag)

    # We construct an array which can map old labels to the new ones.
    # All the labels within a connected component are assigned to a single
    # label in the output.
    map_array = np.arange(labels.max() + 1, dtype=labels.dtype)
    for i, nodes in enumerate(comps):
        for node in nodes:
            for label in rag.nodes[node]['labels']:
                map_array[label] = i

    return map_array[labels]


def cut_normalized(labels, rag, thresh=0.001, num_cuts=10, in_place=True,
                   max_edge=1.0,
                   *,
                   random_state=None,
                   ):
    """Perform Normalized Graph cut on the Region Adjacency Graph.

    Given an image's labels and its similarity RAG, recursively perform
    a 2-way normalized cut on it. All nodes belonging to a subgraph
    that cannot be cut further are assigned a unique label in the
    output.

    Parameters
    ----------
    labels : ndarray
        The array of labels.
    rag : RAG
        The region adjacency graph.
    thresh : float
        The threshold. A subgraph won't be further subdivided if the
        value of the N-cut exceeds `thresh`.
    num_cuts : int
        The number or N-cuts to perform before determining the optimal one.
    in_place : bool
        If set, modifies `rag` in place. For each node `n` the function will
        set a new attribute ``rag.nodes[n]['ncut label']``.
    max_edge : float, optional
        The maximum possible value of an edge in the RAG. This corresponds to
        an edge between identical regions. This is used to put self
        edges in the RAG.
    random_state : {None, int, `numpy.random.Generator`}, optional
        If `random_state` is None the `numpy.random.Generator` singleton is
        used.
        If `random_state` is an int, a new ``Generator`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` instance then that
        instance is used.

        The `random_state` is used for the starting point
        of `scipy.sparse.linalg.eigsh`.

    Returns
    -------
    out : ndarray
        The new labeled array.

    Examples
    --------
    >>> from skimage import data, segmentation
    >>> from skimage.future import graph
    >>> img = data.astronaut()
    >>> labels = segmentation.slic(img)
    >>> rag = graph.rag_mean_color(img, labels, mode='similarity')
    >>> new_labels = graph.cut_normalized(labels, rag)

    References
    ----------
    .. [1] Shi, J.; Malik, J., "Normalized cuts and image segmentation",
           Pattern Analysis and Machine Intelligence,
           IEEE Transactions on, vol. 22, no. 8, pp. 888-905, August 2000.

    """
    random_state = np.random.default_rng(random_state)
    if not in_place:
        rag = rag.copy()

    for node in rag.nodes():
        rag.add_edge(node, node, weight=max_edge)

    _ncut_relabel(rag, thresh, num_cuts, random_state)

    map_array = np.zeros(labels.max() + 1, dtype=labels.dtype)
    # Mapping from old labels to new
    for n, d in rag.nodes(data=True):
        map_array[d['labels']] = d['ncut label']

    return map_array[labels]


def partition_by_cut(cut, rag):
    """Compute resulting subgraphs from given bi-partition.

    Parameters
    ----------
    cut : array
        A array of booleans. Elements set to `True` belong to one
        set.
    rag : RAG
        The Region Adjacency Graph.

    Returns
    -------
    sub1, sub2 : RAG
        The two resulting subgraphs from the bi-partition.
    """
    # `cut` is derived from `D` and `W` matrices, which also follow the
    # ordering returned by `rag.nodes()` because we use
    # nx.to_scipy_sparse_matrix.

    # Example
    # rag.nodes() = [3, 7, 9, 13]
    # cut = [True, False, True, False]
    # nodes1 = [3, 9]
    # nodes2 = [7, 10]

    nodes1 = [n for i, n in enumerate(rag.nodes()) if cut[i]]
    nodes2 = [n for i, n in enumerate(rag.nodes()) if not cut[i]]

    sub1 = rag.subgraph(nodes1)
    sub2 = rag.subgraph(nodes2)

    return sub1, sub2


def get_min_ncut(ev, d, w, num_cuts):
    """Threshold an eigenvector evenly, to determine minimum ncut.

    Parameters
    ----------
    ev : array
        The eigenvector to threshold.
    d : ndarray
        The diagonal matrix of the graph.
    w : ndarray
        The weight matrix of the graph.
    num_cuts : int
        The number of evenly spaced thresholds to check for.

    Returns
    -------
    mask : array
        The array of booleans which denotes the bi-partition.
    mcut : float
        The value of the minimum ncut.
    """
    mcut = np.inf
    mn = ev.min()
    mx = ev.max()

    # If all values in `ev` are equal, it implies that the graph can't be
    # further sub-divided. In this case the bi-partition is the the graph
    # itself and an empty set.
    min_mask = np.zeros_like(ev, dtype=bool)
    if np.allclose(mn, mx):
        return min_mask, mcut

    # Refer Shi & Malik 2001, Section 3.1.3, Page 892
    # Perform evenly spaced n-cuts and determine the optimal one.
    for t in np.linspace(mn, mx, num_cuts, endpoint=False):
        mask = ev > t
        cost = _ncut.ncut_cost(mask, d, w)
        if cost < mcut:
            min_mask = mask
            mcut = cost

    return min_mask, mcut


def _label_all(rag, attr_name):
    """Assign a unique integer to the given attribute in the RAG.

    This function assumes that all labels in `rag` are unique. It
    picks up a random label from them and assigns it to the `attr_name`
    attribute of all the nodes.

    rag : RAG
        The Region Adjacency Graph.
    attr_name : string
        The attribute to which a unique integer is assigned.
    """
    node = min(rag.nodes())
    new_label = rag.nodes[node]['labels'][0]
    for n, d in rag.nodes(data=True):
        d[attr_name] = new_label


def _ncut_relabel(rag, thresh, num_cuts, random_state):
    """Perform Normalized Graph cut on the Region Adjacency Graph.

    Recursively partition the graph into 2, until further subdivision
    yields a cut greater than `thresh` or such a cut cannot be computed.
    For such a subgraph, indices to labels of all its nodes map to a single
    unique value.

    Parameters
    ----------
    rag : RAG
        The region adjacency graph.
    thresh : float
        The threshold. A subgraph won't be further subdivided if the
        value of the N-cut exceeds `thresh`.
    num_cuts : int
        The number or N-cuts to perform before determining the optimal one.
    random_state : `numpy.random.Generator`
        Provides initial values for eigenvalue solver.
    """
    d, w = _ncut.DW_matrices(rag)
    m = w.shape[0]

    if m > 2:
        d2 = d.copy()
        # Since d is diagonal, we can directly operate on its data
        # the inverse of the square root
        d2.data = np.reciprocal(np.sqrt(d2.data, out=d2.data), out=d2.data)

        # Refer Shi & Malik 2001, Equation 7, Page 891
        A = d2 * (d - w) * d2
        # Initialize the vector to ensure reproducibility.
        v0 = random_state.random(A.shape[0])
        vals, vectors = linalg.eigsh(A, which='SM', v0=v0,
                                     k=min(100, m - 2))

        # Pick second smallest eigenvector.
        # Refer Shi & Malik 2001, Section 3.2.3, Page 893
        vals, vectors = np.real(vals), np.real(vectors)
        index2 = _ncut_cy.argmin2(vals)
        ev = vectors[:, index2]

        cut_mask, mcut = get_min_ncut(ev, d, w, num_cuts)
        if (mcut < thresh):
            # Sub divide and perform N-cut again
            # Refer Shi & Malik 2001, Section 3.2.5, Page 893
            sub1, sub2 = partition_by_cut(cut_mask, rag)

            _ncut_relabel(sub1, thresh, num_cuts, random_state)
            _ncut_relabel(sub2, thresh, num_cuts, random_state)
            return

    # The N-cut wasn't small enough, or could not be computed.
    # The remaining graph is a region.
    # Assign `ncut label` by picking any label from the existing nodes, since
    # `labels` are unique, `new_label` is also unique.
    _label_all(rag, 'ncut label')
