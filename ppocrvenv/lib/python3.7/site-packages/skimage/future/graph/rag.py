import networkx as nx
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import ndimage as ndi
from scipy import sparse
import math

from ... import measure, segmentation, util, color
from ..._shared.version_requirements import require


def _edge_generator_from_csr(csr_matrix):
    """Yield weighted edge triples for use by NetworkX from a CSR matrix.

    This function is a straight rewrite of
    `networkx.convert_matrix._csr_gen_triples`. Since that is a private
    function, it is safer to include our own here.

    Parameters
    ----------
    csr_matrix : scipy.sparse.csr_matrix
        The input matrix. An edge (i, j, w) will be yielded if there is a
        data value for coordinates (i, j) in the matrix, even if that value
        is 0.

    Yields
    ------
    i, j, w : (int, int, float) tuples
        Each value `w` in the matrix along with its coordinates (i, j).

    Examples
    --------

    >>> dense = np.eye(2, dtype=float)
    >>> csr = sparse.csr_matrix(dense)
    >>> edges = _edge_generator_from_csr(csr)
    >>> list(edges)
    [(0, 0, 1.0), (1, 1, 1.0)]
    """
    nrows = csr_matrix.shape[0]
    values = csr_matrix.data
    indptr = csr_matrix.indptr
    col_indices = csr_matrix.indices
    for i in range(nrows):
        for j in range(indptr[i], indptr[i + 1]):
            yield i, col_indices[j], values[j]


def min_weight(graph, src, dst, n):
    """Callback to handle merging nodes by choosing minimum weight.

    Returns a dictionary with `"weight"` set as either the weight between
    (`src`, `n`) or (`dst`, `n`) in `graph` or the minimum of the two when
    both exist.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The verices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dict with the `"weight"` attribute set the weight between
        (`src`, `n`) or (`dst`, `n`) in `graph` or the minimum of the two when
        both exist.

    """

    # cover the cases where n only has edge to either `src` or `dst`
    default = {'weight': np.inf}
    w1 = graph[n].get(src, default)['weight']
    w2 = graph[n].get(dst, default)['weight']
    return {'weight': min(w1, w2)}


def _add_edge_filter(values, graph):
    """Create edge in `graph` between central element of `values` and the rest.

    Add an edge between the middle element in `values` and
    all other elements of `values` into `graph`.  ``values[len(values) // 2]``
    is expected to be the central value of the footprint used.

    Parameters
    ----------
    values : array
        The array to process.
    graph : RAG
        The graph to add edges in.

    Returns
    -------
    0 : float
        Always returns 0. The return value is required so that `generic_filter`
        can put it in the output array, but it is ignored by this filter.
    """
    values = values.astype(int)
    center = values[len(values) // 2]
    for value in values:
        if value != center and not graph.has_edge(center, value):
            graph.add_edge(center, value)
    return 0.


class RAG(nx.Graph):

    """
    The Region Adjacency Graph (RAG) of an image, subclasses
    `networx.Graph <http://networkx.github.io/documentation/latest/reference/classes/graph.html>`_

    Parameters
    ----------
    label_image : array of int
        An initial segmentation, with each region labeled as a different
        integer. Every unique value in ``label_image`` will correspond to
        a node in the graph.
    connectivity : int in {1, ..., ``label_image.ndim``}, optional
        The connectivity between pixels in ``label_image``. For a 2D image,
        a connectivity of 1 corresponds to immediate neighbors up, down,
        left, and right, while a connectivity of 2 also includes diagonal
        neighbors. See `scipy.ndimage.generate_binary_structure`.
    data : networkx Graph specification, optional
        Initial or additional edges to pass to the NetworkX Graph
        constructor. See `networkx.Graph`. Valid edge specifications
        include edge list (list of tuples), NumPy arrays, and SciPy
        sparse matrices.
    **attr : keyword arguments, optional
        Additional attributes to add to the graph.
    """

    def __init__(self, label_image=None, connectivity=1, data=None, **attr):

        super(RAG, self).__init__(data, **attr)
        if self.number_of_nodes() == 0:
            self.max_id = 0
        else:
            self.max_id = max(self.nodes())

        if label_image is not None:
            fp = ndi.generate_binary_structure(label_image.ndim, connectivity)
            # In the next ``ndi.generic_filter`` function, the kwarg
            # ``output`` is used to provide a strided array with a single
            # 64-bit floating point number, to which the function repeatedly
            # writes. This is done because even if we don't care about the
            # output, without this, a float array of the same shape as the
            # input image will be created and that could be expensive in
            # memory consumption.
            ndi.generic_filter(
                label_image,
                function=_add_edge_filter,
                footprint=fp,
                mode='nearest',
                output=as_strided(np.empty((1,), dtype=float),
                                  shape=label_image.shape,
                                  strides=((0,) * label_image.ndim)),
                extra_arguments=(self,))

    def merge_nodes(self, src, dst, weight_func=min_weight, in_place=True,
                    extra_arguments=[], extra_keywords={}):
        """Merge node `src` and `dst`.

        The new combined node is adjacent to all the neighbors of `src`
        and `dst`. `weight_func` is called to decide the weight of edges
        incident on the new node.

        Parameters
        ----------
        src, dst : int
            Nodes to be merged.
        weight_func : callable, optional
            Function to decide the attributes of edges incident on the new
            node. For each neighbor `n` for `src and `dst`, `weight_func` will
            be called as follows: `weight_func(src, dst, n, *extra_arguments,
            **extra_keywords)`. `src`, `dst` and `n` are IDs of vertices in the
            RAG object which is in turn a subclass of `networkx.Graph`. It is
            expected to return a dict of attributes of the resulting edge.
        in_place : bool, optional
            If set to `True`, the merged node has the id `dst`, else merged
            node has a new id which is returned.
        extra_arguments : sequence, optional
            The sequence of extra positional arguments passed to
            `weight_func`.
        extra_keywords : dictionary, optional
            The dict of keyword arguments passed to the `weight_func`.

        Returns
        -------
        id : int
            The id of the new node.

        Notes
        -----
        If `in_place` is `False` the resulting node has a new id, rather than
        `dst`.
        """
        src_nbrs = set(self.neighbors(src))
        dst_nbrs = set(self.neighbors(dst))
        neighbors = (src_nbrs | dst_nbrs) - {src, dst}

        if in_place:
            new = dst
        else:
            new = self.next_id()
            self.add_node(new)

        for neighbor in neighbors:
            data = weight_func(self, src, new, neighbor, *extra_arguments,
                               **extra_keywords)
            self.add_edge(neighbor, new, attr_dict=data)

        self.nodes[new]['labels'] = (self.nodes[src]['labels'] +
                                     self.nodes[dst]['labels'])
        self.remove_node(src)

        if not in_place:
            self.remove_node(dst)

        return new

    def add_node(self, n, attr_dict=None, **attr):
        """Add node `n` while updating the maximum node id.

        .. seealso:: :func:`networkx.Graph.add_node`."""
        if attr_dict is None:  # compatibility with old networkx
            attr_dict = attr
        else:
            attr_dict.update(attr)
        super(RAG, self).add_node(n, **attr_dict)
        self.max_id = max(n, self.max_id)

    def add_edge(self, u, v, attr_dict=None, **attr):
        """Add an edge between `u` and `v` while updating max node id.

        .. seealso:: :func:`networkx.Graph.add_edge`."""
        if attr_dict is None:  # compatibility with old networkx
            attr_dict = attr
        else:
            attr_dict.update(attr)
        super(RAG, self).add_edge(u, v, **attr_dict)
        self.max_id = max(u, v, self.max_id)

    def copy(self):
        """Copy the graph with its max node id.

        .. seealso:: :func:`networkx.Graph.copy`."""
        g = super(RAG, self).copy()
        g.max_id = self.max_id
        return g

    def fresh_copy(self):
        """Return a fresh copy graph with the same data structure.

        A fresh copy has no nodes, edges or graph attributes. It is
        the same data structure as the current graph. This method is
        typically used to create an empty version of the graph.

        This is required when subclassing Graph with networkx v2 and
        does not cause problems for v1. Here is more detail from
        the network migrating from 1.x to 2.x document::

            With the new GraphViews (SubGraph, ReversedGraph, etc)
            you can't assume that ``G.__class__()`` will create a new
            instance of the same graph type as ``G``. In fact, the
            call signature for ``__class__`` differs depending on
            whether ``G`` is a view or a base class. For v2.x you
            should use ``G.fresh_copy()`` to create a null graph of
            the correct type---ready to fill with nodes and edges.

        """
        return RAG()

    def next_id(self):
        """Returns the `id` for the new node to be inserted.

        The current implementation returns one more than the maximum `id`.

        Returns
        -------
        id : int
            The `id` of the new node to be inserted.
        """
        return self.max_id + 1

    def _add_node_silent(self, n):
        """Add node `n` without updating the maximum node id.

        This is a convenience method used internally.

        .. seealso:: :func:`networkx.Graph.add_node`."""
        super(RAG, self).add_node(n)


def rag_mean_color(image, labels, connectivity=2, mode='distance',
                   sigma=255.0):
    """Compute the Region Adjacency Graph using mean colors.

    Given an image and its initial segmentation, this method constructs the
    corresponding Region Adjacency Graph (RAG). Each node in the RAG
    represents a set of pixels within `image` with the same label in `labels`.
    The weight between two adjacent regions represents how similar or
    dissimilar two regions are depending on the `mode` parameter.

    Parameters
    ----------
    image : ndarray, shape(M, N, [..., P,] 3)
        Input image.
    labels : ndarray, shape(M, N, [..., P])
        The labelled image. This should have one dimension less than
        `image`. If `image` has dimensions `(M, N, 3)` `labels` should have
        dimensions `(M, N)`.
    connectivity : int, optional
        Pixels with a squared distance less than `connectivity` from each other
        are considered adjacent. It can range from 1 to `labels.ndim`. Its
        behavior is the same as `connectivity` parameter in
        ``scipy.ndimage.generate_binary_structure``.
    mode : {'distance', 'similarity'}, optional
        The strategy to assign edge weights.

            'distance' : The weight between two adjacent regions is the
            :math:`|c_1 - c_2|`, where :math:`c_1` and :math:`c_2` are the mean
            colors of the two regions. It represents the Euclidean distance in
            their average color.

            'similarity' : The weight between two adjacent is
            :math:`e^{-d^2/sigma}` where :math:`d=|c_1 - c_2|`, where
            :math:`c_1` and :math:`c_2` are the mean colors of the two regions.
            It represents how similar two regions are.
    sigma : float, optional
        Used for computation when `mode` is "similarity". It governs how
        close to each other two colors should be, for their corresponding edge
        weight to be significant. A very large value of `sigma` could make
        any two colors behave as though they were similar.

    Returns
    -------
    out : RAG
        The region adjacency graph.

    Examples
    --------
    >>> from skimage import data, segmentation
    >>> from skimage.future import graph
    >>> img = data.astronaut()
    >>> labels = segmentation.slic(img)
    >>> rag = graph.rag_mean_color(img, labels)

    References
    ----------
    .. [1] Alain Tremeau and Philippe Colantoni
           "Regions Adjacency Graph Applied To Color Image Segmentation"
           :DOI:`10.1109/83.841950`
    """
    graph = RAG(labels, connectivity=connectivity)

    for n in graph:
        graph.nodes[n].update({'labels': [n],
                               'pixel count': 0,
                               'total color': np.array([0, 0, 0],
                                                      dtype=np.double)})

    for index in np.ndindex(labels.shape):
        current = labels[index]
        graph.nodes[current]['pixel count'] += 1
        graph.nodes[current]['total color'] += image[index]

    for n in graph:
        graph.nodes[n]['mean color'] = (graph.nodes[n]['total color'] /
                                        graph.nodes[n]['pixel count'])

    for x, y, d in graph.edges(data=True):
        diff = graph.nodes[x]['mean color'] - graph.nodes[y]['mean color']
        diff = np.linalg.norm(diff)
        if mode == 'similarity':
            d['weight'] = math.e ** (-(diff ** 2) / sigma)
        elif mode == 'distance':
            d['weight'] = diff
        else:
            raise ValueError("The mode '%s' is not recognised" % mode)

    return graph


def rag_boundary(labels, edge_map, connectivity=2):
    """ Comouter RAG based on region boundaries

    Given an image's initial segmentation and its edge map this method
    constructs the corresponding Region Adjacency Graph (RAG). Each node in the
    RAG represents a set of pixels within the image with the same label in
    `labels`. The weight between two adjacent regions is the average value
    in `edge_map` along their boundary.

    labels : ndarray
        The labelled image.
    edge_map : ndarray
        This should have the same shape as that of `labels`. For all pixels
        along the boundary between 2 adjacent regions, the average value of the
        corresponding pixels in `edge_map` is the edge weight between them.
    connectivity : int, optional
        Pixels with a squared distance less than `connectivity` from each other
        are considered adjacent. It can range from 1 to `labels.ndim`. Its
        behavior is the same as `connectivity` parameter in
        `scipy.ndimage.filters.generate_binary_structure`.

    Examples
    --------
    >>> from skimage import data, segmentation, filters, color
    >>> from skimage.future import graph
    >>> img = data.chelsea()
    >>> labels = segmentation.slic(img)
    >>> edge_map = filters.sobel(color.rgb2gray(img))
    >>> rag = graph.rag_boundary(labels, edge_map)

    """

    conn = ndi.generate_binary_structure(labels.ndim, connectivity)
    eroded = ndi.grey_erosion(labels, footprint=conn)
    dilated = ndi.grey_dilation(labels, footprint=conn)
    boundaries0 = (eroded != labels)
    boundaries1 = (dilated != labels)
    labels_small = np.concatenate((eroded[boundaries0], labels[boundaries1]))
    labels_large = np.concatenate((labels[boundaries0], dilated[boundaries1]))
    n = np.max(labels_large) + 1

    # use a dummy broadcast array as data for RAG
    ones = as_strided(np.ones((1,), dtype=float), shape=labels_small.shape,
                      strides=(0,))
    count_matrix = sparse.coo_matrix((ones, (labels_small, labels_large)),
                                     dtype=int, shape=(n, n)).tocsr()
    data = np.concatenate((edge_map[boundaries0], edge_map[boundaries1]))

    data_coo = sparse.coo_matrix((data, (labels_small, labels_large)))
    graph_matrix = data_coo.tocsr()
    graph_matrix.data /= count_matrix.data

    rag = RAG()
    rag.add_weighted_edges_from(_edge_generator_from_csr(graph_matrix),
                                weight='weight')
    rag.add_weighted_edges_from(_edge_generator_from_csr(count_matrix),
                                weight='count')

    for n in rag.nodes():
        rag.nodes[n].update({'labels': [n]})

    return rag


@require("matplotlib", ">=3.0.3")
def show_rag(labels, rag, image, border_color='black', edge_width=1.5,
             edge_cmap='magma', img_cmap='bone', in_place=True, ax=None):
    """Show a Region Adjacency Graph on an image.

    Given a labelled image and its corresponding RAG, show the nodes and edges
    of the RAG on the image with the specified colors. Edges are displayed between
    the centroid of the 2 adjacent regions in the image.

    Parameters
    ----------
    labels : ndarray, shape (M, N)
        The labelled image.
    rag : RAG
        The Region Adjacency Graph.
    image : ndarray, shape (M, N[, 3])
        Input image. If `colormap` is `None`, the image should be in RGB
        format.
    border_color : color spec, optional
        Color with which the borders between regions are drawn.
    edge_width : float, optional
        The thickness with which the RAG edges are drawn.
    edge_cmap : :py:class:`matplotlib.colors.Colormap`, optional
        Any matplotlib colormap with which the edges are drawn.
    img_cmap : :py:class:`matplotlib.colors.Colormap`, optional
        Any matplotlib colormap with which the image is draw. If set to `None`
        the image is drawn as it is.
    in_place : bool, optional
        If set, the RAG is modified in place. For each node `n` the function
        will set a new attribute ``rag.nodes[n]['centroid']``.
    ax : :py:class:`matplotlib.axes.Axes`, optional
        The axes to draw on. If not specified, new axes are created and drawn
        on.

    Returns
    -------
    lc : :py:class:`matplotlib.collections.LineCollection`
         A colection of lines that represent the edges of the graph. It can be
         passed to the :meth:`matplotlib.figure.Figure.colorbar` function.

    Examples
    --------
    >>> from skimage import data, segmentation
    >>> from skimage.future import graph
    >>> import matplotlib.pyplot as plt
    >>>
    >>> img = data.coffee()
    >>> labels = segmentation.slic(img)
    >>> g =  graph.rag_mean_color(img, labels)
    >>> lc = graph.show_rag(labels, g, img)
    >>> cbar = plt.colorbar(lc)
    """
    from matplotlib import colors, cm
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection

    if not in_place:
        rag = rag.copy()

    if ax is None:
        fig, ax = plt.subplots()
    out = util.img_as_float(image, force_copy=True)

    if img_cmap is None:
        if image.ndim < 3 or image.shape[2] not in [3, 4]:
            msg = 'If colormap is `None`, an RGB or RGBA image should be given'
            raise ValueError(msg)
        # Ignore the alpha channel
        out = image[:, :, :3]
    else:
        img_cmap = cm.get_cmap(img_cmap)
        out = color.rgb2gray(image)
        # Ignore the alpha channel
        out = img_cmap(out)[:, :, :3]

    edge_cmap = cm.get_cmap(edge_cmap)

    # Handling the case where one node has multiple labels
    # offset is 1 so that regionprops does not ignore 0
    offset = 1
    map_array = np.arange(labels.max() + 1)
    for n, d in rag.nodes(data=True):
        for label in d['labels']:
            map_array[label] = offset
        offset += 1

    rag_labels = map_array[labels]
    regions = measure.regionprops(rag_labels)

    for (n, data), region in zip(rag.nodes(data=True), regions):
        data['centroid'] = tuple(map(int, region['centroid']))

    cc = colors.ColorConverter()
    if border_color is not None:
        border_color = cc.to_rgb(border_color)
        out = segmentation.mark_boundaries(out, rag_labels, color=border_color)

    ax.imshow(out)

    # Defining the end points of the edges
    # The tuple[::-1] syntax reverses a tuple as matplotlib uses (x,y)
    # convention while skimage uses (row, column)
    lines = [[rag.nodes[n1]['centroid'][::-1], rag.nodes[n2]['centroid'][::-1]]
              for (n1, n2) in rag.edges()]

    lc = LineCollection(lines, linewidths=edge_width, cmap=edge_cmap)
    edge_weights = [d['weight'] for x, y, d in rag.edges(data=True)]
    lc.set_array(np.array(edge_weights))
    ax.add_collection(lc)

    return lc
