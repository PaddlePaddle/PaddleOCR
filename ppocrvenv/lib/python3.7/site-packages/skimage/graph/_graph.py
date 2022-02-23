import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from ..morphology._util import _raveled_offsets_and_distances
from ..util._map_array import map_array


def _weighted_abs_diff(values0, values1, distances):
    """A default edge function for complete image graphs.

    A pixel graph on an image with no edge values and no mask is a very
    boring regular lattice, so we define a default edge weight to be the
    absolute difference between values *weighted* by the distance
    between them.

    Parameters
    ----------
    values0 : array
        The pixel values for each node.
    values1 : array
        The pixel values for each neighbor.
    distances : array
        The distance between each node and its neighbor.

    Returns
    -------
    edge_values : array of float
        The computed values: abs(values0 - values1) * distances.
    """
    return np.abs(values0 - values1) * distances


def pixel_graph(
        image, *, mask=None, edge_function=None, connectivity=1, spacing=None
        ):
    """Create an adjacency graph of pixels in an image.

    Pixels where the mask is True are nodes in the returned graph, and they are
    connected by edges to their neighbors according to the connectivity
    parameter. By default, the *value* of an edge when a mask is given, or when
    the image is itself the mask, is the euclidean distance betwene the pixels.

    However, if an int- or float-valued image is given with no mask, the value
    of the edges is the absolute difference in intensity between adjacent
    pixels, weighted by the euclidean distance.

    Parameters
    ----------
    image : array
        The input image. If the image is of type bool, it will be used as the
        mask as well.
    mask : array of bool
        Which pixels to use. If None, the graph for the whole image is used.
    edge_function : callable
        A function taking an array of pixel values, and an array of neighbor
        pixel values, and an array of distances, and returning a value for the
        edge. If no function is given, the value of an edge is just the
        distance.
    connectivity : int
        The square connectivity of the pixel neighborhood: the number of
        orthogonal steps allowed to consider a pixel a neigbor. See
        `scipy.ndimage.generate_binary_structure` for details.
    spacing : tuple of float
        The spacing between pixels along each axis.

    Returns
    -------
    graph : scipy.sparse.csr_matrix
        A sparse adjacency matrix in which entry (i, j) is 1 if nodes i and j
        are neighbors, 0 otherwise.
    nodes : array of int
        The nodes of the graph. These correspond to the raveled indices of the
        nonzero pixels in the mask.
    """
    if image.dtype == bool and mask is None:
        mask = image
    if mask is None and edge_function is None:
        mask = np.ones_like(image, dtype=bool)
        edge_function = _weighted_abs_diff

    # Strategy: we are going to build the (i, j, data) arrays of a scipy
    # sparse COO matrix, then convert to CSR (which is fast).
    # - grab the raveled IDs of the foreground (mask == True) parts of the
    #   image **in the padded space**.
    # - broadcast them together with the raveled offsets to their neighbors.
    #   This gives us for each foreground pixel a list of neighbors (that
    #   may or may not be selected by the mask). (We also track the *distance*
    #   to each neighbor.)
    # - select "valid" entries in the neighbors and distance arrays by indexing
    #   into the mask, which we can do since these are raveled indices.
    # - use np.repeat() to repeat each source index according to the number
    #   of neighbors selected by the mask it has. Each of these repeated
    #   indices will be lined up with its neighbor, i.e. **this is the i
    #   array** of the COO format matrix.
    # - use the mask as a boolean index to get a 1D view of the selected
    #   neighbors. **This is the j array.**
    # - by default, the same boolean indexing can be applied to the distances
    #   to each neighbor, to give the **data array.** Optionally, a
    #   provided edge function can be computed on the pixel values and the
    #   distances to give a different value for the edges.
    # Note, we use map_array to map the raveled coordinates in the padded
    # image to the ones in the original image, and those are the returned
    # nodes.
    padded = np.pad(mask, 1, mode='constant', constant_values=False)
    nodes_padded = np.flatnonzero(padded)
    neighbor_offsets_padded, distances_padded = _raveled_offsets_and_distances(
            padded.shape, connectivity=connectivity, spacing=spacing
            )
    neighbors_padded = nodes_padded[:, np.newaxis] + neighbor_offsets_padded
    neighbor_distances_full = np.broadcast_to(
            distances_padded, neighbors_padded.shape
            )
    nodes = np.flatnonzero(mask)
    nodes_sequential = np.arange(nodes.size)
    # neighbors outside the mask get mapped to 0, which is a valid index,
    # BUT, they will be masked out in the next step.
    neighbors = map_array(neighbors_padded, nodes_padded, nodes)
    neighbors_mask = padded.reshape(-1)[neighbors_padded]
    num_neighbors = np.sum(neighbors_mask, axis=1)
    indices = np.repeat(nodes, num_neighbors)
    indices_sequential = np.repeat(nodes_sequential, num_neighbors)
    neighbor_indices = neighbors[neighbors_mask]
    neighbor_distances = neighbor_distances_full[neighbors_mask]
    neighbor_indices_sequential = map_array(
            neighbor_indices, nodes, nodes_sequential
            )
    if edge_function is None:
        data = neighbor_distances
    else:
        image_r = image.reshape(-1)
        data = edge_function(
                image_r[indices], image_r[neighbor_indices], neighbor_distances
                )
    m = nodes_sequential.size
    mat = sparse.coo_matrix(
            (data, (indices_sequential, neighbor_indices_sequential)),
            shape=(m, m)
            )
    graph = mat.tocsr()
    return graph, nodes


def central_pixel(graph, nodes=None, shape=None, partition_size=100):
    """Find the pixel with the highest closeness centrality.

    Closeness centrality is the inverse of the total sum of shortest distances
    from a node to every other node.

    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
        The sparse matrix representation of the graph.
    nodes : array of int
        The raveled index of each node in graph in the image. If not provided,
        the returned value will be the index in the input graph.
    shape : tuple of int
        The shape of the image in which the nodes are embedded. If provided,
        the returned coordinates are a NumPy multi-index of the same
        dimensionality as the input shape. Otherwise, the returned coordinate
        is the raveled index provided in `nodes`.
    partition_size : int
        This function computes the shortest path distance between every pair
        of nodes in the graph. This can result in a very large (N*N) matrix.
        As a simple performance tweak, the distance values are computed in
        lots of `partition_size`, resulting in a memory requirement of only
        partition_size*N.

    Returns
    -------
    position : int or tuple of int
        If shape is given, the coordinate of the central pixel in the image.
        Otherwise, the raveled index of that pixel.
    distances : array of float
        The total sum of distances from each node to each other reachable
        node.
    """
    if nodes is None:
        nodes = np.arange(graph.shape[0])
    if partition_size is None:
        num_splits = 1
    else:
        num_splits = max(2, graph.shape[0] // partition_size)
    idxs = np.arange(graph.shape[0])
    total_shortest_path_len_list = []
    for partition in np.array_split(idxs, num_splits):
        shortest_paths = csgraph.shortest_path(
                graph, directed=False, indices=partition
                )
        shortest_paths_no_inf = np.nan_to_num(shortest_paths)
        total_shortest_path_len_list.append(
                np.sum(shortest_paths_no_inf, axis=1)
                )
    total_shortest_path_len = np.concatenate(total_shortest_path_len_list)
    nonzero = np.flatnonzero(total_shortest_path_len)
    min_sp = np.argmin(total_shortest_path_len[nonzero])
    raveled_index = nodes[nonzero[min_sp]]
    if shape is not None:
        central = np.unravel_index(raveled_index, shape)
    else:
        central = raveled_index
    return central, total_shortest_path_len
