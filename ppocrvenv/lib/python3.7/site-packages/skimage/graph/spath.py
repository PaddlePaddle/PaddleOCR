import numpy as np
from . import _spath


def shortest_path(arr, reach=1, axis=-1, output_indexlist=False):
    """Find the shortest path through an n-d array from one side to another.

    Parameters
    ----------
    arr : ndarray of float64
    reach : int, optional
        By default (``reach = 1``), the shortest path can only move
        one row up or down for every step it moves forward (i.e.,
        the path gradient is limited to 1). `reach` defines the
        number of elements that can be skipped along each non-axis
        dimension at each step.
    axis : int, optional
        The axis along which the path must always move forward (default -1)
    output_indexlist : bool, optional
        See return value `p` for explanation.

    Returns
    -------
    p : iterable of int
        For each step along `axis`, the coordinate of the shortest path.
        If `output_indexlist` is True, then the path is returned as a list of
        n-d tuples that index into `arr`. If False, then the path is returned
        as an array listing the coordinates of the path along the non-axis
        dimensions for each step along the axis dimension. That is,
        `p.shape == (arr.shape[axis], arr.ndim-1)` except that p is squeezed
        before returning so if `arr.ndim == 2`, then
        `p.shape == (arr.shape[axis],)`
    cost : float
        Cost of path.  This is the absolute sum of all the
        differences along the path.

    """
    # First: calculate the valid moves from any given position. Basically,
    # always move +1 along the given axis, and then can move anywhere within
    # a grid defined by the reach.
    if axis < 0:
        axis += arr.ndim
    offset_ind_shape = (2 * reach + 1,) * (arr.ndim - 1)
    offset_indices = np.indices(offset_ind_shape) - reach
    offset_indices = np.insert(offset_indices, axis,
                               np.ones(offset_ind_shape), axis=0)
    offset_size = np.multiply.reduce(offset_ind_shape)
    offsets = np.reshape(offset_indices, (arr.ndim, offset_size), order='F').T

    # Valid starting positions are anywhere on the hyperplane defined by
    # position 0 on the given axis. Ending positions are anywhere on the
    # hyperplane at position -1 along the same.
    non_axis_shape = arr.shape[:axis] + arr.shape[axis + 1:]
    non_axis_indices = np.indices(non_axis_shape)
    non_axis_size = np.multiply.reduce(non_axis_shape)
    start_indices = np.insert(non_axis_indices, axis,
                              np.zeros(non_axis_shape), axis=0)
    starts = np.reshape(start_indices, (arr.ndim, non_axis_size), order='F').T
    end_indices = np.insert(non_axis_indices, axis,
                            np.full(non_axis_shape, -1,
                                    dtype=non_axis_indices.dtype), axis=0)
    ends = np.reshape(end_indices, (arr.ndim, non_axis_size), order='F').T

    # Find the minimum-cost path to one of the end-points
    m = _spath.MCP_Diff(arr, offsets=offsets)
    costs, traceback = m.find_costs(starts, ends, find_all_ends=False)

    # Figure out which end-point was found
    for end in ends:
        cost = costs[tuple(end)]
        if cost != np.inf:
            break
    traceback = m.traceback(end)

    if not output_indexlist:
        traceback = np.array(traceback)
        traceback = np.concatenate([traceback[:, :axis],
                                    traceback[:, axis + 1:]], axis=1)
        traceback = np.squeeze(traceback)

    return traceback, cost
