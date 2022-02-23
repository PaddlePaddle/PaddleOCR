from ._mcp import MCP, MCP_Geometric, MCP_Connect, MCP_Flexible


def route_through_array(array, start, end, fully_connected=True,
                        geometric=True):
    """Simple example of how to use the MCP and MCP_Geometric classes.

    See the MCP and MCP_Geometric class documentation for explanation of the
    path-finding algorithm.

    Parameters
    ----------
    array : ndarray
        Array of costs.
    start : iterable
        n-d index into `array` defining the starting point
    end : iterable
        n-d index into `array` defining the end point
    fully_connected : bool (optional)
        If True, diagonal moves are permitted, if False, only axial moves.
    geometric : bool (optional)
        If True, the MCP_Geometric class is used to calculate costs, if False,
        the MCP base class is used. See the class documentation for
        an explanation of the differences between MCP and MCP_Geometric.

    Returns
    -------
    path : list
        List of n-d index tuples defining the path from `start` to `end`.
    cost : float
        Cost of the path. If `geometric` is False, the cost of the path is
        the sum of the values of `array` along the path. If `geometric` is
        True, a finer computation is made (see the documentation of the
        MCP_Geometric class).

    See Also
    --------
    MCP, MCP_Geometric

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.graph import route_through_array
    >>>
    >>> image = np.array([[1, 3], [10, 12]])
    >>> image
    array([[ 1,  3],
           [10, 12]])
    >>> # Forbid diagonal steps
    >>> route_through_array(image, [0, 0], [1, 1], fully_connected=False)
    ([(0, 0), (0, 1), (1, 1)], 9.5)
    >>> # Now allow diagonal steps: the path goes directly from start to end
    >>> route_through_array(image, [0, 0], [1, 1])
    ([(0, 0), (1, 1)], 9.19238815542512)
    >>> # Cost is the sum of array values along the path (16 = 1 + 3 + 12)
    >>> route_through_array(image, [0, 0], [1, 1], fully_connected=False,
    ... geometric=False)
    ([(0, 0), (0, 1), (1, 1)], 16.0)
    >>> # Larger array where we display the path that is selected
    >>> image = np.arange((36)).reshape((6, 6))
    >>> image
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29],
           [30, 31, 32, 33, 34, 35]])
    >>> # Find the path with lowest cost
    >>> indices, weight = route_through_array(image, (0, 0), (5, 5))
    >>> indices = np.stack(indices, axis=-1)
    >>> path = np.zeros_like(image)
    >>> path[indices[0], indices[1]] = 1
    >>> path
    array([[1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 1]])

    """
    start, end = tuple(start), tuple(end)
    if geometric:
        mcp_class = MCP_Geometric
    else:
        mcp_class = MCP
    m = mcp_class(array, fully_connected=fully_connected)
    costs, traceback_array = m.find_costs([start], [end])
    return m.traceback(end), costs[end]
