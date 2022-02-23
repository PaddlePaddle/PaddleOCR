import numpy

from .._shared.utils import channel_as_last_axis, deprecate_multichannel_kwarg

__all__ = ['apply_parallel']


def _get_chunks(shape, ncpu):
    """Split the array into equal sized chunks based on the number of
    available processors. The last chunk in each dimension absorbs the
    remainder array elements if the number of CPUs does not divide evenly into
    the number of array elements.

    Examples
    --------
    >>> _get_chunks((4, 4), 4)
    ((2, 2), (2, 2))
    >>> _get_chunks((4, 4), 2)
    ((2, 2), (4,))
    >>> _get_chunks((5, 5), 2)
    ((2, 3), (5,))
    >>> _get_chunks((2, 4), 2)
    ((1, 1), (4,))
    """
    # since apply_parallel is in the critical import path, we lazy import
    # math just when we need it.
    from math import ceil

    chunks = []
    nchunks_per_dim = int(ceil(ncpu ** (1./len(shape))))

    used_chunks = 1
    for i in shape:
        if used_chunks < ncpu:
            regular_chunk = i // nchunks_per_dim
            remainder_chunk = regular_chunk + (i % nchunks_per_dim)

            if regular_chunk == 0:
                chunk_lens = (remainder_chunk,)
            else:
                chunk_lens = ((regular_chunk,) * (nchunks_per_dim - 1) +
                              (remainder_chunk,))
        else:
            chunk_lens = (i,)

        chunks.append(chunk_lens)
        used_chunks *= nchunks_per_dim
    return tuple(chunks)


def _ensure_dask_array(array, chunks=None):
    import dask.array as da
    if isinstance(array, da.Array):
        return array

    return da.from_array(array, chunks=chunks)


@deprecate_multichannel_kwarg()
def apply_parallel(function, array, chunks=None, depth=0, mode=None,
                   extra_arguments=(), extra_keywords={}, *, dtype=None,
                   compute=None, channel_axis=None,
                   multichannel=False):
    """Map a function in parallel across an array.

    Split an array into possibly overlapping chunks of a given depth and
    boundary type, call the given function in parallel on the chunks, combine
    the chunks and return the resulting array.

    Parameters
    ----------
    function : function
        Function to be mapped which takes an array as an argument.
    array : numpy array or dask array
        Array which the function will be applied to.
    chunks : int, tuple, or tuple of tuples, optional
        A single integer is interpreted as the length of one side of a square
        chunk that should be tiled across the array.  One tuple of length
        ``array.ndim`` represents the shape of a chunk, and it is tiled across
        the array.  A list of tuples of length ``ndim``, where each sub-tuple
        is a sequence of chunk sizes along the corresponding dimension. If
        None, the array is broken up into chunks based on the number of
        available cpus. More information about chunks is in the documentation
        `here <https://dask.pydata.org/en/latest/array-design.html>`_. When
        `channel_axis` is not None, the tuples can be length ``ndim - 1`` and
        a single chunk will be used along the channel axis.
    depth : int or sequence of int, optional
        The depth of the added boundary cells. A tuple can be used to specify a
        different depth per array axis. Defaults to zero. When `channel_axis`
        is not None, and a tuple of length ``ndim - 1`` is provided, a depth of
        0 will be used along the channel axis.
    mode : {'reflect', 'symmetric', 'periodic', 'wrap', 'nearest', 'edge'}, optional
        Type of external boundary padding.
    extra_arguments : tuple, optional
        Tuple of arguments to be passed to the function.
    extra_keywords : dictionary, optional
        Dictionary of keyword arguments to be passed to the function.
    dtype : data-type or None, optional
        The data-type of the `function` output. If None, Dask will attempt to
        infer this by calling the function on data of shape ``(1,) * ndim``.
        For functions expecting RGB or multichannel data this may be
        problematic. In such cases, the user should manually specify this dtype
        argument instead.

        .. versionadded:: 0.18
           ``dtype`` was added in 0.18.
    compute : bool, optional
        If ``True``, compute eagerly returning a NumPy Array.
        If ``False``, compute lazily returning a Dask Array.
        If ``None`` (default), compute based on array type provided
        (eagerly for NumPy Arrays and lazily for Dask Arrays).
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.
    multichannel : bool, optional
        If `chunks` is None and `multichannel` is True, this function will keep
        only a single chunk along the channels axis. When `depth` is specified
        as a scalar value, that depth will be applied only to the non-channels
        axes (a depth of 0 will be used along the channels axis). If the user
        manually specified both `chunks` and a `depth` tuple, then this
        argument will have no effect. This argument is deprecated: specify
        `channel_axis` instead.

        .. versionadded:: 0.18
           ``multichannel`` was added in 0.18.

    Returns
    -------
    out : ndarray or dask Array
        Returns the result of the applying the operation.
        Type is dependent on the ``compute`` argument.

    Notes
    -----
    Numpy edge modes 'symmetric', 'wrap', and 'edge' are converted to the
    equivalent ``dask`` boundary modes 'reflect', 'periodic' and 'nearest',
    respectively.
    Setting ``compute=False`` can be useful for chaining later operations.
    For example region selection to preview a result or storing large data
    to disk instead of loading in memory.

    """
    try:
        # Importing dask takes time. since apply_parallel is on the
        # minimum import path of skimage, we lazy attempt to import dask
        import dask.array as da
    except ImportError:
        raise RuntimeError("Could not import 'dask'.  Please install "
                           "using 'pip install dask'")

    if compute is None:
        compute = not isinstance(array, da.Array)

    if channel_axis is not None:
        channel_axis = channel_axis % array.ndim

    if chunks is None:
        shape = array.shape
        try:
            # since apply_parallel is in the critical import path, we lazy
            # import multiprocessing just when we need it.
            from multiprocessing import cpu_count
            ncpu = cpu_count()
        except NotImplementedError:
            ncpu = 4
        if channel_axis is not None:
            # use a single chunk along the channel axis
            spatial_shape = shape[:channel_axis] + shape[channel_axis + 1:]
            chunks = list(_get_chunks(spatial_shape, ncpu))
            chunks.insert(channel_axis, shape[channel_axis])
            chunks = tuple(chunks)
        else:
            chunks = _get_chunks(shape, ncpu)
    elif channel_axis is not None and len(chunks) == array.ndim - 1:
        # insert a single chunk along the channel axis
        chunks = list(chunks)
        chunks.insert(channel_axis, array.shape[channel_axis])
        chunks = tuple(chunks)

    if mode == 'wrap':
        mode = 'periodic'
    elif mode == 'symmetric':
        mode = 'reflect'
    elif mode == 'edge':
        mode = 'nearest'
    elif mode is None:
        # default value for Dask.
        # Note: that for dask >= 2022.03 it will change to 'none' so we set it
        #       here for consistent behavior across Dask versions.
        mode = 'reflect'

    if channel_axis is not None:
        if numpy.isscalar(depth):
            # depth is zero along channel_axis
            depth = [depth] * (array.ndim - 1)
        depth = list(depth)
        if len(depth) == array.ndim - 1:
            depth.insert(channel_axis, 0)
        depth = tuple(depth)

    def wrapped_func(arr):
        return function(arr, *extra_arguments, **extra_keywords)

    darr = _ensure_dask_array(array, chunks=chunks)

    res = darr.map_overlap(wrapped_func, depth, boundary=mode, dtype=dtype)
    if compute:
        res = res.compute()

    return res
