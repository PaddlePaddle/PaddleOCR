# Copyright (C) 2003-2005 Peter J. Verveer
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy
import numpy as np
from . import _ni_support
from . import _ni_label
from . import _nd_image
from . import morphology

__all__ = ['label', 'find_objects', 'labeled_comprehension', 'sum', 'mean',
           'variance', 'standard_deviation', 'minimum', 'maximum', 'median',
           'minimum_position', 'maximum_position', 'extrema', 'center_of_mass',
           'histogram', 'watershed_ift', 'sum_labels']


def label(input, structure=None, output=None):
    """
    Label features in an array.

    Parameters
    ----------
    input : array_like
        An array-like object to be labeled. Any non-zero values in `input` are
        counted as features and zero values are considered the background.
    structure : array_like, optional
        A structuring element that defines feature connections.
        `structure` must be centrosymmetric
        (see Notes).
        If no structuring element is provided,
        one is automatically generated with a squared connectivity equal to
        one.  That is, for a 2-D `input` array, the default structuring element
        is::

            [[0,1,0],
             [1,1,1],
             [0,1,0]]

    output : (None, data-type, array_like), optional
        If `output` is a data type, it specifies the type of the resulting
        labeled feature array.
        If `output` is an array-like object, then `output` will be updated
        with the labeled features from this function.  This function can
        operate in-place, by passing output=input.
        Note that the output must be able to store the largest label, or this
        function will raise an Exception.

    Returns
    -------
    label : ndarray or int
        An integer ndarray where each unique feature in `input` has a unique
        label in the returned array.
    num_features : int
        How many objects were found.

        If `output` is None, this function returns a tuple of
        (`labeled_array`, `num_features`).

        If `output` is a ndarray, then it will be updated with values in
        `labeled_array` and only `num_features` will be returned by this
        function.

    See Also
    --------
    find_objects : generate a list of slices for the labeled features (or
                   objects); useful for finding features' position or
                   dimensions

    Notes
    -----
    A centrosymmetric matrix is a matrix that is symmetric about the center.
    See [1]_ for more information.

    The `structure` matrix must be centrosymmetric to ensure
    two-way connections.
    For instance, if the `structure` matrix is not centrosymmetric
    and is defined as::

        [[0,1,0],
         [1,1,0],
         [0,0,0]]

    and the `input` is::

        [[1,2],
         [0,3]]

    then the structure matrix would indicate the
    entry 2 in the input is connected to 1,
    but 1 is not connected to 2.

    Examples
    --------
    Create an image with some features, then label it using the default
    (cross-shaped) structuring element:

    >>> from scipy.ndimage import label, generate_binary_structure
    >>> a = np.array([[0,0,1,1,0,0],
    ...               [0,0,0,1,0,0],
    ...               [1,1,0,0,1,0],
    ...               [0,0,0,1,0,0]])
    >>> labeled_array, num_features = label(a)

    Each of the 4 features are labeled with a different integer:

    >>> num_features
    4
    >>> labeled_array
    array([[0, 0, 1, 1, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [2, 2, 0, 0, 3, 0],
           [0, 0, 0, 4, 0, 0]])

    Generate a structuring element that will consider features connected even
    if they touch diagonally:

    >>> s = generate_binary_structure(2,2)

    or,

    >>> s = [[1,1,1],
    ...      [1,1,1],
    ...      [1,1,1]]

    Label the image using the new structuring element:

    >>> labeled_array, num_features = label(a, structure=s)

    Show the 2 labeled features (note that features 1, 3, and 4 from above are
    now considered a single feature):

    >>> num_features
    2
    >>> labeled_array
    array([[0, 0, 1, 1, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [2, 2, 0, 0, 1, 0],
           [0, 0, 0, 1, 0, 0]])

    References
    ----------

    .. [1] James R. Weaver, "Centrosymmetric (cross-symmetric)
       matrices, their basic properties, eigenvalues, and
       eigenvectors." The American Mathematical Monthly 92.10
       (1985): 711-717.

    """
    input = numpy.asarray(input)
    if numpy.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    if structure is None:
        structure = morphology.generate_binary_structure(input.ndim, 1)
    structure = numpy.asarray(structure, dtype=bool)
    if structure.ndim != input.ndim:
        raise RuntimeError('structure and input must have equal rank')
    for ii in structure.shape:
        if ii != 3:
            raise ValueError('structure dimensions must be equal to 3')

    # Use 32 bits if it's large enough for this image.
    # _ni_label.label() needs two entries for background and
    # foreground tracking
    need_64bits = input.size >= (2**31 - 2)

    if isinstance(output, numpy.ndarray):
        if output.shape != input.shape:
            raise ValueError("output shape not correct")
        caller_provided_output = True
    else:
        caller_provided_output = False
        if output is None:
            output = np.empty(input.shape, np.intp if need_64bits else np.int32)
        else:
            output = np.empty(input.shape, output)

    # handle scalars, 0-D arrays
    if input.ndim == 0 or input.size == 0:
        if input.ndim == 0:
            # scalar
            maxlabel = 1 if (input != 0) else 0
            output[...] = maxlabel
        else:
            # 0-D
            maxlabel = 0
        if caller_provided_output:
            return maxlabel
        else:
            return output, maxlabel

    try:
        max_label = _ni_label._label(input, structure, output)
    except _ni_label.NeedMoreBits as e:
        # Make another attempt with enough bits, then try to cast to the
        # new type.
        tmp_output = np.empty(input.shape, np.intp if need_64bits else np.int32)
        max_label = _ni_label._label(input, structure, tmp_output)
        output[...] = tmp_output[...]
        if not np.all(output == tmp_output):
            # refuse to return bad results
            raise RuntimeError(
                "insufficient bit-depth in requested output type"
            ) from e

    if caller_provided_output:
        # result was written in-place
        return max_label
    else:
        return output, max_label


def find_objects(input, max_label=0):
    """
    Find objects in a labeled array.

    Parameters
    ----------
    input : ndarray of ints
        Array containing objects defined by different labels. Labels with
        value 0 are ignored.
    max_label : int, optional
        Maximum label to be searched for in `input`. If max_label is not
        given, the positions of all objects are returned.

    Returns
    -------
    object_slices : list of tuples
        A list of tuples, with each tuple containing N slices (with N the
        dimension of the input array). Slices correspond to the minimal
        parallelepiped that contains the object. If a number is missing,
        None is returned instead of a slice.

    See Also
    --------
    label, center_of_mass

    Notes
    -----
    This function is very useful for isolating a volume of interest inside
    a 3-D array, that cannot be "seen through".

    Examples
    --------
    >>> from scipy import ndimage
    >>> a = np.zeros((6,6), dtype=int)
    >>> a[2:4, 2:4] = 1
    >>> a[4, 4] = 1
    >>> a[:2, :3] = 2
    >>> a[0, 5] = 3
    >>> a
    array([[2, 2, 2, 0, 0, 3],
           [2, 2, 2, 0, 0, 0],
           [0, 0, 1, 1, 0, 0],
           [0, 0, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0]])
    >>> ndimage.find_objects(a)
    [(slice(2, 5, None), slice(2, 5, None)), (slice(0, 2, None), slice(0, 3, None)), (slice(0, 1, None), slice(5, 6, None))]
    >>> ndimage.find_objects(a, max_label=2)
    [(slice(2, 5, None), slice(2, 5, None)), (slice(0, 2, None), slice(0, 3, None))]
    >>> ndimage.find_objects(a == 1, max_label=2)
    [(slice(2, 5, None), slice(2, 5, None)), None]

    >>> loc = ndimage.find_objects(a)[0]
    >>> a[loc]
    array([[1, 1, 0],
           [1, 1, 0],
           [0, 0, 1]])

    """
    input = numpy.asarray(input)
    if numpy.iscomplexobj(input):
        raise TypeError('Complex type not supported')

    if max_label < 1:
        max_label = input.max()

    return _nd_image.find_objects(input, max_label)


def labeled_comprehension(input, labels, index, func, out_dtype, default, pass_positions=False):
    """
    Roughly equivalent to [func(input[labels == i]) for i in index].

    Sequentially applies an arbitrary function (that works on array_like input)
    to subsets of an N-D image array specified by `labels` and `index`.
    The option exists to provide the function with positional parameters as the
    second argument.

    Parameters
    ----------
    input : array_like
        Data from which to select `labels` to process.
    labels : array_like or None
        Labels to objects in `input`.
        If not None, array must be same shape as `input`.
        If None, `func` is applied to raveled `input`.
    index : int, sequence of ints or None
        Subset of `labels` to which to apply `func`.
        If a scalar, a single value is returned.
        If None, `func` is applied to all non-zero values of `labels`.
    func : callable
        Python function to apply to `labels` from `input`.
    out_dtype : dtype
        Dtype to use for `result`.
    default : int, float or None
        Default return value when a element of `index` does not exist
        in `labels`.
    pass_positions : bool, optional
        If True, pass linear indices to `func` as a second argument.
        Default is False.

    Returns
    -------
    result : ndarray
        Result of applying `func` to each of `labels` to `input` in `index`.

    Examples
    --------
    >>> a = np.array([[1, 2, 0, 0],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> from scipy import ndimage
    >>> lbl, nlbl = ndimage.label(a)
    >>> lbls = np.arange(1, nlbl+1)
    >>> ndimage.labeled_comprehension(a, lbl, lbls, np.mean, float, 0)
    array([ 2.75,  5.5 ,  6.  ])

    Falling back to `default`:

    >>> lbls = np.arange(1, nlbl+2)
    >>> ndimage.labeled_comprehension(a, lbl, lbls, np.mean, float, -1)
    array([ 2.75,  5.5 ,  6.  , -1.  ])

    Passing positions:

    >>> def fn(val, pos):
    ...     print("fn says: %s : %s" % (val, pos))
    ...     return (val.sum()) if (pos.sum() % 2 == 0) else (-val.sum())
    ...
    >>> ndimage.labeled_comprehension(a, lbl, lbls, fn, float, 0, True)
    fn says: [1 2 5 3] : [0 1 4 5]
    fn says: [4 7] : [ 7 11]
    fn says: [9 3] : [12 13]
    array([ 11.,  11., -12.,   0.])

    """

    as_scalar = numpy.isscalar(index)
    input = numpy.asarray(input)

    if pass_positions:
        positions = numpy.arange(input.size).reshape(input.shape)

    if labels is None:
        if index is not None:
            raise ValueError("index without defined labels")
        if not pass_positions:
            return func(input.ravel())
        else:
            return func(input.ravel(), positions.ravel())

    try:
        input, labels = numpy.broadcast_arrays(input, labels)
    except ValueError as e:
        raise ValueError("input and labels must have the same shape "
                            "(excepting dimensions with width 1)") from e

    if index is None:
        if not pass_positions:
            return func(input[labels > 0])
        else:
            return func(input[labels > 0], positions[labels > 0])

    index = numpy.atleast_1d(index)
    if np.any(index.astype(labels.dtype).astype(index.dtype) != index):
        raise ValueError("Cannot convert index values from <%s> to <%s> "
                            "(labels' type) without loss of precision" %
                            (index.dtype, labels.dtype))

    index = index.astype(labels.dtype)

    # optimization: find min/max in index, and select those parts of labels, input, and positions
    lo = index.min()
    hi = index.max()
    mask = (labels >= lo) & (labels <= hi)

    # this also ravels the arrays
    labels = labels[mask]
    input = input[mask]
    if pass_positions:
        positions = positions[mask]

    # sort everything by labels
    label_order = labels.argsort()
    labels = labels[label_order]
    input = input[label_order]
    if pass_positions:
        positions = positions[label_order]

    index_order = index.argsort()
    sorted_index = index[index_order]

    def do_map(inputs, output):
        """labels must be sorted"""
        nidx = sorted_index.size

        # Find boundaries for each stretch of constant labels
        # This could be faster, but we already paid N log N to sort labels.
        lo = numpy.searchsorted(labels, sorted_index, side='left')
        hi = numpy.searchsorted(labels, sorted_index, side='right')

        for i, l, h in zip(range(nidx), lo, hi):
            if l == h:
                continue
            output[i] = func(*[inp[l:h] for inp in inputs])

    temp = numpy.empty(index.shape, out_dtype)
    temp[:] = default
    if not pass_positions:
        do_map([input], temp)
    else:
        do_map([input, positions], temp)

    output = numpy.zeros(index.shape, out_dtype)
    output[index_order] = temp
    if as_scalar:
        output = output[0]

    return output


def _safely_castable_to_int(dt):
    """Test whether the NumPy data type `dt` can be safely cast to an int."""
    int_size = np.dtype(int).itemsize
    safe = ((np.issubdtype(dt, np.signedinteger) and dt.itemsize <= int_size) or
            (np.issubdtype(dt, np.unsignedinteger) and dt.itemsize < int_size))
    return safe


def _stats(input, labels=None, index=None, centered=False):
    """Count, sum, and optionally compute (sum - centre)^2 of input by label

    Parameters
    ----------
    input : array_like, N-D
        The input data to be analyzed.
    labels : array_like (N-D), optional
        The labels of the data in `input`. This array must be broadcast
        compatible with `input`; typically, it is the same shape as `input`.
        If `labels` is None, all nonzero values in `input` are treated as
        the single labeled group.
    index : label or sequence of labels, optional
        These are the labels of the groups for which the stats are computed.
        If `index` is None, the stats are computed for the single group where
        `labels` is greater than 0.
    centered : bool, optional
        If True, the centered sum of squares for each labeled group is
        also returned. Default is False.

    Returns
    -------
    counts : int or ndarray of ints
        The number of elements in each labeled group.
    sums : scalar or ndarray of scalars
        The sums of the values in each labeled group.
    sums_c : scalar or ndarray of scalars, optional
        The sums of mean-centered squares of the values in each labeled group.
        This is only returned if `centered` is True.

    """
    def single_group(vals):
        if centered:
            vals_c = vals - vals.mean()
            return vals.size, vals.sum(), (vals_c * vals_c.conjugate()).sum()
        else:
            return vals.size, vals.sum()

    if labels is None:
        return single_group(input)

    # ensure input and labels match sizes
    input, labels = numpy.broadcast_arrays(input, labels)

    if index is None:
        return single_group(input[labels > 0])

    if numpy.isscalar(index):
        return single_group(input[labels == index])

    def _sum_centered(labels):
        # `labels` is expected to be an ndarray with the same shape as `input`.
        # It must contain the label indices (which are not necessarily the labels
        # themselves).
        means = sums / counts
        centered_input = input - means[labels]
        # bincount expects 1-D inputs, so we ravel the arguments.
        bc = numpy.bincount(labels.ravel(),
                              weights=(centered_input *
                                       centered_input.conjugate()).ravel())
        return bc

    # Remap labels to unique integers if necessary, or if the largest
    # label is larger than the number of values.

    if (not _safely_castable_to_int(labels.dtype) or
            labels.min() < 0 or labels.max() > labels.size):
        # Use numpy.unique to generate the label indices.  `new_labels` will
        # be 1-D, but it should be interpreted as the flattened N-D array of
        # label indices.
        unique_labels, new_labels = numpy.unique(labels, return_inverse=True)
        counts = numpy.bincount(new_labels)
        sums = numpy.bincount(new_labels, weights=input.ravel())
        if centered:
            # Compute the sum of the mean-centered squares.
            # We must reshape new_labels to the N-D shape of `input` before
            # passing it _sum_centered.
            sums_c = _sum_centered(new_labels.reshape(labels.shape))
        idxs = numpy.searchsorted(unique_labels, index)
        # make all of idxs valid
        idxs[idxs >= unique_labels.size] = 0
        found = (unique_labels[idxs] == index)
    else:
        # labels are an integer type allowed by bincount, and there aren't too
        # many, so call bincount directly.
        counts = numpy.bincount(labels.ravel())
        sums = numpy.bincount(labels.ravel(), weights=input.ravel())
        if centered:
            sums_c = _sum_centered(labels)
        # make sure all index values are valid
        idxs = numpy.asanyarray(index, numpy.int_).copy()
        found = (idxs >= 0) & (idxs < counts.size)
        idxs[~found] = 0

    counts = counts[idxs]
    counts[~found] = 0
    sums = sums[idxs]
    sums[~found] = 0

    if not centered:
        return (counts, sums)
    else:
        sums_c = sums_c[idxs]
        sums_c[~found] = 0
        return (counts, sums, sums_c)


def sum(input, labels=None, index=None):
    """
    Calculate the sum of the values of the array.

    Notes
    -----
    This is an alias for `ndimage.sum_labels` kept for backwards compatibility
    reasons, for new code please prefer `sum_labels`.  See the `sum_labels`
    docstring for more details.

    """
    return sum_labels(input, labels, index)


def sum_labels(input, labels=None, index=None):
    """
    Calculate the sum of the values of the array.

    Parameters
    ----------
    input : array_like
        Values of `input` inside the regions defined by `labels`
        are summed together.
    labels : array_like of ints, optional
        Assign labels to the values of the array. Has to have the same shape as
        `input`.
    index : array_like, optional
        A single label number or a sequence of label numbers of
        the objects to be measured.

    Returns
    -------
    sum : ndarray or scalar
        An array of the sums of values of `input` inside the regions defined
        by `labels` with the same shape as `index`. If 'index' is None or scalar,
        a scalar is returned.

    See also
    --------
    mean, median

    Examples
    --------
    >>> from scipy import ndimage
    >>> input =  [0,1,2,3]
    >>> labels = [1,1,2,2]
    >>> ndimage.sum(input, labels, index=[1,2])
    [1.0, 5.0]
    >>> ndimage.sum(input, labels, index=1)
    1
    >>> ndimage.sum(input, labels)
    6


    """
    count, sum = _stats(input, labels, index)
    return sum


def mean(input, labels=None, index=None):
    """
    Calculate the mean of the values of an array at labels.

    Parameters
    ----------
    input : array_like
        Array on which to compute the mean of elements over distinct
        regions.
    labels : array_like, optional
        Array of labels of same shape, or broadcastable to the same shape as
        `input`. All elements sharing the same label form one region over
        which the mean of the elements is computed.
    index : int or sequence of ints, optional
        Labels of the objects over which the mean is to be computed.
        Default is None, in which case the mean for all values where label is
        greater than 0 is calculated.

    Returns
    -------
    out : list
        Sequence of same length as `index`, with the mean of the different
        regions labeled by the labels in `index`.

    See also
    --------
    variance, standard_deviation, minimum, maximum, sum, label

    Examples
    --------
    >>> from scipy import ndimage
    >>> a = np.arange(25).reshape((5,5))
    >>> labels = np.zeros_like(a)
    >>> labels[3:5,3:5] = 1
    >>> index = np.unique(labels)
    >>> labels
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1],
           [0, 0, 0, 1, 1]])
    >>> index
    array([0, 1])
    >>> ndimage.mean(a, labels=labels, index=index)
    [10.285714285714286, 21.0]

    """

    count, sum = _stats(input, labels, index)
    return sum / numpy.asanyarray(count).astype(numpy.float64)


def variance(input, labels=None, index=None):
    """
    Calculate the variance of the values of an N-D image array, optionally at
    specified sub-regions.

    Parameters
    ----------
    input : array_like
        Nd-image data to process.
    labels : array_like, optional
        Labels defining sub-regions in `input`.
        If not None, must be same shape as `input`.
    index : int or sequence of ints, optional
        `labels` to include in output.  If None (default), all values where
        `labels` is non-zero are used.

    Returns
    -------
    variance : float or ndarray
        Values of variance, for each sub-region if `labels` and `index` are
        specified.

    See Also
    --------
    label, standard_deviation, maximum, minimum, extrema

    Examples
    --------
    >>> a = np.array([[1, 2, 0, 0],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> from scipy import ndimage
    >>> ndimage.variance(a)
    7.609375

    Features to process can be specified using `labels` and `index`:

    >>> lbl, nlbl = ndimage.label(a)
    >>> ndimage.variance(a, lbl, index=np.arange(1, nlbl+1))
    array([ 2.1875,  2.25  ,  9.    ])

    If no index is given, all non-zero `labels` are processed:

    >>> ndimage.variance(a, lbl)
    6.1875

    """
    count, sum, sum_c_sq = _stats(input, labels, index, centered=True)
    return sum_c_sq / np.asanyarray(count).astype(float)


def standard_deviation(input, labels=None, index=None):
    """
    Calculate the standard deviation of the values of an N-D image array,
    optionally at specified sub-regions.

    Parameters
    ----------
    input : array_like
        N-D image data to process.
    labels : array_like, optional
        Labels to identify sub-regions in `input`.
        If not None, must be same shape as `input`.
    index : int or sequence of ints, optional
        `labels` to include in output. If None (default), all values where
        `labels` is non-zero are used.

    Returns
    -------
    standard_deviation : float or ndarray
        Values of standard deviation, for each sub-region if `labels` and
        `index` are specified.

    See Also
    --------
    label, variance, maximum, minimum, extrema

    Examples
    --------
    >>> a = np.array([[1, 2, 0, 0],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> from scipy import ndimage
    >>> ndimage.standard_deviation(a)
    2.7585095613392387

    Features to process can be specified using `labels` and `index`:

    >>> lbl, nlbl = ndimage.label(a)
    >>> ndimage.standard_deviation(a, lbl, index=np.arange(1, nlbl+1))
    array([ 1.479,  1.5  ,  3.   ])

    If no index is given, non-zero `labels` are processed:

    >>> ndimage.standard_deviation(a, lbl)
    2.4874685927665499

    """
    return numpy.sqrt(variance(input, labels, index))


def _select(input, labels=None, index=None, find_min=False, find_max=False,
            find_min_positions=False, find_max_positions=False,
            find_median=False):
    """Returns min, max, or both, plus their positions (if requested), and
    median."""

    input = numpy.asanyarray(input)

    find_positions = find_min_positions or find_max_positions
    positions = None
    if find_positions:
        positions = numpy.arange(input.size).reshape(input.shape)

    def single_group(vals, positions):
        result = []
        if find_min:
            result += [vals.min()]
        if find_min_positions:
            result += [positions[vals == vals.min()][0]]
        if find_max:
            result += [vals.max()]
        if find_max_positions:
            result += [positions[vals == vals.max()][0]]
        if find_median:
            result += [numpy.median(vals)]
        return result

    if labels is None:
        return single_group(input, positions)

    # ensure input and labels match sizes
    input, labels = numpy.broadcast_arrays(input, labels)

    if index is None:
        mask = (labels > 0)
        masked_positions = None
        if find_positions:
            masked_positions = positions[mask]
        return single_group(input[mask], masked_positions)

    if numpy.isscalar(index):
        mask = (labels == index)
        masked_positions = None
        if find_positions:
            masked_positions = positions[mask]
        return single_group(input[mask], masked_positions)

    # remap labels to unique integers if necessary, or if the largest
    # label is larger than the number of values.
    if (not _safely_castable_to_int(labels.dtype) or
            labels.min() < 0 or labels.max() > labels.size):
        # remap labels, and indexes
        unique_labels, labels = numpy.unique(labels, return_inverse=True)
        idxs = numpy.searchsorted(unique_labels, index)

        # make all of idxs valid
        idxs[idxs >= unique_labels.size] = 0
        found = (unique_labels[idxs] == index)
    else:
        # labels are an integer type, and there aren't too many
        idxs = numpy.asanyarray(index, numpy.int_).copy()
        found = (idxs >= 0) & (idxs <= labels.max())

    idxs[~ found] = labels.max() + 1

    if find_median:
        order = numpy.lexsort((input.ravel(), labels.ravel()))
    else:
        order = input.ravel().argsort()
    input = input.ravel()[order]
    labels = labels.ravel()[order]
    if find_positions:
        positions = positions.ravel()[order]

    result = []
    if find_min:
        mins = numpy.zeros(labels.max() + 2, input.dtype)
        mins[labels[::-1]] = input[::-1]
        result += [mins[idxs]]
    if find_min_positions:
        minpos = numpy.zeros(labels.max() + 2, int)
        minpos[labels[::-1]] = positions[::-1]
        result += [minpos[idxs]]
    if find_max:
        maxs = numpy.zeros(labels.max() + 2, input.dtype)
        maxs[labels] = input
        result += [maxs[idxs]]
    if find_max_positions:
        maxpos = numpy.zeros(labels.max() + 2, int)
        maxpos[labels] = positions
        result += [maxpos[idxs]]
    if find_median:
        locs = numpy.arange(len(labels))
        lo = numpy.zeros(labels.max() + 2, numpy.int_)
        lo[labels[::-1]] = locs[::-1]
        hi = numpy.zeros(labels.max() + 2, numpy.int_)
        hi[labels] = locs
        lo = lo[idxs]
        hi = hi[idxs]
        # lo is an index to the lowest value in input for each label,
        # hi is an index to the largest value.
        # move them to be either the same ((hi - lo) % 2 == 0) or next
        # to each other ((hi - lo) % 2 == 1), then average.
        step = (hi - lo) // 2
        lo += step
        hi -= step
        if (np.issubdtype(input.dtype, np.integer)
                or np.issubdtype(input.dtype, np.bool_)):
            # avoid integer overflow or boolean addition (gh-12836)
            result += [(input[lo].astype('d') + input[hi].astype('d')) / 2.0]
        else:
            result += [(input[lo] + input[hi]) / 2.0]

    return result


def minimum(input, labels=None, index=None):
    """
    Calculate the minimum of the values of an array over labeled regions.

    Parameters
    ----------
    input : array_like
        Array_like of values. For each region specified by `labels`, the
        minimal values of `input` over the region is computed.
    labels : array_like, optional
        An array_like of integers marking different regions over which the
        minimum value of `input` is to be computed. `labels` must have the
        same shape as `input`. If `labels` is not specified, the minimum
        over the whole array is returned.
    index : array_like, optional
        A list of region labels that are taken into account for computing the
        minima. If index is None, the minimum over all elements where `labels`
        is non-zero is returned.

    Returns
    -------
    minimum : float or list of floats
        List of minima of `input` over the regions determined by `labels` and
        whose index is in `index`. If `index` or `labels` are not specified, a
        float is returned: the minimal value of `input` if `labels` is None,
        and the minimal value of elements where `labels` is greater than zero
        if `index` is None.

    See also
    --------
    label, maximum, median, minimum_position, extrema, sum, mean, variance,
    standard_deviation

    Notes
    -----
    The function returns a Python list and not a NumPy array, use
    `np.array` to convert the list to an array.

    Examples
    --------
    >>> from scipy import ndimage
    >>> a = np.array([[1, 2, 0, 0],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> labels, labels_nb = ndimage.label(a)
    >>> labels
    array([[1, 1, 0, 0],
           [1, 1, 0, 2],
           [0, 0, 0, 2],
           [3, 3, 0, 0]])
    >>> ndimage.minimum(a, labels=labels, index=np.arange(1, labels_nb + 1))
    [1.0, 4.0, 3.0]
    >>> ndimage.minimum(a)
    0.0
    >>> ndimage.minimum(a, labels=labels)
    1.0

    """
    return _select(input, labels, index, find_min=True)[0]


def maximum(input, labels=None, index=None):
    """
    Calculate the maximum of the values of an array over labeled regions.

    Parameters
    ----------
    input : array_like
        Array_like of values. For each region specified by `labels`, the
        maximal values of `input` over the region is computed.
    labels : array_like, optional
        An array of integers marking different regions over which the
        maximum value of `input` is to be computed. `labels` must have the
        same shape as `input`. If `labels` is not specified, the maximum
        over the whole array is returned.
    index : array_like, optional
        A list of region labels that are taken into account for computing the
        maxima. If index is None, the maximum over all elements where `labels`
        is non-zero is returned.

    Returns
    -------
    output : float or list of floats
        List of maxima of `input` over the regions determined by `labels` and
        whose index is in `index`. If `index` or `labels` are not specified, a
        float is returned: the maximal value of `input` if `labels` is None,
        and the maximal value of elements where `labels` is greater than zero
        if `index` is None.

    See also
    --------
    label, minimum, median, maximum_position, extrema, sum, mean, variance,
    standard_deviation

    Notes
    -----
    The function returns a Python list and not a NumPy array, use
    `np.array` to convert the list to an array.

    Examples
    --------
    >>> a = np.arange(16).reshape((4,4))
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> labels = np.zeros_like(a)
    >>> labels[:2,:2] = 1
    >>> labels[2:, 1:3] = 2
    >>> labels
    array([[1, 1, 0, 0],
           [1, 1, 0, 0],
           [0, 2, 2, 0],
           [0, 2, 2, 0]])
    >>> from scipy import ndimage
    >>> ndimage.maximum(a)
    15.0
    >>> ndimage.maximum(a, labels=labels, index=[1,2])
    [5.0, 14.0]
    >>> ndimage.maximum(a, labels=labels)
    14.0

    >>> b = np.array([[1, 2, 0, 0],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> labels, labels_nb = ndimage.label(b)
    >>> labels
    array([[1, 1, 0, 0],
           [1, 1, 0, 2],
           [0, 0, 0, 2],
           [3, 3, 0, 0]])
    >>> ndimage.maximum(b, labels=labels, index=np.arange(1, labels_nb + 1))
    [5.0, 7.0, 9.0]

    """
    return _select(input, labels, index, find_max=True)[0]


def median(input, labels=None, index=None):
    """
    Calculate the median of the values of an array over labeled regions.

    Parameters
    ----------
    input : array_like
        Array_like of values. For each region specified by `labels`, the
        median value of `input` over the region is computed.
    labels : array_like, optional
        An array_like of integers marking different regions over which the
        median value of `input` is to be computed. `labels` must have the
        same shape as `input`. If `labels` is not specified, the median
        over the whole array is returned.
    index : array_like, optional
        A list of region labels that are taken into account for computing the
        medians. If index is None, the median over all elements where `labels`
        is non-zero is returned.

    Returns
    -------
    median : float or list of floats
        List of medians of `input` over the regions determined by `labels` and
        whose index is in `index`. If `index` or `labels` are not specified, a
        float is returned: the median value of `input` if `labels` is None,
        and the median value of elements where `labels` is greater than zero
        if `index` is None.

    See also
    --------
    label, minimum, maximum, extrema, sum, mean, variance, standard_deviation

    Notes
    -----
    The function returns a Python list and not a NumPy array, use
    `np.array` to convert the list to an array.

    Examples
    --------
    >>> from scipy import ndimage
    >>> a = np.array([[1, 2, 0, 1],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> labels, labels_nb = ndimage.label(a)
    >>> labels
    array([[1, 1, 0, 2],
           [1, 1, 0, 2],
           [0, 0, 0, 2],
           [3, 3, 0, 0]])
    >>> ndimage.median(a, labels=labels, index=np.arange(1, labels_nb + 1))
    [2.5, 4.0, 6.0]
    >>> ndimage.median(a)
    1.0
    >>> ndimage.median(a, labels=labels)
    3.0

    """
    return _select(input, labels, index, find_median=True)[0]


def minimum_position(input, labels=None, index=None):
    """
    Find the positions of the minimums of the values of an array at labels.

    Parameters
    ----------
    input : array_like
        Array_like of values.
    labels : array_like, optional
        An array of integers marking different regions over which the
        position of the minimum value of `input` is to be computed.
        `labels` must have the same shape as `input`. If `labels` is not
        specified, the location of the first minimum over the whole
        array is returned.

        The `labels` argument only works when `index` is specified.
    index : array_like, optional
        A list of region labels that are taken into account for finding the
        location of the minima. If `index` is None, the ``first`` minimum
        over all elements where `labels` is non-zero is returned.

        The `index` argument only works when `labels` is specified.

    Returns
    -------
    output : list of tuples of ints
        Tuple of ints or list of tuples of ints that specify the location
        of minima of `input` over the regions determined by `labels` and
        whose index is in `index`.

        If `index` or `labels` are not specified, a tuple of ints is
        returned specifying the location of the first minimal value of `input`.

    See also
    --------
    label, minimum, median, maximum_position, extrema, sum, mean, variance,
    standard_deviation

    Examples
    --------
    >>> a = np.array([[10, 20, 30],
    ...               [40, 80, 100],
    ...               [1, 100, 200]])
    >>> b = np.array([[1, 2, 0, 1],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])

    >>> from scipy import ndimage

    >>> ndimage.minimum_position(a)
    (2, 0)
    >>> ndimage.minimum_position(b)
    (0, 2)

    Features to process can be specified using `labels` and `index`:

    >>> label, pos = ndimage.label(a)
    >>> ndimage.minimum_position(a, label, index=np.arange(1, pos+1))
    [(2, 0)]

    >>> label, pos = ndimage.label(b)
    >>> ndimage.minimum_position(b, label, index=np.arange(1, pos+1))
    [(0, 0), (0, 3), (3, 1)]

    """
    dims = numpy.array(numpy.asarray(input).shape)
    # see numpy.unravel_index to understand this line.
    dim_prod = numpy.cumprod([1] + list(dims[:0:-1]))[::-1]

    result = _select(input, labels, index, find_min_positions=True)[0]

    if numpy.isscalar(result):
        return tuple((result // dim_prod) % dims)

    return [tuple(v) for v in (result.reshape(-1, 1) // dim_prod) % dims]


def maximum_position(input, labels=None, index=None):
    """
    Find the positions of the maximums of the values of an array at labels.

    For each region specified by `labels`, the position of the maximum
    value of `input` within the region is returned.

    Parameters
    ----------
    input : array_like
        Array_like of values.
    labels : array_like, optional
        An array of integers marking different regions over which the
        position of the maximum value of `input` is to be computed.
        `labels` must have the same shape as `input`. If `labels` is not
        specified, the location of the first maximum over the whole
        array is returned.

        The `labels` argument only works when `index` is specified.
    index : array_like, optional
        A list of region labels that are taken into account for finding the
        location of the maxima. If `index` is None, the first maximum
        over all elements where `labels` is non-zero is returned.

        The `index` argument only works when `labels` is specified.

    Returns
    -------
    output : list of tuples of ints
        List of tuples of ints that specify the location of maxima of
        `input` over the regions determined by `labels` and whose index
        is in `index`.

        If `index` or `labels` are not specified, a tuple of ints is
        returned specifying the location of the ``first`` maximal value
        of `input`.

    See also
    --------
    label, minimum, median, maximum_position, extrema, sum, mean, variance,
    standard_deviation

    Examples
    --------
    >>> from scipy import ndimage
    >>> a = np.array([[1, 2, 0, 0],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> ndimage.maximum_position(a)
    (3, 0)

    Features to process can be specified using `labels` and `index`:

    >>> lbl = np.array([[0, 1, 2, 3],
    ...                 [0, 1, 2, 3],
    ...                 [0, 1, 2, 3],
    ...                 [0, 1, 2, 3]])
    >>> ndimage.maximum_position(a, lbl, 1)
    (1, 1)

    If no index is given, non-zero `labels` are processed:

    >>> ndimage.maximum_position(a, lbl)
    (2, 3)

    If there are no maxima, the position of the first element is returned:

    >>> ndimage.maximum_position(a, lbl, 2)
    (0, 2)

    """
    dims = numpy.array(numpy.asarray(input).shape)
    # see numpy.unravel_index to understand this line.
    dim_prod = numpy.cumprod([1] + list(dims[:0:-1]))[::-1]

    result = _select(input, labels, index, find_max_positions=True)[0]

    if numpy.isscalar(result):
        return tuple((result // dim_prod) % dims)

    return [tuple(v) for v in (result.reshape(-1, 1) // dim_prod) % dims]


def extrema(input, labels=None, index=None):
    """
    Calculate the minimums and maximums of the values of an array
    at labels, along with their positions.

    Parameters
    ----------
    input : ndarray
        N-D image data to process.
    labels : ndarray, optional
        Labels of features in input.
        If not None, must be same shape as `input`.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero `labels` are used.

    Returns
    -------
    minimums, maximums : int or ndarray
        Values of minimums and maximums in each feature.
    min_positions, max_positions : tuple or list of tuples
        Each tuple gives the N-D coordinates of the corresponding minimum
        or maximum.

    See Also
    --------
    maximum, minimum, maximum_position, minimum_position, center_of_mass

    Examples
    --------
    >>> a = np.array([[1, 2, 0, 0],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> from scipy import ndimage
    >>> ndimage.extrema(a)
    (0, 9, (0, 2), (3, 0))

    Features to process can be specified using `labels` and `index`:

    >>> lbl, nlbl = ndimage.label(a)
    >>> ndimage.extrema(a, lbl, index=np.arange(1, nlbl+1))
    (array([1, 4, 3]),
     array([5, 7, 9]),
     [(0, 0), (1, 3), (3, 1)],
     [(1, 0), (2, 3), (3, 0)])

    If no index is given, non-zero `labels` are processed:

    >>> ndimage.extrema(a, lbl)
    (1, 9, (0, 0), (3, 0))

    """
    dims = numpy.array(numpy.asarray(input).shape)
    # see numpy.unravel_index to understand this line.
    dim_prod = numpy.cumprod([1] + list(dims[:0:-1]))[::-1]

    minimums, min_positions, maximums, max_positions = _select(input, labels,
                                                               index,
                                                               find_min=True,
                                                               find_max=True,
                                                               find_min_positions=True,
                                                               find_max_positions=True)

    if numpy.isscalar(minimums):
        return (minimums, maximums, tuple((min_positions // dim_prod) % dims),
                tuple((max_positions // dim_prod) % dims))

    min_positions = [tuple(v) for v in (min_positions.reshape(-1, 1) // dim_prod) % dims]
    max_positions = [tuple(v) for v in (max_positions.reshape(-1, 1) // dim_prod) % dims]

    return minimums, maximums, min_positions, max_positions


def center_of_mass(input, labels=None, index=None):
    """
    Calculate the center of mass of the values of an array at labels.

    Parameters
    ----------
    input : ndarray
        Data from which to calculate center-of-mass. The masses can either
        be positive or negative.
    labels : ndarray, optional
        Labels for objects in `input`, as generated by `ndimage.label`.
        Only used with `index`. Dimensions must be the same as `input`.
    index : int or sequence of ints, optional
        Labels for which to calculate centers-of-mass. If not specified,
        all labels greater than zero are used. Only used with `labels`.

    Returns
    -------
    center_of_mass : tuple, or list of tuples
        Coordinates of centers-of-mass.

    Examples
    --------
    >>> a = np.array(([0,0,0,0],
    ...               [0,1,1,0],
    ...               [0,1,1,0],
    ...               [0,1,1,0]))
    >>> from scipy import ndimage
    >>> ndimage.measurements.center_of_mass(a)
    (2.0, 1.5)

    Calculation of multiple objects in an image

    >>> b = np.array(([0,1,1,0],
    ...               [0,1,0,0],
    ...               [0,0,0,0],
    ...               [0,0,1,1],
    ...               [0,0,1,1]))
    >>> lbl = ndimage.label(b)[0]
    >>> ndimage.measurements.center_of_mass(b, lbl, [1,2])
    [(0.33333333333333331, 1.3333333333333333), (3.5, 2.5)]

    Negative masses are also accepted, which can occur for example when
    bias is removed from measured data due to random noise.

    >>> c = np.array(([-1,0,0,0],
    ...               [0,-1,-1,0],
    ...               [0,1,-1,0],
    ...               [0,1,1,0]))
    >>> ndimage.measurements.center_of_mass(c)
    (-4.0, 1.0)

    If there are division by zero issues, the function does not raise an
    error but rather issues a RuntimeWarning before returning inf and/or NaN.

    >>> d = np.array([-1, 1])
    >>> ndimage.measurements.center_of_mass(d)
    (inf,)
    """
    normalizer = sum(input, labels, index)
    grids = numpy.ogrid[[slice(0, i) for i in input.shape]]

    results = [sum(input * grids[dir].astype(float), labels, index) / normalizer
               for dir in range(input.ndim)]

    if numpy.isscalar(results[0]):
        return tuple(results)

    return [tuple(v) for v in numpy.array(results).T]


def histogram(input, min, max, bins, labels=None, index=None):
    """
    Calculate the histogram of the values of an array, optionally at labels.

    Histogram calculates the frequency of values in an array within bins
    determined by `min`, `max`, and `bins`. The `labels` and `index`
    keywords can limit the scope of the histogram to specified sub-regions
    within the array.

    Parameters
    ----------
    input : array_like
        Data for which to calculate histogram.
    min, max : int
        Minimum and maximum values of range of histogram bins.
    bins : int
        Number of bins.
    labels : array_like, optional
        Labels for objects in `input`.
        If not None, must be same shape as `input`.
    index : int or sequence of ints, optional
        Label or labels for which to calculate histogram. If None, all values
        where label is greater than zero are used

    Returns
    -------
    hist : ndarray
        Histogram counts.

    Examples
    --------
    >>> a = np.array([[ 0.    ,  0.2146,  0.5962,  0.    ],
    ...               [ 0.    ,  0.7778,  0.    ,  0.    ],
    ...               [ 0.    ,  0.    ,  0.    ,  0.    ],
    ...               [ 0.    ,  0.    ,  0.7181,  0.2787],
    ...               [ 0.    ,  0.    ,  0.6573,  0.3094]])
    >>> from scipy import ndimage
    >>> ndimage.measurements.histogram(a, 0, 1, 10)
    array([13,  0,  2,  1,  0,  1,  1,  2,  0,  0])

    With labels and no indices, non-zero elements are counted:

    >>> lbl, nlbl = ndimage.label(a)
    >>> ndimage.measurements.histogram(a, 0, 1, 10, lbl)
    array([0, 0, 2, 1, 0, 1, 1, 2, 0, 0])

    Indices can be used to count only certain objects:

    >>> ndimage.measurements.histogram(a, 0, 1, 10, lbl, 2)
    array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])

    """
    _bins = numpy.linspace(min, max, bins + 1)

    def _hist(vals):
        return numpy.histogram(vals, _bins)[0]

    return labeled_comprehension(input, labels, index, _hist, object, None,
                                 pass_positions=False)


def watershed_ift(input, markers, structure=None, output=None):
    """
    Apply watershed from markers using image foresting transform algorithm.

    Parameters
    ----------
    input : array_like
        Input.
    markers : array_like
        Markers are points within each watershed that form the beginning
        of the process. Negative markers are considered background markers
        which are processed after the other markers.
    structure : structure element, optional
        A structuring element defining the connectivity of the object can be
        provided. If None, an element is generated with a squared
        connectivity equal to one.
    output : ndarray, optional
        An output array can optionally be provided. The same shape as input.

    Returns
    -------
    watershed_ift : ndarray
        Output.  Same shape as `input`.

    References
    ----------
    .. [1] A.X. Falcao, J. Stolfi and R. de Alencar Lotufo, "The image
           foresting transform: theory, algorithms, and applications",
           Pattern Analysis and Machine Intelligence, vol. 26, pp. 19-29, 2004.

    """
    input = numpy.asarray(input)
    if input.dtype.type not in [numpy.uint8, numpy.uint16]:
        raise TypeError('only 8 and 16 unsigned inputs are supported')

    if structure is None:
        structure = morphology.generate_binary_structure(input.ndim, 1)
    structure = numpy.asarray(structure, dtype=bool)
    if structure.ndim != input.ndim:
        raise RuntimeError('structure and input must have equal rank')
    for ii in structure.shape:
        if ii != 3:
            raise RuntimeError('structure dimensions must be equal to 3')

    if not structure.flags.contiguous:
        structure = structure.copy()
    markers = numpy.asarray(markers)
    if input.shape != markers.shape:
        raise RuntimeError('input and markers must have equal shape')

    integral_types = [numpy.int0,
                      numpy.int8,
                      numpy.int16,
                      numpy.int32,
                      numpy.int_,
                      numpy.int64,
                      numpy.intc,
                      numpy.intp]

    if markers.dtype.type not in integral_types:
        raise RuntimeError('marker should be of integer type')

    if isinstance(output, numpy.ndarray):
        if output.dtype.type not in integral_types:
            raise RuntimeError('output should be of integer type')
    else:
        output = markers.dtype

    output = _ni_support._get_output(output, input)
    _nd_image.watershed_ift(input, markers, structure, output)
    return output
