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

import warnings
import operator

import numpy
from . import _ni_support
from . import _nd_image
from . import filters

__all__ = ['iterate_structure', 'generate_binary_structure', 'binary_erosion',
           'binary_dilation', 'binary_opening', 'binary_closing',
           'binary_hit_or_miss', 'binary_propagation', 'binary_fill_holes',
           'grey_erosion', 'grey_dilation', 'grey_opening', 'grey_closing',
           'morphological_gradient', 'morphological_laplace', 'white_tophat',
           'black_tophat', 'distance_transform_bf', 'distance_transform_cdt',
           'distance_transform_edt']


def _center_is_true(structure, origin):
    structure = numpy.array(structure)
    coor = tuple([oo + ss // 2 for ss, oo in zip(structure.shape,
                                                 origin)])
    return bool(structure[coor])


def iterate_structure(structure, iterations, origin=None):
    """
    Iterate a structure by dilating it with itself.

    Parameters
    ----------
    structure : array_like
       Structuring element (an array of bools, for example), to be dilated with
       itself.
    iterations : int
       number of dilations performed on the structure with itself
    origin : optional
        If origin is None, only the iterated structure is returned. If
        not, a tuple of the iterated structure and the modified origin is
        returned.

    Returns
    -------
    iterate_structure : ndarray of bools
        A new structuring element obtained by dilating `structure`
        (`iterations` - 1) times with itself.

    See also
    --------
    generate_binary_structure

    Examples
    --------
    >>> from scipy import ndimage
    >>> struct = ndimage.generate_binary_structure(2, 1)
    >>> struct.astype(int)
    array([[0, 1, 0],
           [1, 1, 1],
           [0, 1, 0]])
    >>> ndimage.iterate_structure(struct, 2).astype(int)
    array([[0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [1, 1, 1, 1, 1],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0]])
    >>> ndimage.iterate_structure(struct, 3).astype(int)
    array([[0, 0, 0, 1, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 0]])

    """
    structure = numpy.asarray(structure)
    if iterations < 2:
        return structure.copy()
    ni = iterations - 1
    shape = [ii + ni * (ii - 1) for ii in structure.shape]
    pos = [ni * (structure.shape[ii] // 2) for ii in range(len(shape))]
    slc = tuple(slice(pos[ii], pos[ii] + structure.shape[ii], None)
                for ii in range(len(shape)))
    out = numpy.zeros(shape, bool)
    out[slc] = structure != 0
    out = binary_dilation(out, structure, iterations=ni)
    if origin is None:
        return out
    else:
        origin = _ni_support._normalize_sequence(origin, structure.ndim)
        origin = [iterations * o for o in origin]
        return out, origin


def generate_binary_structure(rank, connectivity):
    """
    Generate a binary structure for binary morphological operations.

    Parameters
    ----------
    rank : int
         Number of dimensions of the array to which the structuring element
         will be applied, as returned by `np.ndim`.
    connectivity : int
         `connectivity` determines which elements of the output array belong
         to the structure, i.e., are considered as neighbors of the central
         element. Elements up to a squared distance of `connectivity` from
         the center are considered neighbors. `connectivity` may range from 1
         (no diagonal elements are neighbors) to `rank` (all elements are
         neighbors).

    Returns
    -------
    output : ndarray of bools
         Structuring element which may be used for binary morphological
         operations, with `rank` dimensions and all dimensions equal to 3.

    See also
    --------
    iterate_structure, binary_dilation, binary_erosion

    Notes
    -----
    `generate_binary_structure` can only create structuring elements with
    dimensions equal to 3, i.e., minimal dimensions. For larger structuring
    elements, that are useful e.g., for eroding large objects, one may either
    use `iterate_structure`, or create directly custom arrays with
    numpy functions such as `numpy.ones`.

    Examples
    --------
    >>> from scipy import ndimage
    >>> struct = ndimage.generate_binary_structure(2, 1)
    >>> struct
    array([[False,  True, False],
           [ True,  True,  True],
           [False,  True, False]], dtype=bool)
    >>> a = np.zeros((5,5))
    >>> a[2, 2] = 1
    >>> a
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> b = ndimage.binary_dilation(a, structure=struct).astype(a.dtype)
    >>> b
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> ndimage.binary_dilation(b, structure=struct).astype(a.dtype)
    array([[ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.]])
    >>> struct = ndimage.generate_binary_structure(2, 2)
    >>> struct
    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)
    >>> struct = ndimage.generate_binary_structure(3, 1)
    >>> struct # no diagonal elements
    array([[[False, False, False],
            [False,  True, False],
            [False, False, False]],
           [[False,  True, False],
            [ True,  True,  True],
            [False,  True, False]],
           [[False, False, False],
            [False,  True, False],
            [False, False, False]]], dtype=bool)

    """
    if connectivity < 1:
        connectivity = 1
    if rank < 1:
        return numpy.array(True, dtype=bool)
    output = numpy.fabs(numpy.indices([3] * rank) - 1)
    output = numpy.add.reduce(output, 0)
    return output <= connectivity


def _binary_erosion(input, structure, iterations, mask, output,
                    border_value, origin, invert, brute_force):
    try:
        iterations = operator.index(iterations)
    except TypeError as e:
        raise TypeError('iterations parameter should be an integer') from e

    input = numpy.asarray(input)
    if numpy.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    if structure is None:
        structure = generate_binary_structure(input.ndim, 1)
    else:
        structure = numpy.asarray(structure, dtype=bool)
    if structure.ndim != input.ndim:
        raise RuntimeError('structure and input must have same dimensionality')
    if not structure.flags.contiguous:
        structure = structure.copy()
    if numpy.prod(structure.shape, axis=0) < 1:
        raise RuntimeError('structure must not be empty')
    if mask is not None:
        mask = numpy.asarray(mask)
        if mask.shape != input.shape:
            raise RuntimeError('mask and input must have equal sizes')
    origin = _ni_support._normalize_sequence(origin, input.ndim)
    cit = _center_is_true(structure, origin)
    if isinstance(output, numpy.ndarray):
        if numpy.iscomplexobj(output):
            raise TypeError('Complex output type not supported')
    else:
        output = bool
    output = _ni_support._get_output(output, input)
    temp_needed = numpy.may_share_memory(input, output)
    if temp_needed:
        # input and output arrays cannot share memory
        temp = output
        output = _ni_support._get_output(output.dtype, input)
    if iterations == 1:
        _nd_image.binary_erosion(input, structure, mask, output,
                                 border_value, origin, invert, cit, 0)
        return output
    elif cit and not brute_force:
        changed, coordinate_list = _nd_image.binary_erosion(
            input, structure, mask, output,
            border_value, origin, invert, cit, 1)
        structure = structure[tuple([slice(None, None, -1)] *
                                    structure.ndim)]
        for ii in range(len(origin)):
            origin[ii] = -origin[ii]
            if not structure.shape[ii] & 1:
                origin[ii] -= 1
        if mask is not None:
            mask = numpy.asarray(mask, dtype=numpy.int8)
        if not structure.flags.contiguous:
            structure = structure.copy()
        _nd_image.binary_erosion2(output, structure, mask, iterations - 1,
                                  origin, invert, coordinate_list)
    else:
        tmp_in = numpy.empty_like(input, dtype=bool)
        tmp_out = output
        if iterations >= 1 and not iterations & 1:
            tmp_in, tmp_out = tmp_out, tmp_in
        changed = _nd_image.binary_erosion(
            input, structure, mask, tmp_out,
            border_value, origin, invert, cit, 0)
        ii = 1
        while ii < iterations or (iterations < 1 and changed):
            tmp_in, tmp_out = tmp_out, tmp_in
            changed = _nd_image.binary_erosion(
                tmp_in, structure, mask, tmp_out,
                border_value, origin, invert, cit, 0)
            ii += 1
    if temp_needed:
        temp[...] = output
        output = temp
    return output


def binary_erosion(input, structure=None, iterations=1, mask=None, output=None,
                   border_value=0, origin=0, brute_force=False):
    """
    Multidimensional binary erosion with a given structuring element.

    Binary erosion is a mathematical morphology operation used for image
    processing.

    Parameters
    ----------
    input : array_like
        Binary image to be eroded. Non-zero (True) elements form
        the subset to be eroded.
    structure : array_like, optional
        Structuring element used for the erosion. Non-zero elements are
        considered True. If no structuring element is provided, an element
        is generated with a square connectivity equal to one.
    iterations : int, optional
        The erosion is repeated `iterations` times (one, by default).
        If iterations is less than 1, the erosion is repeated until the
        result does not change anymore.
    mask : array_like, optional
        If a mask is given, only those elements with a True value at
        the corresponding mask element are modified at each iteration.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
    border_value : int (cast to 0 or 1), optional
        Value at the border in the output array.
    origin : int or tuple of ints, optional
        Placement of the filter, by default 0.
    brute_force : boolean, optional
        Memory condition: if False, only the pixels whose value was changed in
        the last iteration are tracked as candidates to be updated (eroded) in
        the current iteration; if True all pixels are considered as candidates
        for erosion, regardless of what happened in the previous iteration.
        False by default.

    Returns
    -------
    binary_erosion : ndarray of bools
        Erosion of the input by the structuring element.

    See also
    --------
    grey_erosion, binary_dilation, binary_closing, binary_opening,
    generate_binary_structure

    Notes
    -----
    Erosion [1]_ is a mathematical morphology operation [2]_ that uses a
    structuring element for shrinking the shapes in an image. The binary
    erosion of an image by a structuring element is the locus of the points
    where a superimposition of the structuring element centered on the point
    is entirely contained in the set of non-zero elements of the image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Erosion_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> from scipy import ndimage
    >>> a = np.zeros((7,7), dtype=int)
    >>> a[1:6, 2:5] = 1
    >>> a
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.binary_erosion(a).astype(a.dtype)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> #Erosion removes objects smaller than the structure
    >>> ndimage.binary_erosion(a, structure=np.ones((5,5))).astype(a.dtype)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])

    """
    return _binary_erosion(input, structure, iterations, mask,
                           output, border_value, origin, 0, brute_force)


def binary_dilation(input, structure=None, iterations=1, mask=None,
                    output=None, border_value=0, origin=0,
                    brute_force=False):
    """
    Multidimensional binary dilation with the given structuring element.

    Parameters
    ----------
    input : array_like
        Binary array_like to be dilated. Non-zero (True) elements form
        the subset to be dilated.
    structure : array_like, optional
        Structuring element used for the dilation. Non-zero elements are
        considered True. If no structuring element is provided an element
        is generated with a square connectivity equal to one.
    iterations : int, optional
        The dilation is repeated `iterations` times (one, by default).
        If iterations is less than 1, the dilation is repeated until the
        result does not change anymore. Only an integer of iterations is
        accepted.
    mask : array_like, optional
        If a mask is given, only those elements with a True value at
        the corresponding mask element are modified at each iteration.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
    border_value : int (cast to 0 or 1), optional
        Value at the border in the output array.
    origin : int or tuple of ints, optional
        Placement of the filter, by default 0.
    brute_force : boolean, optional
        Memory condition: if False, only the pixels whose value was changed in
        the last iteration are tracked as candidates to be updated (dilated)
        in the current iteration; if True all pixels are considered as
        candidates for dilation, regardless of what happened in the previous
        iteration. False by default.

    Returns
    -------
    binary_dilation : ndarray of bools
        Dilation of the input by the structuring element.

    See also
    --------
    grey_dilation, binary_erosion, binary_closing, binary_opening,
    generate_binary_structure

    Notes
    -----
    Dilation [1]_ is a mathematical morphology operation [2]_ that uses a
    structuring element for expanding the shapes in an image. The binary
    dilation of an image by a structuring element is the locus of the points
    covered by the structuring element, when its center lies within the
    non-zero points of the image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Dilation_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> from scipy import ndimage
    >>> a = np.zeros((5, 5))
    >>> a[2, 2] = 1
    >>> a
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> ndimage.binary_dilation(a)
    array([[False, False, False, False, False],
           [False, False,  True, False, False],
           [False,  True,  True,  True, False],
           [False, False,  True, False, False],
           [False, False, False, False, False]], dtype=bool)
    >>> ndimage.binary_dilation(a).astype(a.dtype)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> # 3x3 structuring element with connectivity 1, used by default
    >>> struct1 = ndimage.generate_binary_structure(2, 1)
    >>> struct1
    array([[False,  True, False],
           [ True,  True,  True],
           [False,  True, False]], dtype=bool)
    >>> # 3x3 structuring element with connectivity 2
    >>> struct2 = ndimage.generate_binary_structure(2, 2)
    >>> struct2
    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)
    >>> ndimage.binary_dilation(a, structure=struct1).astype(a.dtype)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> ndimage.binary_dilation(a, structure=struct2).astype(a.dtype)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> ndimage.binary_dilation(a, structure=struct1,\\
    ... iterations=2).astype(a.dtype)
    array([[ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.]])

    """
    input = numpy.asarray(input)
    if structure is None:
        structure = generate_binary_structure(input.ndim, 1)
    origin = _ni_support._normalize_sequence(origin, input.ndim)
    structure = numpy.asarray(structure)
    structure = structure[tuple([slice(None, None, -1)] *
                                structure.ndim)]
    for ii in range(len(origin)):
        origin[ii] = -origin[ii]
        if not structure.shape[ii] & 1:
            origin[ii] -= 1

    return _binary_erosion(input, structure, iterations, mask,
                           output, border_value, origin, 1, brute_force)


def binary_opening(input, structure=None, iterations=1, output=None,
                   origin=0, mask=None, border_value=0, brute_force=False):
    """
    Multidimensional binary opening with the given structuring element.

    The *opening* of an input image by a structuring element is the
    *dilation* of the *erosion* of the image by the structuring element.

    Parameters
    ----------
    input : array_like
        Binary array_like to be opened. Non-zero (True) elements form
        the subset to be opened.
    structure : array_like, optional
        Structuring element used for the opening. Non-zero elements are
        considered True. If no structuring element is provided an element
        is generated with a square connectivity equal to one (i.e., only
        nearest neighbors are connected to the center, diagonally-connected
        elements are not considered neighbors).
    iterations : int, optional
        The erosion step of the opening, then the dilation step are each
        repeated `iterations` times (one, by default). If `iterations` is
        less than 1, each operation is repeated until the result does
        not change anymore. Only an integer of iterations is accepted.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
    origin : int or tuple of ints, optional
        Placement of the filter, by default 0.
    mask : array_like, optional
        If a mask is given, only those elements with a True value at
        the corresponding mask element are modified at each iteration.

        .. versionadded:: 1.1.0
    border_value : int (cast to 0 or 1), optional
        Value at the border in the output array.

        .. versionadded:: 1.1.0
    brute_force : boolean, optional
        Memory condition: if False, only the pixels whose value was changed in
        the last iteration are tracked as candidates to be updated in the
        current iteration; if true all pixels are considered as candidates for
        update, regardless of what happened in the previous iteration.
        False by default.

        .. versionadded:: 1.1.0

    Returns
    -------
    binary_opening : ndarray of bools
        Opening of the input by the structuring element.

    See also
    --------
    grey_opening, binary_closing, binary_erosion, binary_dilation,
    generate_binary_structure

    Notes
    -----
    *Opening* [1]_ is a mathematical morphology operation [2]_ that
    consists in the succession of an erosion and a dilation of the
    input with the same structuring element. Opening, therefore, removes
    objects smaller than the structuring element.

    Together with *closing* (`binary_closing`), opening can be used for
    noise removal.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Opening_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> from scipy import ndimage
    >>> a = np.zeros((5,5), dtype=int)
    >>> a[1:4, 1:4] = 1; a[4, 4] = 1
    >>> a
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 1]])
    >>> # Opening removes small objects
    >>> ndimage.binary_opening(a, structure=np.ones((3,3))).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]])
    >>> # Opening can also smooth corners
    >>> ndimage.binary_opening(a).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> # Opening is the dilation of the erosion of the input
    >>> ndimage.binary_erosion(a).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> ndimage.binary_dilation(ndimage.binary_erosion(a)).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]])

    """
    input = numpy.asarray(input)
    if structure is None:
        rank = input.ndim
        structure = generate_binary_structure(rank, 1)

    tmp = binary_erosion(input, structure, iterations, mask, None,
                         border_value, origin, brute_force)
    return binary_dilation(tmp, structure, iterations, mask, output,
                           border_value, origin, brute_force)


def binary_closing(input, structure=None, iterations=1, output=None,
                   origin=0, mask=None, border_value=0, brute_force=False):
    """
    Multidimensional binary closing with the given structuring element.

    The *closing* of an input image by a structuring element is the
    *erosion* of the *dilation* of the image by the structuring element.

    Parameters
    ----------
    input : array_like
        Binary array_like to be closed. Non-zero (True) elements form
        the subset to be closed.
    structure : array_like, optional
        Structuring element used for the closing. Non-zero elements are
        considered True. If no structuring element is provided an element
        is generated with a square connectivity equal to one (i.e., only
        nearest neighbors are connected to the center, diagonally-connected
        elements are not considered neighbors).
    iterations : int, optional
        The dilation step of the closing, then the erosion step are each
        repeated `iterations` times (one, by default). If iterations is
        less than 1, each operations is repeated until the result does
        not change anymore. Only an integer of iterations is accepted.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
    origin : int or tuple of ints, optional
        Placement of the filter, by default 0.
    mask : array_like, optional
        If a mask is given, only those elements with a True value at
        the corresponding mask element are modified at each iteration.

        .. versionadded:: 1.1.0
    border_value : int (cast to 0 or 1), optional
        Value at the border in the output array.

        .. versionadded:: 1.1.0
    brute_force : boolean, optional
        Memory condition: if False, only the pixels whose value was changed in
        the last iteration are tracked as candidates to be updated in the
        current iteration; if true al pixels are considered as candidates for
        update, regardless of what happened in the previous iteration.
        False by default.

        .. versionadded:: 1.1.0

    Returns
    -------
    binary_closing : ndarray of bools
        Closing of the input by the structuring element.

    See also
    --------
    grey_closing, binary_opening, binary_dilation, binary_erosion,
    generate_binary_structure

    Notes
    -----
    *Closing* [1]_ is a mathematical morphology operation [2]_ that
    consists in the succession of a dilation and an erosion of the
    input with the same structuring element. Closing therefore fills
    holes smaller than the structuring element.

    Together with *opening* (`binary_opening`), closing can be used for
    noise removal.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Closing_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> from scipy import ndimage
    >>> a = np.zeros((5,5), dtype=int)
    >>> a[1:-1, 1:-1] = 1; a[2,2] = 0
    >>> a
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 0, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]])
    >>> # Closing removes small holes
    >>> ndimage.binary_closing(a).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]])
    >>> # Closing is the erosion of the dilation of the input
    >>> ndimage.binary_dilation(a).astype(int)
    array([[0, 1, 1, 1, 0],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [0, 1, 1, 1, 0]])
    >>> ndimage.binary_erosion(ndimage.binary_dilation(a)).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]])


    >>> a = np.zeros((7,7), dtype=int)
    >>> a[1:6, 2:5] = 1; a[1:3,3] = 0
    >>> a
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> # In addition to removing holes, closing can also
    >>> # coarsen boundaries with fine hollows.
    >>> ndimage.binary_closing(a).astype(int)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.binary_closing(a, structure=np.ones((2,2))).astype(int)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])

    """
    input = numpy.asarray(input)
    if structure is None:
        rank = input.ndim
        structure = generate_binary_structure(rank, 1)

    tmp = binary_dilation(input, structure, iterations, mask, None,
                          border_value, origin, brute_force)
    return binary_erosion(tmp, structure, iterations, mask, output,
                          border_value, origin, brute_force)


def binary_hit_or_miss(input, structure1=None, structure2=None,
                       output=None, origin1=0, origin2=None):
    """
    Multidimensional binary hit-or-miss transform.

    The hit-or-miss transform finds the locations of a given pattern
    inside the input image.

    Parameters
    ----------
    input : array_like (cast to booleans)
        Binary image where a pattern is to be detected.
    structure1 : array_like (cast to booleans), optional
        Part of the structuring element to be fitted to the foreground
        (non-zero elements) of `input`. If no value is provided, a
        structure of square connectivity 1 is chosen.
    structure2 : array_like (cast to booleans), optional
        Second part of the structuring element that has to miss completely
        the foreground. If no value is provided, the complementary of
        `structure1` is taken.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
    origin1 : int or tuple of ints, optional
        Placement of the first part of the structuring element `structure1`,
        by default 0 for a centered structure.
    origin2 : int or tuple of ints, optional
        Placement of the second part of the structuring element `structure2`,
        by default 0 for a centered structure. If a value is provided for
        `origin1` and not for `origin2`, then `origin2` is set to `origin1`.

    Returns
    -------
    binary_hit_or_miss : ndarray
        Hit-or-miss transform of `input` with the given structuring
        element (`structure1`, `structure2`).

    See also
    --------
    binary_erosion

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Hit-or-miss_transform

    Examples
    --------
    >>> from scipy import ndimage
    >>> a = np.zeros((7,7), dtype=int)
    >>> a[1, 1] = 1; a[2:4, 2:4] = 1; a[4:6, 4:6] = 1
    >>> a
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 0],
           [0, 0, 0, 0, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> structure1 = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]])
    >>> structure1
    array([[1, 0, 0],
           [0, 1, 1],
           [0, 1, 1]])
    >>> # Find the matches of structure1 in the array a
    >>> ndimage.binary_hit_or_miss(a, structure1=structure1).astype(int)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> # Change the origin of the filter
    >>> # origin1=1 is equivalent to origin1=(1,1) here
    >>> ndimage.binary_hit_or_miss(a, structure1=structure1,\\
    ... origin1=1).astype(int)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0]])

    """
    input = numpy.asarray(input)
    if structure1 is None:
        structure1 = generate_binary_structure(input.ndim, 1)
    if structure2 is None:
        structure2 = numpy.logical_not(structure1)
    origin1 = _ni_support._normalize_sequence(origin1, input.ndim)
    if origin2 is None:
        origin2 = origin1
    else:
        origin2 = _ni_support._normalize_sequence(origin2, input.ndim)

    tmp1 = _binary_erosion(input, structure1, 1, None, None, 0, origin1,
                           0, False)
    inplace = isinstance(output, numpy.ndarray)
    result = _binary_erosion(input, structure2, 1, None, output, 0,
                             origin2, 1, False)
    if inplace:
        numpy.logical_not(output, output)
        numpy.logical_and(tmp1, output, output)
    else:
        numpy.logical_not(result, result)
        return numpy.logical_and(tmp1, result)


def binary_propagation(input, structure=None, mask=None,
                       output=None, border_value=0, origin=0):
    """
    Multidimensional binary propagation with the given structuring element.

    Parameters
    ----------
    input : array_like
        Binary image to be propagated inside `mask`.
    structure : array_like, optional
        Structuring element used in the successive dilations. The output
        may depend on the structuring element, especially if `mask` has
        several connex components. If no structuring element is
        provided, an element is generated with a squared connectivity equal
        to one.
    mask : array_like, optional
        Binary mask defining the region into which `input` is allowed to
        propagate.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
    border_value : int (cast to 0 or 1), optional
        Value at the border in the output array.
    origin : int or tuple of ints, optional
        Placement of the filter, by default 0.

    Returns
    -------
    binary_propagation : ndarray
        Binary propagation of `input` inside `mask`.

    Notes
    -----
    This function is functionally equivalent to calling binary_dilation
    with the number of iterations less than one: iterative dilation until
    the result does not change anymore.

    The succession of an erosion and propagation inside the original image
    can be used instead of an *opening* for deleting small objects while
    keeping the contours of larger objects untouched.

    References
    ----------
    .. [1] http://cmm.ensmp.fr/~serra/cours/pdf/en/ch6en.pdf, slide 15.
    .. [2] I.T. Young, J.J. Gerbrands, and L.J. van Vliet, "Fundamentals of
        image processing", 1998
        ftp://qiftp.tudelft.nl/DIPimage/docs/FIP2.3.pdf

    Examples
    --------
    >>> from scipy import ndimage
    >>> input = np.zeros((8, 8), dtype=int)
    >>> input[2, 2] = 1
    >>> mask = np.zeros((8, 8), dtype=int)
    >>> mask[1:4, 1:4] = mask[4, 4]  = mask[6:8, 6:8] = 1
    >>> input
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]])
    >>> mask
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1]])
    >>> ndimage.binary_propagation(input, mask=mask).astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.binary_propagation(input, mask=mask,\\
    ... structure=np.ones((3,3))).astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]])

    >>> # Comparison between opening and erosion+propagation
    >>> a = np.zeros((6,6), dtype=int)
    >>> a[2:5, 2:5] = 1; a[0, 0] = 1; a[5, 5] = 1
    >>> a
    array([[1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 1]])
    >>> ndimage.binary_opening(a).astype(int)
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [0, 0, 1, 1, 1, 0],
           [0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0]])
    >>> b = ndimage.binary_erosion(a)
    >>> b.astype(int)
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0]])
    >>> ndimage.binary_propagation(b, mask=a).astype(int)
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0]])

    """
    return binary_dilation(input, structure, -1, mask, output,
                           border_value, origin)


def binary_fill_holes(input, structure=None, output=None, origin=0):
    """
    Fill the holes in binary objects.


    Parameters
    ----------
    input : array_like
        N-D binary array with holes to be filled
    structure : array_like, optional
        Structuring element used in the computation; large-size elements
        make computations faster but may miss holes separated from the
        background by thin regions. The default element (with a square
        connectivity equal to one) yields the intuitive result where all
        holes in the input have been filled.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
    origin : int, tuple of ints, optional
        Position of the structuring element.

    Returns
    -------
    out : ndarray
        Transformation of the initial image `input` where holes have been
        filled.

    See also
    --------
    binary_dilation, binary_propagation, label

    Notes
    -----
    The algorithm used in this function consists in invading the complementary
    of the shapes in `input` from the outer boundary of the image,
    using binary dilations. Holes are not connected to the boundary and are
    therefore not invaded. The result is the complementary subset of the
    invaded region.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Mathematical_morphology


    Examples
    --------
    >>> from scipy import ndimage
    >>> a = np.zeros((5, 5), dtype=int)
    >>> a[1:4, 1:4] = 1
    >>> a[2,2] = 0
    >>> a
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 0, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]])
    >>> ndimage.binary_fill_holes(a).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]])
    >>> # Too big structuring element
    >>> ndimage.binary_fill_holes(a, structure=np.ones((5,5))).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 0, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]])

    """
    mask = numpy.logical_not(input)
    tmp = numpy.zeros(mask.shape, bool)
    inplace = isinstance(output, numpy.ndarray)
    if inplace:
        binary_dilation(tmp, structure, -1, mask, output, 1, origin)
        numpy.logical_not(output, output)
    else:
        output = binary_dilation(tmp, structure, -1, mask, None, 1,
                                 origin)
        numpy.logical_not(output, output)
        return output


def grey_erosion(input, size=None, footprint=None, structure=None,
                 output=None, mode="reflect", cval=0.0, origin=0):
    """
    Calculate a greyscale erosion, using either a structuring element,
    or a footprint corresponding to a flat structuring element.

    Grayscale erosion is a mathematical morphology operation. For the
    simple case of a full and flat structuring element, it can be viewed
    as a minimum filter over a sliding window.

    Parameters
    ----------
    input : array_like
        Array over which the grayscale erosion is to be computed.
    size : tuple of ints
        Shape of a flat and full structuring element used for the grayscale
        erosion. Optional if `footprint` or `structure` is provided.
    footprint : array of ints, optional
        Positions of non-infinite elements of a flat structuring element
        used for the grayscale erosion. Non-zero values give the set of
        neighbors of the center over which the minimum is chosen.
    structure : array of ints, optional
        Structuring element used for the grayscale erosion. `structure`
        may be a non-flat structuring element.
    output : array, optional
        An array used for storing the output of the erosion may be provided.
    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.
    origin : scalar, optional
        The `origin` parameter controls the placement of the filter.
        Default 0

    Returns
    -------
    output : ndarray
        Grayscale erosion of `input`.

    See also
    --------
    binary_erosion, grey_dilation, grey_opening, grey_closing
    generate_binary_structure, minimum_filter

    Notes
    -----
    The grayscale erosion of an image input by a structuring element s defined
    over a domain E is given by:

    (input+s)(x) = min {input(y) - s(x-y), for y in E}

    In particular, for structuring elements defined as
    s(y) = 0 for y in E, the grayscale erosion computes the minimum of the
    input image inside a sliding window defined by E.

    Grayscale erosion [1]_ is a *mathematical morphology* operation [2]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Erosion_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> from scipy import ndimage
    >>> a = np.zeros((7,7), dtype=int)
    >>> a[1:6, 1:6] = 3
    >>> a[4,4] = 2; a[2,3] = 1
    >>> a
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 3, 3, 3, 3, 3, 0],
           [0, 3, 3, 1, 3, 3, 0],
           [0, 3, 3, 3, 3, 3, 0],
           [0, 3, 3, 3, 2, 3, 0],
           [0, 3, 3, 3, 3, 3, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.grey_erosion(a, size=(3,3))
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 3, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> footprint = ndimage.generate_binary_structure(2, 1)
    >>> footprint
    array([[False,  True, False],
           [ True,  True,  True],
           [False,  True, False]], dtype=bool)
    >>> # Diagonally-connected elements are not considered neighbors
    >>> ndimage.grey_erosion(a, size=(3,3), footprint=footprint)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 3, 1, 2, 0, 0],
           [0, 0, 3, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])

    """
    if size is None and footprint is None and structure is None:
        raise ValueError("size, footprint, or structure must be specified")

    return filters._min_or_max_filter(input, size, footprint, structure,
                                      output, mode, cval, origin, 1)


def grey_dilation(input, size=None, footprint=None, structure=None,
                  output=None, mode="reflect", cval=0.0, origin=0):
    """
    Calculate a greyscale dilation, using either a structuring element,
    or a footprint corresponding to a flat structuring element.

    Grayscale dilation is a mathematical morphology operation. For the
    simple case of a full and flat structuring element, it can be viewed
    as a maximum filter over a sliding window.

    Parameters
    ----------
    input : array_like
        Array over which the grayscale dilation is to be computed.
    size : tuple of ints
        Shape of a flat and full structuring element used for the grayscale
        dilation. Optional if `footprint` or `structure` is provided.
    footprint : array of ints, optional
        Positions of non-infinite elements of a flat structuring element
        used for the grayscale dilation. Non-zero values give the set of
        neighbors of the center over which the maximum is chosen.
    structure : array of ints, optional
        Structuring element used for the grayscale dilation. `structure`
        may be a non-flat structuring element.
    output : array, optional
        An array used for storing the output of the dilation may be provided.
    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.
    origin : scalar, optional
        The `origin` parameter controls the placement of the filter.
        Default 0

    Returns
    -------
    grey_dilation : ndarray
        Grayscale dilation of `input`.

    See also
    --------
    binary_dilation, grey_erosion, grey_closing, grey_opening
    generate_binary_structure, maximum_filter

    Notes
    -----
    The grayscale dilation of an image input by a structuring element s defined
    over a domain E is given by:

    (input+s)(x) = max {input(y) + s(x-y), for y in E}

    In particular, for structuring elements defined as
    s(y) = 0 for y in E, the grayscale dilation computes the maximum of the
    input image inside a sliding window defined by E.

    Grayscale dilation [1]_ is a *mathematical morphology* operation [2]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Dilation_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> from scipy import ndimage
    >>> a = np.zeros((7,7), dtype=int)
    >>> a[2:5, 2:5] = 1
    >>> a[4,4] = 2; a[2,3] = 3
    >>> a
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 3, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.grey_dilation(a, size=(3,3))
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 3, 3, 3, 1, 0],
           [0, 1, 3, 3, 3, 1, 0],
           [0, 1, 3, 3, 3, 2, 0],
           [0, 1, 1, 2, 2, 2, 0],
           [0, 1, 1, 2, 2, 2, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.grey_dilation(a, footprint=np.ones((3,3)))
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 3, 3, 3, 1, 0],
           [0, 1, 3, 3, 3, 1, 0],
           [0, 1, 3, 3, 3, 2, 0],
           [0, 1, 1, 2, 2, 2, 0],
           [0, 1, 1, 2, 2, 2, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> s = ndimage.generate_binary_structure(2,1)
    >>> s
    array([[False,  True, False],
           [ True,  True,  True],
           [False,  True, False]], dtype=bool)
    >>> ndimage.grey_dilation(a, footprint=s)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 3, 1, 0, 0],
           [0, 1, 3, 3, 3, 1, 0],
           [0, 1, 1, 3, 2, 1, 0],
           [0, 1, 1, 2, 2, 2, 0],
           [0, 0, 1, 1, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.grey_dilation(a, size=(3,3), structure=np.ones((3,3)))
    array([[1, 1, 1, 1, 1, 1, 1],
           [1, 2, 4, 4, 4, 2, 1],
           [1, 2, 4, 4, 4, 2, 1],
           [1, 2, 4, 4, 4, 3, 1],
           [1, 2, 2, 3, 3, 3, 1],
           [1, 2, 2, 3, 3, 3, 1],
           [1, 1, 1, 1, 1, 1, 1]])

    """
    if size is None and footprint is None and structure is None:
        raise ValueError("size, footprint, or structure must be specified")
    if structure is not None:
        structure = numpy.asarray(structure)
        structure = structure[tuple([slice(None, None, -1)] *
                                    structure.ndim)]
    if footprint is not None:
        footprint = numpy.asarray(footprint)
        footprint = footprint[tuple([slice(None, None, -1)] *
                                    footprint.ndim)]

    input = numpy.asarray(input)
    origin = _ni_support._normalize_sequence(origin, input.ndim)
    for ii in range(len(origin)):
        origin[ii] = -origin[ii]
        if footprint is not None:
            sz = footprint.shape[ii]
        elif structure is not None:
            sz = structure.shape[ii]
        elif numpy.isscalar(size):
            sz = size
        else:
            sz = size[ii]
        if not sz & 1:
            origin[ii] -= 1

    return filters._min_or_max_filter(input, size, footprint, structure,
                                      output, mode, cval, origin, 0)


def grey_opening(input, size=None, footprint=None, structure=None,
                 output=None, mode="reflect", cval=0.0, origin=0):
    """
    Multidimensional grayscale opening.

    A grayscale opening consists in the succession of a grayscale erosion,
    and a grayscale dilation.

    Parameters
    ----------
    input : array_like
        Array over which the grayscale opening is to be computed.
    size : tuple of ints
        Shape of a flat and full structuring element used for the grayscale
        opening. Optional if `footprint` or `structure` is provided.
    footprint : array of ints, optional
        Positions of non-infinite elements of a flat structuring element
        used for the grayscale opening.
    structure : array of ints, optional
        Structuring element used for the grayscale opening. `structure`
        may be a non-flat structuring element.
    output : array, optional
        An array used for storing the output of the opening may be provided.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.
    origin : scalar, optional
        The `origin` parameter controls the placement of the filter.
        Default 0

    Returns
    -------
    grey_opening : ndarray
        Result of the grayscale opening of `input` with `structure`.

    See also
    --------
    binary_opening, grey_dilation, grey_erosion, grey_closing
    generate_binary_structure

    Notes
    -----
    The action of a grayscale opening with a flat structuring element amounts
    to smoothen high local maxima, whereas binary opening erases small objects.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> from scipy import ndimage
    >>> a = np.arange(36).reshape((6,6))
    >>> a[3, 3] = 50
    >>> a
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20, 50, 22, 23],
           [24, 25, 26, 27, 28, 29],
           [30, 31, 32, 33, 34, 35]])
    >>> ndimage.grey_opening(a, size=(3,3))
    array([[ 0,  1,  2,  3,  4,  4],
           [ 6,  7,  8,  9, 10, 10],
           [12, 13, 14, 15, 16, 16],
           [18, 19, 20, 22, 22, 22],
           [24, 25, 26, 27, 28, 28],
           [24, 25, 26, 27, 28, 28]])
    >>> # Note that the local maximum a[3,3] has disappeared

    """
    if (size is not None) and (footprint is not None):
        warnings.warn("ignoring size because footprint is set", UserWarning, stacklevel=2)
    tmp = grey_erosion(input, size, footprint, structure, None, mode,
                       cval, origin)
    return grey_dilation(tmp, size, footprint, structure, output, mode,
                         cval, origin)


def grey_closing(input, size=None, footprint=None, structure=None,
                 output=None, mode="reflect", cval=0.0, origin=0):
    """
    Multidimensional grayscale closing.

    A grayscale closing consists in the succession of a grayscale dilation,
    and a grayscale erosion.

    Parameters
    ----------
    input : array_like
        Array over which the grayscale closing is to be computed.
    size : tuple of ints
        Shape of a flat and full structuring element used for the grayscale
        closing. Optional if `footprint` or `structure` is provided.
    footprint : array of ints, optional
        Positions of non-infinite elements of a flat structuring element
        used for the grayscale closing.
    structure : array of ints, optional
        Structuring element used for the grayscale closing. `structure`
        may be a non-flat structuring element.
    output : array, optional
        An array used for storing the output of the closing may be provided.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.
    origin : scalar, optional
        The `origin` parameter controls the placement of the filter.
        Default 0

    Returns
    -------
    grey_closing : ndarray
        Result of the grayscale closing of `input` with `structure`.

    See also
    --------
    binary_closing, grey_dilation, grey_erosion, grey_opening,
    generate_binary_structure

    Notes
    -----
    The action of a grayscale closing with a flat structuring element amounts
    to smoothen deep local minima, whereas binary closing fills small holes.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> from scipy import ndimage
    >>> a = np.arange(36).reshape((6,6))
    >>> a[3,3] = 0
    >>> a
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20,  0, 22, 23],
           [24, 25, 26, 27, 28, 29],
           [30, 31, 32, 33, 34, 35]])
    >>> ndimage.grey_closing(a, size=(3,3))
    array([[ 7,  7,  8,  9, 10, 11],
           [ 7,  7,  8,  9, 10, 11],
           [13, 13, 14, 15, 16, 17],
           [19, 19, 20, 20, 22, 23],
           [25, 25, 26, 27, 28, 29],
           [31, 31, 32, 33, 34, 35]])
    >>> # Note that the local minimum a[3,3] has disappeared

    """
    if (size is not None) and (footprint is not None):
        warnings.warn("ignoring size because footprint is set", UserWarning, stacklevel=2)
    tmp = grey_dilation(input, size, footprint, structure, None, mode,
                        cval, origin)
    return grey_erosion(tmp, size, footprint, structure, output, mode,
                        cval, origin)


def morphological_gradient(input, size=None, footprint=None, structure=None,
                           output=None, mode="reflect", cval=0.0, origin=0):
    """
    Multidimensional morphological gradient.

    The morphological gradient is calculated as the difference between a
    dilation and an erosion of the input with a given structuring element.

    Parameters
    ----------
    input : array_like
        Array over which to compute the morphlogical gradient.
    size : tuple of ints
        Shape of a flat and full structuring element used for the mathematical
        morphology operations. Optional if `footprint` or `structure` is
        provided. A larger `size` yields a more blurred gradient.
    footprint : array of ints, optional
        Positions of non-infinite elements of a flat structuring element
        used for the morphology operations. Larger footprints
        give a more blurred morphological gradient.
    structure : array of ints, optional
        Structuring element used for the morphology operations.
        `structure` may be a non-flat structuring element.
    output : array, optional
        An array used for storing the output of the morphological gradient
        may be provided.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.
    origin : scalar, optional
        The `origin` parameter controls the placement of the filter.
        Default 0

    Returns
    -------
    morphological_gradient : ndarray
        Morphological gradient of `input`.

    See also
    --------
    grey_dilation, grey_erosion, gaussian_gradient_magnitude

    Notes
    -----
    For a flat structuring element, the morphological gradient
    computed at a given point corresponds to the maximal difference
    between elements of the input among the elements covered by the
    structuring element centered on the point.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> from scipy import ndimage
    >>> a = np.zeros((7,7), dtype=int)
    >>> a[2:5, 2:5] = 1
    >>> ndimage.morphological_gradient(a, size=(3,3))
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 0, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> # The morphological gradient is computed as the difference
    >>> # between a dilation and an erosion
    >>> ndimage.grey_dilation(a, size=(3,3)) -\\
    ...  ndimage.grey_erosion(a, size=(3,3))
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 0, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> a = np.zeros((7,7), dtype=int)
    >>> a[2:5, 2:5] = 1
    >>> a[4,4] = 2; a[2,3] = 3
    >>> a
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 3, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.morphological_gradient(a, size=(3,3))
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 3, 3, 3, 1, 0],
           [0, 1, 3, 3, 3, 1, 0],
           [0, 1, 3, 2, 3, 2, 0],
           [0, 1, 1, 2, 2, 2, 0],
           [0, 1, 1, 2, 2, 2, 0],
           [0, 0, 0, 0, 0, 0, 0]])

    """
    tmp = grey_dilation(input, size, footprint, structure, None, mode,
                        cval, origin)
    if isinstance(output, numpy.ndarray):
        grey_erosion(input, size, footprint, structure, output, mode,
                     cval, origin)
        return numpy.subtract(tmp, output, output)
    else:
        return (tmp - grey_erosion(input, size, footprint, structure,
                                   None, mode, cval, origin))


def morphological_laplace(input, size=None, footprint=None,
                          structure=None, output=None,
                          mode="reflect", cval=0.0, origin=0):
    """
    Multidimensional morphological laplace.

    Parameters
    ----------
    input : array_like
        Input.
    size : int or sequence of ints, optional
        See `structure`.
    footprint : bool or ndarray, optional
        See `structure`.
    structure : structure, optional
        Either `size`, `footprint`, or the `structure` must be provided.
    output : ndarray, optional
        An output array can optionally be provided.
    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
        The mode parameter determines how the array borders are handled.
        For 'constant' mode, values beyond borders are set to be `cval`.
        Default is 'reflect'.
    cval : scalar, optional
        Value to fill past edges of input if mode is 'constant'.
        Default is 0.0
    origin : origin, optional
        The origin parameter controls the placement of the filter.

    Returns
    -------
    morphological_laplace : ndarray
        Output

    """
    tmp1 = grey_dilation(input, size, footprint, structure, None, mode,
                         cval, origin)
    if isinstance(output, numpy.ndarray):
        grey_erosion(input, size, footprint, structure, output, mode,
                     cval, origin)
        numpy.add(tmp1, output, output)
        numpy.subtract(output, input, output)
        return numpy.subtract(output, input, output)
    else:
        tmp2 = grey_erosion(input, size, footprint, structure, None, mode,
                            cval, origin)
        numpy.add(tmp1, tmp2, tmp2)
        numpy.subtract(tmp2, input, tmp2)
        numpy.subtract(tmp2, input, tmp2)
        return tmp2


def white_tophat(input, size=None, footprint=None, structure=None,
                 output=None, mode="reflect", cval=0.0, origin=0):
    """
    Multidimensional white tophat filter.

    Parameters
    ----------
    input : array_like
        Input.
    size : tuple of ints
        Shape of a flat and full structuring element used for the filter.
        Optional if `footprint` or `structure` is provided.
    footprint : array of ints, optional
        Positions of elements of a flat structuring element
        used for the white tophat filter.
    structure : array of ints, optional
        Structuring element used for the filter. `structure`
        may be a non-flat structuring element.
    output : array, optional
        An array used for storing the output of the filter may be provided.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'.
        Default is 0.0.
    origin : scalar, optional
        The `origin` parameter controls the placement of the filter.
        Default is 0.

    Returns
    -------
    output : ndarray
        Result of the filter of `input` with `structure`.

    Examples
    --------
    Subtract gray background from a bright peak.

    >>> from scipy.ndimage import generate_binary_structure, white_tophat
    >>> square = generate_binary_structure(rank=2, connectivity=3)
    >>> bright_on_gray = np.array([[2, 3, 3, 3, 2],
    ...                            [3, 4, 5, 4, 3],
    ...                            [3, 5, 9, 5, 3],
    ...                            [3, 4, 5, 4, 3],
    ...                            [2, 3, 3, 3, 2]])
    >>> white_tophat(input=bright_on_gray, structure=square)
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 5, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]])

    See also
    --------
    black_tophat

    """
    if (size is not None) and (footprint is not None):
        warnings.warn("ignoring size because footprint is set", UserWarning, stacklevel=2)
    tmp = grey_erosion(input, size, footprint, structure, None, mode,
                       cval, origin)
    tmp = grey_dilation(tmp, size, footprint, structure, output, mode,
                        cval, origin)
    if tmp is None:
        tmp = output

    if input.dtype == numpy.bool_ and tmp.dtype == numpy.bool_:
        numpy.bitwise_xor(input, tmp, out=tmp)
    else:
        numpy.subtract(input, tmp, out=tmp)
    return tmp


def black_tophat(input, size=None, footprint=None,
                 structure=None, output=None, mode="reflect",
                 cval=0.0, origin=0):
    """
    Multidimensional black tophat filter.

    Parameters
    ----------
    input : array_like
        Input.
    size : tuple of ints, optional
        Shape of a flat and full structuring element used for the filter.
        Optional if `footprint` or `structure` is provided.
    footprint : array of ints, optional
        Positions of non-infinite elements of a flat structuring element
        used for the black tophat filter.
    structure : array of ints, optional
        Structuring element used for the filter. `structure`
        may be a non-flat structuring element.
    output : array, optional
        An array used for storing the output of the filter may be provided.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.
    origin : scalar, optional
        The `origin` parameter controls the placement of the filter.
        Default 0

    Returns
    -------
    black_tophat : ndarray
        Result of the filter of `input` with `structure`.

    Examples
    --------
    Change dark peak to bright peak and subtract background.

    >>> from scipy.ndimage import generate_binary_structure, black_tophat
    >>> square = generate_binary_structure(rank=2, connectivity=3)
    >>> dark_on_gray = np.array([[7, 6, 6, 6, 7],
    ...                          [6, 5, 4, 5, 6],
    ...                          [6, 4, 0, 4, 6],
    ...                          [6, 5, 4, 5, 6],
    ...                          [7, 6, 6, 6, 7]])
    >>> black_tophat(input=dark_on_gray, structure=square)
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 5, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]])

    See also
    --------
    white_tophat, grey_opening, grey_closing

    """
    if (size is not None) and (footprint is not None):
        warnings.warn("ignoring size because footprint is set", UserWarning, stacklevel=2)
    tmp = grey_dilation(input, size, footprint, structure, None, mode,
                        cval, origin)
    tmp = grey_erosion(tmp, size, footprint, structure, output, mode,
                       cval, origin)
    if tmp is None:
        tmp = output

    if input.dtype == numpy.bool_ and tmp.dtype == numpy.bool_:
        numpy.bitwise_xor(tmp, input, out=tmp)
    else:
        numpy.subtract(tmp, input, out=tmp)
    return tmp


def distance_transform_bf(input, metric="euclidean", sampling=None,
                          return_distances=True, return_indices=False,
                          distances=None, indices=None):
    """
    Distance transform function by a brute force algorithm.

    This function calculates the distance transform of the `input`, by
    replacing each foreground (non-zero) element, with its
    shortest distance to the background (any zero-valued element).

    In addition to the distance transform, the feature transform can
    be calculated. In this case the index of the closest background
    element to each foreground element is returned in a separate array.

    Parameters
    ----------
    input : array_like
        Input
    metric : {'euclidean', 'taxicab', 'chessboard'}, optional
        'cityblock' and 'manhattan' are also valid, and map to 'taxicab'.
        The default is 'euclidean'.
    sampling : float, or sequence of float, optional
        This parameter is only used when `metric` is 'euclidean'.
        Spacing of elements along each dimension. If a sequence, must be of
        length equal to the input rank; if a single number, this is used for
        all axes. If not specified, a grid spacing of unity is implied.
    return_distances : bool, optional
        Whether to calculate the distance transform.
        Default is True.
    return_indices : bool, optional
        Whether to calculate the feature transform.
        Default is False.
    distances : ndarray, optional
        An output array to store the calculated distance transform, instead of
        returning it.
        `return_distances` must be True.
        It must be the same shape as `input`, and of type float64 if `metric`
        is 'euclidean', uint32 otherwise.
    indices : int32 ndarray, optional
        An output array to store the calculated feature transform, instead of
        returning it.
        `return_indicies` must be True.
        Its shape must be `(input.ndim,) + input.shape`.

    Returns
    -------
    distances : ndarray, optional
        The calculated distance transform. Returned only when
        `return_distances` is True and `distances` is not supplied.
        It will have the same shape as the input array.
    indices : int32 ndarray, optional
        The calculated feature transform. It has an input-shaped array for each
        dimension of the input. See distance_transform_edt documentation for an
        example.
        Returned only when `return_indices` is True and `indices` is not
        supplied.

    Notes
    -----
    This function employs a slow brute force algorithm, see also the
    function distance_transform_cdt for more efficient taxicab and
    chessboard algorithms.

    """
    ft_inplace = isinstance(indices, numpy.ndarray)
    dt_inplace = isinstance(distances, numpy.ndarray)
    _distance_tranform_arg_check(
        dt_inplace, ft_inplace, return_distances, return_indices
    )

    tmp1 = numpy.asarray(input) != 0
    struct = generate_binary_structure(tmp1.ndim, tmp1.ndim)
    tmp2 = binary_dilation(tmp1, struct)
    tmp2 = numpy.logical_xor(tmp1, tmp2)
    tmp1 = tmp1.astype(numpy.int8) - tmp2.astype(numpy.int8)
    metric = metric.lower()
    if metric == 'euclidean':
        metric = 1
    elif metric in ['taxicab', 'cityblock', 'manhattan']:
        metric = 2
    elif metric == 'chessboard':
        metric = 3
    else:
        raise RuntimeError('distance metric not supported')
    if sampling is not None:
        sampling = _ni_support._normalize_sequence(sampling, tmp1.ndim)
        sampling = numpy.asarray(sampling, dtype=numpy.float64)
        if not sampling.flags.contiguous:
            sampling = sampling.copy()
    if return_indices:
        ft = numpy.zeros(tmp1.shape, dtype=numpy.int32)
    else:
        ft = None
    if return_distances:
        if distances is None:
            if metric == 1:
                dt = numpy.zeros(tmp1.shape, dtype=numpy.float64)
            else:
                dt = numpy.zeros(tmp1.shape, dtype=numpy.uint32)
        else:
            if distances.shape != tmp1.shape:
                raise RuntimeError('distances array has wrong shape')
            if metric == 1:
                if distances.dtype.type != numpy.float64:
                    raise RuntimeError('distances array must be float64')
            else:
                if distances.dtype.type != numpy.uint32:
                    raise RuntimeError('distances array must be uint32')
            dt = distances
    else:
        dt = None

    _nd_image.distance_transform_bf(tmp1, metric, sampling, dt, ft)
    if return_indices:
        if isinstance(indices, numpy.ndarray):
            if indices.dtype.type != numpy.int32:
                raise RuntimeError('indices array must be int32')
            if indices.shape != (tmp1.ndim,) + tmp1.shape:
                raise RuntimeError('indices array has wrong shape')
            tmp2 = indices
        else:
            tmp2 = numpy.indices(tmp1.shape, dtype=numpy.int32)
        ft = numpy.ravel(ft)
        for ii in range(tmp2.shape[0]):
            rtmp = numpy.ravel(tmp2[ii, ...])[ft]
            rtmp.shape = tmp1.shape
            tmp2[ii, ...] = rtmp
        ft = tmp2

    # construct and return the result
    result = []
    if return_distances and not dt_inplace:
        result.append(dt)
    if return_indices and not ft_inplace:
        result.append(ft)

    if len(result) == 2:
        return tuple(result)
    elif len(result) == 1:
        return result[0]
    else:
        return None


def distance_transform_cdt(input, metric='chessboard', return_distances=True,
                           return_indices=False, distances=None, indices=None):
    """
    Distance transform for chamfer type of transforms.

    In addition to the distance transform, the feature transform can
    be calculated. In this case the index of the closest background
    element to each foreground element is returned in a separate array.

    Parameters
    ----------
    input : array_like
        Input
    metric : {'chessboard', 'taxicab'} or array_like, optional
        The `metric` determines the type of chamfering that is done. If the
        `metric` is equal to 'taxicab' a structure is generated using
        generate_binary_structure with a squared distance equal to 1. If
        the `metric` is equal to 'chessboard', a `metric` is generated
        using generate_binary_structure with a squared distance equal to
        the dimensionality of the array. These choices correspond to the
        common interpretations of the 'taxicab' and the 'chessboard'
        distance metrics in two dimensions.
        A custom metric may be provided, in the form of a matrix where
        each dimension has a length of three.
        'cityblock' and 'manhattan' are also valid, and map to 'taxicab'.
        The default is 'chessboard'.
    return_distances : bool, optional
        Whether to calculate the distance transform.
        Default is True.
    return_indices : bool, optional
        Whether to calculate the feature transform.
        Default is False.
    distances : int32 ndarray, optional
        An output array to store the calculated distance transform, instead of
        returning it.
        `return_distances` must be True.
        It must be the same shape as `input`.
    indices : int32 ndarray, optional
        An output array to store the calculated feature transform, instead of
        returning it.
        `return_indicies` must be True.
        Its shape must be `(input.ndim,) + input.shape`.

    Returns
    -------
    distances : int32 ndarray, optional
        The calculated distance transform. Returned only when
        `return_distances` is True, and `distances` is not supplied.
        It will have the same shape as the input array.
    indices : int32 ndarray, optional
        The calculated feature transform. It has an input-shaped array for each
        dimension of the input. See distance_transform_edt documentation for an
        example.
        Returned only when `return_indices` is True, and `indices` is not
        supplied.

    """
    ft_inplace = isinstance(indices, numpy.ndarray)
    dt_inplace = isinstance(distances, numpy.ndarray)
    _distance_tranform_arg_check(
        dt_inplace, ft_inplace, return_distances, return_indices
    )
    input = numpy.asarray(input)
    if metric in ['taxicab', 'cityblock', 'manhattan']:
        rank = input.ndim
        metric = generate_binary_structure(rank, 1)
    elif metric == 'chessboard':
        rank = input.ndim
        metric = generate_binary_structure(rank, rank)
    else:
        try:
            metric = numpy.asarray(metric)
        except Exception as e:
            raise RuntimeError('invalid metric provided') from e
        for s in metric.shape:
            if s != 3:
                raise RuntimeError('metric sizes must be equal to 3')

    if not metric.flags.contiguous:
        metric = metric.copy()
    if dt_inplace:
        if distances.dtype.type != numpy.int32:
            raise RuntimeError('distances must be of int32 type')
        if distances.shape != input.shape:
            raise RuntimeError('distances has wrong shape')
        dt = distances
        dt[...] = numpy.where(input, -1, 0).astype(numpy.int32)
    else:
        dt = numpy.where(input, -1, 0).astype(numpy.int32)

    rank = dt.ndim
    if return_indices:
        sz = numpy.prod(dt.shape, axis=0)
        ft = numpy.arange(sz, dtype=numpy.int32)
        ft.shape = dt.shape
    else:
        ft = None

    _nd_image.distance_transform_op(metric, dt, ft)
    dt = dt[tuple([slice(None, None, -1)] * rank)]
    if return_indices:
        ft = ft[tuple([slice(None, None, -1)] * rank)]
    _nd_image.distance_transform_op(metric, dt, ft)
    dt = dt[tuple([slice(None, None, -1)] * rank)]
    if return_indices:
        ft = ft[tuple([slice(None, None, -1)] * rank)]
        ft = numpy.ravel(ft)
        if ft_inplace:
            if indices.dtype.type != numpy.int32:
                raise RuntimeError('indices array must be int32')
            if indices.shape != (dt.ndim,) + dt.shape:
                raise RuntimeError('indices array has wrong shape')
            tmp = indices
        else:
            tmp = numpy.indices(dt.shape, dtype=numpy.int32)
        for ii in range(tmp.shape[0]):
            rtmp = numpy.ravel(tmp[ii, ...])[ft]
            rtmp.shape = dt.shape
            tmp[ii, ...] = rtmp
        ft = tmp

    # construct and return the result
    result = []
    if return_distances and not dt_inplace:
        result.append(dt)
    if return_indices and not ft_inplace:
        result.append(ft)

    if len(result) == 2:
        return tuple(result)
    elif len(result) == 1:
        return result[0]
    else:
        return None


def distance_transform_edt(input, sampling=None, return_distances=True,
                           return_indices=False, distances=None, indices=None):
    """
    Exact Euclidean distance transform.

    In addition to the distance transform, the feature transform can
    be calculated. In this case the index of the closest background
    element to each foreground element is returned in a separate array.

    Parameters
    ----------
    input : array_like
        Input data to transform. Can be any type but will be converted
        into binary: 1 wherever input equates to True, 0 elsewhere.
    sampling : float, or sequence of float, optional
        Spacing of elements along each dimension. If a sequence, must be of
        length equal to the input rank; if a single number, this is used for
        all axes. If not specified, a grid spacing of unity is implied.
    return_distances : bool, optional
        Whether to calculate the distance transform.
        Default is True.
    return_indices : bool, optional
        Whether to calculate the feature transform.
        Default is False.
    distances : float64 ndarray, optional
        An output array to store the calculated distance transform, instead of
        returning it.
        `return_distances` must be True.
        It must be the same shape as `input`.
    indices : int32 ndarray, optional
        An output array to store the calculated feature transform, instead of
        returning it.
        `return_indicies` must be True.
        Its shape must be `(input.ndim,) + input.shape`.

    Returns
    -------
    distances : float64 ndarray, optional
        The calculated distance transform. Returned only when
        `return_distances` is True and `distances` is not supplied.
        It will have the same shape as the input array.
    indices : int32 ndarray, optional
        The calculated feature transform. It has an input-shaped array for each
        dimension of the input. See example below.
        Returned only when `return_indices` is True and `indices` is not
        supplied.

    Notes
    -----
    The Euclidean distance transform gives values of the Euclidean
    distance::

                    n
      y_i = sqrt(sum (x[i]-b[i])**2)
                    i

    where b[i] is the background point (value 0) with the smallest
    Euclidean distance to input points x[i], and n is the
    number of dimensions.

    Examples
    --------
    >>> from scipy import ndimage
    >>> a = np.array(([0,1,1,1,1],
    ...               [0,0,1,1,1],
    ...               [0,1,1,1,1],
    ...               [0,1,1,1,0],
    ...               [0,1,1,0,0]))
    >>> ndimage.distance_transform_edt(a)
    array([[ 0.    ,  1.    ,  1.4142,  2.2361,  3.    ],
           [ 0.    ,  0.    ,  1.    ,  2.    ,  2.    ],
           [ 0.    ,  1.    ,  1.4142,  1.4142,  1.    ],
           [ 0.    ,  1.    ,  1.4142,  1.    ,  0.    ],
           [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])

    With a sampling of 2 units along x, 1 along y:

    >>> ndimage.distance_transform_edt(a, sampling=[2,1])
    array([[ 0.    ,  1.    ,  2.    ,  2.8284,  3.6056],
           [ 0.    ,  0.    ,  1.    ,  2.    ,  3.    ],
           [ 0.    ,  1.    ,  2.    ,  2.2361,  2.    ],
           [ 0.    ,  1.    ,  2.    ,  1.    ,  0.    ],
           [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])

    Asking for indices as well:

    >>> edt, inds = ndimage.distance_transform_edt(a, return_indices=True)
    >>> inds
    array([[[0, 0, 1, 1, 3],
            [1, 1, 1, 1, 3],
            [2, 2, 1, 3, 3],
            [3, 3, 4, 4, 3],
            [4, 4, 4, 4, 4]],
           [[0, 0, 1, 1, 4],
            [0, 1, 1, 1, 4],
            [0, 0, 1, 4, 4],
            [0, 0, 3, 3, 4],
            [0, 0, 3, 3, 4]]])

    With arrays provided for inplace outputs:

    >>> indices = np.zeros(((np.ndim(a),) + a.shape), dtype=np.int32)
    >>> ndimage.distance_transform_edt(a, return_indices=True, indices=indices)
    array([[ 0.    ,  1.    ,  1.4142,  2.2361,  3.    ],
           [ 0.    ,  0.    ,  1.    ,  2.    ,  2.    ],
           [ 0.    ,  1.    ,  1.4142,  1.4142,  1.    ],
           [ 0.    ,  1.    ,  1.4142,  1.    ,  0.    ],
           [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])
    >>> indices
    array([[[0, 0, 1, 1, 3],
            [1, 1, 1, 1, 3],
            [2, 2, 1, 3, 3],
            [3, 3, 4, 4, 3],
            [4, 4, 4, 4, 4]],
           [[0, 0, 1, 1, 4],
            [0, 1, 1, 1, 4],
            [0, 0, 1, 4, 4],
            [0, 0, 3, 3, 4],
            [0, 0, 3, 3, 4]]])

    """
    ft_inplace = isinstance(indices, numpy.ndarray)
    dt_inplace = isinstance(distances, numpy.ndarray)
    _distance_tranform_arg_check(
        dt_inplace, ft_inplace, return_distances, return_indices
    )

    # calculate the feature transform
    input = numpy.atleast_1d(numpy.where(input, 1, 0).astype(numpy.int8))
    if sampling is not None:
        sampling = _ni_support._normalize_sequence(sampling, input.ndim)
        sampling = numpy.asarray(sampling, dtype=numpy.float64)
        if not sampling.flags.contiguous:
            sampling = sampling.copy()

    if ft_inplace:
        ft = indices
        if ft.shape != (input.ndim,) + input.shape:
            raise RuntimeError('indices array has wrong shape')
        if ft.dtype.type != numpy.int32:
            raise RuntimeError('indices array must be int32')
    else:
        ft = numpy.zeros((input.ndim,) + input.shape, dtype=numpy.int32)

    _nd_image.euclidean_feature_transform(input, sampling, ft)
    # if requested, calculate the distance transform
    if return_distances:
        dt = ft - numpy.indices(input.shape, dtype=ft.dtype)
        dt = dt.astype(numpy.float64)
        if sampling is not None:
            for ii in range(len(sampling)):
                dt[ii, ...] *= sampling[ii]
        numpy.multiply(dt, dt, dt)
        if dt_inplace:
            dt = numpy.add.reduce(dt, axis=0)
            if distances.shape != dt.shape:
                raise RuntimeError('distances array has wrong shape')
            if distances.dtype.type != numpy.float64:
                raise RuntimeError('distances array must be float64')
            numpy.sqrt(dt, distances)
        else:
            dt = numpy.add.reduce(dt, axis=0)
            dt = numpy.sqrt(dt)

    # construct and return the result
    result = []
    if return_distances and not dt_inplace:
        result.append(dt)
    if return_indices and not ft_inplace:
        result.append(ft)

    if len(result) == 2:
        return tuple(result)
    elif len(result) == 1:
        return result[0]
    else:
        return None


def _distance_tranform_arg_check(distances_out, indices_out,
                                 return_distances, return_indices):
    """Raise a RuntimeError if the arguments are invalid"""
    error_msgs = []
    if (not return_distances) and (not return_indices):
        error_msgs.append(
            'at least one of return_distances/return_indices must be True')
    if distances_out and not return_distances:
        error_msgs.append(
            'return_distances must be True if distances is supplied'
        )
    if indices_out and not return_indices:
        error_msgs.append('return_indices must be True if indices is supplied')
    if error_msgs:
        raise RuntimeError(', '.join(error_msgs))
