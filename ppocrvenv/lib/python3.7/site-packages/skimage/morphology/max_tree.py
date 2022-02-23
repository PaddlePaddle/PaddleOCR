"""max_tree.py - max_tree representation of images.

This module provides operators based on the max-tree representation of images.
A grayscale image can be seen as a pile of nested sets, each of which is the
result of a threshold operation. These sets can be efficiently represented by
max-trees, where the inclusion relation between connected components at
different levels are represented by parent-child relationships.

These representations allow efficient implementations of many algorithms, such
as attribute operators. Unlike morphological openings and closings, these
operators do not require a fixed footprint, but rather act with a flexible
footprint that meets a certain criterion.

This implementation provides functions for:
1. max-tree generation
2. area openings / closings
3. diameter openings / closings
4. local maxima

References:
    .. [1] Salembier, P., Oliveras, A., & Garrido, L. (1998). Antiextensive
           Connected Operators for Image and Sequence Processing.
           IEEE Transactions on Image Processing, 7(4), 555-570.
           :DOI:10.1109/83.663500
    .. [2] Berger, C., Geraud, T., Levillain, R., Widynski, N., Baillard, A.,
           Bertin, E. (2007). Effective Component Tree Computation with
           Application to Pattern Recognition in Astronomical Imaging.
           In International Conference on Image Processing (ICIP) (pp. 41-44).
           :DOI:10.1109/ICIP.2007.4379949
    .. [3] Najman, L., & Couprie, M. (2006). Building the component tree in
           quasi-linear time. IEEE Transactions on Image Processing, 15(11),
           3531-3539.
           :DOI:10.1109/TIP.2006.877518
    .. [4] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
           Component Tree Computation Algorithms. IEEE Transactions on Image
           Processing, 23(9), 3885-3895.
           :DOI:10.1109/TIP.2014.2336551
"""

import numpy as np

from ._util import _validate_connectivity, _offsets_to_raveled_neighbors
from ..util import invert

from . import _max_tree

unsigned_int_types = [np.uint8, np.uint16, np.uint32, np.uint64]
signed_int_types = [np.int8, np.int16, np.int32, np.int64]
signed_float_types = [np.float16, np.float32, np.float64]


# building the max tree.
def max_tree(image, connectivity=1):
    """Build the max tree from an image.

    Component trees represent the hierarchical structure of the connected
    components resulting from sequential thresholding operations applied to an
    image. A connected component at one level is parent of a component at a
    higher level if the latter is included in the first. A max-tree is an
    efficient representation of a component tree. A connected component at
    one level is represented by one reference pixel at this level, which is
    parent to all other pixels at that level and to the reference pixel at the
    level above. The max-tree is the basis for many morphological operators,
    namely connected operators.

    Parameters
    ----------
    image : ndarray
        The input image for which the max-tree is to be calculated.
        This image can be of any type.
    connectivity : unsigned int, optional
        The neighborhood connectivity. The integer represents the maximum
        number of orthogonal steps to reach a neighbor. In 2D, it is 1 for
        a 4-neighborhood and 2 for a 8-neighborhood. Default value is 1.

    Returns
    -------
    parent : ndarray, int64
        Array of same shape as image. The value of each pixel is the index of
        its parent in the ravelled array.
    tree_traverser : 1D array, int64
        The ordered pixel indices (referring to the ravelled array). The pixels
        are ordered such that every pixel is preceded by its parent (except for
        the root which has no parent).

    References
    ----------
    .. [1] Salembier, P., Oliveras, A., & Garrido, L. (1998). Antiextensive
           Connected Operators for Image and Sequence Processing.
           IEEE Transactions on Image Processing, 7(4), 555-570.
           :DOI:`10.1109/83.663500`
    .. [2] Berger, C., Geraud, T., Levillain, R., Widynski, N., Baillard, A.,
           Bertin, E. (2007). Effective Component Tree Computation with
           Application to Pattern Recognition in Astronomical Imaging.
           In International Conference on Image Processing (ICIP) (pp. 41-44).
           :DOI:`10.1109/ICIP.2007.4379949`
    .. [3] Najman, L., & Couprie, M. (2006). Building the component tree in
           quasi-linear time. IEEE Transactions on Image Processing, 15(11),
           3531-3539.
           :DOI:`10.1109/TIP.2006.877518`
    .. [4] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
           Component Tree Computation Algorithms. IEEE Transactions on Image
           Processing, 23(9), 3885-3895.
           :DOI:`10.1109/TIP.2014.2336551`

    Examples
    --------
    We create a small sample image (Figure 1 from [4]) and build the max-tree.

    >>> image = np.array([[15, 13, 16], [12, 12, 10], [16, 12, 14]])
    >>> P, S = max_tree(image, connectivity=2)
    """
    # User defined masks are not allowed, as there might be more than one
    # connected component in the mask (and therefore not a single tree that
    # represents the image). Mask here is an image that is 0 on the border
    # and 1 everywhere else.
    mask = np.ones(image.shape)
    for k in range(len(image.shape)):
        np.moveaxis(mask, k, 0)[0] = 0
        np.moveaxis(mask, k, 0)[-1] = 0

    neighbors, offset = _validate_connectivity(image.ndim, connectivity,
                                               offset=None)

    # initialization of the parent image
    parent = np.zeros(image.shape, dtype=np.int64)

    # flat_neighborhood contains a list of offsets allowing one to find the
    # neighbors in the ravelled image.
    flat_neighborhood = _offsets_to_raveled_neighbors(image.shape, neighbors,
                                                      offset).astype(np.int32)

    # pixels need to be sorted according to their gray level.
    tree_traverser = np.argsort(image.ravel()).astype(np.int64)

    # call of cython function.
    _max_tree._max_tree(image.ravel(), mask.ravel().astype(np.uint8),
                        flat_neighborhood, offset.astype(np.int32),
                        np.array(image.shape, dtype=np.int32),
                        parent.ravel(), tree_traverser)

    return parent, tree_traverser


def area_opening(image, area_threshold=64, connectivity=1,
                 parent=None, tree_traverser=None):
    """Perform an area opening of the image.

    Area opening removes all bright structures of an image with
    a surface smaller than area_threshold.
    The output image is thus the largest image smaller than the input
    for which all local maxima have at least a surface of
    area_threshold pixels.

    Area openings are similar to morphological openings, but
    they do not use a fixed footprint, but rather a deformable
    one, with surface = area_threshold. Consequently, the area_opening
    with area_threshold=1 is the identity.

    In the binary case, area openings are equivalent to
    remove_small_objects; this operator is thus extended to gray-level images.

    Technically, this operator is based on the max-tree representation of
    the image.

    Parameters
    ----------
    image : ndarray
        The input image for which the area_opening is to be calculated.
        This image can be of any type.
    area_threshold : unsigned int
        The size parameter (number of pixels). The default value is arbitrarily
        chosen to be 64.
    connectivity : unsigned int, optional
        The neighborhood connectivity. The integer represents the maximum
        number of orthogonal steps to reach a neighbor. In 2D, it is 1 for
        a 4-neighborhood and 2 for a 8-neighborhood. Default value is 1.
    parent : ndarray, int64, optional
        Parent image representing the max tree of the image. The
        value of each pixel is the index of its parent in the ravelled array.
    tree_traverser : 1D array, int64, optional
        The ordered pixel indices (referring to the ravelled array). The pixels
        are ordered such that every pixel is preceded by its parent (except for
        the root which has no parent).

    Returns
    -------
    output : ndarray
        Output image of the same shape and type as the input image.

    See Also
    --------
    skimage.morphology.area_closing
    skimage.morphology.diameter_opening
    skimage.morphology.diameter_closing
    skimage.morphology.max_tree
    skimage.morphology.remove_small_objects
    skimage.morphology.remove_small_holes

    References
    ----------
    .. [1] Vincent L., Proc. "Grayscale area openings and closings,
           their efficient implementation and applications",
           EURASIP Workshop on Mathematical Morphology and its
           Applications to Signal Processing, Barcelona, Spain, pp.22-27,
           May 1993.
    .. [2] Soille, P., "Morphological Image Analysis: Principles and
           Applications" (Chapter 6), 2nd edition (2003), ISBN 3540429883.
           :DOI:10.1007/978-3-662-05088-0
    .. [3] Salembier, P., Oliveras, A., & Garrido, L. (1998). Antiextensive
           Connected Operators for Image and Sequence Processing.
           IEEE Transactions on Image Processing, 7(4), 555-570.
           :DOI:10.1109/83.663500
    .. [4] Najman, L., & Couprie, M. (2006). Building the component tree in
           quasi-linear time. IEEE Transactions on Image Processing, 15(11),
           3531-3539.
           :DOI:10.1109/TIP.2006.877518
    .. [5] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
           Component Tree Computation Algorithms. IEEE Transactions on Image
           Processing, 23(9), 3885-3895.
           :DOI:10.1109/TIP.2014.2336551

    Examples
    --------
    We create an image (quadratic function with a maximum in the center and
    4 additional local maxima.

    >>> w = 12
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 20 - 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:3,1:5] = 40; f[2:4,9:11] = 60; f[9:11,2:4] = 80
    >>> f[9:10,9:11] = 100; f[10,10] = 100
    >>> f = f.astype(int)

    We can calculate the area opening:

    >>> open = area_opening(f, 8, connectivity=1)

    The peaks with a surface smaller than 8 are removed.
    """
    output = image.copy()

    if parent is None or tree_traverser is None:
        parent, tree_traverser = max_tree(image, connectivity)

    area = _max_tree._compute_area(image.ravel(),
                                   parent.ravel(), tree_traverser)

    _max_tree._direct_filter(image.ravel(), output.ravel(), parent.ravel(),
                             tree_traverser, area, area_threshold)
    return output


def diameter_opening(image, diameter_threshold=8, connectivity=1,
                     parent=None, tree_traverser=None):
    """Perform a diameter opening of the image.

    Diameter opening removes all bright structures of an image with
    maximal extension smaller than diameter_threshold. The maximal
    extension is defined as the maximal extension of the bounding box.
    The operator is also called Bounding Box Opening. In practice,
    the result is similar to a morphological opening, but long and thin
    structures are not removed.

    Technically, this operator is based on the max-tree representation of
    the image.

    Parameters
    ----------
    image : ndarray
        The input image for which the area_opening is to be calculated.
        This image can be of any type.
    diameter_threshold : unsigned int
        The maximal extension parameter (number of pixels). The default value
        is 8.
    connectivity : unsigned int, optional
        The neighborhood connectivity. The integer represents the maximum
        number of orthogonal steps to reach a neighbor. In 2D, it is 1 for
        a 4-neighborhood and 2 for a 8-neighborhood. Default value is 1.
    parent : ndarray, int64, optional
        Parent image representing the max tree of the image. The
        value of each pixel is the index of its parent in the ravelled array.
    tree_traverser : 1D array, int64, optional
        The ordered pixel indices (referring to the ravelled array). The pixels
        are ordered such that every pixel is preceded by its parent (except for
        the root which has no parent).

    Returns
    -------
    output : ndarray
        Output image of the same shape and type as the input image.

    See Also
    --------
    skimage.morphology.area_opening
    skimage.morphology.area_closing
    skimage.morphology.diameter_closing
    skimage.morphology.max_tree

    References
    ----------
    .. [1] Walter, T., & Klein, J.-C. (2002). Automatic Detection of
           Microaneurysms in Color Fundus Images of the Human Retina by Means
           of the Bounding Box Closing. In A. Colosimo, P. Sirabella,
           A. Giuliani (Eds.), Medical Data Analysis. Lecture Notes in Computer
           Science, vol 2526, pp. 210-220. Springer Berlin Heidelberg.
           :DOI:`10.1007/3-540-36104-9_23`
    .. [2] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
           Component Tree Computation Algorithms. IEEE Transactions on Image
           Processing, 23(9), 3885-3895.
           :DOI:`10.1109/TIP.2014.2336551`

    Examples
    --------
    We create an image (quadratic function with a maximum in the center and
    4 additional local maxima.

    >>> w = 12
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 20 - 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:3,1:5] = 40; f[2:4,9:11] = 60; f[9:11,2:4] = 80
    >>> f[9:10,9:11] = 100; f[10,10] = 100
    >>> f = f.astype(int)

    We can calculate the diameter opening:

    >>> open = diameter_opening(f, 3, connectivity=1)

    The peaks with a maximal extension of 2 or less are removed.
    The remaining peaks have all a maximal extension of at least 3.
    """
    output = image.copy()

    if parent is None or tree_traverser is None:
        parent, tree_traverser = max_tree(image, connectivity)

    diam = _max_tree._compute_extension(image.ravel(),
                                        np.array(image.shape, dtype=np.int32),
                                        parent.ravel(), tree_traverser)

    _max_tree._direct_filter(image.ravel(), output.ravel(), parent.ravel(),
                             tree_traverser, diam, diameter_threshold)
    return output


def area_closing(image, area_threshold=64, connectivity=1,
                 parent=None, tree_traverser=None):
    """Perform an area closing of the image.

    Area closing removes all dark structures of an image with
    a surface smaller than area_threshold.
    The output image is larger than or equal to the input image
    for every pixel and all local minima have at least a surface of
    area_threshold pixels.

    Area closings are similar to morphological closings, but
    they do not use a fixed footprint, but rather a deformable
    one, with surface = area_threshold.

    In the binary case, area closings are equivalent to
    remove_small_holes; this operator is thus extended to gray-level images.

    Technically, this operator is based on the max-tree representation of
    the image.

    Parameters
    ----------
    image : ndarray
        The input image for which the area_closing is to be calculated.
        This image can be of any type.
    area_threshold : unsigned int
        The size parameter (number of pixels). The default value is arbitrarily
        chosen to be 64.
    connectivity : unsigned int, optional
        The neighborhood connectivity. The integer represents the maximum
        number of orthogonal steps to reach a neighbor. In 2D, it is 1 for
        a 4-neighborhood and 2 for a 8-neighborhood. Default value is 1.
    parent : ndarray, int64, optional
        Parent image representing the max tree of the inverted image. The
        value of each pixel is the index of its parent in the ravelled array.
        See Note for further details.
    tree_traverser : 1D array, int64, optional
        The ordered pixel indices (referring to the ravelled array). The pixels
        are ordered such that every pixel is preceded by its parent (except for
        the root which has no parent).

    Returns
    -------
    output : ndarray
        Output image of the same shape and type as input image.

    See Also
    --------
    skimage.morphology.area_opening
    skimage.morphology.diameter_opening
    skimage.morphology.diameter_closing
    skimage.morphology.max_tree
    skimage.morphology.remove_small_objects
    skimage.morphology.remove_small_holes

    References
    ----------
    .. [1] Vincent L., Proc. "Grayscale area openings and closings,
           their efficient implementation and applications",
           EURASIP Workshop on Mathematical Morphology and its
           Applications to Signal Processing, Barcelona, Spain, pp.22-27,
           May 1993.
    .. [2] Soille, P., "Morphological Image Analysis: Principles and
           Applications" (Chapter 6), 2nd edition (2003), ISBN 3540429883.
           :DOI:`10.1007/978-3-662-05088-0`
    .. [3] Salembier, P., Oliveras, A., & Garrido, L. (1998). Antiextensive
           Connected Operators for Image and Sequence Processing.
           IEEE Transactions on Image Processing, 7(4), 555-570.
           :DOI:`10.1109/83.663500`
    .. [4] Najman, L., & Couprie, M. (2006). Building the component tree in
           quasi-linear time. IEEE Transactions on Image Processing, 15(11),
           3531-3539.
           :DOI:`10.1109/TIP.2006.877518`
    .. [5] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
           Component Tree Computation Algorithms. IEEE Transactions on Image
           Processing, 23(9), 3885-3895.
           :DOI:`10.1109/TIP.2014.2336551`

    Examples
    --------
    We create an image (quadratic function with a minimum in the center and
    4 additional local minima.

    >>> w = 12
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 180 + 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:3,1:5] = 160; f[2:4,9:11] = 140; f[9:11,2:4] = 120
    >>> f[9:10,9:11] = 100; f[10,10] = 100
    >>> f = f.astype(int)

    We can calculate the area closing:

    >>> closed = area_closing(f, 8, connectivity=1)

    All small minima are removed, and the remaining minima have at least
    a size of 8.

    Notes
    -----
    If a max-tree representation (parent and tree_traverser) are given to the
    function, they must be calculated from the inverted image for this
    function, i.e.:
    >>> P, S = max_tree(invert(f))
    >>> closed = diameter_closing(f, 3, parent=P, tree_traverser=S)
    """
    # inversion of the input image
    image_inv = invert(image)
    output = image_inv.copy()

    if parent is None or tree_traverser is None:
        parent, tree_traverser = max_tree(image_inv, connectivity)

    area = _max_tree._compute_area(image_inv.ravel(),
                                   parent.ravel(), tree_traverser)

    _max_tree._direct_filter(image_inv.ravel(), output.ravel(), parent.ravel(),
                             tree_traverser, area, area_threshold)

    # inversion of the output image
    output = invert(output)

    return output


def diameter_closing(image, diameter_threshold=8, connectivity=1,
                     parent=None, tree_traverser=None):
    """Perform a diameter closing of the image.

    Diameter closing removes all dark structures of an image with
    maximal extension smaller than diameter_threshold. The maximal
    extension is defined as the maximal extension of the bounding box.
    The operator is also called Bounding Box Closing. In practice,
    the result is similar to a morphological closing, but long and thin
    structures are not removed.

    Technically, this operator is based on the max-tree representation of
    the image.

    Parameters
    ----------
    image : ndarray
        The input image for which the diameter_closing is to be calculated.
        This image can be of any type.
    diameter_threshold : unsigned int
        The maximal extension parameter (number of pixels). The default value
        is 8.
    connectivity : unsigned int, optional
        The neighborhood connectivity. The integer represents the maximum
        number of orthogonal steps to reach a neighbor. In 2D, it is 1 for
        a 4-neighborhood and 2 for a 8-neighborhood. Default value is 1.
    parent : ndarray, int64, optional
        Precomputed parent image representing the max tree of the inverted
        image. This function is fast, if precomputed parent and tree_traverser
        are provided. See Note for further details.
    tree_traverser : 1D array, int64, optional
        Precomputed traverser, where the pixels are ordered such that every
        pixel is preceded by its parent (except for the root which has no
        parent). This function is fast, if precomputed parent and
        tree_traverser are provided. See Note for further details.

    Returns
    -------
    output : ndarray
        Output image of the same shape and type as input image.

    See Also
    --------
    skimage.morphology.area_opening
    skimage.morphology.area_closing
    skimage.morphology.diameter_opening
    skimage.morphology.max_tree

    References
    ----------
    .. [1] Walter, T., & Klein, J.-C. (2002). Automatic Detection of
           Microaneurysms in Color Fundus Images of the Human Retina by Means
           of the Bounding Box Closing. In A. Colosimo, P. Sirabella,
           A. Giuliani (Eds.), Medical Data Analysis. Lecture Notes in Computer
           Science, vol 2526, pp. 210-220. Springer Berlin Heidelberg.
           :DOI:`10.1007/3-540-36104-9_23`
    .. [2] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
           Component Tree Computation Algorithms. IEEE Transactions on Image
           Processing, 23(9), 3885-3895.
           :DOI:`10.1109/TIP.2014.2336551`

    Examples
    --------
    We create an image (quadratic function with a minimum in the center and
    4 additional local minima.

    >>> w = 12
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 180 + 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:3,1:5] = 160; f[2:4,9:11] = 140; f[9:11,2:4] = 120
    >>> f[9:10,9:11] = 100; f[10,10] = 100
    >>> f = f.astype(int)

    We can calculate the diameter closing:

    >>> closed = diameter_closing(f, 3, connectivity=1)

    All small minima with a maximal extension of 2 or less are removed.
    The remaining minima have all a maximal extension of at least 3.

    Notes
    -----
    If a max-tree representation (parent and tree_traverser) are given to the
    function, they must be calculated from the inverted image for this
    function, i.e.:
    >>> P, S = max_tree(invert(f))
    >>> closed = diameter_closing(f, 3, parent=P, tree_traverser=S)
    """
    # inversion of the input image
    image_inv = invert(image)
    output = image_inv.copy()

    if parent is None or tree_traverser is None:
        parent, tree_traverser = max_tree(image_inv, connectivity)

    diam = _max_tree._compute_extension(image_inv.ravel(),
                                        np.array(image_inv.shape,
                                                 dtype=np.int32),
                                        parent.ravel(), tree_traverser)

    _max_tree._direct_filter(image_inv.ravel(), output.ravel(), parent.ravel(),
                             tree_traverser, diam, diameter_threshold)
    output = invert(output)
    return output


def max_tree_local_maxima(image, connectivity=1,
                          parent=None, tree_traverser=None):
    """Determine all local maxima of the image.

    The local maxima are defined as connected sets of pixels with equal
    gray level strictly greater than the gray levels of all pixels in direct
    neighborhood of the set. The function labels the local maxima.

    Technically, the implementation is based on the max-tree representation
    of an image. The function is very efficient if the max-tree representation
    has already been computed. Otherwise, it is preferable to use
    the function local_maxima.

    Parameters
    ----------
    image : ndarray
        The input image for which the maxima are to be calculated.
    connectivity : unsigned int, optional
        The neighborhood connectivity. The integer represents the maximum
        number of orthogonal steps to reach a neighbor. In 2D, it is 1 for
        a 4-neighborhood and 2 for a 8-neighborhood. Default value is 1.
    parent : ndarray, int64, optional
        The value of each pixel is the index of its parent in the ravelled
        array.
    tree_traverser : 1D array, int64, optional
        The ordered pixel indices (referring to the ravelled array). The pixels
        are ordered such that every pixel is preceded by its parent (except for
        the root which has no parent).

    Returns
    -------
    local_max : ndarray, uint64
        Labeled local maxima of the image.

    See Also
    --------
    skimage.morphology.local_maxima
    skimage.morphology.max_tree

    References
    ----------
    .. [1] Vincent L., Proc. "Grayscale area openings and closings,
           their efficient implementation and applications",
           EURASIP Workshop on Mathematical Morphology and its
           Applications to Signal Processing, Barcelona, Spain, pp.22-27,
           May 1993.
    .. [2] Soille, P., "Morphological Image Analysis: Principles and
           Applications" (Chapter 6), 2nd edition (2003), ISBN 3540429883.
           :DOI:`10.1007/978-3-662-05088-0`
    .. [3] Salembier, P., Oliveras, A., & Garrido, L. (1998). Antiextensive
           Connected Operators for Image and Sequence Processing.
           IEEE Transactions on Image Processing, 7(4), 555-570.
           :DOI:`10.1109/83.663500`
    .. [4] Najman, L., & Couprie, M. (2006). Building the component tree in
           quasi-linear time. IEEE Transactions on Image Processing, 15(11),
           3531-3539.
           :DOI:`10.1109/TIP.2006.877518`
    .. [5] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
           Component Tree Computation Algorithms. IEEE Transactions on Image
           Processing, 23(9), 3885-3895.
           :DOI:`10.1109/TIP.2014.2336551`

    Examples
    --------
    We create an image (quadratic function with a maximum in the center and
    4 additional constant maxima.

    >>> w = 10
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 20 - 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:4,2:4] = 40; f[2:4,7:9] = 60; f[7:9,2:4] = 80; f[7:9,7:9] = 100
    >>> f = f.astype(int)

    We can calculate all local maxima:

    >>> maxima = max_tree_local_maxima(f)

    The resulting image contains the labeled local maxima.
    """

    output = np.ones(image.shape, dtype=np.uint64)

    if parent is None or tree_traverser is None:
        parent, tree_traverser = max_tree(image, connectivity)

    _max_tree._max_tree_local_maxima(image.ravel(), output.ravel(),
                                     parent.ravel(), tree_traverser)

    return output
