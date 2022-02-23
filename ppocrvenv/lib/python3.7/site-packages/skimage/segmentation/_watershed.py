"""watershed.py - watershed algorithm

This module implements a watershed algorithm that apportions pixels into
marked basins. The algorithm uses a priority queue to hold the pixels
with the metric for the priority queue being pixel value, then the time
of entry into the queue - this settles ties in favor of the closest marker.

Some ideas taken from
Soille, "Automated Basin Delineation from Digital Elevation Models Using
Mathematical Morphology", Signal Processing 20 (1990) 171-182.

The most important insight in the paper is that entry time onto the queue
solves two problems: a pixel should be assigned to the neighbor with the
largest gradient or, if there is no gradient, pixels on a plateau should
be split between markers on opposite sides.

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Original author: Lee Kamentsky
"""

import numpy as np
from scipy import ndimage as ndi

from . import _watershed_cy
from ..morphology.extrema import local_minima
from ..morphology._util import (_validate_connectivity,
                                _offsets_to_raveled_neighbors)
from ..util import crop, regular_seeds


def _validate_inputs(image, markers, mask, connectivity):
    """Ensure that all inputs to watershed have matching shapes and types.

    Parameters
    ----------
    image : array
        The input image.
    markers : int or array of int
        The marker image.
    mask : array, or None
        A boolean mask, True where we want to compute the watershed.
    connectivity : int in {1, ..., image.ndim}
        The connectivity of the neighborhood of a pixel.

    Returns
    -------
    image, markers, mask : arrays
        The validated and formatted arrays. Image will have dtype float64,
        markers int32, and mask int8. If ``None`` was given for the mask,
        it is a volume of all 1s.

    Raises
    ------
    ValueError
        If the shapes of the given arrays don't match.
    """
    n_pixels = image.size
    if mask is None:
        # Use a complete `True` mask if none is provided
        mask = np.ones(image.shape, bool)
    else:
        mask = np.asanyarray(mask, dtype=bool)
        n_pixels = np.sum(mask)
        if mask.shape != image.shape:
            message = (f'`mask` (shape {mask.shape}) must have same shape '
                       f'as `image` (shape {image.shape})')
            raise ValueError(message)
    if markers is None:
        markers_bool = local_minima(image, connectivity=connectivity) * mask
        footprint = ndi.generate_binary_structure(markers_bool.ndim, connectivity)
        markers = ndi.label(markers_bool, structure=footprint)[0]
    elif not isinstance(markers, (np.ndarray, list, tuple)):
        # not array-like, assume int
        # given int, assume that number of markers *within mask*.
        markers = regular_seeds(image.shape,
                                int(markers / (n_pixels / image.size)))
        markers *= mask
    else:
        markers = np.asanyarray(markers) * mask
        if markers.shape != image.shape:
            message = (f'`markers` (shape {markers.shape}) must have same '
                       f'shape as `image` (shape {image.shape})')
            raise ValueError(message)
    return (image.astype(np.float64),
            markers.astype(np.int32),
            mask.astype(np.int8))


def watershed(image, markers=None, connectivity=1, offset=None, mask=None,
              compactness=0, watershed_line=False):
    """Find watershed basins in `image` flooded from given `markers`.

    Parameters
    ----------
    image : ndarray (2-D, 3-D, ...)
        Data array where the lowest value points are labeled first.
    markers : int, or ndarray of int, same shape as `image`, optional
        The desired number of markers, or an array marking the basins with the
        values to be assigned in the label matrix. Zero means not a marker. If
        ``None`` (no markers given), the local minima of the image are used as
        markers.
    connectivity : ndarray, optional
        An array with the same number of dimensions as `image` whose
        non-zero elements indicate neighbors for connection.
        Following the scipy convention, default is a one-connected array of
        the dimension of the image.
    offset : array_like of shape image.ndim, optional
        offset of the connectivity (one offset per dimension)
    mask : ndarray of bools or 0s and 1s, optional
        Array of same shape as `image`. Only points at which mask == True
        will be labeled.
    compactness : float, optional
        Use compact watershed [3]_ with given compactness parameter.
        Higher values result in more regularly-shaped watershed basins.
    watershed_line : bool, optional
        If watershed_line is True, a one-pixel wide line separates the regions
        obtained by the watershed algorithm. The line has the label 0.

    Returns
    -------
    out : ndarray
        A labeled matrix of the same type and shape as markers

    See Also
    --------
    skimage.segmentation.random_walker : random walker segmentation
        A segmentation algorithm based on anisotropic diffusion, usually
        slower than the watershed but with good results on noisy data and
        boundaries with holes.

    Notes
    -----
    This function implements a watershed algorithm [1]_ [2]_ that apportions
    pixels into marked basins. The algorithm uses a priority queue to hold
    the pixels with the metric for the priority queue being pixel value, then
    the time of entry into the queue - this settles ties in favor of the
    closest marker.

    Some ideas taken from
    Soille, "Automated Basin Delineation from Digital Elevation Models Using
    Mathematical Morphology", Signal Processing 20 (1990) 171-182

    The most important insight in the paper is that entry time onto the queue
    solves two problems: a pixel should be assigned to the neighbor with the
    largest gradient or, if there is no gradient, pixels on a plateau should
    be split between markers on opposite sides.

    This implementation converts all arguments to specific, lowest common
    denominator types, then passes these to a C algorithm.

    Markers can be determined manually, or automatically using for example
    the local minima of the gradient of the image, or the local maxima of the
    distance function to the background for separating overlapping objects
    (see example).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Watershed_%28image_processing%29

    .. [2] http://cmm.ensmp.fr/~beucher/wtshed.html

    .. [3] Peer Neubert & Peter Protzel (2014). Compact Watershed and
           Preemptive SLIC: On Improving Trade-offs of Superpixel Segmentation
           Algorithms. ICPR 2014, pp 996-1001. :DOI:`10.1109/ICPR.2014.181`
           https://www.tu-chemnitz.de/etit/proaut/publications/cws_pSLIC_ICPR.pdf

    Examples
    --------
    The watershed algorithm is useful to separate overlapping objects.

    We first generate an initial image with two overlapping circles:

    >>> x, y = np.indices((80, 80))
    >>> x1, y1, x2, y2 = 28, 28, 44, 52
    >>> r1, r2 = 16, 20
    >>> mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
    >>> mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
    >>> image = np.logical_or(mask_circle1, mask_circle2)

    Next, we want to separate the two circles. We generate markers at the
    maxima of the distance to the background:

    >>> from scipy import ndimage as ndi
    >>> distance = ndi.distance_transform_edt(image)
    >>> from skimage.feature import peak_local_max
    >>> max_coords = peak_local_max(distance, labels=image,
    ...                             footprint=np.ones((3, 3)))
    >>> local_maxima = np.zeros_like(image, dtype=bool)
    >>> local_maxima[tuple(max_coords.T)] = True
    >>> markers = ndi.label(local_maxima)[0]

    Finally, we run the watershed on the image and markers:

    >>> labels = watershed(-distance, markers, mask=image)

    The algorithm works also for 3-D images, and can be used for example to
    separate overlapping spheres.
    """
    image, markers, mask = _validate_inputs(image, markers, mask, connectivity)
    connectivity, offset = _validate_connectivity(image.ndim, connectivity,
                                                  offset)

    # pad the image, markers, and mask so that we can use the mask to
    # keep from running off the edges
    pad_width = [(p, p) for p in offset]
    image = np.pad(image, pad_width, mode='constant')
    mask = np.pad(mask, pad_width, mode='constant').ravel()
    output = np.pad(markers, pad_width, mode='constant')

    flat_neighborhood = _offsets_to_raveled_neighbors(
        image.shape, connectivity, center=offset)
    marker_locations = np.flatnonzero(output)
    image_strides = np.array(image.strides, dtype=np.intp) // image.itemsize

    _watershed_cy.watershed_raveled(image.ravel(),
                                    marker_locations, flat_neighborhood,
                                    mask, image_strides, compactness,
                                    output.ravel(),
                                    watershed_line)

    output = crop(output, pad_width, copy=True)

    return output
