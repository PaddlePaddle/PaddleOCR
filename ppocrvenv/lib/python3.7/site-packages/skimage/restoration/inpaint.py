import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.ndimage as ndi
from scipy.ndimage.filters import laplace

import skimage
from .._shared import utils
from ..measure import label
from ._inpaint import _build_matrix_inner


def _get_neighborhood(nd_idx, radius, nd_shape):
    bounds_lo = np.maximum(nd_idx - radius, 0)
    bounds_hi = np.minimum(nd_idx + radius + 1, nd_shape)
    return bounds_lo, bounds_hi


def _get_neigh_coef(shape, center, dtype=float):
    # Create biharmonic coefficients ndarray
    neigh_coef = np.zeros(shape, dtype=dtype)
    neigh_coef[center] = 1
    neigh_coef = laplace(laplace(neigh_coef))

    # extract non-zero locations and values
    coef_idx = np.where(neigh_coef)
    coef_vals = neigh_coef[coef_idx]

    coef_idx = np.stack(coef_idx, axis=0)
    return neigh_coef, coef_idx, coef_vals


def _inpaint_biharmonic_single_region(image, mask, out, neigh_coef_full,
                                      coef_vals, raveled_offsets):
    """Solve a (sparse) linear system corresponding to biharmonic inpainting.

    This function creates a linear system of the form:

    ``A @ u = b``

    where ``A`` is a sparse matrix, ``b`` is a vector enforcing smoothness and
    boundary constraints and ``u`` is the vector of inpainted values to be
    (uniquely) determined by solving the linear system.

    ``A`` is a sparse matrix of shape (n_mask, n_mask) where `n_mask``
    corresponds to the number of non-zero values in ``mask`` (i.e. the number
    of pixels to be inpainted). Each row in A will have a number of non-zero
    values equal to the number of non-zero values in the biharmonic kernel,
    ``neigh_coef_full``. In practice, biharmonic kernels with reduced extent
    are used at the image borders. This matrix, ``A`` is the same for all
    image channels (since the same inpainting mask is currently used for all
    channels).

    ``u`` is a dense matrix of shape ``(n_mask, n_channels)`` and represents
    the vector of unknown values for each channel.

    ``b`` is a dense matrix of shape ``(n_mask, n_channels)`` and represents
    the desired output of convolving the solution with the biharmonic kernel.
    At mask locations where there is no overlap with known values, ``b`` will
    have a value of 0. This enforces the biharmonic smoothness constraint in
    the interior of inpainting regions. For regions near the boundary that
    overlap with known values, the entries in ``b`` enforce boundary conditions
    designed to avoid discontinuity with the known values.
    """

    n_channels = out.shape[-1]
    radius = neigh_coef_full.shape[0] // 2

    edge_mask = np.ones(mask.shape, dtype=bool)
    edge_mask[(slice(radius, -radius),) * mask.ndim] = 0
    boundary_mask = edge_mask * mask
    center_mask = ~edge_mask * mask

    boundary_pts = np.where(boundary_mask)
    boundary_i = np.flatnonzero(boundary_mask)
    center_i = np.flatnonzero(center_mask)
    mask_i = np.concatenate((boundary_i, center_i))

    center_pts = np.where(center_mask)
    mask_pts = tuple(
        [np.concatenate((b, c)) for b, c in zip(boundary_pts, center_pts)]
    )

    # Use convolution to predetermine the number of non-zero entries in the
    # sparse system matrix.
    structure = neigh_coef_full != 0
    tmp = ndi.convolve(mask, structure, output=np.uint8, mode='constant')
    nnz_matrix = tmp[mask].sum()

    # Need to estimate the number of zeros for the right hand side vector.
    # The computation below will slightly overestimate the true number of zeros
    # due to edge effects (the kernel itself gets shrunk in size near the
    # edges, but that isn't accounted for here). We can trim any excess entries
    # later.
    n_mask = np.count_nonzero(mask)
    n_struct = np.count_nonzero(structure)
    nnz_rhs_vector_max = n_mask - np.count_nonzero(tmp == n_struct)

    # pre-allocate arrays storing sparse matrix indices and values
    row_idx_known = np.empty(nnz_rhs_vector_max, dtype=np.intp)
    data_known = np.zeros((nnz_rhs_vector_max, n_channels), dtype=out.dtype)
    row_idx_unknown = np.empty(nnz_matrix, dtype=np.intp)
    col_idx_unknown = np.empty(nnz_matrix, dtype=np.intp)
    data_unknown = np.empty(nnz_matrix, dtype=out.dtype)

    # cache the various small, non-square Laplacians used near the boundary
    coef_cache = {}

    # Iterate over masked points near the boundary
    mask_flat = mask.reshape(-1)
    out_flat = np.ascontiguousarray(out.reshape((-1, n_channels)))
    idx_known = 0
    idx_unknown = 0
    mask_pt_n = -1
    boundary_pts = np.stack(boundary_pts, axis=1)
    for mask_pt_n, nd_idx in enumerate(boundary_pts):
        # Get bounded neighborhood of selected radius
        b_lo, b_hi = _get_neighborhood(nd_idx, radius, mask.shape)

        # Create (truncated) biharmonic coefficients ndarray
        coef_shape = tuple(b_hi - b_lo)
        coef_center = tuple(nd_idx - b_lo)
        coef_idx, coefs = coef_cache.get((coef_shape, coef_center),
                                         (None, None))
        if coef_idx is None:
            _ , coef_idx, coefs = _get_neigh_coef(coef_shape,
                                                  coef_center,
                                                  dtype=out.dtype)
            coef_cache[(coef_shape, coef_center)] = (coef_idx, coefs)

        # compute corresponding 1d indices into the mask
        coef_idx = coef_idx + b_lo[:, np.newaxis]
        index1d = np.ravel_multi_index(coef_idx, mask.shape)

        # Iterate over masked point's neighborhood
        nvals = 0
        for coef, i in zip(coefs, index1d):
            if mask_flat[i]:
                row_idx_unknown[idx_unknown] = mask_pt_n
                col_idx_unknown[idx_unknown] = i
                data_unknown[idx_unknown] = coef
                idx_unknown += 1
            else:
                data_known[idx_known, :] -= coef * out_flat[i, :]
                nvals += 1
        if nvals:
            row_idx_known[idx_known] = mask_pt_n
            idx_known += 1

    # Call an efficient Cython-based implementation for all interior points
    row_start = mask_pt_n + 1
    known_start_idx = idx_known
    unknown_start_idx = idx_unknown
    nnz_rhs = _build_matrix_inner(
        # starting indices
        row_start, known_start_idx, unknown_start_idx,
        # input arrays
        center_i, raveled_offsets, coef_vals, mask_flat,
        out_flat,
        # output arrays
        row_idx_known, data_known, row_idx_unknown, col_idx_unknown,
        data_unknown
    )

    # trim RHS vector values and indices to the exact length
    row_idx_known = row_idx_known[:nnz_rhs]
    data_known = data_known[:nnz_rhs, :]

    # Form sparse matrix of unknown values
    sp_shape = (n_mask, out.size)
    matrix_unknown = sparse.coo_matrix(
        (data_unknown, (row_idx_unknown, col_idx_unknown)), shape=sp_shape
    ).tocsr()

    # Solve linear system for masked points
    matrix_unknown = matrix_unknown[:, mask_i]

    # dense vectors representing the right hand side for each channel
    rhs = np.zeros((n_mask, n_channels), dtype=out.dtype)
    rhs[row_idx_known, :] = data_known

    # set use_umfpack to False so float32 data is supported
    result = spsolve(matrix_unknown, rhs, use_umfpack=False,
                     permc_spec='MMD_ATA')
    if result.ndim == 1:
        result = result[:, np.newaxis]

    out[mask_pts] = result
    return out


@utils.channel_as_last_axis()
@utils.deprecate_multichannel_kwarg(multichannel_position=2)
def inpaint_biharmonic(image, mask, multichannel=False, *,
                       split_into_regions=False, channel_axis=None):
    """Inpaint masked points in image with biharmonic equations.

    Parameters
    ----------
    image : (M[, N[, ..., P]][, C]) ndarray
        Input image.
    mask : (M[, N[, ..., P]]) ndarray
        Array of pixels to be inpainted. Have to be the same shape as one
        of the 'image' channels. Unknown pixels have to be represented with 1,
        known pixels - with 0.
    multichannel : boolean, optional
        If True, the last `image` dimension is considered as a color channel,
        otherwise as spatial. This argument is deprecated: specify
        `channel_axis` instead.
    split_into_regions : boolean, optional
        If True, inpainting is performed on a region-by-region basis. This is
        likely to be slower, but will have reduced memory requirements.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (M[, N[, ..., P]][, C]) ndarray
        Input image with masked pixels inpainted.

    References
    ----------
    .. [1]  S.B.Damelin and N.S.Hoang. "On Surface Completion and Image
            Inpainting by Biharmonic Functions: Numerical Aspects",
            International Journal of Mathematics and Mathematical Sciences,
            Vol. 2018, Article ID 3950312
            :DOI:`10.1155/2018/3950312`
    .. [2]  C. K. Chui and H. N. Mhaskar, MRA Contextual-Recovery Extension of
            Smooth Functions on Manifolds, Appl. and Comp. Harmonic Anal.,
            28 (2010), 104-113,
            :DOI:`10.1016/j.acha.2009.04.004`

    Examples
    --------
    >>> img = np.tile(np.square(np.linspace(0, 1, 5)), (5, 1))
    >>> mask = np.zeros_like(img)
    >>> mask[2, 2:] = 1
    >>> mask[1, 3:] = 1
    >>> mask[0, 4:] = 1
    >>> out = inpaint_biharmonic(img, mask)
    """

    if image.ndim < 1:
        raise ValueError('Input array has to be at least 1D')

    multichannel = channel_axis is not None
    img_baseshape = image.shape[:-1] if multichannel else image.shape
    if img_baseshape != mask.shape:
        raise ValueError('Input arrays have to be the same shape')

    if np.ma.isMaskedArray(image):
        raise TypeError('Masked arrays are not supported')

    image = skimage.img_as_float(image)

    # float16->float32 and float128->float64
    float_dtype = utils._supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    mask = mask.astype(bool, copy=False)
    if not multichannel:
        image = image[..., np.newaxis]
    out = np.copy(image)

    # Create biharmonic coefficients ndarray
    radius = 2
    coef_shape = (2 * radius + 1,) * mask.ndim
    coef_center = (radius,) * mask.ndim
    neigh_coef_full, coef_idx, coef_vals = _get_neigh_coef(coef_shape,
                                                           coef_center,
                                                           dtype=out.dtype)

    # stride for the last spatial dimension
    channel_stride_bytes = out.strides[-2]

    # offsets to all neighboring non-zero elements in the footprint
    offsets = coef_idx - radius

    # determine per-channel intensity limits
    known_points = image[~mask]
    limits = (known_points.min(axis=0), known_points.max(axis=0))

    if split_into_regions:
        # Split inpainting mask into independent regions
        kernel = ndi.generate_binary_structure(mask.ndim, 1)
        mask_dilated = ndi.binary_dilation(mask, structure=kernel)
        mask_labeled = label(mask_dilated)
        mask_labeled *= mask

        bbox_slices = ndi.find_objects(mask_labeled)

        for idx_region, bb_slice in enumerate(bbox_slices, 1):
            # expand object bounding boxes by the biharmonic kernel radius
            roi_sl = tuple(slice(max(sl.start - radius, 0),
                                 min(sl.stop + radius, size))
                           for sl, size in zip(bb_slice, mask_labeled.shape))
            # extract only the region surrounding the label of interest
            mask_region = mask_labeled[roi_sl] == idx_region
            # add slice for axes
            roi_sl += (slice(None), )
            # copy for contiguity and to account for possible ROI overlap
            otmp = out[roi_sl].copy()

            # compute raveled offsets for the ROI
            ostrides = np.array([s // channel_stride_bytes
                                 for s in otmp[..., 0].strides])
            raveled_offsets = np.sum(offsets * ostrides[..., np.newaxis],
                                     axis=0)

            _inpaint_biharmonic_single_region(
                image[roi_sl], mask_region, otmp,
                neigh_coef_full, coef_vals, raveled_offsets
            )
            # assign output to the
            out[roi_sl] = otmp
    else:
        # compute raveled offsets for output image
        ostrides = np.array([s // channel_stride_bytes
                             for s in out[..., 0].strides])
        raveled_offsets = np.sum(offsets * ostrides[..., np.newaxis], axis=0)

        _inpaint_biharmonic_single_region(
            image, mask, out, neigh_coef_full, coef_vals, raveled_offsets
        )

    # Handle enormous values on a per-channel basis
    np.clip(out, a_min=limits[0], a_max=limits[1], out=out)

    if not multichannel:
        out = out[..., 0]

    return out
