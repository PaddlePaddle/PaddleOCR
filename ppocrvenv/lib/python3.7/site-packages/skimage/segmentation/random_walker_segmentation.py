"""
Random walker segmentation algorithm

from *Random walks for image segmentation*, Leo Grady, IEEE Trans
Pattern Anal Mach Intell. 2006 Nov;28(11):1768-83.

Installing pyamg and using the 'cg_mg' mode of random_walker improves
significantly the performance.
"""

import numpy as np
from scipy import sparse, ndimage as ndi

from .._shared import utils
from .._shared.utils import warn

# executive summary for next code block: try to import umfpack from
# scipy, but make sure not to raise a fuss if it fails since it's only
# needed to speed up a few cases.
# See discussions at:
# https://groups.google.com/d/msg/scikit-image/FrM5IGP6wh4/1hp-FtVZmfcJ
# https://stackoverflow.com/questions/13977970/ignore-exceptions-printed-to-stderr-in-del/13977992?noredirect=1#comment28386412_13977992
try:
    from scipy.sparse.linalg.dsolve.linsolve import umfpack
    old_del = umfpack.UmfpackContext.__del__

    def new_del(self):
        try:
            old_del(self)
        except AttributeError:
            pass
    umfpack.UmfpackContext.__del__ = new_del
    UmfpackContext = umfpack.UmfpackContext()
except ImportError:
    UmfpackContext = None

try:
    from pyamg import ruge_stuben_solver
    amg_loaded = True
except ImportError:
    amg_loaded = False

from ..util import img_as_float

from scipy.sparse.linalg import cg, spsolve


def _make_graph_edges_3d(n_x, n_y, n_z):
    """Returns a list of edges for a 3D image.

    Parameters
    ----------
    n_x : integer
        The size of the grid in the x direction.
    n_y : integer
        The size of the grid in the y direction
    n_z : integer
        The size of the grid in the z direction

    Returns
    -------
    edges : (2, N) ndarray
        with the total number of edges::

            N = n_x * n_y * (nz - 1) +
                n_x * (n_y - 1) * nz +
                (n_x - 1) * n_y * nz

        Graph edges with each column describing a node-id pair.
    """
    vertices = np.arange(n_x * n_y * n_z).reshape((n_x, n_y, n_z))
    edges_deep = np.vstack((vertices[..., :-1].ravel(),
                            vertices[..., 1:].ravel()))
    edges_right = np.vstack((vertices[:, :-1].ravel(),
                             vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
    edges = np.hstack((edges_deep, edges_right, edges_down))
    return edges


def _compute_weights_3d(data, spacing, beta, eps, multichannel):
    # Weight calculation is main difference in multispectral version
    # Original gradient**2 replaced with sum of gradients ** 2
    gradients = np.concatenate(
        [np.diff(data[..., 0], axis=ax).ravel() / spacing[ax]
         for ax in [2, 1, 0] if data.shape[ax] > 1], axis=0) ** 2
    for channel in range(1, data.shape[-1]):
        gradients += np.concatenate(
            [np.diff(data[..., channel], axis=ax).ravel() / spacing[ax]
             for ax in [2, 1, 0] if data.shape[ax] > 1], axis=0) ** 2

    # All channels considered together in this standard deviation
    scale_factor = -beta / (10 * data.std())
    if multichannel:
        # New final term in beta to give == results in trivial case where
        # multiple identical spectra are passed.
        scale_factor /= np.sqrt(data.shape[-1])
    weights = np.exp(scale_factor * gradients)
    weights += eps
    return -weights


def _build_laplacian(data, spacing, mask, beta, multichannel):
    l_x, l_y, l_z = data.shape[:3]
    edges = _make_graph_edges_3d(l_x, l_y, l_z)
    weights = _compute_weights_3d(data, spacing, beta=beta, eps=1.e-10,
                                  multichannel=multichannel)
    if mask is not None:
        # Remove edges of the graph connected to masked nodes, as well
        # as corresponding weights of the edges.
        mask0 = np.hstack([mask[..., :-1].ravel(), mask[:, :-1].ravel(),
                           mask[:-1].ravel()])
        mask1 = np.hstack([mask[..., 1:].ravel(), mask[:, 1:].ravel(),
                           mask[1:].ravel()])
        ind_mask = np.logical_and(mask0, mask1)
        edges, weights = edges[:, ind_mask], weights[ind_mask]

        # Reassign edges labels to 0, 1, ... edges_number - 1
        _, inv_idx = np.unique(edges, return_inverse=True)
        edges = inv_idx.reshape(edges.shape)

    # Build the sparse linear system
    pixel_nb = l_x * l_y * l_z
    i_indices = edges.ravel()
    j_indices = edges[::-1].ravel()
    data = np.hstack((weights, weights))
    lap = sparse.coo_matrix((data, (i_indices, j_indices)),
                            shape=(pixel_nb, pixel_nb))
    lap.setdiag(-np.ravel(lap.sum(axis=0)))
    return lap.tocsr()


def _build_linear_system(data, spacing, labels, nlabels, mask,
                         beta, multichannel):
    """
    Build the matrix A and rhs B of the linear system to solve.
    A and B are two block of the laplacian of the image graph.
    """
    if mask is None:
        labels = labels.ravel()
    else:
        labels = labels[mask]

    indices = np.arange(labels.size)
    seeds_mask = labels > 0
    unlabeled_indices = indices[~seeds_mask]
    seeds_indices = indices[seeds_mask]

    lap_sparse = _build_laplacian(data, spacing, mask=mask,
                                  beta=beta, multichannel=multichannel)

    rows = lap_sparse[unlabeled_indices, :]
    lap_sparse = rows[:, unlabeled_indices]
    B = -rows[:, seeds_indices]

    seeds = labels[seeds_mask]
    seeds_mask = sparse.csc_matrix(np.hstack(
        [np.atleast_2d(seeds == lab).T for lab in range(1, nlabels + 1)]))
    rhs = B.dot(seeds_mask)

    return lap_sparse, rhs


def _solve_linear_system(lap_sparse, B, tol, mode):

    if mode is None:
        mode = 'cg_j'

    if mode == 'cg_mg' and not amg_loaded:
        warn('"cg_mg" not available, it requires pyamg to be installed. '
             'The "cg_j" mode will be used instead.',
             stacklevel=2)
        mode = 'cg_j'

    if mode == 'bf':
        X = spsolve(lap_sparse, B.toarray()).T
    else:
        maxiter = None
        if mode == 'cg':
            if UmfpackContext is None:
                warn('"cg" mode may be slow because UMFPACK is not available. '
                     'Consider building Scipy with UMFPACK or use a '
                     'preconditioned version of CG ("cg_j" or "cg_mg" modes).',
                     stacklevel=2)
            M = None
        elif mode == 'cg_j':
            M = sparse.diags(1.0 / lap_sparse.diagonal())
        else:
            # mode == 'cg_mg'
            lap_sparse = lap_sparse.tocsr()
            ml = ruge_stuben_solver(lap_sparse, coarse_solver='pinv')
            M = ml.aspreconditioner(cycle='V')
            maxiter = 30
        cg_out = [
            cg(lap_sparse, B[:, i].toarray(),
               tol=tol, atol=0, M=M, maxiter=maxiter)
            for i in range(B.shape[1])]
        if np.any([info > 0 for _, info in cg_out]):
            warn("Conjugate gradient convergence to tolerance not achieved. "
                 "Consider decreasing beta to improve system conditionning.",
                 stacklevel=2)
        X = np.asarray([x for x, _ in cg_out])

    return X


def _preprocess(labels):

    label_values, inv_idx = np.unique(labels, return_inverse=True)

    if not (label_values == 0).any():
        warn('Random walker only segments unlabeled areas, where '
             'labels == 0. No zero valued areas in labels were '
             'found. Returning provided labels.',
             stacklevel=2)

        return labels, None, None, None, None

    # If some labeled pixels are isolated inside pruned zones, prune them
    # as well and keep the labels for the final output

    null_mask = labels == 0
    pos_mask = labels > 0
    mask = labels >= 0

    fill = ndi.binary_propagation(null_mask, mask=mask)
    isolated = np.logical_and(pos_mask, np.logical_not(fill))

    pos_mask[isolated] = False

    # If the array has pruned zones, be sure that no isolated pixels
    # exist between pruned zones (they could not be determined)
    if label_values[0] < 0 or np.any(isolated):
        isolated = np.logical_and(
            np.logical_not(ndi.binary_propagation(pos_mask, mask=mask)),
            null_mask)

        labels[isolated] = -1
        if np.all(isolated[null_mask]):
            warn('All unlabeled pixels are isolated, they could not be '
                 'determined by the random walker algorithm.',
                 stacklevel=2)
            return labels, None, None, None, None

        mask[isolated] = False
        mask = np.atleast_3d(mask)
    else:
        mask = None

    # Reorder label values to have consecutive integers (no gaps)
    zero_idx = np.searchsorted(label_values, 0)
    labels = np.atleast_3d(inv_idx.reshape(labels.shape) - zero_idx)

    nlabels = label_values[zero_idx + 1:].shape[0]

    inds_isolated_seeds = np.nonzero(isolated)
    isolated_values = labels[inds_isolated_seeds]

    return labels, nlabels, mask, inds_isolated_seeds, isolated_values


@utils.channel_as_last_axis(multichannel_output=False)
@utils.deprecate_multichannel_kwarg(multichannel_position=6)
def random_walker(data, labels, beta=130, mode='cg_j', tol=1.e-3, copy=True,
                  multichannel=False, return_full_prob=False, spacing=None,
                  *, prob_tol=1e-3, channel_axis=None):
    """Random walker algorithm for segmentation from markers.

    Random walker algorithm is implemented for gray-level or multichannel
    images.

    Parameters
    ----------
    data : array_like
        Image to be segmented in phases. Gray-level `data` can be two- or
        three-dimensional; multichannel data can be three- or four-
        dimensional with `channel_axis` specifying the dimension containing
        channels. Data spacing is assumed isotropic unless the `spacing`
        keyword argument is used.
    labels : array of ints, of same shape as `data` without channels dimension
        Array of seed markers labeled with different positive integers
        for different phases. Zero-labeled pixels are unlabeled pixels.
        Negative labels correspond to inactive pixels that are not taken
        into account (they are removed from the graph). If labels are not
        consecutive integers, the labels array will be transformed so that
        labels are consecutive. In the multichannel case, `labels` should have
        the same shape as a single channel of `data`, i.e. without the final
        dimension denoting channels.
    beta : float, optional
        Penalization coefficient for the random walker motion
        (the greater `beta`, the more difficult the diffusion).
    mode : string, available options {'cg', 'cg_j', 'cg_mg', 'bf'}
        Mode for solving the linear system in the random walker algorithm.

        - 'bf' (brute force): an LU factorization of the Laplacian is
          computed. This is fast for small images (<1024x1024), but very slow
          and memory-intensive for large images (e.g., 3-D volumes).
        - 'cg' (conjugate gradient): the linear system is solved iteratively
          using the Conjugate Gradient method from scipy.sparse.linalg. This is
          less memory-consuming than the brute force method for large images,
          but it is quite slow.
        - 'cg_j' (conjugate gradient with Jacobi preconditionner): the
          Jacobi preconditionner is applyed during the Conjugate
          gradient method iterations. This may accelerate the
          convergence of the 'cg' method.
        - 'cg_mg' (conjugate gradient with multigrid preconditioner): a
          preconditioner is computed using a multigrid solver, then the
          solution is computed with the Conjugate Gradient method. This mode
          requires that the pyamg module is installed.
    tol : float, optional
        Tolerance to achieve when solving the linear system using
        the conjugate gradient based modes ('cg', 'cg_j' and 'cg_mg').
    copy : bool, optional
        If copy is False, the `labels` array will be overwritten with
        the result of the segmentation. Use copy=False if you want to
        save on memory.
    multichannel : bool, optional
        If True, input data is parsed as multichannel data (see 'data' above
        for proper input format in this case). This argument is deprecated:
        specify `channel_axis` instead.
    return_full_prob : bool, optional
        If True, the probability that a pixel belongs to each of the
        labels will be returned, instead of only the most likely
        label.
    spacing : iterable of floats, optional
        Spacing between voxels in each spatial dimension. If `None`, then
        the spacing between pixels/voxels in each dimension is assumed 1.
    prob_tol : float, optional
        Tolerance on the resulting probability to be in the interval [0, 1].
        If the tolerance is not satisfied, a warning is displayed.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    output : ndarray
        * If `return_full_prob` is False, array of ints of same shape
          and data type as `labels`, in which each pixel has been
          labeled according to the marker that reached the pixel first
          by anisotropic diffusion.
        * If `return_full_prob` is True, array of floats of shape
          `(nlabels, labels.shape)`. `output[label_nb, i, j]` is the
          probability that label `label_nb` reaches the pixel `(i, j)`
          first.

    See Also
    --------
    skimage.morphology.watershed : watershed segmentation
        A segmentation algorithm based on mathematical morphology
        and "flooding" of regions from markers.

    Notes
    -----
    Multichannel inputs are scaled with all channel data combined. Ensure all
    channels are separately normalized prior to running this algorithm.

    The `spacing` argument is specifically for anisotropic datasets, where
    data points are spaced differently in one or more spatial dimensions.
    Anisotropic data is commonly encountered in medical imaging.

    The algorithm was first proposed in [1]_.

    The algorithm solves the diffusion equation at infinite times for
    sources placed on markers of each phase in turn. A pixel is labeled with
    the phase that has the greatest probability to diffuse first to the pixel.

    The diffusion equation is solved by minimizing x.T L x for each phase,
    where L is the Laplacian of the weighted graph of the image, and x is
    the probability that a marker of the given phase arrives first at a pixel
    by diffusion (x=1 on markers of the phase, x=0 on the other markers, and
    the other coefficients are looked for). Each pixel is attributed the label
    for which it has a maximal value of x. The Laplacian L of the image
    is defined as:

       - L_ii = d_i, the number of neighbors of pixel i (the degree of i)
       - L_ij = -w_ij if i and j are adjacent pixels

    The weight w_ij is a decreasing function of the norm of the local gradient.
    This ensures that diffusion is easier between pixels of similar values.

    When the Laplacian is decomposed into blocks of marked and unmarked
    pixels::

        L = M B.T
            B A

    with first indices corresponding to marked pixels, and then to unmarked
    pixels, minimizing x.T L x for one phase amount to solving::

        A x = - B x_m

    where x_m = 1 on markers of the given phase, and 0 on other markers.
    This linear system is solved in the algorithm using a direct method for
    small images, and an iterative method for larger images.

    References
    ----------
    .. [1] Leo Grady, Random walks for image segmentation, IEEE Trans Pattern
        Anal Mach Intell. 2006 Nov;28(11):1768-83.
        :DOI:`10.1109/TPAMI.2006.233`.

    Examples
    --------
    >>> rng = np.random.default_rng()
    >>> a = np.zeros((10, 10)) + 0.2 * rng.random((10, 10))
    >>> a[5:8, 5:8] += 1
    >>> b = np.zeros_like(a, dtype=np.int32)
    >>> b[3, 3] = 1  # Marker for first phase
    >>> b[6, 6] = 2  # Marker for second phase
    >>> random_walker(a, b)  # doctest: +SKIP
    array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],
           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],
           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=int32)

    """
    # Parse input data
    if mode not in ('cg_mg', 'cg', 'bf', 'cg_j', None):
        raise ValueError(
            f"{mode} is not a valid mode. Valid modes are 'cg_mg', "
            f"'cg', 'cg_j', 'bf', and None")

    # Spacing kwarg checks
    if spacing is None:
        spacing = np.ones(3)
    elif len(spacing) == labels.ndim:
        if len(spacing) == 2:
            # Need a dummy spacing for singleton 3rd dim
            spacing = np.r_[spacing, 1.]
        spacing = np.asarray(spacing)
    else:
        raise ValueError('Input argument `spacing` incorrect, should be an '
                         'iterable with one number per spatial dimension.')

    # This algorithm expects 4-D arrays of floats, where the first three
    # dimensions are spatial and the final denotes channels. 2-D images have
    # a singleton placeholder dimension added for the third spatial dimension,
    # and single channel images likewise have a singleton added for channels.
    # The following block ensures valid input and coerces it to the correct
    # form.
    multichannel = channel_axis is not None
    if not multichannel:
        if data.ndim not in (2, 3):
            raise ValueError('For non-multichannel input, data must be of '
                             'dimension 2 or 3.')
        if data.shape != labels.shape:
            raise ValueError('Incompatible data and labels shapes.')
        data = np.atleast_3d(img_as_float(data))[..., np.newaxis]
    else:
        if data.ndim not in (3, 4):
            raise ValueError('For multichannel input, data must have 3 or 4 '
                             'dimensions.')
        if data.shape[:-1] != labels.shape:
            raise ValueError('Incompatible data and labels shapes.')
        data = img_as_float(data)
        if data.ndim == 3:  # 2D multispectral, needs singleton in 3rd axis
            data = data[:, :, np.newaxis, :]

    labels_shape = labels.shape
    labels_dtype = labels.dtype

    if copy:
        labels = np.copy(labels)

    (labels, nlabels, mask,
     inds_isolated_seeds, isolated_values) = _preprocess(labels)

    if isolated_values is None:
        # No non isolated zero valued areas in labels were
        # found. Returning provided labels.
        if return_full_prob:
            # Return the concatenation of the masks of each unique label
            return np.concatenate([np.atleast_3d(labels == lab)
                                   for lab in np.unique(labels) if lab > 0],
                                  axis=-1)
        return labels

    # Build the linear system (lap_sparse, B)
    lap_sparse, B = _build_linear_system(data, spacing, labels, nlabels, mask,
                                         beta, multichannel)

    # Solve the linear system lap_sparse X = B
    # where X[i, j] is the probability that a marker of label i arrives
    # first at pixel j by anisotropic diffusion.
    X = _solve_linear_system(lap_sparse, B, tol, mode)

    if X.min() < -prob_tol or X.max() > 1 + prob_tol:
        warn('The probability range is outside [0, 1] given the tolerance '
             '`prob_tol`. Consider decreasing `beta` and/or decreasing '
             '`tol`.')

    # Build the output according to return_full_prob value
    # Put back labels of isolated seeds
    labels[inds_isolated_seeds] = isolated_values
    labels = labels.reshape(labels_shape)

    mask = labels == 0
    mask[inds_isolated_seeds] = False

    if return_full_prob:
        out = np.zeros((nlabels,) + labels_shape)
        for lab, (label_prob, prob) in enumerate(zip(out, X), start=1):
            label_prob[mask] = prob
            label_prob[labels == lab] = 1
    else:
        X = np.argmax(X, axis=0) + 1
        out = labels.astype(labels_dtype)
        out[mask] = X

    return out
