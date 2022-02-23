import numpy as np
import scipy.sparse as sparse
from ._contingency_table import contingency_table
from .._shared.utils import check_shape_equality

__all__ = ['variation_of_information']


def variation_of_information(image0=None, image1=None, *, table=None,
                             ignore_labels=()):
    """Return symmetric conditional entropies associated with the VI. [1]_

    The variation of information is defined as VI(X,Y) = H(X|Y) + H(Y|X).
    If X is the ground-truth segmentation, then H(X|Y) can be interpreted
    as the amount of under-segmentation and H(X|Y) as the amount
    of over-segmentation. In other words, a perfect over-segmentation
    will have H(X|Y)=0 and a perfect under-segmentation will have H(Y|X)=0.

    Parameters
    ----------
    image0, image1 : ndarray of int
        Label images / segmentations, must have same shape.
    table : scipy.sparse array in csr format, optional
        A contingency table built with skimage.evaluate.contingency_table.
        If None, it will be computed with skimage.evaluate.contingency_table.
        If given, the entropies will be computed from this table and any images
        will be ignored.
    ignore_labels : sequence of int, optional
        Labels to ignore. Any part of the true image labeled with any of these
        values will not be counted in the score.

    Returns
    -------
    vi : ndarray of float, shape (2,)
        The conditional entropies of image1|image0 and image0|image1.

    References
    ----------
    .. [1] Marina Meilă (2007), Comparing clusterings—an information based
        distance, Journal of Multivariate Analysis, Volume 98, Issue 5,
        Pages 873-895, ISSN 0047-259X, :DOI:`10.1016/j.jmva.2006.11.013`.
    """
    h0g1, h1g0 = _vi_tables(image0, image1, table=table,
                            ignore_labels=ignore_labels)
    # false splits, false merges
    return np.array([h1g0.sum(), h0g1.sum()])


def _xlogx(x):
    """Compute x * log_2(x).

    We define 0 * log_2(0) = 0

    Parameters
    ----------
    x : ndarray or scipy.sparse.csc_matrix or csr_matrix
        The input array.

    Returns
    -------
    y : same type as x
        Result of x * log_2(x).
    """
    y = x.copy()
    if isinstance(y, sparse.csc_matrix) or isinstance(y, sparse.csr_matrix):
        z = y.data
    else:
        z = np.asarray(y)  # ensure np.matrix converted to np.array
    nz = z.nonzero()
    z[nz] *= np.log2(z[nz])
    return y


def _vi_tables(im_true, im_test, table=None, ignore_labels=()):
    """Compute probability tables used for calculating VI.

    Parameters
    ----------
    im_true, im_test : ndarray of int
        Input label images, any dimensionality.
    table : csr matrix, optional
        Pre-computed contingency table.
    ignore_labels : sequence of int, optional
        Labels to ignore when computing scores.

    Returns
    -------
    hxgy, hygx : ndarray of float
        Per-segment conditional entropies of ``im_true`` given ``im_test`` and
        vice-versa.
    """
    check_shape_equality(im_true, im_test)

    if table is None:
        # normalize, since it is an identity op if already done
        pxy = contingency_table(
            im_true, im_test,
            ignore_labels=ignore_labels, normalize=True
        )

    else:
        pxy = table

    # compute marginal probabilities, converting to 1D array
    px = np.ravel(pxy.sum(axis=1))
    py = np.ravel(pxy.sum(axis=0))

    # use sparse matrix linear algebra to compute VI
    # first, compute the inverse diagonal matrices
    px_inv = sparse.diags(_invert_nonzero(px))
    py_inv = sparse.diags(_invert_nonzero(py))

    # then, compute the entropies
    hygx = -px @ _xlogx(px_inv @ pxy).sum(axis=1)
    hxgy = -_xlogx(pxy @ py_inv).sum(axis=0) @ py

    return list(map(np.asarray, [hxgy, hygx]))


def _invert_nonzero(arr):
    """Compute the inverse of the non-zero elements of arr, not changing 0.

    Parameters
    ----------
    arr : ndarray

    Returns
    -------
    arr_inv : ndarray
        Array containing the inverse of the non-zero elements of arr, and
        zero elsewhere.
    """
    arr_inv = arr.copy()
    nz = np.nonzero(arr)
    arr_inv[nz] = 1 / arr[nz]
    return arr_inv
