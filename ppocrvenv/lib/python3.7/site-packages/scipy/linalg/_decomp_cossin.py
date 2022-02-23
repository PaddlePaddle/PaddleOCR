# -*- coding: utf-8 -*-
from collections.abc import Iterable
import numpy as np

from scipy._lib._util import _asarray_validated
from scipy.linalg import block_diag, LinAlgError
from .lapack import _compute_lwork, get_lapack_funcs

__all__ = ['cossin']


def cossin(X, p=None, q=None, separate=False,
           swap_sign=False, compute_u=True, compute_vh=True):
    """
    Compute the cosine-sine (CS) decomposition of an orthogonal/unitary matrix.

    X is an ``(m, m)`` orthogonal/unitary matrix, partitioned as the following
    where upper left block has the shape of ``(p, q)``::

                                   ┌                   ┐
                                   │ I  0  0 │ 0  0  0 │
        ┌           ┐   ┌         ┐│ 0  C  0 │ 0 -S  0 │┌         ┐*
        │ X11 │ X12 │   │ U1 │    ││ 0  0  0 │ 0  0 -I ││ V1 │    │
        │ ────┼──── │ = │────┼────││─────────┼─────────││────┼────│
        │ X21 │ X22 │   │    │ U2 ││ 0  0  0 │ I  0  0 ││    │ V2 │
        └           ┘   └         ┘│ 0  S  0 │ 0  C  0 │└         ┘
                                   │ 0  0  I │ 0  0  0 │
                                   └                   ┘

    ``U1``, ``U2``, ``V1``, ``V2`` are square orthogonal/unitary matrices of
    dimensions ``(p,p)``, ``(m-p,m-p)``, ``(q,q)``, and ``(m-q,m-q)``
    respectively, and ``C`` and ``S`` are ``(r, r)`` nonnegative diagonal
    matrices satisfying ``C^2 + S^2 = I`` where ``r = min(p, m-p, q, m-q)``.

    Moreover, the rank of the identity matrices are ``min(p, q) - r``,
    ``min(p, m - q) - r``, ``min(m - p, q) - r``, and ``min(m - p, m - q) - r``
    respectively.

    X can be supplied either by itself and block specifications p, q or its
    subblocks in an iterable from which the shapes would be derived. See the
    examples below.

    Parameters
    ----------
    X : array_like, iterable
        complex unitary or real orthogonal matrix to be decomposed, or iterable
        of subblocks ``X11``, ``X12``, ``X21``, ``X22``, when ``p``, ``q`` are
        omitted.
    p : int, optional
        Number of rows of the upper left block ``X11``, used only when X is
        given as an array.
    q : int, optional
        Number of columns of the upper left block ``X11``, used only when X is
        given as an array.
    separate : bool, optional
        if ``True``, the low level components are returned instead of the
        matrix factors, i.e. ``(u1,u2)``, ``theta``, ``(v1h,v2h)`` instead of
        ``u``, ``cs``, ``vh``.
    swap_sign : bool, optional
        if ``True``, the ``-S``, ``-I`` block will be the bottom left,
        otherwise (by default) they will be in the upper right block.
    compute_u : bool, optional
        if ``False``, ``u`` won't be computed and an empty array is returned.
    compute_vh : bool, optional
        if ``False``, ``vh`` won't be computed and an empty array is returned.

    Returns
    -------
    u : ndarray
        When ``compute_u=True``, contains the block diagonal orthogonal/unitary
        matrix consisting of the blocks ``U1`` (``p`` x ``p``) and ``U2``
        (``m-p`` x ``m-p``) orthogonal/unitary matrices. If ``separate=True``,
        this contains the tuple of ``(U1, U2)``.
    cs : ndarray
        The cosine-sine factor with the structure described above.
         If ``separate=True``, this contains the ``theta`` array containing the
         angles in radians.
    vh : ndarray
        When ``compute_vh=True`, contains the block diagonal orthogonal/unitary
        matrix consisting of the blocks ``V1H`` (``q`` x ``q``) and ``V2H``
        (``m-q`` x ``m-q``) orthogonal/unitary matrices. If ``separate=True``,
        this contains the tuple of ``(V1H, V2H)``.

    Examples
    --------
    >>> from scipy.linalg import cossin
    >>> from scipy.stats import unitary_group
    >>> x = unitary_group.rvs(4)
    >>> u, cs, vdh = cossin(x, p=2, q=2)
    >>> np.allclose(x, u @ cs @ vdh)
    True

    Same can be entered via subblocks without the need of ``p`` and ``q``. Also
    let's skip the computation of ``u``

    >>> ue, cs, vdh = cossin((x[:2, :2], x[:2, 2:], x[2:, :2], x[2:, 2:]),
    ...                      compute_u=False)
    >>> print(ue)
    []
    >>> np.allclose(x, u @ cs @ vdh)
    True

    References
    ----------
    .. [1] : Brian D. Sutton. Computing the complete CS decomposition. Numer.
           Algorithms, 50(1):33-65, 2009.

    """

    if p or q:
        p = 1 if p is None else int(p)
        q = 1 if q is None else int(q)
        X = _asarray_validated(X, check_finite=True)
        if not np.equal(*X.shape):
            raise ValueError("Cosine Sine decomposition only supports square"
                             " matrices, got {}".format(X.shape))
        m = X.shape[0]
        if p >= m or p <= 0:
            raise ValueError("invalid p={}, 0<p<{} must hold"
                             .format(p, X.shape[0]))
        if q >= m or q <= 0:
            raise ValueError("invalid q={}, 0<q<{} must hold"
                             .format(q, X.shape[0]))

        x11, x12, x21, x22 = X[:p, :q], X[:p, q:], X[p:, :q], X[p:, q:]
    elif not isinstance(X, Iterable):
        raise ValueError("When p and q are None, X must be an Iterable"
                         " containing the subblocks of X")
    else:
        if len(X) != 4:
            raise ValueError("When p and q are None, exactly four arrays"
                             " should be in X, got {}".format(len(X)))

        x11, x12, x21, x22 = [np.atleast_2d(x) for x in X]
        for name, block in zip(["x11", "x12", "x21", "x22"],
                               [x11, x12, x21, x22]):
            if block.shape[1] == 0:
                raise ValueError("{} can't be empty".format(name))
        p, q = x11.shape
        mmp, mmq = x22.shape

        if x12.shape != (p, mmq):
            raise ValueError("Invalid x12 dimensions: desired {}, "
                             "got {}".format((p, mmq), x12.shape))

        if x21.shape != (mmp, q):
            raise ValueError("Invalid x21 dimensions: desired {}, "
                             "got {}".format((mmp, q), x21.shape))

        if p + mmp != q + mmq:
            raise ValueError("The subblocks have compatible sizes but "
                             "don't form a square array (instead they form a"
                             " {}x{} array). This might be due to missing "
                             "p, q arguments.".format(p + mmp, q + mmq))

        m = p + mmp

    cplx = any([np.iscomplexobj(x) for x in [x11, x12, x21, x22]])
    driver = "uncsd" if cplx else "orcsd"
    csd, csd_lwork = get_lapack_funcs([driver, driver + "_lwork"],
                                      [x11, x12, x21, x22])
    lwork = _compute_lwork(csd_lwork, m=m, p=p, q=q)
    lwork_args = ({'lwork': lwork[0], 'lrwork': lwork[1]} if cplx else
                  {'lwork': lwork})
    *_, theta, u1, u2, v1h, v2h, info = csd(x11=x11, x12=x12, x21=x21, x22=x22,
                                            compute_u1=compute_u,
                                            compute_u2=compute_u,
                                            compute_v1t=compute_vh,
                                            compute_v2t=compute_vh,
                                            trans=False, signs=swap_sign,
                                            **lwork_args)

    method_name = csd.typecode + driver
    if info < 0:
        raise ValueError('illegal value in argument {} of internal {}'
                         .format(-info, method_name))
    if info > 0:
        raise LinAlgError("{} did not converge: {}".format(method_name, info))

    if separate:
        return (u1, u2), theta, (v1h, v2h)

    U = block_diag(u1, u2)
    VDH = block_diag(v1h, v2h)

    # Construct the middle factor CS
    c = np.diag(np.cos(theta))
    s = np.diag(np.sin(theta))
    r = min(p, q, m - p, m - q)
    n11 = min(p, q) - r
    n12 = min(p, m - q) - r
    n21 = min(m - p, q) - r
    n22 = min(m - p, m - q) - r
    Id = np.eye(np.max([n11, n12, n21, n22, r]), dtype=theta.dtype)
    CS = np.zeros((m, m), dtype=theta.dtype)

    CS[:n11, :n11] = Id[:n11, :n11]

    xs = n11 + r
    xe = n11 + r + n12
    ys = n11 + n21 + n22 + 2 * r
    ye = n11 + n21 + n22 + 2 * r + n12
    CS[xs: xe, ys:ye] = Id[:n12, :n12] if swap_sign else -Id[:n12, :n12]

    xs = p + n22 + r
    xe = p + n22 + r + + n21
    ys = n11 + r
    ye = n11 + r + n21
    CS[xs:xe, ys:ye] = -Id[:n21, :n21] if swap_sign else Id[:n21, :n21]

    CS[p:p + n22, q:q + n22] = Id[:n22, :n22]
    CS[n11:n11 + r, n11:n11 + r] = c
    CS[p + n22:p + n22 + r, r + n21 + n22:2 * r + n21 + n22] = c

    xs = n11
    xe = n11 + r
    ys = n11 + n21 + n22 + r
    ye = n11 + n21 + n22 + 2 * r
    CS[xs:xe, ys:ye] = s if swap_sign else -s

    CS[p + n22:p + n22 + r, n11:n11 + r] = -s if swap_sign else s

    return U, CS, VDH
