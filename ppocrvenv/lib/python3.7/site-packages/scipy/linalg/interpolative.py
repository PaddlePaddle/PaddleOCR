#******************************************************************************
#   Copyright (C) 2013 Kenneth L. Ho
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer. Redistributions in binary
#   form must reproduce the above copyright notice, this list of conditions and
#   the following disclaimer in the documentation and/or other materials
#   provided with the distribution.
#
#   None of the names of the copyright holders may be used to endorse or
#   promote products derived from this software without specific prior written
#   permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#   POSSIBILITY OF SUCH DAMAGE.
#******************************************************************************

# Python module for interfacing with `id_dist`.

r"""
======================================================================
Interpolative matrix decomposition (:mod:`scipy.linalg.interpolative`)
======================================================================

.. moduleauthor:: Kenneth L. Ho <klho@stanford.edu>

.. versionadded:: 0.13

.. currentmodule:: scipy.linalg.interpolative

An interpolative decomposition (ID) of a matrix :math:`A \in
\mathbb{C}^{m \times n}` of rank :math:`k \leq \min \{ m, n \}` is a
factorization

.. math::
  A \Pi =
  \begin{bmatrix}
   A \Pi_{1} & A \Pi_{2}
  \end{bmatrix} =
  A \Pi_{1}
  \begin{bmatrix}
   I & T
  \end{bmatrix},

where :math:`\Pi = [\Pi_{1}, \Pi_{2}]` is a permutation matrix with
:math:`\Pi_{1} \in \{ 0, 1 \}^{n \times k}`, i.e., :math:`A \Pi_{2} =
A \Pi_{1} T`. This can equivalently be written as :math:`A = BP`,
where :math:`B = A \Pi_{1}` and :math:`P = [I, T] \Pi^{\mathsf{T}}`
are the *skeleton* and *interpolation matrices*, respectively.

If :math:`A` does not have exact rank :math:`k`, then there exists an
approximation in the form of an ID such that :math:`A = BP + E`, where
:math:`\| E \| \sim \sigma_{k + 1}` is on the order of the :math:`(k +
1)`-th largest singular value of :math:`A`. Note that :math:`\sigma_{k
+ 1}` is the best possible error for a rank-:math:`k` approximation
and, in fact, is achieved by the singular value decomposition (SVD)
:math:`A \approx U S V^{*}`, where :math:`U \in \mathbb{C}^{m \times
k}` and :math:`V \in \mathbb{C}^{n \times k}` have orthonormal columns
and :math:`S = \mathop{\mathrm{diag}} (\sigma_{i}) \in \mathbb{C}^{k
\times k}` is diagonal with nonnegative entries. The principal
advantages of using an ID over an SVD are that:

- it is cheaper to construct;
- it preserves the structure of :math:`A`; and
- it is more efficient to compute with in light of the identity submatrix of :math:`P`.

Routines
========

Main functionality:

.. autosummary::
   :toctree: generated/

   interp_decomp
   reconstruct_matrix_from_id
   reconstruct_interp_matrix
   reconstruct_skel_matrix
   id_to_svd
   svd
   estimate_spectral_norm
   estimate_spectral_norm_diff
   estimate_rank

Support functions:

.. autosummary::
   :toctree: generated/

   seed
   rand


References
==========

This module uses the ID software package [1]_ by Martinsson, Rokhlin,
Shkolnisky, and Tygert, which is a Fortran library for computing IDs
using various algorithms, including the rank-revealing QR approach of
[2]_ and the more recent randomized methods described in [3]_, [4]_,
and [5]_. This module exposes its functionality in a way convenient
for Python users. Note that this module does not add any functionality
beyond that of organizing a simpler and more consistent interface.

We advise the user to consult also the `documentation for the ID package
<http://tygert.com/id_doc.4.pdf>`_.

.. [1] P.G. Martinsson, V. Rokhlin, Y. Shkolnisky, M. Tygert. "ID: a
    software package for low-rank approximation of matrices via interpolative
    decompositions, version 0.2." http://tygert.com/id_doc.4.pdf.

.. [2] H. Cheng, Z. Gimbutas, P.G. Martinsson, V. Rokhlin. "On the
    compression of low rank matrices." *SIAM J. Sci. Comput.* 26 (4): 1389--1404,
    2005. :doi:`10.1137/030602678`.

.. [3] E. Liberty, F. Woolfe, P.G. Martinsson, V. Rokhlin, M.
    Tygert. "Randomized algorithms for the low-rank approximation of matrices."
    *Proc. Natl. Acad. Sci. U.S.A.* 104 (51): 20167--20172, 2007.
    :doi:`10.1073/pnas.0709640104`.

.. [4] P.G. Martinsson, V. Rokhlin, M. Tygert. "A randomized
    algorithm for the decomposition of matrices." *Appl. Comput. Harmon. Anal.* 30
    (1): 47--68,  2011. :doi:`10.1016/j.acha.2010.02.003`.

.. [5] F. Woolfe, E. Liberty, V. Rokhlin, M. Tygert. "A fast
    randomized algorithm for the approximation of matrices." *Appl. Comput.
    Harmon. Anal.* 25 (3): 335--366, 2008. :doi:`10.1016/j.acha.2007.12.002`.


Tutorial
========

Initializing
------------

The first step is to import :mod:`scipy.linalg.interpolative` by issuing the
command:

>>> import scipy.linalg.interpolative as sli

Now let's build a matrix. For this, we consider a Hilbert matrix, which is well
know to have low rank:

>>> from scipy.linalg import hilbert
>>> n = 1000
>>> A = hilbert(n)

We can also do this explicitly via:

>>> import numpy as np
>>> n = 1000
>>> A = np.empty((n, n), order='F')
>>> for j in range(n):
>>>     for i in range(m):
>>>         A[i,j] = 1. / (i + j + 1)

Note the use of the flag ``order='F'`` in :func:`numpy.empty`. This
instantiates the matrix in Fortran-contiguous order and is important for
avoiding data copying when passing to the backend.

We then define multiplication routines for the matrix by regarding it as a
:class:`scipy.sparse.linalg.LinearOperator`:

>>> from scipy.sparse.linalg import aslinearoperator
>>> L = aslinearoperator(A)

This automatically sets up methods describing the action of the matrix and its
adjoint on a vector.

Computing an ID
---------------

We have several choices of algorithm to compute an ID. These fall largely
according to two dichotomies:

1. how the matrix is represented, i.e., via its entries or via its action on a
   vector; and
2. whether to approximate it to a fixed relative precision or to a fixed rank.

We step through each choice in turn below.

In all cases, the ID is represented by three parameters:

1. a rank ``k``;
2. an index array ``idx``; and
3. interpolation coefficients ``proj``.

The ID is specified by the relation
``np.dot(A[:,idx[:k]], proj) == A[:,idx[k:]]``.

From matrix entries
...................

We first consider a matrix given in terms of its entries.

To compute an ID to a fixed precision, type:

>>> k, idx, proj = sli.interp_decomp(A, eps)

where ``eps < 1`` is the desired precision.

To compute an ID to a fixed rank, use:

>>> idx, proj = sli.interp_decomp(A, k)

where ``k >= 1`` is the desired rank.

Both algorithms use random sampling and are usually faster than the
corresponding older, deterministic algorithms, which can be accessed via the
commands:

>>> k, idx, proj = sli.interp_decomp(A, eps, rand=False)

and:

>>> idx, proj = sli.interp_decomp(A, k, rand=False)

respectively.

From matrix action
..................

Now consider a matrix given in terms of its action on a vector as a
:class:`scipy.sparse.linalg.LinearOperator`.

To compute an ID to a fixed precision, type:

>>> k, idx, proj = sli.interp_decomp(L, eps)

To compute an ID to a fixed rank, use:

>>> idx, proj = sli.interp_decomp(L, k)

These algorithms are randomized.

Reconstructing an ID
--------------------

The ID routines above do not output the skeleton and interpolation matrices
explicitly but instead return the relevant information in a more compact (and
sometimes more useful) form. To build these matrices, write:

>>> B = sli.reconstruct_skel_matrix(A, k, idx)

for the skeleton matrix and:

>>> P = sli.reconstruct_interp_matrix(idx, proj)

for the interpolation matrix. The ID approximation can then be computed as:

>>> C = np.dot(B, P)

This can also be constructed directly using:

>>> C = sli.reconstruct_matrix_from_id(B, idx, proj)

without having to first compute ``P``.

Alternatively, this can be done explicitly as well using:

>>> B = A[:,idx[:k]]
>>> P = np.hstack([np.eye(k), proj])[:,np.argsort(idx)]
>>> C = np.dot(B, P)

Computing an SVD
----------------

An ID can be converted to an SVD via the command:

>>> U, S, V = sli.id_to_svd(B, idx, proj)

The SVD approximation is then:

>>> C = np.dot(U, np.dot(np.diag(S), np.dot(V.conj().T)))

The SVD can also be computed "fresh" by combining both the ID and conversion
steps into one command. Following the various ID algorithms above, there are
correspondingly various SVD algorithms that one can employ.

From matrix entries
...................

We consider first SVD algorithms for a matrix given in terms of its entries.

To compute an SVD to a fixed precision, type:

>>> U, S, V = sli.svd(A, eps)

To compute an SVD to a fixed rank, use:

>>> U, S, V = sli.svd(A, k)

Both algorithms use random sampling; for the determinstic versions, issue the
keyword ``rand=False`` as above.

From matrix action
..................

Now consider a matrix given in terms of its action on a vector.

To compute an SVD to a fixed precision, type:

>>> U, S, V = sli.svd(L, eps)

To compute an SVD to a fixed rank, use:

>>> U, S, V = sli.svd(L, k)

Utility routines
----------------

Several utility routines are also available.

To estimate the spectral norm of a matrix, use:

>>> snorm = sli.estimate_spectral_norm(A)

This algorithm is based on the randomized power method and thus requires only
matrix-vector products. The number of iterations to take can be set using the
keyword ``its`` (default: ``its=20``). The matrix is interpreted as a
:class:`scipy.sparse.linalg.LinearOperator`, but it is also valid to supply it
as a :class:`numpy.ndarray`, in which case it is trivially converted using
:func:`scipy.sparse.linalg.aslinearoperator`.

The same algorithm can also estimate the spectral norm of the difference of two
matrices ``A1`` and ``A2`` as follows:

>>> diff = sli.estimate_spectral_norm_diff(A1, A2)

This is often useful for checking the accuracy of a matrix approximation.

Some routines in :mod:`scipy.linalg.interpolative` require estimating the rank
of a matrix as well. This can be done with either:

>>> k = sli.estimate_rank(A, eps)

or:

>>> k = sli.estimate_rank(L, eps)

depending on the representation. The parameter ``eps`` controls the definition
of the numerical rank.

Finally, the random number generation required for all randomized routines can
be controlled via :func:`scipy.linalg.interpolative.seed`. To reset the seed
values to their original values, use:

>>> sli.seed('default')

To specify the seed values, use:

>>> sli.seed(s)

where ``s`` must be an integer or array of 55 floats. If an integer, the array
of floats is obtained by using ``numpy.random.rand`` with the given integer
seed.

To simply generate some random numbers, type:

>>> sli.rand(n)

where ``n`` is the number of random numbers to generate.

Remarks
-------

The above functions all automatically detect the appropriate interface and work
with both real and complex data types, passing input arguments to the proper
backend routine.

"""

import scipy.linalg._interpolative_backend as backend
import numpy as np
import sys

_DTYPE_ERROR = ValueError("invalid input dtype (input must be float64 or complex128)")
_TYPE_ERROR = TypeError("invalid input type (must be array or LinearOperator)")
_32BIT_ERROR = ValueError("interpolative decomposition on 32-bit systems "
                          "with complex128 is buggy")
_IS_32BIT = (sys.maxsize < 2**32)


def _is_real(A):
    try:
        if A.dtype == np.complex128:
            return False
        elif A.dtype == np.float64:
            return True
        else:
            raise _DTYPE_ERROR
    except AttributeError as e:
        raise _TYPE_ERROR from e


def seed(seed=None):
    """
    Seed the internal random number generator used in this ID package.

    The generator is a lagged Fibonacci method with 55-element internal state.

    Parameters
    ----------
    seed : int, sequence, 'default', optional
        If 'default', the random seed is reset to a default value.

        If `seed` is a sequence containing 55 floating-point numbers
        in range [0,1], these are used to set the internal state of
        the generator.

        If the value is an integer, the internal state is obtained
        from `numpy.random.RandomState` (MT19937) with the integer
        used as the initial seed.

        If `seed` is omitted (None), ``numpy.random.rand`` is used to
        initialize the generator.

    """
    # For details, see :func:`backend.id_srand`, :func:`backend.id_srandi`,
    # and :func:`backend.id_srando`.

    if isinstance(seed, str) and seed == 'default':
        backend.id_srando()
    elif hasattr(seed, '__len__'):
        state = np.asfortranarray(seed, dtype=float)
        if state.shape != (55,):
            raise ValueError("invalid input size")
        elif state.min() < 0 or state.max() > 1:
            raise ValueError("values not in range [0,1]")
        backend.id_srandi(state)
    elif seed is None:
        backend.id_srandi(np.random.rand(55))
    else:
        rnd = np.random.RandomState(seed)
        backend.id_srandi(rnd.rand(55))


def rand(*shape):
    """
    Generate standard uniform pseudorandom numbers via a very efficient lagged
    Fibonacci method.

    This routine is used for all random number generation in this package and
    can affect ID and SVD results.

    Parameters
    ----------
    shape
        Shape of output array

    """
    # For details, see :func:`backend.id_srand`, and :func:`backend.id_srando`.
    return backend.id_srand(np.prod(shape)).reshape(shape)


def interp_decomp(A, eps_or_k, rand=True):
    """
    Compute ID of a matrix.

    An ID of a matrix `A` is a factorization defined by a rank `k`, a column
    index array `idx`, and interpolation coefficients `proj` such that::

        numpy.dot(A[:,idx[:k]], proj) = A[:,idx[k:]]

    The original matrix can then be reconstructed as::

        numpy.hstack([A[:,idx[:k]],
                                    numpy.dot(A[:,idx[:k]], proj)]
                                )[:,numpy.argsort(idx)]

    or via the routine :func:`reconstruct_matrix_from_id`. This can
    equivalently be written as::

        numpy.dot(A[:,idx[:k]],
                            numpy.hstack([numpy.eye(k), proj])
                          )[:,np.argsort(idx)]

    in terms of the skeleton and interpolation matrices::

        B = A[:,idx[:k]]

    and::

        P = numpy.hstack([numpy.eye(k), proj])[:,np.argsort(idx)]

    respectively. See also :func:`reconstruct_interp_matrix` and
    :func:`reconstruct_skel_matrix`.

    The ID can be computed to any relative precision or rank (depending on the
    value of `eps_or_k`). If a precision is specified (`eps_or_k < 1`), then
    this function has the output signature::

        k, idx, proj = interp_decomp(A, eps_or_k)

    Otherwise, if a rank is specified (`eps_or_k >= 1`), then the output
    signature is::

        idx, proj = interp_decomp(A, eps_or_k)

    ..  This function automatically detects the form of the input parameters
        and passes them to the appropriate backend. For details, see
        :func:`backend.iddp_id`, :func:`backend.iddp_aid`,
        :func:`backend.iddp_rid`, :func:`backend.iddr_id`,
        :func:`backend.iddr_aid`, :func:`backend.iddr_rid`,
        :func:`backend.idzp_id`, :func:`backend.idzp_aid`,
        :func:`backend.idzp_rid`, :func:`backend.idzr_id`,
        :func:`backend.idzr_aid`, and :func:`backend.idzr_rid`.

    Parameters
    ----------
    A : :class:`numpy.ndarray` or :class:`scipy.sparse.linalg.LinearOperator` with `rmatvec`
        Matrix to be factored
    eps_or_k : float or int
        Relative error (if `eps_or_k < 1`) or rank (if `eps_or_k >= 1`) of
        approximation.
    rand : bool, optional
        Whether to use random sampling if `A` is of type :class:`numpy.ndarray`
        (randomized algorithms are always used if `A` is of type
        :class:`scipy.sparse.linalg.LinearOperator`).

    Returns
    -------
    k : int
        Rank required to achieve specified relative precision if
        `eps_or_k < 1`.
    idx : :class:`numpy.ndarray`
        Column index array.
    proj : :class:`numpy.ndarray`
        Interpolation coefficients.
    """
    from scipy.sparse.linalg import LinearOperator

    real = _is_real(A)

    if isinstance(A, np.ndarray):
        if eps_or_k < 1:
            eps = eps_or_k
            if rand:
                if real:
                    k, idx, proj = backend.iddp_aid(eps, A)
                else:
                    if _IS_32BIT:
                        raise _32BIT_ERROR
                    k, idx, proj = backend.idzp_aid(eps, A)
            else:
                if real:
                    k, idx, proj = backend.iddp_id(eps, A)
                else:
                    k, idx, proj = backend.idzp_id(eps, A)
            return k, idx - 1, proj
        else:
            k = int(eps_or_k)
            if rand:
                if real:
                    idx, proj = backend.iddr_aid(A, k)
                else:
                    if _IS_32BIT:
                        raise _32BIT_ERROR
                    idx, proj = backend.idzr_aid(A, k)
            else:
                if real:
                    idx, proj = backend.iddr_id(A, k)
                else:
                    idx, proj = backend.idzr_id(A, k)
            return idx - 1, proj
    elif isinstance(A, LinearOperator):
        m, n = A.shape
        matveca = A.rmatvec
        if eps_or_k < 1:
            eps = eps_or_k
            if real:
                k, idx, proj = backend.iddp_rid(eps, m, n, matveca)
            else:
                if _IS_32BIT:
                    raise _32BIT_ERROR
                k, idx, proj = backend.idzp_rid(eps, m, n, matveca)
            return k, idx - 1, proj
        else:
            k = int(eps_or_k)
            if real:
                idx, proj = backend.iddr_rid(m, n, matveca, k)
            else:
                if _IS_32BIT:
                    raise _32BIT_ERROR
                idx, proj = backend.idzr_rid(m, n, matveca, k)
            return idx - 1, proj
    else:
        raise _TYPE_ERROR


def reconstruct_matrix_from_id(B, idx, proj):
    """
    Reconstruct matrix from its ID.

    A matrix `A` with skeleton matrix `B` and ID indices and coefficients `idx`
    and `proj`, respectively, can be reconstructed as::

        numpy.hstack([B, numpy.dot(B, proj)])[:,numpy.argsort(idx)]

    See also :func:`reconstruct_interp_matrix` and
    :func:`reconstruct_skel_matrix`.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`backend.idd_reconid` and
        :func:`backend.idz_reconid`.

    Parameters
    ----------
    B : :class:`numpy.ndarray`
        Skeleton matrix.
    idx : :class:`numpy.ndarray`
        Column index array.
    proj : :class:`numpy.ndarray`
        Interpolation coefficients.

    Returns
    -------
    :class:`numpy.ndarray`
        Reconstructed matrix.
    """
    if _is_real(B):
        return backend.idd_reconid(B, idx + 1, proj)
    else:
        return backend.idz_reconid(B, idx + 1, proj)


def reconstruct_interp_matrix(idx, proj):
    """
    Reconstruct interpolation matrix from ID.

    The interpolation matrix can be reconstructed from the ID indices and
    coefficients `idx` and `proj`, respectively, as::

        P = numpy.hstack([numpy.eye(proj.shape[0]), proj])[:,numpy.argsort(idx)]

    The original matrix can then be reconstructed from its skeleton matrix `B`
    via::

        numpy.dot(B, P)

    See also :func:`reconstruct_matrix_from_id` and
    :func:`reconstruct_skel_matrix`.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`backend.idd_reconint` and
        :func:`backend.idz_reconint`.

    Parameters
    ----------
    idx : :class:`numpy.ndarray`
        Column index array.
    proj : :class:`numpy.ndarray`
        Interpolation coefficients.

    Returns
    -------
    :class:`numpy.ndarray`
        Interpolation matrix.
    """
    if _is_real(proj):
        return backend.idd_reconint(idx + 1, proj)
    else:
        return backend.idz_reconint(idx + 1, proj)


def reconstruct_skel_matrix(A, k, idx):
    """
    Reconstruct skeleton matrix from ID.

    The skeleton matrix can be reconstructed from the original matrix `A` and its
    ID rank and indices `k` and `idx`, respectively, as::

        B = A[:,idx[:k]]

    The original matrix can then be reconstructed via::

        numpy.hstack([B, numpy.dot(B, proj)])[:,numpy.argsort(idx)]

    See also :func:`reconstruct_matrix_from_id` and
    :func:`reconstruct_interp_matrix`.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`backend.idd_copycols` and
        :func:`backend.idz_copycols`.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Original matrix.
    k : int
        Rank of ID.
    idx : :class:`numpy.ndarray`
        Column index array.

    Returns
    -------
    :class:`numpy.ndarray`
        Skeleton matrix.
    """
    if _is_real(A):
        return backend.idd_copycols(A, k, idx + 1)
    else:
        return backend.idz_copycols(A, k, idx + 1)


def id_to_svd(B, idx, proj):
    """
    Convert ID to SVD.

    The SVD reconstruction of a matrix with skeleton matrix `B` and ID indices and
    coefficients `idx` and `proj`, respectively, is::

        U, S, V = id_to_svd(B, idx, proj)
        A = numpy.dot(U, numpy.dot(numpy.diag(S), V.conj().T))

    See also :func:`svd`.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`backend.idd_id2svd` and
        :func:`backend.idz_id2svd`.

    Parameters
    ----------
    B : :class:`numpy.ndarray`
        Skeleton matrix.
    idx : :class:`numpy.ndarray`
        Column index array.
    proj : :class:`numpy.ndarray`
        Interpolation coefficients.

    Returns
    -------
    U : :class:`numpy.ndarray`
        Left singular vectors.
    S : :class:`numpy.ndarray`
        Singular values.
    V : :class:`numpy.ndarray`
        Right singular vectors.
    """
    if _is_real(B):
        U, V, S = backend.idd_id2svd(B, idx + 1, proj)
    else:
        U, V, S = backend.idz_id2svd(B, idx + 1, proj)
    return U, S, V


def estimate_spectral_norm(A, its=20):
    """
    Estimate spectral norm of a matrix by the randomized power method.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`backend.idd_snorm` and
        :func:`backend.idz_snorm`.

    Parameters
    ----------
    A : :class:`scipy.sparse.linalg.LinearOperator`
        Matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with the
        `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).
    its : int, optional
        Number of power method iterations.

    Returns
    -------
    float
        Spectral norm estimate.
    """
    from scipy.sparse.linalg import aslinearoperator
    A = aslinearoperator(A)
    m, n = A.shape
    matvec = lambda x: A. matvec(x)
    matveca = lambda x: A.rmatvec(x)
    if _is_real(A):
        return backend.idd_snorm(m, n, matveca, matvec, its=its)
    else:
        return backend.idz_snorm(m, n, matveca, matvec, its=its)


def estimate_spectral_norm_diff(A, B, its=20):
    """
    Estimate spectral norm of the difference of two matrices by the randomized
    power method.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`backend.idd_diffsnorm` and
        :func:`backend.idz_diffsnorm`.

    Parameters
    ----------
    A : :class:`scipy.sparse.linalg.LinearOperator`
        First matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with the
        `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).
    B : :class:`scipy.sparse.linalg.LinearOperator`
        Second matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with
        the `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).
    its : int, optional
        Number of power method iterations.

    Returns
    -------
    float
        Spectral norm estimate of matrix difference.
    """
    from scipy.sparse.linalg import aslinearoperator
    A = aslinearoperator(A)
    B = aslinearoperator(B)
    m, n = A.shape
    matvec1 = lambda x: A. matvec(x)
    matveca1 = lambda x: A.rmatvec(x)
    matvec2 = lambda x: B. matvec(x)
    matveca2 = lambda x: B.rmatvec(x)
    if _is_real(A):
        return backend.idd_diffsnorm(
            m, n, matveca1, matveca2, matvec1, matvec2, its=its)
    else:
        return backend.idz_diffsnorm(
            m, n, matveca1, matveca2, matvec1, matvec2, its=its)


def svd(A, eps_or_k, rand=True):
    """
    Compute SVD of a matrix via an ID.

    An SVD of a matrix `A` is a factorization::

        A = numpy.dot(U, numpy.dot(numpy.diag(S), V.conj().T))

    where `U` and `V` have orthonormal columns and `S` is nonnegative.

    The SVD can be computed to any relative precision or rank (depending on the
    value of `eps_or_k`).

    See also :func:`interp_decomp` and :func:`id_to_svd`.

    ..  This function automatically detects the form of the input parameters and
        passes them to the appropriate backend. For details, see
        :func:`backend.iddp_svd`, :func:`backend.iddp_asvd`,
        :func:`backend.iddp_rsvd`, :func:`backend.iddr_svd`,
        :func:`backend.iddr_asvd`, :func:`backend.iddr_rsvd`,
        :func:`backend.idzp_svd`, :func:`backend.idzp_asvd`,
        :func:`backend.idzp_rsvd`, :func:`backend.idzr_svd`,
        :func:`backend.idzr_asvd`, and :func:`backend.idzr_rsvd`.

    Parameters
    ----------
    A : :class:`numpy.ndarray` or :class:`scipy.sparse.linalg.LinearOperator`
        Matrix to be factored, given as either a :class:`numpy.ndarray` or a
        :class:`scipy.sparse.linalg.LinearOperator` with the `matvec` and
        `rmatvec` methods (to apply the matrix and its adjoint).
    eps_or_k : float or int
        Relative error (if `eps_or_k < 1`) or rank (if `eps_or_k >= 1`) of
        approximation.
    rand : bool, optional
        Whether to use random sampling if `A` is of type :class:`numpy.ndarray`
        (randomized algorithms are always used if `A` is of type
        :class:`scipy.sparse.linalg.LinearOperator`).

    Returns
    -------
    U : :class:`numpy.ndarray`
        Left singular vectors.
    S : :class:`numpy.ndarray`
        Singular values.
    V : :class:`numpy.ndarray`
        Right singular vectors.
    """
    from scipy.sparse.linalg import LinearOperator

    real = _is_real(A)

    if isinstance(A, np.ndarray):
        if eps_or_k < 1:
            eps = eps_or_k
            if rand:
                if real:
                    U, V, S = backend.iddp_asvd(eps, A)
                else:
                    if _IS_32BIT:
                        raise _32BIT_ERROR
                    U, V, S = backend.idzp_asvd(eps, A)
            else:
                if real:
                    U, V, S = backend.iddp_svd(eps, A)
                else:
                    U, V, S = backend.idzp_svd(eps, A)
        else:
            k = int(eps_or_k)
            if k > min(A.shape):
                raise ValueError("Approximation rank %s exceeds min(A.shape) = "
                                 " %s " % (k, min(A.shape)))
            if rand:
                if real:
                    U, V, S = backend.iddr_asvd(A, k)
                else:
                    if _IS_32BIT:
                        raise _32BIT_ERROR
                    U, V, S = backend.idzr_asvd(A, k)
            else:
                if real:
                    U, V, S = backend.iddr_svd(A, k)
                else:
                    U, V, S = backend.idzr_svd(A, k)
    elif isinstance(A, LinearOperator):
        m, n = A.shape
        matvec = lambda x: A.matvec(x)
        matveca = lambda x: A.rmatvec(x)
        if eps_or_k < 1:
            eps = eps_or_k
            if real:
                U, V, S = backend.iddp_rsvd(eps, m, n, matveca, matvec)
            else:
                if _IS_32BIT:
                    raise _32BIT_ERROR
                U, V, S = backend.idzp_rsvd(eps, m, n, matveca, matvec)
        else:
            k = int(eps_or_k)
            if real:
                U, V, S = backend.iddr_rsvd(m, n, matveca, matvec, k)
            else:
                if _IS_32BIT:
                    raise _32BIT_ERROR
                U, V, S = backend.idzr_rsvd(m, n, matveca, matvec, k)
    else:
        raise _TYPE_ERROR
    return U, S, V


def estimate_rank(A, eps):
    """
    Estimate matrix rank to a specified relative precision using randomized
    methods.

    The matrix `A` can be given as either a :class:`numpy.ndarray` or a
    :class:`scipy.sparse.linalg.LinearOperator`, with different algorithms used
    for each case. If `A` is of type :class:`numpy.ndarray`, then the output
    rank is typically about 8 higher than the actual numerical rank.

    ..  This function automatically detects the form of the input parameters and
        passes them to the appropriate backend. For details,
        see :func:`backend.idd_estrank`, :func:`backend.idd_findrank`,
        :func:`backend.idz_estrank`, and :func:`backend.idz_findrank`.

    Parameters
    ----------
    A : :class:`numpy.ndarray` or :class:`scipy.sparse.linalg.LinearOperator`
        Matrix whose rank is to be estimated, given as either a
        :class:`numpy.ndarray` or a :class:`scipy.sparse.linalg.LinearOperator`
        with the `rmatvec` method (to apply the matrix adjoint).
    eps : float
        Relative error for numerical rank definition.

    Returns
    -------
    int
        Estimated matrix rank.
    """
    from scipy.sparse.linalg import LinearOperator

    real = _is_real(A)

    if isinstance(A, np.ndarray):
        if real:
            rank = backend.idd_estrank(eps, A)
        else:
            rank = backend.idz_estrank(eps, A)
        if rank == 0:
            # special return value for nearly full rank
            rank = min(A.shape)
        return rank
    elif isinstance(A, LinearOperator):
        m, n = A.shape
        matveca = A.rmatvec
        if real:
            return backend.idd_findrank(eps, m, n, matveca)
        else:
            return backend.idz_findrank(eps, m, n, matveca)
    else:
        raise _TYPE_ERROR
