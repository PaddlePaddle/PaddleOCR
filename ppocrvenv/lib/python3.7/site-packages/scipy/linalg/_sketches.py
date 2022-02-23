""" Sketching-based Matrix Computations """

# Author: Jordi Montes <jomsdev@gmail.com>
# August 28, 2017

import numpy as np

from scipy._lib._util import check_random_state, rng_integers
from scipy.sparse import csc_matrix

__all__ = ['clarkson_woodruff_transform']


def cwt_matrix(n_rows, n_columns, seed=None):
    r"""
    Generate a matrix S which represents a Clarkson-Woodruff transform.

    Given the desired size of matrix, the method returns a matrix S of size
    (n_rows, n_columns) where each column has all the entries set to 0
    except for one position which has been randomly set to +1 or -1 with
    equal probability.

    Parameters
    ----------
    n_rows: int
        Number of rows of S
    n_columns: int
        Number of columns of S
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    S : (n_rows, n_columns) csc_matrix
        The returned matrix has ``n_columns`` nonzero entries.

    Notes
    -----
    Given a matrix A, with probability at least 9/10,
    .. math:: \|SA\| = (1 \pm \epsilon)\|A\|
    Where the error epsilon is related to the size of S.
    """
    rng = check_random_state(seed)
    rows = rng_integers(rng, 0, n_rows, n_columns)
    cols = np.arange(n_columns+1)
    signs = rng.choice([1, -1], n_columns)
    S = csc_matrix((signs, rows, cols),shape=(n_rows, n_columns))
    return S


def clarkson_woodruff_transform(input_matrix, sketch_size, seed=None):
    r"""
    Applies a Clarkson-Woodruff Transform/sketch to the input matrix.

    Given an input_matrix ``A`` of size ``(n, d)``, compute a matrix ``A'`` of
    size (sketch_size, d) so that

    .. math:: \|Ax\| \approx \|A'x\|

    with high probability via the Clarkson-Woodruff Transform, otherwise
    known as the CountSketch matrix.

    Parameters
    ----------
    input_matrix: array_like
        Input matrix, of shape ``(n, d)``.
    sketch_size: int
        Number of rows for the sketch.
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    A' : array_like
        Sketch of the input matrix ``A``, of size ``(sketch_size, d)``.

    Notes
    -----
    To make the statement

    .. math:: \|Ax\| \approx \|A'x\|

    precise, observe the following result which is adapted from the
    proof of Theorem 14 of [2]_ via Markov's Inequality. If we have
    a sketch size ``sketch_size=k`` which is at least

    .. math:: k \geq \frac{2}{\epsilon^2\delta}

    Then for any fixed vector ``x``,

    .. math:: \|Ax\| = (1\pm\epsilon)\|A'x\|

    with probability at least one minus delta.

    This implementation takes advantage of sparsity: computing
    a sketch takes time proportional to ``A.nnz``. Data ``A`` which
    is in ``scipy.sparse.csc_matrix`` format gives the quickest
    computation time for sparse input.

    >>> from scipy import linalg
    >>> from scipy import sparse
    >>> rng = np.random.default_rng()
    >>> n_rows, n_columns, density, sketch_n_rows = 15000, 100, 0.01, 200
    >>> A = sparse.rand(n_rows, n_columns, density=density, format='csc')
    >>> B = sparse.rand(n_rows, n_columns, density=density, format='csr')
    >>> C = sparse.rand(n_rows, n_columns, density=density, format='coo')
    >>> D = rng.standard_normal((n_rows, n_columns))
    >>> SA = linalg.clarkson_woodruff_transform(A, sketch_n_rows) # fastest
    >>> SB = linalg.clarkson_woodruff_transform(B, sketch_n_rows) # fast
    >>> SC = linalg.clarkson_woodruff_transform(C, sketch_n_rows) # slower
    >>> SD = linalg.clarkson_woodruff_transform(D, sketch_n_rows) # slowest

    That said, this method does perform well on dense inputs, just slower
    on a relative scale.

    Examples
    --------
    Given a big dense matrix ``A``:

    >>> from scipy import linalg
    >>> n_rows, n_columns, sketch_n_rows = 15000, 100, 200
    >>> rng = np.random.default_rng()
    >>> A = rng.standard_normal((n_rows, n_columns))
    >>> sketch = linalg.clarkson_woodruff_transform(A, sketch_n_rows)
    >>> sketch.shape
    (200, 100)
    >>> norm_A = np.linalg.norm(A)
    >>> norm_sketch = np.linalg.norm(sketch)

    Now with high probability, the true norm ``norm_A`` is close to
    the sketched norm ``norm_sketch`` in absolute value.

    Similarly, applying our sketch preserves the solution to a linear
    regression of :math:`\min \|Ax - b\|`.

    >>> from scipy import linalg
    >>> n_rows, n_columns, sketch_n_rows = 15000, 100, 200
    >>> rng = np.random.default_rng()
    >>> A = rng.standard_normal((n_rows, n_columns))
    >>> b = rng.standard_normal(n_rows)
    >>> x = np.linalg.lstsq(A, b, rcond=None)
    >>> Ab = np.hstack((A, b.reshape(-1,1)))
    >>> SAb = linalg.clarkson_woodruff_transform(Ab, sketch_n_rows)
    >>> SA, Sb = SAb[:,:-1], SAb[:,-1]
    >>> x_sketched = np.linalg.lstsq(SA, Sb, rcond=None)

    As with the matrix norm example, ``np.linalg.norm(A @ x - b)``
    is close to ``np.linalg.norm(A @ x_sketched - b)`` with high
    probability.

    References
    ----------
    .. [1] Kenneth L. Clarkson and David P. Woodruff. Low rank approximation and
           regression in input sparsity time. In STOC, 2013.

    .. [2] David P. Woodruff. Sketching as a tool for numerical linear algebra.
           In Foundations and Trends in Theoretical Computer Science, 2014.

    """
    S = cwt_matrix(sketch_size, input_matrix.shape[0], seed)
    return S.dot(input_matrix)
