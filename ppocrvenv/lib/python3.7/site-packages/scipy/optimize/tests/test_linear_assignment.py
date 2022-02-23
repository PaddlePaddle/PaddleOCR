# Author: Brian M. Clapper, G. Varoquaux, Lars Buitinck
# License: BSD

from numpy.testing import assert_array_equal
from pytest import raises as assert_raises
import pytest

import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.sparse import random
from scipy.sparse.sputils import matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.sparse.csgraph.tests.test_matching import (
    linear_sum_assignment_assertions, linear_sum_assignment_test_cases
)


def test_linear_sum_assignment_input_validation():

    assert_raises(ValueError, linear_sum_assignment, [1, 2, 3])

    C = [[1, 2, 3], [4, 5, 6]]
    assert_array_equal(linear_sum_assignment(C),
                       linear_sum_assignment(np.asarray(C)))
    assert_array_equal(linear_sum_assignment(C),
                       linear_sum_assignment(matrix(C)))

    I = np.identity(3)
    assert_array_equal(linear_sum_assignment(I.astype(np.bool_)),
                       linear_sum_assignment(I))
    assert_raises(ValueError, linear_sum_assignment, I.astype(str))

    I[0][0] = np.nan
    with pytest.raises(ValueError, match="contains invalid numeric entries"):
        linear_sum_assignment(I)

    I = np.identity(3)
    I[1][1] = -np.inf
    with pytest.raises(ValueError, match="contains invalid numeric entries"):
        linear_sum_assignment(I)

    I = np.identity(3)
    I[:, 0] = np.inf
    with pytest.raises(ValueError, match="cost matrix is infeasible"):
        linear_sum_assignment(I)


def test_constant_cost_matrix():
    # Fixes #11602
    n = 8
    C = np.ones((n, n))
    row_ind, col_ind = linear_sum_assignment(C)
    assert_array_equal(row_ind, np.arange(n))
    assert_array_equal(col_ind, np.arange(n))


@pytest.mark.parametrize('num_rows,num_cols', [(0, 0), (2, 0), (0, 3)])
def test_linear_sum_assignment_trivial_cost(num_rows, num_cols):
    C = np.empty(shape=(num_cols, num_rows))
    row_ind, col_ind = linear_sum_assignment(C)
    assert len(row_ind) == 0
    assert len(col_ind) == 0


@pytest.mark.parametrize('sign,test_case', linear_sum_assignment_test_cases)
def test_linear_sum_assignment_small_inputs(sign, test_case):
    linear_sum_assignment_assertions(
        linear_sum_assignment, np.array, sign, test_case)


# Tests that combine scipy.optimize.linear_sum_assignment and
# scipy.sparse.csgraph.min_weight_full_bipartite_matching
def test_two_methods_give_same_result_on_many_sparse_inputs():
    # As opposed to the test above, here we do not spell out the expected
    # output; only assert that the two methods give the same result.
    # Concretely, the below tests 100 cases of size 100x100, out of which
    # 36 are infeasible.
    np.random.seed(1234)
    for _ in range(100):
        lsa_raises = False
        mwfbm_raises = False
        sparse = random(100, 100, density=0.06,
                        data_rvs=lambda size: np.random.randint(1, 100, size))
        # In csgraph, zeros correspond to missing edges, so we explicitly
        # replace those with infinities
        dense = np.full(sparse.shape, np.inf)
        dense[sparse.row, sparse.col] = sparse.data
        sparse = sparse.tocsr()
        try:
            row_ind, col_ind = linear_sum_assignment(dense)
            lsa_cost = dense[row_ind, col_ind].sum()
        except ValueError:
            lsa_raises = True
        try:
            row_ind, col_ind = min_weight_full_bipartite_matching(sparse)
            mwfbm_cost = sparse[row_ind, col_ind].sum()
        except ValueError:
            mwfbm_raises = True
        # Ensure that if one method raises, so does the other one.
        assert lsa_raises == mwfbm_raises
        if not lsa_raises:
            assert lsa_cost == mwfbm_cost
