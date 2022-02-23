import numpy as np
from numpy.testing import (assert_almost_equal,
                           assert_array_equal,
                           assert_equal,
                           assert_)
import pytest
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import distance
from scipy._lib._util import check_random_state

class TestHausdorff:
    # Test various properties of the directed Hausdorff code.

    def setup_method(self):
        np.random.seed(1234)
        random_angles = np.random.random(100) * np.pi * 2
        random_columns = np.column_stack(
            (random_angles, random_angles, np.zeros(100)))
        random_columns[..., 0] = np.cos(random_columns[..., 0])
        random_columns[..., 1] = np.sin(random_columns[..., 1])
        random_columns_2 = np.column_stack(
            (random_angles, random_angles, np.zeros(100)))
        random_columns_2[1:, 0] = np.cos(random_columns_2[1:, 0]) * 2.0
        random_columns_2[1:, 1] = np.sin(random_columns_2[1:, 1]) * 2.0
        # move one point farther out so we don't have two perfect circles
        random_columns_2[0, 0] = np.cos(random_columns_2[0, 0]) * 3.3
        random_columns_2[0, 1] = np.sin(random_columns_2[0, 1]) * 3.3
        self.path_1 = random_columns
        self.path_2 = random_columns_2
        self.path_1_4d = np.insert(self.path_1, 3, 5, axis=1)
        self.path_2_4d = np.insert(self.path_2, 3, 27, axis=1)

    def test_symmetry(self):
        # Ensure that the directed (asymmetric) Hausdorff distance is
        # actually asymmetric

        forward = directed_hausdorff(self.path_1, self.path_2)[0]
        reverse = directed_hausdorff(self.path_2, self.path_1)[0]
        assert_(forward != reverse)

    def test_brute_force_comparison_forward(self):
        # Ensure that the algorithm for directed_hausdorff gives the
        # same result as the simple / brute force approach in the
        # forward direction.
        actual = directed_hausdorff(self.path_1, self.path_2)[0]
        # brute force over rows:
        expected = max(np.amin(distance.cdist(self.path_1, self.path_2),
                               axis=1))
        assert_almost_equal(actual, expected, decimal=9)

    def test_brute_force_comparison_reverse(self):
        # Ensure that the algorithm for directed_hausdorff gives the
        # same result as the simple / brute force approach in the
        # reverse direction.
        actual = directed_hausdorff(self.path_2, self.path_1)[0]
        # brute force over columns:
        expected = max(np.amin(distance.cdist(self.path_1, self.path_2), 
                               axis=0))
        assert_almost_equal(actual, expected, decimal=9)

    def test_degenerate_case(self):
        # The directed Hausdorff distance must be zero if both input
        # data arrays match.
        actual = directed_hausdorff(self.path_1, self.path_1)[0]
        assert_almost_equal(actual, 0.0, decimal=9)

    def test_2d_data_forward(self):
        # Ensure that 2D data is handled properly for a simple case
        # relative to brute force approach.
        actual = directed_hausdorff(self.path_1[..., :2],
                                    self.path_2[..., :2])[0]
        expected = max(np.amin(distance.cdist(self.path_1[..., :2],
                                              self.path_2[..., :2]),
                               axis=1))
        assert_almost_equal(actual, expected, decimal=9)

    def test_4d_data_reverse(self):
        # Ensure that 4D data is handled properly for a simple case
        # relative to brute force approach.
        actual = directed_hausdorff(self.path_2_4d, self.path_1_4d)[0]
        # brute force over columns:
        expected = max(np.amin(distance.cdist(self.path_1_4d, self.path_2_4d), 
                               axis=0))
        assert_almost_equal(actual, expected, decimal=9)

    def test_indices(self):
        # Ensure that correct point indices are returned -- they should
        # correspond to the Hausdorff pair
        path_simple_1 = np.array([[-1,-12],[0,0], [1,1], [3,7], [1,2]])
        path_simple_2 = np.array([[0,0], [1,1], [4,100], [10,9]])
        actual = directed_hausdorff(path_simple_2, path_simple_1)[1:]
        expected = (2, 3)
        assert_array_equal(actual, expected)

    def test_random_state(self):
        # ensure that the global random state is not modified because
        # the directed Hausdorff algorithm uses randomization
        rs = check_random_state(None)
        old_global_state = rs.get_state()
        directed_hausdorff(self.path_1, self.path_2)
        rs2 = check_random_state(None)
        new_global_state = rs2.get_state()
        assert_equal(new_global_state, old_global_state)

    def test_random_state_None_int(self):
        # check that seed values of None or int do not alter global
        # random state
        for seed in [None, 27870671]:
            rs = check_random_state(None)
            old_global_state = rs.get_state()
            directed_hausdorff(self.path_1, self.path_2, seed)
            rs2 = check_random_state(None)
            new_global_state = rs2.get_state()
            assert_equal(new_global_state, old_global_state)

    def test_invalid_dimensions(self):
        # Ensure that a ValueError is raised when the number of columns
        # is not the same
        np.random.seed(1234)
        A = np.random.rand(3, 2)
        B = np.random.rand(4, 5)
        with pytest.raises(ValueError):
            directed_hausdorff(A, B)

    @pytest.mark.parametrize("A, B, seed, expected", [
        # the two cases from gh-11332
        ([(0,0)],
         [(0,1), (0,0)],
         0,
         (0.0, 0, 1)),
        ([(0,0)],
         [(0,1), (0,0)],
         1,
         (0.0, 0, 1)),
        # slightly more complex case
        ([(-5, 3), (0,0)],
         [(0,1), (0,0), (-5, 3)],
         77098,
         # the maximum minimum distance will
         # be the last one found, but a unique
         # solution is not guaranteed more broadly
         (0.0, 1, 1)),
    ])
    def test_subsets(self, A, B, seed, expected):
        # verify fix for gh-11332
        actual = directed_hausdorff(u=A, v=B, seed=seed)
        # check distance
        assert_almost_equal(actual[0], expected[0], decimal=9)
        # check indices
        assert actual[1:] == expected[1:]
