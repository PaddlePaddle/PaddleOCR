import pytest
import numpy as np
from numpy.testing import assert_allclose

from sklearn.ensemble._hist_gradient_boosting._bitset import (
    set_bitset_memoryview,
    in_bitset_memoryview,
    set_raw_bitset_from_binned_bitset,
)
from sklearn.ensemble._hist_gradient_boosting.common import X_DTYPE


@pytest.mark.parametrize(
    "values_to_insert, expected_bitset",
    [
        ([0, 4, 33], np.array([2 ** 0 + 2 ** 4, 2 ** 1, 0], dtype=np.uint32)),
        (
            [31, 32, 33, 79],
            np.array([2 ** 31, 2 ** 0 + 2 ** 1, 2 ** 15], dtype=np.uint32),
        ),
    ],
)
def test_set_get_bitset(values_to_insert, expected_bitset):
    n_32bits_ints = 3
    bitset = np.zeros(n_32bits_ints, dtype=np.uint32)
    for value in values_to_insert:
        set_bitset_memoryview(bitset, value)
    assert_allclose(expected_bitset, bitset)
    for value in range(32 * n_32bits_ints):
        if value in values_to_insert:
            assert in_bitset_memoryview(bitset, value)
        else:
            assert not in_bitset_memoryview(bitset, value)


@pytest.mark.parametrize(
    "raw_categories, binned_cat_to_insert, expected_raw_bitset",
    [
        (
            [3, 4, 5, 10, 31, 32, 43],
            [0, 2, 4, 5, 6],
            [2 ** 3 + 2 ** 5 + 2 ** 31, 2 ** 0 + 2 ** 11],
        ),
        ([3, 33, 50, 52], [1, 3], [0, 2 ** 1 + 2 ** 20]),
    ],
)
def test_raw_bitset_from_binned_bitset(
    raw_categories, binned_cat_to_insert, expected_raw_bitset
):
    binned_bitset = np.zeros(2, dtype=np.uint32)
    raw_bitset = np.zeros(2, dtype=np.uint32)
    raw_categories = np.asarray(raw_categories, dtype=X_DTYPE)

    for val in binned_cat_to_insert:
        set_bitset_memoryview(binned_bitset, val)

    set_raw_bitset_from_binned_bitset(raw_bitset, binned_bitset, raw_categories)

    assert_allclose(expected_raw_bitset, raw_bitset)
    for binned_cat_val, raw_cat_val in enumerate(raw_categories):
        if binned_cat_val in binned_cat_to_insert:
            assert in_bitset_memoryview(raw_bitset, raw_cat_val)
        else:
            assert not in_bitset_memoryview(raw_bitset, raw_cat_val)
