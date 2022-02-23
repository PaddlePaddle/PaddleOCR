import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    MultiIndex,
)


# Note: identical the the "multi" entry in the top-level "index" fixture
@pytest.fixture
def idx():
    # a MultiIndex used to test the general functionality of the
    # general functionality of this object
    major_axis = Index(["foo", "bar", "baz", "qux"])
    minor_axis = Index(["one", "two"])

    major_codes = np.array([0, 0, 1, 2, 3, 3])
    minor_codes = np.array([0, 1, 0, 1, 0, 1])
    index_names = ["first", "second"]
    mi = MultiIndex(
        levels=[major_axis, minor_axis],
        codes=[major_codes, minor_codes],
        names=index_names,
        verify_integrity=False,
    )
    return mi


@pytest.fixture
def idx_dup():
    # compare tests/indexes/multi/conftest.py
    major_axis = Index(["foo", "bar", "baz", "qux"])
    minor_axis = Index(["one", "two"])

    major_codes = np.array([0, 0, 1, 0, 1, 1])
    minor_codes = np.array([0, 1, 0, 1, 0, 1])
    index_names = ["first", "second"]
    mi = MultiIndex(
        levels=[major_axis, minor_axis],
        codes=[major_codes, minor_codes],
        names=index_names,
        verify_integrity=False,
    )
    return mi


@pytest.fixture
def index_names():
    # names that match those in the idx fixture for testing equality of
    # names assigned to the idx
    return ["first", "second"]


@pytest.fixture
def narrow_multi_index():
    """
    Return a MultiIndex that is narrower than the display (<80 characters).
    """
    n = 1000
    ci = pd.CategoricalIndex(list("a" * n) + (["abc"] * n))
    dti = pd.date_range("2000-01-01", freq="s", periods=n * 2)
    return MultiIndex.from_arrays([ci, ci.codes + 9, dti], names=["a", "b", "dti"])


@pytest.fixture
def wide_multi_index():
    """
    Return a MultiIndex that is wider than the display (>80 characters).
    """
    n = 1000
    ci = pd.CategoricalIndex(list("a" * n) + (["abc"] * n))
    dti = pd.date_range("2000-01-01", freq="s", periods=n * 2)
    levels = [ci, ci.codes + 9, dti, dti, dti]
    names = ["a", "b", "dti_1", "dti_2", "dti_3"]
    return MultiIndex.from_arrays(levels, names=names)
