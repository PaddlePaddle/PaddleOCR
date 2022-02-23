import numpy as np
import pytest

from pandas import CategoricalIndex
import pandas._testing as tm


class TestFillNA:
    def test_fillna_categorical(self):
        # GH#11343
        idx = CategoricalIndex([1.0, np.nan, 3.0, 1.0], name="x")
        # fill by value in categories
        exp = CategoricalIndex([1.0, 1.0, 3.0, 1.0], name="x")
        tm.assert_index_equal(idx.fillna(1.0), exp)

        cat = idx._data

        # fill by value not in categories raises ValueError on EA, casts on CI
        msg = "Cannot setitem on a Categorical with a new category"
        with pytest.raises(ValueError, match=msg):
            cat.fillna(2.0)

        result = idx.fillna(2.0)
        expected = idx.astype(object).fillna(2.0)
        tm.assert_index_equal(result, expected)

    def test_fillna_copies_with_no_nas(self):
        # Nothing to fill, should still get a copy
        ci = CategoricalIndex([0, 1, 1])
        cat = ci._data
        result = ci.fillna(0)
        assert result._values._ndarray is not cat._ndarray
        assert result._values._ndarray.base is None

        # Same check directly on the Categorical object
        result = cat.fillna(0)
        assert result._ndarray is not cat._ndarray
        assert result._ndarray.base is None

    def test_fillna_validates_with_no_nas(self):
        # We validate the fill value even if fillna is a no-op
        ci = CategoricalIndex([2, 3, 3])
        cat = ci._data

        msg = "Cannot setitem on a Categorical with a new category"
        res = ci.fillna(False)
        # nothing to fill, so we dont cast
        tm.assert_index_equal(res, ci)

        # Same check directly on the Categorical
        with pytest.raises(ValueError, match=msg):
            cat.fillna(False)
