import numpy as np
import pytest

from pandas import (
    IntervalIndex,
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.tests.indexes.common import Base


class TestBase(Base):
    """
    Tests specific to the shared common index tests; unrelated tests should be placed
    in test_interval.py or the specific test file (e.g. test_astype.py)
    """

    _index_cls = IntervalIndex

    @pytest.fixture
    def simple_index(self) -> IntervalIndex:
        return self._index_cls.from_breaks(range(11), closed="right")

    @pytest.fixture
    def index(self):
        return tm.makeIntervalIndex(10)

    def create_index(self, *, closed="right"):
        return IntervalIndex.from_breaks(range(11), closed=closed)

    def test_repr_max_seq_item_setting(self):
        # override base test: not a valid repr as we use interval notation
        pass

    def test_repr_roundtrip(self):
        # override base test: not a valid repr as we use interval notation
        pass

    def test_take(self, closed):
        index = self.create_index(closed=closed)

        result = index.take(range(10))
        tm.assert_index_equal(result, index)

        result = index.take([0, 0, 1])
        expected = IntervalIndex.from_arrays([0, 0, 1], [1, 1, 2], closed=closed)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("klass", [list, tuple, np.array, Series])
    def test_where(self, simple_index, klass):
        idx = simple_index
        cond = [True] * len(idx)
        expected = idx
        result = expected.where(klass(cond))
        tm.assert_index_equal(result, expected)

        cond = [False] + [True] * len(idx[1:])
        expected = IntervalIndex([np.nan] + idx[1:].tolist())
        result = idx.where(klass(cond))
        tm.assert_index_equal(result, expected)

    def test_getitem_2d_deprecated(self, simple_index):
        # GH#30588 multi-dim indexing is deprecated, but raising is also acceptable
        idx = simple_index
        with pytest.raises(ValueError, match="multi-dimensional indexing not allowed"):
            with tm.assert_produces_warning(FutureWarning):
                idx[:, None]


class TestPutmask:
    @pytest.mark.parametrize("tz", ["US/Pacific", None])
    def test_putmask_dt64(self, tz):
        # GH#37968
        dti = date_range("2016-01-01", periods=9, tz=tz)
        idx = IntervalIndex.from_breaks(dti)
        mask = np.zeros(idx.shape, dtype=bool)
        mask[0:3] = True

        result = idx.putmask(mask, idx[-1])
        expected = IntervalIndex([idx[-1]] * 3 + list(idx[3:]))
        tm.assert_index_equal(result, expected)

    def test_putmask_td64(self):
        # GH#37968
        dti = date_range("2016-01-01", periods=9)
        tdi = dti - dti[0]
        idx = IntervalIndex.from_breaks(tdi)
        mask = np.zeros(idx.shape, dtype=bool)
        mask[0:3] = True

        result = idx.putmask(mask, idx[-1])
        expected = IntervalIndex([idx[-1]] * 3 + list(idx[3:]))
        tm.assert_index_equal(result, expected)
