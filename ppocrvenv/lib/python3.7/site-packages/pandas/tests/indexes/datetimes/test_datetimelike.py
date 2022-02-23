""" generic tests from the Datetimelike class """
import pytest

from pandas import (
    DatetimeIndex,
    date_range,
)
import pandas._testing as tm
from pandas.tests.indexes.datetimelike import DatetimeLike


class TestDatetimeIndex(DatetimeLike):
    _index_cls = DatetimeIndex

    @pytest.fixture
    def simple_index(self) -> DatetimeIndex:
        return date_range("20130101", periods=5)

    @pytest.fixture(
        params=[tm.makeDateIndex(10), date_range("20130110", periods=10, freq="-1D")],
        ids=["index_inc", "index_dec"],
    )
    def index(self, request):
        return request.param

    def test_format(self, simple_index):
        # GH35439
        idx = simple_index
        expected = [f"{x:%Y-%m-%d}" for x in idx]
        assert idx.format() == expected

    def test_shift(self):
        pass  # handled in test_ops

    def test_intersection(self):
        pass  # handled in test_setops

    def test_union(self):
        pass  # handled in test_setops
