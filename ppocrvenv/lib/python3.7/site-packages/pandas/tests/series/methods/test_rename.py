from datetime import datetime

import numpy as np

from pandas import (
    Index,
    Series,
)
import pandas._testing as tm


class TestRename:
    def test_rename(self, datetime_series):
        ts = datetime_series
        renamer = lambda x: x.strftime("%Y%m%d")
        renamed = ts.rename(renamer)
        assert renamed.index[0] == renamer(ts.index[0])

        # dict
        rename_dict = dict(zip(ts.index, renamed.index))
        renamed2 = ts.rename(rename_dict)
        tm.assert_series_equal(renamed, renamed2)

        # partial dict
        s = Series(np.arange(4), index=["a", "b", "c", "d"], dtype="int64")
        renamed = s.rename({"b": "foo", "d": "bar"})
        tm.assert_index_equal(renamed.index, Index(["a", "foo", "c", "bar"]))

        # index with name
        renamer = Series(
            np.arange(4), index=Index(["a", "b", "c", "d"], name="name"), dtype="int64"
        )
        renamed = renamer.rename({})
        assert renamed.index.name == renamer.index.name

    def test_rename_by_series(self):
        s = Series(range(5), name="foo")
        renamer = Series({1: 10, 2: 20})
        result = s.rename(renamer)
        expected = Series(range(5), index=[0, 10, 20, 3, 4], name="foo")
        tm.assert_series_equal(result, expected)

    def test_rename_set_name(self):
        s = Series(range(4), index=list("abcd"))
        for name in ["foo", 123, 123.0, datetime(2001, 11, 11), ("foo",)]:
            result = s.rename(name)
            assert result.name == name
            tm.assert_numpy_array_equal(result.index.values, s.index.values)
            assert s.name is None

    def test_rename_set_name_inplace(self):
        s = Series(range(3), index=list("abc"))
        for name in ["foo", 123, 123.0, datetime(2001, 11, 11), ("foo",)]:
            s.rename(name, inplace=True)
            assert s.name == name

            exp = np.array(["a", "b", "c"], dtype=np.object_)
            tm.assert_numpy_array_equal(s.index.values, exp)

    def test_rename_axis_supported(self):
        # Supporting axis for compatibility, detailed in GH-18589
        s = Series(range(5))
        s.rename({}, axis=0)
        s.rename({}, axis="index")
        # FIXME: dont leave commenred-out
        # TODO: clean up shared index validation
        # with pytest.raises(ValueError, match="No axis named 5"):
        #     s.rename({}, axis=5)

    def test_rename_inplace(self, datetime_series):
        renamer = lambda x: x.strftime("%Y%m%d")
        expected = renamer(datetime_series.index[0])

        datetime_series.rename(renamer, inplace=True)
        assert datetime_series.index[0] == expected

    def test_rename_with_custom_indexer(self):
        # GH 27814
        class MyIndexer:
            pass

        ix = MyIndexer()
        s = Series([1, 2, 3]).rename(ix)
        assert s.name is ix

    def test_rename_with_custom_indexer_inplace(self):
        # GH 27814
        class MyIndexer:
            pass

        ix = MyIndexer()
        s = Series([1, 2, 3])
        s.rename(ix, inplace=True)
        assert s.name is ix

    def test_rename_callable(self):
        # GH 17407
        s = Series(range(1, 6), index=Index(range(2, 7), name="IntIndex"))
        result = s.rename(str)
        expected = s.rename(lambda i: str(i))
        tm.assert_series_equal(result, expected)

        assert result.name == expected.name
