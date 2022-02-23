import numpy as np

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm
from pandas.tests.io.pytables.common import ensure_clean_path

from pandas.io.pytables import (
    HDFStore,
    read_hdf,
)


class TestHDFStoreSubclass:
    # GH 33748
    def test_supported_for_subclass_dataframe(self):
        data = {"a": [1, 2], "b": [3, 4]}
        sdf = tm.SubclassedDataFrame(data, dtype=np.intp)

        expected = DataFrame(data, dtype=np.intp)

        with ensure_clean_path("temp.h5") as path:
            sdf.to_hdf(path, "df")
            result = read_hdf(path, "df")
            tm.assert_frame_equal(result, expected)

        with ensure_clean_path("temp.h5") as path:
            with HDFStore(path) as store:
                store.put("df", sdf)
            result = read_hdf(path, "df")
            tm.assert_frame_equal(result, expected)

    def test_supported_for_subclass_series(self):
        data = [1, 2, 3]
        sser = tm.SubclassedSeries(data, dtype=np.intp)

        expected = Series(data, dtype=np.intp)

        with ensure_clean_path("temp.h5") as path:
            sser.to_hdf(path, "ser")
            result = read_hdf(path, "ser")
            tm.assert_series_equal(result, expected)

        with ensure_clean_path("temp.h5") as path:
            with HDFStore(path) as store:
                store.put("ser", sser)
            result = read_hdf(path, "ser")
            tm.assert_series_equal(result, expected)
