from datetime import datetime
import io
import os
from pathlib import Path

import dateutil.parser
import numpy as np
import pytest

from pandas.errors import (
    EmptyDataError,
    PerformanceWarning,
)
import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm


# https://github.com/cython/cython/issues/1720
@pytest.mark.filterwarnings("ignore:can't resolve package:ImportWarning")
class TestSAS7BDAT:
    @pytest.fixture(autouse=True)
    def setup_method(self, datapath):
        self.dirpath = datapath("io", "sas", "data")
        self.data = []
        self.test_ix = [list(range(1, 16)), [16]]
        for j in 1, 2:
            fname = os.path.join(self.dirpath, f"test_sas7bdat_{j}.csv")
            df = pd.read_csv(fname)
            epoch = datetime(1960, 1, 1)
            t1 = pd.to_timedelta(df["Column4"], unit="d")
            df["Column4"] = epoch + t1
            t2 = pd.to_timedelta(df["Column12"], unit="d")
            df["Column12"] = epoch + t2
            for k in range(df.shape[1]):
                col = df.iloc[:, k]
                if col.dtype == np.int64:
                    df.iloc[:, k] = df.iloc[:, k].astype(np.float64)
            self.data.append(df)

    @pytest.mark.slow
    def test_from_file(self):
        for j in 0, 1:
            df0 = self.data[j]
            for k in self.test_ix[j]:
                fname = os.path.join(self.dirpath, f"test{k}.sas7bdat")
                df = pd.read_sas(fname, encoding="utf-8")
                tm.assert_frame_equal(df, df0)

    @pytest.mark.slow
    def test_from_buffer(self):
        for j in 0, 1:
            df0 = self.data[j]
            for k in self.test_ix[j]:
                fname = os.path.join(self.dirpath, f"test{k}.sas7bdat")
                with open(fname, "rb") as f:
                    byts = f.read()
                buf = io.BytesIO(byts)
                with pd.read_sas(
                    buf, format="sas7bdat", iterator=True, encoding="utf-8"
                ) as rdr:
                    df = rdr.read()
                tm.assert_frame_equal(df, df0, check_exact=False)

    @pytest.mark.slow
    def test_from_iterator(self):
        for j in 0, 1:
            df0 = self.data[j]
            for k in self.test_ix[j]:
                fname = os.path.join(self.dirpath, f"test{k}.sas7bdat")
                with pd.read_sas(fname, iterator=True, encoding="utf-8") as rdr:
                    df = rdr.read(2)
                    tm.assert_frame_equal(df, df0.iloc[0:2, :])
                    df = rdr.read(3)
                    tm.assert_frame_equal(df, df0.iloc[2:5, :])

    @pytest.mark.slow
    def test_path_pathlib(self):
        for j in 0, 1:
            df0 = self.data[j]
            for k in self.test_ix[j]:
                fname = Path(os.path.join(self.dirpath, f"test{k}.sas7bdat"))
                df = pd.read_sas(fname, encoding="utf-8")
                tm.assert_frame_equal(df, df0)

    @td.skip_if_no("py.path")
    @pytest.mark.slow
    def test_path_localpath(self):
        from py.path import local as LocalPath

        for j in 0, 1:
            df0 = self.data[j]
            for k in self.test_ix[j]:
                fname = LocalPath(os.path.join(self.dirpath, f"test{k}.sas7bdat"))
                df = pd.read_sas(fname, encoding="utf-8")
                tm.assert_frame_equal(df, df0)

    @pytest.mark.slow
    def test_iterator_loop(self):
        # github #13654
        for j in 0, 1:
            for k in self.test_ix[j]:
                for chunksize in (3, 5, 10, 11):
                    fname = os.path.join(self.dirpath, f"test{k}.sas7bdat")
                    with pd.read_sas(
                        fname, chunksize=chunksize, encoding="utf-8"
                    ) as rdr:
                        y = 0
                        for x in rdr:
                            y += x.shape[0]
                    assert y == rdr.row_count

    def test_iterator_read_too_much(self):
        # github #14734
        k = self.test_ix[0][0]
        fname = os.path.join(self.dirpath, f"test{k}.sas7bdat")
        with pd.read_sas(
            fname, format="sas7bdat", iterator=True, encoding="utf-8"
        ) as rdr:
            d1 = rdr.read(rdr.row_count + 20)

        with pd.read_sas(fname, iterator=True, encoding="utf-8") as rdr:
            d2 = rdr.read(rdr.row_count + 20)
        tm.assert_frame_equal(d1, d2)


def test_encoding_options(datapath):
    fname = datapath("io", "sas", "data", "test1.sas7bdat")
    df1 = pd.read_sas(fname)
    df2 = pd.read_sas(fname, encoding="utf-8")
    for col in df1.columns:
        try:
            df1[col] = df1[col].str.decode("utf-8")
        except AttributeError:
            pass
    tm.assert_frame_equal(df1, df2)

    from pandas.io.sas.sas7bdat import SAS7BDATReader

    rdr = SAS7BDATReader(fname, convert_header_text=False)
    df3 = rdr.read()
    rdr.close()
    for x, y in zip(df1.columns, df3.columns):
        assert x == y.decode()


def test_productsales(datapath):
    fname = datapath("io", "sas", "data", "productsales.sas7bdat")
    df = pd.read_sas(fname, encoding="utf-8")
    fname = datapath("io", "sas", "data", "productsales.csv")
    df0 = pd.read_csv(fname, parse_dates=["MONTH"])
    vn = ["ACTUAL", "PREDICT", "QUARTER", "YEAR"]
    df0[vn] = df0[vn].astype(np.float64)
    tm.assert_frame_equal(df, df0)


def test_12659(datapath):
    fname = datapath("io", "sas", "data", "test_12659.sas7bdat")
    df = pd.read_sas(fname)
    fname = datapath("io", "sas", "data", "test_12659.csv")
    df0 = pd.read_csv(fname)
    df0 = df0.astype(np.float64)
    tm.assert_frame_equal(df, df0)


def test_airline(datapath):
    fname = datapath("io", "sas", "data", "airline.sas7bdat")
    df = pd.read_sas(fname)
    fname = datapath("io", "sas", "data", "airline.csv")
    df0 = pd.read_csv(fname)
    df0 = df0.astype(np.float64)
    tm.assert_frame_equal(df, df0, check_exact=False)


def test_date_time(datapath):
    # Support of different SAS date/datetime formats (PR #15871)
    fname = datapath("io", "sas", "data", "datetime.sas7bdat")
    df = pd.read_sas(fname)
    fname = datapath("io", "sas", "data", "datetime.csv")
    df0 = pd.read_csv(
        fname, parse_dates=["Date1", "Date2", "DateTime", "DateTimeHi", "Taiw"]
    )
    # GH 19732: Timestamps imported from sas will incur floating point errors
    df.iloc[:, 3] = df.iloc[:, 3].dt.round("us")
    tm.assert_frame_equal(df, df0)


def test_compact_numerical_values(datapath):
    # Regression test for #21616
    fname = datapath("io", "sas", "data", "cars.sas7bdat")
    df = pd.read_sas(fname, encoding="latin-1")
    # The two columns CYL and WGT in cars.sas7bdat have column
    # width < 8 and only contain integral values.
    # Test that pandas doesn't corrupt the numbers by adding
    # decimals.
    result = df["WGT"]
    expected = df["WGT"].round()
    tm.assert_series_equal(result, expected, check_exact=True)
    result = df["CYL"]
    expected = df["CYL"].round()
    tm.assert_series_equal(result, expected, check_exact=True)


def test_many_columns(datapath, using_array_manager):
    # Test for looking for column information in more places (PR #22628)
    fname = datapath("io", "sas", "data", "many_columns.sas7bdat")
    expected_warning = None
    if not using_array_manager:
        expected_warning = PerformanceWarning
    with tm.assert_produces_warning(expected_warning):
        # Many DataFrame.insert calls
        df = pd.read_sas(fname, encoding="latin-1")

    fname = datapath("io", "sas", "data", "many_columns.csv")
    df0 = pd.read_csv(fname, encoding="latin-1")
    tm.assert_frame_equal(df, df0)


def test_inconsistent_number_of_rows(datapath):
    # Regression test for issue #16615. (PR #22628)
    fname = datapath("io", "sas", "data", "load_log.sas7bdat")
    df = pd.read_sas(fname, encoding="latin-1")
    assert len(df) == 2097


def test_zero_variables(datapath):
    # Check if the SAS file has zero variables (PR #18184)
    fname = datapath("io", "sas", "data", "zero_variables.sas7bdat")
    with pytest.raises(EmptyDataError, match="No columns to parse from file"):
        pd.read_sas(fname)


def test_corrupt_read(datapath):
    # We don't really care about the exact failure, the important thing is
    # that the resource should be cleaned up afterwards (BUG #35566)
    fname = datapath("io", "sas", "data", "corrupt.sas7bdat")
    msg = "'SAS7BDATReader' object has no attribute 'row_count'"
    with pytest.raises(AttributeError, match=msg):
        pd.read_sas(fname)


def round_datetime_to_ms(ts):
    if isinstance(ts, datetime):
        return ts.replace(microsecond=int(round(ts.microsecond, -3) / 1000) * 1000)
    elif isinstance(ts, str):
        _ts = dateutil.parser.parse(timestr=ts)
        return _ts.replace(microsecond=int(round(_ts.microsecond, -3) / 1000) * 1000)
    else:
        return ts


def test_max_sas_date(datapath):
    # GH 20927
    # NB. max datetime in SAS dataset is 31DEC9999:23:59:59.999
    #    but this is read as 29DEC9999:23:59:59.998993 by a buggy
    #    sas7bdat module
    fname = datapath("io", "sas", "data", "max_sas_date.sas7bdat")
    df = pd.read_sas(fname, encoding="iso-8859-1")

    # SAS likes to left pad strings with spaces - lstrip before comparing
    df = df.applymap(lambda x: x.lstrip() if isinstance(x, str) else x)
    # GH 19732: Timestamps imported from sas will incur floating point errors
    try:
        df["dt_as_dt"] = df["dt_as_dt"].dt.round("us")
    except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:
        df = df.applymap(round_datetime_to_ms)
    except AttributeError:
        df["dt_as_dt"] = df["dt_as_dt"].apply(round_datetime_to_ms)
    # if there are any date/times > pandas.Timestamp.max then ALL in that chunk
    # are returned as datetime.datetime
    expected = pd.DataFrame(
        {
            "text": ["max", "normal"],
            "dt_as_float": [253717747199.999, 1880323199.999],
            "dt_as_dt": [
                datetime(9999, 12, 29, 23, 59, 59, 999000),
                datetime(2019, 8, 1, 23, 59, 59, 999000),
            ],
            "date_as_float": [2936547.0, 21762.0],
            "date_as_date": [datetime(9999, 12, 29), datetime(2019, 8, 1)],
        },
        columns=["text", "dt_as_float", "dt_as_dt", "date_as_float", "date_as_date"],
    )
    tm.assert_frame_equal(df, expected)


def test_max_sas_date_iterator(datapath):
    # GH 20927
    # when called as an iterator, only those chunks with a date > pd.Timestamp.max
    # are returned as datetime.datetime, if this happens that whole chunk is returned
    # as datetime.datetime
    col_order = ["text", "dt_as_float", "dt_as_dt", "date_as_float", "date_as_date"]
    fname = datapath("io", "sas", "data", "max_sas_date.sas7bdat")
    results = []
    for df in pd.read_sas(fname, encoding="iso-8859-1", chunksize=1):
        # SAS likes to left pad strings with spaces - lstrip before comparing
        df = df.applymap(lambda x: x.lstrip() if isinstance(x, str) else x)
        # GH 19732: Timestamps imported from sas will incur floating point errors
        try:
            df["dt_as_dt"] = df["dt_as_dt"].dt.round("us")
        except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:
            df = df.applymap(round_datetime_to_ms)
        except AttributeError:
            df["dt_as_dt"] = df["dt_as_dt"].apply(round_datetime_to_ms)
        df.reset_index(inplace=True, drop=True)
        results.append(df)
    expected = [
        pd.DataFrame(
            {
                "text": ["max"],
                "dt_as_float": [253717747199.999],
                "dt_as_dt": [datetime(9999, 12, 29, 23, 59, 59, 999000)],
                "date_as_float": [2936547.0],
                "date_as_date": [datetime(9999, 12, 29)],
            },
            columns=col_order,
        ),
        pd.DataFrame(
            {
                "text": ["normal"],
                "dt_as_float": [1880323199.999],
                "dt_as_dt": [np.datetime64("2019-08-01 23:59:59.999")],
                "date_as_float": [21762.0],
                "date_as_date": [np.datetime64("2019-08-01")],
            },
            columns=col_order,
        ),
    ]
    for result, expected in zip(results, expected):
        tm.assert_frame_equal(result, expected)


def test_null_date(datapath):
    fname = datapath("io", "sas", "data", "dates_null.sas7bdat")
    df = pd.read_sas(fname, encoding="utf-8")

    expected = pd.DataFrame(
        {
            "datecol": [
                datetime(9999, 12, 29),
                pd.NaT,
            ],
            "datetimecol": [
                datetime(9999, 12, 29, 23, 59, 59, 998993),
                pd.NaT,
            ],
        },
    )
    tm.assert_frame_equal(df, expected)
