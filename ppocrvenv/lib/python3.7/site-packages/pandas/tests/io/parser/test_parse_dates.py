"""
Tests date parsing functionality for all of the
parsers defined in parsers.py
"""

from datetime import (
    date,
    datetime,
)
from io import StringIO

from dateutil.parser import parse as du_parse
from hypothesis import (
    given,
    settings,
    strategies as st,
)
import numpy as np
import pytest
import pytz

from pandas._libs.tslibs import parsing
from pandas._libs.tslibs.parsing import parse_datetime_string
from pandas.compat import (
    is_platform_windows,
    np_array_datetime64_compat,
)

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range

import pandas.io.date_converters as conv

# constant
_DEFAULT_DATETIME = datetime(1, 1, 1)

# Strategy for hypothesis
if is_platform_windows():
    date_strategy = st.datetimes(min_value=datetime(1900, 1, 1))
else:
    date_strategy = st.datetimes()


def test_read_csv_with_custom_date_parser(all_parsers):
    # GH36111
    def __custom_date_parser(time):
        time = time.astype(np.float_)
        time = time.astype(np.int_)  # convert float seconds to int type
        return pd.to_timedelta(time, unit="s")

    testdata = StringIO(
        """time e n h
        41047.00 -98573.7297 871458.0640 389.0089
        41048.00 -98573.7299 871458.0640 389.0089
        41049.00 -98573.7300 871458.0642 389.0088
        41050.00 -98573.7299 871458.0643 389.0088
        41051.00 -98573.7302 871458.0640 389.0086
        """
    )
    result = all_parsers.read_csv(
        testdata,
        delim_whitespace=True,
        parse_dates=True,
        date_parser=__custom_date_parser,
        index_col="time",
    )
    time = [41047, 41048, 41049, 41050, 41051]
    time = pd.TimedeltaIndex([pd.to_timedelta(i, unit="s") for i in time], name="time")
    expected = DataFrame(
        {
            "e": [-98573.7297, -98573.7299, -98573.7300, -98573.7299, -98573.7302],
            "n": [871458.0640, 871458.0640, 871458.0642, 871458.0643, 871458.0640],
            "h": [389.0089, 389.0089, 389.0088, 389.0088, 389.0086],
        },
        index=time,
    )

    tm.assert_frame_equal(result, expected)


def test_separator_date_conflict(all_parsers):
    # Regression test for gh-4678
    #
    # Make sure thousands separator and
    # date parsing do not conflict.
    parser = all_parsers
    data = "06-02-2013;13:00;1-000.215"
    expected = DataFrame(
        [[datetime(2013, 6, 2, 13, 0, 0), 1000.215]], columns=["Date", 2]
    )

    df = parser.read_csv(
        StringIO(data),
        sep=";",
        thousands="-",
        parse_dates={"Date": [0, 1]},
        header=None,
    )
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("keep_date_col", [True, False])
def test_multiple_date_col_custom(all_parsers, keep_date_col):
    data = """\
KORD,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000
"""
    parser = all_parsers

    def date_parser(*date_cols):
        """
        Test date parser.

        Parameters
        ----------
        date_cols : args
            The list of data columns to parse.

        Returns
        -------
        parsed : Series
        """
        return parsing.try_parse_dates(parsing.concat_date_cols(date_cols))

    result = parser.read_csv(
        StringIO(data),
        header=None,
        date_parser=date_parser,
        prefix="X",
        parse_dates={"actual": [1, 2], "nominal": [1, 3]},
        keep_date_col=keep_date_col,
    )
    expected = DataFrame(
        [
            [
                datetime(1999, 1, 27, 19, 0),
                datetime(1999, 1, 27, 18, 56),
                "KORD",
                "19990127",
                " 19:00:00",
                " 18:56:00",
                0.81,
                2.81,
                7.2,
                0.0,
                280.0,
            ],
            [
                datetime(1999, 1, 27, 20, 0),
                datetime(1999, 1, 27, 19, 56),
                "KORD",
                "19990127",
                " 20:00:00",
                " 19:56:00",
                0.01,
                2.21,
                7.2,
                0.0,
                260.0,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                datetime(1999, 1, 27, 20, 56),
                "KORD",
                "19990127",
                " 21:00:00",
                " 20:56:00",
                -0.59,
                2.21,
                5.7,
                0.0,
                280.0,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                datetime(1999, 1, 27, 21, 18),
                "KORD",
                "19990127",
                " 21:00:00",
                " 21:18:00",
                -0.99,
                2.01,
                3.6,
                0.0,
                270.0,
            ],
            [
                datetime(1999, 1, 27, 22, 0),
                datetime(1999, 1, 27, 21, 56),
                "KORD",
                "19990127",
                " 22:00:00",
                " 21:56:00",
                -0.59,
                1.71,
                5.1,
                0.0,
                290.0,
            ],
            [
                datetime(1999, 1, 27, 23, 0),
                datetime(1999, 1, 27, 22, 56),
                "KORD",
                "19990127",
                " 23:00:00",
                " 22:56:00",
                -0.59,
                1.71,
                4.6,
                0.0,
                280.0,
            ],
        ],
        columns=[
            "actual",
            "nominal",
            "X0",
            "X1",
            "X2",
            "X3",
            "X4",
            "X5",
            "X6",
            "X7",
            "X8",
        ],
    )

    if not keep_date_col:
        expected = expected.drop(["X1", "X2", "X3"], axis=1)
    elif parser.engine == "python":
        expected["X1"] = expected["X1"].astype(np.int64)

    # Python can sometimes be flaky about how
    # the aggregated columns are entered, so
    # this standardizes the order.
    result = result[expected.columns]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("container", [list, tuple, Index, Series])
@pytest.mark.parametrize("dim", [1, 2])
def test_concat_date_col_fail(container, dim):
    msg = "not all elements from date_cols are numpy arrays"
    value = "19990127"

    date_cols = tuple(container([value]) for _ in range(dim))

    with pytest.raises(ValueError, match=msg):
        parsing.concat_date_cols(date_cols)


@pytest.mark.parametrize("keep_date_col", [True, False])
def test_multiple_date_col(all_parsers, keep_date_col):
    data = """\
KORD,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000
"""
    parser = all_parsers
    result = parser.read_csv(
        StringIO(data),
        header=None,
        prefix="X",
        parse_dates=[[1, 2], [1, 3]],
        keep_date_col=keep_date_col,
    )
    expected = DataFrame(
        [
            [
                datetime(1999, 1, 27, 19, 0),
                datetime(1999, 1, 27, 18, 56),
                "KORD",
                "19990127",
                " 19:00:00",
                " 18:56:00",
                0.81,
                2.81,
                7.2,
                0.0,
                280.0,
            ],
            [
                datetime(1999, 1, 27, 20, 0),
                datetime(1999, 1, 27, 19, 56),
                "KORD",
                "19990127",
                " 20:00:00",
                " 19:56:00",
                0.01,
                2.21,
                7.2,
                0.0,
                260.0,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                datetime(1999, 1, 27, 20, 56),
                "KORD",
                "19990127",
                " 21:00:00",
                " 20:56:00",
                -0.59,
                2.21,
                5.7,
                0.0,
                280.0,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                datetime(1999, 1, 27, 21, 18),
                "KORD",
                "19990127",
                " 21:00:00",
                " 21:18:00",
                -0.99,
                2.01,
                3.6,
                0.0,
                270.0,
            ],
            [
                datetime(1999, 1, 27, 22, 0),
                datetime(1999, 1, 27, 21, 56),
                "KORD",
                "19990127",
                " 22:00:00",
                " 21:56:00",
                -0.59,
                1.71,
                5.1,
                0.0,
                290.0,
            ],
            [
                datetime(1999, 1, 27, 23, 0),
                datetime(1999, 1, 27, 22, 56),
                "KORD",
                "19990127",
                " 23:00:00",
                " 22:56:00",
                -0.59,
                1.71,
                4.6,
                0.0,
                280.0,
            ],
        ],
        columns=[
            "X1_X2",
            "X1_X3",
            "X0",
            "X1",
            "X2",
            "X3",
            "X4",
            "X5",
            "X6",
            "X7",
            "X8",
        ],
    )

    if not keep_date_col:
        expected = expected.drop(["X1", "X2", "X3"], axis=1)
    elif parser.engine == "python":
        expected["X1"] = expected["X1"].astype(np.int64)

    tm.assert_frame_equal(result, expected)


def test_date_col_as_index_col(all_parsers):
    data = """\
KORD,19990127 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
"""
    parser = all_parsers
    result = parser.read_csv(
        StringIO(data), header=None, prefix="X", parse_dates=[1], index_col=1
    )

    index = Index(
        [
            datetime(1999, 1, 27, 19, 0),
            datetime(1999, 1, 27, 20, 0),
            datetime(1999, 1, 27, 21, 0),
            datetime(1999, 1, 27, 21, 0),
            datetime(1999, 1, 27, 22, 0),
        ],
        name="X1",
    )
    expected = DataFrame(
        [
            ["KORD", " 18:56:00", 0.81, 2.81, 7.2, 0.0, 280.0],
            ["KORD", " 19:56:00", 0.01, 2.21, 7.2, 0.0, 260.0],
            ["KORD", " 20:56:00", -0.59, 2.21, 5.7, 0.0, 280.0],
            ["KORD", " 21:18:00", -0.99, 2.01, 3.6, 0.0, 270.0],
            ["KORD", " 21:56:00", -0.59, 1.71, 5.1, 0.0, 290.0],
        ],
        columns=["X0", "X2", "X3", "X4", "X5", "X6", "X7"],
        index=index,
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "date_parser, warning",
    ([conv.parse_date_time, FutureWarning], [pd.to_datetime, None]),
)
def test_multiple_date_cols_int_cast(all_parsers, date_parser, warning):
    data = (
        "KORD,19990127, 19:00:00, 18:56:00, 0.8100\n"
        "KORD,19990127, 20:00:00, 19:56:00, 0.0100\n"
        "KORD,19990127, 21:00:00, 20:56:00, -0.5900\n"
        "KORD,19990127, 21:00:00, 21:18:00, -0.9900\n"
        "KORD,19990127, 22:00:00, 21:56:00, -0.5900\n"
        "KORD,19990127, 23:00:00, 22:56:00, -0.5900"
    )
    parse_dates = {"actual": [1, 2], "nominal": [1, 3]}
    parser = all_parsers

    with tm.assert_produces_warning(warning, check_stacklevel=False):
        result = parser.read_csv(
            StringIO(data),
            header=None,
            date_parser=date_parser,
            parse_dates=parse_dates,
            prefix="X",
        )

    expected = DataFrame(
        [
            [datetime(1999, 1, 27, 19, 0), datetime(1999, 1, 27, 18, 56), "KORD", 0.81],
            [datetime(1999, 1, 27, 20, 0), datetime(1999, 1, 27, 19, 56), "KORD", 0.01],
            [
                datetime(1999, 1, 27, 21, 0),
                datetime(1999, 1, 27, 20, 56),
                "KORD",
                -0.59,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                datetime(1999, 1, 27, 21, 18),
                "KORD",
                -0.99,
            ],
            [
                datetime(1999, 1, 27, 22, 0),
                datetime(1999, 1, 27, 21, 56),
                "KORD",
                -0.59,
            ],
            [
                datetime(1999, 1, 27, 23, 0),
                datetime(1999, 1, 27, 22, 56),
                "KORD",
                -0.59,
            ],
        ],
        columns=["actual", "nominal", "X0", "X4"],
    )

    # Python can sometimes be flaky about how
    # the aggregated columns are entered, so
    # this standardizes the order.
    result = result[expected.columns]
    tm.assert_frame_equal(result, expected)


def test_multiple_date_col_timestamp_parse(all_parsers):
    parser = all_parsers
    data = """05/31/2012,15:30:00.029,1306.25,1,E,0,,1306.25
05/31/2012,15:30:00.029,1306.25,8,E,0,,1306.25"""

    result = parser.read_csv(
        StringIO(data), parse_dates=[[0, 1]], header=None, date_parser=Timestamp
    )
    expected = DataFrame(
        [
            [
                Timestamp("05/31/2012, 15:30:00.029"),
                1306.25,
                1,
                "E",
                0,
                np.nan,
                1306.25,
            ],
            [
                Timestamp("05/31/2012, 15:30:00.029"),
                1306.25,
                8,
                "E",
                0,
                np.nan,
                1306.25,
            ],
        ],
        columns=["0_1", 2, 3, 4, 5, 6, 7],
    )
    tm.assert_frame_equal(result, expected)


def test_multiple_date_cols_with_header(all_parsers):
    parser = all_parsers
    data = """\
ID,date,NominalTime,ActualTime,TDew,TAir,Windspeed,Precip,WindDir
KORD,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000"""

    result = parser.read_csv(StringIO(data), parse_dates={"nominal": [1, 2]})
    expected = DataFrame(
        [
            [
                datetime(1999, 1, 27, 19, 0),
                "KORD",
                " 18:56:00",
                0.81,
                2.81,
                7.2,
                0.0,
                280.0,
            ],
            [
                datetime(1999, 1, 27, 20, 0),
                "KORD",
                " 19:56:00",
                0.01,
                2.21,
                7.2,
                0.0,
                260.0,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                "KORD",
                " 20:56:00",
                -0.59,
                2.21,
                5.7,
                0.0,
                280.0,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                "KORD",
                " 21:18:00",
                -0.99,
                2.01,
                3.6,
                0.0,
                270.0,
            ],
            [
                datetime(1999, 1, 27, 22, 0),
                "KORD",
                " 21:56:00",
                -0.59,
                1.71,
                5.1,
                0.0,
                290.0,
            ],
            [
                datetime(1999, 1, 27, 23, 0),
                "KORD",
                " 22:56:00",
                -0.59,
                1.71,
                4.6,
                0.0,
                280.0,
            ],
        ],
        columns=[
            "nominal",
            "ID",
            "ActualTime",
            "TDew",
            "TAir",
            "Windspeed",
            "Precip",
            "WindDir",
        ],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data,parse_dates,msg",
    [
        (
            """\
date_NominalTime,date,NominalTime
KORD1,19990127, 19:00:00
KORD2,19990127, 20:00:00""",
            [[1, 2]],
            ("New date column already in dict date_NominalTime"),
        ),
        (
            """\
ID,date,nominalTime
KORD,19990127, 19:00:00
KORD,19990127, 20:00:00""",
            {"ID": [1, 2]},
            "Date column ID already in dict",
        ),
    ],
)
def test_multiple_date_col_name_collision(all_parsers, data, parse_dates, msg):
    parser = all_parsers

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), parse_dates=parse_dates)


def test_date_parser_int_bug(all_parsers):
    # see gh-3071
    parser = all_parsers
    data = (
        "posix_timestamp,elapsed,sys,user,queries,query_time,rows,"
        "accountid,userid,contactid,level,silo,method\n"
        "1343103150,0.062353,0,4,6,0.01690,3,"
        "12345,1,-1,3,invoice_InvoiceResource,search\n"
    )

    result = parser.read_csv(
        StringIO(data),
        index_col=0,
        parse_dates=[0],
        date_parser=lambda x: datetime.utcfromtimestamp(int(x)),
    )
    expected = DataFrame(
        [
            [
                0.062353,
                0,
                4,
                6,
                0.01690,
                3,
                12345,
                1,
                -1,
                3,
                "invoice_InvoiceResource",
                "search",
            ]
        ],
        columns=[
            "elapsed",
            "sys",
            "user",
            "queries",
            "query_time",
            "rows",
            "accountid",
            "userid",
            "contactid",
            "level",
            "silo",
            "method",
        ],
        index=Index([Timestamp("2012-07-24 04:12:30")], name="posix_timestamp"),
    )
    tm.assert_frame_equal(result, expected)


def test_nat_parse(all_parsers):
    # see gh-3062
    parser = all_parsers
    df = DataFrame(
        dict({"A": np.arange(10, dtype="float64"), "B": Timestamp("20010101")})
    )
    df.iloc[3:6, :] = np.nan

    with tm.ensure_clean("__nat_parse_.csv") as path:
        df.to_csv(path)

        result = parser.read_csv(path, index_col=0, parse_dates=["B"])
        tm.assert_frame_equal(result, df)


def test_csv_custom_parser(all_parsers):
    data = """A,B,C
20090101,a,1,2
20090102,b,3,4
20090103,c,4,5
"""
    parser = all_parsers
    result = parser.read_csv(
        StringIO(data), date_parser=lambda x: datetime.strptime(x, "%Y%m%d")
    )
    expected = parser.read_csv(StringIO(data), parse_dates=True)
    tm.assert_frame_equal(result, expected)


def test_parse_dates_implicit_first_col(all_parsers):
    data = """A,B,C
20090101,a,1,2
20090102,b,3,4
20090103,c,4,5
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), parse_dates=True)

    expected = parser.read_csv(StringIO(data), index_col=0, parse_dates=True)
    tm.assert_frame_equal(result, expected)


def test_parse_dates_string(all_parsers):
    data = """date,A,B,C
20090101,a,1,2
20090102,b,3,4
20090103,c,4,5
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col="date", parse_dates=["date"])
    # freq doesn't round-trip
    index = DatetimeIndex(
        list(date_range("1/1/2009", periods=3)), name="date", freq=None
    )

    expected = DataFrame(
        {"A": ["a", "b", "c"], "B": [1, 3, 4], "C": [2, 4, 5]}, index=index
    )
    tm.assert_frame_equal(result, expected)


# Bug in https://github.com/dateutil/dateutil/issues/217
# has been addressed, but we just don't pass in the `yearfirst`
@pytest.mark.xfail(reason="yearfirst is not surfaced in read_*")
@pytest.mark.parametrize("parse_dates", [[["date", "time"]], [[0, 1]]])
def test_yy_format_with_year_first(all_parsers, parse_dates):
    data = """date,time,B,C
090131,0010,1,2
090228,1020,3,4
090331,0830,5,6
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=0, parse_dates=parse_dates)
    index = DatetimeIndex(
        [
            datetime(2009, 1, 31, 0, 10, 0),
            datetime(2009, 2, 28, 10, 20, 0),
            datetime(2009, 3, 31, 8, 30, 0),
        ],
        dtype=object,
        name="date_time",
    )
    expected = DataFrame({"B": [1, 3, 5], "C": [2, 4, 6]}, index=index)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("parse_dates", [[0, 2], ["a", "c"]])
def test_parse_dates_column_list(all_parsers, parse_dates):
    data = "a,b,c\n01/01/2010,1,15/02/2010"
    parser = all_parsers

    expected = DataFrame(
        {"a": [datetime(2010, 1, 1)], "b": [1], "c": [datetime(2010, 2, 15)]}
    )
    expected = expected.set_index(["a", "b"])

    result = parser.read_csv(
        StringIO(data), index_col=[0, 1], parse_dates=parse_dates, dayfirst=True
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("index_col", [[0, 1], [1, 0]])
def test_multi_index_parse_dates(all_parsers, index_col):
    data = """index1,index2,A,B,C
20090101,one,a,1,2
20090101,two,b,3,4
20090101,three,c,4,5
20090102,one,a,1,2
20090102,two,b,3,4
20090102,three,c,4,5
20090103,one,a,1,2
20090103,two,b,3,4
20090103,three,c,4,5
"""
    parser = all_parsers
    index = MultiIndex.from_product(
        [
            (datetime(2009, 1, 1), datetime(2009, 1, 2), datetime(2009, 1, 3)),
            ("one", "two", "three"),
        ],
        names=["index1", "index2"],
    )

    # Out of order.
    if index_col == [1, 0]:
        index = index.swaplevel(0, 1)

    expected = DataFrame(
        [
            ["a", 1, 2],
            ["b", 3, 4],
            ["c", 4, 5],
            ["a", 1, 2],
            ["b", 3, 4],
            ["c", 4, 5],
            ["a", 1, 2],
            ["b", 3, 4],
            ["c", 4, 5],
        ],
        columns=["A", "B", "C"],
        index=index,
    )
    result = parser.read_csv(StringIO(data), index_col=index_col, parse_dates=True)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("kwargs", [{"dayfirst": True}, {"day_first": True}])
def test_parse_dates_custom_euro_format(all_parsers, kwargs):
    parser = all_parsers
    data = """foo,bar,baz
31/01/2010,1,2
01/02/2010,1,NA
02/02/2010,1,2
"""
    if "dayfirst" in kwargs:
        df = parser.read_csv(
            StringIO(data),
            names=["time", "Q", "NTU"],
            date_parser=lambda d: du_parse(d, **kwargs),
            header=0,
            index_col=0,
            parse_dates=True,
            na_values=["NA"],
        )
        exp_index = Index(
            [datetime(2010, 1, 31), datetime(2010, 2, 1), datetime(2010, 2, 2)],
            name="time",
        )
        expected = DataFrame(
            {"Q": [1, 1, 1], "NTU": [2, np.nan, 2]},
            index=exp_index,
            columns=["Q", "NTU"],
        )
        tm.assert_frame_equal(df, expected)
    else:
        msg = "got an unexpected keyword argument 'day_first'"
        with pytest.raises(TypeError, match=msg), tm.assert_produces_warning(
            FutureWarning
        ):
            parser.read_csv(
                StringIO(data),
                names=["time", "Q", "NTU"],
                date_parser=lambda d: du_parse(d, **kwargs),
                skiprows=[0],
                index_col=0,
                parse_dates=True,
                na_values=["NA"],
            )


def test_parse_tz_aware(all_parsers):
    # See gh-1693
    parser = all_parsers
    data = "Date,x\n2012-06-13T01:39:00Z,0.5"

    result = parser.read_csv(StringIO(data), index_col=0, parse_dates=True)
    expected = DataFrame(
        {"x": [0.5]}, index=Index([Timestamp("2012-06-13 01:39:00+00:00")], name="Date")
    )
    tm.assert_frame_equal(result, expected)
    assert result.index.tz is pytz.utc


@pytest.mark.parametrize(
    "parse_dates,index_col",
    [({"nominal": [1, 2]}, "nominal"), ({"nominal": [1, 2]}, 0), ([[1, 2]], 0)],
)
def test_multiple_date_cols_index(all_parsers, parse_dates, index_col):
    parser = all_parsers
    data = """
ID,date,NominalTime,ActualTime,TDew,TAir,Windspeed,Precip,WindDir
KORD1,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD2,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD3,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD4,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD5,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD6,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000
"""
    expected = DataFrame(
        [
            [
                datetime(1999, 1, 27, 19, 0),
                "KORD1",
                " 18:56:00",
                0.81,
                2.81,
                7.2,
                0.0,
                280.0,
            ],
            [
                datetime(1999, 1, 27, 20, 0),
                "KORD2",
                " 19:56:00",
                0.01,
                2.21,
                7.2,
                0.0,
                260.0,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                "KORD3",
                " 20:56:00",
                -0.59,
                2.21,
                5.7,
                0.0,
                280.0,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                "KORD4",
                " 21:18:00",
                -0.99,
                2.01,
                3.6,
                0.0,
                270.0,
            ],
            [
                datetime(1999, 1, 27, 22, 0),
                "KORD5",
                " 21:56:00",
                -0.59,
                1.71,
                5.1,
                0.0,
                290.0,
            ],
            [
                datetime(1999, 1, 27, 23, 0),
                "KORD6",
                " 22:56:00",
                -0.59,
                1.71,
                4.6,
                0.0,
                280.0,
            ],
        ],
        columns=[
            "nominal",
            "ID",
            "ActualTime",
            "TDew",
            "TAir",
            "Windspeed",
            "Precip",
            "WindDir",
        ],
    )
    expected = expected.set_index("nominal")

    if not isinstance(parse_dates, dict):
        expected.index.name = "date_NominalTime"

    result = parser.read_csv(
        StringIO(data), parse_dates=parse_dates, index_col=index_col
    )
    tm.assert_frame_equal(result, expected)


def test_multiple_date_cols_chunked(all_parsers):
    parser = all_parsers
    data = """\
ID,date,nominalTime,actualTime,A,B,C,D,E
KORD,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000
"""

    expected = DataFrame(
        [
            [
                datetime(1999, 1, 27, 19, 0),
                "KORD",
                " 18:56:00",
                0.81,
                2.81,
                7.2,
                0.0,
                280.0,
            ],
            [
                datetime(1999, 1, 27, 20, 0),
                "KORD",
                " 19:56:00",
                0.01,
                2.21,
                7.2,
                0.0,
                260.0,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                "KORD",
                " 20:56:00",
                -0.59,
                2.21,
                5.7,
                0.0,
                280.0,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                "KORD",
                " 21:18:00",
                -0.99,
                2.01,
                3.6,
                0.0,
                270.0,
            ],
            [
                datetime(1999, 1, 27, 22, 0),
                "KORD",
                " 21:56:00",
                -0.59,
                1.71,
                5.1,
                0.0,
                290.0,
            ],
            [
                datetime(1999, 1, 27, 23, 0),
                "KORD",
                " 22:56:00",
                -0.59,
                1.71,
                4.6,
                0.0,
                280.0,
            ],
        ],
        columns=["nominal", "ID", "actualTime", "A", "B", "C", "D", "E"],
    )
    expected = expected.set_index("nominal")

    with parser.read_csv(
        StringIO(data),
        parse_dates={"nominal": [1, 2]},
        index_col="nominal",
        chunksize=2,
    ) as reader:
        chunks = list(reader)

    tm.assert_frame_equal(chunks[0], expected[:2])
    tm.assert_frame_equal(chunks[1], expected[2:4])
    tm.assert_frame_equal(chunks[2], expected[4:])


def test_multiple_date_col_named_index_compat(all_parsers):
    parser = all_parsers
    data = """\
ID,date,nominalTime,actualTime,A,B,C,D,E
KORD,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000
"""

    with_indices = parser.read_csv(
        StringIO(data), parse_dates={"nominal": [1, 2]}, index_col="nominal"
    )
    with_names = parser.read_csv(
        StringIO(data),
        index_col="nominal",
        parse_dates={"nominal": ["date", "nominalTime"]},
    )
    tm.assert_frame_equal(with_indices, with_names)


def test_multiple_date_col_multiple_index_compat(all_parsers):
    parser = all_parsers
    data = """\
ID,date,nominalTime,actualTime,A,B,C,D,E
KORD,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000
"""
    result = parser.read_csv(
        StringIO(data), index_col=["nominal", "ID"], parse_dates={"nominal": [1, 2]}
    )
    expected = parser.read_csv(StringIO(data), parse_dates={"nominal": [1, 2]})

    expected = expected.set_index(["nominal", "ID"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("kwargs", [{}, {"index_col": "C"}])
def test_read_with_parse_dates_scalar_non_bool(all_parsers, kwargs):
    # see gh-5636
    parser = all_parsers
    msg = (
        "Only booleans, lists, and dictionaries "
        "are accepted for the 'parse_dates' parameter"
    )
    data = """A,B,C
    1,2,2003-11-1"""

    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), parse_dates="C", **kwargs)


@pytest.mark.parametrize("parse_dates", [(1,), np.array([4, 5]), {1, 3, 3}])
def test_read_with_parse_dates_invalid_type(all_parsers, parse_dates):
    parser = all_parsers
    msg = (
        "Only booleans, lists, and dictionaries "
        "are accepted for the 'parse_dates' parameter"
    )
    data = """A,B,C
    1,2,2003-11-1"""

    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), parse_dates=(1,))


@pytest.mark.parametrize("cache_dates", [True, False])
@pytest.mark.parametrize("value", ["nan", "0", ""])
def test_bad_date_parse(all_parsers, cache_dates, value):
    # if we have an invalid date make sure that we handle this with
    # and w/o the cache properly
    parser = all_parsers
    s = StringIO((f"{value},\n") * 50000)

    parser.read_csv(
        s,
        header=None,
        names=["foo", "bar"],
        parse_dates=["foo"],
        infer_datetime_format=False,
        cache_dates=cache_dates,
    )


def test_parse_dates_empty_string(all_parsers):
    # see gh-2263
    parser = all_parsers
    data = "Date,test\n2012-01-01,1\n,2"
    result = parser.read_csv(StringIO(data), parse_dates=["Date"], na_filter=False)

    expected = DataFrame(
        [[datetime(2012, 1, 1), 1], [pd.NaT, 2]], columns=["Date", "test"]
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        (
            "a\n04.15.2016",
            {"parse_dates": ["a"]},
            DataFrame([datetime(2016, 4, 15)], columns=["a"]),
        ),
        (
            "a\n04.15.2016",
            {"parse_dates": True, "index_col": 0},
            DataFrame(index=DatetimeIndex(["2016-04-15"], name="a")),
        ),
        (
            "a,b\n04.15.2016,09.16.2013",
            {"parse_dates": ["a", "b"]},
            DataFrame(
                [[datetime(2016, 4, 15), datetime(2013, 9, 16)]], columns=["a", "b"]
            ),
        ),
        (
            "a,b\n04.15.2016,09.16.2013",
            {"parse_dates": True, "index_col": [0, 1]},
            DataFrame(
                index=MultiIndex.from_tuples(
                    [(datetime(2016, 4, 15), datetime(2013, 9, 16))], names=["a", "b"]
                )
            ),
        ),
    ],
)
def test_parse_dates_no_convert_thousands(all_parsers, data, kwargs, expected):
    # see gh-14066
    parser = all_parsers

    result = parser.read_csv(StringIO(data), thousands=".", **kwargs)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "date_parser, warning",
    ([conv.parse_date_time, FutureWarning], [pd.to_datetime, None]),
)
def test_parse_date_time_multi_level_column_name(all_parsers, date_parser, warning):
    data = """\
D,T,A,B
date, time,a,b
2001-01-05, 09:00:00, 0.0, 10.
2001-01-06, 00:00:00, 1.0, 11.
"""
    parser = all_parsers
    with tm.assert_produces_warning(warning, check_stacklevel=False):
        result = parser.read_csv(
            StringIO(data),
            header=[0, 1],
            parse_dates={"date_time": [0, 1]},
            date_parser=date_parser,
        )

    expected_data = [
        [datetime(2001, 1, 5, 9, 0, 0), 0.0, 10.0],
        [datetime(2001, 1, 6, 0, 0, 0), 1.0, 11.0],
    ]
    expected = DataFrame(expected_data, columns=["date_time", ("A", "a"), ("B", "b")])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "date_parser, warning",
    ([conv.parse_date_time, FutureWarning], [pd.to_datetime, None]),
)
@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        (
            """\
date,time,a,b
2001-01-05, 10:00:00, 0.0, 10.
2001-01-05, 00:00:00, 1., 11.
""",
            {"header": 0, "parse_dates": {"date_time": [0, 1]}},
            DataFrame(
                [
                    [datetime(2001, 1, 5, 10, 0, 0), 0.0, 10],
                    [datetime(2001, 1, 5, 0, 0, 0), 1.0, 11.0],
                ],
                columns=["date_time", "a", "b"],
            ),
        ),
        (
            (
                "KORD,19990127, 19:00:00, 18:56:00, 0.8100\n"
                "KORD,19990127, 20:00:00, 19:56:00, 0.0100\n"
                "KORD,19990127, 21:00:00, 20:56:00, -0.5900\n"
                "KORD,19990127, 21:00:00, 21:18:00, -0.9900\n"
                "KORD,19990127, 22:00:00, 21:56:00, -0.5900\n"
                "KORD,19990127, 23:00:00, 22:56:00, -0.5900"
            ),
            {"header": None, "parse_dates": {"actual": [1, 2], "nominal": [1, 3]}},
            DataFrame(
                [
                    [
                        datetime(1999, 1, 27, 19, 0),
                        datetime(1999, 1, 27, 18, 56),
                        "KORD",
                        0.81,
                    ],
                    [
                        datetime(1999, 1, 27, 20, 0),
                        datetime(1999, 1, 27, 19, 56),
                        "KORD",
                        0.01,
                    ],
                    [
                        datetime(1999, 1, 27, 21, 0),
                        datetime(1999, 1, 27, 20, 56),
                        "KORD",
                        -0.59,
                    ],
                    [
                        datetime(1999, 1, 27, 21, 0),
                        datetime(1999, 1, 27, 21, 18),
                        "KORD",
                        -0.99,
                    ],
                    [
                        datetime(1999, 1, 27, 22, 0),
                        datetime(1999, 1, 27, 21, 56),
                        "KORD",
                        -0.59,
                    ],
                    [
                        datetime(1999, 1, 27, 23, 0),
                        datetime(1999, 1, 27, 22, 56),
                        "KORD",
                        -0.59,
                    ],
                ],
                columns=["actual", "nominal", 0, 4],
            ),
        ),
    ],
)
def test_parse_date_time(all_parsers, data, kwargs, expected, date_parser, warning):
    parser = all_parsers
    with tm.assert_produces_warning(warning, check_stacklevel=False):
        result = parser.read_csv(StringIO(data), date_parser=date_parser, **kwargs)

    # Python can sometimes be flaky about how
    # the aggregated columns are entered, so
    # this standardizes the order.
    result = result[expected.columns]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "date_parser, warning",
    ([conv.parse_date_fields, FutureWarning], [pd.to_datetime, None]),
)
def test_parse_date_fields(all_parsers, date_parser, warning):
    parser = all_parsers
    data = "year,month,day,a\n2001,01,10,10.\n2001,02,1,11."
    with tm.assert_produces_warning(warning, check_stacklevel=False):
        result = parser.read_csv(
            StringIO(data),
            header=0,
            parse_dates={"ymd": [0, 1, 2]},
            date_parser=date_parser,
        )

    expected = DataFrame(
        [[datetime(2001, 1, 10), 10.0], [datetime(2001, 2, 1), 11.0]],
        columns=["ymd", "a"],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "date_parser, warning",
    (
        [conv.parse_all_fields, FutureWarning],
        [lambda x: pd.to_datetime(x, format="%Y %m %d %H %M %S"), None],
    ),
)
def test_parse_date_all_fields(all_parsers, date_parser, warning):
    parser = all_parsers
    data = """\
year,month,day,hour,minute,second,a,b
2001,01,05,10,00,0,0.0,10.
2001,01,5,10,0,00,1.,11.
"""
    with tm.assert_produces_warning(warning, check_stacklevel=False):
        result = parser.read_csv(
            StringIO(data),
            header=0,
            date_parser=date_parser,
            parse_dates={"ymdHMS": [0, 1, 2, 3, 4, 5]},
        )
    expected = DataFrame(
        [
            [datetime(2001, 1, 5, 10, 0, 0), 0.0, 10.0],
            [datetime(2001, 1, 5, 10, 0, 0), 1.0, 11.0],
        ],
        columns=["ymdHMS", "a", "b"],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "date_parser, warning",
    (
        [conv.parse_all_fields, FutureWarning],
        [lambda x: pd.to_datetime(x, format="%Y %m %d %H %M %S.%f"), None],
    ),
)
def test_datetime_fractional_seconds(all_parsers, date_parser, warning):
    parser = all_parsers
    data = """\
year,month,day,hour,minute,second,a,b
2001,01,05,10,00,0.123456,0.0,10.
2001,01,5,10,0,0.500000,1.,11.
"""
    with tm.assert_produces_warning(warning, check_stacklevel=False):
        result = parser.read_csv(
            StringIO(data),
            header=0,
            date_parser=date_parser,
            parse_dates={"ymdHMS": [0, 1, 2, 3, 4, 5]},
        )
    expected = DataFrame(
        [
            [datetime(2001, 1, 5, 10, 0, 0, microsecond=123456), 0.0, 10.0],
            [datetime(2001, 1, 5, 10, 0, 0, microsecond=500000), 1.0, 11.0],
        ],
        columns=["ymdHMS", "a", "b"],
    )
    tm.assert_frame_equal(result, expected)


def test_generic(all_parsers):
    parser = all_parsers
    data = "year,month,day,a\n2001,01,10,10.\n2001,02,1,11."

    with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
        result = parser.read_csv(
            StringIO(data),
            header=0,
            parse_dates={"ym": [0, 1]},
            date_parser=lambda y, m: date(year=int(y), month=int(m), day=1),
        )
    expected = DataFrame(
        [[date(2001, 1, 1), 10, 10.0], [date(2001, 2, 1), 1, 11.0]],
        columns=["ym", "day", "a"],
    )
    tm.assert_frame_equal(result, expected)


def test_date_parser_resolution_if_not_ns(all_parsers):
    # see gh-10245
    parser = all_parsers
    data = """\
date,time,prn,rxstatus
2013-11-03,19:00:00,126,00E80000
2013-11-03,19:00:00,23,00E80000
2013-11-03,19:00:00,13,00E80000
"""

    def date_parser(dt, time):
        return np_array_datetime64_compat(dt + "T" + time + "Z", dtype="datetime64[s]")

    result = parser.read_csv(
        StringIO(data),
        date_parser=date_parser,
        parse_dates={"datetime": ["date", "time"]},
        index_col=["datetime", "prn"],
    )

    datetimes = np_array_datetime64_compat(
        ["2013-11-03T19:00:00Z"] * 3, dtype="datetime64[s]"
    )
    expected = DataFrame(
        data={"rxstatus": ["00E80000"] * 3},
        index=MultiIndex.from_tuples(
            [(datetimes[0], 126), (datetimes[1], 23), (datetimes[2], 13)],
            names=["datetime", "prn"],
        ),
    )
    tm.assert_frame_equal(result, expected)


def test_parse_date_column_with_empty_string(all_parsers):
    # see gh-6428
    parser = all_parsers
    data = "case,opdate\n7,10/18/2006\n7,10/18/2008\n621, "
    result = parser.read_csv(StringIO(data), parse_dates=["opdate"])

    expected_data = [[7, "10/18/2006"], [7, "10/18/2008"], [621, " "]]
    expected = DataFrame(expected_data, columns=["case", "opdate"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            "a\n135217135789158401\n1352171357E+5",
            DataFrame({"a": [135217135789158401, 135217135700000]}, dtype="float64"),
        ),
        (
            "a\n99999999999\n123456789012345\n1234E+0",
            DataFrame({"a": [99999999999, 123456789012345, 1234]}, dtype="float64"),
        ),
    ],
)
@pytest.mark.parametrize("parse_dates", [True, False])
def test_parse_date_float(all_parsers, data, expected, parse_dates):
    # see gh-2697
    #
    # Date parsing should fail, so we leave the data untouched
    # (i.e. float precision should remain unchanged).
    parser = all_parsers

    result = parser.read_csv(StringIO(data), parse_dates=parse_dates)
    tm.assert_frame_equal(result, expected)


def test_parse_timezone(all_parsers):
    # see gh-22256
    parser = all_parsers
    data = """dt,val
              2018-01-04 09:01:00+09:00,23350
              2018-01-04 09:02:00+09:00,23400
              2018-01-04 09:03:00+09:00,23400
              2018-01-04 09:04:00+09:00,23400
              2018-01-04 09:05:00+09:00,23400"""
    result = parser.read_csv(StringIO(data), parse_dates=["dt"])

    dti = DatetimeIndex(
        list(
            date_range(
                start="2018-01-04 09:01:00",
                end="2018-01-04 09:05:00",
                freq="1min",
                tz=pytz.FixedOffset(540),
            )
        ),
        freq=None,
    )
    expected_data = {"dt": dti, "val": [23350, 23400, 23400, 23400, 23400]}

    expected = DataFrame(expected_data)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "date_string",
    ["32/32/2019", "02/30/2019", "13/13/2019", "13/2019", "a3/11/2018", "10/11/2o17"],
)
def test_invalid_parse_delimited_date(all_parsers, date_string):
    parser = all_parsers
    expected = DataFrame({0: [date_string]}, dtype="object")
    result = parser.read_csv(StringIO(date_string), header=None, parse_dates=[0])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "date_string,dayfirst,expected",
    [
        # %d/%m/%Y; month > 12 thus replacement
        ("13/02/2019", False, datetime(2019, 2, 13)),
        ("13/02/2019", True, datetime(2019, 2, 13)),
        # %m/%d/%Y; day > 12 thus there will be no replacement
        ("02/13/2019", False, datetime(2019, 2, 13)),
        ("02/13/2019", True, datetime(2019, 2, 13)),
        # %d/%m/%Y; dayfirst==True thus replacement
        ("04/02/2019", True, datetime(2019, 2, 4)),
    ],
)
def test_parse_delimited_date_swap(all_parsers, date_string, dayfirst, expected):
    parser = all_parsers
    expected = DataFrame({0: [expected]}, dtype="datetime64[ns]")
    result = parser.read_csv(
        StringIO(date_string), header=None, dayfirst=dayfirst, parse_dates=[0]
    )
    tm.assert_frame_equal(result, expected)


def _helper_hypothesis_delimited_date(call, date_string, **kwargs):
    msg, result = None, None
    try:
        result = call(date_string, **kwargs)
    except ValueError as er:
        msg = str(er)
        pass
    return msg, result


@given(date_strategy)
@settings(deadline=None)
@pytest.mark.parametrize("delimiter", list(" -./"))
@pytest.mark.parametrize("dayfirst", [True, False])
@pytest.mark.parametrize(
    "date_format",
    ["%d %m %Y", "%m %d %Y", "%m %Y", "%Y %m %d", "%y %m %d", "%Y%m%d", "%y%m%d"],
)
def test_hypothesis_delimited_date(date_format, dayfirst, delimiter, test_datetime):
    if date_format == "%m %Y" and delimiter == ".":
        pytest.skip(
            "parse_datetime_string cannot reliably tell whether \
        e.g. %m.%Y is a float or a date, thus we skip it"
        )
    result, expected = None, None
    except_in_dateutil, except_out_dateutil = None, None
    date_string = test_datetime.strftime(date_format.replace(" ", delimiter))

    except_out_dateutil, result = _helper_hypothesis_delimited_date(
        parse_datetime_string, date_string, dayfirst=dayfirst
    )
    except_in_dateutil, expected = _helper_hypothesis_delimited_date(
        du_parse,
        date_string,
        default=_DEFAULT_DATETIME,
        dayfirst=dayfirst,
        yearfirst=False,
    )

    assert except_out_dateutil == except_in_dateutil
    assert result == expected


@pytest.mark.parametrize(
    "names, usecols, parse_dates, missing_cols",
    [
        (None, ["val"], ["date", "time"], "date, time"),
        (None, ["val"], [0, "time"], "time"),
        (None, ["val"], [["date", "time"]], "date, time"),
        (None, ["val"], [[0, "time"]], "time"),
        (None, ["val"], {"date": [0, "time"]}, "time"),
        (None, ["val"], {"date": ["date", "time"]}, "date, time"),
        (None, ["val"], [["date", "time"], "date"], "date, time"),
        (["date1", "time1", "temperature"], None, ["date", "time"], "date, time"),
        (
            ["date1", "time1", "temperature"],
            ["date1", "temperature"],
            ["date1", "time"],
            "time",
        ),
    ],
)
def test_missing_parse_dates_column_raises(
    all_parsers, names, usecols, parse_dates, missing_cols
):
    # gh-31251 column names provided in parse_dates could be missing.
    parser = all_parsers
    content = StringIO("date,time,val\n2020-01-31,04:20:32,32\n")
    msg = f"Missing column provided to 'parse_dates': '{missing_cols}'"
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(
            content, sep=",", names=names, usecols=usecols, parse_dates=parse_dates
        )


def test_date_parser_and_names(all_parsers):
    # GH#33699
    parser = all_parsers
    data = StringIO("""x,y\n1,2""")
    result = parser.read_csv(data, parse_dates=["B"], names=["B"])
    expected = DataFrame({"B": ["y", "2"]}, index=["x", "1"])
    tm.assert_frame_equal(result, expected)


def test_date_parser_usecols_thousands(all_parsers):
    # GH#39365
    data = """A,B,C
    1,3,20-09-01-01
    2,4,20-09-01-01
    """

    parser = all_parsers
    result = parser.read_csv(
        StringIO(data),
        parse_dates=[1],
        usecols=[1, 2],
        thousands="-",
    )
    expected = DataFrame({"B": [3, 4], "C": [Timestamp("20-09-2001 01:00:00")] * 2})
    tm.assert_frame_equal(result, expected)
