from itertools import product

import numpy as np
import pytest

from pandas import (
    DataFrame,
    NaT,
    date_range,
)
import pandas._testing as tm


@pytest.fixture(params=product([True, False], [True, False]))
def close_open_fixture(request):
    return request.param


@pytest.fixture
def float_frame_with_na():
    """
    Fixture for DataFrame of floats with index of unique strings

    Columns are ['A', 'B', 'C', 'D']; some entries are missing

                       A         B         C         D
    ABwBzA0ljw -1.128865 -0.897161  0.046603  0.274997
    DJiRzmbyQF  0.728869  0.233502  0.722431 -0.890872
    neMgPD5UBF  0.486072 -1.027393 -0.031553  1.449522
    0yWA4n8VeX -1.937191 -1.142531  0.805215 -0.462018
    3slYUbbqU1  0.153260  1.164691  1.489795 -0.545826
    soujjZ0A08       NaN       NaN       NaN       NaN
    7W6NLGsjB9       NaN       NaN       NaN       NaN
    ...              ...       ...       ...       ...
    uhfeaNkCR1 -0.231210 -0.340472  0.244717 -0.901590
    n6p7GYuBIV -0.419052  1.922721 -0.125361 -0.727717
    ZhzAeY6p1y  1.234374 -1.425359 -0.827038 -0.633189
    uWdPsORyUh  0.046738 -0.980445 -1.102965  0.605503
    3DJA6aN590 -0.091018 -1.684734 -1.100900  0.215947
    2GBPAzdbMk -2.883405 -1.021071  1.209877  1.633083
    sHadBoyVHw -2.223032 -0.326384  0.258931  0.245517

    [30 rows x 4 columns]
    """
    df = DataFrame(tm.getSeriesData())
    # set some NAs
    df.iloc[5:10] = np.nan
    df.iloc[15:20, -2:] = np.nan
    return df


@pytest.fixture
def bool_frame_with_na():
    """
    Fixture for DataFrame of booleans with index of unique strings

    Columns are ['A', 'B', 'C', 'D']; some entries are missing

                    A      B      C      D
    zBZxY2IDGd  False  False  False  False
    IhBWBMWllt  False   True   True   True
    ctjdvZSR6R   True  False   True   True
    AVTujptmxb  False   True  False   True
    G9lrImrSWq  False  False  False   True
    sFFwdIUfz2    NaN    NaN    NaN    NaN
    s15ptEJnRb    NaN    NaN    NaN    NaN
    ...           ...    ...    ...    ...
    UW41KkDyZ4   True   True  False  False
    l9l6XkOdqV   True  False  False  False
    X2MeZfzDYA  False   True  False  False
    xWkIKU7vfX  False   True  False   True
    QOhL6VmpGU  False  False  False   True
    22PwkRJdat  False   True  False  False
    kfboQ3VeIK   True  False   True  False

    [30 rows x 4 columns]
    """
    df = DataFrame(tm.getSeriesData()) > 0
    df = df.astype(object)
    # set some NAs
    df.iloc[5:10] = np.nan
    df.iloc[15:20, -2:] = np.nan

    # For `any` tests we need to have at least one True before the first NaN
    #  in each column
    for i in range(4):
        df.iloc[i, i] = True
    return df


@pytest.fixture
def float_string_frame():
    """
    Fixture for DataFrame of floats and strings with index of unique strings

    Columns are ['A', 'B', 'C', 'D', 'foo'].

                       A         B         C         D  foo
    w3orJvq07g -1.594062 -1.084273 -1.252457  0.356460  bar
    PeukuVdmz2  0.109855 -0.955086 -0.809485  0.409747  bar
    ahp2KvwiM8 -1.533729 -0.142519 -0.154666  1.302623  bar
    3WSJ7BUCGd  2.484964  0.213829  0.034778 -2.327831  bar
    khdAmufk0U -0.193480 -0.743518 -0.077987  0.153646  bar
    LE2DZiFlrE -0.193566 -1.343194 -0.107321  0.959978  bar
    HJXSJhVn7b  0.142590  1.257603 -0.659409 -0.223844  bar
    ...              ...       ...       ...       ...  ...
    9a1Vypttgw -1.316394  1.601354  0.173596  1.213196  bar
    h5d1gVFbEy  0.609475  1.106738 -0.155271  0.294630  bar
    mK9LsTQG92  1.303613  0.857040 -1.019153  0.369468  bar
    oOLksd9gKH  0.558219 -0.134491 -0.289869 -0.951033  bar
    9jgoOjKyHg  0.058270 -0.496110 -0.413212 -0.852659  bar
    jZLDHclHAO  0.096298  1.267510  0.549206 -0.005235  bar
    lR0nxDp1C2 -2.119350 -0.794384  0.544118  0.145849  bar

    [30 rows x 5 columns]
    """
    df = DataFrame(tm.getSeriesData())
    df["foo"] = "bar"
    return df


@pytest.fixture
def mixed_float_frame():
    """
    Fixture for DataFrame of different float types with index of unique strings

    Columns are ['A', 'B', 'C', 'D'].

                       A         B         C         D
    GI7bbDaEZe -0.237908 -0.246225 -0.468506  0.752993
    KGp9mFepzA -1.140809 -0.644046 -1.225586  0.801588
    VeVYLAb1l2 -1.154013 -1.677615  0.690430 -0.003731
    kmPME4WKhO  0.979578  0.998274 -0.776367  0.897607
    CPyopdXTiz  0.048119 -0.257174  0.836426  0.111266
    0kJZQndAj0  0.274357 -0.281135 -0.344238  0.834541
    tqdwQsaHG8 -0.979716 -0.519897  0.582031  0.144710
    ...              ...       ...       ...       ...
    7FhZTWILQj -2.906357  1.261039 -0.780273 -0.537237
    4pUDPM4eGq -2.042512 -0.464382 -0.382080  1.132612
    B8dUgUzwTi -1.506637 -0.364435  1.087891  0.297653
    hErlVYjVv9  1.477453 -0.495515 -0.713867  1.438427
    1BKN3o7YLs  0.127535 -0.349812 -0.881836  0.489827
    9S4Ekn7zga  1.445518 -2.095149  0.031982  0.373204
    xN1dNn6OV6  1.425017 -0.983995 -0.363281 -0.224502

    [30 rows x 4 columns]
    """
    df = DataFrame(tm.getSeriesData())
    df.A = df.A.astype("float32")
    df.B = df.B.astype("float32")
    df.C = df.C.astype("float16")
    df.D = df.D.astype("float64")
    return df


@pytest.fixture
def mixed_int_frame():
    """
    Fixture for DataFrame of different int types with index of unique strings

    Columns are ['A', 'B', 'C', 'D'].

                A  B    C    D
    mUrCZ67juP  0  1    2    2
    rw99ACYaKS  0  1    0    0
    7QsEcpaaVU  0  1    1    1
    xkrimI2pcE  0  1    0    0
    dz01SuzoS8  0  1  255  255
    ccQkqOHX75 -1  1    0    0
    DN0iXaoDLd  0  1    0    0
    ...        .. ..  ...  ...
    Dfb141wAaQ  1  1  254  254
    IPD8eQOVu5  0  1    0    0
    CcaKulsCmv  0  1    0    0
    rIBa8gu7E5  0  1    0    0
    RP6peZmh5o  0  1    1    1
    NMb9pipQWQ  0  1    0    0
    PqgbJEzjib  0  1    3    3

    [30 rows x 4 columns]
    """
    df = DataFrame({k: v.astype(int) for k, v in tm.getSeriesData().items()})
    df.A = df.A.astype("int32")
    df.B = np.ones(len(df.B), dtype="uint64")
    df.C = df.C.astype("uint8")
    df.D = df.C.astype("int64")
    return df


@pytest.fixture
def timezone_frame():
    """
    Fixture for DataFrame of date_range Series with different time zones

    Columns are ['A', 'B', 'C']; some entries are missing

               A                         B                         C
    0 2013-01-01 2013-01-01 00:00:00-05:00 2013-01-01 00:00:00+01:00
    1 2013-01-02                       NaT                       NaT
    2 2013-01-03 2013-01-03 00:00:00-05:00 2013-01-03 00:00:00+01:00
    """
    df = DataFrame(
        {
            "A": date_range("20130101", periods=3),
            "B": date_range("20130101", periods=3, tz="US/Eastern"),
            "C": date_range("20130101", periods=3, tz="CET"),
        }
    )
    df.iloc[1, 1] = NaT
    df.iloc[1, 2] = NaT
    return df


@pytest.fixture
def uint64_frame():
    """
    Fixture for DataFrame with uint64 values

    Columns are ['A', 'B']
    """
    return DataFrame(
        {"A": np.arange(3), "B": [2 ** 63, 2 ** 63 + 5, 2 ** 63 + 10]}, dtype=np.uint64
    )


@pytest.fixture
def simple_frame():
    """
    Fixture for simple 3x3 DataFrame

    Columns are ['one', 'two', 'three'], index is ['a', 'b', 'c'].

       one  two  three
    a  1.0  2.0    3.0
    b  4.0  5.0    6.0
    c  7.0  8.0    9.0
    """
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    return DataFrame(arr, columns=["one", "two", "three"], index=["a", "b", "c"])


@pytest.fixture
def frame_of_index_cols():
    """
    Fixture for DataFrame of columns that can be used for indexing

    Columns are ['A', 'B', 'C', 'D', 'E', ('tuple', 'as', 'label')];
    'A' & 'B' contain duplicates (but are jointly unique), the rest are unique.

         A      B  C         D         E  (tuple, as, label)
    0  foo    one  a  0.608477 -0.012500           -1.664297
    1  foo    two  b -0.633460  0.249614           -0.364411
    2  foo  three  c  0.615256  2.154968           -0.834666
    3  bar    one  d  0.234246  1.085675            0.718445
    4  bar    two  e  0.533841 -0.005702           -3.533912
    """
    df = DataFrame(
        {
            "A": ["foo", "foo", "foo", "bar", "bar"],
            "B": ["one", "two", "three", "one", "two"],
            "C": ["a", "b", "c", "d", "e"],
            "D": np.random.randn(5),
            "E": np.random.randn(5),
            ("tuple", "as", "label"): np.random.randn(5),
        }
    )
    return df
