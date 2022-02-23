import numpy as np
import pytest

from pandas.errors import UnsupportedFunctionCall
import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Series,
    Timedelta,
    concat,
    date_range,
)
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer


@td.skip_if_no_scipy
def test_constructor(frame_or_series):
    # GH 12669
    c = frame_or_series(range(5)).rolling

    # valid
    c(win_type="boxcar", window=2, min_periods=1)
    c(win_type="boxcar", window=2, min_periods=1, center=True)
    c(win_type="boxcar", window=2, min_periods=1, center=False)


@pytest.mark.parametrize("w", [2.0, "foo", np.array([2])])
@td.skip_if_no_scipy
def test_invalid_constructor(frame_or_series, w):
    # not valid

    c = frame_or_series(range(5)).rolling
    with pytest.raises(ValueError, match="min_periods must be an integer"):
        c(win_type="boxcar", window=2, min_periods=w)
    with pytest.raises(ValueError, match="center must be a boolean"):
        c(win_type="boxcar", window=2, min_periods=1, center=w)


@pytest.mark.parametrize("wt", ["foobar", 1])
@td.skip_if_no_scipy
def test_invalid_constructor_wintype(frame_or_series, wt):
    c = frame_or_series(range(5)).rolling
    with pytest.raises(ValueError, match="Invalid win_type"):
        c(win_type=wt, window=2)


@td.skip_if_no_scipy
def test_constructor_with_win_type(frame_or_series, win_types):
    # GH 12669
    c = frame_or_series(range(5)).rolling
    c(win_type=win_types, window=2)


@pytest.mark.parametrize("method", ["sum", "mean"])
def test_numpy_compat(method):
    # see gh-12811
    w = Series([2, 4, 6]).rolling(window=2)

    msg = "numpy operations are not valid with window objects"

    with pytest.raises(UnsupportedFunctionCall, match=msg):
        getattr(w, method)(1, 2, 3)
    with pytest.raises(UnsupportedFunctionCall, match=msg):
        getattr(w, method)(dtype=np.float64)


@td.skip_if_no_scipy
@pytest.mark.parametrize("arg", ["median", "kurt", "skew"])
def test_agg_function_support(arg):
    df = DataFrame({"A": np.arange(5)})
    roll = df.rolling(2, win_type="triang")

    msg = f"'{arg}' is not a valid function for 'Window' object"
    with pytest.raises(AttributeError, match=msg):
        roll.agg(arg)

    with pytest.raises(AttributeError, match=msg):
        roll.agg([arg])

    with pytest.raises(AttributeError, match=msg):
        roll.agg({"A": arg})


@td.skip_if_no_scipy
def test_invalid_scipy_arg():
    # This error is raised by scipy
    msg = r"boxcar\(\) got an unexpected"
    with pytest.raises(TypeError, match=msg):
        Series(range(3)).rolling(1, win_type="boxcar").mean(foo="bar")


@td.skip_if_no_scipy
def test_constructor_with_win_type_invalid(frame_or_series):
    # GH 13383
    c = frame_or_series(range(5)).rolling

    msg = "window must be an integer 0 or greater"

    with pytest.raises(ValueError, match=msg):
        c(-1, win_type="boxcar")


@td.skip_if_no_scipy
@pytest.mark.filterwarnings("ignore:can't resolve:ImportWarning")
def test_window_with_args():
    # make sure that we are aggregating window functions correctly with arg
    r = Series(np.random.randn(100)).rolling(
        window=10, min_periods=1, win_type="gaussian"
    )
    expected = concat([r.mean(std=10), r.mean(std=0.01)], axis=1)
    expected.columns = ["<lambda>", "<lambda>"]
    result = r.aggregate([lambda x: x.mean(std=10), lambda x: x.mean(std=0.01)])
    tm.assert_frame_equal(result, expected)

    def a(x):
        return x.mean(std=10)

    def b(x):
        return x.mean(std=0.01)

    expected = concat([r.mean(std=10), r.mean(std=0.01)], axis=1)
    expected.columns = ["a", "b"]
    result = r.aggregate([a, b])
    tm.assert_frame_equal(result, expected)


@td.skip_if_no_scipy
def test_win_type_with_method_invalid():
    with pytest.raises(
        NotImplementedError, match="'single' is the only supported method type."
    ):
        Series(range(1)).rolling(1, win_type="triang", method="table")


@td.skip_if_no_scipy
@pytest.mark.parametrize("arg", [2000000000, "2s", Timedelta("2s")])
def test_consistent_win_type_freq(arg):
    # GH 15969
    s = Series(range(1))
    with pytest.raises(ValueError, match="Invalid win_type freq"):
        s.rolling(arg, win_type="freq")


def test_win_type_freq_return_deprecation():
    freq_roll = Series(range(2), index=date_range("2020", periods=2)).rolling("2s")
    with tm.assert_produces_warning(FutureWarning):
        assert freq_roll.win_type == "freq"


@td.skip_if_no_scipy
def test_win_type_not_implemented():
    class CustomIndexer(BaseIndexer):
        def get_window_bounds(self, num_values, min_periods, center, closed):
            return np.array([0, 1]), np.array([1, 2])

    df = DataFrame({"values": range(2)})
    indexer = CustomIndexer()
    with pytest.raises(NotImplementedError, match="BaseIndexer subclasses not"):
        df.rolling(indexer, win_type="boxcar")
