import numpy as np
import pytest

from pandas import Series
from pandas.core.construction import create_series_with_explicit_dtype


@pytest.mark.parametrize(
    "data, expected",
    [
        (np.random.randint(0, 10, size=1000), False),
        (np.arange(1000), True),
        ([], True),
        ([np.nan], True),
        (["foo", "bar", np.nan], True),
        (["foo", "foo", np.nan], False),
        (["foo", "bar", np.nan, np.nan], False),
    ],
)
def test_is_unique(data, expected):
    # GH#11946 / GH#25180
    ser = create_series_with_explicit_dtype(data, dtype_if_empty=object)
    assert ser.is_unique is expected


def test_is_unique_class_ne(capsys):
    # GH#20661
    class Foo:
        def __init__(self, val):
            self._value = val

        def __ne__(self, other):
            raise Exception("NEQ not supported")

    with capsys.disabled():
        li = [Foo(i) for i in range(5)]
        ser = Series(li, index=list(range(5)))

    ser.is_unique
    captured = capsys.readouterr()
    assert len(captured.err) == 0
