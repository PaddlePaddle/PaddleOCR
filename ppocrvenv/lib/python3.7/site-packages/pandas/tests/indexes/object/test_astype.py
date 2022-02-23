from pandas import Index
import pandas._testing as tm


def test_astype_str_from_bytes():
    # https://github.com/pandas-dev/pandas/issues/38607
    idx = Index(["あ", b"a"], dtype="object")
    result = idx.astype(str)
    expected = Index(["あ", "a"], dtype="object")
    tm.assert_index_equal(result, expected)
