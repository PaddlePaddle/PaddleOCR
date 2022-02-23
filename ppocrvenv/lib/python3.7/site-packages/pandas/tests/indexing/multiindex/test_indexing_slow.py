from typing import (
    Any,
    List,
)
import warnings

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm

m = 50
n = 1000
cols = ["jim", "joe", "jolie", "joline", "jolia"]

vals: List[Any] = [
    np.random.randint(0, 10, n),
    np.random.choice(list("abcdefghij"), n),
    np.random.choice(pd.date_range("20141009", periods=10).tolist(), n),
    np.random.choice(list("ZYXWVUTSRQ"), n),
    np.random.randn(n),
]
vals = list(map(tuple, zip(*vals)))

# bunch of keys for testing
keys: List[Any] = [
    np.random.randint(0, 11, m),
    np.random.choice(list("abcdefghijk"), m),
    np.random.choice(pd.date_range("20141009", periods=11).tolist(), m),
    np.random.choice(list("ZYXWVUTSRQP"), m),
]
keys = list(map(tuple, zip(*keys)))
keys += list(map(lambda t: t[:-1], vals[:: n // m]))


# covers both unique index and non-unique index
df = DataFrame(vals, columns=cols)
a = pd.concat([df, df])
b = df.drop_duplicates(subset=cols[:-1])


def validate(mi, df, key):
    # check indexing into a multi-index before & past the lexsort depth

    mask = np.ones(len(df)).astype("bool")

    # test for all partials of this key
    for i, k in enumerate(key):
        mask &= df.iloc[:, i] == k

        if not mask.any():
            assert key[: i + 1] not in mi.index
            continue

        assert key[: i + 1] in mi.index
        right = df[mask].copy()

        if i + 1 != len(key):  # partial key
            return_value = right.drop(cols[: i + 1], axis=1, inplace=True)
            assert return_value is None
            return_value = right.set_index(cols[i + 1 : -1], inplace=True)
            assert return_value is None
            tm.assert_frame_equal(mi.loc[key[: i + 1]], right)

        else:  # full key
            return_value = right.set_index(cols[:-1], inplace=True)
            assert return_value is None
            if len(right) == 1:  # single hit
                right = Series(
                    right["jolia"].values, name=right.index[0], index=["jolia"]
                )
                tm.assert_series_equal(mi.loc[key[: i + 1]], right)
            else:  # multi hit
                tm.assert_frame_equal(mi.loc[key[: i + 1]], right)


@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
@pytest.mark.parametrize("lexsort_depth", list(range(5)))
@pytest.mark.parametrize("key", keys)
@pytest.mark.parametrize("frame", [a, b])
def test_multiindex_get_loc(lexsort_depth, key, frame):
    # GH7724, GH2646

    with warnings.catch_warnings(record=True):
        if lexsort_depth == 0:
            df = frame.copy()
        else:
            df = frame.sort_values(by=cols[:lexsort_depth])

        mi = df.set_index(cols[:-1])
        assert not mi.index._lexsort_depth < lexsort_depth
        validate(mi, df, key)
