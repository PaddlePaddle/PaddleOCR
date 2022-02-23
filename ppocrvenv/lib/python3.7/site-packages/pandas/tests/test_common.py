import collections
from functools import partial
import string

import numpy as np
import pytest

from pandas.compat import np_version_under1p18

import pandas as pd
from pandas import Series
import pandas._testing as tm
from pandas.core import ops
import pandas.core.common as com
from pandas.util.version import Version


def test_get_callable_name():
    getname = com.get_callable_name

    def fn(x):
        return x

    lambda_ = lambda x: x
    part1 = partial(fn)
    part2 = partial(part1)

    class somecall:
        def __call__(self):
            return x  # noqa

    assert getname(fn) == "fn"
    assert getname(lambda_)
    assert getname(part1) == "fn"
    assert getname(part2) == "fn"
    assert getname(somecall()) == "somecall"
    assert getname(1) is None


def test_any_none():
    assert com.any_none(1, 2, 3, None)
    assert not com.any_none(1, 2, 3, 4)


def test_all_not_none():
    assert com.all_not_none(1, 2, 3, 4)
    assert not com.all_not_none(1, 2, 3, None)
    assert not com.all_not_none(None, None, None, None)


def test_random_state():
    import numpy.random as npr

    # Check with seed
    state = com.random_state(5)
    assert state.uniform() == npr.RandomState(5).uniform()

    # Check with random state object
    state2 = npr.RandomState(10)
    assert com.random_state(state2).uniform() == npr.RandomState(10).uniform()

    # check with no arg random state
    assert com.random_state() is np.random

    # check array-like
    # GH32503
    state_arr_like = npr.randint(0, 2 ** 31, size=624, dtype="uint32")
    assert (
        com.random_state(state_arr_like).uniform()
        == npr.RandomState(state_arr_like).uniform()
    )

    # Check BitGenerators
    # GH32503
    if not np_version_under1p18:
        assert (
            com.random_state(npr.MT19937(3)).uniform()
            == npr.RandomState(npr.MT19937(3)).uniform()
        )
        assert (
            com.random_state(npr.PCG64(11)).uniform()
            == npr.RandomState(npr.PCG64(11)).uniform()
        )

    # Error for floats or strings
    msg = (
        "random_state must be an integer, array-like, a BitGenerator, "
        "a numpy RandomState, or None"
    )
    with pytest.raises(ValueError, match=msg):
        com.random_state("test")

    with pytest.raises(ValueError, match=msg):
        com.random_state(5.5)


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (Series([1], name="x"), Series([2], name="x"), "x"),
        (Series([1], name="x"), Series([2], name="y"), None),
        (Series([1]), Series([2], name="x"), None),
        (Series([1], name="x"), Series([2]), None),
        (Series([1], name="x"), [2], "x"),
        ([1], Series([2], name="y"), "y"),
    ],
)
def test_maybe_match_name(left, right, expected):
    assert ops.common._maybe_match_name(left, right) == expected


def test_standardize_mapping():
    # No uninitialized defaultdicts
    msg = r"to_dict\(\) only accepts initialized defaultdicts"
    with pytest.raises(TypeError, match=msg):
        com.standardize_mapping(collections.defaultdict)

    # No non-mapping subtypes, instance
    msg = "unsupported type: <class 'list'>"
    with pytest.raises(TypeError, match=msg):
        com.standardize_mapping([])

    # No non-mapping subtypes, class
    with pytest.raises(TypeError, match=msg):
        com.standardize_mapping(list)

    fill = {"bad": "data"}
    assert com.standardize_mapping(fill) == dict

    # Convert instance to type
    assert com.standardize_mapping({}) == dict

    dd = collections.defaultdict(list)
    assert isinstance(com.standardize_mapping(dd), partial)


def test_git_version():
    # GH 21295
    git_version = pd.__git_version__
    assert len(git_version) == 40
    assert all(c in string.hexdigits for c in git_version)


def test_version_tag():
    version = Version(pd.__version__)
    try:
        version > Version("0.0.1")
    except TypeError:
        raise ValueError(
            "No git tags exist, please sync tags between upstream and your repo"
        )


@pytest.mark.parametrize(
    "obj", [(obj,) for obj in pd.__dict__.values() if callable(obj)]
)
def test_serializable(obj):
    # GH 35611
    unpickled = tm.round_trip_pickle(obj)
    assert type(obj) == type(unpickled)


class TestIsBoolIndexer:
    def test_non_bool_array_with_na(self):
        # in particular, this should not raise
        arr = np.array(["A", "B", np.nan], dtype=object)
        assert not com.is_bool_indexer(arr)

    def test_list_subclass(self):
        # GH#42433

        class MyList(list):
            pass

        val = MyList(["a"])

        assert not com.is_bool_indexer(val)

        val = MyList([True])
        assert com.is_bool_indexer(val)

    def test_frozenlist(self):
        # GH#42461
        data = {"col1": [1, 2], "col2": [3, 4]}
        df = pd.DataFrame(data=data)

        frozen = df.index.names[1:]
        assert not com.is_bool_indexer(frozen)

        result = df[frozen]
        expected = df[[]]
        tm.assert_frame_equal(result, expected)
