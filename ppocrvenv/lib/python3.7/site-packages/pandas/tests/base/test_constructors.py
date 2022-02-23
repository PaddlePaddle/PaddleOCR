from datetime import datetime
import sys

import numpy as np
import pytest

from pandas.compat import PYPY

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm
from pandas.core.accessor import PandasDelegate
from pandas.core.base import (
    NoNewAttributesMixin,
    PandasObject,
)


@pytest.fixture(
    params=[
        Series,
        lambda x, **kwargs: DataFrame({"a": x}, **kwargs)["a"],
        lambda x, **kwargs: DataFrame(x, **kwargs)[0],
        Index,
    ],
    ids=["Series", "DataFrame-dict", "DataFrame-array", "Index"],
)
def constructor(request):
    return request.param


class TestPandasDelegate:
    class Delegator:
        _properties = ["foo"]
        _methods = ["bar"]

        def _set_foo(self, value):
            self.foo = value

        def _get_foo(self):
            return self.foo

        foo = property(_get_foo, _set_foo, doc="foo property")

        def bar(self, *args, **kwargs):
            """a test bar method"""
            pass

    class Delegate(PandasDelegate, PandasObject):
        def __init__(self, obj):
            self.obj = obj

    def setup_method(self, method):
        pass

    def test_invalid_delegation(self):
        # these show that in order for the delegation to work
        # the _delegate_* methods need to be overridden to not raise
        # a TypeError

        self.Delegate._add_delegate_accessors(
            delegate=self.Delegator,
            accessors=self.Delegator._properties,
            typ="property",
        )
        self.Delegate._add_delegate_accessors(
            delegate=self.Delegator, accessors=self.Delegator._methods, typ="method"
        )

        delegate = self.Delegate(self.Delegator())

        msg = "You cannot access the property foo"
        with pytest.raises(TypeError, match=msg):
            delegate.foo

        msg = "The property foo cannot be set"
        with pytest.raises(TypeError, match=msg):
            delegate.foo = 5

        msg = "You cannot access the property foo"
        with pytest.raises(TypeError, match=msg):
            delegate.foo()

    @pytest.mark.skipif(PYPY, reason="not relevant for PyPy")
    def test_memory_usage(self):
        # Delegate does not implement memory_usage.
        # Check that we fall back to in-built `__sizeof__`
        # GH 12924
        delegate = self.Delegate(self.Delegator())
        sys.getsizeof(delegate)


class TestNoNewAttributesMixin:
    def test_mixin(self):
        class T(NoNewAttributesMixin):
            pass

        t = T()
        assert not hasattr(t, "__frozen")

        t.a = "test"
        assert t.a == "test"

        t._freeze()
        assert "__frozen" in dir(t)
        assert getattr(t, "__frozen")
        msg = "You cannot add any new attribute"
        with pytest.raises(AttributeError, match=msg):
            t.b = "test"

        assert not hasattr(t, "b")


class TestConstruction:
    # test certain constructor behaviours on dtype inference across Series,
    # Index and DataFrame

    @pytest.mark.parametrize(
        "klass",
        [
            Series,
            lambda x, **kwargs: DataFrame({"a": x}, **kwargs)["a"],
            lambda x, **kwargs: DataFrame(x, **kwargs)[0],
            Index,
        ],
    )
    @pytest.mark.parametrize(
        "a",
        [
            np.array(["2263-01-01"], dtype="datetime64[D]"),
            np.array([datetime(2263, 1, 1)], dtype=object),
            np.array([np.datetime64("2263-01-01", "D")], dtype=object),
            np.array(["2263-01-01"], dtype=object),
        ],
        ids=[
            "datetime64[D]",
            "object-datetime.datetime",
            "object-numpy-scalar",
            "object-string",
        ],
    )
    def test_constructor_datetime_outofbound(self, a, klass):
        # GH-26853 (+ bug GH-26206 out of bound non-ns unit)

        # No dtype specified (dtype inference)
        # datetime64[non-ns] raise error, other cases result in object dtype
        # and preserve original data
        if a.dtype.kind == "M":
            msg = "Out of bounds"
            with pytest.raises(pd.errors.OutOfBoundsDatetime, match=msg):
                klass(a)
        else:
            result = klass(a)
            assert result.dtype == "object"
            tm.assert_numpy_array_equal(result.to_numpy(), a)

        # Explicit dtype specified
        # Forced conversion fails for all -> all cases raise error
        msg = "Out of bounds"
        with pytest.raises(pd.errors.OutOfBoundsDatetime, match=msg):
            klass(a, dtype="datetime64[ns]")

    def test_constructor_datetime_nonns(self, constructor):
        arr = np.array(["2020-01-01T00:00:00.000000"], dtype="datetime64[us]")
        expected = constructor(pd.to_datetime(["2020-01-01"]))
        result = constructor(arr)
        tm.assert_equal(result, expected)

        # https://github.com/pandas-dev/pandas/issues/34843
        arr.flags.writeable = False
        result = constructor(arr)
        tm.assert_equal(result, expected)
