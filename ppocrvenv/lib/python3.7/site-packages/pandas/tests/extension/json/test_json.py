import collections
import operator

import pytest

import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
    JSONArray,
    JSONDtype,
    make_data,
)


@pytest.fixture
def dtype():
    return JSONDtype()


@pytest.fixture
def data():
    """Length-100 PeriodArray for semantics test."""
    data = make_data()

    # Why the while loop? NumPy is unable to construct an ndarray from
    # equal-length ndarrays. Many of our operations involve coercing the
    # EA to an ndarray of objects. To avoid random test failures, we ensure
    # that our data is coercible to an ndarray. Several tests deal with only
    # the first two elements, so that's what we'll check.

    while len(data[0]) == len(data[1]):
        data = make_data()

    return JSONArray(data)


@pytest.fixture
def data_missing():
    """Length 2 array with [NA, Valid]"""
    return JSONArray([{}, {"a": 10}])


@pytest.fixture
def data_for_sorting():
    return JSONArray([{"b": 1}, {"c": 4}, {"a": 2, "c": 3}])


@pytest.fixture
def data_missing_for_sorting():
    return JSONArray([{"b": 1}, {}, {"a": 4}])


@pytest.fixture
def na_value(dtype):
    return dtype.na_value


@pytest.fixture
def na_cmp():
    return operator.eq


@pytest.fixture
def data_for_grouping():
    return JSONArray(
        [
            {"b": 1},
            {"b": 1},
            {},
            {},
            {"a": 0, "c": 2},
            {"a": 0, "c": 2},
            {"b": 1},
            {"c": 2},
        ]
    )


class BaseJSON:
    # NumPy doesn't handle an array of equal-length UserDicts.
    # The default assert_series_equal eventually does a
    # Series.values, which raises. We work around it by
    # converting the UserDicts to dicts.
    @classmethod
    def assert_series_equal(cls, left, right, *args, **kwargs):
        if left.dtype.name == "json":
            assert left.dtype == right.dtype
            left = pd.Series(
                JSONArray(left.values.astype(object)), index=left.index, name=left.name
            )
            right = pd.Series(
                JSONArray(right.values.astype(object)),
                index=right.index,
                name=right.name,
            )
        tm.assert_series_equal(left, right, *args, **kwargs)

    @classmethod
    def assert_frame_equal(cls, left, right, *args, **kwargs):
        obj_type = kwargs.get("obj", "DataFrame")
        tm.assert_index_equal(
            left.columns,
            right.columns,
            exact=kwargs.get("check_column_type", "equiv"),
            check_names=kwargs.get("check_names", True),
            check_exact=kwargs.get("check_exact", False),
            check_categorical=kwargs.get("check_categorical", True),
            obj=f"{obj_type}.columns",
        )

        jsons = (left.dtypes == "json").index

        for col in jsons:
            cls.assert_series_equal(left[col], right[col], *args, **kwargs)

        left = left.drop(columns=jsons)
        right = right.drop(columns=jsons)
        tm.assert_frame_equal(left, right, *args, **kwargs)


class TestDtype(BaseJSON, base.BaseDtypeTests):
    pass


class TestInterface(BaseJSON, base.BaseInterfaceTests):
    def test_custom_asserts(self):
        # This would always trigger the KeyError from trying to put
        # an array of equal-length UserDicts inside an ndarray.
        data = JSONArray(
            [
                collections.UserDict({"a": 1}),
                collections.UserDict({"b": 2}),
                collections.UserDict({"c": 3}),
            ]
        )
        a = pd.Series(data)
        self.assert_series_equal(a, a)
        self.assert_frame_equal(a.to_frame(), a.to_frame())

        b = pd.Series(data.take([0, 0, 1]))
        msg = r"ExtensionArray are different"
        with pytest.raises(AssertionError, match=msg):
            self.assert_series_equal(a, b)

        with pytest.raises(AssertionError, match=msg):
            self.assert_frame_equal(a.to_frame(), b.to_frame())

    @pytest.mark.xfail(
        reason="comparison method not implemented for JSONArray (GH-37867)"
    )
    def test_contains(self, data):
        # GH-37867
        super().test_contains(data)


class TestConstructors(BaseJSON, base.BaseConstructorsTests):
    @pytest.mark.skip(reason="not implemented constructor from dtype")
    def test_from_dtype(self, data):
        # construct from our dtype & string dtype
        pass

    @pytest.mark.xfail(reason="RecursionError, GH-33900")
    def test_series_constructor_no_data_with_index(self, dtype, na_value):
        # RecursionError: maximum recursion depth exceeded in comparison
        super().test_series_constructor_no_data_with_index(dtype, na_value)

    @pytest.mark.xfail(reason="RecursionError, GH-33900")
    def test_series_constructor_scalar_na_with_index(self, dtype, na_value):
        # RecursionError: maximum recursion depth exceeded in comparison
        super().test_series_constructor_scalar_na_with_index(dtype, na_value)

    @pytest.mark.xfail(reason="collection as scalar, GH-33901")
    def test_series_constructor_scalar_with_index(self, data, dtype):
        # TypeError: All values must be of type <class 'collections.abc.Mapping'>
        super().test_series_constructor_scalar_with_index(data, dtype)


class TestReshaping(BaseJSON, base.BaseReshapingTests):
    @pytest.mark.skip(reason="Different definitions of NA")
    def test_stack(self):
        """
        The test does .astype(object).stack(). If we happen to have
        any missing values in `data`, then we'll end up with different
        rows since we consider `{}` NA, but `.astype(object)` doesn't.
        """

    @pytest.mark.xfail(reason="dict for NA")
    def test_unstack(self, data, index):
        # The base test has NaN for the expected NA value.
        # this matches otherwise
        return super().test_unstack(data, index)


class TestGetitem(BaseJSON, base.BaseGetitemTests):
    pass


class TestMissing(BaseJSON, base.BaseMissingTests):
    @pytest.mark.skip(reason="Setting a dict as a scalar")
    def test_fillna_series(self):
        """We treat dictionaries as a mapping in fillna, not a scalar."""

    @pytest.mark.skip(reason="Setting a dict as a scalar")
    def test_fillna_frame(self):
        """We treat dictionaries as a mapping in fillna, not a scalar."""


unhashable = pytest.mark.skip(reason="Unhashable")


class TestReduce(base.BaseNoReduceTests):
    pass


class TestMethods(BaseJSON, base.BaseMethodsTests):
    @unhashable
    def test_value_counts(self, all_data, dropna):
        pass

    @unhashable
    def test_value_counts_with_normalize(self, data):
        pass

    @unhashable
    def test_sort_values_frame(self):
        # TODO (EA.factorize): see if _values_for_factorize allows this.
        pass

    def test_argsort(self, data_for_sorting):
        super().test_argsort(data_for_sorting)

    def test_argsort_missing(self, data_missing_for_sorting):
        super().test_argsort_missing(data_missing_for_sorting)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values(self, data_for_sorting, ascending, sort_by_key):
        super().test_sort_values(data_for_sorting, ascending, sort_by_key)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values_missing(
        self, data_missing_for_sorting, ascending, sort_by_key
    ):
        super().test_sort_values_missing(
            data_missing_for_sorting, ascending, sort_by_key
        )

    @pytest.mark.skip(reason="combine for JSONArray not supported")
    def test_combine_le(self, data_repeated):
        pass

    @pytest.mark.skip(reason="combine for JSONArray not supported")
    def test_combine_add(self, data_repeated):
        pass

    @pytest.mark.skip(reason="combine for JSONArray not supported")
    def test_combine_first(self, data):
        pass

    @unhashable
    def test_hash_pandas_object_works(self, data, kind):
        super().test_hash_pandas_object_works(data, kind)

    @pytest.mark.skip(reason="broadcasting error")
    def test_where_series(self, data, na_value):
        # Fails with
        # *** ValueError: operands could not be broadcast together
        # with shapes (4,) (4,) (0,)
        super().test_where_series(data, na_value)

    @pytest.mark.skip(reason="Can't compare dicts.")
    def test_searchsorted(self, data_for_sorting):
        super().test_searchsorted(data_for_sorting)

    @pytest.mark.skip(reason="Can't compare dicts.")
    def test_equals(self, data, na_value, as_series):
        pass


class TestCasting(BaseJSON, base.BaseCastingTests):
    @pytest.mark.skip(reason="failing on np.array(self, dtype=str)")
    def test_astype_str(self):
        """This currently fails in NumPy on np.array(self, dtype=str) with

        *** ValueError: setting an array element with a sequence
        """


# We intentionally don't run base.BaseSetitemTests because pandas'
# internals has trouble setting sequences of values into scalar positions.


class TestGroupby(BaseJSON, base.BaseGroupbyTests):
    @unhashable
    def test_groupby_extension_transform(self):
        """
        This currently fails in Series.name.setter, since the
        name must be hashable, but the value is a dictionary.
        I think this is what we want, i.e. `.name` should be the original
        values, and not the values for factorization.
        """

    @unhashable
    def test_groupby_extension_apply(self):
        """
        This fails in Index._do_unique_check with

        >   hash(val)
        E   TypeError: unhashable type: 'UserDict' with

        I suspect that once we support Index[ExtensionArray],
        we'll be able to dispatch unique.
        """

    @pytest.mark.parametrize("as_index", [True, False])
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        super().test_groupby_extension_agg(as_index, data_for_grouping)

    @pytest.mark.xfail(reason="GH#39098: Converts agg result to object")
    def test_groupby_agg_extension(self, data_for_grouping):
        super().test_groupby_agg_extension(data_for_grouping)


class TestArithmeticOps(BaseJSON, base.BaseArithmeticOpsTests):
    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators, request):
        if len(data[0]) != 1:
            mark = pytest.mark.xfail(reason="raises in coercing to Series")
            request.node.add_marker(mark)
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    def test_add_series_with_extension_array(self, data):
        ser = pd.Series(data)
        with pytest.raises(TypeError, match="unsupported"):
            ser + data

    def test_divmod_series_array(self):
        # GH 23287
        # skipping because it is not implemented
        pass

    def _check_divmod_op(self, s, op, other, exc=NotImplementedError):
        return super()._check_divmod_op(s, op, other, exc=TypeError)


class TestComparisonOps(BaseJSON, base.BaseComparisonOpsTests):
    pass


class TestPrinting(BaseJSON, base.BasePrintingTests):
    pass
