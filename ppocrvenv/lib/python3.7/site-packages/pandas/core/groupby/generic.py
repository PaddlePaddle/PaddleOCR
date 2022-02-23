"""
Define the SeriesGroupBy and DataFrameGroupBy
classes that hold the groupby interfaces (and some implementations).

These are user facing as the result of the ``df.groupby(...)`` operations,
which here returns a DataFrameGroupBy object.
"""
from __future__ import annotations

from collections import (
    abc,
    namedtuple,
)
from functools import partial
from textwrap import dedent
from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    Mapping,
    TypeVar,
    Union,
)
import warnings

import numpy as np

from pandas._libs import (
    lib,
    reduction as libreduction,
)
from pandas._typing import (
    ArrayLike,
    FrameOrSeries,
    FrameOrSeriesUnion,
    Manager2D,
)
from pandas.util._decorators import (
    Appender,
    Substitution,
    doc,
)

from pandas.core.dtypes.common import (
    ensure_int64,
    is_bool,
    is_categorical_dtype,
    is_dict_like,
    is_integer_dtype,
    is_interval_dtype,
    is_numeric_dtype,
    is_scalar,
)
from pandas.core.dtypes.missing import (
    isna,
    notna,
)

from pandas.core import (
    algorithms,
    nanops,
)
from pandas.core.aggregation import (
    maybe_mangle_lambdas,
    reconstruct_func,
    validate_func_kwargs,
)
from pandas.core.apply import GroupByApply
from pandas.core.base import SpecificationError
import pandas.core.common as com
from pandas.core.construction import create_series_with_explicit_dtype
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby import base
from pandas.core.groupby.groupby import (
    GroupBy,
    _agg_template,
    _apply_docs,
    _transform_template,
    group_selection_context,
)
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    all_indexes_same,
)
from pandas.core.series import Series
from pandas.core.util.numba_ import maybe_use_numba

from pandas.plotting import boxplot_frame_groupby

NamedAgg = namedtuple("NamedAgg", ["column", "aggfunc"])
# TODO(typing) the return value on this callable should be any *scalar*.
AggScalar = Union[str, Callable[..., Any]]
# TODO: validate types on ScalarResult and move to _typing
# Blocked from using by https://github.com/python/mypy/issues/1484
# See note at _mangle_lambda_list
ScalarResult = TypeVar("ScalarResult")


def generate_property(name: str, klass: type[FrameOrSeries]):
    """
    Create a property for a GroupBy subclass to dispatch to DataFrame/Series.

    Parameters
    ----------
    name : str
    klass : {DataFrame, Series}

    Returns
    -------
    property
    """

    def prop(self):
        return self._make_wrapper(name)

    parent_method = getattr(klass, name)
    prop.__doc__ = parent_method.__doc__ or ""
    prop.__name__ = name
    return property(prop)


def pin_allowlisted_properties(klass: type[FrameOrSeries], allowlist: frozenset[str]):
    """
    Create GroupBy member defs for DataFrame/Series names in a allowlist.

    Parameters
    ----------
    klass : DataFrame or Series class
        class where members are defined.
    allowlist : frozenset[str]
        Set of names of klass methods to be constructed

    Returns
    -------
    class decorator

    Notes
    -----
    Since we don't want to override methods explicitly defined in the
    base class, any such name is skipped.
    """

    def pinner(cls):
        for name in allowlist:
            if hasattr(cls, name):
                # don't override anything that was explicitly defined
                #  in the base class
                continue

            prop = generate_property(name, klass)
            setattr(cls, name, prop)

        return cls

    return pinner


@pin_allowlisted_properties(Series, base.series_apply_allowlist)
class SeriesGroupBy(GroupBy[Series]):
    _apply_allowlist = base.series_apply_allowlist

    def _iterate_slices(self) -> Iterable[Series]:
        yield self._selected_obj

    _agg_examples_doc = dedent(
        """
    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4])

    >>> s
    0    1
    1    2
    2    3
    3    4
    dtype: int64

    >>> s.groupby([1, 1, 2, 2]).min()
    1    1
    2    3
    dtype: int64

    >>> s.groupby([1, 1, 2, 2]).agg('min')
    1    1
    2    3
    dtype: int64

    >>> s.groupby([1, 1, 2, 2]).agg(['min', 'max'])
       min  max
    1    1    2
    2    3    4

    The output column names can be controlled by passing
    the desired column names and aggregations as keyword arguments.

    >>> s.groupby([1, 1, 2, 2]).agg(
    ...     minimum='min',
    ...     maximum='max',
    ... )
       minimum  maximum
    1        1        2
    2        3        4

    .. versionchanged:: 1.3.0

        The resulting dtype will reflect the return value of the aggregating function.

    >>> s.groupby([1, 1, 2, 2]).agg(lambda x: x.astype(float).min())
    1    1.0
    2    3.0
    dtype: float64"""
    )

    @Appender(
        _apply_docs["template"].format(
            input="series", examples=_apply_docs["series_examples"]
        )
    )
    def apply(self, func, *args, **kwargs):
        return super().apply(func, *args, **kwargs)

    @doc(_agg_template, examples=_agg_examples_doc, klass="Series")
    def aggregate(self, func=None, *args, engine=None, engine_kwargs=None, **kwargs):

        if maybe_use_numba(engine):
            with group_selection_context(self):
                data = self._selected_obj
            result, index = self._aggregate_with_numba(
                data.to_frame(), func, *args, engine_kwargs=engine_kwargs, **kwargs
            )
            return self.obj._constructor(result.ravel(), index=index, name=data.name)

        relabeling = func is None
        columns = None
        if relabeling:
            columns, func = validate_func_kwargs(kwargs)
            kwargs = {}

        if isinstance(func, str):
            return getattr(self, func)(*args, **kwargs)

        elif isinstance(func, abc.Iterable):
            # Catch instances of lists / tuples
            # but not the class list / tuple itself.
            func = maybe_mangle_lambdas(func)
            ret = self._aggregate_multiple_funcs(func)
            if relabeling:
                # error: Incompatible types in assignment (expression has type
                # "Optional[List[str]]", variable has type "Index")
                ret.columns = columns  # type: ignore[assignment]
            return ret

        else:
            cyfunc = com.get_cython_func(func)
            if cyfunc and not args and not kwargs:
                return getattr(self, cyfunc)()

            if self.grouper.nkeys > 1:
                return self._python_agg_general(func, *args, **kwargs)

            try:
                return self._python_agg_general(func, *args, **kwargs)
            except KeyError:
                # TODO: KeyError is raised in _python_agg_general,
                #  see test_groupby.test_basic
                result = self._aggregate_named(func, *args, **kwargs)

                index = Index(sorted(result), name=self.grouper.names[0])
                return create_series_with_explicit_dtype(
                    result, index=index, dtype_if_empty=object
                )

    agg = aggregate

    def _aggregate_multiple_funcs(self, arg) -> DataFrame:
        if isinstance(arg, dict):

            # show the deprecation, but only if we
            # have not shown a higher level one
            # GH 15931
            raise SpecificationError("nested renamer is not supported")

        elif any(isinstance(x, (tuple, list)) for x in arg):
            arg = [(x, x) if not isinstance(x, (tuple, list)) else x for x in arg]

            # indicated column order
            columns = next(zip(*arg))
        else:
            # list of functions / function names
            columns = []
            for f in arg:
                columns.append(com.get_callable_name(f) or f)

            arg = zip(columns, arg)

        results: dict[base.OutputKey, FrameOrSeriesUnion] = {}
        for idx, (name, func) in enumerate(arg):

            key = base.OutputKey(label=name, position=idx)
            results[key] = self.aggregate(func)

        if any(isinstance(x, DataFrame) for x in results.values()):
            from pandas import concat

            res_df = concat(
                results.values(), axis=1, keys=[key.label for key in results.keys()]
            )
            # error: Incompatible return value type (got "Union[DataFrame, Series]",
            # expected "DataFrame")
            return res_df  # type: ignore[return-value]

        indexed_output = {key.position: val for key, val in results.items()}
        output = self.obj._constructor_expanddim(indexed_output, index=None)
        output.columns = Index(key.label for key in results)

        output = self._reindex_output(output)
        return output

    def _cython_agg_general(
        self, how: str, alt: Callable, numeric_only: bool, min_count: int = -1
    ):

        obj = self._selected_obj
        objvals = obj._values
        data = obj._mgr

        if numeric_only and not is_numeric_dtype(obj.dtype):
            # GH#41291 match Series behavior
            raise NotImplementedError(
                f"{type(self).__name__}.{how} does not implement numeric_only."
            )

        # This is overkill because it is only called once, but is here to
        #  mirror the array_func used in DataFrameGroupBy._cython_agg_general
        def array_func(values: ArrayLike) -> ArrayLike:
            try:
                result = self.grouper._cython_operation(
                    "aggregate", values, how, axis=data.ndim - 1, min_count=min_count
                )
            except NotImplementedError:
                # generally if we have numeric_only=False
                # and non-applicable functions
                # try to python agg
                # TODO: shouldn't min_count matter?
                result = self._agg_py_fallback(values, ndim=data.ndim, alt=alt)

            return result

        result = array_func(objvals)

        ser = self.obj._constructor(
            result, index=self.grouper.result_index, name=obj.name
        )
        return self._reindex_output(ser)

    def _wrap_aggregated_output(
        self,
        output: Mapping[base.OutputKey, Series | ArrayLike],
    ) -> Series:
        """
        Wraps the output of a SeriesGroupBy aggregation into the expected result.

        Parameters
        ----------
        output : Mapping[base.OutputKey, Union[Series, ArrayLike]]
            Data to wrap.

        Returns
        -------
        Series

        Notes
        -----
        In the vast majority of cases output will only contain one element.
        The exception is operations that expand dimensions, like ohlc.
        """
        assert len(output) == 1

        name = self.obj.name
        index = self.grouper.result_index
        values = next(iter(output.values()))

        result = self.obj._constructor(values, index=index, name=name)
        return self._reindex_output(result)

    def _wrap_transformed_output(
        self, output: Mapping[base.OutputKey, Series | ArrayLike]
    ) -> Series:
        """
        Wraps the output of a SeriesGroupBy aggregation into the expected result.

        Parameters
        ----------
        output : dict[base.OutputKey, Union[Series, np.ndarray, ExtensionArray]]
            Dict with a sole key of 0 and a value of the result values.

        Returns
        -------
        Series

        Notes
        -----
        output should always contain one element. It is specified as a dict
        for consistency with DataFrame methods and _wrap_aggregated_output.
        """
        assert len(output) == 1

        name = self.obj.name
        values = next(iter(output.values()))
        result = self.obj._constructor(values, index=self.obj.index, name=name)

        # No transformations increase the ndim of the result
        assert isinstance(result, Series)
        return result

    def _wrap_applied_output(
        self,
        data: Series,
        keys: Index,
        values: list[Any] | None,
        not_indexed_same: bool = False,
    ) -> FrameOrSeriesUnion:
        """
        Wrap the output of SeriesGroupBy.apply into the expected result.

        Parameters
        ----------
        data : Series
            Input data for groupby operation.
        keys : Index
            Keys of groups that Series was grouped by.
        values : Optional[List[Any]]
            Applied output for each group.
        not_indexed_same : bool, default False
            Whether the applied outputs are not indexed the same as the group axes.

        Returns
        -------
        DataFrame or Series
        """
        if len(keys) == 0:
            # GH #6265
            return self.obj._constructor(
                [],
                name=self.obj.name,
                index=self.grouper.result_index,
                dtype=data.dtype,
            )
        assert values is not None

        def _get_index() -> Index:
            if self.grouper.nkeys > 1:
                index = MultiIndex.from_tuples(keys, names=self.grouper.names)
            else:
                index = Index(keys, name=self.grouper.names[0])
            return index

        if isinstance(values[0], dict):
            # GH #823 #24880
            index = _get_index()
            res_df = self.obj._constructor_expanddim(values, index=index)
            res_df = self._reindex_output(res_df)
            # if self.observed is False,
            # keep all-NaN rows created while re-indexing
            res_ser = res_df.stack(dropna=self.observed)
            res_ser.name = self.obj.name
            return res_ser
        elif isinstance(values[0], (Series, DataFrame)):
            return self._concat_objects(keys, values, not_indexed_same=not_indexed_same)
        else:
            # GH #6265 #24880
            result = self.obj._constructor(
                data=values, index=_get_index(), name=self.obj.name
            )
            return self._reindex_output(result)

    def _aggregate_named(self, func, *args, **kwargs):
        # Note: this is very similar to _aggregate_series_pure_python,
        #  but that does not pin group.name
        result = {}
        initialized = False

        for name, group in self:
            # Each step of this loop corresponds to
            #  libreduction._BaseGrouper._apply_to_group
            # NB: libreduction does not pin name
            object.__setattr__(group, "name", name)

            output = func(group, *args, **kwargs)
            output = libreduction.extract_result(output)
            if not initialized:
                # We only do this validation on the first iteration
                libreduction.check_result_array(output, group.dtype)
                initialized = True
            result[name] = output

        return result

    @Substitution(klass="Series")
    @Appender(_transform_template)
    def transform(self, func, *args, engine=None, engine_kwargs=None, **kwargs):
        return self._transform(
            func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs
        )

    def _cython_transform(
        self, how: str, numeric_only: bool = True, axis: int = 0, **kwargs
    ):
        assert axis == 0  # handled by caller

        obj = self._selected_obj

        try:
            result = self.grouper._cython_operation(
                "transform", obj._values, how, axis, **kwargs
            )
        except NotImplementedError as err:
            raise TypeError(f"{how} is not supported for {obj.dtype} dtype") from err

        return obj._constructor(result, index=self.obj.index, name=obj.name)

    def _transform_general(self, func: Callable, *args, **kwargs) -> Series:
        """
        Transform with a callable func`.
        """
        assert callable(func)
        klass = type(self.obj)

        results = []
        for name, group in self:
            # this setattr is needed for test_transform_lambda_with_datetimetz
            object.__setattr__(group, "name", name)
            res = func(group, *args, **kwargs)

            results.append(klass(res, index=group.index))

        # check for empty "results" to avoid concat ValueError
        if results:
            from pandas.core.reshape.concat import concat

            concatenated = concat(results)
            result = self._set_result_index_ordered(concatenated)
        else:
            result = self.obj._constructor(dtype=np.float64)

        result.name = self.obj.name
        # error: Incompatible return value type (got "Union[DataFrame, Series]",
        # expected "Series")
        return result  # type: ignore[return-value]

    def _can_use_transform_fast(self, result) -> bool:
        return True

    def _wrap_transform_fast_result(self, result: Series) -> Series:
        """
        fast version of transform, only applicable to
        builtin/cythonizable functions
        """
        ids, _, _ = self.grouper.group_info
        result = result.reindex(self.grouper.result_index, copy=False)
        out = algorithms.take_nd(result._values, ids)
        return self.obj._constructor(out, index=self.obj.index, name=self.obj.name)

    def filter(self, func, dropna: bool = True, *args, **kwargs):
        """
        Return a copy of a Series excluding elements from groups that
        do not satisfy the boolean criterion specified by func.

        Parameters
        ----------
        func : function
            To apply to each group. Should return True or False.
        dropna : Drop groups that do not pass the filter. True by default;
            if False, groups that evaluate False are filled with NaNs.

        Notes
        -----
        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
        ...                           'foo', 'bar'],
        ...                    'B' : [1, 2, 3, 4, 5, 6],
        ...                    'C' : [2.0, 5., 8., 1., 2., 9.]})
        >>> grouped = df.groupby('A')
        >>> df.groupby('A').B.filter(lambda x: x.mean() > 3.)
        1    2
        3    4
        5    6
        Name: B, dtype: int64

        Returns
        -------
        filtered : Series
        """
        if isinstance(func, str):
            wrapper = lambda x: getattr(x, func)(*args, **kwargs)
        else:
            wrapper = lambda x: func(x, *args, **kwargs)

        # Interpret np.nan as False.
        def true_and_notna(x) -> bool:
            b = wrapper(x)
            return b and notna(b)

        try:
            indices = [
                self._get_index(name) for name, group in self if true_and_notna(group)
            ]
        except (ValueError, TypeError) as err:
            raise TypeError("the filter must return a boolean result") from err

        filtered = self._apply_filter(indices, dropna)
        return filtered

    def nunique(self, dropna: bool = True) -> Series:
        """
        Return number of unique elements in the group.

        Returns
        -------
        Series
            Number of unique values within each group.
        """
        ids, _, _ = self.grouper.group_info

        val = self.obj._values

        codes, _ = algorithms.factorize(val, sort=False)
        sorter = np.lexsort((codes, ids))
        codes = codes[sorter]
        ids = ids[sorter]

        # group boundaries are where group ids change
        # unique observations are where sorted values change
        idx = np.r_[0, 1 + np.nonzero(ids[1:] != ids[:-1])[0]]
        inc = np.r_[1, codes[1:] != codes[:-1]]

        # 1st item of each group is a new unique observation
        mask = codes == -1
        if dropna:
            inc[idx] = 1
            inc[mask] = 0
        else:
            inc[mask & np.r_[False, mask[:-1]]] = 0
            inc[idx] = 1

        out = np.add.reduceat(inc, idx).astype("int64", copy=False)
        if len(ids):
            # NaN/NaT group exists if the head of ids is -1,
            # so remove it from res and exclude its index from idx
            if ids[0] == -1:
                res = out[1:]
                idx = idx[np.flatnonzero(idx)]
            else:
                res = out
        else:
            res = out[1:]
        ri = self.grouper.result_index

        # we might have duplications among the bins
        if len(res) != len(ri):
            res, out = np.zeros(len(ri), dtype=out.dtype), res
            res[ids[idx]] = out

        result = self.obj._constructor(res, index=ri, name=self.obj.name)
        return self._reindex_output(result, fill_value=0)

    @doc(Series.describe)
    def describe(self, **kwargs):
        result = self.apply(lambda x: x.describe(**kwargs))
        if self.axis == 1:
            return result.T
        return result.unstack()

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins=None,
        dropna: bool = True,
    ):

        from pandas.core.reshape.merge import get_join_indexers
        from pandas.core.reshape.tile import cut

        ids, _, _ = self.grouper.group_info
        val = self.obj._values

        def apply_series_value_counts():
            return self.apply(
                Series.value_counts,
                normalize=normalize,
                sort=sort,
                ascending=ascending,
                bins=bins,
            )

        if bins is not None:
            if not np.iterable(bins):
                # scalar bins cannot be done at top level
                # in a backward compatible way
                return apply_series_value_counts()
        elif is_categorical_dtype(val.dtype):
            # GH38672
            return apply_series_value_counts()

        # groupby removes null keys from groupings
        mask = ids != -1
        ids, val = ids[mask], val[mask]

        if bins is None:
            lab, lev = algorithms.factorize(val, sort=True)
            llab = lambda lab, inc: lab[inc]
        else:

            # lab is a Categorical with categories an IntervalIndex
            lab = cut(Series(val), bins, include_lowest=True)
            # error: "ndarray" has no attribute "cat"
            lev = lab.cat.categories  # type: ignore[attr-defined]
            # error: No overload variant of "take" of "_ArrayOrScalarCommon" matches
            # argument types "Any", "bool", "Union[Any, float]"
            lab = lev.take(  # type: ignore[call-overload]
                # error: "ndarray" has no attribute "cat"
                lab.cat.codes,  # type: ignore[attr-defined]
                allow_fill=True,
                # error: Item "ndarray" of "Union[ndarray, Index]" has no attribute
                # "_na_value"
                fill_value=lev._na_value,  # type: ignore[union-attr]
            )
            llab = lambda lab, inc: lab[inc]._multiindex.codes[-1]

        if is_interval_dtype(lab.dtype):
            # TODO: should we do this inside II?

            # error: "ndarray" has no attribute "left"
            # error: "ndarray" has no attribute "right"
            sorter = np.lexsort(
                (lab.left, lab.right, ids)  # type: ignore[attr-defined]
            )
        else:
            sorter = np.lexsort((lab, ids))

        ids, lab = ids[sorter], lab[sorter]

        # group boundaries are where group ids change
        idchanges = 1 + np.nonzero(ids[1:] != ids[:-1])[0]
        idx = np.r_[0, idchanges]
        if not len(ids):
            idx = idchanges

        # new values are where sorted labels change
        lchanges = llab(lab, slice(1, None)) != llab(lab, slice(None, -1))
        inc = np.r_[True, lchanges]
        if not len(val):
            inc = lchanges
        inc[idx] = True  # group boundaries are also new values
        out = np.diff(np.nonzero(np.r_[inc, True])[0])  # value counts

        # num. of times each group should be repeated
        rep = partial(np.repeat, repeats=np.add.reduceat(inc, idx))

        # multi-index components
        codes = self.grouper.reconstructed_codes
        codes = [rep(level_codes) for level_codes in codes] + [llab(lab, inc)]
        levels = [ping.group_index for ping in self.grouper.groupings] + [lev]
        names = self.grouper.names + [self.obj.name]

        if dropna:
            mask = codes[-1] != -1
            if mask.all():
                dropna = False
            else:
                out, codes = out[mask], [level_codes[mask] for level_codes in codes]

        if normalize:
            out = out.astype("float")
            d = np.diff(np.r_[idx, len(ids)])
            if dropna:
                m = ids[lab == -1]
                np.add.at(d, m, -1)
                acc = rep(d)[mask]
            else:
                acc = rep(d)
            out /= acc

        if sort and bins is None:
            cat = ids[inc][mask] if dropna else ids[inc]
            sorter = np.lexsort((out if ascending else -out, cat))
            out, codes[-1] = out[sorter], codes[-1][sorter]

        if bins is not None:
            # for compat. with libgroupby.value_counts need to ensure every
            # bin is present at every index level, null filled with zeros
            diff = np.zeros(len(out), dtype="bool")
            for level_codes in codes[:-1]:
                diff |= np.r_[True, level_codes[1:] != level_codes[:-1]]

            ncat, nbin = diff.sum(), len(levels[-1])

            left = [np.repeat(np.arange(ncat), nbin), np.tile(np.arange(nbin), ncat)]

            right = [diff.cumsum() - 1, codes[-1]]

            _, idx = get_join_indexers(left, right, sort=False, how="left")
            out = np.where(idx != -1, out[idx], 0)

            if sort:
                sorter = np.lexsort((out if ascending else -out, left[0]))
                out, left[-1] = out[sorter], left[-1][sorter]

            # build the multi-index w/ full levels
            def build_codes(lev_codes: np.ndarray) -> np.ndarray:
                return np.repeat(lev_codes[diff], nbin)

            codes = [build_codes(lev_codes) for lev_codes in codes[:-1]]
            codes.append(left[-1])

        mi = MultiIndex(levels=levels, codes=codes, names=names, verify_integrity=False)

        if is_integer_dtype(out.dtype):
            out = ensure_int64(out)
        return self.obj._constructor(out, index=mi, name=self.obj.name)

    def count(self) -> Series:
        """
        Compute count of group, excluding missing values.

        Returns
        -------
        Series
            Count of values within each group.
        """
        ids, _, ngroups = self.grouper.group_info
        val = self.obj._values

        mask = (ids != -1) & ~isna(val)
        minlength = ngroups or 0
        out = np.bincount(ids[mask], minlength=minlength)

        result = self.obj._constructor(
            out,
            index=self.grouper.result_index,
            name=self.obj.name,
            dtype="int64",
        )
        return self._reindex_output(result, fill_value=0)

    def pct_change(self, periods=1, fill_method="pad", limit=None, freq=None):
        """Calculate pct_change of each value to previous entry in group"""
        # TODO: Remove this conditional when #23918 is fixed
        if freq:
            return self.apply(
                lambda x: x.pct_change(
                    periods=periods, fill_method=fill_method, limit=limit, freq=freq
                )
            )
        if fill_method is None:  # GH30463
            fill_method = "pad"
            limit = 0
        filled = getattr(self, fill_method)(limit=limit)
        fill_grp = filled.groupby(self.grouper.codes)
        shifted = fill_grp.shift(periods=periods, freq=freq)

        return (filled / shifted) - 1


@pin_allowlisted_properties(DataFrame, base.dataframe_apply_allowlist)
class DataFrameGroupBy(GroupBy[DataFrame]):

    _apply_allowlist = base.dataframe_apply_allowlist

    _agg_examples_doc = dedent(
        """
    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {
    ...         "A": [1, 1, 2, 2],
    ...         "B": [1, 2, 3, 4],
    ...         "C": [0.362838, 0.227877, 1.267767, -0.562860],
    ...     }
    ... )

    >>> df
       A  B         C
    0  1  1  0.362838
    1  1  2  0.227877
    2  2  3  1.267767
    3  2  4 -0.562860

    The aggregation is for each column.

    >>> df.groupby('A').agg('min')
       B         C
    A
    1  1  0.227877
    2  3 -0.562860

    Multiple aggregations

    >>> df.groupby('A').agg(['min', 'max'])
        B             C
      min max       min       max
    A
    1   1   2  0.227877  0.362838
    2   3   4 -0.562860  1.267767

    Select a column for aggregation

    >>> df.groupby('A').B.agg(['min', 'max'])
       min  max
    A
    1    1    2
    2    3    4

    Different aggregations per column

    >>> df.groupby('A').agg({'B': ['min', 'max'], 'C': 'sum'})
        B             C
      min max       sum
    A
    1   1   2  0.590715
    2   3   4  0.704907

    To control the output names with different aggregations per column,
    pandas supports "named aggregation"

    >>> df.groupby("A").agg(
    ...     b_min=pd.NamedAgg(column="B", aggfunc="min"),
    ...     c_sum=pd.NamedAgg(column="C", aggfunc="sum"))
       b_min     c_sum
    A
    1      1  0.590715
    2      3  0.704907

    - The keywords are the *output* column names
    - The values are tuples whose first element is the column to select
      and the second element is the aggregation to apply to that column.
      Pandas provides the ``pandas.NamedAgg`` namedtuple with the fields
      ``['column', 'aggfunc']`` to make it clearer what the arguments are.
      As usual, the aggregation can be a callable or a string alias.

    See :ref:`groupby.aggregate.named` for more.

    .. versionchanged:: 1.3.0

        The resulting dtype will reflect the return value of the aggregating function.

    >>> df.groupby("A")[["B"]].agg(lambda x: x.astype(float).min())
          B
    A
    1   1.0
    2   3.0"""
    )

    @doc(_agg_template, examples=_agg_examples_doc, klass="DataFrame")
    def aggregate(self, func=None, *args, engine=None, engine_kwargs=None, **kwargs):

        if maybe_use_numba(engine):
            with group_selection_context(self):
                data = self._selected_obj
            result, index = self._aggregate_with_numba(
                data, func, *args, engine_kwargs=engine_kwargs, **kwargs
            )
            return self.obj._constructor(result, index=index, columns=data.columns)

        relabeling, func, columns, order = reconstruct_func(func, **kwargs)
        func = maybe_mangle_lambdas(func)

        op = GroupByApply(self, func, args, kwargs)
        result = op.agg()
        if not is_dict_like(func) and result is not None:
            return result
        elif relabeling and result is not None:
            # this should be the only (non-raising) case with relabeling
            # used reordered index of columns
            result = result.iloc[:, order]
            result.columns = columns

        if result is None:

            # grouper specific aggregations
            if self.grouper.nkeys > 1:
                # test_groupby_as_index_series_scalar gets here with 'not self.as_index'
                return self._python_agg_general(func, *args, **kwargs)
            elif args or kwargs:
                # test_pass_args_kwargs gets here (with and without as_index)
                # can't return early
                result = self._aggregate_frame(func, *args, **kwargs)

            elif self.axis == 1:
                # _aggregate_multiple_funcs does not allow self.axis == 1
                # Note: axis == 1 precludes 'not self.as_index', see __init__
                result = self._aggregate_frame(func)
                return result

            else:

                # try to treat as if we are passing a list
                gba = GroupByApply(self, [func], args=(), kwargs={})
                try:
                    result = gba.agg()

                except ValueError as err:
                    if "no results" not in str(err):
                        # raised directly by _aggregate_multiple_funcs
                        raise
                    result = self._aggregate_frame(func)

                else:
                    sobj = self._selected_obj

                    if isinstance(sobj, Series):
                        # GH#35246 test_groupby_as_index_select_column_sum_empty_df
                        result.columns = self._obj_with_exclusions.columns.copy()
                    else:
                        # Retain our column names
                        result.columns._set_names(
                            sobj.columns.names, level=list(range(sobj.columns.nlevels))
                        )
                        # select everything except for the last level, which is the one
                        # containing the name of the function(s), see GH#32040
                        result.columns = result.columns.droplevel(-1)

        if not self.as_index:
            self._insert_inaxis_grouper_inplace(result)
            result.index = Index(range(len(result)))

        return result._convert(datetime=True)

    agg = aggregate

    def _iterate_slices(self) -> Iterable[Series]:
        obj = self._selected_obj
        if self.axis == 1:
            obj = obj.T

        if isinstance(obj, Series) and obj.name not in self.exclusions:
            # Occurs when doing DataFrameGroupBy(...)["X"]
            yield obj
        else:
            for label, values in obj.items():
                if label in self.exclusions:
                    continue

                yield values

    def _cython_agg_general(
        self, how: str, alt: Callable, numeric_only: bool, min_count: int = -1
    ) -> DataFrame:
        # Note: we never get here with how="ohlc"; that goes through SeriesGroupBy

        data: Manager2D = self._get_data_to_aggregate()

        if numeric_only:
            data = data.get_numeric_data(copy=False)

        def array_func(values: ArrayLike) -> ArrayLike:
            try:
                result = self.grouper._cython_operation(
                    "aggregate", values, how, axis=data.ndim - 1, min_count=min_count
                )
            except NotImplementedError:
                # generally if we have numeric_only=False
                # and non-applicable functions
                # try to python agg
                # TODO: shouldn't min_count matter?
                result = self._agg_py_fallback(values, ndim=data.ndim, alt=alt)

            return result

        # TypeError -> we may have an exception in trying to aggregate
        #  continue and exclude the block
        new_mgr = data.grouped_reduce(array_func, ignore_failures=True)

        if len(new_mgr) < len(data):
            warnings.warn(
                f"Dropping invalid columns in {type(self).__name__}.{how} "
                "is deprecated. In a future version, a TypeError will be raised. "
                f"Before calling .{how}, select only columns which should be "
                "valid for the function.",
                FutureWarning,
                stacklevel=4,
            )

        return self._wrap_agged_manager(new_mgr)

    def _aggregate_frame(self, func, *args, **kwargs) -> DataFrame:
        if self.grouper.nkeys != 1:
            raise AssertionError("Number of keys must be 1")

        obj = self._obj_with_exclusions

        result: dict[Hashable, NDFrame | np.ndarray] = {}
        if self.axis == 0:
            # test_pass_args_kwargs_duplicate_columns gets here with non-unique columns
            for name, data in self:
                fres = func(data, *args, **kwargs)
                result[name] = fres
        else:
            # we get here in a number of test_multilevel tests
            for name in self.indices:
                grp_df = self.get_group(name, obj=obj)
                fres = func(grp_df, *args, **kwargs)
                result[name] = fres

        result_index = self.grouper.result_index
        other_ax = obj.axes[1 - self.axis]
        out = self.obj._constructor(result, index=other_ax, columns=result_index)
        if self.axis == 0:
            out = out.T

        return out

    def _aggregate_item_by_item(self, func, *args, **kwargs) -> DataFrame:
        # only for axis==0
        # tests that get here with non-unique cols:
        #  test_resample_with_timedelta_yields_no_empty_groups,
        #  test_resample_apply_product

        obj = self._obj_with_exclusions
        result: dict[int | str, NDFrame] = {}
        for i, item in enumerate(obj):
            ser = obj.iloc[:, i]
            colg = SeriesGroupBy(
                ser, selection=item, grouper=self.grouper, exclusions=self.exclusions
            )

            result[i] = colg.aggregate(func, *args, **kwargs)

        res_df = self.obj._constructor(result)
        res_df.columns = obj.columns
        return res_df

    def _wrap_applied_output(self, data, keys, values, not_indexed_same=False):
        if len(keys) == 0:
            result = self.obj._constructor(
                index=self.grouper.result_index, columns=data.columns
            )
            result = result.astype(data.dtypes.to_dict(), copy=False)
            return result

        # GH12824
        first_not_none = next(com.not_none(*values), None)

        if first_not_none is None:
            # GH9684 - All values are None, return an empty frame.
            return self.obj._constructor()
        elif isinstance(first_not_none, DataFrame):
            return self._concat_objects(keys, values, not_indexed_same=not_indexed_same)

        key_index = self.grouper.result_index if self.as_index else None

        if isinstance(first_not_none, (np.ndarray, Index)):
            # GH#1738: values is list of arrays of unequal lengths
            #  fall through to the outer else clause
            # TODO: sure this is right?  we used to do this
            #  after raising AttributeError above
            return self.obj._constructor_sliced(
                values, index=key_index, name=self._selection
            )
        elif not isinstance(first_not_none, Series):
            # values are not series or array-like but scalars
            # self._selection not passed through to Series as the
            # result should not take the name of original selection
            # of columns
            if self.as_index:
                return self.obj._constructor_sliced(values, index=key_index)
            else:
                result = self.obj._constructor(
                    values, index=key_index, columns=[self._selection]
                )
                self._insert_inaxis_grouper_inplace(result)
                return result
        else:
            # values are Series
            return self._wrap_applied_output_series(
                keys, values, not_indexed_same, first_not_none, key_index
            )

    def _wrap_applied_output_series(
        self,
        keys,
        values: list[Series],
        not_indexed_same: bool,
        first_not_none,
        key_index,
    ) -> FrameOrSeriesUnion:
        # this is to silence a DeprecationWarning
        # TODO: Remove when default dtype of empty Series is object
        kwargs = first_not_none._construct_axes_dict()
        backup = create_series_with_explicit_dtype(dtype_if_empty=object, **kwargs)
        values = [x if (x is not None) else backup for x in values]

        all_indexed_same = all_indexes_same(x.index for x in values)

        # GH3596
        # provide a reduction (Frame -> Series) if groups are
        # unique
        if self.squeeze:
            applied_index = self._selected_obj._get_axis(self.axis)
            singular_series = len(values) == 1 and applied_index.nlevels == 1

            # assign the name to this series
            if singular_series:
                values[0].name = keys[0]

                # GH2893
                # we have series in the values array, we want to
                # produce a series:
                # if any of the sub-series are not indexed the same
                # OR we don't have a multi-index and we have only a
                # single values
                return self._concat_objects(
                    keys, values, not_indexed_same=not_indexed_same
                )

            # still a series
            # path added as of GH 5545
            elif all_indexed_same:
                from pandas.core.reshape.concat import concat

                return concat(values)

        if not all_indexed_same:
            # GH 8467
            return self._concat_objects(keys, values, not_indexed_same=True)

        # Combine values
        # vstack+constructor is faster than concat and handles MI-columns
        stacked_values = np.vstack([np.asarray(v) for v in values])

        if self.axis == 0:
            index = key_index
            columns = first_not_none.index.copy()
            if columns.name is None:
                # GH6124 - propagate name of Series when it's consistent
                names = {v.name for v in values}
                if len(names) == 1:
                    columns.name = list(names)[0]
        else:
            index = first_not_none.index
            columns = key_index
            stacked_values = stacked_values.T

        if stacked_values.dtype == object:
            # We'll have the DataFrame constructor do inference
            stacked_values = stacked_values.tolist()
        result = self.obj._constructor(stacked_values, index=index, columns=columns)

        if not self.as_index:
            self._insert_inaxis_grouper_inplace(result)

        return self._reindex_output(result)

    def _cython_transform(
        self, how: str, numeric_only: bool = True, axis: int = 0, **kwargs
    ) -> DataFrame:
        assert axis == 0  # handled by caller
        # TODO: no tests with self.ndim == 1 for DataFrameGroupBy

        # With self.axis == 0, we have multi-block tests
        #  e.g. test_rank_min_int, test_cython_transform_frame
        #  test_transform_numeric_ret
        # With self.axis == 1, _get_data_to_aggregate does a transpose
        #  so we always have a single block.
        mgr: Manager2D = self._get_data_to_aggregate()
        if numeric_only:
            mgr = mgr.get_numeric_data(copy=False)

        def arr_func(bvalues: ArrayLike) -> ArrayLike:
            return self.grouper._cython_operation(
                "transform", bvalues, how, 1, **kwargs
            )

        # We could use `mgr.apply` here and not have to set_axis, but
        #  we would have to do shape gymnastics for ArrayManager compat
        res_mgr = mgr.grouped_reduce(arr_func, ignore_failures=True)
        res_mgr.set_axis(1, mgr.axes[1])

        if len(res_mgr) < len(mgr):
            warnings.warn(
                f"Dropping invalid columns in {type(self).__name__}.{how} "
                "is deprecated. In a future version, a TypeError will be raised. "
                f"Before calling .{how}, select only columns which should be "
                "valid for the transforming function.",
                FutureWarning,
                stacklevel=4,
            )

        res_df = self.obj._constructor(res_mgr)
        if self.axis == 1:
            res_df = res_df.T
        return res_df

    def _transform_general(self, func, *args, **kwargs):
        from pandas.core.reshape.concat import concat

        applied = []
        obj = self._obj_with_exclusions
        gen = self.grouper.get_iterator(obj, axis=self.axis)
        fast_path, slow_path = self._define_paths(func, *args, **kwargs)

        for name, group in gen:
            object.__setattr__(group, "name", name)

            # Try slow path and fast path.
            try:
                path, res = self._choose_path(fast_path, slow_path, group)
            except TypeError:
                return self._transform_item_by_item(obj, fast_path)
            except ValueError as err:
                msg = "transform must return a scalar value for each group"
                raise ValueError(msg) from err

            if isinstance(res, Series):

                # we need to broadcast across the
                # other dimension; this will preserve dtypes
                # GH14457
                if not np.prod(group.shape):
                    continue
                elif res.index.is_(obj.index):
                    r = concat([res] * len(group.columns), axis=1)
                    r.columns = group.columns
                    r.index = group.index
                else:
                    r = self.obj._constructor(
                        np.concatenate([res.values] * len(group.index)).reshape(
                            group.shape
                        ),
                        columns=group.columns,
                        index=group.index,
                    )

                applied.append(r)
            else:
                applied.append(res)

        concat_index = obj.columns if self.axis == 0 else obj.index
        other_axis = 1 if self.axis == 0 else 0  # switches between 0 & 1
        concatenated = concat(applied, axis=self.axis, verify_integrity=False)
        concatenated = concatenated.reindex(concat_index, axis=other_axis, copy=False)
        return self._set_result_index_ordered(concatenated)

    @Substitution(klass="DataFrame")
    @Appender(_transform_template)
    def transform(self, func, *args, engine=None, engine_kwargs=None, **kwargs):
        return self._transform(
            func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs
        )

    def _can_use_transform_fast(self, result) -> bool:
        return isinstance(result, DataFrame) and result.columns.equals(
            self._obj_with_exclusions.columns
        )

    def _wrap_transform_fast_result(self, result: DataFrame) -> DataFrame:
        """
        Fast transform path for aggregations
        """
        obj = self._obj_with_exclusions

        # for each col, reshape to size of original frame by take operation
        ids, _, _ = self.grouper.group_info
        result = result.reindex(self.grouper.result_index, copy=False)
        output = result.take(ids, axis=0)
        output.index = obj.index
        return output

    def _define_paths(self, func, *args, **kwargs):
        if isinstance(func, str):
            fast_path = lambda group: getattr(group, func)(*args, **kwargs)
            slow_path = lambda group: group.apply(
                lambda x: getattr(x, func)(*args, **kwargs), axis=self.axis
            )
        else:
            fast_path = lambda group: func(group, *args, **kwargs)
            slow_path = lambda group: group.apply(
                lambda x: func(x, *args, **kwargs), axis=self.axis
            )
        return fast_path, slow_path

    def _choose_path(self, fast_path: Callable, slow_path: Callable, group: DataFrame):
        path = slow_path
        res = slow_path(group)

        # if we make it here, test if we can use the fast path
        try:
            res_fast = fast_path(group)
        except AssertionError:
            raise  # pragma: no cover
        except Exception:
            # GH#29631 For user-defined function, we can't predict what may be
            #  raised; see test_transform.test_transform_fastpath_raises
            return path, res

        # verify fast path does not change columns (and names), otherwise
        # its results cannot be joined with those of the slow path
        if not isinstance(res_fast, DataFrame):
            return path, res

        if not res_fast.columns.equals(group.columns):
            return path, res

        if res_fast.equals(res):
            path = fast_path

        return path, res

    def _transform_item_by_item(self, obj: DataFrame, wrapper) -> DataFrame:
        # iterate through columns, see test_transform_exclude_nuisance
        #  gets here with non-unique columns
        output = {}
        inds = []
        for i, col in enumerate(obj):
            subset = obj.iloc[:, i]
            sgb = SeriesGroupBy(
                subset,
                selection=col,
                grouper=self.grouper,
                exclusions=self.exclusions,
            )
            try:
                output[i] = sgb.transform(wrapper)
            except TypeError:
                # e.g. trying to call nanmean with string values
                warnings.warn(
                    f"Dropping invalid columns in {type(self).__name__}.transform "
                    "is deprecated. In a future version, a TypeError will be raised. "
                    "Before calling .transform, select only columns which should be "
                    "valid for the transforming function.",
                    FutureWarning,
                    stacklevel=5,
                )
            else:
                inds.append(i)

        if not output:
            raise TypeError("Transform function invalid for data types")

        columns = obj.columns.take(inds)

        result = self.obj._constructor(output, index=obj.index)
        result.columns = columns
        return result

    def filter(self, func, dropna=True, *args, **kwargs):
        """
        Return a copy of a DataFrame excluding filtered elements.

        Elements from groups are filtered if they do not satisfy the
        boolean criterion specified by func.

        Parameters
        ----------
        func : function
            Function to apply to each subframe. Should return True or False.
        dropna : Drop groups that do not pass the filter. True by default;
            If False, groups that evaluate False are filled with NaNs.

        Returns
        -------
        filtered : DataFrame

        Notes
        -----
        Each subframe is endowed the attribute 'name' in case you need to know
        which group you are working on.

        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
        ...                           'foo', 'bar'],
        ...                    'B' : [1, 2, 3, 4, 5, 6],
        ...                    'C' : [2.0, 5., 8., 1., 2., 9.]})
        >>> grouped = df.groupby('A')
        >>> grouped.filter(lambda x: x['B'].mean() > 3.)
             A  B    C
        1  bar  2  5.0
        3  bar  4  1.0
        5  bar  6  9.0
        """
        indices = []

        obj = self._selected_obj
        gen = self.grouper.get_iterator(obj, axis=self.axis)

        for name, group in gen:
            object.__setattr__(group, "name", name)

            res = func(group, *args, **kwargs)

            try:
                res = res.squeeze()
            except AttributeError:  # allow e.g., scalars and frames to pass
                pass

            # interpret the result of the filter
            if is_bool(res) or (is_scalar(res) and isna(res)):
                if res and notna(res):
                    indices.append(self._get_index(name))
            else:
                # non scalars aren't allowed
                raise TypeError(
                    f"filter function returned a {type(res).__name__}, "
                    "but expected a scalar bool"
                )

        return self._apply_filter(indices, dropna)

    def __getitem__(self, key) -> DataFrameGroupBy | SeriesGroupBy:
        if self.axis == 1:
            # GH 37725
            raise ValueError("Cannot subset columns when using axis=1")
        # per GH 23566
        if isinstance(key, tuple) and len(key) > 1:
            # if len == 1, then it becomes a SeriesGroupBy and this is actually
            # valid syntax, so don't raise warning
            warnings.warn(
                "Indexing with multiple keys (implicitly converted to a tuple "
                "of keys) will be deprecated, use a list instead.",
                FutureWarning,
                stacklevel=2,
            )
        return super().__getitem__(key)

    def _gotitem(self, key, ndim: int, subset=None):
        """
        sub-classes to define
        return a sliced object

        Parameters
        ----------
        key : string / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        if ndim == 2:
            if subset is None:
                subset = self.obj
            return DataFrameGroupBy(
                subset,
                self.grouper,
                axis=self.axis,
                level=self.level,
                grouper=self.grouper,
                exclusions=self.exclusions,
                selection=key,
                as_index=self.as_index,
                sort=self.sort,
                group_keys=self.group_keys,
                squeeze=self.squeeze,
                observed=self.observed,
                mutated=self.mutated,
                dropna=self.dropna,
            )
        elif ndim == 1:
            if subset is None:
                subset = self.obj[key]
            return SeriesGroupBy(
                subset,
                level=self.level,
                grouper=self.grouper,
                selection=key,
                sort=self.sort,
                group_keys=self.group_keys,
                squeeze=self.squeeze,
                observed=self.observed,
                dropna=self.dropna,
            )

        raise AssertionError("invalid ndim for _gotitem")

    def _get_data_to_aggregate(self) -> Manager2D:
        obj = self._obj_with_exclusions
        if self.axis == 1:
            return obj.T._mgr
        else:
            return obj._mgr

    def _insert_inaxis_grouper_inplace(self, result: DataFrame) -> None:
        # zip in reverse so we can always insert at loc 0
        columns = result.columns
        for name, lev, in_axis in zip(
            reversed(self.grouper.names),
            reversed(self.grouper.get_group_levels()),
            reversed([grp.in_axis for grp in self.grouper.groupings]),
        ):
            # GH #28549
            # When using .apply(-), name will be in columns already
            if in_axis and name not in columns:
                result.insert(0, name, lev)

    def _wrap_aggregated_output(
        self,
        output: Mapping[base.OutputKey, Series | ArrayLike],
    ) -> DataFrame:
        """
        Wraps the output of DataFrameGroupBy aggregations into the expected result.

        Parameters
        ----------
        output : Mapping[base.OutputKey, Union[Series, np.ndarray]]
           Data to wrap.

        Returns
        -------
        DataFrame
        """
        indexed_output = {key.position: val for key, val in output.items()}
        columns = Index([key.label for key in output])
        columns._set_names(self._obj_with_exclusions._get_axis(1 - self.axis).names)

        result = self.obj._constructor(indexed_output)
        result.columns = columns

        if not self.as_index:
            self._insert_inaxis_grouper_inplace(result)
            result = result._consolidate()
        else:
            result.index = self.grouper.result_index

        if self.axis == 1:
            result = result.T
            if result.index.equals(self.obj.index):
                # Retain e.g. DatetimeIndex/TimedeltaIndex freq
                result.index = self.obj.index.copy()
                # TODO: Do this more systematically

        return self._reindex_output(result)

    def _wrap_transformed_output(
        self, output: Mapping[base.OutputKey, Series | ArrayLike]
    ) -> DataFrame:
        """
        Wraps the output of DataFrameGroupBy transformations into the expected result.

        Parameters
        ----------
        output : Mapping[base.OutputKey, Union[Series, np.ndarray, ExtensionArray]]
            Data to wrap.

        Returns
        -------
        DataFrame
        """
        indexed_output = {key.position: val for key, val in output.items()}
        result = self.obj._constructor(indexed_output)

        if self.axis == 1:
            result = result.T
            result.columns = self.obj.columns
        else:
            columns = Index(key.label for key in output)
            columns._set_names(self.obj._get_axis(1 - self.axis).names)
            result.columns = columns

        result.index = self.obj.index

        return result

    def _wrap_agged_manager(self, mgr: Manager2D) -> DataFrame:
        if not self.as_index:
            # GH 41998 - empty mgr always gets index of length 0
            rows = mgr.shape[1] if mgr.shape[0] > 0 else 0
            index = Index(range(rows))
            mgr.set_axis(1, index)
            result = self.obj._constructor(mgr)

            self._insert_inaxis_grouper_inplace(result)
            result = result._consolidate()
        else:
            index = self.grouper.result_index
            mgr.set_axis(1, index)
            result = self.obj._constructor(mgr)

        if self.axis == 1:
            result = result.T

        return self._reindex_output(result)._convert(datetime=True)

    def _iterate_column_groupbys(self, obj: FrameOrSeries):
        for i, colname in enumerate(obj.columns):
            yield colname, SeriesGroupBy(
                obj.iloc[:, i],
                selection=colname,
                grouper=self.grouper,
                exclusions=self.exclusions,
            )

    def _apply_to_column_groupbys(self, func, obj: FrameOrSeries) -> DataFrame:
        from pandas.core.reshape.concat import concat

        columns = obj.columns
        results = [
            func(col_groupby) for _, col_groupby in self._iterate_column_groupbys(obj)
        ]

        if not len(results):
            # concat would raise
            return DataFrame([], columns=columns, index=self.grouper.result_index)
        else:
            return concat(results, keys=columns, axis=1)

    def count(self) -> DataFrame:
        """
        Compute count of group, excluding missing values.

        Returns
        -------
        DataFrame
            Count of values within each group.
        """
        data = self._get_data_to_aggregate()
        ids, _, ngroups = self.grouper.group_info
        mask = ids != -1

        def hfunc(bvalues: ArrayLike) -> ArrayLike:
            # TODO(2DEA): reshape would not be necessary with 2D EAs
            if bvalues.ndim == 1:
                # EA
                masked = mask & ~isna(bvalues).reshape(1, -1)
            else:
                masked = mask & ~isna(bvalues)

            counted = lib.count_level_2d(masked, labels=ids, max_bin=ngroups, axis=1)
            return counted

        new_mgr = data.grouped_reduce(hfunc)

        # If we are grouping on categoricals we want unobserved categories to
        # return zero, rather than the default of NaN which the reindexing in
        # _wrap_agged_manager() returns. GH 35028
        with com.temp_setattr(self, "observed", True):
            result = self._wrap_agged_manager(new_mgr)

        return self._reindex_output(result, fill_value=0)

    def nunique(self, dropna: bool = True) -> DataFrame:
        """
        Return DataFrame with counts of unique elements in each position.

        Parameters
        ----------
        dropna : bool, default True
            Don't include NaN in the counts.

        Returns
        -------
        nunique: DataFrame

        Examples
        --------
        >>> df = pd.DataFrame({'id': ['spam', 'egg', 'egg', 'spam',
        ...                           'ham', 'ham'],
        ...                    'value1': [1, 5, 5, 2, 5, 5],
        ...                    'value2': list('abbaxy')})
        >>> df
             id  value1 value2
        0  spam       1      a
        1   egg       5      b
        2   egg       5      b
        3  spam       2      a
        4   ham       5      x
        5   ham       5      y

        >>> df.groupby('id').nunique()
              value1  value2
        id
        egg        1       1
        ham        1       2
        spam       2       1

        Check for rows with the same id but conflicting values:

        >>> df.groupby('id').filter(lambda g: (g.nunique() > 1).any())
             id  value1 value2
        0  spam       1      a
        3  spam       2      a
        4   ham       5      x
        5   ham       5      y
        """

        if self.axis != 0:
            # see test_groupby_crash_on_nunique
            return self._python_agg_general(lambda sgb: sgb.nunique(dropna))

        obj = self._obj_with_exclusions
        results = self._apply_to_column_groupbys(
            lambda sgb: sgb.nunique(dropna), obj=obj
        )

        if not self.as_index:
            results.index = Index(range(len(results)))
            self._insert_inaxis_grouper_inplace(results)

        return results

    @Appender(DataFrame.idxmax.__doc__)
    def idxmax(self, axis=0, skipna: bool = True):
        axis = DataFrame._get_axis_number(axis)
        numeric_only = None if axis == 0 else False

        def func(df):
            # NB: here we use numeric_only=None, in DataFrame it is False GH#38217
            res = df._reduce(
                nanops.nanargmax,
                "argmax",
                axis=axis,
                skipna=skipna,
                numeric_only=numeric_only,
            )
            indices = res._values
            index = df._get_axis(axis)
            result = [index[i] if i >= 0 else np.nan for i in indices]
            return df._constructor_sliced(result, index=res.index)

        return self._python_apply_general(func, self._obj_with_exclusions)

    @Appender(DataFrame.idxmin.__doc__)
    def idxmin(self, axis=0, skipna: bool = True):
        axis = DataFrame._get_axis_number(axis)
        numeric_only = None if axis == 0 else False

        def func(df):
            # NB: here we use numeric_only=None, in DataFrame it is False GH#38217
            res = df._reduce(
                nanops.nanargmin,
                "argmin",
                axis=axis,
                skipna=skipna,
                numeric_only=numeric_only,
            )
            indices = res._values
            index = df._get_axis(axis)
            result = [index[i] if i >= 0 else np.nan for i in indices]
            return df._constructor_sliced(result, index=res.index)

        return self._python_apply_general(func, self._obj_with_exclusions)

    boxplot = boxplot_frame_groupby
