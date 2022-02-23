"""
Provide the groupby split-apply-combine paradigm. Define the GroupBy
class providing the base-class of operations.

The SeriesGroupBy and DataFrameGroupBy sub-class
(defined in pandas.core.groupby.generic)
expose these user-facing objects to provide specific functionality.
"""
from __future__ import annotations

from contextlib import contextmanager
import datetime
from functools import (
    partial,
    wraps,
)
import inspect
from textwrap import dedent
import types
from typing import (
    TYPE_CHECKING,
    Callable,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Sequence,
    TypeVar,
    Union,
    cast,
)
import warnings

import numpy as np

from pandas._config.config import option_context

from pandas._libs import (
    Timestamp,
    lib,
)
import pandas._libs.groupby as libgroupby
from pandas._typing import (
    ArrayLike,
    F,
    FrameOrSeries,
    FrameOrSeriesUnion,
    IndexLabel,
    Scalar,
    T,
    final,
)
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
    Appender,
    Substitution,
    cache_readonly,
    doc,
)

from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_datetime64_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_scalar,
    is_timedelta64_dtype,
)
from pandas.core.dtypes.missing import (
    isna,
    notna,
)

from pandas.core import nanops
import pandas.core.algorithms as algorithms
from pandas.core.arrays import (
    BaseMaskedArray,
    BooleanArray,
    Categorical,
    ExtensionArray,
)
from pandas.core.base import (
    DataError,
    PandasObject,
    SelectionMixin,
)
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby import (
    base,
    numba_,
    ops,
)
from pandas.core.indexes.api import (
    CategoricalIndex,
    Index,
    MultiIndex,
)
from pandas.core.internals.blocks import ensure_block_shape
from pandas.core.series import Series
from pandas.core.sorting import get_group_index_sorter
from pandas.core.util.numba_ import (
    NUMBA_FUNC_CACHE,
    maybe_use_numba,
)

if TYPE_CHECKING:
    from typing import Literal

_common_see_also = """
        See Also
        --------
        Series.%(name)s : Apply a function %(name)s to a Series.
        DataFrame.%(name)s : Apply a function %(name)s
            to each row or column of a DataFrame.
"""

_apply_docs = {
    "template": """
    Apply function ``func`` group-wise and combine the results together.

    The function passed to ``apply`` must take a {input} as its first
    argument and return a DataFrame, Series or scalar. ``apply`` will
    then take care of combining the results back together into a single
    dataframe or series. ``apply`` is therefore a highly flexible
    grouping method.

    While ``apply`` is a very flexible method, its downside is that
    using it can be quite a bit slower than using more specific methods
    like ``agg`` or ``transform``. Pandas offers a wide range of method that will
    be much faster than using ``apply`` for their specific purposes, so try to
    use them before reaching for ``apply``.

    Parameters
    ----------
    func : callable
        A callable that takes a {input} as its first argument, and
        returns a dataframe, a series or a scalar. In addition the
        callable may take positional and keyword arguments.
    args, kwargs : tuple and dict
        Optional positional and keyword arguments to pass to ``func``.

    Returns
    -------
    applied : Series or DataFrame

    See Also
    --------
    pipe : Apply function to the full GroupBy object instead of to each
        group.
    aggregate : Apply aggregate function to the GroupBy object.
    transform : Apply function column-by-column to the GroupBy object.
    Series.apply : Apply a function to a Series.
    DataFrame.apply : Apply a function to each row or column of a DataFrame.

    Notes
    -----
    In the current implementation ``apply`` calls ``func`` twice on the
    first group to decide whether it can take a fast or slow code
    path. This can lead to unexpected behavior if ``func`` has
    side-effects, as they will take effect twice for the first
    group.

    .. versionchanged:: 1.3.0

        The resulting dtype will reflect the return value of the passed ``func``,
        see the examples below.

    Examples
    --------
    {examples}
    """,
    "dataframe_examples": """
    >>> df = pd.DataFrame({'A': 'a a b'.split(),
    ...                    'B': [1,2,3],
    ...                    'C': [4,6,5]})
    >>> g = df.groupby('A')

    Notice that ``g`` has two groups, ``a`` and ``b``.
    Calling `apply` in various ways, we can get different grouping results:

    Example 1: below the function passed to `apply` takes a DataFrame as
    its argument and returns a DataFrame. `apply` combines the result for
    each group together into a new DataFrame:

    >>> g[['B', 'C']].apply(lambda x: x / x.sum())
              B    C
    0  0.333333  0.4
    1  0.666667  0.6
    2  1.000000  1.0

    Example 2: The function passed to `apply` takes a DataFrame as
    its argument and returns a Series.  `apply` combines the result for
    each group together into a new DataFrame.

    .. versionchanged:: 1.3.0

        The resulting dtype will reflect the return value of the passed ``func``.

    >>> g[['B', 'C']].apply(lambda x: x.astype(float).max() - x.min())
         B    C
    A
    a  1.0  2.0
    b  0.0  0.0

    Example 3: The function passed to `apply` takes a DataFrame as
    its argument and returns a scalar. `apply` combines the result for
    each group together into a Series, including setting the index as
    appropriate:

    >>> g.apply(lambda x: x.C.max() - x.B.min())
    A
    a    5
    b    2
    dtype: int64""",
    "series_examples": """
    >>> s = pd.Series([0, 1, 2], index='a a b'.split())
    >>> g = s.groupby(s.index)

    From ``s`` above we can see that ``g`` has two groups, ``a`` and ``b``.
    Calling `apply` in various ways, we can get different grouping results:

    Example 1: The function passed to `apply` takes a Series as
    its argument and returns a Series.  `apply` combines the result for
    each group together into a new Series.

    .. versionchanged:: 1.3.0

        The resulting dtype will reflect the return value of the passed ``func``.

    >>> g.apply(lambda x: x*2 if x.name == 'a' else x/2)
    a    0.0
    a    2.0
    b    1.0
    dtype: float64

    Example 2: The function passed to `apply` takes a Series as
    its argument and returns a scalar. `apply` combines the result for
    each group together into a Series, including setting the index as
    appropriate:

    >>> g.apply(lambda x: x.max() - x.min())
    a    1
    b    0
    dtype: int64""",
}

_groupby_agg_method_template = """
Compute {fname} of group values.

Parameters
----------
numeric_only : bool, default {no}
    Include only float, int, boolean columns. If None, will attempt to use
    everything, then use only numeric data.
min_count : int, default {mc}
    The required number of valid values to perform the operation. If fewer
    than ``min_count`` non-NA values are present the result will be NA.

Returns
-------
Series or DataFrame
    Computed {fname} of values within each group.
"""

_pipe_template = """
Apply a function `func` with arguments to this %(klass)s object and return
the function's result.

Use `.pipe` when you want to improve readability by chaining together
functions that expect Series, DataFrames, GroupBy or Resampler objects.
Instead of writing

>>> h(g(f(df.groupby('group')), arg1=a), arg2=b, arg3=c)  # doctest: +SKIP

You can write

>>> (df.groupby('group')
...    .pipe(f)
...    .pipe(g, arg1=a)
...    .pipe(h, arg2=b, arg3=c))  # doctest: +SKIP

which is much more readable.

Parameters
----------
func : callable or tuple of (callable, str)
    Function to apply to this %(klass)s object or, alternatively,
    a `(callable, data_keyword)` tuple where `data_keyword` is a
    string indicating the keyword of `callable` that expects the
    %(klass)s object.
args : iterable, optional
       Positional arguments passed into `func`.
kwargs : dict, optional
         A dictionary of keyword arguments passed into `func`.

Returns
-------
object : the return type of `func`.

See Also
--------
Series.pipe : Apply a function with arguments to a series.
DataFrame.pipe: Apply a function with arguments to a dataframe.
apply : Apply function to each group instead of to the
    full %(klass)s object.

Notes
-----
See more `here
<https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#piping-function-calls>`_

Examples
--------
%(examples)s
"""

_transform_template = """
Call function producing a like-indexed %(klass)s on each group and
return a %(klass)s having the same indexes as the original object
filled with the transformed values

Parameters
----------
f : function
    Function to apply to each group.

    Can also accept a Numba JIT function with
    ``engine='numba'`` specified.

    If the ``'numba'`` engine is chosen, the function must be
    a user defined function with ``values`` and ``index`` as the
    first and second arguments respectively in the function signature.
    Each group's index will be passed to the user defined function
    and optionally available for use.

    .. versionchanged:: 1.1.0
*args
    Positional arguments to pass to func.
engine : str, default None
    * ``'cython'`` : Runs the function through C-extensions from cython.
    * ``'numba'`` : Runs the function through JIT compiled code from numba.
    * ``None`` : Defaults to ``'cython'`` or the global setting ``compute.use_numba``

    .. versionadded:: 1.1.0
engine_kwargs : dict, default None
    * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
    * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
      and ``parallel`` dictionary keys. The values must either be ``True`` or
      ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
      ``{'nopython': True, 'nogil': False, 'parallel': False}`` and will be
      applied to the function

    .. versionadded:: 1.1.0
**kwargs
    Keyword arguments to be passed into func.

Returns
-------
%(klass)s

See Also
--------
%(klass)s.groupby.apply : Apply function ``func`` group-wise and combine
    the results together.
%(klass)s.groupby.aggregate : Aggregate using one or more
    operations over the specified axis.
%(klass)s.transform : Call ``func`` on self producing a %(klass)s with
    transformed values.

Notes
-----
Each group is endowed the attribute 'name' in case you need to know
which group you are working on.

The current implementation imposes three requirements on f:

* f must return a value that either has the same shape as the input
  subframe or can be broadcast to the shape of the input subframe.
  For example, if `f` returns a scalar it will be broadcast to have the
  same shape as the input subframe.
* if this is a DataFrame, f must support application column-by-column
  in the subframe. If f also supports application to the entire subframe,
  then a fast path is used starting from the second chunk.
* f must not mutate groups. Mutation is not supported and may
  produce unexpected results. See :ref:`gotchas.udf-mutation` for more details.

When using ``engine='numba'``, there will be no "fall back" behavior internally.
The group data and group index will be passed as numpy arrays to the JITed
user defined function, and no alternative execution attempts will be tried.

.. versionchanged:: 1.3.0

    The resulting dtype will reflect the return value of the passed ``func``,
    see the examples below.

Examples
--------

>>> df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
...                           'foo', 'bar'],
...                    'B' : ['one', 'one', 'two', 'three',
...                           'two', 'two'],
...                    'C' : [1, 5, 5, 2, 5, 5],
...                    'D' : [2.0, 5., 8., 1., 2., 9.]})
>>> grouped = df.groupby('A')
>>> grouped.transform(lambda x: (x - x.mean()) / x.std())
          C         D
0 -1.154701 -0.577350
1  0.577350  0.000000
2  0.577350  1.154701
3 -1.154701 -1.000000
4  0.577350 -0.577350
5  0.577350  1.000000

Broadcast result of the transformation

>>> grouped.transform(lambda x: x.max() - x.min())
   C    D
0  4  6.0
1  3  8.0
2  4  6.0
3  3  8.0
4  4  6.0
5  3  8.0

.. versionchanged:: 1.3.0

    The resulting dtype will reflect the return value of the passed ``func``,
    for example:

>>> grouped[['C', 'D']].transform(lambda x: x.astype(int).max())
   C  D
0  5  8
1  5  9
2  5  8
3  5  9
4  5  8
5  5  9
"""

_agg_template = """
Aggregate using one or more operations over the specified axis.

Parameters
----------
func : function, str, list or dict
    Function to use for aggregating the data. If a function, must either
    work when passed a {klass} or when passed to {klass}.apply.

    Accepted combinations are:

    - function
    - string function name
    - list of functions and/or function names, e.g. ``[np.sum, 'mean']``
    - dict of axis labels -> functions, function names or list of such.

    Can also accept a Numba JIT function with
    ``engine='numba'`` specified. Only passing a single function is supported
    with this engine.

    If the ``'numba'`` engine is chosen, the function must be
    a user defined function with ``values`` and ``index`` as the
    first and second arguments respectively in the function signature.
    Each group's index will be passed to the user defined function
    and optionally available for use.

    .. versionchanged:: 1.1.0
*args
    Positional arguments to pass to func.
engine : str, default None
    * ``'cython'`` : Runs the function through C-extensions from cython.
    * ``'numba'`` : Runs the function through JIT compiled code from numba.
    * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``

    .. versionadded:: 1.1.0
engine_kwargs : dict, default None
    * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
    * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
      and ``parallel`` dictionary keys. The values must either be ``True`` or
      ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
      ``{{'nopython': True, 'nogil': False, 'parallel': False}}`` and will be
      applied to the function

    .. versionadded:: 1.1.0
**kwargs
    Keyword arguments to be passed into func.

Returns
-------
{klass}

See Also
--------
{klass}.groupby.apply : Apply function func group-wise
    and combine the results together.
{klass}.groupby.transform : Aggregate using one or more
    operations over the specified axis.
{klass}.aggregate : Transforms the Series on each group
    based on the given function.

Notes
-----
When using ``engine='numba'``, there will be no "fall back" behavior internally.
The group data and group index will be passed as numpy arrays to the JITed
user defined function, and no alternative execution attempts will be tried.

Functions that mutate the passed object can produce unexpected
behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
for more details.

.. versionchanged:: 1.3.0

    The resulting dtype will reflect the return value of the passed ``func``,
    see the examples below.
{examples}"""


@final
class GroupByPlot(PandasObject):
    """
    Class implementing the .plot attribute for groupby objects.
    """

    def __init__(self, groupby: GroupBy):
        self._groupby = groupby

    def __call__(self, *args, **kwargs):
        def f(self):
            return self.plot(*args, **kwargs)

        f.__name__ = "plot"
        return self._groupby.apply(f)

    def __getattr__(self, name: str):
        def attr(*args, **kwargs):
            def f(self):
                return getattr(self.plot, name)(*args, **kwargs)

            return self._groupby.apply(f)

        return attr


@contextmanager
def group_selection_context(groupby: GroupBy) -> Iterator[GroupBy]:
    """
    Set / reset the group_selection_context.
    """
    groupby._set_group_selection()
    try:
        yield groupby
    finally:
        groupby._reset_group_selection()


_KeysArgType = Union[
    Hashable,
    List[Hashable],
    Callable[[Hashable], Hashable],
    List[Callable[[Hashable], Hashable]],
    Mapping[Hashable, Hashable],
]


class BaseGroupBy(PandasObject, SelectionMixin[FrameOrSeries]):
    _group_selection: IndexLabel | None = None
    _apply_allowlist: frozenset[str] = frozenset()
    _hidden_attrs = PandasObject._hidden_attrs | {
        "as_index",
        "axis",
        "dropna",
        "exclusions",
        "grouper",
        "group_keys",
        "keys",
        "level",
        "mutated",
        "obj",
        "observed",
        "sort",
        "squeeze",
    }

    axis: int
    grouper: ops.BaseGrouper
    group_keys: bool

    @final
    def __len__(self) -> int:
        return len(self.groups)

    @final
    def __repr__(self) -> str:
        # TODO: Better repr for GroupBy object
        return object.__repr__(self)

    @final
    @property
    def groups(self) -> dict[Hashable, np.ndarray]:
        """
        Dict {group name -> group labels}.
        """
        return self.grouper.groups

    @final
    @property
    def ngroups(self) -> int:
        return self.grouper.ngroups

    @final
    @property
    def indices(self):
        """
        Dict {group name -> group indices}.
        """
        return self.grouper.indices

    @final
    def _get_indices(self, names):
        """
        Safe get multiple indices, translate keys for
        datelike to underlying repr.
        """

        def get_converter(s):
            # possibly convert to the actual key types
            # in the indices, could be a Timestamp or a np.datetime64
            if isinstance(s, datetime.datetime):
                return lambda key: Timestamp(key)
            elif isinstance(s, np.datetime64):
                return lambda key: Timestamp(key).asm8
            else:
                return lambda key: key

        if len(names) == 0:
            return []

        if len(self.indices) > 0:
            index_sample = next(iter(self.indices))
        else:
            index_sample = None  # Dummy sample

        name_sample = names[0]
        if isinstance(index_sample, tuple):
            if not isinstance(name_sample, tuple):
                msg = "must supply a tuple to get_group with multiple grouping keys"
                raise ValueError(msg)
            if not len(name_sample) == len(index_sample):
                try:
                    # If the original grouper was a tuple
                    return [self.indices[name] for name in names]
                except KeyError as err:
                    # turns out it wasn't a tuple
                    msg = (
                        "must supply a same-length tuple to get_group "
                        "with multiple grouping keys"
                    )
                    raise ValueError(msg) from err

            converters = [get_converter(s) for s in index_sample]
            names = (tuple(f(n) for f, n in zip(converters, name)) for name in names)

        else:
            converter = get_converter(index_sample)
            names = (converter(name) for name in names)

        return [self.indices.get(name, []) for name in names]

    @final
    def _get_index(self, name):
        """
        Safe get index, translate keys for datelike to underlying repr.
        """
        return self._get_indices([name])[0]

    @final
    @cache_readonly
    def _selected_obj(self):
        # Note: _selected_obj is always just `self.obj` for SeriesGroupBy

        if self._selection is None or isinstance(self.obj, Series):
            if self._group_selection is not None:
                return self.obj[self._group_selection]
            return self.obj
        else:
            return self.obj[self._selection]

    @final
    def _dir_additions(self) -> set[str]:
        return self.obj._dir_additions() | self._apply_allowlist

    @Substitution(
        klass="GroupBy",
        examples=dedent(
            """\
        >>> df = pd.DataFrame({'A': 'a b a b'.split(), 'B': [1, 2, 3, 4]})
        >>> df
           A  B
        0  a  1
        1  b  2
        2  a  3
        3  b  4

        To get the difference between each groups maximum and minimum value in one
        pass, you can do

        >>> df.groupby('A').pipe(lambda x: x.max() - x.min())
           B
        A
        a  2
        b  2"""
        ),
    )
    @Appender(_pipe_template)
    def pipe(
        self,
        func: Callable[..., T] | tuple[Callable[..., T], str],
        *args,
        **kwargs,
    ) -> T:
        return com.pipe(self, func, *args, **kwargs)

    plot = property(GroupByPlot)

    @final
    def get_group(self, name, obj=None) -> FrameOrSeriesUnion:
        """
        Construct DataFrame from group with provided name.

        Parameters
        ----------
        name : object
            The name of the group to get as a DataFrame.
        obj : DataFrame, default None
            The DataFrame to take the DataFrame out of.  If
            it is None, the object groupby was called on will
            be used.

        Returns
        -------
        group : same type as obj
        """
        if obj is None:
            obj = self._selected_obj

        inds = self._get_index(name)
        if not len(inds):
            raise KeyError(name)

        return obj._take_with_is_copy(inds, axis=self.axis)

    @final
    def __iter__(self) -> Iterator[tuple[Hashable, FrameOrSeries]]:
        """
        Groupby iterator.

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """
        return self.grouper.get_iterator(self.obj, axis=self.axis)


# To track operations that expand dimensions, like ohlc
OutputFrameOrSeries = TypeVar("OutputFrameOrSeries", bound=NDFrame)


class GroupBy(BaseGroupBy[FrameOrSeries]):
    """
    Class for grouping and aggregating relational data.

    See aggregate, transform, and apply functions on this object.

    It's easiest to use obj.groupby(...) to use GroupBy, but you can also do:

    ::

        grouped = groupby(obj, ...)

    Parameters
    ----------
    obj : pandas object
    axis : int, default 0
    level : int, default None
        Level of MultiIndex
    groupings : list of Grouping objects
        Most users should ignore this
    exclusions : array-like, optional
        List of columns to exclude
    name : str
        Most users should ignore this

    Returns
    -------
    **Attributes**
    groups : dict
        {group name -> group labels}
    len(grouped) : int
        Number of groups

    Notes
    -----
    After grouping, see aggregate, apply, and transform functions. Here are
    some other brief notes about usage. When grouping by multiple groups, the
    result index will be a MultiIndex (hierarchical) by default.

    Iteration produces (key, group) tuples, i.e. chunking the data by group. So
    you can write code like:

    ::

        grouped = obj.groupby(keys, axis=axis)
        for key, group in grouped:
            # do something with the data

    Function calls on GroupBy, if not specially implemented, "dispatch" to the
    grouped data. So if you group a DataFrame and wish to invoke the std()
    method on each group, you can simply do:

    ::

        df.groupby(mapper).std()

    rather than

    ::

        df.groupby(mapper).aggregate(np.std)

    You can pass arguments to these "wrapped" functions, too.

    See the online documentation for full exposition on these topics and much
    more
    """

    grouper: ops.BaseGrouper
    as_index: bool

    @final
    def __init__(
        self,
        obj: FrameOrSeries,
        keys: _KeysArgType | None = None,
        axis: int = 0,
        level: IndexLabel | None = None,
        grouper: ops.BaseGrouper | None = None,
        exclusions: frozenset[Hashable] | None = None,
        selection: IndexLabel | None = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        squeeze: bool = False,
        observed: bool = False,
        mutated: bool = False,
        dropna: bool = True,
    ):

        self._selection = selection

        assert isinstance(obj, NDFrame), type(obj)

        self.level = level

        if not as_index:
            if not isinstance(obj, DataFrame):
                raise TypeError("as_index=False only valid with DataFrame")
            if axis != 0:
                raise ValueError("as_index=False only valid for axis=0")

        self.as_index = as_index
        self.keys = keys
        self.sort = sort
        self.group_keys = group_keys
        self.squeeze = squeeze
        self.observed = observed
        self.mutated = mutated
        self.dropna = dropna

        if grouper is None:
            from pandas.core.groupby.grouper import get_grouper

            grouper, exclusions, obj = get_grouper(
                obj,
                keys,
                axis=axis,
                level=level,
                sort=sort,
                observed=observed,
                mutated=self.mutated,
                dropna=self.dropna,
            )

        self.obj = obj
        self.axis = obj._get_axis_number(axis)
        self.grouper = grouper
        self.exclusions = frozenset(exclusions) if exclusions else frozenset()

    def __getattr__(self, attr: str):
        if attr in self._internal_names_set:
            return object.__getattribute__(self, attr)
        if attr in self.obj:
            return self[attr]

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    @final
    def _make_wrapper(self, name: str) -> Callable:
        assert name in self._apply_allowlist

        with group_selection_context(self):
            # need to setup the selection
            # as are not passed directly but in the grouper
            f = getattr(self._obj_with_exclusions, name)
            if not isinstance(f, types.MethodType):
                return self.apply(lambda self: getattr(self, name))

        f = getattr(type(self._obj_with_exclusions), name)
        sig = inspect.signature(f)

        def wrapper(*args, **kwargs):
            # a little trickery for aggregation functions that need an axis
            # argument
            if "axis" in sig.parameters:
                if kwargs.get("axis", None) is None:
                    kwargs["axis"] = self.axis

            def curried(x):
                return f(x, *args, **kwargs)

            # preserve the name so we can detect it when calling plot methods,
            # to avoid duplicates
            curried.__name__ = name

            # special case otherwise extra plots are created when catching the
            # exception below
            if name in base.plotting_methods:
                return self.apply(curried)

            return self._python_apply_general(curried, self._obj_with_exclusions)

        wrapper.__name__ = name
        return wrapper

    # -----------------------------------------------------------------
    # Selection

    @final
    def _set_group_selection(self) -> None:
        """
        Create group based selection.

        Used when selection is not passed directly but instead via a grouper.

        NOTE: this should be paired with a call to _reset_group_selection
        """
        # This is a no-op for SeriesGroupBy
        grp = self.grouper
        if not (
            self.as_index
            and grp.groupings is not None
            and self.obj.ndim > 1
            and self._group_selection is None
        ):
            return

        groupers = [g.name for g in grp.groupings if g.level is None and g.in_axis]

        if len(groupers):
            # GH12839 clear selected obj cache when group selection changes
            ax = self.obj._info_axis
            self._group_selection = ax.difference(Index(groupers), sort=False).tolist()
            self._reset_cache("_selected_obj")

    @final
    def _reset_group_selection(self) -> None:
        """
        Clear group based selection.

        Used for methods needing to return info on each group regardless of
        whether a group selection was previously set.
        """
        if self._group_selection is not None:
            # GH12839 clear cached selection too when changing group selection
            self._group_selection = None
            self._reset_cache("_selected_obj")

    def _iterate_slices(self) -> Iterable[Series]:
        raise AbstractMethodError(self)

    # -----------------------------------------------------------------
    # Dispatch/Wrapping

    @final
    def _concat_objects(self, keys, values, not_indexed_same: bool = False):
        from pandas.core.reshape.concat import concat

        def reset_identity(values):
            # reset the identities of the components
            # of the values to prevent aliasing
            for v in com.not_none(*values):
                ax = v._get_axis(self.axis)
                ax._reset_identity()
            return values

        if not not_indexed_same:
            result = concat(values, axis=self.axis)
            ax = (
                self.filter(lambda x: True).axes[self.axis]
                if self.dropna
                else self._selected_obj._get_axis(self.axis)
            )

            # this is a very unfortunate situation
            # we can't use reindex to restore the original order
            # when the ax has duplicates
            # so we resort to this
            # GH 14776, 30667
            if ax.has_duplicates and not result.axes[self.axis].equals(ax):
                indexer, _ = result.index.get_indexer_non_unique(ax._values)
                indexer = algorithms.unique1d(indexer)
                result = result.take(indexer, axis=self.axis)
            else:
                result = result.reindex(ax, axis=self.axis, copy=False)

        elif self.group_keys:

            values = reset_identity(values)
            if self.as_index:

                # possible MI return case
                group_keys = keys
                group_levels = self.grouper.levels
                group_names = self.grouper.names

                result = concat(
                    values,
                    axis=self.axis,
                    keys=group_keys,
                    levels=group_levels,
                    names=group_names,
                    sort=False,
                )
            else:

                # GH5610, returns a MI, with the first level being a
                # range index
                keys = list(range(len(values)))
                result = concat(values, axis=self.axis, keys=keys)
        else:
            values = reset_identity(values)
            result = concat(values, axis=self.axis)

        name = self.obj.name if self.obj.ndim == 1 else self._selection
        if isinstance(result, Series) and name is not None:

            result.name = name

        return result

    @final
    def _set_result_index_ordered(
        self, result: OutputFrameOrSeries
    ) -> OutputFrameOrSeries:
        # set the result index on the passed values object and
        # return the new object, xref 8046

        if self.grouper.is_monotonic:
            # shortcut if we have an already ordered grouper
            result.set_axis(self.obj._get_axis(self.axis), axis=self.axis, inplace=True)
            return result

        # row order is scrambled => sort the rows by position in original index
        original_positions = Index(
            np.concatenate(self._get_indices(self.grouper.result_index))
        )
        result.set_axis(original_positions, axis=self.axis, inplace=True)
        result = result.sort_index(axis=self.axis)

        dropped_rows = len(result.index) < len(self.obj.index)

        if dropped_rows:
            # get index by slicing original index according to original positions
            # slice drops attrs => use set_axis when no rows were dropped
            sorted_indexer = result.index
            result.index = self._selected_obj.index[sorted_indexer]
        else:
            result.set_axis(self.obj._get_axis(self.axis), axis=self.axis, inplace=True)

        return result

    def _wrap_aggregated_output(self, output: Mapping[base.OutputKey, ArrayLike]):
        raise AbstractMethodError(self)

    def _wrap_transformed_output(self, output: Mapping[base.OutputKey, ArrayLike]):
        raise AbstractMethodError(self)

    def _wrap_applied_output(self, data, keys, values, not_indexed_same: bool = False):
        raise AbstractMethodError(self)

    def _resolve_numeric_only(self, numeric_only: bool | lib.NoDefault) -> bool:
        """
        Determine subclass-specific default value for 'numeric_only'.

        For SeriesGroupBy we want the default to be False (to match Series behavior).
        For DataFrameGroupBy we want it to be True (for backwards-compat).

        Parameters
        ----------
        numeric_only : bool or lib.no_default

        Returns
        -------
        bool
        """
        # GH#41291
        if numeric_only is lib.no_default:
            # i.e. not explicitly passed by user
            if self.obj.ndim == 2:
                # i.e. DataFrameGroupBy
                numeric_only = True
                # GH#42395 GH#43108 GH#43154
                # Regression from 1.2.5 to 1.3 caused object columns to be dropped
                if self.axis:
                    obj = self._obj_with_exclusions.T
                else:
                    obj = self._obj_with_exclusions
                check = obj._get_numeric_data()
                if len(obj.columns) and not len(check.columns) and not obj.empty:
                    numeric_only = False
                    # TODO: v1.4+ Add FutureWarning

            else:
                numeric_only = False

        # error: Incompatible return value type (got "Union[bool, NoDefault]",
        # expected "bool")
        return numeric_only  # type: ignore[return-value]

    # -----------------------------------------------------------------
    # numba

    @final
    def _numba_prep(self, func, data):
        if not callable(func):
            raise NotImplementedError(
                "Numba engine can only be used with a single function."
            )
        ids, _, ngroups = self.grouper.group_info
        sorted_index = get_group_index_sorter(ids, ngroups)
        sorted_ids = algorithms.take_nd(ids, sorted_index, allow_fill=False)

        sorted_data = data.take(sorted_index, axis=self.axis).to_numpy()
        sorted_index_data = data.index.take(sorted_index).to_numpy()

        starts, ends = lib.generate_slices(sorted_ids, ngroups)
        return (
            starts,
            ends,
            sorted_index_data,
            sorted_data,
        )

    @final
    def _transform_with_numba(self, data, func, *args, engine_kwargs=None, **kwargs):
        """
        Perform groupby transform routine with the numba engine.

        This routine mimics the data splitting routine of the DataSplitter class
        to generate the indices of each group in the sorted data and then passes the
        data and indices into a Numba jitted function.
        """
        starts, ends, sorted_index, sorted_data = self._numba_prep(func, data)
        group_keys = self.grouper._get_group_keys()

        numba_transform_func = numba_.generate_numba_transform_func(
            kwargs, func, engine_kwargs
        )
        result = numba_transform_func(
            sorted_data,
            sorted_index,
            starts,
            ends,
            len(group_keys),
            len(data.columns),
            *args,
        )

        cache_key = (func, "groupby_transform")
        if cache_key not in NUMBA_FUNC_CACHE:
            NUMBA_FUNC_CACHE[cache_key] = numba_transform_func

        # result values needs to be resorted to their original positions since we
        # evaluated the data sorted by group
        return result.take(np.argsort(sorted_index), axis=0)

    @final
    def _aggregate_with_numba(self, data, func, *args, engine_kwargs=None, **kwargs):
        """
        Perform groupby aggregation routine with the numba engine.

        This routine mimics the data splitting routine of the DataSplitter class
        to generate the indices of each group in the sorted data and then passes the
        data and indices into a Numba jitted function.
        """
        starts, ends, sorted_index, sorted_data = self._numba_prep(func, data)
        group_keys = self.grouper._get_group_keys()

        numba_agg_func = numba_.generate_numba_agg_func(kwargs, func, engine_kwargs)
        result = numba_agg_func(
            sorted_data,
            sorted_index,
            starts,
            ends,
            len(group_keys),
            len(data.columns),
            *args,
        )

        cache_key = (func, "groupby_agg")
        if cache_key not in NUMBA_FUNC_CACHE:
            NUMBA_FUNC_CACHE[cache_key] = numba_agg_func

        if self.grouper.nkeys > 1:
            index = MultiIndex.from_tuples(group_keys, names=self.grouper.names)
        else:
            index = Index(group_keys, name=self.grouper.names[0])
        return result, index

    # -----------------------------------------------------------------
    # apply/agg/transform

    @Appender(
        _apply_docs["template"].format(
            input="dataframe", examples=_apply_docs["dataframe_examples"]
        )
    )
    def apply(self, func, *args, **kwargs):

        func = com.is_builtin_func(func)

        # this is needed so we don't try and wrap strings. If we could
        # resolve functions to their callable functions prior, this
        # wouldn't be needed
        if args or kwargs:
            if callable(func):

                @wraps(func)
                def f(g):
                    with np.errstate(all="ignore"):
                        return func(g, *args, **kwargs)

            elif hasattr(nanops, "nan" + func):
                # TODO: should we wrap this in to e.g. _is_builtin_func?
                f = getattr(nanops, "nan" + func)

            else:
                raise ValueError(
                    "func must be a callable if args or kwargs are supplied"
                )
        else:
            f = func

        # ignore SettingWithCopy here in case the user mutates
        with option_context("mode.chained_assignment", None):
            try:
                result = self._python_apply_general(f, self._selected_obj)
            except TypeError:
                # gh-20949
                # try again, with .apply acting as a filtering
                # operation, by excluding the grouping column
                # This would normally not be triggered
                # except if the udf is trying an operation that
                # fails on *some* columns, e.g. a numeric operation
                # on a string grouper column

                with group_selection_context(self):
                    return self._python_apply_general(f, self._selected_obj)

        return result

    @final
    def _python_apply_general(
        self, f: F, data: FrameOrSeriesUnion
    ) -> FrameOrSeriesUnion:
        """
        Apply function f in python space

        Parameters
        ----------
        f : callable
            Function to apply
        data : Series or DataFrame
            Data to apply f to

        Returns
        -------
        Series or DataFrame
            data after applying f
        """
        keys, values, mutated = self.grouper.apply(f, data, self.axis)

        return self._wrap_applied_output(
            data, keys, values, not_indexed_same=mutated or self.mutated
        )

    @final
    def _python_agg_general(self, func, *args, **kwargs):
        func = com.is_builtin_func(func)
        f = lambda x: func(x, *args, **kwargs)

        # iterate through "columns" ex exclusions to populate output dict
        output: dict[base.OutputKey, ArrayLike] = {}

        if self.ngroups == 0:
            # agg_series below assumes ngroups > 0
            return self._python_apply_general(f, self._selected_obj)

        for idx, obj in enumerate(self._iterate_slices()):
            name = obj.name

            try:
                # if this function is invalid for this dtype, we will ignore it.
                result = self.grouper.agg_series(obj, f)
            except TypeError:
                warnings.warn(
                    f"Dropping invalid columns in {type(self).__name__}.agg "
                    "is deprecated. In a future version, a TypeError will be raised. "
                    "Before calling .agg, select only columns which should be "
                    "valid for the aggregating function.",
                    FutureWarning,
                    stacklevel=3,
                )
                continue

            key = base.OutputKey(label=name, position=idx)
            output[key] = result

        if not output:
            return self._python_apply_general(f, self._selected_obj)

        return self._wrap_aggregated_output(output)

    @final
    def _agg_general(
        self,
        numeric_only: bool = True,
        min_count: int = -1,
        *,
        alias: str,
        npfunc: Callable,
    ):

        with group_selection_context(self):
            # try a cython aggregation if we can
            result = self._cython_agg_general(
                how=alias,
                alt=npfunc,
                numeric_only=numeric_only,
                min_count=min_count,
            )
            return result.__finalize__(self.obj, method="groupby")

    def _agg_py_fallback(
        self, values: ArrayLike, ndim: int, alt: Callable
    ) -> ArrayLike:
        """
        Fallback to pure-python aggregation if _cython_operation raises
        NotImplementedError.
        """
        # We get here with a) EADtypes and b) object dtype

        if values.ndim == 1:
            # For DataFrameGroupBy we only get here with ExtensionArray
            ser = Series(values)
        else:
            # We only get here with values.dtype == object
            # TODO: special case not needed with ArrayManager
            df = DataFrame(values.T)
            # bc we split object blocks in grouped_reduce, we have only 1 col
            # otherwise we'd have to worry about block-splitting GH#39329
            assert df.shape[1] == 1
            # Avoid call to self.values that can occur in DataFrame
            #  reductions; see GH#28949
            ser = df.iloc[:, 0]

        # We do not get here with UDFs, so we know that our dtype
        #  should always be preserved by the implemented aggregations
        # TODO: Is this exactly right; see WrappedCythonOp get_result_dtype?
        res_values = self.grouper.agg_series(ser, alt, preserve_dtype=True)

        if isinstance(values, Categorical):
            # Because we only get here with known dtype-preserving
            #  reductions, we cast back to Categorical.
            # TODO: if we ever get "rank" working, exclude it here.
            res_values = type(values)._from_sequence(res_values, dtype=values.dtype)

        # If we are DataFrameGroupBy and went through a SeriesGroupByPath
        # then we need to reshape
        # GH#32223 includes case with IntegerArray values, ndarray res_values
        # test_groupby_duplicate_columns with object dtype values
        return ensure_block_shape(res_values, ndim=ndim)

    def _cython_agg_general(
        self, how: str, alt: Callable, numeric_only: bool, min_count: int = -1
    ):
        raise AbstractMethodError(self)

    def _cython_transform(
        self, how: str, numeric_only: bool = True, axis: int = 0, **kwargs
    ):
        raise AbstractMethodError(self)

    @final
    def _transform(self, func, *args, engine=None, engine_kwargs=None, **kwargs):

        if maybe_use_numba(engine):
            # TODO: tests with self._selected_obj.ndim == 1 on DataFrameGroupBy
            with group_selection_context(self):
                data = self._selected_obj
            df = data if data.ndim == 2 else data.to_frame()
            result = self._transform_with_numba(
                df, func, *args, engine_kwargs=engine_kwargs, **kwargs
            )
            if self.obj.ndim == 2:
                return cast(DataFrame, self.obj)._constructor(
                    result, index=data.index, columns=data.columns
                )
            else:
                return cast(Series, self.obj)._constructor(
                    result.ravel(), index=data.index, name=data.name
                )

        # optimized transforms
        func = com.get_cython_func(func) or func

        if not isinstance(func, str):
            return self._transform_general(func, *args, **kwargs)

        elif func not in base.transform_kernel_allowlist:
            msg = f"'{func}' is not a valid function name for transform(name)"
            raise ValueError(msg)
        elif func in base.cythonized_kernels or func in base.transformation_kernels:
            # cythonized transform or canned "agg+broadcast"
            return getattr(self, func)(*args, **kwargs)

        else:
            # i.e. func in base.reduction_kernels

            # GH#30918 Use _transform_fast only when we know func is an aggregation
            # If func is a reduction, we need to broadcast the
            # result to the whole group. Compute func result
            # and deal with possible broadcasting below.
            # Temporarily set observed for dealing with categoricals.
            with com.temp_setattr(self, "observed", True):
                result = getattr(self, func)(*args, **kwargs)

            if self._can_use_transform_fast(result):
                return self._wrap_transform_fast_result(result)

            # only reached for DataFrameGroupBy
            return self._transform_general(func, *args, **kwargs)

    # -----------------------------------------------------------------
    # Utilities

    @final
    def _apply_filter(self, indices, dropna):
        if len(indices) == 0:
            indices = np.array([], dtype="int64")
        else:
            indices = np.sort(np.concatenate(indices))
        if dropna:
            filtered = self._selected_obj.take(indices, axis=self.axis)
        else:
            mask = np.empty(len(self._selected_obj.index), dtype=bool)
            mask.fill(False)
            mask[indices.astype(int)] = True
            # mask fails to broadcast when passed to where; broadcast manually.
            mask = np.tile(mask, list(self._selected_obj.shape[1:]) + [1]).T
            filtered = self._selected_obj.where(mask)  # Fill with NaNs.
        return filtered

    @final
    def _cumcount_array(self, ascending: bool = True) -> np.ndarray:
        """
        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from length of group - 1 to 0.

        Notes
        -----
        this is currently implementing sort=False
        (though the default is sort=True) for groupby in general
        """
        ids, _, ngroups = self.grouper.group_info
        sorter = get_group_index_sorter(ids, ngroups)
        ids, count = ids[sorter], len(ids)

        if count == 0:
            return np.empty(0, dtype=np.int64)

        run = np.r_[True, ids[:-1] != ids[1:]]
        rep = np.diff(np.r_[np.nonzero(run)[0], count])
        out = (~run).cumsum()

        if ascending:
            out -= np.repeat(out[run], rep)
        else:
            out = np.repeat(out[np.r_[run[1:], True]], rep) - out

        rev = np.empty(count, dtype=np.intp)
        rev[sorter] = np.arange(count, dtype=np.intp)
        return out[rev].astype(np.int64, copy=False)

    # -----------------------------------------------------------------

    @final
    @property
    def _obj_1d_constructor(self) -> type[Series]:
        # GH28330 preserve subclassed Series/DataFrames
        if isinstance(self.obj, DataFrame):
            return self.obj._constructor_sliced
        assert isinstance(self.obj, Series)
        return self.obj._constructor

    @final
    def _bool_agg(self, val_test, skipna):
        """
        Shared func to call any / all Cython GroupBy implementations.
        """

        def objs_to_bool(vals: ArrayLike) -> tuple[np.ndarray, type]:
            if is_object_dtype(vals):
                # GH#37501: don't raise on pd.NA when skipna=True
                if skipna:
                    vals = np.array([bool(x) if not isna(x) else True for x in vals])
                else:
                    vals = np.array([bool(x) for x in vals])
            elif isinstance(vals, BaseMaskedArray):
                vals = vals._data.astype(bool, copy=False)
            else:
                vals = vals.astype(bool)

            return vals.view(np.int8), bool

        def result_to_bool(
            result: np.ndarray,
            inference: type,
            nullable: bool = False,
        ) -> ArrayLike:
            if nullable:
                return BooleanArray(result.astype(bool, copy=False), result == -1)
            else:
                return result.astype(inference, copy=False)

        return self._get_cythonized_result(
            "group_any_all",
            aggregate=True,
            numeric_only=False,
            cython_dtype=np.dtype(np.int8),
            needs_values=True,
            needs_mask=True,
            needs_nullable=True,
            pre_processing=objs_to_bool,
            post_processing=result_to_bool,
            val_test=val_test,
            skipna=skipna,
        )

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def any(self, skipna: bool = True):
        """
        Return True if any value in the group is truthful, else False.

        Parameters
        ----------
        skipna : bool, default True
            Flag to ignore nan values during truth testing.

        Returns
        -------
        Series or DataFrame
            DataFrame or Series of boolean values, where a value is True if any element
            is True within its respective group, False otherwise.
        """
        return self._bool_agg("any", skipna)

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def all(self, skipna: bool = True):
        """
        Return True if all values in the group are truthful, else False.

        Parameters
        ----------
        skipna : bool, default True
            Flag to ignore nan values during truth testing.

        Returns
        -------
        Series or DataFrame
            DataFrame or Series of boolean values, where a value is True if all elements
            are True within its respective group, False otherwise.
        """
        return self._bool_agg("all", skipna)

    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def count(self):
        """
        Compute count of group, excluding missing values.

        Returns
        -------
        Series or DataFrame
            Count of values within each group.
        """
        # defined here for API doc
        raise NotImplementedError

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def mean(self, numeric_only: bool | lib.NoDefault = lib.no_default):
        """
        Compute mean of groups, excluding missing values.

        Parameters
        ----------
        numeric_only : bool, default True
            Include only float, int, boolean columns. If None, will attempt to use
            everything, then use only numeric data.

        Returns
        -------
        pandas.Series or pandas.DataFrame
        %(see_also)s
        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 1, 2, 1, 2],
        ...                    'B': [np.nan, 2, 3, 4, 5],
        ...                    'C': [1, 2, 1, 1, 2]}, columns=['A', 'B', 'C'])

        Groupby one column and return the mean of the remaining columns in
        each group.

        >>> df.groupby('A').mean()
             B         C
        A
        1  3.0  1.333333
        2  4.0  1.500000

        Groupby two columns and return the mean of the remaining column.

        >>> df.groupby(['A', 'B']).mean()
                 C
        A B
        1 2.0  2.0
          4.0  1.0
        2 3.0  1.0
          5.0  2.0

        Groupby one column and return the mean of only particular column in
        the group.

        >>> df.groupby('A')['B'].mean()
        A
        1    3.0
        2    4.0
        Name: B, dtype: float64
        """
        numeric_only = self._resolve_numeric_only(numeric_only)

        result = self._cython_agg_general(
            "mean",
            alt=lambda x: Series(x).mean(numeric_only=numeric_only),
            numeric_only=numeric_only,
        )
        return result.__finalize__(self.obj, method="groupby")

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def median(self, numeric_only: bool | lib.NoDefault = lib.no_default):
        """
        Compute median of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex

        Parameters
        ----------
        numeric_only : bool, default True
            Include only float, int, boolean columns. If None, will attempt to use
            everything, then use only numeric data.

        Returns
        -------
        Series or DataFrame
            Median of values within each group.
        """
        numeric_only = self._resolve_numeric_only(numeric_only)

        result = self._cython_agg_general(
            "median",
            alt=lambda x: Series(x).median(numeric_only=numeric_only),
            numeric_only=numeric_only,
        )
        return result.__finalize__(self.obj, method="groupby")

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def std(self, ddof: int = 1):
        """
        Compute standard deviation of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        Returns
        -------
        Series or DataFrame
            Standard deviation of values within each group.
        """
        return self._get_cythonized_result(
            "group_var",
            aggregate=True,
            needs_counts=True,
            needs_values=True,
            needs_2d=True,
            cython_dtype=np.dtype(np.float64),
            post_processing=lambda vals, inference: np.sqrt(vals),
            ddof=ddof,
        )

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def var(self, ddof: int = 1):
        """
        Compute variance of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        Returns
        -------
        Series or DataFrame
            Variance of values within each group.
        """
        if ddof == 1:
            numeric_only = self._resolve_numeric_only(lib.no_default)
            return self._cython_agg_general(
                "var", alt=lambda x: Series(x).var(ddof=ddof), numeric_only=numeric_only
            )
        else:
            func = lambda x: x.var(ddof=ddof)
            with group_selection_context(self):
                return self._python_agg_general(func)

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def sem(self, ddof: int = 1):
        """
        Compute standard error of the mean of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        Returns
        -------
        Series or DataFrame
            Standard error of the mean of values within each group.
        """
        result = self.std(ddof=ddof)
        if result.ndim == 1:
            result /= np.sqrt(self.count())
        else:
            cols = result.columns.difference(self.exclusions).unique()
            counts = self.count()
            result_ilocs = result.columns.get_indexer_for(cols)
            count_ilocs = counts.columns.get_indexer_for(cols)
            result.iloc[:, result_ilocs] /= np.sqrt(counts.iloc[:, count_ilocs])
        return result

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def size(self) -> FrameOrSeriesUnion:
        """
        Compute group sizes.

        Returns
        -------
        DataFrame or Series
            Number of rows in each group as a Series if as_index is True
            or a DataFrame if as_index is False.
        """
        result = self.grouper.size()

        # GH28330 preserve subclassed Series/DataFrames through calls
        if issubclass(self.obj._constructor, Series):
            result = self._obj_1d_constructor(result, name=self.obj.name)
        else:
            result = self._obj_1d_constructor(result)

        if not self.as_index:
            result = result.rename("size").reset_index()

        return self._reindex_output(result, fill_value=0)

    @final
    @doc(_groupby_agg_method_template, fname="sum", no=True, mc=0)
    def sum(
        self, numeric_only: bool | lib.NoDefault = lib.no_default, min_count: int = 0
    ):
        numeric_only = self._resolve_numeric_only(numeric_only)

        # If we are grouping on categoricals we want unobserved categories to
        # return zero, rather than the default of NaN which the reindexing in
        # _agg_general() returns. GH #31422
        with com.temp_setattr(self, "observed", True):
            result = self._agg_general(
                numeric_only=numeric_only,
                min_count=min_count,
                alias="add",
                npfunc=np.sum,
            )

        return self._reindex_output(result, fill_value=0)

    @final
    @doc(_groupby_agg_method_template, fname="prod", no=True, mc=0)
    def prod(
        self, numeric_only: bool | lib.NoDefault = lib.no_default, min_count: int = 0
    ):
        numeric_only = self._resolve_numeric_only(numeric_only)

        return self._agg_general(
            numeric_only=numeric_only, min_count=min_count, alias="prod", npfunc=np.prod
        )

    @final
    @doc(_groupby_agg_method_template, fname="min", no=False, mc=-1)
    def min(self, numeric_only: bool = False, min_count: int = -1):
        return self._agg_general(
            numeric_only=numeric_only, min_count=min_count, alias="min", npfunc=np.min
        )

    @final
    @doc(_groupby_agg_method_template, fname="max", no=False, mc=-1)
    def max(self, numeric_only: bool = False, min_count: int = -1):
        return self._agg_general(
            numeric_only=numeric_only, min_count=min_count, alias="max", npfunc=np.max
        )

    @final
    @doc(_groupby_agg_method_template, fname="first", no=False, mc=-1)
    def first(self, numeric_only: bool = False, min_count: int = -1):
        def first_compat(obj: FrameOrSeries, axis: int = 0):
            def first(x: Series):
                """Helper function for first item that isn't NA."""
                arr = x.array[notna(x.array)]
                if not len(arr):
                    return np.nan
                return arr[0]

            if isinstance(obj, DataFrame):
                return obj.apply(first, axis=axis)
            elif isinstance(obj, Series):
                return first(obj)
            else:  # pragma: no cover
                raise TypeError(type(obj))

        return self._agg_general(
            numeric_only=numeric_only,
            min_count=min_count,
            alias="first",
            npfunc=first_compat,
        )

    @final
    @doc(_groupby_agg_method_template, fname="last", no=False, mc=-1)
    def last(self, numeric_only: bool = False, min_count: int = -1):
        def last_compat(obj: FrameOrSeries, axis: int = 0):
            def last(x: Series):
                """Helper function for last item that isn't NA."""
                arr = x.array[notna(x.array)]
                if not len(arr):
                    return np.nan
                return arr[-1]

            if isinstance(obj, DataFrame):
                return obj.apply(last, axis=axis)
            elif isinstance(obj, Series):
                return last(obj)
            else:  # pragma: no cover
                raise TypeError(type(obj))

        return self._agg_general(
            numeric_only=numeric_only,
            min_count=min_count,
            alias="last",
            npfunc=last_compat,
        )

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def ohlc(self) -> DataFrame:
        """
        Compute open, high, low and close values of a group, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex

        Returns
        -------
        DataFrame
            Open, high, low and close values within each group.
        """
        if self.obj.ndim == 1:
            # self._iterate_slices() yields only self._selected_obj
            obj = self._selected_obj

            is_numeric = is_numeric_dtype(obj.dtype)
            if not is_numeric:
                raise DataError("No numeric types to aggregate")

            res_values = self.grouper._cython_operation(
                "aggregate", obj._values, "ohlc", axis=0, min_count=-1
            )

            agg_names = ["open", "high", "low", "close"]
            result = self.obj._constructor_expanddim(
                res_values, index=self.grouper.result_index, columns=agg_names
            )
            return self._reindex_output(result)

        return self._apply_to_column_groupbys(
            lambda x: x.ohlc(), self._obj_with_exclusions
        )

    @final
    @doc(DataFrame.describe)
    def describe(self, **kwargs):
        with group_selection_context(self):
            result = self.apply(lambda x: x.describe(**kwargs))
            if self.axis == 1:
                return result.T
            return result.unstack()

    @final
    def resample(self, rule, *args, **kwargs):
        """
        Provide resampling when using a TimeGrouper.

        Given a grouper, the function resamples it according to a string
        "string" -> "frequency".

        See the :ref:`frequency aliases <timeseries.offset_aliases>`
        documentation for more details.

        Parameters
        ----------
        rule : str or DateOffset
            The offset string or object representing target grouper conversion.
        *args, **kwargs
            Possible arguments are `how`, `fill_method`, `limit`, `kind` and
            `on`, and other arguments of `TimeGrouper`.

        Returns
        -------
        Grouper
            Return a new grouper with our resampler appended.

        See Also
        --------
        Grouper : Specify a frequency to resample with when
            grouping by a key.
        DatetimeIndex.resample : Frequency conversion and resampling of
            time series.

        Examples
        --------
        >>> idx = pd.date_range('1/1/2000', periods=4, freq='T')
        >>> df = pd.DataFrame(data=4 * [range(2)],
        ...                   index=idx,
        ...                   columns=['a', 'b'])
        >>> df.iloc[2, 0] = 5
        >>> df
                            a  b
        2000-01-01 00:00:00  0  1
        2000-01-01 00:01:00  0  1
        2000-01-01 00:02:00  5  1
        2000-01-01 00:03:00  0  1

        Downsample the DataFrame into 3 minute bins and sum the values of
        the timestamps falling into a bin.

        >>> df.groupby('a').resample('3T').sum()
                                 a  b
        a
        0   2000-01-01 00:00:00  0  2
            2000-01-01 00:03:00  0  1
        5   2000-01-01 00:00:00  5  1

        Upsample the series into 30 second bins.

        >>> df.groupby('a').resample('30S').sum()
                            a  b
        a
        0   2000-01-01 00:00:00  0  1
            2000-01-01 00:00:30  0  0
            2000-01-01 00:01:00  0  1
            2000-01-01 00:01:30  0  0
            2000-01-01 00:02:00  0  0
            2000-01-01 00:02:30  0  0
            2000-01-01 00:03:00  0  1
        5   2000-01-01 00:02:00  5  1

        Resample by month. Values are assigned to the month of the period.

        >>> df.groupby('a').resample('M').sum()
                    a  b
        a
        0   2000-01-31  0  3
        5   2000-01-31  5  1

        Downsample the series into 3 minute bins as above, but close the right
        side of the bin interval.

        >>> df.groupby('a').resample('3T', closed='right').sum()
                                 a  b
        a
        0   1999-12-31 23:57:00  0  1
            2000-01-01 00:00:00  0  2
        5   2000-01-01 00:00:00  5  1

        Downsample the series into 3 minute bins and close the right side of
        the bin interval, but label each bin using the right edge instead of
        the left.

        >>> df.groupby('a').resample('3T', closed='right', label='right').sum()
                                 a  b
        a
        0   2000-01-01 00:00:00  0  1
            2000-01-01 00:03:00  0  2
        5   2000-01-01 00:03:00  5  1
        """
        from pandas.core.resample import get_resampler_for_grouping

        return get_resampler_for_grouping(self, rule, *args, **kwargs)

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def rolling(self, *args, **kwargs):
        """
        Return a rolling grouper, providing rolling functionality per group.
        """
        from pandas.core.window import RollingGroupby

        return RollingGroupby(
            self._selected_obj,
            *args,
            _grouper=self.grouper,
            _as_index=self.as_index,
            **kwargs,
        )

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def expanding(self, *args, **kwargs):
        """
        Return an expanding grouper, providing expanding
        functionality per group.
        """
        from pandas.core.window import ExpandingGroupby

        return ExpandingGroupby(
            self._selected_obj,
            *args,
            _grouper=self.grouper,
            **kwargs,
        )

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def ewm(self, *args, **kwargs):
        """
        Return an ewm grouper, providing ewm functionality per group.
        """
        from pandas.core.window import ExponentialMovingWindowGroupby

        return ExponentialMovingWindowGroupby(
            self._selected_obj,
            *args,
            _grouper=self.grouper,
            **kwargs,
        )

    @final
    def _fill(self, direction: Literal["ffill", "bfill"], limit=None):
        """
        Shared function for `pad` and `backfill` to call Cython method.

        Parameters
        ----------
        direction : {'ffill', 'bfill'}
            Direction passed to underlying Cython function. `bfill` will cause
            values to be filled backwards. `ffill` and any other values will
            default to a forward fill
        limit : int, default None
            Maximum number of consecutive values to fill. If `None`, this
            method will convert to -1 prior to passing to Cython

        Returns
        -------
        `Series` or `DataFrame` with filled values

        See Also
        --------
        pad : Returns Series with minimum number of char in object.
        backfill : Backward fill the missing values in the dataset.
        """
        # Need int value for Cython
        if limit is None:
            limit = -1

        return self._get_cythonized_result(
            "group_fillna_indexer",
            numeric_only=False,
            needs_mask=True,
            cython_dtype=np.dtype(np.int64),
            result_is_index=True,
            direction=direction,
            limit=limit,
            dropna=self.dropna,
        )

    @final
    @Substitution(name="groupby")
    def pad(self, limit=None):
        """
        Forward fill the values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series or DataFrame
            Object with missing values filled.

        See Also
        --------
        Series.pad: Returns Series with minimum number of char in object.
        DataFrame.pad: Object with missing values filled or None if inplace=True.
        Series.fillna: Fill NaN values of a Series.
        DataFrame.fillna: Fill NaN values of a DataFrame.
        """
        return self._fill("ffill", limit=limit)

    ffill = pad

    @final
    @Substitution(name="groupby")
    def backfill(self, limit=None):
        """
        Backward fill the values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series or DataFrame
            Object with missing values filled.

        See Also
        --------
        Series.backfill :  Backward fill the missing values in the dataset.
        DataFrame.backfill:  Backward fill the missing values in the dataset.
        Series.fillna: Fill NaN values of a Series.
        DataFrame.fillna: Fill NaN values of a DataFrame.
        """
        return self._fill("bfill", limit=limit)

    bfill = backfill

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def nth(
        self, n: int | list[int], dropna: Literal["any", "all", None] = None
    ) -> DataFrame:
        """
        Take the nth row from each group if n is an int, or a subset of rows
        if n is a list of ints.

        If dropna, will take the nth non-null row, dropna is either
        'all' or 'any'; this is equivalent to calling dropna(how=dropna)
        before the groupby.

        Parameters
        ----------
        n : int or list of ints
            A single nth value for the row or a list of nth values.
        dropna : {'any', 'all', None}, default None
            Apply the specified dropna operation before counting which row is
            the nth row.

        Returns
        -------
        Series or DataFrame
            N-th value within each group.
        %(see_also)s
        Examples
        --------

        >>> df = pd.DataFrame({'A': [1, 1, 2, 1, 2],
        ...                    'B': [np.nan, 2, 3, 4, 5]}, columns=['A', 'B'])
        >>> g = df.groupby('A')
        >>> g.nth(0)
             B
        A
        1  NaN
        2  3.0
        >>> g.nth(1)
             B
        A
        1  2.0
        2  5.0
        >>> g.nth(-1)
             B
        A
        1  4.0
        2  5.0
        >>> g.nth([0, 1])
             B
        A
        1  NaN
        1  2.0
        2  3.0
        2  5.0

        Specifying `dropna` allows count ignoring ``NaN``

        >>> g.nth(0, dropna='any')
             B
        A
        1  2.0
        2  3.0

        NaNs denote group exhausted when using dropna

        >>> g.nth(3, dropna='any')
            B
        A
        1 NaN
        2 NaN

        Specifying `as_index=False` in `groupby` keeps the original index.

        >>> df.groupby('A', as_index=False).nth(1)
           A    B
        1  1  2.0
        4  2  5.0
        """
        valid_containers = (set, list, tuple)
        if not isinstance(n, (valid_containers, int)):
            raise TypeError("n needs to be an int or a list/set/tuple of ints")

        if not dropna:

            if isinstance(n, int):
                nth_values = [n]
            elif isinstance(n, valid_containers):
                nth_values = list(set(n))

            nth_array = np.array(nth_values, dtype=np.intp)
            with group_selection_context(self):

                mask_left = np.in1d(self._cumcount_array(), nth_array)
                mask_right = np.in1d(
                    self._cumcount_array(ascending=False) + 1, -nth_array
                )
                mask = mask_left | mask_right

                ids, _, _ = self.grouper.group_info

                # Drop NA values in grouping
                mask = mask & (ids != -1)

                out = self._selected_obj[mask]
                if not self.as_index:
                    return out

                result_index = self.grouper.result_index
                out.index = result_index[ids[mask]]

                if not self.observed and isinstance(result_index, CategoricalIndex):
                    out = out.reindex(result_index)

                out = self._reindex_output(out)
                return out.sort_index() if self.sort else out

        # dropna is truthy
        if isinstance(n, valid_containers):
            raise ValueError("dropna option with a list of nth values is not supported")

        if dropna not in ["any", "all"]:
            # Note: when agg-ing picker doesn't raise this, just returns NaN
            raise ValueError(
                "For a DataFrame groupby, dropna must be "
                "either None, 'any' or 'all', "
                f"(was passed {dropna})."
            )

        # old behaviour, but with all and any support for DataFrames.
        # modified in GH 7559 to have better perf
        max_len = n if n >= 0 else -1 - n
        dropped = self.obj.dropna(how=dropna, axis=self.axis)

        # get a new grouper for our dropped obj
        if self.keys is None and self.level is None:

            # we don't have the grouper info available
            # (e.g. we have selected out
            # a column that is not in the current object)
            axis = self.grouper.axis
            grouper = axis[axis.isin(dropped.index)]

        else:

            # create a grouper with the original parameters, but on dropped
            # object
            from pandas.core.groupby.grouper import get_grouper

            grouper, _, _ = get_grouper(
                dropped,
                key=self.keys,
                axis=self.axis,
                level=self.level,
                sort=self.sort,
                mutated=self.mutated,
            )

        grb = dropped.groupby(grouper, as_index=self.as_index, sort=self.sort)
        sizes, result = grb.size(), grb.nth(n)
        mask = (sizes < max_len)._values

        # set the results which don't meet the criteria
        if len(result) and mask.any():
            result.loc[mask] = np.nan

        # reset/reindex to the original groups
        if len(self.obj) == len(dropped) or len(result) == len(
            self.grouper.result_index
        ):
            result.index = self.grouper.result_index
        else:
            result = result.reindex(self.grouper.result_index)

        return result

    @final
    def quantile(self, q=0.5, interpolation: str = "linear"):
        """
        Return group values at the given quantile, a la numpy.percentile.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            Value(s) between 0 and 1 providing the quantile(s) to compute.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            Method to use when the desired quantile falls between two points.

        Returns
        -------
        Series or DataFrame
            Return type determined by caller of GroupBy object.

        See Also
        --------
        Series.quantile : Similar method for Series.
        DataFrame.quantile : Similar method for DataFrame.
        numpy.percentile : NumPy method to compute qth percentile.

        Examples
        --------
        >>> df = pd.DataFrame([
        ...     ['a', 1], ['a', 2], ['a', 3],
        ...     ['b', 1], ['b', 3], ['b', 5]
        ... ], columns=['key', 'val'])
        >>> df.groupby('key').quantile()
            val
        key
        a    2.0
        b    3.0
        """
        from pandas import concat

        def pre_processor(vals: ArrayLike) -> tuple[np.ndarray, np.dtype | None]:
            if is_object_dtype(vals):
                raise TypeError(
                    "'quantile' cannot be performed against 'object' dtypes!"
                )

            inference: np.dtype | None = None
            if is_integer_dtype(vals.dtype):
                if isinstance(vals, ExtensionArray):
                    out = vals.to_numpy(dtype=float, na_value=np.nan)
                else:
                    out = vals
                inference = np.dtype(np.int64)
            elif is_bool_dtype(vals.dtype) and isinstance(vals, ExtensionArray):
                out = vals.to_numpy(dtype=float, na_value=np.nan)
            elif is_datetime64_dtype(vals.dtype):
                inference = np.dtype("datetime64[ns]")
                out = np.asarray(vals).astype(float)
            elif is_timedelta64_dtype(vals.dtype):
                inference = np.dtype("timedelta64[ns]")
                out = np.asarray(vals).astype(float)
            elif isinstance(vals, ExtensionArray) and is_float_dtype(vals):
                inference = np.dtype(np.float64)
                out = vals.to_numpy(dtype=float, na_value=np.nan)
            else:
                out = np.asarray(vals)

            return out, inference

        def post_processor(vals: np.ndarray, inference: type | None) -> np.ndarray:
            if inference:
                # Check for edge case
                if not (
                    is_integer_dtype(inference)
                    and interpolation in {"linear", "midpoint"}
                ):
                    vals = vals.astype(inference)

            return vals

        if is_scalar(q):
            return self._get_cythonized_result(
                "group_quantile",
                aggregate=True,
                numeric_only=False,
                needs_values=True,
                needs_mask=True,
                cython_dtype=np.dtype(np.float64),
                pre_processing=pre_processor,
                post_processing=post_processor,
                q=q,
                interpolation=interpolation,
            )
        else:
            results = [
                self._get_cythonized_result(
                    "group_quantile",
                    aggregate=True,
                    needs_values=True,
                    needs_mask=True,
                    cython_dtype=np.dtype(np.float64),
                    pre_processing=pre_processor,
                    post_processing=post_processor,
                    q=qi,
                    interpolation=interpolation,
                )
                for qi in q
            ]
            result = concat(results, axis=self.axis, keys=q)
            # fix levels to place quantiles on the inside
            # TODO(GH-10710): Ideally, we could write this as
            #  >>> result.stack(0).loc[pd.IndexSlice[:, ..., q], :]
            #  but this hits https://github.com/pandas-dev/pandas/issues/10710
            #  which doesn't reorder the list-like `q` on the inner level.
            order = list(range(1, result.axes[self.axis].nlevels)) + [0]

            # temporarily saves the index names
            index_names = np.array(result.axes[self.axis].names)

            # set index names to positions to avoid confusion
            result.axes[self.axis].names = np.arange(len(index_names))

            # place quantiles on the inside
            if isinstance(result, Series):
                result = result.reorder_levels(order)
            else:
                result = result.reorder_levels(order, axis=self.axis)

            # restore the index names in order
            result.axes[self.axis].names = index_names[order]

            # reorder rows to keep things sorted
            indices = (
                np.arange(result.shape[self.axis])
                .reshape([len(q), self.ngroups])
                .T.flatten()
            )
            return result.take(indices, axis=self.axis)

    @final
    @Substitution(name="groupby")
    def ngroup(self, ascending: bool = True):
        """
        Number each group from 0 to the number of groups - 1.

        This is the enumerative complement of cumcount.  Note that the
        numbers given to the groups match the order in which the groups
        would be seen when iterating over the groupby object, not the
        order they are first observed.

        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from number of group - 1 to 0.

        Returns
        -------
        Series
            Unique numbers for each group.

        See Also
        --------
        .cumcount : Number the rows in each group.

        Examples
        --------
        >>> df = pd.DataFrame({"A": list("aaabba")})
        >>> df
           A
        0  a
        1  a
        2  a
        3  b
        4  b
        5  a
        >>> df.groupby('A').ngroup()
        0    0
        1    0
        2    0
        3    1
        4    1
        5    0
        dtype: int64
        >>> df.groupby('A').ngroup(ascending=False)
        0    1
        1    1
        2    1
        3    0
        4    0
        5    1
        dtype: int64
        >>> df.groupby(["A", [1,1,2,3,2,1]]).ngroup()
        0    0
        1    0
        2    1
        3    3
        4    2
        5    0
        dtype: int64
        """
        with group_selection_context(self):
            index = self._selected_obj.index
            result = self._obj_1d_constructor(
                self.grouper.group_info[0], index, dtype=np.int64
            )
            if not ascending:
                result = self.ngroups - 1 - result
            return result

    @final
    @Substitution(name="groupby")
    def cumcount(self, ascending: bool = True):
        """
        Number each item in each group from 0 to the length of that group - 1.

        Essentially this is equivalent to

        .. code-block:: python

            self.apply(lambda x: pd.Series(np.arange(len(x)), x.index))

        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from length of group - 1 to 0.

        Returns
        -------
        Series
            Sequence number of each element within each group.

        See Also
        --------
        .ngroup : Number the groups themselves.

        Examples
        --------
        >>> df = pd.DataFrame([['a'], ['a'], ['a'], ['b'], ['b'], ['a']],
        ...                   columns=['A'])
        >>> df
           A
        0  a
        1  a
        2  a
        3  b
        4  b
        5  a
        >>> df.groupby('A').cumcount()
        0    0
        1    1
        2    2
        3    0
        4    1
        5    3
        dtype: int64
        >>> df.groupby('A').cumcount(ascending=False)
        0    3
        1    2
        2    1
        3    1
        4    0
        5    0
        dtype: int64
        """
        with group_selection_context(self):
            index = self._selected_obj._get_axis(self.axis)
            cumcounts = self._cumcount_array(ascending=ascending)
            return self._obj_1d_constructor(cumcounts, index)

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def rank(
        self,
        method: str = "average",
        ascending: bool = True,
        na_option: str = "keep",
        pct: bool = False,
        axis: int = 0,
    ):
        """
        Provide the rank of values within each group.

        Parameters
        ----------
        method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
            * average: average rank of group.
            * min: lowest rank in group.
            * max: highest rank in group.
            * first: ranks assigned in order they appear in the array.
            * dense: like 'min', but rank always increases by 1 between groups.
        ascending : bool, default True
            False for ranks by high (1) to low (N).
        na_option : {'keep', 'top', 'bottom'}, default 'keep'
            * keep: leave NA values where they are.
            * top: smallest rank if ascending.
            * bottom: smallest rank if descending.
        pct : bool, default False
            Compute percentage rank of data within each group.
        axis : int, default 0
            The axis of the object over which to compute the rank.

        Returns
        -------
        DataFrame with ranking of values within each group
        """
        if na_option not in {"keep", "top", "bottom"}:
            msg = "na_option must be one of 'keep', 'top', or 'bottom'"
            raise ValueError(msg)

        kwargs = {
            "ties_method": method,
            "ascending": ascending,
            "na_option": na_option,
            "pct": pct,
        }
        if axis != 0:
            # DataFrame uses different keyword name
            kwargs["method"] = kwargs.pop("ties_method")
            return self.apply(lambda x: x.rank(axis=axis, numeric_only=False, **kwargs))

        return self._cython_transform(
            "rank",
            numeric_only=False,
            axis=axis,
            **kwargs,
        )

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def cumprod(self, axis=0, *args, **kwargs):
        """
        Cumulative product for each group.

        Returns
        -------
        Series or DataFrame
        """
        nv.validate_groupby_func("cumprod", args, kwargs, ["numeric_only", "skipna"])
        if axis != 0:
            return self.apply(lambda x: x.cumprod(axis=axis, **kwargs))

        return self._cython_transform("cumprod", **kwargs)

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def cumsum(self, axis=0, *args, **kwargs):
        """
        Cumulative sum for each group.

        Returns
        -------
        Series or DataFrame
        """
        nv.validate_groupby_func("cumsum", args, kwargs, ["numeric_only", "skipna"])
        if axis != 0:
            return self.apply(lambda x: x.cumsum(axis=axis, **kwargs))

        return self._cython_transform("cumsum", **kwargs)

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def cummin(self, axis=0, **kwargs):
        """
        Cumulative min for each group.

        Returns
        -------
        Series or DataFrame
        """
        if axis != 0:
            return self.apply(lambda x: np.minimum.accumulate(x, axis))

        return self._cython_transform("cummin", numeric_only=False)

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def cummax(self, axis=0, **kwargs):
        """
        Cumulative max for each group.

        Returns
        -------
        Series or DataFrame
        """
        if axis != 0:
            return self.apply(lambda x: np.maximum.accumulate(x, axis))

        return self._cython_transform("cummax", numeric_only=False)

    @final
    def _get_cythonized_result(
        self,
        how: str,
        cython_dtype: np.dtype,
        aggregate: bool = False,
        numeric_only: bool | lib.NoDefault = lib.no_default,
        needs_counts: bool = False,
        needs_values: bool = False,
        needs_2d: bool = False,
        needs_nullable: bool = False,
        min_count: int | None = None,
        needs_mask: bool = False,
        needs_ngroups: bool = False,
        result_is_index: bool = False,
        pre_processing=None,
        post_processing=None,
        **kwargs,
    ):
        """
        Get result for Cythonized functions.

        Parameters
        ----------
        how : str, Cythonized function name to be called
        cython_dtype : np.dtype
            Type of the array that will be modified by the Cython call.
        aggregate : bool, default False
            Whether the result should be aggregated to match the number of
            groups
        numeric_only : bool, default True
            Whether only numeric datatypes should be computed
        needs_counts : bool, default False
            Whether the counts should be a part of the Cython call
        needs_values : bool, default False
            Whether the values should be a part of the Cython call
            signature
        needs_2d : bool, default False
            Whether the values and result of the Cython call signature
            are 2-dimensional.
        min_count : int, default None
            When not None, min_count for the Cython call
        needs_mask : bool, default False
            Whether boolean mask needs to be part of the Cython call
            signature
        needs_ngroups : bool, default False
            Whether number of groups is part of the Cython call signature
        needs_nullable : bool, default False
            Whether a bool specifying if the input is nullable is part
            of the Cython call signature
        result_is_index : bool, default False
            Whether the result of the Cython operation is an index of
            values to be retrieved, instead of the actual values themselves
        pre_processing : function, default None
            Function to be applied to `values` prior to passing to Cython.
            Function should return a tuple where the first element is the
            values to be passed to Cython and the second element is an optional
            type which the values should be converted to after being returned
            by the Cython operation. This function is also responsible for
            raising a TypeError if the values have an invalid type. Raises
            if `needs_values` is False.
        post_processing : function, default None
            Function to be applied to result of Cython function. Should accept
            an array of values as the first argument and type inferences as its
            second argument, i.e. the signature should be
            (ndarray, Type). If `needs_nullable=True`, a third argument should be
            `nullable`, to allow for processing specific to nullable values.
        **kwargs : dict
            Extra arguments to be passed back to Cython funcs

        Returns
        -------
        `Series` or `DataFrame`  with filled values
        """
        numeric_only = self._resolve_numeric_only(numeric_only)

        if result_is_index and aggregate:
            raise ValueError("'result_is_index' and 'aggregate' cannot both be True!")
        if post_processing and not callable(post_processing):
            raise ValueError("'post_processing' must be a callable!")
        if pre_processing:
            if not callable(pre_processing):
                raise ValueError("'pre_processing' must be a callable!")
            if not needs_values:
                raise ValueError(
                    "Cannot use 'pre_processing' without specifying 'needs_values'!"
                )

        grouper = self.grouper

        ids, _, ngroups = grouper.group_info
        output: dict[base.OutputKey, np.ndarray] = {}
        base_func = getattr(libgroupby, how)

        error_msg = ""
        for idx, obj in enumerate(self._iterate_slices()):
            name = obj.name
            values = obj._values

            if numeric_only and not is_numeric_dtype(values.dtype):
                continue

            if aggregate:
                result_sz = ngroups
            else:
                result_sz = len(values)

            result = np.zeros(result_sz, dtype=cython_dtype)
            if needs_2d:
                result = result.reshape((-1, 1))
            func = partial(base_func, result)

            inferences = None

            if needs_counts:
                counts = np.zeros(self.ngroups, dtype=np.int64)
                func = partial(func, counts)

            if needs_values:
                vals = values
                if pre_processing:
                    try:
                        vals, inferences = pre_processing(vals)
                    except TypeError as err:
                        error_msg = str(err)
                        howstr = how.replace("group_", "")
                        warnings.warn(
                            "Dropping invalid columns in "
                            f"{type(self).__name__}.{howstr} is deprecated. "
                            "In a future version, a TypeError will be raised. "
                            f"Before calling .{howstr}, select only columns which "
                            "should be valid for the function.",
                            FutureWarning,
                            stacklevel=3,
                        )
                        continue
                vals = vals.astype(cython_dtype, copy=False)
                if needs_2d:
                    vals = vals.reshape((-1, 1))
                func = partial(func, vals)

            func = partial(func, ids)

            if min_count is not None:
                func = partial(func, min_count)

            if needs_mask:
                mask = isna(values).view(np.uint8)
                func = partial(func, mask)

            if needs_ngroups:
                func = partial(func, ngroups)

            if needs_nullable:
                is_nullable = isinstance(values, BaseMaskedArray)
                func = partial(func, nullable=is_nullable)
                if post_processing:
                    post_processing = partial(post_processing, nullable=is_nullable)

            func(**kwargs)  # Call func to modify indexer values in place

            if needs_2d:
                result = result.reshape(-1)

            if result_is_index:
                result = algorithms.take_nd(values, result)

            if post_processing:
                result = post_processing(result, inferences)

            key = base.OutputKey(label=name, position=idx)
            output[key] = result

        # error_msg is "" on an frame/series with no rows or columns
        if not output and error_msg != "":
            raise TypeError(error_msg)

        if aggregate:
            return self._wrap_aggregated_output(output)
        else:
            return self._wrap_transformed_output(output)

    @final
    @Substitution(name="groupby")
    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        """
        Shift each group by periods observations.

        If freq is passed, the index will be increased using the periods and the freq.

        Parameters
        ----------
        periods : int, default 1
            Number of periods to shift.
        freq : str, optional
            Frequency string.
        axis : axis to shift, default 0
            Shift direction.
        fill_value : optional
            The scalar value to use for newly introduced missing values.

        Returns
        -------
        Series or DataFrame
            Object shifted within each group.

        See Also
        --------
        Index.shift : Shift values of Index.
        tshift : Shift the time index, using the index’s frequency
            if available.
        """
        if freq is not None or axis != 0 or not isna(fill_value):
            return self.apply(lambda x: x.shift(periods, freq, axis, fill_value))

        return self._get_cythonized_result(
            "group_shift_indexer",
            numeric_only=False,
            cython_dtype=np.dtype(np.int64),
            needs_ngroups=True,
            result_is_index=True,
            periods=periods,
        )

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def pct_change(self, periods=1, fill_method="pad", limit=None, freq=None, axis=0):
        """
        Calculate pct_change of each value to previous entry in group.

        Returns
        -------
        Series or DataFrame
            Percentage changes within each group.
        """
        if freq is not None or axis != 0:
            return self.apply(
                lambda x: x.pct_change(
                    periods=periods,
                    fill_method=fill_method,
                    limit=limit,
                    freq=freq,
                    axis=axis,
                )
            )
        if fill_method is None:  # GH30463
            fill_method = "pad"
            limit = 0
        filled = getattr(self, fill_method)(limit=limit)
        fill_grp = filled.groupby(self.grouper.codes, axis=self.axis)
        shifted = fill_grp.shift(periods=periods, freq=freq, axis=self.axis)
        return (filled / shifted) - 1

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def head(self, n=5):
        """
        Return first n rows of each group.

        Similar to ``.apply(lambda x: x.head(n))``, but it returns a subset of rows
        from the original DataFrame with original index and order preserved
        (``as_index`` flag is ignored).

        Does not work for negative values of `n`.

        Returns
        -------
        Series or DataFrame
        %(see_also)s
        Examples
        --------

        >>> df = pd.DataFrame([[1, 2], [1, 4], [5, 6]],
        ...                   columns=['A', 'B'])
        >>> df.groupby('A').head(1)
           A  B
        0  1  2
        2  5  6
        >>> df.groupby('A').head(-1)
        Empty DataFrame
        Columns: [A, B]
        Index: []
        """
        self._reset_group_selection()
        mask = self._cumcount_array() < n
        if self.axis == 0:
            return self._selected_obj[mask]
        else:
            return self._selected_obj.iloc[:, mask]

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def tail(self, n=5):
        """
        Return last n rows of each group.

        Similar to ``.apply(lambda x: x.tail(n))``, but it returns a subset of rows
        from the original DataFrame with original index and order preserved
        (``as_index`` flag is ignored).

        Does not work for negative values of `n`.

        Returns
        -------
        Series or DataFrame
        %(see_also)s
        Examples
        --------

        >>> df = pd.DataFrame([['a', 1], ['a', 2], ['b', 1], ['b', 2]],
        ...                   columns=['A', 'B'])
        >>> df.groupby('A').tail(1)
           A  B
        1  a  2
        3  b  2
        >>> df.groupby('A').tail(-1)
        Empty DataFrame
        Columns: [A, B]
        Index: []
        """
        self._reset_group_selection()
        mask = self._cumcount_array(ascending=False) < n
        if self.axis == 0:
            return self._selected_obj[mask]
        else:
            return self._selected_obj.iloc[:, mask]

    @final
    def _reindex_output(
        self, output: OutputFrameOrSeries, fill_value: Scalar = np.NaN
    ) -> OutputFrameOrSeries:
        """
        If we have categorical groupers, then we might want to make sure that
        we have a fully re-indexed output to the levels. This means expanding
        the output space to accommodate all values in the cartesian product of
        our groups, regardless of whether they were observed in the data or
        not. This will expand the output space if there are missing groups.

        The method returns early without modifying the input if the number of
        groupings is less than 2, self.observed == True or none of the groupers
        are categorical.

        Parameters
        ----------
        output : Series or DataFrame
            Object resulting from grouping and applying an operation.
        fill_value : scalar, default np.NaN
            Value to use for unobserved categories if self.observed is False.

        Returns
        -------
        Series or DataFrame
            Object (potentially) re-indexed to include all possible groups.
        """
        groupings = self.grouper.groupings
        if len(groupings) == 1:
            return output

        # if we only care about the observed values
        # we are done
        elif self.observed:
            return output

        # reindexing only applies to a Categorical grouper
        elif not any(
            isinstance(ping.grouping_vector, (Categorical, CategoricalIndex))
            for ping in groupings
        ):
            return output

        levels_list = [ping.group_index for ping in groupings]
        index, _ = MultiIndex.from_product(
            levels_list, names=self.grouper.names
        ).sortlevel()

        if self.as_index:
            d = {
                self.obj._get_axis_name(self.axis): index,
                "copy": False,
                "fill_value": fill_value,
            }
            return output.reindex(**d)

        # GH 13204
        # Here, the categorical in-axis groupers, which need to be fully
        # expanded, are columns in `output`. An idea is to do:
        # output = output.set_index(self.grouper.names)
        #                .reindex(index).reset_index()
        # but special care has to be taken because of possible not-in-axis
        # groupers.
        # So, we manually select and drop the in-axis grouper columns,
        # reindex `output`, and then reset the in-axis grouper columns.

        # Select in-axis groupers
        in_axis_grps = (
            (i, ping.name) for (i, ping) in enumerate(groupings) if ping.in_axis
        )
        g_nums, g_names = zip(*in_axis_grps)

        output = output.drop(labels=list(g_names), axis=1)

        # Set a temp index and reindex (possibly expanding)
        output = output.set_index(self.grouper.result_index).reindex(
            index, copy=False, fill_value=fill_value
        )

        # Reset in-axis grouper columns
        # (using level numbers `g_nums` because level names may not be unique)
        output = output.reset_index(level=g_nums)

        return output.reset_index(drop=True)

    @final
    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        replace: bool = False,
        weights: Sequence | Series | None = None,
        random_state=None,
    ):
        """
        Return a random sample of items from each group.

        You can use `random_state` for reproducibility.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        n : int, optional
            Number of items to return for each group. Cannot be used with
            `frac` and must be no larger than the smallest group unless
            `replace` is True. Default is one if `frac` is None.
        frac : float, optional
            Fraction of items to return. Cannot be used with `n`.
        replace : bool, default False
            Allow or disallow sampling of the same row more than once.
        weights : list-like, optional
            Default None results in equal probability weighting.
            If passed a list-like then values must have the same length as
            the underlying DataFrame or Series object and will be used as
            sampling probabilities after normalization within each group.
            Values must be non-negative with at least one positive element
            within each group.
        random_state : int, array-like, BitGenerator, np.random.RandomState, optional
            If int, array-like, or BitGenerator (NumPy>=1.17), seed for
            random number generator
            If np.random.RandomState, use as numpy RandomState object.

        Returns
        -------
        Series or DataFrame
            A new object of same type as caller containing items randomly
            sampled within each group from the caller object.

        See Also
        --------
        DataFrame.sample: Generate random samples from a DataFrame object.
        numpy.random.choice: Generate a random sample from a given 1-D numpy
            array.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"a": ["red"] * 2 + ["blue"] * 2 + ["black"] * 2, "b": range(6)}
        ... )
        >>> df
               a  b
        0    red  0
        1    red  1
        2   blue  2
        3   blue  3
        4  black  4
        5  black  5

        Select one row at random for each distinct value in column a. The
        `random_state` argument can be used to guarantee reproducibility:

        >>> df.groupby("a").sample(n=1, random_state=1)
               a  b
        4  black  4
        2   blue  2
        1    red  1

        Set `frac` to sample fixed proportions rather than counts:

        >>> df.groupby("a")["b"].sample(frac=0.5, random_state=2)
        5    5
        2    2
        0    0
        Name: b, dtype: int64

        Control sample probabilities within groups by setting weights:

        >>> df.groupby("a").sample(
        ...     n=1,
        ...     weights=[1, 1, 1, 0, 0, 1],
        ...     random_state=1,
        ... )
               a  b
        5  black  5
        2   blue  2
        0    red  0
        """
        from pandas.core.reshape.concat import concat

        if weights is not None:
            weights = Series(weights, index=self._selected_obj.index)
            ws = [weights.iloc[idx] for idx in self.indices.values()]
        else:
            ws = [None] * self.ngroups

        if random_state is not None:
            random_state = com.random_state(random_state)

        group_iterator = self.grouper.get_iterator(self._selected_obj, self.axis)
        samples = [
            obj.sample(
                n=n, frac=frac, replace=replace, weights=w, random_state=random_state
            )
            for (_, obj), w in zip(group_iterator, ws)
        ]

        return concat(samples, axis=self.axis)


@doc(GroupBy)
def get_groupby(
    obj: NDFrame,
    by: _KeysArgType | None = None,
    axis: int = 0,
    level=None,
    grouper: ops.BaseGrouper | None = None,
    exclusions=None,
    selection=None,
    as_index: bool = True,
    sort: bool = True,
    group_keys: bool = True,
    squeeze: bool = False,
    observed: bool = False,
    mutated: bool = False,
    dropna: bool = True,
) -> GroupBy:

    klass: type[GroupBy]
    if isinstance(obj, Series):
        from pandas.core.groupby.generic import SeriesGroupBy

        klass = SeriesGroupBy
    elif isinstance(obj, DataFrame):
        from pandas.core.groupby.generic import DataFrameGroupBy

        klass = DataFrameGroupBy
    else:  # pragma: no cover
        raise TypeError(f"invalid type: {obj}")

    return klass(
        obj=obj,
        keys=by,
        axis=axis,
        level=level,
        grouper=grouper,
        exclusions=exclusions,
        selection=selection,
        as_index=as_index,
        sort=sort,
        group_keys=group_keys,
        squeeze=squeeze,
        observed=observed,
        mutated=mutated,
        dropna=dropna,
    )
