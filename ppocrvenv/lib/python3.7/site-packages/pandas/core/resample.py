from __future__ import annotations

import copy
from datetime import timedelta
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Callable,
    Hashable,
    no_type_check,
)

import numpy as np

from pandas._libs import lib
from pandas._libs.tslibs import (
    BaseOffset,
    IncompatibleFrequency,
    NaT,
    Period,
    Timedelta,
    Timestamp,
    to_offset,
)
from pandas._typing import (
    FrameOrSeries,
    IndexLabel,
    T,
    TimedeltaConvertibleTypes,
    TimestampConvertibleTypes,
    final,
)
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
    Appender,
    Substitution,
    deprecate_nonkeyword_arguments,
    doc,
)

from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

import pandas.core.algorithms as algos
from pandas.core.apply import ResamplerWindowApply
from pandas.core.base import (
    DataError,
    PandasObject,
)
import pandas.core.common as com
from pandas.core.generic import (
    NDFrame,
    _shared_docs,
)
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.groupby.groupby import (
    BaseGroupBy,
    GroupBy,
    _pipe_template,
    get_groupby,
)
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
from pandas.core.indexes.api import Index
from pandas.core.indexes.datetimes import (
    DatetimeIndex,
    date_range,
)
from pandas.core.indexes.period import (
    PeriodIndex,
    period_range,
)
from pandas.core.indexes.timedeltas import (
    TimedeltaIndex,
    timedelta_range,
)

from pandas.tseries.frequencies import (
    is_subperiod,
    is_superperiod,
)
from pandas.tseries.offsets import (
    DateOffset,
    Day,
    Nano,
    Tick,
)

if TYPE_CHECKING:
    from typing import Literal

_shared_docs_kwargs: dict[str, str] = {}


class Resampler(BaseGroupBy, PandasObject):
    """
    Class for resampling datetimelike data, a groupby-like operation.
    See aggregate, transform, and apply functions on this object.

    It's easiest to use obj.resample(...) to use Resampler.

    Parameters
    ----------
    obj : Series or DataFrame
    groupby : TimeGrouper
    axis : int, default 0
    kind : str or None
        'period', 'timestamp' to override default index treatment

    Returns
    -------
    a Resampler of the appropriate type

    Notes
    -----
    After resampling, see aggregate, apply, and transform functions.
    """

    grouper: BinGrouper
    exclusions: frozenset[Hashable] = frozenset()  # for SelectionMixin compat

    # to the groupby descriptor
    _attributes = [
        "freq",
        "axis",
        "closed",
        "label",
        "convention",
        "loffset",
        "kind",
        "origin",
        "offset",
    ]

    def __init__(
        self,
        obj: FrameOrSeries,
        groupby: TimeGrouper,
        axis: int = 0,
        kind=None,
        *,
        selection=None,
        **kwargs,
    ):
        self.groupby = groupby
        self.keys = None
        self.sort = True
        self.axis = axis
        self.kind = kind
        self.squeeze = False
        self.group_keys = True
        self.as_index = True

        self.groupby._set_grouper(self._convert_obj(obj), sort=True)
        self.binner, self.grouper = self._get_binner()
        self._selection = selection

    @final
    def _shallow_copy(self, obj, **kwargs):
        """
        return a new object with the replacement attributes
        """
        if isinstance(obj, self._constructor):
            obj = obj.obj
        for attr in self._attributes:
            if attr not in kwargs:
                kwargs[attr] = getattr(self, attr)
        return self._constructor(obj, **kwargs)

    def __str__(self) -> str:
        """
        Provide a nice str repr of our rolling object.
        """
        attrs = (
            f"{k}={getattr(self.groupby, k)}"
            for k in self._attributes
            if getattr(self.groupby, k, None) is not None
        )
        return f"{type(self).__name__} [{', '.join(attrs)}]"

    def __getattr__(self, attr: str):
        if attr in self._internal_names_set:
            return object.__getattribute__(self, attr)
        if attr in self._attributes:
            return getattr(self.groupby, attr)
        if attr in self.obj:
            return self[attr]

        return object.__getattribute__(self, attr)

    # error: Signature of "obj" incompatible with supertype "BaseGroupBy"
    @property
    def obj(self) -> FrameOrSeries:  # type: ignore[override]
        # error: Incompatible return value type (got "Optional[Any]",
        # expected "FrameOrSeries")
        return self.groupby.obj  # type: ignore[return-value]

    @property
    def ax(self):
        # we can infer that this is a PeriodIndex/DatetimeIndex/TimedeltaIndex,
        #  but skipping annotating bc the overrides overwhelming
        return self.groupby.ax

    @property
    def _from_selection(self) -> bool:
        """
        Is the resampling from a DataFrame column or MultiIndex level.
        """
        # upsampling and PeriodIndex resampling do not work
        # with selection, this state used to catch and raise an error
        return self.groupby is not None and (
            self.groupby.key is not None or self.groupby.level is not None
        )

    def _convert_obj(self, obj: FrameOrSeries) -> FrameOrSeries:
        """
        Provide any conversions for the object in order to correctly handle.

        Parameters
        ----------
        obj : Series or DataFrame

        Returns
        -------
        Series or DataFrame
        """
        return obj._consolidate()

    def _get_binner_for_time(self):
        raise AbstractMethodError(self)

    @final
    def _get_binner(self):
        """
        Create the BinGrouper, assume that self.set_grouper(obj)
        has already been called.
        """
        binner, bins, binlabels = self._get_binner_for_time()
        assert len(bins) == len(binlabels)
        bin_grouper = BinGrouper(bins, binlabels, indexer=self.groupby.indexer)
        return binner, bin_grouper

    @Substitution(
        klass="Resampler",
        examples="""
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4]},
    ...                   index=pd.date_range('2012-08-02', periods=4))
    >>> df
                A
    2012-08-02  1
    2012-08-03  2
    2012-08-04  3
    2012-08-05  4

    To get the difference between each 2-day period's maximum and minimum
    value in one pass, you can do

    >>> df.resample('2D').pipe(lambda x: x.max() - x.min())
                A
    2012-08-02  1
    2012-08-04  1""",
    )
    @Appender(_pipe_template)
    def pipe(
        self,
        func: Callable[..., T] | tuple[Callable[..., T], str],
        *args,
        **kwargs,
    ) -> T:
        return super().pipe(func, *args, **kwargs)

    _agg_see_also_doc = dedent(
        """
    See Also
    --------
    DataFrame.groupby.aggregate : Aggregate using callable, string, dict,
        or list of string/callables.
    DataFrame.resample.transform : Transforms the Series on each group
        based on the given function.
    DataFrame.aggregate: Aggregate using one or more
        operations over the specified axis.
    """
    )

    _agg_examples_doc = dedent(
        """
    Examples
    --------
    >>> s = pd.Series([1,2,3,4,5],
                      index=pd.date_range('20130101', periods=5,freq='s'))
    2013-01-01 00:00:00    1
    2013-01-01 00:00:01    2
    2013-01-01 00:00:02    3
    2013-01-01 00:00:03    4
    2013-01-01 00:00:04    5
    Freq: S, dtype: int64

    >>> r = s.resample('2s')
    DatetimeIndexResampler [freq=<2 * Seconds>, axis=0, closed=left,
                            label=left, convention=start]

    >>> r.agg(np.sum)
    2013-01-01 00:00:00    3
    2013-01-01 00:00:02    7
    2013-01-01 00:00:04    5
    Freq: 2S, dtype: int64

    >>> r.agg(['sum','mean','max'])
                         sum  mean  max
    2013-01-01 00:00:00    3   1.5    2
    2013-01-01 00:00:02    7   3.5    4
    2013-01-01 00:00:04    5   5.0    5

    >>> r.agg({'result' : lambda x: x.mean() / x.std(),
               'total' : np.sum})
                         total    result
    2013-01-01 00:00:00      3  2.121320
    2013-01-01 00:00:02      7  4.949747
    2013-01-01 00:00:04      5       NaN
    """
    )

    @doc(
        _shared_docs["aggregate"],
        see_also=_agg_see_also_doc,
        examples=_agg_examples_doc,
        klass="DataFrame",
        axis="",
    )
    def aggregate(self, func, *args, **kwargs):

        result = ResamplerWindowApply(self, func, args=args, kwargs=kwargs).agg()
        if result is None:
            how = func
            grouper = None
            result = self._groupby_and_aggregate(how, grouper, *args, **kwargs)

        result = self._apply_loffset(result)
        return result

    agg = aggregate
    apply = aggregate

    def transform(self, arg, *args, **kwargs):
        """
        Call function producing a like-indexed Series on each group and return
        a Series with the transformed values.

        Parameters
        ----------
        arg : function
            To apply to each group. Should return a Series with the same index.

        Returns
        -------
        transformed : Series

        Examples
        --------
        >>> resampled.transform(lambda x: (x - x.mean()) / x.std())
        """
        return self._selected_obj.groupby(self.groupby).transform(arg, *args, **kwargs)

    def _downsample(self, f):
        raise AbstractMethodError(self)

    def _upsample(self, f, limit=None, fill_value=None):
        raise AbstractMethodError(self)

    def _gotitem(self, key, ndim: int, subset=None):
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : string / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        grouper = self.grouper
        if subset is None:
            subset = self.obj
        grouped = get_groupby(subset, by=None, grouper=grouper, axis=self.axis)

        # try the key selection
        try:
            return grouped[key]
        except KeyError:
            return grouped

    def _groupby_and_aggregate(self, how, grouper=None, *args, **kwargs):
        """
        Re-evaluate the obj with a groupby aggregation.
        """
        if grouper is None:
            grouper = self.grouper

        obj = self._selected_obj

        grouped = get_groupby(obj, by=None, grouper=grouper, axis=self.axis)

        try:
            if isinstance(obj, ABCDataFrame) and callable(how):
                # Check if the function is reducing or not.
                result = grouped._aggregate_item_by_item(how, *args, **kwargs)
            else:
                result = grouped.aggregate(how, *args, **kwargs)
        except DataError:
            # got TypeErrors on aggregation
            result = grouped.apply(how, *args, **kwargs)
        except (AttributeError, KeyError):
            # we have a non-reducing function; try to evaluate
            # alternatively we want to evaluate only a column of the input

            # test_apply_to_one_column_of_df the function being applied references
            #  a DataFrame column, but aggregate_item_by_item operates column-wise
            #  on Series, raising AttributeError or KeyError
            #  (depending on whether the column lookup uses getattr/__getitem__)
            result = grouped.apply(how, *args, **kwargs)

        except ValueError as err:
            if "Must produce aggregated value" in str(err):
                # raised in _aggregate_named
                # see test_apply_without_aggregation, test_apply_with_mutated_index
                pass
            else:
                raise

            # we have a non-reducing function
            # try to evaluate
            result = grouped.apply(how, *args, **kwargs)

        result = self._apply_loffset(result)
        return self._wrap_result(result)

    def _apply_loffset(self, result):
        """
        If loffset is set, offset the result index.

        This is NOT an idempotent routine, it will be applied
        exactly once to the result.

        Parameters
        ----------
        result : Series or DataFrame
            the result of resample
        """
        # error: Cannot determine type of 'loffset'
        needs_offset = (
            isinstance(
                self.loffset,  # type: ignore[has-type]
                (DateOffset, timedelta, np.timedelta64),
            )
            and isinstance(result.index, DatetimeIndex)
            and len(result.index) > 0
        )

        if needs_offset:
            # error: Cannot determine type of 'loffset'
            result.index = result.index + self.loffset  # type: ignore[has-type]

        self.loffset = None
        return result

    def _get_resampler_for_grouping(self, groupby):
        """
        Return the correct class for resampling with groupby.
        """
        return self._resampler_for_grouping(self, groupby=groupby)

    def _wrap_result(self, result):
        """
        Potentially wrap any results.
        """
        if isinstance(result, ABCSeries) and self._selection is not None:
            result.name = self._selection

        if isinstance(result, ABCSeries) and result.empty:
            obj = self.obj
            # When index is all NaT, result is empty but index is not
            result.index = _asfreq_compat(obj.index[:0], freq=self.freq)
            result.name = getattr(obj, "name", None)

        return result

    def pad(self, limit=None):
        """
        Forward fill the values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        An upsampled Series.

        See Also
        --------
        Series.fillna: Fill NA/NaN values using the specified method.
        DataFrame.fillna: Fill NA/NaN values using the specified method.
        """
        return self._upsample("pad", limit=limit)

    ffill = pad

    def nearest(self, limit=None):
        """
        Resample by using the nearest value.

        When resampling data, missing values may appear (e.g., when the
        resampling frequency is higher than the original frequency).
        The `nearest` method will replace ``NaN`` values that appeared in
        the resampled data with the value from the nearest member of the
        sequence, based on the index value.
        Missing values that existed in the original data will not be modified.
        If `limit` is given, fill only this many values in each direction for
        each of the original values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series or DataFrame
            An upsampled Series or DataFrame with ``NaN`` values filled with
            their nearest value.

        See Also
        --------
        backfill : Backward fill the new missing values in the resampled data.
        pad : Forward fill ``NaN`` values.

        Examples
        --------
        >>> s = pd.Series([1, 2],
        ...               index=pd.date_range('20180101',
        ...                                   periods=2,
        ...                                   freq='1h'))
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        Freq: H, dtype: int64

        >>> s.resample('15min').nearest()
        2018-01-01 00:00:00    1
        2018-01-01 00:15:00    1
        2018-01-01 00:30:00    2
        2018-01-01 00:45:00    2
        2018-01-01 01:00:00    2
        Freq: 15T, dtype: int64

        Limit the number of upsampled values imputed by the nearest:

        >>> s.resample('15min').nearest(limit=1)
        2018-01-01 00:00:00    1.0
        2018-01-01 00:15:00    1.0
        2018-01-01 00:30:00    NaN
        2018-01-01 00:45:00    2.0
        2018-01-01 01:00:00    2.0
        Freq: 15T, dtype: float64
        """
        return self._upsample("nearest", limit=limit)

    def backfill(self, limit=None):
        """
        Backward fill the new missing values in the resampled data.

        In statistics, imputation is the process of replacing missing data with
        substituted values [1]_. When resampling data, missing values may
        appear (e.g., when the resampling frequency is higher than the original
        frequency). The backward fill will replace NaN values that appeared in
        the resampled data with the next value in the original sequence.
        Missing values that existed in the original data will not be modified.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series, DataFrame
            An upsampled Series or DataFrame with backward filled NaN values.

        See Also
        --------
        bfill : Alias of backfill.
        fillna : Fill NaN values using the specified method, which can be
            'backfill'.
        nearest : Fill NaN values with nearest neighbor starting from center.
        pad : Forward fill NaN values.
        Series.fillna : Fill NaN values in the Series using the
            specified method, which can be 'backfill'.
        DataFrame.fillna : Fill NaN values in the DataFrame using the
            specified method, which can be 'backfill'.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Imputation_(statistics)

        Examples
        --------
        Resampling a Series:

        >>> s = pd.Series([1, 2, 3],
        ...               index=pd.date_range('20180101', periods=3, freq='h'))
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        2018-01-01 02:00:00    3
        Freq: H, dtype: int64

        >>> s.resample('30min').backfill()
        2018-01-01 00:00:00    1
        2018-01-01 00:30:00    2
        2018-01-01 01:00:00    2
        2018-01-01 01:30:00    3
        2018-01-01 02:00:00    3
        Freq: 30T, dtype: int64

        >>> s.resample('15min').backfill(limit=2)
        2018-01-01 00:00:00    1.0
        2018-01-01 00:15:00    NaN
        2018-01-01 00:30:00    2.0
        2018-01-01 00:45:00    2.0
        2018-01-01 01:00:00    2.0
        2018-01-01 01:15:00    NaN
        2018-01-01 01:30:00    3.0
        2018-01-01 01:45:00    3.0
        2018-01-01 02:00:00    3.0
        Freq: 15T, dtype: float64

        Resampling a DataFrame that has missing values:

        >>> df = pd.DataFrame({'a': [2, np.nan, 6], 'b': [1, 3, 5]},
        ...                   index=pd.date_range('20180101', periods=3,
        ...                                       freq='h'))
        >>> df
                               a  b
        2018-01-01 00:00:00  2.0  1
        2018-01-01 01:00:00  NaN  3
        2018-01-01 02:00:00  6.0  5

        >>> df.resample('30min').backfill()
                               a  b
        2018-01-01 00:00:00  2.0  1
        2018-01-01 00:30:00  NaN  3
        2018-01-01 01:00:00  NaN  3
        2018-01-01 01:30:00  6.0  5
        2018-01-01 02:00:00  6.0  5

        >>> df.resample('15min').backfill(limit=2)
                               a    b
        2018-01-01 00:00:00  2.0  1.0
        2018-01-01 00:15:00  NaN  NaN
        2018-01-01 00:30:00  NaN  3.0
        2018-01-01 00:45:00  NaN  3.0
        2018-01-01 01:00:00  NaN  3.0
        2018-01-01 01:15:00  NaN  NaN
        2018-01-01 01:30:00  6.0  5.0
        2018-01-01 01:45:00  6.0  5.0
        2018-01-01 02:00:00  6.0  5.0
        """
        return self._upsample("backfill", limit=limit)

    bfill = backfill

    def fillna(self, method, limit=None):
        """
        Fill missing values introduced by upsampling.

        In statistics, imputation is the process of replacing missing data with
        substituted values [1]_. When resampling data, missing values may
        appear (e.g., when the resampling frequency is higher than the original
        frequency).

        Missing values that existed in the original data will
        not be modified.

        Parameters
        ----------
        method : {'pad', 'backfill', 'ffill', 'bfill', 'nearest'}
            Method to use for filling holes in resampled data

            * 'pad' or 'ffill': use previous valid observation to fill gap
              (forward fill).
            * 'backfill' or 'bfill': use next valid observation to fill gap.
            * 'nearest': use nearest valid observation to fill gap.

        limit : int, optional
            Limit of how many consecutive missing values to fill.

        Returns
        -------
        Series or DataFrame
            An upsampled Series or DataFrame with missing values filled.

        See Also
        --------
        backfill : Backward fill NaN values in the resampled data.
        pad : Forward fill NaN values in the resampled data.
        nearest : Fill NaN values in the resampled data
            with nearest neighbor starting from center.
        interpolate : Fill NaN values using interpolation.
        Series.fillna : Fill NaN values in the Series using the
            specified method, which can be 'bfill' and 'ffill'.
        DataFrame.fillna : Fill NaN values in the DataFrame using the
            specified method, which can be 'bfill' and 'ffill'.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Imputation_(statistics)

        Examples
        --------
        Resampling a Series:

        >>> s = pd.Series([1, 2, 3],
        ...               index=pd.date_range('20180101', periods=3, freq='h'))
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        2018-01-01 02:00:00    3
        Freq: H, dtype: int64

        Without filling the missing values you get:

        >>> s.resample("30min").asfreq()
        2018-01-01 00:00:00    1.0
        2018-01-01 00:30:00    NaN
        2018-01-01 01:00:00    2.0
        2018-01-01 01:30:00    NaN
        2018-01-01 02:00:00    3.0
        Freq: 30T, dtype: float64

        >>> s.resample('30min').fillna("backfill")
        2018-01-01 00:00:00    1
        2018-01-01 00:30:00    2
        2018-01-01 01:00:00    2
        2018-01-01 01:30:00    3
        2018-01-01 02:00:00    3
        Freq: 30T, dtype: int64

        >>> s.resample('15min').fillna("backfill", limit=2)
        2018-01-01 00:00:00    1.0
        2018-01-01 00:15:00    NaN
        2018-01-01 00:30:00    2.0
        2018-01-01 00:45:00    2.0
        2018-01-01 01:00:00    2.0
        2018-01-01 01:15:00    NaN
        2018-01-01 01:30:00    3.0
        2018-01-01 01:45:00    3.0
        2018-01-01 02:00:00    3.0
        Freq: 15T, dtype: float64

        >>> s.resample('30min').fillna("pad")
        2018-01-01 00:00:00    1
        2018-01-01 00:30:00    1
        2018-01-01 01:00:00    2
        2018-01-01 01:30:00    2
        2018-01-01 02:00:00    3
        Freq: 30T, dtype: int64

        >>> s.resample('30min').fillna("nearest")
        2018-01-01 00:00:00    1
        2018-01-01 00:30:00    2
        2018-01-01 01:00:00    2
        2018-01-01 01:30:00    3
        2018-01-01 02:00:00    3
        Freq: 30T, dtype: int64

        Missing values present before the upsampling are not affected.

        >>> sm = pd.Series([1, None, 3],
        ...               index=pd.date_range('20180101', periods=3, freq='h'))
        >>> sm
        2018-01-01 00:00:00    1.0
        2018-01-01 01:00:00    NaN
        2018-01-01 02:00:00    3.0
        Freq: H, dtype: float64

        >>> sm.resample('30min').fillna('backfill')
        2018-01-01 00:00:00    1.0
        2018-01-01 00:30:00    NaN
        2018-01-01 01:00:00    NaN
        2018-01-01 01:30:00    3.0
        2018-01-01 02:00:00    3.0
        Freq: 30T, dtype: float64

        >>> sm.resample('30min').fillna('pad')
        2018-01-01 00:00:00    1.0
        2018-01-01 00:30:00    1.0
        2018-01-01 01:00:00    NaN
        2018-01-01 01:30:00    NaN
        2018-01-01 02:00:00    3.0
        Freq: 30T, dtype: float64

        >>> sm.resample('30min').fillna('nearest')
        2018-01-01 00:00:00    1.0
        2018-01-01 00:30:00    NaN
        2018-01-01 01:00:00    NaN
        2018-01-01 01:30:00    3.0
        2018-01-01 02:00:00    3.0
        Freq: 30T, dtype: float64

        DataFrame resampling is done column-wise. All the same options are
        available.

        >>> df = pd.DataFrame({'a': [2, np.nan, 6], 'b': [1, 3, 5]},
        ...                   index=pd.date_range('20180101', periods=3,
        ...                                       freq='h'))
        >>> df
                               a  b
        2018-01-01 00:00:00  2.0  1
        2018-01-01 01:00:00  NaN  3
        2018-01-01 02:00:00  6.0  5

        >>> df.resample('30min').fillna("bfill")
                               a  b
        2018-01-01 00:00:00  2.0  1
        2018-01-01 00:30:00  NaN  3
        2018-01-01 01:00:00  NaN  3
        2018-01-01 01:30:00  6.0  5
        2018-01-01 02:00:00  6.0  5
        """
        return self._upsample(method, limit=limit)

    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "method"])
    @doc(NDFrame.interpolate, **_shared_docs_kwargs)
    def interpolate(
        self,
        method="linear",
        axis=0,
        limit=None,
        inplace=False,
        limit_direction="forward",
        limit_area=None,
        downcast=None,
        **kwargs,
    ):
        """
        Interpolate values according to different methods.
        """
        result = self._upsample("asfreq")
        return result.interpolate(
            method=method,
            axis=axis,
            limit=limit,
            inplace=inplace,
            limit_direction=limit_direction,
            limit_area=limit_area,
            downcast=downcast,
            **kwargs,
        )

    def asfreq(self, fill_value=None):
        """
        Return the values at the new freq, essentially a reindex.

        Parameters
        ----------
        fill_value : scalar, optional
            Value to use for missing values, applied during upsampling (note
            this does not fill NaNs that already were present).

        Returns
        -------
        DataFrame or Series
            Values at the specified freq.

        See Also
        --------
        Series.asfreq: Convert TimeSeries to specified frequency.
        DataFrame.asfreq: Convert TimeSeries to specified frequency.
        """
        return self._upsample("asfreq", fill_value=fill_value)

    def std(self, ddof=1, *args, **kwargs):
        """
        Compute standard deviation of groups, excluding missing values.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        Returns
        -------
        DataFrame or Series
            Standard deviation of values within each group.
        """
        nv.validate_resampler_func("std", args, kwargs)
        # error: Unexpected keyword argument "ddof" for "_downsample"
        return self._downsample("std", ddof=ddof)  # type: ignore[call-arg]

    def var(self, ddof=1, *args, **kwargs):
        """
        Compute variance of groups, excluding missing values.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        Returns
        -------
        DataFrame or Series
            Variance of values within each group.
        """
        nv.validate_resampler_func("var", args, kwargs)
        # error: Unexpected keyword argument "ddof" for "_downsample"
        return self._downsample("var", ddof=ddof)  # type: ignore[call-arg]

    @doc(GroupBy.size)
    def size(self):
        result = self._downsample("size")
        if not len(self.ax):
            from pandas import Series

            if self._selected_obj.ndim == 1:
                name = self._selected_obj.name
            else:
                name = None
            result = Series([], index=result.index, dtype="int64", name=name)
        return result

    @doc(GroupBy.count)
    def count(self):
        result = self._downsample("count")
        if not len(self.ax):
            if self._selected_obj.ndim == 1:
                result = type(self._selected_obj)(
                    [], index=result.index, dtype="int64", name=self._selected_obj.name
                )
            else:
                from pandas import DataFrame

                result = DataFrame(
                    [], index=result.index, columns=result.columns, dtype="int64"
                )

        return result

    def quantile(self, q=0.5, **kwargs):
        """
        Return value at the given quantile.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)

        Returns
        -------
        DataFrame or Series
            Quantile of values within each group.

        See Also
        --------
        Series.quantile
            Return a series, where the index is q and the values are the quantiles.
        DataFrame.quantile
            Return a DataFrame, where the columns are the columns of self,
            and the values are the quantiles.
        DataFrameGroupBy.quantile
            Return a DataFrame, where the coulmns are groupby columns,
            and the values are its quantiles.
        """
        # error: Unexpected keyword argument "q" for "_downsample"
        # error: Too many arguments for "_downsample"
        return self._downsample("quantile", q=q, **kwargs)  # type: ignore[call-arg]


# downsample methods
for method in ["sum", "prod", "min", "max", "first", "last"]:

    def f(self, _method=method, min_count=0, *args, **kwargs):
        nv.validate_resampler_func(_method, args, kwargs)
        return self._downsample(_method, min_count=min_count)

    f.__doc__ = getattr(GroupBy, method).__doc__
    setattr(Resampler, method, f)


# downsample methods
for method in ["mean", "sem", "median", "ohlc"]:

    def g(self, _method=method, *args, **kwargs):
        nv.validate_resampler_func(_method, args, kwargs)
        return self._downsample(_method)

    g.__doc__ = getattr(GroupBy, method).__doc__
    setattr(Resampler, method, g)


# series only methods
for method in ["nunique"]:

    def h(self, _method=method):
        return self._downsample(_method)

    h.__doc__ = getattr(SeriesGroupBy, method).__doc__
    setattr(Resampler, method, h)


class _GroupByMixin(PandasObject):
    """
    Provide the groupby facilities.
    """

    _attributes: list[str]  # in practice the same as Resampler._attributes
    _selection: IndexLabel | None = None

    def __init__(self, obj, parent=None, groupby=None, **kwargs):
        # reached via ._gotitem and _get_resampler_for_grouping

        if parent is None:
            parent = obj

        # initialize our GroupByMixin object with
        # the resampler attributes
        for attr in self._attributes:
            setattr(self, attr, kwargs.get(attr, getattr(parent, attr)))
        self._selection = kwargs.get("selection")

        self.binner = parent.binner

        self._groupby = groupby
        self._groupby.mutated = True
        self._groupby.grouper.mutated = True
        self.groupby = copy.copy(parent.groupby)

    @no_type_check
    def _apply(self, f, grouper=None, *args, **kwargs):
        """
        Dispatch to _upsample; we are stripping all of the _upsample kwargs and
        performing the original function call on the grouped object.
        """

        def func(x):
            x = self._shallow_copy(x, groupby=self.groupby)

            if isinstance(f, str):
                return getattr(x, f)(**kwargs)

            return x.apply(f, *args, **kwargs)

        result = self._groupby.apply(func)
        return self._wrap_result(result)

    _upsample = _apply
    _downsample = _apply
    _groupby_and_aggregate = _apply

    @final
    def _gotitem(self, key, ndim, subset=None):
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : string / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        # create a new object to prevent aliasing
        if subset is None:
            # error: "GotItemMixin" has no attribute "obj"
            subset = self.obj  # type: ignore[attr-defined]

        # we need to make a shallow copy of ourselves
        # with the same groupby
        kwargs = {attr: getattr(self, attr) for attr in self._attributes}

        # Try to select from a DataFrame, falling back to a Series
        try:
            groupby = self._groupby[key]
        except IndexError:
            groupby = self._groupby

        selection = None
        if subset.ndim == 2 and (
            (lib.is_scalar(key) and key in subset) or lib.is_list_like(key)
        ):
            selection = key

        new_rs = type(self)(
            subset, groupby=groupby, parent=self, selection=selection, **kwargs
        )
        return new_rs


class DatetimeIndexResampler(Resampler):
    @property
    def _resampler_for_grouping(self):
        return DatetimeIndexResamplerGroupby

    def _get_binner_for_time(self):

        # this is how we are actually creating the bins
        if self.kind == "period":
            return self.groupby._get_time_period_bins(self.ax)
        return self.groupby._get_time_bins(self.ax)

    def _downsample(self, how, **kwargs):
        """
        Downsample the cython defined function.

        Parameters
        ----------
        how : string / cython mapped function
        **kwargs : kw args passed to how function
        """
        how = com.get_cython_func(how) or how
        ax = self.ax
        obj = self._selected_obj

        if not len(ax):
            # reset to the new freq
            obj = obj.copy()
            obj.index = obj.index._with_freq(self.freq)
            assert obj.index.freq == self.freq, (obj.index.freq, self.freq)
            return obj

        # do we have a regular frequency

        # error: Item "None" of "Optional[Any]" has no attribute "binlabels"
        if (
            (ax.freq is not None or ax.inferred_freq is not None)
            and len(self.grouper.binlabels) > len(ax)
            and how is None
        ):

            # let's do an asfreq
            return self.asfreq()

        # we are downsampling
        # we want to call the actual grouper method here
        result = obj.groupby(self.grouper, axis=self.axis).aggregate(how, **kwargs)

        result = self._apply_loffset(result)
        return self._wrap_result(result)

    def _adjust_binner_for_upsample(self, binner):
        """
        Adjust our binner when upsampling.

        The range of a new index should not be outside specified range
        """
        if self.closed == "right":
            binner = binner[1:]
        else:
            binner = binner[:-1]
        return binner

    def _upsample(self, method, limit=None, fill_value=None):
        """
        Parameters
        ----------
        method : string {'backfill', 'bfill', 'pad',
            'ffill', 'asfreq'} method for upsampling
        limit : int, default None
            Maximum size gap to fill when reindexing
        fill_value : scalar, default None
            Value to use for missing values

        See Also
        --------
        .fillna: Fill NA/NaN values using the specified method.

        """
        if self.axis:
            raise AssertionError("axis must be 0")
        if self._from_selection:
            raise ValueError(
                "Upsampling from level= or on= selection "
                "is not supported, use .set_index(...) "
                "to explicitly set index to datetime-like"
            )

        ax = self.ax
        obj = self._selected_obj
        binner = self.binner
        res_index = self._adjust_binner_for_upsample(binner)

        # if we have the same frequency as our axis, then we are equal sampling
        if (
            limit is None
            and to_offset(ax.inferred_freq) == self.freq
            and len(obj) == len(res_index)
        ):
            result = obj.copy()
            result.index = res_index
        else:
            result = obj.reindex(
                res_index, method=method, limit=limit, fill_value=fill_value
            )

        result = self._apply_loffset(result)
        return self._wrap_result(result)

    def _wrap_result(self, result):
        result = super()._wrap_result(result)

        # we may have a different kind that we were asked originally
        # convert if needed
        if self.kind == "period" and not isinstance(result.index, PeriodIndex):
            result.index = result.index.to_period(self.freq)
        return result


class DatetimeIndexResamplerGroupby(_GroupByMixin, DatetimeIndexResampler):
    """
    Provides a resample of a groupby implementation
    """

    @property
    def _constructor(self):
        return DatetimeIndexResampler


class PeriodIndexResampler(DatetimeIndexResampler):
    @property
    def _resampler_for_grouping(self):
        return PeriodIndexResamplerGroupby

    def _get_binner_for_time(self):
        if self.kind == "timestamp":
            return super()._get_binner_for_time()
        return self.groupby._get_period_bins(self.ax)

    def _convert_obj(self, obj: FrameOrSeries) -> FrameOrSeries:
        obj = super()._convert_obj(obj)

        if self._from_selection:
            # see GH 14008, GH 12871
            msg = (
                "Resampling from level= or on= selection "
                "with a PeriodIndex is not currently supported, "
                "use .set_index(...) to explicitly set index"
            )
            raise NotImplementedError(msg)

        if self.loffset is not None:
            # Cannot apply loffset/timedelta to PeriodIndex -> convert to
            # timestamps
            self.kind = "timestamp"

        # convert to timestamp
        if self.kind == "timestamp":
            obj = obj.to_timestamp(how=self.convention)

        return obj

    def _downsample(self, how, **kwargs):
        """
        Downsample the cython defined function.

        Parameters
        ----------
        how : string / cython mapped function
        **kwargs : kw args passed to how function
        """
        # we may need to actually resample as if we are timestamps
        if self.kind == "timestamp":
            return super()._downsample(how, **kwargs)

        how = com.get_cython_func(how) or how
        ax = self.ax

        if is_subperiod(ax.freq, self.freq):
            # Downsampling
            return self._groupby_and_aggregate(how, grouper=self.grouper, **kwargs)
        elif is_superperiod(ax.freq, self.freq):
            if how == "ohlc":
                # GH #13083
                # upsampling to subperiods is handled as an asfreq, which works
                # for pure aggregating/reducing methods
                # OHLC reduces along the time dimension, but creates multiple
                # values for each period -> handle by _groupby_and_aggregate()
                return self._groupby_and_aggregate(how, grouper=self.grouper)
            return self.asfreq()
        elif ax.freq == self.freq:
            return self.asfreq()

        raise IncompatibleFrequency(
            f"Frequency {ax.freq} cannot be resampled to {self.freq}, "
            "as they are not sub or super periods"
        )

    def _upsample(self, method, limit=None, fill_value=None):
        """
        Parameters
        ----------
        method : {'backfill', 'bfill', 'pad', 'ffill'}
            Method for upsampling.
        limit : int, default None
            Maximum size gap to fill when reindexing.
        fill_value : scalar, default None
            Value to use for missing values.

        See Also
        --------
        .fillna: Fill NA/NaN values using the specified method.

        """
        # we may need to actually resample as if we are timestamps
        if self.kind == "timestamp":
            return super()._upsample(method, limit=limit, fill_value=fill_value)

        ax = self.ax
        obj = self.obj
        new_index = self.binner

        # Start vs. end of period
        memb = ax.asfreq(self.freq, how=self.convention)

        # Get the fill indexer
        indexer = memb.get_indexer(new_index, method=method, limit=limit)
        new_obj = _take_new_index(
            obj,
            indexer,
            new_index,
            axis=self.axis,
        )
        return self._wrap_result(new_obj)


class PeriodIndexResamplerGroupby(_GroupByMixin, PeriodIndexResampler):
    """
    Provides a resample of a groupby implementation.
    """

    @property
    def _constructor(self):
        return PeriodIndexResampler


class TimedeltaIndexResampler(DatetimeIndexResampler):
    @property
    def _resampler_for_grouping(self):
        return TimedeltaIndexResamplerGroupby

    def _get_binner_for_time(self):
        return self.groupby._get_time_delta_bins(self.ax)

    def _adjust_binner_for_upsample(self, binner):
        """
        Adjust our binner when upsampling.

        The range of a new index is allowed to be greater than original range
        so we don't need to change the length of a binner, GH 13022
        """
        return binner


class TimedeltaIndexResamplerGroupby(_GroupByMixin, TimedeltaIndexResampler):
    """
    Provides a resample of a groupby implementation.
    """

    @property
    def _constructor(self):
        return TimedeltaIndexResampler


def get_resampler(obj, kind=None, **kwds):
    """
    Create a TimeGrouper and return our resampler.
    """
    tg = TimeGrouper(**kwds)
    return tg._get_resampler(obj, kind=kind)


get_resampler.__doc__ = Resampler.__doc__


def get_resampler_for_grouping(
    groupby, rule, how=None, fill_method=None, limit=None, kind=None, on=None, **kwargs
):
    """
    Return our appropriate resampler when grouping as well.
    """
    # .resample uses 'on' similar to how .groupby uses 'key'
    tg = TimeGrouper(freq=rule, key=on, **kwargs)
    resampler = tg._get_resampler(groupby.obj, kind=kind)
    return resampler._get_resampler_for_grouping(groupby=groupby)


class TimeGrouper(Grouper):
    """
    Custom groupby class for time-interval grouping.

    Parameters
    ----------
    freq : pandas date offset or offset alias for identifying bin edges
    closed : closed end of interval; 'left' or 'right'
    label : interval boundary to use for labeling; 'left' or 'right'
    convention : {'start', 'end', 'e', 's'}
        If axis is PeriodIndex
    """

    _attributes = Grouper._attributes + (
        "closed",
        "label",
        "how",
        "loffset",
        "kind",
        "convention",
        "origin",
        "offset",
    )

    def __init__(
        self,
        freq="Min",
        closed: Literal["left", "right"] | None = None,
        label: str | None = None,
        how="mean",
        axis=0,
        fill_method=None,
        limit=None,
        loffset=None,
        kind: str | None = None,
        convention: str | None = None,
        base: int | None = None,
        origin: str | TimestampConvertibleTypes = "start_day",
        offset: TimedeltaConvertibleTypes | None = None,
        **kwargs,
    ):
        # Check for correctness of the keyword arguments which would
        # otherwise silently use the default if misspelled
        if label not in {None, "left", "right"}:
            raise ValueError(f"Unsupported value {label} for `label`")
        if closed not in {None, "left", "right"}:
            raise ValueError(f"Unsupported value {closed} for `closed`")
        if convention not in {None, "start", "end", "e", "s"}:
            raise ValueError(f"Unsupported value {convention} for `convention`")

        freq = to_offset(freq)

        end_types = {"M", "A", "Q", "BM", "BA", "BQ", "W"}
        rule = freq.rule_code
        if rule in end_types or ("-" in rule and rule[: rule.find("-")] in end_types):
            if closed is None:
                closed = "right"
            if label is None:
                label = "right"
        else:
            # The backward resample sets ``closed`` to ``'right'`` by default
            # since the last value should be considered as the edge point for
            # the last bin. When origin in "end" or "end_day", the value for a
            # specific ``Timestamp`` index stands for the resample result from
            # the current ``Timestamp`` minus ``freq`` to the current
            # ``Timestamp`` with a right close.
            if origin in ["end", "end_day"]:
                if closed is None:
                    closed = "right"
                if label is None:
                    label = "right"
            else:
                if closed is None:
                    closed = "left"
                if label is None:
                    label = "left"

        self.closed = closed
        self.label = label
        self.kind = kind

        self.convention = convention or "E"
        self.convention = self.convention.lower()

        self.how = how
        self.fill_method = fill_method
        self.limit = limit

        if origin in ("epoch", "start", "start_day", "end", "end_day"):
            self.origin = origin
        else:
            try:
                self.origin = Timestamp(origin)
            except (ValueError, TypeError) as err:
                raise ValueError(
                    "'origin' should be equal to 'epoch', 'start', 'start_day', "
                    "'end', 'end_day' or "
                    f"should be a Timestamp convertible type. Got '{origin}' instead."
                ) from err

        try:
            self.offset = Timedelta(offset) if offset is not None else None
        except (ValueError, TypeError) as err:
            raise ValueError(
                "'offset' should be a Timedelta convertible type. "
                f"Got '{offset}' instead."
            ) from err

        # always sort time groupers
        kwargs["sort"] = True

        # Handle deprecated arguments since v1.1.0 of `base` and `loffset` (GH #31809)
        if base is not None and offset is not None:
            raise ValueError("'offset' and 'base' cannot be present at the same time")

        if base and isinstance(freq, Tick):
            # this conversion handle the default behavior of base and the
            # special case of GH #10530. Indeed in case when dealing with
            # a TimedeltaIndex base was treated as a 'pure' offset even though
            # the default behavior of base was equivalent of a modulo on
            # freq_nanos.
            self.offset = Timedelta(base * freq.nanos // freq.n)

        if isinstance(loffset, str):
            loffset = to_offset(loffset)
        self.loffset = loffset

        super().__init__(freq=freq, axis=axis, **kwargs)

    def _get_resampler(self, obj, kind=None):
        """
        Return my resampler or raise if we have an invalid axis.

        Parameters
        ----------
        obj : input object
        kind : string, optional
            'period','timestamp','timedelta' are valid

        Returns
        -------
        a Resampler

        Raises
        ------
        TypeError if incompatible axis

        """
        self._set_grouper(obj)

        ax = self.ax
        if isinstance(ax, DatetimeIndex):
            return DatetimeIndexResampler(obj, groupby=self, kind=kind, axis=self.axis)
        elif isinstance(ax, PeriodIndex) or kind == "period":
            return PeriodIndexResampler(obj, groupby=self, kind=kind, axis=self.axis)
        elif isinstance(ax, TimedeltaIndex):
            return TimedeltaIndexResampler(obj, groupby=self, axis=self.axis)

        raise TypeError(
            "Only valid with DatetimeIndex, "
            "TimedeltaIndex or PeriodIndex, "
            f"but got an instance of '{type(ax).__name__}'"
        )

    def _get_grouper(self, obj, validate: bool = True):
        # create the resampler and return our binner
        r = self._get_resampler(obj)
        return r.binner, r.grouper, r.obj

    def _get_time_bins(self, ax: DatetimeIndex):
        if not isinstance(ax, DatetimeIndex):
            raise TypeError(
                "axis must be a DatetimeIndex, but got "
                f"an instance of {type(ax).__name__}"
            )

        if len(ax) == 0:
            binner = labels = DatetimeIndex(data=[], freq=self.freq, name=ax.name)
            return binner, [], labels

        first, last = _get_timestamp_range_edges(
            ax.min(),
            ax.max(),
            self.freq,
            closed=self.closed,
            origin=self.origin,
            offset=self.offset,
        )
        # GH #12037
        # use first/last directly instead of call replace() on them
        # because replace() will swallow the nanosecond part
        # thus last bin maybe slightly before the end if the end contains
        # nanosecond part and lead to `Values falls after last bin` error
        # GH 25758: If DST lands at midnight (e.g. 'America/Havana'), user feedback
        # has noted that ambiguous=True provides the most sensible result
        binner = labels = date_range(
            freq=self.freq,
            start=first,
            end=last,
            tz=ax.tz,
            name=ax.name,
            ambiguous=True,
            nonexistent="shift_forward",
        )

        ax_values = ax.asi8
        binner, bin_edges = self._adjust_bin_edges(binner, ax_values)

        # general version, knowing nothing about relative frequencies
        bins = lib.generate_bins_dt64(
            ax_values, bin_edges, self.closed, hasnans=ax.hasnans
        )

        if self.closed == "right":
            labels = binner
            if self.label == "right":
                labels = labels[1:]
        elif self.label == "right":
            labels = labels[1:]

        if ax.hasnans:
            binner = binner.insert(0, NaT)
            labels = labels.insert(0, NaT)

        # if we end up with more labels than bins
        # adjust the labels
        # GH4076
        if len(bins) < len(labels):
            labels = labels[: len(bins)]

        return binner, bins, labels

    def _adjust_bin_edges(self, binner, ax_values):
        # Some hacks for > daily data, see #1471, #1458, #1483

        if self.freq != "D" and is_superperiod(self.freq, "D"):
            if self.closed == "right":
                # GH 21459, GH 9119: Adjust the bins relative to the wall time
                bin_edges = binner.tz_localize(None)
                bin_edges = bin_edges + timedelta(1) - Nano(1)
                bin_edges = bin_edges.tz_localize(binner.tz).asi8
            else:
                bin_edges = binner.asi8

            # intraday values on last day
            if bin_edges[-2] > ax_values.max():
                bin_edges = bin_edges[:-1]
                binner = binner[:-1]
        else:
            bin_edges = binner.asi8
        return binner, bin_edges

    def _get_time_delta_bins(self, ax: TimedeltaIndex):
        if not isinstance(ax, TimedeltaIndex):
            raise TypeError(
                "axis must be a TimedeltaIndex, but got "
                f"an instance of {type(ax).__name__}"
            )

        if not len(ax):
            binner = labels = TimedeltaIndex(data=[], freq=self.freq, name=ax.name)
            return binner, [], labels

        start, end = ax.min(), ax.max()
        labels = binner = timedelta_range(
            start=start, end=end, freq=self.freq, name=ax.name
        )

        end_stamps = labels + self.freq
        bins = ax.searchsorted(end_stamps, side="left")

        if self.offset:
            # GH 10530 & 31809
            labels += self.offset
        if self.loffset:
            # GH 33498
            labels += self.loffset

        return binner, bins, labels

    def _get_time_period_bins(self, ax: DatetimeIndex):
        if not isinstance(ax, DatetimeIndex):
            raise TypeError(
                "axis must be a DatetimeIndex, but got "
                f"an instance of {type(ax).__name__}"
            )

        freq = self.freq

        if not len(ax):
            binner = labels = PeriodIndex(data=[], freq=freq, name=ax.name)
            return binner, [], labels

        labels = binner = period_range(start=ax[0], end=ax[-1], freq=freq, name=ax.name)

        end_stamps = (labels + freq).asfreq(freq, "s").to_timestamp()
        if ax.tz:
            end_stamps = end_stamps.tz_localize(ax.tz)
        bins = ax.searchsorted(end_stamps, side="left")

        return binner, bins, labels

    def _get_period_bins(self, ax: PeriodIndex):
        if not isinstance(ax, PeriodIndex):
            raise TypeError(
                "axis must be a PeriodIndex, but got "
                f"an instance of {type(ax).__name__}"
            )

        memb = ax.asfreq(self.freq, how=self.convention)

        # NaT handling as in pandas._lib.lib.generate_bins_dt64()
        nat_count = 0
        if memb.hasnans:
            nat_count = np.sum(memb._isnan)
            memb = memb[~memb._isnan]

        if not len(memb):
            # index contains no valid (non-NaT) values
            bins = np.array([], dtype=np.int64)
            binner = labels = PeriodIndex(data=[], freq=self.freq, name=ax.name)
            if len(ax) > 0:
                # index is all NaT
                binner, bins, labels = _insert_nat_bin(binner, bins, labels, len(ax))
            return binner, bins, labels

        freq_mult = self.freq.n

        start = ax.min().asfreq(self.freq, how=self.convention)
        end = ax.max().asfreq(self.freq, how="end")
        bin_shift = 0

        if isinstance(self.freq, Tick):
            # GH 23882 & 31809: get adjusted bin edge labels with 'origin'
            # and 'origin' support. This call only makes sense if the freq is a
            # Tick since offset and origin are only used in those cases.
            # Not doing this check could create an extra empty bin.
            p_start, end = _get_period_range_edges(
                start,
                end,
                self.freq,
                closed=self.closed,
                origin=self.origin,
                offset=self.offset,
            )

            # Get offset for bin edge (not label edge) adjustment
            start_offset = Period(start, self.freq) - Period(p_start, self.freq)
            # error: Item "Period" of "Union[Period, Any]" has no attribute "n"
            bin_shift = start_offset.n % freq_mult  # type: ignore[union-attr]
            start = p_start

        labels = binner = period_range(
            start=start, end=end, freq=self.freq, name=ax.name
        )

        i8 = memb.asi8

        # when upsampling to subperiods, we need to generate enough bins
        expected_bins_count = len(binner) * freq_mult
        i8_extend = expected_bins_count - (i8[-1] - i8[0])
        rng = np.arange(i8[0], i8[-1] + i8_extend, freq_mult)
        rng += freq_mult
        # adjust bin edge indexes to account for base
        rng -= bin_shift

        # Wrap in PeriodArray for PeriodArray.searchsorted
        prng = type(memb._data)(rng, dtype=memb.dtype)
        bins = memb.searchsorted(prng, side="left")

        if nat_count > 0:
            binner, bins, labels = _insert_nat_bin(binner, bins, labels, nat_count)

        return binner, bins, labels


def _take_new_index(
    obj: FrameOrSeries, indexer: np.ndarray, new_index: Index, axis: int = 0
) -> FrameOrSeries:
    # indexer: np.ndarray[np.intp]

    if isinstance(obj, ABCSeries):
        new_values = algos.take_nd(obj._values, indexer)
        # error: Incompatible return value type (got "Series", expected "FrameOrSeries")
        return obj._constructor(  # type: ignore[return-value]
            new_values, index=new_index, name=obj.name
        )
    elif isinstance(obj, ABCDataFrame):
        if axis == 1:
            raise NotImplementedError("axis 1 is not supported")
        new_mgr = obj._mgr.reindex_indexer(new_axis=new_index, indexer=indexer, axis=1)
        # error: Incompatible return value type
        # (got "DataFrame", expected "FrameOrSeries")
        return obj._constructor(new_mgr)  # type: ignore[return-value]
    else:
        raise ValueError("'obj' should be either a Series or a DataFrame")


def _get_timestamp_range_edges(
    first: Timestamp,
    last: Timestamp,
    freq: BaseOffset,
    closed: Literal["right", "left"] = "left",
    origin="start_day",
    offset: Timedelta | None = None,
) -> tuple[Timestamp, Timestamp]:
    """
    Adjust the `first` Timestamp to the preceding Timestamp that resides on
    the provided offset. Adjust the `last` Timestamp to the following
    Timestamp that resides on the provided offset. Input Timestamps that
    already reside on the offset will be adjusted depending on the type of
    offset and the `closed` parameter.

    Parameters
    ----------
    first : pd.Timestamp
        The beginning Timestamp of the range to be adjusted.
    last : pd.Timestamp
        The ending Timestamp of the range to be adjusted.
    freq : pd.DateOffset
        The dateoffset to which the Timestamps will be adjusted.
    closed : {'right', 'left'}, default "left"
        Which side of bin interval is closed.
    origin : {'epoch', 'start', 'start_day'} or Timestamp, default 'start_day'
        The timestamp on which to adjust the grouping. The timezone of origin must
        match the timezone of the index.
        If a timestamp is not used, these values are also supported:

        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries
    offset : pd.Timedelta, default is None
        An offset timedelta added to the origin.

    Returns
    -------
    A tuple of length 2, containing the adjusted pd.Timestamp objects.
    """
    if isinstance(freq, Tick):
        index_tz = first.tz
        if isinstance(origin, Timestamp) and (origin.tz is None) != (index_tz is None):
            raise ValueError("The origin must have the same timezone as the index.")
        elif origin == "epoch":
            # set the epoch based on the timezone to have similar bins results when
            # resampling on the same kind of indexes on different timezones
            origin = Timestamp("1970-01-01", tz=index_tz)

        if isinstance(freq, Day):
            # _adjust_dates_anchored assumes 'D' means 24H, but first/last
            # might contain a DST transition (23H, 24H, or 25H).
            # So "pretend" the dates are naive when adjusting the endpoints
            first = first.tz_localize(None)
            last = last.tz_localize(None)
            if isinstance(origin, Timestamp):
                origin = origin.tz_localize(None)

        first, last = _adjust_dates_anchored(
            first, last, freq, closed=closed, origin=origin, offset=offset
        )
        if isinstance(freq, Day):
            first = first.tz_localize(index_tz)
            last = last.tz_localize(index_tz)
    else:
        first = first.normalize()
        last = last.normalize()

        if closed == "left":
            first = Timestamp(freq.rollback(first))
        else:
            first = Timestamp(first - freq)

        last = Timestamp(last + freq)

    return first, last


def _get_period_range_edges(
    first: Period,
    last: Period,
    freq: BaseOffset,
    closed: Literal["right", "left"] = "left",
    origin="start_day",
    offset: Timedelta | None = None,
) -> tuple[Period, Period]:
    """
    Adjust the provided `first` and `last` Periods to the respective Period of
    the given offset that encompasses them.

    Parameters
    ----------
    first : pd.Period
        The beginning Period of the range to be adjusted.
    last : pd.Period
        The ending Period of the range to be adjusted.
    freq : pd.DateOffset
        The freq to which the Periods will be adjusted.
    closed : {'right', 'left'}, default "left"
        Which side of bin interval is closed.
    origin : {'epoch', 'start', 'start_day'}, Timestamp, default 'start_day'
        The timestamp on which to adjust the grouping. The timezone of origin must
        match the timezone of the index.

        If a timestamp is not used, these values are also supported:

        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries
    offset : pd.Timedelta, default is None
        An offset timedelta added to the origin.

    Returns
    -------
    A tuple of length 2, containing the adjusted pd.Period objects.
    """
    if not all(isinstance(obj, Period) for obj in [first, last]):
        raise TypeError("'first' and 'last' must be instances of type Period")

    # GH 23882
    first_ts = first.to_timestamp()
    last_ts = last.to_timestamp()
    adjust_first = not freq.is_on_offset(first_ts)
    adjust_last = freq.is_on_offset(last_ts)

    first_ts, last_ts = _get_timestamp_range_edges(
        first_ts, last_ts, freq, closed=closed, origin=origin, offset=offset
    )

    first = (first_ts + int(adjust_first) * freq).to_period(freq)
    last = (last_ts - int(adjust_last) * freq).to_period(freq)
    return first, last


def _insert_nat_bin(
    binner: PeriodIndex, bins: np.ndarray, labels: PeriodIndex, nat_count: int
) -> tuple[PeriodIndex, np.ndarray, PeriodIndex]:
    # NaT handling as in pandas._lib.lib.generate_bins_dt64()
    # shift bins by the number of NaT
    assert nat_count > 0
    bins += nat_count
    bins = np.insert(bins, 0, nat_count)
    binner = binner.insert(0, NaT)
    labels = labels.insert(0, NaT)
    return binner, bins, labels


def _adjust_dates_anchored(
    first: Timestamp,
    last: Timestamp,
    freq: Tick,
    closed: Literal["right", "left"] = "right",
    origin="start_day",
    offset: Timedelta | None = None,
) -> tuple[Timestamp, Timestamp]:
    # First and last offsets should be calculated from the start day to fix an
    # error cause by resampling across multiple days when a one day period is
    # not a multiple of the frequency. See GH 8683
    # To handle frequencies that are not multiple or divisible by a day we let
    # the possibility to define a fixed origin timestamp. See GH 31809
    origin_nanos = 0  # origin == "epoch"
    if origin == "start_day":
        origin_nanos = first.normalize().value
    elif origin == "start":
        origin_nanos = first.value
    elif isinstance(origin, Timestamp):
        origin_nanos = origin.value
    elif origin in ["end", "end_day"]:
        origin = last if origin == "end" else last.ceil("D")
        sub_freq_times = (origin.value - first.value) // freq.nanos
        if closed == "left":
            sub_freq_times += 1
        first = origin - sub_freq_times * freq
        origin_nanos = first.value
    origin_nanos += offset.value if offset else 0

    # GH 10117 & GH 19375. If first and last contain timezone information,
    # Perform the calculation in UTC in order to avoid localizing on an
    # Ambiguous or Nonexistent time.
    first_tzinfo = first.tzinfo
    last_tzinfo = last.tzinfo
    if first_tzinfo is not None:
        first = first.tz_convert("UTC")
    if last_tzinfo is not None:
        last = last.tz_convert("UTC")

    foffset = (first.value - origin_nanos) % freq.nanos
    loffset = (last.value - origin_nanos) % freq.nanos

    if closed == "right":
        if foffset > 0:
            # roll back
            fresult = first.value - foffset
        else:
            fresult = first.value - freq.nanos

        if loffset > 0:
            # roll forward
            lresult = last.value + (freq.nanos - loffset)
        else:
            # already the end of the road
            lresult = last.value
    else:  # closed == 'left'
        if foffset > 0:
            fresult = first.value - foffset
        else:
            # start of the road
            fresult = first.value

        if loffset > 0:
            # roll forward
            lresult = last.value + (freq.nanos - loffset)
        else:
            lresult = last.value + freq.nanos
    fresult = Timestamp(fresult)
    lresult = Timestamp(lresult)
    if first_tzinfo is not None:
        fresult = fresult.tz_localize("UTC").tz_convert(first_tzinfo)
    if last_tzinfo is not None:
        lresult = lresult.tz_localize("UTC").tz_convert(last_tzinfo)
    return fresult, lresult


def asfreq(
    obj: FrameOrSeries,
    freq,
    method=None,
    how=None,
    normalize: bool = False,
    fill_value=None,
) -> FrameOrSeries:
    """
    Utility frequency conversion method for Series/DataFrame.

    See :meth:`pandas.NDFrame.asfreq` for full documentation.
    """
    if isinstance(obj.index, PeriodIndex):
        if method is not None:
            raise NotImplementedError("'method' argument is not supported")

        if how is None:
            how = "E"

        new_obj = obj.copy()
        new_obj.index = obj.index.asfreq(freq, how=how)

    elif len(obj.index) == 0:
        new_obj = obj.copy()

        new_obj.index = _asfreq_compat(obj.index, freq)
    else:
        dti = date_range(obj.index.min(), obj.index.max(), freq=freq)
        dti.name = obj.index.name
        new_obj = obj.reindex(dti, method=method, fill_value=fill_value)
        if normalize:
            new_obj.index = new_obj.index.normalize()

    return new_obj


def _asfreq_compat(index: DatetimeIndex | PeriodIndex | TimedeltaIndex, freq):
    """
    Helper to mimic asfreq on (empty) DatetimeIndex and TimedeltaIndex.

    Parameters
    ----------
    index : PeriodIndex, DatetimeIndex, or TimedeltaIndex
    freq : DateOffset

    Returns
    -------
    same type as index
    """
    if len(index) != 0:
        # This should never be reached, always checked by the caller
        raise ValueError(
            "Can only set arbitrary freq for empty DatetimeIndex or TimedeltaIndex"
        )
    new_index: Index
    if isinstance(index, PeriodIndex):
        new_index = index.asfreq(freq=freq)
    elif isinstance(index, DatetimeIndex):
        new_index = DatetimeIndex([], dtype=index.dtype, freq=freq, name=index.name)
    elif isinstance(index, TimedeltaIndex):
        new_index = TimedeltaIndex([], dtype=index.dtype, freq=freq, name=index.name)
    else:  # pragma: no cover
        raise TypeError(type(index))
    return new_index
