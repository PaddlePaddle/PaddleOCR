# TODO: Use the fact that axis can have units to simplify the process

from __future__ import annotations

import functools
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np

from pandas._libs.tslibs import (
    BaseOffset,
    Period,
    to_offset,
)
from pandas._libs.tslibs.dtypes import FreqGroup
from pandas._typing import FrameOrSeriesUnion

from pandas.core.dtypes.generic import (
    ABCDatetimeIndex,
    ABCPeriodIndex,
    ABCTimedeltaIndex,
)

from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.converter import (
    TimeSeries_DateFormatter,
    TimeSeries_DateLocator,
    TimeSeries_TimedeltaFormatter,
)
from pandas.tseries.frequencies import (
    get_period_alias,
    is_subperiod,
    is_superperiod,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from pandas import (
        DatetimeIndex,
        Index,
        Series,
    )

# ---------------------------------------------------------------------
# Plotting functions and monkey patches


def maybe_resample(series: Series, ax: Axes, kwargs):
    # resample against axes freq if necessary
    freq, ax_freq = _get_freq(ax, series)

    if freq is None:  # pragma: no cover
        raise ValueError("Cannot use dynamic axis without frequency info")

    # Convert DatetimeIndex to PeriodIndex
    if isinstance(series.index, ABCDatetimeIndex):
        series = series.to_period(freq=freq)

    if ax_freq is not None and freq != ax_freq:
        if is_superperiod(freq, ax_freq):  # upsample input
            series = series.copy()
            # error: "Index" has no attribute "asfreq"
            series.index = series.index.asfreq(  # type: ignore[attr-defined]
                ax_freq, how="s"
            )
            freq = ax_freq
        elif _is_sup(freq, ax_freq):  # one is weekly
            how = kwargs.pop("how", "last")
            series = getattr(series.resample("D"), how)().dropna()
            series = getattr(series.resample(ax_freq), how)().dropna()
            freq = ax_freq
        elif is_subperiod(freq, ax_freq) or _is_sub(freq, ax_freq):
            _upsample_others(ax, freq, kwargs)
        else:  # pragma: no cover
            raise ValueError("Incompatible frequency conversion")
    return freq, series


def _is_sub(f1: str, f2: str) -> bool:
    return (f1.startswith("W") and is_subperiod("D", f2)) or (
        f2.startswith("W") and is_subperiod(f1, "D")
    )


def _is_sup(f1: str, f2: str) -> bool:
    return (f1.startswith("W") and is_superperiod("D", f2)) or (
        f2.startswith("W") and is_superperiod(f1, "D")
    )


def _upsample_others(ax: Axes, freq, kwargs):
    legend = ax.get_legend()
    lines, labels = _replot_ax(ax, freq, kwargs)
    _replot_ax(ax, freq, kwargs)

    other_ax = None
    if hasattr(ax, "left_ax"):
        other_ax = ax.left_ax
    if hasattr(ax, "right_ax"):
        other_ax = ax.right_ax

    if other_ax is not None:
        rlines, rlabels = _replot_ax(other_ax, freq, kwargs)
        lines.extend(rlines)
        labels.extend(rlabels)

    if legend is not None and kwargs.get("legend", True) and len(lines) > 0:
        title = legend.get_title().get_text()
        if title == "None":
            title = None
        ax.legend(lines, labels, loc="best", title=title)


def _replot_ax(ax: Axes, freq, kwargs):
    data = getattr(ax, "_plot_data", None)

    # clear current axes and data
    ax._plot_data = []
    ax.clear()

    decorate_axes(ax, freq, kwargs)

    lines = []
    labels = []
    if data is not None:
        for series, plotf, kwds in data:
            series = series.copy()
            idx = series.index.asfreq(freq, how="S")
            series.index = idx
            ax._plot_data.append((series, plotf, kwds))

            # for tsplot
            if isinstance(plotf, str):
                from pandas.plotting._matplotlib import PLOT_CLASSES

                plotf = PLOT_CLASSES[plotf]._plot

            lines.append(plotf(ax, series.index._mpl_repr(), series.values, **kwds)[0])
            labels.append(pprint_thing(series.name))

    return lines, labels


def decorate_axes(ax: Axes, freq, kwargs):
    """Initialize axes for time-series plotting"""
    if not hasattr(ax, "_plot_data"):
        ax._plot_data = []

    ax.freq = freq
    xaxis = ax.get_xaxis()
    xaxis.freq = freq
    if not hasattr(ax, "legendlabels"):
        ax.legendlabels = [kwargs.get("label", None)]
    else:
        ax.legendlabels.append(kwargs.get("label", None))
    ax.view_interval = None
    ax.date_axis_info = None


def _get_ax_freq(ax: Axes):
    """
    Get the freq attribute of the ax object if set.
    Also checks shared axes (eg when using secondary yaxis, sharex=True
    or twinx)
    """
    ax_freq = getattr(ax, "freq", None)
    if ax_freq is None:
        # check for left/right ax in case of secondary yaxis
        if hasattr(ax, "left_ax"):
            ax_freq = getattr(ax.left_ax, "freq", None)
        elif hasattr(ax, "right_ax"):
            ax_freq = getattr(ax.right_ax, "freq", None)
    if ax_freq is None:
        # check if a shared ax (sharex/twinx) has already freq set
        shared_axes = ax.get_shared_x_axes().get_siblings(ax)
        if len(shared_axes) > 1:
            for shared_ax in shared_axes:
                ax_freq = getattr(shared_ax, "freq", None)
                if ax_freq is not None:
                    break
    return ax_freq


def _get_period_alias(freq) -> str | None:
    freqstr = to_offset(freq).rule_code

    freq = get_period_alias(freqstr)
    return freq


def _get_freq(ax: Axes, series: Series):
    # get frequency from data
    freq = getattr(series.index, "freq", None)
    if freq is None:
        freq = getattr(series.index, "inferred_freq", None)
        freq = to_offset(freq)

    ax_freq = _get_ax_freq(ax)

    # use axes freq if no data freq
    if freq is None:
        freq = ax_freq

    # get the period frequency
    freq = _get_period_alias(freq)
    return freq, ax_freq


def use_dynamic_x(ax: Axes, data: FrameOrSeriesUnion) -> bool:
    freq = _get_index_freq(data.index)
    ax_freq = _get_ax_freq(ax)

    if freq is None:  # convert irregular if axes has freq info
        freq = ax_freq
    else:  # do not use tsplot if irregular was plotted first
        if (ax_freq is None) and (len(ax.get_lines()) > 0):
            return False

    if freq is None:
        return False

    freq = _get_period_alias(freq)

    if freq is None:
        return False

    # FIXME: hack this for 0.10.1, creating more technical debt...sigh
    if isinstance(data.index, ABCDatetimeIndex):
        base = to_offset(freq)._period_dtype_code
        x = data.index
        if base <= FreqGroup.FR_DAY.value:
            return x[:1].is_normalized
        return Period(x[0], freq).to_timestamp().tz_localize(x.tz) == x[0]
    return True


def _get_index_freq(index: Index) -> BaseOffset | None:
    freq = getattr(index, "freq", None)
    if freq is None:
        freq = getattr(index, "inferred_freq", None)
        if freq == "B":
            # error: "Index" has no attribute "dayofweek"
            weekdays = np.unique(index.dayofweek)  # type: ignore[attr-defined]
            if (5 in weekdays) or (6 in weekdays):
                freq = None

    freq = to_offset(freq)
    return freq


def maybe_convert_index(ax: Axes, data):
    # tsplot converts automatically, but don't want to convert index
    # over and over for DataFrames
    if isinstance(data.index, (ABCDatetimeIndex, ABCPeriodIndex)):
        freq = data.index.freq

        if freq is None:
            # We only get here for DatetimeIndex
            data.index = cast("DatetimeIndex", data.index)
            freq = data.index.inferred_freq
            freq = to_offset(freq)

        if freq is None:
            freq = _get_ax_freq(ax)

        if freq is None:
            raise ValueError("Could not get frequency alias for plotting")

        freq = _get_period_alias(freq)

        if isinstance(data.index, ABCDatetimeIndex):
            data = data.tz_localize(None).to_period(freq=freq)
        elif isinstance(data.index, ABCPeriodIndex):
            data.index = data.index.asfreq(freq=freq)
    return data


# Patch methods for subplot. Only format_dateaxis is currently used.
# Do we need the rest for convenience?


def _format_coord(freq, t, y) -> str:
    time_period = Period(ordinal=int(t), freq=freq)
    return f"t = {time_period}  y = {y:8f}"


def format_dateaxis(subplot, freq, index):
    """
    Pretty-formats the date axis (x-axis).

    Major and minor ticks are automatically set for the frequency of the
    current underlying series.  As the dynamic mode is activated by
    default, changing the limits of the x axis will intelligently change
    the positions of the ticks.
    """
    from matplotlib import pylab

    # handle index specific formatting
    # Note: DatetimeIndex does not use this
    # interface. DatetimeIndex uses matplotlib.date directly
    if isinstance(index, ABCPeriodIndex):

        majlocator = TimeSeries_DateLocator(
            freq, dynamic_mode=True, minor_locator=False, plot_obj=subplot
        )
        minlocator = TimeSeries_DateLocator(
            freq, dynamic_mode=True, minor_locator=True, plot_obj=subplot
        )
        subplot.xaxis.set_major_locator(majlocator)
        subplot.xaxis.set_minor_locator(minlocator)

        majformatter = TimeSeries_DateFormatter(
            freq, dynamic_mode=True, minor_locator=False, plot_obj=subplot
        )
        minformatter = TimeSeries_DateFormatter(
            freq, dynamic_mode=True, minor_locator=True, plot_obj=subplot
        )
        subplot.xaxis.set_major_formatter(majformatter)
        subplot.xaxis.set_minor_formatter(minformatter)

        # x and y coord info
        subplot.format_coord = functools.partial(_format_coord, freq)

    elif isinstance(index, ABCTimedeltaIndex):
        subplot.xaxis.set_major_formatter(TimeSeries_TimedeltaFormatter())
    else:
        raise TypeError("index type not supported")

    pylab.draw_if_interactive()
