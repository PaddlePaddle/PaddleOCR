from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Hashable,
)
import warnings

from matplotlib.artist import Artist
import numpy as np

from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.common import (
    is_categorical_dtype,
    is_extension_array_dtype,
    is_float,
    is_float_dtype,
    is_hashable,
    is_integer,
    is_integer_dtype,
    is_iterator,
    is_list_like,
    is_number,
    is_numeric_dtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCMultiIndex,
    ABCPeriodIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    isna,
    notna,
)

import pandas.core.common as com

from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.compat import mpl_ge_3_0_0
from pandas.plotting._matplotlib.converter import register_pandas_matplotlib_converters
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.timeseries import (
    decorate_axes,
    format_dateaxis,
    maybe_convert_index,
    maybe_resample,
    use_dynamic_x,
)
from pandas.plotting._matplotlib.tools import (
    create_subplots,
    flatten_axes,
    format_date_labels,
    get_all_lines,
    get_xlim,
    handle_shared_axes,
    table,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.axis import Axis


def _color_in_style(style: str) -> bool:
    """
    Check if there is a color letter in the style string.
    """
    from matplotlib.colors import BASE_COLORS

    return not set(BASE_COLORS).isdisjoint(style)


class MPLPlot:
    """
    Base class for assembling a pandas plot using matplotlib

    Parameters
    ----------
    data :

    """

    @property
    def _kind(self):
        """Specify kind str. Must be overridden in child class"""
        raise NotImplementedError

    _layout_type = "vertical"
    _default_rot = 0
    orientation: str | None = None

    axes: np.ndarray  # of Axes objects

    def __init__(
        self,
        data,
        kind=None,
        by=None,
        subplots=False,
        sharex=None,
        sharey=False,
        use_index=True,
        figsize=None,
        grid=None,
        legend=True,
        rot=None,
        ax=None,
        fig=None,
        title=None,
        xlim=None,
        ylim=None,
        xticks=None,
        yticks=None,
        xlabel: Hashable | None = None,
        ylabel: Hashable | None = None,
        sort_columns=False,
        fontsize=None,
        secondary_y=False,
        colormap=None,
        table=False,
        layout=None,
        include_bool=False,
        **kwds,
    ):

        import matplotlib.pyplot as plt

        self.data = data
        self.by = by

        self.kind = kind

        self.sort_columns = sort_columns

        self.subplots = subplots

        if sharex is None:
            if ax is None:
                self.sharex = True
            else:
                # if we get an axis, the users should do the visibility
                # setting...
                self.sharex = False
        else:
            self.sharex = sharex

        self.sharey = sharey
        self.figsize = figsize
        self.layout = layout

        self.xticks = xticks
        self.yticks = yticks
        self.xlim = xlim
        self.ylim = ylim
        self.title = title
        self.use_index = use_index
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.fontsize = fontsize

        if rot is not None:
            self.rot = rot
            # need to know for format_date_labels since it's rotated to 30 by
            # default
            self._rot_set = True
        else:
            self._rot_set = False
            self.rot = self._default_rot

        if grid is None:
            grid = False if secondary_y else plt.rcParams["axes.grid"]

        self.grid = grid
        self.legend = legend
        self.legend_handles: list[Artist] = []
        self.legend_labels: list[Hashable] = []

        self.logx = kwds.pop("logx", False)
        self.logy = kwds.pop("logy", False)
        self.loglog = kwds.pop("loglog", False)
        self.label = kwds.pop("label", None)
        self.style = kwds.pop("style", None)
        self.mark_right = kwds.pop("mark_right", True)
        self.stacked = kwds.pop("stacked", False)

        self.ax = ax
        self.fig = fig
        self.axes = np.array([], dtype=object)  # "real" version get set in `generate`

        # parse errorbar input if given
        xerr = kwds.pop("xerr", None)
        yerr = kwds.pop("yerr", None)
        self.errors = {
            kw: self._parse_errorbars(kw, err)
            for kw, err in zip(["xerr", "yerr"], [xerr, yerr])
        }

        if not isinstance(secondary_y, (bool, tuple, list, np.ndarray, ABCIndex)):
            secondary_y = [secondary_y]
        self.secondary_y = secondary_y

        # ugly TypeError if user passes matplotlib's `cmap` name.
        # Probably better to accept either.
        if "cmap" in kwds and colormap:
            raise TypeError("Only specify one of `cmap` and `colormap`.")
        elif "cmap" in kwds:
            self.colormap = kwds.pop("cmap")
        else:
            self.colormap = colormap

        self.table = table
        self.include_bool = include_bool

        self.kwds = kwds

        self._validate_color_args()

    def _validate_color_args(self):
        if (
            "color" in self.kwds
            and self.nseries == 1
            and not is_list_like(self.kwds["color"])
        ):
            # support series.plot(color='green')
            self.kwds["color"] = [self.kwds["color"]]

        if (
            "color" in self.kwds
            and isinstance(self.kwds["color"], tuple)
            and self.nseries == 1
            and len(self.kwds["color"]) in (3, 4)
        ):
            # support RGB and RGBA tuples in series plot
            self.kwds["color"] = [self.kwds["color"]]

        if (
            "color" in self.kwds or "colors" in self.kwds
        ) and self.colormap is not None:
            warnings.warn(
                "'color' and 'colormap' cannot be used simultaneously. Using 'color'"
            )

        if "color" in self.kwds and self.style is not None:
            if is_list_like(self.style):
                styles = self.style
            else:
                styles = [self.style]
            # need only a single match
            for s in styles:
                if _color_in_style(s):
                    raise ValueError(
                        "Cannot pass 'style' string with a color symbol and "
                        "'color' keyword argument. Please use one or the "
                        "other or pass 'style' without a color symbol"
                    )

    def _iter_data(self, data=None, keep_index=False, fillna=None):
        if data is None:
            data = self.data
        if fillna is not None:
            data = data.fillna(fillna)

        for col, values in data.items():
            if keep_index is True:
                yield col, values
            else:
                yield col, values.values

    @property
    def nseries(self) -> int:
        if self.data.ndim == 1:
            return 1
        else:
            return self.data.shape[1]

    def draw(self):
        self.plt.draw_if_interactive()

    def generate(self):
        self._args_adjust()
        self._compute_plot_data()
        self._setup_subplots()
        self._make_plot()
        self._add_table()
        self._make_legend()
        self._adorn_subplots()

        for ax in self.axes:
            self._post_plot_logic_common(ax, self.data)
            self._post_plot_logic(ax, self.data)

    def _args_adjust(self):
        pass

    def _has_plotted_object(self, ax: Axes) -> bool:
        """check whether ax has data"""
        return len(ax.lines) != 0 or len(ax.artists) != 0 or len(ax.containers) != 0

    def _maybe_right_yaxis(self, ax: Axes, axes_num):
        if not self.on_right(axes_num):
            # secondary axes may be passed via ax kw
            return self._get_ax_layer(ax)

        if hasattr(ax, "right_ax"):
            # if it has right_ax property, ``ax`` must be left axes
            return ax.right_ax
        elif hasattr(ax, "left_ax"):
            # if it has left_ax property, ``ax`` must be right axes
            return ax
        else:
            # otherwise, create twin axes
            orig_ax, new_ax = ax, ax.twinx()
            # TODO: use Matplotlib public API when available
            new_ax._get_lines = orig_ax._get_lines
            new_ax._get_patches_for_fill = orig_ax._get_patches_for_fill
            orig_ax.right_ax, new_ax.left_ax = new_ax, orig_ax

            if not self._has_plotted_object(orig_ax):  # no data on left y
                orig_ax.get_yaxis().set_visible(False)

            if self.logy is True or self.loglog is True:
                new_ax.set_yscale("log")
            elif self.logy == "sym" or self.loglog == "sym":
                new_ax.set_yscale("symlog")
            return new_ax

    def _setup_subplots(self):
        if self.subplots:
            fig, axes = create_subplots(
                naxes=self.nseries,
                sharex=self.sharex,
                sharey=self.sharey,
                figsize=self.figsize,
                ax=self.ax,
                layout=self.layout,
                layout_type=self._layout_type,
            )
        else:
            if self.ax is None:
                fig = self.plt.figure(figsize=self.figsize)
                axes = fig.add_subplot(111)
            else:
                fig = self.ax.get_figure()
                if self.figsize is not None:
                    fig.set_size_inches(self.figsize)
                axes = self.ax

        axes = flatten_axes(axes)

        valid_log = {False, True, "sym", None}
        input_log = {self.logx, self.logy, self.loglog}
        if input_log - valid_log:
            invalid_log = next(iter(input_log - valid_log))
            raise ValueError(
                f"Boolean, None and 'sym' are valid options, '{invalid_log}' is given."
            )

        if self.logx is True or self.loglog is True:
            [a.set_xscale("log") for a in axes]
        elif self.logx == "sym" or self.loglog == "sym":
            [a.set_xscale("symlog") for a in axes]

        if self.logy is True or self.loglog is True:
            [a.set_yscale("log") for a in axes]
        elif self.logy == "sym" or self.loglog == "sym":
            [a.set_yscale("symlog") for a in axes]

        self.fig = fig
        self.axes = axes

    @property
    def result(self):
        """
        Return result axes
        """
        if self.subplots:
            if self.layout is not None and not is_list_like(self.ax):
                return self.axes.reshape(*self.layout)
            else:
                return self.axes
        else:
            sec_true = isinstance(self.secondary_y, bool) and self.secondary_y
            all_sec = (
                is_list_like(self.secondary_y) and len(self.secondary_y) == self.nseries
            )
            if sec_true or all_sec:
                # if all data is plotted on secondary, return right axes
                return self._get_ax_layer(self.axes[0], primary=False)
            else:
                return self.axes[0]

    def _convert_to_ndarray(self, data):
        # GH31357: categorical columns are processed separately
        if is_categorical_dtype(data):
            return data

        # GH32073: cast to float if values contain nulled integers
        if (
            is_integer_dtype(data.dtype) or is_float_dtype(data.dtype)
        ) and is_extension_array_dtype(data.dtype):
            return data.to_numpy(dtype="float", na_value=np.nan)

        # GH25587: cast ExtensionArray of pandas (IntegerArray, etc.) to
        # np.ndarray before plot.
        if len(data) > 0:
            return np.asarray(data)

        return data

    def _compute_plot_data(self):
        data = self.data

        if isinstance(data, ABCSeries):
            label = self.label
            if label is None and data.name is None:
                label = "None"
            data = data.to_frame(name=label)

        # GH16953, _convert is needed as fallback, for ``Series``
        # with ``dtype == object``
        data = data._convert(datetime=True, timedelta=True)
        include_type = [np.number, "datetime", "datetimetz", "timedelta"]

        # GH23719, allow plotting boolean
        if self.include_bool is True:
            include_type.append(np.bool_)

        # GH22799, exclude datetime-like type for boxplot
        exclude_type = None
        if self._kind == "box":
            # TODO: change after solving issue 27881
            include_type = [np.number]
            exclude_type = ["timedelta"]

        # GH 18755, include object and category type for scatter plot
        if self._kind == "scatter":
            include_type.extend(["object", "category"])

        numeric_data = data.select_dtypes(include=include_type, exclude=exclude_type)

        try:
            is_empty = numeric_data.columns.empty
        except AttributeError:
            is_empty = not len(numeric_data)

        # no non-numeric frames or series allowed
        if is_empty:
            raise TypeError("no numeric data to plot")

        self.data = numeric_data.apply(self._convert_to_ndarray)

    def _make_plot(self):
        raise AbstractMethodError(self)

    def _add_table(self):
        if self.table is False:
            return
        elif self.table is True:
            data = self.data.transpose()
        else:
            data = self.table
        ax = self._get_ax(0)
        table(ax, data)

    def _post_plot_logic_common(self, ax, data):
        """Common post process for each axes"""
        if self.orientation == "vertical" or self.orientation is None:
            self._apply_axis_properties(ax.xaxis, rot=self.rot, fontsize=self.fontsize)
            self._apply_axis_properties(ax.yaxis, fontsize=self.fontsize)

            if hasattr(ax, "right_ax"):
                self._apply_axis_properties(ax.right_ax.yaxis, fontsize=self.fontsize)

        elif self.orientation == "horizontal":
            self._apply_axis_properties(ax.yaxis, rot=self.rot, fontsize=self.fontsize)
            self._apply_axis_properties(ax.xaxis, fontsize=self.fontsize)

            if hasattr(ax, "right_ax"):
                self._apply_axis_properties(ax.right_ax.yaxis, fontsize=self.fontsize)
        else:  # pragma no cover
            raise ValueError

    def _post_plot_logic(self, ax, data):
        """Post process for each axes. Overridden in child classes"""
        pass

    def _adorn_subplots(self):
        """Common post process unrelated to data"""
        if len(self.axes) > 0:
            all_axes = self._get_subplots()
            nrows, ncols = self._get_axes_layout()
            handle_shared_axes(
                axarr=all_axes,
                nplots=len(all_axes),
                naxes=nrows * ncols,
                nrows=nrows,
                ncols=ncols,
                sharex=self.sharex,
                sharey=self.sharey,
            )

        for ax in self.axes:
            if self.yticks is not None:
                ax.set_yticks(self.yticks)

            if self.xticks is not None:
                ax.set_xticks(self.xticks)

            if self.ylim is not None:
                ax.set_ylim(self.ylim)

            if self.xlim is not None:
                ax.set_xlim(self.xlim)

            # GH9093, currently Pandas does not show ylabel, so if users provide
            # ylabel will set it as ylabel in the plot.
            if self.ylabel is not None:
                ax.set_ylabel(pprint_thing(self.ylabel))

            ax.grid(self.grid)

        if self.title:
            if self.subplots:
                if is_list_like(self.title):
                    if len(self.title) != self.nseries:
                        raise ValueError(
                            "The length of `title` must equal the number "
                            "of columns if using `title` of type `list` "
                            "and `subplots=True`.\n"
                            f"length of title = {len(self.title)}\n"
                            f"number of columns = {self.nseries}"
                        )

                    for (ax, title) in zip(self.axes, self.title):
                        ax.set_title(title)
                else:
                    self.fig.suptitle(self.title)
            else:
                if is_list_like(self.title):
                    msg = (
                        "Using `title` of type `list` is not supported "
                        "unless `subplots=True` is passed"
                    )
                    raise ValueError(msg)
                self.axes[0].set_title(self.title)

    def _apply_axis_properties(self, axis: Axis, rot=None, fontsize=None):
        """
        Tick creation within matplotlib is reasonably expensive and is
        internally deferred until accessed as Ticks are created/destroyed
        multiple times per draw. It's therefore beneficial for us to avoid
        accessing unless we will act on the Tick.
        """
        if rot is not None or fontsize is not None:
            # rot=0 is a valid setting, hence the explicit None check
            labels = axis.get_majorticklabels() + axis.get_minorticklabels()
            for label in labels:
                if rot is not None:
                    label.set_rotation(rot)
                if fontsize is not None:
                    label.set_fontsize(fontsize)

    @property
    def legend_title(self) -> str | None:
        if not isinstance(self.data.columns, ABCMultiIndex):
            name = self.data.columns.name
            if name is not None:
                name = pprint_thing(name)
            return name
        else:
            stringified = map(pprint_thing, self.data.columns.names)
            return ",".join(stringified)

    def _mark_right_label(self, label: str, index: int) -> str:
        """
        Append ``(right)`` to the label of a line if it's plotted on the right axis.

        Note that ``(right)`` is only appended when ``subplots=False``.
        """
        if not self.subplots and self.mark_right and self.on_right(index):
            label += " (right)"
        return label

    def _append_legend_handles_labels(self, handle: Artist, label: str) -> None:
        """
        Append current handle and label to ``legend_handles`` and ``legend_labels``.

        These will be used to make the legend.
        """
        self.legend_handles.append(handle)
        self.legend_labels.append(label)

    def _make_legend(self):
        ax, leg = self._get_ax_legend(self.axes[0])

        handles = []
        labels = []
        title = ""

        if not self.subplots:
            if leg is not None:
                title = leg.get_title().get_text()
                # Replace leg.LegendHandles because it misses marker info
                handles = leg.legendHandles
                labels = [x.get_text() for x in leg.get_texts()]

            if self.legend:
                if self.legend == "reverse":
                    # error: Incompatible types in assignment (expression has type
                    # "Iterator[Any]", variable has type "List[Any]")
                    self.legend_handles = reversed(  # type: ignore[assignment]
                        self.legend_handles
                    )
                    # error: Incompatible types in assignment (expression has type
                    # "Iterator[Hashable]", variable has type
                    # "List[Hashable]")
                    self.legend_labels = reversed(  # type: ignore[assignment]
                        self.legend_labels
                    )

                handles += self.legend_handles
                labels += self.legend_labels

                if self.legend_title is not None:
                    title = self.legend_title

            if len(handles) > 0:
                ax.legend(handles, labels, loc="best", title=title)

        elif self.subplots and self.legend:
            for ax in self.axes:
                if ax.get_visible():
                    ax.legend(loc="best")

    def _get_ax_legend(self, ax: Axes):
        """
        Take in axes and return ax and legend under different scenarios
        """
        leg = ax.get_legend()

        other_ax = getattr(ax, "left_ax", None) or getattr(ax, "right_ax", None)
        other_leg = None
        if other_ax is not None:
            other_leg = other_ax.get_legend()
        if leg is None and other_leg is not None:
            leg = other_leg
            ax = other_ax
        return ax, leg

    @cache_readonly
    def plt(self):
        import matplotlib.pyplot as plt

        return plt

    _need_to_set_index = False

    def _get_xticks(self, convert_period: bool = False):
        index = self.data.index
        is_datetype = index.inferred_type in ("datetime", "date", "datetime64", "time")

        if self.use_index:
            if convert_period and isinstance(index, ABCPeriodIndex):
                self.data = self.data.reindex(index=index.sort_values())
                x = self.data.index.to_timestamp()._mpl_repr()
            elif index.is_numeric():
                """
                Matplotlib supports numeric values or datetime objects as
                xaxis values. Taking LBYL approach here, by the time
                matplotlib raises exception when using non numeric/datetime
                values for xaxis, several actions are already taken by plt.
                """
                x = index._mpl_repr()
            elif is_datetype:
                self.data = self.data[notna(self.data.index)]
                self.data = self.data.sort_index()
                x = self.data.index._mpl_repr()
            else:
                self._need_to_set_index = True
                x = list(range(len(index)))
        else:
            x = list(range(len(index)))

        return x

    @classmethod
    @register_pandas_matplotlib_converters
    def _plot(cls, ax: Axes, x, y, style=None, is_errorbar: bool = False, **kwds):
        mask = isna(y)
        if mask.any():
            y = np.ma.array(y)
            y = np.ma.masked_where(mask, y)

        if isinstance(x, ABCIndex):
            x = x._mpl_repr()

        if is_errorbar:
            if "xerr" in kwds:
                kwds["xerr"] = np.array(kwds.get("xerr"))
            if "yerr" in kwds:
                kwds["yerr"] = np.array(kwds.get("yerr"))
            return ax.errorbar(x, y, **kwds)
        else:
            # prevent style kwarg from going to errorbar, where it is unsupported
            args = (x, y, style) if style is not None else (x, y)
            return ax.plot(*args, **kwds)

    def _get_index_name(self) -> str | None:
        if isinstance(self.data.index, ABCMultiIndex):
            name = self.data.index.names
            if com.any_not_none(*name):
                name = ",".join(pprint_thing(x) for x in name)
            else:
                name = None
        else:
            name = self.data.index.name
            if name is not None:
                name = pprint_thing(name)

        # GH 9093, override the default xlabel if xlabel is provided.
        if self.xlabel is not None:
            name = pprint_thing(self.xlabel)

        return name

    @classmethod
    def _get_ax_layer(cls, ax, primary=True):
        """get left (primary) or right (secondary) axes"""
        if primary:
            return getattr(ax, "left_ax", ax)
        else:
            return getattr(ax, "right_ax", ax)

    def _get_ax(self, i: int):
        # get the twinx ax if appropriate
        if self.subplots:
            ax = self.axes[i]
            ax = self._maybe_right_yaxis(ax, i)
            self.axes[i] = ax
        else:
            ax = self.axes[0]
            ax = self._maybe_right_yaxis(ax, i)

        ax.get_yaxis().set_visible(True)
        return ax

    @classmethod
    def get_default_ax(cls, ax):
        import matplotlib.pyplot as plt

        if ax is None and len(plt.get_fignums()) > 0:
            with plt.rc_context():
                ax = plt.gca()
            ax = cls._get_ax_layer(ax)

    def on_right(self, i):
        if isinstance(self.secondary_y, bool):
            return self.secondary_y

        if isinstance(self.secondary_y, (tuple, list, np.ndarray, ABCIndex)):
            return self.data.columns[i] in self.secondary_y

    def _apply_style_colors(self, colors, kwds, col_num, label):
        """
        Manage style and color based on column number and its label.
        Returns tuple of appropriate style and kwds which "color" may be added.
        """
        style = None
        if self.style is not None:
            if isinstance(self.style, list):
                try:
                    style = self.style[col_num]
                except IndexError:
                    pass
            elif isinstance(self.style, dict):
                style = self.style.get(label, style)
            else:
                style = self.style

        has_color = "color" in kwds or self.colormap is not None
        nocolor_style = style is None or not _color_in_style(style)
        if (has_color or self.subplots) and nocolor_style:
            if isinstance(colors, dict):
                kwds["color"] = colors[label]
            else:
                kwds["color"] = colors[col_num % len(colors)]
        return style, kwds

    def _get_colors(self, num_colors=None, color_kwds="color"):
        if num_colors is None:
            num_colors = self.nseries

        return get_standard_colors(
            num_colors=num_colors,
            colormap=self.colormap,
            color=self.kwds.get(color_kwds),
        )

    def _parse_errorbars(self, label, err):
        """
        Look for error keyword arguments and return the actual errorbar data
        or return the error DataFrame/dict

        Error bars can be specified in several ways:
            Series: the user provides a pandas.Series object of the same
                    length as the data
            ndarray: provides a np.ndarray of the same length as the data
            DataFrame/dict: error values are paired with keys matching the
                    key in the plotted DataFrame
            str: the name of the column within the plotted DataFrame

        Asymmetrical error bars are also supported, however raw error values
        must be provided in this case. For a ``N`` length :class:`Series`, a
        ``2xN`` array should be provided indicating lower and upper (or left
        and right) errors. For a ``MxN`` :class:`DataFrame`, asymmetrical errors
        should be in a ``Mx2xN`` array.
        """
        if err is None:
            return None

        def match_labels(data, e):
            e = e.reindex(data.index)
            return e

        # key-matched DataFrame
        if isinstance(err, ABCDataFrame):

            err = match_labels(self.data, err)
        # key-matched dict
        elif isinstance(err, dict):
            pass

        # Series of error values
        elif isinstance(err, ABCSeries):
            # broadcast error series across data
            err = match_labels(self.data, err)
            err = np.atleast_2d(err)
            err = np.tile(err, (self.nseries, 1))

        # errors are a column in the dataframe
        elif isinstance(err, str):
            evalues = self.data[err].values
            self.data = self.data[self.data.columns.drop(err)]
            err = np.atleast_2d(evalues)
            err = np.tile(err, (self.nseries, 1))

        elif is_list_like(err):
            if is_iterator(err):
                err = np.atleast_2d(list(err))
            else:
                # raw error values
                err = np.atleast_2d(err)

            err_shape = err.shape

            # asymmetrical error bars
            if isinstance(self.data, ABCSeries) and err_shape[0] == 2:
                err = np.expand_dims(err, 0)
                err_shape = err.shape
                if err_shape[2] != len(self.data):
                    raise ValueError(
                        "Asymmetrical error bars should be provided "
                        f"with the shape (2, {len(self.data)})"
                    )
            elif isinstance(self.data, ABCDataFrame) and err.ndim == 3:
                if (
                    (err_shape[0] != self.nseries)
                    or (err_shape[1] != 2)
                    or (err_shape[2] != len(self.data))
                ):
                    raise ValueError(
                        "Asymmetrical error bars should be provided "
                        f"with the shape ({self.nseries}, 2, {len(self.data)})"
                    )

            # broadcast errors to each data series
            if len(err) == 1:
                err = np.tile(err, (self.nseries, 1))

        elif is_number(err):
            err = np.tile([err], (self.nseries, len(self.data)))

        else:
            msg = f"No valid {label} detected"
            raise ValueError(msg)

        return err

    def _get_errorbars(self, label=None, index=None, xerr=True, yerr=True):
        errors = {}

        for kw, flag in zip(["xerr", "yerr"], [xerr, yerr]):
            if flag:
                err = self.errors[kw]
                # user provided label-matched dataframe of errors
                if isinstance(err, (ABCDataFrame, dict)):
                    if label is not None and label in err.keys():
                        err = err[label]
                    else:
                        err = None
                elif index is not None and err is not None:
                    err = err[index]

                if err is not None:
                    errors[kw] = err
        return errors

    def _get_subplots(self):
        from matplotlib.axes import Subplot

        return [
            ax for ax in self.axes[0].get_figure().get_axes() if isinstance(ax, Subplot)
        ]

    def _get_axes_layout(self) -> tuple[int, int]:
        axes = self._get_subplots()
        x_set = set()
        y_set = set()
        for ax in axes:
            # check axes coordinates to estimate layout
            points = ax.get_position().get_points()
            x_set.add(points[0][0])
            y_set.add(points[0][1])
        return (len(y_set), len(x_set))


class PlanePlot(MPLPlot):
    """
    Abstract class for plotting on plane, currently scatter and hexbin.
    """

    _layout_type = "single"

    def __init__(self, data, x, y, **kwargs):
        MPLPlot.__init__(self, data, **kwargs)
        if x is None or y is None:
            raise ValueError(self._kind + " requires an x and y column")
        if is_integer(x) and not self.data.columns.holds_integer():
            x = self.data.columns[x]
        if is_integer(y) and not self.data.columns.holds_integer():
            y = self.data.columns[y]

        # Scatter plot allows to plot objects data
        if self._kind == "hexbin":
            if len(self.data[x]._get_numeric_data()) == 0:
                raise ValueError(self._kind + " requires x column to be numeric")
            if len(self.data[y]._get_numeric_data()) == 0:
                raise ValueError(self._kind + " requires y column to be numeric")

        self.x = x
        self.y = y

    @property
    def nseries(self) -> int:
        return 1

    def _post_plot_logic(self, ax: Axes, data):
        x, y = self.x, self.y
        xlabel = self.xlabel if self.xlabel is not None else pprint_thing(x)
        ylabel = self.ylabel if self.ylabel is not None else pprint_thing(y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def _plot_colorbar(self, ax: Axes, **kwds):
        # Addresses issues #10611 and #10678:
        # When plotting scatterplots and hexbinplots in IPython
        # inline backend the colorbar axis height tends not to
        # exactly match the parent axis height.
        # The difference is due to small fractional differences
        # in floating points with similar representation.
        # To deal with this, this method forces the colorbar
        # height to take the height of the parent axes.
        # For a more detailed description of the issue
        # see the following link:
        # https://github.com/ipython/ipython/issues/11215

        # GH33389, if ax is used multiple times, we should always
        # use the last one which contains the latest information
        # about the ax
        img = ax.collections[-1]
        cbar = self.fig.colorbar(img, ax=ax, **kwds)

        if mpl_ge_3_0_0():
            # The workaround below is no longer necessary.
            return cbar

        points = ax.get_position().get_points()
        cbar_points = cbar.ax.get_position().get_points()

        cbar.ax.set_position(
            [
                cbar_points[0, 0],
                points[0, 1],
                cbar_points[1, 0] - cbar_points[0, 0],
                points[1, 1] - points[0, 1],
            ]
        )
        # To see the discrepancy in axis heights uncomment
        # the following two lines:
        # print(points[1, 1] - points[0, 1])
        # print(cbar_points[1, 1] - cbar_points[0, 1])

        return cbar


class ScatterPlot(PlanePlot):
    _kind = "scatter"

    def __init__(self, data, x, y, s=None, c=None, **kwargs):
        if s is None:
            # hide the matplotlib default for size, in case we want to change
            # the handling of this argument later
            s = 20
        elif is_hashable(s) and s in data.columns:
            s = data[s]
        super().__init__(data, x, y, s=s, **kwargs)
        if is_integer(c) and not self.data.columns.holds_integer():
            c = self.data.columns[c]
        self.c = c

    def _make_plot(self):
        x, y, c, data = self.x, self.y, self.c, self.data
        ax = self.axes[0]

        c_is_column = is_hashable(c) and c in self.data.columns

        color_by_categorical = c_is_column and is_categorical_dtype(self.data[c])

        # pandas uses colormap, matplotlib uses cmap.
        cmap = self.colormap or "Greys"
        cmap = self.plt.cm.get_cmap(cmap)
        color = self.kwds.pop("color", None)
        if c is not None and color is not None:
            raise TypeError("Specify exactly one of `c` and `color`")
        elif c is None and color is None:
            c_values = self.plt.rcParams["patch.facecolor"]
        elif color is not None:
            c_values = color
        elif color_by_categorical:
            c_values = self.data[c].cat.codes
        elif c_is_column:
            c_values = self.data[c].values
        else:
            c_values = c

        if color_by_categorical:
            from matplotlib import colors

            n_cats = len(self.data[c].cat.categories)
            cmap = colors.ListedColormap([cmap(i) for i in range(cmap.N)])
            bounds = np.linspace(0, n_cats, n_cats + 1)
            norm = colors.BoundaryNorm(bounds, cmap.N)
        else:
            norm = None
        # plot colorbar if
        # 1. colormap is assigned, and
        # 2.`c` is a column containing only numeric values
        plot_colorbar = self.colormap or c_is_column
        cb = self.kwds.pop("colorbar", is_numeric_dtype(c_values) and plot_colorbar)

        if self.legend and hasattr(self, "label"):
            label = self.label
        else:
            label = None
        scatter = ax.scatter(
            data[x].values,
            data[y].values,
            c=c_values,
            label=label,
            cmap=cmap,
            norm=norm,
            **self.kwds,
        )
        if cb:
            cbar_label = c if c_is_column else ""
            cbar = self._plot_colorbar(ax, label=cbar_label)
            if color_by_categorical:
                cbar.set_ticks(np.linspace(0.5, n_cats - 0.5, n_cats))
                cbar.ax.set_yticklabels(self.data[c].cat.categories)

        if label is not None:
            self._append_legend_handles_labels(scatter, label)
        else:
            self.legend = False

        errors_x = self._get_errorbars(label=x, index=0, yerr=False)
        errors_y = self._get_errorbars(label=y, index=0, xerr=False)
        if len(errors_x) > 0 or len(errors_y) > 0:
            err_kwds = dict(errors_x, **errors_y)
            err_kwds["ecolor"] = scatter.get_facecolor()[0]
            ax.errorbar(data[x].values, data[y].values, linestyle="none", **err_kwds)


class HexBinPlot(PlanePlot):
    _kind = "hexbin"

    def __init__(self, data, x, y, C=None, **kwargs):
        super().__init__(data, x, y, **kwargs)
        if is_integer(C) and not self.data.columns.holds_integer():
            C = self.data.columns[C]
        self.C = C

    def _make_plot(self):
        x, y, data, C = self.x, self.y, self.data, self.C
        ax = self.axes[0]
        # pandas uses colormap, matplotlib uses cmap.
        cmap = self.colormap or "BuGn"
        cmap = self.plt.cm.get_cmap(cmap)
        cb = self.kwds.pop("colorbar", True)

        if C is None:
            c_values = None
        else:
            c_values = data[C].values

        ax.hexbin(data[x].values, data[y].values, C=c_values, cmap=cmap, **self.kwds)
        if cb:
            self._plot_colorbar(ax)

    def _make_legend(self):
        pass


class LinePlot(MPLPlot):
    _kind = "line"
    _default_rot = 0
    orientation = "vertical"

    def __init__(self, data, **kwargs):
        from pandas.plotting import plot_params

        MPLPlot.__init__(self, data, **kwargs)
        if self.stacked:
            self.data = self.data.fillna(value=0)
        self.x_compat = plot_params["x_compat"]
        if "x_compat" in self.kwds:
            self.x_compat = bool(self.kwds.pop("x_compat"))

    def _is_ts_plot(self) -> bool:
        # this is slightly deceptive
        return not self.x_compat and self.use_index and self._use_dynamic_x()

    def _use_dynamic_x(self):
        return use_dynamic_x(self._get_ax(0), self.data)

    def _make_plot(self):
        if self._is_ts_plot():
            data = maybe_convert_index(self._get_ax(0), self.data)

            x = data.index  # dummy, not used
            plotf = self._ts_plot
            it = self._iter_data(data=data, keep_index=True)
        else:
            x = self._get_xticks(convert_period=True)
            # error: Incompatible types in assignment (expression has type
            # "Callable[[Any, Any, Any, Any, Any, Any, KwArg(Any)], Any]", variable has
            # type "Callable[[Any, Any, Any, Any, KwArg(Any)], Any]")
            plotf = self._plot  # type: ignore[assignment]
            it = self._iter_data()

        stacking_id = self._get_stacking_id()
        is_errorbar = com.any_not_none(*self.errors.values())

        colors = self._get_colors()
        for i, (label, y) in enumerate(it):
            ax = self._get_ax(i)
            kwds = self.kwds.copy()
            style, kwds = self._apply_style_colors(colors, kwds, i, label)

            errors = self._get_errorbars(label=label, index=i)
            kwds = dict(kwds, **errors)

            label = pprint_thing(label)  # .encode('utf-8')
            label = self._mark_right_label(label, index=i)
            kwds["label"] = label

            newlines = plotf(
                ax,
                x,
                y,
                style=style,
                column_num=i,
                stacking_id=stacking_id,
                is_errorbar=is_errorbar,
                **kwds,
            )
            self._append_legend_handles_labels(newlines[0], label)

            if self._is_ts_plot():

                # reset of xlim should be used for ts data
                # TODO: GH28021, should find a way to change view limit on xaxis
                lines = get_all_lines(ax)
                left, right = get_xlim(lines)
                ax.set_xlim(left, right)

    @classmethod
    def _plot(
        cls, ax: Axes, x, y, style=None, column_num=None, stacking_id=None, **kwds
    ):
        # column_num is used to get the target column from plotf in line and
        # area plots
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(y))
        y_values = cls._get_stacked_values(ax, stacking_id, y, kwds["label"])
        lines = MPLPlot._plot(ax, x, y_values, style=style, **kwds)
        cls._update_stacker(ax, stacking_id, y)
        return lines

    @classmethod
    def _ts_plot(cls, ax: Axes, x, data, style=None, **kwds):
        # accept x to be consistent with normal plot func,
        # x is not passed to tsplot as it uses data.index as x coordinate
        # column_num must be in kwds for stacking purpose
        freq, data = maybe_resample(data, ax, kwds)

        # Set ax with freq info
        decorate_axes(ax, freq, kwds)
        # digging deeper
        if hasattr(ax, "left_ax"):
            decorate_axes(ax.left_ax, freq, kwds)
        if hasattr(ax, "right_ax"):
            decorate_axes(ax.right_ax, freq, kwds)
        ax._plot_data.append((data, cls._kind, kwds))

        lines = cls._plot(ax, data.index, data.values, style=style, **kwds)
        # set date formatter, locators and rescale limits
        format_dateaxis(ax, ax.freq, data.index)
        return lines

    def _get_stacking_id(self):
        if self.stacked:
            return id(self.data)
        else:
            return None

    @classmethod
    def _initialize_stacker(cls, ax: Axes, stacking_id, n: int):
        if stacking_id is None:
            return
        if not hasattr(ax, "_stacker_pos_prior"):
            ax._stacker_pos_prior = {}
        if not hasattr(ax, "_stacker_neg_prior"):
            ax._stacker_neg_prior = {}
        ax._stacker_pos_prior[stacking_id] = np.zeros(n)
        ax._stacker_neg_prior[stacking_id] = np.zeros(n)

    @classmethod
    def _get_stacked_values(cls, ax: Axes, stacking_id, values, label):
        if stacking_id is None:
            return values
        if not hasattr(ax, "_stacker_pos_prior"):
            # stacker may not be initialized for subplots
            cls._initialize_stacker(ax, stacking_id, len(values))

        if (values >= 0).all():
            return ax._stacker_pos_prior[stacking_id] + values
        elif (values <= 0).all():
            return ax._stacker_neg_prior[stacking_id] + values

        raise ValueError(
            "When stacked is True, each column must be either "
            "all positive or all negative. "
            f"Column '{label}' contains both positive and negative values"
        )

    @classmethod
    def _update_stacker(cls, ax: Axes, stacking_id, values):
        if stacking_id is None:
            return
        if (values >= 0).all():
            ax._stacker_pos_prior[stacking_id] += values
        elif (values <= 0).all():
            ax._stacker_neg_prior[stacking_id] += values

    def _post_plot_logic(self, ax: Axes, data):
        from matplotlib.ticker import FixedLocator

        def get_label(i):
            if is_float(i) and i.is_integer():
                i = int(i)
            try:
                return pprint_thing(data.index[i])
            except Exception:
                return ""

        if self._need_to_set_index:
            xticks = ax.get_xticks()
            xticklabels = [get_label(x) for x in xticks]
            ax.xaxis.set_major_locator(FixedLocator(xticks))
            ax.set_xticklabels(xticklabels)

        # If the index is an irregular time series, then by default
        # we rotate the tick labels. The exception is if there are
        # subplots which don't share their x-axes, in which we case
        # we don't rotate the ticklabels as by default the subplots
        # would be too close together.
        condition = (
            not self._use_dynamic_x()
            and (data.index._is_all_dates and self.use_index)
            and (not self.subplots or (self.subplots and self.sharex))
        )

        index_name = self._get_index_name()

        if condition:
            # irregular TS rotated 30 deg. by default
            # probably a better place to check / set this.
            if not self._rot_set:
                self.rot = 30
            format_date_labels(ax, rot=self.rot)

        if index_name is not None and self.use_index:
            ax.set_xlabel(index_name)


class AreaPlot(LinePlot):
    _kind = "area"

    def __init__(self, data, **kwargs):
        kwargs.setdefault("stacked", True)
        data = data.fillna(value=0)
        LinePlot.__init__(self, data, **kwargs)

        if not self.stacked:
            # use smaller alpha to distinguish overlap
            self.kwds.setdefault("alpha", 0.5)

        if self.logy or self.loglog:
            raise ValueError("Log-y scales are not supported in area plot")

    @classmethod
    def _plot(
        cls,
        ax: Axes,
        x,
        y,
        style=None,
        column_num=None,
        stacking_id=None,
        is_errorbar=False,
        **kwds,
    ):

        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(y))
        y_values = cls._get_stacked_values(ax, stacking_id, y, kwds["label"])

        # need to remove label, because subplots uses mpl legend as it is
        line_kwds = kwds.copy()
        line_kwds.pop("label")
        lines = MPLPlot._plot(ax, x, y_values, style=style, **line_kwds)

        # get data from the line to get coordinates for fill_between
        xdata, y_values = lines[0].get_data(orig=False)

        # unable to use ``_get_stacked_values`` here to get starting point
        if stacking_id is None:
            start = np.zeros(len(y))
        elif (y >= 0).all():
            start = ax._stacker_pos_prior[stacking_id]
        elif (y <= 0).all():
            start = ax._stacker_neg_prior[stacking_id]
        else:
            start = np.zeros(len(y))

        if "color" not in kwds:
            kwds["color"] = lines[0].get_color()

        rect = ax.fill_between(xdata, start, y_values, **kwds)
        cls._update_stacker(ax, stacking_id, y)

        # LinePlot expects list of artists
        res = [rect]
        return res

    def _post_plot_logic(self, ax: Axes, data):
        LinePlot._post_plot_logic(self, ax, data)

        is_shared_y = len(list(ax.get_shared_y_axes())) > 0
        # do not override the default axis behaviour in case of shared y axes
        if self.ylim is None and not is_shared_y:
            if (data >= 0).all().all():
                ax.set_ylim(0, None)
            elif (data <= 0).all().all():
                ax.set_ylim(None, 0)


class BarPlot(MPLPlot):
    _kind = "bar"
    _default_rot = 90
    orientation = "vertical"

    def __init__(self, data, **kwargs):
        # we have to treat a series differently than a
        # 1-column DataFrame w.r.t. color handling
        self._is_series = isinstance(data, ABCSeries)
        self.bar_width = kwargs.pop("width", 0.5)
        pos = kwargs.pop("position", 0.5)
        kwargs.setdefault("align", "center")
        self.tick_pos = np.arange(len(data))

        self.bottom = kwargs.pop("bottom", 0)
        self.left = kwargs.pop("left", 0)

        self.log = kwargs.pop("log", False)
        MPLPlot.__init__(self, data, **kwargs)

        if self.stacked or self.subplots:
            self.tickoffset = self.bar_width * pos
            if kwargs["align"] == "edge":
                self.lim_offset = self.bar_width / 2
            else:
                self.lim_offset = 0
        else:
            if kwargs["align"] == "edge":
                w = self.bar_width / self.nseries
                self.tickoffset = self.bar_width * (pos - 0.5) + w * 0.5
                self.lim_offset = w * 0.5
            else:
                self.tickoffset = self.bar_width * pos
                self.lim_offset = 0

        self.ax_pos = self.tick_pos - self.tickoffset

    def _args_adjust(self):
        if is_list_like(self.bottom):
            self.bottom = np.array(self.bottom)
        if is_list_like(self.left):
            self.left = np.array(self.left)

    @classmethod
    def _plot(cls, ax: Axes, x, y, w, start=0, log=False, **kwds):
        return ax.bar(x, y, w, bottom=start, log=log, **kwds)

    @property
    def _start_base(self):
        return self.bottom

    def _make_plot(self):
        import matplotlib as mpl

        colors = self._get_colors()
        ncolors = len(colors)

        pos_prior = neg_prior = np.zeros(len(self.data))
        K = self.nseries

        for i, (label, y) in enumerate(self._iter_data(fillna=0)):
            ax = self._get_ax(i)
            kwds = self.kwds.copy()
            if self._is_series:
                kwds["color"] = colors
            elif isinstance(colors, dict):
                kwds["color"] = colors[label]
            else:
                kwds["color"] = colors[i % ncolors]

            errors = self._get_errorbars(label=label, index=i)
            kwds = dict(kwds, **errors)

            label = pprint_thing(label)
            label = self._mark_right_label(label, index=i)

            if (("yerr" in kwds) or ("xerr" in kwds)) and (kwds.get("ecolor") is None):
                kwds["ecolor"] = mpl.rcParams["xtick.color"]

            start = 0
            if self.log and (y >= 1).all():
                start = 1
            start = start + self._start_base

            if self.subplots:
                w = self.bar_width / 2
                rect = self._plot(
                    ax,
                    self.ax_pos + w,
                    y,
                    self.bar_width,
                    start=start,
                    label=label,
                    log=self.log,
                    **kwds,
                )
                ax.set_title(label)
            elif self.stacked:
                mask = y > 0
                start = np.where(mask, pos_prior, neg_prior) + self._start_base
                w = self.bar_width / 2
                rect = self._plot(
                    ax,
                    self.ax_pos + w,
                    y,
                    self.bar_width,
                    start=start,
                    label=label,
                    log=self.log,
                    **kwds,
                )
                pos_prior = pos_prior + np.where(mask, y, 0)
                neg_prior = neg_prior + np.where(mask, 0, y)
            else:
                w = self.bar_width / K
                rect = self._plot(
                    ax,
                    self.ax_pos + (i + 0.5) * w,
                    y,
                    w,
                    start=start,
                    label=label,
                    log=self.log,
                    **kwds,
                )
            self._append_legend_handles_labels(rect, label)

    def _post_plot_logic(self, ax: Axes, data):
        if self.use_index:
            str_index = [pprint_thing(key) for key in data.index]
        else:
            str_index = [pprint_thing(key) for key in range(data.shape[0])]
        name = self._get_index_name()

        s_edge = self.ax_pos[0] - 0.25 + self.lim_offset
        e_edge = self.ax_pos[-1] + 0.25 + self.bar_width + self.lim_offset

        self._decorate_ticks(ax, name, str_index, s_edge, e_edge)

    def _decorate_ticks(self, ax: Axes, name, ticklabels, start_edge, end_edge):
        ax.set_xlim((start_edge, end_edge))

        if self.xticks is not None:
            ax.set_xticks(np.array(self.xticks))
        else:
            ax.set_xticks(self.tick_pos)
            ax.set_xticklabels(ticklabels)

        if name is not None and self.use_index:
            ax.set_xlabel(name)


class BarhPlot(BarPlot):
    _kind = "barh"
    _default_rot = 0
    orientation = "horizontal"

    @property
    def _start_base(self):
        return self.left

    @classmethod
    def _plot(cls, ax: Axes, x, y, w, start=0, log=False, **kwds):
        return ax.barh(x, y, w, left=start, log=log, **kwds)

    def _decorate_ticks(self, ax: Axes, name, ticklabels, start_edge, end_edge):
        # horizontal bars
        ax.set_ylim((start_edge, end_edge))
        ax.set_yticks(self.tick_pos)
        ax.set_yticklabels(ticklabels)
        if name is not None and self.use_index:
            ax.set_ylabel(name)


class PiePlot(MPLPlot):
    _kind = "pie"
    _layout_type = "horizontal"

    def __init__(self, data, kind=None, **kwargs):
        data = data.fillna(value=0)
        if (data < 0).any().any():
            raise ValueError(f"{self._kind} plot doesn't allow negative values")
        MPLPlot.__init__(self, data, kind=kind, **kwargs)

    def _args_adjust(self):
        self.grid = False
        self.logy = False
        self.logx = False
        self.loglog = False

    def _validate_color_args(self):
        pass

    def _make_plot(self):
        colors = self._get_colors(num_colors=len(self.data), color_kwds="colors")
        self.kwds.setdefault("colors", colors)

        for i, (label, y) in enumerate(self._iter_data()):
            ax = self._get_ax(i)
            if label is not None:
                label = pprint_thing(label)
                ax.set_ylabel(label)

            kwds = self.kwds.copy()

            def blank_labeler(label, value):
                if value == 0:
                    return ""
                else:
                    return label

            idx = [pprint_thing(v) for v in self.data.index]
            labels = kwds.pop("labels", idx)
            # labels is used for each wedge's labels
            # Blank out labels for values of 0 so they don't overlap
            # with nonzero wedges
            if labels is not None:
                blabels = [blank_labeler(left, value) for left, value in zip(labels, y)]
            else:
                # error: Incompatible types in assignment (expression has type "None",
                # variable has type "List[Any]")
                blabels = None  # type: ignore[assignment]
            results = ax.pie(y, labels=blabels, **kwds)

            if kwds.get("autopct", None) is not None:
                patches, texts, autotexts = results
            else:
                patches, texts = results
                autotexts = []

            if self.fontsize is not None:
                for t in texts + autotexts:
                    t.set_fontsize(self.fontsize)

            # leglabels is used for legend labels
            leglabels = labels if labels is not None else idx
            for p, l in zip(patches, leglabels):
                self._append_legend_handles_labels(p, l)
