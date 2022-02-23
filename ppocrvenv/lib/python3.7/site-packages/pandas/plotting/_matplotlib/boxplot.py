from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING
import warnings

from matplotlib.artist import setp
import numpy as np

from pandas.core.dtypes.common import is_dict_like
from pandas.core.dtypes.missing import remove_na_arraylike

import pandas as pd
import pandas.core.common as com

from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import (
    LinePlot,
    MPLPlot,
)
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.tools import (
    create_subplots,
    flatten_axes,
    maybe_adjust_figure,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class BoxPlot(LinePlot):
    _kind = "box"
    _layout_type = "horizontal"

    _valid_return_types = (None, "axes", "dict", "both")
    # namedtuple to hold results
    BP = namedtuple("BP", ["ax", "lines"])

    def __init__(self, data, return_type="axes", **kwargs):
        # Do not call LinePlot.__init__ which may fill nan
        if return_type not in self._valid_return_types:
            raise ValueError("return_type must be {None, 'axes', 'dict', 'both'}")

        self.return_type = return_type
        MPLPlot.__init__(self, data, **kwargs)

    def _args_adjust(self):
        if self.subplots:
            # Disable label ax sharing. Otherwise, all subplots shows last
            # column label
            if self.orientation == "vertical":
                self.sharex = False
            else:
                self.sharey = False

    @classmethod
    def _plot(cls, ax, y, column_num=None, return_type="axes", **kwds):
        if y.ndim == 2:
            y = [remove_na_arraylike(v) for v in y]
            # Boxplot fails with empty arrays, so need to add a NaN
            #   if any cols are empty
            # GH 8181
            y = [v if v.size > 0 else np.array([np.nan]) for v in y]
        else:
            y = remove_na_arraylike(y)
        bp = ax.boxplot(y, **kwds)

        if return_type == "dict":
            return bp, bp
        elif return_type == "both":
            return cls.BP(ax=ax, lines=bp), bp
        else:
            return ax, bp

    def _validate_color_args(self):
        if "color" in self.kwds:
            if self.colormap is not None:
                warnings.warn(
                    "'color' and 'colormap' cannot be used "
                    "simultaneously. Using 'color'"
                )
            self.color = self.kwds.pop("color")

            if isinstance(self.color, dict):
                valid_keys = ["boxes", "whiskers", "medians", "caps"]
                for key in self.color:
                    if key not in valid_keys:
                        raise ValueError(
                            f"color dict contains invalid key '{key}'. "
                            f"The key must be either {valid_keys}"
                        )
        else:
            self.color = None

        # get standard colors for default
        colors = get_standard_colors(num_colors=3, colormap=self.colormap, color=None)
        # use 2 colors by default, for box/whisker and median
        # flier colors isn't needed here
        # because it can be specified by ``sym`` kw
        self._boxes_c = colors[0]
        self._whiskers_c = colors[0]
        self._medians_c = colors[2]
        self._caps_c = colors[0]

    def _get_colors(self, num_colors=None, color_kwds="color"):
        pass

    def maybe_color_bp(self, bp):
        if isinstance(self.color, dict):
            boxes = self.color.get("boxes", self._boxes_c)
            whiskers = self.color.get("whiskers", self._whiskers_c)
            medians = self.color.get("medians", self._medians_c)
            caps = self.color.get("caps", self._caps_c)
        else:
            # Other types are forwarded to matplotlib
            # If None, use default colors
            boxes = self.color or self._boxes_c
            whiskers = self.color or self._whiskers_c
            medians = self.color or self._medians_c
            caps = self.color or self._caps_c

        # GH 30346, when users specifying those arguments explicitly, our defaults
        # for these four kwargs should be overridden; if not, use Pandas settings
        if not self.kwds.get("boxprops"):
            setp(bp["boxes"], color=boxes, alpha=1)
        if not self.kwds.get("whiskerprops"):
            setp(bp["whiskers"], color=whiskers, alpha=1)
        if not self.kwds.get("medianprops"):
            setp(bp["medians"], color=medians, alpha=1)
        if not self.kwds.get("capprops"):
            setp(bp["caps"], color=caps, alpha=1)

    def _make_plot(self):
        if self.subplots:
            self._return_obj = pd.Series(dtype=object)

            for i, (label, y) in enumerate(self._iter_data()):
                ax = self._get_ax(i)
                kwds = self.kwds.copy()

                ret, bp = self._plot(
                    ax, y, column_num=i, return_type=self.return_type, **kwds
                )
                self.maybe_color_bp(bp)
                self._return_obj[label] = ret

                label = [pprint_thing(label)]
                self._set_ticklabels(ax, label)
        else:
            y = self.data.values.T
            ax = self._get_ax(0)
            kwds = self.kwds.copy()

            ret, bp = self._plot(
                ax, y, column_num=0, return_type=self.return_type, **kwds
            )
            self.maybe_color_bp(bp)
            self._return_obj = ret

            labels = [left for left, _ in self._iter_data()]
            labels = [pprint_thing(left) for left in labels]
            if not self.use_index:
                labels = [pprint_thing(key) for key in range(len(labels))]
            self._set_ticklabels(ax, labels)

    def _set_ticklabels(self, ax: Axes, labels):
        if self.orientation == "vertical":
            ax.set_xticklabels(labels)
        else:
            ax.set_yticklabels(labels)

    def _make_legend(self):
        pass

    def _post_plot_logic(self, ax, data):
        pass

    @property
    def orientation(self):
        if self.kwds.get("vert", True):
            return "vertical"
        else:
            return "horizontal"

    @property
    def result(self):
        if self.return_type is None:
            return super().result
        else:
            return self._return_obj


def _grouped_plot_by_column(
    plotf,
    data,
    columns=None,
    by=None,
    numeric_only=True,
    grid=False,
    figsize=None,
    ax=None,
    layout=None,
    return_type=None,
    **kwargs,
):
    grouped = data.groupby(by)
    if columns is None:
        if not isinstance(by, (list, tuple)):
            by = [by]
        columns = data._get_numeric_data().columns.difference(by)
    naxes = len(columns)
    fig, axes = create_subplots(
        naxes=naxes, sharex=True, sharey=True, figsize=figsize, ax=ax, layout=layout
    )

    _axes = flatten_axes(axes)

    ax_values = []

    for i, col in enumerate(columns):
        ax = _axes[i]
        gp_col = grouped[col]
        keys, values = zip(*gp_col)
        re_plotf = plotf(keys, values, ax, **kwargs)
        ax.set_title(col)
        ax.set_xlabel(pprint_thing(by))
        ax_values.append(re_plotf)
        ax.grid(grid)

    result = pd.Series(ax_values, index=columns)

    # Return axes in multiplot case, maybe revisit later # 985
    if return_type is None:
        result = axes

    byline = by[0] if len(by) == 1 else by
    fig.suptitle(f"Boxplot grouped by {byline}")
    maybe_adjust_figure(fig, bottom=0.15, top=0.9, left=0.1, right=0.9, wspace=0.2)

    return result


def boxplot(
    data,
    column=None,
    by=None,
    ax=None,
    fontsize=None,
    rot=0,
    grid=True,
    figsize=None,
    layout=None,
    return_type=None,
    **kwds,
):

    import matplotlib.pyplot as plt

    # validate return_type:
    if return_type not in BoxPlot._valid_return_types:
        raise ValueError("return_type must be {'axes', 'dict', 'both'}")

    if isinstance(data, pd.Series):
        data = data.to_frame("x")
        column = "x"

    def _get_colors():
        #  num_colors=3 is required as method maybe_color_bp takes the colors
        #  in positions 0 and 2.
        #  if colors not provided, use same defaults as DataFrame.plot.box
        result = get_standard_colors(num_colors=3)
        result = np.take(result, [0, 0, 2])
        result = np.append(result, "k")

        colors = kwds.pop("color", None)
        if colors:
            if is_dict_like(colors):
                # replace colors in result array with user-specified colors
                # taken from the colors dict parameter
                # "boxes" value placed in position 0, "whiskers" in 1, etc.
                valid_keys = ["boxes", "whiskers", "medians", "caps"]
                key_to_index = dict(zip(valid_keys, range(4)))
                for key, value in colors.items():
                    if key in valid_keys:
                        result[key_to_index[key]] = value
                    else:
                        raise ValueError(
                            f"color dict contains invalid key '{key}'. "
                            f"The key must be either {valid_keys}"
                        )
            else:
                result.fill(colors)

        return result

    def maybe_color_bp(bp, **kwds):
        # GH 30346, when users specifying those arguments explicitly, our defaults
        # for these four kwargs should be overridden; if not, use Pandas settings
        if not kwds.get("boxprops"):
            setp(bp["boxes"], color=colors[0], alpha=1)
        if not kwds.get("whiskerprops"):
            setp(bp["whiskers"], color=colors[1], alpha=1)
        if not kwds.get("medianprops"):
            setp(bp["medians"], color=colors[2], alpha=1)
        if not kwds.get("capprops"):
            setp(bp["caps"], color=colors[3], alpha=1)

    def plot_group(keys, values, ax: Axes):
        keys = [pprint_thing(x) for x in keys]
        values = [np.asarray(remove_na_arraylike(v), dtype=object) for v in values]
        bp = ax.boxplot(values, **kwds)
        if fontsize is not None:
            ax.tick_params(axis="both", labelsize=fontsize)
        if kwds.get("vert", 1):
            ticks = ax.get_xticks()
            if len(ticks) != len(keys):
                i, remainder = divmod(len(ticks), len(keys))
                assert remainder == 0, remainder
                keys *= i
            ax.set_xticklabels(keys, rotation=rot)
        else:
            ax.set_yticklabels(keys, rotation=rot)
        maybe_color_bp(bp, **kwds)

        # Return axes in multiplot case, maybe revisit later # 985
        if return_type == "dict":
            return bp
        elif return_type == "both":
            return BoxPlot.BP(ax=ax, lines=bp)
        else:
            return ax

    colors = _get_colors()
    if column is None:
        columns = None
    else:
        if isinstance(column, (list, tuple)):
            columns = column
        else:
            columns = [column]

    if by is not None:
        # Prefer array return type for 2-D plots to match the subplot layout
        # https://github.com/pandas-dev/pandas/pull/12216#issuecomment-241175580
        result = _grouped_plot_by_column(
            plot_group,
            data,
            columns=columns,
            by=by,
            grid=grid,
            figsize=figsize,
            ax=ax,
            layout=layout,
            return_type=return_type,
        )
    else:
        if return_type is None:
            return_type = "axes"
        if layout is not None:
            raise ValueError("The 'layout' keyword is not supported when 'by' is None")

        if ax is None:
            rc = {"figure.figsize": figsize} if figsize is not None else {}
            with plt.rc_context(rc):
                ax = plt.gca()
        data = data._get_numeric_data()
        if columns is None:
            columns = data.columns
        else:
            data = data[columns]

        result = plot_group(columns, data.values.T, ax)
        ax.grid(grid)

    return result


def boxplot_frame(
    self,
    column=None,
    by=None,
    ax=None,
    fontsize=None,
    rot=0,
    grid=True,
    figsize=None,
    layout=None,
    return_type=None,
    **kwds,
):
    import matplotlib.pyplot as plt

    ax = boxplot(
        self,
        column=column,
        by=by,
        ax=ax,
        fontsize=fontsize,
        grid=grid,
        rot=rot,
        figsize=figsize,
        layout=layout,
        return_type=return_type,
        **kwds,
    )
    plt.draw_if_interactive()
    return ax


def boxplot_frame_groupby(
    grouped,
    subplots=True,
    column=None,
    fontsize=None,
    rot=0,
    grid=True,
    ax=None,
    figsize=None,
    layout=None,
    sharex=False,
    sharey=True,
    **kwds,
):
    if subplots is True:
        naxes = len(grouped)
        fig, axes = create_subplots(
            naxes=naxes,
            squeeze=False,
            ax=ax,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            layout=layout,
        )
        axes = flatten_axes(axes)

        ret = pd.Series(dtype=object)

        for (key, group), ax in zip(grouped, axes):
            d = group.boxplot(
                ax=ax, column=column, fontsize=fontsize, rot=rot, grid=grid, **kwds
            )
            ax.set_title(pprint_thing(key))
            ret.loc[key] = d
        maybe_adjust_figure(fig, bottom=0.15, top=0.9, left=0.1, right=0.9, wspace=0.2)
    else:
        keys, frames = zip(*grouped)
        if grouped.axis == 0:
            df = pd.concat(frames, keys=keys, axis=1)
        else:
            if len(frames) > 1:
                df = frames[0].join(frames[1::])
            else:
                df = frames[0]

        # GH 16748, DataFrameGroupby fails when subplots=False and `column` argument
        # is assigned, and in this case, since `df` here becomes MI after groupby,
        # so we need to couple the keys (grouped values) and column (original df
        # column) together to search for subset to plot
        if column is not None:
            column = com.convert_to_list_like(column)
            multi_key = pd.MultiIndex.from_product([keys, column])
            column = list(multi_key.values)
        ret = df.boxplot(
            column=column,
            fontsize=fontsize,
            rot=rot,
            grid=grid,
            ax=ax,
            figsize=figsize,
            layout=layout,
            **kwds,
        )
    return ret
