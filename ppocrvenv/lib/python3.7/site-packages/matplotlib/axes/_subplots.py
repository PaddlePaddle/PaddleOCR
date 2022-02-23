import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib.axes._axes import Axes
from matplotlib.gridspec import GridSpec, SubplotSpec


class SubplotBase:
    """
    Base class for subplots, which are :class:`Axes` instances with
    additional methods to facilitate generating and manipulating a set
    of :class:`Axes` within a figure.
    """

    def __init__(self, fig, *args, **kwargs):
        """
        Parameters
        ----------
        fig : `matplotlib.figure.Figure`

        *args : tuple (*nrows*, *ncols*, *index*) or int
            The array of subplots in the figure has dimensions ``(nrows,
            ncols)``, and *index* is the index of the subplot being created.
            *index* starts at 1 in the upper left corner and increases to the
            right.

            If *nrows*, *ncols*, and *index* are all single digit numbers, then
            *args* can be passed as a single 3-digit number (e.g. 234 for
            (2, 3, 4)).

        **kwargs
            Keyword arguments are passed to the Axes (sub)class constructor.
        """
        # _axes_class is set in the subplot_class_factory
        self._axes_class.__init__(self, fig, [0, 0, 1, 1], **kwargs)
        # This will also update the axes position.
        self.set_subplotspec(SubplotSpec._from_subplot_args(fig, args))

    @_api.deprecated(
        "3.4", alternative="get_subplotspec",
        addendum="(get_subplotspec returns a SubplotSpec instance.)")
    def get_geometry(self):
        """Get the subplot geometry, e.g., (2, 2, 3)."""
        rows, cols, num1, num2 = self.get_subplotspec().get_geometry()
        return rows, cols, num1 + 1  # for compatibility

    @_api.deprecated("3.4", alternative="set_subplotspec")
    def change_geometry(self, numrows, numcols, num):
        """Change subplot geometry, e.g., from (1, 1, 1) to (2, 2, 3)."""
        self._subplotspec = GridSpec(numrows, numcols,
                                     figure=self.figure)[num - 1]
        self.update_params()
        self.set_position(self.figbox)

    def get_subplotspec(self):
        """Return the `.SubplotSpec` instance associated with the subplot."""
        return self._subplotspec

    def set_subplotspec(self, subplotspec):
        """Set the `.SubplotSpec`. instance associated with the subplot."""
        self._subplotspec = subplotspec
        self._set_position(subplotspec.get_position(self.figure))

    def get_gridspec(self):
        """Return the `.GridSpec` instance associated with the subplot."""
        return self._subplotspec.get_gridspec()

    @_api.deprecated(
        "3.4", alternative="get_position()")
    @property
    def figbox(self):
        return self.get_position()

    @_api.deprecated("3.4", alternative="get_gridspec().nrows")
    @property
    def numRows(self):
        return self.get_gridspec().nrows

    @_api.deprecated("3.4", alternative="get_gridspec().ncols")
    @property
    def numCols(self):
        return self.get_gridspec().ncols

    @_api.deprecated("3.4")
    def update_params(self):
        """Update the subplot position from ``self.figure.subplotpars``."""
        # Now a no-op, as figbox/numRows/numCols are (deprecated) auto-updating
        # properties.

    @_api.deprecated("3.4", alternative="ax.get_subplotspec().is_first_row()")
    def is_first_row(self):
        return self.get_subplotspec().rowspan.start == 0

    @_api.deprecated("3.4", alternative="ax.get_subplotspec().is_last_row()")
    def is_last_row(self):
        return self.get_subplotspec().rowspan.stop == self.get_gridspec().nrows

    @_api.deprecated("3.4", alternative="ax.get_subplotspec().is_first_col()")
    def is_first_col(self):
        return self.get_subplotspec().colspan.start == 0

    @_api.deprecated("3.4", alternative="ax.get_subplotspec().is_last_col()")
    def is_last_col(self):
        return self.get_subplotspec().colspan.stop == self.get_gridspec().ncols

    def label_outer(self):
        """
        Only show "outer" labels and tick labels.

        x-labels are only kept for subplots on the last row (or first row, if
        labels are on the top side); y-labels only for subplots on the first
        column (or last column, if labels are on the right side).
        """
        self._label_outer_xaxis(check_patch=False)
        self._label_outer_yaxis(check_patch=False)

    def _label_outer_xaxis(self, *, check_patch):
        # see documentation in label_outer.
        if check_patch and not isinstance(self.patch, mpl.patches.Rectangle):
            return
        ss = self.get_subplotspec()
        label_position = self.xaxis.get_label_position()
        if not ss.is_first_row():  # Remove top label/ticklabels/offsettext.
            if label_position == "top":
                self.set_xlabel("")
            self.xaxis.set_tick_params(which="both", labeltop=False)
            if self.xaxis.offsetText.get_position()[1] == 1:
                self.xaxis.offsetText.set_visible(False)
        if not ss.is_last_row():  # Remove bottom label/ticklabels/offsettext.
            if label_position == "bottom":
                self.set_xlabel("")
            self.xaxis.set_tick_params(which="both", labelbottom=False)
            if self.xaxis.offsetText.get_position()[1] == 0:
                self.xaxis.offsetText.set_visible(False)

    def _label_outer_yaxis(self, *, check_patch):
        # see documentation in label_outer.
        if check_patch and not isinstance(self.patch, mpl.patches.Rectangle):
            return
        ss = self.get_subplotspec()
        label_position = self.yaxis.get_label_position()
        if not ss.is_first_col():  # Remove left label/ticklabels/offsettext.
            if label_position == "left":
                self.set_ylabel("")
            self.yaxis.set_tick_params(which="both", labelleft=False)
            if self.yaxis.offsetText.get_position()[0] == 0:
                self.yaxis.offsetText.set_visible(False)
        if not ss.is_last_col():  # Remove right label/ticklabels/offsettext.
            if label_position == "right":
                self.set_ylabel("")
            self.yaxis.set_tick_params(which="both", labelright=False)
            if self.yaxis.offsetText.get_position()[0] == 1:
                self.yaxis.offsetText.set_visible(False)

    def _make_twin_axes(self, *args, **kwargs):
        """Make a twinx axes of self. This is used for twinx and twiny."""
        if 'sharex' in kwargs and 'sharey' in kwargs:
            # The following line is added in v2.2 to avoid breaking Seaborn,
            # which currently uses this internal API.
            if kwargs["sharex"] is not self and kwargs["sharey"] is not self:
                raise ValueError("Twinned Axes may share only one axis")
        twin = self.figure.add_subplot(self.get_subplotspec(), *args, **kwargs)
        self.set_adjustable('datalim')
        twin.set_adjustable('datalim')
        self._twinned_axes.join(self, twin)
        return twin


subplot_class_factory = cbook._make_class_factory(
    SubplotBase, "{}Subplot", "_axes_class")
Subplot = subplot_class_factory(Axes)  # Provided for backward compatibility.
