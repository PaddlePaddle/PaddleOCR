import numpy as np

from matplotlib import _api
import matplotlib.docstring as docstring
import matplotlib.ticker as mticker
from matplotlib.axes._base import _AxesBase, _TransformedBoundsLocator
from matplotlib.axis import Axis


class SecondaryAxis(_AxesBase):
    """
    General class to hold a Secondary_X/Yaxis.
    """

    def __init__(self, parent, orientation, location, functions, **kwargs):
        """
        See `.secondary_xaxis` and `.secondary_yaxis` for the doc string.
        While there is no need for this to be private, it should really be
        called by those higher level functions.
        """

        self._functions = functions
        self._parent = parent
        self._orientation = orientation
        self._ticks_set = False

        if self._orientation == 'x':
            super().__init__(self._parent.figure, [0, 1., 1, 0.0001], **kwargs)
            self._axis = self.xaxis
            self._locstrings = ['top', 'bottom']
            self._otherstrings = ['left', 'right']
        elif self._orientation == 'y':
            super().__init__(self._parent.figure, [0, 1., 0.0001, 1], **kwargs)
            self._axis = self.yaxis
            self._locstrings = ['right', 'left']
            self._otherstrings = ['top', 'bottom']
        self._parentscale = None
        # this gets positioned w/o constrained_layout so exclude:

        self.set_location(location)
        self.set_functions(functions)

        # styling:
        if self._orientation == 'x':
            otheraxis = self.yaxis
        else:
            otheraxis = self.xaxis

        otheraxis.set_major_locator(mticker.NullLocator())
        otheraxis.set_ticks_position('none')

        self.spines[self._otherstrings].set_visible(False)
        self.spines[self._locstrings].set_visible(True)

        if self._pos < 0.5:
            # flip the location strings...
            self._locstrings = self._locstrings[::-1]
        self.set_alignment(self._locstrings[0])

    def set_alignment(self, align):
        """
        Set if axes spine and labels are drawn at top or bottom (or left/right)
        of the axes.

        Parameters
        ----------
        align : str
            either 'top' or 'bottom' for orientation='x' or
            'left' or 'right' for orientation='y' axis.
        """
        _api.check_in_list(self._locstrings, align=align)
        if align == self._locstrings[1]:  # Need to change the orientation.
            self._locstrings = self._locstrings[::-1]
        self.spines[self._locstrings[0]].set_visible(True)
        self.spines[self._locstrings[1]].set_visible(False)
        self._axis.set_ticks_position(align)
        self._axis.set_label_position(align)

    def set_location(self, location):
        """
        Set the vertical or horizontal location of the axes in
        parent-normalized coordinates.

        Parameters
        ----------
        location : {'top', 'bottom', 'left', 'right'} or float
            The position to put the secondary axis.  Strings can be 'top' or
            'bottom' for orientation='x' and 'right' or 'left' for
            orientation='y'. A float indicates the relative position on the
            parent axes to put the new axes, 0.0 being the bottom (or left)
            and 1.0 being the top (or right).
        """

        # This puts the rectangle into figure-relative coordinates.
        if isinstance(location, str):
            if location in ['top', 'right']:
                self._pos = 1.
            elif location in ['bottom', 'left']:
                self._pos = 0.
            else:
                raise ValueError(
                    f"location must be {self._locstrings[0]!r}, "
                    f"{self._locstrings[1]!r}, or a float, not {location!r}")
        else:
            self._pos = location
        self._loc = location

        if self._orientation == 'x':
            # An x-secondary axes is like an inset axes from x = 0 to x = 1 and
            # from y = pos to y = pos + eps, in the parent's transAxes coords.
            bounds = [0, self._pos, 1., 1e-10]
        else:
            bounds = [self._pos, 0, 1e-10, 1]

        # this locator lets the axes move in the parent axes coordinates.
        # so it never needs to know where the parent is explicitly in
        # figure coordinates.
        # it gets called in ax.apply_aspect() (of all places)
        self.set_axes_locator(
            _TransformedBoundsLocator(bounds, self._parent.transAxes))

    def apply_aspect(self, position=None):
        # docstring inherited.
        self._set_lims()
        super().apply_aspect(position)

    @docstring.copy(Axis.set_ticks)
    def set_ticks(self, ticks, labels=None, *, minor=False, **kwargs):
        ret = self._axis.set_ticks(ticks, labels, minor=minor, **kwargs)
        self.stale = True
        self._ticks_set = True
        return ret

    def set_functions(self, functions):
        """
        Set how the secondary axis converts limits from the parent axes.

        Parameters
        ----------
        functions : 2-tuple of func, or `Transform` with an inverse.
            Transform between the parent axis values and the secondary axis
            values.

            If supplied as a 2-tuple of functions, the first function is
            the forward transform function and the second is the inverse
            transform.

            If a transform is supplied, then the transform must have an
            inverse.
        """
        if (isinstance(functions, tuple) and len(functions) == 2 and
                callable(functions[0]) and callable(functions[1])):
            # make an arbitrary convert from a two-tuple of functions
            # forward and inverse.
            self._functions = functions
        elif functions is None:
            self._functions = (lambda x: x, lambda x: x)
        else:
            raise ValueError('functions argument of secondary axes '
                             'must be a two-tuple of callable functions '
                             'with the first function being the transform '
                             'and the second being the inverse')
        self._set_scale()

    # Should be changed to draw(self, renderer) once the deprecation of
    # renderer=None and of inframe expires.
    def draw(self, *args, **kwargs):
        """
        Draw the secondary axes.

        Consults the parent axes for its limits and converts them
        using the converter specified by
        `~.axes._secondary_axes.set_functions` (or *functions*
        parameter when axes initialized.)
        """
        self._set_lims()
        # this sets the scale in case the parent has set its scale.
        self._set_scale()
        super().draw(*args, **kwargs)

    def _set_scale(self):
        """
        Check if parent has set its scale
        """

        if self._orientation == 'x':
            pscale = self._parent.xaxis.get_scale()
            set_scale = self.set_xscale
        if self._orientation == 'y':
            pscale = self._parent.yaxis.get_scale()
            set_scale = self.set_yscale
        if pscale == self._parentscale:
            return

        if pscale == 'log':
            defscale = 'functionlog'
        else:
            defscale = 'function'

        if self._ticks_set:
            ticks = self._axis.get_ticklocs()

        # need to invert the roles here for the ticks to line up.
        set_scale(defscale, functions=self._functions[::-1])

        # OK, set_scale sets the locators, but if we've called
        # axsecond.set_ticks, we want to keep those.
        if self._ticks_set:
            self._axis.set_major_locator(mticker.FixedLocator(ticks))

        # If the parent scale doesn't change, we can skip this next time.
        self._parentscale = pscale

    def _set_lims(self):
        """
        Set the limits based on parent limits and the convert method
        between the parent and this secondary axes.
        """
        if self._orientation == 'x':
            lims = self._parent.get_xlim()
            set_lim = self.set_xlim
        if self._orientation == 'y':
            lims = self._parent.get_ylim()
            set_lim = self.set_ylim
        order = lims[0] < lims[1]
        lims = self._functions[0](np.array(lims))
        neworder = lims[0] < lims[1]
        if neworder != order:
            # Flip because the transform will take care of the flipping.
            lims = lims[::-1]
        set_lim(lims)

    def set_aspect(self, *args, **kwargs):
        """
        Secondary axes cannot set the aspect ratio, so calling this just
        sets a warning.
        """
        _api.warn_external("Secondary axes can't set the aspect ratio")

    def set_color(self, color):
        """
        Change the color of the secondary axes and all decorators.

        Parameters
        ----------
        color : color
        """
        if self._orientation == 'x':
            self.tick_params(axis='x', colors=color)
            self.spines.bottom.set_color(color)
            self.spines.top.set_color(color)
            self.xaxis.label.set_color(color)
        else:
            self.tick_params(axis='y', colors=color)
            self.spines.left.set_color(color)
            self.spines.right.set_color(color)
            self.yaxis.label.set_color(color)


_secax_docstring = '''
Warnings
--------
This method is experimental as of 3.1, and the API may change.

Parameters
----------
location : {'top', 'bottom', 'left', 'right'} or float
    The position to put the secondary axis.  Strings can be 'top' or
    'bottom' for orientation='x' and 'right' or 'left' for
    orientation='y'. A float indicates the relative position on the
    parent axes to put the new axes, 0.0 being the bottom (or left)
    and 1.0 being the top (or right).

functions : 2-tuple of func, or Transform with an inverse

    If a 2-tuple of functions, the user specifies the transform
    function and its inverse.  i.e.
    ``functions=(lambda x: 2 / x, lambda x: 2 / x)`` would be an
    reciprocal transform with a factor of 2.

    The user can also directly supply a subclass of
    `.transforms.Transform` so long as it has an inverse.

    See :doc:`/gallery/subplots_axes_and_figures/secondary_axis`
    for examples of making these conversions.

Returns
-------
ax : axes._secondary_axes.SecondaryAxis

Other Parameters
----------------
**kwargs : `~matplotlib.axes.Axes` properties.
    Other miscellaneous axes parameters.
'''
docstring.interpd.update(_secax_docstring=_secax_docstring)
