from matplotlib import _api
from .axislines import (
    Axes, AxesZero, AxisArtistHelper, AxisArtistHelperRectlinear,
    GridHelperBase, GridHelperRectlinear, Subplot, SubplotZero)
from .axis_artist import AxisArtist, GridlinesCollection
from .grid_helper_curvelinear import GridHelperCurveLinear
from .floating_axes import FloatingAxes, FloatingSubplot
from mpl_toolkits.axes_grid1.parasite_axes import (
    host_axes_class_factory, parasite_axes_class_factory,
    parasite_axes_auxtrans_class_factory, subplot_class_factory)


ParasiteAxes = parasite_axes_class_factory(Axes)
HostAxes = host_axes_class_factory(Axes)
SubplotHost = subplot_class_factory(HostAxes)
with _api.suppress_matplotlib_deprecation_warning():
    ParasiteAxesAuxTrans = parasite_axes_auxtrans_class_factory(ParasiteAxes)
