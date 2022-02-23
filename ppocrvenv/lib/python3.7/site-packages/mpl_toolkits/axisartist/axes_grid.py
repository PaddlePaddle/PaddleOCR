from matplotlib import _api
import mpl_toolkits.axes_grid1.axes_grid as axes_grid_orig
from .axislines import Axes


@_api.deprecated("3.5")
class CbarAxes(axes_grid_orig.CbarAxesBase, Axes):
    pass


class Grid(axes_grid_orig.Grid):
    _defaultAxesClass = Axes


class ImageGrid(axes_grid_orig.ImageGrid):
    _defaultAxesClass = Axes


AxesGrid = ImageGrid
