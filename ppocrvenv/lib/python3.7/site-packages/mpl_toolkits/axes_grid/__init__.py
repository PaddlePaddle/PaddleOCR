from . import axes_size as Size
from .axes_divider import Divider, SubplotDivider, make_axes_locatable
from .axes_grid import Grid, ImageGrid, AxesGrid
from matplotlib import _api
_api.warn_deprecated(since='2.1',
                     name='mpl_toolkits.axes_grid',
                     alternative='mpl_toolkits.axes_grid1 and'
                                 ' mpl_toolkits.axisartist, which provide'
                                 ' the same functionality',
                     obj_type='module')
