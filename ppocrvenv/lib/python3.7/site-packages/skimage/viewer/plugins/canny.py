import numpy as np

import skimage
from ...feature import canny
from .overlayplugin import OverlayPlugin
from ..widgets import Slider, ComboBox


class CannyPlugin(OverlayPlugin):
    """Canny filter plugin to show edges of an image."""

    name = 'Canny Filter'

    def __init__(self, *args, **kwargs):
        super(CannyPlugin, self).__init__(image_filter=canny, **kwargs)

    def attach(self, image_viewer):
        image = image_viewer.image
        imin, imax = skimage.dtype_limits(image, clip_negative=False)
        itype = 'float' if np.issubdtype(image.dtype, np.floating) else 'int'
        self.add_widget(Slider('sigma', 0, 5, update_on='release'))
        self.add_widget(Slider('low threshold', imin, imax, value_type=itype,
                        update_on='release'))
        self.add_widget(Slider('high threshold', imin, imax, value_type=itype,
                        update_on='release'))
        self.add_widget(ComboBox('color', self.color_names, ptype='plugin'))
        # Call parent method at end b/c it calls `filter_image`, which needs
        # the values specified by the widgets. Alternatively, move call to
        # parent method to beginning and add a call to `self.filter_image()`
        super(CannyPlugin,self).attach(image_viewer)
