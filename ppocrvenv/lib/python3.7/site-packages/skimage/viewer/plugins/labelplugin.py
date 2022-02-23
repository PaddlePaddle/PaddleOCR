import numpy as np

from .base import Plugin
from ..widgets import ComboBox, Slider
from ..canvastools import PaintTool


__all__ = ['LabelPainter']


rad2deg = 180 / np.pi


class LabelPainter(Plugin):
    name = 'LabelPainter'

    def __init__(self, max_radius=20, **kwargs):
        super(LabelPainter, self).__init__(**kwargs)

        # These widgets adjust plugin properties instead of an image filter.
        self._radius_widget = Slider('radius', low=1, high=max_radius,
                                     value=5, value_type='int', ptype='plugin')
        labels = [str(i) for i in range(6)]
        labels[0] = 'Erase'
        self._label_widget = ComboBox('label', labels, ptype='plugin')
        self.add_widget(self._radius_widget)
        self.add_widget(self._label_widget)

        print(self.help())

    def help(self):
        helpstr = ("Label painter",
                   "Hold left-mouse button and paint on canvas.")
        return '\n'.join(helpstr)

    def attach(self, image_viewer):
        super(LabelPainter, self).attach(image_viewer)

        image = image_viewer.original_image
        self.paint_tool = PaintTool(image_viewer, image.shape,
                                    on_enter=self.on_enter)
        self.paint_tool.radius = self.radius
        self.paint_tool.label = self._label_widget.index = 1
        self.artists.append(self.paint_tool)

    def _on_new_image(self, image):
        """Update plugin for new images."""
        self.paint_tool.shape = image.shape

    def on_enter(self, overlay):
        pass

    @property
    def radius(self):
        return self._radius_widget.val

    @radius.setter
    def radius(self, val):
        self.paint_tool.radius = val

    @property
    def label(self):
        return self._label_widget.val

    @label.setter
    def label(self, val):
        self.paint_tool.label = val
