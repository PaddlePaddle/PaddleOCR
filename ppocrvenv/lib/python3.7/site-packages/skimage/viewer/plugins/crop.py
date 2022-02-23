from .base import Plugin
from ..canvastools import RectangleTool
from ...viewer.widgets import SaveButtons, Button


__all__ = ['Crop']


class Crop(Plugin):
    name = 'Crop'

    def __init__(self, maxdist=10, **kwargs):
        super(Crop, self).__init__(**kwargs)
        self.maxdist = maxdist
        self.add_widget(SaveButtons())
        print(self.help())

    def attach(self, image_viewer):
        super(Crop, self).attach(image_viewer)

        self.rect_tool = RectangleTool(image_viewer,
                                       maxdist=self.maxdist,
                                       on_enter=self.crop)
        self.artists.append(self.rect_tool)

        self.reset_button = Button('Reset', self.reset)
        self.add_widget(self.reset_button)

    def help(self):
        helpstr = ("Crop tool",
                   "Select rectangular region and press enter to crop.")
        return '\n'.join(helpstr)

    def crop(self, extents):
        xmin, xmax, ymin, ymax = extents
        if xmin == xmax or ymin == ymax:
            return
        image = self.image_viewer.image[ymin:ymax+1, xmin:xmax+1]
        self.image_viewer.image = image
        self.image_viewer.ax.relim()

    def reset(self):
        self.rect_tool.extents = -10, -10, -10, -10
        self.image_viewer.image = self.image_viewer.original_image
        self.image_viewer.ax.relim()
