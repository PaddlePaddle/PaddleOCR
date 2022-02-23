import numpy as np

from qtpy.QtGui import QPainter, QColor
from qtpy.QtWidgets import QWidget, QGridLayout, QFrame

from .util import histograms


class ColorHistogram(QWidget):
    '''A Class which draws a scaling histogram in
    a widget.

    Where counts are the bin values in the histogram
    and colormap is a tuple of (R, G, B) tuples the same length
    as counts. These are the colors to apply to the histogram bars.
    Colormap can also contain a single tuple (R, G, B), in which case this is
    the color applied to all bars of that histogram.

    The histogram assumes the bins were evenly spaced.
    '''

    def __init__(self, counts, colormap):
        QWidget.__init__(self)
        self._validate_input(counts, colormap)
        self.counts = counts
        self.n = np.sum(self.counts)
        self.colormap = colormap
        self.setMinimumSize(100, 50)

    def _validate_input(self, counts, colormap):
        if len(counts) != len(colormap):
            if len(colormap) != 3:
                msg = '''Colormap must be a list of 3-tuples the same
                          length as counts or a 3-tuple'''
                raise ValueError(msg)

    def paintEvent(self, evt):
        # get the widget dimensions
        orig_width = self.width()
        orig_height = self.height()

        # fill perc % of the widget
        perc = 1
        width = int(orig_width * perc)
        height = int(orig_height * perc)

        # get the starting origin
        x_orig = int((orig_width - width) / 2)
        # we want to start at the bottom and draw up.
        y_orig = orig_height - int((orig_height - height) / 2)

        # a running x-position
        running_pos = x_orig

        # calculate to number of bars
        nbars = len(self.counts)

        # calculate the bar widths, this compilcation is
        # necessary because integer trunction severely cripples
        # the layout.
        remainder = width % nbars
        bar_width = [int(width / nbars)] * nbars
        for i in range(remainder):
            bar_width[i] += 1

        paint = QPainter()
        paint.begin(self)

        # determine the scaling factor
        max_val = np.max(self.counts)
        scale = 1. * height / max_val

        # determine if we have a colormap and drop into the appropriate
        # loop.
        if hasattr(self.colormap[0], '__iter__'):
            # assume we have a colormap
            for i in range(len(self.counts)):
                bar_height = self.counts[i]
                r, g, b = self.colormap[i]
                paint.setPen(QColor(r, g, b))
                paint.setBrush(QColor(r, g, b))
                paint.drawRect(running_pos, y_orig, bar_width[i],
                               -bar_height)
                running_pos += bar_width[i]

        else:
            # we have a tuple
            r, g, b = self.colormap
            paint.setPen(QColor(r, g, b))
            paint.setBrush(QColor(r, g, b))
            for i in range(len(self.counts)):
                bar_height = self.counts[i] * scale
                paint.drawRect(running_pos, y_orig, bar_width[i],
                               -bar_height)
                running_pos += bar_width[i]

        paint.end()

    def update_hist(self, counts, cmap):
        self._validate_input(counts, cmap)
        self.counts = counts
        self.colormap = cmap
        self.repaint()


class QuadHistogram(QFrame):
    '''A class which uses ColorHistogram to draw
    the 4 histograms of an image. R, G, B, and Value.

    The 4 histograms are layout out in a grid,
    and can be specified horizontal or vertical,
    and in which order ie. ['R', 'G', 'B', 'V']
    '''

    def __init__(self, img, layout='vertical', order=['R', 'G', 'B', 'V']):
        QFrame.__init__(self)

        r, g, b, v = histograms(img, 100)
        self.r_hist = ColorHistogram(r, (255, 0, 0))
        self.g_hist = ColorHistogram(g, (0, 255, 0))
        self.b_hist = ColorHistogram(b, (0, 0, 255))
        self.v_hist = ColorHistogram(v, (0, 0, 0))

        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        order_map = {'R': self.r_hist, 'G': self.g_hist, 'B': self.b_hist,
                     'V': self.v_hist}

        if layout == 'vertical':
            for i in range(len(order)):
                self.layout.addWidget(order_map[order[i]], i, 0)
        elif layout == 'horizontal':
            for i in range(len(order)):
                self.layout.addWidget(order_map[order[i]], 0, i)

    def update_hists(self, img):
        r, g, b, v = histograms(img, 100)
        self.r_hist.update_hist(r, (255, 0, 0))
        self.g_hist.update_hist(g, (0, 255, 0))
        self.b_hist.update_hist(b, (0, 0, 255))
        self.v_hist.update_hist(v, (0, 0, 0))
