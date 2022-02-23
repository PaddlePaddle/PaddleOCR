r"""
A module for parsing a subset of the TeX math syntax and rendering it to a
Matplotlib backend.

For a tutorial of its usage, see :doc:`/tutorials/text/mathtext`.  This
document is primarily concerned with implementation details.

The module uses pyparsing_ to parse the TeX expression.

.. _pyparsing: https://pypi.org/project/pyparsing/

The Bakoma distribution of the TeX Computer Modern fonts, and STIX
fonts are supported.  There is experimental support for using
arbitrary fonts, but results may vary without proper tweaking and
metrics for those fonts.
"""

from collections import namedtuple
import functools
from io import StringIO
import logging
import types

import numpy as np
from PIL import Image

from matplotlib import (
    _api, colors as mcolors, rcParams, _mathtext, _mathtext_data)
from matplotlib.ft2font import FT2Image, LOAD_NO_HINTING
from matplotlib.font_manager import FontProperties

_log = logging.getLogger(__name__)


@_api.caching_module_getattr
class __getattr__:
    locals().update({
        name: _api.deprecated("3.4")(
            property(lambda self, _mod=mod, _name=name: getattr(_mod, _name)))
        for mod, names in [
            (_mathtext, ["SHRINK_FACTOR", "GROW_FACTOR", "NUM_SIZE_LEVELS"]),
            (_mathtext_data, [
                "latex_to_bakoma", "latex_to_cmex", "latex_to_standard",
                "stix_virtual_fonts", "tex2uni"])]
        for name in names})


get_unicode_index = _mathtext.get_unicode_index
get_unicode_index.__module__ = __name__


class MathtextBackend:
    """
    The base class for the mathtext backend-specific code.  `MathtextBackend`
    subclasses interface between mathtext and specific Matplotlib graphics
    backends.

    Subclasses need to override the following:

    - :meth:`render_glyph`
    - :meth:`render_rect_filled`
    - :meth:`get_results`

    And optionally, if you need to use a FreeType hinting style:

    - :meth:`get_hinting_type`
    """
    def __init__(self):
        self.width = 0
        self.height = 0
        self.depth = 0

    def set_canvas_size(self, w, h, d):
        """Set the dimension of the drawing canvas."""
        self.width  = w
        self.height = h
        self.depth  = d

    def render_glyph(self, ox, oy, info):
        """
        Draw a glyph described by *info* to the reference point (*ox*,
        *oy*).
        """
        raise NotImplementedError()

    def render_rect_filled(self, x1, y1, x2, y2):
        """
        Draw a filled black rectangle from (*x1*, *y1*) to (*x2*, *y2*).
        """
        raise NotImplementedError()

    def get_results(self, box):
        """
        Return a backend-specific tuple to return to the backend after
        all processing is done.
        """
        raise NotImplementedError()

    def get_hinting_type(self):
        """
        Get the FreeType hinting type to use with this particular
        backend.
        """
        return LOAD_NO_HINTING


class MathtextBackendAgg(MathtextBackend):
    """
    Render glyphs and rectangles to an FTImage buffer, which is later
    transferred to the Agg image by the Agg backend.
    """
    def __init__(self):
        self.ox = 0
        self.oy = 0
        self.image = None
        self.mode = 'bbox'
        self.bbox = [0, 0, 0, 0]
        super().__init__()

    def _update_bbox(self, x1, y1, x2, y2):
        self.bbox = [min(self.bbox[0], x1),
                     min(self.bbox[1], y1),
                     max(self.bbox[2], x2),
                     max(self.bbox[3], y2)]

    def set_canvas_size(self, w, h, d):
        super().set_canvas_size(w, h, d)
        if self.mode != 'bbox':
            self.image = FT2Image(np.ceil(w), np.ceil(h + max(d, 0)))

    def render_glyph(self, ox, oy, info):
        if self.mode == 'bbox':
            self._update_bbox(ox + info.metrics.xmin,
                              oy - info.metrics.ymax,
                              ox + info.metrics.xmax,
                              oy - info.metrics.ymin)
        else:
            info.font.draw_glyph_to_bitmap(
                self.image, ox, oy - info.metrics.iceberg, info.glyph,
                antialiased=rcParams['text.antialiased'])

    def render_rect_filled(self, x1, y1, x2, y2):
        if self.mode == 'bbox':
            self._update_bbox(x1, y1, x2, y2)
        else:
            height = max(int(y2 - y1) - 1, 0)
            if height == 0:
                center = (y2 + y1) / 2.0
                y = int(center - (height + 1) / 2.0)
            else:
                y = int(y1)
            self.image.draw_rect_filled(int(x1), y, np.ceil(x2), y + height)

    def get_results(self, box, used_characters):
        self.mode = 'bbox'
        orig_height = box.height
        orig_depth  = box.depth
        _mathtext.ship(0, 0, box)
        bbox = self.bbox
        bbox = [bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1]
        self.mode = 'render'
        self.set_canvas_size(
            bbox[2] - bbox[0],
            (bbox[3] - bbox[1]) - orig_depth,
            (bbox[3] - bbox[1]) - orig_height)
        _mathtext.ship(-bbox[0], -bbox[1], box)
        result = (self.ox,
                  self.oy,
                  self.width,
                  self.height + self.depth,
                  self.depth,
                  self.image,
                  used_characters)
        self.image = None
        return result

    def get_hinting_type(self):
        from matplotlib.backends import backend_agg
        return backend_agg.get_hinting_flag()


@_api.deprecated("3.4", alternative="mathtext.math_to_image")
class MathtextBackendBitmap(MathtextBackendAgg):
    def get_results(self, box, used_characters):
        ox, oy, width, height, depth, image, characters = \
            super().get_results(box, used_characters)
        return image, depth


@_api.deprecated("3.4", alternative="MathtextBackendPath")
class MathtextBackendPs(MathtextBackend):
    """
    Store information to write a mathtext rendering to the PostScript backend.
    """

    _PSResult = namedtuple(
        "_PSResult", "width height depth pswriter used_characters")

    def __init__(self):
        super().__init__()
        self.pswriter = StringIO()
        self.lastfont = None

    def render_glyph(self, ox, oy, info):
        oy = self.height - oy + info.offset
        postscript_name = info.postscript_name
        fontsize = info.fontsize

        if (postscript_name, fontsize) != self.lastfont:
            self.lastfont = postscript_name, fontsize
            self.pswriter.write(
                f"/{postscript_name} findfont\n"
                f"{fontsize} scalefont\n"
                f"setfont\n")

        self.pswriter.write(
            f"{ox:f} {oy:f} moveto\n"
            f"/{info.symbol_name} glyphshow\n")

    def render_rect_filled(self, x1, y1, x2, y2):
        ps = "%f %f %f %f rectfill\n" % (
            x1, self.height - y2, x2 - x1, y2 - y1)
        self.pswriter.write(ps)

    def get_results(self, box, used_characters):
        _mathtext.ship(0, 0, box)
        return self._PSResult(self.width,
                              self.height + self.depth,
                              self.depth,
                              self.pswriter,
                              used_characters)


@_api.deprecated("3.4", alternative="MathtextBackendPath")
class MathtextBackendPdf(MathtextBackend):
    """Store information to write a mathtext rendering to the PDF backend."""

    _PDFResult = namedtuple(
        "_PDFResult", "width height depth glyphs rects used_characters")

    def __init__(self):
        super().__init__()
        self.glyphs = []
        self.rects = []

    def render_glyph(self, ox, oy, info):
        filename = info.font.fname
        oy = self.height - oy + info.offset
        self.glyphs.append(
            (ox, oy, filename, info.fontsize,
             info.num, info.symbol_name))

    def render_rect_filled(self, x1, y1, x2, y2):
        self.rects.append((x1, self.height - y2, x2 - x1, y2 - y1))

    def get_results(self, box, used_characters):
        _mathtext.ship(0, 0, box)
        return self._PDFResult(self.width,
                               self.height + self.depth,
                               self.depth,
                               self.glyphs,
                               self.rects,
                               used_characters)


@_api.deprecated("3.4", alternative="MathtextBackendPath")
class MathtextBackendSvg(MathtextBackend):
    """
    Store information to write a mathtext rendering to the SVG
    backend.
    """
    def __init__(self):
        super().__init__()
        self.svg_glyphs = []
        self.svg_rects = []

    def render_glyph(self, ox, oy, info):
        oy = self.height - oy + info.offset

        self.svg_glyphs.append(
            (info.font, info.fontsize, info.num, ox, oy, info.metrics))

    def render_rect_filled(self, x1, y1, x2, y2):
        self.svg_rects.append(
            (x1, self.height - y1 + 1, x2 - x1, y2 - y1))

    def get_results(self, box, used_characters):
        _mathtext.ship(0, 0, box)
        svg_elements = types.SimpleNamespace(svg_glyphs=self.svg_glyphs,
                                             svg_rects=self.svg_rects)
        return (self.width,
                self.height + self.depth,
                self.depth,
                svg_elements,
                used_characters)


class MathtextBackendPath(MathtextBackend):
    """
    Store information to write a mathtext rendering to the text path
    machinery.
    """

    _Result = namedtuple("_Result", "width height depth glyphs rects")

    def __init__(self):
        super().__init__()
        self.glyphs = []
        self.rects = []

    def render_glyph(self, ox, oy, info):
        oy = self.height - oy + info.offset
        self.glyphs.append((info.font, info.fontsize, info.num, ox, oy))

    def render_rect_filled(self, x1, y1, x2, y2):
        self.rects.append((x1, self.height - y2, x2 - x1, y2 - y1))

    def get_results(self, box, used_characters):
        _mathtext.ship(0, 0, box)
        return self._Result(self.width,
                            self.height + self.depth,
                            self.depth,
                            self.glyphs,
                            self.rects)


@_api.deprecated("3.4", alternative="MathtextBackendPath")
class MathtextBackendCairo(MathtextBackend):
    """
    Store information to write a mathtext rendering to the Cairo
    backend.
    """

    def __init__(self):
        super().__init__()
        self.glyphs = []
        self.rects = []

    def render_glyph(self, ox, oy, info):
        oy = oy - info.offset - self.height
        thetext = chr(info.num)
        self.glyphs.append(
            (info.font, info.fontsize, thetext, ox, oy))

    def render_rect_filled(self, x1, y1, x2, y2):
        self.rects.append(
            (x1, y1 - self.height, x2 - x1, y2 - y1))

    def get_results(self, box, used_characters):
        _mathtext.ship(0, 0, box)
        return (self.width,
                self.height + self.depth,
                self.depth,
                self.glyphs,
                self.rects)


for _cls_name in [
        "Fonts",
        *[c.__name__ for c in _mathtext.Fonts.__subclasses__()],
        "FontConstantsBase",
        *[c.__name__ for c in _mathtext.FontConstantsBase.__subclasses__()],
        "Node",
        *[c.__name__ for c in _mathtext.Node.__subclasses__()],
        "Ship", "Parser",
]:
    globals()[_cls_name] = _api.deprecated("3.4")(
        type(_cls_name, (getattr(_mathtext, _cls_name),), {}))


class MathTextWarning(Warning):
    pass


@_api.deprecated("3.4")
def ship(ox, oy, box):
    _mathtext.ship(ox, oy, box)


##############################################################################
# MAIN


class MathTextParser:
    _parser = None

    _backend_mapping = {
        'bitmap': MathtextBackendBitmap,
        'agg':    MathtextBackendAgg,
        'ps':     MathtextBackendPs,
        'pdf':    MathtextBackendPdf,
        'svg':    MathtextBackendSvg,
        'path':   MathtextBackendPath,
        'cairo':  MathtextBackendCairo,
        'macosx': MathtextBackendAgg,
    }
    _font_type_mapping = {
        'cm':          _mathtext.BakomaFonts,
        'dejavuserif': _mathtext.DejaVuSerifFonts,
        'dejavusans':  _mathtext.DejaVuSansFonts,
        'stix':        _mathtext.StixFonts,
        'stixsans':    _mathtext.StixSansFonts,
        'custom':      _mathtext.UnicodeFonts,
    }

    def __init__(self, output):
        """Create a MathTextParser for the given backend *output*."""
        self._output = output.lower()

    def parse(self, s, dpi=72, prop=None, *, _force_standard_ps_fonts=False):
        """
        Parse the given math expression *s* at the given *dpi*.  If *prop* is
        provided, it is a `.FontProperties` object specifying the "default"
        font to use in the math expression, used for all non-math text.

        The results are cached, so multiple calls to `parse`
        with the same expression should be fast.
        """
        if _force_standard_ps_fonts:
            _api.warn_deprecated(
                "3.4",
                removal="3.5",
                message=(
                    "Mathtext using only standard PostScript fonts has "
                    "been likely to produce wrong output for a while, "
                    "has been deprecated in %(since)s and will be removed "
                    "in %(removal)s, after which ps.useafm will have no "
                    "effect on mathtext."
                )
            )

        # lru_cache can't decorate parse() directly because the ps.useafm and
        # mathtext.fontset rcParams also affect the parse (e.g. by affecting
        # the glyph metrics).
        return self._parse_cached(s, dpi, prop, _force_standard_ps_fonts)

    @functools.lru_cache(50)
    def _parse_cached(self, s, dpi, prop, force_standard_ps_fonts):
        if prop is None:
            prop = FontProperties()

        fontset_class = (
            _mathtext.StandardPsFonts if force_standard_ps_fonts
            else _api.check_getitem(
                self._font_type_mapping, fontset=prop.get_math_fontfamily()))
        backend = self._backend_mapping[self._output]()
        font_output = fontset_class(prop, backend)

        fontsize = prop.get_size_in_points()

        # This is a class variable so we don't rebuild the parser
        # with each request.
        if self._parser is None:
            self.__class__._parser = _mathtext.Parser()

        box = self._parser.parse(s, font_output, fontsize, dpi)
        font_output.set_canvas_size(box.width, box.height, box.depth)
        return font_output.get_results(box)

    @_api.deprecated("3.4", alternative="mathtext.math_to_image")
    def to_mask(self, texstr, dpi=120, fontsize=14):
        r"""
        Convert a mathtext string to a grayscale array and depth.

        Parameters
        ----------
        texstr : str
            A valid mathtext string, e.g., r'IQ: $\sigma_i=15$'.
        dpi : float
            The dots-per-inch setting used to render the text.
        fontsize : int
            The font size in points

        Returns
        -------
        array : 2D uint8 alpha
            Mask array of rasterized tex.
        depth : int
            Offset of the baseline from the bottom of the image, in pixels.
        """
        assert self._output == "bitmap"
        prop = FontProperties(size=fontsize)
        ftimage, depth = self.parse(texstr, dpi=dpi, prop=prop)
        return np.asarray(ftimage), depth

    @_api.deprecated("3.4", alternative="mathtext.math_to_image")
    def to_rgba(self, texstr, color='black', dpi=120, fontsize=14):
        r"""
        Convert a mathtext string to an RGBA array and depth.

        Parameters
        ----------
        texstr : str
            A valid mathtext string, e.g., r'IQ: $\sigma_i=15$'.
        color : color
            The text color.
        dpi : float
            The dots-per-inch setting used to render the text.
        fontsize : int
            The font size in points.

        Returns
        -------
        array : (M, N, 4) array
            RGBA color values of rasterized tex, colorized with *color*.
        depth : int
            Offset of the baseline from the bottom of the image, in pixels.
        """
        x, depth = self.to_mask(texstr, dpi=dpi, fontsize=fontsize)

        r, g, b, a = mcolors.to_rgba(color)
        RGBA = np.zeros((x.shape[0], x.shape[1], 4), dtype=np.uint8)
        RGBA[:, :, 0] = 255 * r
        RGBA[:, :, 1] = 255 * g
        RGBA[:, :, 2] = 255 * b
        RGBA[:, :, 3] = x
        return RGBA, depth

    @_api.deprecated("3.4", alternative="mathtext.math_to_image")
    def to_png(self, filename, texstr, color='black', dpi=120, fontsize=14):
        r"""
        Render a tex expression to a PNG file.

        Parameters
        ----------
        filename
            A writable filename or fileobject.
        texstr : str
            A valid mathtext string, e.g., r'IQ: $\sigma_i=15$'.
        color : color
            The text color.
        dpi : float
            The dots-per-inch setting used to render the text.
        fontsize : int
            The font size in points.

        Returns
        -------
        int
            Offset of the baseline from the bottom of the image, in pixels.
        """
        rgba, depth = self.to_rgba(
            texstr, color=color, dpi=dpi, fontsize=fontsize)
        Image.fromarray(rgba).save(filename, format="png")
        return depth

    @_api.deprecated("3.4", alternative="mathtext.math_to_image")
    def get_depth(self, texstr, dpi=120, fontsize=14):
        r"""
        Get the depth of a mathtext string.

        Parameters
        ----------
        texstr : str
            A valid mathtext string, e.g., r'IQ: $\sigma_i=15$'.
        dpi : float
            The dots-per-inch setting used to render the text.

        Returns
        -------
        int
            Offset of the baseline from the bottom of the image, in pixels.
        """
        assert self._output == "bitmap"
        prop = FontProperties(size=fontsize)
        ftimage, depth = self.parse(texstr, dpi=dpi, prop=prop)
        return depth


def math_to_image(s, filename_or_obj, prop=None, dpi=None, format=None):
    """
    Given a math expression, renders it in a closely-clipped bounding
    box to an image file.

    Parameters
    ----------
    s : str
        A math expression.  The math portion must be enclosed in dollar signs.
    filename_or_obj : str or path-like or file-like
        Where to write the image data.
    prop : `.FontProperties`, optional
        The size and style of the text.
    dpi : float, optional
        The output dpi.  If not set, the dpi is determined as for
        `.Figure.savefig`.
    format : str, optional
        The output format, e.g., 'svg', 'pdf', 'ps' or 'png'.  If not set, the
        format is determined as for `.Figure.savefig`.
    """
    from matplotlib import figure
    # backend_agg supports all of the core output formats
    from matplotlib.backends import backend_agg

    if prop is None:
        prop = FontProperties()

    parser = MathTextParser('path')
    width, height, depth, _, _ = parser.parse(s, dpi=72, prop=prop)

    fig = figure.Figure(figsize=(width / 72.0, height / 72.0))
    fig.text(0, depth/height, s, fontproperties=prop)
    backend_agg.FigureCanvasAgg(fig)
    fig.savefig(filename_or_obj, dpi=dpi, format=format)

    return depth
