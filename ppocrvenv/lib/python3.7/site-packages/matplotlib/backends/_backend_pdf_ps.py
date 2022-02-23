"""
Common functionality between the PDF and PS backends.
"""

from io import BytesIO
import functools

from fontTools import subset

import matplotlib as mpl
from .. import font_manager, ft2font
from ..afm import AFM
from ..backend_bases import RendererBase


@functools.lru_cache(50)
def _cached_get_afm_from_fname(fname):
    with open(fname, "rb") as fh:
        return AFM(fh)


def get_glyphs_subset(fontfile, characters):
    """
    Subset a TTF font

    Reads the named fontfile and restricts the font to the characters.
    Returns a serialization of the subset font as file-like object.

    Parameters
    ----------
    symbol : str
        Path to the font file
    characters : str
        Continuous set of characters to include in subset
    """

    options = subset.Options(glyph_names=True, recommended_glyphs=True)

    # prevent subsetting FontForge Timestamp and other tables
    options.drop_tables += ['FFTM', 'PfEd']

    with subset.load_font(fontfile, options) as font:
        subsetter = subset.Subsetter(options=options)
        subsetter.populate(text=characters)
        subsetter.subset(font)
        fh = BytesIO()
        font.save(fh, reorderTables=False)
        return fh


class CharacterTracker:
    """
    Helper for font subsetting by the pdf and ps backends.

    Maintains a mapping of font paths to the set of character codepoints that
    are being used from that font.
    """

    def __init__(self):
        self.used = {}

    def track(self, font, s):
        """Record that string *s* is being typeset using font *font*."""
        self.used.setdefault(font.fname, set()).update(map(ord, s))

    def track_glyph(self, font, glyph):
        """Record that codepoint *glyph* is being typeset using font *font*."""
        self.used.setdefault(font.fname, set()).add(glyph)


class RendererPDFPSBase(RendererBase):
    # The following attributes must be defined by the subclasses:
    # - _afm_font_dir
    # - _use_afm_rc_name

    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height

    def flipy(self):
        # docstring inherited
        return False  # y increases from bottom to top.

    def option_scale_image(self):
        # docstring inherited
        return True  # PDF and PS support arbitrary image scaling.

    def option_image_nocomposite(self):
        # docstring inherited
        # Decide whether to composite image based on rcParam value.
        return not mpl.rcParams["image.composite_image"]

    def get_canvas_width_height(self):
        # docstring inherited
        return self.width * 72.0, self.height * 72.0

    def get_text_width_height_descent(self, s, prop, ismath):
        # docstring inherited
        if ismath == "TeX":
            texmanager = self.get_texmanager()
            fontsize = prop.get_size_in_points()
            w, h, d = texmanager.get_text_width_height_descent(
                s, fontsize, renderer=self)
            return w, h, d
        elif ismath:
            # Circular import.
            from matplotlib.backends.backend_ps import RendererPS
            parse = self._text2path.mathtext_parser.parse(
                s, 72, prop,
                _force_standard_ps_fonts=(isinstance(self, RendererPS)
                                          and mpl.rcParams["ps.useafm"]))
            return parse.width, parse.height, parse.depth
        elif mpl.rcParams[self._use_afm_rc_name]:
            font = self._get_font_afm(prop)
            l, b, w, h, d = font.get_str_bbox_and_descent(s)
            scale = prop.get_size_in_points() / 1000
            w *= scale
            h *= scale
            d *= scale
            return w, h, d
        else:
            font = self._get_font_ttf(prop)
            font.set_text(s, 0.0, flags=ft2font.LOAD_NO_HINTING)
            w, h = font.get_width_height()
            d = font.get_descent()
            scale = 1 / 64
            w *= scale
            h *= scale
            d *= scale
            return w, h, d

    def _get_font_afm(self, prop):
        fname = font_manager.findfont(
            prop, fontext="afm", directory=self._afm_font_dir)
        return _cached_get_afm_from_fname(fname)

    def _get_font_ttf(self, prop):
        fname = font_manager.findfont(prop)
        font = font_manager.get_font(fname)
        font.clear()
        font.set_size(prop.get_size_in_points(), 72)
        return font
