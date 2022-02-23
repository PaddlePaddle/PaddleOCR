"""
Implementation details for :mod:`.mathtext`.
"""

from collections import namedtuple
import enum
import functools
from io import StringIO
import logging
import os
import types
import unicodedata

import numpy as np
from pyparsing import (
    Combine, Empty, Forward, Group, Literal, oneOf, OneOrMore,
    Optional, ParseBaseException, ParseFatalException, ParserElement,
    ParseResults, QuotedString, Regex, StringEnd, Suppress, White, ZeroOrMore)

import matplotlib as mpl
from . import _api, cbook
from ._mathtext_data import (
    latex_to_bakoma, latex_to_standard, stix_glyph_fixes, stix_virtual_fonts,
    tex2uni)
from .afm import AFM
from .font_manager import FontProperties, findfont, get_font
from .ft2font import KERNING_DEFAULT


ParserElement.enablePackrat()
_log = logging.getLogger("matplotlib.mathtext")


##############################################################################
# FONTS


def get_unicode_index(symbol, math=True):
    r"""
    Return the integer index (from the Unicode table) of *symbol*.

    Parameters
    ----------
    symbol : str
        A single unicode character, a TeX command (e.g. r'\pi') or a Type1
        symbol name (e.g. 'phi').
    math : bool, default: True
        If False, always treat as a single unicode character.
    """
    # for a non-math symbol, simply return its unicode index
    if not math:
        return ord(symbol)
    # From UTF #25: U+2212 minus sign is the preferred
    # representation of the unary and binary minus sign rather than
    # the ASCII-derived U+002D hyphen-minus, because minus sign is
    # unambiguous and because it is rendered with a more desirable
    # length, usually longer than a hyphen.
    if symbol == '-':
        return 0x2212
    try:  # This will succeed if symbol is a single unicode char
        return ord(symbol)
    except TypeError:
        pass
    try:  # Is symbol a TeX symbol (i.e. \alpha)
        return tex2uni[symbol.strip("\\")]
    except KeyError as err:
        raise ValueError(
            "'{}' is not a valid Unicode character or TeX/Type1 symbol"
            .format(symbol)) from err


class Fonts:
    """
    An abstract base class for a system of fonts to use for mathtext.

    The class must be able to take symbol keys and font file names and
    return the character metrics.  It also delegates to a backend class
    to do the actual drawing.
    """

    def __init__(self, default_font_prop, mathtext_backend):
        """
        Parameters
        ----------
        default_font_prop : `~.font_manager.FontProperties`
            The default non-math font, or the base font for Unicode (generic)
            font rendering.
        mathtext_backend : `MathtextBackend` subclass
            Backend to which rendering is actually delegated.
        """
        self.default_font_prop = default_font_prop
        self.mathtext_backend = mathtext_backend
        self.used_characters = {}

    @_api.deprecated("3.4")
    def destroy(self):
        """
        Fix any cyclical references before the object is about
        to be destroyed.
        """
        self.used_characters = None

    def get_kern(self, font1, fontclass1, sym1, fontsize1,
                 font2, fontclass2, sym2, fontsize2, dpi):
        """
        Get the kerning distance for font between *sym1* and *sym2*.

        See `~.Fonts.get_metrics` for a detailed description of the parameters.
        """
        return 0.

    def get_metrics(self, font, font_class, sym, fontsize, dpi, math=True):
        r"""
        Parameters
        ----------
        font : str
            One of the TeX font names: "tt", "it", "rm", "cal", "sf", "bf",
            "default", "regular", "bb", "frak", "scr".  "default" and "regular"
            are synonyms and use the non-math font.
        font_class : str
            One of the TeX font names (as for *font*), but **not** "bb",
            "frak", or "scr".  This is used to combine two font classes.  The
            only supported combination currently is ``get_metrics("frak", "bf",
            ...)``.
        sym : str
            A symbol in raw TeX form, e.g., "1", "x", or "\sigma".
        fontsize : float
            Font size in points.
        dpi : float
            Rendering dots-per-inch.
        math : bool
            Whether we are currently in math mode or not.

        Returns
        -------
        object

            The returned object has the following attributes (all floats,
            except *slanted*):

            - *advance*: The advance distance (in points) of the glyph.
            - *height*: The height of the glyph in points.
            - *width*: The width of the glyph in points.
            - *xmin*, *xmax*, *ymin*, *ymax*: The ink rectangle of the glyph
            - *iceberg*: The distance from the baseline to the top of the
              glyph.  (This corresponds to TeX's definition of "height".)
            - *slanted*: Whether the glyph should be considered as "slanted"
              (currently used for kerning sub/superscripts).
        """
        info = self._get_info(font, font_class, sym, fontsize, dpi, math)
        return info.metrics

    def set_canvas_size(self, w, h, d):
        """
        Set the size of the buffer used to render the math expression.
        Only really necessary for the bitmap backends.
        """
        self.width, self.height, self.depth = np.ceil([w, h, d])
        self.mathtext_backend.set_canvas_size(
            self.width, self.height, self.depth)

    @_api.rename_parameter("3.4", "facename", "font")
    def render_glyph(self, ox, oy, font, font_class, sym, fontsize, dpi):
        """
        At position (*ox*, *oy*), draw the glyph specified by the remaining
        parameters (see `get_metrics` for their detailed description).
        """
        info = self._get_info(font, font_class, sym, fontsize, dpi)
        self.used_characters.setdefault(info.font.fname, set()).add(info.num)
        self.mathtext_backend.render_glyph(ox, oy, info)

    def render_rect_filled(self, x1, y1, x2, y2):
        """
        Draw a filled rectangle from (*x1*, *y1*) to (*x2*, *y2*).
        """
        self.mathtext_backend.render_rect_filled(x1, y1, x2, y2)

    def get_xheight(self, font, fontsize, dpi):
        """
        Get the xheight for the given *font* and *fontsize*.
        """
        raise NotImplementedError()

    def get_underline_thickness(self, font, fontsize, dpi):
        """
        Get the line thickness that matches the given font.  Used as a
        base unit for drawing lines such as in a fraction or radical.
        """
        raise NotImplementedError()

    def get_used_characters(self):
        """
        Get the set of characters that were used in the math
        expression.  Used by backends that need to subset fonts so
        they know which glyphs to include.
        """
        return self.used_characters

    def get_results(self, box):
        """
        Get the data needed by the backend to render the math
        expression.  The return value is backend-specific.
        """
        result = self.mathtext_backend.get_results(
            box, self.get_used_characters())
        if self.destroy != TruetypeFonts.destroy.__get__(self):
            destroy = _api.deprecate_method_override(
                __class__.destroy, self, since="3.4")
            if destroy:
                destroy()
        return result

    def get_sized_alternatives_for_symbol(self, fontname, sym):
        """
        Override if your font provides multiple sizes of the same
        symbol.  Should return a list of symbols matching *sym* in
        various sizes.  The expression renderer will select the most
        appropriate size for a given situation from this list.
        """
        return [(fontname, sym)]


class TruetypeFonts(Fonts):
    """
    A generic base class for all font setups that use Truetype fonts
    (through FT2Font).
    """
    def __init__(self, default_font_prop, mathtext_backend):
        super().__init__(default_font_prop, mathtext_backend)
        self.glyphd = {}
        self._fonts = {}

        filename = findfont(default_font_prop)
        default_font = get_font(filename)
        self._fonts['default'] = default_font
        self._fonts['regular'] = default_font

    @_api.deprecated("3.4")
    def destroy(self):
        self.glyphd = None
        super().destroy()

    def _get_font(self, font):
        if font in self.fontmap:
            basename = self.fontmap[font]
        else:
            basename = font
        cached_font = self._fonts.get(basename)
        if cached_font is None and os.path.exists(basename):
            cached_font = get_font(basename)
            self._fonts[basename] = cached_font
            self._fonts[cached_font.postscript_name] = cached_font
            self._fonts[cached_font.postscript_name.lower()] = cached_font
        return cached_font

    def _get_offset(self, font, glyph, fontsize, dpi):
        if font.postscript_name == 'Cmex10':
            return (glyph.height / 64 / 2) + (fontsize/3 * dpi/72)
        return 0.

    def _get_info(self, fontname, font_class, sym, fontsize, dpi, math=True):
        key = fontname, font_class, sym, fontsize, dpi
        bunch = self.glyphd.get(key)
        if bunch is not None:
            return bunch

        font, num, symbol_name, fontsize, slanted = \
            self._get_glyph(fontname, font_class, sym, fontsize, math)

        font.set_size(fontsize, dpi)
        glyph = font.load_char(
            num,
            flags=self.mathtext_backend.get_hinting_type())

        xmin, ymin, xmax, ymax = [val/64.0 for val in glyph.bbox]
        offset = self._get_offset(font, glyph, fontsize, dpi)
        metrics = types.SimpleNamespace(
            advance = glyph.linearHoriAdvance/65536.0,
            height  = glyph.height/64.0,
            width   = glyph.width/64.0,
            xmin    = xmin,
            xmax    = xmax,
            ymin    = ymin+offset,
            ymax    = ymax+offset,
            # iceberg is the equivalent of TeX's "height"
            iceberg = glyph.horiBearingY/64.0 + offset,
            slanted = slanted
            )

        result = self.glyphd[key] = types.SimpleNamespace(
            font            = font,
            fontsize        = fontsize,
            postscript_name = font.postscript_name,
            metrics         = metrics,
            symbol_name     = symbol_name,
            num             = num,
            glyph           = glyph,
            offset          = offset
            )
        return result

    def get_xheight(self, fontname, fontsize, dpi):
        font = self._get_font(fontname)
        font.set_size(fontsize, dpi)
        pclt = font.get_sfnt_table('pclt')
        if pclt is None:
            # Some fonts don't store the xHeight, so we do a poor man's xHeight
            metrics = self.get_metrics(
                fontname, mpl.rcParams['mathtext.default'], 'x', fontsize, dpi)
            return metrics.iceberg
        xHeight = (pclt['xHeight'] / 64.0) * (fontsize / 12.0) * (dpi / 100.0)
        return xHeight

    def get_underline_thickness(self, font, fontsize, dpi):
        # This function used to grab underline thickness from the font
        # metrics, but that information is just too un-reliable, so it
        # is now hardcoded.
        return ((0.75 / 12.0) * fontsize * dpi) / 72.0

    def get_kern(self, font1, fontclass1, sym1, fontsize1,
                 font2, fontclass2, sym2, fontsize2, dpi):
        if font1 == font2 and fontsize1 == fontsize2:
            info1 = self._get_info(font1, fontclass1, sym1, fontsize1, dpi)
            info2 = self._get_info(font2, fontclass2, sym2, fontsize2, dpi)
            font = info1.font
            return font.get_kerning(info1.num, info2.num, KERNING_DEFAULT) / 64
        return super().get_kern(font1, fontclass1, sym1, fontsize1,
                                font2, fontclass2, sym2, fontsize2, dpi)


class BakomaFonts(TruetypeFonts):
    """
    Use the Bakoma TrueType fonts for rendering.

    Symbols are strewn about a number of font files, each of which has
    its own proprietary 8-bit encoding.
    """
    _fontmap = {
        'cal': 'cmsy10',
        'rm':  'cmr10',
        'tt':  'cmtt10',
        'it':  'cmmi10',
        'bf':  'cmb10',
        'sf':  'cmss10',
        'ex':  'cmex10',
    }

    def __init__(self, *args, **kwargs):
        self._stix_fallback = StixFonts(*args, **kwargs)

        super().__init__(*args, **kwargs)
        self.fontmap = {}
        for key, val in self._fontmap.items():
            fullpath = findfont(val)
            self.fontmap[key] = fullpath
            self.fontmap[val] = fullpath

    _slanted_symbols = set(r"\int \oint".split())

    def _get_glyph(self, fontname, font_class, sym, fontsize, math=True):
        symbol_name = None
        font = None
        if fontname in self.fontmap and sym in latex_to_bakoma:
            basename, num = latex_to_bakoma[sym]
            slanted = (basename == "cmmi10") or sym in self._slanted_symbols
            font = self._get_font(basename)
        elif len(sym) == 1:
            slanted = (fontname == "it")
            font = self._get_font(fontname)
            if font is not None:
                num = ord(sym)

        if font is not None:
            gid = font.get_char_index(num)
            if gid != 0:
                symbol_name = font.get_glyph_name(gid)

        if symbol_name is None:
            return self._stix_fallback._get_glyph(
                fontname, font_class, sym, fontsize, math)

        return font, num, symbol_name, fontsize, slanted

    # The Bakoma fonts contain many pre-sized alternatives for the
    # delimiters.  The AutoSizedChar class will use these alternatives
    # and select the best (closest sized) glyph.
    _size_alternatives = {
        '(':           [('rm', '('), ('ex', '\xa1'), ('ex', '\xb3'),
                        ('ex', '\xb5'), ('ex', '\xc3')],
        ')':           [('rm', ')'), ('ex', '\xa2'), ('ex', '\xb4'),
                        ('ex', '\xb6'), ('ex', '\x21')],
        '{':           [('cal', '{'), ('ex', '\xa9'), ('ex', '\x6e'),
                        ('ex', '\xbd'), ('ex', '\x28')],
        '}':           [('cal', '}'), ('ex', '\xaa'), ('ex', '\x6f'),
                        ('ex', '\xbe'), ('ex', '\x29')],
        # The fourth size of '[' is mysteriously missing from the BaKoMa
        # font, so I've omitted it for both '[' and ']'
        '[':           [('rm', '['), ('ex', '\xa3'), ('ex', '\x68'),
                        ('ex', '\x22')],
        ']':           [('rm', ']'), ('ex', '\xa4'), ('ex', '\x69'),
                        ('ex', '\x23')],
        r'\lfloor':    [('ex', '\xa5'), ('ex', '\x6a'),
                        ('ex', '\xb9'), ('ex', '\x24')],
        r'\rfloor':    [('ex', '\xa6'), ('ex', '\x6b'),
                        ('ex', '\xba'), ('ex', '\x25')],
        r'\lceil':     [('ex', '\xa7'), ('ex', '\x6c'),
                        ('ex', '\xbb'), ('ex', '\x26')],
        r'\rceil':     [('ex', '\xa8'), ('ex', '\x6d'),
                        ('ex', '\xbc'), ('ex', '\x27')],
        r'\langle':    [('ex', '\xad'), ('ex', '\x44'),
                        ('ex', '\xbf'), ('ex', '\x2a')],
        r'\rangle':    [('ex', '\xae'), ('ex', '\x45'),
                        ('ex', '\xc0'), ('ex', '\x2b')],
        r'\__sqrt__':  [('ex', '\x70'), ('ex', '\x71'),
                        ('ex', '\x72'), ('ex', '\x73')],
        r'\backslash': [('ex', '\xb2'), ('ex', '\x2f'),
                        ('ex', '\xc2'), ('ex', '\x2d')],
        r'/':          [('rm', '/'), ('ex', '\xb1'), ('ex', '\x2e'),
                        ('ex', '\xcb'), ('ex', '\x2c')],
        r'\widehat':   [('rm', '\x5e'), ('ex', '\x62'), ('ex', '\x63'),
                        ('ex', '\x64')],
        r'\widetilde': [('rm', '\x7e'), ('ex', '\x65'), ('ex', '\x66'),
                        ('ex', '\x67')],
        r'<':          [('cal', 'h'), ('ex', 'D')],
        r'>':          [('cal', 'i'), ('ex', 'E')]
        }

    for alias, target in [(r'\leftparen', '('),
                          (r'\rightparent', ')'),
                          (r'\leftbrace', '{'),
                          (r'\rightbrace', '}'),
                          (r'\leftbracket', '['),
                          (r'\rightbracket', ']'),
                          (r'\{', '{'),
                          (r'\}', '}'),
                          (r'\[', '['),
                          (r'\]', ']')]:
        _size_alternatives[alias] = _size_alternatives[target]

    def get_sized_alternatives_for_symbol(self, fontname, sym):
        return self._size_alternatives.get(sym, [(fontname, sym)])


class UnicodeFonts(TruetypeFonts):
    """
    An abstract base class for handling Unicode fonts.

    While some reasonably complete Unicode fonts (such as DejaVu) may
    work in some situations, the only Unicode font I'm aware of with a
    complete set of math symbols is STIX.

    This class will "fallback" on the Bakoma fonts when a required
    symbol can not be found in the font.
    """
    use_cmex = True  # Unused; delete once mathtext becomes private.

    def __init__(self, *args, **kwargs):
        # This must come first so the backend's owner is set correctly
        fallback_rc = mpl.rcParams['mathtext.fallback']
        font_cls = {'stix': StixFonts,
                    'stixsans': StixSansFonts,
                    'cm': BakomaFonts
                    }.get(fallback_rc)
        self.cm_fallback = font_cls(*args, **kwargs) if font_cls else None

        super().__init__(*args, **kwargs)
        self.fontmap = {}
        for texfont in "cal rm tt it bf sf".split():
            prop = mpl.rcParams['mathtext.' + texfont]
            font = findfont(prop)
            self.fontmap[texfont] = font
        prop = FontProperties('cmex10')
        font = findfont(prop)
        self.fontmap['ex'] = font

        # include STIX sized alternatives for glyphs if fallback is STIX
        if isinstance(self.cm_fallback, StixFonts):
            stixsizedaltfonts = {
                 0: 'STIXGeneral',
                 1: 'STIXSizeOneSym',
                 2: 'STIXSizeTwoSym',
                 3: 'STIXSizeThreeSym',
                 4: 'STIXSizeFourSym',
                 5: 'STIXSizeFiveSym'}

            for size, name in stixsizedaltfonts.items():
                fullpath = findfont(name)
                self.fontmap[size] = fullpath
                self.fontmap[name] = fullpath

    _slanted_symbols = set(r"\int \oint".split())

    def _map_virtual_font(self, fontname, font_class, uniindex):
        return fontname, uniindex

    def _get_glyph(self, fontname, font_class, sym, fontsize, math=True):
        try:
            uniindex = get_unicode_index(sym, math)
            found_symbol = True
        except ValueError:
            uniindex = ord('?')
            found_symbol = False
            _log.warning("No TeX to unicode mapping for {!a}.".format(sym))

        fontname, uniindex = self._map_virtual_font(
            fontname, font_class, uniindex)

        new_fontname = fontname

        # Only characters in the "Letter" class should be italicized in 'it'
        # mode.  Greek capital letters should be Roman.
        if found_symbol:
            if fontname == 'it' and uniindex < 0x10000:
                char = chr(uniindex)
                if (unicodedata.category(char)[0] != "L"
                        or unicodedata.name(char).startswith("GREEK CAPITAL")):
                    new_fontname = 'rm'

            slanted = (new_fontname == 'it') or sym in self._slanted_symbols
            found_symbol = False
            font = self._get_font(new_fontname)
            if font is not None:
                if font.family_name == "cmr10" and uniindex == 0x2212:
                    # minus sign exists in cmsy10 (not cmr10)
                    font = get_font(
                        cbook._get_data_path("fonts/ttf/cmsy10.ttf"))
                    uniindex = 0xa1
                glyphindex = font.get_char_index(uniindex)
                if glyphindex != 0:
                    found_symbol = True

        if not found_symbol:
            if self.cm_fallback:
                if (fontname in ('it', 'regular')
                        and isinstance(self.cm_fallback, StixFonts)):
                    fontname = 'rm'

                g = self.cm_fallback._get_glyph(fontname, font_class,
                                                sym, fontsize)
                fname = g[0].family_name
                if fname in list(BakomaFonts._fontmap.values()):
                    fname = "Computer Modern"
                _log.info("Substituting symbol %s from %s", sym, fname)
                return g

            else:
                if (fontname in ('it', 'regular')
                        and isinstance(self, StixFonts)):
                    return self._get_glyph('rm', font_class, sym, fontsize)
                _log.warning("Font {!r} does not have a glyph for {!a} "
                             "[U+{:x}], substituting with a dummy "
                             "symbol.".format(new_fontname, sym, uniindex))
                fontname = 'rm'
                font = self._get_font(fontname)
                uniindex = 0xA4  # currency char, for lack of anything better
                glyphindex = font.get_char_index(uniindex)
                slanted = False

        symbol_name = font.get_glyph_name(glyphindex)
        return font, uniindex, symbol_name, fontsize, slanted

    def get_sized_alternatives_for_symbol(self, fontname, sym):
        if self.cm_fallback:
            return self.cm_fallback.get_sized_alternatives_for_symbol(
                fontname, sym)
        return [(fontname, sym)]


class DejaVuFonts(UnicodeFonts):
    use_cmex = False  # Unused; delete once mathtext becomes private.

    def __init__(self, *args, **kwargs):
        # This must come first so the backend's owner is set correctly
        if isinstance(self, DejaVuSerifFonts):
            self.cm_fallback = StixFonts(*args, **kwargs)
        else:
            self.cm_fallback = StixSansFonts(*args, **kwargs)
        self.bakoma = BakomaFonts(*args, **kwargs)
        TruetypeFonts.__init__(self, *args, **kwargs)
        self.fontmap = {}
        # Include Stix sized alternatives for glyphs
        self._fontmap.update({
            1: 'STIXSizeOneSym',
            2: 'STIXSizeTwoSym',
            3: 'STIXSizeThreeSym',
            4: 'STIXSizeFourSym',
            5: 'STIXSizeFiveSym',
        })
        for key, name in self._fontmap.items():
            fullpath = findfont(name)
            self.fontmap[key] = fullpath
            self.fontmap[name] = fullpath

    def _get_glyph(self, fontname, font_class, sym, fontsize, math=True):
        # Override prime symbol to use Bakoma.
        if sym == r'\prime':
            return self.bakoma._get_glyph(
                fontname, font_class, sym, fontsize, math)
        else:
            # check whether the glyph is available in the display font
            uniindex = get_unicode_index(sym)
            font = self._get_font('ex')
            if font is not None:
                glyphindex = font.get_char_index(uniindex)
                if glyphindex != 0:
                    return super()._get_glyph(
                        'ex', font_class, sym, fontsize, math)
            # otherwise return regular glyph
            return super()._get_glyph(
                fontname, font_class, sym, fontsize, math)


class DejaVuSerifFonts(DejaVuFonts):
    """
    A font handling class for the DejaVu Serif fonts

    If a glyph is not found it will fallback to Stix Serif
    """
    _fontmap = {
        'rm': 'DejaVu Serif',
        'it': 'DejaVu Serif:italic',
        'bf': 'DejaVu Serif:weight=bold',
        'sf': 'DejaVu Sans',
        'tt': 'DejaVu Sans Mono',
        'ex': 'DejaVu Serif Display',
        0:    'DejaVu Serif',
    }


class DejaVuSansFonts(DejaVuFonts):
    """
    A font handling class for the DejaVu Sans fonts

    If a glyph is not found it will fallback to Stix Sans
    """
    _fontmap = {
        'rm': 'DejaVu Sans',
        'it': 'DejaVu Sans:italic',
        'bf': 'DejaVu Sans:weight=bold',
        'sf': 'DejaVu Sans',
        'tt': 'DejaVu Sans Mono',
        'ex': 'DejaVu Sans Display',
        0:    'DejaVu Sans',
    }


class StixFonts(UnicodeFonts):
    """
    A font handling class for the STIX fonts.

    In addition to what UnicodeFonts provides, this class:

    - supports "virtual fonts" which are complete alpha numeric
      character sets with different font styles at special Unicode
      code points, such as "Blackboard".

    - handles sized alternative characters for the STIXSizeX fonts.
    """
    _fontmap = {
        'rm': 'STIXGeneral',
        'it': 'STIXGeneral:italic',
        'bf': 'STIXGeneral:weight=bold',
        'nonunirm': 'STIXNonUnicode',
        'nonuniit': 'STIXNonUnicode:italic',
        'nonunibf': 'STIXNonUnicode:weight=bold',
        0: 'STIXGeneral',
        1: 'STIXSizeOneSym',
        2: 'STIXSizeTwoSym',
        3: 'STIXSizeThreeSym',
        4: 'STIXSizeFourSym',
        5: 'STIXSizeFiveSym',
    }
    use_cmex = False  # Unused; delete once mathtext becomes private.
    cm_fallback = False
    _sans = False

    def __init__(self, *args, **kwargs):
        TruetypeFonts.__init__(self, *args, **kwargs)
        self.fontmap = {}
        for key, name in self._fontmap.items():
            fullpath = findfont(name)
            self.fontmap[key] = fullpath
            self.fontmap[name] = fullpath

    def _map_virtual_font(self, fontname, font_class, uniindex):
        # Handle these "fonts" that are actually embedded in
        # other fonts.
        mapping = stix_virtual_fonts.get(fontname)
        if (self._sans and mapping is None
                and fontname not in ('regular', 'default')):
            mapping = stix_virtual_fonts['sf']
            doing_sans_conversion = True
        else:
            doing_sans_conversion = False

        if mapping is not None:
            if isinstance(mapping, dict):
                try:
                    mapping = mapping[font_class]
                except KeyError:
                    mapping = mapping['rm']

            # Binary search for the source glyph
            lo = 0
            hi = len(mapping)
            while lo < hi:
                mid = (lo+hi)//2
                range = mapping[mid]
                if uniindex < range[0]:
                    hi = mid
                elif uniindex <= range[1]:
                    break
                else:
                    lo = mid + 1

            if range[0] <= uniindex <= range[1]:
                uniindex = uniindex - range[0] + range[3]
                fontname = range[2]
            elif not doing_sans_conversion:
                # This will generate a dummy character
                uniindex = 0x1
                fontname = mpl.rcParams['mathtext.default']

        # Fix some incorrect glyphs.
        if fontname in ('rm', 'it'):
            uniindex = stix_glyph_fixes.get(uniindex, uniindex)

        # Handle private use area glyphs
        if fontname in ('it', 'rm', 'bf') and 0xe000 <= uniindex <= 0xf8ff:
            fontname = 'nonuni' + fontname

        return fontname, uniindex

    @functools.lru_cache()
    def get_sized_alternatives_for_symbol(self, fontname, sym):
        fixes = {
            '\\{': '{', '\\}': '}', '\\[': '[', '\\]': ']',
            '<': '\N{MATHEMATICAL LEFT ANGLE BRACKET}',
            '>': '\N{MATHEMATICAL RIGHT ANGLE BRACKET}',
        }
        sym = fixes.get(sym, sym)
        try:
            uniindex = get_unicode_index(sym)
        except ValueError:
            return [(fontname, sym)]
        alternatives = [(i, chr(uniindex)) for i in range(6)
                        if self._get_font(i).get_char_index(uniindex) != 0]
        # The largest size of the radical symbol in STIX has incorrect
        # metrics that cause it to be disconnected from the stem.
        if sym == r'\__sqrt__':
            alternatives = alternatives[:-1]
        return alternatives


class StixSansFonts(StixFonts):
    """
    A font handling class for the STIX fonts (that uses sans-serif
    characters by default).
    """
    _sans = True


class StandardPsFonts(Fonts):
    """
    Use the standard postscript fonts for rendering to backend_ps

    Unlike the other font classes, BakomaFont and UnicodeFont, this
    one requires the Ps backend.
    """
    basepath = str(cbook._get_data_path('fonts/afm'))

    fontmap = {
        'cal': 'pzcmi8a',  # Zapf Chancery
        'rm':  'pncr8a',   # New Century Schoolbook
        'tt':  'pcrr8a',   # Courier
        'it':  'pncri8a',  # New Century Schoolbook Italic
        'sf':  'phvr8a',   # Helvetica
        'bf':  'pncb8a',   # New Century Schoolbook Bold
        None:  'psyr',     # Symbol
    }

    def __init__(self, default_font_prop, mathtext_backend=None):
        if mathtext_backend is None:
            # Circular import, can be dropped after public access to
            # StandardPsFonts is removed and mathtext_backend made a required
            # parameter.
            from . import mathtext
            mathtext_backend = mathtext.MathtextBackendPath()
        super().__init__(default_font_prop, mathtext_backend)
        self.glyphd = {}
        self.fonts = {}

        filename = findfont(default_font_prop, fontext='afm',
                            directory=self.basepath)
        if filename is None:
            filename = findfont('Helvetica', fontext='afm',
                                directory=self.basepath)
        with open(filename, 'rb') as fd:
            default_font = AFM(fd)
        default_font.fname = filename

        self.fonts['default'] = default_font
        self.fonts['regular'] = default_font

    pswriter = _api.deprecated("3.4")(property(lambda self: StringIO()))

    def _get_font(self, font):
        if font in self.fontmap:
            basename = self.fontmap[font]
        else:
            basename = font

        cached_font = self.fonts.get(basename)
        if cached_font is None:
            fname = os.path.join(self.basepath, basename + ".afm")
            with open(fname, 'rb') as fd:
                cached_font = AFM(fd)
            cached_font.fname = fname
            self.fonts[basename] = cached_font
            self.fonts[cached_font.get_fontname()] = cached_font
        return cached_font

    def _get_info(self, fontname, font_class, sym, fontsize, dpi, math=True):
        """Load the cmfont, metrics and glyph with caching."""
        key = fontname, sym, fontsize, dpi
        tup = self.glyphd.get(key)

        if tup is not None:
            return tup

        # Only characters in the "Letter" class should really be italicized.
        # This class includes greek letters, so we're ok
        if (fontname == 'it' and
                (len(sym) > 1
                 or not unicodedata.category(sym).startswith("L"))):
            fontname = 'rm'

        found_symbol = False

        if sym in latex_to_standard:
            fontname, num = latex_to_standard[sym]
            glyph = chr(num)
            found_symbol = True
        elif len(sym) == 1:
            glyph = sym
            num = ord(glyph)
            found_symbol = True
        else:
            _log.warning(
                "No TeX to built-in Postscript mapping for {!r}".format(sym))

        slanted = (fontname == 'it')
        font = self._get_font(fontname)

        if found_symbol:
            try:
                symbol_name = font.get_name_char(glyph)
            except KeyError:
                _log.warning(
                    "No glyph in standard Postscript font {!r} for {!r}"
                    .format(font.get_fontname(), sym))
                found_symbol = False

        if not found_symbol:
            glyph = '?'
            num = ord(glyph)
            symbol_name = font.get_name_char(glyph)

        offset = 0

        scale = 0.001 * fontsize

        xmin, ymin, xmax, ymax = [val * scale
                                  for val in font.get_bbox_char(glyph)]
        metrics = types.SimpleNamespace(
            advance  = font.get_width_char(glyph) * scale,
            width    = font.get_width_char(glyph) * scale,
            height   = font.get_height_char(glyph) * scale,
            xmin = xmin,
            xmax = xmax,
            ymin = ymin+offset,
            ymax = ymax+offset,
            # iceberg is the equivalent of TeX's "height"
            iceberg = ymax + offset,
            slanted = slanted
            )

        self.glyphd[key] = types.SimpleNamespace(
            font            = font,
            fontsize        = fontsize,
            postscript_name = font.get_fontname(),
            metrics         = metrics,
            symbol_name     = symbol_name,
            num             = num,
            glyph           = glyph,
            offset          = offset
            )

        return self.glyphd[key]

    def get_kern(self, font1, fontclass1, sym1, fontsize1,
                 font2, fontclass2, sym2, fontsize2, dpi):
        if font1 == font2 and fontsize1 == fontsize2:
            info1 = self._get_info(font1, fontclass1, sym1, fontsize1, dpi)
            info2 = self._get_info(font2, fontclass2, sym2, fontsize2, dpi)
            font = info1.font
            return (font.get_kern_dist(info1.glyph, info2.glyph)
                    * 0.001 * fontsize1)
        return super().get_kern(font1, fontclass1, sym1, fontsize1,
                                font2, fontclass2, sym2, fontsize2, dpi)

    def get_xheight(self, font, fontsize, dpi):
        font = self._get_font(font)
        return font.get_xheight() * 0.001 * fontsize

    def get_underline_thickness(self, font, fontsize, dpi):
        font = self._get_font(font)
        return font.get_underline_thickness() * 0.001 * fontsize


##############################################################################
# TeX-LIKE BOX MODEL

# The following is based directly on the document 'woven' from the
# TeX82 source code.  This information is also available in printed
# form:
#
#    Knuth, Donald E.. 1986.  Computers and Typesetting, Volume B:
#    TeX: The Program.  Addison-Wesley Professional.
#
# The most relevant "chapters" are:
#    Data structures for boxes and their friends
#    Shipping pages out (Ship class)
#    Packaging (hpack and vpack)
#    Data structures for math mode
#    Subroutines for math mode
#    Typesetting math formulas
#
# Many of the docstrings below refer to a numbered "node" in that
# book, e.g., node123
#
# Note that (as TeX) y increases downward, unlike many other parts of
# matplotlib.

# How much text shrinks when going to the next-smallest level.  GROW_FACTOR
# must be the inverse of SHRINK_FACTOR.
SHRINK_FACTOR   = 0.7
GROW_FACTOR     = 1 / SHRINK_FACTOR
# The number of different sizes of chars to use, beyond which they will not
# get any smaller
NUM_SIZE_LEVELS = 6


class FontConstantsBase:
    """
    A set of constants that controls how certain things, such as sub-
    and superscripts are laid out.  These are all metrics that can't
    be reliably retrieved from the font metrics in the font itself.
    """
    # Percentage of x-height of additional horiz. space after sub/superscripts
    script_space = 0.05

    # Percentage of x-height that sub/superscripts drop below the baseline
    subdrop = 0.4

    # Percentage of x-height that superscripts are raised from the baseline
    sup1 = 0.7

    # Percentage of x-height that subscripts drop below the baseline
    sub1 = 0.3

    # Percentage of x-height that subscripts drop below the baseline when a
    # superscript is present
    sub2 = 0.5

    # Percentage of x-height that sub/supercripts are offset relative to the
    # nucleus edge for non-slanted nuclei
    delta = 0.025

    # Additional percentage of last character height above 2/3 of the
    # x-height that supercripts are offset relative to the subscript
    # for slanted nuclei
    delta_slanted = 0.2

    # Percentage of x-height that supercripts and subscripts are offset for
    # integrals
    delta_integral = 0.1


class ComputerModernFontConstants(FontConstantsBase):
    script_space = 0.075
    subdrop = 0.2
    sup1 = 0.45
    sub1 = 0.2
    sub2 = 0.3
    delta = 0.075
    delta_slanted = 0.3
    delta_integral = 0.3


class STIXFontConstants(FontConstantsBase):
    script_space = 0.1
    sup1 = 0.8
    sub2 = 0.6
    delta = 0.05
    delta_slanted = 0.3
    delta_integral = 0.3


class STIXSansFontConstants(FontConstantsBase):
    script_space = 0.05
    sup1 = 0.8
    delta_slanted = 0.6
    delta_integral = 0.3


class DejaVuSerifFontConstants(FontConstantsBase):
    pass


class DejaVuSansFontConstants(FontConstantsBase):
    pass


# Maps font family names to the FontConstantBase subclass to use
_font_constant_mapping = {
    'DejaVu Sans': DejaVuSansFontConstants,
    'DejaVu Sans Mono': DejaVuSansFontConstants,
    'DejaVu Serif': DejaVuSerifFontConstants,
    'cmb10': ComputerModernFontConstants,
    'cmex10': ComputerModernFontConstants,
    'cmmi10': ComputerModernFontConstants,
    'cmr10': ComputerModernFontConstants,
    'cmss10': ComputerModernFontConstants,
    'cmsy10': ComputerModernFontConstants,
    'cmtt10': ComputerModernFontConstants,
    'STIXGeneral': STIXFontConstants,
    'STIXNonUnicode': STIXFontConstants,
    'STIXSizeFiveSym': STIXFontConstants,
    'STIXSizeFourSym': STIXFontConstants,
    'STIXSizeThreeSym': STIXFontConstants,
    'STIXSizeTwoSym': STIXFontConstants,
    'STIXSizeOneSym': STIXFontConstants,
    # Map the fonts we used to ship, just for good measure
    'Bitstream Vera Sans': DejaVuSansFontConstants,
    'Bitstream Vera': DejaVuSansFontConstants,
    }


def _get_font_constant_set(state):
    constants = _font_constant_mapping.get(
        state.font_output._get_font(state.font).family_name,
        FontConstantsBase)
    # STIX sans isn't really its own fonts, just different code points
    # in the STIX fonts, so we have to detect this one separately.
    if (constants is STIXFontConstants and
            isinstance(state.font_output, StixSansFonts)):
        return STIXSansFontConstants
    return constants


class Node:
    """A node in the TeX box model."""

    def __init__(self):
        self.size = 0

    def __repr__(self):
        return self.__class__.__name__

    def get_kerning(self, next):
        return 0.0

    def shrink(self):
        """
        Shrinks one level smaller.  There are only three levels of
        sizes, after which things will no longer get smaller.
        """
        self.size += 1

    def grow(self):
        """
        Grows one level larger.  There is no limit to how big
        something can get.
        """
        self.size -= 1

    def render(self, x, y):
        pass


class Box(Node):
    """A node with a physical location."""

    def __init__(self, width, height, depth):
        super().__init__()
        self.width  = width
        self.height = height
        self.depth  = depth

    def shrink(self):
        super().shrink()
        if self.size < NUM_SIZE_LEVELS:
            self.width  *= SHRINK_FACTOR
            self.height *= SHRINK_FACTOR
            self.depth  *= SHRINK_FACTOR

    def grow(self):
        super().grow()
        self.width  *= GROW_FACTOR
        self.height *= GROW_FACTOR
        self.depth  *= GROW_FACTOR

    def render(self, x1, y1, x2, y2):
        pass


class Vbox(Box):
    """A box with only height (zero width)."""

    def __init__(self, height, depth):
        super().__init__(0., height, depth)


class Hbox(Box):
    """A box with only width (zero height and depth)."""

    def __init__(self, width):
        super().__init__(width, 0., 0.)


class Char(Node):
    """
    A single character.

    Unlike TeX, the font information and metrics are stored with each `Char`
    to make it easier to lookup the font metrics when needed.  Note that TeX
    boxes have a width, height, and depth, unlike Type1 and TrueType which use
    a full bounding box and an advance in the x-direction.  The metrics must
    be converted to the TeX model, and the advance (if different from width)
    must be converted into a `Kern` node when the `Char` is added to its parent
    `Hlist`.
    """

    def __init__(self, c, state, math=True):
        super().__init__()
        self.c = c
        self.font_output = state.font_output
        self.font = state.font
        self.font_class = state.font_class
        self.fontsize = state.fontsize
        self.dpi = state.dpi
        self.math = math
        # The real width, height and depth will be set during the
        # pack phase, after we know the real fontsize
        self._update_metrics()

    def __repr__(self):
        return '`%s`' % self.c

    def _update_metrics(self):
        metrics = self._metrics = self.font_output.get_metrics(
            self.font, self.font_class, self.c, self.fontsize, self.dpi,
            self.math)
        if self.c == ' ':
            self.width = metrics.advance
        else:
            self.width = metrics.width
        self.height = metrics.iceberg
        self.depth = -(metrics.iceberg - metrics.height)

    def is_slanted(self):
        return self._metrics.slanted

    def get_kerning(self, next):
        """
        Return the amount of kerning between this and the given character.

        This method is called when characters are strung together into `Hlist`
        to create `Kern` nodes.
        """
        advance = self._metrics.advance - self.width
        kern = 0.
        if isinstance(next, Char):
            kern = self.font_output.get_kern(
                self.font, self.font_class, self.c, self.fontsize,
                next.font, next.font_class, next.c, next.fontsize,
                self.dpi)
        return advance + kern

    def render(self, x, y):
        """
        Render the character to the canvas
        """
        self.font_output.render_glyph(
            x, y,
            self.font, self.font_class, self.c, self.fontsize, self.dpi)

    def shrink(self):
        super().shrink()
        if self.size < NUM_SIZE_LEVELS:
            self.fontsize *= SHRINK_FACTOR
            self.width    *= SHRINK_FACTOR
            self.height   *= SHRINK_FACTOR
            self.depth    *= SHRINK_FACTOR

    def grow(self):
        super().grow()
        self.fontsize *= GROW_FACTOR
        self.width    *= GROW_FACTOR
        self.height   *= GROW_FACTOR
        self.depth    *= GROW_FACTOR


class Accent(Char):
    """
    The font metrics need to be dealt with differently for accents,
    since they are already offset correctly from the baseline in
    TrueType fonts.
    """
    def _update_metrics(self):
        metrics = self._metrics = self.font_output.get_metrics(
            self.font, self.font_class, self.c, self.fontsize, self.dpi)
        self.width = metrics.xmax - metrics.xmin
        self.height = metrics.ymax - metrics.ymin
        self.depth = 0

    def shrink(self):
        super().shrink()
        self._update_metrics()

    def grow(self):
        super().grow()
        self._update_metrics()

    def render(self, x, y):
        """
        Render the character to the canvas.
        """
        self.font_output.render_glyph(
            x - self._metrics.xmin, y + self._metrics.ymin,
            self.font, self.font_class, self.c, self.fontsize, self.dpi)


class List(Box):
    """A list of nodes (either horizontal or vertical)."""

    def __init__(self, elements):
        super().__init__(0., 0., 0.)
        self.shift_amount = 0.   # An arbitrary offset
        self.children     = elements  # The child nodes of this list
        # The following parameters are set in the vpack and hpack functions
        self.glue_set     = 0.   # The glue setting of this list
        self.glue_sign    = 0    # 0: normal, -1: shrinking, 1: stretching
        self.glue_order   = 0    # The order of infinity (0 - 3) for the glue

    def __repr__(self):
        return '[%s <%.02f %.02f %.02f %.02f> %s]' % (
            super().__repr__(),
            self.width, self.height,
            self.depth, self.shift_amount,
            ' '.join([repr(x) for x in self.children]))

    @staticmethod
    def _determine_order(totals):
        """
        Determine the highest order of glue used by the members of this list.

        Helper function used by vpack and hpack.
        """
        for i in range(len(totals))[::-1]:
            if totals[i] != 0:
                return i
        return 0

    def _set_glue(self, x, sign, totals, error_type):
        o = self._determine_order(totals)
        self.glue_order = o
        self.glue_sign = sign
        if totals[o] != 0.:
            self.glue_set = x / totals[o]
        else:
            self.glue_sign = 0
            self.glue_ratio = 0.
        if o == 0:
            if len(self.children):
                _log.warning("%s %s: %r",
                             error_type, self.__class__.__name__, self)

    def shrink(self):
        for child in self.children:
            child.shrink()
        super().shrink()
        if self.size < NUM_SIZE_LEVELS:
            self.shift_amount *= SHRINK_FACTOR
            self.glue_set     *= SHRINK_FACTOR

    def grow(self):
        for child in self.children:
            child.grow()
        super().grow()
        self.shift_amount *= GROW_FACTOR
        self.glue_set     *= GROW_FACTOR


class Hlist(List):
    """A horizontal list of boxes."""

    def __init__(self, elements, w=0., m='additional', do_kern=True):
        super().__init__(elements)
        if do_kern:
            self.kern()
        self.hpack()

    def kern(self):
        """
        Insert `Kern` nodes between `Char` nodes to set kerning.

        The `Char` nodes themselves determine the amount of kerning they need
        (in `~Char.get_kerning`), and this function just creates the correct
        linked list.
        """
        new_children = []
        num_children = len(self.children)
        if num_children:
            for i in range(num_children):
                elem = self.children[i]
                if i < num_children - 1:
                    next = self.children[i + 1]
                else:
                    next = None

                new_children.append(elem)
                kerning_distance = elem.get_kerning(next)
                if kerning_distance != 0.:
                    kern = Kern(kerning_distance)
                    new_children.append(kern)
            self.children = new_children

    # This is a failed experiment to fake cross-font kerning.
#     def get_kerning(self, next):
#         if len(self.children) >= 2 and isinstance(self.children[-2], Char):
#             if isinstance(next, Char):
#                 print "CASE A"
#                 return self.children[-2].get_kerning(next)
#             elif (isinstance(next, Hlist) and len(next.children)
#                   and isinstance(next.children[0], Char)):
#                 print "CASE B"
#                 result = self.children[-2].get_kerning(next.children[0])
#                 print result
#                 return result
#         return 0.0

    def hpack(self, w=0., m='additional'):
        r"""
        Compute the dimensions of the resulting boxes, and adjust the glue if
        one of those dimensions is pre-specified.  The computed sizes normally
        enclose all of the material inside the new box; but some items may
        stick out if negative glue is used, if the box is overfull, or if a
        ``\vbox`` includes other boxes that have been shifted left.

        Parameters
        ----------
        w : float, default: 0
            A width.
        m : {'exactly', 'additional'}, default: 'additional'
            Whether to produce a box whose width is 'exactly' *w*; or a box
            with the natural width of the contents, plus *w* ('additional').

        Notes
        -----
        The defaults produce a box with the natural width of the contents.
        """
        # I don't know why these get reset in TeX.  Shift_amount is pretty
        # much useless if we do.
        # self.shift_amount = 0.
        h = 0.
        d = 0.
        x = 0.
        total_stretch = [0.] * 4
        total_shrink = [0.] * 4
        for p in self.children:
            if isinstance(p, Char):
                x += p.width
                h = max(h, p.height)
                d = max(d, p.depth)
            elif isinstance(p, Box):
                x += p.width
                if not np.isinf(p.height) and not np.isinf(p.depth):
                    s = getattr(p, 'shift_amount', 0.)
                    h = max(h, p.height - s)
                    d = max(d, p.depth + s)
            elif isinstance(p, Glue):
                glue_spec = p.glue_spec
                x += glue_spec.width
                total_stretch[glue_spec.stretch_order] += glue_spec.stretch
                total_shrink[glue_spec.shrink_order] += glue_spec.shrink
            elif isinstance(p, Kern):
                x += p.width
        self.height = h
        self.depth = d

        if m == 'additional':
            w += x
        self.width = w
        x = w - x

        if x == 0.:
            self.glue_sign = 0
            self.glue_order = 0
            self.glue_ratio = 0.
            return
        if x > 0.:
            self._set_glue(x, 1, total_stretch, "Overfull")
        else:
            self._set_glue(x, -1, total_shrink, "Underfull")


class Vlist(List):
    """A vertical list of boxes."""

    def __init__(self, elements, h=0., m='additional'):
        super().__init__(elements)
        self.vpack()

    def vpack(self, h=0., m='additional', l=np.inf):
        """
        Compute the dimensions of the resulting boxes, and to adjust the glue
        if one of those dimensions is pre-specified.

        Parameters
        ----------
        h : float, default: 0
            A height.
        m : {'exactly', 'additional'}, default: 'additional'
            Whether to produce a box whose height is 'exactly' *w*; or a box
            with the natural height of the contents, plus *w* ('additional').
        l : float, default: np.inf
            The maximum height.

        Notes
        -----
        The defaults produce a box with the natural height of the contents.
        """
        # I don't know why these get reset in TeX.  Shift_amount is pretty
        # much useless if we do.
        # self.shift_amount = 0.
        w = 0.
        d = 0.
        x = 0.
        total_stretch = [0.] * 4
        total_shrink = [0.] * 4
        for p in self.children:
            if isinstance(p, Box):
                x += d + p.height
                d = p.depth
                if not np.isinf(p.width):
                    s = getattr(p, 'shift_amount', 0.)
                    w = max(w, p.width + s)
            elif isinstance(p, Glue):
                x += d
                d = 0.
                glue_spec = p.glue_spec
                x += glue_spec.width
                total_stretch[glue_spec.stretch_order] += glue_spec.stretch
                total_shrink[glue_spec.shrink_order] += glue_spec.shrink
            elif isinstance(p, Kern):
                x += d + p.width
                d = 0.
            elif isinstance(p, Char):
                raise RuntimeError(
                    "Internal mathtext error: Char node found in Vlist")

        self.width = w
        if d > l:
            x += d - l
            self.depth = l
        else:
            self.depth = d

        if m == 'additional':
            h += x
        self.height = h
        x = h - x

        if x == 0:
            self.glue_sign = 0
            self.glue_order = 0
            self.glue_ratio = 0.
            return

        if x > 0.:
            self._set_glue(x, 1, total_stretch, "Overfull")
        else:
            self._set_glue(x, -1, total_shrink, "Underfull")


class Rule(Box):
    """
    A solid black rectangle.

    It has *width*, *depth*, and *height* fields just as in an `Hlist`.
    However, if any of these dimensions is inf, the actual value will be
    determined by running the rule up to the boundary of the innermost
    enclosing box.  This is called a "running dimension".  The width is never
    running in an `Hlist`; the height and depth are never running in a `Vlist`.
    """

    def __init__(self, width, height, depth, state):
        super().__init__(width, height, depth)
        self.font_output = state.font_output

    def render(self, x, y, w, h):
        self.font_output.render_rect_filled(x, y, x + w, y + h)


class Hrule(Rule):
    """Convenience class to create a horizontal rule."""

    def __init__(self, state, thickness=None):
        if thickness is None:
            thickness = state.font_output.get_underline_thickness(
                state.font, state.fontsize, state.dpi)
        height = depth = thickness * 0.5
        super().__init__(np.inf, height, depth, state)


class Vrule(Rule):
    """Convenience class to create a vertical rule."""

    def __init__(self, state):
        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)
        super().__init__(thickness, np.inf, np.inf, state)


_GlueSpec = namedtuple(
    "_GlueSpec", "width stretch stretch_order shrink shrink_order")
_GlueSpec._named = {
    'fil':         _GlueSpec(0., 1., 1, 0., 0),
    'fill':        _GlueSpec(0., 1., 2, 0., 0),
    'filll':       _GlueSpec(0., 1., 3, 0., 0),
    'neg_fil':     _GlueSpec(0., 0., 0, 1., 1),
    'neg_fill':    _GlueSpec(0., 0., 0, 1., 2),
    'neg_filll':   _GlueSpec(0., 0., 0, 1., 3),
    'empty':       _GlueSpec(0., 0., 0, 0., 0),
    'ss':          _GlueSpec(0., 1., 1, -1., 1),
}


class Glue(Node):
    """
    Most of the information in this object is stored in the underlying
    ``_GlueSpec`` class, which is shared between multiple glue objects.
    (This is a memory optimization which probably doesn't matter anymore, but
    it's easier to stick to what TeX does.)
    """

    def __init__(self, glue_type):
        super().__init__()
        if isinstance(glue_type, str):
            glue_spec = _GlueSpec._named[glue_type]
        elif isinstance(glue_type, _GlueSpec):
            glue_spec = glue_type
        else:
            raise ValueError("glue_type must be a glue spec name or instance")
        self.glue_spec = glue_spec

    def shrink(self):
        super().shrink()
        if self.size < NUM_SIZE_LEVELS:
            g = self.glue_spec
            self.glue_spec = g._replace(width=g.width * SHRINK_FACTOR)

    def grow(self):
        super().grow()
        g = self.glue_spec
        self.glue_spec = g._replace(width=g.width * GROW_FACTOR)


class HCentered(Hlist):
    """
    A convenience class to create an `Hlist` whose contents are
    centered within its enclosing box.
    """

    def __init__(self, elements):
        super().__init__([Glue('ss'), *elements, Glue('ss')], do_kern=False)


class VCentered(Vlist):
    """
    A convenience class to create a `Vlist` whose contents are
    centered within its enclosing box.
    """

    def __init__(self, elements):
        super().__init__([Glue('ss'), *elements, Glue('ss')])


class Kern(Node):
    """
    A `Kern` node has a width field to specify a (normally
    negative) amount of spacing. This spacing correction appears in
    horizontal lists between letters like A and V when the font
    designer said that it looks better to move them closer together or
    further apart. A kern node can also appear in a vertical list,
    when its *width* denotes additional spacing in the vertical
    direction.
    """

    height = 0
    depth = 0

    def __init__(self, width):
        super().__init__()
        self.width = width

    def __repr__(self):
        return "k%.02f" % self.width

    def shrink(self):
        super().shrink()
        if self.size < NUM_SIZE_LEVELS:
            self.width *= SHRINK_FACTOR

    def grow(self):
        super().grow()
        self.width *= GROW_FACTOR


class SubSuperCluster(Hlist):
    """
    A hack to get around that fact that this code does a two-pass parse like
    TeX.  This lets us store enough information in the hlist itself, namely the
    nucleus, sub- and super-script, such that if another script follows that
    needs to be attached, it can be reconfigured on the fly.
    """

    def __init__(self):
        self.nucleus = None
        self.sub = None
        self.super = None
        super().__init__([])


class AutoHeightChar(Hlist):
    """
    A character as close to the given height and depth as possible.

    When using a font with multiple height versions of some characters (such as
    the BaKoMa fonts), the correct glyph will be selected, otherwise this will
    always just return a scaled version of the glyph.
    """

    def __init__(self, c, height, depth, state, always=False, factor=None):
        alternatives = state.font_output.get_sized_alternatives_for_symbol(
            state.font, c)

        xHeight = state.font_output.get_xheight(
            state.font, state.fontsize, state.dpi)

        state = state.copy()
        target_total = height + depth
        for fontname, sym in alternatives:
            state.font = fontname
            char = Char(sym, state)
            # Ensure that size 0 is chosen when the text is regular sized but
            # with descender glyphs by subtracting 0.2 * xHeight
            if char.height + char.depth >= target_total - 0.2 * xHeight:
                break

        shift = 0
        if state.font != 0:
            if factor is None:
                factor = target_total / (char.height + char.depth)
            state.fontsize *= factor
            char = Char(sym, state)

            shift = (depth - char.depth)

        super().__init__([char])
        self.shift_amount = shift


class AutoWidthChar(Hlist):
    """
    A character as close to the given width as possible.

    When using a font with multiple width versions of some characters (such as
    the BaKoMa fonts), the correct glyph will be selected, otherwise this will
    always just return a scaled version of the glyph.
    """

    def __init__(self, c, width, state, always=False, char_class=Char):
        alternatives = state.font_output.get_sized_alternatives_for_symbol(
            state.font, c)

        state = state.copy()
        for fontname, sym in alternatives:
            state.font = fontname
            char = char_class(sym, state)
            if char.width >= width:
                break

        factor = width / char.width
        state.fontsize *= factor
        char = char_class(sym, state)

        super().__init__([char])
        self.width = char.width


class Ship:
    """
    Ship boxes to output once they have been set up, this sends them to output.

    Since boxes can be inside of boxes inside of boxes, the main work of `Ship`
    is done by two mutually recursive routines, `hlist_out` and `vlist_out`,
    which traverse the `Hlist` nodes and `Vlist` nodes inside of horizontal
    and vertical boxes.  The global variables used in TeX to store state as it
    processes have become member variables here.
    """

    def __call__(self, ox, oy, box):
        self.max_push    = 0  # Deepest nesting of push commands so far
        self.cur_s       = 0
        self.cur_v       = 0.
        self.cur_h       = 0.
        self.off_h       = ox
        self.off_v       = oy + box.height
        self.hlist_out(box)

    @staticmethod
    def clamp(value):
        if value < -1000000000.:
            return -1000000000.
        if value > 1000000000.:
            return 1000000000.
        return value

    def hlist_out(self, box):
        cur_g         = 0
        cur_glue      = 0.
        glue_order    = box.glue_order
        glue_sign     = box.glue_sign
        base_line     = self.cur_v
        left_edge     = self.cur_h
        self.cur_s    += 1
        self.max_push = max(self.cur_s, self.max_push)
        clamp         = self.clamp

        for p in box.children:
            if isinstance(p, Char):
                p.render(self.cur_h + self.off_h, self.cur_v + self.off_v)
                self.cur_h += p.width
            elif isinstance(p, Kern):
                self.cur_h += p.width
            elif isinstance(p, List):
                # node623
                if len(p.children) == 0:
                    self.cur_h += p.width
                else:
                    edge = self.cur_h
                    self.cur_v = base_line + p.shift_amount
                    if isinstance(p, Hlist):
                        self.hlist_out(p)
                    else:
                        # p.vpack(box.height + box.depth, 'exactly')
                        self.vlist_out(p)
                    self.cur_h = edge + p.width
                    self.cur_v = base_line
            elif isinstance(p, Box):
                # node624
                rule_height = p.height
                rule_depth  = p.depth
                rule_width  = p.width
                if np.isinf(rule_height):
                    rule_height = box.height
                if np.isinf(rule_depth):
                    rule_depth = box.depth
                if rule_height > 0 and rule_width > 0:
                    self.cur_v = base_line + rule_depth
                    p.render(self.cur_h + self.off_h,
                             self.cur_v + self.off_v,
                             rule_width, rule_height)
                    self.cur_v = base_line
                self.cur_h += rule_width
            elif isinstance(p, Glue):
                # node625
                glue_spec = p.glue_spec
                rule_width = glue_spec.width - cur_g
                if glue_sign != 0:  # normal
                    if glue_sign == 1:  # stretching
                        if glue_spec.stretch_order == glue_order:
                            cur_glue += glue_spec.stretch
                            cur_g = round(clamp(box.glue_set * cur_glue))
                    elif glue_spec.shrink_order == glue_order:
                        cur_glue += glue_spec.shrink
                        cur_g = round(clamp(box.glue_set * cur_glue))
                rule_width += cur_g
                self.cur_h += rule_width
        self.cur_s -= 1

    def vlist_out(self, box):
        cur_g         = 0
        cur_glue      = 0.
        glue_order    = box.glue_order
        glue_sign     = box.glue_sign
        self.cur_s    += 1
        self.max_push = max(self.max_push, self.cur_s)
        left_edge     = self.cur_h
        self.cur_v    -= box.height
        top_edge      = self.cur_v
        clamp         = self.clamp

        for p in box.children:
            if isinstance(p, Kern):
                self.cur_v += p.width
            elif isinstance(p, List):
                if len(p.children) == 0:
                    self.cur_v += p.height + p.depth
                else:
                    self.cur_v += p.height
                    self.cur_h = left_edge + p.shift_amount
                    save_v = self.cur_v
                    p.width = box.width
                    if isinstance(p, Hlist):
                        self.hlist_out(p)
                    else:
                        self.vlist_out(p)
                    self.cur_v = save_v + p.depth
                    self.cur_h = left_edge
            elif isinstance(p, Box):
                rule_height = p.height
                rule_depth = p.depth
                rule_width = p.width
                if np.isinf(rule_width):
                    rule_width = box.width
                rule_height += rule_depth
                if rule_height > 0 and rule_depth > 0:
                    self.cur_v += rule_height
                    p.render(self.cur_h + self.off_h,
                             self.cur_v + self.off_v,
                             rule_width, rule_height)
            elif isinstance(p, Glue):
                glue_spec = p.glue_spec
                rule_height = glue_spec.width - cur_g
                if glue_sign != 0:  # normal
                    if glue_sign == 1:  # stretching
                        if glue_spec.stretch_order == glue_order:
                            cur_glue += glue_spec.stretch
                            cur_g = round(clamp(box.glue_set * cur_glue))
                    elif glue_spec.shrink_order == glue_order:  # shrinking
                        cur_glue += glue_spec.shrink
                        cur_g = round(clamp(box.glue_set * cur_glue))
                rule_height += cur_g
                self.cur_v += rule_height
            elif isinstance(p, Char):
                raise RuntimeError(
                    "Internal mathtext error: Char node found in vlist")
        self.cur_s -= 1


ship = Ship()


##############################################################################
# PARSER


def Error(msg):
    """Helper class to raise parser errors."""
    def raise_error(s, loc, toks):
        raise ParseFatalException(s, loc, msg)

    empty = Empty()
    empty.setParseAction(raise_error)
    return empty


class Parser:
    """
    A pyparsing-based parser for strings containing math expressions.

    Raw text may also appear outside of pairs of ``$``.

    The grammar is based directly on that in TeX, though it cuts a few corners.
    """

    class _MathStyle(enum.Enum):
        DISPLAYSTYLE = enum.auto()
        TEXTSTYLE = enum.auto()
        SCRIPTSTYLE = enum.auto()
        SCRIPTSCRIPTSTYLE = enum.auto()

    _binary_operators = set('''
      + * -
      \\pm             \\sqcap                   \\rhd
      \\mp             \\sqcup                   \\unlhd
      \\times          \\vee                     \\unrhd
      \\div            \\wedge                   \\oplus
      \\ast            \\setminus                \\ominus
      \\star           \\wr                      \\otimes
      \\circ           \\diamond                 \\oslash
      \\bullet         \\bigtriangleup           \\odot
      \\cdot           \\bigtriangledown         \\bigcirc
      \\cap            \\triangleleft            \\dagger
      \\cup            \\triangleright           \\ddagger
      \\uplus          \\lhd                     \\amalg'''.split())

    _relation_symbols = set('''
      = < > :
      \\leq        \\geq        \\equiv   \\models
      \\prec       \\succ       \\sim     \\perp
      \\preceq     \\succeq     \\simeq   \\mid
      \\ll         \\gg         \\asymp   \\parallel
      \\subset     \\supset     \\approx  \\bowtie
      \\subseteq   \\supseteq   \\cong    \\Join
      \\sqsubset   \\sqsupset   \\neq     \\smile
      \\sqsubseteq \\sqsupseteq \\doteq   \\frown
      \\in         \\ni         \\propto  \\vdash
      \\dashv      \\dots       \\dotplus \\doteqdot'''.split())

    _arrow_symbols = set('''
      \\leftarrow              \\longleftarrow           \\uparrow
      \\Leftarrow              \\Longleftarrow           \\Uparrow
      \\rightarrow             \\longrightarrow          \\downarrow
      \\Rightarrow             \\Longrightarrow          \\Downarrow
      \\leftrightarrow         \\longleftrightarrow      \\updownarrow
      \\Leftrightarrow         \\Longleftrightarrow      \\Updownarrow
      \\mapsto                 \\longmapsto              \\nearrow
      \\hookleftarrow          \\hookrightarrow          \\searrow
      \\leftharpoonup          \\rightharpoonup          \\swarrow
      \\leftharpoondown        \\rightharpoondown        \\nwarrow
      \\rightleftharpoons      \\leadsto'''.split())

    _spaced_symbols = _binary_operators | _relation_symbols | _arrow_symbols

    _punctuation_symbols = set(r', ; . ! \ldotp \cdotp'.split())

    _overunder_symbols = set(r'''
       \sum \prod \coprod \bigcap \bigcup \bigsqcup \bigvee
       \bigwedge \bigodot \bigotimes \bigoplus \biguplus
       '''.split())

    _overunder_functions = set(
        "lim liminf limsup sup max min".split())

    _dropsub_symbols = set(r'''\int \oint'''.split())

    _fontnames = set("rm cal it tt sf bf default bb frak scr regular".split())

    _function_names = set("""
      arccos csc ker min arcsin deg lg Pr arctan det lim sec arg dim
      liminf sin cos exp limsup sinh cosh gcd ln sup cot hom log tan
      coth inf max tanh""".split())

    _ambi_delim = set("""
      | \\| / \\backslash \\uparrow \\downarrow \\updownarrow \\Uparrow
      \\Downarrow \\Updownarrow . \\vert \\Vert \\\\|""".split())

    _left_delim = set(r"( [ \{ < \lfloor \langle \lceil".split())

    _right_delim = set(r") ] \} > \rfloor \rangle \rceil".split())

    def __init__(self):
        p = types.SimpleNamespace()
        # All forward declarations are here
        p.accent           = Forward()
        p.ambi_delim       = Forward()
        p.apostrophe       = Forward()
        p.auto_delim       = Forward()
        p.binom            = Forward()
        p.bslash           = Forward()
        p.c_over_c         = Forward()
        p.customspace      = Forward()
        p.end_group        = Forward()
        p.float_literal    = Forward()
        p.font             = Forward()
        p.frac             = Forward()
        p.dfrac            = Forward()
        p.function         = Forward()
        p.genfrac          = Forward()
        p.group            = Forward()
        p.int_literal      = Forward()
        p.latexfont        = Forward()
        p.lbracket         = Forward()
        p.left_delim       = Forward()
        p.lbrace           = Forward()
        p.main             = Forward()
        p.math             = Forward()
        p.math_string      = Forward()
        p.non_math         = Forward()
        p.operatorname     = Forward()
        p.overline         = Forward()
        p.overset          = Forward()
        p.placeable        = Forward()
        p.rbrace           = Forward()
        p.rbracket         = Forward()
        p.required_group   = Forward()
        p.right_delim      = Forward()
        p.right_delim_safe = Forward()
        p.simple           = Forward()
        p.simple_group     = Forward()
        p.single_symbol    = Forward()
        p.accentprefixed   = Forward()
        p.space            = Forward()
        p.sqrt             = Forward()
        p.start_group      = Forward()
        p.subsuper         = Forward()
        p.subsuperop       = Forward()
        p.symbol           = Forward()
        p.symbol_name      = Forward()
        p.token            = Forward()
        p.underset         = Forward()
        p.unknown_symbol   = Forward()

        # Set names on everything -- very useful for debugging
        for key, val in vars(p).items():
            if not key.startswith('_'):
                val.setName(key)

        p.float_literal <<= Regex(r"[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)")
        p.int_literal   <<= Regex("[-+]?[0-9]+")

        p.lbrace        <<= Literal('{').suppress()
        p.rbrace        <<= Literal('}').suppress()
        p.lbracket      <<= Literal('[').suppress()
        p.rbracket      <<= Literal(']').suppress()
        p.bslash        <<= Literal('\\')

        p.space         <<= oneOf(list(self._space_widths))
        p.customspace   <<= (
            Suppress(Literal(r'\hspace'))
            - ((p.lbrace + p.float_literal + p.rbrace)
               | Error(r"Expected \hspace{n}"))
        )

        unicode_range = "\U00000080-\U0001ffff"
        p.single_symbol <<= Regex(
            r"([a-zA-Z0-9 +\-*/<>=:,.;!\?&'@()\[\]|%s])|(\\[%%${}\[\]_|])" %
            unicode_range)
        p.accentprefixed <<= Suppress(p.bslash) + oneOf(self._accentprefixed)
        p.symbol_name   <<= (
            Combine(p.bslash + oneOf(list(tex2uni)))
            + Suppress(Regex("(?=[^A-Za-z]|$)").leaveWhitespace())
        )
        p.symbol        <<= (p.single_symbol | p.symbol_name).leaveWhitespace()

        p.apostrophe    <<= Regex("'+")

        p.c_over_c      <<= (
            Suppress(p.bslash)
            + oneOf(list(self._char_over_chars))
        )

        p.accent        <<= Group(
            Suppress(p.bslash)
            + oneOf([*self._accent_map, *self._wide_accents])
            + Suppress(Optional(White()))
            - p.placeable
        )

        p.function      <<= (
            Suppress(p.bslash)
            + oneOf(list(self._function_names))
        )

        p.start_group    <<= Optional(p.latexfont) + p.lbrace
        p.end_group      <<= p.rbrace.copy()
        p.simple_group   <<= Group(p.lbrace + ZeroOrMore(p.token) + p.rbrace)
        p.required_group <<= Group(p.lbrace + OneOrMore(p.token) + p.rbrace)
        p.group          <<= Group(
            p.start_group + ZeroOrMore(p.token) + p.end_group
        )

        p.font          <<= Suppress(p.bslash) + oneOf(list(self._fontnames))
        p.latexfont     <<= (
            Suppress(p.bslash)
            + oneOf(['math' + x for x in self._fontnames])
        )

        p.frac          <<= Group(
            Suppress(Literal(r"\frac"))
            - ((p.required_group + p.required_group)
               | Error(r"Expected \frac{num}{den}"))
        )

        p.dfrac         <<= Group(
            Suppress(Literal(r"\dfrac"))
            - ((p.required_group + p.required_group)
               | Error(r"Expected \dfrac{num}{den}"))
        )

        p.binom         <<= Group(
            Suppress(Literal(r"\binom"))
            - ((p.required_group + p.required_group)
               | Error(r"Expected \binom{num}{den}"))
        )

        p.ambi_delim    <<= oneOf(list(self._ambi_delim))
        p.left_delim    <<= oneOf(list(self._left_delim))
        p.right_delim   <<= oneOf(list(self._right_delim))
        p.right_delim_safe <<= oneOf([*(self._right_delim - {'}'}), r'\}'])

        p.genfrac <<= Group(
            Suppress(Literal(r"\genfrac"))
            - (((p.lbrace
                 + Optional(p.ambi_delim | p.left_delim, default='')
                 + p.rbrace)
                + (p.lbrace
                   + Optional(p.ambi_delim | p.right_delim_safe, default='')
                   + p.rbrace)
                + (p.lbrace + p.float_literal + p.rbrace)
                + p.simple_group + p.required_group + p.required_group)
               | Error("Expected "
                       r"\genfrac{ldelim}{rdelim}{rulesize}{style}{num}{den}"))
        )

        p.sqrt <<= Group(
            Suppress(Literal(r"\sqrt"))
            - ((Group(Optional(
                p.lbracket + OneOrMore(~p.rbracket + p.token) + p.rbracket))
                + p.required_group)
               | Error("Expected \\sqrt{value}"))
        )

        p.overline <<= Group(
            Suppress(Literal(r"\overline"))
            - (p.required_group | Error("Expected \\overline{value}"))
        )

        p.overset <<= Group(
            Suppress(Literal(r"\overset"))
            - ((p.simple_group + p.simple_group)
               | Error("Expected \\overset{body}{annotation}"))
        )

        p.underset <<= Group(
            Suppress(Literal(r"\underset"))
            - ((p.simple_group + p.simple_group)
               | Error("Expected \\underset{body}{annotation}"))
        )

        p.unknown_symbol <<= Combine(p.bslash + Regex("[A-Za-z]*"))

        p.operatorname <<= Group(
            Suppress(Literal(r"\operatorname"))
            - ((p.lbrace + ZeroOrMore(p.simple | p.unknown_symbol) + p.rbrace)
               | Error("Expected \\operatorname{value}"))
        )

        p.placeable     <<= (
            p.accentprefixed  # Must be before accent so named symbols that are
                              # prefixed with an accent name work
            | p.accent   # Must be before symbol as all accents are symbols
            | p.symbol   # Must be third to catch all named symbols and single
                         # chars not in a group
            | p.c_over_c
            | p.function
            | p.group
            | p.frac
            | p.dfrac
            | p.binom
            | p.genfrac
            | p.overset
            | p.underset
            | p.sqrt
            | p.overline
            | p.operatorname
        )

        p.simple        <<= (
            p.space
            | p.customspace
            | p.font
            | p.subsuper
        )

        p.subsuperop    <<= oneOf(["_", "^"])

        p.subsuper      <<= Group(
            (Optional(p.placeable)
             + OneOrMore(p.subsuperop - p.placeable)
             + Optional(p.apostrophe))
            | (p.placeable + Optional(p.apostrophe))
            | p.apostrophe
        )

        p.token         <<= (
            p.simple
            | p.auto_delim
            | p.unknown_symbol  # Must be last
        )

        p.auto_delim    <<= (
            Suppress(Literal(r"\left"))
            - ((p.left_delim | p.ambi_delim)
               | Error("Expected a delimiter"))
            + Group(ZeroOrMore(p.simple | p.auto_delim))
            + Suppress(Literal(r"\right"))
            - ((p.right_delim | p.ambi_delim)
               | Error("Expected a delimiter"))
        )

        p.math          <<= OneOrMore(p.token)

        p.math_string   <<= QuotedString('$', '\\', unquoteResults=False)

        p.non_math      <<= Regex(r"(?:(?:\\[$])|[^$])*").leaveWhitespace()

        p.main          <<= (
            p.non_math + ZeroOrMore(p.math_string + p.non_math) + StringEnd()
        )

        # Set actions
        for key, val in vars(p).items():
            if not key.startswith('_'):
                if hasattr(self, key):
                    val.setParseAction(getattr(self, key))

        self._expression = p.main
        self._math_expression = p.math

    def parse(self, s, fonts_object, fontsize, dpi):
        """
        Parse expression *s* using the given *fonts_object* for
        output, at the given *fontsize* and *dpi*.

        Returns the parse tree of `Node` instances.
        """
        self._state_stack = [
            self.State(fonts_object, 'default', 'rm', fontsize, dpi)]
        self._em_width_cache = {}
        try:
            result = self._expression.parseString(s)
        except ParseBaseException as err:
            raise ValueError("\n".join(["",
                                        err.line,
                                        " " * (err.column - 1) + "^",
                                        str(err)])) from err
        self._state_stack = None
        self._em_width_cache = {}
        self._expression.resetCache()
        return result[0]

    # The state of the parser is maintained in a stack.  Upon
    # entering and leaving a group { } or math/non-math, the stack
    # is pushed and popped accordingly.  The current state always
    # exists in the top element of the stack.
    class State:
        """
        Stores the state of the parser.

        States are pushed and popped from a stack as necessary, and
        the "current" state is always at the top of the stack.
        """
        def __init__(self, font_output, font, font_class, fontsize, dpi):
            self.font_output = font_output
            self._font = font
            self.font_class = font_class
            self.fontsize = fontsize
            self.dpi = dpi

        def copy(self):
            return Parser.State(
                self.font_output,
                self.font,
                self.font_class,
                self.fontsize,
                self.dpi)

        @property
        def font(self):
            return self._font

        @font.setter
        def font(self, name):
            if name in ('rm', 'it', 'bf'):
                self.font_class = name
            self._font = name

    def get_state(self):
        """Get the current `State` of the parser."""
        return self._state_stack[-1]

    def pop_state(self):
        """Pop a `State` off of the stack."""
        self._state_stack.pop()

    def push_state(self):
        """Push a new `State` onto the stack, copying the current state."""
        self._state_stack.append(self.get_state().copy())

    def main(self, s, loc, toks):
        return [Hlist(toks)]

    def math_string(self, s, loc, toks):
        return self._math_expression.parseString(toks[0][1:-1])

    def math(self, s, loc, toks):
        hlist = Hlist(toks)
        self.pop_state()
        return [hlist]

    def non_math(self, s, loc, toks):
        s = toks[0].replace(r'\$', '$')
        symbols = [Char(c, self.get_state(), math=False) for c in s]
        hlist = Hlist(symbols)
        # We're going into math now, so set font to 'it'
        self.push_state()
        self.get_state().font = mpl.rcParams['mathtext.default']
        return [hlist]

    def _make_space(self, percentage):
        # All spaces are relative to em width
        state = self.get_state()
        key = (state.font, state.fontsize, state.dpi)
        width = self._em_width_cache.get(key)
        if width is None:
            metrics = state.font_output.get_metrics(
                state.font, mpl.rcParams['mathtext.default'], 'm',
                state.fontsize, state.dpi)
            width = metrics.advance
            self._em_width_cache[key] = width
        return Kern(width * percentage)

    _space_widths = {
        r'\,':         0.16667,   # 3/18 em = 3 mu
        r'\thinspace': 0.16667,   # 3/18 em = 3 mu
        r'\/':         0.16667,   # 3/18 em = 3 mu
        r'\>':         0.22222,   # 4/18 em = 4 mu
        r'\:':         0.22222,   # 4/18 em = 4 mu
        r'\;':         0.27778,   # 5/18 em = 5 mu
        r'\ ':         0.33333,   # 6/18 em = 6 mu
        r'~':          0.33333,   # 6/18 em = 6 mu, nonbreakable
        r'\enspace':   0.5,       # 9/18 em = 9 mu
        r'\quad':      1,         # 1 em = 18 mu
        r'\qquad':     2,         # 2 em = 36 mu
        r'\!':         -0.16667,  # -3/18 em = -3 mu
    }

    def space(self, s, loc, toks):
        tok, = toks
        num = self._space_widths[tok]
        box = self._make_space(num)
        return [box]

    def customspace(self, s, loc, toks):
        return [self._make_space(float(toks[0]))]

    def symbol(self, s, loc, toks):
        c, = toks
        try:
            char = Char(c, self.get_state())
        except ValueError as err:
            raise ParseFatalException(s, loc,
                                      "Unknown symbol: %s" % c) from err

        if c in self._spaced_symbols:
            # iterate until we find previous character, needed for cases
            # such as ${ -2}$, $ -2$, or $   -2$.
            prev_char = next((c for c in s[:loc][::-1] if c != ' '), '')
            # Binary operators at start of string should not be spaced
            if (c in self._binary_operators and
                    (len(s[:loc].split()) == 0 or prev_char == '{' or
                     prev_char in self._left_delim)):
                return [char]
            else:
                return [Hlist([self._make_space(0.2),
                               char,
                               self._make_space(0.2)],
                              do_kern=True)]
        elif c in self._punctuation_symbols:

            # Do not space commas between brackets
            if c == ',':
                prev_char = next((c for c in s[:loc][::-1] if c != ' '), '')
                next_char = next((c for c in s[loc + 1:] if c != ' '), '')
                if prev_char == '{' and next_char == '}':
                    return [char]

            # Do not space dots as decimal separators
            if c == '.' and s[loc - 1].isdigit() and s[loc + 1].isdigit():
                return [char]
            else:
                return [Hlist([char, self._make_space(0.2)], do_kern=True)]
        return [char]

    accentprefixed = symbol

    def unknown_symbol(self, s, loc, toks):
        c, = toks
        raise ParseFatalException(s, loc, "Unknown symbol: %s" % c)

    _char_over_chars = {
        # The first 2 entries in the tuple are (font, char, sizescale) for
        # the two symbols under and over.  The third element is the space
        # (in multiples of underline height)
        r'AA': (('it', 'A', 1.0), (None, '\\circ', 0.5), 0.0),
    }

    def c_over_c(self, s, loc, toks):
        sym, = toks
        state = self.get_state()
        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)

        under_desc, over_desc, space = \
            self._char_over_chars.get(sym, (None, None, 0.0))
        if under_desc is None:
            raise ParseFatalException("Error parsing symbol")

        over_state = state.copy()
        if over_desc[0] is not None:
            over_state.font = over_desc[0]
        over_state.fontsize *= over_desc[2]
        over = Accent(over_desc[1], over_state)

        under_state = state.copy()
        if under_desc[0] is not None:
            under_state.font = under_desc[0]
        under_state.fontsize *= under_desc[2]
        under = Char(under_desc[1], under_state)

        width = max(over.width, under.width)

        over_centered = HCentered([over])
        over_centered.hpack(width, 'exactly')

        under_centered = HCentered([under])
        under_centered.hpack(width, 'exactly')

        return Vlist([
                over_centered,
                Vbox(0., thickness * space),
                under_centered
                ])

    _accent_map = {
        r'hat':            r'\circumflexaccent',
        r'breve':          r'\combiningbreve',
        r'bar':            r'\combiningoverline',
        r'grave':          r'\combininggraveaccent',
        r'acute':          r'\combiningacuteaccent',
        r'tilde':          r'\combiningtilde',
        r'dot':            r'\combiningdotabove',
        r'ddot':           r'\combiningdiaeresis',
        r'dddot':          r'\combiningthreedotsabove',
        r'ddddot':         r'\combiningfourdotsabove',
        r'vec':            r'\combiningrightarrowabove',
        r'"':              r'\combiningdiaeresis',
        r"`":              r'\combininggraveaccent',
        r"'":              r'\combiningacuteaccent',
        r'~':              r'\combiningtilde',
        r'.':              r'\combiningdotabove',
        r'^':              r'\circumflexaccent',
        r'overrightarrow': r'\rightarrow',
        r'overleftarrow':  r'\leftarrow',
        r'mathring':       r'\circ',
    }

    _wide_accents = set(r"widehat widetilde widebar".split())

    # make a lambda and call it to get the namespace right
    _accentprefixed = (lambda am: [
        p for p in tex2uni
        if any(p.startswith(a) and a != p for a in am)
    ])(set(_accent_map))

    def accent(self, s, loc, toks):
        state = self.get_state()
        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)
        (accent, sym), = toks
        if accent in self._wide_accents:
            accent_box = AutoWidthChar(
                '\\' + accent, sym.width, state, char_class=Accent)
        else:
            accent_box = Accent(self._accent_map[accent], state)
        if accent == 'mathring':
            accent_box.shrink()
            accent_box.shrink()
        centered = HCentered([Hbox(sym.width / 4.0), accent_box])
        centered.hpack(sym.width, 'exactly')
        return Vlist([
                centered,
                Vbox(0., thickness * 2.0),
                Hlist([sym])
                ])

    def function(self, s, loc, toks):
        hlist = self.operatorname(s, loc, toks)
        hlist.function_name, = toks
        return hlist

    def operatorname(self, s, loc, toks):
        self.push_state()
        state = self.get_state()
        state.font = 'rm'
        hlist_list = []
        # Change the font of Chars, but leave Kerns alone
        for c in toks[0]:
            if isinstance(c, Char):
                c.font = 'rm'
                c._update_metrics()
                hlist_list.append(c)
            elif isinstance(c, str):
                hlist_list.append(Char(c, state))
            else:
                hlist_list.append(c)
        next_char_loc = loc + len(toks[0]) + 1
        if isinstance(toks[0], ParseResults):
            next_char_loc += len('operatorname{}')
        next_char = next((c for c in s[next_char_loc:] if c != ' '), '')
        delimiters = self._left_delim | self._ambi_delim | self._right_delim
        delimiters |= {'^', '_'}
        if (next_char not in delimiters and
                toks[0] not in self._overunder_functions):
            # Add thin space except when followed by parenthesis, bracket, etc.
            hlist_list += [self._make_space(self._space_widths[r'\,'])]
        self.pop_state()
        return Hlist(hlist_list)

    def start_group(self, s, loc, toks):
        self.push_state()
        # Deal with LaTeX-style font tokens
        if len(toks):
            self.get_state().font = toks[0][4:]
        return []

    def group(self, s, loc, toks):
        grp = Hlist(toks[0])
        return [grp]
    required_group = simple_group = group

    def end_group(self, s, loc, toks):
        self.pop_state()
        return []

    def font(self, s, loc, toks):
        name, = toks
        self.get_state().font = name
        return []

    def is_overunder(self, nucleus):
        if isinstance(nucleus, Char):
            return nucleus.c in self._overunder_symbols
        elif isinstance(nucleus, Hlist) and hasattr(nucleus, 'function_name'):
            return nucleus.function_name in self._overunder_functions
        return False

    def is_dropsub(self, nucleus):
        if isinstance(nucleus, Char):
            return nucleus.c in self._dropsub_symbols
        return False

    def is_slanted(self, nucleus):
        if isinstance(nucleus, Char):
            return nucleus.is_slanted()
        return False

    def is_between_brackets(self, s, loc):
        return False

    def subsuper(self, s, loc, toks):
        assert len(toks) == 1

        nucleus = None
        sub = None
        super = None

        # Pick all of the apostrophes out, including first apostrophes that
        # have been parsed as characters
        napostrophes = 0
        new_toks = []
        for tok in toks[0]:
            if isinstance(tok, str) and tok not in ('^', '_'):
                napostrophes += len(tok)
            elif isinstance(tok, Char) and tok.c == "'":
                napostrophes += 1
            else:
                new_toks.append(tok)
        toks = new_toks

        if len(toks) == 0:
            assert napostrophes
            nucleus = Hbox(0.0)
        elif len(toks) == 1:
            if not napostrophes:
                return toks[0]  # .asList()
            else:
                nucleus = toks[0]
        elif len(toks) in (2, 3):
            # single subscript or superscript
            nucleus = toks[0] if len(toks) == 3 else Hbox(0.0)
            op, next = toks[-2:]
            if op == '_':
                sub = next
            else:
                super = next
        elif len(toks) in (4, 5):
            # subscript and superscript
            nucleus = toks[0] if len(toks) == 5 else Hbox(0.0)
            op1, next1, op2, next2 = toks[-4:]
            if op1 == op2:
                if op1 == '_':
                    raise ParseFatalException("Double subscript")
                else:
                    raise ParseFatalException("Double superscript")
            if op1 == '_':
                sub = next1
                super = next2
            else:
                super = next1
                sub = next2
        else:
            raise ParseFatalException(
                "Subscript/superscript sequence is too long. "
                "Use braces { } to remove ambiguity.")

        state = self.get_state()
        rule_thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)
        xHeight = state.font_output.get_xheight(
            state.font, state.fontsize, state.dpi)

        if napostrophes:
            if super is None:
                super = Hlist([])
            for i in range(napostrophes):
                super.children.extend(self.symbol(s, loc, ['\\prime']))
            # kern() and hpack() needed to get the metrics right after
            # extending
            super.kern()
            super.hpack()

        # Handle over/under symbols, such as sum or prod
        if self.is_overunder(nucleus):
            vlist = []
            shift = 0.
            width = nucleus.width
            if super is not None:
                super.shrink()
                width = max(width, super.width)
            if sub is not None:
                sub.shrink()
                width = max(width, sub.width)

            vgap = rule_thickness * 3.0
            if super is not None:
                hlist = HCentered([super])
                hlist.hpack(width, 'exactly')
                vlist.extend([hlist, Vbox(0, vgap)])
            hlist = HCentered([nucleus])
            hlist.hpack(width, 'exactly')
            vlist.append(hlist)
            if sub is not None:
                hlist = HCentered([sub])
                hlist.hpack(width, 'exactly')
                vlist.extend([Vbox(0, vgap), hlist])
                shift = hlist.height + vgap
            vlist = Vlist(vlist)
            vlist.shift_amount = shift + nucleus.depth
            result = Hlist([vlist])
            return [result]

        # We remove kerning on the last character for consistency (otherwise
        # it will compute kerning based on non-shrunk characters and may put
        # them too close together when superscripted)
        # We change the width of the last character to match the advance to
        # consider some fonts with weird metrics: e.g. stix's f has a width of
        # 7.75 and a kerning of -4.0 for an advance of 3.72, and we want to put
        # the superscript at the advance
        last_char = nucleus
        if isinstance(nucleus, Hlist):
            new_children = nucleus.children
            if len(new_children):
                # remove last kern
                if (isinstance(new_children[-1], Kern) and
                        hasattr(new_children[-2], '_metrics')):
                    new_children = new_children[:-1]
                last_char = new_children[-1]
                if hasattr(last_char, '_metrics'):
                    last_char.width = last_char._metrics.advance
            # create new Hlist without kerning
            nucleus = Hlist(new_children, do_kern=False)
        else:
            if isinstance(nucleus, Char):
                last_char.width = last_char._metrics.advance
            nucleus = Hlist([nucleus])

        # Handle regular sub/superscripts
        constants = _get_font_constant_set(state)
        lc_height   = last_char.height
        lc_baseline = 0
        if self.is_dropsub(last_char):
            lc_baseline = last_char.depth

        # Compute kerning for sub and super
        superkern = constants.delta * xHeight
        subkern = constants.delta * xHeight
        if self.is_slanted(last_char):
            superkern += constants.delta * xHeight
            superkern += (constants.delta_slanted *
                          (lc_height - xHeight * 2. / 3.))
            if self.is_dropsub(last_char):
                subkern = (3 * constants.delta -
                           constants.delta_integral) * lc_height
                superkern = (3 * constants.delta +
                             constants.delta_integral) * lc_height
            else:
                subkern = 0

        if super is None:
            # node757
            x = Hlist([Kern(subkern), sub])
            x.shrink()
            if self.is_dropsub(last_char):
                shift_down = lc_baseline + constants.subdrop * xHeight
            else:
                shift_down = constants.sub1 * xHeight
            x.shift_amount = shift_down
        else:
            x = Hlist([Kern(superkern), super])
            x.shrink()
            if self.is_dropsub(last_char):
                shift_up = lc_height - constants.subdrop * xHeight
            else:
                shift_up = constants.sup1 * xHeight
            if sub is None:
                x.shift_amount = -shift_up
            else:  # Both sub and superscript
                y = Hlist([Kern(subkern), sub])
                y.shrink()
                if self.is_dropsub(last_char):
                    shift_down = lc_baseline + constants.subdrop * xHeight
                else:
                    shift_down = constants.sub2 * xHeight
                # If sub and superscript collide, move super up
                clr = (2.0 * rule_thickness -
                       ((shift_up - x.depth) - (y.height - shift_down)))
                if clr > 0.:
                    shift_up += clr
                x = Vlist([
                    x,
                    Kern((shift_up - x.depth) - (y.height - shift_down)),
                    y])
                x.shift_amount = shift_down

        if not self.is_dropsub(last_char):
            x.width += constants.script_space * xHeight
        result = Hlist([nucleus, x])

        return [result]

    def _genfrac(self, ldelim, rdelim, rule, style, num, den):
        state = self.get_state()
        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)

        rule = float(rule)

        if style is not self._MathStyle.DISPLAYSTYLE:
            num.shrink()
            den.shrink()
        cnum = HCentered([num])
        cden = HCentered([den])
        width = max(num.width, den.width)
        cnum.hpack(width, 'exactly')
        cden.hpack(width, 'exactly')
        vlist = Vlist([cnum,                      # numerator
                       Vbox(0, thickness * 2.0),  # space
                       Hrule(state, rule),        # rule
                       Vbox(0, thickness * 2.0),  # space
                       cden                       # denominator
                       ])

        # Shift so the fraction line sits in the middle of the
        # equals sign
        metrics = state.font_output.get_metrics(
            state.font, mpl.rcParams['mathtext.default'],
            '=', state.fontsize, state.dpi)
        shift = (cden.height -
                 ((metrics.ymax + metrics.ymin) / 2 -
                  thickness * 3.0))
        vlist.shift_amount = shift

        result = [Hlist([vlist, Hbox(thickness * 2.)])]
        if ldelim or rdelim:
            if ldelim == '':
                ldelim = '.'
            if rdelim == '':
                rdelim = '.'
            return self._auto_sized_delimiter(ldelim, result, rdelim)
        return result

    def genfrac(self, s, loc, toks):
        args, = toks
        return self._genfrac(*args)

    def frac(self, s, loc, toks):
        state = self.get_state()
        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)
        (num, den), = toks
        return self._genfrac('', '', thickness, self._MathStyle.TEXTSTYLE,
                             num, den)

    def dfrac(self, s, loc, toks):
        state = self.get_state()
        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)
        (num, den), = toks
        return self._genfrac('', '', thickness, self._MathStyle.DISPLAYSTYLE,
                             num, den)

    def binom(self, s, loc, toks):
        (num, den), = toks
        return self._genfrac('(', ')', 0.0, self._MathStyle.TEXTSTYLE,
                             num, den)

    def _genset(self, s, loc, toks):
        (annotation, body), = toks
        state = self.get_state()
        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)

        annotation.shrink()
        cannotation = HCentered([annotation])
        cbody = HCentered([body])
        width = max(cannotation.width, cbody.width)
        cannotation.hpack(width, 'exactly')
        cbody.hpack(width, 'exactly')

        vgap = thickness * 3
        if s[loc + 1] == "u":  # \underset
            vlist = Vlist([cbody,                       # body
                           Vbox(0, vgap),               # space
                           cannotation                  # annotation
                           ])
            # Shift so the body sits in the same vertical position
            vlist.shift_amount = cbody.depth + cannotation.height + vgap
        else:  # \overset
            vlist = Vlist([cannotation,                 # annotation
                           Vbox(0, vgap),               # space
                           cbody                        # body
                           ])

        # To add horizontal gap between symbols: wrap the Vlist into
        # an Hlist and extend it with an Hbox(0, horizontal_gap)
        return vlist

    overset = underset = _genset

    def sqrt(self, s, loc, toks):
        (root, body), = toks
        state = self.get_state()
        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)

        # Determine the height of the body, and add a little extra to
        # the height so it doesn't seem cramped
        height = body.height - body.shift_amount + thickness * 5.0
        depth = body.depth + body.shift_amount
        check = AutoHeightChar(r'\__sqrt__', height, depth, state, always=True)
        height = check.height - check.shift_amount
        depth = check.depth + check.shift_amount

        # Put a little extra space to the left and right of the body
        padded_body = Hlist([Hbox(2 * thickness), body, Hbox(2 * thickness)])
        rightside = Vlist([Hrule(state), Glue('fill'), padded_body])
        # Stretch the glue between the hrule and the body
        rightside.vpack(height + (state.fontsize * state.dpi) / (100.0 * 12.0),
                        'exactly', depth)

        # Add the root and shift it upward so it is above the tick.
        # The value of 0.6 is a hard-coded hack ;)
        if not root:
            root = Box(check.width * 0.5, 0., 0.)
        else:
            root = Hlist(root)
            root.shrink()
            root.shrink()

        root_vlist = Vlist([Hlist([root])])
        root_vlist.shift_amount = -height * 0.6

        hlist = Hlist([root_vlist,               # Root
                       # Negative kerning to put root over tick
                       Kern(-check.width * 0.5),
                       check,                    # Check
                       rightside])               # Body
        return [hlist]

    def overline(self, s, loc, toks):
        (body,), = toks

        state = self.get_state()
        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)

        height = body.height - body.shift_amount + thickness * 3.0
        depth = body.depth + body.shift_amount

        # Place overline above body
        rightside = Vlist([Hrule(state), Glue('fill'), Hlist([body])])

        # Stretch the glue between the hrule and the body
        rightside.vpack(height + (state.fontsize * state.dpi) / (100.0 * 12.0),
                        'exactly', depth)

        hlist = Hlist([rightside])
        return [hlist]

    def _auto_sized_delimiter(self, front, middle, back):
        state = self.get_state()
        if len(middle):
            height = max(x.height for x in middle)
            depth = max(x.depth for x in middle)
            factor = None
        else:
            height = 0
            depth = 0
            factor = 1.0
        parts = []
        # \left. and \right. aren't supposed to produce any symbols
        if front != '.':
            parts.append(
                AutoHeightChar(front, height, depth, state, factor=factor))
        parts.extend(middle)
        if back != '.':
            parts.append(
                AutoHeightChar(back, height, depth, state, factor=factor))
        hlist = Hlist(parts)
        return hlist

    def auto_delim(self, s, loc, toks):
        front, middle, back = toks

        return self._auto_sized_delimiter(front, middle.asList(), back)
