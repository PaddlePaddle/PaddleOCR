r"""
Support for embedded TeX expressions in Matplotlib.

Requirements:

* LaTeX.
* \*Agg backends: dvipng>=1.6.
* PS backend: PSfrag, dvips, and Ghostscript>=9.0.
* PDF and SVG backends: if LuaTeX is present, it will be used to speed up some
  post-processing steps, but note that it is not used to parse the TeX string
  itself (only LaTeX is supported).

To enable TeX rendering of all text in your Matplotlib figure, set
:rc:`text.usetex` to True.

TeX and dvipng/dvips processing results are cached
in ~/.matplotlib/tex.cache for reuse between sessions.

`TexManager.get_rgba` can also be used to directly obtain raster output as RGBA
NumPy arrays.
"""

import functools
import hashlib
import logging
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from packaging.version import parse as parse_version

import matplotlib as mpl
from matplotlib import _api, cbook, dviread, rcParams

_log = logging.getLogger(__name__)


def _usepackage_if_not_loaded(package, *, option=None):
    """
    Output LaTeX code that loads a package (possibly with an option) if it
    hasn't been loaded yet.

    LaTeX cannot load twice a package with different options, so this helper
    can be used to protect against users loading arbitrary packages/options in
    their custom preamble.
    """
    option = f"[{option}]" if option is not None else ""
    return (
        r"\makeatletter"
        r"\@ifpackageloaded{%(package)s}{}{\usepackage%(option)s{%(package)s}}"
        r"\makeatother"
    ) % {"package": package, "option": option}


class TexManager:
    """
    Convert strings to dvi files using TeX, caching the results to a directory.

    Repeated calls to this constructor always return the same instance.
    """

    texcache = os.path.join(mpl.get_cachedir(), 'tex.cache')

    _grey_arrayd = {}
    _font_family = 'serif'
    _font_families = ('serif', 'sans-serif', 'cursive', 'monospace')
    _font_info = {
        'new century schoolbook': ('pnc', r'\renewcommand{\rmdefault}{pnc}'),
        'bookman': ('pbk', r'\renewcommand{\rmdefault}{pbk}'),
        'times': ('ptm', r'\usepackage{mathptmx}'),
        'palatino': ('ppl', r'\usepackage{mathpazo}'),
        'zapf chancery': ('pzc', r'\usepackage{chancery}'),
        'cursive': ('pzc', r'\usepackage{chancery}'),
        'charter': ('pch', r'\usepackage{charter}'),
        'serif': ('cmr', ''),
        'sans-serif': ('cmss', ''),
        'helvetica': ('phv', r'\usepackage{helvet}'),
        'avant garde': ('pag', r'\usepackage{avant}'),
        'courier': ('pcr', r'\usepackage{courier}'),
        # Loading the type1ec package ensures that cm-super is installed, which
        # is necessary for unicode computer modern.  (It also allows the use of
        # computer modern at arbitrary sizes, but that's just a side effect.)
        'monospace': ('cmtt', r'\usepackage{type1ec}'),
        'computer modern roman': ('cmr', r'\usepackage{type1ec}'),
        'computer modern sans serif': ('cmss', r'\usepackage{type1ec}'),
        'computer modern typewriter': ('cmtt', r'\usepackage{type1ec}')}
    _font_types = {
        'new century schoolbook': 'serif', 'bookman': 'serif',
        'times': 'serif', 'palatino': 'serif', 'charter': 'serif',
        'computer modern roman': 'serif', 'zapf chancery': 'cursive',
        'helvetica': 'sans-serif', 'avant garde': 'sans-serif',
        'computer modern sans serif': 'sans-serif',
        'courier': 'monospace', 'computer modern typewriter': 'monospace'}

    grey_arrayd = _api.deprecate_privatize_attribute("3.5")
    font_family = _api.deprecate_privatize_attribute("3.5")
    font_families = _api.deprecate_privatize_attribute("3.5")
    font_info = _api.deprecate_privatize_attribute("3.5")

    @functools.lru_cache()  # Always return the same instance.
    def __new__(cls):
        Path(cls.texcache).mkdir(parents=True, exist_ok=True)
        return object.__new__(cls)

    def get_font_config(self):
        ff = rcParams['font.family']
        ff_val = ff[0].lower() if len(ff) == 1 else None
        reduced_notation = False
        if len(ff) == 1 and ff_val in self._font_families:
            self._font_family = ff_val
        elif len(ff) == 1 and ff_val in self._font_info:
            reduced_notation = True
            self._font_family = self._font_types[ff_val]
        else:
            _log.info('font.family must be one of (%s) when text.usetex is '
                      'True. serif will be used by default.',
                      ', '.join(self._font_families))
            self._font_family = 'serif'

        fontconfig = [self._font_family]
        fonts = {}
        for font_family in self._font_families:
            if reduced_notation and self._font_family == font_family:
                fonts[font_family] = self._font_info[ff_val]
            else:
                for font in rcParams['font.' + font_family]:
                    if font.lower() in self._font_info:
                        fonts[font_family] = self._font_info[font.lower()]
                        _log.debug(
                            'family: %s, font: %s, info: %s',
                            font_family, font, self._font_info[font.lower()])
                        break
                    else:
                        _log.debug('%s font is not compatible with usetex.',
                                   font)
                else:
                    _log.info('No LaTeX-compatible font found for the %s font'
                              'family in rcParams. Using default.',
                              font_family)
                    fonts[font_family] = self._font_info[font_family]
            fontconfig.append(fonts[font_family][0])
        # Add a hash of the latex preamble to fontconfig so that the
        # correct png is selected for strings rendered with same font and dpi
        # even if the latex preamble changes within the session
        preamble_bytes = self.get_custom_preamble().encode('utf-8')
        fontconfig.append(hashlib.md5(preamble_bytes).hexdigest())

        # The following packages and commands need to be included in the latex
        # file's preamble:
        cmd = {fonts[family][1]
               for family in ['serif', 'sans-serif', 'monospace']}
        if self._font_family == 'cursive':
            cmd.add(fonts['cursive'][1])
        cmd.add(r'\usepackage{type1cm}')
        self._font_preamble = '\n'.join(sorted(cmd))

        return ''.join(fontconfig)

    def get_basefile(self, tex, fontsize, dpi=None):
        """
        Return a filename based on a hash of the string, fontsize, and dpi.
        """
        s = ''.join([tex, self.get_font_config(), '%f' % fontsize,
                     self.get_custom_preamble(), str(dpi or '')])
        return os.path.join(
            self.texcache, hashlib.md5(s.encode('utf-8')).hexdigest())

    def get_font_preamble(self):
        """
        Return a string containing font configuration for the tex preamble.
        """
        return self._font_preamble

    def get_custom_preamble(self):
        """Return a string containing user additions to the tex preamble."""
        return rcParams['text.latex.preamble']

    def _get_preamble(self):
        return "\n".join([
            r"\documentclass{article}",
            # Pass-through \mathdefault, which is used in non-usetex mode to
            # use the default text font but was historically suppressed in
            # usetex mode.
            r"\newcommand{\mathdefault}[1]{#1}",
            self._font_preamble,
            r"\usepackage[utf8]{inputenc}",
            r"\DeclareUnicodeCharacter{2212}{\ensuremath{-}}",
            # geometry is loaded before the custom preamble as convert_psfrags
            # relies on a custom preamble to change the geometry.
            r"\usepackage[papersize=72in, margin=1in]{geometry}",
            self.get_custom_preamble(),
            # Use `underscore` package to take care of underscores in text
            # The [strings] option allows to use underscores in file names
            _usepackage_if_not_loaded("underscore", option="strings"),
            # Custom packages (e.g. newtxtext) may already have loaded textcomp
            # with different options.
            _usepackage_if_not_loaded("textcomp"),
        ])

    def make_tex(self, tex, fontsize):
        """
        Generate a tex file to render the tex string at a specific font size.

        Return the file name.
        """
        basefile = self.get_basefile(tex, fontsize)
        texfile = '%s.tex' % basefile
        fontcmd = {'sans-serif': r'{\sffamily %s}',
                   'monospace': r'{\ttfamily %s}'}.get(self._font_family,
                                                       r'{\rmfamily %s}')

        Path(texfile).write_text(
            r"""
%s
\pagestyle{empty}
\begin{document}
%% The empty hbox ensures that a page is printed even for empty inputs, except
%% when using psfrag which gets confused by it.
\fontsize{%f}{%f}%%
\ifdefined\psfrag\else\hbox{}\fi%%
%s
\end{document}
""" % (self._get_preamble(), fontsize, fontsize * 1.25, fontcmd % tex),
            encoding='utf-8')

        return texfile

    def _run_checked_subprocess(self, command, tex, *, cwd=None):
        _log.debug(cbook._pformat_subprocess(command))
        try:
            report = subprocess.check_output(
                command, cwd=cwd if cwd is not None else self.texcache,
                stderr=subprocess.STDOUT)
        except FileNotFoundError as exc:
            raise RuntimeError(
                'Failed to process string with tex because {} could not be '
                'found'.format(command[0])) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                '{prog} was not able to process the following string:\n'
                '{tex!r}\n\n'
                'Here is the full report generated by {prog}:\n'
                '{exc}\n\n'.format(
                    prog=command[0],
                    tex=tex.encode('unicode_escape'),
                    exc=exc.output.decode('utf-8'))) from exc
        _log.debug(report)
        return report

    def make_dvi(self, tex, fontsize):
        """
        Generate a dvi file containing latex's layout of tex string.

        Return the file name.
        """
        basefile = self.get_basefile(tex, fontsize)
        dvifile = '%s.dvi' % basefile
        if not os.path.exists(dvifile):
            texfile = Path(self.make_tex(tex, fontsize))
            # Generate the dvi in a temporary directory to avoid race
            # conditions e.g. if multiple processes try to process the same tex
            # string at the same time.  Having tmpdir be a subdirectory of the
            # final output dir ensures that they are on the same filesystem,
            # and thus replace() works atomically.  It also allows referring to
            # the texfile with a relative path (for pathological MPLCONFIGDIRs,
            # the absolute path may contain characters (e.g. ~) that TeX does
            # not support.)
            with TemporaryDirectory(dir=Path(dvifile).parent) as tmpdir:
                self._run_checked_subprocess(
                    ["latex", "-interaction=nonstopmode", "--halt-on-error",
                     f"../{texfile.name}"], tex, cwd=tmpdir)
                (Path(tmpdir) / Path(dvifile).name).replace(dvifile)
        return dvifile

    def make_png(self, tex, fontsize, dpi):
        """
        Generate a png file containing latex's rendering of tex string.

        Return the file name.
        """
        basefile = self.get_basefile(tex, fontsize, dpi)
        pngfile = '%s.png' % basefile
        # see get_rgba for a discussion of the background
        if not os.path.exists(pngfile):
            dvifile = self.make_dvi(tex, fontsize)
            cmd = ["dvipng", "-bg", "Transparent", "-D", str(dpi),
                   "-T", "tight", "-o", pngfile, dvifile]
            # When testing, disable FreeType rendering for reproducibility; but
            # dvipng 1.16 has a bug (fixed in f3ff241) that breaks --freetype0
            # mode, so for it we keep FreeType enabled; the image will be
            # slightly off.
            bad_ver = parse_version("1.16")
            if (getattr(mpl, "_called_from_pytest", False)
                    and mpl._get_executable_info("dvipng").version != bad_ver):
                cmd.insert(1, "--freetype0")
            self._run_checked_subprocess(cmd, tex)
        return pngfile

    def get_grey(self, tex, fontsize=None, dpi=None):
        """Return the alpha channel."""
        if not fontsize:
            fontsize = rcParams['font.size']
        if not dpi:
            dpi = rcParams['savefig.dpi']
        key = tex, self.get_font_config(), fontsize, dpi
        alpha = self._grey_arrayd.get(key)
        if alpha is None:
            pngfile = self.make_png(tex, fontsize, dpi)
            rgba = mpl.image.imread(os.path.join(self.texcache, pngfile))
            self._grey_arrayd[key] = alpha = rgba[:, :, -1]
        return alpha

    def get_rgba(self, tex, fontsize=None, dpi=None, rgb=(0, 0, 0)):
        r"""
        Return latex's rendering of the tex string as an rgba array.

        Examples
        --------
        >>> texmanager = TexManager()
        >>> s = r"\TeX\ is $\displaystyle\sum_n\frac{-e^{i\pi}}{2^n}$!"
        >>> Z = texmanager.get_rgba(s, fontsize=12, dpi=80, rgb=(1, 0, 0))
        """
        alpha = self.get_grey(tex, fontsize, dpi)
        rgba = np.empty((*alpha.shape, 4))
        rgba[..., :3] = mpl.colors.to_rgb(rgb)
        rgba[..., -1] = alpha
        return rgba

    def get_text_width_height_descent(self, tex, fontsize, renderer=None):
        """Return width, height and descent of the text."""
        if tex.strip() == '':
            return 0, 0, 0
        dvifile = self.make_dvi(tex, fontsize)
        dpi_fraction = renderer.points_to_pixels(1.) if renderer else 1
        with dviread.Dvi(dvifile, 72 * dpi_fraction) as dvi:
            page, = dvi
        # A total height (including the descent) needs to be returned.
        return page.width, page.height + page.descent, page.descent
