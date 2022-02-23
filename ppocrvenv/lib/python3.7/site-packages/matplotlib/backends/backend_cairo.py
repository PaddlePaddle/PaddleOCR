"""
A Cairo backend for Matplotlib
==============================
:Author: Steve Chaplin and others

This backend depends on cairocffi or pycairo.
"""

import functools
import gzip
import math

import numpy as np

try:
    import cairo
    if cairo.version_info < (1, 11, 0):
        # Introduced create_for_data for Py3.
        raise ImportError
except ImportError:
    try:
        import cairocffi as cairo
    except ImportError as err:
        raise ImportError(
            "cairo backend requires that pycairo>=1.11.0 or cairocffi "
            "is installed") from err

import matplotlib as mpl
from .. import _api, cbook, font_manager
from matplotlib.backend_bases import (
    _Backend, _check_savefig_extra_args, FigureCanvasBase, FigureManagerBase,
    GraphicsContextBase, RendererBase)
from matplotlib.font_manager import ttfFontProperty
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.transforms import Affine2D


backend_version = cairo.version


if cairo.__name__ == "cairocffi":
    # Convert a pycairo context to a cairocffi one.
    def _to_context(ctx):
        if not isinstance(ctx, cairo.Context):
            ctx = cairo.Context._from_pointer(
                cairo.ffi.cast(
                    'cairo_t **',
                    id(ctx) + object.__basicsize__)[0],
                incref=True)
        return ctx
else:
    # Pass-through a pycairo context.
    def _to_context(ctx):
        return ctx


def _append_path(ctx, path, transform, clip=None):
    for points, code in path.iter_segments(
            transform, remove_nans=True, clip=clip):
        if code == Path.MOVETO:
            ctx.move_to(*points)
        elif code == Path.CLOSEPOLY:
            ctx.close_path()
        elif code == Path.LINETO:
            ctx.line_to(*points)
        elif code == Path.CURVE3:
            cur = np.asarray(ctx.get_current_point())
            a = points[:2]
            b = points[-2:]
            ctx.curve_to(*(cur / 3 + a * 2 / 3), *(a * 2 / 3 + b / 3), *b)
        elif code == Path.CURVE4:
            ctx.curve_to(*points)


def _cairo_font_args_from_font_prop(prop):
    """
    Convert a `.FontProperties` or a `.FontEntry` to arguments that can be
    passed to `.Context.select_font_face`.
    """
    def attr(field):
        try:
            return getattr(prop, f"get_{field}")()
        except AttributeError:
            return getattr(prop, field)

    name = attr("name")
    slant = getattr(cairo, f"FONT_SLANT_{attr('style').upper()}")
    weight = attr("weight")
    weight = (cairo.FONT_WEIGHT_NORMAL
              if font_manager.weight_dict.get(weight, weight) < 550
              else cairo.FONT_WEIGHT_BOLD)
    return name, slant, weight


# Mappings used for deprecated properties in RendererCairo, see below.
_f_weights = {
    100:          cairo.FONT_WEIGHT_NORMAL,
    200:          cairo.FONT_WEIGHT_NORMAL,
    300:          cairo.FONT_WEIGHT_NORMAL,
    400:          cairo.FONT_WEIGHT_NORMAL,
    500:          cairo.FONT_WEIGHT_NORMAL,
    600:          cairo.FONT_WEIGHT_BOLD,
    700:          cairo.FONT_WEIGHT_BOLD,
    800:          cairo.FONT_WEIGHT_BOLD,
    900:          cairo.FONT_WEIGHT_BOLD,
    'ultralight': cairo.FONT_WEIGHT_NORMAL,
    'light':      cairo.FONT_WEIGHT_NORMAL,
    'normal':     cairo.FONT_WEIGHT_NORMAL,
    'medium':     cairo.FONT_WEIGHT_NORMAL,
    'regular':    cairo.FONT_WEIGHT_NORMAL,
    'semibold':   cairo.FONT_WEIGHT_BOLD,
    'bold':       cairo.FONT_WEIGHT_BOLD,
    'heavy':      cairo.FONT_WEIGHT_BOLD,
    'ultrabold':  cairo.FONT_WEIGHT_BOLD,
    'black':      cairo.FONT_WEIGHT_BOLD,
}
_f_angles = {
    'italic':  cairo.FONT_SLANT_ITALIC,
    'normal':  cairo.FONT_SLANT_NORMAL,
    'oblique': cairo.FONT_SLANT_OBLIQUE,
}


class RendererCairo(RendererBase):
    mathtext_parser = _api.deprecated("3.4")(
        property(lambda self: MathTextParser('Cairo')))

    def __init__(self, dpi):
        self.dpi = dpi
        self.gc = GraphicsContextCairo(renderer=self)
        self.text_ctx = cairo.Context(
           cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1))
        super().__init__()

    def set_ctx_from_surface(self, surface):
        self.gc.ctx = cairo.Context(surface)
        # Although it may appear natural to automatically call
        # `self.set_width_height(surface.get_width(), surface.get_height())`
        # here (instead of having the caller do so separately), this would fail
        # for PDF/PS/SVG surfaces, which have no way to report their extents.

    def set_width_height(self, width, height):
        self.width = width
        self.height = height

    def _fill_and_stroke(self, ctx, fill_c, alpha, alpha_overrides):
        if fill_c is not None:
            ctx.save()
            if len(fill_c) == 3 or alpha_overrides:
                ctx.set_source_rgba(fill_c[0], fill_c[1], fill_c[2], alpha)
            else:
                ctx.set_source_rgba(fill_c[0], fill_c[1], fill_c[2], fill_c[3])
            ctx.fill_preserve()
            ctx.restore()
        ctx.stroke()

    def draw_path(self, gc, path, transform, rgbFace=None):
        # docstring inherited
        ctx = gc.ctx
        # Clip the path to the actual rendering extents if it isn't filled.
        clip = (ctx.clip_extents()
                if rgbFace is None and gc.get_hatch() is None
                else None)
        transform = (transform
                     + Affine2D().scale(1, -1).translate(0, self.height))
        ctx.new_path()
        _append_path(ctx, path, transform, clip)
        self._fill_and_stroke(
            ctx, rgbFace, gc.get_alpha(), gc.get_forced_alpha())

    def draw_markers(self, gc, marker_path, marker_trans, path, transform,
                     rgbFace=None):
        # docstring inherited

        ctx = gc.ctx
        ctx.new_path()
        # Create the path for the marker; it needs to be flipped here already!
        _append_path(ctx, marker_path, marker_trans + Affine2D().scale(1, -1))
        marker_path = ctx.copy_path_flat()

        # Figure out whether the path has a fill
        x1, y1, x2, y2 = ctx.fill_extents()
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            filled = False
            # No fill, just unset this (so we don't try to fill it later on)
            rgbFace = None
        else:
            filled = True

        transform = (transform
                     + Affine2D().scale(1, -1).translate(0, self.height))

        ctx.new_path()
        for i, (vertices, codes) in enumerate(
                path.iter_segments(transform, simplify=False)):
            if len(vertices):
                x, y = vertices[-2:]
                ctx.save()

                # Translate and apply path
                ctx.translate(x, y)
                ctx.append_path(marker_path)

                ctx.restore()

                # Slower code path if there is a fill; we need to draw
                # the fill and stroke for each marker at the same time.
                # Also flush out the drawing every once in a while to
                # prevent the paths from getting way too long.
                if filled or i % 1000 == 0:
                    self._fill_and_stroke(
                        ctx, rgbFace, gc.get_alpha(), gc.get_forced_alpha())

        # Fast path, if there is no fill, draw everything in one step
        if not filled:
            self._fill_and_stroke(
                ctx, rgbFace, gc.get_alpha(), gc.get_forced_alpha())

    def draw_image(self, gc, x, y, im):
        im = cbook._unmultiplied_rgba8888_to_premultiplied_argb32(im[::-1])
        surface = cairo.ImageSurface.create_for_data(
            im.ravel().data, cairo.FORMAT_ARGB32,
            im.shape[1], im.shape[0], im.shape[1] * 4)
        ctx = gc.ctx
        y = self.height - y - im.shape[0]

        ctx.save()
        ctx.set_source_surface(surface, float(x), float(y))
        ctx.paint()
        ctx.restore()

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # docstring inherited

        # Note: (x, y) are device/display coords, not user-coords, unlike other
        # draw_* methods
        if ismath:
            self._draw_mathtext(gc, x, y, s, prop, angle)

        else:
            ctx = gc.ctx
            ctx.new_path()
            ctx.move_to(x, y)

            ctx.save()
            ctx.select_font_face(*_cairo_font_args_from_font_prop(prop))
            ctx.set_font_size(self.points_to_pixels(prop.get_size_in_points()))
            opts = cairo.FontOptions()
            opts.set_antialias(
                cairo.ANTIALIAS_DEFAULT if mpl.rcParams["text.antialiased"]
                else cairo.ANTIALIAS_NONE)
            ctx.set_font_options(opts)
            if angle:
                ctx.rotate(np.deg2rad(-angle))
            ctx.show_text(s)
            ctx.restore()

    def _draw_mathtext(self, gc, x, y, s, prop, angle):
        ctx = gc.ctx
        width, height, descent, glyphs, rects = \
            self._text2path.mathtext_parser.parse(s, self.dpi, prop)

        ctx.save()
        ctx.translate(x, y)
        if angle:
            ctx.rotate(np.deg2rad(-angle))

        for font, fontsize, idx, ox, oy in glyphs:
            ctx.new_path()
            ctx.move_to(ox, -oy)
            ctx.select_font_face(
                *_cairo_font_args_from_font_prop(ttfFontProperty(font)))
            ctx.set_font_size(self.points_to_pixels(fontsize))
            ctx.show_text(chr(idx))

        for ox, oy, w, h in rects:
            ctx.new_path()
            ctx.rectangle(ox, -oy, w, -h)
            ctx.set_source_rgb(0, 0, 0)
            ctx.fill_preserve()

        ctx.restore()

    def get_canvas_width_height(self):
        # docstring inherited
        return self.width, self.height

    def get_text_width_height_descent(self, s, prop, ismath):
        # docstring inherited

        if ismath == 'TeX':
            return super().get_text_width_height_descent(s, prop, ismath)

        if ismath:
            width, height, descent, *_ = \
                self._text2path.mathtext_parser.parse(s, self.dpi, prop)
            return width, height, descent

        ctx = self.text_ctx
        # problem - scale remembers last setting and font can become
        # enormous causing program to crash
        # save/restore prevents the problem
        ctx.save()
        ctx.select_font_face(*_cairo_font_args_from_font_prop(prop))
        ctx.set_font_size(self.points_to_pixels(prop.get_size_in_points()))

        y_bearing, w, h = ctx.text_extents(s)[1:4]
        ctx.restore()

        return w, h, h + y_bearing

    def new_gc(self):
        # docstring inherited
        self.gc.ctx.save()
        self.gc._alpha = 1
        self.gc._forced_alpha = False  # if True, _alpha overrides A from RGBA
        return self.gc

    def points_to_pixels(self, points):
        # docstring inherited
        return points / 72 * self.dpi


class GraphicsContextCairo(GraphicsContextBase):
    _joind = {
        'bevel':  cairo.LINE_JOIN_BEVEL,
        'miter':  cairo.LINE_JOIN_MITER,
        'round':  cairo.LINE_JOIN_ROUND,
    }

    _capd = {
        'butt':        cairo.LINE_CAP_BUTT,
        'projecting':  cairo.LINE_CAP_SQUARE,
        'round':       cairo.LINE_CAP_ROUND,
    }

    def __init__(self, renderer):
        super().__init__()
        self.renderer = renderer

    def restore(self):
        self.ctx.restore()

    def set_alpha(self, alpha):
        super().set_alpha(alpha)
        _alpha = self.get_alpha()
        rgb = self._rgb
        if self.get_forced_alpha():
            self.ctx.set_source_rgba(rgb[0], rgb[1], rgb[2], _alpha)
        else:
            self.ctx.set_source_rgba(rgb[0], rgb[1], rgb[2], rgb[3])

    def set_antialiased(self, b):
        self.ctx.set_antialias(
            cairo.ANTIALIAS_DEFAULT if b else cairo.ANTIALIAS_NONE)

    def set_capstyle(self, cs):
        self.ctx.set_line_cap(_api.check_getitem(self._capd, capstyle=cs))
        self._capstyle = cs

    def set_clip_rectangle(self, rectangle):
        if not rectangle:
            return
        x, y, w, h = np.round(rectangle.bounds)
        ctx = self.ctx
        ctx.new_path()
        ctx.rectangle(x, self.renderer.height - h - y, w, h)
        ctx.clip()

    def set_clip_path(self, path):
        if not path:
            return
        tpath, affine = path.get_transformed_path_and_affine()
        ctx = self.ctx
        ctx.new_path()
        affine = (affine
                  + Affine2D().scale(1, -1).translate(0, self.renderer.height))
        _append_path(ctx, tpath, affine)
        ctx.clip()

    def set_dashes(self, offset, dashes):
        self._dashes = offset, dashes
        if dashes is None:
            self.ctx.set_dash([], 0)  # switch dashes off
        else:
            self.ctx.set_dash(
                list(self.renderer.points_to_pixels(np.asarray(dashes))),
                offset)

    def set_foreground(self, fg, isRGBA=None):
        super().set_foreground(fg, isRGBA)
        if len(self._rgb) == 3:
            self.ctx.set_source_rgb(*self._rgb)
        else:
            self.ctx.set_source_rgba(*self._rgb)

    def get_rgb(self):
        return self.ctx.get_source().get_rgba()[:3]

    def set_joinstyle(self, js):
        self.ctx.set_line_join(_api.check_getitem(self._joind, joinstyle=js))
        self._joinstyle = js

    def set_linewidth(self, w):
        self._linewidth = float(w)
        self.ctx.set_line_width(self.renderer.points_to_pixels(w))


class _CairoRegion:
    def __init__(self, slices, data):
        self._slices = slices
        self._data = data


class FigureCanvasCairo(FigureCanvasBase):

    def copy_from_bbox(self, bbox):
        surface = self._renderer.gc.ctx.get_target()
        if not isinstance(surface, cairo.ImageSurface):
            raise RuntimeError(
                "copy_from_bbox only works when rendering to an ImageSurface")
        sw = surface.get_width()
        sh = surface.get_height()
        x0 = math.ceil(bbox.x0)
        x1 = math.floor(bbox.x1)
        y0 = math.ceil(sh - bbox.y1)
        y1 = math.floor(sh - bbox.y0)
        if not (0 <= x0 and x1 <= sw and bbox.x0 <= bbox.x1
                and 0 <= y0 and y1 <= sh and bbox.y0 <= bbox.y1):
            raise ValueError("Invalid bbox")
        sls = slice(y0, y0 + max(y1 - y0, 0)), slice(x0, x0 + max(x1 - x0, 0))
        data = (np.frombuffer(surface.get_data(), np.uint32)
                .reshape((sh, sw))[sls].copy())
        return _CairoRegion(sls, data)

    def restore_region(self, region):
        surface = self._renderer.gc.ctx.get_target()
        if not isinstance(surface, cairo.ImageSurface):
            raise RuntimeError(
                "restore_region only works when rendering to an ImageSurface")
        surface.flush()
        sw = surface.get_width()
        sh = surface.get_height()
        sly, slx = region._slices
        (np.frombuffer(surface.get_data(), np.uint32)
         .reshape((sh, sw))[sly, slx]) = region._data
        surface.mark_dirty_rectangle(
            slx.start, sly.start, slx.stop - slx.start, sly.stop - sly.start)

    @_check_savefig_extra_args
    def print_png(self, fobj):
        self._get_printed_image_surface().write_to_png(fobj)

    @_check_savefig_extra_args
    def print_rgba(self, fobj):
        width, height = self.get_width_height()
        buf = self._get_printed_image_surface().get_data()
        fobj.write(cbook._premultiplied_argb32_to_unmultiplied_rgba8888(
            np.asarray(buf).reshape((width, height, 4))))

    print_raw = print_rgba

    def _get_printed_image_surface(self):
        width, height = self.get_width_height()
        renderer = RendererCairo(self.figure.dpi)
        renderer.set_width_height(width, height)
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        renderer.set_ctx_from_surface(surface)
        self.figure.draw(renderer)
        return surface

    @_check_savefig_extra_args
    def _save(self, fmt, fobj, *, orientation='portrait'):
        # save PDF/PS/SVG

        dpi = 72
        self.figure.dpi = dpi
        w_in, h_in = self.figure.get_size_inches()
        width_in_points, height_in_points = w_in * dpi, h_in * dpi

        if orientation == 'landscape':
            width_in_points, height_in_points = (
                height_in_points, width_in_points)

        if fmt == 'ps':
            if not hasattr(cairo, 'PSSurface'):
                raise RuntimeError('cairo has not been compiled with PS '
                                   'support enabled')
            surface = cairo.PSSurface(fobj, width_in_points, height_in_points)
        elif fmt == 'pdf':
            if not hasattr(cairo, 'PDFSurface'):
                raise RuntimeError('cairo has not been compiled with PDF '
                                   'support enabled')
            surface = cairo.PDFSurface(fobj, width_in_points, height_in_points)
        elif fmt in ('svg', 'svgz'):
            if not hasattr(cairo, 'SVGSurface'):
                raise RuntimeError('cairo has not been compiled with SVG '
                                   'support enabled')
            if fmt == 'svgz':
                if isinstance(fobj, str):
                    fobj = gzip.GzipFile(fobj, 'wb')
                else:
                    fobj = gzip.GzipFile(None, 'wb', fileobj=fobj)
            surface = cairo.SVGSurface(fobj, width_in_points, height_in_points)
        else:
            raise ValueError("Unknown format: {!r}".format(fmt))

        # surface.set_dpi() can be used
        renderer = RendererCairo(self.figure.dpi)
        renderer.set_width_height(width_in_points, height_in_points)
        renderer.set_ctx_from_surface(surface)
        ctx = renderer.gc.ctx

        if orientation == 'landscape':
            ctx.rotate(np.pi / 2)
            ctx.translate(0, -height_in_points)
            # Perhaps add an '%%Orientation: Landscape' comment?

        self.figure.draw(renderer)

        ctx.show_page()
        surface.finish()
        if fmt == 'svgz':
            fobj.close()

    print_pdf = functools.partialmethod(_save, "pdf")
    print_ps = functools.partialmethod(_save, "ps")
    print_svg = functools.partialmethod(_save, "svg")
    print_svgz = functools.partialmethod(_save, "svgz")


@_Backend.export
class _BackendCairo(_Backend):
    FigureCanvas = FigureCanvasCairo
    FigureManager = FigureManagerBase
