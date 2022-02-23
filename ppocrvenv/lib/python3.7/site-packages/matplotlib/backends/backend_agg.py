"""
An `Anti-Grain Geometry`_ (AGG) backend.

Features that are implemented:

* capstyles and join styles
* dashes
* linewidth
* lines, rectangles, ellipses
* clipping to a rectangle
* output to RGBA and Pillow-supported image formats
* alpha blending
* DPI scaling properly - everything scales properly (dashes, linewidths, etc)
* draw polygon
* freetype2 w/ ft2font

Still TODO:

* integrate screen dpi w/ ppi and text

.. _Anti-Grain Geometry: http://agg.sourceforge.net/antigrain.com
"""

try:
    import threading
except ImportError:
    import dummy_threading as threading
from contextlib import nullcontext
from math import radians, cos, sin

import numpy as np
from PIL import Image

import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import colors as mcolors
from matplotlib.backend_bases import (
    _Backend, _check_savefig_extra_args, FigureCanvasBase, FigureManagerBase,
    RendererBase)
from matplotlib.font_manager import findfont, get_font
from matplotlib.ft2font import (LOAD_FORCE_AUTOHINT, LOAD_NO_HINTING,
                                LOAD_DEFAULT, LOAD_NO_AUTOHINT)
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.transforms import Bbox, BboxBase
from matplotlib.backends._backend_agg import RendererAgg as _RendererAgg


backend_version = 'v2.2'


def get_hinting_flag():
    mapping = {
        'default': LOAD_DEFAULT,
        'no_autohint': LOAD_NO_AUTOHINT,
        'force_autohint': LOAD_FORCE_AUTOHINT,
        'no_hinting': LOAD_NO_HINTING,
        True: LOAD_FORCE_AUTOHINT,
        False: LOAD_NO_HINTING,
        'either': LOAD_DEFAULT,
        'native': LOAD_NO_AUTOHINT,
        'auto': LOAD_FORCE_AUTOHINT,
        'none': LOAD_NO_HINTING,
    }
    return mapping[mpl.rcParams['text.hinting']]


class RendererAgg(RendererBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles
    """

    # we want to cache the fonts at the class level so that when
    # multiple figures are created we can reuse them.  This helps with
    # a bug on windows where the creation of too many figures leads to
    # too many open file handles.  However, storing them at the class
    # level is not thread safe.  The solution here is to let the
    # FigureCanvas acquire a lock on the fontd at the start of the
    # draw, and release it when it is done.  This allows multiple
    # renderers to share the cached fonts, but only one figure can
    # draw at time and so the font cache is used by only one
    # renderer at a time.

    lock = threading.RLock()

    def __init__(self, width, height, dpi):
        super().__init__()

        self.dpi = dpi
        self.width = width
        self.height = height
        self._renderer = _RendererAgg(int(width), int(height), dpi)
        self._filter_renderers = []

        self._update_methods()
        self.mathtext_parser = MathTextParser('Agg')

        self.bbox = Bbox.from_bounds(0, 0, self.width, self.height)

    def __getstate__(self):
        # We only want to preserve the init keywords of the Renderer.
        # Anything else can be re-created.
        return {'width': self.width, 'height': self.height, 'dpi': self.dpi}

    def __setstate__(self, state):
        self.__init__(state['width'], state['height'], state['dpi'])

    def _update_methods(self):
        self.draw_gouraud_triangle = self._renderer.draw_gouraud_triangle
        self.draw_gouraud_triangles = self._renderer.draw_gouraud_triangles
        self.draw_image = self._renderer.draw_image
        self.draw_markers = self._renderer.draw_markers
        self.draw_path_collection = self._renderer.draw_path_collection
        self.draw_quad_mesh = self._renderer.draw_quad_mesh
        self.copy_from_bbox = self._renderer.copy_from_bbox

    @_api.deprecated("3.4")
    def get_content_extents(self):
        orig_img = np.asarray(self.buffer_rgba())
        slice_y, slice_x = cbook._get_nonzero_slices(orig_img[..., 3])
        return (slice_x.start, slice_y.start,
                slice_x.stop - slice_x.start, slice_y.stop - slice_y.start)

    @_api.deprecated("3.4")
    def tostring_rgba_minimized(self):
        extents = self.get_content_extents()
        bbox = [[extents[0], self.height - (extents[1] + extents[3])],
                [extents[0] + extents[2], self.height - extents[1]]]
        region = self.copy_from_bbox(bbox)
        return np.array(region), extents

    def draw_path(self, gc, path, transform, rgbFace=None):
        # docstring inherited
        nmax = mpl.rcParams['agg.path.chunksize']  # here at least for testing
        npts = path.vertices.shape[0]

        if (npts > nmax > 100 and path.should_simplify and
                rgbFace is None and gc.get_hatch() is None):
            nch = np.ceil(npts / nmax)
            chsize = int(np.ceil(npts / nch))
            i0 = np.arange(0, npts, chsize)
            i1 = np.zeros_like(i0)
            i1[:-1] = i0[1:] - 1
            i1[-1] = npts
            for ii0, ii1 in zip(i0, i1):
                v = path.vertices[ii0:ii1, :]
                c = path.codes
                if c is not None:
                    c = c[ii0:ii1]
                    c[0] = Path.MOVETO  # move to end of last chunk
                p = Path(v, c)
                p.simplify_threshold = path.simplify_threshold
                try:
                    self._renderer.draw_path(gc, p, transform, rgbFace)
                except OverflowError as err:
                    msg = (
                        "Exceeded cell block limit in Agg.\n\n"
                        "Please reduce the value of "
                        f"rcParams['agg.path.chunksize'] (currently {nmax}) "
                        "or increase the path simplification threshold"
                        "(rcParams['path.simplify_threshold'] = "
                        f"{mpl.rcParams['path.simplify_threshold']:.2f} by "
                        "default and path.simplify_threshold = "
                        f"{path.simplify_threshold:.2f} on the input)."
                    )
                    raise OverflowError(msg) from None
        else:
            try:
                self._renderer.draw_path(gc, path, transform, rgbFace)
            except OverflowError as err:
                cant_chunk = ''
                if rgbFace is not None:
                    cant_chunk += "- can not split filled path\n"
                if gc.get_hatch() is not None:
                    cant_chunk += "- can not split hatched path\n"
                if not path.should_simplify:
                    cant_chunk += "- path.should_simplify is False\n"
                if len(cant_chunk):
                    msg = (
                        "Exceeded cell block limit in Agg, however for the "
                        "following reasons:\n\n"
                        f"{cant_chunk}\n"
                        "we can not automatically split up this path to draw."
                        "\n\nPlease manually simplify your path."
                    )

                else:
                    inc_threshold = (
                        "or increase the path simplification threshold"
                        "(rcParams['path.simplify_threshold'] = "
                        f"{mpl.rcParams['path.simplify_threshold']} "
                        "by default and path.simplify_threshold "
                        f"= {path.simplify_threshold} "
                        "on the input)."
                        )
                    if nmax > 100:
                        msg = (
                            "Exceeded cell block limit in Agg.  Please reduce "
                            "the value of rcParams['agg.path.chunksize'] "
                            f"(currently {nmax}) {inc_threshold}"
                        )
                    else:
                        msg = (
                            "Exceeded cell block limit in Agg.  Please set "
                            "the value of rcParams['agg.path.chunksize'], "
                            f"(currently {nmax}) to be greater than 100 "
                            + inc_threshold
                        )

                raise OverflowError(msg) from None

    def draw_mathtext(self, gc, x, y, s, prop, angle):
        """Draw mathtext using :mod:`matplotlib.mathtext`."""
        ox, oy, width, height, descent, font_image, used_characters = \
            self.mathtext_parser.parse(s, self.dpi, prop)

        xd = descent * sin(radians(angle))
        yd = descent * cos(radians(angle))
        x = round(x + ox + xd)
        y = round(y - oy + yd)
        self._renderer.draw_text_image(font_image, x, y + 1, angle, gc)

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # docstring inherited

        if ismath:
            return self.draw_mathtext(gc, x, y, s, prop, angle)

        flags = get_hinting_flag()
        font = self._get_agg_font(prop)

        if font is None:
            return None
        # We pass '0' for angle here, since it will be rotated (in raster
        # space) in the following call to draw_text_image).
        font.set_text(s, 0, flags=flags)
        font.draw_glyphs_to_bitmap(
            antialiased=mpl.rcParams['text.antialiased'])
        d = font.get_descent() / 64.0
        # The descent needs to be adjusted for the angle.
        xo, yo = font.get_bitmap_offset()
        xo /= 64.0
        yo /= 64.0
        xd = d * sin(radians(angle))
        yd = d * cos(radians(angle))
        x = round(x + xo + xd)
        y = round(y + yo + yd)
        self._renderer.draw_text_image(font, x, y + 1, angle, gc)

    def get_text_width_height_descent(self, s, prop, ismath):
        # docstring inherited

        _api.check_in_list(["TeX", True, False], ismath=ismath)
        if ismath == "TeX":
            # todo: handle props
            texmanager = self.get_texmanager()
            fontsize = prop.get_size_in_points()
            w, h, d = texmanager.get_text_width_height_descent(
                s, fontsize, renderer=self)
            return w, h, d

        if ismath:
            ox, oy, width, height, descent, fonts, used_characters = \
                self.mathtext_parser.parse(s, self.dpi, prop)
            return width, height, descent

        flags = get_hinting_flag()
        font = self._get_agg_font(prop)
        font.set_text(s, 0.0, flags=flags)
        w, h = font.get_width_height()  # width and height of unrotated string
        d = font.get_descent()
        w /= 64.0  # convert from subpixels
        h /= 64.0
        d /= 64.0
        return w, h, d

    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=None):
        # docstring inherited
        # todo, handle props, angle, origins
        size = prop.get_size_in_points()

        texmanager = self.get_texmanager()

        Z = texmanager.get_grey(s, size, self.dpi)
        Z = np.array(Z * 255.0, np.uint8)

        w, h, d = self.get_text_width_height_descent(s, prop, ismath="TeX")
        xd = d * sin(radians(angle))
        yd = d * cos(radians(angle))
        x = round(x + xd)
        y = round(y + yd)
        self._renderer.draw_text_image(Z, x, y, angle, gc)

    def get_canvas_width_height(self):
        # docstring inherited
        return self.width, self.height

    def _get_agg_font(self, prop):
        """
        Get the font for text instance t, caching for efficiency
        """
        fname = findfont(prop)
        font = get_font(fname)

        font.clear()
        size = prop.get_size_in_points()
        font.set_size(size, self.dpi)

        return font

    def points_to_pixels(self, points):
        # docstring inherited
        return points * self.dpi / 72

    def buffer_rgba(self):
        return memoryview(self._renderer)

    def tostring_argb(self):
        return np.asarray(self._renderer).take([3, 0, 1, 2], axis=2).tobytes()

    def tostring_rgb(self):
        return np.asarray(self._renderer).take([0, 1, 2], axis=2).tobytes()

    def clear(self):
        self._renderer.clear()

    def option_image_nocomposite(self):
        # docstring inherited

        # It is generally faster to composite each image directly to
        # the Figure, and there's no file size benefit to compositing
        # with the Agg backend
        return True

    def option_scale_image(self):
        # docstring inherited
        return False

    def restore_region(self, region, bbox=None, xy=None):
        """
        Restore the saved region. If bbox (instance of BboxBase, or
        its extents) is given, only the region specified by the bbox
        will be restored. *xy* (a pair of floats) optionally
        specifies the new position (the LLC of the original region,
        not the LLC of the bbox) where the region will be restored.

        >>> region = renderer.copy_from_bbox()
        >>> x1, y1, x2, y2 = region.get_extents()
        >>> renderer.restore_region(region, bbox=(x1+dx, y1, x2, y2),
        ...                         xy=(x1-dx, y1))

        """
        if bbox is not None or xy is not None:
            if bbox is None:
                x1, y1, x2, y2 = region.get_extents()
            elif isinstance(bbox, BboxBase):
                x1, y1, x2, y2 = bbox.extents
            else:
                x1, y1, x2, y2 = bbox

            if xy is None:
                ox, oy = x1, y1
            else:
                ox, oy = xy

            # The incoming data is float, but the _renderer type-checking wants
            # to see integers.
            self._renderer.restore_region(region, int(x1), int(y1),
                                          int(x2), int(y2), int(ox), int(oy))

        else:
            self._renderer.restore_region(region)

    def start_filter(self):
        """
        Start filtering. It simply create a new canvas (the old one is saved).
        """
        self._filter_renderers.append(self._renderer)
        self._renderer = _RendererAgg(int(self.width), int(self.height),
                                      self.dpi)
        self._update_methods()

    def stop_filter(self, post_processing):
        """
        Save the plot in the current canvas as a image and apply
        the *post_processing* function.

           def post_processing(image, dpi):
             # ny, nx, depth = image.shape
             # image (numpy array) has RGBA channels and has a depth of 4.
             ...
             # create a new_image (numpy array of 4 channels, size can be
             # different). The resulting image may have offsets from
             # lower-left corner of the original image
             return new_image, offset_x, offset_y

        The saved renderer is restored and the returned image from
        post_processing is plotted (using draw_image) on it.
        """
        orig_img = np.asarray(self.buffer_rgba())
        slice_y, slice_x = cbook._get_nonzero_slices(orig_img[..., 3])
        cropped_img = orig_img[slice_y, slice_x]

        self._renderer = self._filter_renderers.pop()
        self._update_methods()

        if cropped_img.size:
            img, ox, oy = post_processing(cropped_img / 255, self.dpi)
            gc = self.new_gc()
            if img.dtype.kind == 'f':
                img = np.asarray(img * 255., np.uint8)
            self._renderer.draw_image(
                gc, slice_x.start + ox, int(self.height) - slice_y.stop + oy,
                img[::-1])


class FigureCanvasAgg(FigureCanvasBase):
    # docstring inherited

    def copy_from_bbox(self, bbox):
        renderer = self.get_renderer()
        return renderer.copy_from_bbox(bbox)

    def restore_region(self, region, bbox=None, xy=None):
        renderer = self.get_renderer()
        return renderer.restore_region(region, bbox, xy)

    def draw(self):
        # docstring inherited
        self.renderer = self.get_renderer(cleared=True)
        # Acquire a lock on the shared font cache.
        with RendererAgg.lock, \
             (self.toolbar._wait_cursor_for_draw_cm() if self.toolbar
              else nullcontext()):
            self.figure.draw(self.renderer)
            # A GUI class may be need to update a window using this draw, so
            # don't forget to call the superclass.
            super().draw()

    def get_renderer(self, cleared=False):
        w, h = self.figure.bbox.size
        key = w, h, self.figure.dpi
        reuse_renderer = (hasattr(self, "renderer")
                          and getattr(self, "_lastKey", None) == key)
        if not reuse_renderer:
            self.renderer = RendererAgg(w, h, self.figure.dpi)
            self._lastKey = key
        elif cleared:
            self.renderer.clear()
        return self.renderer

    def tostring_rgb(self):
        """
        Get the image as RGB `bytes`.

        `draw` must be called at least once before this function will work and
        to update the renderer for any subsequent changes to the Figure.
        """
        return self.renderer.tostring_rgb()

    def tostring_argb(self):
        """
        Get the image as ARGB `bytes`.

        `draw` must be called at least once before this function will work and
        to update the renderer for any subsequent changes to the Figure.
        """
        return self.renderer.tostring_argb()

    def buffer_rgba(self):
        """
        Get the image as a `memoryview` to the renderer's buffer.

        `draw` must be called at least once before this function will work and
        to update the renderer for any subsequent changes to the Figure.
        """
        return self.renderer.buffer_rgba()

    @_check_savefig_extra_args
    @_api.delete_parameter("3.5", "args")
    def print_raw(self, filename_or_obj, *args):
        FigureCanvasAgg.draw(self)
        renderer = self.get_renderer()
        with cbook.open_file_cm(filename_or_obj, "wb") as fh:
            fh.write(renderer.buffer_rgba())

    print_rgba = print_raw

    @_check_savefig_extra_args
    @_api.delete_parameter("3.5", "args")
    def print_png(self, filename_or_obj, *args,
                  metadata=None, pil_kwargs=None):
        """
        Write the figure to a PNG file.

        Parameters
        ----------
        filename_or_obj : str or path-like or file-like
            The file to write to.

        metadata : dict, optional
            Metadata in the PNG file as key-value pairs of bytes or latin-1
            encodable strings.
            According to the PNG specification, keys must be shorter than 79
            chars.

            The `PNG specification`_ defines some common keywords that may be
            used as appropriate:

            - Title: Short (one line) title or caption for image.
            - Author: Name of image's creator.
            - Description: Description of image (possibly long).
            - Copyright: Copyright notice.
            - Creation Time: Time of original image creation
              (usually RFC 1123 format).
            - Software: Software used to create the image.
            - Disclaimer: Legal disclaimer.
            - Warning: Warning of nature of content.
            - Source: Device used to create the image.
            - Comment: Miscellaneous comment;
              conversion from other image format.

            Other keywords may be invented for other purposes.

            If 'Software' is not given, an autogenerated value for Matplotlib
            will be used.  This can be removed by setting it to *None*.

            For more details see the `PNG specification`_.

            .. _PNG specification: \
                https://www.w3.org/TR/2003/REC-PNG-20031110/#11keywords

        pil_kwargs : dict, optional
            Keyword arguments passed to `PIL.Image.Image.save`.

            If the 'pnginfo' key is present, it completely overrides
            *metadata*, including the default 'Software' key.
        """
        FigureCanvasAgg.draw(self)
        mpl.image.imsave(
            filename_or_obj, self.buffer_rgba(), format="png", origin="upper",
            dpi=self.figure.dpi, metadata=metadata, pil_kwargs=pil_kwargs)

    def print_to_buffer(self):
        FigureCanvasAgg.draw(self)
        renderer = self.get_renderer()
        return (bytes(renderer.buffer_rgba()),
                (int(renderer.width), int(renderer.height)))

    # Note that these methods should typically be called via savefig() and
    # print_figure(), and the latter ensures that `self.figure.dpi` already
    # matches the dpi kwarg (if any).

    @_check_savefig_extra_args()
    @_api.delete_parameter("3.5", "args")
    def print_jpg(self, filename_or_obj, *args, pil_kwargs=None, **kwargs):
        """
        Write the figure to a JPEG file.

        Parameters
        ----------
        filename_or_obj : str or path-like or file-like
            The file to write to.

        Other Parameters
        ----------------
        pil_kwargs : dict, optional
            Additional keyword arguments that are passed to
            `PIL.Image.Image.save` when saving the figure.
        """
        # Remove transparency by alpha-blending on an assumed white background.
        r, g, b, a = mcolors.to_rgba(self.figure.get_facecolor())
        try:
            self.figure.set_facecolor(a * np.array([r, g, b]) + 1 - a)
            FigureCanvasAgg.draw(self)
        finally:
            self.figure.set_facecolor((r, g, b, a))
        if pil_kwargs is None:
            pil_kwargs = {}
        pil_kwargs.setdefault("dpi", (self.figure.dpi, self.figure.dpi))
        # Drop alpha channel now.
        return (Image.fromarray(np.asarray(self.buffer_rgba())[..., :3])
                .save(filename_or_obj, format='jpeg', **pil_kwargs))

    print_jpeg = print_jpg

    @_check_savefig_extra_args
    def print_tif(self, filename_or_obj, *, pil_kwargs=None):
        FigureCanvasAgg.draw(self)
        if pil_kwargs is None:
            pil_kwargs = {}
        pil_kwargs.setdefault("dpi", (self.figure.dpi, self.figure.dpi))
        return (Image.fromarray(np.asarray(self.buffer_rgba()))
                .save(filename_or_obj, format='tiff', **pil_kwargs))

    print_tiff = print_tif


@_Backend.export
class _BackendAgg(_Backend):
    FigureCanvas = FigureCanvasAgg
    FigureManager = FigureManagerBase
