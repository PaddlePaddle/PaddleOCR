""":mod:`wand.drawing` --- Drawings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The module provides some vector drawing functions.

.. versionadded:: 0.3.0

"""
import collections
import ctypes
import numbers

from . import assertions
from .api import AffineMatrix, PointInfo, library
from .color import Color
from .compat import abc, binary, string_type, text, text_type, xrange
from .exceptions import WandLibraryVersionError
from .image import BaseImage, COMPOSITE_OPERATORS
from .sequence import SingleImage
from .resource import Resource

__all__ = ('CLIP_PATH_UNITS', 'FILL_RULE_TYPES', 'FONT_METRICS_ATTRIBUTES',
           'GRAVITY_TYPES', 'LINE_CAP_TYPES', 'LINE_JOIN_TYPES',
           'PAINT_METHOD_TYPES', 'STRETCH_TYPES', 'STYLE_TYPES',
           'TEXT_ALIGN_TYPES', 'TEXT_DECORATION_TYPES',
           'TEXT_DIRECTION_TYPES', 'Drawing', 'FontMetrics')


#: (:class:`collections.abc.Sequence`) The list of clip path units
#:
#: - ``'undefined_path_units'``
#: - ``'user_space'``
#: - ``'user_space_on_use'``
#: - ``'object_bounding_box'``
CLIP_PATH_UNITS = ('undefined_path_units', 'user_space', 'user_space_on_use',
                   'object_bounding_box')

#: (:class:`collections.abc.Sequence`) The list of text align types.
#:
#: - ``'undefined'``
#: - ``'left'``
#: - ``'center'``
#: - ``'right'``
TEXT_ALIGN_TYPES = 'undefined', 'left', 'center', 'right'

#: (:class:`collections.abc.Sequence`) The list of text decoration types.
#:
#: - ``'undefined'``
#: - ``'no'``
#: - ``'underline'``
#: - ``'overline'``
#: - ``'line_through'``
TEXT_DECORATION_TYPES = ('undefined', 'no', 'underline', 'overline',
                         'line_through')

#: (:class:`collections.abc.Sequence`) The list of text direction types.
#:
#: - ``'undefined'``
#: - ``'right_to_left'``
#: - ``'left_to_right'``
TEXT_DIRECTION_TYPES = ('undefined', 'right_to_left', 'left_to_right')

#: (:class:`collections.abc.Sequence`) The list of text gravity types.
#:
#: - ``'forget'``
#: - ``'north_west'``
#: - ``'north'``
#: - ``'north_east'``
#: - ``'west'``
#: - ``'center'``
#: - ``'east'``
#: - ``'south_west'``
#: - ``'south'``
#: - ``'south_east'``
#: - ``'static'``
GRAVITY_TYPES = ('forget', 'north_west', 'north', 'north_east', 'west',
                 'center', 'east', 'south_west', 'south', 'south_east',
                 'static')

#: (:class:`collections.abc.Sequence`) The list of fill-rule types.
#:
#: - ``'undefined'``
#: - ``'evenodd'``
#: - ``'nonzero'``
FILL_RULE_TYPES = ('undefined', 'evenodd', 'nonzero')

#: (:class:`collections.abc.Sequence`) The attribute names of font metrics.
FONT_METRICS_ATTRIBUTES = ('character_width', 'character_height', 'ascender',
                           'descender', 'text_width', 'text_height',
                           'maximum_horizontal_advance', 'x1', 'y1', 'x2',
                           'y2', 'x', 'y')

#: The tuple subtype which consists of font metrics data.
FontMetrics = collections.namedtuple('FontMetrics', FONT_METRICS_ATTRIBUTES)

#: (:class:`collections.abc.Sequence`) The list of stretch types for fonts
#:
#: - ``'undefined;``
#: - ``'normal'``
#: - ``'ultra_condensed'``
#: - ``'extra_condensed'``
#: - ``'condensed'``
#: - ``'semi_condensed'``
#: - ``'semi_expanded'``
#: - ``'expanded'``
#: - ``'extra_expanded'``
#: - ``'ultra_expanded'``
#: - ``'any'``
STRETCH_TYPES = ('undefined', 'normal', 'ultra_condensed', 'extra_condensed',
                 'condensed', 'semi_condensed', 'semi_expanded', 'expanded',
                 'extra_expanded', 'ultra_expanded', 'any')

#: (:class:`collections.abc.Sequence`) The list of style types for fonts
#:
#: - ``'undefined;``
#: - ``'normal'``
#: - ``'italic'``
#: - ``'oblique'``
#: - ``'any'``
STYLE_TYPES = ('undefined', 'normal', 'italic', 'oblique', 'any')

#: (:class:`collections.abc.Sequence`) The list of LineCap types
#:
#: - ``'undefined;``
#: - ``'butt'``
#: - ``'round'``
#: - ``'square'``
LINE_CAP_TYPES = ('undefined', 'butt', 'round', 'square')

#: (:class:`collections.abc.Sequence`) The list of LineJoin types
#:
#: - ``'undefined'``
#: - ``'miter'``
#: - ``'round'``
#: - ``'bevel'``
LINE_JOIN_TYPES = ('undefined', 'miter', 'round', 'bevel')


#: (:class:`collections.abc.Sequence`) The list of paint method types.
#:
#: - ``'undefined'``
#: - ``'point'``
#: - ``'replace'``
#: - ``'floodfill'``
#: - ``'filltoborder'``
#: - ``'reset'``
PAINT_METHOD_TYPES = ('undefined', 'point', 'replace',
                      'floodfill', 'filltoborder', 'reset')


class Drawing(Resource):
    """Drawing object.  It maintains several vector drawing instructions
    and can get drawn into zero or more :class:`~wand.image.Image` objects
    by calling it.

    For example, the following code draws a diagonal line to the ``image``::

        with Drawing() as draw:
            draw.line((0, 0), image.size)
            draw(image)

    :param drawing: an optional drawing object to clone.
                    use :meth:`clone()` method rather than this parameter
    :type drawing: :class:`Drawing`

    .. versionadded:: 0.3.0

    """

    c_is_resource = library.IsDrawingWand
    c_destroy_resource = library.DestroyDrawingWand
    c_get_exception = library.DrawGetException
    c_clear_exception = library.DrawClearException

    def __init__(self, drawing=None):
        with self.allocate():
            if not drawing:
                wand = library.NewDrawingWand()
            elif not isinstance(drawing, type(self)):
                raise TypeError('drawing must be a wand.drawing.Drawing '
                                'instance, not ' + repr(drawing))
            else:
                wand = library.CloneDrawingWand(drawing.resource)
            self.resource = wand

    @property
    def border_color(self):
        """(:class:`~wand.color.Color`) the current border color. It also can
        be set. This attribute controls the behavior of
        :meth:`~wand.drawing.Drawing.color()` during ``'filltoborder'``
        operation.

        .. versionadded:: 0.4.0
        """
        pixelwand = library.NewPixelWand()
        library.DrawGetBorderColor(self.resource, pixelwand)
        color = Color.from_pixelwand(pixelwand)
        pixelwand = library.DestroyPixelWand(pixelwand)
        return color

    @border_color.setter
    def border_color(self, color):
        if isinstance(color, string_type):
            color = Color(color)
        assertions.assert_color(border_color=color)
        with color:
            library.DrawSetBorderColor(self.resource, color.resource)

    @property
    def clip_path(self):
        """(:class:`basestring`) The current clip path. It also can be set.

        .. versionadded:: 0.4.0

        .. versionchanged: 0.4.1
           Safely release allocated memory with
           :c:func:`MagickRelinquishMemory` instead of :c:func:`libc.free`.

        """
        clip_path_str = None
        clip_path_p = library.DrawGetClipPath(self.resource)
        if clip_path_p:
            clip_path_str = text(ctypes.string_at(clip_path_p))
            clip_path_p = library.MagickRelinquishMemory(clip_path_p)
        return clip_path_str

    @clip_path.setter
    def clip_path(self, path):
        assertions.assert_string(clip_path=path)
        library.DrawSetClipPath(self.resource, binary(path))

    @property
    def clip_rule(self):
        """(:class:`basestring`) The current clip rule. It also can be set.
        It's a string value from :const:`FILL_RULE_TYPES` list.

        .. versionadded:: 0.4.0
        """
        clip_rule = library.DrawGetClipRule(self.resource)
        return FILL_RULE_TYPES[clip_rule]

    @clip_rule.setter
    def clip_rule(self, clip_rule):
        assertions.string_in_list(FILL_RULE_TYPES,
                                  'wand.drawing.FILL_RULE_TYPES',
                                  clip_rule=clip_rule)
        library.DrawSetClipRule(self.resource,
                                FILL_RULE_TYPES.index(clip_rule))

    @property
    def clip_units(self):
        """(:class:`basestring`) The current clip units. It also can be set.
        It's a string value from :const:`CLIP_PATH_UNITS` list.

        .. versionadded:: 0.4.0
        """
        clip_unit = library.DrawGetClipUnits(self.resource)
        return CLIP_PATH_UNITS[clip_unit]

    @clip_units.setter
    def clip_units(self, clip_unit):
        assertions.string_in_list(CLIP_PATH_UNITS,
                                  'wand.drawing.CLIP_PATH_UNITS',
                                  clip_unit=clip_unit)
        library.DrawSetClipUnits(self.resource,
                                 CLIP_PATH_UNITS.index(clip_unit))

    @property
    def fill_color(self):
        """(:class:`~wand.color.Color`) The current color to fill.
        It also can be set.

        """
        pixel = library.NewPixelWand()
        library.DrawGetFillColor(self.resource, pixel)
        color = Color.from_pixelwand(pixel)
        pixel = library.DestroyPixelWand(pixel)
        return color

    @fill_color.setter
    def fill_color(self, color):
        if isinstance(color, string_type):
            color = Color(color)
        assertions.assert_color(fill_color=color)
        with color:
            library.DrawSetFillColor(self.resource, color.resource)

    @property
    def fill_opacity(self):
        """(:class:`~numbers.Real`) The current fill opacity.
        It also can be set.

        .. versionadded:: 0.4.0
        """
        return library.DrawGetFillOpacity(self.resource)

    @fill_opacity.setter
    def fill_opacity(self, opacity):
        assertions.assert_real(fill_opacity=opacity)
        library.DrawSetFillOpacity(self.resource, opacity)

    @property
    def fill_rule(self):
        """(:class:`basestring`) The current fill rule. It can also be set.
        It's a string value from :const:`FILL_RULE_TYPES` list.

        .. versionadded:: 0.4.0
        """
        fill_rule_index = library.DrawGetFillRule(self.resource)
        if fill_rule_index not in FILL_RULE_TYPES:
            self.raise_exception()
        return text(FILL_RULE_TYPES[fill_rule_index])

    @fill_rule.setter
    def fill_rule(self, fill_rule):
        assertions.string_in_list(FILL_RULE_TYPES,
                                  'wand.drawing.FILL_RULE_TYPES',
                                  fill_rule=fill_rule)
        library.DrawSetFillRule(self.resource,
                                FILL_RULE_TYPES.index(fill_rule))

    @property
    def font(self):
        """(:class:`basestring`) The current font name.  It also can be set.

        .. versionchanged: 0.4.1
           Safely release allocated memory with
           :c:func:`MagickRelinquishMemory` instead of :c:func:`libc.free`.

        """
        font_str = None
        font_p = library.DrawGetFont(self.resource)
        if font_p:
            font_str = text(ctypes.string_at(font_p))
            font_p = library.MagickRelinquishMemory(font_p)
        return font_str

    @font.setter
    def font(self, font):
        assertions.assert_string(font=font)
        library.DrawSetFont(self.resource, binary(font))

    @property
    def font_family(self):
        """(:class:`basestring`) The current font family. It also can be set.

        .. versionadded:: 0.4.0

        .. versionchanged: 0.4.1
           Safely release allocated memory with
           :c:func:`MagickRelinquishMemory` instead of :c:func:`libc.free`.

        """
        font_family_str = None
        font_family_p = library.DrawGetFontFamily(self.resource)
        if font_family_p:
            font_family_str = text(ctypes.string_at(font_family_p))
            font_family_p = library.MagickRelinquishMemory(font_family_p)
        return font_family_str

    @font_family.setter
    def font_family(self, family):
        assertions.assert_string(font_family=family)
        library.DrawSetFontFamily(self.resource, binary(family))

    @property
    def font_resolution(self):
        """(:class:`~collections.abc.Sequence`) The current font resolution. It also
        can be set.

        .. versionadded:: 0.4.0
        """
        x, y = ctypes.c_double(0.0), ctypes.c_double(0.0)
        library.DrawGetFontResolution(self.resource,
                                      ctypes.byref(x),
                                      ctypes.byref(y))
        return x.value, y.value

    @font_resolution.setter
    def font_resolution(self, resolution):
        assertions.assert_coordinate(font_resolution=resolution)
        library.DrawSetFontResolution(self.resource, *resolution)

    @property
    def font_size(self):
        """(:class:`numbers.Real`) The font size.  It also can be set."""
        return library.DrawGetFontSize(self.resource)

    @font_size.setter
    def font_size(self, size):
        assertions.assert_real(font_size=size)
        if size < 0.0:
            raise ValueError('cannot be less than 0.0, but got ' + repr(size))
        library.DrawSetFontSize(self.resource, size)

    @property
    def font_stretch(self):
        """(:class:`basestring`) The current font stretch variation.
        It also can be set, but will only apply if the font-family or encoder
        supports the stretch type.

        .. versionadded:: 0.4.0
        """
        stretch_index = library.DrawGetFontStretch(self.resource)
        return text(STRETCH_TYPES[stretch_index])

    @font_stretch.setter
    def font_stretch(self, stretch):
        assertions.string_in_list(STRETCH_TYPES,
                                  'wand.drawing.STRETCH_TYPES',
                                  font_stretch=stretch)
        library.DrawSetFontStretch(self.resource,
                                   STRETCH_TYPES.index(stretch))

    @property
    def font_style(self):
        """(:class:`basestring`) The current font style.
        It also can be set, but will only apply if the font-family
        supports the style.

        .. versionadded:: 0.4.0
        """
        style_index = library.DrawGetFontStyle(self.resource)
        return text(STYLE_TYPES[style_index])

    @font_style.setter
    def font_style(self, style):
        assertions.string_in_list(STYLE_TYPES,
                                  'wand.drawing.STYLE_TYPES',
                                  font_style=style)
        library.DrawSetFontStyle(self.resource,
                                 STYLE_TYPES.index(style))

    @property
    def font_weight(self):
        """(:class:`~numbers.Integral`) The current font weight.
        It also can be set.

        .. versionadded:: 0.4.0
        """
        return library.DrawGetFontWeight(self.resource)

    @font_weight.setter
    def font_weight(self, weight):
        assertions.assert_integer(font_weight=weight)
        library.DrawSetFontWeight(self.resource, weight)

    @property
    def gravity(self):
        """(:class:`basestring`) The text placement gravity used when
        annotating with text.  It's a string from :const:`GRAVITY_TYPES`
        list.  It also can be set.

        """
        gravity_index = library.DrawGetGravity(self.resource)
        if not gravity_index:
            self.raise_exception()
        return text(GRAVITY_TYPES[gravity_index])

    @gravity.setter
    def gravity(self, value):
        assertions.string_in_list(GRAVITY_TYPES,
                                  'wand.drawing.GRAVITY_TYPES',
                                  gravity=value)
        library.DrawSetGravity(self.resource, GRAVITY_TYPES.index(value))

    @property
    def opacity(self):
        """(:class:`~numbers.Real`) returns the opacity used when drawing with
        the fill or stroke color or texture. Fully opaque is 1.0. This method
        only affects vector graphics, and is experimental. To set the opacity
        of a drawing, use
        :attr:`Drawing.fill_opacity` & :attr:`Drawing.stroke_opacity`

        .. versionadded:: 0.4.0
        """
        return library.DrawGetOpacity(self.resource)

    @opacity.setter
    def opacity(self, opaque):
        assertions.assert_real(opacity=opaque)
        library.DrawSetOpacity(self.resource, opaque)

    @property
    def stroke_antialias(self):
        """(:class:`bool`) Controls whether stroked outlines are antialiased.
        Stroked outlines are antialiased by default. When antialiasing is
        disabled stroked pixels are thresholded to determine if the stroke
        color or underlying canvas color should be used.

        It also can be set.

        .. versionadded:: 0.4.0

        """
        stroke_antialias = library.DrawGetStrokeAntialias(self.resource)
        return bool(stroke_antialias)

    @stroke_antialias.setter
    def stroke_antialias(self, stroke_antialias):
        assertions.assert_bool(stroke_antialias=stroke_antialias)
        library.DrawSetStrokeAntialias(self.resource, stroke_antialias)

    @property
    def stroke_color(self):
        """(:class:`~wand.color.Color`) The current color of stroke.
        It also can be set.

        .. versionadded:: 0.3.3

        """
        pixel = library.NewPixelWand()
        library.DrawGetStrokeColor(self.resource, pixel)
        color = Color.from_pixelwand(pixel)
        pixel = library.DestroyPixelWand(pixel)
        return color

    @stroke_color.setter
    def stroke_color(self, color):
        if isinstance(color, string_type):
            color = Color(color)
        assertions.assert_color(stroke_color=color)
        with color:
            library.DrawSetStrokeColor(self.resource, color.resource)

    @property
    def stroke_dash_array(self):
        """(:class:`~collections.abc.Sequence`) - (:class:`numbers.Real`) An
        array representing the pattern of dashes & gaps used to stroke paths.
        It also can be set.

        .. versionadded:: 0.4.0

        .. versionchanged: 0.4.1
           Safely release allocated memory with
           :c:func:`MagickRelinquishMemory` instead of :c:func:`libc.free`.

        """
        number_elements = ctypes.c_size_t(0)
        dash_array_p = library.DrawGetStrokeDashArray(
            self.resource, ctypes.byref(number_elements)
        )
        dash_array = []
        if dash_array_p:
            dash_array = [float(dash_array_p[i])
                          for i in xrange(number_elements.value)]
            dash_array_p = library.MagickRelinquishMemory(dash_array_p)
        return dash_array

    @stroke_dash_array.setter
    def stroke_dash_array(self, dash_array):
        dash_array_l = len(dash_array)
        dash_array_p = (ctypes.c_double * dash_array_l)(*dash_array)
        library.DrawSetStrokeDashArray(self.resource,
                                       dash_array_l,
                                       dash_array_p)

    @property
    def stroke_dash_offset(self):
        """(:class:`numbers.Real`) The stroke dash offset. It also can be set.

        .. versionadded:: 0.4.0
        """
        return library.DrawGetStrokeDashOffset(self.resource)

    @stroke_dash_offset.setter
    def stroke_dash_offset(self, offset):
        assertions.assert_real(stroke_dash_offset=offset)
        library.DrawSetStrokeDashOffset(self.resource, offset)

    @property
    def stroke_line_cap(self):
        """(:class:`basestring`) The stroke line cap. It also can be set.

        .. versionadded:: 0.4.0
        """
        line_cap_index = library.DrawGetStrokeLineCap(self.resource)
        if line_cap_index not in LINE_CAP_TYPES:
            self.raise_exception()
        return text(LINE_CAP_TYPES[line_cap_index])

    @stroke_line_cap.setter
    def stroke_line_cap(self, line_cap):
        assertions.string_in_list(LINE_CAP_TYPES,
                                  'wand.drawing.LINE_CAP_TYPES',
                                  stroke_line_cap=line_cap)
        library.DrawSetStrokeLineCap(self.resource,
                                     LINE_CAP_TYPES.index(line_cap))

    @property
    def stroke_line_join(self):
        """(:class:`basestring`) The stroke line join. It also can be set.

        .. versionadded:: 0.4.0
        """
        line_join_index = library.DrawGetStrokeLineJoin(self.resource)
        if line_join_index not in LINE_JOIN_TYPES:
            self.raise_exception()
        return text(LINE_JOIN_TYPES[line_join_index])

    @stroke_line_join.setter
    def stroke_line_join(self, line_join):
        assertions.string_in_list(LINE_JOIN_TYPES,
                                  'wand.drawing.LINE_JOIN_TYPES',
                                  stroke_line_join=line_join)
        library.DrawSetStrokeLineJoin(self.resource,
                                      LINE_JOIN_TYPES.index(line_join))

    @property
    def stroke_miter_limit(self):
        """(:class:`~numbers.Integral`) The current miter limit.
        It also can be set.

        .. versionadded:: 0.4.0
        """
        return library.DrawGetStrokeMiterLimit(self.resource)

    @stroke_miter_limit.setter
    def stroke_miter_limit(self, miter_limit):
        assertions.assert_integer(stroke_miter_limit=miter_limit)
        library.DrawSetStrokeMiterLimit(self.resource, miter_limit)

    @property
    def stroke_opacity(self):
        """(:class:`~numbers.Real`) The current stroke opacity.
        It also can be set.

        .. versionadded:: 0.4.0
        """
        return library.DrawGetStrokeOpacity(self.resource)

    @stroke_opacity.setter
    def stroke_opacity(self, opacity):
        assertions.assert_real(stroke_opacity=opacity)
        library.DrawSetStrokeOpacity(self.resource, opacity)

    @property
    def stroke_width(self):
        """(:class:`numbers.Real`) The stroke width.  It also can be set.

        .. versionadded:: 0.3.3

        """
        return library.DrawGetStrokeWidth(self.resource)

    @stroke_width.setter
    def stroke_width(self, width):
        assertions.assert_real(stroke_width=width)
        if width < 0.0:
            raise ValueError('cannot be less than 0.0, but got ' + repr(width))
        library.DrawSetStrokeWidth(self.resource, width)

    @property
    def text_alignment(self):
        """(:class:`basestring`) The current text alignment setting.
        It's a string value from :const:`TEXT_ALIGN_TYPES` list.
        It also can be set.

        """
        text_alignment_index = library.DrawGetTextAlignment(self.resource)
        if not text_alignment_index:  # pragma: no cover
            self.raise_exception()
        return text(TEXT_ALIGN_TYPES[text_alignment_index])

    @text_alignment.setter
    def text_alignment(self, align):
        assertions.string_in_list(TEXT_ALIGN_TYPES,
                                  'wand.drawing.TEXT_ALIGN_TYPES',
                                  text_alignment=align)
        library.DrawSetTextAlignment(self.resource,
                                     TEXT_ALIGN_TYPES.index(align))

    @property
    def text_antialias(self):
        """(:class:`bool`) The boolean value which represents whether
        antialiasing is used for text rendering.  It also can be set to
        ``True`` or ``False`` to switch the setting.

        """
        result = library.DrawGetTextAntialias(self.resource)
        return bool(result)

    @text_antialias.setter
    def text_antialias(self, value):
        assertions.assert_bool(text_antialias=value)
        library.DrawSetTextAntialias(self.resource, value)

    @property
    def text_decoration(self):
        """(:class:`basestring`) The text decoration setting, a string
        from :const:`TEXT_DECORATION_TYPES` list.  It also can be set.

        """
        text_decoration_index = library.DrawGetTextDecoration(self.resource)
        if not text_decoration_index:  # pragma: no cover
            self.raise_exception()
        return text(TEXT_DECORATION_TYPES[text_decoration_index])

    @text_decoration.setter
    def text_decoration(self, decoration):
        assertions.string_in_list(TEXT_DECORATION_TYPES,
                                  'wand.drawing.TEXT_DECORATION_TYPES',
                                  text_decoration=decoration)
        library.DrawSetTextDecoration(self.resource,
                                      TEXT_DECORATION_TYPES.index(decoration))

    @property
    def text_direction(self):
        """(:class:`basestring`) The text direction setting. a string
        from :const:`TEXT_DIRECTION_TYPES` list. It also can be set."""
        if library.DrawGetTextDirection is None:  # pragma: no cover
            raise WandLibraryVersionError(
                'the installed version of ImageMagick does not support '
                'this feature'
            )
        text_direction_index = library.DrawGetTextDirection(self.resource)
        if not text_direction_index:  # pragma: no cover
            self.raise_exception()
        return text(TEXT_DIRECTION_TYPES[text_direction_index])

    @text_direction.setter
    def text_direction(self, direction):
        if library.DrawGetTextDirection is None:  # pragma: no cover
            raise WandLibraryVersionError(
                'The installed version of ImageMagick does not support '
                'this feature'
            )
        assertions.string_in_list(TEXT_DIRECTION_TYPES,
                                  'wand.drawing.TEXT_DIRECTION_TYPES',
                                  text_direction=direction)
        library.DrawSetTextDirection(self.resource,
                                     TEXT_DIRECTION_TYPES.index(direction))

    @property
    def text_encoding(self):
        """(:class:`basestring`) The internally used text encoding setting.
        Although it also can be set, but it's not encouraged.

        .. versionchanged: 0.4.1
           Safely release allocated memory with
           :c:func:`MagickRelinquishMemory` instead of :c:func:`libc.free`.

        """
        text_encoding_str = None
        text_encoding_p = library.DrawGetTextEncoding(self.resource)
        if text_encoding_p:
            text_encoding_str = text(ctypes.string_at(text_encoding_p))
            text_encoding_p = library.MagickRelinquishMemory(text_encoding_p)
        return text_encoding_str

    @text_encoding.setter
    def text_encoding(self, encoding):
        if encoding is None:
            # encoding specify an empty string to set text encoding
            # to system's default.
            encoding = b''
        else:
            assertions.assert_string(text_encoding=encoding)
            encoding = binary(encoding)
        library.DrawSetTextEncoding(self.resource, encoding)

    @property
    def text_interline_spacing(self):
        """(:class:`numbers.Real`) The setting of the text line spacing.
        It also can be set.

        """
        if library.DrawGetTextInterlineSpacing is None:  # pragma: no cover
            raise WandLibraryVersionError('The installed version of '
                                          'ImageMagick does not support '
                                          'this feature')
        return library.DrawGetTextInterlineSpacing(self.resource)

    @text_interline_spacing.setter
    def text_interline_spacing(self, spacing):
        if library.DrawSetTextInterlineSpacing is None:  # pragma: no cover
            raise WandLibraryVersionError('The installed version of '
                                          'ImageMagick does not support '
                                          'this feature')
        assertions.assert_real(text_interline_spacing=spacing)
        library.DrawSetTextInterlineSpacing(self.resource, spacing)

    @property
    def text_interword_spacing(self):
        """(:class:`numbers.Real`) The setting of the word spacing.
        It also can be set.

        """
        return library.DrawGetTextInterwordSpacing(self.resource)

    @text_interword_spacing.setter
    def text_interword_spacing(self, spacing):
        assertions.assert_real(text_interword_spacing=spacing)
        library.DrawSetTextInterwordSpacing(self.resource, spacing)

    @property
    def text_kerning(self):
        """(:class:`numbers.Real`) The setting of the text kerning.
        It also can be set.

        """
        return library.DrawGetTextKerning(self.resource)

    @text_kerning.setter
    def text_kerning(self, kerning):
        assertions.assert_real(text_kerning=kerning)
        library.DrawSetTextKerning(self.resource, kerning)

    @property
    def text_under_color(self):
        """(:class:`~wand.color.Color`) The color of a background rectangle
        to place under text annotations.  It also can be set.

        """
        pixel = library.NewPixelWand()
        library.DrawGetTextUnderColor(self.resource, pixel)
        color = Color.from_pixelwand(pixel)
        pixel = library.DestroyPixelWand(pixel)
        return color

    @text_under_color.setter
    def text_under_color(self, color):
        if isinstance(color, string_type):
            color = Color(color)
        assertions.assert_color(text_under_color=color)
        with color:
            library.DrawSetTextUnderColor(self.resource, color.resource)

    @property
    def vector_graphics(self):
        """(:class:`basestring`) The XML text of the Vector Graphics.
        It also can be set.  The drawing-wand XML is experimental,
        and subject to change.

        Setting this property to None will reset all vector graphic properties
        to the default state.

        .. versionadded:: 0.4.0

        .. versionchanged: 0.4.1
           Safely release allocated memory with
           :c:func:`MagickRelinquishMemory` instead of :c:func:`libc.free`.

        """
        vg_p = library.DrawGetVectorGraphics(self.resource)
        if vg_p:
            vg_str = ctypes.string_at(vg_p)
            vg_p = library.MagickRelinquishMemory(vg_p)
        else:
            vg_str = b''
        return '<wand>' + text(vg_str) + '</wand>'

    @vector_graphics.setter
    def vector_graphics(self, vector_graphics):
        if vector_graphics is not None and not isinstance(vector_graphics,
                                                          string_type):
            raise TypeError('expected a string, not ' + repr(vector_graphics))
        elif vector_graphics is None:
            # Reset all vector graphic properties on drawing wand.
            library.DrawResetVectorGraphics(self.resource)
        else:
            vector_graphics = binary(vector_graphics)
            okay = library.DrawSetVectorGraphics(self.resource,
                                                 vector_graphics)
            if okay == 0:  # pragma: no cover
                raise ValueError("Vector graphic not understood.")

    def affine(self, matrix):
        """Adjusts the current affine transformation matrix with the specified
        affine transformation matrix. Note that the current affine transform is
        adjusted rather than replaced.

        .. sourcecode:: text

                                              | sx  rx  0 |
            | x', y', 1 |  =  | x, y, 1 |  *  | ry  sy  0 |
                                              | tx  ty  1 |

        :param matrix: a list of :class:`~numbers.Real` to define affine
                       matrix ``[sx, rx, ry, sy, tx, ty]``
        :type matrix: :class:`collections.abc.Sequence`

        .. versionadded:: 0.4.0

        """
        if not isinstance(matrix, abc.Sequence) or len(matrix) != 6:
            raise ValueError('matrix must be a list of size Real numbers')
        for idx, val in enumerate(matrix):
            if not isinstance(val, numbers.Real):
                raise TypeError('expecting numbers.Real in position #' +
                                repr(idx))
        amx = AffineMatrix(sx=matrix[0], rx=matrix[1],
                           ry=matrix[2], sy=matrix[3],
                           tx=matrix[4], ty=matrix[5])
        library.DrawAffine(self.resource, ctypes.byref(amx))

    def alpha(self, x=None, y=None, paint_method='undefined'):
        """Paints on the image's opacity channel in order to set effected pixels
        to transparent.

         To influence the opacity of pixels. The available methods are:

        - ``'undefined'``
        - ``'point'``
        - ``'replace'``
        - ``'floodfill'``
        - ``'filltoborder'``
        - ``'reset'``

        .. note::

            This method replaces :meth:`matte()` in ImageMagick version 7.
            An :class:`AttributeError` will be raised if attempting
            to call on a library without ``DrawAlpha`` support.

        .. versionadded:: 0.5.0

        """
        if library.DrawAlpha is None:
            raise AttributeError(
                'Method added with ImageMagick version 7. ' +
                'Please use `wand.drawing.Drawing.matte()\' instead.'
            )
        assertions.assert_real(x=x, y=y)
        assertions.string_in_list(PAINT_METHOD_TYPES,
                                  'wand.drawing.PAINT_METHOD_TYPES',
                                  paint_method=paint_method)
        op = PAINT_METHOD_TYPES.index(paint_method)
        library.DrawAlpha(self.resource, x, y, op)

    def arc(self, start, end, degree):
        """Draws a arc using the current :attr:`stroke_color`,
        :attr:`stroke_width`, and :attr:`fill_color`.

        :param start: (:class:`~numbers.Real`, :class:`numbers.Real`)
                      pair which represents starting x and y of the arc
        :type start: :class:`~collections.abc.Sequence`
        :param end: (:class:`~numbers.Real`, :class:`numbers.Real`)
                      pair which represents ending x and y of the arc
        :type end: :class:`~collections.abc.Sequence`
        :param degree: (:class:`~numbers.Real`, :class:`numbers.Real`)
                      pair which represents starting degree, and ending degree
        :type degree: :class:`~collections.abc.Sequence`

        .. versionadded:: 0.4.0

        """
        assertions.assert_coordinate(start=start, end=end, degree=degree)
        start_x, start_y = start
        end_x, end_y = end
        degree_start, degree_end = degree
        library.DrawArc(self.resource,
                        float(start_x), float(start_y),
                        float(end_x), float(end_y),
                        float(degree_start), float(degree_end))

    def bezier(self, points=None):
        """Draws a bezier curve through a set of points on the image, using
        the specified array of coordinates.

        At least four points should be given to complete a bezier path.
        The first & forth point being the start & end point, and the second
        & third point controlling the direction & curve.

        Example bezier on ``image`` ::

            with Drawing() as draw:
                points = [(40,10), # Start point
                          (20,50), # First control
                          (90,10), # Second control
                          (70,40)] # End point
                draw.stroke_color = Color('#000')
                draw.fill_color = Color('#fff')
                draw.bezier(points)
                draw.draw(image)

        :param points: list of x,y tuples
        :type points: :class:`list`

        .. versionadded:: 0.4.0

        """

        (points_l, points_p) = _list_to_point_info(points)
        library.DrawBezier(self.resource, points_l,
                           ctypes.cast(points_p, ctypes.POINTER(PointInfo)))

    def circle(self, origin, perimeter):
        """Draws a circle from ``origin`` to ``perimeter``

        :param origin: (:class:`~numbers.Real`, :class:`numbers.Real`)
                       pair which represents origin x and y of circle
        :type origin: :class:`collections.abc.Sequence`
        :param perimeter: (:class:`~numbers.Real`, :class:`numbers.Real`)
                       pair which represents perimeter x and y of circle
        :type perimeter: :class:`collections.abc.Sequence`

        .. versionadded:: 0.4.0

        """
        assertions.assert_coordinate(origin=origin, perimeter=perimeter)
        origin_x, origin_y = origin
        perimeter_x, perimeter_y = perimeter
        library.DrawCircle(self.resource, origin_x, origin_y,
                           perimeter_x, perimeter_y)

    def clear(self):
        library.ClearDrawingWand(self.resource)

    def clone(self):
        """Copies a drawing object.

        :returns: a duplication
        :rtype: :class:`Drawing`

        """
        return type(self)(drawing=self)

    def color(self, x=0.0, y=0.0, paint_method='undefined'):
        """Draws a color on the image using current fill color, starting
        at specified position & method.

        Available methods in :class:`wand.drawing.PAINT_METHOD_TYPES`:

        - ``'undefined'``
        - ``'point'``
        - ``'replace'``
        - ``'floodfill'``
        - ``'filltoborder'``
        - ``'reset'``

        .. versionadded:: 0.4.0

        """
        assertions.assert_real(x=x, y=y)
        assertions.string_in_list(PAINT_METHOD_TYPES,
                                  'wand.drawing.PAINT_METHOD_TYPES',
                                  paint_method=paint_method)
        op = PAINT_METHOD_TYPES.index(paint_method)
        library.DrawColor(self.resource, x, y, op)

    def comment(self, message=None):
        """Adds a comment to the vector stream.

        :param message: the comment to set.
        :type message: :class:`basestring`

        .. versionadded:: 0.4.0
        """
        if message is None:
            message = b''
        else:
            assertions.assert_string(message=message)
            message = binary(message)
        library.DrawComment(self.resource, message)

    def composite(self, operator, left, top, width, height, image):
        """Composites an image onto the current image, using the specified
        composition operator, specified position, and at the specified size.

        :param operator: the operator that affects how the composite
                         is applied to the image.  available values
                         can be found in the :const:`COMPOSITE_OPERATORS`
                         list
        :param type: :const:`COMPOSITE_OPERATORS`
        :param left: the column offset of the composited drawing source
        :type left: :class:`numbers.Real`
        :param top: the row offset of the composited drawing source
        :type top: :class:`numbers.Real`
        :param width: the total columns to include in the composited source
        :type width: :class:`numbers.Real`
        :param height: the total rows to include in the composited source
        :type height: :class:`numbers.Real`

        .. versionadded:: 0.4.0

        """
        assertions.assert_string(operator=operator)
        assertions.assert_real(left=left, top=top, width=width, height=height)
        assertions.string_in_list(COMPOSITE_OPERATORS,
                                  'wand.drawing.COMPOSITE_OPERATORS',
                                  operator=operator)
        op = COMPOSITE_OPERATORS.index(operator)
        library.DrawComposite(self.resource, op, left, top, width,
                              height, image.wand)
        self.raise_exception()

    def draw(self, image):
        """Renders the current drawing into the ``image``.  You can simply
        call :class:`Drawing` instance rather than calling this method.
        That means the following code which calls :class:`Drawing` object
        itself::

            drawing(image)

        is equivalent to the following code which calls :meth:`draw()` method::

            drawing.draw(image)

        :param image: the image to be drawn
        :type image: :class:`~wand.image.BaseImage`

        """
        if not isinstance(image, BaseImage):
            raise TypeError('image must be a wand.image.BaseImage instance,'
                            ' not ' + repr(image))
        if isinstance(image, SingleImage):
            previous = library.MagickGetIteratorIndex(image.container.wand)
            library.MagickSetIteratorIndex(image.container.wand, image.index)
            res = library.MagickDrawImage(image.container.wand, self.resource)
            library.MagickSetIteratorIndex(image.container.wand, previous)
        else:
            res = library.MagickDrawImage(image.wand, self.resource)
        if not res:
            self.raise_exception()

    def ellipse(self, origin, radius, rotation=None):
        """Draws a ellipse at ``origin`` with independent x & y ``radius``.
        Ellipse can be partial by setting start & end ``rotation``.

        :param origin: (:class:`~numbers.Real`, :class:`numbers.Real`)
                       pair which represents origin x and y of circle
        :type origin: :class:`collections.abc.Sequence`
        :param radius: (:class:`~numbers.Real`, :class:`numbers.Real`)
                       pair which represents radius x and radius y of circle
        :type radius: :class:`collections.abc.Sequence`
        :param rotation: (:class:`~numbers.Real`, :class:`numbers.Real`)
                         pair which represents start and end of ellipse.
                         Default (0,360)
        :type rotation: :class:`collections.abc.Sequence`

        .. versionadded:: 0.4.0

        """
        if rotation is None:
            rotation = (0, 360)
        assertions.assert_coordinate(origin=origin, radius=radius,
                                     rotation=rotation)
        origin_x, origin_y = origin
        radius_x, radius_y = radius
        rotation_start, rotation_end = rotation
        library.DrawEllipse(self.resource,
                            origin_x, origin_y,
                            radius_x, radius_y,
                            rotation_start, rotation_end)

    def get_font_metrics(self, image, text, multiline=False):
        """Queries font metrics from the given ``text``.

        :param image: the image to be drawn
        :type image: :class:`~wand.image.BaseImage`
        :param text: the text string for get font metrics.
        :type text: :class:`basestring`
        :param multiline: text is multiline or not
        :type multiline: `boolean`

        """
        if not isinstance(image, BaseImage):
            raise TypeError('image must be a wand.image.BaseImage instance,'
                            ' not ' + repr(image))
        assertions.assert_string(text=text)
        if multiline:
            font_metrics_f = library.MagickQueryMultilineFontMetrics
        else:
            font_metrics_f = library.MagickQueryFontMetrics
        if isinstance(text, text_type):
            if self.text_encoding:
                text = text.encode(self.text_encoding)
            else:
                text = binary(text)
        result = font_metrics_f(image.wand, self.resource, text)
        if not result:  # pragma: no cover
            # Error on drawing context
            self.raise_exception()
            # Or error on image canvas
            image.raise_exception()
            # Generate a generic error if ImageMagick couldn't emit one.
            raise ValueError('Unable to render text with current font.')
        args = [result[i] for i in xrange(13)]
        result = library.MagickRelinquishMemory(result)
        return FontMetrics(*args)

    def line(self, start, end):
        """Draws a line ``start`` to ``end``.

        :param start: (:class:`~numbers.Integral`, :class:`numbers.Integral`)
                      pair which represents starting x and y of the line
        :type start: :class:`collections.abc.Sequence`
        :param end: (:class:`~numbers.Integral`, :class:`numbers.Integral`)
                    pair which represents ending x and y of the line
        :type end: :class:`collections.abc.Sequence`

        """
        start_x, start_y = start
        end_x, end_y = end
        library.DrawLine(self.resource,
                         int(start_x), int(start_y),
                         int(end_x), int(end_y))

    def matte(self, x=0.0, y=0.0, paint_method='undefined'):
        """Paints on the image's opacity channel in order to set effected pixels
        to transparent.

         To influence the opacity of pixels. The available methods are:

        - ``'undefined'``
        - ``'point'``
        - ``'replace'``
        - ``'floodfill'``
        - ``'filltoborder'``
        - ``'reset'``

        .. note::

            This method has been replace by :meth:`alpha()` in ImageMagick
            version 7. An :class:`AttributeError` will be raised if attempting
            to call on a library without ``DrawMatte`` support.

        .. versionadded:: 0.4.0

        """
        if library.DrawMatte is None:
            raise AttributeError(
                'Method removed from ImageMagick version. ' +
                'Please use `wand.drawing.Drawing.alpha()\' instead.'
            )
        assertions.assert_real(x=x, y=y)
        assertions.string_in_list(PAINT_METHOD_TYPES,
                                  'wand.drawing.PAINT_METHOD_TYPES',
                                  paint_method=paint_method)
        op = PAINT_METHOD_TYPES.index(paint_method)
        library.DrawMatte(self.resource, x, y, op)

    def path_close(self):
        """Adds a path element to the current path which closes
        the current subpath by drawing a straight line from the current point
        to the current subpath's most recent starting point.

        .. versionadded:: 0.4.0

        """
        library.DrawPathClose(self.resource)
        return self

    def path_curve(self, to=None, controls=None, smooth=False, relative=False):
        """Draws a cubic Bezier curve from the current point to given ``to``
        (x,y) coordinate using ``controls`` points at the beginning and
        the end of the curve.
        If ``smooth`` is set to True, only one ``controls`` is expected
        and the previous control is used, else two pair of coordinates are
        expected to define the control points. The ``to`` coordinate then
        becomes the new current point.

        :param to: (:class:`~numbers.Real`, :class:`numbers.Real`)
                   pair which represents coordinates to draw to
        :type to: :class:`collections.abc.Sequence`
        :param controls: (:class:`~numbers.Real`, :class:`numbers.Real`)
                         coordinate to used to influence curve
        :type controls: :class:`collections.abc.Sequence`
        :param smooth: :class:`bool` assume last defined control coordinate
        :type smooth: :class:`bool`
        :param relative: treat given coordinates as relative to current point
        :type relative: :class:`bool`

        .. versionadded:: 0.4.0

        """
        if to is None:
            raise TypeError('to is missing')
        if controls is None:
            raise TypeError('controls is missing')
        x, y = to
        if smooth:
            x2, y2 = controls
        else:
            (x1, y1), (x2, y2) = controls

        if smooth:
            if relative:
                library.DrawPathCurveToSmoothRelative(self.resource,
                                                      x2, y2, x, y)
            else:
                library.DrawPathCurveToSmoothAbsolute(self.resource,
                                                      x2, y2, x, y)
        else:
            if relative:
                library.DrawPathCurveToRelative(self.resource,
                                                x1, y1, x2, y2, x, y)
            else:
                library.DrawPathCurveToAbsolute(self.resource,
                                                x1, y1, x2, y2, x, y)
        return self

    def path_curve_to_quadratic_bezier(self, to=None, control=None,
                                       smooth=False, relative=False):
        """Draws a quadratic Bezier curve from the current point to given
        ``to`` coordinate. The control point is assumed to be the reflection of
        the control point on the previous command if ``smooth`` is True, else a
        pair of ``control`` coordinates must be given. Each coordinates can be
        relative, or absolute, to the current point by setting the ``relative``
        flag. The ``to`` coordinate then becomes the new current point, and the
        ``control`` coordinate will be assumed when called again
        when ``smooth`` is set to true.

        :param to: (:class:`~numbers.Real`, :class:`numbers.Real`)
                   pair which represents coordinates to draw to
        :type to: :class:`collections.abc.Sequence`
        :param control: (:class:`~numbers.Real`, :class:`numbers.Real`)
                        coordinate to used to influence curve
        :type control: :class:`collections.abc.Sequence`
        :param smooth: assume last defined control coordinate
        :type smooth: :class:`bool`
        :param relative: treat given coordinates as relative to current point
        :type relative: :class:`bool`

        .. versionadded:: 0.4.0

        """
        if to is None:
            raise TypeError('to is missing')
        x, y = to

        if smooth:
            if relative:
                library.DrawPathCurveToQuadraticBezierSmoothRelative(
                    self.resource, float(x), float(y)
                )
            else:
                library.DrawPathCurveToQuadraticBezierSmoothAbsolute(
                    self.resource, float(x), float(y)
                )
        else:
            if control is None:
                raise TypeError('control is missing')
            x1, y1 = control
            if relative:
                library.DrawPathCurveToQuadraticBezierRelative(self.resource,
                                                               float(x1),
                                                               float(y1),
                                                               float(x),
                                                               float(y))
            else:
                library.DrawPathCurveToQuadraticBezierAbsolute(self.resource,
                                                               float(x1),
                                                               float(y1),
                                                               float(x),
                                                               float(y))
        return self

    def path_elliptic_arc(self, to=None, radius=None, rotation=0.0,
                          large_arc=False, clockwise=False, relative=False):
        """Draws an elliptical arc from the current point to given ``to``
        coordinates. The ``to`` coordinates can be relative, or absolute,
        to the current point by setting the ``relative`` flag.
        The size and orientation of the ellipse are defined by
        two radii (rx, ry) in ``radius`` and an ``rotation`` parameters,
        which indicates how the ellipse as a whole is
        rotated relative to the current coordinate system. The center of the
        ellipse is calculated automagically to satisfy the constraints imposed
        by the other parameters. ``large_arc`` and ``clockwise`` contribute to
        the automatic calculations and help determine how the arc is drawn.
        If ``large_arc`` is True then draw the larger of the available arcs.
        If ``clockwise`` is true, then draw the arc matching a clock-wise
        rotation.

        :param to: (:class:`~numbers.Real`, :class:`numbers.Real`)
                   pair which represents coordinates to draw to
        :type to: :class:`collections.abc.Sequence`
        :param radius: (:class:`~numbers.Real`, :class:`numbers.Real`)
                       pair which represents the radii of the ellipse to draw
        :type radius: :class:`collections.abc.Sequence`
        :param rotate: degree to rotate ellipse on x-axis
        :type rotate: :class:`~numbers.Real`
        :param large_arc: draw largest available arc
        :type large_arc: :class:`bool`
        :param clockwise: draw arc path clockwise from start to target
        :type clockwise: :class:`bool`
        :param relative: treat given coordinates as relative to current point
        :type relative: :class:`bool`

        .. versionadded:: 0.4.0

        """
        if to is None:
            raise TypeError('to is missing')
        if radius is None:
            raise TypeError('radius is missing')
        x, y = to
        rx, ry = radius
        if relative:
            library.DrawPathEllipticArcRelative(self.resource,
                                                float(rx), float(ry),
                                                float(rotation),
                                                bool(large_arc),
                                                bool(clockwise),
                                                float(x), float(y))
        else:
            library.DrawPathEllipticArcAbsolute(self.resource,
                                                float(rx), float(ry),
                                                float(rotation),
                                                bool(large_arc),
                                                bool(clockwise),
                                                float(x), float(y))
        return self

    def path_finish(self):
        """Terminates the current path.

        .. versionadded:: 0.4.0

        """
        library.DrawPathFinish(self.resource)
        return self

    def path_horizontal_line(self, x=None, relative=False):
        """Draws a horizontal line path from the current point to the target
        point. Given ``x`` parameter can be relative, or absolute, to the
        current point by setting the ``relative`` flag. The target point then
        becomes the new current point.

        :param x: :class:`~numbers.Real`
                      x-axis point to draw to.
        :type x: :class:`~numbers.Real`
        :param relative: :class:`bool`
                    treat given point as relative to current point
        :type relative: :class:`bool`

        .. versionadded:: 0.4.0

        """
        assertions.assert_real(x=x)
        if relative:
            library.DrawPathLineToHorizontalRelative(self.resource, x)
        else:
            library.DrawPathLineToHorizontalAbsolute(self.resource, x)
        return self

    def path_line(self, to=None, relative=False):
        """Draws a line path from the current point to the given ``to``
        coordinate. The ``to`` coordinates can be relative, or absolute, to the
        current point by setting the ``relative`` flag. The coordinate then
        becomes the new current point.

        :param to: (:class:`~numbers.Real`, :class:`numbers.Real`)
                      pair which represents coordinates to draw to.
        :type to: :class:`collections.abc.Sequence`
        :param relative: :class:`bool`
                    treat given coordinates as relative to current point
        :type relative: :class:`bool`

        .. versionadded:: 0.4.0

        """
        assertions.assert_coordinate(to=to)
        x, y = to
        if relative:
            library.DrawPathLineToRelative(self.resource, x, y)
        else:
            library.DrawPathLineToAbsolute(self.resource, x, y)
        return self

    def path_move(self, to=None, relative=False):
        """Starts a new sub-path at the given coordinates. Given ``to``
        parameter can be relative, or absolute, by setting the ``relative``
        flag.

        :param to: (:class:`~numbers.Real`, :class:`numbers.Real`)
                      pair which represents coordinates to draw to.
        :type to: :class:`collections.abc.Sequence`
        :param relative: :class:`bool`
                    treat given coordinates as relative to current point
        :type relative: :class:`bool`

        .. versionadded:: 0.4.0

        """
        assertions.assert_coordinate(to=to)
        x, y = to
        if relative:
            library.DrawPathMoveToRelative(self.resource, x, y)
        else:
            library.DrawPathMoveToAbsolute(self.resource, x, y)
        return self

    def path_start(self):
        """Declares the start of a path drawing list which is terminated by a
        matching :meth:`path_finish()` command. All other `path_*` commands
        must be enclosed between a :meth:`path_start()` and a
        :meth:`path_finish()` command. This is because path drawing commands
        are subordinate commands and they do not function by themselves.

        .. versionadded:: 0.4.0

        """
        library.DrawPathStart(self.resource)
        return self

    def path_vertical_line(self, y=None, relative=False):
        """Draws a vertical line path from the current point to the target
        point. Given ``y`` parameter can be relative, or absolute, to the
        current point by setting the ``relative`` flag. The target point then
        becomes the new current point.

        :param y: :class:`~numbers.Real`
                      y-axis point to draw to.
        :type y: :class:`~numbers.Real`
        :param relative: :class:`bool`
                    treat given point as relative to current point
        :type relative: :class:`bool`

        .. versionadded:: 0.4.0

        """
        assertions.assert_real(y=y)
        if relative:
            library.DrawPathLineToVerticalRelative(self.resource, y)
        else:
            library.DrawPathLineToVerticalAbsolute(self.resource, y)
        return self

    def polygon(self, points=None):
        """Draws a polygon using the current :attr:`stroke_color`,
        :attr:`stroke_width`, and :attr:`fill_color`, using the specified
        array of coordinates.

        Example polygon on ``image`` ::

            with Drawing() as draw:
                points = [(40,10), (20,50), (90,10), (70,40)]
                draw.polygon(points)
                draw.draw(image)

        :param points: list of x,y tuples
        :type points: :class:`list`

        .. versionadded:: 0.4.0

        """

        (points_l, points_p) = _list_to_point_info(points)
        library.DrawPolygon(self.resource, points_l,
                            ctypes.cast(points_p, ctypes.POINTER(PointInfo)))

    def polyline(self, points=None):
        """Draws a polyline using the current :attr:`stroke_color`,
        :attr:`stroke_width`, and :attr:`fill_color`, using the specified
        array of coordinates.

        Identical to :class:`~wand.drawing.Drawing.polygon`, but without closed
        stroke line.

        :param points: list of x,y tuples
        :type points: :class:`list`

        .. versionadded:: 0.4.0

        """

        (points_l, points_p) = _list_to_point_info(points)
        library.DrawPolyline(self.resource, points_l,
                             ctypes.cast(points_p, ctypes.POINTER(PointInfo)))

    def point(self, x, y):
        """Draws a point at given ``x`` and ``y``

        :param x: :class:`~numbers.Real` x of point
        :type x: :class:`~numbers.Real`
        :param y: :class:`~numbers.Real` y of point
        :type y: :class:`~numbers.Real`

        .. versionadded:: 0.4.0

        """
        assertions.assert_real(x=x, y=y)
        library.DrawPoint(self.resource, x, y)

    def pop(self):
        """Pop destroys the current tip of the drawing context stack,
        and restores the parent style context.
        See :meth:`push()` method for an example.

        .. note::

            Popping the graphical context stack will not erase,
            or alter, any previously executed drawing commands.

        :returns: success of pop operation.
        :rtype: :class:`bool`

        .. versionadded:: 0.4.0

        """
        ok = bool(library.PopDrawingWand(self.resource))
        if not ok:  # pragma: no cover
            self.raise_exception()
        return ok

    def pop_clip_path(self):
        """Terminates a clip path definition.

        .. versionadded:: 0.4.0

        """
        library.DrawPopClipPath(self.resource)

    def pop_defs(self):
        """Terminates a definition list.

        .. versionadded:: 0.4.0

        """
        library.DrawPopDefs(self.resource)

    def pop_pattern(self):
        """Terminates a pattern definition.

        .. versionadded:: 0.4.0

        """
        library.DrawPopPattern(self.resource)

    def push(self):
        """Grows the current drawing context stack by one, and inherits
        the previous style attributes. Use :class:`Drawing.pop` to return
        to restore previous style attributes.

        This is useful for drawing shapes with diffrent styles
        without repeatedly setting the similar
        :meth:`fill_color <wand.drawing.Drawing.fill_color>` &
        :meth:`stroke_color <wand.drawing.Drawing.stroke_color>` properties.

        For example::

            with Drawing() as ctx:
                ctx.fill_color = Color('GREEN')
                ctx.stroke_color = Color('ORANGE')
                ctx.push()
                ctx.fill_color = Color('RED')
                ctx.text(x1, y1, 'this is RED with ORANGE outline')
                ctx.push()
                ctx.stroke_color = Color('BLACK')
                ctx.text(x2, y2, 'this is RED with BLACK outline')
                ctx.pop()
                ctx.pop()
                ctx.text(x3, y3, 'this is GREEN with ORANGE outline')

        Which translate to the following MVG::

            push graphic-context
                fill "GREEN"
                stroke "ORANGE"
                push graphic-context
                    fill "RED"
                    text x1,y1 "this is RED with ORANGE outline"
                    push graphic-context
                        stroke "BLACK"
                        text x2,y2 "this is RED with BLACK outline"
                    pop graphic-context
                pop graphic-context
                text x3,y3 "this is GREEN with ORANGE outline"
            pop graphic-context

        .. note::

            Pushing graphical context does not reset any previously
            drawn artifacts.

        :returns: success of push operation.
        :rtype: :class:`bool`

        .. versionadded:: 0.4.0

        """
        ok = bool(library.PushDrawingWand(self.resource))
        if not ok:
            self.raise_exception()
        return ok

    def push_clip_path(self, clip_mask_id):
        """Starts a clip path definition which is comprised of any number of
        drawing commands and terminated by a :class:`Drawing.pop_clip_path`
        command.

        :param clip_mask_id: string identifier to associate with the clip path.
        :type clip_mask_id: :class:`basestring`

        .. versionadded:: 0.4.0

        """
        library.DrawPushClipPath(self.resource, binary(clip_mask_id))

    def push_defs(self):
        """Indicates that commands up to a terminating :class:`Drawing.pop_defs`
        command create named elements (e.g. clip-paths, textures, etc.) which
        may safely be processed earlier for the sake of efficiency.

        .. versionadded:: 0.4.0

        """
        library.DrawPushDefs(self.resource)

    def push_pattern(self, pattern_id, left, top, width, height):
        """Indicates that subsequent commands up to a
        :class:`Drawing.pop_pattern` command comprise the definition of a named
        pattern. The pattern space is assigned top left corner coordinates, a
        width and height, and becomes its own drawing space. Anything which can
        be drawn may be used in a pattern definition.
        Named patterns may be used as stroke or brush definitions.

        :param pattern_id: a unique identifier for the pattern.
        :type pattern_id: :class:`basestring`
        :param left: x ordinate of top left corner.
        :type left: :class:`numbers.Real`
        :param top: y ordinate of top left corner.
        :type top: :class:`numbers.Real`
        :param width: width of pattern space.
        :type width: :class:`numbers.Real`
        :param height: height of pattern space.
        :type height: :class:`numbers.Real`
        :returns: success of push operation
        :rtype: :class:`bool`

        .. versionadded:: 0.4.0

        """
        assertions.assert_string(pattern_id=pattern_id)
        assertions.assert_real(left=left, top=top, width=width, height=height)
        okay = library.DrawPushPattern(self.resource, binary(pattern_id),
                                       left, top,
                                       width, height)
        if not okay:  # pragma: no cover
            self.raise_exception()
        return bool(okay)

    def rectangle(self, left=None, top=None, right=None, bottom=None,
                  width=None, height=None, radius=None, xradius=None,
                  yradius=None):
        """Draws a rectangle using the current :attr:`stroke_color`,
        :attr:`stroke_width`, and :attr:`fill_color`.

        .. sourcecode:: text

           +--------------------------------------------------+
           |              ^                         ^         |
           |              |                         |         |
           |             top                        |         |
           |              |                         |         |
           |              v                         |         |
           | <-- left --> +-------------------+  bottom       |
           |              |             ^     |     |         |
           |              | <-- width --|---> |     |         |
           |              |           height  |     |         |
           |              |             |     |     |         |
           |              |             v     |     |         |
           |              +-------------------+     v         |
           | <--------------- right ---------->               |
           +--------------------------------------------------+

        :param left: x-offset of the rectangle to draw
        :type left: :class:`numbers.Real`
        :param top: y-offset of the rectangle to draw
        :type top: :class:`numbers.Real`
        :param right: second x-offset of the rectangle to draw.
                      this parameter and ``width`` parameter are exclusive
                      each other
        :type right: :class:`numbers.Real`
        :param bottom: second y-offset of the rectangle to draw.
                       this parameter and ``height`` parameter are exclusive
                       each other
        :type bottom: :class:`numbers.Real`
        :param width: the :attr:`width` of the rectangle to draw.
                      this parameter and ``right`` parameter are exclusive
                      each other
        :type width: :class:`numbers.Real`
        :param height: the :attr:`height` of the rectangle to draw.
                       this parameter and ``bottom`` parameter are exclusive
                       each other
        :type height: :class:`numbers.Real`
        :param radius: the corner rounding. this is a short-cut for setting
                       both :attr:`xradius`, and :attr:`yradius`
        :type radius: :class:`numbers.Real`
        :param xradius: the :attr:`xradius` corner in horizontal direction.
        :type xradius: :class:`numbers.Real`
        :param yradius: the :attr:`yradius` corner in vertical direction.
        :type yradius: :class:`numbers.Real`

        .. versionadded:: 0.3.6

        .. versionchanged:: 0.4.0
           Radius keywords added to create rounded rectangle.

        """
        if left is None:
            raise TypeError('left is missing')
        elif top is None:
            raise TypeError('top is missing')
        elif right is None and width is None:
            raise TypeError('right/width is missing')
        elif bottom is None and height is None:
            raise TypeError('bottom/height is missing')
        elif not (right is None or width is None):
            raise TypeError('parameters right and width are exclusive each '
                            'other; use one at a time')
        elif not (bottom is None or height is None):
            raise TypeError('parameters bottom and height are exclusive each '
                            'other; use one at a time')
        elif not isinstance(left, numbers.Real):
            raise TypeError('left must be numbers.Real, not ' + repr(left))
        elif not isinstance(top, numbers.Real):
            raise TypeError('top must be numbers.Real, not ' + repr(top))
        elif not (right is None or isinstance(right, numbers.Real)):
            raise TypeError('right must be numbers.Real, not ' + repr(right))
        elif not (bottom is None or isinstance(bottom, numbers.Real)):
            raise TypeError('bottom must be numbers.Real, not ' + repr(bottom))
        elif not (width is None or isinstance(width, numbers.Real)):
            raise TypeError('width must be numbers.Real, not ' + repr(width))
        elif not (height is None or isinstance(height, numbers.Real)):
            raise TypeError('height must be numbers.Real, not ' + repr(height))
        if right is None:
            if width < 0:
                raise ValueError('width must be positive, not ' + repr(width))
            right = left + width
        elif right < left:
            raise ValueError('right must be more than left ({0!r}), '
                             'not {1!r})'.format(left, right))
        if bottom is None:
            if height < 0:
                raise ValueError('height must be positive, not ' +
                                 repr(height))
            bottom = top + height
        elif bottom < top:
            raise ValueError('bottom must be more than top ({0!r}), '
                             'not {1!r})'.format(top, bottom))
        if radius is not None:
            xradius = yradius = radius
        if xradius is not None or yradius is not None:
            if xradius is None:
                xradius = 0.0
            if yradius is None:
                yradius = 0.0
            assertions.assert_real(xradius=xradius, yradius=yradius)
            library.DrawRoundRectangle(self.resource, left, top, right, bottom,
                                       xradius, yradius)
        else:
            library.DrawRectangle(self.resource, left, top, right, bottom)
        self.raise_exception()

    def rotate(self, degree=0.0):
        """Applies the specified rotation to the current coordinate space.

        :param degree: degree to rotate
        :type degree: :class:`~numbers.Real`

        .. versionadded:: 0.4.0

        """
        assertions.assert_real(degree=degree)
        library.DrawRotate(self.resource, degree)

    def scale(self, x=1.0, y=1.0):
        """
        Adjusts the scaling factor to apply in the horizontal and vertical
        directions to the current coordinate space.

        :param x: Horizontal scale factor. Default `1.0`
        :type x: :class:`~numbers.Real`
        :param y: Vertical scale factor. Default `1.0`
        :type y: :class:`~numbers.Real`

        .. versionadded:: 0.4.0

        """
        assertions.assert_real(x=x, y=y)
        library.DrawScale(self.resource, x, y)

    def set_fill_pattern_url(self, url):
        """Sets the URL to use as a fill pattern for filling objects. Only local
        URLs ("#identifier") are supported at this time. These local URLs are
        normally created by defining a named fill pattern with
        Drawing.push_pattern & Drawing.pop_pattern.

        :param url: URL to use to obtain fill pattern.
        :type url: :class:`basestring`

        .. versionadded:: 0.4.0

        """
        assertions.assert_string(url=url)
        if url[0] != '#':
            raise ValueError('value not a relative URL, '
                             'expecting "#identifier"')
        library.DrawSetFillPatternURL(self.resource, binary(url))
        self.raise_exception()

    def set_stroke_pattern_url(self, url):
        """Sets the pattern used for stroking object outlines. Only local
        URLs ("#identifier") are supported at this time. These local URLs are
        normally created by defining a named stroke pattern with
        Drawing.push_pattern & Drawing.pop_pattern.

        :param url: URL to use to obtain stroke pattern.
        :type url: :class:`basestring`

        .. versionadded:: 0.4.0

        """
        assertions.assert_string(url=url)
        if url[0] != '#':
            raise ValueError('value not a relative URL, '
                             'expecting "#identifier"')
        library.DrawSetStrokePatternURL(self.resource, binary(url))
        self.raise_exception()

    def skew(self, x=None, y=None):
        """Skews the current coordinate system in the horizontal direction if
        ``x`` is given, and vertical direction if ``y`` is given.

        :param x: Skew horizontal direction
        :type x: :class:`~numbers.Real`
        :param y: Skew vertical direction
        :type y: :class:`~numbers.Real`

        .. versionadded:: 0.4.0

        """
        if x is not None:
            assertions.assert_real(x=x)
            library.DrawSkewX(self.resource, x)
        if y is not None:
            assertions.assert_real(y=y)
            library.DrawSkewY(self.resource, y)

    def text(self, x, y, body):
        """Writes a text ``body`` into (``x``, ``y``).

        :param x: the left offset where to start writing a text
        :type x: :class:`numbers.Integral`
        :param y: the baseline where to start writing text
        :type y: :class:`numbers.Integral`
        :param body: the body string to write
        :type body: :class:`basestring`

        """
        assertions.assert_unsigned_integer(x=x, y=y)
        assertions.assert_string(body=body)
        if not body:
            raise ValueError('body string cannot be empty')
        if isinstance(body, text_type):
            # According to ImageMagick C API docs, we can use only UTF-8
            # at this time, so we do hardcoding here.
            # http://imagemagick.org/api/drawing-wand.php#DrawSetTextEncoding
            if not self.text_encoding:
                self.text_encoding = 'UTF-8'
            body = body.encode(self.text_encoding)
        body_p = ctypes.create_string_buffer(body)
        library.DrawAnnotation(
            self.resource, x, y,
            ctypes.cast(body_p, ctypes.POINTER(ctypes.c_ubyte))
        )

    def translate(self, x=0.0, y=0.0):
        """Applies a translation to the current coordinate system which moves
        the coordinate system origin to the specified coordinate.

        :param x: Skew horizontal direction
        :type x: :class:`~numbers.Real`
        :param y: Skew vertical direction
        :type y: :class:`~numbers.Real`

        .. versionadded:: 0.4.0
        """
        assertions.assert_real(x=x, y=y)
        library.DrawTranslate(self.resource, x, y)

    def viewbox(self, left, top, right, bottom):
        """Viewbox sets the overall canvas size to be recorded with the drawing
        vector data. Usually this will be specified using the same size as the
        canvas image. When the vector data is saved to SVG or MVG formats, the
        viewbox is use to specify the size of the canvas image that a viewer
        will render the vector data on.

        :param left: the left most point of the viewbox.
        :type left: :class:`~numbers.Integral`
        :param top: the top most point of the viewbox.
        :type top: :class:`~numbers.Integral`
        :param right: the right most point of the viewbox.
        :type right: :class:`~numbers.Integral`
        :param bottom: the bottom most point of the viewbox.
        :type bottom: :class:`~numbers.Integral`

        .. versionadded:: 0.4.0

        """
        assertions.assert_integer(left=left, top=top,
                                  right=right, bottom=bottom)
        library.DrawSetViewbox(self.resource, left, top, right, bottom)

    def __call__(self, image):
        return self.draw(image)


def _list_to_point_info(points):
    """
    Helper method to convert a list of tuples to ``const * PointInfo``

    :param points: a list of tuples
    :type points: `list`
    :returns: tuple of point length and c_double array
    :rtype: `tuple`
    :raises: `TypeError`

    .. versionadded:: 0.4.0

    """
    if not isinstance(points, list):
        raise TypeError('points must be a list, not ' + repr(points))
    point_length = len(points)
    tuple_size = 2
    point_info_size = point_length * tuple_size
    # Allocate sequence of memory
    point_info = (ctypes.c_double * point_info_size)()
    for double_index in xrange(point_info_size):
        tuple_index = double_index // tuple_size
        tuple_offset = double_index % tuple_size
        point_info[double_index] = ctypes.c_double(
            points[tuple_index][tuple_offset]
        )
    return (point_length, point_info)
