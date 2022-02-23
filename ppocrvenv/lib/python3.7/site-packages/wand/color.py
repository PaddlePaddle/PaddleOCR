""":mod:`wand.color` --- Colors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 0.1.2

"""
import ctypes
import numbers

from .api import library
from .cdefs.structures import MagickPixelPacket, PixelInfo
from .compat import binary, text
from .resource import Resource
from .version import MAGICK_VERSION_NUMBER, MAGICK_HDRI, QUANTUM_DEPTH

__all__ = 'Color', 'scale_quantum_to_int8'


class Color(Resource):
    """Color value.

    Unlike any other objects in Wand, its resource management can be
    implicit when it used outside of :keyword:`with` block. In these case,
    its resource are allocated for every operation which requires a resource
    and destroyed immediately. Of course it is inefficient when the
    operations are much, so to avoid it, you should use color objects
    inside of :keyword:`with` block explicitly e.g.::

        red_count = 0
        with Color('#f00') as red:
            with Image(filename='image.png') as img:
                for row in img:
                    for col in row:
                        if col == red:
                            red_count += 1

    :param string: a color name string e.g. ``'rgb(255, 255, 255)'``,
                   ``'#fff'``, ``'white'``. see `ImageMagick Color Names`_
                   doc also
    :type string: :class:`basestring`

    .. versionchanged:: 0.3.0
       :class:`Color` objects become hashable.

    .. versionchanged:: 0.5.1
       Color channel properties can now be set.

    .. versionchanged:: 0.5.1
       Added :attr:`cyan`, :attr:`magenta`, :attr:`yellow`, & :attr:`black`
       properties for CMYK :class:`Color` instances.

    .. versionchanged:: 0.5.1
       Method :meth:`Color.from_hsl()` can create a RGB color from ``hue``,
       ``saturation``, & ``lightness`` values.

    .. seealso::

       `ImageMagick Color Names`_
          The color can then be given as a color name (there is a limited
          but large set of these; see below) or it can be given as a set
          of numbers (in decimal or hexadecimal), each corresponding to
          a channel in an RGB or RGBA color model. HSL, HSLA, HSB, HSBA,
          CMYK, or CMYKA color models may also be specified. These topics
          are briefly described in the sections below.

    .. _ImageMagick Color Names: http://www.imagemagick.org/script/color.php

    .. describe:: == (other)

       Equality operator.

       :param other: a color another one
       :type color: :class:`Color`
       :returns: ``True`` only if two images equal.
       :rtype: :class:`bool`

    """

    #: (:class:`bool`) Whether the color has changed or not.
    dirty = None

    c_is_resource = library.IsPixelWand
    c_destroy_resource = library.DestroyPixelWand
    c_get_exception = library.PixelGetException
    c_clear_exception = library.PixelClearException

    __slots__ = 'raw', 'c_resource', 'allocated'

    def __init__(self, string=None, raw=None):
        if (string is None and raw is None or
                string is not None and raw is not None):
            raise TypeError('expected one argument')

        # MagickPixelPacket has been deprecated, use PixelInfo
        self.use_pixel = MAGICK_VERSION_NUMBER >= 0x700
        self.dirty = False
        self.allocated = 0
        if raw is None:
            if self.use_pixel:  # pragma: no cover
                self.raw = ctypes.create_string_buffer(
                    ctypes.sizeof(PixelInfo)
                )
            else:
                self.raw = ctypes.create_string_buffer(
                    ctypes.sizeof(MagickPixelPacket)
                )
            with self:
                # Create color from string.
                ok = library.PixelSetColor(self.resource, binary(string))
                if not ok:
                    # Could not understand color-input. Try sending
                    # ImageMagick's exception.
                    self.raise_exception()
                    # That might be only a warning. Try a more generic message.
                    msg = 'Unrecognized color string "{0}"'.format(string)
                    raise ValueError(msg)
                # Copy color value to structure buffer for future read.
                library.PixelGetMagickColor(self.resource, self.raw)
        else:
            self.raw = raw

    def __getinitargs__(self):
        return self.string, None

    def __enter__(self):
        if self.allocated < 1:
            with self.allocate():
                # Initialize resource.
                self.resource = library.NewPixelWand()
                # Restore color value from structure buffer.
                if self.use_pixel:  # pragma: no cover
                    library.PixelSetPixelColor(self.resource, self.raw)
                else:
                    library.PixelSetMagickColor(self.resource, self.raw)
            self.allocated = 1
        else:
            self.allocated += 1
        return Resource.__enter__(self)

    def __exit__(self, type, value, traceback):
        self.allocated -= 1
        if self.dirty:
            library.PixelGetMagickColor(self.resource, self.raw)
            self.dirty = False
        if self.allocated < 1:
            Resource.__exit__(self, type, value, traceback)

    def __eq__(self, other):
        if not isinstance(other, Color):
            return False
        with self as this:
            with other:
                return self.c_equals(this.resource, other.resource)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        if self.alpha:
            return hash(self.normalized_string)
        return hash(None)

    def __str__(self):
        return self.string

    def __repr__(self):
        c = type(self)
        return '{0}.{1}({2!r})'.format(c.__module__, c.__name__, self.string)

    def _assert_double(self, subject):
        """Ensure the given ``subject`` is a float type, and value between
        0.0 & 1.0.

        :param subject: value to assert as a valid double.
        :type subject: :class:`numbers.Real`
        :raises ValueError: if the subject is not between 0.0 and 1.0
        :raises TypeError: if the subject is not a float-point number.

        ..versionadded:: 0.5.1
        """
        if not isinstance(subject, numbers.Real):
            raise TypeError('Expecting a float-point real number, not ' +
                            repr(subject))
        if subject < 0.0 or subject > 1.0:
            raise ValueError('Expecting a real number between 0.0 & 1.0, not' +
                             repr(subject))

    def _assert_int8(self, subject):
        """Ensure the given ``subject`` is a integer type, and value between
        0 & 255.

        :param subject: value to assert as a valid number.
        :type subject: :class:`numbers.Integral`
        :raises ValueError: if the subject is not between 0 and 255
        :raises TypeError: if the subject is not a Integral number.

        ..versionadded:: 0.5.1
        """
        if not isinstance(subject, numbers.Integral):
            raise TypeError('Expecting an integer number, not ' +
                            repr(subject))
        if subject < 0 or subject > 255:
            raise ValueError('Expecting a real number between 0 & 255, not' +
                             repr(subject))

    def _assert_quantum(self, subject):
        """Ensure the given ``subject`` is a number, and value between
        0.0 & QuantumRange.

        The QuantumRange is the max value based on the QuantumDepth of the
        ImageMagick library (i.e. Q16).

        :param subject: value to assert as a valid double.
        :type subject: :class:`numbers.Number`
        :raises ValueError: if the subject is not between 0 and QuantumRange
        :raises TypeError: if the subject is not a number.

        ..versionadded:: 0.5.1
        """
        quantum_range = {
            8: 255.0,
            16: 65535.0,
            32: 4294967295.0,
            64: 18446744073709551615.0
        }
        if not isinstance(subject, numbers.Number):
            raise TypeError('Expecting a number, not ' + repr(subject))
        if subject < 0.0 or subject > quantum_range[QUANTUM_DEPTH]:
            message = 'Expecting a number between 0 & {0}, not {1}'
            raise ValueError(message.format(quantum_range[QUANTUM_DEPTH],
                                            repr(subject)))

    def _repr_html_(self):
        html = """
        <span style="background-color:#{red:02X}{green:02X}{blue:02X};
                     display:inline-block;
                     line-height:1em;
                     width:1em;">&nbsp;</span>
        <strong>#{red:02X}{green:02X}{blue:02X}</strong>
        """
        return html.format(red=self.red_int8,
                           green=self.green_int8,
                           blue=self.blue_int8)

    @staticmethod
    def c_equals(a, b):
        """Raw level version of equality test function for two pixels.

        :param a: a pointer to PixelWand to compare
        :type a: :class:`ctypes.c_void_p`
        :param b: a pointer to PixelWand to compare
        :type b: :class:`ctypes.c_void_p`
        :returns: ``True`` only if two pixels equal
        :rtype: :class:`bool`

        .. note::

           It's only for internal use. Don't use it directly.
           Use ``==`` operator of :class:`Color` instead.

        """
        alpha = library.PixelGetAlpha
        return bool(library.IsPixelWandSimilar(a, b, 0) and
                    alpha(a) == alpha(b))

    @classmethod
    def from_hsl(cls, hue=0.0, saturation=0.0, lightness=0.0):
        """Creates a RGB color from HSL values. The ``hue``, ``saturation``,
        and ``lightness`` must be normalized between 0.0 & 1.0.

        .. code::

            h=0.75  # 270 Degrees
            s=1.0   # 100 Percent
            l=0.5   # 50 Percent
            with Color.from_hsl(hue=h, saturation=s, lightness=l) as color:
                print(color)  #=> srgb(128,0,255)

        :param hue: a normalized double between 0.0 & 1.0.
        :type hue: :class:`numbers.Real`
        :param saturation: a normalized double between 0.0 & 1.0.
        :type saturation: :class:`numbers.Real`
        :param lightness: a normalized double between 0.0 & 1.0.
        :type lightness: :class:`numbers.Real`
        :rtype: :class:`Color`

        .. versionadded:: 0.5.1
        """
        color = cls('WHITE')
        color._assert_double(hue)
        color._assert_double(saturation)
        color._assert_double(lightness)
        color.dirty = True
        with color:
            library.PixelSetHSL(color.resource, hue, saturation, lightness)
        return color

    @classmethod
    def from_pixelwand(cls, pixelwand):
        assert pixelwand
        if MAGICK_VERSION_NUMBER < 0x700:
            pixel_structure = MagickPixelPacket
        else:  # pragma: no cover
            pixel_structure = PixelInfo
        size = ctypes.sizeof(pixel_structure)
        raw_buffer = ctypes.create_string_buffer(size)
        library.PixelGetMagickColor(pixelwand, raw_buffer)
        return cls(raw=raw_buffer)

    @property
    def alpha(self):
        """(:class:`numbers.Real`) Alpha value, from 0.0 to 1.0."""
        with self:
            return library.PixelGetAlpha(self.resource)

    @alpha.setter
    def alpha(self, value):
        self._assert_double(value)
        self.dirty = True
        with self:
            library.PixelSetAlpha(self.resource, value)

    @property
    def alpha_int8(self):
        """(:class:`numbers.Integral`) Alpha value as 8bit integer which is
        a common style.  From 0 to 255.

        .. versionadded:: 0.3.0

        """
        return max(0, min(255, int(255.0 * self.alpha)))

    @alpha_int8.setter
    def alpha_int8(self, value):
        self._assert_int8(value)
        self.alpha = float(value) / 255.0

    @property
    def alpha_quantum(self):
        """(:class:`numbers.Integral`) Alpha value.
        Scale depends on :const:`~wand.version.QUANTUM_DEPTH`.

        .. versionadded:: 0.3.0

        """
        with self:
            return library.PixelGetAlphaQuantum(self.resource)

    @alpha_quantum.setter
    def alpha_quantum(self, value):
        self._assert_quantum(value)
        self.dirty = True
        with self:
            library.PixelSetAlphaQuantum(self.resource, value)

    @property
    def black(self):
        """(:class:`numbers.Real`) Black, or ``'K'``, color channel in CMYK
        colorspace. Unused by RGB colorspace.

        .. versionadded:: 0.5.1
        """
        with self:
            return library.PixelGetBlack(self.resource)

    @black.setter
    def black(self, value):
        self._assert_double(value)
        self.dirty = True
        with self:
            library.PixelSetBlack(self.resource, value)

    @property
    def black_int8(self):
        """(:class:`numbers.Integral`) Black value as 8bit integer which is
        a common style.  From 0 to 255.

        .. versionadded:: 0.5.1
        """
        return max(0, min(255, int(255.0 * self.black)))

    @black_int8.setter
    def black_int8(self, value):
        self._assert_int8(value)
        self.black = float(value) / 255.0

    @property
    def black_quantum(self):
        """(:class:`numbers.Integral`) Black.
        Scale depends on :const:`~wand.version.QUANTUM_DEPTH`.

        .. versionadded:: 0.5.1
        """
        with self:
            return library.PixelGetBlackQuantum(self.resource)

    @black_quantum.setter
    def black_quantum(self, value):
        self._assert_quantum(value)
        self.dirty = True
        with self:
            library.PixelSetBlackQuantum(self.resource, value)

    @property
    def blue(self):
        """(:class:`numbers.Real`) Blue, from 0.0 to 1.0."""
        with self:
            return library.PixelGetBlue(self.resource)

    @blue.setter
    def blue(self, value):
        self._assert_double(value)
        self.dirty = True
        with self:
            library.PixelSetBlue(self.resource, value)

    @property
    def blue_int8(self):
        """(:class:`numbers.Integral`) Blue as 8bit integer which is
        a common style.  From 0 to 255.

        .. versionadded:: 0.3.0

        """
        return max(0, min(255, int(255.0 * self.blue)))

    @blue_int8.setter
    def blue_int8(self, value):
        self._assert_int8(value)
        self.blue = float(value) / 255.0

    @property
    def blue_quantum(self):
        """(:class:`numbers.Integral`) Blue.
        Scale depends on :const:`~wand.version.QUANTUM_DEPTH`.

        .. versionadded:: 0.3.0

        """
        with self:
            return library.PixelGetBlueQuantum(self.resource)

    @blue_quantum.setter
    def blue_quantum(self, value):
        self._assert_quantum(value)
        self.dirty = True
        with self:
            library.PixelSetBlueQuantum(self.resource, value)

    @property
    def cyan(self):
        """(:class:`numbers.Real`) Cyan color channel in CMYK
        colorspace. Unused by RGB colorspace.

        .. versionadded:: 0.5.1
        """
        with self:
            return library.PixelGetCyan(self.resource)

    @cyan.setter
    def cyan(self, value):
        self._assert_double(value)
        self.dirty = True
        with self:
            library.PixelSetCyan(self.resource, value)

    @property
    def cyan_int8(self):
        """(:class:`numbers.Integral`) Cyan value as 8bit integer which is
        a common style.  From 0 to 255.

        .. versionadded:: 0.5.1
        """
        return max(0, min(255, int(255.0 * self.cyan)))

    @cyan_int8.setter
    def cyan_int8(self, value):
        self._assert_int8(value)
        self.cyan = float(value) / 255.0

    @property
    def cyan_quantum(self):
        """(:class:`numbers.Integral`) Cyan.
        Scale depends on :const:`~wand.version.QUANTUM_DEPTH`.

        .. versionadded:: 0.5.1
        """
        with self:
            return library.PixelGetCyanQuantum(self.resource)

    @cyan_quantum.setter
    def cyan_quantum(self, value):
        self._assert_quantum(value)
        self.dirty = True
        with self:
            library.PixelSetCyanQuantum(self.resource, value)

    @property
    def fuzz(self):
        with self:
            return library.PixelGetFuzz(self.resource)

    @fuzz.setter
    def fuzz(self, value):
        if not isinstance(value, numbers.Real):
            raise TypeError('Expecting a float-point real number, not ' +
                            repr(value))
        self.dirty = True
        with self:
            library.PixelSetFuzz(self.resource, value)

    @property
    def green(self):
        """(:class:`numbers.Real`) Green, from 0.0 to 1.0."""
        with self:
            return library.PixelGetGreen(self.resource)

    @green.setter
    def green(self, value):
        self._assert_double(value)
        self.dirty = True
        with self:
            library.PixelSetGreen(self.resource, value)

    @property
    def green_int8(self):
        """(:class:`numbers.Integral`) Green as 8bit integer which is
        a common style.  From 0 to 255.

        .. versionadded:: 0.3.0

        """
        return max(0, min(255, int(255.0 * self.green)))

    @green_int8.setter
    def green_int8(self, value):
        self._assert_int8(value)
        self.green = float(value) / 255.0

    @property
    def green_quantum(self):
        """(:class:`numbers.Integral`) Green.
        Scale depends on :const:`~wand.version.QUANTUM_DEPTH`.

        .. versionadded:: 0.3.0

        """
        with self:
            return library.PixelGetGreenQuantum(self.resource)

    @green_quantum.setter
    def green_quantum(self, value):
        self._assert_quantum(value)
        self.dirty = True
        with self:
            library.PixelSetGreenQuantum(self.resource, value)

    @property
    def magenta(self):
        """(:class:`numbers.Real`) Magenta color channel in CMYK
        colorspace. Unused by RGB colorspace.

        .. versionadded:: 0.5.1
        """
        with self:
            return library.PixelGetMagenta(self.resource)

    @magenta.setter
    def magenta(self, value):
        self._assert_double(value)
        self.dirty = True
        with self:
            library.PixelSetMagenta(self.resource, value)

    @property
    def magenta_int8(self):
        """(:class:`numbers.Integral`) Magenta value as 8bit integer which is
        a common style.  From 0 to 255.

        .. versionadded:: 0.5.1
        """
        return max(0, min(255, int(255.0 * self.magenta)))

    @magenta_int8.setter
    def magenta_int8(self, value):
        self._assert_int8(value)
        self.magenta = float(value) / 255.0

    @property
    def magenta_quantum(self):
        with self:
            return library.PixelGetMagentaQuantum(self.resource)

    @magenta_quantum.setter
    def magenta_quantum(self, value):
        """(:class:`numbers.Integral`) Magenta.
        Scale depends on :const:`~wand.version.QUANTUM_DEPTH`.

        .. versionadded:: 0.5.1
        """
        self._assert_quantum(value)
        self.dirty = True
        with self:
            library.PixelSetMagentaQuantum(self.resource, value)

    @property
    def normalized_string(self):
        """(:class:`basestring`) The normalized string representation of
        the color.  The same color is always represented to the same
        string.

        .. versionadded:: 0.3.0

        """
        with self:
            string = None
            ptr = library.PixelGetColorAsNormalizedString(self.resource)
            if ptr:
                string = text(ctypes.string_at(ptr))
                ptr = library.MagickRelinquishMemory(ptr)
            return string

    @property
    def red(self):
        """(:class:`numbers.Real`) Red, from 0.0 to 1.0."""
        with self:
            return library.PixelGetRed(self.resource)

    @red.setter
    def red(self, value):
        self._assert_double(value)
        self.dirty = True
        with self:
            library.PixelSetRed(self.resource, value)

    @property
    def red_int8(self):
        """(:class:`numbers.Integral`) Red as 8bit integer which is a common
        style.  From 0 to 255.

        .. versionadded:: 0.3.0

        """
        return max(0, min(255, int(255.0 * self.red)))

    @red_int8.setter
    def red_int8(self, value):
        self._assert_int8(value)
        self.red = float(value) / 255.0

    @property
    def red_quantum(self):
        """(:class:`numbers.Integral`) Red.
        Scale depends on :const:`~wand.version.QUANTUM_DEPTH`.

        .. versionadded:: 0.3.0

        """
        with self:
            return library.PixelGetRedQuantum(self.resource)

    @red_quantum.setter
    def red_quantum(self, value):
        self._assert_quantum(value)
        self.dirty = True
        with self:
            library.PixelSetRedQuantum(self.resource, value)

    @property
    def string(self):
        """(:class:`basestring`) The string representation of the color."""
        with self:
            color_string = None
            ptr = library.PixelGetColorAsString(self.resource)
            if ptr:
                color_string = text(ctypes.string_at(ptr))
                ptr = library.MagickRelinquishMemory(ptr)
            return color_string

    @property
    def yellow(self):
        """(:class:`numbers.Real`) Yellow color channel in CMYK
        colorspace. Unused by RGB colorspace.

        .. versionadded:: 0.5.1
        """
        with self:
            return library.PixelGetYellow(self.resource)

    @yellow.setter
    def yellow(self, value):
        self._assert_double(value)
        self.dirty = True
        with self:
            library.PixelSetYellow(self.resource, value)

    @property
    def yellow_int8(self):
        """(:class:`numbers.Integral`) Yellow as 8bit integer which is a common
        style. From 0 to 255.

        .. versionadded:: 0.5.1
        """
        return max(0, min(255, int(255.0 * self.yellow)))

    @yellow_int8.setter
    def yellow_int8(self, value):
        self._assert_int8(value)
        self.yellow = float(value) / 255.0

    @property
    def yellow_quantum(self):
        """(:class:`numbers.Integral`) Yellow.
        Scale depends on :const:`~wand.version.QUANTUM_DEPTH`.

        .. versionadded:: 0.5.1
        """
        with self:
            return library.PixelGetYellowQuantum(self.resource)

    @yellow_quantum.setter
    def yellow_quantum(self, value):
        self._assert_quantum(value)
        self.dirty = True
        with self:
            library.PixelSetYellowQuantum(self.resource, value)

    def hsl(self):
        """Calculate the HSL color values from the RGB color.

        :returns: Tuple containing three normalized doubles, between 0.0 &
                  1.0, representing ``hue``, ``saturation``, and ``lightness``.
        :rtype: :class:`collections.Sequence`

        .. versionadded:: 0.5.1
        """
        hue = ctypes.c_double(0.0)
        saturation = ctypes.c_double(0.0)
        lightness = ctypes.c_double(0.0)
        with self:
            library.PixelGetHSL(self.resource,
                                ctypes.byref(hue),
                                ctypes.byref(saturation),
                                ctypes.byref(lightness))
        return (hue.value, saturation.value, lightness.value)


def scale_quantum_to_int8(quantum):
    """Straightforward port of :c:func:`ScaleQuantumToChar()` inline
    function.

    .. deprecated:: 0.6.6

    :param quantum: quantum value
    :type quantum: :class:`numbers.Integral`
    :returns: 8bit integer of the given ``quantum`` value
    :rtype: :class:`numbers.Integral`

    .. versionadded:: 0.3.0
    .. versionchanged:: 0.5.0
        Added HDRI support
    """
    if quantum <= 0:
        return 0
    table = {8: 1, 16: 257.0, 32: 16843009.0, 64: 72340172838076673.0}
    if MAGICK_HDRI:  # pragma: no cover
        if QUANTUM_DEPTH == 8:
            v = quantum / table[QUANTUM_DEPTH]
        elif QUANTUM_DEPTH == 16:
            v = ((int(quantum + 128) - (int(quantum + 128) >> 8)) >> 8)
        elif QUANTUM_DEPTH == 32:
            v = ((quantum + 8421504) / table[QUANTUM_DEPTH])
        elif QUANTUM_DEPTH == 64:
            v = quantum / table[QUANTUM_DEPTH]
    else:
        v = quantum / table[QUANTUM_DEPTH]
    if v >= 255:
        return 255
    return int(v + 0.5)
