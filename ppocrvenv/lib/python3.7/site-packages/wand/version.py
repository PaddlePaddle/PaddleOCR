""":mod:`wand.version` --- Version data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can find the current version in the command line interface:

.. sourcecode:: console

   $ python -m wand.version
   0.0.0
   $ python -m wand.version --verbose
   Wand 0.0.0
   ImageMagick 6.7.7-6 2012-06-03 Q16 http://www.imagemagick.org
   $ python -m wand.version --config | grep CC | cut -d : -f 2
   gcc -std=gnu99 -std=gnu99
   $ python -m wand.version --fonts | grep Helvetica
   Helvetica
   Helvetica-Bold
   Helvetica-Light
   Helvetica-Narrow
   Helvetica-Oblique
   $ python -m wand.version --formats | grep CMYK
   CMYK
   CMYKA

.. versionadded:: 0.2.0
   The command line interface.

.. versionadded:: 0.2.2
   The ``--verbose``/``-v`` option which also prints ImageMagick library
   version for CLI.

.. versionadded:: 0.4.1
   The ``--fonts``, ``--formats``, & ``--config`` option allows printing
   additional information about ImageMagick library.

"""
from __future__ import print_function

import ctypes
import datetime
import re
import sys

try:
    from .api import libmagick, library
except ImportError:  # pragma: no cover
    libmagick = None
from .compat import binary, string_type, text


__all__ = ('VERSION', 'VERSION_INFO', 'MAGICK_VERSION',
           'MAGICK_VERSION_DELEGATES', 'MAGICK_VERSION_FEATURES',
           'MAGICK_VERSION_INFO', 'MAGICK_VERSION_NUMBER',
           'MAGICK_RELEASE_DATE', 'MAGICK_RELEASE_DATE_STRING', 'MAGICK_HDRI',
           'QUANTUM_DEPTH', 'QUANTUM_RANGE', 'configure_options',
           'fonts', 'formats')

#: (:class:`tuple`) The version tuple e.g. ``(0, 1, 2)``.
#:
#: .. versionchanged:: 0.1.9
#:    Becomes :class:`tuple`.  (It was string before.)
VERSION_INFO = (0, 6, 7)

#: (:class:`basestring`) The version string e.g. ``'0.1.2'``.
#:
#: .. versionchanged:: 0.1.9
#:    Becomes string.  (It was :class:`tuple` before.)
VERSION = '{0}.{1}.{2}'.format(*VERSION_INFO)

if libmagick:
    c_magick_version = ctypes.c_size_t()
    #: (:class:`basestring`) The version string of the linked ImageMagick
    #: library.  The exactly same string to the result of
    #: :c:func:`GetMagickVersion` function.
    #:
    #: Example::
    #:
    #:    'ImageMagick 6.7.7-6 2012-06-03 Q16 http://www.imagemagick.org'
    #:
    #: .. versionadded:: 0.2.1
    MAGICK_VERSION = text(
        libmagick.GetMagickVersion(ctypes.byref(c_magick_version))
    )

    #: (:class:`numbers.Integral`) The version number of the linked
    #: ImageMagick library.
    #:
    #: .. versionadded:: 0.2.1
    MAGICK_VERSION_NUMBER = c_magick_version.value

    _match = re.match(r'^ImageMagick\s+(\d+)\.(\d+)\.(\d+)(?:-(\d+))?',
                      MAGICK_VERSION)

    #: (:class:`basestring`) A string of all delegates enabled.
    #: This value is identical to what is returned by
    #: :c:func:`GetMagickDelegates`
    #:
    #: Set to empty string if the system uses an older version of
    #: ImageMagick-6, or does not support :c:func:`GetMagickDelegates`.
    #:
    #: .. versionadded:: 0.5.0
    if libmagick.GetMagickDelegates:  # pragma: no cover
        MAGICK_VERSION_DELEGATES = text(libmagick.GetMagickDelegates())
    else:  # pragma: no cover
        MAGICK_VERSION_DELEGATES = ""

    #: (:class:`basestring`) A string of all features enabled.
    #: This value is identical to what is returned by
    #: :c:func:`GetMagickFeatures`
    #:
    #: .. versionadded:: 0.5.0
    MAGICK_VERSION_FEATURES = text(libmagick.GetMagickFeatures())

    #: (:class:`tuple`) The version tuple e.g. ``(6, 7, 7, 6)`` of
    #: :const:`MAGICK_VERSION`.
    #:
    #: .. versionadded:: 0.2.1
    MAGICK_VERSION_INFO = tuple(int(v or 0) for v in _match.groups())

    #: (:class:`basestring`) The date string e.g. ``'2012-06-03'`` of
    #: :const:`MAGICK_RELEASE_DATE_STRING`.  This value is the exactly same
    #: string to the result of :c:func:`GetMagickReleaseDate` function.
    #:
    #: .. versionadded:: 0.2.1
    MAGICK_RELEASE_DATE_STRING = text(libmagick.GetMagickReleaseDate())

    if MAGICK_RELEASE_DATE_STRING:
        _match = re.match(r'^(\d{4})-?(\d\d)-?(\d\d)$',
                          MAGICK_RELEASE_DATE_STRING)
        #: (:class:`datetime.date`) The release date of the linked ImageMagick
        #: library.  Equivalent to the result of :c:func:`GetMagickReleaseDate`
        #: function.
        #:
        #: .. versionadded:: 0.2.1
        MAGICK_RELEASE_DATE = datetime.date(*map(int, _match.groups()))

    c_quantum_depth = ctypes.c_size_t()
    libmagick.GetMagickQuantumDepth(ctypes.byref(c_quantum_depth))
    #: (:class:`numbers.Integral`) The quantum depth configuration of
    #: the linked ImageMagick library.  One of 8, 16, 32, or 64.
    #:
    #: .. versionadded:: 0.3.0
    QUANTUM_DEPTH = c_quantum_depth.value

    c_quantum_range = ctypes.c_size_t()
    libmagick.GetMagickQuantumRange(ctypes.byref(c_quantum_range))
    #: (:class:`numbers.Integral`) The quantum range configuration of
    #: the linked ImageMagick library.
    #:
    #: .. versionadded:: 0.5.0
    QUANTUM_RANGE = c_quantum_range.value

    #: (:class:`bool`) True if ImageMagick is compiled for High Dynamic
    #: Range Image.
    MAGICK_HDRI = 'HDRI' in MAGICK_VERSION_FEATURES

    del c_magick_version, _match, c_quantum_depth, c_quantum_range


def configure_options(pattern='*'):
    """
    Queries ImageMagick library for configurations options given at
    compile-time.

    Example: Find where the ImageMagick documents are installed::

        >>> from wand.version import configure_options
        >>> configure_options('DOC*')
        {'DOCUMENTATION_PATH': '/usr/local/share/doc/ImageMagick-6'}

    :param pattern: A term to filter queries against. Supports wildcard '*'
                    characters. Default patterns '*' for all options.
    :type pattern: :class:`basestring`
    :returns: Directory of configuration options matching given pattern
    :rtype: :class:`collections.defaultdict`
    """
    if not isinstance(pattern, string_type):
        raise TypeError('pattern must be a string, not ' + repr(pattern))
    # We must force init environment to load user config paths.
    library.MagickWandGenesis()
    pattern_p = ctypes.create_string_buffer(binary(pattern))
    config_count = ctypes.c_size_t(0)
    configs = {}
    configs_p = library.MagickQueryConfigureOptions(pattern_p,
                                                    ctypes.byref(config_count))
    for cursor in range(config_count.value):
        config = ctypes.string_at(configs_p[cursor])
        val_p = library.MagickQueryConfigureOption(config)
        if val_p:
            configs[text(config)] = text(ctypes.string_at(val_p))
            val_p = library.MagickRelinquishMemory(val_p)
    if configs_p:
        configs_p = library.MagickRelinquishMemory(configs_p)
    return configs


def fonts(pattern='*'):
    """
    Queries ImageMagick library for available fonts.

    Available fonts can be configured by defining `types.xml`,
    `type-ghostscript.xml`, or `type-windows.xml`.
    Use :func:`wand.version.configure_options` to locate system search path,
    and `resources <http://www.imagemagick.org/script/resources.php>`_
    article for defining xml file.

    Example: List all bold Helvetica fonts::

        >>> from wand.version import fonts
        >>> fonts('*Helvetica*Bold*')
        ['Helvetica-Bold', 'Helvetica-Bold-Oblique', 'Helvetica-BoldOblique',
         'Helvetica-Narrow-Bold', 'Helvetica-Narrow-BoldOblique']


    :param pattern: A term to filter queries against. Supports wildcard '*'
                    characters. Default patterns '*' for all options.
    :type pattern: :class:`basestring`
    :returns: Sequence of matching fonts
    :rtype: :class:`collections.Sequence`
    """
    if not isinstance(pattern, string_type):
        raise TypeError('pattern must be a string, not ' + repr(pattern))
    # We must force init environment to load user config paths.
    library.MagickWandGenesis()
    pattern_p = ctypes.create_string_buffer(binary(pattern))
    number_fonts = ctypes.c_size_t(0)
    fonts = []
    fonts_p = library.MagickQueryFonts(pattern_p,
                                       ctypes.byref(number_fonts))
    for cursor in range(number_fonts.value):
        font = ctypes.string_at(fonts_p[cursor])
        fonts.append(text(font))
    if fonts_p:
        fonts_p = library.MagickRelinquishMemory(fonts_p)
    return fonts


def formats(pattern='*'):
    """
    Queries ImageMagick library for supported formats.

    Example: List supported PNG formats::

        >>> from wand.version import formats
        >>> formats('PNG*')
        ['PNG', 'PNG00', 'PNG8', 'PNG24', 'PNG32', 'PNG48', 'PNG64']


    :param pattern: A term to filter formats against. Supports wildcards '*'
                    characters. Default pattern '*' for all formats.
    :type pattern: :class:`basestring`
    :returns: Sequence of matching formats
    :rtype: :class:`collections.Sequence`
    """
    if not isinstance(pattern, string_type):
        raise TypeError('pattern must be a string, not ' + repr(pattern))
    # We must force init environment to load user config paths.
    library.MagickWandGenesis()
    pattern_p = ctypes.create_string_buffer(binary(pattern))
    number_formats = ctypes.c_size_t(0)
    formats = []
    formats_p = library.MagickQueryFormats(pattern_p,
                                           ctypes.byref(number_formats))
    for cursor in range(number_formats.value):
        value = ctypes.string_at(formats_p[cursor])
        formats.append(text(value))
    if formats_p:
        formats_p = library.MagickRelinquishMemory(formats_p)
    return formats


if __doc__ is not None:
    __doc__ = __doc__.replace('0.0.0', VERSION)

del libmagick


if __name__ == '__main__':  # pragma: no cover
    options = frozenset(sys.argv[1:])
    if '-v' in options or '--verbose' in options:
        print('Wand', VERSION)
        try:
            print(MAGICK_VERSION)
        except NameError:
            pass
    elif '--fonts' in options:
        for font in fonts():
            print(font)
    elif '--formats' in options:
        for supported_format in formats():
            print(supported_format)
    elif '--config' in options:
        config_options = configure_options()
        for key in config_options:
            print('{:24s}: {}'.format(key, config_options[key]))
    else:
        print(VERSION)
