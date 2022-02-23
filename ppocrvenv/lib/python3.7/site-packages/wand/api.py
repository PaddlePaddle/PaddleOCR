""":mod:`wand.api` --- Low-level interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionchanged:: 0.1.10
   Changed to throw :exc:`~exceptions.ImportError` instead of
   :exc:`~exceptions.AttributeError` when the shared library fails to load.

"""
import ctypes
import ctypes.util
import itertools
import os
import os.path
import platform
import sys
import traceback
# Forward import for backwords compatibility.
from .cdefs.structures import (AffineMatrix, MagickPixelPacket, PixelInfo,
                               PointInfo)
if platform.system() == "Windows":
    try:
        import winreg
    except ImportError:
        import _winreg as winreg

__all__ = ('AffineMatrix', 'MagickPixelPacket', 'library', 'libc', 'libmagick',
           'load_library', 'PixelInfo', 'PointInfo')


def library_paths():
    """Iterates for library paths to try loading.  The result paths are not
    guaranteed that they exist.

    :returns: a pair of libwand and libmagick paths.  they can be the same.
              path can be ``None`` as well
    :rtype: :class:`tuple`

    """
    libwand = None
    libmagick = None
    versions = '', '-7', '-7.Q8', '-7.Q16', '-6', '-Q16', '-Q8', '-6.Q16'
    options = '', 'HDRI', 'HDRI-2'
    system = platform.system()
    magick_home = os.environ.get('MAGICK_HOME')
    magick_suffix = os.environ.get('WAND_MAGICK_LIBRARY_SUFFIX')

    if system == 'Windows':
        # ImageMagick installers normally install coder and filter DLLs in
        # subfolders, we need to add those folders to PATH, otherwise loading
        # the DLL later will fail.
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SOFTWARE\ImageMagick\Current") as reg_key:
                libPath = winreg.QueryValueEx(reg_key, "LibPath")
                coderPath = winreg.QueryValueEx(reg_key, "CoderModulesPath")
                filterPath = winreg.QueryValueEx(reg_key, "FilterModulesPath")
                magick_home = libPath[0]
                os.environ['PATH'] += str((';' + libPath[0] + ";" +
                                          coderPath[0] + ";" + filterPath[0]))
        except OSError:
            # otherwise use MAGICK_HOME, and we assume the coder and
            # filter DLLs are in the same directory
            pass

    def magick_path(path):
        return os.path.join(magick_home, *path)
    combinations = itertools.product(versions, options)
    suffixes = list()
    if magick_suffix:
        suffixes = str(magick_suffix).split(';')
    # We need to convert the ``combinations`` generator to a list so we can
    # iterate over it twice.
    suffixes.extend(list(version + option for version, option in combinations))
    if magick_home:
        # exhaustively search for libraries in magick_home before calling
        # find_library.
        for suffix in suffixes:
            # On Windows, the API is split between two libs. On other
            # platforms, it's all contained in one.
            if system == 'Windows':
                libwand = 'CORE_RL_wand_{0}.dll'.format(suffix),
                libmagick = 'CORE_RL_magick_{0}.dll'.format(suffix),
                yield magick_path(libwand), magick_path(libmagick)
                libwand = 'CORE_RL_MagickWand_{0}.dll'.format(suffix),
                libmagick = 'CORE_RL_MagickCore_{0}.dll'.format(suffix),
                yield magick_path(libwand), magick_path(libmagick)
                libwand = 'libMagickWand{0}.dll'.format(suffix),
                libmagick = 'libMagickCore{0}.dll'.format(suffix),
                yield magick_path(libwand), magick_path(libmagick)
            elif system == 'Darwin':
                libwand = 'lib', 'libMagickWand{0}.dylib'.format(suffix),
                yield magick_path(libwand), magick_path(libwand)
            else:
                libwand = 'lib', 'libMagickWand{0}.so'.format(suffix),
                libmagick = 'lib', 'libMagickCore{0}.so'.format(suffix),
                yield magick_path(libwand), magick_path(libmagick)
                libwand = 'lib', 'libMagickWand{0}.so.6'.format(suffix),
                libmagick = 'lib', 'libMagickCore{0}.so.6'.format(suffix),
                yield magick_path(libwand), magick_path(libmagick)
    for suffix in suffixes:
        if system == 'Windows':
            libwand = ctypes.util.find_library('CORE_RL_wand_' + suffix)
            libmagick = ctypes.util.find_library('CORE_RL_magick_' + suffix)
            yield libwand, libmagick
            libwand = ctypes.util.find_library('CORE_RL_MagickWand_' + suffix)
            libmagick = ctypes.util.find_library(
                'CORE_RL_MagickCore_' + suffix
            )
            yield libwand, libmagick
            libwand = ctypes.util.find_library('libMagickWand' + suffix)
            libmagick = ctypes.util.find_library('libMagickCore' + suffix)
            yield libwand, libmagick
        else:
            libwand = ctypes.util.find_library('MagickWand' + suffix)
            yield libwand, libwand


def load_library():
    """Loads the MagickWand library.

    :returns: the MagickWand library and the ImageMagick library
    :rtype: :class:`ctypes.CDLL`

    """
    tried_paths = []
    for libwand_path, libmagick_path in library_paths():
        if libwand_path is None or libmagick_path is None:
            continue
        try:
            tried_paths.append(libwand_path)
            libwand = ctypes.CDLL(str(libwand_path))
            if libwand_path == libmagick_path:
                libmagick = libwand
            else:
                tried_paths.append(libmagick_path)
                libmagick = ctypes.CDLL(str(libmagick_path))
        except (IOError, OSError):
            continue
        return libwand, libmagick
    raise IOError('cannot find library; tried paths: ' + repr(tried_paths))


try:
    # Preserve the module itself even if it fails to import
    sys.modules['wand._api'] = sys.modules['wand.api']
except KeyError:
    # Loading the module locally or a non-standard setting
    pass

try:
    libraries = load_library()
except (OSError, IOError):
    msg = 'https://docs.wand-py.org/en/latest/guide/install.html'
    if sys.platform.startswith(('dragonfly', 'freebsd')):
        msg = 'pkg install'
    elif sys.platform == 'win32':
        msg += '#install-imagemagick-on-windows'
    elif sys.platform == 'darwin':
        mac_pkgmgrs = {'brew': 'brew install freetype imagemagick',
                       'port': 'port install imagemagick'}
        for pkgmgr in mac_pkgmgrs:
            with os.popen('which ' + pkgmgr) as f:
                if f.read().strip():
                    msg = mac_pkgmgrs[pkgmgr]
                    break
        else:
            msg += '#install-imagemagick-on-mac'
    elif hasattr(platform, 'linux_distribution'):
        distname, _, __ = platform.linux_distribution()
        distname = (distname or '').lower()
        if distname in ('debian', 'ubuntu'):
            msg = 'apt-get install libmagickwand-dev'
        elif distname in ('fedora', 'centos', 'redhat'):
            msg = 'yum install ImageMagick-devel'
    raise ImportError('MagickWand shared library not found.\n'
                      'You probably had not installed ImageMagick library.\n'
                      'Try to install:\n  ' + msg)

#: (:class:`ctypes.CDLL`) The MagickWand library.
library = libraries[0]

#: (:class:`ctypes.CDLL`) The ImageMagick library.  It is the same with
#: :data:`library` on platforms other than Windows.
#:
#: .. versionadded:: 0.1.10
libmagick = libraries[1]

try:
    from wand.cdefs import (core, magick_wand, magick_image, magick_property,
                            pixel_iterator, pixel_wand, drawing_wand)

    core.load(libmagick)
    # Let's get the magick-version number to pass to load methods.
    IM_VERSION = ctypes.c_size_t()
    libmagick.GetMagickVersion(ctypes.byref(IM_VERSION))
    # Query Quantum Depth (i.e. Q8, Q16, ... etc).
    IM_QUANTUM_DEPTH = ctypes.c_size_t()
    libmagick.GetMagickQuantumDepth(ctypes.byref(IM_QUANTUM_DEPTH))
    # Does the library support HDRI?
    IM_HDRI = 'HDRI' in str(libmagick.GetMagickFeatures())
    core.load_with_version(libmagick, IM_VERSION.value)
    magick_wand.load(library, IM_VERSION.value)
    magick_property.load(library, IM_VERSION.value)
    magick_image.load(library, IM_VERSION.value)
    pixel_iterator.load(library, IM_VERSION.value)
    pixel_wand.load(library, IM_VERSION.value, IM_QUANTUM_DEPTH.value, IM_HDRI)
    drawing_wand.load(library, IM_VERSION.value)
    del IM_HDRI, IM_QUANTUM_DEPTH, IM_VERSION

except AttributeError:
    raise ImportError('MagickWand shared library not found or incompatible\n'
                      'Original exception was raised in:\n' +
                      traceback.format_exc())

#: (:class:`ctypes.CDLL`) The C standard library.
libc = None

if platform.system() == 'Windows':
    msvcrt = ctypes.util.find_msvcrt()
    # workaround -- the newest visual studio DLL is named differently:
    if not msvcrt and '1900' in platform.python_compiler():
        msvcrt = 'vcruntime140.dll'
    if msvcrt:
        libc = ctypes.CDLL(msvcrt)
else:
    libc_path = ctypes.util.find_library('c')
    if libc_path:
        libc = ctypes.cdll.LoadLibrary(libc_path)
    else:
        # Attempt to guess popular versions of libc
        libc_paths = ('libc.so.6', 'libc.so', 'libc.a', 'libc.dylib',
                      '/usr/lib/libc.dylib')
        for libc_path in libc_paths:
            try:
                libc = ctypes.cdll.LoadLibrary(libc_path)
                break
            except (IOError, OSError):
                continue
    if libc:
        libc.fdopen.argtypes = [ctypes.c_int, ctypes.c_char_p]
        libc.fdopen.restype = ctypes.c_void_p
        libc.fflush.argtypes = [ctypes.c_void_p]
