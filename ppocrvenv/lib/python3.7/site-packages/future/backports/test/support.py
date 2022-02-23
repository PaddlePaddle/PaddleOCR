# -*- coding: utf-8 -*-
"""Supporting definitions for the Python regression tests.

Backported for python-future from Python 3.3 test/support.py.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future import utils
from future.builtins import str, range, open, int, map, list

import contextlib
import errno
import functools
import gc
import socket
import sys
import os
import platform
import shutil
import warnings
import unittest
# For Python 2.6 compatibility:
if not hasattr(unittest, 'skip'):
    import unittest2 as unittest

import importlib
# import collections.abc    # not present on Py2.7
import re
import subprocess
import imp
import time
try:
    import sysconfig
except ImportError:
    # sysconfig is not available on Python 2.6. Try using distutils.sysconfig instead:
    from distutils import sysconfig
import fnmatch
import logging.handlers
import struct
import tempfile

try:
    if utils.PY3:
        import _thread, threading
    else:
        import thread as _thread, threading
except ImportError:
    _thread = None
    threading = None
try:
    import multiprocessing.process
except ImportError:
    multiprocessing = None

try:
    import zlib
except ImportError:
    zlib = None

try:
    import gzip
except ImportError:
    gzip = None

try:
    import bz2
except ImportError:
    bz2 = None

try:
    import lzma
except ImportError:
    lzma = None

__all__ = [
    "Error", "TestFailed", "ResourceDenied", "import_module", "verbose",
    "use_resources", "max_memuse", "record_original_stdout",
    "get_original_stdout", "unload", "unlink", "rmtree", "forget",
    "is_resource_enabled", "requires", "requires_freebsd_version",
    "requires_linux_version", "requires_mac_ver", "find_unused_port",
    "bind_port", "IPV6_ENABLED", "is_jython", "TESTFN", "HOST", "SAVEDCWD",
    "temp_cwd", "findfile", "create_empty_file", "sortdict",
    "check_syntax_error", "open_urlresource", "check_warnings", "CleanImport",
    "EnvironmentVarGuard", "TransientResource", "captured_stdout",
    "captured_stdin", "captured_stderr", "time_out", "socket_peer_reset",
    "ioerror_peer_reset", "run_with_locale", 'temp_umask',
    "transient_internet", "set_memlimit", "bigmemtest", "bigaddrspacetest",
    "BasicTestRunner", "run_unittest", "run_doctest", "threading_setup",
    "threading_cleanup", "reap_children", "cpython_only", "check_impl_detail",
    "get_attribute", "swap_item", "swap_attr", "requires_IEEE_754",
    "TestHandler", "Matcher", "can_symlink", "skip_unless_symlink",
    "skip_unless_xattr", "import_fresh_module", "requires_zlib",
    "PIPE_MAX_SIZE", "failfast", "anticipate_failure", "run_with_tz",
    "requires_gzip", "requires_bz2", "requires_lzma", "suppress_crash_popup",
    ]

class Error(Exception):
    """Base class for regression test exceptions."""

class TestFailed(Error):
    """Test failed."""

class ResourceDenied(unittest.SkipTest):
    """Test skipped because it requested a disallowed resource.

    This is raised when a test calls requires() for a resource that
    has not be enabled.  It is used to distinguish between expected
    and unexpected skips.
    """

@contextlib.contextmanager
def _ignore_deprecated_imports(ignore=True):
    """Context manager to suppress package and module deprecation
    warnings when importing them.

    If ignore is False, this context manager has no effect."""
    if ignore:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".+ (module|package)",
                                    DeprecationWarning)
            yield
    else:
        yield


def import_module(name, deprecated=False):
    """Import and return the module to be tested, raising SkipTest if
    it is not available.

    If deprecated is True, any module or package deprecation messages
    will be suppressed."""
    with _ignore_deprecated_imports(deprecated):
        try:
            return importlib.import_module(name)
        except ImportError as msg:
            raise unittest.SkipTest(str(msg))


def _save_and_remove_module(name, orig_modules):
    """Helper function to save and remove a module from sys.modules

    Raise ImportError if the module can't be imported.
    """
    # try to import the module and raise an error if it can't be imported
    if name not in sys.modules:
        __import__(name)
        del sys.modules[name]
    for modname in list(sys.modules):
        if modname == name or modname.startswith(name + '.'):
            orig_modules[modname] = sys.modules[modname]
            del sys.modules[modname]

def _save_and_block_module(name, orig_modules):
    """Helper function to save and block a module in sys.modules

    Return True if the module was in sys.modules, False otherwise.
    """
    saved = True
    try:
        orig_modules[name] = sys.modules[name]
    except KeyError:
        saved = False
    sys.modules[name] = None
    return saved


def anticipate_failure(condition):
    """Decorator to mark a test that is known to be broken in some cases

       Any use of this decorator should have a comment identifying the
       associated tracker issue.
    """
    if condition:
        return unittest.expectedFailure
    return lambda f: f


def import_fresh_module(name, fresh=(), blocked=(), deprecated=False):
    """Import and return a module, deliberately bypassing sys.modules.
    This function imports and returns a fresh copy of the named Python module
    by removing the named module from sys.modules before doing the import.
    Note that unlike reload, the original module is not affected by
    this operation.

    *fresh* is an iterable of additional module names that are also removed
    from the sys.modules cache before doing the import.

    *blocked* is an iterable of module names that are replaced with None
    in the module cache during the import to ensure that attempts to import
    them raise ImportError.

    The named module and any modules named in the *fresh* and *blocked*
    parameters are saved before starting the import and then reinserted into
    sys.modules when the fresh import is complete.

    Module and package deprecation messages are suppressed during this import
    if *deprecated* is True.

    This function will raise ImportError if the named module cannot be
    imported.

    If deprecated is True, any module or package deprecation messages
    will be suppressed.
    """
    # NOTE: test_heapq, test_json and test_warnings include extra sanity checks
    # to make sure that this utility function is working as expected
    with _ignore_deprecated_imports(deprecated):
        # Keep track of modules saved for later restoration as well
        # as those which just need a blocking entry removed
        orig_modules = {}
        names_to_remove = []
        _save_and_remove_module(name, orig_modules)
        try:
            for fresh_name in fresh:
                _save_and_remove_module(fresh_name, orig_modules)
            for blocked_name in blocked:
                if not _save_and_block_module(blocked_name, orig_modules):
                    names_to_remove.append(blocked_name)
            fresh_module = importlib.import_module(name)
        except ImportError:
            fresh_module = None
        finally:
            for orig_name, module in orig_modules.items():
                sys.modules[orig_name] = module
            for name_to_remove in names_to_remove:
                del sys.modules[name_to_remove]
        return fresh_module


def get_attribute(obj, name):
    """Get an attribute, raising SkipTest if AttributeError is raised."""
    try:
        attribute = getattr(obj, name)
    except AttributeError:
        raise unittest.SkipTest("object %r has no attribute %r" % (obj, name))
    else:
        return attribute

verbose = 1              # Flag set to 0 by regrtest.py
use_resources = None     # Flag set to [] by regrtest.py
max_memuse = 0           # Disable bigmem tests (they will still be run with
                         # small sizes, to make sure they work.)
real_max_memuse = 0
failfast = False
match_tests = None

# _original_stdout is meant to hold stdout at the time regrtest began.
# This may be "the real" stdout, or IDLE's emulation of stdout, or whatever.
# The point is to have some flavor of stdout the user can actually see.
_original_stdout = None
def record_original_stdout(stdout):
    global _original_stdout
    _original_stdout = stdout

def get_original_stdout():
    return _original_stdout or sys.stdout

def unload(name):
    try:
        del sys.modules[name]
    except KeyError:
        pass

if sys.platform.startswith("win"):
    def _waitfor(func, pathname, waitall=False):
        # Perform the operation
        func(pathname)
        # Now setup the wait loop
        if waitall:
            dirname = pathname
        else:
            dirname, name = os.path.split(pathname)
            dirname = dirname or '.'
        # Check for `pathname` to be removed from the filesystem.
        # The exponential backoff of the timeout amounts to a total
        # of ~1 second after which the deletion is probably an error
        # anyway.
        # Testing on a i7@4.3GHz shows that usually only 1 iteration is
        # required when contention occurs.
        timeout = 0.001
        while timeout < 1.0:
            # Note we are only testing for the existence of the file(s) in
            # the contents of the directory regardless of any security or
            # access rights.  If we have made it this far, we have sufficient
            # permissions to do that much using Python's equivalent of the
            # Windows API FindFirstFile.
            # Other Windows APIs can fail or give incorrect results when
            # dealing with files that are pending deletion.
            L = os.listdir(dirname)
            if not (L if waitall else name in L):
                return
            # Increase the timeout and try again
            time.sleep(timeout)
            timeout *= 2
        warnings.warn('tests may fail, delete still pending for ' + pathname,
                      RuntimeWarning, stacklevel=4)

    def _unlink(filename):
        _waitfor(os.unlink, filename)

    def _rmdir(dirname):
        _waitfor(os.rmdir, dirname)

    def _rmtree(path):
        def _rmtree_inner(path):
            for name in os.listdir(path):
                fullname = os.path.join(path, name)
                if os.path.isdir(fullname):
                    _waitfor(_rmtree_inner, fullname, waitall=True)
                    os.rmdir(fullname)
                else:
                    os.unlink(fullname)
        _waitfor(_rmtree_inner, path, waitall=True)
        _waitfor(os.rmdir, path)
else:
    _unlink = os.unlink
    _rmdir = os.rmdir
    _rmtree = shutil.rmtree

def unlink(filename):
    try:
        _unlink(filename)
    except OSError as error:
        # The filename need not exist.
        if error.errno not in (errno.ENOENT, errno.ENOTDIR):
            raise

def rmdir(dirname):
    try:
        _rmdir(dirname)
    except OSError as error:
        # The directory need not exist.
        if error.errno != errno.ENOENT:
            raise

def rmtree(path):
    try:
        _rmtree(path)
    except OSError as error:
        if error.errno != errno.ENOENT:
            raise

def make_legacy_pyc(source):
    """Move a PEP 3147 pyc/pyo file to its legacy pyc/pyo location.

    The choice of .pyc or .pyo extension is done based on the __debug__ flag
    value.

    :param source: The file system path to the source file.  The source file
        does not need to exist, however the PEP 3147 pyc file must exist.
    :return: The file system path to the legacy pyc file.
    """
    pyc_file = imp.cache_from_source(source)
    up_one = os.path.dirname(os.path.abspath(source))
    legacy_pyc = os.path.join(up_one, source + ('c' if __debug__ else 'o'))
    os.rename(pyc_file, legacy_pyc)
    return legacy_pyc

def forget(modname):
    """'Forget' a module was ever imported.

    This removes the module from sys.modules and deletes any PEP 3147 or
    legacy .pyc and .pyo files.
    """
    unload(modname)
    for dirname in sys.path:
        source = os.path.join(dirname, modname + '.py')
        # It doesn't matter if they exist or not, unlink all possible
        # combinations of PEP 3147 and legacy pyc and pyo files.
        unlink(source + 'c')
        unlink(source + 'o')
        unlink(imp.cache_from_source(source, debug_override=True))
        unlink(imp.cache_from_source(source, debug_override=False))

# On some platforms, should not run gui test even if it is allowed
# in `use_resources'.
if sys.platform.startswith('win'):
    import ctypes
    import ctypes.wintypes
    def _is_gui_available():
        UOI_FLAGS = 1
        WSF_VISIBLE = 0x0001
        class USEROBJECTFLAGS(ctypes.Structure):
            _fields_ = [("fInherit", ctypes.wintypes.BOOL),
                        ("fReserved", ctypes.wintypes.BOOL),
                        ("dwFlags", ctypes.wintypes.DWORD)]
        dll = ctypes.windll.user32
        h = dll.GetProcessWindowStation()
        if not h:
            raise ctypes.WinError()
        uof = USEROBJECTFLAGS()
        needed = ctypes.wintypes.DWORD()
        res = dll.GetUserObjectInformationW(h,
            UOI_FLAGS,
            ctypes.byref(uof),
            ctypes.sizeof(uof),
            ctypes.byref(needed))
        if not res:
            raise ctypes.WinError()
        return bool(uof.dwFlags & WSF_VISIBLE)
else:
    def _is_gui_available():
        return True

def is_resource_enabled(resource):
    """Test whether a resource is enabled.  Known resources are set by
    regrtest.py."""
    return use_resources is not None and resource in use_resources

def requires(resource, msg=None):
    """Raise ResourceDenied if the specified resource is not available.

    If the caller's module is __main__ then automatically return True.  The
    possibility of False being returned occurs when regrtest.py is
    executing.
    """
    if resource == 'gui' and not _is_gui_available():
        raise unittest.SkipTest("Cannot use the 'gui' resource")
    # see if the caller's module is __main__ - if so, treat as if
    # the resource was set
    if sys._getframe(1).f_globals.get("__name__") == "__main__":
        return
    if not is_resource_enabled(resource):
        if msg is None:
            msg = "Use of the %r resource not enabled" % resource
        raise ResourceDenied(msg)

def _requires_unix_version(sysname, min_version):
    """Decorator raising SkipTest if the OS is `sysname` and the version is less
    than `min_version`.

    For example, @_requires_unix_version('FreeBSD', (7, 2)) raises SkipTest if
    the FreeBSD version is less than 7.2.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            if platform.system() == sysname:
                version_txt = platform.release().split('-', 1)[0]
                try:
                    version = tuple(map(int, version_txt.split('.')))
                except ValueError:
                    pass
                else:
                    if version < min_version:
                        min_version_txt = '.'.join(map(str, min_version))
                        raise unittest.SkipTest(
                            "%s version %s or higher required, not %s"
                            % (sysname, min_version_txt, version_txt))
            return func(*args, **kw)
        wrapper.min_version = min_version
        return wrapper
    return decorator

def requires_freebsd_version(*min_version):
    """Decorator raising SkipTest if the OS is FreeBSD and the FreeBSD version is
    less than `min_version`.

    For example, @requires_freebsd_version(7, 2) raises SkipTest if the FreeBSD
    version is less than 7.2.
    """
    return _requires_unix_version('FreeBSD', min_version)

def requires_linux_version(*min_version):
    """Decorator raising SkipTest if the OS is Linux and the Linux version is
    less than `min_version`.

    For example, @requires_linux_version(2, 6, 32) raises SkipTest if the Linux
    version is less than 2.6.32.
    """
    return _requires_unix_version('Linux', min_version)

def requires_mac_ver(*min_version):
    """Decorator raising SkipTest if the OS is Mac OS X and the OS X
    version if less than min_version.

    For example, @requires_mac_ver(10, 5) raises SkipTest if the OS X version
    is lesser than 10.5.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            if sys.platform == 'darwin':
                version_txt = platform.mac_ver()[0]
                try:
                    version = tuple(map(int, version_txt.split('.')))
                except ValueError:
                    pass
                else:
                    if version < min_version:
                        min_version_txt = '.'.join(map(str, min_version))
                        raise unittest.SkipTest(
                            "Mac OS X %s or higher required, not %s"
                            % (min_version_txt, version_txt))
            return func(*args, **kw)
        wrapper.min_version = min_version
        return wrapper
    return decorator

# Don't use "localhost", since resolving it uses the DNS under recent
# Windows versions (see issue #18792).
HOST = "127.0.0.1"
HOSTv6 = "::1"


def find_unused_port(family=socket.AF_INET, socktype=socket.SOCK_STREAM):
    """Returns an unused port that should be suitable for binding.  This is
    achieved by creating a temporary socket with the same family and type as
    the 'sock' parameter (default is AF_INET, SOCK_STREAM), and binding it to
    the specified host address (defaults to 0.0.0.0) with the port set to 0,
    eliciting an unused ephemeral port from the OS.  The temporary socket is
    then closed and deleted, and the ephemeral port is returned.

    Either this method or bind_port() should be used for any tests where a
    server socket needs to be bound to a particular port for the duration of
    the test.  Which one to use depends on whether the calling code is creating
    a python socket, or if an unused port needs to be provided in a constructor
    or passed to an external program (i.e. the -accept argument to openssl's
    s_server mode).  Always prefer bind_port() over find_unused_port() where
    possible.  Hard coded ports should *NEVER* be used.  As soon as a server
    socket is bound to a hard coded port, the ability to run multiple instances
    of the test simultaneously on the same host is compromised, which makes the
    test a ticking time bomb in a buildbot environment. On Unix buildbots, this
    may simply manifest as a failed test, which can be recovered from without
    intervention in most cases, but on Windows, the entire python process can
    completely and utterly wedge, requiring someone to log in to the buildbot
    and manually kill the affected process.

    (This is easy to reproduce on Windows, unfortunately, and can be traced to
    the SO_REUSEADDR socket option having different semantics on Windows versus
    Unix/Linux.  On Unix, you can't have two AF_INET SOCK_STREAM sockets bind,
    listen and then accept connections on identical host/ports.  An EADDRINUSE
    socket.error will be raised at some point (depending on the platform and
    the order bind and listen were called on each socket).

    However, on Windows, if SO_REUSEADDR is set on the sockets, no EADDRINUSE
    will ever be raised when attempting to bind two identical host/ports. When
    accept() is called on each socket, the second caller's process will steal
    the port from the first caller, leaving them both in an awkwardly wedged
    state where they'll no longer respond to any signals or graceful kills, and
    must be forcibly killed via OpenProcess()/TerminateProcess().

    The solution on Windows is to use the SO_EXCLUSIVEADDRUSE socket option
    instead of SO_REUSEADDR, which effectively affords the same semantics as
    SO_REUSEADDR on Unix.  Given the propensity of Unix developers in the Open
    Source world compared to Windows ones, this is a common mistake.  A quick
    look over OpenSSL's 0.9.8g source shows that they use SO_REUSEADDR when
    openssl.exe is called with the 's_server' option, for example. See
    http://bugs.python.org/issue2550 for more info.  The following site also
    has a very thorough description about the implications of both REUSEADDR
    and EXCLUSIVEADDRUSE on Windows:
    http://msdn2.microsoft.com/en-us/library/ms740621(VS.85).aspx)

    XXX: although this approach is a vast improvement on previous attempts to
    elicit unused ports, it rests heavily on the assumption that the ephemeral
    port returned to us by the OS won't immediately be dished back out to some
    other process when we close and delete our temporary socket but before our
    calling code has a chance to bind the returned port.  We can deal with this
    issue if/when we come across it.
    """

    tempsock = socket.socket(family, socktype)
    port = bind_port(tempsock)
    tempsock.close()
    del tempsock
    return port

def bind_port(sock, host=HOST):
    """Bind the socket to a free port and return the port number.  Relies on
    ephemeral ports in order to ensure we are using an unbound port.  This is
    important as many tests may be running simultaneously, especially in a
    buildbot environment.  This method raises an exception if the sock.family
    is AF_INET and sock.type is SOCK_STREAM, *and* the socket has SO_REUSEADDR
    or SO_REUSEPORT set on it.  Tests should *never* set these socket options
    for TCP/IP sockets.  The only case for setting these options is testing
    multicasting via multiple UDP sockets.

    Additionally, if the SO_EXCLUSIVEADDRUSE socket option is available (i.e.
    on Windows), it will be set on the socket.  This will prevent anyone else
    from bind()'ing to our host/port for the duration of the test.
    """

    if sock.family == socket.AF_INET and sock.type == socket.SOCK_STREAM:
        if hasattr(socket, 'SO_REUSEADDR'):
            if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) == 1:
                raise TestFailed("tests should never set the SO_REUSEADDR "   \
                                 "socket option on TCP/IP sockets!")
        if hasattr(socket, 'SO_REUSEPORT'):
            try:
                if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 1:
                    raise TestFailed("tests should never set the SO_REUSEPORT "   \
                                     "socket option on TCP/IP sockets!")
            except socket.error:
                # Python's socket module was compiled using modern headers
                # thus defining SO_REUSEPORT but this process is running
                # under an older kernel that does not support SO_REUSEPORT.
                pass
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)

    sock.bind((host, 0))
    port = sock.getsockname()[1]
    return port

def _is_ipv6_enabled():
    """Check whether IPv6 is enabled on this host."""
    if socket.has_ipv6:
        sock = None
        try:
            sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            sock.bind(('::1', 0))
            return True
        except (socket.error, socket.gaierror):
            pass
        finally:
            if sock:
                sock.close()
    return False

IPV6_ENABLED = _is_ipv6_enabled()


# A constant likely larger than the underlying OS pipe buffer size, to
# make writes blocking.
# Windows limit seems to be around 512 B, and many Unix kernels have a
# 64 KiB pipe buffer size or 16 * PAGE_SIZE: take a few megs to be sure.
# (see issue #17835 for a discussion of this number).
PIPE_MAX_SIZE = 4 * 1024 * 1024 + 1

# A constant likely larger than the underlying OS socket buffer size, to make
# writes blocking.
# The socket buffer sizes can usually be tuned system-wide (e.g. through sysctl
# on Linux), or on a per-socket basis (SO_SNDBUF/SO_RCVBUF). See issue #18643
# for a discussion of this number).
SOCK_MAX_SIZE = 16 * 1024 * 1024 + 1

# # decorator for skipping tests on non-IEEE 754 platforms
# requires_IEEE_754 = unittest.skipUnless(
#     float.__getformat__("double").startswith("IEEE"),
#     "test requires IEEE 754 doubles")

requires_zlib = unittest.skipUnless(zlib, 'requires zlib')

requires_bz2 = unittest.skipUnless(bz2, 'requires bz2')

requires_lzma = unittest.skipUnless(lzma, 'requires lzma')

is_jython = sys.platform.startswith('java')

# Filename used for testing
if os.name == 'java':
    # Jython disallows @ in module names
    TESTFN = '$test'
else:
    TESTFN = '@test'

# Disambiguate TESTFN for parallel testing, while letting it remain a valid
# module name.
TESTFN = "{0}_{1}_tmp".format(TESTFN, os.getpid())

# # FS_NONASCII: non-ASCII character encodable by os.fsencode(),
# # or None if there is no such character.
# FS_NONASCII = None
# for character in (
#     # First try printable and common characters to have a readable filename.
#     # For each character, the encoding list are just example of encodings able
#     # to encode the character (the list is not exhaustive).
#
#     # U+00E6 (Latin Small Letter Ae): cp1252, iso-8859-1
#     '\u00E6',
#     # U+0130 (Latin Capital Letter I With Dot Above): cp1254, iso8859_3
#     '\u0130',
#     # U+0141 (Latin Capital Letter L With Stroke): cp1250, cp1257
#     '\u0141',
#     # U+03C6 (Greek Small Letter Phi): cp1253
#     '\u03C6',
#     # U+041A (Cyrillic Capital Letter Ka): cp1251
#     '\u041A',
#     # U+05D0 (Hebrew Letter Alef): Encodable to cp424
#     '\u05D0',
#     # U+060C (Arabic Comma): cp864, cp1006, iso8859_6, mac_arabic
#     '\u060C',
#     # U+062A (Arabic Letter Teh): cp720
#     '\u062A',
#     # U+0E01 (Thai Character Ko Kai): cp874
#     '\u0E01',
#
#     # Then try more "special" characters. "special" because they may be
#     # interpreted or displayed differently depending on the exact locale
#     # encoding and the font.
#
#     # U+00A0 (No-Break Space)
#     '\u00A0',
#     # U+20AC (Euro Sign)
#     '\u20AC',
# ):
#     try:
#         os.fsdecode(os.fsencode(character))
#     except UnicodeError:
#         pass
#     else:
#         FS_NONASCII = character
#         break
#
# # TESTFN_UNICODE is a non-ascii filename
# TESTFN_UNICODE = TESTFN + "-\xe0\xf2\u0258\u0141\u011f"
# if sys.platform == 'darwin':
#     # In Mac OS X's VFS API file names are, by definition, canonically
#     # decomposed Unicode, encoded using UTF-8. See QA1173:
#     # http://developer.apple.com/mac/library/qa/qa2001/qa1173.html
#     import unicodedata
#     TESTFN_UNICODE = unicodedata.normalize('NFD', TESTFN_UNICODE)
# TESTFN_ENCODING = sys.getfilesystemencoding()
#
# # TESTFN_UNENCODABLE is a filename (str type) that should *not* be able to be
# # encoded by the filesystem encoding (in strict mode). It can be None if we
# # cannot generate such filename.
# TESTFN_UNENCODABLE = None
# if os.name in ('nt', 'ce'):
#     # skip win32s (0) or Windows 9x/ME (1)
#     if sys.getwindowsversion().platform >= 2:
#         # Different kinds of characters from various languages to minimize the
#         # probability that the whole name is encodable to MBCS (issue #9819)
#         TESTFN_UNENCODABLE = TESTFN + "-\u5171\u0141\u2661\u0363\uDC80"
#         try:
#             TESTFN_UNENCODABLE.encode(TESTFN_ENCODING)
#         except UnicodeEncodeError:
#             pass
#         else:
#             print('WARNING: The filename %r CAN be encoded by the filesystem encoding (%s). '
#                   'Unicode filename tests may not be effective'
#                   % (TESTFN_UNENCODABLE, TESTFN_ENCODING))
#             TESTFN_UNENCODABLE = None
# # Mac OS X denies unencodable filenames (invalid utf-8)
# elif sys.platform != 'darwin':
#     try:
#         # ascii and utf-8 cannot encode the byte 0xff
#         b'\xff'.decode(TESTFN_ENCODING)
#     except UnicodeDecodeError:
#         # 0xff will be encoded using the surrogate character u+DCFF
#         TESTFN_UNENCODABLE = TESTFN \
#             + b'-\xff'.decode(TESTFN_ENCODING, 'surrogateescape')
#     else:
#         # File system encoding (eg. ISO-8859-* encodings) can encode
#         # the byte 0xff. Skip some unicode filename tests.
#         pass
#
# # TESTFN_UNDECODABLE is a filename (bytes type) that should *not* be able to be
# # decoded from the filesystem encoding (in strict mode). It can be None if we
# # cannot generate such filename (ex: the latin1 encoding can decode any byte
# # sequence). On UNIX, TESTFN_UNDECODABLE can be decoded by os.fsdecode() thanks
# # to the surrogateescape error handler (PEP 383), but not from the filesystem
# # encoding in strict mode.
# TESTFN_UNDECODABLE = None
# for name in (
#     # b'\xff' is not decodable by os.fsdecode() with code page 932. Windows
#     # accepts it to create a file or a directory, or don't accept to enter to
#     # such directory (when the bytes name is used). So test b'\xe7' first: it is
#     # not decodable from cp932.
#     b'\xe7w\xf0',
#     # undecodable from ASCII, UTF-8
#     b'\xff',
#     # undecodable from iso8859-3, iso8859-6, iso8859-7, cp424, iso8859-8, cp856
#     # and cp857
#     b'\xae\xd5'
#     # undecodable from UTF-8 (UNIX and Mac OS X)
#     b'\xed\xb2\x80', b'\xed\xb4\x80',
#     # undecodable from shift_jis, cp869, cp874, cp932, cp1250, cp1251, cp1252,
#     # cp1253, cp1254, cp1255, cp1257, cp1258
#     b'\x81\x98',
# ):
#     try:
#         name.decode(TESTFN_ENCODING)
#     except UnicodeDecodeError:
#         TESTFN_UNDECODABLE = os.fsencode(TESTFN) + name
#         break
#
# if FS_NONASCII:
#     TESTFN_NONASCII = TESTFN + '-' + FS_NONASCII
# else:
#     TESTFN_NONASCII = None

# Save the initial cwd
SAVEDCWD = os.getcwd()

@contextlib.contextmanager
def temp_cwd(name='tempcwd', quiet=False, path=None):
    """
    Context manager that temporarily changes the CWD.

    An existing path may be provided as *path*, in which case this
    function makes no changes to the file system.

    Otherwise, the new CWD is created in the current directory and it's
    named *name*. If *quiet* is False (default) and it's not possible to
    create or change the CWD, an error is raised.  If it's True, only a
    warning is raised and the original CWD is used.
    """
    saved_dir = os.getcwd()
    is_temporary = False
    if path is None:
        path = name
        try:
            os.mkdir(name)
            is_temporary = True
        except OSError:
            if not quiet:
                raise
            warnings.warn('tests may fail, unable to create temp CWD ' + name,
                          RuntimeWarning, stacklevel=3)
    try:
        os.chdir(path)
    except OSError:
        if not quiet:
            raise
        warnings.warn('tests may fail, unable to change the CWD to ' + path,
                      RuntimeWarning, stacklevel=3)
    try:
        yield os.getcwd()
    finally:
        os.chdir(saved_dir)
        if is_temporary:
            rmtree(name)


if hasattr(os, "umask"):
    @contextlib.contextmanager
    def temp_umask(umask):
        """Context manager that temporarily sets the process umask."""
        oldmask = os.umask(umask)
        try:
            yield
        finally:
            os.umask(oldmask)


def findfile(file, here=__file__, subdir=None):
    """Try to find a file on sys.path and the working directory.  If it is not
    found the argument passed to the function is returned (this does not
    necessarily signal failure; could still be the legitimate path)."""
    if os.path.isabs(file):
        return file
    if subdir is not None:
        file = os.path.join(subdir, file)
    path = sys.path
    path = [os.path.dirname(here)] + path
    for dn in path:
        fn = os.path.join(dn, file)
        if os.path.exists(fn): return fn
    return file

def create_empty_file(filename):
    """Create an empty file. If the file already exists, truncate it."""
    fd = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.close(fd)

def sortdict(dict):
    "Like repr(dict), but in sorted order."
    items = sorted(dict.items())
    reprpairs = ["%r: %r" % pair for pair in items]
    withcommas = ", ".join(reprpairs)
    return "{%s}" % withcommas

def make_bad_fd():
    """
    Create an invalid file descriptor by opening and closing a file and return
    its fd.
    """
    file = open(TESTFN, "wb")
    try:
        return file.fileno()
    finally:
        file.close()
        unlink(TESTFN)

def check_syntax_error(testcase, statement):
    testcase.assertRaises(SyntaxError, compile, statement,
                          '<test string>', 'exec')

def open_urlresource(url, *args, **kw):
    from future.backports.urllib import (request as urllib_request,
                                         parse as urllib_parse)

    check = kw.pop('check', None)

    filename = urllib_parse.urlparse(url)[2].split('/')[-1] # '/': it's URL!

    fn = os.path.join(os.path.dirname(__file__), "data", filename)

    def check_valid_file(fn):
        f = open(fn, *args, **kw)
        if check is None:
            return f
        elif check(f):
            f.seek(0)
            return f
        f.close()

    if os.path.exists(fn):
        f = check_valid_file(fn)
        if f is not None:
            return f
        unlink(fn)

    # Verify the requirement before downloading the file
    requires('urlfetch')

    print('\tfetching %s ...' % url, file=get_original_stdout())
    f = urllib_request.urlopen(url, timeout=15)
    try:
        with open(fn, "wb") as out:
            s = f.read()
            while s:
                out.write(s)
                s = f.read()
    finally:
        f.close()

    f = check_valid_file(fn)
    if f is not None:
        return f
    raise TestFailed('invalid resource %r' % fn)


class WarningsRecorder(object):
    """Convenience wrapper for the warnings list returned on
       entry to the warnings.catch_warnings() context manager.
    """
    def __init__(self, warnings_list):
        self._warnings = warnings_list
        self._last = 0

    def __getattr__(self, attr):
        if len(self._warnings) > self._last:
            return getattr(self._warnings[-1], attr)
        elif attr in warnings.WarningMessage._WARNING_DETAILS:
            return None
        raise AttributeError("%r has no attribute %r" % (self, attr))

    @property
    def warnings(self):
        return self._warnings[self._last:]

    def reset(self):
        self._last = len(self._warnings)


def _filterwarnings(filters, quiet=False):
    """Catch the warnings, then check if all the expected
    warnings have been raised and re-raise unexpected warnings.
    If 'quiet' is True, only re-raise the unexpected warnings.
    """
    # Clear the warning registry of the calling module
    # in order to re-raise the warnings.
    frame = sys._getframe(2)
    registry = frame.f_globals.get('__warningregistry__')
    if registry:
        if utils.PY3:
            registry.clear()
        else:
            # Py2-compatible:
            for i in range(len(registry)):
                registry.pop()
    with warnings.catch_warnings(record=True) as w:
        # Set filter "always" to record all warnings.  Because
        # test_warnings swap the module, we need to look up in
        # the sys.modules dictionary.
        sys.modules['warnings'].simplefilter("always")
        yield WarningsRecorder(w)
    # Filter the recorded warnings
    reraise = list(w)
    missing = []
    for msg, cat in filters:
        seen = False
        for w in reraise[:]:
            warning = w.message
            # Filter out the matching messages
            if (re.match(msg, str(warning), re.I) and
                issubclass(warning.__class__, cat)):
                seen = True
                reraise.remove(w)
        if not seen and not quiet:
            # This filter caught nothing
            missing.append((msg, cat.__name__))
    if reraise:
        raise AssertionError("unhandled warning %s" % reraise[0])
    if missing:
        raise AssertionError("filter (%r, %s) did not catch any warning" %
                             missing[0])


@contextlib.contextmanager
def check_warnings(*filters, **kwargs):
    """Context manager to silence warnings.

    Accept 2-tuples as positional arguments:
        ("message regexp", WarningCategory)

    Optional argument:
     - if 'quiet' is True, it does not fail if a filter catches nothing
        (default True without argument,
         default False if some filters are defined)

    Without argument, it defaults to:
        check_warnings(("", Warning), quiet=True)
    """
    quiet = kwargs.get('quiet')
    if not filters:
        filters = (("", Warning),)
        # Preserve backward compatibility
        if quiet is None:
            quiet = True
    return _filterwarnings(filters, quiet)


class CleanImport(object):
    """Context manager to force import to return a new module reference.

    This is useful for testing module-level behaviours, such as
    the emission of a DeprecationWarning on import.

    Use like this:

        with CleanImport("foo"):
            importlib.import_module("foo") # new reference
    """

    def __init__(self, *module_names):
        self.original_modules = sys.modules.copy()
        for module_name in module_names:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                # It is possible that module_name is just an alias for
                # another module (e.g. stub for modules renamed in 3.x).
                # In that case, we also need delete the real module to clear
                # the import cache.
                if module.__name__ != module_name:
                    del sys.modules[module.__name__]
                del sys.modules[module_name]

    def __enter__(self):
        return self

    def __exit__(self, *ignore_exc):
        sys.modules.update(self.original_modules)

### Added for python-future:
if utils.PY3:
    import collections.abc
    mybase = collections.abc.MutableMapping
else:
    import UserDict
    mybase = UserDict.DictMixin
###

class EnvironmentVarGuard(mybase):

    """Class to help protect the environment variable properly.  Can be used as
    a context manager."""

    def __init__(self):
        self._environ = os.environ
        self._changed = {}

    def __getitem__(self, envvar):
        return self._environ[envvar]

    def __setitem__(self, envvar, value):
        # Remember the initial value on the first access
        if envvar not in self._changed:
            self._changed[envvar] = self._environ.get(envvar)
        self._environ[envvar] = value

    def __delitem__(self, envvar):
        # Remember the initial value on the first access
        if envvar not in self._changed:
            self._changed[envvar] = self._environ.get(envvar)
        if envvar in self._environ:
            del self._environ[envvar]

    def keys(self):
        return self._environ.keys()

    def __iter__(self):
        return iter(self._environ)

    def __len__(self):
        return len(self._environ)

    def set(self, envvar, value):
        self[envvar] = value

    def unset(self, envvar):
        del self[envvar]

    def __enter__(self):
        return self

    def __exit__(self, *ignore_exc):
        for (k, v) in self._changed.items():
            if v is None:
                if k in self._environ:
                    del self._environ[k]
            else:
                self._environ[k] = v
        os.environ = self._environ


class DirsOnSysPath(object):
    """Context manager to temporarily add directories to sys.path.

    This makes a copy of sys.path, appends any directories given
    as positional arguments, then reverts sys.path to the copied
    settings when the context ends.

    Note that *all* sys.path modifications in the body of the
    context manager, including replacement of the object,
    will be reverted at the end of the block.
    """

    def __init__(self, *paths):
        self.original_value = sys.path[:]
        self.original_object = sys.path
        sys.path.extend(paths)

    def __enter__(self):
        return self

    def __exit__(self, *ignore_exc):
        sys.path = self.original_object
        sys.path[:] = self.original_value


class TransientResource(object):

    """Raise ResourceDenied if an exception is raised while the context manager
    is in effect that matches the specified exception and attributes."""

    def __init__(self, exc, **kwargs):
        self.exc = exc
        self.attrs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, type_=None, value=None, traceback=None):
        """If type_ is a subclass of self.exc and value has attributes matching
        self.attrs, raise ResourceDenied.  Otherwise let the exception
        propagate (if any)."""
        if type_ is not None and issubclass(self.exc, type_):
            for attr, attr_value in self.attrs.items():
                if not hasattr(value, attr):
                    break
                if getattr(value, attr) != attr_value:
                    break
            else:
                raise ResourceDenied("an optional resource is not available")

# Context managers that raise ResourceDenied when various issues
# with the Internet connection manifest themselves as exceptions.
# XXX deprecate these and use transient_internet() instead
time_out = TransientResource(IOError, errno=errno.ETIMEDOUT)
socket_peer_reset = TransientResource(socket.error, errno=errno.ECONNRESET)
ioerror_peer_reset = TransientResource(IOError, errno=errno.ECONNRESET)


@contextlib.contextmanager
def transient_internet(resource_name, timeout=30.0, errnos=()):
    """Return a context manager that raises ResourceDenied when various issues
    with the Internet connection manifest themselves as exceptions."""
    default_errnos = [
        ('ECONNREFUSED', 111),
        ('ECONNRESET', 104),
        ('EHOSTUNREACH', 113),
        ('ENETUNREACH', 101),
        ('ETIMEDOUT', 110),
    ]
    default_gai_errnos = [
        ('EAI_AGAIN', -3),
        ('EAI_FAIL', -4),
        ('EAI_NONAME', -2),
        ('EAI_NODATA', -5),
        # Encountered when trying to resolve IPv6-only hostnames
        ('WSANO_DATA', 11004),
    ]

    denied = ResourceDenied("Resource %r is not available" % resource_name)
    captured_errnos = errnos
    gai_errnos = []
    if not captured_errnos:
        captured_errnos = [getattr(errno, name, num)
                           for (name, num) in default_errnos]
        gai_errnos = [getattr(socket, name, num)
                      for (name, num) in default_gai_errnos]

    def filter_error(err):
        n = getattr(err, 'errno', None)
        if (isinstance(err, socket.timeout) or
            (isinstance(err, socket.gaierror) and n in gai_errnos) or
            n in captured_errnos):
            if not verbose:
                sys.stderr.write(denied.args[0] + "\n")
            # Was: raise denied from err
            # For Python-Future:
            exc = denied
            exc.__cause__ = err
            raise exc

    old_timeout = socket.getdefaulttimeout()
    try:
        if timeout is not None:
            socket.setdefaulttimeout(timeout)
        yield
    except IOError as err:
        # urllib can wrap original socket errors multiple times (!), we must
        # unwrap to get at the original error.
        while True:
            a = err.args
            if len(a) >= 1 and isinstance(a[0], IOError):
                err = a[0]
            # The error can also be wrapped as args[1]:
            #    except socket.error as msg:
            #        raise IOError('socket error', msg).with_traceback(sys.exc_info()[2])
            elif len(a) >= 2 and isinstance(a[1], IOError):
                err = a[1]
            else:
                break
        filter_error(err)
        raise
    # XXX should we catch generic exceptions and look for their
    # __cause__ or __context__?
    finally:
        socket.setdefaulttimeout(old_timeout)


@contextlib.contextmanager
def captured_output(stream_name):
    """Return a context manager used by captured_stdout/stdin/stderr
    that temporarily replaces the sys stream *stream_name* with a StringIO."""
    import io
    orig_stdout = getattr(sys, stream_name)
    setattr(sys, stream_name, io.StringIO())
    try:
        yield getattr(sys, stream_name)
    finally:
        setattr(sys, stream_name, orig_stdout)

def captured_stdout():
    """Capture the output of sys.stdout:

       with captured_stdout() as s:
           print("hello")
       self.assertEqual(s.getvalue(), "hello")
    """
    return captured_output("stdout")

def captured_stderr():
    return captured_output("stderr")

def captured_stdin():
    return captured_output("stdin")


def gc_collect():
    """Force as many objects as possible to be collected.

    In non-CPython implementations of Python, this is needed because timely
    deallocation is not guaranteed by the garbage collector.  (Even in CPython
    this can be the case in case of reference cycles.)  This means that __del__
    methods may be called later than expected and weakrefs may remain alive for
    longer than expected.  This function tries its best to force all garbage
    objects to disappear.
    """
    gc.collect()
    if is_jython:
        time.sleep(0.1)
    gc.collect()
    gc.collect()

@contextlib.contextmanager
def disable_gc():
    have_gc = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if have_gc:
            gc.enable()


def python_is_optimized():
    """Find if Python was built with optimizations."""
    # We don't have sysconfig on Py2.6:
    import sysconfig
    cflags = sysconfig.get_config_var('PY_CFLAGS') or ''
    final_opt = ""
    for opt in cflags.split():
        if opt.startswith('-O'):
            final_opt = opt
    return final_opt != '' and final_opt != '-O0'


_header = 'nP'
_align = '0n'
if hasattr(sys, "gettotalrefcount"):
    _header = '2P' + _header
    _align = '0P'
_vheader = _header + 'n'

def calcobjsize(fmt):
    return struct.calcsize(_header + fmt + _align)

def calcvobjsize(fmt):
    return struct.calcsize(_vheader + fmt + _align)


_TPFLAGS_HAVE_GC = 1<<14
_TPFLAGS_HEAPTYPE = 1<<9

def check_sizeof(test, o, size):
    result = sys.getsizeof(o)
    # add GC header size
    if ((type(o) == type) and (o.__flags__ & _TPFLAGS_HEAPTYPE) or\
        ((type(o) != type) and (type(o).__flags__ & _TPFLAGS_HAVE_GC))):
        size += _testcapi.SIZEOF_PYGC_HEAD
    msg = 'wrong size for %s: got %d, expected %d' \
            % (type(o), result, size)
    test.assertEqual(result, size, msg)

#=======================================================================
# Decorator for running a function in a different locale, correctly resetting
# it afterwards.

def run_with_locale(catstr, *locales):
    def decorator(func):
        def inner(*args, **kwds):
            try:
                import locale
                category = getattr(locale, catstr)
                orig_locale = locale.setlocale(category)
            except AttributeError:
                # if the test author gives us an invalid category string
                raise
            except:
                # cannot retrieve original locale, so do nothing
                locale = orig_locale = None
            else:
                for loc in locales:
                    try:
                        locale.setlocale(category, loc)
                        break
                    except:
                        pass

            # now run the function, resetting the locale on exceptions
            try:
                return func(*args, **kwds)
            finally:
                if locale and orig_locale:
                    locale.setlocale(category, orig_locale)
        inner.__name__ = func.__name__
        inner.__doc__ = func.__doc__
        return inner
    return decorator

#=======================================================================
# Decorator for running a function in a specific timezone, correctly
# resetting it afterwards.

def run_with_tz(tz):
    def decorator(func):
        def inner(*args, **kwds):
            try:
                tzset = time.tzset
            except AttributeError:
                raise unittest.SkipTest("tzset required")
            if 'TZ' in os.environ:
                orig_tz = os.environ['TZ']
            else:
                orig_tz = None
            os.environ['TZ'] = tz
            tzset()

            # now run the function, resetting the tz on exceptions
            try:
                return func(*args, **kwds)
            finally:
                if orig_tz is None:
                    del os.environ['TZ']
                else:
                    os.environ['TZ'] = orig_tz
                time.tzset()

        inner.__name__ = func.__name__
        inner.__doc__ = func.__doc__
        return inner
    return decorator

#=======================================================================
# Big-memory-test support. Separate from 'resources' because memory use
# should be configurable.

# Some handy shorthands. Note that these are used for byte-limits as well
# as size-limits, in the various bigmem tests
_1M = 1024*1024
_1G = 1024 * _1M
_2G = 2 * _1G
_4G = 4 * _1G

MAX_Py_ssize_t = sys.maxsize

def set_memlimit(limit):
    global max_memuse
    global real_max_memuse
    sizes = {
        'k': 1024,
        'm': _1M,
        'g': _1G,
        't': 1024*_1G,
    }
    m = re.match(r'(\d+(\.\d+)?) (K|M|G|T)b?$', limit,
                 re.IGNORECASE | re.VERBOSE)
    if m is None:
        raise ValueError('Invalid memory limit %r' % (limit,))
    memlimit = int(float(m.group(1)) * sizes[m.group(3).lower()])
    real_max_memuse = memlimit
    if memlimit > MAX_Py_ssize_t:
        memlimit = MAX_Py_ssize_t
    if memlimit < _2G - 1:
        raise ValueError('Memory limit %r too low to be useful' % (limit,))
    max_memuse = memlimit

class _MemoryWatchdog(object):
    """An object which periodically watches the process' memory consumption
    and prints it out.
    """

    def __init__(self):
        self.procfile = '/proc/{pid}/statm'.format(pid=os.getpid())
        self.started = False

    def start(self):
        try:
            f = open(self.procfile, 'r')
        except OSError as e:
            warnings.warn('/proc not available for stats: {0}'.format(e),
                          RuntimeWarning)
            sys.stderr.flush()
            return

        watchdog_script = findfile("memory_watchdog.py")
        self.mem_watchdog = subprocess.Popen([sys.executable, watchdog_script],
                                             stdin=f, stderr=subprocess.DEVNULL)
        f.close()
        self.started = True

    def stop(self):
        if self.started:
            self.mem_watchdog.terminate()
            self.mem_watchdog.wait()


def bigmemtest(size, memuse, dry_run=True):
    """Decorator for bigmem tests.

    'minsize' is the minimum useful size for the test (in arbitrary,
    test-interpreted units.) 'memuse' is the number of 'bytes per size' for
    the test, or a good estimate of it.

    if 'dry_run' is False, it means the test doesn't support dummy runs
    when -M is not specified.
    """
    def decorator(f):
        def wrapper(self):
            size = wrapper.size
            memuse = wrapper.memuse
            if not real_max_memuse:
                maxsize = 5147
            else:
                maxsize = size

            if ((real_max_memuse or not dry_run)
                and real_max_memuse < maxsize * memuse):
                raise unittest.SkipTest(
                    "not enough memory: %.1fG minimum needed"
                    % (size * memuse / (1024 ** 3)))

            if real_max_memuse and verbose:
                print()
                print(" ... expected peak memory use: {peak:.1f}G"
                      .format(peak=size * memuse / (1024 ** 3)))
                watchdog = _MemoryWatchdog()
                watchdog.start()
            else:
                watchdog = None

            try:
                return f(self, maxsize)
            finally:
                if watchdog:
                    watchdog.stop()

        wrapper.size = size
        wrapper.memuse = memuse
        return wrapper
    return decorator

def bigaddrspacetest(f):
    """Decorator for tests that fill the address space."""
    def wrapper(self):
        if max_memuse < MAX_Py_ssize_t:
            if MAX_Py_ssize_t >= 2**63 - 1 and max_memuse >= 2**31:
                raise unittest.SkipTest(
                    "not enough memory: try a 32-bit build instead")
            else:
                raise unittest.SkipTest(
                    "not enough memory: %.1fG minimum needed"
                    % (MAX_Py_ssize_t / (1024 ** 3)))
        else:
            return f(self)
    return wrapper

#=======================================================================
# unittest integration.

class BasicTestRunner(object):
    def run(self, test):
        result = unittest.TestResult()
        test(result)
        return result

def _id(obj):
    return obj

def requires_resource(resource):
    if resource == 'gui' and not _is_gui_available():
        return unittest.skip("resource 'gui' is not available")
    if is_resource_enabled(resource):
        return _id
    else:
        return unittest.skip("resource {0!r} is not enabled".format(resource))

def cpython_only(test):
    """
    Decorator for tests only applicable on CPython.
    """
    return impl_detail(cpython=True)(test)

def impl_detail(msg=None, **guards):
    if check_impl_detail(**guards):
        return _id
    if msg is None:
        guardnames, default = _parse_guards(guards)
        if default:
            msg = "implementation detail not available on {0}"
        else:
            msg = "implementation detail specific to {0}"
        guardnames = sorted(guardnames.keys())
        msg = msg.format(' or '.join(guardnames))
    return unittest.skip(msg)

def _parse_guards(guards):
    # Returns a tuple ({platform_name: run_me}, default_value)
    if not guards:
        return ({'cpython': True}, False)
    is_true = list(guards.values())[0]
    assert list(guards.values()) == [is_true] * len(guards)   # all True or all False
    return (guards, not is_true)

# Use the following check to guard CPython's implementation-specific tests --
# or to run them only on the implementation(s) guarded by the arguments.
def check_impl_detail(**guards):
    """This function returns True or False depending on the host platform.
       Examples:
          if check_impl_detail():               # only on CPython (default)
          if check_impl_detail(jython=True):    # only on Jython
          if check_impl_detail(cpython=False):  # everywhere except on CPython
    """
    guards, default = _parse_guards(guards)
    return guards.get(platform.python_implementation().lower(), default)


def no_tracing(func):
    """Decorator to temporarily turn off tracing for the duration of a test."""
    if not hasattr(sys, 'gettrace'):
        return func
    else:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            original_trace = sys.gettrace()
            try:
                sys.settrace(None)
                return func(*args, **kwargs)
            finally:
                sys.settrace(original_trace)
        return wrapper


def refcount_test(test):
    """Decorator for tests which involve reference counting.

    To start, the decorator does not run the test if is not run by CPython.
    After that, any trace function is unset during the test to prevent
    unexpected refcounts caused by the trace function.

    """
    return no_tracing(cpython_only(test))


def _filter_suite(suite, pred):
    """Recursively filter test cases in a suite based on a predicate."""
    newtests = []
    for test in suite._tests:
        if isinstance(test, unittest.TestSuite):
            _filter_suite(test, pred)
            newtests.append(test)
        else:
            if pred(test):
                newtests.append(test)
    suite._tests = newtests

def _run_suite(suite):
    """Run tests from a unittest.TestSuite-derived class."""
    if verbose:
        runner = unittest.TextTestRunner(sys.stdout, verbosity=2,
                                         failfast=failfast)
    else:
        runner = BasicTestRunner()

    result = runner.run(suite)
    if not result.wasSuccessful():
        if len(result.errors) == 1 and not result.failures:
            err = result.errors[0][1]
        elif len(result.failures) == 1 and not result.errors:
            err = result.failures[0][1]
        else:
            err = "multiple errors occurred"
            if not verbose: err += "; run in verbose mode for details"
        raise TestFailed(err)


def run_unittest(*classes):
    """Run tests from unittest.TestCase-derived classes."""
    valid_types = (unittest.TestSuite, unittest.TestCase)
    suite = unittest.TestSuite()
    for cls in classes:
        if isinstance(cls, str):
            if cls in sys.modules:
                suite.addTest(unittest.findTestCases(sys.modules[cls]))
            else:
                raise ValueError("str arguments must be keys in sys.modules")
        elif isinstance(cls, valid_types):
            suite.addTest(cls)
        else:
            suite.addTest(unittest.makeSuite(cls))
    def case_pred(test):
        if match_tests is None:
            return True
        for name in test.id().split("."):
            if fnmatch.fnmatchcase(name, match_tests):
                return True
        return False
    _filter_suite(suite, case_pred)
    _run_suite(suite)

# We don't have sysconfig on Py2.6:
# #=======================================================================
# # Check for the presence of docstrings.
#
# HAVE_DOCSTRINGS = (check_impl_detail(cpython=False) or
#                    sys.platform == 'win32' or
#                    sysconfig.get_config_var('WITH_DOC_STRINGS'))
#
# requires_docstrings = unittest.skipUnless(HAVE_DOCSTRINGS,
#                                           "test requires docstrings")
#
#
# #=======================================================================
# doctest driver.

def run_doctest(module, verbosity=None, optionflags=0):
    """Run doctest on the given module.  Return (#failures, #tests).

    If optional argument verbosity is not specified (or is None), pass
    support's belief about verbosity on to doctest.  Else doctest's
    usual behavior is used (it searches sys.argv for -v).
    """

    import doctest

    if verbosity is None:
        verbosity = verbose
    else:
        verbosity = None

    f, t = doctest.testmod(module, verbose=verbosity, optionflags=optionflags)
    if f:
        raise TestFailed("%d of %d doctests failed" % (f, t))
    if verbose:
        print('doctest (%s) ... %d tests with zero failures' %
              (module.__name__, t))
    return f, t


#=======================================================================
# Support for saving and restoring the imported modules.

def modules_setup():
    return sys.modules.copy(),

def modules_cleanup(oldmodules):
    # Encoders/decoders are registered permanently within the internal
    # codec cache. If we destroy the corresponding modules their
    # globals will be set to None which will trip up the cached functions.
    encodings = [(k, v) for k, v in sys.modules.items()
                 if k.startswith('encodings.')]
    # Was:
    # sys.modules.clear()
    # Py2-compatible:
    for i in range(len(sys.modules)):
        sys.modules.pop()

    sys.modules.update(encodings)
    # XXX: This kind of problem can affect more than just encodings. In particular
    # extension modules (such as _ssl) don't cope with reloading properly.
    # Really, test modules should be cleaning out the test specific modules they
    # know they added (ala test_runpy) rather than relying on this function (as
    # test_importhooks and test_pkg do currently).
    # Implicitly imported *real* modules should be left alone (see issue 10556).
    sys.modules.update(oldmodules)

#=======================================================================
# Backported versions of threading_setup() and threading_cleanup() which don't refer
# to threading._dangling (not available on Py2.7).

# Threading support to prevent reporting refleaks when running regrtest.py -R

# NOTE: we use thread._count() rather than threading.enumerate() (or the
# moral equivalent thereof) because a threading.Thread object is still alive
# until its __bootstrap() method has returned, even after it has been
# unregistered from the threading module.
# thread._count(), on the other hand, only gets decremented *after* the
# __bootstrap() method has returned, which gives us reliable reference counts
# at the end of a test run.

def threading_setup():
    if _thread:
        return _thread._count(),
    else:
        return 1,

def threading_cleanup(nb_threads):
    if not _thread:
        return

    _MAX_COUNT = 10
    for count in range(_MAX_COUNT):
        n = _thread._count()
        if n == nb_threads:
            break
        time.sleep(0.1)
    # XXX print a warning in case of failure?

def reap_threads(func):
    """Use this function when threads are being used.  This will
    ensure that the threads are cleaned up even when the test fails.
    If threading is unavailable this function does nothing.
    """
    if not _thread:
        return func

    @functools.wraps(func)
    def decorator(*args):
        key = threading_setup()
        try:
            return func(*args)
        finally:
            threading_cleanup(*key)
    return decorator

def reap_children():
    """Use this function at the end of test_main() whenever sub-processes
    are started.  This will help ensure that no extra children (zombies)
    stick around to hog resources and create problems when looking
    for refleaks.
    """

    # Reap all our dead child processes so we don't leave zombies around.
    # These hog resources and might be causing some of the buildbots to die.
    if hasattr(os, 'waitpid'):
        any_process = -1
        while True:
            try:
                # This will raise an exception on Windows.  That's ok.
                pid, status = os.waitpid(any_process, os.WNOHANG)
                if pid == 0:
                    break
            except:
                break

@contextlib.contextmanager
def swap_attr(obj, attr, new_val):
    """Temporary swap out an attribute with a new object.

    Usage:
        with swap_attr(obj, "attr", 5):
            ...

        This will set obj.attr to 5 for the duration of the with: block,
        restoring the old value at the end of the block. If `attr` doesn't
        exist on `obj`, it will be created and then deleted at the end of the
        block.
    """
    if hasattr(obj, attr):
        real_val = getattr(obj, attr)
        setattr(obj, attr, new_val)
        try:
            yield
        finally:
            setattr(obj, attr, real_val)
    else:
        setattr(obj, attr, new_val)
        try:
            yield
        finally:
            delattr(obj, attr)

@contextlib.contextmanager
def swap_item(obj, item, new_val):
    """Temporary swap out an item with a new object.

    Usage:
        with swap_item(obj, "item", 5):
            ...

        This will set obj["item"] to 5 for the duration of the with: block,
        restoring the old value at the end of the block. If `item` doesn't
        exist on `obj`, it will be created and then deleted at the end of the
        block.
    """
    if item in obj:
        real_val = obj[item]
        obj[item] = new_val
        try:
            yield
        finally:
            obj[item] = real_val
    else:
        obj[item] = new_val
        try:
            yield
        finally:
            del obj[item]

def strip_python_stderr(stderr):
    """Strip the stderr of a Python process from potential debug output
    emitted by the interpreter.

    This will typically be run on the result of the communicate() method
    of a subprocess.Popen object.
    """
    stderr = re.sub(br"\[\d+ refs\]\r?\n?", b"", stderr).strip()
    return stderr

def args_from_interpreter_flags():
    """Return a list of command-line arguments reproducing the current
    settings in sys.flags and sys.warnoptions."""
    return subprocess._args_from_interpreter_flags()

#============================================================
# Support for assertions about logging.
#============================================================

class TestHandler(logging.handlers.BufferingHandler):
    def __init__(self, matcher):
        # BufferingHandler takes a "capacity" argument
        # so as to know when to flush. As we're overriding
        # shouldFlush anyway, we can set a capacity of zero.
        # You can call flush() manually to clear out the
        # buffer.
        logging.handlers.BufferingHandler.__init__(self, 0)
        self.matcher = matcher

    def shouldFlush(self):
        return False

    def emit(self, record):
        self.format(record)
        self.buffer.append(record.__dict__)

    def matches(self, **kwargs):
        """
        Look for a saved dict whose keys/values match the supplied arguments.
        """
        result = False
        for d in self.buffer:
            if self.matcher.matches(d, **kwargs):
                result = True
                break
        return result

class Matcher(object):

    _partial_matches = ('msg', 'message')

    def matches(self, d, **kwargs):
        """
        Try to match a single dict with the supplied arguments.

        Keys whose values are strings and which are in self._partial_matches
        will be checked for partial (i.e. substring) matches. You can extend
        this scheme to (for example) do regular expression matching, etc.
        """
        result = True
        for k in kwargs:
            v = kwargs[k]
            dv = d.get(k)
            if not self.match_value(k, dv, v):
                result = False
                break
        return result

    def match_value(self, k, dv, v):
        """
        Try to match a single stored value (dv) with a supplied value (v).
        """
        if type(v) != type(dv):
            result = False
        elif type(dv) is not str or k not in self._partial_matches:
            result = (v == dv)
        else:
            result = dv.find(v) >= 0
        return result


_can_symlink = None
def can_symlink():
    global _can_symlink
    if _can_symlink is not None:
        return _can_symlink
    symlink_path = TESTFN + "can_symlink"
    try:
        os.symlink(TESTFN, symlink_path)
        can = True
    except (OSError, NotImplementedError, AttributeError):
        can = False
    else:
        os.remove(symlink_path)
    _can_symlink = can
    return can

def skip_unless_symlink(test):
    """Skip decorator for tests that require functional symlink"""
    ok = can_symlink()
    msg = "Requires functional symlink implementation"
    return test if ok else unittest.skip(msg)(test)

_can_xattr = None
def can_xattr():
    global _can_xattr
    if _can_xattr is not None:
        return _can_xattr
    if not hasattr(os, "setxattr"):
        can = False
    else:
        tmp_fp, tmp_name = tempfile.mkstemp()
        try:
            with open(TESTFN, "wb") as fp:
                try:
                    # TESTFN & tempfile may use different file systems with
                    # different capabilities
                    os.setxattr(tmp_fp, b"user.test", b"")
                    os.setxattr(fp.fileno(), b"user.test", b"")
                    # Kernels < 2.6.39 don't respect setxattr flags.
                    kernel_version = platform.release()
                    m = re.match("2.6.(\d{1,2})", kernel_version)
                    can = m is None or int(m.group(1)) >= 39
                except OSError:
                    can = False
        finally:
            unlink(TESTFN)
            unlink(tmp_name)
    _can_xattr = can
    return can

def skip_unless_xattr(test):
    """Skip decorator for tests that require functional extended attributes"""
    ok = can_xattr()
    msg = "no non-broken extended attribute support"
    return test if ok else unittest.skip(msg)(test)


if sys.platform.startswith('win'):
    @contextlib.contextmanager
    def suppress_crash_popup():
        """Disable Windows Error Reporting dialogs using SetErrorMode."""
        # see http://msdn.microsoft.com/en-us/library/windows/desktop/ms680621%28v=vs.85%29.aspx
        # GetErrorMode is not available on Windows XP and Windows Server 2003,
        # but SetErrorMode returns the previous value, so we can use that
        import ctypes
        k32 = ctypes.windll.kernel32
        SEM_NOGPFAULTERRORBOX = 0x02
        old_error_mode = k32.SetErrorMode(SEM_NOGPFAULTERRORBOX)
        k32.SetErrorMode(old_error_mode | SEM_NOGPFAULTERRORBOX)
        try:
            yield
        finally:
            k32.SetErrorMode(old_error_mode)
else:
    # this is a no-op for other platforms
    @contextlib.contextmanager
    def suppress_crash_popup():
        yield


def patch(test_instance, object_to_patch, attr_name, new_value):
    """Override 'object_to_patch'.'attr_name' with 'new_value'.

    Also, add a cleanup procedure to 'test_instance' to restore
    'object_to_patch' value for 'attr_name'.
    The 'attr_name' should be a valid attribute for 'object_to_patch'.

    """
    # check that 'attr_name' is a real attribute for 'object_to_patch'
    # will raise AttributeError if it does not exist
    getattr(object_to_patch, attr_name)

    # keep a copy of the old value
    attr_is_local = False
    try:
        old_value = object_to_patch.__dict__[attr_name]
    except (AttributeError, KeyError):
        old_value = getattr(object_to_patch, attr_name, None)
    else:
        attr_is_local = True

    # restore the value when the test is done
    def cleanup():
        if attr_is_local:
            setattr(object_to_patch, attr_name, old_value)
        else:
            delattr(object_to_patch, attr_name)

    test_instance.addCleanup(cleanup)

    # actually override the attribute
    setattr(object_to_patch, attr_name, new_value)
