#
#   Cython -- Things that don't belong
#            anywhere else in particular
#

from __future__ import absolute_import

try:
    from __builtin__ import basestring
except ImportError:
    basestring = str

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = OSError

import os
import sys
import re
import io
import codecs
import shutil
import tempfile
from contextlib import contextmanager

modification_time = os.path.getmtime

_function_caches = []
def clear_function_caches():
    for cache in _function_caches:
        cache.clear()

def cached_function(f):
    cache = {}
    _function_caches.append(cache)
    uncomputed = object()
    def wrapper(*args):
        res = cache.get(args, uncomputed)
        if res is uncomputed:
            res = cache[args] = f(*args)
        return res
    wrapper.uncached = f
    return wrapper

def cached_method(f):
    cache_name = '__%s_cache' % f.__name__
    def wrapper(self, *args):
        cache = getattr(self, cache_name, None)
        if cache is None:
            cache = {}
            setattr(self, cache_name, cache)
        if args in cache:
            return cache[args]
        res = cache[args] = f(self, *args)
        return res
    return wrapper

def replace_suffix(path, newsuf):
    base, _ = os.path.splitext(path)
    return base + newsuf


def open_new_file(path):
    if os.path.exists(path):
        # Make sure to create a new file here so we can
        # safely hard link the output files.
        os.unlink(path)

    # we use the ISO-8859-1 encoding here because we only write pure
    # ASCII strings or (e.g. for file names) byte encoded strings as
    # Unicode, so we need a direct mapping from the first 256 Unicode
    # characters to a byte sequence, which ISO-8859-1 provides

    # note: can't use io.open() in Py2 as we may be writing str objects
    return codecs.open(path, "w", encoding="ISO-8859-1")


def castrate_file(path, st):
    #  Remove junk contents from an output file after a
    #  failed compilation.
    #  Also sets access and modification times back to
    #  those specified by st (a stat struct).
    try:
        f = open_new_file(path)
    except EnvironmentError:
        pass
    else:
        f.write(
            "#error Do not use this file, it is the result of a failed Cython compilation.\n")
        f.close()
        if st:
            os.utime(path, (st.st_atime, st.st_mtime-1))

def file_newer_than(path, time):
    ftime = modification_time(path)
    return ftime > time


def safe_makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def copy_file_to_dir_if_newer(sourcefile, destdir):
    """
    Copy file sourcefile to directory destdir (creating it if needed),
    preserving metadata. If the destination file exists and is not
    older than the source file, the copying is skipped.
    """
    destfile = os.path.join(destdir, os.path.basename(sourcefile))
    try:
        desttime = modification_time(destfile)
    except OSError:
        # New file does not exist, destdir may or may not exist
        safe_makedirs(destdir)
    else:
        # New file already exists
        if not file_newer_than(sourcefile, desttime):
            return
    shutil.copy2(sourcefile, destfile)


@cached_function
def find_root_package_dir(file_path):
    dir = os.path.dirname(file_path)
    if file_path == dir:
        return dir
    elif is_package_dir(dir):
        return find_root_package_dir(dir)
    else:
        return dir

@cached_function
def check_package_dir(dir, package_names):
    for dirname in package_names:
        dir = os.path.join(dir, dirname)
        if not is_package_dir(dir):
            return None
    return dir

@cached_function
def is_package_dir(dir_path):
    for filename in ("__init__.py",
                     "__init__.pyc",
                     "__init__.pyx",
                     "__init__.pxd"):
        path = os.path.join(dir_path, filename)
        if path_exists(path):
            return 1

@cached_function
def path_exists(path):
    # try on the filesystem first
    if os.path.exists(path):
        return True
    # figure out if a PEP 302 loader is around
    try:
        loader = __loader__
        # XXX the code below assumes a 'zipimport.zipimporter' instance
        # XXX should be easy to generalize, but too lazy right now to write it
        archive_path = getattr(loader, 'archive', None)
        if archive_path:
            normpath = os.path.normpath(path)
            if normpath.startswith(archive_path):
                arcname = normpath[len(archive_path)+1:]
                try:
                    loader.get_data(arcname)
                    return True
                except IOError:
                    return False
    except NameError:
        pass
    return False

# file name encodings

def decode_filename(filename):
    if isinstance(filename, bytes):
        try:
            filename_encoding = sys.getfilesystemencoding()
            if filename_encoding is None:
                filename_encoding = sys.getdefaultencoding()
            filename = filename.decode(filename_encoding)
        except UnicodeDecodeError:
            pass
    return filename

# support for source file encoding detection

_match_file_encoding = re.compile(br"(\w*coding)[:=]\s*([-\w.]+)").search


def detect_opened_file_encoding(f):
    # PEPs 263 and 3120
    # Most of the time the first two lines fall in the first couple of hundred chars,
    # and this bulk read/split is much faster.
    lines = ()
    start = b''
    while len(lines) < 3:
        data = f.read(500)
        start += data
        lines = start.split(b"\n")
        if not data:
            break
    m = _match_file_encoding(lines[0])
    if m and m.group(1) != b'c_string_encoding':
        return m.group(2).decode('iso8859-1')
    elif len(lines) > 1:
        m = _match_file_encoding(lines[1])
        if m:
            return m.group(2).decode('iso8859-1')
    return "UTF-8"


def skip_bom(f):
    """
    Read past a BOM at the beginning of a source file.
    This could be added to the scanner, but it's *substantially* easier
    to keep it at this level.
    """
    if f.read(1) != u'\uFEFF':
        f.seek(0)


def open_source_file(source_filename, encoding=None, error_handling=None):
    stream = None
    try:
        if encoding is None:
            # Most of the time the encoding is not specified, so try hard to open the file only once.
            f = io.open(source_filename, 'rb')
            encoding = detect_opened_file_encoding(f)
            f.seek(0)
            stream = io.TextIOWrapper(f, encoding=encoding, errors=error_handling)
        else:
            stream = io.open(source_filename, encoding=encoding, errors=error_handling)

    except OSError:
        if os.path.exists(source_filename):
            raise  # File is there, but something went wrong reading from it.
        # Allow source files to be in zip files etc.
        try:
            loader = __loader__
            if source_filename.startswith(loader.archive):
                stream = open_source_from_loader(
                    loader, source_filename,
                    encoding, error_handling)
        except (NameError, AttributeError):
            pass

    if stream is None:
        raise FileNotFoundError(source_filename)
    skip_bom(stream)
    return stream


def open_source_from_loader(loader,
                            source_filename,
                            encoding=None, error_handling=None):
    nrmpath = os.path.normpath(source_filename)
    arcname = nrmpath[len(loader.archive)+1:]
    data = loader.get_data(arcname)
    return io.TextIOWrapper(io.BytesIO(data),
                            encoding=encoding,
                            errors=error_handling)


def str_to_number(value):
    # note: this expects a string as input that was accepted by the
    # parser already, with an optional "-" sign in front
    is_neg = False
    if value[:1] == '-':
        is_neg = True
        value = value[1:]
    if len(value) < 2:
        value = int(value, 0)
    elif value[0] == '0':
        literal_type = value[1]  # 0'o' - 0'b' - 0'x'
        if literal_type in 'xX':
            # hex notation ('0x1AF')
            value = int(value[2:], 16)
        elif literal_type in 'oO':
            # Py3 octal notation ('0o136')
            value = int(value[2:], 8)
        elif literal_type in 'bB':
            # Py3 binary notation ('0b101')
            value = int(value[2:], 2)
        else:
            # Py2 octal notation ('0136')
            value = int(value, 8)
    else:
        value = int(value, 0)
    return -value if is_neg else value


def long_literal(value):
    if isinstance(value, basestring):
        value = str_to_number(value)
    return not -2**31 <= value < 2**31


@cached_function
def get_cython_cache_dir():
    r"""
    Return the base directory containing Cython's caches.

    Priority:

    1. CYTHON_CACHE_DIR
    2. (OS X): ~/Library/Caches/Cython
       (posix not OS X): XDG_CACHE_HOME/cython if XDG_CACHE_HOME defined
    3. ~/.cython

    """
    if 'CYTHON_CACHE_DIR' in os.environ:
        return os.environ['CYTHON_CACHE_DIR']

    parent = None
    if os.name == 'posix':
        if sys.platform == 'darwin':
            parent = os.path.expanduser('~/Library/Caches')
        else:
            # this could fallback on ~/.cache
            parent = os.environ.get('XDG_CACHE_HOME')

    if parent and os.path.isdir(parent):
        return os.path.join(parent, 'cython')

    # last fallback: ~/.cython
    return os.path.expanduser(os.path.join('~', '.cython'))


@contextmanager
def captured_fd(stream=2, encoding=None):
    orig_stream = os.dup(stream)  # keep copy of original stream
    try:
        with tempfile.TemporaryFile(mode="a+b") as temp_file:
            def read_output(_output=[b'']):
                if not temp_file.closed:
                    temp_file.seek(0)
                    _output[0] = temp_file.read()
                return _output[0]

            os.dup2(temp_file.fileno(), stream)  # replace stream by copy of pipe
            try:
                def get_output():
                    result = read_output()
                    return result.decode(encoding) if encoding else result

                yield get_output
            finally:
                os.dup2(orig_stream, stream)  # restore original stream
                read_output()  # keep the output in case it's used after closing the context manager
    finally:
        os.close(orig_stream)


def print_bytes(s, header_text=None, end=b'\n', file=sys.stdout, flush=True):
    if header_text:
        file.write(header_text)  # note: text! => file.write() instead of out.write()
    file.flush()
    try:
        out = file.buffer  # Py3
    except AttributeError:
        out = file         # Py2
    out.write(s)
    if end:
        out.write(end)
    if flush:
        out.flush()

class LazyStr:
    def __init__(self, callback):
        self.callback = callback
    def __str__(self):
        return self.callback()
    def __repr__(self):
        return self.callback()
    def __add__(self, right):
        return self.callback() + right
    def __radd__(self, left):
        return left + self.callback()


class OrderedSet(object):
  def __init__(self, elements=()):
    self._list = []
    self._set = set()
    self.update(elements)
  def __iter__(self):
    return iter(self._list)
  def update(self, elements):
    for e in elements:
      self.add(e)
  def add(self, e):
    if e not in self._set:
      self._list.append(e)
      self._set.add(e)


# Class decorator that adds a metaclass and recreates the class with it.
# Copied from 'six'.
def add_metaclass(metaclass):
    """Class decorator for creating a class with a metaclass."""
    def wrapper(cls):
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get('__slots__')
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        orig_vars.pop('__dict__', None)
        orig_vars.pop('__weakref__', None)
        return metaclass(cls.__name__, cls.__bases__, orig_vars)
    return wrapper


def raise_error_if_module_name_forbidden(full_module_name):
    #it is bad idea to call the pyx-file cython.pyx, so fail early
    if full_module_name == 'cython' or full_module_name.startswith('cython.'):
        raise ValueError('cython is a special module, cannot be used as a module name')


def build_hex_version(version_string):
    """
    Parse and translate '4.3a1' into the readable hex representation '0x040300A1' (like PY_VERSION_HEX).
    """
    # First, parse '4.12a1' into [4, 12, 0, 0xA01].
    digits = []
    release_status = 0xF0
    for digit in re.split('([.abrc]+)', version_string):
        if digit in ('a', 'b', 'rc'):
            release_status = {'a': 0xA0, 'b': 0xB0, 'rc': 0xC0}[digit]
            digits = (digits + [0, 0])[:3]  # 1.2a1 -> 1.2.0a1
        elif digit != '.':
            digits.append(int(digit))
    digits = (digits + [0] * 3)[:4]
    digits[3] += release_status

    # Then, build a single hex value, two hex digits per version part.
    hexversion = 0
    for digit in digits:
        hexversion = (hexversion << 8) + digit

    return '0x%08X' % hexversion
