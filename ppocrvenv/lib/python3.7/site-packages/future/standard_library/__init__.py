"""
Python 3 reorganized the standard library (PEP 3108). This module exposes
several standard library modules to Python 2 under their new Python 3
names.

It is designed to be used as follows::

    from future import standard_library
    standard_library.install_aliases()

And then these normal Py3 imports work on both Py3 and Py2::

    import builtins
    import copyreg
    import queue
    import reprlib
    import socketserver
    import winreg    # on Windows only
    import test.support
    import html, html.parser, html.entites
    import http, http.client, http.server
    import http.cookies, http.cookiejar
    import urllib.parse, urllib.request, urllib.response, urllib.error, urllib.robotparser
    import xmlrpc.client, xmlrpc.server

    import _thread
    import _dummy_thread
    import _markupbase

    from itertools import filterfalse, zip_longest
    from sys import intern
    from collections import UserDict, UserList, UserString
    from collections import OrderedDict, Counter, ChainMap     # even on Py2.6
    from subprocess import getoutput, getstatusoutput
    from subprocess import check_output              # even on Py2.6

(The renamed modules and functions are still available under their old
names on Python 2.)

This is a cleaner alternative to this idiom (see
http://docs.pythonsprints.com/python3_porting/py-porting.html)::

    try:
        import queue
    except ImportError:
        import Queue as queue


Limitations
-----------
We don't currently support these modules, but would like to::

    import dbm
    import dbm.dumb
    import dbm.gnu
    import collections.abc  # on Py33
    import pickle     # should (optionally) bring in cPickle on Python 2

"""

from __future__ import absolute_import, division, print_function

import sys
import logging
import imp
import contextlib
import types
import copy
import os

# Make a dedicated logger; leave the root logger to be configured
# by the application.
flog = logging.getLogger('future_stdlib')
_formatter = logging.Formatter(logging.BASIC_FORMAT)
_handler = logging.StreamHandler()
_handler.setFormatter(_formatter)
flog.addHandler(_handler)
flog.setLevel(logging.WARN)

from future.utils import PY2, PY3

# The modules that are defined under the same names on Py3 but with
# different contents in a significant way (e.g. submodules) are:
#   pickle (fast one)
#   dbm
#   urllib
#   test
#   email

REPLACED_MODULES = set(['test', 'urllib', 'pickle', 'dbm'])  # add email and dbm when we support it

# The following module names are not present in Python 2.x, so they cause no
# potential clashes between the old and new names:
#   http
#   html
#   tkinter
#   xmlrpc
# Keys: Py2 / real module names
# Values: Py3 / simulated module names
RENAMES = {
           # 'cStringIO': 'io',  # there's a new io module in Python 2.6
                                 # that provides StringIO and BytesIO
           # 'StringIO': 'io',   # ditto
           # 'cPickle': 'pickle',
           '__builtin__': 'builtins',
           'copy_reg': 'copyreg',
           'Queue': 'queue',
           'future.moves.socketserver': 'socketserver',
           'ConfigParser': 'configparser',
           'repr': 'reprlib',
           # 'FileDialog': 'tkinter.filedialog',
           # 'tkFileDialog': 'tkinter.filedialog',
           # 'SimpleDialog': 'tkinter.simpledialog',
           # 'tkSimpleDialog': 'tkinter.simpledialog',
           # 'tkColorChooser': 'tkinter.colorchooser',
           # 'tkCommonDialog': 'tkinter.commondialog',
           # 'Dialog': 'tkinter.dialog',
           # 'Tkdnd': 'tkinter.dnd',
           # 'tkFont': 'tkinter.font',
           # 'tkMessageBox': 'tkinter.messagebox',
           # 'ScrolledText': 'tkinter.scrolledtext',
           # 'Tkconstants': 'tkinter.constants',
           # 'Tix': 'tkinter.tix',
           # 'ttk': 'tkinter.ttk',
           # 'Tkinter': 'tkinter',
           '_winreg': 'winreg',
           'thread': '_thread',
           'dummy_thread': '_dummy_thread',
           # 'anydbm': 'dbm',   # causes infinite import loop
           # 'whichdb': 'dbm',  # causes infinite import loop
           # anydbm and whichdb are handled by fix_imports2
           # 'dbhash': 'dbm.bsd',
           # 'dumbdbm': 'dbm.dumb',
           # 'dbm': 'dbm.ndbm',
           # 'gdbm': 'dbm.gnu',
           'future.moves.xmlrpc': 'xmlrpc',
           # 'future.backports.email': 'email',    # for use by urllib
           # 'DocXMLRPCServer': 'xmlrpc.server',
           # 'SimpleXMLRPCServer': 'xmlrpc.server',
           # 'httplib': 'http.client',
           # 'htmlentitydefs' : 'html.entities',
           # 'HTMLParser' : 'html.parser',
           # 'Cookie': 'http.cookies',
           # 'cookielib': 'http.cookiejar',
           # 'BaseHTTPServer': 'http.server',
           # 'SimpleHTTPServer': 'http.server',
           # 'CGIHTTPServer': 'http.server',
           # 'future.backports.test': 'test',  # primarily for renaming test_support to support
           # 'commands': 'subprocess',
           # 'urlparse' : 'urllib.parse',
           # 'robotparser' : 'urllib.robotparser',
           # 'abc': 'collections.abc',   # for Py33
           # 'future.utils.six.moves.html': 'html',
           # 'future.utils.six.moves.http': 'http',
           'future.moves.html': 'html',
           'future.moves.http': 'http',
           # 'future.backports.urllib': 'urllib',
           # 'future.utils.six.moves.urllib': 'urllib',
           'future.moves._markupbase': '_markupbase',
          }


# It is complicated and apparently brittle to mess around with the
# ``sys.modules`` cache in order to support "import urllib" meaning two
# different things (Py2.7 urllib and backported Py3.3-like urllib) in different
# contexts. So we require explicit imports for these modules.
assert len(set(RENAMES.values()) & set(REPLACED_MODULES)) == 0


# Harmless renames that we can insert.
# These modules need names from elsewhere being added to them:
#   subprocess: should provide getoutput and other fns from commands
#               module but these fns are missing: getstatus, mk2arg,
#               mkarg
#   re:         needs an ASCII constant that works compatibly with Py3

# etc: see lib2to3/fixes/fix_imports.py

# (New module name, new object name, old module name, old object name)
MOVES = [('collections', 'UserList', 'UserList', 'UserList'),
         ('collections', 'UserDict', 'UserDict', 'UserDict'),
         ('collections', 'UserString','UserString', 'UserString'),
         ('collections', 'ChainMap', 'future.backports.misc', 'ChainMap'),
         ('itertools', 'filterfalse','itertools', 'ifilterfalse'),
         ('itertools', 'zip_longest','itertools', 'izip_longest'),
         ('sys', 'intern','__builtin__', 'intern'),
         # The re module has no ASCII flag in Py2, but this is the default.
         # Set re.ASCII to a zero constant. stat.ST_MODE just happens to be one
         # (and it exists on Py2.6+).
         ('re', 'ASCII','stat', 'ST_MODE'),
         ('base64', 'encodebytes','base64', 'encodestring'),
         ('base64', 'decodebytes','base64', 'decodestring'),
         ('subprocess', 'getoutput', 'commands', 'getoutput'),
         ('subprocess', 'getstatusoutput', 'commands', 'getstatusoutput'),
         ('subprocess', 'check_output', 'future.backports.misc', 'check_output'),
         ('math', 'ceil', 'future.backports.misc', 'ceil'),
         ('collections', 'OrderedDict', 'future.backports.misc', 'OrderedDict'),
         ('collections', 'Counter', 'future.backports.misc', 'Counter'),
         ('collections', 'ChainMap', 'future.backports.misc', 'ChainMap'),
         ('itertools', 'count', 'future.backports.misc', 'count'),
         ('reprlib', 'recursive_repr', 'future.backports.misc', 'recursive_repr'),
         ('functools', 'cmp_to_key', 'future.backports.misc', 'cmp_to_key'),

# This is no use, since "import urllib.request" etc. still fails:
#          ('urllib', 'error', 'future.moves.urllib', 'error'),
#          ('urllib', 'parse', 'future.moves.urllib', 'parse'),
#          ('urllib', 'request', 'future.moves.urllib', 'request'),
#          ('urllib', 'response', 'future.moves.urllib', 'response'),
#          ('urllib', 'robotparser', 'future.moves.urllib', 'robotparser'),
        ]


# A minimal example of an import hook:
# class WarnOnImport(object):
#     def __init__(self, *args):
#         self.module_names = args
#
#     def find_module(self, fullname, path=None):
#         if fullname in self.module_names:
#             self.path = path
#             return self
#         return None
#
#     def load_module(self, name):
#         if name in sys.modules:
#             return sys.modules[name]
#         module_info = imp.find_module(name, self.path)
#         module = imp.load_module(name, *module_info)
#         sys.modules[name] = module
#         flog.warning("Imported deprecated module %s", name)
#         return module


class RenameImport(object):
    """
    A class for import hooks mapping Py3 module names etc. to the Py2 equivalents.
    """
    # Different RenameImport classes are created when importing this module from
    # different source files. This causes isinstance(hook, RenameImport) checks
    # to produce inconsistent results. We add this RENAMER attribute here so
    # remove_hooks() and install_hooks() can find instances of these classes
    # easily:
    RENAMER = True

    def __init__(self, old_to_new):
        '''
        Pass in a dictionary-like object mapping from old names to new
        names. E.g. {'ConfigParser': 'configparser', 'cPickle': 'pickle'}
        '''
        self.old_to_new = old_to_new
        both = set(old_to_new.keys()) & set(old_to_new.values())
        assert (len(both) == 0 and
                len(set(old_to_new.values())) == len(old_to_new.values())), \
               'Ambiguity in renaming (handler not implemented)'
        self.new_to_old = dict((new, old) for (old, new) in old_to_new.items())

    def find_module(self, fullname, path=None):
        # Handles hierarchical importing: package.module.module2
        new_base_names = set([s.split('.')[0] for s in self.new_to_old])
        # Before v0.12: Was: if fullname in set(self.old_to_new) | new_base_names:
        if fullname in new_base_names:
            return self
        return None

    def load_module(self, name):
        path = None
        if name in sys.modules:
            return sys.modules[name]
        elif name in self.new_to_old:
            # New name. Look up the corresponding old (Py2) name:
            oldname = self.new_to_old[name]
            module = self._find_and_load_module(oldname)
            # module.__future_module__ = True
        else:
            module = self._find_and_load_module(name)
        # In any case, make it available under the requested (Py3) name
        sys.modules[name] = module
        return module

    def _find_and_load_module(self, name, path=None):
        """
        Finds and loads it. But if there's a . in the name, handles it
        properly.
        """
        bits = name.split('.')
        while len(bits) > 1:
            # Treat the first bit as a package
            packagename = bits.pop(0)
            package = self._find_and_load_module(packagename, path)
            try:
                path = package.__path__
            except AttributeError:
                # This could be e.g. moves.
                flog.debug('Package {0} has no __path__.'.format(package))
                if name in sys.modules:
                    return sys.modules[name]
                flog.debug('What to do here?')

        name = bits[0]
        module_info = imp.find_module(name, path)
        return imp.load_module(name, *module_info)


class hooks(object):
    """
    Acts as a context manager. Saves the state of sys.modules and restores it
    after the 'with' block.

    Use like this:

    >>> from future import standard_library
    >>> with standard_library.hooks():
    ...     import http.client
    >>> import requests

    For this to work, http.client will be scrubbed from sys.modules after the
    'with' block. That way the modules imported in the 'with' block will
    continue to be accessible in the current namespace but not from any
    imported modules (like requests).
    """
    def __enter__(self):
        # flog.debug('Entering hooks context manager')
        self.old_sys_modules = copy.copy(sys.modules)
        self.hooks_were_installed = detect_hooks()
        # self.scrubbed = scrub_py2_sys_modules()
        install_hooks()
        return self

    def __exit__(self, *args):
        # flog.debug('Exiting hooks context manager')
        # restore_sys_modules(self.scrubbed)
        if not self.hooks_were_installed:
            remove_hooks()
        # scrub_future_sys_modules()

# Sanity check for is_py2_stdlib_module(): We aren't replacing any
# builtin modules names:
if PY2:
    assert len(set(RENAMES.values()) & set(sys.builtin_module_names)) == 0


def is_py2_stdlib_module(m):
    """
    Tries to infer whether the module m is from the Python 2 standard library.
    This may not be reliable on all systems.
    """
    if PY3:
        return False
    if not 'stdlib_path' in is_py2_stdlib_module.__dict__:
        stdlib_files = [contextlib.__file__, os.__file__, copy.__file__]
        stdlib_paths = [os.path.split(f)[0] for f in stdlib_files]
        if not len(set(stdlib_paths)) == 1:
            # This seems to happen on travis-ci.org. Very strange. We'll try to
            # ignore it.
            flog.warn('Multiple locations found for the Python standard '
                         'library: %s' % stdlib_paths)
        # Choose the first one arbitrarily
        is_py2_stdlib_module.stdlib_path = stdlib_paths[0]

    if m.__name__ in sys.builtin_module_names:
        return True

    if hasattr(m, '__file__'):
        modpath = os.path.split(m.__file__)
        if (modpath[0].startswith(is_py2_stdlib_module.stdlib_path) and
            'site-packages' not in modpath[0]):
            return True

    return False


def scrub_py2_sys_modules():
    """
    Removes any Python 2 standard library modules from ``sys.modules`` that
    would interfere with Py3-style imports using import hooks. Examples are
    modules with the same names (like urllib or email).

    (Note that currently import hooks are disabled for modules like these
    with ambiguous names anyway ...)
    """
    if PY3:
        return {}
    scrubbed = {}
    for modulename in REPLACED_MODULES & set(RENAMES.keys()):
        if not modulename in sys.modules:
            continue

        module = sys.modules[modulename]

        if is_py2_stdlib_module(module):
            flog.debug('Deleting (Py2) {} from sys.modules'.format(modulename))
            scrubbed[modulename] = sys.modules[modulename]
            del sys.modules[modulename]
    return scrubbed


def scrub_future_sys_modules():
    """
    Deprecated.
    """
    return {}

class suspend_hooks(object):
    """
    Acts as a context manager. Use like this:

    >>> from future import standard_library
    >>> standard_library.install_hooks()
    >>> import http.client
    >>> # ...
    >>> with standard_library.suspend_hooks():
    >>>     import requests     # incompatible with ``future``'s standard library hooks

    If the hooks were disabled before the context, they are not installed when
    the context is left.
    """
    def __enter__(self):
        self.hooks_were_installed = detect_hooks()
        remove_hooks()
        # self.scrubbed = scrub_future_sys_modules()
        return self

    def __exit__(self, *args):
        if self.hooks_were_installed:
            install_hooks()
        # restore_sys_modules(self.scrubbed)


def restore_sys_modules(scrubbed):
    """
    Add any previously scrubbed modules back to the sys.modules cache,
    but only if it's safe to do so.
    """
    clash = set(sys.modules) & set(scrubbed)
    if len(clash) != 0:
        # If several, choose one arbitrarily to raise an exception about
        first = list(clash)[0]
        raise ImportError('future module {} clashes with Py2 module'
                          .format(first))
    sys.modules.update(scrubbed)


def install_aliases():
    """
    Monkey-patches the standard library in Py2.6/7 to provide
    aliases for better Py3 compatibility.
    """
    if PY3:
        return
    # if hasattr(install_aliases, 'run_already'):
    #     return
    for (newmodname, newobjname, oldmodname, oldobjname) in MOVES:
        __import__(newmodname)
        # We look up the module in sys.modules because __import__ just returns the
        # top-level package:
        newmod = sys.modules[newmodname]
        # newmod.__future_module__ = True

        __import__(oldmodname)
        oldmod = sys.modules[oldmodname]

        obj = getattr(oldmod, oldobjname)
        setattr(newmod, newobjname, obj)

    # Hack for urllib so it appears to have the same structure on Py2 as on Py3
    import urllib
    from future.backports.urllib import request
    from future.backports.urllib import response
    from future.backports.urllib import parse
    from future.backports.urllib import error
    from future.backports.urllib import robotparser
    urllib.request = request
    urllib.response = response
    urllib.parse = parse
    urllib.error = error
    urllib.robotparser = robotparser
    sys.modules['urllib.request'] = request
    sys.modules['urllib.response'] = response
    sys.modules['urllib.parse'] = parse
    sys.modules['urllib.error'] = error
    sys.modules['urllib.robotparser'] = robotparser

    # Patch the test module so it appears to have the same structure on Py2 as on Py3
    try:
        import test
    except ImportError:
        pass
    try:
        from future.moves.test import support
    except ImportError:
        pass
    else:
        test.support = support
        sys.modules['test.support'] = support

    # Patch the dbm module so it appears to have the same structure on Py2 as on Py3
    try:
        import dbm
    except ImportError:
        pass
    else:
        from future.moves.dbm import dumb
        dbm.dumb = dumb
        sys.modules['dbm.dumb'] = dumb
        try:
            from future.moves.dbm import gnu
        except ImportError:
            pass
        else:
            dbm.gnu = gnu
            sys.modules['dbm.gnu'] = gnu
        try:
            from future.moves.dbm import ndbm
        except ImportError:
            pass
        else:
            dbm.ndbm = ndbm
            sys.modules['dbm.ndbm'] = ndbm

    # install_aliases.run_already = True


def install_hooks():
    """
    This function installs the future.standard_library import hook into
    sys.meta_path.
    """
    if PY3:
        return

    install_aliases()

    flog.debug('sys.meta_path was: {0}'.format(sys.meta_path))
    flog.debug('Installing hooks ...')

    # Add it unless it's there already
    newhook = RenameImport(RENAMES)
    if not detect_hooks():
        sys.meta_path.append(newhook)
    flog.debug('sys.meta_path is now: {0}'.format(sys.meta_path))


def enable_hooks():
    """
    Deprecated. Use install_hooks() instead. This will be removed by
    ``future`` v1.0.
    """
    install_hooks()


def remove_hooks(scrub_sys_modules=False):
    """
    This function removes the import hook from sys.meta_path.
    """
    if PY3:
        return
    flog.debug('Uninstalling hooks ...')
    # Loop backwards, so deleting items keeps the ordering:
    for i, hook in list(enumerate(sys.meta_path))[::-1]:
        if hasattr(hook, 'RENAMER'):
            del sys.meta_path[i]

    # Explicit is better than implicit. In the future the interface should
    # probably change so that scrubbing the import hooks requires a separate
    # function call. Left as is for now for backward compatibility with
    # v0.11.x.
    if scrub_sys_modules:
        scrub_future_sys_modules()


def disable_hooks():
    """
    Deprecated. Use remove_hooks() instead. This will be removed by
    ``future`` v1.0.
    """
    remove_hooks()


def detect_hooks():
    """
    Returns True if the import hooks are installed, False if not.
    """
    flog.debug('Detecting hooks ...')
    present = any([hasattr(hook, 'RENAMER') for hook in sys.meta_path])
    if present:
        flog.debug('Detected.')
    else:
        flog.debug('Not detected.')
    return present


# As of v0.12, this no longer happens implicitly:
# if not PY3:
#     install_hooks()


if not hasattr(sys, 'py2_modules'):
    sys.py2_modules = {}

def cache_py2_modules():
    """
    Currently this function is unneeded, as we are not attempting to provide import hooks
    for modules with ambiguous names: email, urllib, pickle.
    """
    if len(sys.py2_modules) != 0:
        return
    assert not detect_hooks()
    import urllib
    sys.py2_modules['urllib'] = urllib

    import email
    sys.py2_modules['email'] = email

    import pickle
    sys.py2_modules['pickle'] = pickle

    # Not all Python installations have test module. (Anaconda doesn't, for example.)
    # try:
    #     import test
    # except ImportError:
    #     sys.py2_modules['test'] = None
    # sys.py2_modules['test'] = test

    # import dbm
    # sys.py2_modules['dbm'] = dbm


def import_(module_name, backport=False):
    """
    Pass a (potentially dotted) module name of a Python 3 standard library
    module. This function imports the module compatibly on Py2 and Py3 and
    returns the top-level module.

    Example use:
        >>> http = import_('http.client')
        >>> http = import_('http.server')
        >>> urllib = import_('urllib.request')

    Then:
        >>> conn = http.client.HTTPConnection(...)
        >>> response = urllib.request.urlopen('http://mywebsite.com')
        >>> # etc.

    Use as follows:
        >>> package_name = import_(module_name)

    On Py3, equivalent to this:

        >>> import module_name

    On Py2, equivalent to this if backport=False:

        >>> from future.moves import module_name

    or to this if backport=True:

        >>> from future.backports import module_name

    except that it also handles dotted module names such as ``http.client``
    The effect then is like this:

        >>> from future.backports import module
        >>> from future.backports.module import submodule
        >>> module.submodule = submodule

    Note that this would be a SyntaxError in Python:

        >>> from future.backports import http.client

    """
    # Python 2.6 doesn't have importlib in the stdlib, so it requires
    # the backported ``importlib`` package from PyPI as a dependency to use
    # this function:
    import importlib

    if PY3:
        return __import__(module_name)
    else:
        # client.blah = blah
        # Then http.client = client
        # etc.
        if backport:
            prefix = 'future.backports'
        else:
            prefix = 'future.moves'
        parts = prefix.split('.') + module_name.split('.')

        modules = []
        for i, part in enumerate(parts):
            sofar = '.'.join(parts[:i+1])
            modules.append(importlib.import_module(sofar))
        for i, part in reversed(list(enumerate(parts))):
            if i == 0:
                break
            setattr(modules[i-1], part, modules[i])

        # Return the next-most top-level module after future.backports / future.moves:
        return modules[2]


def from_import(module_name, *symbol_names, **kwargs):
    """
    Example use:
        >>> HTTPConnection = from_import('http.client', 'HTTPConnection')
        >>> HTTPServer = from_import('http.server', 'HTTPServer')
        >>> urlopen, urlparse = from_import('urllib.request', 'urlopen', 'urlparse')

    Equivalent to this on Py3:

        >>> from module_name import symbol_names[0], symbol_names[1], ...

    and this on Py2:

        >>> from future.moves.module_name import symbol_names[0], ...

    or:

        >>> from future.backports.module_name import symbol_names[0], ...

    except that it also handles dotted module names such as ``http.client``.
    """

    if PY3:
        return __import__(module_name)
    else:
        if 'backport' in kwargs and bool(kwargs['backport']):
            prefix = 'future.backports'
        else:
            prefix = 'future.moves'
        parts = prefix.split('.') + module_name.split('.')
        module = importlib.import_module(prefix + '.' + module_name)
        output = [getattr(module, name) for name in symbol_names]
        if len(output) == 1:
            return output[0]
        else:
            return output


class exclude_local_folder_imports(object):
    """
    A context-manager that prevents standard library modules like configparser
    from being imported from the local python-future source folder on Py3.

    (This was need prior to v0.16.0 because the presence of a configparser
    folder would otherwise have prevented setuptools from running on Py3. Maybe
    it's not needed any more?)
    """
    def __init__(self, *args):
        assert len(args) > 0
        self.module_names = args
        # Disallow dotted module names like http.client:
        if any(['.' in m for m in self.module_names]):
            raise NotImplementedError('Dotted module names are not supported')

    def __enter__(self):
        self.old_sys_path = copy.copy(sys.path)
        self.old_sys_modules = copy.copy(sys.modules)
        if sys.version_info[0] < 3:
            return
        # The presence of all these indicates we've found our source folder,
        # because `builtins` won't have been installed in site-packages by setup.py:
        FUTURE_SOURCE_SUBFOLDERS = ['future', 'past', 'libfuturize', 'libpasteurize', 'builtins']

        # Look for the future source folder:
        for folder in self.old_sys_path:
            if all([os.path.exists(os.path.join(folder, subfolder))
                    for subfolder in FUTURE_SOURCE_SUBFOLDERS]):
                # Found it. Remove it.
                sys.path.remove(folder)

        # Ensure we import the system module:
        for m in self.module_names:
            # Delete the module and any submodules from sys.modules:
            # for key in list(sys.modules):
            #     if key == m or key.startswith(m + '.'):
            #         try:
            #             del sys.modules[key]
            #         except KeyError:
            #             pass
            try:
                module = __import__(m, level=0)
            except ImportError:
                # There's a problem importing the system module. E.g. the
                # winreg module is not available except on Windows.
                pass

    def __exit__(self, *args):
        # Restore sys.path and sys.modules:
        sys.path = self.old_sys_path
        for m in set(self.old_sys_modules.keys()) - set(sys.modules.keys()):
            sys.modules[m] = self.old_sys_modules[m]

TOP_LEVEL_MODULES = ['builtins',
                     'copyreg',
                     'html',
                     'http',
                     'queue',
                     'reprlib',
                     'socketserver',
                     'test',
                     'tkinter',
                     'winreg',
                     'xmlrpc',
                     '_dummy_thread',
                     '_markupbase',
                     '_thread',
                    ]

def import_top_level_modules():
    with exclude_local_folder_imports(*TOP_LEVEL_MODULES):
        for m in TOP_LEVEL_MODULES:
            try:
                __import__(m)
            except ImportError:     # e.g. winreg
                pass
