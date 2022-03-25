"""
"""
import warnings
import os
import sys
import posixpath
import fnmatch
import py

# Moved from local.py.
iswin32 = sys.platform == "win32" or (getattr(os, '_name', False) == 'nt')

try:
    # FileNotFoundError might happen in py34, and is not available with py27.
    import_errors = (ImportError, FileNotFoundError)
except NameError:
    import_errors = (ImportError,)

try:
    from os import fspath
except ImportError:
    def fspath(path):
        """
        Return the string representation of the path.
        If str or bytes is passed in, it is returned unchanged.
        This code comes from PEP 519, modified to support earlier versions of
        python.

        This is required for python < 3.6.
        """
        if isinstance(path, (py.builtin.text, py.builtin.bytes)):
            return path

        # Work from the object's type to match method resolution of other magic
        # methods.
        path_type = type(path)
        try:
            return path_type.__fspath__(path)
        except AttributeError:
            if hasattr(path_type, '__fspath__'):
                raise
            try:
                import pathlib
            except import_errors:
                pass
            else:
                if isinstance(path, pathlib.PurePath):
                    return py.builtin.text(path)

            raise TypeError("expected str, bytes or os.PathLike object, not "
                            + path_type.__name__)

class Checkers:
    _depend_on_existence = 'exists', 'link', 'dir', 'file'

    def __init__(self, path):
        self.path = path

    def dir(self):
        raise NotImplementedError

    def file(self):
        raise NotImplementedError

    def dotfile(self):
        return self.path.basename.startswith('.')

    def ext(self, arg):
        if not arg.startswith('.'):
            arg = '.' + arg
        return self.path.ext == arg

    def exists(self):
        raise NotImplementedError

    def basename(self, arg):
        return self.path.basename == arg

    def basestarts(self, arg):
        return self.path.basename.startswith(arg)

    def relto(self, arg):
        return self.path.relto(arg)

    def fnmatch(self, arg):
        return self.path.fnmatch(arg)

    def endswith(self, arg):
        return str(self.path).endswith(arg)

    def _evaluate(self, kw):
        for name, value in kw.items():
            invert = False
            meth = None
            try:
                meth = getattr(self, name)
            except AttributeError:
                if name[:3] == 'not':
                    invert = True
                    try:
                        meth = getattr(self, name[3:])
                    except AttributeError:
                        pass
            if meth is None:
                raise TypeError(
                    "no %r checker available for %r" % (name, self.path))
            try:
                if py.code.getrawcode(meth).co_argcount > 1:
                    if (not meth(value)) ^ invert:
                        return False
                else:
                    if bool(value) ^ bool(meth()) ^ invert:
                        return False
            except (py.error.ENOENT, py.error.ENOTDIR, py.error.EBUSY):
                # EBUSY feels not entirely correct,
                # but its kind of necessary since ENOMEDIUM
                # is not accessible in python
                for name in self._depend_on_existence:
                    if name in kw:
                        if kw.get(name):
                            return False
                    name = 'not' + name
                    if name in kw:
                        if not kw.get(name):
                            return False
        return True

class NeverRaised(Exception):
    pass

class PathBase(object):
    """ shared implementation for filesystem path objects."""
    Checkers = Checkers

    def __div__(self, other):
        return self.join(fspath(other))
    __truediv__ = __div__ # py3k

    def basename(self):
        """ basename part of path. """
        return self._getbyspec('basename')[0]
    basename = property(basename, None, None, basename.__doc__)

    def dirname(self):
        """ dirname part of path. """
        return self._getbyspec('dirname')[0]
    dirname = property(dirname, None, None, dirname.__doc__)

    def purebasename(self):
        """ pure base name of the path."""
        return self._getbyspec('purebasename')[0]
    purebasename = property(purebasename, None, None, purebasename.__doc__)

    def ext(self):
        """ extension of the path (including the '.')."""
        return self._getbyspec('ext')[0]
    ext = property(ext, None, None, ext.__doc__)

    def dirpath(self, *args, **kwargs):
        """ return the directory path joined with any given path arguments.  """
        return self.new(basename='').join(*args, **kwargs)

    def read_binary(self):
        """ read and return a bytestring from reading the path. """
        with self.open('rb') as f:
            return f.read()

    def read_text(self, encoding):
        """ read and return a Unicode string from reading the path. """
        with self.open("r", encoding=encoding) as f:
            return f.read()


    def read(self, mode='r'):
        """ read and return a bytestring from reading the path. """
        with self.open(mode) as f:
            return f.read()

    def readlines(self, cr=1):
        """ read and return a list of lines from the path. if cr is False, the
newline will be removed from the end of each line. """
        if sys.version_info < (3, ):
            mode = 'rU'
        else:  # python 3 deprecates mode "U" in favor of "newline" option
            mode = 'r'

        if not cr:
            content = self.read(mode)
            return content.split('\n')
        else:
            f = self.open(mode)
            try:
                return f.readlines()
            finally:
                f.close()

    def load(self):
        """ (deprecated) return object unpickled from self.read() """
        f = self.open('rb')
        try:
            import pickle
            return py.error.checked_call(pickle.load, f)
        finally:
            f.close()

    def move(self, target):
        """ move this path to target. """
        if target.relto(self):
            raise py.error.EINVAL(
                target,
                "cannot move path into a subdirectory of itself")
        try:
            self.rename(target)
        except py.error.EXDEV:  # invalid cross-device link
            self.copy(target)
            self.remove()

    def __repr__(self):
        """ return a string representation of this path. """
        return repr(str(self))

    def check(self, **kw):
        """ check a path for existence and properties.

            Without arguments, return True if the path exists, otherwise False.

            valid checkers::

                file=1    # is a file
                file=0    # is not a file (may not even exist)
                dir=1     # is a dir
                link=1    # is a link
                exists=1  # exists

            You can specify multiple checker definitions, for example::

                path.check(file=1, link=1)  # a link pointing to a file
        """
        if not kw:
            kw = {'exists': 1}
        return self.Checkers(self)._evaluate(kw)

    def fnmatch(self, pattern):
        """return true if the basename/fullname matches the glob-'pattern'.

        valid pattern characters::

            *       matches everything
            ?       matches any single character
            [seq]   matches any character in seq
            [!seq]  matches any char not in seq

        If the pattern contains a path-separator then the full path
        is used for pattern matching and a '*' is prepended to the
        pattern.

        if the pattern doesn't contain a path-separator the pattern
        is only matched against the basename.
        """
        return FNMatcher(pattern)(self)

    def relto(self, relpath):
        """ return a string which is the relative part of the path
        to the given 'relpath'.
        """
        if not isinstance(relpath, (str, PathBase)):
            raise TypeError("%r: not a string or path object" %(relpath,))
        strrelpath = str(relpath)
        if strrelpath and strrelpath[-1] != self.sep:
            strrelpath += self.sep
        #assert strrelpath[-1] == self.sep
        #assert strrelpath[-2] != self.sep
        strself = self.strpath
        if sys.platform == "win32" or getattr(os, '_name', None) == 'nt':
            if os.path.normcase(strself).startswith(
               os.path.normcase(strrelpath)):
                return strself[len(strrelpath):]
        elif strself.startswith(strrelpath):
            return strself[len(strrelpath):]
        return ""

    def ensure_dir(self, *args):
        """ ensure the path joined with args is a directory. """
        return self.ensure(*args, **{"dir": True})

    def bestrelpath(self, dest):
        """ return a string which is a relative path from self
            (assumed to be a directory) to dest such that
            self.join(bestrelpath) == dest and if not such
            path can be determined return dest.
        """
        try:
            if self == dest:
                return os.curdir
            base = self.common(dest)
            if not base:  # can be the case on windows
                return str(dest)
            self2base = self.relto(base)
            reldest = dest.relto(base)
            if self2base:
                n = self2base.count(self.sep) + 1
            else:
                n = 0
            l = [os.pardir] * n
            if reldest:
                l.append(reldest)
            target = dest.sep.join(l)
            return target
        except AttributeError:
            return str(dest)

    def exists(self):
        return self.check()

    def isdir(self):
        return self.check(dir=1)

    def isfile(self):
        return self.check(file=1)

    def parts(self, reverse=False):
        """ return a root-first list of all ancestor directories
            plus the path itself.
        """
        current = self
        l = [self]
        while 1:
            last = current
            current = current.dirpath()
            if last == current:
                break
            l.append(current)
        if not reverse:
            l.reverse()
        return l

    def common(self, other):
        """ return the common part shared with the other path
            or None if there is no common part.
        """
        last = None
        for x, y in zip(self.parts(), other.parts()):
            if x != y:
                return last
            last = x
        return last

    def __add__(self, other):
        """ return new path object with 'other' added to the basename"""
        return self.new(basename=self.basename+str(other))

    def __cmp__(self, other):
        """ return sort value (-1, 0, +1). """
        try:
            return cmp(self.strpath, other.strpath)
        except AttributeError:
            return cmp(str(self), str(other)) # self.path, other.path)

    def __lt__(self, other):
        try:
            return self.strpath < other.strpath
        except AttributeError:
            return str(self) < str(other)

    def visit(self, fil=None, rec=None, ignore=NeverRaised, bf=False, sort=False):
        """ yields all paths below the current one

            fil is a filter (glob pattern or callable), if not matching the
            path will not be yielded, defaulting to None (everything is
            returned)

            rec is a filter (glob pattern or callable) that controls whether
            a node is descended, defaulting to None

            ignore is an Exception class that is ignoredwhen calling dirlist()
            on any of the paths (by default, all exceptions are reported)

            bf if True will cause a breadthfirst search instead of the
            default depthfirst. Default: False

            sort if True will sort entries within each directory level.
        """
        for x in Visitor(fil, rec, ignore, bf, sort).gen(self):
            yield x

    def _sortlist(self, res, sort):
        if sort:
            if hasattr(sort, '__call__'):
                warnings.warn(DeprecationWarning(
                    "listdir(sort=callable) is deprecated and breaks on python3"
                ), stacklevel=3)
                res.sort(sort)
            else:
                res.sort()

    def samefile(self, other):
        """ return True if other refers to the same stat object as self. """
        return self.strpath == str(other)

    def __fspath__(self):
        return self.strpath

class Visitor:
    def __init__(self, fil, rec, ignore, bf, sort):
        if isinstance(fil, py.builtin._basestring):
            fil = FNMatcher(fil)
        if isinstance(rec, py.builtin._basestring):
            self.rec = FNMatcher(rec)
        elif not hasattr(rec, '__call__') and rec:
            self.rec = lambda path: True
        else:
            self.rec = rec
        self.fil = fil
        self.ignore = ignore
        self.breadthfirst = bf
        self.optsort = sort and sorted or (lambda x: x)

    def gen(self, path):
        try:
            entries = path.listdir()
        except self.ignore:
            return
        rec = self.rec
        dirs = self.optsort([p for p in entries
                    if p.check(dir=1) and (rec is None or rec(p))])
        if not self.breadthfirst:
            for subdir in dirs:
                for p in self.gen(subdir):
                    yield p
        for p in self.optsort(entries):
            if self.fil is None or self.fil(p):
                yield p
        if self.breadthfirst:
            for subdir in dirs:
                for p in self.gen(subdir):
                    yield p

class FNMatcher:
    def __init__(self, pattern):
        self.pattern = pattern

    def __call__(self, path):
        pattern = self.pattern

        if (pattern.find(path.sep) == -1 and
        iswin32 and
        pattern.find(posixpath.sep) != -1):
            # Running on Windows, the pattern has no Windows path separators,
            # and the pattern has one or more Posix path separators. Replace
            # the Posix path separators with the Windows path separator.
            pattern = pattern.replace(posixpath.sep, path.sep)

        if pattern.find(path.sep) == -1:
            name = path.basename
        else:
            name = str(path) # path.strpath # XXX svn?
            if not os.path.isabs(pattern):
                pattern = '*' + path.sep + pattern
        return fnmatch.fnmatch(name, pattern)
