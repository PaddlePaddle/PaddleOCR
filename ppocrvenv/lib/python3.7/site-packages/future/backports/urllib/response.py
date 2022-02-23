"""Response classes used by urllib.

The base class, addbase, defines a minimal file-like interface,
including read() and readline().  The typical response object is an
addinfourl instance, which defines an info() method that returns
headers and a geturl() method that returns the url.
"""
from __future__ import absolute_import, division, unicode_literals
from future.builtins import object

class addbase(object):
    """Base class for addinfo and addclosehook."""

    # XXX Add a method to expose the timeout on the underlying socket?

    def __init__(self, fp):
        # TODO(jhylton): Is there a better way to delegate using io?
        self.fp = fp
        self.read = self.fp.read
        self.readline = self.fp.readline
        # TODO(jhylton): Make sure an object with readlines() is also iterable
        if hasattr(self.fp, "readlines"):
            self.readlines = self.fp.readlines
        if hasattr(self.fp, "fileno"):
            self.fileno = self.fp.fileno
        else:
            self.fileno = lambda: None

    def __iter__(self):
        # Assigning `__iter__` to the instance doesn't work as intended
        # because the iter builtin does something like `cls.__iter__(obj)`
        # and thus fails to find the _bound_ method `obj.__iter__`.
        # Returning just `self.fp` works for built-in file objects but
        # might not work for general file-like objects.
        return iter(self.fp)

    def __repr__(self):
        return '<%s at %r whose fp = %r>' % (self.__class__.__name__,
                                             id(self), self.fp)

    def close(self):
        if self.fp:
            self.fp.close()
        self.fp = None
        self.read = None
        self.readline = None
        self.readlines = None
        self.fileno = None
        self.__iter__ = None
        self.__next__ = None

    def __enter__(self):
        if self.fp is None:
            raise ValueError("I/O operation on closed file")
        return self

    def __exit__(self, type, value, traceback):
        self.close()

class addclosehook(addbase):
    """Class to add a close hook to an open file."""

    def __init__(self, fp, closehook, *hookargs):
        addbase.__init__(self, fp)
        self.closehook = closehook
        self.hookargs = hookargs

    def close(self):
        if self.closehook:
            self.closehook(*self.hookargs)
            self.closehook = None
            self.hookargs = None
        addbase.close(self)

class addinfo(addbase):
    """class to add an info() method to an open file."""

    def __init__(self, fp, headers):
        addbase.__init__(self, fp)
        self.headers = headers

    def info(self):
        return self.headers

class addinfourl(addbase):
    """class to add info() and geturl() methods to an open file."""

    def __init__(self, fp, headers, url, code=None):
        addbase.__init__(self, fp)
        self.headers = headers
        self.url = url
        self.code = code

    def info(self):
        return self.headers

    def getcode(self):
        return self.code

    def geturl(self):
        return self.url

del absolute_import, division, unicode_literals, object
