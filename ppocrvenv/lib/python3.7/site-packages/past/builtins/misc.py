from __future__ import unicode_literals

import inspect

from future.utils import PY2, PY3, exec_

if PY2:
    from collections import Mapping
else:
    from collections.abc import Mapping

if PY3:
    import builtins
    from collections.abc import Mapping

    def apply(f, *args, **kw):
        return f(*args, **kw)

    from past.builtins import str as oldstr

    def chr(i):
        """
        Return a byte-string of one character with ordinal i; 0 <= i <= 256
        """
        return oldstr(bytes((i,)))

    def cmp(x, y):
        """
        cmp(x, y) -> integer

        Return negative if x<y, zero if x==y, positive if x>y.
        """
        return (x > y) - (x < y)

    from sys import intern

    def oct(number):
        """oct(number) -> string

        Return the octal representation of an integer
        """
        return '0' + builtins.oct(number)[2:]

    raw_input = input
    from imp import reload
    unicode = str
    unichr = chr
    xrange = range
else:
    import __builtin__
    from collections import Mapping
    apply = __builtin__.apply
    chr = __builtin__.chr
    cmp = __builtin__.cmp
    execfile = __builtin__.execfile
    intern = __builtin__.intern
    oct = __builtin__.oct
    raw_input = __builtin__.raw_input
    reload = __builtin__.reload
    unicode = __builtin__.unicode
    unichr = __builtin__.unichr
    xrange = __builtin__.xrange


if PY3:
    def execfile(filename, myglobals=None, mylocals=None):
        """
        Read and execute a Python script from a file in the given namespaces.
        The globals and locals are dictionaries, defaulting to the current
        globals and locals. If only globals is given, locals defaults to it.
        """
        if myglobals is None:
            # There seems to be no alternative to frame hacking here.
            caller_frame = inspect.stack()[1]
            myglobals = caller_frame[0].f_globals
            mylocals = caller_frame[0].f_locals
        elif mylocals is None:
            # Only if myglobals is given do we set mylocals to it.
            mylocals = myglobals
        if not isinstance(myglobals, Mapping):
            raise TypeError('globals must be a mapping')
        if not isinstance(mylocals, Mapping):
            raise TypeError('locals must be a mapping')
        with open(filename, "rb") as fin:
             source = fin.read()
        code = compile(source, filename, "exec")
        exec_(code, myglobals, mylocals)


if PY3:
    __all__ = ['apply', 'chr', 'cmp', 'execfile', 'intern', 'raw_input',
               'reload', 'unichr', 'unicode', 'xrange']
else:
    __all__ = []
