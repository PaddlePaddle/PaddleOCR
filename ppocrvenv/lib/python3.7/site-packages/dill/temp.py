#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE
"""
Methods for serialized objects (or source code) stored in temporary files
and file-like objects.
"""
#XXX: better instead to have functions write to any given file-like object ?
#XXX: currently, all file-like objects are created by the function...

__all__ = ['dump_source', 'dump', 'dumpIO_source', 'dumpIO',\
           'load_source', 'load', 'loadIO_source', 'loadIO',\
           'capture']

import contextlib
from ._dill import PY3


@contextlib.contextmanager
def capture(stream='stdout'):
    """builds a context that temporarily replaces the given stream name

    >>> with capture('stdout') as out:
    ...   print ("foo!")
    ... 
    >>> print (out.getvalue())
    foo!

    """
    import sys
    if PY3:
        from io import StringIO
    else:
        from StringIO import StringIO
    orig = getattr(sys, stream)
    setattr(sys, stream, StringIO())
    try:
        yield getattr(sys, stream)
    finally:
        setattr(sys, stream, orig)


def b(x): # deal with b'foo' versus 'foo'
    import codecs
    return codecs.latin_1_encode(x)[0]

def load_source(file, **kwds):
    """load an object that was stored with dill.temp.dump_source

    file: filehandle
    alias: string name of stored object
    mode: mode to open the file, one of: {'r', 'rb'}

    >>> f = lambda x: x**2
    >>> pyfile = dill.temp.dump_source(f, alias='_f')
    >>> _f = dill.temp.load_source(pyfile)
    >>> _f(4)
    16
    """
    alias = kwds.pop('alias', None)
    mode = kwds.pop('mode', 'r')
    fname = getattr(file, 'name', file) # fname=file.name or fname=file (if str)
    source = open(fname, mode=mode, **kwds).read()
    if not alias:
        tag = source.strip().splitlines()[-1].split()
        if tag[0] != '#NAME:':
            stub = source.splitlines()[0]
            raise IOError("unknown name for code: %s" % stub)
        alias = tag[-1]
    local = {}
    exec(source, local)
    _ = eval("%s" % alias, local)
    return _

def dump_source(object, **kwds):
    """write object source to a NamedTemporaryFile (instead of dill.dump)
Loads with "import" or "dill.temp.load_source".  Returns the filehandle.

    >>> f = lambda x: x**2
    >>> pyfile = dill.temp.dump_source(f, alias='_f')
    >>> _f = dill.temp.load_source(pyfile)
    >>> _f(4)
    16

    >>> f = lambda x: x**2
    >>> pyfile = dill.temp.dump_source(f, dir='.')
    >>> modulename = os.path.basename(pyfile.name).split('.py')[0]
    >>> exec('from %s import f as _f' % modulename)
    >>> _f(4)
    16

Optional kwds:
    If 'alias' is specified, the object will be renamed to the given string.

    If 'prefix' is specified, the file name will begin with that prefix,
    otherwise a default prefix is used.
    
    If 'dir' is specified, the file will be created in that directory,
    otherwise a default directory is used.
    
    If 'text' is specified and true, the file is opened in text
    mode.  Else (the default) the file is opened in binary mode.  On
    some operating systems, this makes no difference.

NOTE: Keep the return value for as long as you want your file to exist !
    """ #XXX: write a "load_source"?
    from .source import importable, getname
    import tempfile
    kwds.pop('suffix', '') # this is *always* '.py'
    alias = kwds.pop('alias', '') #XXX: include an alias so a name is known
    name = str(alias) or getname(object)
    name = "\n#NAME: %s\n" % name
    #XXX: assumes kwds['dir'] is writable and on $PYTHONPATH
    file = tempfile.NamedTemporaryFile(suffix='.py', **kwds)
    file.write(b(''.join([importable(object, alias=alias),name])))
    file.flush()
    return file

def load(file, **kwds):
    """load an object that was stored with dill.temp.dump

    file: filehandle
    mode: mode to open the file, one of: {'r', 'rb'}

    >>> dumpfile = dill.temp.dump([1, 2, 3, 4, 5])
    >>> dill.temp.load(dumpfile)
    [1, 2, 3, 4, 5]
    """
    import dill as pickle
    mode = kwds.pop('mode', 'rb')
    name = getattr(file, 'name', file) # name=file.name or name=file (if str)
    return pickle.load(open(name, mode=mode, **kwds))

def dump(object, **kwds):
    """dill.dump of object to a NamedTemporaryFile.
Loads with "dill.temp.load".  Returns the filehandle.

    >>> dumpfile = dill.temp.dump([1, 2, 3, 4, 5])
    >>> dill.temp.load(dumpfile)
    [1, 2, 3, 4, 5]

Optional kwds:
    If 'suffix' is specified, the file name will end with that suffix,
    otherwise there will be no suffix.
    
    If 'prefix' is specified, the file name will begin with that prefix,
    otherwise a default prefix is used.
    
    If 'dir' is specified, the file will be created in that directory,
    otherwise a default directory is used.
    
    If 'text' is specified and true, the file is opened in text
    mode.  Else (the default) the file is opened in binary mode.  On
    some operating systems, this makes no difference.

NOTE: Keep the return value for as long as you want your file to exist !
    """
    import dill as pickle
    import tempfile
    file = tempfile.NamedTemporaryFile(**kwds)
    pickle.dump(object, file)
    file.flush()
    return file

def loadIO(buffer, **kwds):
    """load an object that was stored with dill.temp.dumpIO

    buffer: buffer object

    >>> dumpfile = dill.temp.dumpIO([1, 2, 3, 4, 5])
    >>> dill.temp.loadIO(dumpfile)
    [1, 2, 3, 4, 5]
    """
    import dill as pickle
    if PY3:
        from io import BytesIO as StringIO
    else:
        from StringIO import StringIO
    value = getattr(buffer, 'getvalue', buffer) # value or buffer.getvalue
    if value != buffer: value = value() # buffer.getvalue()
    return pickle.load(StringIO(value))

def dumpIO(object, **kwds):
    """dill.dump of object to a buffer.
Loads with "dill.temp.loadIO".  Returns the buffer object.

    >>> dumpfile = dill.temp.dumpIO([1, 2, 3, 4, 5])
    >>> dill.temp.loadIO(dumpfile)
    [1, 2, 3, 4, 5]
    """
    import dill as pickle
    if PY3:
        from io import BytesIO as StringIO
    else:
        from StringIO import StringIO
    file = StringIO()
    pickle.dump(object, file)
    file.flush()
    return file

def loadIO_source(buffer, **kwds):
    """load an object that was stored with dill.temp.dumpIO_source

    buffer: buffer object
    alias: string name of stored object

    >>> f = lambda x:x**2
    >>> pyfile = dill.temp.dumpIO_source(f, alias='_f')
    >>> _f = dill.temp.loadIO_source(pyfile)
    >>> _f(4)
    16
    """
    alias = kwds.pop('alias', None)
    source = getattr(buffer, 'getvalue', buffer) # source or buffer.getvalue
    if source != buffer: source = source() # buffer.getvalue()
    if PY3: source = source.decode() # buffer to string
    if not alias:
        tag = source.strip().splitlines()[-1].split()
        if tag[0] != '#NAME:':
            stub = source.splitlines()[0]
            raise IOError("unknown name for code: %s" % stub)
        alias = tag[-1]
    local = {}
    exec(source, local)
    _ = eval("%s" % alias, local)
    return _

def dumpIO_source(object, **kwds):
    """write object source to a buffer (instead of dill.dump)
Loads by with dill.temp.loadIO_source.  Returns the buffer object.

    >>> f = lambda x:x**2
    >>> pyfile = dill.temp.dumpIO_source(f, alias='_f')
    >>> _f = dill.temp.loadIO_source(pyfile)
    >>> _f(4)
    16

Optional kwds:
    If 'alias' is specified, the object will be renamed to the given string.
    """
    from .source import importable, getname
    if PY3:
        from io import BytesIO as StringIO
    else:
        from StringIO import StringIO
    alias = kwds.pop('alias', '') #XXX: include an alias so a name is known
    name = str(alias) or getname(object)
    name = "\n#NAME: %s\n" % name
    #XXX: assumes kwds['dir'] is writable and on $PYTHONPATH
    file = StringIO()
    file.write(b(''.join([importable(object, alias=alias),name])))
    file.flush()
    return file


del contextlib


# EOF
