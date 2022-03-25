import sys


# Passthrough for builtins supported with py27.
BaseException = BaseException
GeneratorExit = GeneratorExit
_sysex = (KeyboardInterrupt, SystemExit, MemoryError, GeneratorExit)
all = all
any = any
callable = callable
enumerate = enumerate
reversed = reversed
set, frozenset = set, frozenset
sorted = sorted


if sys.version_info >= (3, 0):
    exec("print_ = print ; exec_=exec")
    import builtins

    # some backward compatibility helpers
    _basestring = str
    def _totext(obj, encoding=None, errors=None):
        if isinstance(obj, bytes):
            if errors is None:
                obj = obj.decode(encoding)
            else:
                obj = obj.decode(encoding, errors)
        elif not isinstance(obj, str):
            obj = str(obj)
        return obj

    def _isbytes(x):
        return isinstance(x, bytes)

    def _istext(x):
        return isinstance(x, str)

    text = str
    bytes = bytes

    def _getimself(function):
        return getattr(function, '__self__', None)

    def _getfuncdict(function):
        return getattr(function, "__dict__", None)

    def _getcode(function):
        return getattr(function, "__code__", None)

    def execfile(fn, globs=None, locs=None):
        if globs is None:
            back = sys._getframe(1)
            globs = back.f_globals
            locs = back.f_locals
            del back
        elif locs is None:
            locs = globs
        fp = open(fn, "r")
        try:
            source = fp.read()
        finally:
            fp.close()
        co = compile(source, fn, "exec", dont_inherit=True)
        exec_(co, globs, locs)

else:
    import __builtin__ as builtins
    _totext = unicode
    _basestring = basestring
    text = unicode
    bytes = str
    execfile = execfile
    callable = callable
    def _isbytes(x):
        return isinstance(x, str)
    def _istext(x):
        return isinstance(x, unicode)

    def _getimself(function):
        return getattr(function, 'im_self', None)

    def _getfuncdict(function):
        return getattr(function, "__dict__", None)

    def _getcode(function):
        try:
            return getattr(function, "__code__")
        except AttributeError:
            return getattr(function, "func_code", None)

    def print_(*args, **kwargs):
        """ minimal backport of py3k print statement. """
        sep = ' '
        if 'sep' in kwargs:
            sep = kwargs.pop('sep')
        end = '\n'
        if 'end' in kwargs:
            end = kwargs.pop('end')
        file = 'file' in kwargs and kwargs.pop('file') or sys.stdout
        if kwargs:
            args = ", ".join([str(x) for x in kwargs])
            raise TypeError("invalid keyword arguments: %s" % args)
        at_start = True
        for x in args:
            if not at_start:
                file.write(sep)
            file.write(str(x))
            at_start = False
        file.write(end)

    def exec_(obj, globals=None, locals=None):
        """ minimal backport of py3k exec statement. """
        __tracebackhide__ = True
        if globals is None:
            frame = sys._getframe(1)
            globals = frame.f_globals
            if locals is None:
                locals = frame.f_locals
        elif locals is None:
            locals = globals
        exec2(obj, globals, locals)

if sys.version_info >= (3, 0):
    def _reraise(cls, val, tb):
        __tracebackhide__ = True
        assert hasattr(val, '__traceback__')
        raise cls.with_traceback(val, tb)
else:
    exec ("""
def _reraise(cls, val, tb):
    __tracebackhide__ = True
    raise cls, val, tb
def exec2(obj, globals, locals):
    __tracebackhide__ = True
    exec obj in globals, locals
""")

def _tryimport(*names):
    """ return the first successfully imported module. """
    assert names
    for name in names:
        try:
            __import__(name)
        except ImportError:
            excinfo = sys.exc_info()
        else:
            return sys.modules[name]
    _reraise(*excinfo)
