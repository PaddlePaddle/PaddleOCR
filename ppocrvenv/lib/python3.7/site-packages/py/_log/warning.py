import py, sys

class DeprecationWarning(DeprecationWarning):
    def __init__(self, msg, path, lineno):
        self.msg = msg
        self.path = path
        self.lineno = lineno
    def __repr__(self):
        return "%s:%d: %s" %(self.path, self.lineno+1, self.msg)
    def __str__(self):
        return self.msg

def _apiwarn(startversion, msg, stacklevel=2, function=None):
    # below is mostly COPIED from python2.4/warnings.py's def warn()
    # Get context information
    if isinstance(stacklevel, str):
        frame = sys._getframe(1)
        level = 1
        found = frame.f_code.co_filename.find(stacklevel) != -1
        while frame:
            co = frame.f_code
            if co.co_filename.find(stacklevel) == -1:
                if found:
                    stacklevel = level
                    break
            else:
                found = True
            level += 1
            frame = frame.f_back
        else:
            stacklevel = 1
    msg = "%s (since version %s)" %(msg, startversion)
    warn(msg, stacklevel=stacklevel+1, function=function)


def warn(msg, stacklevel=1, function=None):
    if function is not None:
        import inspect
        filename = inspect.getfile(function)
        lineno = py.code.getrawcode(function).co_firstlineno
    else:
        try:
            caller = sys._getframe(stacklevel)
        except ValueError:
            globals = sys.__dict__
            lineno = 1
        else:
            globals = caller.f_globals
            lineno = caller.f_lineno
        if '__name__' in globals:
            module = globals['__name__']
        else:
            module = "<string>"
        filename = globals.get('__file__')
    if filename:
        fnl = filename.lower()
        if fnl.endswith(".pyc") or fnl.endswith(".pyo"):
            filename = filename[:-1]
        elif fnl.endswith("$py.class"):
            filename = filename.replace('$py.class', '.py')
    else:
        if module == "__main__":
            try:
                filename = sys.argv[0]
            except AttributeError:
                # embedded interpreters don't have sys.argv, see bug #839151
                filename = '__main__'
        if not filename:
            filename = module
    path = py.path.local(filename)
    warning = DeprecationWarning(msg, path, lineno)
    import warnings
    warnings.warn_explicit(warning, category=Warning,
        filename=str(warning.path),
        lineno=warning.lineno,
        registry=warnings.__dict__.setdefault(
            "__warningsregistry__", {})
    )

