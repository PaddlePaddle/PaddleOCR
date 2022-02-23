"""
A substitute for the Python 3 open() function.

Note that io.open() is more complete but maybe slower. Even so, the
completeness may be a better default. TODO: compare these
"""

_builtin_open = open

class newopen(object):
    """Wrapper providing key part of Python 3 open() interface.

    From IPython's py3compat.py module. License: BSD.
    """
    def __init__(self, fname, mode="r", encoding="utf-8"):
        self.f = _builtin_open(fname, mode)
        self.enc = encoding

    def write(self, s):
        return self.f.write(s.encode(self.enc))

    def read(self, size=-1):
        return self.f.read(size).decode(self.enc)

    def close(self):
        return self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, etype, value, traceback):
        self.f.close()
