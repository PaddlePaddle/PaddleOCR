import os
from pathlib import Path


def isPath(f):
    return isinstance(f, (bytes, str, Path))


# Checks if an object is a string, and that it points to a directory.
def isDirectory(f):
    return isPath(f) and os.path.isdir(f)


class deferred_error:
    def __init__(self, ex):
        self.ex = ex

    def __getattr__(self, elt):
        raise self.ex
