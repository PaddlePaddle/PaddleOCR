from __future__ import absolute_import, unicode_literals

import sys

import six

if six.PY3:
    from pathlib import Path
else:
    if sys.platform == "win32":
        # workaround for https://github.com/mcmtroffaes/pathlib2/issues/56
        from .via_os_path import Path
    else:
        from pathlib2 import Path


__all__ = ("Path",)
