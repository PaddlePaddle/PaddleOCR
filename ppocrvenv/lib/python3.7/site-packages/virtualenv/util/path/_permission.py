from __future__ import absolute_import, unicode_literals

import os
from stat import S_IXGRP, S_IXOTH, S_IXUSR

from virtualenv.util.six import ensure_text


def make_exe(filename):
    original_mode = filename.stat().st_mode
    levels = [S_IXUSR, S_IXGRP, S_IXOTH]
    for at in range(len(levels), 0, -1):
        try:
            mode = original_mode
            for level in levels[:at]:
                mode |= level
            filename.chmod(mode)
            break
        except OSError:
            continue


def set_tree(folder, stat):
    for root, _, files in os.walk(ensure_text(str(folder))):
        for filename in files:
            os.chmod(os.path.join(root, filename), stat)


__all__ = (
    "make_exe",
    "set_tree",
)
