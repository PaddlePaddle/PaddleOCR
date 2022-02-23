from __future__ import absolute_import, unicode_literals

import subprocess
import sys

import six

if six.PY2 and sys.platform == "win32":
    from . import _win_subprocess

    Popen = _win_subprocess.Popen
else:
    Popen = subprocess.Popen


CREATE_NO_WINDOW = 0x80000000


def run_cmd(cmd):
    try:
        process = Popen(
            cmd,
            universal_newlines=True,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        out, err = process.communicate()  # input disabled
        code = process.returncode
    except OSError as os_error:
        code, out, err = os_error.errno, "", os_error.strerror
    return code, out, err


__all__ = (
    "subprocess",
    "Popen",
    "run_cmd",
    "CREATE_NO_WINDOW",
)
