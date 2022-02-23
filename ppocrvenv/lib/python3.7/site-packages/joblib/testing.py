"""
Helper for testing.
"""

import sys
import warnings
import os.path
import re
import subprocess
import threading

import pytest
import _pytest


raises = pytest.raises
warns = pytest.warns
SkipTest = _pytest.runner.Skipped
skipif = pytest.mark.skipif
fixture = pytest.fixture
parametrize = pytest.mark.parametrize
timeout = pytest.mark.timeout
xfail = pytest.mark.xfail
param = pytest.param


def warnings_to_stdout():
    """ Redirect all warnings to stdout.
    """
    showwarning_orig = warnings.showwarning

    def showwarning(msg, cat, fname, lno, file=None, line=0):
        showwarning_orig(msg, cat, os.path.basename(fname), line, sys.stdout)

    warnings.showwarning = showwarning
    # warnings.simplefilter('always')


def check_subprocess_call(cmd, timeout=5, stdout_regex=None,
                          stderr_regex=None):
    """Runs a command in a subprocess with timeout in seconds.

    Also checks returncode is zero, stdout if stdout_regex is set, and
    stderr if stderr_regex is set.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    def kill_process():
        warnings.warn("Timeout running {}".format(cmd))
        proc.kill()

    timer = threading.Timer(timeout, kill_process)
    try:
        timer.start()
        stdout, stderr = proc.communicate()
        stdout, stderr = stdout.decode(), stderr.decode()
        if proc.returncode != 0:
            message = (
                'Non-zero return code: {}.\nStdout:\n{}\n'
                'Stderr:\n{}').format(
                    proc.returncode, stdout, stderr)
            raise ValueError(message)

        if (stdout_regex is not None and
                not re.search(stdout_regex, stdout)):
            raise ValueError(
                "Unexpected stdout: {!r} does not match:\n{!r}".format(
                    stdout_regex, stdout))
        if (stderr_regex is not None and
                not re.search(stderr_regex, stderr)):
            raise ValueError(
                "Unexpected stderr: {!r} does not match:\n{!r}".format(
                    stderr_regex, stderr))

    finally:
        timer.cancel()
