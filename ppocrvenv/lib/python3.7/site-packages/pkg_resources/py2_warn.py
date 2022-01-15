import sys
import warnings
import textwrap


msg = textwrap.dedent("""
    Encountered a version of Setuptools that no longer supports
    this version of Python. Please head to
    https://bit.ly/setuptools-py2-sunset for support.
    """)

pre = "Setuptools no longer works on Python 2\n"

if sys.version_info < (3,):
    warnings.warn(pre + "*" * 60 + msg + "*" * 60)
    raise SystemExit(32)
