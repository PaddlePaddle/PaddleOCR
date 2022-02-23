"""
Entrypoint for testing from the top-level namespace.
"""
import os
import sys

PKG = os.path.dirname(os.path.dirname(__file__))


def test(extra_args=None):
    try:
        import pytest
    except ImportError as err:
        raise ImportError("Need pytest>=5.0.1 to run tests") from err
    try:
        import hypothesis  # noqa
    except ImportError as err:
        raise ImportError("Need hypothesis>=3.58 to run tests") from err
    cmd = ["--skip-slow", "--skip-network", "--skip-db"]
    if extra_args:
        if not isinstance(extra_args, list):
            extra_args = [extra_args]
        cmd = extra_args
    cmd += [PKG]
    joined = " ".join(cmd)
    print(f"running: pytest {joined}")
    sys.exit(pytest.main(cmd))


__all__ = ["test"]
