"""Tests for the optional typing-extensions dependency."""

import sys
import textwrap
import subprocess

CODE = textwrap.dedent(r"""
    import sys
    import importlib

    assert "typing_extensions" not in sys.modules
    assert "numpy.typing" not in sys.modules

    # Importing `typing_extensions` will now raise an `ImportError`
    sys.modules["typing_extensions"] = None
    assert importlib.import_module("numpy.typing")
""")


def test_no_typing_extensions() -> None:
    """Import `numpy.typing` in the absence of typing-extensions.

    Notes
    -----
    Ideally, we'd just run the normal typing tests in an environment where
    typing-extensions is not installed, but unfortunatelly this is currently
    impossible as it is an indirect hard dependency of pytest.

    """
    p = subprocess.run([sys.executable, '-c', CODE], capture_output=True)
    if p.returncode:
        raise AssertionError(
            f"Non-zero return code: {p.returncode!r}\n\n{p.stderr.decode()}"
        )

