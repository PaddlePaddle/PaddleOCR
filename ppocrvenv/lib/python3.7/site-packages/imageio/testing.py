# -*- coding: utf-8 -*-
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

""" Functionality used for testing. This code itself is not covered in tests.
"""

import os
import sys
import inspect
import shutil
import atexit

import pytest

# Get root dir
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = THIS_DIR
for i in range(9):
    ROOT_DIR = os.path.dirname(ROOT_DIR)
    if os.path.isfile(os.path.join(ROOT_DIR, ".gitignore")):
        break


# Functions to use in tests


def run_tests_if_main(show_coverage=False):
    """Run tests in a given file if it is run as a script

    Coverage is reported for running this single test. Set show_coverage to
    launch the report in the web browser.
    """
    local_vars = inspect.currentframe().f_back.f_locals
    if not local_vars.get("__name__", "") == "__main__":
        return
    # we are in a "__main__"
    os.chdir(ROOT_DIR)
    fname = str(local_vars["__file__"])
    _clear_imageio()
    _enable_faulthandler()
    pytest.main(
        [
            "-v",
            "-x",
            "--color=yes",
            "--cov",
            "imageio",
            "--cov-config",
            ".coveragerc",
            "--cov-report",
            "html",
            fname,
        ]
    )
    if show_coverage:
        import webbrowser

        fname = os.path.join(ROOT_DIR, "htmlcov", "index.html")
        webbrowser.open_new_tab(fname)


_the_test_dir = None


def get_test_dir():
    global _the_test_dir
    if _the_test_dir is None:
        # Define dir
        from imageio.core import appdata_dir

        _the_test_dir = os.path.join(appdata_dir("imageio"), "testdir")
        # Clear and create it now
        clean_test_dir(True)
        os.makedirs(_the_test_dir)
        os.makedirs(os.path.join(_the_test_dir, "images"))
        # And later
        atexit.register(clean_test_dir)
    return _the_test_dir


def clean_test_dir(strict=False):
    if os.path.isdir(_the_test_dir):
        try:
            shutil.rmtree(_the_test_dir)
        except Exception:
            if strict:
                raise


def need_internet():
    if os.getenv("IMAGEIO_NO_INTERNET", "").lower() in ("1", "true", "yes"):
        pytest.skip("No internet")


# Functions to use from invoke tasks


def test_unit(cov_report="term"):
    """Run all unit tests. Returns exit code."""
    orig_dir = os.getcwd()
    os.chdir(ROOT_DIR)
    try:
        _clear_imageio()
        _enable_faulthandler()
        return pytest.main(
            [
                "-v",
                "--cov",
                "imageio",
                "--cov-config",
                ".coveragerc",
                "--cov-report",
                cov_report,
                "tests",
            ]
        )
    finally:
        os.chdir(orig_dir)
        import imageio

        print("Tests were performed on", str(imageio))


# Requirements


def _enable_faulthandler():
    """Enable faulthandler (if we can), so that we get tracebacks
    on segfaults.
    """
    try:
        import faulthandler

        faulthandler.enable()
        print("Faulthandler enabled")
    except Exception:
        print("Could not enable faulthandler")


def _clear_imageio():
    # Remove ourselves from sys.modules to force an import
    for key in list(sys.modules.keys()):
        if key.startswith("imageio"):
            del sys.modules[key]
