"""Helpers for OpenMP support during the build."""

# This code is adapted for a large part from the astropy openmp helpers, which
# can be found at: https://github.com/astropy/extension-helpers/blob/master/extension_helpers/_openmp_helpers.py  # noqa


import os
import sys
import textwrap
import warnings
import subprocess

from distutils.errors import CompileError, LinkError

from .pre_build_helpers import compile_test_program


def get_openmp_flag(compiler):
    if hasattr(compiler, "compiler"):
        compiler = compiler.compiler[0]
    else:
        compiler = compiler.__class__.__name__

    if sys.platform == "win32" and ("icc" in compiler or "icl" in compiler):
        return ["/Qopenmp"]
    elif sys.platform == "win32":
        return ["/openmp"]
    elif sys.platform in ("darwin", "linux") and "icc" in compiler:
        return ["-qopenmp"]
    elif sys.platform == "darwin" and "openmp" in os.getenv("CPPFLAGS", ""):
        # -fopenmp can't be passed as compile flag when using Apple-clang.
        # OpenMP support has to be enabled during preprocessing.
        #
        # For example, our macOS wheel build jobs use the following environment
        # variables to build with Apple-clang and the brew installed "libomp":
        #
        # export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
        # export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
        # export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
        # export LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/local/opt/libomp/lib
        #                          -L/usr/local/opt/libomp/lib -lomp"
        return []
    # Default flag for GCC and clang:
    return ["-fopenmp"]


def check_openmp_support():
    """Check whether OpenMP test code can be compiled and run"""
    if "PYODIDE_PACKAGE_ABI" in os.environ:
        # Pyodide doesn't support OpenMP
        return False

    code = textwrap.dedent(
        """\
        #include <omp.h>
        #include <stdio.h>
        int main(void) {
        #pragma omp parallel
        printf("nthreads=%d\\n", omp_get_num_threads());
        return 0;
        }
        """
    )

    extra_preargs = os.getenv("LDFLAGS", None)
    if extra_preargs is not None:
        extra_preargs = extra_preargs.strip().split(" ")
        # FIXME: temporary fix to link against system libraries on linux
        # "-Wl,--sysroot=/" should be removed
        extra_preargs = [
            flag
            for flag in extra_preargs
            if flag.startswith(("-L", "-Wl,-rpath", "-l", "-Wl,--sysroot=/"))
        ]

    extra_postargs = get_openmp_flag

    try:
        output = compile_test_program(
            code, extra_preargs=extra_preargs, extra_postargs=extra_postargs
        )

        if output and "nthreads=" in output[0]:
            nthreads = int(output[0].strip().split("=")[1])
            openmp_supported = len(output) == nthreads
        elif "PYTHON_CROSSENV" in os.environ:
            # Since we can't run the test program when cross-compiling
            # assume that openmp is supported if the program can be
            # compiled.
            openmp_supported = True
        else:
            openmp_supported = False

    except (CompileError, LinkError, subprocess.CalledProcessError):
        openmp_supported = False

    if not openmp_supported:
        if os.getenv("SKLEARN_FAIL_NO_OPENMP"):
            raise CompileError("Failed to build with OpenMP")
        else:
            message = textwrap.dedent(
                """

                                ***********
                                * WARNING *
                                ***********

                It seems that scikit-learn cannot be built with OpenMP.

                - Make sure you have followed the installation instructions:

                    https://scikit-learn.org/dev/developers/advanced_installation.html

                - If your compiler supports OpenMP but you still see this
                  message, please submit a bug report at:

                    https://github.com/scikit-learn/scikit-learn/issues

                - The build will continue with OpenMP-based parallelism
                  disabled. Note however that some estimators will run in
                  sequential mode instead of leveraging thread-based
                  parallelism.

                                    ***
                """
            )
            warnings.warn(message)

    return openmp_supported
