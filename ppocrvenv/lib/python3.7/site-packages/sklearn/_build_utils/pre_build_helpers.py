"""Helpers to check build environment before actual build of scikit-learn"""

import os
import sys
import glob
import tempfile
import textwrap
import setuptools  # noqa
import subprocess

from distutils.dist import Distribution
from distutils.sysconfig import customize_compiler
from numpy.distutils.ccompiler import new_compiler
from numpy.distutils.command.config_compiler import config_cc


def _get_compiler():
    """Get a compiler equivalent to the one that will be used to build sklearn

    Handles compiler specified as follows:
        - python setup.py build_ext --compiler=<compiler>
        - CC=<compiler> python setup.py build_ext
    """
    dist = Distribution(
        {
            "script_name": os.path.basename(sys.argv[0]),
            "script_args": sys.argv[1:],
            "cmdclass": {"config_cc": config_cc},
        }
    )
    dist.parse_config_files()
    dist.parse_command_line()

    cmd_opts = dist.command_options.get("build_ext")
    if cmd_opts is not None and "compiler" in cmd_opts:
        compiler = cmd_opts["compiler"][1]
    else:
        compiler = None

    ccompiler = new_compiler(compiler=compiler)
    customize_compiler(ccompiler)

    return ccompiler


def compile_test_program(code, extra_preargs=[], extra_postargs=[]):
    """Check that some C code can be compiled and run"""
    ccompiler = _get_compiler()

    # extra_(pre/post)args can be a callable to make it possible to get its
    # value from the compiler
    if callable(extra_preargs):
        extra_preargs = extra_preargs(ccompiler)
    if callable(extra_postargs):
        extra_postargs = extra_postargs(ccompiler)

    start_dir = os.path.abspath(".")

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            os.chdir(tmp_dir)

            # Write test program
            with open("test_program.c", "w") as f:
                f.write(code)

            os.mkdir("objects")

            # Compile, test program
            ccompiler.compile(
                ["test_program.c"], output_dir="objects", extra_postargs=extra_postargs
            )

            # Link test program
            objects = glob.glob(os.path.join("objects", "*" + ccompiler.obj_extension))
            ccompiler.link_executable(
                objects,
                "test_program",
                extra_preargs=extra_preargs,
                extra_postargs=extra_postargs,
            )

            if "PYTHON_CROSSENV" not in os.environ:
                # Run test program if not cross compiling
                # will raise a CalledProcessError if return code was non-zero
                output = subprocess.check_output("./test_program")
                output = output.decode(sys.stdout.encoding or "utf-8").splitlines()
            else:
                # Return an empty output if we are cross compiling
                # as we cannot run the test_program
                output = []
        except Exception:
            raise
        finally:
            os.chdir(start_dir)

    return output


def basic_check_build():
    """Check basic compilation and linking of C code"""
    if "PYODIDE_PACKAGE_ABI" in os.environ:
        # The following check won't work in pyodide
        return

    code = textwrap.dedent(
        """\
        #include <stdio.h>
        int main(void) {
        return 0;
        }
        """
    )
    compile_test_program(code)
