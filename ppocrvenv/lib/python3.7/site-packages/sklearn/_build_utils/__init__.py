"""
Utilities useful during the build.
"""
# author: Andy Mueller, Gael Varoquaux
# license: BSD


import os
import sklearn
import contextlib

from distutils.version import LooseVersion

from .pre_build_helpers import basic_check_build
from .openmp_helpers import check_openmp_support
from .._min_dependencies import CYTHON_MIN_VERSION


DEFAULT_ROOT = "sklearn"


def _check_cython_version():
    message = (
        "Please install Cython with a version >= {0} in order "
        "to build a scikit-learn from source."
    ).format(CYTHON_MIN_VERSION)
    try:
        import Cython
    except ModuleNotFoundError as e:
        # Re-raise with more informative error message instead:
        raise ModuleNotFoundError(message) from e

    if LooseVersion(Cython.__version__) < CYTHON_MIN_VERSION:
        message += " The current version of Cython is {} installed in {}.".format(
            Cython.__version__, Cython.__path__
        )
        raise ValueError(message)


def cythonize_extensions(top_path, config):
    """Check that a recent Cython is available and cythonize extensions"""
    _check_cython_version()
    from Cython.Build import cythonize

    # Fast fail before cythonization if compiler fails compiling basic test
    # code even without OpenMP
    basic_check_build()

    # check simple compilation with OpenMP. If it fails scikit-learn will be
    # built without OpenMP and the test test_openmp_supported in the test suite
    # will fail.
    # `check_openmp_support` compiles a small test program to see if the
    # compilers are properly configured to build with OpenMP. This is expensive
    # and we only want to call this function once.
    # The result of this check is cached as a private attribute on the sklearn
    # module (only at build-time) to be used twice:
    # - First to set the value of SKLEARN_OPENMP_PARALLELISM_ENABLED, the
    #   cython build-time variable passed to the cythonize() call.
    # - Then in the build_ext subclass defined in the top-level setup.py file
    #   to actually build the compiled extensions with OpenMP flags if needed.
    sklearn._OPENMP_SUPPORTED = check_openmp_support()

    n_jobs = 1
    with contextlib.suppress(ImportError):
        import joblib

        if LooseVersion(joblib.__version__) > LooseVersion("0.13.0"):
            # earlier joblib versions don't account for CPU affinity
            # constraints, and may over-estimate the number of available
            # CPU particularly in CI (cf loky#114)
            n_jobs = joblib.cpu_count()

    # Additional checks for Cython
    cython_enable_debug_directives = (
        os.environ.get("SKLEARN_ENABLE_DEBUG_CYTHON_DIRECTIVES", "0") != "0"
    )

    config.ext_modules = cythonize(
        config.ext_modules,
        nthreads=n_jobs,
        compile_time_env={
            "SKLEARN_OPENMP_PARALLELISM_ENABLED": sklearn._OPENMP_SUPPORTED
        },
        compiler_directives={
            "language_level": 3,
            "boundscheck": cython_enable_debug_directives,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
            "cdivision": True,
        },
    )


def gen_from_templates(templates):
    """Generate cython files from a list of templates"""
    # Lazy import because cython is not a runtime dependency.
    from Cython import Tempita

    for template in templates:
        outfile = template.replace(".tp", "")

        # if the template is not updated, no need to output the cython file
        if not (
            os.path.exists(outfile)
            and os.stat(template).st_mtime < os.stat(outfile).st_mtime
        ):

            with open(template, "r") as f:
                tmpl = f.read()

            tmpl_ = Tempita.sub(tmpl)

            with open(outfile, "w") as f:
                f.write(tmpl_)
