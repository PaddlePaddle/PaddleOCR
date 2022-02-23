"""All minimum dependencies for scikit-learn."""
import platform
import argparse


# numpy scipy and cython should by in sync with pyproject.toml
if platform.python_implementation() == "PyPy":
    NUMPY_MIN_VERSION = "1.19.0"
else:
    # We pinned PyWavelet (a scikit-image dependence) to 1.1.1 in the minimum
    # documentation CI builds that is the latest version that support our
    # minimum NumPy version required. If PyWavelets 1.2+ is installed, it would
    # require NumPy 1.17+ that trigger a bug with Pandas 0.25:
    # https://github.com/numpy/numpy/issues/18355#issuecomment-774610226
    # When upgrading NumPy, we can unpin PyWavelets but we need to update the
    # minimum version of Pandas >= 1.0.5.
    NUMPY_MIN_VERSION = "1.14.6"

SCIPY_MIN_VERSION = "1.1.0"
JOBLIB_MIN_VERSION = "0.11"
THREADPOOLCTL_MIN_VERSION = "2.0.0"
PYTEST_MIN_VERSION = "5.0.1"
CYTHON_MIN_VERSION = "0.29.24"


# 'build' and 'install' is included to have structured metadata for CI.
# It will NOT be included in setup's extras_require
# The values are (version_spec, comma separated tags)
dependent_packages = {
    "numpy": (NUMPY_MIN_VERSION, "build, install"),
    "scipy": (SCIPY_MIN_VERSION, "build, install"),
    "joblib": (JOBLIB_MIN_VERSION, "install"),
    "threadpoolctl": (THREADPOOLCTL_MIN_VERSION, "install"),
    "cython": (CYTHON_MIN_VERSION, "build"),
    "matplotlib": ("2.2.3", "benchmark, docs, examples, tests"),
    "scikit-image": ("0.14.5", "docs, examples, tests"),
    "pandas": ("0.25.0", "benchmark, docs, examples, tests"),
    "seaborn": ("0.9.0", "docs, examples"),
    "memory_profiler": ("0.57.0", "benchmark, docs"),
    "pytest": (PYTEST_MIN_VERSION, "tests"),
    "pytest-cov": ("2.9.0", "tests"),
    "flake8": ("3.8.2", "tests"),
    "black": ("21.6b0", "tests"),
    "mypy": ("0.770", "tests"),
    "pyamg": ("4.0.0", "tests"),
    "sphinx": ("4.0.1", "docs"),
    "sphinx-gallery": ("0.7.0", "docs"),
    "numpydoc": ("1.0.0", "docs"),
    "Pillow": ("7.1.2", "docs"),
    "sphinx-prompt": ("1.3.0", "docs"),
    "sphinxext-opengraph": ("0.4.2", "docs"),
}


# create inverse mapping for setuptools
tag_to_packages: dict = {
    extra: []
    for extra in ["build", "install", "docs", "examples", "tests", "benchmark"]
}
for package, (min_version, extras) in dependent_packages.items():
    for extra in extras.split(", "):
        tag_to_packages[extra].append("{}>={}".format(package, min_version))


# Used by CI to get the min dependencies
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get min dependencies for a package")

    parser.add_argument("package", choices=dependent_packages)
    args = parser.parse_args()
    min_version = dependent_packages[args.package][0]
    print(min_version)
