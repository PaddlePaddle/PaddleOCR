import os
from os.path import join
import numpy


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration("svm", parent_package, top_path)

    config.add_subpackage("tests")

    # newrand wrappers
    config.add_extension(
        "_newrand",
        sources=["_newrand.pyx"],
        include_dirs=[numpy.get_include(), join("src", "newrand")],
        depends=[join("src", "newrand", "newrand.h")],
        language="c++",
        # Use C++11 random number generator fix
        extra_compile_args=["-std=c++11"],
    )

    # Section LibSVM

    # we compile both libsvm and libsvm_sparse
    config.add_library(
        "libsvm-skl",
        sources=[join("src", "libsvm", "libsvm_template.cpp")],
        depends=[
            join("src", "libsvm", "svm.cpp"),
            join("src", "libsvm", "svm.h"),
            join("src", "newrand", "newrand.h"),
        ],
        # Force C++ linking in case gcc is picked up instead
        # of g++ under windows with some versions of MinGW
        extra_link_args=["-lstdc++"],
        # Use C++11 to use the random number generator fix
        extra_compiler_args=["-std=c++11"],
    )

    libsvm_sources = ["_libsvm.pyx"]
    libsvm_depends = [
        join("src", "libsvm", "libsvm_helper.c"),
        join("src", "libsvm", "libsvm_template.cpp"),
        join("src", "libsvm", "svm.cpp"),
        join("src", "libsvm", "svm.h"),
        join("src", "newrand", "newrand.h"),
    ]

    config.add_extension(
        "_libsvm",
        sources=libsvm_sources,
        include_dirs=[
            numpy.get_include(),
            join("src", "libsvm"),
            join("src", "newrand"),
        ],
        libraries=["libsvm-skl"],
        depends=libsvm_depends,
    )

    # liblinear module
    libraries = []
    if os.name == "posix":
        libraries.append("m")

    # precompile liblinear to use C++11 flag
    config.add_library(
        "liblinear-skl",
        sources=[
            join("src", "liblinear", "linear.cpp"),
            join("src", "liblinear", "tron.cpp"),
        ],
        depends=[
            join("src", "liblinear", "linear.h"),
            join("src", "liblinear", "tron.h"),
            join("src", "newrand", "newrand.h"),
        ],
        # Force C++ linking in case gcc is picked up instead
        # of g++ under windows with some versions of MinGW
        extra_link_args=["-lstdc++"],
        # Use C++11 to use the random number generator fix
        extra_compiler_args=["-std=c++11"],
    )

    liblinear_sources = ["_liblinear.pyx"]
    liblinear_depends = [
        join("src", "liblinear", "*.h"),
        join("src", "newrand", "newrand.h"),
        join("src", "liblinear", "liblinear_helper.c"),
    ]

    config.add_extension(
        "_liblinear",
        sources=liblinear_sources,
        libraries=["liblinear-skl"] + libraries,
        include_dirs=[
            join(".", "src", "liblinear"),
            join(".", "src", "newrand"),
            join("..", "utils"),
            numpy.get_include(),
        ],
        depends=liblinear_depends,
        # extra_compile_args=['-O0 -fno-inline'],
    )

    # end liblinear module

    # this should go *after* libsvm-skl
    libsvm_sparse_sources = ["_libsvm_sparse.pyx"]
    config.add_extension(
        "_libsvm_sparse",
        libraries=["libsvm-skl"],
        sources=libsvm_sparse_sources,
        include_dirs=[
            numpy.get_include(),
            join("src", "libsvm"),
            join("src", "newrand"),
        ],
        depends=[
            join("src", "libsvm", "svm.h"),
            join("src", "newrand", "newrand.h"),
            join("src", "libsvm", "libsvm_sparse_helper.c"),
        ],
    )

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration(top_path="").todict())
