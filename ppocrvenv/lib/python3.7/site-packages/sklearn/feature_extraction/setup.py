import os
import platform


def configuration(parent_package="", top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration("feature_extraction", parent_package, top_path)
    libraries = []
    if os.name == "posix":
        libraries.append("m")

    if platform.python_implementation() != "PyPy":
        config.add_extension(
            "_hashing_fast",
            sources=["_hashing_fast.pyx"],
            include_dirs=[numpy.get_include()],
            libraries=libraries,
        )
    config.add_subpackage("tests")

    return config
