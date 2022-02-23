import os

import numpy
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("cluster", parent_package, top_path)
    libraries = []
    if os.name == "posix":
        libraries.append("m")
    config.add_extension(
        "_expected_mutual_info_fast",
        sources=["_expected_mutual_info_fast.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
    )

    config.add_subpackage("tests")

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration().todict())
