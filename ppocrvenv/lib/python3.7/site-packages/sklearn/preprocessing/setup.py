import os


def configuration(parent_package="", top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration("preprocessing", parent_package, top_path)
    libraries = []
    if os.name == "posix":
        libraries.append("m")

    config.add_extension(
        "_csr_polynomial_expansion",
        sources=["_csr_polynomial_expansion.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
    )

    config.add_subpackage("tests")

    return config
