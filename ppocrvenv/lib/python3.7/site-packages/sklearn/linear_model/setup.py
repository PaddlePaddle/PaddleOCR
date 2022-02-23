import os
import numpy

from sklearn._build_utils import gen_from_templates


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration("linear_model", parent_package, top_path)

    libraries = []
    if os.name == "posix":
        libraries.append("m")

    config.add_extension(
        "_cd_fast",
        sources=["_cd_fast.pyx"],
        include_dirs=numpy.get_include(),
        libraries=libraries,
    )

    config.add_extension(
        "_sgd_fast",
        sources=["_sgd_fast.pyx"],
        include_dirs=numpy.get_include(),
        libraries=libraries,
    )

    # generate sag_fast from template
    templates = ["sklearn/linear_model/_sag_fast.pyx.tp"]
    gen_from_templates(templates)

    config.add_extension(
        "_sag_fast", sources=["_sag_fast.pyx"], include_dirs=numpy.get_include()
    )

    # add other directories
    config.add_subpackage("tests")
    config.add_subpackage("_glm")
    config.add_subpackage("_glm/tests")

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration(top_path="").todict())
