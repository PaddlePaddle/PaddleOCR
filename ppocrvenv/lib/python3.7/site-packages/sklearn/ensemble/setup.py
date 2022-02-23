import numpy
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("ensemble", parent_package, top_path)

    config.add_extension(
        "_gradient_boosting",
        sources=["_gradient_boosting.pyx"],
        include_dirs=[numpy.get_include()],
    )

    config.add_subpackage("tests")

    # Histogram-based gradient boosting files
    config.add_extension(
        "_hist_gradient_boosting._gradient_boosting",
        sources=["_hist_gradient_boosting/_gradient_boosting.pyx"],
        include_dirs=[numpy.get_include()],
    )

    config.add_extension(
        "_hist_gradient_boosting.histogram",
        sources=["_hist_gradient_boosting/histogram.pyx"],
        include_dirs=[numpy.get_include()],
    )

    config.add_extension(
        "_hist_gradient_boosting.splitting",
        sources=["_hist_gradient_boosting/splitting.pyx"],
        include_dirs=[numpy.get_include()],
    )

    config.add_extension(
        "_hist_gradient_boosting._binning",
        sources=["_hist_gradient_boosting/_binning.pyx"],
        include_dirs=[numpy.get_include()],
    )

    config.add_extension(
        "_hist_gradient_boosting._predictor",
        sources=["_hist_gradient_boosting/_predictor.pyx"],
        include_dirs=[numpy.get_include()],
    )

    config.add_extension(
        "_hist_gradient_boosting._loss",
        sources=["_hist_gradient_boosting/_loss.pyx"],
        include_dirs=[numpy.get_include()],
    )

    config.add_extension(
        "_hist_gradient_boosting._bitset",
        sources=["_hist_gradient_boosting/_bitset.pyx"],
        include_dirs=[numpy.get_include()],
    )

    config.add_extension(
        "_hist_gradient_boosting.common",
        sources=["_hist_gradient_boosting/common.pyx"],
        include_dirs=[numpy.get_include()],
    )

    config.add_extension(
        "_hist_gradient_boosting.utils",
        sources=["_hist_gradient_boosting/utils.pyx"],
        include_dirs=[numpy.get_include()],
    )

    config.add_subpackage("_hist_gradient_boosting.tests")

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration().todict())
