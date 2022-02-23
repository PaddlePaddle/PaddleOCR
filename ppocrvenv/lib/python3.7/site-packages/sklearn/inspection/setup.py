from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("inspection", parent_package, top_path)

    config.add_subpackage("_plot")
    config.add_subpackage("_plot.tests")

    config.add_subpackage("tests")

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration().todict())
