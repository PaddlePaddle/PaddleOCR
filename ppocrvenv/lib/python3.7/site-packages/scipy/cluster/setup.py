DEFINE_MACROS = [("SCIPY_PY3K", None)]


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    config = Configuration('cluster', parent_package, top_path)

    config.add_data_dir('tests')

    config.add_extension('_vq',
        sources=[('_vq.c')],
        include_dirs=[get_numpy_include_dirs()])

    config.add_extension('_hierarchy',
        sources=[('_hierarchy.c')],
        include_dirs=[get_numpy_include_dirs()])

    config.add_extension('_optimal_leaf_ordering',
        sources=[('_optimal_leaf_ordering.c')],
        include_dirs=[get_numpy_include_dirs()])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
