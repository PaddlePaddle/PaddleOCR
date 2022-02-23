
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('cython_optimize', parent_package, top_path)

    config.add_data_files('*.pxd')
    config.add_extension('_zeros', sources='_zeros.c')
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
