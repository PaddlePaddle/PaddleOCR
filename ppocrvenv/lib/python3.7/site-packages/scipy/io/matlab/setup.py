
def configuration(parent_package='io',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('matlab', parent_package, top_path)
    config.add_extension('streams', sources=['streams.c'])
    config.add_extension('mio_utils', sources=['mio_utils.c'])
    config.add_extension('mio5_utils', sources=['mio5_utils.c'])
    config.add_data_dir('tests')
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
