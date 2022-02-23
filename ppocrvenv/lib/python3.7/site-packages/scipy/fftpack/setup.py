# Created by Pearu Peterson, August 2002

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('fftpack',parent_package, top_path)

    config.add_data_dir('tests')

    config.add_extension('convolve', sources=['convolve.c'])
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
