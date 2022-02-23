def configuration(parent_package='',top_path=None):
    from scipy._build_utils.system_info import get_info, NotFoundError
    lapack_opt = get_info("lapack_opt")

    from numpy.distutils.misc_util import Configuration
    config = Configuration('scipy',parent_package,top_path)
    config.add_subpackage('cluster')
    config.add_subpackage('constants')
    config.add_subpackage('fft')
    config.add_subpackage('fftpack')
    config.add_subpackage('integrate')
    config.add_subpackage('interpolate')
    config.add_subpackage('io')
    config.add_subpackage('linalg')
    config.add_data_files('*.pxd')
    config.add_subpackage('misc')
    config.add_subpackage('odr')
    config.add_subpackage('optimize')
    config.add_subpackage('signal')
    config.add_subpackage('sparse')
    config.add_subpackage('spatial')
    config.add_subpackage('special')
    config.add_subpackage('stats')
    config.add_subpackage('ndimage')
    config.add_subpackage('_build_utils')
    config.add_subpackage('_lib')
    config.make_config_py()
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
