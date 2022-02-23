
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('windows', parent_package, top_path)

    config.add_data_dir('tests')

    return config
