
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('transform', parent_package, top_path)

    config.add_data_dir('tests')

    config.add_data_files('rotation.pyi')
    config.add_extension('rotation',
                         sources=['rotation.c'])

    return config
