
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('eigen',parent_package,top_path)

    config.add_subpackage(('arpack'))
    config.add_subpackage(('lobpcg'))

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
