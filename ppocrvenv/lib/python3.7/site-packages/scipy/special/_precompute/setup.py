
def configuration(parent_name='special', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('_precompute', parent_name, top_path)
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration().todict())

