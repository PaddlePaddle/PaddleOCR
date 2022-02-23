def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('skimage', parent_package, top_path)

    config.add_subpackage('_shared')
    config.add_subpackage('color')
    config.add_subpackage('data')
    config.add_subpackage('draw')
    config.add_subpackage('exposure')
    config.add_subpackage('feature')
    config.add_subpackage('restoration')
    config.add_subpackage('filters')
    config.add_subpackage('future')
    config.add_subpackage('graph')
    config.add_subpackage('io')
    config.add_subpackage('measure')
    config.add_subpackage('metrics')
    config.add_subpackage('morphology')
    config.add_subpackage('transform')
    config.add_subpackage('util')
    config.add_subpackage('segmentation')

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup

    config = configuration(top_path='').todict()
    setup(**config)
