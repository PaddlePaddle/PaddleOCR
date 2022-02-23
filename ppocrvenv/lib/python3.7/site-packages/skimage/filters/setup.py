#!/usr/bin/env python

import os
from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('filters', parent_package, top_path)

    cython(['rank/core_cy.pyx',
            'rank/core_cy_3d.pyx',
            'rank/generic_cy.pyx',
            'rank/percentile_cy.pyx',
            'rank/bilateral_cy.pyx',
            '_multiotsu.pyx'], working_path=base_path)

    config.add_extension('rank.core_cy', sources=['rank/core_cy.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('rank.core_cy_3d', sources=['rank/core_cy_3d.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_multiotsu', sources=['_multiotsu.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('rank.generic_cy', sources=['rank/generic_cy.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension(
        'rank.percentile_cy', sources=['rank/percentile_cy.c'],
        include_dirs=[get_numpy_include_dirs()])
    config.add_extension(
        'rank.bilateral_cy', sources=['rank/bilateral_cy.c'],
        include_dirs=[get_numpy_include_dirs()])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='scikit-image Developers',
          author='scikit-image Developers',
          maintainer_email='scikit-image@python.org',
          description='Filters',
          url='https://github.com/scikit-image/scikit-image',
          license='SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )
