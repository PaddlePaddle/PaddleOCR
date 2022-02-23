#!/usr/bin/env python

import os
from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('segmentation', parent_package, top_path)

    cython(['_watershed_cy.pyx',
            '_felzenszwalb_cy.pyx',
            '_quickshift_cy.pyx',
            '_slic.pyx',
            ], working_path=base_path)
    config.add_extension('_watershed_cy', sources=['_watershed_cy.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_felzenszwalb_cy', sources=['_felzenszwalb_cy.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_quickshift_cy', sources=['_quickshift_cy.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_slic', sources=['_slic.c'],
                         include_dirs=[get_numpy_include_dirs()])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='scikit-image Developers',
          maintainer_email='scikit-image@python.org',
          description='Segmentation Algorithms',
          url='https://github.com/scikit-image/scikit-image',
          license='SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )
