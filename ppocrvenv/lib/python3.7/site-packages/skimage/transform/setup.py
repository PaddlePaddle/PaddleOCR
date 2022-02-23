#!/usr/bin/env python

import os

from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('transform', parent_package, top_path)

    cython(['_hough_transform.pyx',
            '_warps_cy.pyx',
            '_radon_transform.pyx'], working_path=base_path)

    config.add_extension('_hough_transform', sources=['_hough_transform.c'],
                         include_dirs=[get_numpy_include_dirs()])

    config.add_extension('_warps_cy', sources=['_warps_cy.c'],
                         include_dirs=[get_numpy_include_dirs(), '../_shared'])

    config.add_extension('_radon_transform',
                         sources=['_radon_transform.c'],
                         include_dirs=[get_numpy_include_dirs()])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='scikit-image Developers',
          author='scikit-image Developers',
          maintainer_email='scikit-image@python.org',
          description='Transforms',
          url='https://github.com/scikit-image/scikit-image',
          license='SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )
