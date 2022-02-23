#!/usr/bin/env python

import os
from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('draw', parent_package, top_path)

    cython(['_draw.pyx'], working_path=base_path)

    config.add_extension('_draw', sources=['_draw.c'],
                         include_dirs=[get_numpy_include_dirs(), '../_shared'])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='scikit-image developers',
          author='scikit-image developers',
          maintainer_email='scikit-image@python.org',
          description='Drawing',
          url='https://github.com/scikit-image/scikit-image',
          license='SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )
