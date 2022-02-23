#!/usr/bin/env python

from skimage._build import cython

import os.path

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('io', parent_package, top_path)
    config.add_data_files('_plugins/*.ini')

    # This function tries to create C files from the given .pyx files.  If
    # it fails, we build the checked-in .c files.
    cython(['_plugins/_colormixer.pyx', '_plugins/_histograms.pyx'],
           working_path=base_path)

    config.add_extension('_plugins._colormixer',
                         sources=['_plugins/_colormixer.c'],
                         include_dirs=[get_numpy_include_dirs()])

    config.add_extension('_plugins._histograms',
                         sources=['_plugins/_histograms.c'],
                         include_dirs=[get_numpy_include_dirs()])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='scikit-image Developers',
          maintainer_email='scikit-image@python.org',
          description='Image I/O Routines',
          url='https://github.com/scikit-image/scikit-image',
          license='Modified BSD',
          **(configuration(top_path='').todict())
          )
