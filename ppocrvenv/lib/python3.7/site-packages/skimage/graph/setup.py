#!/usr/bin/env python

from skimage._build import cython
import os.path

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('graph', parent_package, top_path)

    # This function tries to create C files from the given .pyx files.  If
    # it fails, try to build with pre-generated .c files.
    cython(['_spath.pyx',
            '_mcp.pyx',
            'heap.pyx'], working_path=base_path)

    config.add_extension('_spath', sources=['_spath.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_mcp', sources=['_mcp.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('heap', sources=['heap.c'],
                         include_dirs=[get_numpy_include_dirs()])
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='scikit-image Developers',
          maintainer_email='scikit-image@python.org',
          description='Graph-based Image-processing Algorithms',
          url='https://github.com/scikit-image/scikit-image',
          license='Modified BSD',
          **(configuration(top_path='').todict())
          )
