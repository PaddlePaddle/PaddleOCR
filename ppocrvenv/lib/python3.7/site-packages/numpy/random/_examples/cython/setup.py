#!/usr/bin/env python3
"""
Build the Cython demonstrations of low-level access to NumPy random

Usage: python setup.py build_ext -i
"""
from distutils.core import setup
from os.path import dirname, join, abspath

import numpy as np
from Cython.Build import cythonize
from numpy.distutils.misc_util import get_info
from setuptools.extension import Extension

path = dirname(__file__)
src_dir = join(dirname(path), '..', 'src')
defs = [('NPY_NO_DEPRECATED_API', 0)]
inc_path = np.get_include()
lib_path = [abspath(join(np.get_include(), '..', '..', 'random', 'lib'))]
lib_path += get_info('npymath')['library_dirs']

extending = Extension("extending",
                      sources=[join('.', 'extending.pyx')],
                      include_dirs=[
                            np.get_include(),
                            join(path, '..', '..')
                        ],
                      define_macros=defs,
                      )
distributions = Extension("extending_distributions",
                          sources=[join('.', 'extending_distributions.pyx')],
                          include_dirs=[inc_path],
                          library_dirs=lib_path,
                          libraries=['npyrandom', 'npymath'],
                          define_macros=defs,
                          )

extensions = [extending, distributions]

setup(
    ext_modules=cythonize(extensions)
)
