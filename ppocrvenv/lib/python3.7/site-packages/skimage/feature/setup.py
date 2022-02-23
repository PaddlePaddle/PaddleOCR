#!/usr/bin/env python

import os
from skimage._build import cython
import pythran, logging

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('feature', parent_package, top_path)
    config.add_data_files('orb_descriptor_positions.txt')

    cython(['corner_cy.pyx',
            'censure_cy.pyx',
            'orb_cy.pyx',
            '_texture.pyx',
            '_hoghistogram.pyx',
            '_sift.pyx',
            ], working_path=base_path)
    # _haar uses c++, so it must be cythonized separately
    cython(['_cascade.pyx',
            '_haar.pyx'], working_path=base_path)

    config.add_extension('_cascade', sources=['_cascade.cpp'],
                         include_dirs=[get_numpy_include_dirs()],
                         language="c++")
    config.add_extension('corner_cy', sources=['corner_cy.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('censure_cy', sources=['censure_cy.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('orb_cy', sources=['orb_cy.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_texture', sources=['_texture.c'],
                         include_dirs=[get_numpy_include_dirs(), '../_shared'])
    config.add_extension('_hoghistogram', sources=['_hoghistogram.c'],
                         include_dirs=[get_numpy_include_dirs(), '../_shared'])
    config.add_extension('_haar', sources=['_haar.cpp'],
                         include_dirs=[get_numpy_include_dirs(), '../_shared'],
                         language="c++")
    config.add_extension('_sift', sources=['_sift.c'],
                         include_dirs=[get_numpy_include_dirs(), '../_shared'])

    # pythran submodules
    pythran.config.logger.setLevel(logging.INFO)
    ext = pythran.dist.PythranExtension(
        'skimage.feature.brief_cy',
        sources=["skimage/feature/brief_pythran.py"],
        config=['compiler.blas=none'])
    config.ext_modules.append(ext)

    ext = pythran.dist.PythranExtension(
        'skimage.feature._hessian_det_appx',
        sources=["skimage/feature/_hessian_det_appx_pythran.py"],
        config=['compiler.blas=none'])
    config.ext_modules.append(ext)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='scikit-image Developers',
          author='scikit-image Developers',
          maintainer_email='scikit-image@python.org',
          description='Features',
          url='https://github.com/scikit-image/scikit-image',
          license='SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )
