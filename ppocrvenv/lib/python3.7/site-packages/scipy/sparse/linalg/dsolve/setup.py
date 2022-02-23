from os.path import join, dirname
import sys
import glob


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from scipy._build_utils.system_info import get_info
    from scipy._build_utils import numpy_nodepr_api

    config = Configuration('dsolve',parent_package,top_path)
    config.add_data_dir('tests')

    lapack_opt = get_info('lapack_opt',notfound_action=2)
    if sys.platform == 'win32':
        superlu_defs = [('NO_TIMER',1)]
    else:
        superlu_defs = []
    superlu_defs.append(('USE_VENDOR_BLAS',1))

    superlu_src = join(dirname(__file__), 'SuperLU', 'SRC')

    sources = sorted(glob.glob(join(superlu_src, '*.c')))
    headers = list(glob.glob(join(superlu_src, '*.h')))

    config.add_library('superlu_src',
                       sources=sources,
                       macros=superlu_defs,
                       include_dirs=[superlu_src],
                       )

    # Extension
    ext_sources = ['_superlumodule.c',
                   '_superlu_utils.c',
                   '_superluobject.c']

    config.add_extension('_superlu',
                         sources=ext_sources,
                         libraries=['superlu_src'],
                         depends=(sources + headers),
                         extra_info=lapack_opt,
                         **numpy_nodepr_api
                         )

    # Add license files
    config.add_data_files('SuperLU/License.txt')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
