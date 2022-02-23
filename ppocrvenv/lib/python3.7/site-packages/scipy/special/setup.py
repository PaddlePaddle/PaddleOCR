import os
import sys
from os.path import join, dirname
from distutils.sysconfig import get_python_inc
import subprocess
import numpy
from numpy.distutils.misc_util import get_numpy_include_dirs, get_info

from scipy._build_utils.compiler_helper import set_c_flags_hook


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from scipy._build_utils.system_info import get_info as get_system_info
    from scipy._build_utils import combine_dict, uses_blas64

    config = Configuration('special', parent_package, top_path)

    if uses_blas64():
        lapack_opt = get_system_info('lapack_ilp64_opt')
    else:
        lapack_opt = get_system_info('lapack_opt')

    define_macros = []
    if sys.platform == 'win32':
        # define_macros.append(('NOINFINITIES',None))
        # define_macros.append(('NONANS',None))
        define_macros.append(('_USE_MATH_DEFINES',None))

    curdir = os.path.abspath(os.path.dirname(__file__))
    python_inc_dirs = get_python_inc()
    plat_specific_python_inc_dirs = get_python_inc(plat_specific=1)
    inc_dirs = [get_numpy_include_dirs(), python_inc_dirs]
    if python_inc_dirs != plat_specific_python_inc_dirs:
        inc_dirs.append(plat_specific_python_inc_dirs)
    inc_dirs.append(join(dirname(dirname(__file__)), '_lib'))
    inc_dirs.append(join(dirname(dirname(__file__)), '_build_utils', 'src'))

    # C libraries
    cephes_src = [join('cephes','*.c')]
    cephes_hdr = [join('cephes', '*.h')]
    config.add_library('sc_cephes',sources=cephes_src,
                       include_dirs=[curdir] + inc_dirs,
                       depends=(cephes_hdr + ['*.h']),
                       macros=define_macros)

    # Fortran/C++ libraries
    mach_src = [join('mach','*.f')]
    amos_src = [join('amos','*.f')]
    cdf_src = [join('cdflib','*.f')]
    specfun_src = [join('specfun','*.f')]
    config.add_library('sc_mach',sources=mach_src,
                       config_fc={'noopt':(__file__,1)})
    config.add_library('sc_amos',sources=amos_src)
    config.add_library('sc_cdf',sources=cdf_src)
    config.add_library('sc_specfun',sources=specfun_src)

    # Extension specfun
    config.add_extension('specfun',
                         sources=['specfun.pyf'],
                         f2py_options=['--no-wrap-functions'],
                         depends=specfun_src,
                         define_macros=[],
                         libraries=['sc_specfun'])

    # Extension _ufuncs
    headers = ['*.h', join('cephes', '*.h')]
    ufuncs_src = ['_ufuncs.c', 'sf_error.c',
                  'amos_wrappers.c', 'cdf_wrappers.c', 'specfun_wrappers.c',
                  '_cosine.c']

    ufuncs_dep = (
        headers
        + ufuncs_src
        + amos_src
        + cephes_src
        + mach_src
        + cdf_src
        + specfun_src
    )
    cfg = combine_dict(lapack_opt,
                       include_dirs=[curdir] + inc_dirs + [numpy.get_include()],
                       libraries=['sc_amos', 'sc_cephes', 'sc_mach',
                                  'sc_cdf', 'sc_specfun'],
                       define_macros=define_macros)
    _ufuncs = config.add_extension('_ufuncs',
                                   depends=ufuncs_dep,
                                   sources=ufuncs_src,
                                   extra_info=get_info("npymath"),
                                   **cfg)
    _ufuncs._pre_build_hook = set_c_flags_hook

    # Extension _ufuncs_cxx
    ufuncs_cxx_src = ['_ufuncs_cxx.cxx', 'sf_error.c',
                      '_faddeeva.cxx', 'Faddeeva.cc',
                      '_wright.cxx', 'wright.cc']
    ufuncs_cxx_dep = (headers + ufuncs_cxx_src + cephes_src
                      + ['*.hh'])
    config.add_extension('_ufuncs_cxx',
                         sources=ufuncs_cxx_src,
                         depends=ufuncs_cxx_dep,
                         include_dirs=[curdir] + inc_dirs,
                         define_macros=define_macros,
                         extra_info=get_info("npymath"))

    cfg = combine_dict(lapack_opt, include_dirs=inc_dirs)
    config.add_extension('_ellip_harm_2',
                         sources=['_ellip_harm_2.c', 'sf_error.c',],
                         **cfg)

    # Cython API
    config.add_data_files('cython_special.pxd')

    cython_special_src = ['cython_special.c', 'sf_error.c',
                          'amos_wrappers.c', 'cdf_wrappers.c',
                          'specfun_wrappers.c', '_cosine.c']
    cython_special_dep = (
        headers
        + ufuncs_src
        + ufuncs_cxx_src
        + amos_src
        + cephes_src
        + mach_src
        + cdf_src
        + specfun_src
    )
    cfg = combine_dict(lapack_opt,
                       include_dirs=[curdir] + inc_dirs + [numpy.get_include()],
                       libraries=['sc_amos', 'sc_cephes', 'sc_mach',
                                  'sc_cdf', 'sc_specfun'],
                       define_macros=define_macros)
    cython_special = config.add_extension('cython_special',
                                          depends=cython_special_dep,
                                          sources=cython_special_src,
                                          extra_info=get_info("npymath"),
                                          **cfg)
    cython_special._pre_build_hook = set_c_flags_hook

    # combinatorics
    config.add_extension('_comb',
                         sources=['_comb.c'])

    # testing for _round.h
    config.add_extension('_test_round',
                         sources=['_test_round.c'],
                         depends=['_round.h', 'cephes/dd_idefs.h'],
                         include_dirs=[numpy.get_include()] + inc_dirs,
                         extra_info=get_info('npymath'))

    config.add_data_files('tests/*.py')
    config.add_data_files('tests/data/README')

    # regenerate npz data files
    makenpz = os.path.join(os.path.dirname(__file__),
                           'utils', 'makenpz.py')
    data_dir = os.path.join(os.path.dirname(__file__),
                            'tests', 'data')
    for name in ['boost', 'gsl', 'local']:
        subprocess.check_call([sys.executable, makenpz,
                               '--use-timestamp',
                               os.path.join(data_dir, name)])

    config.add_data_files('tests/data/*.npz')

    config.add_subpackage('_precompute')

    # Type stubs
    config.add_data_files('*.pyi')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
