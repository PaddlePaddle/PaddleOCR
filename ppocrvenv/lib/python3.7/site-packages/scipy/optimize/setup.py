import sys
import os.path
from os.path import join

from scipy._build_utils import numpy_nodepr_api


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from scipy._build_utils.system_info import get_info
    from scipy._build_utils import (gfortran_legacy_flag_hook,
                                    blas_ilp64_pre_build_hook, combine_dict,
                                    uses_blas64, get_f2py_int64_options)
    from scipy._build_utils.compiler_helper import (
        set_cxx_flags_hook, set_cxx_flags_clib_hook, set_c_flags_hook)

    config = Configuration('optimize', parent_package, top_path)

    include_dirs = [join(os.path.dirname(__file__), '..', '_lib', 'src')]

    minpack_src = [join('minpack', '*f')]
    config.add_library('minpack', sources=minpack_src)
    config.add_extension('_minpack',
                         sources=['_minpackmodule.c'],
                         libraries=['minpack'],
                         depends=(["minpack.h", "__minpack.h"] + minpack_src),
                         include_dirs=include_dirs,
                         **numpy_nodepr_api)

    config.add_library('rectangular_lsap',
                       sources='rectangular_lsap/rectangular_lsap.cpp',
                       headers='rectangular_lsap/rectangular_lsap.h',
                       _pre_build_hook=set_cxx_flags_clib_hook)
    _lsap = config.add_extension(
        '_lsap_module',
        sources=['_lsap_module.c'],
        libraries=['rectangular_lsap'],
        depends=(['rectangular_lsap/rectangular_lsap.cpp',
                  'rectangular_lsap/rectangular_lsap.h']),
        include_dirs=include_dirs,
        **numpy_nodepr_api)
    _lsap._pre_build_hook = set_c_flags_hook

    rootfind_src = [join('Zeros', '*.c')]
    rootfind_hdr = [join('Zeros', 'zeros.h')]
    config.add_library('rootfind',
                       sources=rootfind_src,
                       headers=rootfind_hdr, **numpy_nodepr_api)

    config.add_extension('_zeros',
                         sources=['zeros.c'],
                         libraries=['rootfind'],
                         depends=(rootfind_src + rootfind_hdr),
                         **numpy_nodepr_api)

    if uses_blas64():
        lapack = get_info('lapack_ilp64_opt')
        f2py_options = get_f2py_int64_options()
        pre_build_hook = blas_ilp64_pre_build_hook(lapack)
    else:
        lapack = get_info('lapack_opt')
        f2py_options = None
        pre_build_hook = None

    lapack = combine_dict(lapack, numpy_nodepr_api)

    sources = ['lbfgsb.pyf', 'lbfgsb.f', 'linpack.f', 'timer.f']
    ext = config.add_extension('_lbfgsb',
                               sources=[join('lbfgsb_src', x)
                                        for x in sources],
                               f2py_options=f2py_options,
                               **lapack)
    ext._pre_build_hook = pre_build_hook

    sources = ['moduleTNC.c', 'tnc.c']
    config.add_extension('moduleTNC',
                         sources=[join('tnc', x) for x in sources],
                         depends=[join('tnc', 'tnc.h')],
                         **numpy_nodepr_api)

    config.add_extension('_cobyla',
                         sources=[join('cobyla', x) for x in [
                             'cobyla.pyf', 'cobyla2.f', 'trstlp.f']],
                         **numpy_nodepr_api)

    sources = ['minpack2.pyf', 'dcsrch.f', 'dcstep.f']
    config.add_extension('minpack2',
                         sources=[join('minpack2', x) for x in sources],
                         **numpy_nodepr_api)

    sources = ['slsqp.pyf', 'slsqp_optmz.f']
    ext = config.add_extension('_slsqp', sources=[
        join('slsqp', x) for x in sources], **numpy_nodepr_api)
    ext._pre_build_hook = gfortran_legacy_flag_hook

    config.add_data_files('__nnls.pyi')
    ext = config.add_extension('__nnls', sources=[
        join('__nnls', x) for x in ["nnls.f", "nnls.pyf"]], **numpy_nodepr_api)
    ext._pre_build_hook = gfortran_legacy_flag_hook

    if int(os.environ.get('SCIPY_USE_PYTHRAN', 1)):
        import pythran
        ext = pythran.dist.PythranExtension(
            'scipy.optimize._group_columns',
            sources=["scipy/optimize/_group_columns.py"],
            config=['compiler.blas=none'])
        config.ext_modules.append(ext)
    else:
        config.add_extension('_group_columns', sources=['_group_columns.c'],)

    config.add_extension('_bglu_dense', sources=['_bglu_dense.c'])

    config.add_subpackage('_lsq')

    config.add_subpackage('_trlib')

    config.add_subpackage('_trustregion_constr')

    # Cython optimize API for zeros functions
    config.add_subpackage('cython_optimize')
    config.add_data_files('cython_optimize.pxd')

    config.add_subpackage('_shgo_lib')
    config.add_data_dir('_shgo_lib')

    # HiGHS linear programming libraries and extensions
    if 'sdist' not in sys.argv:
        # Avoid running this during sdist creation - it makes numpy.distutils
        # create an empty cython/src top-level directory.
        config.add_subpackage('_highs')

    config.add_data_dir('tests')

    # Add license files
    config.add_data_files('lbfgsb_src/README')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
