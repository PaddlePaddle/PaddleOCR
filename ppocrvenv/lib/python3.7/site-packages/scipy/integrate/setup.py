import os
from os.path import join

from scipy._build_utils import numpy_nodepr_api


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from scipy._build_utils.system_info import get_info
    from scipy._build_utils import (uses_blas64, blas_ilp64_pre_build_hook,
                                    combine_dict, get_f2py_int64_options)

    config = Configuration('integrate', parent_package, top_path)

    if uses_blas64():
        lapack_opt = get_info('lapack_ilp64_opt', 2)
        pre_build_hook = blas_ilp64_pre_build_hook(lapack_opt)
        f2py_options = get_f2py_int64_options()
    else:
        lapack_opt = get_info('lapack_opt')
        pre_build_hook = None
        f2py_options = None

    mach_src = [join('mach','*.f')]
    quadpack_src = [join('quadpack', '*.f')]
    lsoda_src = [join('odepack', fn) for fn in [
        'blkdta000.f', 'bnorm.f', 'cfode.f',
        'ewset.f', 'fnorm.f', 'intdy.f',
        'lsoda.f', 'prja.f', 'solsy.f', 'srcma.f',
        'stoda.f', 'vmnorm.f', 'xerrwv.f', 'xsetf.f',
        'xsetun.f']]
    vode_src = [join('odepack', 'vode.f'), join('odepack', 'zvode.f')]
    dop_src = [join('dop','*.f')]
    quadpack_test_src = [join('tests','_test_multivariate.c')]
    odeint_banded_test_src = [join('tests', 'banded5x5.f')]

    config.add_library('mach', sources=mach_src, config_fc={'noopt': (__file__, 1)},
                       _pre_build_hook=pre_build_hook)
    config.add_library('quadpack', sources=quadpack_src, _pre_build_hook=pre_build_hook)
    config.add_library('lsoda', sources=lsoda_src, _pre_build_hook=pre_build_hook)
    config.add_library('vode', sources=vode_src, _pre_build_hook=pre_build_hook)
    config.add_library('dop', sources=dop_src, _pre_build_hook=pre_build_hook)

    # Extensions
    # quadpack:
    include_dirs = [join(os.path.dirname(__file__), '..', '_lib', 'src')]
    cfg = combine_dict(lapack_opt,
                       include_dirs=include_dirs,
                       libraries=['quadpack', 'mach'])
    config.add_extension('_quadpack',
                         sources=['_quadpackmodule.c'],
                         depends=(['__quadpack.h']
                                  + quadpack_src + mach_src),
                         **cfg)

    # odepack/lsoda-odeint
    cfg = combine_dict(lapack_opt, numpy_nodepr_api,
                       libraries=['lsoda', 'mach'])
    config.add_extension('_odepack',
                         sources=['_odepackmodule.c'],
                         depends=(lsoda_src + mach_src),
                         **cfg)

    # vode
    cfg = combine_dict(lapack_opt,
                       libraries=['vode'])
    ext = config.add_extension('vode',
                               sources=['vode.pyf'],
                               depends=vode_src,
                               f2py_options=f2py_options,
                               **cfg)
    ext._pre_build_hook = pre_build_hook

    # lsoda
    cfg = combine_dict(lapack_opt,
                       libraries=['lsoda', 'mach'])
    ext = config.add_extension('lsoda',
                               sources=['lsoda.pyf'],
                               depends=(lsoda_src + mach_src),
                               f2py_options=f2py_options,
                               **cfg)
    ext._pre_build_hook = pre_build_hook

    # dop
    ext = config.add_extension('_dop',
                               sources=['dop.pyf'],
                               libraries=['dop'],
                               depends=dop_src,
                               f2py_options=f2py_options)
    ext._pre_build_hook = pre_build_hook

    config.add_extension('_test_multivariate',
                         sources=quadpack_test_src)

    # Fortran+f2py extension module for testing odeint.
    cfg = combine_dict(lapack_opt,
                       libraries=['lsoda', 'mach'])
    ext = config.add_extension('_test_odeint_banded',
                               sources=odeint_banded_test_src,
                               depends=(lsoda_src + mach_src),
                               f2py_options=f2py_options,
                               **cfg)
    ext._pre_build_hook = pre_build_hook

    config.add_subpackage('_ivp')

    config.add_data_dir('tests')
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
