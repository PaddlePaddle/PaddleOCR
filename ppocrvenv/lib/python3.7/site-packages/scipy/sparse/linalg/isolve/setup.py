from os.path import join


def configuration(parent_package='',top_path=None):
    from scipy._build_utils.system_info import get_info
    from numpy.distutils.misc_util import Configuration
    from scipy._build_utils import (get_g77_abi_wrappers, uses_blas64,
                                    blas_ilp64_pre_build_hook, get_f2py_int64_options)

    config = Configuration('isolve',parent_package,top_path)

    if uses_blas64():
        lapack_opt = get_info('lapack_ilp64_opt')
        f2py_options = get_f2py_int64_options()
        pre_build_hook = blas_ilp64_pre_build_hook(lapack_opt)
    else:
        lapack_opt = get_info('lapack_opt')
        f2py_options = None
        pre_build_hook = None

    # iterative methods
    methods = ['BiCGREVCOM.f.src',
               'BiCGSTABREVCOM.f.src',
               'CGREVCOM.f.src',
               'CGSREVCOM.f.src',
#               'ChebyREVCOM.f.src',
               'GMRESREVCOM.f.src',
#               'JacobiREVCOM.f.src',
               'QMRREVCOM.f.src',
#               'SORREVCOM.f.src'
               ]

    Util = ['getbreak.f.src']
    sources = Util + methods + ['_iterative.pyf.src']
    sources = [join('iterative', x) for x in sources]
    sources += get_g77_abi_wrappers(lapack_opt)

    ext = config.add_extension('_iterative',
                               sources=sources,
                               f2py_options=f2py_options,
                               extra_info=lapack_opt)
    ext._pre_build_hook = pre_build_hook

    config.add_data_dir('tests')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
