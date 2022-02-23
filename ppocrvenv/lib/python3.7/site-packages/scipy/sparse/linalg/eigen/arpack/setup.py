from os.path import join


def configuration(parent_package='',top_path=None):
    from scipy._build_utils.system_info import get_info
    from numpy.distutils.misc_util import Configuration
    from scipy._build_utils import (get_g77_abi_wrappers,
                                    gfortran_legacy_flag_hook,
                                    blas_ilp64_pre_build_hook,
                                    uses_blas64, get_f2py_int64_options)

    if uses_blas64():
        lapack_opt = get_info('lapack_ilp64_opt', 2)
        pre_build_hook = (gfortran_legacy_flag_hook,
                          blas_ilp64_pre_build_hook(lapack_opt))
        f2py_options = get_f2py_int64_options()
    else:
        lapack_opt = get_info('lapack_opt')
        pre_build_hook = gfortran_legacy_flag_hook
        f2py_options = None

    config = Configuration('arpack', parent_package, top_path)

    arpack_sources = [join('ARPACK','SRC', '*.f')]
    arpack_sources.extend([join('ARPACK','UTIL', '*.f')])

    arpack_sources += get_g77_abi_wrappers(lapack_opt)

    config.add_library('arpack_scipy', sources=arpack_sources,
                       include_dirs=[join('ARPACK', 'SRC')],
                       _pre_build_hook=pre_build_hook)

    ext = config.add_extension('_arpack',
                               sources=['arpack.pyf.src'],
                               libraries=['arpack_scipy'],
                               f2py_options=f2py_options,
                               extra_info=lapack_opt,
                               depends=arpack_sources)
    ext._pre_build_hook = pre_build_hook

    config.add_data_dir('tests')

    # Add license files
    config.add_data_files('ARPACK/COPYING')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
